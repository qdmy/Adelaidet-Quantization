# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling.quantization import dorefa_clip
from detectron2.modeling.quantization import linear_quantization

from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from .qresnet import build_qresnet_backbone
from .resnet_lpf import build_resnet_lpf_backbone

__all__ = ["build_qresnet_fpn_backbone",
           "build_retinanet_qresnet_qfpn_backbone",
           "build_fcos_qresnet_qfpn_backbone",
           "QFPN"]


class QFPN(Backbone):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, bottom_up, in_features, out_channels, norm="", activation=False, top_block=None, fuse_type="sum", bits_weights=32, bits_activations=32, merge_bn=False, quan_type="dorefa_clip"
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(QFPN, self).__init__()
        assert isinstance(bottom_up, Backbone)

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        in_strides = [bottom_up.out_feature_strides[f] for f in in_features]
        in_channels = [bottom_up.out_feature_channels[f] for f in in_features]

        _assert_strides_are_log2_contiguous(in_strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            if quan_type in "dorefa_clip":
                conv = dorefa_clip.QConv2d
            else:
                conv = linear_quantization.QConv2d

            lateral_conv = conv(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm,  bits_weights=bits_weights, bits_activations=bits_activations,
                merge_bn=merge_bn
            )
            output_conv = conv(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                bits_weights=bits_weights,
                bits_activations=bits_activations,
                merge_bn=merge_bn
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(in_strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = in_features
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in in_strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = in_strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type
        self.use_relu = activation

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str: Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str: Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        bottom_up_features = self.bottom_up(x)
        x = [bottom_up_features[f] for f in self.in_features[::-1]]
        results = []
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))
        for features, lateral_conv, output_conv in zip(
            x[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            if self.use_relu:
                prev_features = F.relu_(prev_features)
                results.insert(0, F.relu_(output_conv(prev_features)))
            else:
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class QLastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet and FCOS to generate extra layers, P6 and P7 from
    C5 or P5 feature.
    """

    def __init__(self, in_channels, out_channels, in_features="res5", norm="", activation=False,
                 bits_weights=32, bits_activations=32, quan_type="dorefa_clip", merge_bn=False):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_features
        self.use_relu = activation
        use_bias = norm == ""

        if quan_type in "dorefa_clip":
            conv = dorefa_clip.QConv2d
        else:
            conv = linear_quantization.QConv2d

        self.p6 = conv(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=use_bias, bits_weights=bits_weights, bits_activations=bits_activations, 
        norm=get_norm(norm, out_channels), merge_bn=merge_bn)
        self.p7 = conv(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=use_bias, bits_weights=bits_weights, bits_activations=bits_activations,
        norm=get_norm(norm, out_channels), merge_bn=merge_bn)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, x):
        p6 = self.p6(x)
        p7 = self.p7(F.relu_(p6))
        if self.use_relu:
            p7 = F.relu_(p7)
        return [p6, p7]


class QLastLevelP6(nn.Module):
    """
    This module is used in FCOS to generate extra layers
    """

    def __init__(self, in_channels, out_channels, in_features="res5", norm="", activation=False, bits_weights=32, bits_activations=32, quan_type="dorefa_clip", merge_bn=False):
        super().__init__()
        self.num_levels = 1
        self.in_feature = in_features
        self.use_relu = activation
        use_bias = norm == ""

        if quan_type in "dorefa_clip":
            conv = dorefa_clip.QConv2d
        else:
            conv = linear_quantization.QConv2d

        self.p6 = conv(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=use_bias, bits_weights=bits_weights, bits_activations=bits_activations, 
        norm=get_norm(norm, out_channels), merge_bn=merge_bn)
        for module in [self.p6]:
            weight_init.c2_xavier_fill(module)

    def forward(self, x):
        p6 = self.p6(x)
        if self.use_relu:
            p6 = F.relu_(p6)
        return [p6]


@BACKBONE_REGISTRY.register()
def build_qresnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_qresnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    bits_weights        = cfg.MODEL.QUANTIZATION.wt_bit
    bits_activations    = cfg.MODEL.QUANTIZATION.fm_bit
    quan_type = cfg.MODEL.QUANTIZATION.QUAN_TYPE
    merge_bn = cfg.MODEL.QUANTIZATION.MERGE_BN

    backbone = QFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        activation=cfg.MODEL.FPN.USE_RELU,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        bits_weights=bits_weights, 
        bits_activations=bits_activations,
        merge_bn=merge_bn,
        quan_type=quan_type,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_retinanet_qresnet_qfpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_qresnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    norm = cfg.MODEL.FPN.NORM
    activation = cfg.MODEL.FPN.USE_RELU
    in_channels_p6p7 = bottom_up.out_feature_channels["res5"]
    bits_weights        = cfg.MODEL.QUANTIZATION.wt_bit
    bits_activations    = cfg.MODEL.QUANTIZATION.fm_bit
    quan_type = cfg.MODEL.QUANTIZATION.QUAN_TYPE
    merge_bn = cfg.MODEL.QUANTIZATION.MERGE_BN

    backbone = QFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=norm,
        activation=activation,
        top_block=QLastLevelP6P7(in_channels_p6p7, out_channels, norm=norm, activation=activation, bits_weights=bits_weights, bits_activations=bits_activations, quan_type=quan_type, merge_bn=merge_bn),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        bits_weights=bits_weights, 
        bits_activations=bits_activations,
        quan_type=quan_type,
        merge_bn=merge_bn
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_fcos_qresnet_qfpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_qresnet_backbone(cfg, input_shape)
    in_features         = cfg.MODEL.FPN.IN_FEATURES
    out_channels        = cfg.MODEL.FPN.OUT_CHANNELS
    top_levels          = cfg.MODEL.FCOS.TOP_LEVELS
    in_channels_top = out_channels
    norm                = cfg.MODEL.FPN.NORM
    activation          = cfg.MODEL.FPN.USE_RELU
    bits_weights        = cfg.MODEL.QUANTIZATION.wt_bit
    bits_activations    = cfg.MODEL.QUANTIZATION.fm_bit
    quan_type = cfg.MODEL.QUANTIZATION.QUAN_TYPE
    merge_bn = cfg.MODEL.QUANTIZATION.MERGE_BN

    if top_levels == 2:
        top_block = QLastLevelP6P7(in_channels_top, out_channels, "p5", norm=norm, activation=activation, bits_weights=bits_weights, bits_activations=bits_activations, quan_type=quan_type, merge_bn=merge_bn)
    if top_levels == 1:
        top_block = QLastLevelP6(in_channels_top, out_channels, "p5", norm=norm, activation=activation, bits_weights=bits_weights, bits_activations=bits_activations, quan_type=quan_type, merge_bn=merge_bn)
    elif top_levels == 0:
        top_block = None
    backbone = QFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=norm,
        activation=activation,
        top_block=top_block,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        bits_weights=bits_weights, 
        bits_activations=bits_activations,
        quan_type=quan_type,
        merge_bn=merge_bn
    )
    return backbone

