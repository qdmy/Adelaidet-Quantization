import math
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import DFConv2d, IOULoss, get_norm, Conv2d
from detectron2.layers import ShapeSpec
from detectron2.modeling.quantization import dorefa_clip
from detectron2.modeling.quantization import linear_quantization
from .fcos_outputs import FCOSOutputs
from ..build import PROPOSAL_GENERATOR_REGISTRY

__all__ = ["QFCOS"]

INF = 100000000


@PROPOSAL_GENERATOR_REGISTRY.register()
class QFCOS(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        # fmt: off
        self.in_features          = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides          = cfg.MODEL.FCOS.FPN_STRIDES
        self.focal_loss_alpha     = cfg.MODEL.FCOS.LOSS_ALPHA
        self.focal_loss_gamma     = cfg.MODEL.FCOS.LOSS_GAMMA
        self.center_sample        = cfg.MODEL.FCOS.CENTER_SAMPLE
        self.strides              = cfg.MODEL.FCOS.FPN_STRIDES
        self.radius               = cfg.MODEL.FCOS.POS_RADIUS
        self.pre_nms_thresh_train = cfg.MODEL.FCOS.INFERENCE_TH_TRAIN
        self.pre_nms_thresh_test  = cfg.MODEL.FCOS.INFERENCE_TH_TEST
        self.pre_nms_topk_train   = cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN
        self.pre_nms_topk_test    = cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST
        self.nms_thresh           = cfg.MODEL.FCOS.NMS_TH
        self.yield_proposal       = cfg.MODEL.FCOS.YIELD_PROPOSAL
        self.post_nms_topk_train  = cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN
        self.post_nms_topk_test   = cfg.MODEL.FCOS.POST_NMS_TOPK_TEST
        # fmt: on
        self.iou_loss = IOULoss(cfg.MODEL.FCOS.LOC_LOSS_TYPE)
        # generate sizes of interest
        soi = []
        prev_size = -1
        for s in cfg.MODEL.FCOS.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi
        if cfg.MODEL.FCOS.NOT_SHARED_HEAD:
            self.fcos_head = FCOSNotSharedHead(cfg, [input_shape[f] for f in self.in_features])
        elif cfg.MODEL.QUANTIZATION.QUANTIZE_ONE_HEAD:
            self.fcos_head = FCOSNotSharedOneHead(cfg, [input_shape[f] for f in self.in_features])
        else:
            self.fcos_head = FCOSHead(cfg, [input_shape[f] for f in self.in_features])

    def forward(self, images, features, gt_instances=None, top_module=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        features = [features[f] for f in self.in_features]
        locations = self.compute_locations(features)
        pred_class_logits, pred_deltas, pred_centerness, top_feats, bbox_towers = self.fcos_head(
            features, top_module, self.yield_proposal)

        if self.training:
            pre_nms_thresh = self.pre_nms_thresh_train
            pre_nms_topk = self.pre_nms_topk_train
            post_nms_topk = self.post_nms_topk_train
        else:
            pre_nms_thresh = self.pre_nms_thresh_test
            pre_nms_topk = self.pre_nms_topk_test
            post_nms_topk = self.post_nms_topk_test

        outputs = FCOSOutputs(
            images,
            locations,
            pred_class_logits,
            pred_deltas,
            pred_centerness,
            self.focal_loss_alpha,
            self.focal_loss_gamma,
            self.iou_loss,
            self.center_sample,
            self.sizes_of_interest,
            self.strides,
            self.radius,
            self.fcos_head.num_classes,
            pre_nms_thresh,
            pre_nms_topk,
            self.nms_thresh,
            post_nms_topk,
            gt_instances,
        )

        results = {}
        if self.yield_proposal:
            results["features"] = {
                f: b for f, b in zip(self.in_features, bbox_towers)}

        if self.training:
            losses, extras = outputs.losses()
            # losses = {k: v * self.loss_weight for k, v in losses.items()}
            if top_module is not None:
                results["extras"] = extras
                results["top_feats"] = top_feats
            if self.yield_proposal:
                with torch.no_grad():
                    results["proposals"] = outputs.predict_proposals(top_feats)
        else:
            losses = {}
            with torch.no_grad():
                proposals = outputs.predict_proposals(top_feats)
            if self.yield_proposal:
                results["proposals"] = proposals
            else:
                results = proposals
        return results, losses

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


class FCOSHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        # TODO: Implement the sigmoid version first.
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        head_configs = {"cls": (cfg.MODEL.FCOS.NUM_CLS_CONVS,
                                False),
                        "bbox": (cfg.MODEL.FCOS.NUM_BOX_CONVS,
                                 cfg.MODEL.FCOS.USE_DEFORMABLE),
                        "share": (cfg.MODEL.FCOS.NUM_SHARE_CONVS,
                                  cfg.MODEL.FCOS.USE_DEFORMABLE)}
        self.bits_weights         = cfg.MODEL.QUANTIZATION.wt_bit
        self.bits_activations     = cfg.MODEL.QUANTIZATION.fm_bit
        self.quan_type            = cfg.MODEL.QUANTIZATION.QUAN_TYPE
        self.head_configs = head_configs
        # norm = None if cfg.MODEL.FCOS.NORM == "None" else cfg.MODEL.FCOS.NORM
        self.norm = cfg.MODEL.FCOS.NORM
        self.feature_num = len(self.fpn_strides)
        self.shared_norm = cfg.MODEL.FCOS.SHARED_NORM
        self.quantize_first_and_last = cfg.MODEL.QUANTIZATION.QUANTIZE_FIRST_AND_LAST
        self.merge_bn = cfg.MODEL.QUANTIZATION.MERGE_BN

        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            if use_deformable:
                conv_func = DFConv2d
            else:
                if self.quan_type in "dorefa_clip":
                    conv_func = dorefa_clip.QConv2d
                else:
                    conv_func = linear_quantization.QConv2d
            for i in range(num_convs):
                tower.append(
                    conv_func(
                        in_channels, in_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=False, 
                        bits_weights=self.bits_weights,
                        bits_activations=self.bits_activations,
                        merge_bn=self.merge_bn))
                tower.append(get_norm(self.norm, in_channels))
                tower.append(nn.ReLU(inplace=True))
                if self.norm in ['BN', 'SyncBN', 'GN'] and not self.shared_norm:
                    self.add_module('{}_norm{}'.format(head, i),
                            nn.ModuleList([get_norm(self.norm, in_channels) for j in range(self.feature_num)]))
                    tower[-2] = getattr(self, "{}_norm{}".format(head, i))[0]
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        if self.quantize_first_and_last:
            self.cls_logits = conv_func(
                in_channels, self.num_classes, kernel_size=3, stride=1,
                padding=1, bits_weights=self.bits_weights, 
                bits_activations=self.bits_activations,
                merge_bn=self.merge_b)
            self.bbox_pred = conv_func(
                in_channels, 4, kernel_size=3, stride=1,
                padding=1, bits_weights=self.bits_weights, 
                bits_activations=self.bits_activations,
                merge_bn=self.merge_b)
            self.centerness = conv_func(
                in_channels, 1, kernel_size=3, stride=1,
                padding=1, bits_weights=self.bits_weights, 
                bits_activations=self.bits_activations,
                merge_bn=self.merge_b)
        else:
            self.cls_logits = nn.Conv2d(
                in_channels, self.num_classes, kernel_size=3, stride=1,
                padding=1)
            self.bbox_pred = nn.Conv2d(
                in_channels, 4, kernel_size=3, stride=1,
                padding=1)
            self.centerness = nn.Conv2d(
                in_channels, 1, kernel_size=3, stride=1,
                padding=1)

        for modules in [self.cls_tower, self.bbox_tower, self.share_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def update_norm_layer(self, index=0):
        if self.norm in ['BN', 'SyncBN', 'GN'] and not self.shared_norm:
            for head in self.head_configs:
                num_convs, use_deformable = self.head_configs[head]
                for i in range(num_convs):
                    getattr(self, "{}_tower".format(head))[1 + 3*i] = getattr(self, "{}_norm{}".format(head, i))[index]
        pass

    def forward(self, x, top_module=None, yield_bbox_towers=False):
        """
        offset related operations are messy
        """
        logits = []
        bbox_reg = []
        centerness = []
        top_feats = []
        bbox_towers = []
        for l, feature in enumerate(x):
            self.update_norm_layer(l)
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)
            if yield_bbox_towers:
                bbox_towers.append(bbox_tower)

            logits.append(self.cls_logits(cls_tower))
            centerness.append(self.centerness(bbox_tower))
            reg = F.relu_(self.bbox_pred(bbox_tower))
            bbox_reg.append(reg)
            if top_module is not None:
                top_feats.append(top_module(bbox_tower))
        return logits, bbox_reg, centerness, top_feats, bbox_towers


class FCOSNotSharedHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        # TODO: Implement the sigmoid version first.
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.bits_weights         = cfg.MODEL.QUANTIZATION.wt_bit
        self.bits_activations     = cfg.MODEL.QUANTIZATION.fm_bit
        self.quan_type            = cfg.MODEL.QUANTIZATION.QUAN_TYPE
        self.quantize_first_and_last = cfg.MODEL.QUANTIZATION.QUANTIZE_FIRST_AND_LAST
        self.merge_bn = cfg.MODEL.QUANTIZATION.MERGE_BN
        # norm = None if cfg.MODEL.FCOS.NORM == "None" else cfg.MODEL.FCOS.NORM
        self.norm = cfg.MODEL.FCOS.NORM
        self.feature_num = len(self.fpn_strides)

        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        self.head_cls_tower = nn.ModuleList([])
        self.head_bbox_tower = nn.ModuleList([])

        num_heads = 5

        if self.quan_type in "dorefa_clip":
            conv_func = dorefa_clip.QConv2d
        else:
            conv_func = linear_quantization.QConv2d
        
        for i in range(num_heads):
            cls_tower = nn.ModuleList([])
            bbox_tower = nn.ModuleList([])

            for j in range(cfg.MODEL.FCOS.NUM_CLS_CONVS):
                cls_tower.append(conv_func(
                    in_channels, in_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=False, 
                    bits_weights=self.bits_weights,
                    bits_activations=self.bits_activations,
                    merge_bn=self.merge_bn))
                cls_tower.append(get_norm(self.norm, in_channels))
                cls_tower.append(nn.ReLU(inplace=True))

            for j in range(cfg.MODEL.FCOS.NUM_BOX_CONVS):
                bbox_tower.append(conv_func(
                    in_channels, in_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=False, 
                    bits_weights=self.bits_weights,
                    bits_activations=self.bits_activations,
                    merge_bn=self.merge_bn))
                bbox_tower.append(get_norm(self.norm, in_channels))
                bbox_tower.append(nn.ReLU(inplace=True))
            
            self.head_cls_tower.append(nn.Sequential(*cls_tower))
            self.head_bbox_tower.append(nn.Sequential(*bbox_tower))

        if self.quantize_first_and_last:
            self.cls_logits = conv_func(
                in_channels, self.num_classes, kernel_size=3, stride=1,
                padding=1, bits_weights=self.bits_weights,
                bits_activations=self.bits_activations,
                merge_bn=self.merge_bn)
            self.bbox_pred = conv_func(
                in_channels, 4, kernel_size=3, stride=1,
                padding=1, bits_weights=self.bits_weights,
                bits_activations=self.bits_activations,
                merge_bn=self.merge_bn)
            self.centerness = conv_func(
                in_channels, 1, kernel_size=3, stride=1,
                padding=1, bits_weights=self.bits_weights,
                bits_activations=self.bits_activations,
                merge_bn=self.merge_bn)
        else:
            self.cls_logits = nn.Conv2d(
                in_channels, self.num_classes, kernel_size=3, stride=1,
                padding=1)
            self.bbox_pred = nn.Conv2d(
                in_channels, 4, kernel_size=3, stride=1,
                padding=1)
            self.centerness = nn.Conv2d(
                in_channels, 1, kernel_size=3, stride=1,
                padding=1)

        for tower in [self.head_cls_tower, self.head_bbox_tower]:
            for head in tower:
                for l in head:
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        if l.bias is not None:
                            torch.nn.init.constant_(l.bias, 0)
        

        for modules in [self.cls_logits, self.bbox_pred, self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                            torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x, top_module=None, yield_bbox_towers=False):
        """
        offset related operations are messy
        """
        logits = []
        bbox_reg = []
        centerness = []
        top_feats = []
        bbox_towers = []
        for l, feature in enumerate(x):
            cls_tower = self.head_cls_tower[l](feature)
            bbox_tower = self.head_bbox_tower[l](feature)
            if yield_bbox_towers:
                bbox_towers.append(bbox_tower)

            logits.append(self.cls_logits(cls_tower))
            centerness.append(self.centerness(bbox_tower))
            reg = F.relu_(self.bbox_pred(bbox_tower))
            bbox_reg.append(reg)
            if top_module is not None:
                top_feats.append(top_module(bbox_tower))
        return logits, bbox_reg, centerness, top_feats, bbox_towers


class FCOSNotSharedOneHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        # TODO: Implement the sigmoid version first.
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.bits_weights         = cfg.MODEL.QUANTIZATION.wt_bit
        self.bits_activations     = cfg.MODEL.QUANTIZATION.fm_bit
        self.quan_type            = cfg.MODEL.QUANTIZATION.QUAN_TYPE
        # norm = None if cfg.MODEL.FCOS.NORM == "None" else cfg.MODEL.FCOS.NORM
        self.norm = cfg.MODEL.FCOS.NORM
        self.quantize_first_and_last = cfg.MODEL.QUANTIZATION.QUANTIZE_FIRST_AND_LAST
        self.merge_bn = cfg.MODEL.QUANTIZATION.MERGE_BN
        self.feature_num = len(self.fpn_strides)

        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        self.head_cls_tower = nn.ModuleList([])
        self.head_bbox_tower = nn.ModuleList([])

        num_heads = 5

        # if use_sigmoid_quantization:
        #     conv_func = dorefa_clip_sigmoid.QConv2d
        # else:
        #     conv_func = dorefa_clip.QConv2d
        
        for i in range(num_heads):
            cls_tower = nn.ModuleList([])
            bbox_tower = nn.ModuleList([])

            if cfg.MODEL.QUANTIZATION.QUANTIZE_HEAD == i:
                if self.quan_type in "dorefa_clip":
                    conv_func = dorefa_clip.QConv2d
                else:
                    conv_func = linear_quantization.QConv2d
            else:
                conv_func = nn.Conv2d

            for j in range(cfg.MODEL.FCOS.NUM_CLS_CONVS):
                if isinstance(conv_func, (dorefa_clip.QConv2d, linear_quantization.QConv2d)):
                    cls_tower.append(conv_func(
                        in_channels, in_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=False, 
                        bits_weights=self.bits_weights,
                        bits_activations=self.bits_activations,
                        merge_bn=self.merge_bn))
                else:
                    cls_tower.append(conv_func(
                        in_channels, in_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=False))
                cls_tower.append(get_norm(self.norm, in_channels))
                cls_tower.append(nn.ReLU(inplace=True))

            for j in range(cfg.MODEL.FCOS.NUM_BOX_CONVS):
                if isinstance(conv_func, (dorefa_clip.QConv2d, linear_quantization.QConv2d)):
                    bbox_tower.append(conv_func(
                        in_channels, in_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=False, 
                        bits_weights=self.bits_weights,
                        bits_activations=self.bits_activations,
                        merge_bn=self.merge_bn))
                else:
                    bbox_tower.append(conv_func(
                        in_channels, in_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=False))
                bbox_tower.append(get_norm(self.norm, in_channels))
                bbox_tower.append(nn.ReLU(inplace=True))
            
            self.head_cls_tower.append(nn.Sequential(*cls_tower))
            self.head_bbox_tower.append(nn.Sequential(*bbox_tower))

        self.cls_logits = nn.Conv2d(
            in_channels, self.num_classes, kernel_size=3, stride=1,
            padding=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1)
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1)

        for tower in [self.head_cls_tower, self.head_bbox_tower]:
            for head in tower:
                for l in head:
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        if l.bias is not None:
                            torch.nn.init.constant_(l.bias, 0)
        

        for modules in [self.cls_logits, self.bbox_pred, self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                            torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x, top_module=None, yield_bbox_towers=False):
        """
        offset related operations are messy
        """
        logits = []
        bbox_reg = []
        centerness = []
        top_feats = []
        bbox_towers = []
        for l, feature in enumerate(x):
            cls_tower = self.head_cls_tower[l](feature)
            bbox_tower = self.head_bbox_tower[l](feature)
            if yield_bbox_towers:
                bbox_towers.append(bbox_tower)

            logits.append(self.cls_logits(cls_tower))
            centerness.append(self.centerness(bbox_tower))
            reg = F.relu_(self.bbox_pred(bbox_tower))
            bbox_reg.append(reg)
            if top_module is not None:
                top_feats.append(top_module(bbox_tower))
        return logits, bbox_reg, centerness, top_feats, bbox_towers