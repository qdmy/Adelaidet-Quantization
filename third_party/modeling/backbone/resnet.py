# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from fvcore.common.config import CfgNode

from third_party.cls.resnet_ import (
        ResNet as ResNet_,
        BottleNeck,
        BasicBlock,
        )

from codebase.third_party.spos_ofa.ofa.imagenet_classification.networks.resnets import ResNet50, ResNet50D

__all__ = [
    "build_cls_resnet_backbone",
]

class Wrapper(ResNet_, Backbone):
    def __init__(self, block, layers, args, out_features=None):
        super(ResNet_, self).__init__()
        super(Wrapper, self).__init__(block, layers, args)

        stride = 4
        if 'cifar10' in args.keyword or 'cifar100' in args.keyword:
            stride = 1

        self._out_feature_strides  = { 'stem': stride,
                'layer1': stride, 'layer2': stride*2, 'layer3': stride*4, 'layer4': stride*8 }
        self._out_feature_channels = { 'stem': self.input_channel,
                'layer1': self.input_channel * 1 * block.expansion,
                'layer2': self.input_channel * 2 * block.expansion,
                'layer3': self.input_channel * 4 * block.expansion,
                'layer4': self.input_channel * 8 * block.expansion } # define each stage's out_channel

        if out_features is None:
            out_features = ['linear']
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        return super(Wrapper, self).forward(x)

    def output_shape(self):
        return {
                name: ShapeSpec(
                    channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
                    )
                for name in self._out_features
            }


@BACKBONE_REGISTRY.register()
def build_cls_resnet_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    args = CfgNode()
    # fmt: off
    depth        = cfg.MODEL.RESNETS.DEPTH
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    args.keyword = cfg.MODEL.CLS.KEYWORD
    args.verbose = cfg.MODEL.CLS.VERBOSE
    args.base    = 1
    args.num_classes = 0
    assert hasattr(args, 'keyword'), 'args should have attribution of "keyword"'
    if isinstance(args.keyword, str):
        args.keyword = [x.strip() for x in args.keyword.split(',')]
    # fmt: on

    stages = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
            }[depth]

    block = BasicBlock if depth <= 34 else BottleNeck
    assert cfg.MODEL.RESNETS.RES2_OUT_CHANNELS == block.expansion * 64, 'unexpect channel out for layer1'

    return Wrapper(block, stages, args=args, out_features=out_features)


