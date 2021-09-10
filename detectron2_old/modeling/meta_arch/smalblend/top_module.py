import numpy as np

import torch
from torch import nn

from detectron2.modeling.poolers import ROIPooler


def build_top_module(cfg, attn_len, input_shape):
    return SmalHead(cfg, attn_len, input_shape)


class SmalHead(nn.Module):
    def __init__(self, cfg, attn_len, input_shape):
        super().__init__()

        # fmt: off
        pooler_resolution     = cfg.MODEL.SMALBLEND.POOLER_RESOLUTION
        self.in_features      = cfg.MODEL.SMALBLEND.IN_FEATURES
        self.feature_strides  = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels = {k: v.channels for k, v in input_shape.items()}
        pooler_scales         = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio        = cfg.MODEL.SMALBLEND.POOLER_SAMPLING_RATIO
        pooler_type           = cfg.MODEL.SMALBLEND.POOLER_TYPE
        canonical_lengths     = cfg.MODEL.SMALBLEND.CANONICAL_LENGTHS
        canonical_level       = cfg.MODEL.SMALBLEND.CANONICAL_LEVEL
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]
        self.feat_size = np.prod((in_channels, pooler_resolution, pooler_resolution))

        self.poolers = []
        for canonical_length in canonical_lengths:
            self.poolers.append(ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
                canonical_box_size=canonical_length,
                canonical_level=canonical_level,
                assign_crit="length"))

        self.predictor = nn.Linear(self.feat_size, attn_len)
        nn.init.normal_(self.predictor.weight, std=0.001)
        nn.init.constant_(self.predictor.bias, 0)

    def forward(self, proposals, targets=None):
        features = proposals["features"]
        pred_instances = proposals["proposals"]

        features_list = [features[f] for f in self.in_features]

        if self.training:
            # During training, gt boxes are used by the mask head as proposal.
            boxes = [x.gt_boxes for x in targets]
            results = []
            for pooler in self.poolers:
                mask_rois = pooler(features_list, boxes)
                mask_logits = self.predictor(torch.flatten(mask_rois, start_dim=1))
                results.append(mask_logits)
            return results
        else:
            boxes = [x.pred_boxes for x in pred_instances]
            mask_rois = self.poolers[0](features_list, boxes)
            mask_logits = self.predictor(torch.flatten(mask_rois, start_dim=1))
            return mask_logits
