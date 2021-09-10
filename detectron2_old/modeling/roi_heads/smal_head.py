import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import cat
from .roi_heads import ROI_HEADS_REGISTRY, ROIHeads
from ..poolers import ROIPooler
from .mask_head import mask_rcnn_loss, mask_rcnn_inference


@ROI_HEADS_REGISTRY.register()
class SmalHead(ROIHeads):
    def __init__(self, cfg, input_shape):
        super(SmalHead, self).__init__(cfg, input_shape)
        # fmt: off
        pooler_resolution   = cfg.MODEL.SMALMASK.POOLER_RESOLUTION
        pooler_scales       = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio      = cfg.MODEL.SMALMASK.POOLER_SAMPLING_RATIO
        pooler_type         = cfg.MODEL.SMALMASK.POOLER_TYPE
        self.mask_size      = cfg.MODEL.SMALMASK.MASK_SIZE
        canonical_size_box  = cfg.MODEL.SMALMASK.CANONICAL_SIZE_BOX
        canonical_size_mask = cfg.MODEL.SMALMASK.CANONICAL_SIZE_MASK
        canonical_level     = cfg.MODEL.SMALMASK.CANONICAL_LEVEL
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]
        self.feat_size = np.prod((in_channels, pooler_resolution, pooler_resolution))

        self.pooler_mask_l = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            canonical_box_size=canonical_size_mask,
            canonical_level=canonical_level,
            assign_crit="length")

        self.pooler_mask_s = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            canonical_box_size=canonical_size_box,
            canonical_level=canonical_level,
            assign_crit="length")

        self.mask_dim = self.mask_size ** 2
        self.mask_predictor = nn.Linear(self.feat_size, self.mask_dim)
        nn.init.normal_(self.mask_predictor.weight, std=0.001)
        for l in [self.mask_predictor, self.box_predictor]:
            nn.init.constant_(l.bias, 0)

    def forward(self, images, features, proposals, targets=None):
        del features

        features = proposals["features"]
        pred_instances = proposals["proposals"]

        del images
        features_list = [features[f] for f in self.in_features]

        if self.training:
            losses = {}
            # During training, gt boxes are used by the mask head as proposal.
            for key, pooler in zip(['s', 'l'], [self.pooler_mask_s, self.pooler_mask_l]):
                mask_rois = pooler(features_list, [x.gt_boxes for x in targets])
                mask_logits = self.mask_predictor(torch.flatten(mask_rois, start_dim=1)).view(
                    -1, 1, self.mask_size, self.mask_size)
                # process targets for loss
                for x in targets:
                    x.proposal_boxes = x.gt_boxes
                losses["loss_mask_{}".format(key)] = mask_rcnn_loss(mask_logits, targets)
            return proposals, losses
        else:
            pred_boxes = [x.pred_boxes for x in pred_instances]
            mask_rois = self.pooler_mask_l(features_list, pred_boxes)
            mask_logits = self.mask_predictor(torch.flatten(mask_rois, start_dim=1)).view(
                -1, 1, self.mask_size, self.mask_size)
            mask_rcnn_inference(mask_logits, pred_instances)
            return pred_instances, {}
