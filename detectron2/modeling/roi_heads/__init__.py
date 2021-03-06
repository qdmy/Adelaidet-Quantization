# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .box_head import ROI_BOX_HEAD_REGISTRY, build_box_head
from .keypoint_head import ROI_KEYPOINT_HEAD_REGISTRY, build_keypoint_head
from .mask_head import ROI_MASK_HEAD_REGISTRY, build_mask_head
from .roi_heads import ROI_HEADS_REGISTRY, ROIHeads, StandardROIHeads, build_roi_heads
from .rotated_fast_rcnn import RROIHeads
from .light_head import LightROIHeads
from .smal_head import SmalHead

from . import cascade_rcnn  # isort:skip
