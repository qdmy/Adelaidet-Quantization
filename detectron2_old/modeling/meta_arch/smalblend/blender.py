import torch
from torch.nn import functional as F

from detectron2.layers import cat
from detectron2.modeling.poolers import ROIPooler


def build_blender(cfg):
    return Blender(cfg)


class Blender(object):
    def __init__(self, cfg):

        # fmt: off
        self.pooler_resolution = cfg.MODEL.BLENDMASK.BOTTOM_RESOLUTION
        sampling_ratio         = cfg.MODEL.BLENDMASK.POOLER_SAMPLING_RATIO
        pooler_type            = cfg.MODEL.BLENDMASK.POOLER_TYPE
        pooler_scales          = cfg.MODEL.BLENDMASK.POOLER_SCALES
        self.attn_size         = cfg.MODEL.BLENDMASK.ATTN_SIZE
        self.top_interp        = cfg.MODEL.BLENDMASK.TOP_INTERP
        num_bases              = cfg.MODEL.BASIS_MODULE.NUM_BASES
        # fmt: on

        self.attn_len = num_bases * self.attn_size * self.attn_size

        self.pooler = ROIPooler(
            output_size=self.pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            canonical_level=2)

    def __call__(self, bases, attns, proposals, gt_instances):
        if gt_instances is not None:
            # training
            rois = self.pooler(bases, [x.gt_boxes for x in gt_instances])

            # gen targets
            gt_masks = []
            for instances_per_image in gt_instances:
                if len(instances_per_image.gt_boxes.tensor) == 0:
                    continue
                gt_mask_per_image = instances_per_image.gt_masks.crop_and_resize(
                    instances_per_image.gt_boxes.tensor, self.pooler_resolution
                ).to(device=rois.device)
                gt_masks.append(gt_mask_per_image)
            gt_masks = cat(gt_masks, dim=0)
            N = gt_masks.size(0)
            gt_masks = gt_masks.view(N, -1)

            losses = {}
            for i, attn in enumerate(attns):
                pred_mask_logits = self.merge_bases(rois, attn)
                mask_loss = F.binary_cross_entropy_with_logits(
                    pred_mask_logits, gt_masks.to(dtype=torch.float32))
                losses["loss_mask_{}".format(i)] = mask_loss
            return None, losses
        else:
            # no proposals
            proposals = proposals["proposals"]
            total_instances = sum([len(x) for x in proposals])
            if total_instances == 0:
                return proposals, {}
            rois = self.pooler(bases, [x.pred_boxes for x in proposals])
            pred_mask_logits = self.merge_bases(rois, attns).sigmoid()
            pred_mask_logits = pred_mask_logits.view(
                -1, 1, self.pooler_resolution, self.pooler_resolution)
            start_ind = 0
            for box in proposals:
                end_ind = start_ind + len(box)
                box.pred_masks = pred_mask_logits[start_ind:end_ind]
                start_ind = end_ind
            return proposals, {}

    def merge_bases(self, rois, coeffs, location_to_inds=None):
        # merge predictions
        N = coeffs.size(0)
        if location_to_inds is not None:
            rois = rois[location_to_inds]
        N, B, H, W = rois.size()

        coeffs = coeffs.view(N, -1, self.attn_size, self.attn_size)
        coeffs = F.interpolate(coeffs, (H, W),
                               mode=self.top_interp).softmax(dim=1)
        masks_preds = (rois * coeffs).sum(dim=1)
        return masks_preds.view(N, -1)
