import torch
from torch import nn
from torch.nn import functional as F


class SegLoss(nn.Module):
    def __init__(self, min_cls=0, max_cls=-1, scale_factor=1, use_bce=False, ignore_index=-100):
        super(SegLoss, self).__init__()
        self.min_cls = min_cls
        self.max_cls = max_cls
        self.scale_factor = scale_factor
        self.use_bce = use_bce
        self.ignore_index = ignore_index

    def prepare_target(self, targets, mask):
        labels = []

        for t in targets:
            t = t["basis_sem"].to(mask.device).unsqueeze(0)
            # t = t.get_field("basis_seg_masks").get_mask_tensor().unsqueeze(0)
            if self.min_cls > 0:
                t = t - self.min_cls
                t = torch.clamp(t, min=0)
            if self.max_cls > 0:
                t = torch.clamp(t, max=self.max_cls)
            if self.scale_factor != 1:
                t = F.interpolate(
                    t.unsqueeze(0),
                    scale_factor=self.scale_factor,
                    mode='nearest').long().squeeze()
            labels.append(t)

        batched_labels = mask.new_full(
            (mask.size(0), mask.size(2), mask.size(3)),
            0,  # ignore 0 !!!
            dtype=torch.long)
        for label, pad_label in zip(labels, batched_labels):
            pad_label[: label.shape[0], : label.shape[1]].copy_(label)

        return batched_labels

    def prepare_target_inst(self, targets, mask):
        labels = []

        for t in targets:
            t = t["basis_ins"].to(mask.device).unsqueeze(0)
            # t = t.get_field("ins_masks").get_mask_tensor().unsqueeze(0)
            if self.scale_factor != 1:
                t = F.interpolate(
                    t,
                    scale_factor=self.scale_factor,
                    mode='nearest').squeeze()
            labels.append(t)

        batched_labels = mask.new_full((mask.size(0), 2, mask.size(2), mask.size(3)), 0, dtype=torch.float)
        for label, pad_label in zip(labels, batched_labels):
            pad_label[:, :label.shape[1], :label.shape[2]].copy_(label)

        return batched_labels

    def forward(self, mask, target, ins=None):
        '''
            mask : Tensor
            target : list[Boxlist]
        '''
        losses = {}
        mask = F.interpolate(mask, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        gt_sem = target["gt_sem"]
        if self.min_cls > 0:
            gt_sem = gt_sem - self.min_cls
            gt_sem = torch.clamp(gt_sem, min=0)
        if self.max_cls > 0:
            gt_sem = torch.clamp(gt_sem, max=self.max_cls)
        if self.use_bce:
            one_hot = torch.zeros_like(mask)
            one_hot.scatter_(1, gt_sem.unsqueeze(1), 1.0)
            losses['seg'] = F.binary_cross_entropy_with_logits(mask, one_hot)
        else:
            losses['seg'] = F.cross_entropy(mask, gt_sem, ignore_index=self.ignore_index)
        if ins is not None:
            target_ins = target["gt_ins"]
            ins = F.interpolate(ins, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
            is_bg = (gt_sem == 0).float()
            losses['ins'] = F.l1_loss(ins[:, :2], target_ins)
            losses['bg'] = F.binary_cross_entropy_with_logits(ins[:, 2], is_bg)
        return losses
