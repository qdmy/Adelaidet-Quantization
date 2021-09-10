import logging
import torch
import torch.nn.functional as F

from detectron2.layers import cat
from detectron2.structures import Instances, Boxes
from detectron2.utils.comm import reduce_sum, get_world_size
from fvcore.nn import sigmoid_focal_loss_jit

logger = logging.getLogger(__name__)

INF = 100000000

# TODO: comments for future refactoring of this module
#
# From @rbg:
# This code involves a significant amount of tensor reshaping and permuting. Look for
# ways to simplify this.

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization

Naming convention:

    class: refers to the classification of an position.

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box offsets.

    pred_class_logits: predicted classification scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(object).

    gt_class_logits: ground-truth classification labels for objectness

    pred_deltas: predicted box offsets

    gt_deltas: ground-truth box offsets
"""


def compute_centerness_targets(reg_targets):
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                 (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness)


def fcos_losses(
        gt_class,
        gt_deltas,
        gt_inds,
        pred_class_logits,
        pred_deltas,
        pred_centerness,
        focal_loss_alpha,
        focal_loss_gamma,
        iou_loss,
):
    """
    Args:
        gt_class_logits (Tensor): shape (N,), each element in {-1, 0, 1} representing
            ground-truth objectness labels with: -1 = ignore; 0 = not object; 1 = object.
        gt_deltas (Tensor): shape (N, box_dim), row i represents ground-truth
            box2box transform targets (dx, dy, dw, dh) or (dx, dy, dw, dh, da) that map anchor i to
            its matched ground-truth box.
        pred_class_logits (Tensor): shape (N,), each element is a predicted objectness
            logit.
        pred_deltas (Tensor): shape (N, box_dim), each row is a predicted box2box
            transform (dx, dy, dw, dh) or (dx, dy, dw, dh, da)

    Returns:
        objectness_loss, localization_loss, both unnormalized (summed over samples).
    """
    num_classes = pred_class_logits.size(1)
    gt_class = gt_class.flatten()
    pos_inds = torch.nonzero(gt_class != num_classes).squeeze(1)
    num_pos_per_gpu = pos_inds.numel()
    num_gpus = get_world_size()
    total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_per_gpu])).item()
    pred_deltas = pred_deltas[pos_inds]
    pred_centerness = pred_centerness[pos_inds]

    gt_deltas = gt_deltas[pos_inds]
    gt_inds = gt_inds[pos_inds]

    # prepare one_hot
    gt_class_target = torch.zeros_like(pred_class_logits)
    gt_class_target[pos_inds, gt_class[pos_inds]] = 1

    class_loss = sigmoid_focal_loss_jit(
        pred_class_logits,
        gt_class_target,
        alpha=focal_loss_alpha,
        gamma=focal_loss_gamma,
        reduction="sum",
    ) / max(total_num_pos / num_gpus, 1.0)  # add N to avoid dividing by a zero

    if pos_inds.numel() > 0:
        gt_centerness = compute_centerness_targets(gt_deltas)
        sum_gt_centerness = gt_centerness.sum()
        loss_denorm = reduce_sum(sum_gt_centerness).item() / num_gpus
        reg_loss = iou_loss(
            pred_deltas,
            gt_deltas,
            gt_centerness
        ) / loss_denorm
        centerness_loss = F.binary_cross_entropy_with_logits(
            pred_centerness,
            gt_centerness,
            reduction="sum"
        ) / max(total_num_pos / num_gpus, 1.0)
    else:
        reg_loss = pred_deltas.sum() * 0
        centerness_loss = pred_centerness.sum() * 0
        gt_centerness = None
        loss_denorm = None

    losses = {"loss_fcos_cls": class_loss,
              "loss_fcos_loc": reg_loss,
              "loss_fcos_ctr": centerness_loss}
    extras = {"pos_inds": pos_inds,
              "gt_inds": gt_inds,
              "gt_ctr": gt_centerness,
              "loss_denorm": loss_denorm}
    return losses, extras


class FCOSOutputs(object):
    def __init__(
            self,
            images,
            locations,
            pred_class_logits,
            pred_deltas,
            pred_centerness,
            focal_loss_alpha,
            focal_loss_gamma,
            iou_loss,
            center_sample,
            sizes_of_interest,
            strides,
            radius,
            num_classes,
            pre_nms_thresh,
            pre_nms_top_n,
            nms_thresh,
            fpn_post_nms_top_n,
            gt_instances=None,
    ):
        """
        Args:
            images (ImageList): :class:`ImageList` instance representing N input images
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for anchors.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, A*4, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
            anchors (list[list[Boxes]]): A list of N elements. Each element is a list of L
                Boxes. The Boxes at (n, l) stores the entire anchor array for feature map l in image
                n (i.e. the cell anchors repeated over all locations in feature map (n, l)).
            boundary_threshold (int): if >= 0, then anchors that extend beyond the image
                boundary by more than boundary_thresh are not used in training. Set to a very large
                number or < 0 to disable this behavior. Only needed in training.
            gt_boxes (list[Boxes], optional): A list of N elements. Element i a Boxes storing
                the ground-truth ("gt") boxes for image i.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.pred_class_logits = pred_class_logits
        self.pred_deltas = pred_deltas
        self.pred_centerness = pred_centerness
        self.locations = locations

        self.gt_instances = gt_instances
        self.num_feature_maps = len(pred_class_logits)
        self.num_images = len(images)
        self.image_sizes = images.image_sizes
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.iou_loss = iou_loss
        self.center_sample = center_sample
        self.sizes_of_interest = sizes_of_interest
        self.strides = strides
        self.radius = radius
        self.num_classes = num_classes
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n

    def _get_ground_truth(self):
        object_sizes_of_interest = self.sizes_of_interest
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(self.locations):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in self.locations]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(self.locations, dim=0)
        labels, reg_targets, target_inds = self.compute_targets_for_locations(
            points_all_level, self.gt_instances, expanded_object_sizes_of_interest)

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)
            target_inds[i] = torch.split(target_inds[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        target_inds_level_first = []
        for level in range(len(self.locations)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            reg_targets_level_first.append(
                torch.cat([reg_targets_per_im[level]
                           for reg_targets_per_im in reg_targets],
                          dim=0) / self.strides[level]
            )
            target_inds_level_first.append(
                torch.cat([target_inds_per_im[level] for target_inds_per_im in target_inds], dim=0)
            )

        return labels_level_first, reg_targets_level_first, target_inds_level_first

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1):
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax)
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        target_inds = []
        xs, ys = locations[:, 0], locations[:, 1]

        num_targets = 0
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes
            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                target_inds.append(labels_per_im.new_zeros(locations.size(0)) - 1)
                continue
            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sample:
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, self.num_points_per_level,
                    xs, ys, radius=self.radius)
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            target_inds_per_im = locations_to_gt_inds + num_targets
            num_targets += len(targets_per_im)

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            target_inds.append(target_inds_per_im)

        return labels, reg_targets, target_inds

    def losses(self):
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """

        gt_class, gt_deltas, gt_inds = self._get_ground_truth()
        """
        gt_class_logits: list of N tensors. Tensor i is a vector whose length is the
            total number of anchors in image i (i.e., len(anchors[i]))
        gt_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), B),
            where B is the box dimension
        """

        # Collect all objectness logits and delta predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W, A from slowest to fastest axis.
        pred_class_logits = cat(
            [
                # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
                x.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
                for x in self.pred_class_logits
            ], dim=0,)
        pred_deltas = cat(
            [
                # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
                x.permute(0, 2, 3, 1).reshape(-1, 4)
                for x in self.pred_deltas
            ], dim=0,)
        pred_centerness = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1) for x in self.pred_centerness
            ], dim=0,)

        gt_class = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1) for x in gt_class
            ], dim=0,)

        gt_inds = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1) for x in gt_inds
            ], dim=0,)

        gt_deltas = cat(
            [
                # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
                x.reshape(-1, 4) for x in gt_deltas
            ], dim=0,)

        return fcos_losses(
            gt_class,
            gt_deltas,
            gt_inds,
            pred_class_logits,
            pred_deltas,
            pred_centerness,
            self.focal_loss_alpha,
            self.focal_loss_gamma,
            self.iou_loss
        )

    def predict_proposals(self, top_feats):
        sampled_boxes = []
        if len(top_feats) > 0:
            for i, (l, o, d, c, s, t) in enumerate(zip(
                    self.locations, self.pred_class_logits,
                    self.pred_deltas, self.pred_centerness,
                    self.strides, top_feats)):
                sampled_boxes.append(
                    self.forward_for_single_feature_map(
                        l, o, d * s, c, self.image_sizes, t))
        else:
            for i, (l, o, d, c, s) in enumerate(zip(
                    self.locations, self.pred_class_logits,
                    self.pred_deltas, self.pred_centerness,
                    self.strides)):
                sampled_boxes.append(
                    self.forward_for_single_feature_map(
                        l, o, d * s, c, self.image_sizes))

        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)
        return boxlists

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            image_sizes, top_feat=None):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()
        if top_feat is not None:
            top_feat = top_feat.view(N, -1, H, W).permute(0, 2, 3, 1)
            top_feat = top_feat.reshape(N, H * W, -1)

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]
            if top_feat is not None:
                per_top_feat = top_feat[i]
                per_top_feat = per_top_feat[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                per_box_loc = per_box_loc[top_k_indices]
                if top_feat is not None:
                    per_top_feat = per_top_feat[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            boxlist = Instances(image_sizes[i])
            boxlist.pred_boxes = Boxes(detections)
            boxlist.scores = per_box_cls
            boxlist.pred_classes = per_class
            if top_feat is not None:
                boxlist.top_feat = per_top_feat
            results.append(boxlist)

        return results

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = Instances.ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results
