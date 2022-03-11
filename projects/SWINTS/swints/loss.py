import torch
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit

from .util import box_ops
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from scipy.optimize import linear_sum_assignment


class SetCriterion(nn.Module):
    def __init__(self, cfg, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.cfg = cfg

        self.focal_loss_alpha = cfg.MODEL.SWINTS.ALPHA
        self.focal_loss_gamma = cfg.MODEL.SWINTS.GAMMA

    def loss_labels(self, outputs, targets, indices, num_boxes, mask_encoding):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.flatten(0, 1)

        target_classes = target_classes.flatten(0, 1)
        pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
        labels = torch.zeros_like(src_logits)
        labels[pos_inds, target_classes[pos_inds]] = 1

        class_loss = sigmoid_focal_loss_jit(
            src_logits,
            labels,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_boxes
        losses = {'loss_ce': class_loss}

        return losses


    def loss_boxes(self, outputs, targets, indices, num_boxes, mask_encoding):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes_xyxy'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_boxes, target_boxes))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        image_size = torch.cat([v["image_size_xyxy_tgt"] for v in targets])
        src_boxes_ = src_boxes / image_size
        target_boxes_ = target_boxes / image_size

        loss_bbox = F.l1_loss(src_boxes_, target_boxes_, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes, mask_encoding):
        assert 'pred_masks' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_masks_feat = outputs['pred_masks'][idx]
        target_masks = torch.cat([t['gt_masks'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        mask_loss_func = nn.MSELoss(reduction="none")

        target_masks_feat = mask_encoding.encoder(target_masks.flatten(1))
        loss = mask_loss_func(src_masks_feat, target_masks_feat)
        
        losses = {}
        losses['loss_feat'] =  loss.sum() / num_boxes / self.cfg.MODEL.SWINTS.MASK_DIM

        eps = 1e-5
        src_masks = mask_encoding.decoder(src_masks_feat.flatten(1))
        n_inst = src_masks.size(0)
        target_masks = target_masks.flatten(1)
        intersection = (src_masks * target_masks).sum(dim=1)
        union = (src_masks ** 2.0).sum(dim=1) + (target_masks ** 2.0).sum(dim=1) + eps
        loss = 1. - (2 * intersection / union)
        losses['loss_dice'] = loss.sum() / num_boxes

        return losses
    def loss_rec(self, outputs, targets, indices, num_boxes, mask_encoding):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_rec = outputs['pred_rec']
        losses = {}
        losses['loss_rec'] = src_rec
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, mask_encoding, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'rec': self.loss_rec
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, mask_encoding, **kwargs)

    def forward(self, outputs, targets, mask_encoding):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, mask_encoding)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, mask_encoding))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, mask_encoding)
                for loss in self.losses:
                    # if loss == 'masks':
                    #     # Intermediate masks losses are too costly to compute, we ignore them.
                    #     continue
                    if loss == 'rec':
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, mask_encoding, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses



class HungarianMatcher(nn.Module):
    def __init__(self, cfg, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_mask: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_mask = cost_mask
        self.focal_loss_alpha = cfg.MODEL.SWINTS.ALPHA
        self.focal_loss_gamma = cfg.MODEL.SWINTS.GAMMA
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, mask_encoding):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes_xyxy"] for v in targets])

        
        alpha = self.focal_loss_alpha
        gamma = self.focal_loss_gamma
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        image_size_out = torch.cat([v["image_size_xyxy"].unsqueeze(0) for v in targets])
        image_size_out = image_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)
        image_size_tgt = torch.cat([v["image_size_xyxy_tgt"] for v in targets])

        out_bbox_ = out_bbox / image_size_out
        tgt_bbox_ = tgt_bbox / image_size_tgt
        cost_bbox = torch.cdist(out_bbox_, tgt_bbox_, p=1)

        # Compute the giou cost betwen boxes
        # cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

        # mask loss
        tgt_mask = torch.cat([v["gt_masks"] for v in targets]).flatten(1)
        tgt_mask_feat = mask_encoding.encoder(tgt_mask)
        out_mask_feat = outputs["pred_masks"].flatten(0, 1).flatten(1)

        tgt_mask_feat = nn.functional.normalize(tgt_mask_feat, p=2)
        out_mask_feat = nn.functional.normalize(out_mask_feat, p=2)
        
        # cost_mask = -torch.mm(out_mask, tgt_mask.T)
        cost_mask = -(torch.mm(out_mask_feat, tgt_mask_feat.T) + 1.0) / 2.0

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + self.cost_mask * cost_mask
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
