import numpy as np
import jittor as jt
import jittor.nn as nn

from jdet.utils.registry import LOSSES
from jdet.models.losses.cross_entropy_loss import cross_entropy_loss, binary_cross_entropy_loss
from jdet.models.losses.smooth_l1_loss import smooth_l1_loss
from jdet.models.losses.l1_loss import l1_loss


def _split_views(x, num_views):
    if num_views <= 1:
        return [x]
    n = x.shape[0] // num_views
    return [x[i * n:(i + 1) * n] for i in range(num_views)]


def _reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        avg_factor = max(loss.shape[0], 1)
    if reduction == 'mean':
        loss = loss.sum() / avg_factor
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def _eye(n, dtype=jt.float32):
    # Jittor does not always expose jt.eye; use numpy fallback.
    return jt.array(np.eye(n, dtype=np.float32)).cast(dtype)


def _jsd_from_logits(pred_list, use_sigmoid=False, eps=1e-7):
    if len(pred_list) < 2:
        return jt.array(0.0)
    probs = []
    for pred in pred_list:
        if use_sigmoid:
            p = jt.sigmoid(pred)
            if p.ndim == 1:
                p = p.reshape((-1, 1))
            # binary -> 2-class distribution
            p = jt.concat([p, 1.0 - p], dim=1)
        else:
            p = nn.softmax(pred, dim=1)
        probs.append(p)
    mixture = jt.zeros_like(probs[0])
    for p in probs:
        mixture += p
    mixture = mixture / float(len(probs))
    mixture = jt.clamp(mixture, eps, 1.0)
    log_m = jt.log(mixture)
    jsd = 0.0
    for p in probs:
        p = jt.clamp(p, eps, 1.0)
        log_p = jt.log(p)
        kl = jt.sum(p * (log_p - log_m), dim=1)
        jsd = jsd + kl
    jsd = jsd / float(len(probs))
    return jsd.mean()


def _normalize(feats, eps=1e-8):
    norm = jt.sqrt((feats * feats).sum(dim=1, keepdims=True) + eps)
    return feats / norm


def _supcontrast(features, labels, num_views=2, temper=0.07, min_samples=10):
    # labels: (N, 1) or (N,)
    if features.numel() == 0:
        return jt.array(0.0)
    if labels.ndim == 1:
        labels = labels.reshape((-1, 1))
    labels = labels.stop_grad()
    device = features

    batch_size = features.shape[0]
    ori_size = batch_size // num_views if num_views > 0 else batch_size
    rp_total_size = batch_size - ori_size * num_views
    rp_size = rp_total_size // num_views if num_views > 0 else 0

    # foreground/background masks (bg = 0)
    fg = (labels != 0).float()
    bg = (labels == 0).float()

    # mask for same instance across views (bg only)
    mask_same_instance = jt.zeros((batch_size, batch_size), dtype=jt.float32)
    if ori_size > 0 and num_views >= 2:
        eye_ori = _eye(ori_size, dtype=jt.float32)
        mask_same_instance[:ori_size, ori_size:ori_size * 2] = eye_ori
        mask_same_instance[ori_size:ori_size * 2, :ori_size] = eye_ori
    if rp_size > 0 and num_views >= 2:
        start = ori_size * num_views
        eye_rp = _eye(rp_size, dtype=jt.float32)
        mask_same_instance[start:start + rp_size, start + rp_size:start + rp_size * 2] = eye_rp
        mask_same_instance[start + rp_size:start + rp_size * 2, start:start + rp_size] = eye_rp

    mask_bg = jt.matmul(bg, bg.transpose(0, 1))
    mask_anchor_bg = mask_same_instance * mask_bg

    fg_count = int((labels != 0).sum().item())
    if fg_count <= min_samples:
        # Keep a tiny gradient path for the contrastive head on sparse-foreground batches.
        return (features * features).mean() * 1e-6

    mask_fg = jt.matmul(fg, fg.transpose(0, 1))
    mask_eye = _eye(batch_size, dtype=jt.float32)
    mask_anchor = (labels == labels.transpose(0, 1)).float()
    mask_anchor_except_eye = mask_anchor - mask_eye
    mask_anchor_fg = mask_anchor_except_eye * mask_fg
    mask_anchor = (mask_anchor_fg + mask_anchor_bg).stop_grad()

    mask_contrast = (jt.ones((batch_size, batch_size)) - mask_eye).stop_grad()

    feats = _normalize(features)
    logits = jt.matmul(feats, feats.transpose(0, 1)) / temper
    logits_max = jt.max(logits, dim=1, keepdims=True)
    logits = logits - logits_max

    exp_logits = jt.exp(logits) * mask_contrast
    log_prob = logits - jt.log(exp_logits.sum(dim=1, keepdims=True) + 1e-12)
    pos_per_anchor = mask_anchor.sum(dim=1)
    valid = (pos_per_anchor > 0).float()
    if int(valid.sum().item()) == 0:
        return (features * features).mean() * 1e-6
    mean_log_prob_pos = (mask_anchor * log_prob).sum(dim=1) / (pos_per_anchor + 1e-8)
    loss = -(mean_log_prob_pos * valid).sum() / jt.maximum(valid.sum(), jt.array(1.0, dtype=valid.dtype))
    return loss


@LOSSES.register_module()
class CrossEntropyLossPlus(nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 loss_weight=1.0,
                 reduction='mean',
                 additional_loss=None,
                 lambda_weight=0.0,
                 num_views=2,
                 **kwargs):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.additional_loss = additional_loss
        self.lambda_weight = lambda_weight
        self.num_views = num_views

    def execute(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        reduction = reduction_override if reduction_override else self.reduction
        if avg_factor is not None and self.num_views > 1:
            avg_factor = avg_factor / float(self.num_views)
        pred_chunks = _split_views(pred, self.num_views)
        target_chunks = _split_views(target, self.num_views)
        weight_chunks = _split_views(weight, self.num_views) if weight is not None else None

        pred0 = pred_chunks[0]
        target0 = target_chunks[0]
        weight0 = weight_chunks[0] if weight_chunks is not None else None

        if self.use_sigmoid:
            if pred0.ndim != target0.ndim:
                if pred0.ndim == 2 and pred0.shape[1] == 1:
                    target0 = target0.reshape((-1, 1))
                    if weight0 is not None and weight0.ndim == 1:
                        weight0 = weight0.reshape((-1, 1))
                else:
                    # expand labels/weights to match pred channels
                    if target0.ndim == 1:
                        n, c = pred0.shape[0], pred0.shape[1]
                        tgt = jt.zeros((n, c), dtype=pred0.dtype)
                        inds = jt.nonzero((target0 >= 1) & (target0 <= c)).squeeze(-1)
                        if inds.numel() > 0:
                            tgt[inds, target0[inds] - 1] = 1
                        target0 = tgt
                    if weight0 is not None and weight0.ndim == 1:
                        weight0 = weight0.reshape((-1, 1)) * jt.ones((1, pred0.shape[1]))
            elif weight0 is not None and weight0.ndim == 1 and pred0.ndim == 2:
                if pred0.shape[1] == 1:
                    weight0 = weight0.reshape((-1, 1))
                else:
                    weight0 = weight0.reshape((-1, 1)) * jt.ones((1, pred0.shape[1]))
            loss_main = binary_cross_entropy_loss(
                pred0, target0, weight0, avg_factor=avg_factor, reduction=reduction
            )
        else:
            loss_main = cross_entropy_loss(
                pred0, target0, weight0, avg_factor=avg_factor, reduction=reduction
            )

        loss = self.loss_weight * loss_main

        if self.additional_loss in ('jsdv1_3', 'jsdv1_3_2aug', 'jsd') and self.num_views > 1:
            jsd = _jsd_from_logits(pred_chunks, use_sigmoid=self.use_sigmoid)
            loss = loss + self.lambda_weight * jsd

        return loss


@LOSSES.register_module()
class SmoothL1LossPlus(nn.Module):
    def __init__(self, beta=1.0, loss_weight=1.0, reduction='mean', num_views=2, **kwargs):
        super().__init__()
        self.beta = beta
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.num_views = num_views

    def execute(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        reduction = reduction_override if reduction_override else self.reduction
        if avg_factor is not None and self.num_views > 1:
            avg_factor = avg_factor / float(self.num_views)
        pred0 = _split_views(pred, self.num_views)[0]
        target0 = _split_views(target, self.num_views)[0]
        weight0 = _split_views(weight, self.num_views)[0] if weight is not None else None
        loss = smooth_l1_loss(pred0, target0, weight0, beta=self.beta, avg_factor=avg_factor, reduction=reduction)
        return self.loss_weight * loss


@LOSSES.register_module()
class L1LossPlus(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', num_views=2, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.num_views = num_views

    def execute(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        reduction = reduction_override if reduction_override else self.reduction
        if avg_factor is not None and self.num_views > 1:
            avg_factor = avg_factor / float(self.num_views)
        pred0 = _split_views(pred, self.num_views)[0]
        target0 = _split_views(target, self.num_views)[0]
        weight0 = _split_views(weight, self.num_views)[0] if weight is not None else None
        loss = l1_loss(pred0, target0, weight0, avg_factor=avg_factor, reduction=reduction)
        return self.loss_weight * loss


@LOSSES.register_module()
class ContrastiveLossPlus(nn.Module):
    def __init__(self, loss_weight=1.0, temperature=0.07, num_views=2, min_samples=10, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.num_views = num_views
        self.min_samples = min_samples

    def execute(self, cont_feats, labels):
        if cont_feats is None or cont_feats.numel() == 0:
            return jt.array(0.0)
        loss = _supcontrast(cont_feats, labels, num_views=self.num_views, temper=self.temperature, min_samples=self.min_samples)
        return self.loss_weight * loss
