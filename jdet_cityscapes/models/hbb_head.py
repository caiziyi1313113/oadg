import jittor as jt
import jittor.nn as nn

from jdet.utils.registry import HEADS, LOSSES, build_from_cfg
from jdet.utils.general import multi_apply
from jdet.ops.bbox_transforms import bbox2delta, delta2bbox
from jdet.ops.nms import multiclass_nms


def bbox_target_hbb(pos_bboxes_list,
                    neg_bboxes_list,
                    pos_assigned_gt_inds_list,
                    gt_bboxes_list,
                    pos_gt_labels_list,
                    cfg,
                    reg_classes=1,
                    target_means=[.0, .0, .0, .0],
                    target_stds=[1.0, 1.0, 1.0, 1.0],
                    concat=True):
    labels, label_weights, bbox_targets, bbox_weights = multi_apply(
        bbox_target_hbb_single,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_assigned_gt_inds_list,
        gt_bboxes_list,
        pos_gt_labels_list,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds)

    if concat:
        labels = jt.contrib.concat(labels, 0)
        label_weights = jt.contrib.concat(label_weights, 0)
        bbox_targets = jt.contrib.concat(bbox_targets, 0)
        bbox_weights = jt.contrib.concat(bbox_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights


def bbox_target_hbb_single(pos_bboxes,
                           neg_bboxes,
                           pos_assigned_gt_inds,
                           gt_bboxes,
                           pos_gt_labels,
                           cfg,
                           reg_classes=1,
                           target_means=[.0, .0, .0, .0],
                           target_stds=[1.0, 1.0, 1.0, 1.0]):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    labels = jt.zeros(num_samples, dtype=jt.int)
    label_weights = jt.zeros(num_samples)
    bbox_targets = jt.zeros((num_samples, 4))
    bbox_weights = jt.zeros((num_samples, 4))

    pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds]

    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg['pos_weight'] <= 0 else cfg['pos_weight']
        label_weights[:num_pos] = pos_weight
        pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes, target_means, target_stds)
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0

    return labels, label_weights, bbox_targets, bbox_weights


def accuracy(pred, target, topk=1):
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, 1, True, True)
    pred_label = pred_label.t()
    correct = pred_label.equal(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdims=True)
        correct_k *= 100.0 / pred.shape[0]
        res.append(correct_k)
    return res[0] if return_single else res


@HEADS.register_module()
class BBoxHeadHBB(nn.Module):
    """Simplest HBB RoI head with two fc layers for cls/reg."""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=9,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False,
                 with_module=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super().__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic

        self.loss_cls = build_from_cfg(loss_cls, LOSSES)
        self.loss_bbox = build_from_cfg(loss_bbox, LOSSES)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(roi_feat_size)
        else:
            if isinstance(self.roi_feat_size, int):
                in_channels *= (self.roi_feat_size * self.roi_feat_size)
            elif isinstance(self.roi_feat_size, tuple):
                in_channels *= (self.roi_feat_size[0] * self.roi_feat_size[1])
        if self.with_cls:
            self.fc_cls = nn.Linear(in_channels, num_classes)
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)

    def init_weights(self):
        if self.with_cls:
            nn.init.gauss_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.gauss_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    def execute(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def get_target(self, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target_hbb(
            pos_proposals,
            neg_proposals,
            pos_assigned_gt_inds,
            gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets

    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = nn.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means, self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:]

        if rescale:
            if isinstance(scale_factor, (list, tuple)) and len(scale_factor) == 4:
                scale = jt.array(scale_factor)
                bboxes = bboxes / scale
            else:
                bboxes = bboxes / scale_factor

        det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels

    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduce=True):
        losses = dict()
        if cls_score is not None:
            reduction_override = "mean" if reduce else "none"
            losses['loss_cls'] = self.loss_cls(
                cls_score, labels, label_weights, reduction_override=reduction_override)
            losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[pos_inds, labels[pos_inds]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0))
        return losses


@HEADS.register_module()
class ConvFCBBoxHeadHBB(BBoxHeadHBB):
    """HBB bbox head with shared fc layers (similar to mmdet Shared2FC)."""

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= (self.roi_feat_size * self.roi_feat_size)
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= (self.roi_feat_size * self.roi_feat_size)

        self.relu = nn.ReLU()
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 * self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    nn.Conv(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,))
            last_layer_dim = self.conv_out_channels
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                if isinstance(self.roi_feat_size, int):
                    last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
                elif isinstance(self.roi_feat_size, tuple):
                    last_layer_dim *= (self.roi_feat_size[0] * self.roi_feat_size[1])
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super().init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def execute(self, x):
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.ndim > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.ndim > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


@HEADS.register_module()
class SharedFCBBoxHeadHBB(ConvFCBBoxHeadHBB):
    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super().__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class Shared2FCContrastiveHeadHBB(ConvFCBBoxHeadHBB):
    """Shared 2FC head with contrastive branch (OA-Loss)."""

    def __init__(self,
                 with_cont=True,
                 cont_predictor_cfg=dict(num_linear=2, feat_channels=256, return_relu=True),
                 out_dim_cont=256,
                 loss_cont=dict(type='ContrastiveLossPlus', loss_weight=0.01, num_views=2, temperature=0.07),
                 num_fcs=2,
                 fc_out_channels=1024,
                 *args,
                 **kwargs):
        assert num_fcs >= 1
        super().__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        self.with_cont = with_cont
        self.cont_predictor_cfg = cont_predictor_cfg
        self.out_dim_cont = out_dim_cont
        self.loss_cont = build_from_cfg(loss_cont, LOSSES)
        if self.with_cont:
            self.fc_cont = self._add_linear_relu(in_channels=self.cls_last_dim,
                                                 **self.cont_predictor_cfg)

    def _add_linear_relu(self, num_linear, in_channels, feat_channels, return_relu=False):
        layers = []
        num_relu = num_linear if return_relu else num_linear - 1
        for i in range(num_linear):
            in_c = in_channels if i == 0 else feat_channels
            layers.append(nn.Linear(in_c, feat_channels))
            if i < num_relu - 1:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def execute(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        x_cls = x
        x_reg = x
        x_cont = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.ndim > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.ndim > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        self.cls_feats = x_cls

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        cont_feats = self.fc_cont(x_cont) if self.with_cont else None
        return cls_score, bbox_pred, cont_feats

    def loss(self,
             cls_score,
             bbox_pred,
             cont_feats,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduce=True,
             labels_cont=None):
        losses = dict()
        if cls_score is not None:
            reduction_override = "mean" if reduce else "none"
            losses['loss_cls'] = self.loss_cls(
                cls_score, labels, label_weights, reduction_override=reduction_override)
            losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[pos_inds, labels[pos_inds]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0))

        if cont_feats is not None:
            if labels_cont is None:
                labels_cont = labels
            if cont_feats.shape[0] != labels_cont.shape[0]:
                raise ValueError(
                    f"cont_feats and labels_cont length mismatch: "
                    f"{cont_feats.shape[0]} vs {labels_cont.shape[0]}"
                )
            labels_cont = labels_cont.view(-1, 1)
            loss_cont = self.loss_cont(cont_feats, labels_cont)
            losses['loss_cont'] = loss_cont
        return losses
