import numpy as np
from PIL import Image
import jittor as jt
from contextlib import nullcontext

from jdet.utils.registry import MODELS, build_from_cfg, BOXES
from jdet.ops.bbox_transforms import bbox2roi

from .faster_rcnn_hbb import FasterRCNNHBB
from .fdd_modules import supcon_one_side, supcon_two_side, FDDProjector


@MODELS.register_module()
class FasterRCNNHBBFDD(FasterRCNNHBB):
    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 fdd_cfg=None,
                 fdd_loss_cfg=None,
                 fdd_detach=True,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super().__init__(
            backbone=backbone,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            neck=neck,
            shared_head=shared_head,
            pretrained=pretrained,
        )
        if fdd_cfg is None:
            raise ValueError("fdd_cfg must be provided for FasterRCNNHBBFDD.")
        self.fdd = build_from_cfg(fdd_cfg, MODELS)
        self.fdd_detach = fdd_detach
        fdd_loss_cfg = fdd_loss_cfg or {}
        self.fdd_temperature = fdd_loss_cfg.get("temperature", 0.07)
        self.fdd_loss_weight_img = fdd_loss_cfg.get("loss_weight_img", 1.0)
        self.fdd_loss_weight_ins = fdd_loss_cfg.get("loss_weight_ins", 1.0)
        self.fdd_loss_weight_reg = fdd_loss_cfg.get("loss_weight_reg", 0.0)
        self.fdd_loss_ins_cap = float(fdd_loss_cfg.get("loss_ins_cap", 0.0))
        self.fdd_two_side = fdd_loss_cfg.get("two_side", True)
        self.fdd_use_instance = fdd_loss_cfg.get("use_instance", True)
        self.fdd_ins_from_input = fdd_loss_cfg.get("ins_from_input", True)
        self.fdd_ins_max_rois = int(fdd_loss_cfg.get("ins_max_rois", 512))
        self.fdd_img_from_input = fdd_loss_cfg.get("img_from_input", True)
        proj_dim = fdd_loss_cfg.get("proj_dim", 128)
        proj_in_channels = fdd_loss_cfg.get("proj_in_channels", 256)
        img_proj_in_channels = fdd_loss_cfg.get("img_proj_in_channels", 3)

        need_feat_proj = (
            ((self.fdd_use_instance and self.fdd_loss_weight_ins > 0) and (not self.fdd_ins_from_input))
            or (not self.fdd_img_from_input)
        )
        self.fdd_proj = FDDProjector(proj_in_channels, proj_dim) if need_feat_proj else None
        self.fdd_img_proj = (
            FDDProjector(img_proj_in_channels, proj_dim)
            if ((self.fdd_loss_weight_img > 0 and self.fdd_img_from_input)
                or (self.fdd_use_instance and self.fdd_loss_weight_ins > 0 and self.fdd_ins_from_input))
            else None
        )

    def _stack_views(self, images, targets):
        if targets is None or len(targets) == 0:
            return images, targets, 1, images.shape[0]
        if 'img2' not in targets[0]:
            return images, targets, 1, images.shape[0]

        imgs2 = []
        targets2 = []
        for t in targets:
            img2 = t.get('img2', None)
            if img2 is None:
                continue
            if isinstance(img2, Image.Image):
                arr = np.asarray(img2)
                if arr.ndim == 3 and arr.shape[-1] in (1, 3):
                    arr = arr.transpose((2, 0, 1))
                if arr.ndim == 2:
                    arr = arr[None, ...]
                if arr.shape[0] == 1:
                    arr = np.repeat(arr, 3, axis=0)
                img2_var = jt.array(arr.astype(np.float32, copy=False))
            elif isinstance(img2, jt.Var):
                img2_var = img2
                if img2_var.ndim == 3 and int(img2_var.shape[-1]) in (1, 3):
                    img2_var = img2_var.transpose((2, 0, 1))
                if img2_var.ndim == 2:
                    img2_var = img2_var.unsqueeze(0)
                if int(img2_var.shape[0]) == 1:
                    img2_var = jt.contrib.concat([img2_var, img2_var, img2_var], dim=0)
                img2_var = img2_var.float32()
            else:
                arr = np.asarray(img2)
                if arr.ndim == 3 and arr.shape[-1] in (1, 3):
                    arr = arr.transpose((2, 0, 1))
                if arr.ndim == 2:
                    arr = arr[None, ...]
                if arr.shape[0] == 1:
                    arr = np.repeat(arr, 3, axis=0)
                img2_var = jt.array(arr.astype(np.float32, copy=False))
            imgs2.append(img2_var)
            t2 = dict(t)
            t2['bboxes'] = t.get('bboxes2', t.get('bboxes'))
            t2['labels'] = t.get('labels2', t.get('labels'))
            t2.pop('img2', None)
            t2.pop('bboxes2', None)
            t2.pop('labels2', None)
            targets2.append(t2)

        if len(imgs2) != len(targets):
            return images, targets, 1, images.shape[0]

        h_max, w_max = int(images.shape[-2]), int(images.shape[-1])
        images2 = jt.zeros((len(imgs2), 3, h_max, w_max), dtype=images.dtype)
        for i, arr in enumerate(imgs2):
            h = min(int(arr.shape[1]), h_max)
            w = min(int(arr.shape[2]), w_max)
            images2[i, :, :h, :w] = arr[:, :h, :w]

        images_all = jt.concat([images, images2], dim=0)
        targets_all = targets + targets2
        return images_all, targets_all, 2, images.shape[0]

    def _pool_rois_from_input(self, images, rois):
        if rois.numel() == 0:
            c = int(images.shape[1])
            return jt.zeros((0, c, 1, 1), dtype=images.dtype)
        n, c, h, w = [int(v) for v in images.shape]
        rois_np = rois.numpy()
        pooled = []
        for r in rois_np:
            b = int(r[0])
            if b < 0 or b >= n:
                continue
            x1 = max(0, min(w - 1, int(np.floor(r[1]))))
            y1 = max(0, min(h - 1, int(np.floor(r[2]))))
            x2 = max(x1 + 1, min(w, int(np.ceil(r[3]))))
            y2 = max(y1 + 1, min(h, int(np.ceil(r[4]))))
            crop = images[b:b + 1, :, y1:y2, x1:x2]
            feat = crop.mean(dim=3, keepdims=True).mean(dim=2, keepdims=True)
            pooled.append(feat)
        if len(pooled) == 0:
            return jt.zeros((0, c, 1, 1), dtype=images.dtype)
        return jt.contrib.concat(pooled, dim=0)

    def _forward_detector(self, images, targets, num_views, batch_size):
        image_meta = []
        gt_labels = []
        gt_bboxes = []
        gt_bboxes_ignore = []
        for target in targets:
            img_size = target["img_size"]
            ori_size = target["ori_img_size"]
            pad_size = target.get("pad_shape", img_size)
            meta = dict(
                ori_shape=(ori_size[1], ori_size[0]),
                img_shape=(img_size[1], img_size[0]),
                pad_shape=(pad_size[1], pad_size[0]),
                img_file=target.get("img_file", target.get("filename", "")),
                to_bgr=target.get("to_bgr", True),
                scale_factor=target.get("scale_factor", 1.0),
            )
            image_meta.append(meta)
            gt_bboxes.append(target["bboxes"])
            gt_labels.append(target["labels"])
            gt_bboxes_ignore.append(target.get("bboxes_ignore", None))

        losses = dict()
        features = self.backbone(images)
        if self.neck:
            features = self.neck(features)

        if self.rpn_head.__class__.__name__ == "RPNHead":
            rpn_targets = []
            for t in targets:
                rt = dict(t)
                rt["hboxes"] = t["bboxes"]
                rt["hboxes_ignore"] = t.get("bboxes_ignore", None)
                rpn_targets.append(rt)
            proposal_list, rpn_losses = self.rpn_head(features, rpn_targets)
            losses.update(rpn_losses)
        elif self.rpn_head.__class__.__name__ in ("FasterrcnnHead", "FasterrcnnHeadFixed"):
            rpn_outs = self.rpn_head(features)
            rpn_losses = self.rpn_head.loss(
                *rpn_outs, gt_bboxes, image_meta, self.train_cfg.rpn, gt_bboxes_ignore=gt_bboxes_ignore
            )
            losses.update(rpn_losses)
            proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*rpn_outs, image_meta, proposal_cfg)
        else:
            rpn_outs = self.rpn_head(features)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, image_meta, self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)
            proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (image_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)

        bbox_assigner = build_from_cfg(self.train_cfg.rcnn.assigner, BOXES)
        bbox_sampler = build_from_cfg(self.train_cfg.rcnn.sampler, BOXES)
        num_imgs = images.shape[0]
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []

        if num_views > 1:
            sampling_results_batch = []
            for proposal, gt_bbox, gt_bbox_ignore, gt_label in zip(
                proposal_list[:batch_size],
                gt_bboxes[:batch_size],
                gt_bboxes_ignore[:batch_size],
                gt_labels[:batch_size],
            ):
                if isinstance(proposal, (list, tuple)):
                    proposal = jt.contrib.concat(proposal, dim=0)
                assign_result = bbox_assigner.assign(proposal[:, :4], gt_bbox, gt_bbox_ignore, gt_label)
                sampling_result = bbox_sampler.sample(assign_result, proposal, gt_bbox, gt_label)
                sampling_results_batch.append(sampling_result)
            for _ in range(num_views):
                sampling_results.extend(sampling_results_batch)
        else:
            for proposal, gt_bbox, gt_bbox_ignore, gt_label in zip(
                proposal_list, gt_bboxes, gt_bboxes_ignore, gt_labels
            ):
                if isinstance(proposal, (list, tuple)):
                    proposal = jt.contrib.concat(proposal, dim=0)
                assign_result = bbox_assigner.assign(proposal[:, :4], gt_bbox, gt_bbox_ignore, gt_label)
                sampling_result = bbox_sampler.sample(assign_result, proposal, gt_bbox, gt_label)
                sampling_results.append(sampling_result)

        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_feats = self.bbox_roi_extractor(features[:self.bbox_roi_extractor.num_inputs], rois)
        head_out = self.bbox_head(bbox_feats)
        if isinstance(head_out, (list, tuple)) and len(head_out) == 3:
            cls_score, bbox_pred, cont_feats = head_out
        else:
            cls_score, bbox_pred = head_out
            cont_feats = None

        random_proposal_list = None
        extra_cont = 0
        if self.train_cfg is not None and "random_proposal_cfg" in self.train_cfg:
            img_shape = (targets[0]["img_size"][1], targets[0]["img_size"][0])
            random_proposal_list = self._get_random_proposal_list(img_shape, gt_bboxes, targets, num_views)
            if random_proposal_list is not None and cont_feats is not None:
                rois2 = bbox2roi([rp[:, :4] for rp in random_proposal_list])
                bbox_feats2 = self.bbox_roi_extractor(
                    features[:self.bbox_roi_extractor.num_inputs], rois2
                )
                head_out2 = self.bbox_head(bbox_feats2)
                if isinstance(head_out2, (list, tuple)) and len(head_out2) == 3:
                    _, _, cont_feats2 = head_out2
                    cont_feats = jt.contrib.concat([cont_feats, cont_feats2], dim=0)
                    extra_cont = cont_feats2.shape[0]

        bbox_targets = self.bbox_head.get_target(sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
        labels, label_weights, bbox_t, bbox_w = bbox_targets
        if cont_feats is not None:
            labels_cont = labels
            if extra_cont > 0:
                pad = jt.zeros((extra_cont,), dtype=labels.dtype)
                labels_cont = jt.contrib.concat([labels, pad], dim=0)
            loss_bbox = self.bbox_head.loss(
                cls_score, bbox_pred, cont_feats, labels, label_weights, bbox_t, bbox_w, labels_cont=labels_cont
            )
        else:
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, labels, label_weights, bbox_t, bbox_w)
        losses.update(loss_bbox)
        return losses

    def _compute_fdd_losses(self, inv_all, spe_all, targets, batch_size):
        losses = {}
        if self.fdd_loss_weight_img <= 0 and self.fdd_loss_weight_ins <= 0 and self.fdd_loss_weight_reg <= 0:
            return losses

        need_feat = (
            (self.fdd_loss_weight_img > 0 and (not self.fdd_img_from_input))
            or (self.fdd_use_instance and self.fdd_loss_weight_ins > 0 and (not self.fdd_ins_from_input))
        )
        feat_inv_list = None
        feat_spe_list = None
        if need_feat:
            feat_inv = self.backbone(inv_all)
            if self.neck:
                feat_inv = self.neck(feat_inv)
            feat_spe = self.backbone(spe_all)
            if self.neck:
                feat_spe = self.neck(feat_spe)
            feat_inv_list = feat_inv if isinstance(feat_inv, (list, tuple)) else [feat_inv]
            feat_spe_list = feat_spe if isinstance(feat_spe, (list, tuple)) else [feat_spe]

        if self.fdd_loss_weight_img > 0:
            if self.fdd_img_from_input:
                if self.fdd_img_proj is not None:
                    inv_vec = self.fdd_img_proj(inv_all)
                    spe_vec = self.fdd_img_proj(spe_all)
                else:
                    inv_vec = None
                    spe_vec = None
            else:
                if self.fdd_proj is not None:
                    if feat_inv_list is None or feat_spe_list is None:
                        feat_inv = self.backbone(inv_all)
                        if self.neck:
                            feat_inv = self.neck(feat_inv)
                        feat_spe = self.backbone(spe_all)
                        if self.neck:
                            feat_spe = self.neck(feat_spe)
                        feat_inv_list = feat_inv if isinstance(feat_inv, (list, tuple)) else [feat_inv]
                        feat_spe_list = feat_spe if isinstance(feat_spe, (list, tuple)) else [feat_spe]
                    inv_vec = self.fdd_proj(feat_inv_list[-1])
                    spe_vec = self.fdd_proj(feat_spe_list[-1])
                else:
                    inv_vec = None
                    spe_vec = None
            if inv_vec is None or spe_vec is None:
                loss_img = (inv_all.sum() + spe_all.sum()) * 0.0
            elif self.fdd_two_side:
                loss_img = supcon_two_side(inv_vec, spe_vec, self.fdd_temperature)
            else:
                loss_img = supcon_one_side(inv_vec, spe_vec, self.fdd_temperature)
            losses["loss_fdd_img"] = loss_img * self.fdd_loss_weight_img

        if self.fdd_use_instance and self.fdd_loss_weight_ins > 0:
            loss_ins = (inv_all.sum() + spe_all.sum()) * 0.0
            rois = bbox2roi([t["bboxes"] for t in targets])
            if rois.numel() > 0:
                roi_labels = jt.contrib.concat([t["labels"] for t in targets], dim=0)
                if self.fdd_ins_max_rois > 0 and int(rois.shape[0]) > self.fdd_ins_max_rois:
                    idx = np.random.choice(int(rois.shape[0]), self.fdd_ins_max_rois, replace=False).astype(np.int32)
                    idx_var = jt.array(idx)
                    rois = rois[idx_var]
                    roi_labels = roi_labels[idx_var]
                if self.fdd_ins_from_input:
                    if self.fdd_img_proj is not None:
                        roi_inv = self._pool_rois_from_input(inv_all, rois)
                        roi_spe = self._pool_rois_from_input(spe_all, rois)
                        inv_vec = self.fdd_img_proj(roi_inv)
                        spe_vec = self.fdd_img_proj(roi_spe)
                        if self.fdd_two_side:
                            loss_ins = supcon_two_side(
                                inv_vec, spe_vec, self.fdd_temperature, labels=roi_labels
                            )
                        else:
                            loss_ins = supcon_one_side(
                                inv_vec, spe_vec, self.fdd_temperature, labels=roi_labels
                            )
                elif self.fdd_proj is not None:
                    roi_inv = self.bbox_roi_extractor(
                        feat_inv_list[: self.bbox_roi_extractor.num_inputs], rois
                    )
                    roi_spe = self.bbox_roi_extractor(
                        feat_spe_list[: self.bbox_roi_extractor.num_inputs], rois
                    )
                    inv_vec = self.fdd_proj(roi_inv)
                    spe_vec = self.fdd_proj(roi_spe)
                    if self.fdd_two_side:
                        loss_ins = supcon_two_side(
                            inv_vec, spe_vec, self.fdd_temperature, labels=roi_labels
                        )
                    else:
                        loss_ins = supcon_one_side(
                            inv_vec, spe_vec, self.fdd_temperature, labels=roi_labels
                        )
            if self.fdd_loss_ins_cap > 0:
                cap = jt.array(float(self.fdd_loss_ins_cap), dtype=loss_ins.dtype)
                # Smooth cap keeps gradient while preventing runaway magnitude.
                loss_ins = cap * jt.tanh(loss_ins / cap)
            losses["loss_fdd_ins"] = loss_ins * self.fdd_loss_weight_ins
        if self.fdd_loss_weight_reg > 0:
            # Tiny always-on regularizer to keep a valid gradient path for FDD params.
            losses["loss_fdd_reg"] = (inv_all * inv_all).mean() * self.fdd_loss_weight_reg

        return losses

    def execute_train(self, images, targets=None):
        self.backbone.train()
        images_all, targets_all, num_views, batch_size = self._stack_views(images, targets)

        if self.fdd_detach:
            # Save memory: run detector on detached FDD output from a no-grad pass.
            no_grad_ctx = jt.no_grad() if hasattr(jt, "no_grad") else nullcontext()
            with no_grad_ctx:
                inv_det, _ = self.fdd(images_all)
            losses = self._forward_detector(inv_det, targets_all, num_views, batch_size)
            inv_all, spe_all = self.fdd(images_all)
        else:
            inv_all, spe_all = self.fdd(images_all)
            losses = self._forward_detector(inv_all, targets_all, num_views, batch_size)
        fdd_losses = self._compute_fdd_losses(inv_all, spe_all, targets_all, batch_size)
        losses.update(fdd_losses)
        return losses

    def execute_test(self, images, targets=None, rescale=True):
        inv, _ = self.fdd(images)
        return super().execute_test(inv, targets, rescale=rescale)
