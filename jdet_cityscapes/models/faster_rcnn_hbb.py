import jittor as jt
from jittor import nn
import numpy as np
from PIL import Image

from jdet.utils.registry import BOXES, MODELS, build_from_cfg, BACKBONES, HEADS, NECKS, ROI_EXTRACTORS
from jdet.ops.bbox_transforms import bbox2roi
from jdet.utils.general import parse_losses
from jdet.models.boxes.iou_calculator import bbox_overlaps_np


def generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy=None,
                              scales=(0.01, 0.2), ratios=(0.3, 1/0.3),
                              max_iters=100, eps=1e-6, iou_max=0.7, iou_min=0.0):
    if isinstance(num_bboxes, (tuple, list)):
        num_bboxes = np.random.randint(num_bboxes[0], num_bboxes[1] + 1)
    img_w, img_h = img_size
    random_bboxes_xy = np.zeros((num_bboxes, 4), dtype=np.float32)
    total = 0
    for _ in range(max_iters):
        if total >= num_bboxes:
            break
        x1, y1 = np.random.randint(0, img_w), np.random.randint(0, img_h)
        scale = np.random.uniform(*scales) * img_h * img_w
        ratio = np.random.uniform(*ratios)
        bbox_w, bbox_h = int(np.sqrt(scale / ratio)), int(np.sqrt(scale * ratio))
        x2, y2 = min(x1 + bbox_w, img_w), min(y1 + bbox_h, img_h)
        random_bbox = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        if bboxes_xy is not None and len(bboxes_xy) > 0:
            ious = bbox_overlaps_np(random_bbox, bboxes_xy)
            if np.max(ious) > iou_max:
                continue
        random_bboxes_xy[total, :] = random_bbox[0]
        total += 1
    if total != num_bboxes:
        random_bboxes_xy = random_bboxes_xy[:total, :]
    return random_bboxes_xy


@MODELS.register_module()
class FasterRCNNHBB(nn.Module):
    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super().__init__()
        self.backbone = build_from_cfg(backbone, BACKBONES)
        self.neck = build_from_cfg(neck, NECKS)
        self.shared_head = build_from_cfg(shared_head, NECKS)
        self.rpn_head = build_from_cfg(rpn_head, HEADS)
        self.bbox_roi_extractor = build_from_cfg(bbox_roi_extractor, ROI_EXTRACTORS)
        self.bbox_head = build_from_cfg(bbox_head, HEADS)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

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
                img2 = np.array(img2).transpose((2, 0, 1))
            else:
                if img2.ndim == 3 and img2.shape[-1] in (1, 3):
                    img2 = img2.transpose((2, 0, 1))
            imgs2.append(img2.astype(np.float32))
            t2 = dict(t)
            t2['bboxes'] = t.get('bboxes2', t.get('bboxes'))
            t2['labels'] = t.get('labels2', t.get('labels'))
            t2.pop('img2', None)
            t2.pop('bboxes2', None)
            t2.pop('labels2', None)
            targets2.append(t2)

        if len(imgs2) != len(targets):
            return images, targets, 1, images.shape[0]

        images2 = jt.array(np.stack(imgs2))
        images_all = jt.concat([images, images2], dim=0)
        targets_all = targets + targets2
        return images_all, targets_all, 2, images.shape[0]

    def _get_random_proposal_list(self, img_shape, gt_bboxes, targets, num_views):
        cfg = self.train_cfg.get('random_proposal_cfg', None)
        if cfg is None:
            return None
        num_imgs = len(gt_bboxes)
        batch_size = num_imgs // num_views
        random_list = [None for _ in range(num_imgs)]
        for i in range(num_imgs):
            base_idx = i % batch_size
            base_t = targets[base_idx]
            base_gt = gt_bboxes[base_idx]
            boxes_all = []
            multilevel = base_t.get('multilevel_boxes', None)
            if multilevel is not None and len(multilevel) > 0:
                ious = bbox_overlaps_np(multilevel, base_gt)
                mask_np = np.max(ious, axis=1) < cfg.get('iou_max', 0.7)
                if isinstance(multilevel, jt.Var):
                    keep = multilevel[jt.array(mask_np)]
                else:
                    keep = multilevel[mask_np]
                if len(keep) > 0:
                    boxes_all.append(keep)
            oamix_boxes = base_t.get('oamix_boxes', None)
            if oamix_boxes is not None and len(oamix_boxes) > 0:
                ious = bbox_overlaps_np(oamix_boxes, base_gt)
                mask_np = np.max(ious, axis=1) < cfg.get('iou_max', 0.7)
                if isinstance(oamix_boxes, jt.Var):
                    keep = oamix_boxes[jt.array(mask_np)]
                else:
                    keep = oamix_boxes[mask_np]
                if len(keep) > 0:
                    boxes_all.append(keep)

            random_bg = generate_random_bboxes_xy(
                img_shape[::-1],
                num_bboxes=cfg.get('num_bboxes', 10),
                bboxes_xy=base_gt,
                scales=cfg.get('scales', (0.01, 0.3)),
                ratios=cfg.get('ratios', (0.3, 1/0.3)),
                iou_max=cfg.get('iou_max', 0.7),
                iou_min=cfg.get('iou_min', 0.0),
            )
            boxes_all.append(random_bg)

            if len(boxes_all) > 0:
                boxes_np = np.concatenate(boxes_all, axis=0).astype(np.float32)
            else:
                boxes_np = np.zeros((0, 4), dtype=np.float32)
            random_list[i] = jt.array(boxes_np)
        return random_list

    def execute_train(self, images, targets=None):
        # print(">>> Model start forward...") # 加入这行
        self.backbone.train()

        images, targets, num_views, batch_size = self._stack_views(images, targets)

        image_meta = []
        gt_labels = []
        gt_bboxes = []
        gt_bboxes_ignore = []
        for target in targets:
            img_size = target['img_size']
            ori_size = target['ori_img_size']
            pad_size = target.get('pad_shape', img_size)
            meta = dict(
                ori_shape=(ori_size[1], ori_size[0]),
                img_shape=(img_size[1], img_size[0]),
                pad_shape=(pad_size[1], pad_size[0]),
                img_file=target.get('img_file', target.get('filename', '')),
                to_bgr=target.get('to_bgr', True),
                scale_factor=target.get('scale_factor', 1.0)
            )
            image_meta.append(meta)
            gt_bboxes.append(target['bboxes'])
            gt_labels.append(target['labels'])
            gt_bboxes_ignore.append(target.get('bboxes_ignore', None))

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
            rpn_losses = self.rpn_head.loss(*rpn_outs, gt_bboxes, image_meta,
                                            self.train_cfg.rpn, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*rpn_outs, image_meta, proposal_cfg)
        else:
            rpn_outs = self.rpn_head(features)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, image_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
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
                    proposal_list[:batch_size], gt_bboxes[:batch_size], gt_bboxes_ignore[:batch_size], gt_labels[:batch_size]):
                if isinstance(proposal, (list, tuple)):
                    proposal = jt.contrib.concat(proposal, dim=0)
                assign_result = bbox_assigner.assign(
                    proposal[:, :4], gt_bbox, gt_bbox_ignore, gt_label
                )
                sampling_result = bbox_sampler.sample(
                    assign_result, proposal, gt_bbox, gt_label
                )
                sampling_results_batch.append(sampling_result)
            for _ in range(num_views):
                sampling_results.extend(sampling_results_batch)
        else:
            for proposal, gt_bbox, gt_bbox_ignore, gt_label in zip(proposal_list, gt_bboxes, gt_bboxes_ignore, gt_labels):
                if isinstance(proposal, (list, tuple)):
                    proposal = jt.contrib.concat(proposal, dim=0)
                assign_result = bbox_assigner.assign(
                    proposal[:, :4], gt_bbox, gt_bbox_ignore, gt_label
                )
                sampling_result = bbox_sampler.sample(
                    assign_result, proposal, gt_bbox, gt_label
                )
                sampling_results.append(sampling_result)

        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_feats = self.bbox_roi_extractor(
            features[:self.bbox_roi_extractor.num_inputs], rois)
        head_out = self.bbox_head(bbox_feats)
        if isinstance(head_out, (list, tuple)) and len(head_out) == 3:
            cls_score, bbox_pred, cont_feats = head_out
        else:
            cls_score, bbox_pred = head_out
            cont_feats = None

        random_proposal_list = None
        extra_cont = 0
        if self.train_cfg is not None and 'random_proposal_cfg' in self.train_cfg:
            img_shape = (targets[0]['img_size'][1], targets[0]['img_size'][0])
            random_proposal_list = self._get_random_proposal_list(img_shape, gt_bboxes, targets, num_views)
            if random_proposal_list is not None and cont_feats is not None:
                rois2 = bbox2roi([rp[:, :4] for rp in random_proposal_list])
                bbox_feats2 = self.bbox_roi_extractor(
                    features[:self.bbox_roi_extractor.num_inputs], rois2)
                head_out2 = self.bbox_head(bbox_feats2)
                if isinstance(head_out2, (list, tuple)) and len(head_out2) == 3:
                    _, _, cont_feats2 = head_out2
                    cont_feats = jt.contrib.concat([cont_feats, cont_feats2], dim=0)
                    extra_cont = cont_feats2.shape[0]

        bbox_targets = self.bbox_head.get_target(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
        labels, label_weights, bbox_t, bbox_w = bbox_targets
        if cont_feats is not None:
            labels_cont = labels
            if extra_cont > 0:
                pad = jt.zeros((extra_cont,), dtype=labels.dtype)
                labels_cont = jt.contrib.concat([labels, pad], dim=0)
            loss_bbox = self.bbox_head.loss(
                cls_score, bbox_pred, cont_feats,
                labels, label_weights, bbox_t, bbox_w,
                labels_cont=labels_cont
            )
        else:
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, labels, label_weights, bbox_t, bbox_w)
        losses.update(loss_bbox)
        return losses

    def execute_test(self, images, targets=None, rescale=True):
        img_meta = []
        img_shape = []
        scale_factor = []
        for target in targets:
            ori_img_size = target['ori_img_size']
            img_size = target.get('img_size', ori_img_size)
            pad_shape = target.get('pad_shape', img_size)
            # meta uses (h, w)
            meta = dict(
                ori_shape=(ori_img_size[1], ori_img_size[0]),
                img_shape=(img_size[1], img_size[0]),
                pad_shape=(pad_shape[1], pad_shape[0]),
                scale_factor=target.get('scale_factor', 1.0),
                img_file=target.get('img_file', target.get('filename', ''))
            )
            img_meta.append(meta)
            img_shape.append((img_size[1], img_size[0]))
            scale_factor.append(target.get('scale_factor', 1.0))

        x = self.backbone(images)
        if self.neck:
            x = self.neck(x)

        if self.rpn_head.__class__.__name__ == "RPNHead":
            rpn_targets = []
            for t in targets:
                rt = dict(t)
                rt["hboxes"] = t["bboxes"]
                rt["hboxes_ignore"] = t.get("bboxes_ignore", None)
                rpn_targets.append(rt)
            proposal_list, _ = self.rpn_head(x, rpn_targets)
        elif self.rpn_head.__class__.__name__ in ("FasterrcnnHead", "FasterrcnnHeadFixed"):
            rpn_outs = self.rpn_head(x)
            proposal_list = self.rpn_head.get_bboxes(*rpn_outs, img_meta, self.test_cfg.rpn)
        else:
            rpn_outs = self.rpn_head(x)
            proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)

        if len(proposal_list) > 0 and isinstance(proposal_list[0], (list, tuple)):
            proposal_list = [jt.contrib.concat(p, dim=0) for p in proposal_list]
        rois = bbox2roi(proposal_list)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        head_out = self.bbox_head(roi_feats)
        if isinstance(head_out, (list, tuple)) and len(head_out) == 3:
            cls_score, bbox_pred, _ = head_out
        else:
            cls_score, bbox_pred = head_out

        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape[0],
            scale_factor[0],
            rescale=rescale,
            cfg=self.test_cfg.rcnn)

        if det_bboxes.numel() == 0:
            det_bboxes = jt.zeros((0, 5))
            det_labels = jt.zeros((0,), dtype=jt.int32)

        result = dict(
            img_id=targets[0].get("img_id", 0),
            boxes=det_bboxes[:, :4],
            scores=det_bboxes[:, 4],
            labels=det_labels,
        )
        return [result]

    def execute(self, images, targets=None):
        if self.is_training():
            return self.execute_train(images, targets)
        else:
            return self.execute_test(images, targets)


@MODELS.register_module()
class FasterRCNNHBBRescale(FasterRCNNHBB):
    def execute_test(self, images, targets=None, rescale=True):
        return super().execute_test(images, targets, rescale=True)
