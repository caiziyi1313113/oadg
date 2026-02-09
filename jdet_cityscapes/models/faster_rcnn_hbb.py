import jittor as jt
from jittor import nn

from jdet.utils.registry import BOXES, MODELS, build_from_cfg, BACKBONES, HEADS, NECKS, ROI_EXTRACTORS
from jdet.ops.bbox_transforms import bbox2roi
from jdet.utils.general import parse_losses


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

    def execute_train(self, images, targets=None):
        # print(">>> Model start forward...") # 加入这行
        self.backbone.train()

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
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_targets = self.bbox_head.get_target(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
        loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
        losses.update(loss_bbox)
        return losses

    def execute_test(self, images, targets=None, rescale=True):
        img_meta = []
        img_shape = []
        scale_factor = []
        for target in targets:
            ori_img_size = target['ori_img_size']
            meta = dict(
                ori_shape=(ori_img_size[1], ori_img_size[0]),
                img_shape=(ori_img_size[1], ori_img_size[0]),
                pad_shape=(ori_img_size[1], ori_img_size[0]),
                scale_factor=target.get('scale_factor', 1.0),
                img_file=target.get('img_file', target.get('filename', ''))
            )
            img_meta.append(meta)
            img_shape.append((target['img_size'][1], target['img_size'][0]))
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
        cls_score, bbox_pred = self.bbox_head(roi_feats)

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
