import os
import numpy as np

from jdet.data.coco import COCODataset
from jdet.utils.registry import DATASETS
'''
把 COCO Cityscapes 标注读出来，过滤无效框，返回 image + anno dict

为 OBB 模型额外生成 rboxes（旋转框 targets），并在 transforms 后重新生成保证一致

把预测结果保存成 COCO json（即使模型输出 polygon，也能转成 HBB）用于评测
'''

@DATASETS.register_module()
class CityscapesCOCOOBBDataset(COCODataset):
    """COCO-style Cityscapes dataset that also provides rboxes for FasterRCNNOBB.

    It uses COCO annotations for HBB and derives rotated boxes with a fixed angle.
    """

    @staticmethod
    def _hbb_to_rbox(hboxes):
        if hboxes is None or len(hboxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        hboxes = hboxes.astype(np.float32)
        # Match hbb2obb_v2 in jdet.ops.bbox_transforms for target consistency.
        ex_h = hboxes[:, 2] - hboxes[:, 0] + 1.0
        ex_w = hboxes[:, 3] - hboxes[:, 1] + 1.0
        ctr_x = hboxes[:, 0] + 0.5 * (ex_h - 1.0)
        ctr_y = hboxes[:, 1] + 0.5 * (ex_w - 1.0)
        angle = -np.pi / 2.0
        angles = np.full_like(ctr_x, angle, dtype=np.float32)
        rboxes = np.stack([ctr_x, ctr_y, ex_w, ex_h, angles], axis=1)
        return rboxes.astype(np.float32)

    @staticmethod
    def _poly_to_xyxy(polys):
        polys = np.asarray(polys, dtype=np.float32).reshape(-1, 8)
        xs = polys[:, 0::2]
        ys = polys[:, 1::2]
        x1 = xs.min(axis=1)
        y1 = ys.min(axis=1)
        x2 = xs.max(axis=1)
        y2 = ys.max(axis=1)
        return np.stack([x1, y1, x2, y2], axis=1)

    def _read_ann_info(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info["file_name"])
        image = self._load_img(img_path)

        ann_info = self.coco.loadAnns(self.coco.getAnnIds(img_id))

        width, height = image.size
        assert width == img_info["width"] and height == img_info["height"], "image size is different from annotations"

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for ann in ann_info:
            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            inter_w = max(0, min(x1 + w, img_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info["height"]) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get("iscrowd", False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann["category_id"]])
                gt_masks_ann.append(ann.get("segmentation", None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int32)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int32)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            img_id=img_id,
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            classes=self.CLASSES,
            ori_img_size=(width, height),
            img_size=(width, height),
            scale_factor=1.0,
            img_file=img_path,
        )

        # Extra keys for FasterRCNNOBB
        ann["hboxes"] = gt_bboxes
        ann["hboxes_ignore"] = gt_bboxes_ignore
        ann["rboxes"] = self._hbb_to_rbox(gt_bboxes)
        ann["rboxes_ignore"] = self._hbb_to_rbox(gt_bboxes_ignore)

        return image, ann

    @staticmethod
    def _load_img(path):
        from PIL import Image
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        image, anno = self._read_ann_info(img_id)

        if self.transforms is not None:
            image, anno = self.transforms(image, anno)

        # Rebuild rboxes from hboxes after transforms to keep HBB geometry consistent.
        if anno is not None and "hboxes" in anno:
            anno["rboxes"] = self._hbb_to_rbox(anno.get("hboxes"))
            anno["rboxes_ignore"] = self._hbb_to_rbox(anno.get("hboxes_ignore"))

        return image, anno

    def save_results(self, results, save_file):
        """Convert detection results to COCO json style.

        Accepts either:
        - dict with keys: boxes, scores, labels
        - tuple/list: (polys, scores, labels) where polys is Nx8
        """
        json_results = []
        for result, target in results:
            img_id = target.get("img_id")
            img_info = self.coco.loadImgs(img_id)[0]
            img_w, img_h = img_info["width"], img_info["height"]
            if isinstance(result, dict):
                boxes = np.asarray(result.get("boxes", []))
                scores = np.asarray(result.get("scores", []))
                labels = np.asarray(result.get("labels", []))
            else:
                if result is None or len(result) != 3:
                    continue
                polys, scores, labels = result
                if polys is None or len(polys) == 0:
                    continue
                boxes = self._poly_to_xyxy(polys)
                scores = np.asarray(scores)
                labels = np.asarray(labels)

            if boxes is None or len(boxes) == 0:
                continue

            for box, score, label in zip(boxes, scores, labels):
                label = int(label)
                if label < 0 or label >= len(self.cat_ids):
                    continue
                x1, y1, x2, y2 = box.tolist()
                # clip to image bounds to avoid invalid COCO boxes
                x1 = max(0.0, min(x1, img_w))
                y1 = max(0.0, min(y1, img_h))
                x2 = max(0.0, min(x2, img_w))
                y2 = max(0.0, min(y2, img_h))
                if x2 <= x1 or y2 <= y1:
                    continue
                data = dict(
                    image_id=img_id,
                    bbox=[x1, y1, x2 - x1, y2 - y1],
                    score=float(score),
                    category_id=self.cat_ids[label],
                )
                json_results.append(data)

        import json
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(json_results, f)
