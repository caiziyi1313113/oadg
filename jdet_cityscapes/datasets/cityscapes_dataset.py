import os

from jdet.utils.registry import DATASETS
from jdet.data.coco import COCODataset


@DATASETS.register_module()
class CityscapesDataset(COCODataset):
    """Cityscapes dataset (COCO-format) with fixed 8 detection classes."""

    CLASSES = (
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    )

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        ids_with_ann = set(_["image_id"] for _ in self.coco.anns.values())
        ids_in_cat = set()
        for class_id in self.cat_ids:
            ids_in_cat |= set(self.coco.catToImgs[class_id])
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for img_id in self.img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            ann_info = self.coco.loadAnns(ann_ids)
            all_iscrowd = all([_["iscrowd"] for _ in ann_info]) if ann_info else False
            if self.filter_empty_gt and (img_id not in ids_in_cat or all_iscrowd):
                continue
            if min(img_info["width"], img_info["height"]) >= min_size:
                valid_img_ids.append(img_id)

        self.img_ids = valid_img_ids
        return valid_img_ids

    def __init__(self, *args, **kwargs):
        self.filter_empty_gt = kwargs.get("filter_empty_gt", True)
        super().__init__(*args, **kwargs)
