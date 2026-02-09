import random
from PIL import Image
import numpy as np

from jdet.utils.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ResizeByImgScale:
    """Resize using mmdet-style img_scale list with keep_ratio=True."""

    def __init__(self, img_scale, keep_ratio=True):
        if not isinstance(img_scale, (list, tuple)):
            raise TypeError("img_scale must be list/tuple")
        if isinstance(img_scale, tuple) and isinstance(img_scale[0], int):
            self.img_scale = [img_scale]
        else:
            self.img_scale = list(img_scale)
        self.keep_ratio = keep_ratio

    def _resize_boxes(self, target, size):
        for key in ["bboxes", "polys"]:
            if key not in target:
                continue
            bboxes = target[key]
            if bboxes is None or bboxes.ndim < 2:
                continue
            width, height = target["img_size"]
            new_w, new_h = size
            bboxes[:, 0::2] = bboxes[:, 0::2] * float(new_w / width)
            bboxes[:, 1::2] = bboxes[:, 1::2] * float(new_h / height)
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, new_w - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, new_h - 1)
            target[key] = bboxes

    def __call__(self, image, target=None):
        # print("[dbg] ResizeByImgScale: start", flush=True)
        if isinstance(image, Image.Image):
            w, h = image.size
        else:
            h, w = image.shape[-2:]

        scale_w, scale_h = random.choice(self.img_scale)
        if self.keep_ratio:
            scale = min(scale_w / w, scale_h / h)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            scale_factor = scale
        else:
            new_w, new_h = int(scale_w), int(scale_h)
            scale_factor = [new_w / w, new_h / h, new_w / w, new_h / h]

        if isinstance(image, Image.Image):
            image = image.resize((new_w, new_h), Image.BILINEAR)
        else:
            image = Image.fromarray(image.transpose(1, 2, 0))
            image = image.resize((new_w, new_h), Image.BILINEAR)

        if target is not None:
            self._resize_boxes(target, (new_w, new_h))
            target["img_size"] = (new_w, new_h)
            target["scale_factor"] = scale_factor
            target["pad_shape"] = (new_w, new_h)
            target["keep_ratio"] = self.keep_ratio
        # print("[dbg] ResizeByImgScale: end", flush=True)
        return image, target
