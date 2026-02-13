import numpy as np
import jittor as jt
from PIL import Image

from jdet.utils.registry import MODELS
from .faster_rcnn_hbb import FasterRCNNHBB


@MODELS.register_module()
class FasterRCNNHBBMulti(FasterRCNNHBB):
    """FasterRCNNHBB variant that supports batch_size > 1 with multi-view (img2)."""

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
                img2 = np.array(img2)
            elif isinstance(img2, jt.Var):
                img2 = img2.numpy()
            else:
                img2 = np.asarray(img2)
            if img2.ndim == 3 and img2.shape[-1] in (1, 3):
                img2 = img2.transpose((2, 0, 1))
            if img2.ndim == 2:
                img2 = img2[None, ...]
            if img2.shape[0] == 1:
                img2 = np.repeat(img2, 3, axis=0)
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

        # pad img2 to batch max size (same as collated images)
        h_max, w_max = images.shape[-2], images.shape[-1]
        padded = np.zeros((len(imgs2), 3, h_max, w_max), dtype=np.float32)
        for i, arr in enumerate(imgs2):
            h = min(arr.shape[1], h_max)
            w = min(arr.shape[2], w_max)
            padded[i, :, :h, :w] = arr[:, :h, :w]

        images2 = jt.array(padded)
        images_all = jt.concat([images, images2], dim=0)
        targets_all = targets + targets2
        return images_all, targets_all, 2, images.shape[0]
