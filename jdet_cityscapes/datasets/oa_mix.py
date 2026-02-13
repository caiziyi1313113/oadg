import warnings

import numpy as np
from PIL import Image, ImageOps, ImageEnhance

import cv2

from jdet.utils.registry import TRANSFORMS
from jdet.models.boxes.iou_calculator import bbox_overlaps_np

# ----------------------------
# Local AugMix ops (no mmdet dependency)
# ----------------------------
def int_parameter(level, maxval):
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    return float(level) * maxval / 10.0


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, **kwargs):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, **kwargs):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level, **kwargs):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def invert(pil_img, **kwargs):
    return ImageOps.invert(pil_img)


def solarize(pil_img, level, **kwargs):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def color(pil_img, level, **kwargs):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


def contrast(pil_img, level, **kwargs):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


def brightness(pil_img, level, **kwargs):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


def sharpness(pil_img, level, **kwargs):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


def rotate(pil_img, level, img_size, fillcolor=None, center=None, mask=None, bbox_xy=None, return_bbox=False, **kwargs):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    if center is None:
        center = (img_size[0] / 2, img_size[1] / 2)
    M = cv2.getRotationMatrix2D(center, degrees, 1.0)
    outputs = dict(img=cv2.warpAffine(pil_img, M, img_size))
    if mask is not None:
        outputs['mask'] = cv2.warpAffine(mask, M, img_size)
    if return_bbox:
        outputs['gt_bbox'] = bbox_xy
    return outputs


def shear_x(pil_img, level, img_size, fillcolor=None, center=None, mask=None, bbox_xy=None, return_bbox=False, **kwargs):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    tx = 0 if center is None else -level * center[1]
    M = np.float32([[1, -level, -tx], [0, 1, 0]])
    outputs = dict(img=cv2.warpAffine(pil_img, M, (0, 0)))
    if mask is not None:
        outputs['mask'] = cv2.warpAffine(mask, M, (0, 0))
    if return_bbox:
        outputs['gt_bbox'] = bbox_xy
    return outputs


def shear_y(pil_img, level, img_size, fillcolor=None, center=None, mask=None, bbox_xy=None, return_bbox=False, **kwargs):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    ty = 0 if center is None else -level * center[0]
    M = np.float32([[1, 0, 0], [-level, 1, -ty]])
    outputs = dict(img=cv2.warpAffine(pil_img, M, (0, 0)))
    if mask is not None:
        outputs['mask'] = cv2.warpAffine(mask, M, (0, 0))
    if return_bbox:
        outputs['gt_bbox'] = bbox_xy
    return outputs


def translate_x(pil_img, level, img_size, fillcolor=None, img_size_for_level=None, mask=None, bbox_xy=None, return_bbox=False, **kwargs):
    maxval = img_size[0] if img_size_for_level is None else img_size_for_level[0]
    level = int_parameter(sample_level(level), maxval / 3)
    if np.random.random() > 0.5:
        level = -level
    M = np.float32([[1, 0, -level], [0, 1, 0]])
    outputs = dict(img=cv2.warpAffine(pil_img, M, (0, 0)))
    if mask is not None:
        outputs['mask'] = cv2.warpAffine(mask, M, (0, 0))
    if return_bbox:
        bbox_xy[0] = max(bbox_xy[0], bbox_xy[0] - level)
        bbox_xy[2] = min(bbox_xy[2], bbox_xy[2] - level)
        outputs['gt_bbox'] = bbox_xy
    return outputs


def translate_y(pil_img, level, img_size, fillcolor=None, img_size_for_level=None, mask=None, bbox_xy=None, return_bbox=False, **kwargs):
    maxval = img_size[1] if img_size_for_level is None else img_size_for_level[1]
    level = int_parameter(sample_level(level), maxval / 3)
    if np.random.random() > 0.5:
        level = -level
    M = np.float32([[1, 0, 0], [0, 1, -level]])
    outputs = dict(img=cv2.warpAffine(pil_img, M, (0, 0)))
    if mask is not None:
        outputs['mask'] = cv2.warpAffine(mask, M, (0, 0))
    if return_bbox:
        bbox_xy[1] = max(bbox_xy[1], bbox_xy[1] - level)
        bbox_xy[3] = min(bbox_xy[3], bbox_xy[3] - level)
        outputs['gt_bbox'] = bbox_xy
    return outputs


# BBox-only augmentation helpers (ported from mmdet)
def _apply_bbox_only_augmentation(img, bbox_xy, aug_func, fillmode=None, fillcolor=None,
                                  return_bbox=False, radius=10, radius_ratio=None,
                                  margin=3, sigma_ratio=None, times=3, blur_bbox=None, **kwargs):
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    x1, y1, x2, y2 = int(bbox_xy[0]), int(bbox_xy[1]), int(bbox_xy[2]), int(bbox_xy[3])
    if (x2 - x1) < 1 or (y2 - y1) < 1:
        return (np.asarray(img, dtype=np.uint8), bbox_xy) if return_bbox else np.asarray(img, dtype=np.uint8)

    bbox_content = img
    img_height, img_width = img.shape[0], img.shape[1]
    kwargs['img_size'] = (img_width, img_height)
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    kwargs['img_size_for_level'] = (x2 - x1 + 1, y2 - y1 + 1)
    outputs = aug_func(bbox_content, **kwargs, fillcolor=fillcolor, center=center,
                       bbox_xy=bbox_xy, return_bbox=return_bbox)
    augmented_bbox_content = np.asarray(outputs['img'])
    augmented_gt_bbox = outputs['gt_bbox'] if 'gt_bbox' in outputs else bbox_xy

    if blur_bbox is None:
        mask = np.ones_like(img, dtype=np.float32)
    else:
        mask = 1.0 - np.asarray(blur_bbox, dtype=np.float32)
    img = img * mask + augmented_bbox_content * (1.0 - mask)

    if return_bbox:
        return np.asarray(img, dtype=np.uint8), augmented_gt_bbox
    else:
        return np.asarray(img, dtype=np.uint8)


def _apply_bboxes_only_augmentation(img, bboxes_xy, aug_func, mask_bboxes=None, **kwargs):
    if mask_bboxes is None:
        mask_bboxes = [None] * len(bboxes_xy)
    assert len(bboxes_xy) == len(mask_bboxes)
    for i in range(len(bboxes_xy)):
        blur_bbox = None if mask_bboxes is None else mask_bboxes[i]
        img = _apply_bbox_only_augmentation(img, bboxes_xy[i], aug_func, blur_bbox=blur_bbox, **kwargs)
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    else:
        return img


def bboxes_only_rotate(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bboxes_only_augmentation(pil_img, bboxes_xy, rotate, level=level, img_size=img_size, **kwargs)


def bboxes_only_shear_x(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bboxes_only_augmentation(pil_img, bboxes_xy, shear_x, level=level, img_size=img_size, **kwargs)


def bboxes_only_shear_y(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bboxes_only_augmentation(pil_img, bboxes_xy, shear_y, level=level, img_size=img_size, **kwargs)


def bboxes_only_shear_xy(pil_img, bboxes_xy, level, img_size, **kwargs):
    func = bboxes_only_shear_x if np.random.rand() < 0.5 else bboxes_only_shear_y
    return func(pil_img, bboxes_xy, level, img_size, **kwargs)


def bboxes_only_translate_x(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bboxes_only_augmentation(pil_img, bboxes_xy, translate_x, level=level, img_size=img_size, **kwargs)


def bboxes_only_translate_y(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bboxes_only_augmentation(pil_img, bboxes_xy, translate_y, level=level, img_size=img_size, **kwargs)


def bboxes_only_translate_xy(pil_img, bboxes_xy, level, img_size, **kwargs):
    func = bboxes_only_translate_x if np.random.rand() < 0.5 else bboxes_only_translate_y
    return func(pil_img, bboxes_xy, level, img_size, **kwargs)


# Background-only augmentation (ported from mmdet)
def _apply_bg_only_augmentation(img, bboxes_xy, aug_func, mask_bboxes=None, fillmode=None,
                                fillcolor=0, return_bbox=False, radius=10,
                                radius_ratio=None, bg_margin=3, times=3, margin_bg=False,
                                sigma_ratio=None, blur_bboxes=None, **kwargs):
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    bbox_content = img.copy()
    img_shape = img.copy().shape
    kwargs['img_size'] = (img_shape[1], img_shape[0])

    if mask_bboxes is None or len(mask_bboxes) == 0:
        mask = np.zeros_like(img)
    else:
        mask = np.max(mask_bboxes, axis=0)

    outputs = aug_func(bbox_content, return_bbox=False, **kwargs, fillcolor=fillcolor,
                       mask=np.asarray(mask * 255, dtype=np.uint8))
    augmented_bbox_content = outputs['img']
    augmented_mask = np.asarray(outputs['mask']) / 255

    maintained_mask = np.maximum(mask, augmented_mask)
    img = maintained_mask * img + (1.0 - maintained_mask) * augmented_bbox_content
    return np.asarray(img, dtype=np.uint8)


def bg_only_rotate(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bg_only_augmentation(pil_img, bboxes_xy, rotate, level=level, img_size=img_size, **kwargs)


def bg_only_shear_x(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bg_only_augmentation(pil_img, bboxes_xy, shear_x, level=level, img_size=img_size, **kwargs)


def bg_only_shear_y(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bg_only_augmentation(pil_img, bboxes_xy, shear_y, level=level, img_size=img_size, **kwargs)


def bg_only_shear_xy(pil_img, bboxes_xy, level, img_size, **kwargs):
    func = bg_only_shear_x if np.random.rand() < 0.5 else bg_only_shear_y
    return func(pil_img, bboxes_xy, level, img_size, **kwargs)


def bg_only_translate_x(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bg_only_augmentation(pil_img, bboxes_xy, translate_x, level=level, img_size=img_size, **kwargs)


def bg_only_translate_y(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bg_only_augmentation(pil_img, bboxes_xy, translate_y, level=level, img_size=img_size, **kwargs)


def bg_only_translate_xy(pil_img, bboxes_xy, level, img_size, **kwargs):
    func = bg_only_translate_x if np.random.rand() < 0.5 else bg_only_translate_y
    return func(pil_img, bboxes_xy, level, img_size, **kwargs)


def get_aug_list(version):
    if version == 'augmix':
        aug_list = [autocontrast, equalize, posterize, solarize,
                    bboxes_only_rotate, bboxes_only_shear_xy, bboxes_only_translate_xy,
                    bg_only_rotate, bg_only_shear_xy, bg_only_translate_xy]
    elif version == 'augmix.all':
        aug_list = [autocontrast, equalize, posterize, solarize, invert,
                    color, contrast, brightness, sharpness,
                    bboxes_only_rotate, bboxes_only_shear_xy, bboxes_only_translate_xy,
                    bg_only_rotate, bg_only_shear_xy, bg_only_translate_xy]
    else:
        raise NotImplementedError
    return aug_list


@TRANSFORMS.register_module()
class OAMix:
    def __init__(self,
                 version='augmix',
                 num_views=1, keep_orig=False, severity=10,
                 mixture_width=3, mixture_depth=-1,
                 random_box_scale=(0.01, 0.1), random_box_ratio=(3, 1 / 3),
                 oa_random_box_scale=(0.005, 0.1), oa_random_box_ratio=(3, 1 / 3), num_bboxes=(3, 5),
                 spatial_ratio=4, sigma_ratio=0.3,
                 **kwargs):
        super().__init__()
        self.aug_list = get_aug_list(version)

        self.num_views = num_views
        self.keep_orig = keep_orig
        if self.num_views == 1 and self.keep_orig:
            warnings.warn('No augmentation will be applied since num_views=1 and keep_orig=True')

        self.severity = severity
        self.aug_prob_coeff = 1.0
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth

        self.random_box_scale = random_box_scale
        self.random_box_ratio = random_box_ratio

        self.oa_random_box_scale = oa_random_box_scale
        self.oa_random_box_ratio = oa_random_box_ratio

        self.score_thresh = 10

        self.spatial_ratio = spatial_ratio
        self.sigma_ratio = sigma_ratio

        self._history = {}
        self.kwargs = kwargs

    @staticmethod
    def _get_mask(box, target_shape, spatial_ratio=None, sigma_ratio=None):
        h_img, w_img, c_img = target_shape
        use_blur = (spatial_ratio is not None) and (sigma_ratio is not None)
        if use_blur:
            x1, y1, x2, y2 = np.array(box // spatial_ratio, dtype=np.int32)
            mask = np.zeros((h_img // spatial_ratio, w_img // spatial_ratio, c_img), dtype=np.float32)
        else:
            x1, y1, x2, y2 = box
            mask = np.zeros(target_shape, dtype=np.float32)

        mask[y1:y2, x1:x2, :] = 1.0
        if use_blur:
            sigma_x = (x2 - x1) * sigma_ratio / 3 * 2
            sigma_y = (y2 - y1) * sigma_ratio / 3 * 2
            if not (sigma_x <= 0 or sigma_y <= 0):
                mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
            mask = cv2.resize(mask, (w_img, h_img))

        return mask

    def get_fg_regions(self, img, gt_bboxes):
        if hasattr(self._history, 'fg_box_list'):
            return self._history['fg_box_list'], self._history['fg_mask_list'], self._history['fg_score_list']

        if gt_bboxes is None or len(gt_bboxes) == 0:
            self._history.update({
                'fg_box_list': [],
                'fg_mask_list': [],
                'fg_score_list': []
            })
            return [], [], []

        fg_box_list, fg_mask_list, fg_score_list = gt_bboxes, [], []
        for gt_bbox in gt_bboxes:
            x1, y1, x2, y2 = np.array(gt_bbox, dtype=np.int32)
            if x2 - x1 < self.spatial_ratio or y2 - y1 < self.spatial_ratio:
                fg_score_list.append(-1)
            else:
                if hasattr(cv2, 'saliency'):
                    bbox_img = img[y1:y2, x1:x2]
                    try:
                        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
                        (_, saliency_map) = saliency.computeSaliency(bbox_img)
                        saliency_score = np.mean((saliency_map * 255).astype('uint8'))
                    except Exception:
                        saliency_score = 0
                else:
                    saliency_score = 0
                fg_score_list.append(saliency_score)

            fg_mask = self._get_mask(gt_bbox, img.shape, spatial_ratio=self.spatial_ratio, sigma_ratio=self.sigma_ratio)
            fg_mask_list.append(fg_mask)

        self._history.update({'fg_box_list': fg_box_list})
        self._history.update({'fg_mask_list': fg_mask_list})
        self._history.update({'fg_score_list': fg_score_list})
        return fg_box_list, fg_mask_list, fg_score_list

    def get_random_regions(self, img, scale, ratio,
                           num_bboxes=None, use_blur=False,
                           return_score=False, fg_box_list=None, fg_score_list=None,
                           max_iters=50, eps=1e-6):
        if return_score:
            assert fg_box_list is not None and fg_score_list is not None
        (h_img, w_img, c_img) = img.shape

        random_box_list, random_mask_list, random_score_list = [], [], []

        target_num_bboxes = np.random.randint(*num_bboxes) if isinstance(num_bboxes, tuple) else num_bboxes
        for _ in range(max_iters):
            if len(random_mask_list) >= target_num_bboxes:
                break

            x1, y1 = np.random.randint(0, w_img), np.random.randint(0, h_img)
            _scale = np.random.uniform(*scale) * h_img * w_img
            _ratio = np.random.uniform(*ratio)
            bbox_w, bbox_h = int(np.sqrt(_scale / _ratio)), int(np.sqrt(_scale * _ratio))

            if x1 + bbox_w > w_img or y1 + bbox_h > h_img:
                continue

            x2, y2 = min(x1 + bbox_w, w_img), min(y1 + bbox_h, h_img)
            random_box = np.array([[x1, y1, x2, y2]])

            if len(random_box_list) > 0:
                ious = bbox_overlaps_np(random_box, np.asarray(random_box_list))
                if np.sum(ious) > eps:
                    continue

            if return_score:
                if fg_box_list is None or len(fg_box_list) == 0:
                    random_score_list.append(float('inf'))
                else:
                    ious = bbox_overlaps_np(random_box, np.asarray(fg_box_list))
                    final_score = float('inf')
                    if np.sum(ious) > eps:
                        for i, (iou, fg_box, fg_score) in enumerate(zip(ious[0], fg_box_list, fg_score_list)):
                            x1_fg, y1_fg, x2_fg, y2_fg = fg_box
                            if iou == 0.0 or x2_fg - x1_fg < 1 or y2_fg - y1_fg < 1:
                                continue
                            if fg_score < final_score:
                                final_score = fg_score
                    random_score_list.append(final_score)

            if use_blur:
                random_mask = self._get_mask(random_box[0], img.shape, spatial_ratio=self.spatial_ratio, sigma_ratio=self.sigma_ratio)
            else:
                random_mask = self._get_mask(random_box[0], img.shape)
            random_mask_list.append(random_mask)
            random_box_list += list(random_box)

        if return_score:
            return random_box_list, random_mask_list, random_score_list
        else:
            return random_box_list, random_mask_list

    def __call__(self, image, target=None):
        if target is None or 'bboxes' not in target:
            return image, target

        self._history = {}

        if self.num_views <= 1:
            if not self.keep_orig:
                img_np = np.asarray(image, dtype=np.uint8)
                img_np = self.oamix(img_np, target['bboxes'].copy())
                image = Image.fromarray(img_np)
            return image, target

        # multi-view: keep original as img, generate img2
        img_np = np.asarray(image, dtype=np.uint8)
        img_aug = self.oamix(img_np, target['bboxes'].copy())
        if self.keep_orig:
            target['img2'] = Image.fromarray(np.asarray(img_aug, dtype=np.uint8))
        else:
            # swap: use aug as primary, store orig as img2
            target['img2'] = Image.fromarray(img_np.copy())
            image = Image.fromarray(np.asarray(img_aug, dtype=np.uint8))

        # duplicate labels/boxes for view2
        target['bboxes2'] = target['bboxes'].copy()
        target['labels2'] = target['labels'].copy()

        # store oamix/meta boxes for random proposals
        if 'random_box_list' in self._history and len(self._history['random_box_list']) > 0:
            target['multilevel_boxes'] = self._history['random_box_list'].copy()
        else:
            target['multilevel_boxes'] = np.zeros((0, 4), dtype=np.float32)
        if 'oa_random_box_list' in self._history and len(self._history['oa_random_box_list']) > 0:
            target['oamix_boxes'] = np.stack(self._history['oa_random_box_list'], axis=0).astype(np.float32)
        else:
            target['oamix_boxes'] = np.zeros((0, 4), dtype=np.float32)

        return image, target

    def oamix(self, img, gt_bboxes):
        img = np.asarray(img, dtype=np.uint8)
        if gt_bboxes is None or len(gt_bboxes) == 0:
            return img
        h_img, w_img, _ = img.shape
        img_size = (w_img, h_img)

        ws = np.float32(np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))

        random_box_list, random_mask_list = self.get_random_regions(
            img, self.random_box_scale, self.random_box_ratio, num_bboxes=(1, 3))
        if len(random_box_list) > 0:
            self._history.update({'random_box_list': np.stack(random_box_list, axis=0)})
        else:
            self._history.update({'random_box_list': np.zeros((0, 4), dtype=np.float32)})

        fg_box_list, fg_mask_list, fg_score_list = self.get_fg_regions(img=img, gt_bboxes=gt_bboxes)

        img_oamix = np.zeros_like(img.copy(), dtype=np.float32)
        for i in range(self.mixture_width):
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            img_aug = Image.fromarray(img.copy(), 'RGB')
            for _ in range(depth):
                img_tmp = np.zeros_like(img, dtype=np.float32)
                for _randbox, _randmask in zip(random_box_list, random_mask_list):
                    img_tmp += _randmask * self.aug(img_aug, img_size, fg_box_list, fg_mask_list)

                if len(random_mask_list) > 0:
                    union_mask = np.max(random_mask_list, axis=0)
                else:
                    union_mask = np.zeros_like(img, dtype=np.float32)
                img_aug = np.asarray(
                    img_tmp + (1.0 - union_mask) * self.aug(img_aug, img_size, fg_box_list, fg_mask_list),
                    dtype=np.uint8)

            img_oamix += ws[i] * np.asarray(img_aug, dtype=np.float32)

        oa_target_box_list, oa_target_mask_list, oa_target_score_list = self.get_regions_for_object_aware_mixing(
            img, fg_box_list, fg_mask_list, fg_score_list)
        img_oamix = self.object_aware_mixing(img, img_oamix, oa_target_mask_list, oa_target_score_list)

        return np.asarray(img_oamix, dtype=np.uint8)

    def get_regions_for_object_aware_mixing(self, img, fg_box_list, fg_mask_list, fg_score_list):
        oa_target_box_list, oa_target_mask_list, oa_target_score_list = [], [], []
        for box, mask, score in zip(fg_box_list, fg_mask_list, fg_score_list):
            if score <= self.score_thresh:
                oa_target_box_list.append(box)
                oa_target_mask_list.append(mask)
                oa_target_score_list.append(score)
        oa_random_box_list, oa_random_mask_list, oa_random_score_list = self.get_random_regions(
            img, self.oa_random_box_scale, self.oa_random_box_ratio,
            num_bboxes=min(max(len(oa_target_box_list), 1), 5),
            return_score=True, fg_box_list=fg_box_list, fg_score_list=fg_score_list)
        oa_target_box_list += oa_random_box_list
        oa_target_mask_list += oa_random_mask_list
        oa_target_score_list += oa_random_score_list
        self._history.update({'oa_random_box_list': oa_random_box_list})
        return oa_target_box_list, oa_target_mask_list, oa_target_score_list

    def aug(self, img, img_size, fg_box_list, fg_mask_list):
        op = np.random.choice(self.aug_list)
        if op in [autocontrast, equalize, posterize, solarize, color, contrast, brightness, sharpness]:
            if type(img) == np.ndarray:
                img = Image.fromarray(img, 'RGB')
            pil_img = op(img, level=self.severity, img_size=img_size)
        elif op in [invert]:
            if not isinstance(img, np.ndarray):
                img = np.asarray(img, dtype=np.uint8)
            tx = 1 if np.random.random() > 0.5 else -1
            ty = 1 if np.random.random() > 0.5 else -1
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            pil_img = -cv2.warpAffine(img, M, (0, 0))
        else:
            pil_img = op(img, level=self.severity, img_size=img_size, bboxes_xy=fg_box_list, mask_bboxes=fg_mask_list, fillmode='oa')
        return pil_img

    def object_aware_mixing(self, img, img_aug, mask_list, score_list):
        m = np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff)

        orig, aug = np.zeros_like(img, dtype=np.float32), np.zeros_like(img, dtype=np.float32)
        mask_sum = np.zeros_like(img, dtype=np.float32)
        mask_max_list = []
        for mask, score in zip(mask_list, score_list):
            mask_sum += mask
            mask_max_list.append(mask)
            mask_max = np.max(mask_max_list, axis=0)
            mask_overlap = mask_sum - mask_max

            if score <= self.score_thresh:
                m_oa = np.float32(np.random.uniform(0.0, 0.5))
            else:
                m_oa = np.float32(np.random.uniform(0.0, 1.0))
            orig += (1.0 - m_oa) * img * (mask - mask_overlap * 0.5)
            aug += m_oa * img_aug * (mask - mask_overlap * 0.5)
            mask_sum = mask_max

        img_oamix = orig + aug

        img_oamix += (1.0 - m) * img * (1.0 - mask_sum)
        img_oamix += m * img_aug * (1.0 - mask_sum)
        img_oamix = np.clip(img_oamix, 0, 255)

        return img_oamix

    def __repr__(self):
        return self.__class__.__name__
