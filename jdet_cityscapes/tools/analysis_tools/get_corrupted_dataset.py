import argparse
import copy
import os
import sys

import numpy as np
from PIL import Image

# Ensure local project and JDet are on sys.path before importing jdet.*
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
JDET_PY = r"D:\JDet\python" if os.name == "nt" else "/mnt/d/sim2real/OA-DG/JDet/python"
if os.path.isdir(JDET_PY) and JDET_PY not in sys.path:
    sys.path.insert(0, JDET_PY)

import jittor as jt

from jdet.config import init_cfg, get_cfg
from jdet.utils.registry import build_from_cfg, DATASETS
import jdet.data.transforms  # noqa: F401

# Register custom dataset/model/transforms
import datasets.cityscapes_dataset  # noqa: F401
import datasets.resize_by_img_scale  # noqa: F401
import datasets.repeat_dataset  # noqa: F401
import models.hbb_head  # noqa: F401
import models.faster_rcnn_hbb  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="JDet get corrupted dataset")
    parser.add_argument("config", help="config file path")
    parser.add_argument(
        "--show-dir",
        default="/mnt/d/sim2real/OA-DG/ws/data/cityscapes-c",
        help="directory to save images",
    )
    parser.add_argument(
        "--corruptions",
        type=str,
        nargs="+",
        default="benchmark",
        choices=[
            "all", "benchmark", "noise", "blur", "weather", "digital",
            "holdout", "None", "gaussian_noise", "shot_noise", "impulse_noise",
            "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow",
            "frost", "fog", "brightness", "contrast", "elastic_transform",
            "pixelate", "jpeg_compression", "speckle_noise", "gaussian_blur",
            "spatter", "saturate",
        ],
    )
    parser.add_argument(
        "--severities",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
        help="corruption severity levels",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="skip saving if target file already exists",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="do not skip existing files (overwrite)",
    )
    return parser.parse_args()


def _save_image(image, save_path):
    if isinstance(image, Image.Image):
        img = image
    else:
        if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = image.transpose(1, 2, 0)
        img = Image.fromarray(image.astype(np.uint8))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path.endswith(".jpg"):
        save_path = save_path[:-4] + ".png"
    if os.path.exists(save_path):
        return False
    img.save(save_path, quality="keep")
    return True


def _get_filename(dataset, target, idx=None):
    if isinstance(target, dict):
        img_file = target.get("img_file", target.get("filename", ""))
        if img_file:
            return img_file
        img_id = target.get("img_id", None)
    else:
        img_id = None
    if img_id is None and idx is not None and hasattr(dataset, "img_ids"):
        img_id = dataset.img_ids[idx]
    if img_id is not None and hasattr(dataset, "coco"):
        info = dataset.coco.loadImgs(img_id)[0]
        return info.get("file_name", "")
    return ""


def _get_filename_by_idx(dataset, idx):
    if hasattr(dataset, "img_ids") and hasattr(dataset, "coco"):
        img_id = dataset.img_ids[idx]
        info = dataset.coco.loadImgs(img_id)[0]
        return info.get("file_name", "")
    return ""


def main():
    args = parse_args()
    if not args.no_cuda:
        jt.flags.use_cuda = 1
    jt.set_global_seed(args.seed)
    if args.no_skip_existing:
        args.skip_existing = False

    init_cfg(args.config)
    cfg = get_cfg()

    if "all" in args.corruptions:
        corruptions = [
            "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
            "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
            "brightness", "contrast", "elastic_transform", "pixelate",
            "jpeg_compression", "speckle_noise", "gaussian_blur", "spatter",
            "saturate",
        ]
    elif "benchmark" in args.corruptions:
        corruptions = [
            "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
            "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
            "brightness", "contrast", "elastic_transform", "pixelate",
            "jpeg_compression",
        ]
    elif "noise" in args.corruptions:
        corruptions = ["gaussian_noise", "shot_noise", "impulse_noise"]
    elif "blur" in args.corruptions:
        corruptions = ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"]
    elif "weather" in args.corruptions:
        corruptions = ["snow", "frost", "fog", "brightness"]
    elif "digital" in args.corruptions:
        corruptions = ["contrast", "elastic_transform", "pixelate", "jpeg_compression"]
    elif "holdout" in args.corruptions:
        corruptions = ["speckle_noise", "gaussian_blur", "spatter", "saturate"]
    elif "None" in args.corruptions:
        corruptions = ["None"]
        args.severities = [0]
    else:
        corruptions = args.corruptions

    for corruption in corruptions:
        for severity in args.severities:
            if corruption != "None" and severity == 0:
                continue

            test_cfg = copy.deepcopy(cfg.dataset.test)
            if severity > 0 and corruption != "None":
                test_cfg["transforms"] = list(test_cfg.get("transforms", []))
                test_cfg["transforms"].insert(0, dict(
                    type="Corrupt",
                    corruption=corruption,
                    severity=severity,
                ))

            dataset = build_from_cfg(test_cfg, DATASETS, drop_last=False)
            print(f"\nTesting {corruption} at severity {severity}")
            print("root:", getattr(dataset, "root", None))
            print("ann:", getattr(dataset, "anno_file", None))
            print("len:", len(dataset))

            out_dir = os.path.join(args.show_dir, corruption, str(severity))
            os.makedirs(out_dir, exist_ok=True)

            start_idx = 0
            if args.skip_existing:
                for idx in range(len(dataset)):
                    rel = _get_filename_by_idx(dataset, idx)
                    if not rel:
                        start_idx = idx
                        break
                    save_path = os.path.join(out_dir, rel)
                    if not os.path.exists(save_path):
                        start_idx = idx
                        break
                else:
                    start_idx = len(dataset)

            for idx in range(start_idx, len(dataset)):
                image, target = dataset[idx]
                rel = _get_filename(dataset, target, idx=idx)
                if not rel:
                    continue
                save_path = os.path.join(out_dir, rel)
                if _save_image(image, save_path):
                    print(save_path)


if __name__ == "__main__":
    main()
