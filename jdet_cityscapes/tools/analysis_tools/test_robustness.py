import argparse
import copy
import json
import os
import sys

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
from jdet.utils.registry import build_from_cfg, MODELS, DATASETS
from jdet.utils.general import sync

# Register custom dataset/model/extensions
import datasets.cityscapes_dataset  # noqa: F401
import datasets.resize_by_img_scale  # noqa: F401
import models.hbb_head  # noqa: F401
import models.faster_rcnn_hbb  # noqa: F401
import models.oadg_losses  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="JDet robustness test")
    parser.add_argument("config", help="config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--out", required=True, help="output json file")
    parser.add_argument(
        "--corruptions",
        type=str,
        nargs="+",
        default=["benchmark"],
        choices=[
            "all", "benchmark", "noise", "blur", "weather", "digital", "holdout", "None",
            "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
            "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
            "brightness", "contrast", "elastic_transform", "pixelate",
            "jpeg_compression", "speckle_noise", "gaussian_blur", "spatter", "saturate",
        ],
    )
    parser.add_argument(
        "--severities",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
        help="corruption severity levels",
    )
    parser.add_argument(
        "--load-dataset",
        type=str,
        choices=["original", "corrupted"],
        default="corrupted",
        help="use original dataset with online corruption or prebuilt corrupted dataset",
    )
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def _load_model(cfg, checkpoint_path):
    model = build_from_cfg(cfg.model, MODELS)
    resume_data = jt.load(checkpoint_path)
    if "model" in resume_data:
        model.load_parameters(resume_data["model"])
    elif "state_dict" in resume_data:
        model.load_parameters(resume_data["state_dict"])
    else:
        model.load_parameters(resume_data)
    model.eval()
    return model


def _run_eval(model, dataset, work_dir, epoch):
    results = []
    for images, targets in dataset:
        result = model(images, targets)
        results.extend([(r, t) for r, t in zip(sync(result), sync(targets))])
    eval_results = dataset.evaluate(results, work_dir, epoch, logger="print")
    return eval_results, results


def main():
    args = parse_args()
    if not args.no_cuda:
        jt.flags.use_cuda = 1

    init_cfg(args.config)
    cfg = get_cfg()
    cfg.resume_path = ""
    cfg.pretrained_weights = None
    # Prevent loading the config's pretrained weights during robustness eval.
    if hasattr(cfg, "load_from"):
        cfg.load_from = None
    if hasattr(cfg, "model"):
        if isinstance(cfg.model, dict):
            cfg.model["pretrained"] = None
            if "backbone" in cfg.model and isinstance(cfg.model["backbone"], dict):
                cfg.model["backbone"]["pretrained"] = None

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

    model = _load_model(cfg, args.checkpoint)
    aggregated_results = {}

    for corr_i, corruption in enumerate(corruptions):
        aggregated_results[corruption] = {}
        for severity in args.severities:
            if corr_i > 0 and severity == 0:
                aggregated_results[corruption][0] = aggregated_results[corruptions[0]][0]
                continue

            test_cfg = copy.deepcopy(cfg.dataset.test)
            if args.load_dataset == "corrupted":
                root = test_cfg.get("root", "")
                if root:
                    # original root like .../cityscapes/leftImg8bit/val
                    base_root = root.replace("cityscapes-c", "cityscapes")
                    cc_root = base_root.replace("cityscapes", "cityscapes-c")
                    cc_parent = cc_root
                    if cc_parent.endswith(os.path.join("leftImg8bit", "val")):
                        cc_parent = os.path.dirname(os.path.dirname(cc_parent))
                    elif cc_parent.endswith(os.path.join("leftImg8bit", "train")):
                        cc_parent = os.path.dirname(os.path.dirname(cc_parent))

                    candidates = [
                        os.path.join(cc_root, corruption, str(severity)),
                        os.path.join(cc_parent, corruption, str(severity)),
                        os.path.join(cc_root, "leftImg8bit", "val", corruption, str(severity)),
                        os.path.join(cc_root, "leftImg8bit", "train", corruption, str(severity)),
                    ]
                    # If severity==0 and corrupted dataset not found, fallback to original clean root
                    chosen = None
                    for cand in candidates:
                        if os.path.isdir(cand):
                            chosen = cand
                            break
                    if chosen is None and severity == 0:
                        chosen = base_root
                    if chosen is None:
                        raise FileNotFoundError(
                            f"Corrupted dataset root not found. Tried: {candidates}"
                        )
                    test_cfg["root"] = chosen
            else:
                if severity != 0:
                    raise NotImplementedError(
                        "Online corruption not implemented for JDet transforms. "
                        "Use --load-dataset corrupted."
                    )

            dataset = build_from_cfg(test_cfg, DATASETS, drop_last=False)
            print(f"\nTesting {corruption} at severity {severity}")
            print("root:", getattr(dataset, "root", None))
            print("ann:", getattr(dataset, "anno_file", None))
            print("len:", len(dataset))

            eval_results, results = _run_eval(model, dataset, cfg.work_dir, severity)
            aggregated_results[corruption][severity] = {"bbox": eval_results}
            # diagnostics: count predicted boxes and preview json
            try:
                # build json from eval save file created in evaluate
                det_json = os.path.join(cfg.work_dir, "detections", f"val_{severity}.json")
                if os.path.exists(det_json):
                    with open(det_json, "r") as f:
                        dets = json.load(f)
                    num_boxes = len(dets)
                    avg_boxes = num_boxes / max(len(dataset), 1)
                    print(f"[diag] pred_boxes={num_boxes}, avg_per_img={avg_boxes:.4f}")
                    print("[diag] sample preds:", dets[:10])
                    # bbox range check vs image size
                    if hasattr(dataset, "coco"):
                        id2wh = {img["id"]: (img["width"], img["height"]) for img in dataset.coco.dataset.get("images", [])}
                        oob = 0
                        for d in dets[:2000]:
                            w, h = id2wh.get(d["image_id"], (None, None))
                            if w is None:
                                continue
                            x, y, bw, bh = d["bbox"]
                            if (x < 0) or (y < 0) or (x + bw > w) or (y + bh > h):
                                oob += 1
                        print(f"[diag] oob_in_first2000={oob}")
            except Exception as e:
                print(f"[diag] failed to read detections json: {e}")
            # incremental save for resume
            with open(args.out, "w") as f:
                json.dump(aggregated_results, f, indent=2)

    print("Saved:", args.out)


if __name__ == "__main__":
    main()
