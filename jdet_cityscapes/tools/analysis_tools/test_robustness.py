import argparse
import copy
import json
import os
import shutil
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
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="do not load existing --out file and always recompute all settings",
    )
    parser.add_argument(
        "--det-out-dir",
        default=None,
        help="directory to save per-corruption detection boxes "
             "(default: <dirname(--out)>/det_boxes)",
    )
    parser.add_argument(
        "--no-save-det-boxes",
        action="store_true",
        help="disable saving per-corruption detection json files",
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


def _det_json_path(det_out_dir, corruption, severity):
    return os.path.join(det_out_dir, corruption, f"severity_{severity}.json")


def _det_meta_path(det_out_dir, corruption, severity):
    return os.path.join(det_out_dir, corruption, f"severity_{severity}.meta.json")


def _update_det_index(det_out_dir, record):
    index_path = os.path.join(det_out_dir, "index.json")
    index_data = {}
    if os.path.exists(index_path):
        try:
            with open(index_path, "r") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                index_data = raw
        except Exception:
            index_data = {}
    key = f"{record['corruption']}|{record['severity']}"
    index_data[key] = record
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)


def _save_detection_boxes(dataset, results, det_out_dir, corruption, severity, ann_file):
    pred_path = _det_json_path(det_out_dir, corruption, severity)
    meta_path = _det_meta_path(det_out_dir, corruption, severity)
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    dataset.save_results(results, pred_path)
    num_preds = 0
    for result, _ in results:
        boxes = result.get("boxes", None)
        if boxes is None:
            continue
        if hasattr(boxes, "shape"):
            num_preds += int(boxes.shape[0])
        elif hasattr(boxes, "__len__"):
            num_preds += len(boxes)

    vis_out = os.path.join(det_out_dir, "vis", corruption, f"severity_{severity}")
    record = {
        "corruption": corruption,
        "severity": int(severity),
        "pred_file": os.path.relpath(pred_path, det_out_dir),
        "ann_file": ann_file,
        "img_root": getattr(dataset, "root", None),
        "num_images": len(dataset),
        "num_preds": num_preds,
        "vis_command": (
            "python jdet_cityscapes/tools/vis_coco_boxes.py "
            f"--ann-file \"{ann_file}\" "
            f"--img-root \"{getattr(dataset, 'root', '')}\" "
            f"--pred-file \"{pred_path}\" "
            f"--mode both --out-dir \"{vis_out}\""
        ),
    }
    with open(meta_path, "w") as f:
        json.dump(record, f, indent=2)
    _update_det_index(det_out_dir, record)
    return pred_path


def _copy_detection_boxes(det_out_dir, src_corr, src_sev, dst_corr, dst_sev, ann_file, img_root):
    src_json = _det_json_path(det_out_dir, src_corr, src_sev)
    dst_json = _det_json_path(det_out_dir, dst_corr, dst_sev)
    if not os.path.exists(src_json):
        return None
    os.makedirs(os.path.dirname(dst_json), exist_ok=True)
    shutil.copyfile(src_json, dst_json)
    meta_path = _det_meta_path(det_out_dir, dst_corr, dst_sev)
    vis_out = os.path.join(det_out_dir, "vis", dst_corr, f"severity_{dst_sev}")
    record = {
        "corruption": dst_corr,
        "severity": int(dst_sev),
        "pred_file": os.path.relpath(dst_json, det_out_dir),
        "ann_file": ann_file,
        "img_root": img_root,
        "copied_from": f"{src_corr}|{src_sev}",
        "vis_command": (
            "python jdet_cityscapes/tools/vis_coco_boxes.py "
            f"--ann-file \"{ann_file}\" "
            f"--img-root \"{img_root}\" "
            f"--pred-file \"{dst_json}\" "
            f"--mode both --out-dir \"{vis_out}\""
        ),
    }
    with open(meta_path, "w") as f:
        json.dump(record, f, indent=2)
    _update_det_index(det_out_dir, record)
    return dst_json


def _load_existing_results(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return {}

    normalized = {}
    for corruption, sev_dict in raw.items():
        if not isinstance(sev_dict, dict):
            continue
        normalized[corruption] = {}
        for severity, payload in sev_dict.items():
            try:
                sev_key = int(severity)
            except (TypeError, ValueError):
                continue
            normalized[corruption][sev_key] = payload
    return normalized


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

    save_det_boxes = not args.no_save_det_boxes
    det_out_dir = args.det_out_dir
    if det_out_dir is None:
        det_out_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "det_boxes")
    if save_det_boxes:
        os.makedirs(det_out_dir, exist_ok=True)
        print(f"Detection boxes will be saved to: {det_out_dir}")

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

    aggregated_results = {}
    if not args.no_resume:
        try:
            aggregated_results = _load_existing_results(args.out)
            if aggregated_results:
                done_count = sum(len(v) for v in aggregated_results.values())
                print(f"Loaded existing results from {args.out} (entries={done_count})")
        except Exception as e:
            print(f"Failed to load existing results from {args.out}: {e}")
            print("Proceeding with a fresh run.")
            aggregated_results = {}

    model = _load_model(cfg, args.checkpoint)
    saved_det_paths = {}

    for corr_i, corruption in enumerate(corruptions):
        if corruption not in aggregated_results:
            aggregated_results[corruption] = {}
        for severity in args.severities:
            if severity in aggregated_results[corruption]:
                print(f"\nSkipping {corruption} at severity {severity} (already done)")
                if save_det_boxes:
                    expected = _det_json_path(det_out_dir, corruption, severity)
                    if os.path.exists(expected):
                        saved_det_paths[(corruption, severity)] = expected
                    elif corr_i > 0 and severity == 0:
                        test_cfg = copy.deepcopy(cfg.dataset.test)
                        ann_file = test_cfg.get("anno_file", None)
                        img_root = test_cfg.get("root", None)
                        recovered = _copy_detection_boxes(
                            det_out_dir,
                            src_corr=corruptions[0],
                            src_sev=0,
                            dst_corr=corruption,
                            dst_sev=0,
                            ann_file=ann_file,
                            img_root=img_root,
                        )
                        if recovered is not None:
                            saved_det_paths[(corruption, severity)] = recovered
                            print(
                                f"[det] recovered {corruption}/severity_{severity} "
                                f"from {corruptions[0]}/severity_0"
                            )
                continue

            if corr_i > 0 and severity == 0:
                first_corr_clean = aggregated_results.get(corruptions[0], {}).get(0)
                if first_corr_clean is not None:
                    aggregated_results[corruption][0] = first_corr_clean
                    print(
                        f"\nSkipping {corruption} at severity 0 "
                        f"(reusing clean result from {corruptions[0]})"
                    )
                    if save_det_boxes:
                        test_cfg = copy.deepcopy(cfg.dataset.test)
                        ann_file = test_cfg.get("anno_file", None)
                        img_root = test_cfg.get("root", None)
                        copied = _copy_detection_boxes(
                            det_out_dir,
                            src_corr=corruptions[0],
                            src_sev=0,
                            dst_corr=corruption,
                            dst_sev=0,
                            ann_file=ann_file,
                            img_root=img_root,
                        )
                        if copied is not None:
                            saved_det_paths[(corruption, severity)] = copied
                            print(f"[det] copied clean boxes -> {copied}")
                    with open(args.out, "w") as f:
                        json.dump(aggregated_results, f, indent=2)
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
            saved_det_json = None
            if save_det_boxes:
                ann_file = test_cfg.get("anno_file", None)
                saved_det_json = _save_detection_boxes(
                    dataset=dataset,
                    results=results,
                    det_out_dir=det_out_dir,
                    corruption=corruption,
                    severity=severity,
                    ann_file=ann_file,
                )
                saved_det_paths[(corruption, severity)] = saved_det_json
                print(f"[det] saved: {saved_det_json}")
            # diagnostics: count predicted boxes and preview json
            try:
                # build json from eval save file created in evaluate
                det_json = saved_det_json
                if det_json is None or (not os.path.exists(det_json)):
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
