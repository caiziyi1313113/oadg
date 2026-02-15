#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2


def _load_coco(ann_path):
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    imgid2info = {int(im["id"]): im for im in coco.get("images", [])}
    catid2name = {int(c["id"]): c.get("name", str(c["id"])) for c in coco.get("categories", [])}
    anns_by_img = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("bbox") is None:
            continue
        anns_by_img[int(ann["image_id"])].append(ann)
    return imgid2info, catid2name, anns_by_img


def _load_preds(pred_path):
    with open(pred_path, "r", encoding="utf-8") as f:
        preds = json.load(f)
    by_img = defaultdict(list)
    for d in preds:
        if "image_id" not in d or "bbox" not in d:
            continue
        by_img[int(d["image_id"])].append(d)
    return by_img


def _xywh_to_xyxy(box):
    x, y, w, h = box
    return float(x), float(y), float(x + w), float(y + h)


def _iou_xywh(box1, box2):
    x11, y11, x12, y12 = _xywh_to_xyxy(box1)
    x21, y21, x22, y22 = _xywh_to_xyxy(box2)
    inter_x1 = max(x11, x21)
    inter_y1 = max(y11, y21)
    inter_x2 = min(x12, x22)
    inter_y2 = min(y12, y22)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area1 = max(0.0, x12 - x11) * max(0.0, y12 - y11)
    area2 = max(0.0, x22 - x21) * max(0.0, y22 - y21)
    union = area1 + area2 - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _clamp_xywh(box, width, height):
    x, y, w, h = box
    x1 = max(0, min(int(round(x)), width - 1))
    y1 = max(0, min(int(round(y)), height - 1))
    x2 = max(0, min(int(round(x + w)), width - 1))
    y2 = max(0, min(int(round(y + h)), height - 1))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def _find_fn_gts(gt_anns, pred_anns, iou_thr):
    gt_by_cat = defaultdict(list)
    for i, ann in enumerate(gt_anns):
        gt_by_cat[int(ann["category_id"])].append(i)

    pred_by_cat = defaultdict(list)
    for pred in pred_anns:
        pred_by_cat[int(pred["category_id"])].append(pred)

    matched_gt = [False] * len(gt_anns)

    for cat_id, gt_indices in gt_by_cat.items():
        preds = pred_by_cat.get(cat_id, [])
        if not preds:
            continue
        preds = sorted(preds, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        used = [False] * len(gt_indices)
        for pred in preds:
            best_j = -1
            best_iou = iou_thr
            for j, gt_i in enumerate(gt_indices):
                if used[j]:
                    continue
                iou = _iou_xywh(pred["bbox"], gt_anns[gt_i]["bbox"])
                if iou >= best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0:
                used[best_j] = True
        for j, gt_i in enumerate(gt_indices):
            if used[j]:
                matched_gt[gt_i] = True

    return [i for i, ok in enumerate(matched_gt) if not ok]


def _draw_fn_boxes(img, fn_anns, catid2name):
    out = img
    h, w = out.shape[:2]
    for ann in fn_anns:
        x1, y1, x2, y2 = _clamp_xywh(ann["bbox"], w, h)
        cid = int(ann["category_id"])
        cname = catid2name.get(cid, str(cid))
        color = (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"FN:{cname}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y1 - th - baseline - 4)
        cv2.rectangle(out, (x1, y_text), (x1 + tw + 4, y_text + th + baseline + 4), color, -1)
        cv2.putText(
            out,
            text,
            (x1 + 2, y_text + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export FN-only images from COCO GT + prediction json."
    )
    parser.add_argument("--ann-file", required=True, help="COCO annotation json")
    parser.add_argument("--img-root", required=True, help="Image root directory")
    parser.add_argument("--pred-file", required=True, help="COCO detection json list")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--score-thr", type=float, default=0.05, help="Prediction score threshold")
    parser.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold to match GT and pred")
    parser.add_argument("--max-det", type=int, default=300, help="Max predictions per image after score filter")
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img-ids", type=int, nargs="*", default=None, help="Optional image id list")
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Save every image (if false, save only images that contain FN boxes)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    imgid2info, catid2name, anns_by_img = _load_coco(args.ann_file)
    preds_by_img = _load_preds(args.pred_file)

    img_ids = list(imgid2info.keys())
    if args.img_ids:
        img_ids = [i for i in args.img_ids if i in imgid2info]
    if args.max_images and args.max_images > 0 and len(img_ids) > args.max_images:
        random.seed(args.seed)
        img_ids = random.sample(img_ids, args.max_images)

    img_root = Path(args.img_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    missing = 0
    total_fn = 0
    summary = []

    for img_id in img_ids:
        info = imgid2info[img_id]
        file_name = info.get("file_name")
        if not file_name:
            missing += 1
            continue

        img_path = img_root / file_name
        if not img_path.exists():
            missing += 1
            continue

        gt_anns = anns_by_img.get(img_id, [])
        pred_anns = preds_by_img.get(img_id, [])
        pred_anns = [p for p in pred_anns if float(p.get("score", 0.0)) >= args.score_thr]
        pred_anns = sorted(pred_anns, key=lambda x: float(x.get("score", 0.0)), reverse=True)[: args.max_det]

        fn_indices = _find_fn_gts(gt_anns, pred_anns, args.iou_thr)
        fn_anns = [gt_anns[i] for i in fn_indices]

        if not args.save_all and len(fn_anns) == 0:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            missing += 1
            continue

        vis = _draw_fn_boxes(img, fn_anns, catid2name)
        out_path = out_dir / file_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)
        saved += 1
        total_fn += len(fn_anns)

        summary.append(
            {
                "image_id": int(img_id),
                "file_name": file_name,
                "fn_count": len(fn_anns),
                "fn_objects": [
                    {
                        "category_id": int(a["category_id"]),
                        "category_name": catid2name.get(int(a["category_id"]), str(a["category_id"])),
                        "bbox": a["bbox"],
                    }
                    for a in fn_anns
                ],
            }
        )

    summary_path = out_dir / "fn_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ann_file": args.ann_file,
                "pred_file": args.pred_file,
                "score_thr": args.score_thr,
                "iou_thr": args.iou_thr,
                "max_det": args.max_det,
                "saved_images": saved,
                "total_fn_boxes": total_fn,
                "records": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] saved {saved} images to: {out_dir}")
    print(f"[OK] total FN boxes: {total_fn}")
    print(f"[OK] summary json: {summary_path}")
    if missing:
        print(f"[WARN] missing mapping/file for {missing} image_ids")


if __name__ == "__main__":
    main()
