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


def _clamp_bbox_xywh(x, y, w, h, W, H):
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(0, min(x2, W - 1))
    y2 = max(0, min(y2, H - 1))
    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2


_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 128, 0),
    (128, 0, 255), (0, 255, 255), (255, 0, 255), (128, 255, 0),
    (0, 128, 128), (128, 128, 0), (128, 0, 0), (0, 0, 128),
    (0, 128, 0), (128, 0, 128), (64, 64, 255), (255, 64, 64),
]


def _color_for_cid(catid):
    if catid is None or catid < 0:
        return (255, 255, 255)
    return _PALETTE[catid % len(_PALETTE)]


def _draw_boxes(img, dets, catid2name, color, score_thr=0.0, max_det=200, prefix="", fixed_color=False):
    H, W = img.shape[:2]
    out = img
    # sort by score if present
    def _score(d):
        return float(d.get("score", 1.0))
    dets = [d for d in dets if _score(d) >= score_thr]
    dets.sort(key=_score, reverse=True)
    dets = dets[:max_det]
    for d in dets:
        x, y, w, h = d["bbox"]
        cid = int(d.get("category_id", -1))
        cname = catid2name.get(cid, str(cid))
        score = d.get("score", None)
        box_color = color if fixed_color else _color_for_cid(cid)
        x1, y1, x2, y2 = _clamp_bbox_xywh(x, y, w, h, W, H)
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
        if score is None:
            text = f"{prefix}{cname}"
        else:
            text = f"{prefix}{cname}:{float(score):.3f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y1 - th - baseline - 4)
        cv2.rectangle(out, (x1, y_text), (x1 + tw + 4, y_text + th + baseline + 4), box_color, -1)
        cv2.putText(out, text, (x1 + 2, y_text + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description="Visualize COCO GT/pred boxes")
    ap.add_argument("--ann-file", required=True, help="COCO annotation json")
    ap.add_argument("--img-root", required=True, help="image root (img_prefix)")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--mode", choices=["gt", "pred", "both"], default="gt")
    ap.add_argument("--pred-file", help="COCO detection json list (required for pred/both)")
    ap.add_argument("--score-thr", type=float, default=0.05)
    ap.add_argument("--max-det", type=int, default=200)
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-ids", type=int, nargs="*", default=None, help="optional list of image ids")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--fixed-color", action="store_true",
                    help="use single color per mode (disable per-class colors)")
    args = ap.parse_args()

    if args.mode in ("pred", "both") and not args.pred_file:
        raise SystemExit("--pred-file is required for mode=pred/both")

    imgid2info, catid2name, anns_by_img = _load_coco(args.ann_file)
    preds_by_img = _load_preds(args.pred_file) if args.pred_file else {}

    img_ids = list(imgid2info.keys())
    if args.img_ids:
        img_ids = [i for i in args.img_ids if i in imgid2info]

    if args.max_images and args.max_images > 0 and len(img_ids) > args.max_images:
        random.seed(args.seed)
        img_ids = random.sample(img_ids, args.max_images)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_root = Path(args.img_root)

    missing = 0
    done = 0
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
        out_path = out_dir / file_name
        if args.skip_existing and out_path.exists():
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_path))
        if img is None:
            missing += 1
            continue

        if args.mode in ("gt", "both"):
            gt = anns_by_img.get(img_id, [])
            img = _draw_boxes(img, gt, catid2name, color=(0, 0, 255),
                              prefix="GT:", fixed_color=args.fixed_color)
        if args.mode in ("pred", "both"):
            dets = preds_by_img.get(img_id, [])
            img = _draw_boxes(img, dets, catid2name, color=(0, 255, 0),
                              score_thr=args.score_thr, max_det=args.max_det,
                              prefix="P:", fixed_color=args.fixed_color)

        cv2.imwrite(str(out_path), img)
        done += 1

    print(f"[OK] saved {done} images to: {out_dir}")
    if missing:
        print(f"[WARN] missing mapping/file for {missing} image_ids (check ann_file and img_root).")


if __name__ == "__main__":
    main()
