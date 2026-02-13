import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_coco(ann_path):
    data = json.loads(Path(ann_path).read_text())
    images = {img["id"]: img for img in data.get("images", [])}
    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    anns_by_img = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        anns_by_img.setdefault(img_id, []).append(ann)
    return images, categories, anns_by_img


def _load_preds(pred_path):
    preds = json.loads(Path(pred_path).read_text())
    preds_by_img = {}
    for p in preds:
        img_id = p["image_id"]
        preds_by_img.setdefault(img_id, []).append(p)
    return preds_by_img


def _xywh_to_xyxy(b):
    x, y, w, h = b
    return x, y, x + w, y + h


def _get_font(size=14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_boxes(draw, boxes, labels, color, scores=None, prefix=""):
    font = _get_font()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = _xywh_to_xyxy(box)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"{prefix}{labels[i]}"
        if scores is not None:
            text += f" {scores[i]:.2f}"
        draw.text((x1 + 2, y1 + 2), text, fill=color, font=font)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Cityscapes COCO boxes")
    parser.add_argument("--ann", required=True, help="COCO annotation json")
    parser.add_argument("--img-root", required=True, help="image root directory")
    parser.add_argument("--out-dir", required=True, help="output directory")
    parser.add_argument("--pred", default=None, help="prediction json (COCO bbox)")
    parser.add_argument("--mode", choices=["gt", "pred", "both"], default="gt")
    parser.add_argument("--score-thr", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0, help="limit number of images (0 = all)")
    return parser.parse_args()


def main():
    args = parse_args()
    images, categories, anns_by_img = _load_coco(args.ann)
    preds_by_img = _load_preds(args.pred) if args.pred else {}

    img_root = Path(args.img_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_id, info in images.items():
        rel = info.get("file_name", "")
        if not rel:
            continue
        img_path = img_root / Path(rel)
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        if args.mode in ("gt", "both"):
            gt_anns = anns_by_img.get(img_id, [])
            gt_boxes = [a["bbox"] for a in gt_anns]
            gt_labels = [categories.get(a["category_id"], str(a["category_id"])) for a in gt_anns]
            _draw_boxes(draw, gt_boxes, gt_labels, color="green", prefix="gt:")

        if args.mode in ("pred", "both") and preds_by_img:
            preds = preds_by_img.get(img_id, [])
            preds = [p for p in preds if p.get("score", 1.0) >= args.score_thr]
            preds = sorted(preds, key=lambda x: x.get("score", 0.0), reverse=True)[: args.topk]
            pred_boxes = [p["bbox"] for p in preds]
            pred_labels = [categories.get(p["category_id"], str(p["category_id"])) for p in preds]
            pred_scores = [p.get("score", 0.0) for p in preds]
            _draw_boxes(draw, pred_boxes, pred_labels, color="red", scores=pred_scores, prefix="pred:")

        out_path = out_dir / Path(rel)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        count += 1
        if args.limit and count >= args.limit:
            break

    print(f"Saved {count} images to {out_dir}")


if __name__ == "__main__":
    main()
