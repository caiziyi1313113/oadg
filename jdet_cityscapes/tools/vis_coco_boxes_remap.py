#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False
    from PIL import Image, ImageDraw, ImageFont


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
    return by_img, preds


def _iou_xywh(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    iw = max(0.0, xb - xa)
    ih = max(0.0, yb - ya)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = w1 * h1 + w2 * h2 - inter
    if union <= 0.0:
        return 0.0
    return inter / union


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
    if _HAS_CV2:
        return _draw_boxes_cv2(
            img, dets, catid2name, color, score_thr=score_thr,
            max_det=max_det, prefix=prefix, fixed_color=fixed_color
        )
    return _draw_boxes_pil(
        img, dets, catid2name, color, score_thr=score_thr,
        max_det=max_det, prefix=prefix, fixed_color=fixed_color
    )


def _draw_boxes_cv2(img, dets, catid2name, color, score_thr=0.0, max_det=200, prefix="", fixed_color=False):
    H, W = img.shape[:2]
    out = img

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


def _bgr_to_rgb(c):
    return (int(c[2]), int(c[1]), int(c[0]))


def _draw_boxes_pil(img, dets, catid2name, color, score_thr=0.0, max_det=200, prefix="", fixed_color=False):
    out = img
    W, H = out.size
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

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
        box_color_bgr = color if fixed_color else _color_for_cid(cid)
        box_color = _bgr_to_rgb(box_color_bgr)
        x1, y1, x2, y2 = _clamp_bbox_xywh(x, y, w, h, W, H)
        try:
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
        except TypeError:
            draw.rectangle([x1, y1, x2, y2], outline=box_color)
        if score is None:
            text = f"{prefix}{cname}"
        else:
            text = f"{prefix}{cname}:{float(score):.3f}"
        tw, th = draw.textsize(text, font=font)
        y_text = max(0, y1 - th - 4)
        draw.rectangle([x1, y_text, x1 + tw + 4, y_text + th + 4], fill=box_color)
        draw.text((x1 + 2, y_text + 2), text, fill=(0, 0, 0), font=font)
    return out


def _topk_preds_for_fit(preds_by_img, fit_score_thr, fit_topk):
    out = {}
    for img_id, dets in preds_by_img.items():
        ds = [d for d in dets if float(d.get("score", 1.0)) >= fit_score_thr]
        ds.sort(key=lambda x: float(x.get("score", 1.0)), reverse=True)
        out[img_id] = ds[:fit_topk]
    return out


def _pair_score(preds, gts, class_aware=True):
    if not preds or not gts:
        return 0.0
    if class_aware:
        gts_by_cat = defaultdict(list)
        for g in gts:
            gts_by_cat[int(g.get("category_id", -1))].append(g["bbox"])
    else:
        gboxes = [g["bbox"] for g in gts]

    weighted_sum = 0.0
    weight = 0.0
    for p in preds:
        pb = p["bbox"]
        pw = float(p.get("score", 1.0))
        cid = int(p.get("category_id", -1))
        if class_aware:
            candidates = gts_by_cat.get(cid, [])
        else:
            candidates = gboxes
        best = 0.0
        for gb in candidates:
            v = _iou_xywh(pb, gb)
            if v > best:
                best = v
        weighted_sum += pw * best
        weight += pw
    if weight <= 0.0:
        return 0.0
    return weighted_sum / weight


def _compute_similarity_matrix(pred_ids, ann_ids, fit_preds_by_img, anns_by_img, class_aware=True):
    matrix = []
    for pid in pred_ids:
        preds = fit_preds_by_img.get(pid, [])
        row = []
        for aid in ann_ids:
            gts = anns_by_img.get(aid, [])
            row.append(_pair_score(preds, gts, class_aware=class_aware))
        matrix.append(row)
    return matrix


def _greedy_assignment(sim_matrix):
    n_rows = len(sim_matrix)
    n_cols = len(sim_matrix[0]) if n_rows > 0 else 0
    used_r = set()
    used_c = set()
    assigned = {}

    pairs = []
    for r in range(n_rows):
        row = sim_matrix[r]
        for c in range(n_cols):
            pairs.append((row[c], r, c))
    pairs.sort(key=lambda x: x[0], reverse=True)

    for score, r, c in pairs:
        if r in used_r or c in used_c:
            continue
        assigned[r] = c
        used_r.add(r)
        used_c.add(c)
        if len(assigned) == n_rows:
            break

    if len(assigned) < n_rows:
        all_cols = set(range(n_cols))
        remaining_cols = list(all_cols - used_c)
        for r in range(n_rows):
            if r in assigned:
                continue
            if remaining_cols:
                c = max(remaining_cols, key=lambda x: sim_matrix[r][x])
                remaining_cols.remove(c)
            else:
                c = max(range(n_cols), key=lambda x: sim_matrix[r][x]) if n_cols > 0 else 0
            assigned[r] = c
    return assigned


def _hungarian_assignment(sim_matrix):
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return None
    cost = -np.array(sim_matrix, dtype=float)
    row_ind, col_ind = linear_sum_assignment(cost)
    out = {}
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        out[r] = c
    return out


def _mean_score_for_map(pred_ids, ann_id_to_col, sim_matrix, id_map):
    vals = []
    for r, pid in enumerate(pred_ids):
        aid = id_map.get(pid, None)
        if aid is None:
            continue
        c = ann_id_to_col.get(aid, None)
        if c is None:
            continue
        vals.append(sim_matrix[r][c])
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def infer_image_id_map(
    imgid2info,
    anns_by_img,
    preds_by_img,
    fit_score_thr=0.5,
    fit_topk=25,
    class_aware=True,
):
    all_pred_ids = sorted(preds_by_img.keys())
    ann_ids = sorted(imgid2info.keys())
    ann_id_to_col = {aid: i for i, aid in enumerate(ann_ids)}

    fit_preds_by_img = _topk_preds_for_fit(preds_by_img, fit_score_thr=fit_score_thr, fit_topk=fit_topk)
    active_pred_ids = [pid for pid in all_pred_ids if fit_preds_by_img.get(pid)]
    inactive_pred_ids = [pid for pid in all_pred_ids if not fit_preds_by_img.get(pid)]

    sim_matrix = _compute_similarity_matrix(
        active_pred_ids, ann_ids, fit_preds_by_img, anns_by_img, class_aware=class_aware
    )

    assigned = _hungarian_assignment(sim_matrix)
    used_hungarian = assigned is not None
    if assigned is None:
        assigned = _greedy_assignment(sim_matrix)

    id_map = {}
    for r, pid in enumerate(active_pred_ids):
        c = assigned.get(r, None)
        if c is None:
            id_map[pid] = pid
        else:
            id_map[pid] = ann_ids[c]

    # for rows with no confident preds, keep identity as fallback
    for pid in inactive_pred_ids:
        id_map[pid] = pid

    identity_map = {pid: pid for pid in all_pred_ids}
    remap_mean = _mean_score_for_map(active_pred_ids, ann_id_to_col, sim_matrix, id_map)
    identity_mean = _mean_score_for_map(active_pred_ids, ann_id_to_col, sim_matrix, identity_map)

    stats = {
        "num_pred_image_ids": len(all_pred_ids),
        "num_active_pred_image_ids": len(active_pred_ids),
        "num_ann_image_ids": len(ann_ids),
        "fit_score_thr": fit_score_thr,
        "fit_topk": fit_topk,
        "class_aware": class_aware,
        "solver": "hungarian(scipy)" if used_hungarian else "greedy",
        "identity_mean_score": identity_mean,
        "remap_mean_score": remap_mean,
        "improvement": remap_mean - identity_mean,
    }
    return id_map, stats


def _load_id_map(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "id_map" in raw and isinstance(raw["id_map"], dict):
        raw = raw["id_map"]
    out = {}
    for k, v in raw.items():
        out[int(k)] = int(v)
    return out


def _save_id_map(path, id_map, stats):
    payload = {
        "id_map": {str(k): int(v) for k, v in sorted(id_map.items())},
        "stats": stats,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _remap_preds(preds, id_map):
    remapped = []
    for d in preds:
        if "image_id" not in d:
            continue
        old_id = int(d["image_id"])
        new_id = int(id_map.get(old_id, old_id))
        dd = dict(d)
        dd["image_id"] = new_id
        remapped.append(dd)
    return remapped


def parse_args():
    ap = argparse.ArgumentParser(
        description="Visualize COCO boxes with optional auto image_id remapping for mismatched prediction json."
    )
    ap.add_argument("--ann-file", required=True, help="COCO annotation json")
    ap.add_argument("--img-root", required=True, help="image root (img_prefix)")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--mode", choices=["gt", "pred", "both"], default="pred")
    ap.add_argument("--pred-file", help="COCO detection json list (required for pred/both)")
    ap.add_argument("--score-thr", type=float, default=0.05)
    ap.add_argument("--max-det", type=int, default=200)
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-ids", type=int, nargs="*", default=None, help="optional list of image ids")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--fixed-color", action="store_true",
                    help="use single color per mode (disable per-class colors)")

    ap.add_argument("--auto-remap", action="store_true",
                    help="infer image_id mapping from pred to ann via box matching")
    ap.add_argument("--force-remap", action="store_true",
                    help="use inferred remap even if improvement is small")
    ap.add_argument("--fit-score-thr", type=float, default=0.5,
                    help="score threshold used only for remap fitting")
    ap.add_argument("--fit-topk", type=int, default=25,
                    help="top-k preds per image used only for remap fitting")
    ap.add_argument("--fit-class-agnostic", action="store_true",
                    help="ignore class id during remap fitting")
    ap.add_argument("--id-map-in", help="load fixed image_id map json (key: old_id, value: new_id)")
    ap.add_argument("--id-map-out", help="save inferred/used image_id map json")
    ap.add_argument("--save-remapped-pred", help="optional path to dump remapped prediction json")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.mode in ("pred", "both") and not args.pred_file:
        raise SystemExit("--pred-file is required for mode=pred/both")

    imgid2info, catid2name, anns_by_img = _load_coco(args.ann_file)
    preds_by_img, raw_preds = _load_preds(args.pred_file) if args.pred_file else ({}, [])

    id_map = {}
    remap_stats = None
    if args.mode in ("pred", "both"):
        if args.id_map_in:
            id_map = _load_id_map(args.id_map_in)
            print(f"[map] loaded image_id map from: {args.id_map_in} (size={len(id_map)})")
        elif args.auto_remap:
            id_map, remap_stats = infer_image_id_map(
                imgid2info=imgid2info,
                anns_by_img=anns_by_img,
                preds_by_img=preds_by_img,
                fit_score_thr=args.fit_score_thr,
                fit_topk=args.fit_topk,
                class_aware=not args.fit_class_agnostic,
            )
            print(f"[map] solver={remap_stats['solver']}")
            print(f"[map] active_ids={remap_stats['num_active_pred_image_ids']}, ann_ids={remap_stats['num_ann_image_ids']}")
            print(f"[map] identity_mean={remap_stats['identity_mean_score']:.6f}")
            print(f"[map] remap_mean={remap_stats['remap_mean_score']:.6f}")
            print(f"[map] improvement={remap_stats['improvement']:.6f}")
            if not args.force_remap and remap_stats["improvement"] <= 1e-3:
                print("[map] improvement too small, fallback to identity mapping")
                id_map = {}

        remapped_preds = _remap_preds(raw_preds, id_map) if id_map else raw_preds
        remapped_by_img = defaultdict(list)
        for d in remapped_preds:
            remapped_by_img[int(d["image_id"])].append(d)
        preds_by_img = remapped_by_img

        if args.id_map_out:
            payload_stats = remap_stats if remap_stats is not None else {"note": "provided/existing map used"}
            _save_id_map(args.id_map_out, id_map, payload_stats)
            print(f"[map] saved image_id map to: {args.id_map_out}")
        if args.save_remapped_pred:
            p = Path(args.save_remapped_pred)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(remapped_preds), encoding="utf-8")
            print(f"[map] saved remapped predictions to: {args.save_remapped_pred}")

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

        if _HAS_CV2:
            img = cv2.imread(str(img_path))
            if img is None:
                missing += 1
                continue
        else:
            try:
                img = Image.open(str(img_path)).convert("RGB")
            except Exception:
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

        if _HAS_CV2:
            cv2.imwrite(str(out_path), img)
        else:
            img.save(str(out_path))
        done += 1

    print(f"[OK] saved {done} images to: {out_dir}")
    if missing:
        print(f"[WARN] missing mapping/file for {missing} image_ids (check ann_file and img_root).")


if __name__ == "__main__":
    main()
