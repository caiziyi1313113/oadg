import argparse
import json
import os


CITYSCAPES_C_BENCHMARK = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}


def _detect_metric_style(sample_task):
    """Return mapping from canonical metric names to keys in file."""
    if "bbox_mAP" in sample_task:
        return {
            "AP": "bbox_mAP",
            "AP50": "bbox_mAP_50",
            "AP75": "bbox_mAP_75",
            "APs": "bbox_mAP_s",
            "APm": "bbox_mAP_m",
            "APl": "bbox_mAP_l",
        }
    # standard COCO-style keys
    return {
        "AP": "AP",
        "AP50": "AP50",
        "AP75": "AP75",
        "APs": "APs",
        "APm": "APm",
        "APl": "APl",
    }


def _mean(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return 0.0
    return float(sum(vals)) / float(len(vals))


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def compute_cityscapes_c_metrics(data, task="bbox", metrics=None):
    # detect metric keys
    first_corr = next(iter(data))
    first_sev = next(iter(data[first_corr]))
    sample_task = data[first_corr][first_sev].get(task, {})
    metric_map = _detect_metric_style(sample_task)

    if metrics is None:
        metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

    # clean (severity 0) metrics
    clean = {}
    if "0" in data[first_corr]:
        for m in metrics:
            key = metric_map[m]
            clean[m] = data[first_corr]["0"][task].get(key, 0.0)

    # per corruption (mean over severity 1..5)
    per_corruption = {}
    for corr in data:
        sev_vals = data[corr]
        per_corruption[corr] = {}
        for m in metrics:
            key = metric_map[m]
            vals = []
            for s in sorted(sev_vals.keys(), key=lambda x: int(x)):
                if int(s) == 0:
                    continue
                vals.append(sev_vals[s][task].get(key, 0.0))
            per_corruption[corr][m] = _mean(vals)

    # per domain (mean over corruptions + severities)
    per_domain = {}
    for domain, corr_list in CITYSCAPES_C_BENCHMARK.items():
        per_domain[domain] = {}
        for m in metrics:
            key = metric_map[m]
            vals = []
            for corr in corr_list:
                if corr not in data:
                    continue
                for s in data[corr]:
                    if int(s) == 0:
                        continue
                    vals.append(data[corr][s][task].get(key, 0.0))
            per_domain[domain][m] = _mean(vals)

    # overall mPC across all benchmark corruptions + severities
    all_corrs = [c for v in CITYSCAPES_C_BENCHMARK.values() for c in v]
    overall = {}
    for m in metrics:
        key = metric_map[m]
        vals = []
        for corr in all_corrs:
            if corr not in data:
                continue
            for s in data[corr]:
                if int(s) == 0:
                    continue
                vals.append(data[corr][s][task].get(key, 0.0))
        overall[m] = _mean(vals)

    return {
        "clean": clean,
        "per_corruption": per_corruption,
        "per_domain": per_domain,
        "mpc": overall,
        "metrics": metrics,
        "task": task,
    }


def _print_block(title, metrics, values):
    print(f"\n{title}")
    for m in metrics:
        v = values.get(m, 0.0)
        print(f"{m:6s}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Cityscapes-C domain metrics")
    parser.add_argument("input", help="robustness json file")
    parser.add_argument("--out", help="output json file")
    parser.add_argument("--task", default="bbox")
    parser.add_argument("--metrics", nargs="+", default=None)
    args = parser.parse_args()

    data = _load_json(args.input)
    result = compute_cityscapes_c_metrics(data, task=args.task, metrics=args.metrics)

    metrics = result["metrics"]
    _print_block("Clean (severity 0)", metrics, result["clean"])
    _print_block("mPC (all corruptions)", metrics, result["mpc"])
    for domain in CITYSCAPES_C_BENCHMARK:
        _print_block(f"Domain: {domain}", metrics, result["per_domain"][domain])

    out_path = args.out
    if not out_path:
        base, _ = os.path.splitext(args.input)
        out_path = base + "_domain_metrics.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
