import argparse
import json
from pathlib import Path


CITYSCAPES_C_BENCHMARK = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]

DOMAIN_GROUPS = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}


def _get_metric(entry, metric_key):
    if entry is None:
        return None
    if isinstance(entry, dict) and "bbox" in entry:
        entry = entry["bbox"]
    if isinstance(entry, dict):
        return entry.get(metric_key, None)
    return None


def compute_metrics(data, metric_key, severities, corruptions=None):
    corruptions = corruptions or list(data.keys())
    missing = []
    per_corr = {}
    all_values = []

    for corr in corruptions:
        corr_data = data.get(corr, {})
        vals = []
        for s in severities:
            entry = corr_data.get(str(s))
            val = _get_metric(entry, metric_key)
            if val is None:
                missing.append((corr, s))
                continue
            vals.append(float(val))
            all_values.append(float(val))
        per_corr[corr] = {
            "mean": sum(vals) / len(vals) if vals else None,
            "values": {str(s): _get_metric(corr_data.get(str(s)), metric_key) for s in severities},
        }

    overall = sum(all_values) / len(all_values) if all_values else None
    return per_corr, overall, missing


def compute_domains(per_corr, available_corrs):
    domain_out = {}
    for domain, corr_list in DOMAIN_GROUPS.items():
        corr_list = [c for c in corr_list if c in available_corrs]
        if not corr_list:
            continue
        vals = [per_corr[c]["mean"] for c in corr_list if per_corr[c]["mean"] is not None]
        domain_out[domain] = {
            "corruptions": corr_list,
            "mean": sum(vals) / len(vals) if vals else None,
        }
    return domain_out


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Cityscapes-C metrics (mPC) from robustness JSON.")
    parser.add_argument("input", help="robustness json file (e.g., robustness_ckpt17.json)")
    parser.add_argument("--output", help="output json file path")
    parser.add_argument("--metric", default="bbox_mAP", help="metric key (default: bbox_mAP)")
    parser.add_argument(
        "--severities",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="severity levels to aggregate (default: 1..5)",
    )
    parser.add_argument(
        "--corruptions",
        type=str,
        nargs="+",
        default=CITYSCAPES_C_BENCHMARK,
        help="corruption list to use (default: Cityscapes-C benchmark 15)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)
    out = Path(args.output) if args.output else inp.with_suffix(".mpc.json")

    data = json.loads(inp.read_text())

    per_corr, overall, missing = compute_metrics(
        data, args.metric, args.severities, corruptions=args.corruptions
    )
    domains = compute_domains(per_corr, set(args.corruptions))

    result = {
        "input": str(inp),
        "metric": args.metric,
        "severities": args.severities,
        "corruptions": per_corr,
        "domains": domains,
        "mPC": overall,
        "missing": missing,
    }

    out.write_text(json.dumps(result, indent=2))
    print(f"Saved: {out}")
    if overall is not None:
        print(f"mPC({args.metric}) = {overall:.6f}")
    if domains:
        for name, stats in domains.items():
            print(f"{name}: {stats['mean']}")


if __name__ == "__main__":
    main()
