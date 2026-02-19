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

SIZE_ORDER = ["small", "medium", "large"]


def _detect_size_metric_keys(data):
    # robustly detect whether file uses bbox_mAP_* or AP* naming.
    first_corr = next(iter(data.values()))
    first_sev = next(iter(first_corr.values()))
    sample_task = first_sev.get("bbox", first_sev)
    if not isinstance(sample_task, dict):
        raise ValueError("Invalid robustness entry format.")
    if "bbox_mAP_s" in sample_task:
        return {
            "small": "bbox_mAP_s",
            "medium": "bbox_mAP_m",
            "large": "bbox_mAP_l",
        }
    return {
        "small": "APs",
        "medium": "APm",
        "large": "APl",
    }


def _get_metric(entry, metric_key):
    if entry is None:
        return None
    if isinstance(entry, dict) and "bbox" in entry:
        entry = entry["bbox"]
    if isinstance(entry, dict):
        return entry.get(metric_key, None)
    return None


def _compute_per_corruption(data, metric_key, severities, corruptions):
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
            val = float(val)
            vals.append(val)
            all_values.append(val)

        per_corr[corr] = {
            "mean": sum(vals) / len(vals) if vals else None,
            "values": {str(s): _get_metric(corr_data.get(str(s)), metric_key) for s in severities},
        }

    overall = sum(all_values) / len(all_values) if all_values else None
    return per_corr, overall, missing


def _compute_clean(data, metric_key, corruption_list, clean_severity):
    per_corr = {}
    vals = []
    missing = []
    sev_key = str(clean_severity)

    for corr in corruption_list:
        corr_data = data.get(corr, {})
        val = _get_metric(corr_data.get(sev_key), metric_key)
        per_corr[corr] = val
        if val is None:
            missing.append((corr, clean_severity))
            continue
        vals.append(float(val))

    overall = sum(vals) / len(vals) if vals else None
    return per_corr, overall, missing


def _compute_domains(per_corr, available_corrs):
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


def _build_size_result(data, size_name, metric_key, severities, clean_severity, corruptions):
    per_corr, mpc, missing = _compute_per_corruption(
        data, metric_key, severities, corruptions
    )
    clean_per_corr, clean_overall, missing_clean = _compute_clean(
        data, metric_key, corruptions, clean_severity
    )
    domains = _compute_domains(per_corr, set(corruptions))
    if clean_overall is None or clean_overall == 0:
        rpc = None
    else:
        rpc = mpc / clean_overall if mpc is not None else None

    return {
        "size": size_name,
        "metric": metric_key,
        "severities": severities,
        "clean_severity": clean_severity,
        "P": clean_overall,
        "mPC": mpc,
        "rPC": rpc,
        "clean": {
            "overall": clean_overall,
            "per_corruption": clean_per_corr,
        },
        "corruptions": per_corr,
        "domains": domains,
        "missing": missing,
        "missing_clean": missing_clean,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Cityscapes-C metrics for different object sizes (small/medium/large)."
    )
    parser.add_argument("input", help="robustness json file (e.g., robustness_last.json)")
    parser.add_argument("--output", help="output json file path")
    parser.add_argument(
        "--sizes",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "small", "medium", "large"],
        help="which object size groups to compute (default: all)",
    )
    parser.add_argument(
        "--severities",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="severity levels used for mPC aggregation (default: 1..5)",
    )
    parser.add_argument(
        "--clean-severity",
        type=int,
        default=0,
        help="severity index used as clean metric P (default: 0)",
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
    out = Path(args.output) if args.output else inp.with_suffix(".size.json")
    data = json.loads(inp.read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid or empty robustness file: {inp}")
    first_corr_data = next(iter(data.values()))
    if not isinstance(first_corr_data, dict) or not first_corr_data:
        raise ValueError(
            "Invalid input format. Please use raw robustness json from test_robustness.py "
            "(e.g., robustness_last.json), not summarized mpc files."
        )
    first_sev_data = next(iter(first_corr_data.values()))
    if not isinstance(first_sev_data, dict):
        raise ValueError(
            "Invalid input format. Please use raw robustness json from test_robustness.py "
            "(e.g., robustness_last.json), not summarized mpc files."
        )

    metric_keys = _detect_size_metric_keys(data)
    size_names = SIZE_ORDER if "all" in args.sizes else args.sizes

    result = {
        "input": str(inp),
        "sizes": size_names,
        "severities": args.severities,
        "clean_severity": args.clean_severity,
        "corruptions": args.corruptions,
        "metrics": {},
    }

    for size_name in size_names:
        metric_key = metric_keys[size_name]
        result["metrics"][size_name] = _build_size_result(
            data,
            size_name=size_name,
            metric_key=metric_key,
            severities=args.severities,
            clean_severity=args.clean_severity,
            corruptions=args.corruptions,
        )

    out.write_text(json.dumps(result, indent=2))
    print(f"Saved: {out}")

    for size_name in size_names:
        item = result["metrics"][size_name]
        p_val = item["P"]
        mpc_val = item["mPC"]
        rpc_val = item["rPC"]
        p_str = "None" if p_val is None else f"{p_val:.6f}"
        mpc_str = "None" if mpc_val is None else f"{mpc_val:.6f}"
        rpc_str = "None" if rpc_val is None else f"{rpc_val:.6f}"
        print(f"{size_name}: P={p_str} mPC={mpc_str} rPC={rpc_str}")


if __name__ == "__main__":
    main()
