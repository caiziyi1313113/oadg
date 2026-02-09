import json
import os
from argparse import ArgumentParser

import numpy as np


def _print_coco_results(results):
    def _print(result, ap=1, iou_thr=None, area_rng="all", max_dets=100):
        title_str = "Average Precision" if ap == 1 else "Average Recall"
        type_str = "(AP)" if ap == 1 else "(AR)"
        iou_str = "0.50:0.95" if iou_thr is None else f"{iou_thr:0.2f}"
        i_str = f" {title_str:<18} {type_str} @[ IoU={iou_str:<9} | "
        i_str += f"area={area_rng:>6s} | maxDets={max_dets:>3d} ] = {result:0.3f}"
        print(i_str)

    _print(results[0], 1)
    _print(results[1], 1, iou_thr=0.5)
    _print(results[2], 1, iou_thr=0.75)
    _print(results[3], 1, area_rng="small")
    _print(results[4], 1, area_rng="medium")
    _print(results[5], 1, area_rng="large")
    _print(results[6], 0, max_dets=1)
    _print(results[7], 0, max_dets=10)
    _print(results[8], 0)
    _print(results[9], 0, area_rng="small")
    _print(results[10], 0, area_rng="medium")
    _print(results[11], 0, area_rng="large")


def _load_results(filename):
    with open(filename, "r") as f:
        return json.load(f)


def get_coco_style_results(filename, task="bbox", metric=None, prints="mPC", aggregate="benchmark"):
    assert aggregate in ["benchmark", "all"]
    if prints == "all":
        prints = ["P", "mPC", "rPC"]
    elif isinstance(prints, str):
        prints = [prints]
    for p in prints:
        assert p in ["P", "mPC", "rPC"]

    if metric is None:
        metrics = [
            "AP", "AP50", "AP75", "APs", "APm", "APl",
            "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
        ]
    elif isinstance(metric, list):
        metrics = metric
    else:
        metrics = [metric]

    for metric_name in metrics:
        assert metric_name in [
            "AP", "AP50", "AP75", "APs", "APm", "APl",
            "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
        ]

    eval_output = _load_results(filename)

    num_distortions = len(list(eval_output.keys()))
    results = np.zeros((num_distortions, 6, len(metrics)), dtype="float32")

    for corr_i, distortion in enumerate(eval_output):
        for severity in eval_output[distortion]:
            for metric_j, metric_name in enumerate(metrics):
                mAP = eval_output[distortion][severity][task][metric_name]
                results[corr_i, severity, metric_j] = mAP

    P = results[0, 0, :]
    if aggregate == "benchmark":
        mPC = np.mean(results[:15, 1:, :], axis=(0, 1))
    else:
        mPC = np.mean(results[:, 1:, :], axis=(0, 1))
    rPC = mPC / P

    print(f"\nmodel: {os.path.basename(filename)}")
    if metric is None:
        if "P" in prints:
            print(f"Performance on Clean Data [P] ({task})")
            _print_coco_results(P)
        if "mPC" in prints:
            print(f"Mean Performance under Corruption [mPC] ({task})")
            _print_coco_results(mPC)
        if "rPC" in prints:
            print(f"Relative Performance under Corruption [rPC] ({task})")
            _print_coco_results(rPC)
    else:
        if "P" in prints:
            print(f"Performance on Clean Data [P] ({task})")
            for metric_i, metric_name in enumerate(metrics):
                print(f"{metric_name:5} =  {P[metric_i]:0.3f}")
        if "mPC" in prints:
            print(f"Mean Performance under Corruption [mPC] ({task})")
            for metric_i, metric_name in enumerate(metrics):
                print(f"{metric_name:5} =  {mPC[metric_i]:0.3f}")
        if "rPC" in prints:
            print(f"Relative Performance under Corruption [rPC] ({task})")
            for metric_i, metric_name in enumerate(metrics):
                print(f"{metric_name:5} => {rPC[metric_i] * 100:0.1f} %")

    return results


def get_results(filename, dataset="coco", task="bbox", metric=None, prints="mPC", aggregate="benchmark"):
    assert dataset in ["coco", "cityscapes"]
    return get_coco_style_results(
        filename, task=task, metric=metric, prints=prints, aggregate=aggregate
    )


def main():
    parser = ArgumentParser(description="Corruption Result Analysis (JDet)")
    parser.add_argument("filename", help="result file path (json)")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["coco", "cityscapes"],
        default="coco",
        help="dataset type",
    )
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        choices=["bbox"],
        default=["bbox"],
        help="task to report",
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        choices=[
            None, "AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10",
            "AR100", "ARs", "ARm", "ARl",
        ],
        default=None,
        help="metric to report",
    )
    parser.add_argument(
        "--prints",
        type=str,
        nargs="+",
        choices=["P", "mPC", "rPC"],
        default="mPC",
        help="corruption benchmark metric to print",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        choices=["all", "benchmark"],
        default="benchmark",
        help="aggregate all results or only those for benchmark corruptions",
    )

    args = parser.parse_args()

    for task in args.task:
        get_results(
            args.filename,
            dataset=args.dataset,
            task=task,
            metric=args.metric,
            prints=args.prints,
            aggregate=args.aggregate,
        )


if __name__ == "__main__":
    main()
