"""Instance IoU at fixed confidence, with score-ordered one-to-one matching."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO


def summarize_matches(values, ground_truth_count, prediction_count):
    count = len(values)
    return {
        "matched_mean_iou": float(np.mean(values)) if count else None,
        "gt_mean_iou": float(sum(values) / ground_truth_count) if ground_truth_count else None,
        "recall_at_iou50": count / ground_truth_count if ground_truth_count else None,
        "precision_at_iou50": count / prediction_count if prediction_count else None,
        "matched_objects": count,
        "ground_truth_objects": ground_truth_count,
        "predicted_objects": prediction_count,
    }


def evaluate_instance_iou(ground_truth_path: Path, predictions_path: Path,
                          score_threshold: float = 0.5) -> dict:
    """Match same-class instances at IoU >= .5; exclude crowd/ignored GT.

    Mask and box matches are independent. Unmatched GT contributes zero to
    gt_mean_iou; matched_mean_iou is undefined (None) when no objects match.
    """
    if not 0 <= score_threshold <= 1:
        raise ValueError("IoU score threshold must be between 0 and 1.")
    coco = COCO(str(ground_truth_path))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    grouped = {}
    for prediction in predictions:
        if prediction["score"] >= score_threshold:
            key = (prediction["image_id"], prediction["category_id"])
            grouped.setdefault(key, []).append(prediction)
    report = {"score_threshold": score_threshold, "match_iou_threshold": 0.5,
              "mask_probability_threshold": 0.5,
              "matching": "same-class, descending confidence, one-to-one; crowd/ignored GT excluded"}
    for kind in ("segm", "bbox"):
        per_class = {}
        all_values, total_gt, total_pred = [], 0, 0
        for category_id, category in coco.cats.items():
            values, n_gt, n_pred = [], 0, 0
            for image_id in coco.imgs:
                gt = [ann for ann in coco.imgToAnns.get(image_id, [])
                      if ann["category_id"] == category_id
                      and not ann.get("iscrowd", 0) and not ann.get("ignore", 0)]
                pred = sorted(grouped.get((image_id, category_id), []),
                              key=lambda ann: ann["score"], reverse=True)
                n_gt += len(gt)
                n_pred += len(pred)
                if not gt or not pred:
                    continue
                a = [coco.annToRLE(ann) for ann in pred] if kind == "segm" else [ann["bbox"] for ann in pred]
                b = [coco.annToRLE(ann) for ann in gt] if kind == "segm" else [ann["bbox"] for ann in gt]
                overlaps = mask_utils.iou(a, b, [0] * len(gt))
                used = set()
                for row in overlaps:
                    available = [index for index in range(len(gt)) if index not in used]
                    if not available:
                        break
                    best = max(available, key=lambda index: row[index])
                    if row[best] >= 0.5:
                        used.add(best)
                        values.append(float(row[best]))
            per_class[category["name"]] = summarize_matches(values, n_gt, n_pred)
            all_values.extend(values)
            total_gt += n_gt
            total_pred += n_pred
        report[kind] = {"overall": summarize_matches(all_values, total_gt, total_pred),
                        "per_class": per_class}
    return report


def evaluate_saved_run(output_dir: Path, score_threshold: float = 0.5) -> dict:
    """Evaluate existing test exports without loading a model or dataset."""
    folders = sorted(output_dir.glob("split*"))
    folders = [folder for folder in folders if folder.is_dir()]
    if not folders:
        folders = [output_dir]
    # Validate all inputs before writing any results.
    for folder in folders:
        for name in ("coco_test_ground_truth.json", "coco_test_predictions.json"):
            if not (folder / name).is_file():
                raise FileNotFoundError(f"Missing saved evaluation file: {folder / name}")
    reports = {}
    for folder in folders:
        report = evaluate_instance_iou(folder / "coco_test_ground_truth.json",
                                      folder / "coco_test_predictions.json", score_threshold)
        (folder / "iou_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports[folder.name] = report
    summary = {"splits": reports, "split_statistics": {}}
    for kind in ("segm", "bbox"):
        summary["split_statistics"][kind] = {}
        for metric in ("matched_mean_iou", "gt_mean_iou", "recall_at_iou50", "precision_at_iou50"):
            values = [report[kind]["overall"][metric] for report in reports.values()
                      if report[kind]["overall"][metric] is not None]
            summary["split_statistics"][kind][metric] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "valid_splits": len(values),
            }
    (output_dir / "iou_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    args = parser.parse_args()
    result = evaluate_saved_run(args.output_dir, args.score_threshold)
    print(json.dumps(result["split_statistics"], indent=2))
    print(f"Saved IoU results to {args.output_dir / 'iou_summary.json'}")
