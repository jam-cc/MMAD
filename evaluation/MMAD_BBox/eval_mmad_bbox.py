#!/usr/bin/env python3
"""Score original-pixel prediction JSONL against the full MMAD-BBox manifest."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from bbox_mask_iou import bbox_mask_iou


THRESHOLDS = (0.1, 0.2, 0.3, 0.5)


def load_predictions(path: Path, condition: str | None = None) -> dict:
    predictions = {}
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if condition is not None and row.get("condition") != condition:
                continue
            key = row.get("image_path")
            if not isinstance(key, str) or "pred_boxes" not in row:
                raise ValueError(f"Line {number}: require image_path and pred_boxes")
            if key in predictions:
                raise ValueError(f"Duplicate prediction: {key}; select a single condition/run")
            predictions[key] = row
    return predictions


def aggregate(values: list[float]) -> dict:
    return {"samples": len(values), "miou": sum(values) / len(values),
            "accuracy_at_iou": {str(t): sum(v >= t for v in values) / len(values)
                                for t in THRESHOLDS}}


def evaluate(annotations: dict, predictions: dict) -> tuple[dict, list[dict]]:
    if not annotations:
        raise ValueError("Empty annotation manifest")
    unknown = predictions.keys() - annotations.keys()
    if unknown:
        raise ValueError(f"Unknown prediction image IDs: {sorted(unknown)[:5]}")
    rows, by_dataset, by_category = [], defaultdict(list), defaultdict(list)
    statuses = Counter()
    no_box = missing = 0
    for key, truth in annotations.items():
        present = key in predictions
        row = predictions.get(key, {"pred_boxes": [], "status": "missing"})
        if row.get("coordinate_space", "original") != "original":
            raise ValueError(f"Predictions must be converted to original-image pixels: {key}")
        if row.get("original_size", [truth["width"], truth["height"]]) != [truth["width"], truth["height"]]:
            raise ValueError(f"Original image size mismatch: {key}")
        boxes = row["pred_boxes"]
        if not truth["gt_boxes"]:
            raise ValueError(f"Localization ground truth cannot be empty: {key}")
        value = bbox_mask_iou(boxes, truth["gt_boxes"], width=truth["width"], height=truth["height"])
        status = row.get("status", "ok" if boxes else "no_box")
        statuses[status] += 1
        missing += not present
        no_box += len(boxes) == 0
        by_dataset[truth["dataset"]].append(value)
        by_category[f'{truth["dataset"]}/{truth["category"]}'].append(value)
        rows.append({"image_path": key, "iou": value, "prediction_present": present,
                     "status": status, "n_pred_boxes": len(boxes), "n_gt_boxes": len(truth["gt_boxes"])})
    report = aggregate([r["iou"] for r in rows])
    report.update({
        "prediction_rows": len(predictions), "missing_predictions": missing,
        "prediction_coverage": len(predictions) / len(annotations),
        "images_without_predicted_boxes": no_box,
        "status_counts": dict(statuses), "denominator": "All annotation images; missing/no-box predictions score zero",
        "coordinate_space": "original pixels; xyxy; x2/y2 exclusive",
        "per_dataset": {k: aggregate(v) for k, v in sorted(by_dataset.items())},
        "per_category": {k: aggregate(v) for k, v in sorted(by_category.items())},
    })
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path(__file__).with_name("mmad_bbox.json"))
    parser.add_argument("--predictions", type=Path, required=True, help="JSONL: image_path, pred_boxes in original pixels")
    parser.add_argument("--output", type=Path, required=True, help="Summary JSON; also writes <stem>.per_image.jsonl")
    parser.add_argument("--condition", help="Explicitly select one condition from diagnostic JSONL")
    parser.add_argument("--require-complete", action="store_true", help="Fail if any benchmark image lacks a prediction row")
    args = parser.parse_args()
    per_image = args.output.with_name(args.output.stem + ".per_image.jsonl")
    inputs = {args.annotations.resolve(), args.predictions.resolve()}
    if args.output.resolve() in inputs or per_image.resolve() in inputs:
        raise ValueError("Output paths must differ from annotation and prediction input paths")
    annotations = json.loads(args.annotations.read_text())
    predictions = load_predictions(args.predictions, args.condition)
    if args.require_complete and annotations.keys() - predictions.keys():
        raise ValueError(f"Incomplete predictions: {len(annotations.keys() - predictions.keys())} missing images")
    report, rows = evaluate(annotations, predictions)
    report.update({"annotations_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
                   "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
                   "condition": args.condition})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    per_image.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(json.dumps({"samples": report["samples"], "prediction_rows": report["prediction_rows"],
                      "missing_predictions": report["missing_predictions"], "mIoU_percent": report["miou"] * 100,
                      "ACC_percent": {k: v * 100 for k, v in report["accuracy_at_iou"].items()},
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
