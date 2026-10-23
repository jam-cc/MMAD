#!/usr/bin/env python3
"""Score original-pixel prediction JSONL against the full MMAD-BBox manifest."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from numbers import Integral, Real
import math

import numpy as np


def validate_boxes(boxes) -> None:
    if not isinstance(boxes, (list, tuple, np.ndarray)):
        raise ValueError("Boxes must be a sequence of [x1, y1, x2, y2] coordinates")
    for box in boxes:
        if not isinstance(box, (list, tuple, np.ndarray)) or len(box) != 4:
            raise ValueError(f"Expected four coordinates per box: {box!r}")
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
               or not math.isfinite(v) for v in box):
            raise ValueError(f"Box coordinates must be finite numbers: {box!r}")


def rasterize_boxes(boxes, *, width: int, height: int) -> np.ndarray:
    """Round to nearest integer, clip to the image, order endpoints, fill xyxy.

    This reproduces run_bbox_grounding.py:raster, including Python's round().
    Right and bottom endpoints are exclusive. Overlapping boxes are OR-ed.
    """
    if (isinstance(width, bool) or isinstance(height, bool)
            or not isinstance(width, Integral) or not isinstance(height, Integral)
            or width <= 0 or height <= 0):
        raise ValueError("Image width and height must be positive integers")
    validate_boxes(boxes)
    region = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        x0, x1 = sorted(max(0, min(width, int(round(x)))) for x in (x0, x1))
        y0, y1 = sorted(max(0, min(height, int(round(y)))) for y in (y0, y1))
        region[y0:y1, x0:x1] = True
    return region


def bbox_mask_iou(pred_boxes, gt_boxes, *, width: int, height: int) -> float:
    """|union(pred) & union(GT boxes)| / |union(pred) | union(GT boxes)|.

    GT inputs are the connected-component BOXES, rather than the irregular
    source masks. No instance matching, confidence ranking or NMS is used.
    An empty union scores zero, matching the existing evaluation code.
    """
    validate_boxes(pred_boxes)
    validate_boxes(gt_boxes)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        # Validate dimensions even when there is no region to rasterize.
        if (isinstance(width, bool) or isinstance(height, bool)
                or not isinstance(width, Integral) or not isinstance(height, Integral)
                or width <= 0 or height <= 0):
            raise ValueError("Image width and height must be positive integers")
        return 0.0
    predicted = rasterize_boxes(pred_boxes, width=width, height=height)
    truth = rasterize_boxes(gt_boxes, width=width, height=height)
    union = np.count_nonzero(predicted | truth)
    return float(np.count_nonzero(predicted & truth) / union) if union else 0.0



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


def self_test() -> None:
    """Run the built-in metric and aggregation checks; needs NumPy only."""
    import sys
    import tempfile
    import unittest
    metric = evaluator = sys.modules[__name__]

    class MetricTests(unittest.TestCase):
        def test_covering_two_gt_components_includes_the_gap_in_union(self):
            self.assertAlmostEqual(metric.bbox_mask_iou(
                [[0, 0, 6, 2]], [[0, 0, 2, 2], [4, 0, 6, 2]], width=6, height=2), 2 / 3)

        def test_overlapping_or_duplicate_predictions_are_not_counted_twice(self):
            self.assertEqual(metric.bbox_mask_iou(
                [[0, 0, 3, 2], [1, 0, 4, 2], [1, 0, 4, 2]],
                [[0, 0, 4, 2]], width=4, height=2), 1.0)

        def test_empty_prediction_and_empty_union_score_zero(self):
            self.assertEqual(metric.bbox_mask_iou([], [[0, 0, 1, 1]], width=2, height=2), 0)
            self.assertEqual(metric.bbox_mask_iou([], [], width=2, height=2), 0)

        def test_clip_reverse_round_and_exclusive_boundary(self):
            self.assertEqual(metric.bbox_mask_iou(
                [[5, 3, -1, -2]], [[0, 0, 4, 2]], width=4, height=2), 1)
            self.assertEqual(metric.bbox_mask_iou(
                [[1.6, 1.6, 3, 3]], [[2, 2, 3, 3]], width=3, height=3), 1)

        def test_nonfinite_and_malformed_boxes_fail(self):
            for boxes in [[[0, 0, float("nan"), 1]], [[0, 0, 1]], [[False, 0, 1, 1]]]:
                with self.assertRaises(ValueError):
                    metric.bbox_mask_iou(boxes, [[0, 0, 1, 1]], width=2, height=2)


    class EvaluationTests(unittest.TestCase):
        def setUp(self):
            self.annotations = {k: {"width": 2, "height": 2, "gt_boxes": [[0, 0, 2, 2]],
                                      "dataset": "test", "category": "part"} for k in ["a", "b"]}

        def test_missing_rows_stay_in_the_full_benchmark_denominator(self):
            report, rows = evaluator.evaluate(self.annotations, {"a": {"pred_boxes": [[0, 0, 2, 2]]}})
            self.assertEqual(report["samples"], 2)
            self.assertEqual(report["missing_predictions"], 1)
            self.assertEqual(report["miou"], 0.5)
            self.assertEqual(report["accuracy_at_iou"]["0.5"], 0.5)
            self.assertEqual(len(rows), 2)

        def test_duplicate_unknown_and_unconverted_coordinates_fail(self):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "predictions.jsonl"
                path.write_text('{"image_path":"a","pred_boxes":[]}\n' * 2)
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    evaluator.load_predictions(path)
            with self.assertRaisesRegex(ValueError, "Unknown"):
                evaluator.evaluate(self.annotations, {"c": {"pred_boxes": []}})
            with self.assertRaisesRegex(ValueError, "original"):
                evaluator.evaluate(self.annotations, {"a": {"pred_boxes": [], "coordinate_space": "processed"}})


    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case)
                               for case in (MetricTests, EvaluationTests))
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("mmad_bbox.json"))
    parser.add_argument("--predictions", type=Path, help="JSONL: image_path, pred_boxes in original pixels")
    parser.add_argument("--output", type=Path, help="Summary JSON; also writes <stem>.per_image.jsonl")
    parser.add_argument("--condition", help="Explicitly select one condition from diagnostic JSONL")
    parser.add_argument("--require-complete", action="store_true", help="Fail if any benchmark image lacks a prediction row")
    parser.add_argument("--self-test", action="store_true", help="Run built-in checks without annotations or predictions")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.predictions is None or args.output is None:
        parser.error("--predictions and --output are required unless --self-test is used")
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
