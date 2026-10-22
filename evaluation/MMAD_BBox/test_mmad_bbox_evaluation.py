import tempfile
import unittest
from pathlib import Path

import bbox_mask_iou as metric
import eval_mmad_bbox as evaluator
import run_mmad_bbox as runner


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


class RunnerTests(unittest.TestCase):
    def test_answer_tags_take_precedence_over_reasoning_boxes(self):
        text = '<think>{"bbox_2d":[1,2,3,4]}</think><answer>[{"bbox_2d":[5,6,7,8]}]</answer>'
        self.assertEqual(runner.parse_boxes(text), [[5, 6, 7, 8]])

    def test_markdown_json_and_empty_box_list(self):
        self.assertEqual(runner.parse_boxes('```json\n[{"bbox_2d":[1,2,3,4]}]\n```'), [[1, 2, 3, 4]])
        self.assertEqual(runner.parse_boxes("[]"), [])
        self.assertIsNone(runner.parse_boxes("I cannot locate it."))

    def test_truncated_reasoning_is_not_a_final_box_prediction(self):
        self.assertIsNone(runner.parse_boxes('<think>Perhaps {"bbox_2d":[1,2,3,4]}'))

    def test_processed_and_normalized_coordinates_convert_to_original(self):
        self.assertEqual(runner.to_original([[10, 5, 20, 10]], "processed", (400, 200), (200, 100)),
                         [[20, 10, 40, 20]])
        self.assertEqual(runner.to_original([[250, 250, 750, 750]], "normalized_1000", (400, 200), None),
                         [[100, 50, 300, 150]])


if __name__ == "__main__":
    unittest.main()
