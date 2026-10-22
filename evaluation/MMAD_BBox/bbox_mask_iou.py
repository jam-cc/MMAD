"""MMAD-BBox: IoU of rasterized box unions, in original-image pixels."""

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
