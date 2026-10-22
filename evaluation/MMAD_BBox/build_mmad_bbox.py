#!/usr/bin/env python3
"""Export MMAD localization ground truth using connected-mask-region boxes.

Paths in the output are relative to --mmad-root. The source MMAD JSON and
all source images/masks are read only. No question/answer text selects samples.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from scipy import ndimage


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def boxes_from_mask(mask: np.ndarray) -> list[list[int]]:
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    return [
        [sl[1].start, sl[0].start, sl[1].stop, sl[0].stop]
        for sl in ndimage.find_objects(labels)
        if sl is not None
    ]


def resolve_mask(root: Path, image: str, annotation: str) -> tuple[Path, list[Path]]:
    dataset, category = image.split("/")[:2]
    path = root / dataset / category / annotation
    # Four LOCO entries name <id>.png although their GT is the directory <id>/.
    if (not path.exists() and dataset == "MVTec-LOCO"
            and path.suffix.lower() == ".png" and path.with_suffix("").is_dir()):
        path = path.with_suffix("")
    if not path.exists():
        raise FileNotFoundError(f"Missing mask for {image}: {path}")
    files = ([path] if path.is_file() else
             sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES))
    if not files:
        raise ValueError(f"No mask images in {path}")
    return path, files


def load_mask(files: list[Path], image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    foreground = np.zeros((height, width), dtype=bool)
    for path in files:
        with Image.open(path) as image:
            if image.size != image_size:
                raise ValueError(f"Mask size {image.size} != query size {image_size}: {path}")
            # Palette masks are decoded to colors; alpha is never a defect label.
            array = np.asarray(image.convert("RGB") if image.mode == "P" else image)
            binary = np.any(array[..., :3] > 0, axis=-1) if array.ndim == 3 else array > 0
        foreground |= binary
    return foreground


def build_sample(root: Path, key: str, source: dict) -> tuple[dict | None, dict | None]:
    with Image.open(root / key) as image:
        width, height = image.size
    mask_path, mask_files = resolve_mask(root, key, source["mask_path"])
    mask = load_mask(mask_files, (width, height))
    boxes = boxes_from_mask(mask)
    if not boxes:
        return None, {"image_path": key, "reason": "empty_ground_truth_mask"}
    for field in ("similar_templates", "random_templates"):
        if not source.get(field):
            raise ValueError(f"Empty {field}: {key}")
        for template in source[field]:
            if not (root / template).is_file():
                raise FileNotFoundError(f"Missing {field} image: {template}")
    dataset, category = key.split("/")[:2]
    record = {
        "image_path": key,
        "dataset": dataset,
        "category": category,
        "width": width,
        "height": height,
        "mask_path": mask_path.relative_to(root).as_posix(),
        "mask_files": [p.relative_to(root).as_posix() for p in mask_files],
        "gt_boxes": boxes,
        "similar_templates": source["similar_templates"],
        "random_templates": source["random_templates"],
    }
    original = "/".join([dataset, category, source["mask_path"]])
    correction = None
    if record["mask_path"] != original:
        correction = {"image_path": key, "source_mask_path": original,
                      "resolved_mask_path": record["mask_path"]}
    return record, correction


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as temporary:
        json.dump(value, temporary, ensure_ascii=False, indent=2)
        temporary.write("\n")
        name = temporary.name
    os.replace(name, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmad-root", type=Path, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    source_bytes = args.source_json.read_bytes()
    source = json.loads(source_bytes)
    candidates = [(key, value) for key, value in source.items() if value.get("mask_path")]
    output, corrections, exclusions, errors = {}, [], [], []

    def process(item):
        key, value = item
        try:
            return key, *build_sample(args.mmad_root, key, value), None
        except (OSError, ValueError) as error:
            return key, None, None, f"{type(error).__name__}: {error}"

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (key, record, note, error) in enumerate(executor.map(process, candidates), 1):
            if error:
                errors.append({"image_path": key, "error": error})
            elif record is None:
                exclusions.append(note)
            else:
                output[key] = record
                if note:
                    corrections.append(note)
            if index % 500 == 0 or index == len(candidates):
                print(f"Processed {index}/{len(candidates)} GT candidates", flush=True)

    if errors:
        write_json(args.output.with_suffix(".errors.json"), {"errors": errors})
        raise RuntimeError(f"{len(errors)} invalid samples; no benchmark JSON was written")
    if not output:
        raise RuntimeError("No nonempty localization ground truth found")
    write_json(args.output, output)
    metadata = {
        "name": "MMAD-BBox",
        "schema_version": "1.0",
        "source_mmad_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "paper_sha256": hashlib.sha256(args.paper.read_bytes()).hexdigest() if args.paper else None,
        "source_records": len(source),
        "mask_candidates": len(candidates),
        "samples": len(output),
        "ground_truth_boxes": sum(len(v["gt_boxes"]) for v in output.values()),
        "source_mask_files": sum(len(v["mask_files"]) for v in output.values()),
        "datasets": dict(sorted(Counter(v["dataset"] for v in output.values()).items())),
        "categories": dict(sorted(Counter(f'{v["dataset"]}/{v["category"]}' for v in output.values()).items())),
        "gt_box_count_histogram": dict(sorted(Counter(len(v["gt_boxes"]) for v in output.values()).items())),
        "selection": "Every source query with an existing nonempty GT mask; no VQA labels used; all sources including MVTec-LOCO",
        "excluded_without_ground_truth_mask": len(source) - len(candidates),
        "excluded_empty_masks": exclusions,
        "mask_path_corrections": corrections,
        "paper_rules": {
            "gt_boxes": "One tight box per connected foreground region of the GT mask",
            "scoring": "IoU between the union of rasterized predicted boxes and the union of rasterized GT boxes; no box matching",
            "metrics": ["mean per-image IoU", "ACC@IoU>=0.1", "ACC@IoU>=0.2", "ACC@IoU>=0.3", "ACC@IoU>=0.5"],
            "default_setting": "1-shot, first similar_templates image, without domain knowledge",
        },
        "implementation_conventions": {
            "path_base": "MMAD dataset root; all paths, including mask_path and mask_files, are root-relative",
            "coordinate_format": "xyxy in original-image pixels, zero-based, x2/y2 exclusive",
            "foreground": "Any nonzero label/channel; ignore alpha",
            "mask_directory": "Pixelwise union of all per-image masks, then connected components",
            "connectivity": 8,
            "minimum_component_area_pixels": 1,
            "mask_resizing": "None; reject mismatched query/mask sizes",
            "empty_prediction_iou": 0,
            "reference_code": "rebuttal/structure_diagnostics/run_bbox_grounding.py:gt_boxes, raster, region_iou",
        },
    }
    write_json(args.output.with_name(args.output.stem + "_meta.json"), metadata)
    print(json.dumps({k: metadata[k] for k in ("samples", "ground_truth_boxes", "datasets", "output_sha256")}, indent=2))


if __name__ == "__main__":
    main()
