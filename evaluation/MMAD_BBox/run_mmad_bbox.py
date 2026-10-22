#!/usr/bin/env python3
"""Run AD-Copilot/Qwen2.5-VL on MMAD-BBox and save original-pixel JSONL."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import time

from PIL import Image

from bbox_mask_iou import validate_boxes
from eval_mmad_bbox import load_predictions


PROMPT_1SHOT = ("The first image is a normal sample. Please locate the defects within the "
                "second image with bounding box in JSON format.")
PROMPT_0SHOT = "Please locate the defects within the image with bounding box in JSON format."


def parse_boxes(text: str) -> list | None:
    """Read JSON boxes from the answer, excluding any preceding reasoning."""
    answers = re.findall(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    if not answers and re.search(r"<think>", text, re.I) and not re.search(r"</think>", text, re.I):
        return None
    text = answers[-1] if answers else re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()

    def unpack(value):
        if isinstance(value, dict):
            if "bbox_2d" in value:
                return [value["bbox_2d"]]
            if "boxes" in value:
                return unpack(value["boxes"])
            raise ValueError("No bbox_2d field")
        if not isinstance(value, list):
            raise ValueError("Expected JSON array/object")
        if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
            return [value]
        if value and isinstance(value[0], dict):
            return [item["bbox_2d"] for item in value]
        return value

    decoder = json.JSONDecoder()
    # Permit a short textual prefix/fenced JSON, but not guessed coordinate regexes.
    for match in re.finditer(r"[\[{]", text):
        try:
            payload, _ = decoder.raw_decode(text[match.start():])
            boxes = unpack(payload)
            validate_boxes(boxes)
            return boxes
        except (ValueError, KeyError, TypeError):
            continue
    return None


def to_original(boxes, coordinate_space: str, original_size: tuple,
                processed_size: tuple | None) -> list:
    validate_boxes(boxes)
    width, height = original_size
    if coordinate_space == "processed":
        if not processed_size or min(processed_size) <= 0:
            raise ValueError("Processed-pixel boxes require the processor's image size")
        sx, sy = width / processed_size[0], height / processed_size[1]
    elif coordinate_space == "normalized_1000":
        sx, sy = width / 1000, height / 1000
    elif coordinate_space == "normalized_1":
        sx, sy = width, height
    elif coordinate_space == "original":
        sx = sy = 1
    else:
        raise ValueError(f"Unknown coordinate space: {coordinate_space}")
    return [[x0 * sx, y0 * sy, x1 * sx, y1 * sy] for x0, y0, x1, y1 in boxes]


def load_image(path: Path, max_side: int):
    with Image.open(path) as source:
        image = source.convert("RGB")
    if max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path(__file__).with_name("mmad_bbox.json"))
    parser.add_argument("--mmad-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, choices=(0, 1), default=1)
    parser.add_argument("--coordinate-space", choices=("processed", "original", "normalized_1000", "normalized_1"),
                        default="processed", help="Coordinate space emitted by the model; output JSONL is always original pixels")
    parser.add_argument("--max-image-side", type=int, default=512, help="Existing public-demo thumbnail setting; 0 disables thumbnailing")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("default", "sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--revision", help="Pin a Hub model/processor revision")
    parser.add_argument("--trust-remote-code", action="store_true", help="Required for the custom AD-Copilot checkpoint code")
    parser.add_argument("--limit", type=int, default=0, help="Smoke test: first N images; 0 selects the full manifest")
    parser.add_argument("--resume", action="store_true", help="Resume only if the saved run configuration matches")
    parser.add_argument("--dry-run", action="store_true", help="Validate selected query/template paths and sizes without loading a model")
    args = parser.parse_args()
    if args.max_image_side < 0 or args.limit < 0 or args.max_new_tokens <= 0:
        parser.error("Image side/limit must be nonnegative and max-new-tokens positive")
    annotation_bytes = args.annotations.read_bytes()
    annotations = json.loads(annotation_bytes)
    selected = list(annotations.items())
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError("No selected images")
    for key, sample in selected:
        with Image.open(args.mmad_root / key) as image:
            if image.size != (sample["width"], sample["height"]):
                raise ValueError(f"Query image size mismatch: {key}")
        if args.shots and (not sample["similar_templates"]
                           or not (args.mmad_root / sample["similar_templates"][0]).is_file()):
            raise FileNotFoundError(f"Missing normal reference for {key}")
    selection = {"samples": len(selected), "datasets": dict(Counter(v["dataset"] for _, v in selected)),
                 "shots": args.shots, "model_output_coordinates": args.coordinate_space,
                 "prediction_coordinates": "original", "max_image_side": args.max_image_side}
    if args.dry_run:
        print(json.dumps({"dry_run": True, **selection}, indent=2))
        return

    settings = {name: getattr(args, name) for name in
                ("model", "revision", "shots", "coordinate_space", "max_image_side", "max_new_tokens",
                 "dtype", "attn_implementation", "trust_remote_code", "limit")}
    if Path(args.model).exists():
        settings["model"] = str(Path(args.model).resolve())
    settings["annotations_sha256"] = hashlib.sha256(annotation_bytes).hexdigest()
    run_path = args.output.with_suffix(".run.json")
    done = {}
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"Output exists; use --resume with the same run settings: {args.output}")
        if not run_path.exists() or json.loads(run_path.read_text())["settings"] != settings:
            raise ValueError("Resume configuration differs from the saved run")
        done = load_predictions(args.output)
        if done.keys() - dict(selected).keys():
            raise ValueError("Resume output contains images outside the selected manifest")
        for row in done.values():
            validate_boxes(row["pred_boxes"])
            if row.get("coordinate_space") != "original":
                raise ValueError("Resume rows must be in original coordinates")
    if len(done) == len(selected):
        print("All selected predictions already exist")
        return

    # Heavy inference dependencies are imported only after dry-run/preflight.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoModelForVision2Seq, AutoProcessor
    from qwen_vl_utils import process_vision_info
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; --dry-run works without loading a model")
    common = {"trust_remote_code": args.trust_remote_code, "revision": args.revision}
    config = AutoConfig.from_pretrained(args.model, **common)
    mapping = getattr(config, "auto_map", {}) or {}
    model_class = (AutoModelForVision2Seq if "AutoModelForVision2Seq" in mapping
                   and "AutoModelForImageTextToText" not in mapping else AutoModelForImageTextToText)
    kwargs = {**common, "config": config, "dtype": getattr(torch, args.dtype), "low_cpu_mem_usage": True}
    if args.attn_implementation != "default":
        kwargs["attn_implementation"] = args.attn_implementation
    model = model_class.from_pretrained(args.model, **kwargs).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=64 * 28 * 28,
                                            max_pixels=1280 * 28 * 28, **common)
    checkpoint_commit = getattr(config, "_commit_hash", None)
    if args.resume and run_path.exists():
        previous_commit = json.loads(run_path.read_text()).get("model_commit")
        if previous_commit != checkpoint_commit:
            raise ValueError("Checkpoint revision changed since this run started; pin --revision")
    vision_config = getattr(config, "vision_config", None)
    patch_size = (vision_config.get("patch_size") if isinstance(vision_config, dict)
                  else getattr(vision_config, "patch_size", None))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps({"settings": settings, "selection": selection,
                                    "model_class": type(model).__name__, "model_commit": checkpoint_commit,
                                    "device": args.device, "prompt": PROMPT_1SHOT if args.shots else PROMPT_0SHOT}, indent=2) + "\n")
    with args.output.open("a" if args.resume else "w") as handle:
        for number, (key, sample) in enumerate(selected, 1):
            if key in done:
                continue
            started = time.monotonic()
            content = []
            template = sample["similar_templates"][0] if args.shots else None
            if template:
                content.append({"type": "image", "image": load_image(args.mmad_root / template, args.max_image_side)})
            content.extend([{"type": "image", "image": load_image(args.mmad_root / key, args.max_image_side)},
                            {"type": "text", "text": PROMPT_1SHOT if args.shots else PROMPT_0SHOT}])
            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(args.device)
            processed_size = None
            if "image_grid_thw" in inputs and isinstance(patch_size, int):
                _, gh, gw = [int(v) for v in inputs["image_grid_thw"][args.shots]]
                processed_size = (gw * patch_size, gh * patch_size)
            if args.coordinate_space == "processed" and processed_size is None:
                raise ValueError("Model does not expose image_grid_thw/patch_size; specify its output coordinate space")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                           return_dict_in_generate=False)
            count = int(generated.shape[1] - inputs.input_ids.shape[1])
            raw = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            parsed = parse_boxes(raw)
            status = "parse_error" if parsed is None else ("ok" if parsed else "no_box")
            boxes = to_original(parsed or [], args.coordinate_space,
                                (sample["width"], sample["height"]), processed_size)
            row = {"image_path": key, "pred_boxes": boxes, "coordinate_space": "original",
                   "original_size": [sample["width"], sample["height"]], "processed_size": processed_size,
                   "template_path": template, "raw": raw, "status": status,
                   "generated_tokens": count, "generation_hit_token_limit": count >= args.max_new_tokens,
                   "seconds": round(time.monotonic() - started, 3)}
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            print(f"{number}/{len(selected)} {key}: {status}, {len(boxes)} boxes", flush=True)


if __name__ == "__main__":
    main()
