# MMAD-BBox evaluation annotations

`mmad_bbox.json` contains 5,076 localization queries, 10,450 ground-truth boxes, and 38 source/category pairs. It is derived from the corrected official MMAD annotations, GitHub revision `ee10818eab0bb3ade6e1dae26e0e8f3da2095cd7` (identical JSON on Hugging Face at `a69500650a49c6cead7f26137c7e3eb0e274e7ed`). The source JSON SHA-256 is `c219c6b0c8794c017519d398374b72522fe18649d0fbac1bc0c563501de08143`.

## Scope and paper protocol

The manuscript's MMAD-BBox section (`AD_Copilot-latex/AD-Copilot/main_tip.tex`, sections “MMAD-BBox Benchmark” and “BBox-Mask IoU”) defines a localization task using the MMAD image sources. Each connected GT foreground region is enclosed by a tight box. Predicted boxes and GT boxes are separately rasterized into binary regions, and their union-mask IoU is computed without box matching. Report mean per-image IoU and accuracy at IoU thresholds 0.1, 0.2, 0.3, and 0.5.

This manifest includes every MMAD query with a nonempty source GT mask, including MVTec-LOCO logical and structural anomalies. Normal reference images are prompt templates, not normal test queries; this localization score does not measure false positives on normal images. The 3,290 queries without GT masks do not supply positive localization ground truth and are excluded. None of the 5,076 mask candidates is missing or empty. Selection uses source masks rather than generated multiple-choice answers. This is a full localization manifest, independent of the earlier 120-image diagnostic subsets.

| Source | Queries |
|---|---:|
| DS-MVTec | 1,226 |
| GoodsAD | 1,659 |
| MVTec-LOCO | 993 |
| VisA | 1,198 |

## Format and coordinates

The top-level object maps the query image's relative path to its annotation. **All paths are relative to the MMAD image root**, including `mask_path` and each `mask_files` entry. This differs from the category-relative `mask_path` field in the original `mmad.json`.

- `image_path`, `dataset`, `category`: query identity.
- `width`, `height`: original image dimensions.
- `mask_path`: actual mask file or per-query mask directory.
- `mask_files`: complete list of concrete source mask images.
- `gt_boxes`: integer `[x1, y1, x2, y2]` boxes in original-image pixel coordinates. Coordinates start at zero; the right/bottom endpoints are exclusive, so rasterization is `region[y1:y2, x1:x2] = 1`.
- `similar_templates`, `random_templates`: unchanged lists from the corrected MMAD release. The default 1-shot setting uses `similar_templates[0]`, without domain knowledge. Use no template for 0-shot. Lists can contain fewer references than a requested multi-shot setting.

The paper specifies connected-region boxes. The following conventions make that rule executable and match the existing `run_bbox_grounding.py` implementation: 8-connectivity; retain every component, including a one-pixel component; nonzero mask labels or color channels indicate foreground; merge a LOCO query's mask files with a pixelwise OR before labeling; no mask or query resizing during GT extraction. Predictions made on resized or normalized model inputs must be transformed into original-image coordinates before scoring. An absent prediction receives IoU 0; samples receive equal weight in the reported mean.

Example (template lists shortened here only):

```json
{
  "DS-MVTec/bottle/image/broken_large/000.png": {
    "image_path": "DS-MVTec/bottle/image/broken_large/000.png",
    "dataset": "DS-MVTec",
    "category": "bottle",
    "width": 900,
    "height": 900,
    "mask_path": "DS-MVTec/bottle/rbg_mask/broken_large/000_rbg_mask.png",
    "mask_files": [
      "DS-MVTec/bottle/rbg_mask/broken_large/000_rbg_mask.png"
    ],
    "gt_boxes": [
      [
        350,
        284,
        771,
        820
      ]
    ],
    "similar_templates": [
      "MVTec-AD/bottle/train/good/001.png"
    ],
    "random_templates": [
      "MVTec-AD/bottle/train/good/004.png"
    ]
  }
}
```

## Mask path corrections

Four MVTec-LOCO source annotations name a `.png` file where the actual GT is a same-stem directory. Their resolved paths are recorded in this file and in `mmad_bbox_meta.json`; the original `mmad.json` and source masks are not modified.

- `MVTec-LOCO/breakfast_box/test/structural_anomalies/001.png`: `MVTec-LOCO/breakfast_box/ground_truth/structural_anomalies/001.png` → `MVTec-LOCO/breakfast_box/ground_truth/structural_anomalies/001`.
- `MVTec-LOCO/screw_bag/test/logical_anomalies/029.png`: `MVTec-LOCO/screw_bag/ground_truth/logical_anomalies/029.png` → `MVTec-LOCO/screw_bag/ground_truth/logical_anomalies/029`.
- `MVTec-LOCO/splicing_connectors/test/logical_anomalies/089.png`: `MVTec-LOCO/splicing_connectors/ground_truth/logical_anomalies/089.png` → `MVTec-LOCO/splicing_connectors/ground_truth/logical_anomalies/089`.
- `MVTec-LOCO/splicing_connectors/test/logical_anomalies/096.png`: `MVTec-LOCO/splicing_connectors/ground_truth/logical_anomalies/096.png` → `MVTec-LOCO/splicing_connectors/ground_truth/logical_anomalies/096`.

## Reproduction and validation

The companion `build_mmad_bbox.py` takes `--mmad-root`, `--source-json`, `--output`, and optionally `--paper` and `--workers`. It reads source images/masks, validates every referenced template, rejects size mismatches and missing files, then writes the JSON and `_meta.json` sidecar. It does not need a GPU. Dependencies: Python, NumPy, Pillow, SciPy. Example:

```bash
python build_mmad_bbox.py \
  --mmad-root /path/to/dataset/MMAD \
  --source-json /path/to/corrected/mmad.json \
  --output /path/to/output/mmad_bbox.json \
  --paper /path/to/main_tip.tex
```

`mmad_bbox_meta.json` records source/output/paper/builder hashes, source counts, box counts, coordinate conventions, and path corrections. `mmad_bbox_validation.json` records an independent full-dataset check using OpenCV's 8-connected-component statistics, alongside image/mask dimensions, template integrity and coordinate bounds. `test_build_mmad_bbox.py` covers diagonal connectivity, a single-pixel boundary component, color masks, LOCO mask union, path resolution and rejection of size mismatches.

This file freezes the current corrected MMAD template selection. Record its revision/hash when reporting results: historical paper evaluations may have used the earlier GoodsAD template lists.


## Inference and scoring

Three additional scripts implement the evaluation interface:

- `run_mmad_bbox.py`: AD-Copilot / AD-Copilot-Thinking / Qwen2.5-VL inference, 0-shot or 1-shot, greedy decoding, JSONL output, optional resume and CPU-only preflight.
- `bbox_mask_iou.py`: standalone box-union rasterization and `bbox_mask_iou(pred_boxes, gt_boxes, width=..., height=...)`.
- `eval_mmad_bbox.py`: score prediction JSONL against the full frozen manifest; save summary and per-image scores. Requires NumPy only beyond the Python standard library. The inference runner additionally needs Pillow, PyTorch, Transformers, Accelerate and qwen-vl-utils. It disables optional TensorFlow/Flax backend discovery (`USE_TF=0`, `USE_FLAX=0`) before importing Transformers; these models do not require TensorFlow.

For image $i$, let $P_i$ be the pixelwise union of all rasterized predicted boxes and $G_i$ the union of all rasterized GT connected-region boxes. The paper's score is

$$
\operatorname{BBoxMaskIoU}_i = \frac{|P_i\cap G_i|}{|P_i\cup G_i|},\qquad
\operatorname{mIoU}=\frac{1}{N}\sum_i\operatorname{BBoxMaskIoU}_i.
$$

$$
\operatorname{ACC}@t=\frac{1}{N}\sum_i\mathbf{1}[\operatorname{BBoxMaskIoU}_i\ge t],
\quad t\in\{0.1,0.2,0.3,0.5\}.
$$

Both sides of the IoU are **box-union masks**. Overlapping or duplicate boxes occupy each pixel once; there is no instance matching. The implementation rounds coordinates with Python `round`, clips to the image dimensions, orders reversed endpoints, and fills `[y1:y2, x1:x2]`, exactly as the existing `run_bbox_grounding.py:raster` implementation. A missing/empty prediction or zero-area union receives zero. The summary averages over all annotation images and also reports per-source and per-category means. JSON metrics are fractions in `[0,1]`; console scores are percentages.

For example, two disjoint GT boxes `[0,0,2,2]` and `[4,0,6,2]` cover 8 pixels. A single prediction `[0,0,6,2]` covers 12 pixels, so its BBox-Mask IoU is `8/12 = 2/3`. This directly penalizes the predicted gap while retaining the GT's fragmented extent.

### Run the released model

Use Python 3.10 or newer. Scoring requires NumPy only. Rebuilding annotations and running tests also require Pillow and SciPy. Inference additionally requires PyTorch, Transformers, Accelerate and qwen-vl-utils; the smoke test used PyTorch 2.6.0 and Transformers 4.57.3. Annotations and metadata are in `dataset/MMAD/`; scripts are in `evaluation/MMAD_BBox/`.

From the repository root:

```bash
cd evaluation/MMAD_BBox
```

Then run:

```bash
python run_mmad_bbox.py \
  --annotations ../../dataset/MMAD/mmad_bbox.json \
  --mmad-root /path/to/MMAD/dataset/MMAD \
  --model /path/to/AD-Copilot \
  --trust-remote-code \
  --shots 1 \
  --coordinate-space processed \
  --device cuda:0 \
  --output results/adcopilot_1shot.jsonl

python eval_mmad_bbox.py \
  --annotations ../../dataset/MMAD/mmad_bbox.json \
  --predictions results/adcopilot_1shot.jsonl \
  --output results/adcopilot_1shot.metrics.json \
  --require-complete
```

The runner uses the checkpoint's native implementation. AD-Copilot-Thinking is loaded through its registered AutoModel class; select its model directory and set `--max-new-tokens 2048` if the chosen protocol needs the longer output budget. Choose an idle GPU when running inference. Actual CPU BF16 generation has passed a two-query smoke test for both AD-Copilot and Qwen2.5-VL, with exact agreement between the new scorer and the original parser/GT/scoring functions on all fresh outputs. This functional check does not reproduce the paper’s benchmark scores. The full 5,076-query evaluation and the planned 120-query GPU check have not been run.

By default the runner follows the existing public-demo preprocessing: thumbnail each image to longest side 512, then apply the checkpoint processor (`min_pixels=64*28*28`, `max_pixels=1280*28*28`). Set `--max-image-side 0` to omit thumbnailing, and record that setting as a different run. `--coordinate-space processed` is for the AD-Copilot/Qwen2.5-VL pixel output convention: processed query width/height come from `image_grid_thw` times the vision patch size, and boxes are scaled back to the original query size. `processed_size` and `original_size` are recorded with every prediction. Other coordinate conventions must be specified explicitly with `original`, `normalized_1000`, or `normalized_1`; the script never guesses the convention from coordinate magnitudes.

`--dry-run` checks all selected query sizes and template paths without loading model weights or importing the inference libraries. `--limit 2` prepares a two-image smoke test; use a separate output filename for a full run. `--shots 0` uses only the query image. With `--resume`, existing rows are reused only when the annotation hash and generation settings agree; saved Hub checkpoint revisions are also checked. Pin `--revision` for Hub models when reproducibility is required.

The runner saves raw responses, parsed original-coordinate boxes, response status, generated-token counts and truncation flags. Final `<answer>` content takes precedence over reasoning; an unfinished `<think>` block is not parsed as a final prediction. Invalid/no-box responses are recorded with empty predicted boxes and score zero. An inference exception stops the run; missing rows remain visible in the scorer's coverage, and a matching configuration can resume completed work.

### Score outputs from other models

Write one JSON object per image/line, with all boxes already converted to original-image pixels:

```json
{"image_path":"DS-MVTec/bottle/image/broken_large/000.png","pred_boxes":[[100,100,200,200]],"coordinate_space":"original","original_size":[900,900]}
```

The coordinates above illustrate the schema, not a model result. An attempted image with no output boxes must still have an explicit `"pred_boxes": []` row. The scorer rejects duplicate IDs, unknown query IDs, nonfinite/malformed coordinates, mismatched original sizes and explicitly unconverted coordinate spaces. With `--require-complete`, missing image rows fail the evaluation. Without that flag, missing rows score zero in the **full manifest denominator**, and coverage/missing counts are reported; a partial run cannot silently turn into a smaller scored subset.

Historical diagnostic JSONL can be selected with `--condition intact`; only one condition should be scored at a time. The summary's annotation/prediction SHA-256 values identify the scored inputs. The scorer writes `<output-stem>.per_image.jsonl` alongside the summary for auditing individual scores.

### Regression checks

```bash
python -m unittest discover -s . -p 'test_*.py'
```

`test_mmad_bbox_evaluation.py` checks fragmented GT, overlapping/duplicate predictions, clipping, reversed/fractional coordinates, empty predictions, malformed coordinates, full-denominator aggregation, duplicate/unknown IDs, answer parsing and coordinate conversion. Existing dataset-building tests cover 8-connected GT extraction and LOCO mask handling. `mmad_bbox_metric_parity.json` records an exact numerical comparison with the existing metric on cached prediction rows; this is a compatibility check, not a new model benchmark result.
