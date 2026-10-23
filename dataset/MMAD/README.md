
## Download the datasets in the following link and extract them into corresponding folders.
+ [DS-MVTec](https://huggingface.co/datasets/DefectSpectrum/Defect_Spectrum/tree/main/DS-MVTec)
+ [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
+ [MVTec-LOCO](https://www.mvtec.com/company/research/datasets/mvtec-loco)
+ [VisA](https://github.com/amazon-science/spot-diff)(1-class setup)
+ [GoodsAD](https://github.com/jianzhang96/GoodsAD)

For research purposes, we have provided a ZIP file containing all the datasets: [ZIP file](https://huggingface.co/datasets/jiang-cc/MMAD/resolve/refs%2Fpr%2F1/ALL_DATA.zip?download=true). Please note that these datasets are for research use only and should be used in accordance with their respective copyright and licensing agreements.

## Dataset Directory Structure

After downloading and extracting, ensure that each dataset folder directly contains the category folders. You can also use symbolic links to achieve this structure if preferred.

#### DS-MVTec and MVTec-AD Directory Structure
Both the **DS-MVTec** and **MVTec-AD** datasets should be organized with the same folder structure:

```
└── DS-MVTec
    ├── bottle
    ├── cable
    ├── capsule
    ├── carpet
    ├── grid
    ├── hazelnut
    ├── leather
    ├── metal_nut
    ├── pill
    ├── screw
    ├── tile
    ├── toothbrush
    ├── transistor
    ├── wood
    └── zipper
```

#### MVTec-LOCO Directory Structure


```
└── MVTec-LOCO
    ├── breakfast_box
    ├── juice_bottle
    ├── pushpins
    ├── screw_bag
    └── splicing_connectors
```

#### VisA Directory Structure


```
└── VisA
    ├── candle
    ├── capsules
    ├── cashew
    ├── chewinggum
    ├── fryum
    ├── macaroni1
    ├── macaroni2
    ├── pcb1
    ├── pcb2
    ├── pcb3
    ├── pcb4
    └── pipe_fryum
```

#### GoodsAD Directory Structure


```
└── GoodsAD
    ├── cigarette_box
    ├── drink_bottle
    ├── drink_can
    ├── food_bottle
    ├── food_box
    └── food_package
```

After organizing the datasets, each folder should contain the category subfolders. 

## Annotation corrections (2026-10-22)

- Updated `similar_templates` for all 2,968 GoodsAD query images to normal training images with the same category and product ID (the filename prefix before `_`). For example, `GoodsAD/food_package/test/surface_anomaly/091_006.jpg` now references `091_001.jpg` through `091_004.jpg` in `GoodsAD/food_package/train/good/`.
- Templates are ordered by numeric filename and include all available matching normal training images. Counts vary from 1 to 26; 62 query images have only one matching reference. Evaluation code should use the available references when fewer than the requested number of shots exist.
- Changed the remaining incorrect query key `DS-MVTec/pill/image/good/008.png` and its `image_path` to `MVTec-AD/pill/test/good/008.png`, consistent with the existing pill annotations. The DS-MVTec image is a metal nut.

Question and answer annotations, `random_templates`, and the total of 8,366 query images are unchanged. Existing question corrections are retained. Results using the revised GoodsAD references may differ from earlier evaluations; record the dataset revision when reporting results.

For an existing download, replace `mmad.json` with the current version. If you use a derived `metadata.csv`, regenerate it with `python generate_metadata.py` from this directory after updating the JSON. The source image archives do not need to be changed.

## MMAD-BBox localization annotations

`mmad_bbox.json` contains 5,076 localization queries, 10,450 GT boxes, and 38 source/category pairs: DS-MVTec (1,226), GoodsAD (1,659), MVTec-LOCO (993), and VisA (1,198). It selects queries with nonempty source GT masks; 3,290 queries without masks are excluded. Normal reference images are prompt templates, not normal test queries. This localization score does not measure false positives on normal images.

The release has four files:

- `mmad_bbox.json`: frozen query annotations.
- `mmad_bbox_meta.json`: source hashes, construction rules and counts.
- `build_mmad_bbox.py`: optional reconstruction from the original images and masks.
- `eval_mmad_bbox.py`: BBox-Mask IoU operator, JSONL scoring and built-in self-tests in one standalone script.

### Annotation and prediction format

The annotation object maps each query's relative image path to `image_path`, `dataset`, `category`, `width`, `height`, `mask_path`, `mask_files`, `gt_boxes`, `similar_templates` and `random_templates`. All image/mask/template paths are relative to the MMAD image root. This differs from category-relative mask paths in the original `mmad.json`.

Boxes use original-image pixels and zero-based `[x1,y1,x2,y2]` coordinates; right/bottom endpoints are exclusive. Predictions from resized or normalized inputs must be converted to original coordinates before scoring. Write one JSON object per line:

```json
{"image_path":"DS-MVTec/bottle/image/broken_large/000.png","pred_boxes":[[350,284,771,820]],"coordinate_space":"original","original_size":[900,900]}
```

This example illustrates the schema, not a model result. Use `"pred_boxes": []` for an attempted query with no predicted boxes. For the default 1-shot setting, use `similar_templates[0]` as the normal reference, without domain knowledge; 0-shot uses only the query. Template lists retain the corrected MMAD same-product GoodsAD references, so historical scores may differ.

### BBox-Mask IoU

Each GT foreground connected region is enclosed by a tight box. Predicted and GT boxes are separately rasterized and merged with pixelwise OR:

$$\mathrm{IoU}_i=\frac{|P_i\cap G_i|}{|P_i\cup G_i|},\qquad
\mathrm{mIoU}=\frac1N\sum_i\mathrm{IoU}_i.$$

Both sides are **unions of box regions**, not irregular source masks. No instance matching or NMS is used. Report mean per-image IoU and `ACC@t = mean(IoU >= t)` for `t = 0.1, 0.2, 0.3, 0.5`. The scorer rounds coordinates with Python `round`, clips to image bounds, orders reversed endpoints, and fills `[y1:y2,x1:x2]`. Empty predictions score zero. Missing rows also score zero in the full manifest denominator; `--require-complete` instead rejects missing rows. Duplicate/unknown IDs and malformed/unconverted coordinates are rejected. JSON scores are fractions in `[0,1]`; console scores are percentages.

### Score and self-test

Use Python 3.10+ and NumPy (`python -m pip install numpy`). The scorer needs neither model weights nor source images/masks. From the repository root:

```bash
python evaluation/MMAD_BBox/eval_mmad_bbox.py --self-test

python evaluation/MMAD_BBox/eval_mmad_bbox.py \
  --annotations dataset/MMAD/mmad_bbox.json \
  --predictions predictions.jsonl \
  --output metrics.json \
  --require-complete
```

It writes a summary (overall, per-source and per-category scores, coverage and input hashes) plus `metrics.per_image.jsonl`. `bbox_mask_iou(pred_boxes, gt_boxes, width=..., height=...)` can also be imported from this same script. Run model inference separately and provide the prediction format above. No separate test files or validation-report files are needed.

### Optional annotation reconstruction

Rebuilding additionally requires Pillow and SciPy. From the repository root:

```bash
python evaluation/MMAD_BBox/build_mmad_bbox.py \
  --mmad-root /path/to/MMAD/images \
  --source-json dataset/MMAD/mmad.json \
  --output /path/to/output/mmad_bbox.json
```

GT extraction uses 8-connectivity and every nonzero component, including a single pixel. RGB channels indicate foreground; alpha is ignored. For LOCO, per-query masks are OR-ed before component extraction. Image/mask size mismatches fail; no GT resizing is performed. Four LOCO source paths ending in `.png` resolve to same-stem mask directories; details are recorded in the metadata. The builder writes the annotation JSON and `_meta.json` sidecar and leaves source files untouched. Metadata records the frozen source/output hashes; retain the dataset revision when reporting results.
