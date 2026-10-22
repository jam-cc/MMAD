
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

[`mmad_bbox.json`](mmad_bbox.json) contains 5,076 queries with nonempty GT masks and 10,450 connected-region boxes. See [the evaluation guide](../../evaluation/MMAD_BBox/README.md) for coordinates, template selection, inference and BBox-Mask IoU scoring. [`mmad_bbox_meta.json`](mmad_bbox_meta.json) records the frozen source hash and construction rules. Normal-image false positives are outside this localization manifest.
