
## Download the datasets in the following link and extract them into corresponding folders.
+ [DS-MVTec](https://huggingface.co/datasets/DefectSpectrum/Defect_Spectrum/tree/main/DS-MVTec)
+ [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
+ [MVTec-LOCO](https://www.mvtec.com/company/research/datasets/mvtec-loco)
+ [VisA](https://github.com/amazon-science/spot-diff)(1-class setup)
+ [GoodsAD](https://github.com/jianzhang96/GoodsAD)


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
