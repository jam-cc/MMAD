from __future__ import division

import json
import logging
import os
from collections import defaultdict

import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from datasets.base_dataset import BaseDataset, TestBaseTransform, TrainBaseTransform
from datasets.image_reader import build_image_reader
from datasets.transforms import RandomColorJitter

logger = logging.getLogger("global_logger")


def build_coco_dataloader(cfg, training, distributed=True):

    image_reader = build_image_reader(cfg.image_reader)

    normalize_fn = transforms.Normalize(mean=cfg["pixel_mean"], std=cfg["pixel_std"])
    if training:
        transform_fn = TrainBaseTransform(
            cfg["input_size"], cfg["hflip"], cfg["vflip"], cfg["rotate"]
        )
    else:
        transform_fn = TestBaseTransform(cfg["input_size"])

    colorjitter_fn = None
    if cfg.get("colorjitter", None) and training:
        colorjitter_fn = RandomColorJitter.from_params(cfg["colorjitter"])

    logger.info("building CustomDataset from: {}".format(cfg["meta_file"]))

    dataset = CocoDataset(
        image_reader,
        cfg["meta_file"],
        training,
        transform_fn=transform_fn,
        normalize_fn=normalize_fn,
        colorjitter_fn=colorjitter_fn,
    )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
    )

    return data_loader


class CocoDataset(BaseDataset):
    def __init__(
        self,
        image_reader,
        meta_file,
        training,
        transform_fn,
        normalize_fn,
        colorjitter_fn=None,
    ):
        self.image_reader = image_reader
        self.meta_file = meta_file
        self.training = training
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.colorjitter_fn = colorjitter_fn

        self.coco = COCO(meta_file)

        with open(meta_file, "r") as f_r:
            json_data = json.load(f_r)
        self.metas = json_data["images"]
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        input = {}
        (
            classname,
            label,
            filename,
            anns,
            fg_mask,
            crop_roi,
            height,
            width,
            instance,
        ) = self.data_to_iterate[index]
        # read image
        image = self.image_reader(filename)

        input.update(
            {
                "filename": filename,
                "height": image.shape[0],
                "width": image.shape[1],
                "label": label,
            }
        )
        input["clsname"] = classname
        input["instance"] = instance

        image = Image.fromarray(image, "RGB")

        if anns is not None:
            mask = np.zeros((height, width)).astype(np.uint8)
            for ann in anns:
                if "OUTER" in ann["category_name"]:
                    continue
                bbox = ann["bbox_xyxy"]
                cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, -1)
        else:
            if label == 0:  # good
                mask = np.zeros((image.height, image.width)).astype(np.uint8)
            elif label == 1:  # defective
                mask = (np.ones((image.height, image.width)) * 255).astype(np.uint8)
            else:
                raise ValueError("Labels must be [None, 0, 1]!")

        mask = Image.fromarray(mask, "L")
        fg_mask = Image.fromarray(fg_mask, "L")

        if self.transform_fn:
            image, mask, fg_mask = self.transform_fn(image, mask, fg_mask, crop_roi)
        if self.colorjitter_fn:
            image = self.colorjitter_fn(image)
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        fg_mask = transforms.ToTensor()(fg_mask)
        if self.normalize_fn:
            image = self.normalize_fn(image)
        input.update({"image": image, "mask": mask, "fg_mask": fg_mask})
        return input

    def get_image_data(self):
        imgpaths_per_class = {}
        data_to_iterate = []
        for info in self.metas:
            imagepath = info["file_name"]
            label = 0 if info["sample_cat"] == "OK" else 1

            if "outer_polygon" in info:
                points = (
                    np.array(info["outer_polygon"]).reshape((-1, 2)).astype(np.int32)
                )
            else:
                # points = np.array(info["outer"]).reshape((-1, 2)).astype(np.int32)
                points = np.array(info["crop_info"]).reshape((-1, 2)).astype(np.int32)
            roi = np.array(info["crop_info"]).astype(np.int32)
            # roi = np.array(info["outer"]).astype(np.int32)

            img = Image.new("L", (info["width"], info["height"]), 0)
            ImageDraw.Draw(img).polygon(tuple(map(tuple, points)), outline=1, fill=1)
            foreground_mask = np.array(img)
            if "bottle" in imagepath:
                classname = imagepath.split("/")[-4]
            else:
                classname = imagepath.split("/")[-1].split("_")[2]
            # classname = imagepath.split("/")[-1].split("_")[2]
            instance = imagepath.split("/")[-1].split("_")[0]
            data_tuple = [classname, label, imagepath]
            if label == 1:
                anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=info["id"]))
            else:
                anns = None
            # data_tuple.append(mask)
            data_tuple = [
                classname,
                label,
                imagepath,
                anns,
                foreground_mask,
                roi,
                info["height"],
                info["width"],
                instance,
            ]
            data_to_iterate.append(data_tuple)
            if classname not in imgpaths_per_class:
                imgpaths_per_class[classname] = defaultdict(list)
            imgpaths_per_class[classname][label].append(imagepath)

        return imgpaths_per_class, data_to_iterate
