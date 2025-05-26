import contextlib
import logging
import os
import sys

import time
import random
from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.data import Subset, ConcatDataset

sys.path.append('src')
import src.backbones as backbones
import src.common as common
import src.metrics as metrics
import src.sampler as sampler
import src.utils as utils
import src.softpatch as softpatch
import src.datasets as datasets
LOGGER = logging.getLogger(__name__)

_DATASETS = {"ad": ["datasets.mvtec", "MVTecDataset"],
            "mvtec": ["datasets.mvtec", "MVTecDataset"],
             "btad": ["datasets.btad", "BTADDataset"]}


def parse_args():
    # project
    parser = argparse.ArgumentParser(description='SoftPatch')
    parser.add_argument('--gpu', type=int, default=[], action='append')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_path", type=str, default="result")
    parser.add_argument("--log_project", type=str, default="project")
    parser.add_argument("--log_group", type=str, default="group")
    parser.add_argument("--save_segmentation_images", action='store_true')
    # backbone
    parser.add_argument("--backbone_names", "-b", type=str, action='append', default=['wideresnet50'])
    parser.add_argument("--layers_to_extract_from", "-le", type=str, action='append', default=['layer2', 'layer3'])
    # coreset sampler
    parser.add_argument("--sampler_name", type=str, default="approx_greedy_coreset")
    parser.add_argument("--sampling_ratio", type=float, default=0.1)
    parser.add_argument("--faiss_on_gpu", action='store_true')
    parser.add_argument("--faiss_num_workers", type=int, default=4)
    # SoftPatch hyper-parameter
    parser.add_argument("--weight_method", type=str, default="lof")
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--lof_k", type=int, default=6)
    parser.add_argument("--without_soft_weight", action='store_true')
    # dataset
    parser.add_argument("--dataset", type=str, default="ad")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--subdatasets", "-d", action='append', type=str, required=True)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--resize", default=256, type=int)
    parser.add_argument("--imagesize", default=224, type=int)
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--overlap", action='store_true')
    parser.add_argument("--noise_augmentation", action='store_true')
    parser.add_argument("--fold", type=int, default=0)

    args = parser.parse_args()
    return args


def get_dataloaders(args):
    data_path = args.data_path
    batch_size = args.batch_size
    resize = args.resize
    imagesize = args.imagesize
    noise = args.noise
    overlap = args.overlap
    noise_augmentation = args.noise_augmentation
    fold = args.fold

    dataset_info = _DATASETS[args.dataset]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])
    dataloaders = []
    for subdataset in args.subdatasets:
        train_dataset = dataset_library.__dict__[dataset_info[1]](
            source=data_path,
            classname=subdataset,
            resize=resize,
            imagesize=imagesize,
            split=dataset_library.DatasetSplit.TRAIN,
        )

        test_dataset = dataset_library.__dict__[dataset_info[1]](
            source=data_path,
            classname=subdataset,
            resize=resize,
            imagesize=imagesize,
            split=dataset_library.DatasetSplit.TEST,
        )

        if noise >= 0:
            anomaly_index = [index for index in range(len(test_dataset)) if test_dataset[index]["is_anomaly"]]
            train_length = len(train_dataset)
            noise_number = int(noise * train_length)

            if noise_number > (len(anomaly_index)/2):
                noise_number = int(len(anomaly_index) / 2)
                # Adjust the number of normal samples so that the ratio of noise_number to samples in train_dataset is noise
                train_normal_number = noise_number/noise
                # Randomly select train_normal_number samples from train_dataset as normal samples
                train_normal_index = random.sample(range(len(train_dataset)), int(train_normal_number))
                train_normal_dataset = Subset(train_dataset, train_normal_index)
                train_normal_dataset.imagesize = train_normal_dataset.dataset.imagesize

            else:
                train_normal_dataset = train_dataset


            LOGGER.info("{} anomaly samples are being added into train dataset as noise.".format(noise_number))

            noise_index_path = Path("noise_index" + "/" + str(args.dataset) + "_noise"
                                    + str(noise) + "_fold" + str(fold))

            noise_index_path.mkdir(parents=True, exist_ok=True)
            path = noise_index_path / (subdataset + "-noise" + str(noise) + ".pth")
            if path.exists() and fold != 0:
                noise_index = torch.load(path)
                assert len(noise_index) == noise_number
            else:
                noise_index = random.sample(anomaly_index, noise_number)
                torch.save(noise_index, path)

            noise_dataset = Subset(test_dataset, noise_index)
            if noise_augmentation:
                noise_dataset = datasets.NoiseDataset(noise_dataset)
            train_dataset = ConcatDataset([train_normal_dataset, noise_dataset])

            train_dataset.imagesize = train_dataset.datasets[0].imagesize

            if not overlap:
                new_test_data_index = list(set(range(len(test_dataset))) - set(noise_index))
                test_dataset = Subset(test_dataset, new_test_data_index)
            else:
                test_dataset = Subset(test_dataset, range(len(test_dataset)))


        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

        train_dataloader.name = args.dataset
        if subdataset is not None:
            train_dataloader.name += "_" + subdataset

        dataloader_dict = {
            "training": train_dataloader,
            "testing": test_dataloader,
        }

        dataloaders.append(dataloader_dict)
    return dataloaders


def get_sampler(sampler_name, sampling_ratio, device):
    if sampler_name == "identity":
        return sampler.IdentitySampler()
    elif sampler_name == "greedy_coreset":
        return sampler.GreedyCoresetSampler(sampling_ratio, device)
    elif sampler_name == "approx_greedy_coreset":
        return sampler.ApproximateGreedyCoresetSampler(sampling_ratio, device)


def get_coreset(args, imagesize, sampler, device):
    input_shape = (3, imagesize, imagesize)
    backbone_names = list(args.backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in args.layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [args.layers_to_extract_from]

    loaded_coresets = []
    for backbone_name, layers_to_extract_from in zip(
        backbone_names, layers_to_extract_from_coll
    ):
        backbone_seed = None
        if ".seed-" in backbone_name:
            backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                backbone_name.split("-")[-1]
            )
        backbone = backbones.load(backbone_name)
        backbone.name, backbone.seed = backbone_name, backbone_seed

        nn_method = common.FaissNN(args.faiss_on_gpu, args.faiss_num_workers, device=device.index)

        coreset_instance = softpatch.SoftPatch(device)
        coreset_instance.load(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract_from,
            device=device,
            input_shape=input_shape,
            featuresampler=sampler,
            nn_method=nn_method,
            threshold=args.threshold,
            soft_weight_flag=not args.without_soft_weight,
        )
        loaded_coresets.append(coreset_instance)
    return loaded_coresets


def run(args):

    seed = args.seed
    run_save_path = utils.create_storage_folder(
        args.results_path, args.log_project, args.log_group, mode="iterate"
    )

    device = utils.set_torch_device(args.gpu)
    LOGGER.info(device)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )
    utils.fix_seeds(seed, device)
    result_collect = []

    list_of_dataloaders = get_dataloaders(args)
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        dataset_name = dataloaders["training"].name
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )
        start_time = time.time()

        with device_context:
            torch.cuda.empty_cache()
            sampler = get_sampler(args.sampler_name, args.sampling_ratio, device)
            coreset_list = get_coreset(args, args.imagesize, sampler, device)
            if len(coreset_list) > 1:
                LOGGER.info(
                    "Utilizing Coreset Ensemble (N={}).".format(len(coreset_list))
                )
            for i, coreset in enumerate(coreset_list):
                torch.cuda.empty_cache()
                if coreset.backbone.seed is not None:
                    utils.fix_seeds(coreset.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(coreset_list))
                )
                # for epoch in range(20):
                #     coreset._train(dataloaders["training"])
                coreset.fit(dataloaders["training"])
            train_end = time.time()
            torch.cuda.empty_cache()
            aggregator = {"scores": [], "segmentations": []}
            for i, coreset in enumerate(coreset_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(coreset_list)
                    )
                )
                scores, segmentations, labels_gt, masks_gt = coreset.predict(
                    dataloaders["testing"]
                )
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores + 1e-5)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            # anomaly_labels = [
            #     x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            # ]

            test_end = time.time()
            LOGGER.info("Training time:{}, Testing time:{}".format(train_end - start_time, test_end - train_end))

            # (Optional) Plot example images.
            if args.save_segmentation_images:
                # dataset = dataloaders["testing"].dataset
                image_paths = [
                    x[2] for x in
                    dataloaders["testing"].dataset.dataset.data_to_iterate[dataloaders["testing"].dataset.indices]
                ]
                mask_paths = [
                    x[3] for x in
                    dataloaders["testing"].dataset.dataset.data_to_iterate[dataloaders["testing"].dataset.indices]
                ]

                def image_transform(image):
                    in_std = np.array(
                        dataloaders["testing"].dataset.dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        dataloaders["testing"].dataset.dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.dataset.transform_mask(mask).numpy()

                image_save_path = os.path.join(
                    run_save_path, "segmentation_images", dataset_name
                )
                os.makedirs(image_save_path, exist_ok=True)
                utils.plot_segmentation_images(
                    image_save_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                    # dataset=dataset
                )

            LOGGER.info("Computing evaluation metrics.")
            auroc = metrics.compute_imagewise_retrieval_metrics(
                scores, labels_gt
            )["auroc"]

            # Compute PRO score & PW Auroc for all images
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )
            full_pixel_auroc = pixel_scores["auroc"]

            # Compute PRO score & PW Auroc only images with anomalies
            # sel_idxs = []
            # for i in range(len(masks_gt)):
            #     if np.sum(masks_gt[i]) > 0:
            #         sel_idxs.append(i)
            # pixel_scores = coreset.metrics.compute_pixelwise_retrieval_metrics(
            #     [segmentations[i] for i in sel_idxs],
            #     [masks_gt[i] for i in sel_idxs],
            # )
            # anomaly_pixel_auroc = pixel_scores["auroc"]

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "image_auroc": auroc,
                    "pixel_auroc": full_pixel_auroc,
                    # "anomaly_pixel_auroc": anomaly_pixel_auroc,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:4.4f}".format(key, item))

        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    args = parse_args()
    run(args)
