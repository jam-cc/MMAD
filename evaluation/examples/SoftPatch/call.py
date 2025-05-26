import contextlib
import io
import logging
import os
import sys

import time
import random
from pathlib import Path
import argparse

import cv2
import pynvml
import yaml

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import cm, pyplot as plt

# sys.path.append('src')

from .src import backbones as backbones
from .src import common as common
from .src import utils as utils
from .src import softpatch as softpatch
from .src import sampler


def set_torch_device():
    # Initialize pynvml
    pynvml.nvmlInit()

    device_count = pynvml.nvmlDeviceGetCount()
    max_free_memory = 0
    best_device = 0

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory = mem_info.free

        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_device = i

    # Release pynvml resources
    pynvml.nvmlShutdown()

    # Set PyTorch device
    device = torch.device(f'cuda:{best_device}' if torch.cuda.is_available() else 'cpu')
    return device


# Use automatically selected device
device = set_torch_device()

patchcore_args = argparse.Namespace(
    backbone_names=['wideresnet50'],
    layers_to_extract_from=['layer2', 'layer3'],
    faiss_on_gpu=False,
    faiss_num_workers=4,
    threshold=0,
    without_soft_weight=True,
    imagesize=224,
)

def build_patchcore(args=patchcore_args):
    input_shape = (3, args.imagesize, args.imagesize)
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

        # nn_method = common.SklearnNN()
        nn_method = common.FaissNN(args.faiss_on_gpu, args.faiss_num_workers, device=device.index)
        featuresampler = sampler.ApproximateGreedyCoresetSampler(percentage=0.99, device=device)

        coreset_instance = softpatch.SoftPatch(device)
        coreset_instance.load(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract_from,
            device=device,
            input_shape=input_shape,
            nn_method=nn_method,
            featuresampler=featuresampler,
            threshold=args.threshold,
            soft_weight_flag=not args.without_soft_weight,
        )
        loaded_coresets.append(coreset_instance)
    return loaded_coresets

def call_patchcore(image_path, few_shot, coresets, notation="bbox", args=patchcore_args, visualize=False):
    transform = transforms.Compose([
        transforms.Resize((args.imagesize, args.imagesize)),
        transforms.ToTensor(),  # Convert to Tensor
    ])
    query_image = Image.open(image_path).convert('RGB')
    original_size = query_image.size
    # Scale original_size to longest side 512
    max_size = 512
    if max(original_size) > max_size:
        scale = max_size / max(original_size)
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        query_image = query_image.resize(new_size)
    original_size = query_image.size
    query_image_tensor = transform(query_image)

    template_images = []
    for template_path in few_shot:
        template_image = Image.open(template_path).convert('RGB')
        template_image = transform(template_image)
        template_images.append(template_image)

    template_images = torch.stack(template_images)

    anomaly_map = []
    for coreset_instance in coresets:
        coreset_instance.fit([template_images])
        image_scores, masks = coreset_instance.predict(query_image_tensor.unsqueeze(0))
        anomaly_map.append(masks[0])

    # Calculate average anomaly_map
    anomaly_map = np.mean(anomaly_map, axis=0)
    anomaly_map = cv2.resize(anomaly_map, original_size, interpolation=cv2.INTER_LINEAR)



    ########## 分割前景
    # # 将 query_image 转换为灰度图像
    # query_image_gray = cv2.cvtColor(np.array(query_image), cv2.COLOR_RGB2GRAY)
    # # 使用 Otsu's 二值化方法得到前景掩码
    # _, fg_mask = cv2.threshold(query_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # # 将掩码转换为布尔类型
    # fg_mask = fg_mask == 255

    # 将 query_image 转换为 NumPy 数组
    query_image_np = np.array(query_image)
    # 使用 GrabCut 算法分割前景
    mask = np.zeros(query_image_np.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    # 初始化矩形区域，假设前景大致在图像中心
    height, width = query_image_np.shape[:2]
    rect = (int(width * 0.01), int(height * 0.01), int(width * 0.98), int(height * 0.98))

    # 应用 GrabCut 算法
    cv2.grabCut(query_image_np, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # 将 mask 转换为二值掩码
    # fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    fg_mask = mask != 2

    # 计算前景区域的大小
    foreground_area = np.sum(fg_mask)

    # 设置前景区域的最小阈值
    min_foreground_area = 0.05 * fg_mask.size  # 例如，前景区域至少占图像的5%

    # 如果前景区域太小，认为分割错误，不进行分割
    if foreground_area < min_foreground_area:
        print("前景区域太小，分割错误，不进行分割")
    else:
        # 过滤出前景区域的 anomaly_map
        filtered_anomaly_map = np.zeros_like(anomaly_map)
        filtered_anomaly_map[fg_mask] = anomaly_map[fg_mask]
        anomaly_map = filtered_anomaly_map
    if visualize:
        # 显示fg_mask和anomaly_map
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(fg_mask, cmap='gray')
        ax[0].set_title('Foreground Mask')
        ax[0].axis('off')
        ax[1].imshow(anomaly_map, cmap='viridis')
        ax[1].set_title('Anomaly Map')
        ax[1].axis('off')
        plt.show()

    # 归一化 anomaly_map 到 [0, 1] 范围
    # anomaly_map_normalized = (anomaly_map - np.min(anomaly_map)) / (np.max(anomaly_map) - np.min(anomaly_map))
    anomaly_map_normalized = anomaly_map/np.max(anomaly_map)

    if notation == "contour" or notation == "bbox":
        # 选择初始阈值
        initial_threshold = anomaly_map_normalized.max() * (1 - 1 / 3)
        _, binary_map = cv2.threshold(anomaly_map_normalized, initial_threshold, 1, cv2.THRESH_BINARY)

        # 如果标出的区域大于50%，则选择50%的节点作为阈值
        if np.count_nonzero(binary_map) / (binary_map.shape[0] * binary_map.shape[1]) > 0.5:
            sorted_values = np.sort(anomaly_map_normalized.flatten())
            threshold_value = sorted_values[int(0.5 * len(sorted_values))]
            _, binary_map = cv2.threshold(anomaly_map_normalized, threshold_value, 1, cv2.THRESH_BINARY)

        # 将 binary_map 转换为 8-bit 图像
        binary_map = (binary_map * 255).astype(np.uint8)

        # 查找轮廓
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 去除噪声
        contours = [contour for contour in contours if cv2.contourArea(contour) > 0.001 * query_image.size[0] * query_image.size[1]]

        # 将 query_image 转换为 RGBA
        query_image_np = np.array(query_image.convert("RGBA"))

        if notation == "contour":
            # 画轮廓
            for contour in contours:
                cv2.drawContours(query_image_np, [contour], -1, (255, 0, 0, 255), 2)  # Red contour, width 2

        elif notation == "bbox":
            # 画边界框
            # 获取每个轮廓的边界框
            bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

            # 计算两个边界框之间的距离
            def box_distance(box1, box2):
                x1, y1, w1, h1 = box1
                x2, y2, w2, h2 = box2
                center1 = (x1 + w1 / 2, y1 + h1 / 2)
                center2 = (x2 + w2 / 2, y2 + h2 / 2)
                return np.linalg.norm(np.array(center1) - np.array(center2))

            # 合并两个边界框
            def merge_two_boxes(box1, box2):
                x1, y1, w1, h1 = box1
                x2, y2, w2, h2 = box2
                nx = min(x1, x2)
                ny = min(y1, y2)
                nw = max(x1 + w1, x2 + w2) - nx
                nh = max(y1 + h1, y2 + h2) - ny
                return (nx, ny, nw, nh)

            # 合并边界框直到数量不超过2个
            while len(bounding_boxes) > 2:
                # 找到最接近的两个边界框
                min_distance = float('inf')
                min_pair = (0, 1)
                for i in range(len(bounding_boxes)):
                    for j in range(i + 1, len(bounding_boxes)):
                        dist = box_distance(bounding_boxes[i], bounding_boxes[j])
                        if dist < min_distance:
                            min_distance = dist
                            min_pair = (i, j)

                # 合并最接近的两个边界框
                i, j = min_pair
                new_box = merge_two_boxes(bounding_boxes[i], bounding_boxes[j])
                bounding_boxes = [box for k, box in enumerate(bounding_boxes) if k != i and k != j]
                bounding_boxes.append(new_box)

            # 画边界框
            for x, y, w, h in bounding_boxes:
                cv2.rectangle(query_image_np, (x, y), (x + w, y + h), (255, 0, 0, 255), 2)  # Red bounding box, width 2

        # 转换回 PIL Image
        combined_image = Image.fromarray(query_image_np)

    elif notation == "heatmap":
        # 选择一个 colormap，例如 'viridis'
        colormap = cm.get_cmap('viridis')
        colored_anomaly_map = colormap(anomaly_map_normalized)  # 正则化并应用 colormap
        colored_anomaly_map = (colored_anomaly_map[:, :, :3] * 255).astype(np.uint8)  # Convert to 8-bit RGB

        anomaly_map_image = Image.fromarray(colored_anomaly_map).convert("RGBA")

        combined_image = anomaly_map_image

    elif notation == "highlight":
        # 选择一个 colormap，例如 'viridis'
        colormap = cm.get_cmap('viridis')
        colored_anomaly_map = colormap(anomaly_map_normalized)  # 正则化并应用 colormap
        colored_anomaly_map = (colored_anomaly_map[:, :, :3] * 255).astype(np.uint8)  # 转换为 8-bit RGB

        # 将 anomaly_map 透明化 0.7
        anomaly_map_image = Image.fromarray(colored_anomaly_map).convert("RGBA")
        alpha = 0.7
        anomaly_map_with_alpha = Image.new("RGBA", anomaly_map_image.size)
        anomaly_map_with_alpha = Image.blend(anomaly_map_with_alpha, anomaly_map_image, alpha)

        # 将 query_image 转换为 RGBA
        query_image = query_image.convert("RGBA")

        # 叠加 anomaly_map 到 query_image 上
        combined_image = Image.alpha_composite(query_image, anomaly_map_with_alpha)


    image_buffer = io.BytesIO()
    combined_image.save(image_buffer, format='PNG')
    image_buffer.seek(0)
    if visualize:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(combined_image)
        ax[0].axis('off')  # Turn off coordinate axes
        ax[1].imshow(anomaly_map, cmap='viridis')
        ax[1].axis('off')  # Turn off coordinate axes
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Reduce margins
        plt.show()
    return image_buffer


# 使用 matplotlib 的 colormap 进行颜色映射处理
def apply_colormap(anomaly_map):
    colormap = cm.get_cmap('viridis')  # 选择一个 colormap，例如 'viridis'
    colored_anomaly_map = colormap(anomaly_map / np.max(anomaly_map))  # 正则化并应用 colormap
    colored_anomaly_map = (colored_anomaly_map[:, :, :3] * 255).astype(np.uint8)  # 转换为 8-bit RGB
    return colored_anomaly_map


def call_ground_truth(image_path, gt_path, notation="bbox", visualize=False):
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size

    if gt_path is None:
        # 如果没有 ground truth，直接返回原图
        image_buffer = io.BytesIO()
        image.save(image_buffer, format='PNG')
        image_buffer.seek(0)
        return image_buffer

    # 读取ground truth
    if os.path.isdir(os.path.splitext(gt_path)[0]):
        # 如果gt_path是一个文件夹，读取所有mask图像并合并
        masks = []
        for filename in os.listdir(os.path.splitext(gt_path)[0]):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                mask = cv2.imread(os.path.join(os.path.splitext(gt_path)[0], filename), cv2.IMREAD_GRAYSCALE)
                masks.append(mask)
        gt = np.max(masks, axis=0)  # 合并所有mask
    else:
        # 如果gt_path是一个单一的图像文件
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            gt_path = gt_path.replace(".jpg", ".png")
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    # 统一缩放到最大512
    max_size = 512
    if max(original_size) > max_size:
        scale = max_size / max(original_size)
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        image = image.resize(new_size)
        gt = cv2.resize(gt, new_size, interpolation=cv2.INTER_NEAREST)

    image = np.array(image)

    # 处理不同的mask格式
    if np.max(gt) == 1:
        # 0是背景，1是标注的mask
        mask = gt
    elif np.max(gt) == 255:
        # 0是背景，255是标注的mask
        mask = gt // 255
    else:
        # 0是背景，标注的mask有多种颜色来表示多个瑕疵
        mask = (gt > 0).astype(np.uint8)

    if notation == "mask":
        # 如果是mask，直接输出gt
        # 获取 viridis 颜色映射
        viridis = cm.get_cmap('viridis')
        normalized_gt_image = mask / mask.max()
        colored_gt_image = viridis(normalized_gt_image)
        gt_image = (colored_gt_image[:, :, :3] * 255).astype(np.uint8)
    elif notation == "contour":
        # 如果是contour，找到轮廓并绘制在图像上
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gt_image = cv2.drawContours(image.copy(), contours, -1, (255, 0, 0), 2)
    elif notation == "bbox":
        # 如果是bbox，从mask转换为bbox
        gt_image = image.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 获取每个轮廓的边界框
        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

        # 计算两个边界框之间的距离
        def box_distance(box1, box2):
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            center1 = (x1 + w1 / 2, y1 + h1 / 2)
            center2 = (x2 + w2 / 2, y2 + h2 / 2)
            return np.linalg.norm(np.array(center1) - np.array(center2))

        # 合并两个边界框
        def merge_two_boxes(box1, box2):
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            nx = min(x1, x2)
            ny = min(y1, y2)
            nw = max(x1 + w1, x2 + w2) - nx
            nh = max(y1 + h1, y2 + h2) - ny
            return (nx, ny, nw, nh)

        # 合并边界框直到数量不超过2个
        while len(bounding_boxes) > 3:
            # 找到最接近的两个边界框
            min_distance = float('inf')
            min_pair = (0, 1)
            for i in range(len(bounding_boxes)):
                for j in range(i + 1, len(bounding_boxes)):
                    dist = box_distance(bounding_boxes[i], bounding_boxes[j])
                    if dist < min_distance:
                        min_distance = dist
                        min_pair = (i, j)

            # 合并最接近的两个边界框
            i, j = min_pair
            new_box = merge_two_boxes(bounding_boxes[i], bounding_boxes[j])
            bounding_boxes = [box for k, box in enumerate(bounding_boxes) if k != i and k != j]
            bounding_boxes.append(new_box)

        gt_image = image.copy()
        # 画边界框
        for x, y, w, h in bounding_boxes:
            cv2.rectangle(gt_image, (x, y), (x + w, y + h), (255, 0, 0, 255), 2)  # Red bounding box, width 2
    elif notation == "highlight":
        viridis = cm.get_cmap('viridis')
        normalized_gt_image = mask / mask.max()
        colored_gt_image = viridis(normalized_gt_image)
        heatmap = (colored_gt_image[:, :, :3] * 255).astype(np.uint8)
        gt_image = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    else:
        raise ValueError("Unsupported notation type")

    # 将图像保存到内存缓冲区
    gt_image = Image.fromarray(gt_image)
    image_buffer = io.BytesIO()
    gt_image.save(image_buffer, format='PNG')
    image_buffer.seek(0)

    if visualize:
        plt.imshow(gt_image)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Reduce margins
        plt.show()

    return image_buffer
