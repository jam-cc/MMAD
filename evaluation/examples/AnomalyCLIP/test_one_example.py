from matplotlib import pyplot as plt

from . import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from .prompt_ensemble import AnomalyCLIP_PromptLearner
from PIL import Image

import os
import random
import numpy as np
from .utils import get_transform, normalize

import io
from matplotlib import cm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# from visualization import visualizer
import cv2


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def visualizer(path, anomaly_map, img_size):
    filename = os.path.basename(path)
    dirname = os.path.dirname(path)
    vis = cv2.cvtColor(cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
    mask = normalize(anomaly_map[0])
    vis = apply_ad_scoremap(vis, mask)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
    save_vis = os.path.join(dirname, f'anomaly_map_{filename}')
    print(save_vis)
    cv2.imwrite(save_vis, vis)

from scipy.ndimage import gaussian_filter
def test(args):
    img_size = args.image_size
    features_list = args.features_list
    image_path = args.image_path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    torch_home = os.getenv('TORCH_HOME') or os.getenv('XDG_CACHE_HOME') or os.path.expanduser('~/.cache')
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters, download_root = os.path.join(torch_home, 'clip'))
    model.eval()

    preprocess, target_transform = get_transform(args)


    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
    text_features = text_features/text_features.norm(dim=-1, keepdim=True)

    img = Image.open(image_path)
    img = preprocess(img)
    
    # print("img", img.shape)
    image = img.reshape(1, 3, img_size, img_size).to(device)
   
    with torch.no_grad():
        image_features, patch_features = model.encode_image(image, features_list, DPAM_layer = 20)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_probs = image_features @ text_features.permute(0, 2, 1)
        text_probs = (text_probs/0.07).softmax(-1)
        text_probs = text_probs[:, 0, 1]
        anomaly_map_list = []
        for idx, patch_feature in enumerate(patch_features):
            if idx >= args.feature_map_layer[0]:
                patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
                similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                anomaly_map = (similarity_map[...,1] + 1 - similarity_map[...,0])/2.0
                anomaly_map_list.append(anomaly_map)

        anomaly_map = torch.stack(anomaly_map_list)
        
        anomaly_map = anomaly_map.sum(dim = 0)
      
        anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in anomaly_map.detach().cpu()], dim = 0 )

        # visualizer(image_path, anomaly_map.detach().cpu().numpy(), args.image_size)
        plt.imshow(normalize(anomaly_map[0]))
        plt.axis('off')
        plt.show()
        return anomaly_map[0]

anomalyclip_args = argparse.Namespace(checkpoint_path ='checkpoints/9_12_4_multiscale/epoch_15.pth', features_list = [6, 12, 18, 24], image_size = 518, depth = 9, n_ctx = 12, t_n_ctx = 4, feature_map_layer = [0, 1, 2, 3], seed = 111, sigma = 4)
anomalyclip_args_visa = argparse.Namespace(checkpoint_path ='checkpoints/9_12_4_multiscale_visa/epoch_15.pth', features_list = [6, 12, 18, 24], image_size = 518, depth = 9, n_ctx = 12, t_n_ctx = 4, feature_map_layer = [0, 1, 2, 3], seed = 111, sigma = 4)

def load_model_and_features(args=anomalyclip_args):
    cache = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth,
                              "learnabel_text_embedding_length": args.t_n_ctx}
    torch_home = os.getenv('TORCH_HOME') or os.getenv('XDG_CACHE_HOME') or os.path.expanduser('~/.cache')
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters,
                                    download_root=os.path.join(torch_home, 'clip'))
    model.eval()

    preprocess, target_transform = get_transform(args)

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    cache['model'] = model
    cache['text_features'] = text_features
    cache['preprocess'] = preprocess
    return cache

def call_anomalyclip(image_path, cache=None, notation = "bbox", args=anomalyclip_args, visualize = False):
    anomalyclip_args.image_path = image_path
    # anomaly_map = test(args)
    ############################
    if cache is None:
        cache = load_model_and_features(args)
    model = cache['model']
    text_features = cache['text_features']
    preprocess = cache['preprocess']

    img_size = args.image_size
    image_path = args.image_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = Image.open(image_path)
    img = preprocess(img)
    image = img.reshape(1, 3, img_size, img_size).to(device)

    with torch.no_grad():
        image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_probs = image_features @ text_features.permute(0, 2, 1)
        text_probs = (text_probs / 0.07).softmax(-1)
        text_probs = text_probs[:, 0, 1]
        anomaly_map_list = []
        for idx, patch_feature in enumerate(patch_features):
            if idx >= args.feature_map_layer[0]:
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                anomaly_map_list.append(anomaly_map)

        anomaly_map = torch.stack(anomaly_map_list)
        anomaly_map = anomaly_map.sum(dim=0)
        anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in anomaly_map.detach().cpu()], dim=0)
        anomaly_map = anomaly_map[0]
    ############################

    anomaly_map_normalized = normalize(anomaly_map)
    anomaly_map_normalized = anomaly_map_normalized.numpy()
    # 读取图像
    query_image = Image.open(image_path).convert('RGB')
    original_size = query_image.size
    # 统一缩放到最大512
    max_size = 512
    if max(original_size) > max_size:
        scale = max_size / max(original_size)
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        query_image = query_image.resize(new_size)
        anomaly_map_normalized = cv2.resize(anomaly_map_normalized, new_size, interpolation=cv2.INTER_NEAREST)

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
                cv2.drawContours(query_image_np, [contour], -1, (255, 0, 0, 255), 2)  # 红色轮廓，宽度为2

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
                cv2.rectangle(query_image_np, (x, y), (x + w, y + h), (255, 0, 0, 255), 2)  # 红色边界框，宽度为2

        # 转换回 PIL Image
        combined_image = Image.fromarray(query_image_np)

    elif notation == "heatmap":
        # 选择一个 colormap，例如 'viridis'
        colormap = cm.get_cmap('viridis')
        colored_anomaly_map = colormap(anomaly_map_normalized)  # 正则化并应用 colormap
        colored_anomaly_map = (colored_anomaly_map[:, :, :3] * 255).astype(np.uint8)  # 转换为 8-bit RGB

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
        ax[0].axis('off')  # 关闭坐标轴
        ax[1].imshow(anomaly_map_normalized, cmap='viridis')
        ax[1].axis('off')  # 关闭坐标轴
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 减小边距
        plt.show()
    return image_buffer


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    # paths
    parser.add_argument("--image_path", type=str, default="./data/visa", help="path to test image")
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/9_12_4_multiscale_visa/epoch_15.pth', help='path to checkpoint')
    # model
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)
