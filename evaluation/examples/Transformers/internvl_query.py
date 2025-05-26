import argparse
import base64
import json
from collections import defaultdict

import math
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Set environment variable export HF_HOME=~/.cache/huggingface
os.environ["HF_HOME"] = "~/.cache/huggingface"

import time
import random
import re
import pandas as pd
import torch
from tqdm import tqdm
import sys
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T

sys.path.append("..")
from helper.summary import caculate_accuracy_mmad
from GPT4.gpt4v import GPT4Query, instruction

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVLQuery(GPT4Query):
    def __init__(self, image_path, text_gt, tokenizer, model, few_shot=[], visualization=False, domain_knowledge=None, agent=None, mask_path=None, CoT=None, defect_shot=[], args=None):
        super(InternVLQuery, self).__init__(image_path, text_gt, few_shot, visualization)
        self.tokenizer = tokenizer
        self.model = model
        self.domain_knowledge = domain_knowledge
        self.agent = agent
        self.mask_path = mask_path
        self.CoT = CoT
        self.defect_shot = defect_shot
        self.args = args

    def generate_answer(self):
        questions, answers = self.parse_conversation(self.text_gt)
        if questions == [] or answers == []:
            return questions, answers, None
        query_image = load_image(self.image_path, max_num=1).to(torch.bfloat16).cuda() # Default is 12 patches
        template_image = []
        for ref_image_path in self.few_shot:
            template_image.append(load_image(ref_image_path, max_num=1).to(torch.bfloat16).cuda())
        for ref_image_path in self.defect_shot:
            template_image.append(load_image(ref_image_path, max_num=1).to(torch.bfloat16).cuda())
        images = template_image + [query_image]
        if self.agent:
            if args.agent_model == "GT":
                anomaly_map = call_ground_truth(self.image_path, self.mask_path, notation=self.args.agent_notation, visualize=self.visualization)
            elif args.agent_model == "PatchCore":
                anomaly_map = call_patchcore(self.image_path, self.few_shot, self.agent, notation=args.agent_notation, visualize=self.visualization)
            elif "AnomalyCLIP" in args.agent_model:
                anomaly_map = call_anomalyclip(self.image_path, self.agent, notation=args.agent_notation, visualize=self.visualization)
            images = images + [load_image(anomaly_map, max_num=1).to(torch.bfloat16).cuda()]
        if self.visualization:
            self.visualize_image(cv2.imread(self.image_path))
            for ref_image_path in self.few_shot:
                self.visualize_image(cv2.imread(ref_image_path))
        pixel_values = torch.cat(images, dim=0)

        num_patches_list = [image.shape[0] for image in images]

        gpt_answers = []
        history = None

        if self.CoT:
            chain_of_thought = []
            chain_of_thought.append(
                {
                    # "type": "text",
                    "text": f"Before answering the following questions, we need to understand the query image. \n"
                            f"First, identify the objects in the template image and imagine the possible types of defects. \n"
                            f"Second, simply describe the differences between the query image with the template sample. \n"
                            f"Third, if exist, judge whether those differences are defect or normal. \n"
                            f"Output a simple thoughts in 100 words. Notice that slight movement of objects or changes in image quality fall within the normal range.\n"
                    # f"First, summary the normal and defect samples and answer why defect sample abnormal. \n"
                    # f"Second, simply describe the differences between the test image with the normal and defect sample. \n"
                    # f"Third, judge whether those differences are defect or normal. \n"
                },
            )

            payload, conversation_text = self.get_query(chain_of_thought)
            query = payload + conversation_text
            response, temp_history = model.chat(tokenizer, pixel_values, query,
                                                dict(max_new_tokens=256, do_sample=False),
                                                num_patches_list=num_patches_list,
                                                history=history, return_history=True)
            print(response)
            history = temp_history

        for i in range(len(questions)):
            part_questions = questions[i:i + 1]
            payload, conversation_text = self.get_query(part_questions)
            query = payload + conversation_text
            response, temp_history = model.chat(tokenizer, pixel_values, query,
                                                dict(max_new_tokens=128, do_sample=False),
                                                num_patches_list=num_patches_list,
                                                history=history, return_history=True)
            if args.record_history:
                history = temp_history
            print(response)
            gpt_answer = self.parse_answer(response)
            gpt_answers.append(gpt_answer[-1])
        print(gpt_answers)
        return questions, answers, gpt_answers

    def get_query(self, conversation):
        incontext = ''
        if self.domain_knowledge:
            incontext = incontext + f"Following is the domain knowledge which contains some type of defect and the normal object characteristics: \n {self.domain_knowledge}"

        if self.few_shot:
            incontext = incontext + f"Following is/are {len(self.few_shot)} image of normal sample, which can be used as a template to compare the image being queried."
            for ref_image_path in self.few_shot:
                incontext = incontext + '\n' + ''' <image>\n'''

        if self.defect_shot:
            incontext = incontext + f"Following is/are {len(self.defect_shot)} image of defect sample."
            for ref_image_path in self.defect_shot:
                incontext = incontext + '\n' + ''' <image>\n'''

        if self.visualization:
            print(conversation)
        payload = (instruction + incontext + '\n' +
                   f"Following is the query image: " + '\n' + f"<image> ")
        if self.agent:
            if self.args.agent_notation == "bbox":
                payload = payload + f"Following is the reference image where some suspicious areas are marked by red bounding boxes. Please note that even if the query image does not have any defects, the red boxes still highlight some areas. Therefore, you still need to make a judgement by yourself. <image>"
            elif self.args.agent_notation == "contour":
                payload = payload + f"Following is the reference image where some suspicious areas are marked by red contour. Please note that even if the query image does not have any defects, the image still highlight some areas. Therefore, you still need to make a judgement by yourself. <image>"
            elif self.args.agent_notation == "highlight":
                payload = payload + f"Following is the reference image where some suspicious areas are highlight. Please note that even if the query image does not have any defects, the image still highlight some areas. Therefore, you still need to make a judgement by yourself. <image>"
            elif self.args.agent_notation == "mask":
                payload = payload + f"Following is the mask of some suspicious areas. But you still need to make a judgement by yourself. <image>"

        payload = payload + '\n' + f"Following is the question list. Answer with the option's letter from the given choices directly: " + '\n'

        conversation_text = ''
        for q in conversation:
            conversation_text += f"{q['text']}" + '\n'

        return payload, conversation_text


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="../../InternVL/pretrained/InternVL2-1B")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--few_shot_model", type=int, default=1)
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--similar_template", action="store_true")
    parser.add_argument("--record_history", action="store_true")

    parser.add_argument("--domain_knowledge", action="store_true")
    parser.add_argument("--domain_knowledge_path", type=str, default="../../../dataset/MMAD/domain_knowledge.json")
    parser.add_argument("--agent", action="store_true")
    parser.add_argument("--agent_model", type=str, default="GT", choices=["GT", "PatchCore", "AnomalyCLIP", "AnomalyCLIP_mvtec"])
    parser.add_argument("--agent_notation", type=str, default="bbox", choices=["bbox", "contour", "highlight", "mask", "heatmap"])

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--visualization", action="store_true")
    parser.add_argument("--CoT", action="store_true")
    parser.add_argument("--defect_shot", type=int, default=0)


    args = parser.parse_args()

    torch.manual_seed(1234)
    model_path = args.model_path
    model_name = os.path.split(model_path.rstrip('/'))[-1]

    torch.set_grad_enabled(False)
    if args.num_gpus > 1:
        device_map = split_model(model_name)
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map).eval()
    else:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    if args.similar_template:
        model_name += "_Similar_template"
    if args.domain_knowledge:
        model_name += "_Domain_knowledge"
        with open(args.domain_knowledge_path, "r") as file:
            all_domain_knowledge = json.load(file)
    agent = None
    if args.agent:
        model_name += "_Agent-" + args.agent_model + "-with-" + args.agent_notation
        if args.agent_model == "GT":
            agent = "GT"
        elif args.agent_model == "PatchCore":
            from SoftPatch.call import call_patchcore, build_patchcore, call_ground_truth
            agent = build_patchcore()
        elif args.agent_model == "AnomalyCLIP":
            from AnomalyCLIP.test_one_example import call_anomalyclip, load_model_and_features, anomalyclip_args, \
                anomalyclip_args_visa

            anomalyclip_args_visa.checkpoint_path = os.path.join("../AnomalyCLIP", anomalyclip_args_visa.checkpoint_path)
            agent = load_model_and_features(anomalyclip_args_visa)
        elif args.agent_model == "AnomalyCLIP_mvtec":
            from AnomalyCLIP.test_one_example import call_anomalyclip, load_model_and_features, anomalyclip_args

            anomalyclip_args.checkpoint_path = os.path.join("../AnomalyCLIP", anomalyclip_args.checkpoint_path)
            agent = load_model_and_features(anomalyclip_args)
    if args.CoT:
        model_name += "_CoT"
    if args.debug:
        model_name += "_Debug"
    if args.defect_shot >= 1:
        model_name += f"_{args.defect_shot}_defect_shot"


    answers_json_path = f"result/answers_{args.few_shot_model}_shot_{model_name}.json"
    if not os.path.exists("result"):
        os.makedirs("result")
    print(f"Answers will be saved at {answers_json_path}")
    # For storing all answers
    if os.path.exists(answers_json_path):
        with open(answers_json_path, "r") as file:
            all_answers_json = json.load(file)
    else:
        all_answers_json = []

    existing_images = [a["image"] for a in all_answers_json]

    cfg = {
        "data_path": "../../../dataset/MMAD",
        "json_path": "../../../dataset/MMAD/mmad.json",
    }
    args.data_path = cfg["data_path"]

    with open(cfg["json_path"], "r") as file:
        chat_ad = json.load(file)

    if args.debug:
        # Fix random seed
        random.seed(1)
        # random.seed(10)
        sample_keys = random.sample(list(chat_ad.keys()), 1600)
    else:
        sample_keys = chat_ad.keys()

    # Create a dictionary to store image paths for each defect category
    defect_images = defaultdict(list)
    # Iterate through dictionary keys and store image paths in corresponding defect categories
    for image_path in sample_keys:
        dataset_name = image_path.split("/")[0].replace("DS-MVTec", "MVTec")
        object_name = image_path.split("/")[1]
        defect_name = image_path.split("/")[2]

        # Use (dataset_name, object_name, defect_name) as key
        defect_key = (dataset_name, object_name, defect_name)
        defect_images[defect_key].append(image_path)

    for data_id, image_path in enumerate(tqdm(sample_keys)):
        if image_path in existing_images and not args.reproduce:
            continue
        text_gt = chat_ad[image_path]
        if args.similar_template:
            few_shot = text_gt["similar_templates"][:args.few_shot_model]
        else:
            few_shot = text_gt["random_templates"][:args.few_shot_model]

        rel_image_path = os.path.join(args.data_path, image_path)
        rel_few_shot = [os.path.join(args.data_path, path) for path in few_shot]

        if args.domain_knowledge:
            dataset_name = image_path.split("/")[0].replace("DS-MVTec", "MVTec")
            object_name = image_path.split("/")[1]

            domain_knowledge = '\n'.join(all_domain_knowledge[dataset_name][object_name].values())
        else:
            domain_knowledge = None

        # Get all image paths in the same defect category
        images_in_defect = defect_images[defect_key]
        # Randomly select multiple images, excluding the current image, and ensure the number of selections does not exceed the number of available images
        defect_shot = random.sample([img for img in images_in_defect if img != image_path],
                                        min(args.defect_shot, len(images_in_defect) - 1))
        rel_defect_shot = [os.path.join(args.data_path, path) for path in defect_shot]
        if text_gt["mask_path"]:
            rel_mask_path = os.path.join(args.data_path, image_path.split("/")[0], image_path.split("/")[1],text_gt["mask_path"])
        else:
            rel_mask_path = None
        internvlquery = InternVLQuery(image_path=rel_image_path, text_gt=text_gt,
                                      tokenizer=tokenizer, model=model, few_shot=rel_few_shot, visualization=args.visualization,
                                      domain_knowledge=domain_knowledge, agent=agent, mask_path=rel_mask_path, CoT=args.CoT, defect_shot=rel_defect_shot, args=args)
        questions, answers, gpt_answers = internvlquery.generate_answer()
        if gpt_answers is None or len(gpt_answers) != len(answers):
            print(f"Error at {image_path}")
            continue
        correct = 0
        for i, answer in enumerate(answers):
            if gpt_answers[i] == answer:
                correct += 1
        accuracy = correct / len(answers)
        print(f"Accuracy: {accuracy:.2f}")

        questions_type = [conversion["type"] for conversion in text_gt["conversation"]]
        # Update answer record
        for q, a, ga, qt in zip(questions, answers, gpt_answers, questions_type):
            answer_entry = {
                "image": image_path,
                "question": q,
                "question_type": qt,
                "correct_answer": a,
                "gpt_answer": ga
            }

            all_answers_json.append(answer_entry)

        if data_id % 10 == 0 or data_id == len(chat_ad.keys()) - 1:
            # Save answers as JSON
            with open(answers_json_path, "w") as file:
                json.dump(all_answers_json, file, indent=4)

    caculate_accuracy_mmad(answers_json_path)
