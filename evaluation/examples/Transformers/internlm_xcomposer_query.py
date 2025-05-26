import argparse
import base64
import json

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

import seaborn as sns

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

sys.path.append("..")
# from data.mvtec import ADDataset
from helper.summary import caculate_accuracy_mmad
from GPT4.gpt4v import GPT4Query, instruction
from LLaVA_Query.llava_query import LLaVAQuery


sys.path.append("../")


class InterVLQuery(GPT4Query):
    def __init__(self, image_path, text_gt, tokenizer, model, few_shot=[], visualization=False):
        super(InterVLQuery, self).__init__(image_path, text_gt, few_shot, visualization)
        self.tokenizer = tokenizer
        self.model = model

    def pre_encoder_image(self, image, resize=336*3):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            image = image.resize((resize, resize))
            image = self.model.vis_processor(image).unsqueeze(0).to(self.model.device)
        else:
            assert isinstance(image, torch.Tensor)
        # Check and synchronize model and image precision
        model_dtype = next(self.model.parameters()).dtype
        if image.dtype != model_dtype:
            image = image.to(model_dtype)
        return image

    def generate_answer(self):
        questions, answers = self.parse_conversation(self.text_gt)
        if questions == [] or answers == []:
            return questions, answers, None
        query_image = self.model.encode_img(self.pre_encoder_image(self.image_path))
        template_image = []
        for ref_image_path in self.few_shot:
            template_image.append(self.model.encode_img(self.pre_encoder_image(ref_image_path)))
        image = template_image + [query_image]
        image = torch.stack(image).squeeze(1).cuda()

        gpt_answers = []
        history = []
        for i in range(len(questions)):
            part_questions = questions[i:i + 1]
            payload, conversation_text = self.get_query(part_questions)
            query = payload + conversation_text
            with torch.cuda.amp.autocast():
                inputs, im_mask = self.model.interleav_wrap_chat(self.tokenizer, query=query, image=image, history=history, meta_instruction='')
                response = self.model.generate(
                    inputs_embeds=inputs['inputs_embeds'],
                    streamer=None,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=0.0,
                    repetition_penalty=1.005,
                    im_mask=im_mask,
                )
                response = self.tokenizer.decode(response[0], skip_special_tokens=True)
                response = response.split('[UNUSED_TOKEN_145]')[0]
            print(response)
            gpt_answer = self.parse_answer(response)
            gpt_answers.append(gpt_answer[-1])
        print(gpt_answers)
        return questions, answers, gpt_answers


    def get_query(self, conversation):
        incontext = ''
        if self.few_shot:
            incontext = incontext+ f"Following is/are {len(self.few_shot)} image of normal sample, which can be used as a template to compare the image being queried."
        for ref_image_path in self.few_shot:
                incontext = incontext + '''<ImageHere> '''

        if self.visualization:
            print(conversation)
        payload = instruction + incontext + '\n' + \
                    f"Following is the query image: " + '\n' + f"<ImageHere> " + '\n' + f"Following is the question list. Answer with the option's letter from the given choices directly: " + '\n'

        conversation_text = ''
        for q in conversation:
            conversation_text += f"{q['text']}" + '\n'

        return payload, conversation_text


def auto_configure_device_map(num_gpus):
    # visual_encoder counts as 4 layers
    # internlm_model.model.embed_tokens occupies 1 layer
    # norm and lm_head occupy 1 layer
    # transformer.layers occupy 32 layers
    # Total 34 layers distributed across num_gpus cards
    num_trans_layers = 32
    per_gpu_layers = 38 / num_gpus

    device_map = {
        'vit': 0,
        'vision_proj': 0,
        'model.tok_embeddings': 0,
        'model.norm': num_gpus - 1,
        'output': num_gpus - 1,
    }

    used = 3
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'model.layers.{i}'] = gpu_target
        used += 1

    return device_map


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="internlm/internlm-xcomposer2-vl-7b")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--few_shot_model", type=int, default=1)
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--similar_template", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(1234)
    model_path = args.model_path

    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir="~/.cache/huggingface/hub")
    if args.dtype == "bf16":
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, cache_dir="~/.cache/huggingface/hub")
    else:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, cache_dir="~/.cache/huggingface/hub")

    if args.num_gpus > 1:
        from accelerate import dispatch_model, infer_auto_device_map
        # cuda_list = '0,1'.split(',')
        # memory = '23GiB'
        # no_split_module_classes = model._no_split_modules
        # max_memory = {int(cuda): memory for cuda in cuda_list}
        # device_map = infer_auto_device_map(model, max_memory=max_memory,
        #                                    no_split_module_classes=no_split_module_classes)  # Automatically assign device for each layer
        device_map = auto_configure_device_map(args.num_gpus)
        model = dispatch_model(model, device_map=device_map)
    if args.dtype == "fp16":
        model = model.half().cuda().eval()
    else:
        model = model.cuda().eval()

    model_name = os.path.split(model_path.rstrip('/'))[-1]
    if args.similar_template:
        model_name += "_Similar_template"
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
        "json_path": "../../../dataset/MMAD/mmad.json"
    }
    args.data_path = cfg["data_path"]

    with open(cfg["json_path"], "r") as file:
        chat_ad = json.load(file)


    for data_id, image_path in enumerate(tqdm(chat_ad.keys())):
        if image_path in existing_images and not args.reproduce:
            continue
        text_gt = chat_ad[image_path]
        if args.similar_template:
            few_shot = text_gt["similar_templates"][:args.few_shot_model]
        else:
            few_shot = text_gt["random_templates"][:args.few_shot_model]

        rel_image_path = os.path.join(args.data_path, image_path)
        rel_few_shot = [os.path.join(args.data_path, path) for path in few_shot]
        intervlquery = InterVLQuery(image_path=rel_image_path, text_gt=text_gt,
                           tokenizer=tokenizer, model=model, few_shot=rel_few_shot, visualization=False)
        questions, answers, gpt_answers = intervlquery.generate_answer()
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
