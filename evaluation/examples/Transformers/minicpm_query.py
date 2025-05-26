import argparse
import base64
import json

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
from SoftPatch.call import call_patchcore, build_patchcore


class MiniCPMQuery(GPT4Query):
    def __init__(self, image_path, text_gt, tokenizer, model, few_shot=[], visualization=False, domain_knowledge=None,
                 agent=None):
        super(MiniCPMQuery, self).__init__(image_path, text_gt, few_shot, visualization)
        self.tokenizer = tokenizer
        self.model = model
        self.domain_knowledge = domain_knowledge
        self.agent = agent

    def generate_answer(self):
        questions, answers = self.parse_conversation(self.text_gt)
        if questions == [] or answers == []:
            return questions, answers, None
        template_image = []
        for ref_image_path in self.few_shot:
            template_image.append(Image.open(ref_image_path).convert('RGB'))
        if self.agent:
            anomaly_map = call_patchcore(self.image_path, self.few_shot, self.agent)
            template_image.append(Image.open(anomaly_map).convert('RGB'))
        query_image = Image.open(self.image_path).convert('RGB')
        images = template_image + [query_image]

        gpt_answers = []
        history = ''
        for i in range(len(questions)):
            part_questions = questions[i:i + 1]
            payload, conversation_text = self.get_query(part_questions)
            query = payload + history + conversation_text
            msgs = [{'role': 'user', 'content': images+[query]}]
            response = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer
            )
            print(response)
            gpt_answer = self.parse_answer(response)
            gpt_answers.append(gpt_answer[-1])
            if args.record_history:
                history = history + conversation_text + '\n' + response + '\n'
        print(gpt_answers)
        return questions, answers, gpt_answers

    def get_query(self, conversation):
        incontext = ''
        if self.domain_knowledge:
            incontext = incontext + f"Following is the domain knowledge which contains some type of defect and the normal object characteristics: \n {self.domain_knowledge}"

        if self.few_shot:
            if len(self.few_shot) == 1:
                incontext = incontext + f"Image 1 is a normal sample, which can be used as a template to compare the image being queried."
            else:
                incontext = incontext + f"Image 1-{len(self.few_shot)} are {len(self.few_shot)} normal samples, which can be used as a template to compare the image being queried."

        if self.agent:
            incontext = incontext + f"Image {len(self.few_shot) + 1} is the anomaly map for the query image. The highlighted areas in yellow suggest potential defects. Please note that even if the query image does not have any defects, the anomaly map still show highlighted areas due to normalization effects. So you still need to make a judgement by comparing the images."
        if self.visualization:
            print(conversation)
        payload = instruction + incontext + '\n' + \
                  f"The last image is the query image: " + f"Following is the question list. Answer with the option's letter from the given choices directly: " + '\n'

        conversation_text = ''
        for q in conversation:
            conversation_text += f"{q['text']}" + '\n'

        return payload, conversation_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="openbmb/MiniCPM-V-2_6")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--few_shot_model", type=int, default=1)
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--similar_template", action="store_true")
    parser.add_argument("--record_history", action="store_true")
    parser.add_argument("--domain_knowledge", action="store_true")
    parser.add_argument("--domain_knowledge_path", type=str, default="../../../dataset/MMAD/domain_knowledge.json")
    parser.add_argument("--agent", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(1234)
    model_path = args.model_path
    model_name = os.path.split(model_path.rstrip('/'))[-1]

    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if args.similar_template:
        model_name += "_Similar_template"
    if args.domain_knowledge:
        model_name += "_Domain_knowledge"
        with open(args.domain_knowledge_path, "r") as file:
            all_domain_knowledge = json.load(file)
    if args.agent:
        model_name += "_Agent"
        agent = build_patchcore()
    else:
        agent = None

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

        if args.domain_knowledge:
            dataset_name = image_path.split("/")[0].replace("DS-MVTec", "MVTec")
            object_name = image_path.split("/")[1]

            domain_knowledge = '\n'.join(all_domain_knowledge[dataset_name][object_name].values())
        else:
            domain_knowledge = None

        minicpmquery = MiniCPMQuery(image_path=rel_image_path, text_gt=text_gt,
                                    tokenizer=tokenizer, model=model, few_shot=rel_few_shot, visualization=False,
                                    domain_knowledge=domain_knowledge, agent=agent)
        questions, answers, gpt_answers = minicpmquery.generate_answer()
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
