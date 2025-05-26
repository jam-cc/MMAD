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
import seaborn as sns
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

sys.path.append("..")
# from data.mvtec import ADDataset
from helper.summary import caculate_accuracy_mmad
from GPT4.gpt4v import GPT4Query, instruction

sys.path.append("../")

class QwenQuery(GPT4Query):
    def __init__(self, image_path, text_gt, tokenizer, model, few_shot=[], visualization=False, args=None):
        super(QwenQuery, self).__init__(image_path, text_gt, few_shot, visualization)
        self.tokenizer = tokenizer
        self.model = model
        self.args = args

    def generate_answer(self):
        questions, answers = self.parse_conversation(self.text_gt)
        if questions == [] or answers == []:
            return questions, answers, None

        gpt_answers = []
        for i in range(len(questions)):
            part_questions = questions[i:i + 1]
            if self.args.record_history:
                if i == 0:
                    query = self.tokenizer.from_list_format(self.get_query(part_questions))
                    # input_ids, image_tensor, image_sizes = self.get_query(part_questions)
                    response, history = self.model.chat(tokenizer, query=query, history=None)
                else:
                    query = self.tokenizer.from_list_format(part_questions)
                    response, history = self.model.chat(tokenizer, query=query, history=history)
            else:
                query = self.tokenizer.from_list_format(self.get_query(part_questions))
                response, _ = self.model.chat(tokenizer, query=query, history=None)
            print(response)
            gpt_answer = self.parse_answer(response, part_questions[0]['options'])
            if len(gpt_answer) == 0:
                gpt_answer.append(response)
                logging.error(f"No matching answer at {self.image_path}: {part_questions}")
            gpt_answers.append(gpt_answer[-1])

        return questions, answers, gpt_answers


    def get_query(self, conversation):
        incontext = []
        if self.few_shot:
            incontext.append({
                "text": f"Following is/are {len(self.few_shot)} image of normal sample, which can be used as a template to compare the image being queried."
            })
        for ref_image_path in self.few_shot:
            # if not is_anomaly:
                ref_image = cv2.imread(ref_image_path)
                if self.visualization:
                    self.visualize_image(ref_image)
                ref_base64_image = self.encode_image_to_base64(ref_image)
                incontext.append({
                    "image": ref_image_path
                    })

        image = cv2.imread(self.image_path)
        if self.visualization:
            self.visualize_image(image)
        base64_image = self.encode_image_to_base64(image)

        if self.visualization:
            print(conversation)

        # Build query
        payload = [
            {"text": instruction},
        ] + [
            {"text": f"Answer with the option's letter from the given choices directly."}
        ] + incontext + [
            {"text": f"Following is the query image: "},
            {"image": self.image_path},
            {"text": f"Following is the question list: "}
        ] + conversation

        return payload

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen-VL-Chat")

    parser.add_argument("--few_shot_model", type=int, default=1)
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--similar_template", action="store_true")
    parser.add_argument("--record_history", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(1234)
    model_path = args.model_path
    # Note: The default behavior now has injection attack prevention off.
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir="~/.cache/huggingface/hub")

    # use cuda device
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, cache_dir="~/.cache/huggingface/hub").eval()
    # Specify hyperparameters for generation
    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True, cache_dir="~/.cache/huggingface/hub")

    model_name = os.path.split(model_path.rstrip('/'))[-1]
    if args.similar_template:
        model_name = model_name + "_Similar_template"

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

    for image_path in tqdm(chat_ad.keys()):
        if image_path in existing_images and not args.reproduce:
            continue
        text_gt = chat_ad[image_path]
        if args.similar_template:
            few_shot = text_gt["similar_templates"][:args.few_shot_model]
        else:
            few_shot = text_gt["random_templates"][:args.few_shot_model]

        rel_image_path = os.path.join(args.data_path, image_path)
        rel_few_shot = [os.path.join(args.data_path, path) for path in few_shot]
        qwenquery = QwenQuery(image_path=rel_image_path, text_gt=text_gt,
                           tokenizer=tokenizer, model=model, few_shot=rel_few_shot, visualization=False, args=args)
        questions, answers, gpt_answers = qwenquery.generate_answer()
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

        # Save answers as JSON
        with open(answers_json_path, "w") as file:
            json.dump(all_answers_json, file, indent=4)

    caculate_accuracy_mmad(answers_json_path)
