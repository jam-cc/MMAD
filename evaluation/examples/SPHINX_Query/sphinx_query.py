import argparse
import base64
import json

import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random
import re
import pandas as pd
import torch
from tqdm import tqdm
import sys
import copy
sys.path.append("..")
from helper.summary import caculate_accuracy_mmad
from GPT4.gpt4v import GPT4Query, instruction

# Set environment variable export HF_HOME=~/.cache/huggingface
os.environ["HF_HOME"] = "~/.cache/huggingface"
from SPHINX import SPHINXModel
from PIL import Image

random.seed(0)
torch.random.manual_seed(0)
np.random.seed(0)


class SPHINXQuery(GPT4Query):
    def __init__(self, image_path, text_gt, model, few_shot=[], visualization=False, domain_knowledge=None, args=None):
        super(SPHINXQuery, self).__init__(image_path, text_gt, few_shot, visualization)
        self.model = model
        self.domain_knowledge = domain_knowledge
        self.args = args

    def generate_answer(self):
        questions, answers = self.parse_conversation(self.text_gt)
        if questions == [] or answers == []:
            return questions, answers, None
        query_image = Image.open(self.image_path).convert('RGB')

        gpt_answers = []
        history = ''
        for i in range(len(questions)):
            part_questions = questions[i:i + 1]
            payload, conversation_text = self.get_query(part_questions)
            query = payload + history + conversation_text
            qas = [[query, None]]
            response = model.generate_response(qas, query_image, max_gen_len=1024, temperature=0.9, top_p=0.5, seed=0)
            if args.record_history:
                history = history + conversation_text + '\n' + response + '\n'
            print(response)
            gpt_answer = self.parse_answer(response)
            gpt_answers.append(gpt_answer[-1])
        print(gpt_answers)
        return questions, answers, gpt_answers

    def get_query(self, conversation):
        incontext = ''
        if self.domain_knowledge:
            incontext = incontext + f"Following is the domain knowledge which contains some type of defect and the normal object characteristics: \n {self.domain_knowledge}"

        if self.visualization:
            print(conversation)
        payload = (instruction + incontext + '\n')

        payload = payload + '\n' + f"Following is the question list. Answer with the option's letter from the given choices directly: " + '\n'

        conversation_text = ''
        for q in conversation:
            conversation_text += f"{q['text']}" + '\n'

        return payload, conversation_text


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="../../LLaMA2-Accessory/ckpt/SPHINX-v2-1k")

    parser.add_argument("--single-pred-prompt", action="store_true")

    parser.add_argument("--dtype", type=str, default="fp32")

    parser.add_argument("--few_shot_model", type=int, default=0)
    parser.add_argument("--similar_template", action="store_true")
    parser.add_argument("--reproduce", action="store_true")

    # parser.add_argument("--defect-shot", type=int, default=1)
    parser.add_argument("--record_history", action="store_true")
    parser.add_argument("--CoT", action="store_true")
    parser.add_argument("--domain_knowledge", action="store_true")
    parser.add_argument("--text_only", action="store_true")


    args = parser.parse_args()

    # Model
    model_path = os.path.expanduser(args.model_path)
    model = SPHINXModel.from_pretrained(pretrained_path=model_path, with_visual=True, quant=True)

    model_name = os.path.split(model_path.rstrip('/'))[-1]
    if args.CoT:
        model_name = model_name + "_CoT"
    if args.domain_knowledge:
        model_name = model_name + "_Domain_knowledge"
    if args.text_only:
        model_name = model_name + "_Text_only"
    if args.similar_template:
        model_name = model_name + "_Similar_template"

    answers_json_path = f"result/answers_{args.few_shot_model}_shot_{model_name}.json"
    if not os.path.exists("result"):
        os.makedirs("result")
    print(f"Answers will be saved at {answers_json_path}")
    # Used to store all answers
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
        sphinxquery = SPHINXQuery(image_path=rel_image_path, text_gt=text_gt,
                            model=model, few_shot=rel_few_shot, visualization=False, args=args)
        questions, answers, gpt_answers = sphinxquery.generate_answer()
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
        # Update answer records
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