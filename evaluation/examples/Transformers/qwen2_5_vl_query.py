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
import logging

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

sys.path.append("..")
from helper.summary import caculate_accuracy_mmad
from GPT4V.gpt4v import GPT4Query, instruction
# from LLaVA_Query.llava_query import LLaVAQuery


# sys.path.append("../../LLaVA")

sys.path.append("../")


class QwenQuery(GPT4Query):
    def __init__(self, image_path, text_gt, processor, model, few_shot=[], visualization=False, domain_knowledge=None, args=None):
        super(QwenQuery, self).__init__(image_path, text_gt, few_shot, visualization)
        self.processor = processor
        self.model = model
        self.domain_knowledge = domain_knowledge
        self.args = args

    def generate_answer(self):
        questions, answers = self.parse_conversation(self.text_gt, add_question_id=False)
        if questions == [] or answers == []:
            return questions, answers, None

        gpt_answers = []
        for i in range(len(questions)):
            part_questions = questions[i:i + 1]
            if self.args.record_history:
                pass
            else:
                content = self.get_query(part_questions)
                messages = [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
                # Preparation for inference
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")
                # Inference: Generation of the output
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            print(response)
            gpt_answer = self.parse_answer(response, part_questions[0]['options'])
            if len(gpt_answer) == 0:
                gpt_answer.append(response)
                logging.error(f"No matching answer at {self.image_path}: {part_questions}")
            gpt_answers.append(gpt_answer[-1])

        return questions, answers, gpt_answers


    def get_query(self, conversation):
        incontext = []
        if self.domain_knowledge:
            incontext.append({
                "type": "text",
                "text": f"Following is the domain knowledge which contains some type of defect and the normal object characteristics: \n {self.domain_knowledge}"
            })
        if self.few_shot:
            incontext.append({
                "type": "text",
                "text": f"Following is/are {len(self.few_shot)} image of normal sample, which can be used as a template to compare the image being queried."
            })
        for ref_image_path in self.few_shot:
            ref_image = cv2.imread(ref_image_path)
            if self.visualization:
                self.visualize_image(ref_image)
            # ref_base64_image = self.encode_image_to_base64(ref_image)
            incontext.append({
                "type": "image",
                "image": ref_image_path
                })

        image = cv2.imread(self.image_path)
        if self.visualization:
            self.visualize_image(image)
        # base64_image = self.encode_image_to_base64(image)

        if self.visualization:
            print(conversation)

        # 构建查询
        payload = [
            {"type": "text", "text": instruction},
        ] + [
            {"type": "text", "text": f"Answer with the option's letter from the given choices directly! "}
        ] + incontext + [
            {"type": "text", "text": f"Following is the query image: "},
            {"type": "image", "image": self.image_path},
            {"type": "text", "text": f"Following is the question list: "}
        ] + conversation

        return payload


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen2-VL/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--few_shot_model", type=int, default=1)
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--similar_template", action="store_true")
    parser.add_argument("--record_history", action="store_true")
    parser.add_argument("--domain_knowledge", action="store_true")
    parser.add_argument("--domain_knowledge_path", type=str, default="../../../dataset/MMAD/domain_knowledge.json")

    args = parser.parse_args()

    torch.manual_seed(1234)
    model_path = args.model_path

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    min_pixels = 64*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    model_name = os.path.split(model_path.rstrip('/'))[-1]
    if args.similar_template:
        model_name = model_name + "_Similar_template"
    if args.domain_knowledge:
        model_name = model_name + "_Domain_knowledge"
        with open(args.domain_knowledge_path, "r") as file:
            all_domain_knowledge = json.load(file)

    answers_json_path = f"result/answers_{args.few_shot_model}_shot_{model_name}.json"
    if not os.path.exists("result"):
        os.makedirs("result")
    print(f"Answers will be saved at {answers_json_path}")
    # 用于存储所有答案
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
        if args.domain_knowledge:
            dataset_name = image_path.split("/")[0].replace("DS-MVTec", "MVTec")
            object_name = image_path.split("/")[1]
            domain_knowledge = '\n'.join(all_domain_knowledge[dataset_name][object_name].values())
        else:
            domain_knowledge = None
        qwenquery = QwenQuery(image_path=rel_image_path, text_gt=text_gt,
                           processor=processor, model=model, few_shot=rel_few_shot, visualization=False, domain_knowledge=domain_knowledge, args=args)
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
        # 更新答案记录
        for q, a, ga, qt in zip(questions, answers, gpt_answers, questions_type):
            answer_entry = {
                "image": image_path,
                "question": q,
                "question_type": qt,
                "correct_answer": a,
                "gpt_answer": ga
            }

            all_answers_json.append(answer_entry)

        # 保存答案为JSON
        with open(answers_json_path, "w") as file:
            json.dump(all_answers_json, file, indent=4)

    caculate_accuracy_mmad(answers_json_path)
