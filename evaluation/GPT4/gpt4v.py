import argparse
import base64
import json

import requests
import cv2
import matplotlib.pyplot as plt
import os
import time
import random
import re
import pandas as pd
from requests import RequestException
from tqdm import tqdm
import sys
import re
from difflib import get_close_matches

sys.path.append("..")
from data.mvtec import ADDataset
from helper.summary import caculate_accuracy

error_keywords = ['please', 'sorry', 'today', 'cannot assist']
api = {
    "api_key": "YOUR_API_KEY",
    "url": "https://api.openai.com/v1/chat/completions"
}

instruction = '''
You are an industrial inspector who checks products by images. You should judge whether there is a defect in the query image and answer the questions about it.
Answer with the option's letter from the given choices directly.
'''

# instruction = '''
# You are an industrial inspector who checks products by images. You need to find anomalies by comparing the sample to known information.
# You will first receive information of the known sample, then an image of the query sample and a series of questions for the query image.
# You need to respond to questions based on the information you have.
#
# Each question is followed by a series of choices, you need to choose the correct answer from the choices.
# The answer should begin with 'Answer: ', followed with the your choice letter and end with '.', such as "Answer: A."
# DONOT ANSWER ANYTHING ELSE, OTHERWISE THE SYSTEM WILL NOT RECOGNIZE YOUR ANSWER!
#
# Finally, you should output a list of answer, such as:
# 1. Answer: B.
# 2. Answer: B.
# 3. Answer: A.
# '''


class GPT4Query():
    def __init__(self, image_path, text_gt, few_shot=[], visualization=True):
        self.api_key = api["api_key"]
        self.url = api["url"]
        self.image_path = image_path
        self.text_gt = text_gt
        self.few_shot = few_shot
        self.max_image_size = (512, 512)
        self.api_time_cost = 0
        self.visualization = visualization
        self.max_retries = 3

    def encode_image_to_base64(self, image):
        # 获取图像的尺寸
        height, width = image.shape[:2]
        # 计算缩放比例
        scale = min(self.max_image_size[0] / width, self.max_image_size[1] / height)

        # 使用新的尺寸缩放图像
        new_width, new_height = int(width * scale), int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        _, encoded_image = cv2.imencode('.jpg', resized_image)
        return base64.b64encode(encoded_image).decode('utf-8')

    def visualize_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis('off')  # 关闭坐标轴
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 减小边距
        plt.show()


    def send_request_to_api(self, payload):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        max_retries = self.max_retries
        retry_delay = 1  # Initial delay between retries in seconds
        retries = 0

        while retries < max_retries:
            try:
                before = time.time()
                response = requests.post(self.url, headers=headers, json=payload)

                # 使用get方法从响应中安全获取'choices'字段
                choices = response.json().get('choices', [])
                if choices:
                    if any(word in choices[0]['message']['content'].lower() for word in error_keywords):
                        print("Error respond of ", self.image_path, ": "
                              , choices[0]['message']['content'])
                        retries += 1
                        continue

                    self.api_time_cost += time.time() - before
                    return response.json()
                else:
                    # 如果choices字段不存在或为空，根据需要进行操作
                    print(response.json())
                    retries += 1

            except RequestException as e:
                print(f"Request failed: {e}, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                retries += 1

        # Handle the case where all retries fail
        print("Failed to send request.")
        return None

    def parse_conversation(self, text_gt):
        Question = []
        Answer = []
        # 想要匹配的关键字
        keyword = "conversation"

        # 遍历字典中的所有键
        for key in text_gt.keys():
            # 如果键以关键字开头
            if key.startswith(keyword):
                # 获取对应的值
                conversation = text_gt[key]
                for i, QA in enumerate(conversation):

                    # 打乱选项的顺序
                    options_items = list(QA['Options'].items())
                    # random.shuffle(options_items)  # 随机排序选项

                    # 重建选项文本并创建一个新的选项到答案的映射
                    options_text = ""
                    new_answer_key = None
                    for new_key, (original_key, value) in enumerate(options_items):
                        options_text += f"{chr(65 + new_key)}. {value}\n"  # 65是字母A的ASCII码
                        if QA['Answer'] == original_key:
                            new_answer_key = chr(65 + new_key)  # 更新答案的键
                    option_dict = {chr(65 + new_key): value for new_key, (original_key, value) in enumerate(options_items)}

                    questions_text = QA['Question']
                    # if i == 0:
                    #     questions_text = "Should the query picture be classified as normal or abnormal?"
                    #     options_text = options_text.replace('Yes', 'Abnormal')
                    #     options_text = options_text.replace('No', 'Normal')
                    Question.append(
                        {
                            "type": "text",
                            "text": f"Question {i + 1}: {questions_text} \n"
                                    f"{options_text}",
                            "options": option_dict,
                    },
                    )
                    # 确保我们找到了新的答案键
                    if new_answer_key is not None:
                        Answer.append(new_answer_key)
                    else:
                        raise ValueError("Answer key not found after shuffling options.")
                break
        return Question, Answer


    def parse_answer(self, response_text, options=None):
        # pattern = re.compile(r'\bAnswer:\s*([A-Za-z])[^A-Za-z]*')
        # pattern = re.compile(r'(?:Answer:\s*[^A-D]*)?([A-D])[^\w]*')
        pattern = re.compile(r'\b([A-E])\b')
        # 使用正则表达式提取答案
        answers = pattern.findall(response_text)

        if len(answers) == 0 and options is not None:
            print(f"Failed to extract answer from response: {response_text}")
            # 模糊匹配options字典来得到答案
            options_values = list(options.values())
            # 使用difflib.get_close_matches来找到最接近的匹配项
            closest_matches = get_close_matches(response_text, options_values, n=1, cutoff=0.0)
            if closest_matches:
                # 如果有匹配项，找到对应的键
                closest_match = closest_matches[0]
                for key, value in options.items():
                    if value == closest_match:
                        answers.append(key)
                        break
        return answers

    def parse_multi_answer(self, response_text, options=None):
        # pattern = re.compile(r'\bAnswer:\s*([A-Za-z])[^A-Za-z]*')
        pattern = re.compile(r'(?:Answer:\s*[^A-D]*)?([A-D])[^\w]*')
        # 使用正则表达式提取答案
        answers = pattern.findall(response_text)

        if len(answers) == 0 and options is not None:
            print(f"Failed to extract answer from response: {response_text}")
            # 模糊匹配options字典来得到答案
            options_values = list(options.values())
            # 使用difflib.get_close_matches来找到最接近的匹配项
            closest_matches = get_close_matches(response_text, options_values, n=1, cutoff=0.0)
            if closest_matches:
                # 如果有匹配项，找到对应的键
                closest_match = closest_matches[0]
                for key, value in options.items():
                    if value == closest_match:
                        answers.append(key)
                        break
        return answers

    def generate_answer(self):
        questions, answers = self.parse_conversation(self.text_gt)
        if questions == [] or answers == []:
            return questions, answers, None

        gpt_answers = []
        for i in range(len(questions)):
            part_questions = questions[:i + 1]
            payload = self.get_query(part_questions)
            respond = self.send_request_to_api(payload)
            if respond is None:
                gpt_answers.append('')
                continue
            gpt_answer = self.parse_json(respond)
            gpt_answer = self.parse_answer(gpt_answer)
            gpt_answers.append(gpt_answer[-1])

        # # 先单独问有无的问题再问其他问题
        # first_question = [questions[0]]
        # payload = self.get_query(first_question)
        # respond = self.send_request_to_api(payload)
        # if respond is None:
        #     return questions, answers, None
        #     # first_answer = ['']
        # else:
        #     gpt_answer = self.parse_json(respond)
        #     gpt_answer = self.parse_answer(gpt_answer)
        #     first_answer = [gpt_answer[-1]]
        #
        # payload = self.get_query(questions)
        # respond = self.send_request_to_api(payload)
        # if respond is None:
        #     return questions, answers, None
        #     # gpt_answer = ['' for _ in range(len(questions))]
        # else:
        #     gpt_answer = self.parse_json(respond)
        #     gpt_answer = self.parse_answer(gpt_answer)
        # gpt_answers = first_answer + gpt_answer[1:]

        return questions, answers, gpt_answers

    def parse_json(self, response_json):
        # 从响应中获取'choices'字段
        choices = response_json.get('choices', [])

        # 如果'choices'字段存在且不为空
        if choices:
            # 获取'choices'字段的第一个元素
            first_choice = choices[0]

            # 从第一个元素中获取'message'字段
            message = first_choice.get('message', {})

            # 从'message'字段中获取'content'字段，即caption
            # caption = message.get('content', '')
            caption = message['content']
            if self.visualization:
                print(f"Caption: {caption}")
            return caption

        # 如果'choices'字段不存在或为空，返回空字符串
        return ''

    def get_query(self, conversation):
        incontext = []
        if self.few_shot:
            incontext.append({
                "type": "text",
                "text": f"Following is {len(self.few_shot)} image of normal sample, which can be used as a template to compare."
            })
        for ref_image_path in self.few_shot:
            # if not is_anomaly:
                ref_image = cv2.imread(ref_image_path)
                if self.visualization:
                    self.visualize_image(ref_image)
                ref_base64_image = self.encode_image_to_base64(ref_image)
                incontext.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{ref_base64_image}",
                        "detail": "low"
                    }
                    })
                # incontext.append({
                #         "type": "text",
                #         "text": f"{caption}"
                #     })

        image = cv2.imread(self.image_path)
        if self.visualization:
            self.visualize_image(image)
        base64_image = self.encode_image_to_base64(image)

        if self.visualization:
            print(conversation)

        # 构建查询
        payload = {
            "model": "gpt-4o",
            "messages": [{
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": instruction
                    },
                ]
            },
                {
                    "role": "user",
                    "content": incontext +
                    [
                        {
                            "type": "text",
                            "text": f"Following is the query image: "
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            }
                        },
                        {
                            "type": "text",
                            "text": f"Following is the question list: "
                        },
                    ] + conversation

                }
            ],
            "max_tokens": 600,
        }
        return payload

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--few_shot_model", type=int, default=1)

    args = parser.parse_args()
    cfg = {
        "path": "../../../dataset/Defect_Spectrum/DS-MVTec",
        "splits": ["image"],
        "few_shot_model": args.few_shot_model,
    }
    noraml_cfg = {
        "path": "../../../dataset/MVTec",
        "splits": ["train"],
    }
    few_shot_model = cfg["few_shot_model"]
    dataset_path = cfg["path"]
    #classname 定义为dataset_path下的文件夹
    classname = os.listdir(dataset_path)
    splits = cfg["splits"]
    # 检查是否都是文件夹
    for c in classname:
        if not os.path.isdir(os.path.join(dataset_path, c)):
            classname.remove(c)
    # indices = [0, 2]  # 想从列表中选择的元素的索引
    #
    # # 使用列表推导选择元素
    # classname = [classname[i] for i in indices]


    dataset = ADDataset(source=dataset_path, classnames=classname, splits=splits,training=False, transform_fn=None, normalize_fn=None)

    normal_dataset = ADDataset(source=noraml_cfg["path"], classnames=classname, splits=noraml_cfg["splits"],training=True, transform_fn=None, normalize_fn=None)
    answers_json_path = f"answers_{few_shot_model}_shot.json"

    # 初始化准确率统计字典
    question_stats = [{} for _ in range(5)]
    for i in range(5):
        for cls in classname:
            question_stats[i][cls] = {'correct': 0, 'total': 0}

    # 用于存储所有答案
    if os.path.exists(answers_json_path):
        with open(answers_json_path, "r") as file:
            all_answers_json = json.load(file)
    else:
        all_answers_json = []

    for data_id, data in tqdm(enumerate(dataset), total=len(dataset), desc="Processing dataset"):
        # 计算当前image_path对应的答案数量
        answers_count = sum([a["image"] == data["image_path"] for a in all_answers_json])

        # 如果答案数量达到5个，则跳过
        if answers_count >= 5:
            continue
        else:
            # 如果答案数量少于5个，则删除所有已有的答案
            all_answers_json = [a for a in all_answers_json if a["image"] != data["image_path"]]
        if data["text_gt"] is None:
            continue
        # 从ADDataset中随机抽取数据作为few—shot, 要求类别一致
        imgpaths_per_class = normal_dataset.imgpaths_per_class
        normal_set = imgpaths_per_class[data["clsname"]]["good"]
        # 从normal_set中随机抽取few_shot
        few_shot = random.sample(normal_set, few_shot_model)
        model = GPT4Query(image_path=data["image_path"], text_gt=data["text_gt"], few_shot=few_shot)
        questions, answers, gpt_answers = model.generate_answer()
        if gpt_answers is None or len(gpt_answers) != len(answers):
            continue
        correct = 0
        for i, answer in enumerate(answers):
            if gpt_answers[i] == answer:
                correct += 1
        accuracy = correct / len(answers)
        print(f"Accuracy: {accuracy:.2f}")
        print(f"API time cost: {model.api_time_cost:.2f}s")

        # 更新答案记录
        for q, a, ga in zip(questions, answers, gpt_answers):
            answer_entry = {
                "class": data['clsname'],
                "image": data["image_path"],
                "question": q,
                "correct_answer": a,
                "gpt_answer": ga
            }

            all_answers_json.append(answer_entry)

            # question_index = questions.index(q)
            # question_stats[question_index][data["clsname"]]['total'] += 1
            # if ga == a:
            #     question_stats[question_index][data["clsname"]]['correct'] += 1
        if data_id % 10 == 0 or data_id == len(dataset) - 1:
            # 保存答案为JSON
            with open(answers_json_path, "w") as file:
                json.dump(all_answers_json, file, indent=4)

    caculate_accuracy(answers_json_path, "good")

