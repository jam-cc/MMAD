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
# from data.mvtec import ADDataset
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
        # Get image dimensions
        height, width = image.shape[:2]
        # Calculate scaling ratio
        scale = min(self.max_image_size[0] / width, self.max_image_size[1] / height)

        # Scale image using new dimensions
        new_width, new_height = int(width * scale), int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        _, encoded_image = cv2.imencode('.jpg', resized_image)
        return base64.b64encode(encoded_image).decode('utf-8')

    def visualize_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis('off')  # Turn off coordinate axes
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Reduce margins
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

                # Safely get 'choices' field from response using get method
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
                    # If choices field doesn't exist or is empty, perform operations as needed
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
        # Keywords to match
        keyword = "conversation"

        # Iterate through all keys in the dictionary
        for key in text_gt.keys():
            # If the key starts with the keyword
            if key.startswith(keyword):
                # Get the corresponding value
                conversation = text_gt[key]
                for i, QA in enumerate(conversation):

                    # Shuffle the order of options
                    options_items = list(QA['Options'].items())
                    # random.shuffle(options_items)  # Randomly sort options

                    # Rebuild option text and create a new mapping from options to answers
                    options_text = ""
                    new_answer_key = None
                    for new_key, (original_key, value) in enumerate(options_items):
                        options_text += f"{chr(65 + new_key)}. {value}\n"  # 65 is the ASCII code for letter A
                        if QA['Answer'] == original_key:
                            new_answer_key = chr(65 + new_key)  # Update answer key
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
                    # Ensure we found the new answer key
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
        # Extract answers using regular expressions
        answers = pattern.findall(response_text)

        if len(answers) == 0 and options is not None:
            print(f"Failed to extract answer from response: {response_text}")
            # Use fuzzy matching on options dictionary to get the answer
            options_values = list(options.values())
            # Use difflib.get_close_matches to find the closest match
            closest_matches = get_close_matches(response_text, options_values, n=1, cutoff=0.0)
            if closest_matches:
                # If there's a match, find the corresponding key
                closest_match = closest_matches[0]
                for key, value in options.items():
                    if value == closest_match:
                        answers.append(key)
                        break
        return answers

    def parse_multi_answer(self, response_text, options=None):
        # pattern = re.compile(r'\bAnswer:\s*([A-Za-z])[^A-Za-z]*')
        pattern = re.compile(r'(?:Answer:\s*[^A-D]*)?([A-D])[^\w]*')
        # Extract answers using regular expressions
        answers = pattern.findall(response_text)

        if len(answers) == 0 and options is not None:
            print(f"Failed to extract answer from response: {response_text}")
            # Use fuzzy matching on options dictionary to get the answer
            options_values = list(options.values())
            # Use difflib.get_close_matches to find the closest match
            closest_matches = get_close_matches(response_text, options_values, n=1, cutoff=0.0)
            if closest_matches:
                # If there's a match, find the corresponding key
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

        # # First ask the yes/no question separately, then ask other questions
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
        # Get 'choices' field from response
        choices = response_json.get('choices', [])

        # If 'choices' field exists and is not empty
        if choices:
            # Get the first element of 'choices' field
            first_choice = choices[0]

            # Get 'message' field from the first element
            message = first_choice.get('message', {})

            # Get 'content' field from 'message' field, which is the caption
            # caption = message.get('content', '')
            caption = message['content']
            if self.visualization:
                print(f"Caption: {caption}")
            return caption

        # If 'choices' field doesn't exist or is empty, return empty string
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

        # Build query
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
    # classname is defined as folders under dataset_path
    classname = os.listdir(dataset_path)
    splits = cfg["splits"]
    # Check if they are all folders
    for c in classname:
        if not os.path.isdir(os.path.join(dataset_path, c)):
            classname.remove(c)
    # indices = [0, 2]  # Indices of elements to select from the list
    #
    # # Use list comprehension to select elements
    # classname = [classname[i] for i in indices]


    dataset = ADDataset(source=dataset_path, classnames=classname, splits=splits,training=False, transform_fn=None, normalize_fn=None)

    normal_dataset = ADDataset(source=noraml_cfg["path"], classnames=classname, splits=noraml_cfg["splits"],training=True, transform_fn=None, normalize_fn=None)
    answers_json_path = f"answers_{few_shot_model}_shot.json"

    # Initialize accuracy statistics dictionary
    question_stats = [{} for _ in range(5)]
    for i in range(5):
        for cls in classname:
            question_stats[i][cls] = {'correct': 0, 'total': 0}

    # Used to store all answers
    if os.path.exists(answers_json_path):
        with open(answers_json_path, "r") as file:
            all_answers_json = json.load(file)
    else:
        all_answers_json = []

    for data_id, data in tqdm(enumerate(dataset), total=len(dataset), desc="Processing dataset"):
        # Calculate the number of answers corresponding to current image_path
        answers_count = sum([a["image"] == data["image_path"] for a in all_answers_json])

        # If the number of answers reaches 5, skip
        if answers_count >= 5:
            continue
        else:
            # If the number of answers is less than 5, delete all existing answers
            all_answers_json = [a for a in all_answers_json if a["image"] != data["image_path"]]
        if data["text_gt"] is None:
            continue
        # Randomly sample data from ADDataset as few-shot, requiring consistent categories
        imgpaths_per_class = normal_dataset.imgpaths_per_class
        normal_set = imgpaths_per_class[data["clsname"]]["good"]
        # Randomly sample few_shot from normal_set
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

        # Update answer records
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
            # Save answers as JSON
            with open(answers_json_path, "w") as file:
                json.dump(all_answers_json, file, indent=4)

    caculate_accuracy(answers_json_path, "good")

