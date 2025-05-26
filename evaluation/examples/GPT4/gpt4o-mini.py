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
import concurrent.futures

sys.path.append("..")
# from data.mvtec import ADDataset
from helper.summary import caculate_accuracy_mmad

error_keywords = ['please', 'sorry', 'today', 'cannot assist']
api = {
    "api_key": "YOUR_API_KEY",
    "url": "https://api.openai.com/v1/chat/completions"
}
# Cost estimation: 4o-mini $40 4o $1400 4v $2800
instruction = '''
You are an industrial inspector who checks products by images. You should judge whether there is a defect in the query image and answer the questions about it.
Answer with the option's letter from the given choices directly.

Finally, you should output a list of answer, such as:
1. Answer: B.
2. Answer: B.
3. Answer: A.
...
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
    def __init__(self, image_path, text_gt, few_shot=[], visualization=False):
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
                    options_items = list(QA['Options'].items())
                    options_text = ""
                    for j, (key, value) in enumerate(options_items):
                        options_text += f"{key}. {value}\n"
                    questions_text = QA['Question']
                    Question.append(
                        {
                            "type": "text",
                            "text": f"Question {i + 1}: {questions_text} \n"
                                    f"{options_text}"
                        },
                    )
                    Answer.append(QA['Answer'])

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

        def process_question(i, questions):
            # Use single question for query
            part_questions = questions[i:i + 1]
            payload = self.get_query(part_questions)
            respond = self.send_request_to_api(payload)
            if respond is None:
                return i, ''
            gpt_answer = self.parse_json(respond)
            gpt_answer = self.parse_answer(gpt_answer)
            return i, gpt_answer[-1]

        gpt_answers = [None] * len(questions)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_question, i, questions) for i in range(len(questions))]
            for future in concurrent.futures.as_completed(futures):
                i, answer = future.result()
                gpt_answers[i] = answer

        # Non-concurrent requests
        # gpt_answers = []
        # for i in range(len(questions)):
        #     part_questions = questions[i:i + 1]
        #     payload = self.get_query(part_questions)
        #     respond = self.send_request_to_api(payload)
        #     if respond is None:
        #         gpt_answers.append('')
        #         continue
        #     gpt_answer = self.parse_json(respond)
        #     gpt_answer = self.parse_answer(gpt_answer)
        #     gpt_answers.append(gpt_answer[-1])

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
        incontext = ""
        if self.few_shot:
            incontext = f"The first {len(self.few_shot)} image is the normal sample, which can be used as a template to compare."
            # incontext.append({
            #     "type": "text",
            #     "text": f"Following is {len(self.few_shot)} image of normal sample, which can be used as a template to compare."
            # })
        incontext_image = []
        for ref_image_path in self.few_shot:
            # if not is_anomaly:
                ref_image = cv2.imread(ref_image_path)
                if self.visualization:
                    self.visualize_image(ref_image)
                ref_base64_image = self.encode_image_to_base64(ref_image)
                incontext_image.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{ref_base64_image}",
                        "detail": "low"
                    }
                    })
        conversation_text = ''
        for q in conversation:
            conversation_text += f"{q['text']}" + '\n'
        image = cv2.imread(self.image_path)
        if self.visualization:
            self.visualize_image(image)
        base64_image = self.encode_image_to_base64(image)

        if self.visualization:
            print(conversation)

        # Build query
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content":incontext_image +
                    [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            }
                        },
                        {
                            "type": "text",
                            "text": instruction + incontext + f"The last image is the query image" +
                                    f"Following is the question list: " + '\n' + conversation_text
                        },
                    ]
                }
            ],
            "max_tokens": 600,
        }
        return payload

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--few_shot_model", type=int, default=1)
    parser.add_argument("--similar_template", action="store_true")
    parser.add_argument("--reproduce", action="store_true")

    args = parser.parse_args()
    model_name = "gpt-4o-mini"
    if args.similar_template:
        model_name += "_Similar_template"
    answers_json_path = f"result/answers_{args.few_shot_model}_shot_{model_name}.json"

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
        model = GPT4Query(image_path=rel_image_path, text_gt=text_gt, few_shot=rel_few_shot)
        questions, answers, gpt_answers = model.generate_answer()
        if gpt_answers is None or len(gpt_answers) != len(answers):
            print(f"Error at {image_path}")
            continue
        correct = 0
        for i, answer in enumerate(answers):
            if gpt_answers[i] == answer:
                correct += 1
        accuracy = correct / len(answers)
        print(f"Accuracy: {accuracy:.2f}")
        print(f"API time cost: {model.api_time_cost:.2f}s")

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

        with open(answers_json_path, "w") as file:
            json.dump(all_answers_json, file, indent=4)

    caculate_accuracy_mmad(answers_json_path)
