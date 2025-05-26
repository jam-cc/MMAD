import argparse
import base64
import json


import cv2
import matplotlib.pyplot as plt
import os
import time

from tqdm import tqdm
import sys
import re
from difflib import get_close_matches
import concurrent.futures
import google.generativeai as genai
from google.api_core import retry
import PIL.Image

sys.path.append("..")
from helper.summary import caculate_accuracy_mmad

error_keywords = ['please', 'sorry', 'today', 'cannot assist']

GOOGLE_API_KEY = 'YOUR_API_KEY'
genai.configure(api_key=GOOGLE_API_KEY, transport='rest')

# os.environ["https_proxy"] = "http://127.0.0.1:7890"
# os.environ["http_proxy"] = "http://127.0.0.1:7890"

instruction = '''
You are an industrial inspector who checks products by images. You should judge whether there is a defect in the query image and answer the questions about it.
Answer with the option's letter from the given choices directly.

Finally, you should output a answer list of every question, such as:
1. Answer: B.
2. Answer: B.
3. Answer: A.
...

'''

def get_mime_type(ref_image_path):
    if ref_image_path.lower().endswith(".png"):
        mime_type = "image/png"
    elif ref_image_path.lower().endswith(".jpeg") or ref_image_path.lower().endswith(".jpg"):
        mime_type = "image/jpeg"
    else:
        mime_type = "image/png"
    return mime_type


class GeminiQuery():
    def __init__(self, image_path, text_gt, few_shot=[], visualization=False):
        self.gemini_model = gemini_model
        self.image_path = image_path
        self.text_gt = text_gt
        self.few_shot = few_shot
        self.max_image_size = (512, 512)
        self.api_time_cost = 0
        self.visualization = visualization
        self.max_retries = 5

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
        retry_delay = 1  # Initial delay between retries in seconds
        retries = 0
        while retries < self.max_retries:
            try:
                response = self.gemini_model.generate_content(
                    payload,
                )
                response.resolve()
                pred = response.text
                return pred
            except Exception as e:
                print(e)
                retries += 1
                retry_delay *= 2
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

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
                            "text": f"Question: {questions_text} \n"
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
        # Use regular expressions to extract answer
        answers = pattern.findall(response_text)

        if len(answers) == 0 and options is not None:
            print(f"Failed to extract answer from response: {response_text}")
            # Use fuzzy matching on options dictionary to get answer
            options_values = list(options.values())
            # Use difflib.get_close_matches to find the closest match
            closest_matches = get_close_matches(response_text, options_values, n=1, cutoff=0.0)
            if closest_matches:
                # If there are matches, find the corresponding key
                closest_match = closest_matches[0]
                for key, value in options.items():
                    if value == closest_match:
                        answers.append(key)
                        break
        return answers

    def parse_multi_answer(self, response_text, options=None):
        # pattern = re.compile(r'\bAnswer:\s*([A-Za-z])[^A-Za-z]*')
        pattern = re.compile(r'(?:Answer:\s*[^A-D]*)?([A-D])[^\w]*')
        # Use regular expressions to extract answer
        answers = pattern.findall(response_text)

        if len(answers) == 0 and options is not None:
            print(f"Failed to extract answer from response: {response_text}")
            # Use fuzzy matching on options dictionary to get answer
            options_values = list(options.values())
            # Use difflib.get_close_matches to find the closest match
            closest_matches = get_close_matches(response_text, options_values, n=1, cutoff=0.0)
            if closest_matches:
                # If there are matches, find the corresponding key
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

        # def process_question(i, questions):
        #     # Use single question for query
        #     part_questions = questions[i:i + 1]
        #     payload = self.get_query(part_questions)
        #     respond = self.send_request_to_api(payload)
        #     if respond is None:
        #         return i, ''
        #
        #     gpt_answer = self.parse_answer(respond)
        #     return i, gpt_answer[-1]
        #
        # gpt_answers = [None] * len(questions)
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(process_question, i, questions) for i in range(len(questions))]
        #     for future in concurrent.futures.as_completed(futures):
        #         i, answer = future.result()
        #         gpt_answers[i] = answer

        # First ask if there's a problem before asking other questions
        first_question = [questions[0]]
        payload = self.get_query(first_question)
        respond = self.send_request_to_api(payload)
        if respond is None:
            return questions, answers, None
            # first_answer = ['']
        else:
            print("\nfirst_question", respond)
            gpt_answer = self.parse_answer(respond)
            first_answer = [gpt_answer[-1]]

        payload = self.get_query(questions[1:])
        respond = self.send_request_to_api(payload)
        if respond is None:
            return questions, answers, None
            # gpt_answer = ['' for _ in range(len(questions))]
        else:
            print(respond)
            gpt_answer = self.parse_answer(respond)
        gpt_answers = first_answer + gpt_answer

        return questions, answers, gpt_answers

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
                    "inline_data": {
                        "mime_type": get_mime_type(ref_image_path),
                        "data": ref_base64_image
                    }
                })
        conversation_text = ''
        for q in conversation:
            conversation_text += f"{q['text']}" + '\n'
        image = cv2.imread(self.image_path)
        if self.visualization:
            self.visualize_image(image)
        base64_image = self.encode_image_to_base64(image)
        incontext_image.append({
            "inline_data": {
                "mime_type": get_mime_type(self.image_path),
                "data": base64_image
            }
        })

        if self.visualization:
            print(conversation)

        parts = incontext_image + [{
            "text": instruction + incontext + f"The last image is the query image" +
                    f"Following is the question list: " + '\n' + conversation_text
        }]

        payload = {
            "parts": parts
        }

        return payload

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--few_shot_model", type=int, default=1)
    parser.add_argument("--similar_template", action="store_true")
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--model_name", type=str, default="gemini-1.5-flash", help="The model name of the generative model, [gemini-1.5-pro, gemini-1.5-flash]")

    args = parser.parse_args()

    # model_name = "gemini-1.5-pro"
    # model_name = "gemini-1.5-flash"
    model_name = args.model_name
    gemini_model = genai.GenerativeModel(model_name=model_name)
    if args.similar_template:
        model_name += "_Similar_template"
    answers_json_path = f"result/answers_{args.few_shot_model}_shot_{model_name}.json"
    if not os.path.exists("result"):
        os.makedirs("result")

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
        model = GeminiQuery(image_path=rel_image_path, text_gt=text_gt, few_shot=rel_few_shot)
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
        # print(f"API time cost: {model.api_time_cost:.2f}s")

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

        with open(answers_json_path, "w") as file:
            json.dump(all_answers_json, file, indent=4)

    caculate_accuracy_mmad(answers_json_path)

