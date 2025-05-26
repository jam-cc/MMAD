import argparse

import json

import cv2

import os
# Set environment variable export HF_HOME=~/.cache/huggingface
os.environ["HF_HOME"] = "~/.cache/huggingface"
import random

import torch
from tqdm import tqdm
import sys
sys.path.append("..")
# from data.mvtec import ADDataset
from helper.summary import caculate_accuracy_mmad
from GPT4.gpt4v import GPT4Query, instruction
from SoftPatch.call import call_patchcore, build_patchcore

from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.utils import disable_torch_init
from cambrian.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

class CambrianQuery(GPT4Query):
    def __init__(self, image_path, text_gt, tokenizer, model, image_processor, context_len, few_shot=[], defect_shot=[], visualization=False, domain_knowledge=None, args=None):
        super(CambrianQuery, self).__init__(image_path, text_gt, few_shot, visualization)
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len
        self.defect_shot = defect_shot
        self.args = args
        self.domain_knowledge = domain_knowledge


    def generate_answer(self):
        questions, answers = self.parse_conversation(self.text_gt)
        if questions == [] or answers == []:
            return questions, answers, None
        gpt_answers = []
        history = []

        ########
        if self.args.CoT:
            chain_of_thought = []
            chain_of_thought.append(
                {
                    # "type": "text",
                    "text": f"Before answering the following questions, we need to understand the test image. \n"
                            f"First, simply describe the differences between the test image with the normal and defect sample. \n"
                            f"Second, judge whether those differences are defect or normal. \n"
                            # f"First, summary the normal and defect samples and answer why defect sample abnormal. \n"
                            # f"Second, simply describe the differences between the test image with the normal and defect sample. \n"
                            # f"Third, judge whether those differences are defect or normal. \n"
                },
            )
            input_ids, image_tensor, image_sizes = self.get_query(chain_of_thought, history)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=512,
                    use_cache=True)
                # outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                outputs = self.tokenizer.batch_decode(output_ids)[0].strip()

            history.append((chain_of_thought[0]['text'], outputs))
            print(outputs)

        for i in range(len(questions)):
            part_questions = questions[i:i + 1]
            input_ids, image_tensor, image_sizes = self.get_query(part_questions, history)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=128,
                    use_cache=True)
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if self.args.record_history:
                history.append((part_questions[0]['text'], outputs))
            print(outputs)
            gpt_answer = self.parse_answer(outputs, part_questions[0]['options'])
            gpt_answers.append(gpt_answer[0])

        return questions, answers, gpt_answers

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
                    random.shuffle(options_items)  # Randomly sort options

                    # Rebuild option text and create a new mapping from option to answer
                    options_text = ""
                    new_answer_key = None
                    for new_key, (original_key, value) in enumerate(options_items):
                        options_text += f"{chr(65 + new_key)}. {value}\n"  # 65 is the ASCII code for letter A
                        if QA['Answer'] == original_key:
                            new_answer_key = chr(65 + new_key)  # Update the answer key
                    option_dict = {chr(65 + new_key): value for new_key, (original_key, value) in enumerate(options_items)}

                    questions_text = QA['Question']
                    # if i == 0:
                    #     questions_text = "Should the test picture be classified as normal or abnormal?"
                    #     options_text = options_text.replace('Yes', 'Abnormal')
                    #     options_text = options_text.replace('No', 'Normal')
                    Question.append(
                        {
                            # "type": "text",
                            "text": f"{questions_text} \n"
                                    f"{options_text}",
                                    # f"{options_text} E. Can not determine\n"
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
    def get_query(self, conversations, history=[]):
        hint = instruction
        if self.few_shot:
            question = f"Following is {len(self.few_shot)} image of normal sample:"
                        # , which can be used as a template to compare the test image + DEFAULT_IMAGE_TOKEN*len(self.few_shot) + '\n' +
            for i, ref_image_path in enumerate(self.few_shot):
                question += f"{DEFAULT_IMAGE_TOKEN}\n"
            # if args.domain_knowledge:
            #     normal_summary_path = os.path.dirname(self.few_shot[0]) + '/summary.json'
            #     with open(normal_summary_path, 'r', encoding='utf-8') as file:
            #         summary = json.load(file)
            #     question += f"Summary of the normal pattern: {summary}\n"
        else:
            question = ""

        if self.defect_shot:
            question += f"Following is {len(self.defect_shot)} image of defect sample:"
                        # , which can be used as a template to compare the test image + DEFAULT_IMAGE_TOKEN*len(self.defect_shot) + '\n' +
            for i, ref_image_path in enumerate(self.defect_shot):
                question += f"{DEFAULT_IMAGE_TOKEN}\n"
            # if args.domain_knowledge:
            #     defect_summary_path = os.path.dirname(self.defect_shot[0]) + '/summary.json'
            #     with open(defect_summary_path, 'r', encoding='utf-8') as file:
            #         summary = json.load(file)
            #     question += f"Summary of the defect pattern: {summary}\n"

        if self.domain_knowledge:
            question = f"Following is the domain knowledge which contains some type of defect and the normal object characteristics: \n {self.domain_knowledge}" + question

        question = question +"Test image: \n" + DEFAULT_IMAGE_TOKEN + '\n'

        if self.args.text_only:
            hint = "Although you can not see the image. Answer with the option's letter from the given choices directly. "
            question = ""

        question = hint + '\n' + question

        if history:
            question += "Followings are the previous questions and answers: \n"
        for idx, (q, a) in enumerate(history):
            question += f"{q}\n"
            question += f"Answer: {a}\n"

        question += "Following is new question list: \n"
        for conversation in conversations:
            question += conversation['text']

        qs = question + '\n' + "Answer with the letter."
        # if self.model.config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        # if args.single_pred_prompt:
        #     if args.lang == 'cn':
        #         qs = qs + '\n' + "请直接回答选项字母。"
        #     else:
        #         qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # Process images
        query_image = cv2.imread(self.image_path)
        if self.visualization:
            self.visualize_image(query_image)
        base64_image = self.encode_image_to_base64(query_image)
        query_image = load_image_from_base64(base64_image)

        ref_images = []
        for ref_image_path in self.few_shot:
            # if not is_anomaly:
                ref_image = cv2.imread(ref_image_path)
                if self.visualization:
                    self.visualize_image(ref_image)
                ref_base64_image = self.encode_image_to_base64(ref_image)
                ref_image = load_image_from_base64(ref_base64_image)
                ref_images.append(ref_image)

        for ref_image_path in self.defect_shot:
                ref_image = cv2.imread(ref_image_path)
                if self.visualization:
                    self.visualize_image(ref_image)
                ref_base64_image = self.encode_image_to_base64(ref_image)
                ref_image = load_image_from_base64(ref_base64_image)
                ref_images.append(ref_image)

        images = ref_images + [query_image]

        image_tensor = process_images(images, self.image_processor, self.context_len)

        # 将image_tensor第一维变为list
        # image_tensor = [image_tensor[i].half().cuda() for i in range(len(image_tensor))]
        image_sizes = [image.size for image in images]
        if self.args.text_only:
            return input_ids, None, None
        return input_ids, image_tensor, image_sizes


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="../../Cambrian/ckpt/cambrian-8b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")

    parser.add_argument("--dtype", type=str, default="fp32")

    parser.add_argument("--few_shot_model", type=int, default=0)
    parser.add_argument("--similar_template", action="store_true")
    parser.add_argument("--reproduce", action="store_true")

    # parser.add_argument("--defect-shot", type=int, default=1)
    parser.add_argument("--record_history", action="store_true")
    parser.add_argument("--CoT", action="store_true")
    parser.add_argument("--domain_knowledge", action="store_true")
    parser.add_argument("--text_only", action="store_true")
    parser.add_argument("--domain_knowledge_path", type=str, default="../../../dataset/MMAD/domain_knowledge.json")
    parser.add_argument("--agent", action="store_true")


    args = parser.parse_args()

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    if args.dtype == "4bit":
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, load_4bit=True, cache_dir="~/.cache/huggingface/hub")
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, cache_dir="~/.cache/huggingface/hub")

    model_name = os.path.split(model_path.rstrip('/'))[-1]
    if args.CoT:
        model_name = model_name + "_CoT"
    if args.domain_knowledge:
        model_name = model_name + "_Domain_knowledge"
        with open(args.domain_knowledge_path, "r") as file:
            all_domain_knowledge = json.load(file)
    if args.text_only:
        model_name = model_name + "_Text_only"
    if args.similar_template:
        model_name = model_name + "_Similar_template"
    if args.agent:
        model_name = model_name + "_Agent" # TODO: Add agent

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
        if args.domain_knowledge:
            dataset_name = image_path.split("/")[0].replace("DS-MVTec", "MVTec")
            object_name = image_path.split("/")[1]

            domain_knowledge = '\n'.join(all_domain_knowledge[dataset_name][object_name].values())
        else:
            domain_knowledge = None
        rel_image_path = os.path.join(args.data_path, image_path)
        rel_few_shot = [os.path.join(args.data_path, path) for path in few_shot]
        cambrianquery = CambrianQuery(image_path=rel_image_path, text_gt=text_gt,
                           tokenizer=tokenizer, model=model, image_processor=image_processor, context_len=context_len,
                           few_shot=rel_few_shot, defect_shot=[], visualization=False, domain_knowledge=domain_knowledge, args=args)
        questions, answers, gpt_answers = cambrianquery.generate_answer()
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
