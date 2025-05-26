import json
import os
import sys
import random
from tqdm import tqdm
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from PIL import Image
import numpy as np
import argparse

sys.path.append("../")
# from data.mvtec import ADDataset
from helper.summary import caculate_accuracy_mmad
from GPT4.gpt4v import GPT4Query, instruction

sys.path.append("anomalygpt/code")
from anomalygpt.code.model.openllama import OpenLLAMAPEFTModel

describles = {}
describles['bottle'] = "This is a photo of a bottle for anomaly detection, which should be round, without any damage, flaw, defect, scratch, hole or broken part."
describles['cable'] = "This is a photo of three cables for anomaly detection, cables cannot be missed or swapped, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['capsule'] = "This is a photo of a capsule for anomaly detection, which should be black and orange, with print '500', without any damage, flaw, defect, scratch, hole or broken part."
describles['carpet'] = "This is a photo of carpet for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['grid'] = "This is a photo of grid for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['hazelnut'] = "This is a photo of a hazelnut for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['leather'] = "This is a photo of leather for anomaly detection, which should be brown and without any damage, flaw, defect, scratch, hole or broken part."
describles['metal_nut'] = "This is a photo of a metal nut for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part, and shouldn't be fliped."
describles['pill'] = "This is a photo of a pill for anomaly detection, which should be white, with print 'FF' and red patterns, without any damage, flaw, defect, scratch, hole or broken part."
describles['screw'] = "This is a photo of a screw for anomaly detection, which tail should be sharp, and without any damage, flaw, defect, scratch, hole or broken part."
describles['tile'] = "This is a photo of tile for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['toothbrush'] = "This is a photo of a toothbrush for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['transistor'] = "This is a photo of a transistor for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['wood'] = "This is a photo of wood for anomaly detection, which should be brown with patterns, without any damage, flaw, defect, scratch, hole or broken part."
describles['zipper'] = "This is a photo of a zipper for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."

describles['candle'] = "This is a photo of 4 candles for anomaly detection, every candle should be round, without any damage, flaw, defect, scratch, hole or broken part."
describles['capsules'] = "This is a photo of many small capsules for anomaly detection, every capsule is green, should be without any damage, flaw, defect, scratch, hole or broken part."
describles['cashew'] = "This is a photo of a cashew for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['chewinggum'] = "This is a photo of a chewinggom for anomaly detection, which should be white, without any damage, flaw, defect, scratch, hole or broken part."
describles['fryum'] = "This is a photo of a fryum for anomaly detection on green background, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['macaroni1'] = "This is a photo of 4 macaronis for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['macaroni2'] = "This is a photo of 4 macaronis for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['pcb1'] = "This is a photo of pcb for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['pcb2'] = "This is a photo of pcb for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['pcb3'] = "This is a photo of pcb for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['pcb4'] = "This is a photo of pcb for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['pipe_fryum'] = "This is a photo of a pipe fryum for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."

class AnomlyGPTQuery(GPT4Query):
    def __init__(self, image_path, text_gt, model, few_shot=[], visualization=False, args=None):
        super(AnomlyGPTQuery, self).__init__(image_path, text_gt, few_shot, visualization)
        self.model = model

    def generate_answer(self):
        questions, answers = self.parse_conversation(self.text_gt)
        if questions == [] or answers == []:
            return questions, answers, None
        self.classname = self.image_path.split('/')[-4]
        # prompt_text = instruction
        # if self.classname in describles.keys() and "good" not in self.image_path:
        #     prompt_text = describles[self.classname]
        prompt_text = "This is a photo of a object for anomaly detection, which should be round, without any damage, flaw, defect, scratch, hole or broken part."

        history = []
        gpt_answers = []
        for i in range(len(questions)):
            part_questions = questions[i:i + 1]
            input = part_questions[0]["text"]
            for idx, (q, a) in enumerate(history):
                if idx == 0:
                    prompt_text += f'{q}\n### Assistant: {a}\n###'
                else:
                    prompt_text += f' Human: {q}\n### Assistant: {a}\n###'
            if len(history) == 0:
                prompt_text += f'{input}'
            else:
                prompt_text += f' Human: {input}'
            if i != 0:
                prompt_text += "Answer with the option's letter from the given choices directly."

            response, pixel_output = self.model.generate({
                'prompt': prompt_text,
                'image_paths': [self.image_path] if self.image_path else [],
                'audio_paths': [],
                'video_paths': [],
                'thermal_paths': [],
                'normal_img_paths': self.few_shot if self.few_shot else [],
                'top_p': 0.1,
                'temperature': 0.01,
                'max_tgt_len': 50,
                'modality_embeds': []
            })
            print(response)
            options = part_questions[0]["options"]
            gpt_answer = self.parse_answer(response, options)
            if len(gpt_answer) == 0:
                return questions, answers, None
            gpt_answers.append(gpt_answer[-1])
            if args.record_history:
                history.append((input, gpt_answer[-1]))
        print(gpt_answers)
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
                    #     questions_text = "Should the query picture be classified as normal or abnormal?"
                    #     options_text = options_text.replace('Yes', 'Abnormal')
                    #     options_text = options_text.replace('No', 'Normal')
                    if QA['type'] == 'Anomaly Detection':
                        questions_text = "Is there any anomaly in the image?"
                        Question.append(
                            {
                                "type": "text",
                                "text": f" {questions_text}",
                                        # f"{options_text}",
                                "options": option_dict,
                            },
                        )
                    else:
                        Question.append(
                            {
                                "type": "text",
                                "text": f"{questions_text} \n"
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


# if __name__=="__main__":
#     parser = argparse.ArgumentParser("AnomalyGPT", add_help=True)
#     # paths
#     parser.add_argument("--k_shot", type=int, default=1)
#     parser.add_argument("--ckpt", type=str, default="train_visa", choices=["train_visa", "train_mvtec", "train_supervised"])
#     command_args = parser.parse_args()
#
#     FEW_SHOT = command_args.k_shot
#
#     # init the model
#     args = {
#         'model': 'openllama_peft',
#         'imagebind_ckpt_path': 'anomalygpt/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
#         'vicuna_ckpt_path': 'anomalygpt/pretrained_ckpt/vicuna_ckpt/7b_v0',
#         'anomalygpt_ckpt_path': f'anomalygpt/code/ckpt/{command_args.ckpt}/pytorch_model.pt',
#         'delta_ckpt_path': 'anomalygpt/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
#         'stage': 2,
#         'max_tgt_len': 128,
#         'lora_r': 32,
#         'lora_alpha': 32,
#         'lora_dropout': 0.1,
#     }
#
#     model = OpenLLAMAPEFTModel(**args)
#     delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
#     model.load_state_dict(delta_ckpt, strict=False)
#     delta_ckpt = torch.load(args['anomalygpt_ckpt_path'], map_location=torch.device('cpu'))
#     model.load_state_dict(delta_ckpt, strict=False)
#     model = model.eval().half().cuda()
#
#     print(f'[!] init the 7b model over ...')
#
#     cfg = {
#         "path": "../../../dataset/Defect_Spectrum/DS-MVTec",
#         "splits": ["image"],
#         "few_shot_model": FEW_SHOT,
#         "normal_flag": "good",
#     }
#     noraml_cfg = {
#         "path": "../../../dataset/MVTec",
#         "splits": ["train"],
#     }
#     few_shot_model = cfg["few_shot_model"]
#     dataset_path = cfg["path"]
#     #classname 定义为dataset_path下的文件夹
#     classname = os.listdir(dataset_path)
#     splits = cfg["splits"]
#     # 检查是否都是文件夹
#     for c in classname:
#         if not os.path.isdir(os.path.join(dataset_path, c)):
#             classname.remove(c)
#     # indices = [0, 2]  # 想从列表中选择的元素的索引
#     #
#     # # 使用列表推导选择元素
#     # classname = [classname[i] for i in indices]
#
#     dataset = ADDataset(source=dataset_path, classnames=classname, splits=splits, training=False, transform_fn=None, normalize_fn=None)
#
#     normal_dataset = ADDataset(source=noraml_cfg["path"], classnames=classname, splits=noraml_cfg["splits"],training=True, transform_fn=None, normalize_fn=None)
#     model_name = args['anomalygpt_ckpt_path'].split('/')[-2]
#     answers_json_path = f"answers_{few_shot_model}_shot_{model_name}.json"
#
#     # For storing all answers
#     if os.path.exists(answers_json_path):
#         with open(answers_json_path, "r") as file:
#             all_answers_json = json.load(file)
#     else:
#         all_answers_json = []
#
#     existing_images = [a["image"] for a in all_answers_json]
#
#     Q1_correct = 0
#     processed_items = 0
#
#     for data_id, data in tqdm(enumerate(dataset), total=len(dataset), desc="Processing dataset"):
#         # If already have answers then skip
#         if data["image_path"] in existing_images:
#             continue
#         if data["text_gt"] is None:
#             continue
#         # Randomly draw data from ADDataset as few_shot, require category consistency
#         imgpaths_per_class = normal_dataset.imgpaths_per_class
#         normal_set = imgpaths_per_class[data["clsname"]]["good"]
#         # Randomly draw few_shot from normal_set
#         few_shot = random.sample(normal_set, few_shot_model)
#         anomalygpt_query = AnomlyGPTQuery(image_path=data["image_path"], text_gt=data["text_gt"],
#                            model=model, few_shot=few_shot, visualization=False)
#         questions, answers, gpt_answers = anomalygpt_query.generate_answer()
#         if gpt_answers is None or len(gpt_answers) != len(answers):
#             continue
#         correct = 0
#         for i, answer in enumerate(answers):
#             if gpt_answers[i] == answer:
#                 correct += 1
#         accuracy = correct / len(answers)
#         print(f"Accuracy: {accuracy:.2f}")
#
#         if answers[0] == gpt_answers[0]:
#             Q1_correct += 1
#         processed_items += 1
#         tqdm.write(f"Q1 accuracy: {Q1_correct / processed_items:.2f}")
#
#         # Update answer record
#         for q, a, ga in zip(questions, answers, gpt_answers):
#             answer_entry = {
#                 "class": data['clsname'],
#                 "image": data["image_path"],
#                 "question": q,
#                 "correct_answer": a,
#                 "gpt_answer": ga
#             }
#
#             all_answers_json.append(answer_entry)
#
#         if data_id % 10 == 0 or data_id == len(dataset) - 1:
#             # Save answers as JSON
#             with open(answers_json_path, "w") as file:
#                 json.dump(all_answers_json, file, indent=4)
#
#     caculate_accuracy(answers_json_path, cfg["normal_flag"])

if __name__=="__main__":
    parser = argparse.ArgumentParser("AnomalyGPT", add_help=True)
    # paths
    parser.add_argument("--few_shot_model", type=int, default=1)
    parser.add_argument("--similar_template", action="store_true")
    parser.add_argument("--reproduce", action="store_true")

    # parser.add_argument("--defect-shot", type=int, default=1)
    parser.add_argument("--record_history", action="store_true")
    parser.add_argument("--CoT", action="store_true")
    parser.add_argument("--domain_knowledge", action="store_true")
    parser.add_argument("--text_only", action="store_true")

    parser.add_argument("--ckpt", type=str, default="train_visa", choices=["train_visa", "train_mvtec", "train_supervised"])
    args = parser.parse_args()

    # init the model
    pre_args = {
        'model': 'openllama_peft',
        'imagebind_ckpt_path': 'anomalygpt/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
        'vicuna_ckpt_path': 'anomalygpt/pretrained_ckpt/vicuna_ckpt/7b_v0',
        'anomalygpt_ckpt_path': f'anomalygpt/code/ckpt/{args.ckpt}/pytorch_model.pt',
        'delta_ckpt_path': 'anomalygpt/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
        'stage': 2,
        'max_tgt_len': 128,
        'lora_r': 32,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
    }

    model = OpenLLAMAPEFTModel(**pre_args)
    delta_ckpt = torch.load(pre_args['delta_ckpt_path'], map_location=torch.device('cpu'))
    model.load_state_dict(delta_ckpt, strict=False)
    delta_ckpt = torch.load(pre_args['anomalygpt_ckpt_path'], map_location=torch.device('cpu'))
    model.load_state_dict(delta_ckpt, strict=False)
    model = model.eval().half().cuda()

    print(f'[!] init the 7b model over ...')

    model_name = 'AnomalyGPT' + "_" + args.ckpt
    if args.CoT:
        model_name = model_name + "_CoT"
    if args.domain_knowledge:
        model_name = model_name + "_Domain_knowledge"
    if args.text_only:
        model_name = model_name + "_Text_only"
    if args.similar_template:
        model_name = model_name + "_Similar_template"
    if args.record_history:
        model_name = model_name + "_Record_history"

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
        anomalygpt_query = AnomlyGPTQuery(image_path=rel_image_path, text_gt=text_gt,
                           model=model, few_shot=rel_few_shot, visualization=False, args=args)
        questions, answers, gpt_answers = anomalygpt_query.generate_answer()
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