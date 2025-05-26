import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def caculate_accuracy_mmad(answers_json_path, normal_flag='good', show_overkill_miss=False):
    # For storing all answers
    if os.path.exists(answers_json_path):
        with open(answers_json_path, "r") as file:
            all_answers_json = json.load(file)
    dataset_names = []
    type_list = []
    for answer in all_answers_json:
        dataset_name = answer['image'].split('/')[0]
        question_type = answer['question_type']
        if question_type in ["Object Structure", "Object Details"]:
            question_type = "Object Analysis"
        if dataset_name not in dataset_names:
            dataset_names.append(dataset_name)
        if type not in type_list:
            type_list.append(question_type)

    # Initialize statistics data structure
    question_stats = {dataset_name: {} for dataset_name in dataset_names}
    detection_stats = {dataset_name: {} for dataset_name in dataset_names}
    for dataset_name in dataset_names:
        detection_stats[dataset_name]['normal'] = {'total': 0, 'correct': 0, 'correct_answers': {}, 'answers': {}}
        detection_stats[dataset_name]['abnormal'] = {'total': 0, 'correct': 0, 'correct_answers': {}, 'answers': {}}
        for question_type in type_list:
            question_stats[dataset_name][question_type] = {'total': 0, 'correct': 0, 'correct_answers': {}, 'answers': {}}

    for answer in all_answers_json:
        dataset_name = answer['image'].split('/')[0]
        question_type = answer['question_type']
        if question_type in ["Object Structure", "Object Details"]:
            question_type = "Object Analysis"
        gpt_answer = answer['gpt_answer']
        correct_answer = answer['correct_answer']
        if correct_answer not in ['A', 'B', 'C', 'D', 'E'] or gpt_answer not in ['A', 'B', 'C', 'D', 'E']:
            all_answers_json.remove(answer)
            print("Remove error:", "correct_answer:", correct_answer, "gpt_answer:", gpt_answer)
            continue

        question_stats[dataset_name][question_type]['total'] += 1
        if answer['correct_answer'] == answer['gpt_answer']:
            question_stats[dataset_name][question_type]['correct'] += 1

        if question_type == "Anomaly Detection":
            if normal_flag in answer['image']:
                detection_stats[dataset_name]['normal']['total'] += 1
                if answer['correct_answer'] == answer['gpt_answer']:
                    detection_stats[dataset_name]['normal']['correct'] += 1
            else:
                detection_stats[dataset_name]['abnormal']['total'] += 1
                if answer['correct_answer'] == answer['gpt_answer']:
                    detection_stats[dataset_name]['abnormal']['correct'] += 1


        answers_dict = question_stats[dataset_name][question_type]['answers']
        if gpt_answer not in answers_dict:
            answers_dict[gpt_answer] = 0
        answers_dict[gpt_answer] += 1
        correct_answers_dict = question_stats[dataset_name][question_type]['correct_answers']
        if correct_answer not in correct_answers_dict:
            correct_answers_dict[correct_answer] = 0
        correct_answers_dict[correct_answer] += 1

    # Create accuracy table
    accuracy_df = pd.DataFrame(index=dataset_names)
    for dataset_name in dataset_names:
        for question_type in type_list:
            total = question_stats[dataset_name][question_type]['total']
            correct = question_stats[dataset_name][question_type]['correct']
            cls_accuracy = correct / total if total != 0 else 0
            accuracy_df.at[dataset_name, question_type] = cls_accuracy*100

            if question_type in ['Anomaly Detection']:
                TP = detection_stats[dataset_name]['abnormal']['correct']
                FP = detection_stats[dataset_name]['normal']['total'] - detection_stats[dataset_name]['normal']['correct']
                FN = detection_stats[dataset_name]['abnormal']['total'] - detection_stats[dataset_name]['abnormal']['correct']
                TN = detection_stats[dataset_name]['normal']['correct']
                Precision = TP / (TP + FP) if (TP + FP) != 0 else 0
                Recall = TP / (TP + FN) if (TP + FN) != 0 else 0
                TPR = Recall
                FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
                normal_acc = detection_stats[dataset_name]['normal']['correct'] / detection_stats[dataset_name]['normal']['total'] if detection_stats[dataset_name]['normal']['total'] != 0 else 0
                anomaly_acc = detection_stats[dataset_name]['abnormal']['correct'] / detection_stats[dataset_name]['abnormal']['total'] if detection_stats[dataset_name]['abnormal']['total'] != 0 else 0
                # accuracy_df.at[dataset_name, 'normal_acc'] = normal_acc
                # accuracy_df.at[dataset_name, 'anomaly_acc'] = anomaly_acc
                accuracy_df.at[dataset_name, 'Anomaly Detection'] = (normal_acc+anomaly_acc)/2*100

    # Calculate the average accuracy for each question
    accuracy_df['Average'] = accuracy_df.mean(axis=1)

    if show_overkill_miss:
        for dataset_name in dataset_names:
            normal_acc = detection_stats[dataset_name]['normal']['correct'] / detection_stats[dataset_name]['normal'][
                'total'] if detection_stats[dataset_name]['normal']['total'] != 0 else 0
            anomaly_acc = detection_stats[dataset_name]['abnormal']['correct'] / detection_stats[dataset_name]['abnormal'][
                'total'] if detection_stats[dataset_name]['abnormal']['total'] != 0 else 0
            accuracy_df.at[dataset_name, 'Overkill'] = (1 - normal_acc) * 100
            accuracy_df.at[dataset_name, 'Miss'] = (1 - anomaly_acc) * 100

    accuracy_df.loc['Average'] = accuracy_df.mean()

    # Data visualization
    plt.figure(figsize=(10, 7))
    sns.heatmap(accuracy_df, annot=True, cmap='coolwarm', fmt=".1f", vmax=100, vmin=25)
    plt.title(f'Accuracy of {os.path.split(answers_json_path)[-1].replace(".json", "")}')
    # Rotate X-axis labels
    plt.xticks(rotation=30, ha='right')  # ha='right' can make labels slightly tilted for better readability

    # Automatically adjust borders to reduce whitespace
    plt.tight_layout()
    plt.show()

    # Save accuracy table
    accuracy_path = answers_json_path.replace('.json', '_accuracy.csv')
    accuracy_df.to_csv(accuracy_path)

    print(accuracy_df)
    return question_stats

def caculate_accuracy(answers_json_path, normal_flag='good'): # for mvtec only
    # For storing all answers
    if os.path.exists(answers_json_path):
        with open(answers_json_path, "r") as file:
            all_answers_json = json.load(file)
    # Statistic classname
    classname = []
    for answer in all_answers_json:
        cls = answer['class']
        if cls not in classname:
            classname.append(cls)
    # Initialize statistics data structure
    question_stats = {'normal': {}, 'anomaly': {}}

    for category in ['normal', 'anomaly']:
        for i in range(1, 6):
            question_stats[category][i] = {}
            for cls in classname:
                question_stats[category][i][cls] = {'total': 0, 'correct': 0, 'correct_answers': {}, 'answers': {}}

    count = 0
    question_number = 1
    last_image = ''
    # Fill statistics data structure
    for answer in all_answers_json:
        cls = answer['class']
        question_text = answer['question']['text']
        if 'Question' in question_text:
            # question_number = int(question_text.split(':')[0].split(' ')[1])
            question_number = int(question_text.split('Question')[1].strip()[0])
        elif answer['image'] == last_image:
            question_number += 1
        else:
            question_number = 1
        last_image = answer['image']

        is_normal = normal_flag in answer['image']
        category = 'normal' if is_normal else 'anomaly'
        # if answer['gpt_answer'] == '' or answer['gpt_answer'] == '':
        #     count += 1
        #     continue
        # Update total and correct
        question_stats[category][question_number][cls]['total'] += 1
        if answer['correct_answer'] == answer['gpt_answer']:
            question_stats[category][question_number][cls]['correct'] += 1
        gpt_answer = answer['gpt_answer']
        correct_answer = answer['correct_answer']
        if correct_answer not in ['A', 'B', 'C', 'D', 'E'] or gpt_answer not in ['A', 'B', 'C', 'D', 'E']:
            # Remove from all_answers_json and save
            all_answers_json.remove(answer)
            print("correct_answer:", correct_answer, "gpt_answer:", gpt_answer)

            continue
        # Update answer count
        answers_dict = question_stats[category][question_number][cls]['answers']
        if gpt_answer not in answers_dict:
            answers_dict[gpt_answer] = 0
        answers_dict[gpt_answer] += 1
        # Update correct answer count
        correct_answers_dict = question_stats[category][question_number][cls]['correct_answers']
        if correct_answer not in correct_answers_dict:
            correct_answers_dict[correct_answer] = 0
        correct_answers_dict[correct_answer] += 1
    # with open(answers_json_path, "w") as file:
    #     json.dump(all_answers_json, file, indent=4)

    # Anomaly question: 1 existence 2 type 3 location 4 appearance 5 other
    Anomaly_Question = ["Existence", "Defect Type", "Defect Location", "Defect Appearance", "Other"]
    # Normal question: 1 existence 2-5 other
    Normal_Question = ["Existence", "Other", "Other", "Other", "Other"]

    # Recount based on question and category
    Question_label = ["Existence", "Defect Type", "Defect Location", "Defect Appearance", "Other"]
    new_question_stats = {}
    for cls in classname:
        new_question_stats[cls] = {}
        for question_label in Question_label:
            new_question_stats[cls][question_label] = {'total': 0, 'correct': 0}
    for cls in classname:
        for category in ['normal', 'anomaly']:
            for i in range(1, 6):
                if category == 'normal':
                    question_label = Normal_Question[i - 1]
                else:
                    question_label = Anomaly_Question[i - 1]
                new_question_stats[cls][question_label]['total'] += question_stats[category][i][cls]['total']
                new_question_stats[cls][question_label]['correct'] += question_stats[category][i][cls]['correct']

    # Create accuracy table
    accuracy_df = pd.DataFrame(index=classname)
    for cls in classname:
        for question_label in Question_label:
            total = new_question_stats[cls][question_label]['total']
            correct = new_question_stats[cls][question_label]['correct']
            cls_accuracy = correct / total if total != 0 else 0
            accuracy_df.at[cls, question_label] = cls_accuracy

    # Calculate the average accuracy for each question
    accuracy_df['Average'] = accuracy_df.mean(axis=1)


    for cls in classname:
        TP = question_stats['anomaly'][1][cls]['correct']
        FP = question_stats['normal'][1][cls]['total'] - question_stats['normal'][1][cls]['correct']
        FN = question_stats['anomaly'][1][cls]['total'] - question_stats['anomaly'][1][cls]['correct']
        TN = question_stats['normal'][1][cls]['correct']
        Precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        Recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        # F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) != 0 else 0
        # accuracy_df.at[cls, 'Existence (F1)'] = F1
        TPR = Recall
        FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
        # # auroc = (1 - FPR) * TPR
        # auroc = (1 - FPR + TPR) / 2
        # accuracy_df.at[cls, 'Existence (AUROC)'] = auroc
        # # AUPR = Precision*Recall
        # AUPR = (Precision + Recall) / 2
        # accuracy_df.at[cls, 'Existence (AUPR)'] = AUPR

        normal_acc = question_stats['normal'][1][cls]['correct'] / question_stats['normal'][1][cls]['total'] if question_stats['normal'][1][cls]['total'] != 0 else 0
        anomaly_acc = question_stats['anomaly'][1][cls]['correct'] / question_stats['anomaly'][1][cls]['total'] if question_stats['anomaly'][1][cls]['total'] != 0 else 0
        accuracy_df.at[cls, 'Overkill'] = 1 - normal_acc
        accuracy_df.at[cls, 'Miss'] = 1 - anomaly_acc



    # Calculate the average accuracy for each category
    accuracy_df.loc['Average'] = accuracy_df.mean()
    # Data visualization
    plt.figure(figsize=(10, 9))
    sns.heatmap(accuracy_df, annot=True, cmap='coolwarm', fmt=".2f", vmax=1, vmin=0)
    plt.title(f'Accuracy of {os.path.split(answers_json_path)[-1].replace(".json", "")}')
    # Rotate X-axis labels
    plt.xticks(rotation=30, ha='right')  # ha='right' can make labels slightly tilted for better readability
    plt.show()

    # Save accuracy table
    accuracy_path = answers_json_path.replace('.json', '_accuracy.csv')
    accuracy_df.to_csv(accuracy_path)

    print(accuracy_df)
    return question_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--answers_json_path', type=str, default='~/project/MLLM-AD/Transformers/result/answers_1_shot_InternVL2-40B_Agent-AnomalyCLIP-with-contour_Debug.json')
    parser.add_argument('--normal_flag', type=str, default='good')
    args = parser.parse_args()
    # caculate_accuracy(args.answers_json_path, args.normal_flag)
    caculate_accuracy_mmad(args.answers_json_path, show_overkill_miss=True)
