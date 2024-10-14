# MMAD: The First-Ever Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection

![VideoQA](https://img.shields.io/badge/Task-Industry_Inspection-red)
![Video-MME](https://img.shields.io/badge/Dataset-MMAD-blue)
![Gemini](https://img.shields.io/badge/Model-Gemini--1.5-green)
![GPT-4o](https://img.shields.io/badge/Model-GPT--4-green)

[//]: # (## 💡 Highlights)
 Our benchmark responds to the following questions:
- How well are current MLLMs performing as industrial quality inspectors?
- Which MLLM performs the best in industrial anomaly detection? 
- What are the key challenges in industrial anomaly detection for MLLMs?

## 📜 News

[//]: # (- **[2024-10-13]** MMAD paper is released.)
- **[2024-10-08]** MMAD dataset and evaluation code are released.



## 👀 Overview
In the field of industrial inspection, Multimodal Large Language Models (MLLMs) have a high potential to renew the paradigms in practical applications due to their robust language capabilities and generalization abilities. However, despite their impressive problem-solving skills in many domains, MLLMs' ability in industrial anomaly detection has not been systematically studied. To bridge this gap, we present MMAD, the first-ever full-spectrum MLLMs benchmark in industrial Anomaly Detection. We defined seven key subtasks of MLLMs in industrial inspection and designed a novel pipeline to generate the MMAD dataset with 39,672 questions for 8,366 industrial images. With MMAD, we have conducted a comprehensive, quantitative evaluation of various state-of-the-art MLLMs.


<p align="center">
    <img src="./figs/overview.jpg" width="100%" height="100%">
</p>

## 📐 Dataset Examples
We collected 8,366 samples from 38 classes of industrial products across 4 public datasets, generating a total of 39,672 multiple-choice questions in 7 key subtasks.
<p align="center">
    <img src="./figs/examples.jpg" width="100%" height="100%">
</p>

## 🔮 Evaluation Pipeline

### 1. Data Preparation

Prepare the evaluation dataset by following the instructions provided in the [README.md](dataset/README.md) file located in the dataset folder. 

### 2. Model Configuration

Due to different MLLMs' input and output handling methods, we have created separate example files for each MLLM being tested, which can be found in the evaluation folder.

For Gemini and GPT4, an API KEY is required and should be provided in the respective file.

For Cambrain, LLaVA, and SPHINX, the environment must be set up as per the original repository. (Here are the addresses to refer to: [Cambrain](https://github.com/cambrian-mllm/cambrian), [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT/), [SPHINX](https://github.com/Alpha-VLLM/LLaMA2-Accessory))

For Qwen, MiniCPM, InternVL, and similar models, simply install the transformers library (`pip install transformers`).

### 3. Run Evaluation

Each test file uses the `--model-path` argument to specify the model, and `--few_shot_model` to indicate the number of normal samples in the prompt.

Examples:
```
cd ./evaluation/Transformers
python internvl_query.py --model-path ../../InternVL/pretrained/InternVL2-1B

cd ./evaluation/LLaVA_Query
python llava_query.py --model-path ../../LLaVA/llava-v1.6-34b/ --dtype 4bit
```



## 👨‍💻 Todo
- [x] Release the dataset
- [x] Release the evaluation code
- [ ] Release the paper
- [ ] Release a version with images included on Hugging Face
