# llava-1.6
python llava_query.py --model-path ../../LLaVA/llava-v1.6-34b/ --dtype 4bit

CUDA_VISIBLE_DEVICES=1 python llava_query.py --model-path ../../LLaVA/ckpt/llava-next-interleave-qwen-7b-dpo

CUDA_VISIBLE_DEVICES=0 python llava_query.py --model-path ../../LLaVA/llava-v1.5-13b --text_only

