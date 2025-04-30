from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="OpenGVLab/InternVL2-1B",  # official Hugging Face repo id
    local_dir="../../InternVL/pretrained/InternVL2-1B",  # your target local folder
    local_dir_use_symlinks=False  # force a real copy rather than symlink
)