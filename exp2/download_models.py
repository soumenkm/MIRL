from huggingface_hub import snapshot_download

snapshot_download(
    "Qwen/Qwen2.5-7B-Instruct",
    local_dir="/home/speech-nlp-cse/23m2157/MS_Research/MIRL/models/Qwen2.5-7B-Instruct",
    local_dir_use_symlinks=False,
)

snapshot_download(
    "openai/gpt-oss-20b",
    local_dir="/home/speech-nlp-cse/23m2157/MS_Research/MIRL/models/gpt-oss-20b",
    local_dir_use_symlinks=False,
)

snapshot_download(
    "google/gemma-4-31b-it",
    local_dir="/home/speech-nlp-cse/23m2157/MS_Research/MIRL/models/gemma-4-31b-it",
    local_dir_use_symlinks=False,
)