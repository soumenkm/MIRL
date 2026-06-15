from huggingface_hub import snapshot_download

# snapshot_download(
#     "Qwen/Qwen2.5-7B-Instruct",
#     local_dir="./models/Qwen2.5-7B-Instruct",
#     local_dir_use_symlinks=False,
# )

# snapshot_download(
#     "google/gemma-4-31b-it",
#     local_dir="./models/gemma-4-31b-it",
#     local_dir_use_symlinks=False,
# )

# snapshot_download(
#     "openai/gpt-oss-20b",
#     local_dir="./models/gpt-oss-20b",
#     local_dir_use_symlinks=False,
# )

# snapshot_download(
#     "google/gemma-3-27b-it",
#     local_dir="./models/google/gemma-3-27b-it",
#     local_dir_use_symlinks=False,
# )

snapshot_download(
    "google/gemma-4-26B-A4B-it",
    # local_dir="./models/gemma-4-26B-A4B-it",
    local_dir_use_symlinks=False,
)