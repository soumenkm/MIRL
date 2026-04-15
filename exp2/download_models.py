from huggingface_hub import snapshot_download

snapshot_download(
    "openai/gpt-oss-20b",
    local_dir_use_symlinks=False,
)