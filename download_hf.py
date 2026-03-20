"""
Download HuggingFace models on login node
Run: python download_models.py
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# List of models to download
MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    # Add more models here
]

# List of datasets to download
DATASETS = [
    ("jbross-ibm-research/mgsm", "bn"),
    ("jbross-ibm-research/mgsm", "ca"),
    ("jbross-ibm-research/mgsm", "de"),
    ("jbross-ibm-research/mgsm", "en"),
    ("jbross-ibm-research/mgsm", "es"),
    ("jbross-ibm-research/mgsm", "eu"),
    ("jbross-ibm-research/mgsm", "fr"),
    ("jbross-ibm-research/mgsm", "gl"),
    ("jbross-ibm-research/mgsm", "ja"),
    ("jbross-ibm-research/mgsm", "ru"),
    ("jbross-ibm-research/mgsm", "sw"),
    ("jbross-ibm-research/mgsm", "te"),
    ("jbross-ibm-research/mgsm", "th"),
    ("jbross-ibm-research/mgsm", "zh"),
]

isModel = False
isData = True

if isModel:
    print("Downloading models...")
    for model_name in MODELS:
        print(f"\n[Downloading] {model_name}")
        try:
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"  ✓ Tokenizer downloaded")
            
            # Download model (just config, not weights - saves space)
            # To download full weights, uncomment next line:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            print(f"  ✓ Model config downloaded")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

if isData:
    print("Downloading datasets...")
    for dataset_name, config in DATASETS:
        print(f"\n[Downloading] {dataset_name} ({config})")
        try:
            dataset = load_dataset(dataset_name, config, split="test")
            print(f"  ✓ Dataset downloaded ({len(dataset)} examples)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

