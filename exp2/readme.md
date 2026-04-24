# MIRL Experiment 2: GRPO Training for Cross-Lingual Collapse

This experiment trains a multilingual LLM (Qwen 2.5 7B Instruct) using Group Relative Policy Optimization (GRPO) on the MGSM math dataset, then analyzes cross-lingual representation collapse via mechanistic interpretability tools (TransformerLens). The training saves checkpoints at regular intervals for later layer-wise analysis.

## Directory Structure

```
MIRL/
├── models/                          # Shared model weights (downloaded once)
│   ├── Qwen2.5-7B-Instruct/
│   └── gemma-4-31b-it/
├── exp2/
│   ├── data/                        # Dataset files
│   │   ├── url-nlp/                 # Cloned MGSM repo (auto-created)
│   │   ├── train.json               # 8 few-shot exemplars per language
│   │   ├── test.json                # 250 evaluation problems per language
│   │   ├── grpo_train.json          # 200 training examples per language
│   │   └── grpo_test.json           # 58 test examples per language (50 + 8 exemplars)
│   ├── logs/                        # All log files
│   │   ├── models.log               # Model I/O: prompts and generated responses
│   │   ├── rewards.log              # Reward computation: regex/judge I/O
│   │   ├── train.log                # Training loop: loss, rewards, steps
│   │   └── *.log                    # Dataset and download logs
│   ├── outputs/                     # Training outputs
│   │   ├── checkpoints/             # LoRA checkpoints at each save step
│   │   │   ├── step_10/
│   │   │   │   ├── adapter/         # LoRA weights (for TransformerLens later)
│   │   │   │   └── train_state.pt   # Optimizer + scheduler + metadata
│   │   │   ├── step_20/
│   │   │   └── ...
│   │   ├── training_metrics.json    # All step-level metrics
│   │   └── eval_iter_*.json         # Per-iteration evaluation results
│   ├── download_mgsm.py             # Step 1: Download MGSM dataset
│   ├── mgsm_splitter.py             # Step 2: Split into GRPO train/test
│   ├── download_models.py           # Step 3: Download model weights
│   ├── grpo_dataset.py              # Dataset class (imported by train.py)
│   ├── models.py                    # Model management (imported by train.py)
│   ├── rewards.py                   # Reward computation (imported by train.py)
│   └── train.py                     # Step 4: Main GRPO training loop
```

## Prerequisites

Install the required packages in your conda environment:

```bash
conda create -n mirl python=3.14
conda activate mirl
pip install torch transformers peft bitsandbytes tqdm tiktoken protobuf datasets
```

Note: `bitsandbytes` is needed only if using 4-bit quantization (QLoRA). If training in full precision, it is optional.

## Step-by-Step Execution

### Step 1: Download MGSM Dataset (Login Node — needs internet)

```bash
cd /path/to/MIRL
python exp2/download_mgsm.py
```

This clones the MGSM data from GitHub (`google-research/url-nlp`) and creates `exp2/data/train.json` (8 exemplars × 5 languages) and `exp2/data/test.json` (250 problems × 5 languages).

**Config keys to check** in `download_mgsm.py → main()`:
- `languages`: Which MGSM languages to include. Default: `["en", "bn", "te", "th", "ru"]`
- `data_dir`: Where to save JSON files. Default: `"./exp2/data"`

### Step 2: Prepare GRPO Train/Test Splits (Login Node — no GPU needed)

```bash
python exp2/mgsm_splitter.py
```

Splits the 250 test problems into 200 for GRPO training and 50 for evaluation, then merges the 50 with the 8 exemplars to get 58 test examples. Creates `exp2/data/grpo_train.json` and `exp2/data/grpo_test.json`.

**Config keys to check** in `mgsm_splitter.py → main()`:
- `train_size`: How many of the 250 to use for training. Default: `200`
- `languages`: Must match Step 1. Default: `["en", "bn", "te", "th", "ru"]`

### Step 3: Download Model Weights (Login Node — needs internet)

```bash
python exp2/download_models.py
```

Or manually:

```python
from huggingface_hub import snapshot_download

# Policy + Reference model
snapshot_download(
    "Qwen/Qwen2.5-7B-Instruct",
    local_dir="/path/to/MIRL/models/Qwen2.5-7B-Instruct",
    local_dir_use_symlinks=False,
)

# Judge LLM (only needed if using judge-based reward)
snapshot_download(
    "google/gemma-4-31b-it",
    local_dir="/path/to/MIRL/models/gemma-4-31b-it",
    local_dir_use_symlinks=False,
)
```

Models are saved under `./models/` (shared across experiments, not inside `exp2/`).

### Step 4: Run GRPO Training (Compute Node — needs GPU, no internet)

```bash
# In your SLURM job script or interactive session:
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /path/to/MIRL
python exp2/train.py
```

This runs the full GRPO training loop. All outputs go to `exp2/outputs/` and logs to `exp2/logs/`.

## GPU Configuration Guide

All GPU assignments are in `train.py → main() → config`. Here are the three scenarios:

### Scenario A: 2 GPUs (Recommended)

```python
# GPU 0: Policy model (training)
# GPU 1: Reference model + Judge LLM (inference only)
"model_config": {
    "policy_device": "cuda:0",
    "reference_device": "cuda:1",
    "policy_load_in_4bit": False,      # Full precision policy
    "reference_load_in_4bit": False,    # Full precision reference
},
"reward_config": {
    "judge_device": "cuda:1",
    "judge_load_in_4bit": True,         # 4-bit judge to save memory
},
```

Memory estimate (2 GPUs):
- GPU 0: ~50-60 GB (policy + optimizer + gradients)
- GPU 1: ~14 GB (ref) + ~17 GB (judge 4-bit) = ~31 GB

### Scenario B: 1 GPU (80 GB, uses QLoRA)

```python
"model_config": {
    "policy_device": "cuda:0",
    "reference_device": "cuda:0",
    "policy_load_in_4bit": True,        # QLoRA: 4-bit base + full-precision LoRA
    "reference_load_in_4bit": True,     # 4-bit reference
},
"reward_config": {
    "judge_device": "cuda:0",
    "judge_load_in_4bit": True,         # 4-bit judge
},
```

Memory estimate (1 GPU):
- ~5 GB (policy 4-bit) + ~5 GB (ref 4-bit) + ~17 GB (judge 4-bit) + optimizer/gradients = ~40-50 GB

### Scenario C: 3 GPUs

```python
"model_config": {
    "policy_device": "cuda:0",
    "reference_device": "cuda:1",
    "policy_load_in_4bit": False,
    "reference_load_in_4bit": False,
},
"reward_config": {
    "judge_device": "cuda:2",
    "judge_load_in_4bit": False,        # Full precision judge on dedicated GPU
},
```

### Scenario D: No Judge LLM (fastest, accuracy-only via regex)

```python
"reward_config": {
    "weight_accuracy_regex": 1.0,
    "weight_accuracy_judge": 0.0,       # Judge not loaded at all
},
```

This skips loading the judge model entirely, saving ~17 GB GPU memory and significant inference time per step.

## Key Config Parameters

### Training Hyperparameters (in `train.py → config`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_iterations` | 16 | Number of epochs (fresh rollouts each epoch) |
| `batch_size` | 8 | Minibatch size B for gradient updates |
| `group_size` | 4 | Number of rollout responses G per prompt |
| `epsilon` | 0.2 | Clipping parameter ε for PPO-style objective |
| `beta` | 0.04 | KL penalty coefficient β |
| `learning_rate` | 1e-5 | AdamW learning rate |
| `max_grad_norm` | 1.0 | Gradient clipping norm |
| `checkpoint_save_freq` | 10 | Save checkpoint every N gradient steps |
| `resume_from_step` | None | Set to step number to resume training |

### LoRA Parameters (in `config["model_config"]`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_rank` | 64 | LoRA rank (higher = more parameters) |
| `lora_alpha` | 16 | LoRA scaling factor |
| `lora_dropout` | 0.05 | Dropout on LoRA layers |
| `lora_target_modules` | `["q_proj", "k_proj", "v_proj", "o_proj"]` | Which layers get LoRA adapters |

### Generation Parameters (in `config["model_config"]`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 512 | Maximum response length |
| `temperature` | 0.7 | Sampling temperature for rollouts |
| `top_p` | 0.95 | Nucleus sampling threshold |
| `tokens_after_final_answer` | 10 | Tokens to generate after "Final Answer:" |

### Reward Weights (in `config["reward_config"]`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weight_accuracy_regex` | 1.0 | Weight for regex-based accuracy (0 or 1) |
| `weight_accuracy_judge` | 0.0 | Weight for judge LLM accuracy (0 or 1) |
| `weight_format` | 0.0 | Weight for format reward (Phase 2) |
| `weight_language_consistency` | 0.0 | Weight for language consistency (Phase 2) |

## Resuming Training

If training gets interrupted, set `resume_from_step` to the last saved checkpoint step:

```python
config = {
    "resume_from_step": 150,  # Will load checkpoint from exp2/outputs/checkpoints/step_150/
    ...
}
```

This restores the LoRA weights, optimizer state, and scheduler state.

## Monitoring Training

Since the compute node has no internet (no wandb), monitor via:

```bash
# Watch live training progress
tail -f exp2/logs/train.log

# Check model I/O (what prompts go in, what responses come out)
tail -f exp2/logs/models.log

# Check reward computation details
tail -f exp2/logs/rewards.log

# Check training metrics programmatically
python -c "
import json
metrics = json.load(open('exp2/outputs/training_metrics.json'))
for m in metrics[-5:]:
    print(f'step={m[\"step\"]} loss={m[\"loss\"]:.4f} reward={m[\"avg_reward\"]:.3f}')
"
```



