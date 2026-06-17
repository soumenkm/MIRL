#!/usr/bin/env python3
"""
verify_adapter_switch.py

Fast pre-flight check (≈2 min, one prompt) that the eval.py adapter-switch
fix actually changes generations between checkpoints. Run this BEFORE
launching the full multi-hour eval re-run.

It mimics exactly what the fixed _activate_checkpoint does:
  - load base + initial PEFT wrapper via your model manager,
  - generate once with LoRA disabled (= step 0 / base),
  - load checkpoint-20 and checkpoint-360 as named adapters,
    set_adapter + enable_adapter_layers, generate once each,
  - assert the three token sequences are NOT all identical.

If base == ckpt20 == ckpt360, the fix didn't take (likely a PEFT version
API mismatch) and we debug before wasting a full run. If they differ, the
real eval will produce a non-flat curve.

Edit CONFIG to match the config dict you pass to eval.py's main().
"""

import sys
from pathlib import Path

import torch

# ---- adjust to mirror your eval.py model_config / dataset_config ----
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import GRPOModelManager          # noqa: E402
from grpo_dataset import GRPOMGSMDataset      # noqa: E402

MODEL_CONFIG = {
    # Fill these to match what eval.py uses. Only the fields the manager
    # needs are required here.
    "model_path": "models/Qwen2.5-7B-Instruct",
    "checkpoint_dir": Path("exp3/outputs/checkpoints"),
    # ... add any other keys GRPOModelManager expects (lora rank, device…)
    "model_name": Path("./models/Qwen2.5-7B-Instruct"),
            "policy_device": "cuda:0",
            "reference_device": "cuda:0",   # eval only loads policy; ref unused
            "dtype": "float16",
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "policy_load_in_4bit": False,
            "reference_load_in_4bit": False,
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95,
            "group_size": 1,
            "tokens_after_final_answer": 20,
            "log_max_response_chars": 500,
            "checkpoint_dir": Path("./exp3/outputs/checkpoints"),
            "log_dir":        Path("./exp3/logs"),
}
DATASET_CONFIG = {
    "data_dir": Path("exp3/data"),
    # ... match eval.py's dataset_config
            "log_dir":  Path("./exp3/logs"),
            "languages": ["en", "bn", "te", "th", "ru", "ja", "zh"],
            "num_few_shot": 1,
            "seed": 42,
}
CHECKPOINT_DIR = Path("exp3/outputs/checkpoints")
STEPS_TO_TEST = [20, 360]
MAX_NEW_TOKENS = 200
QWEN_IM_END = 151645


def build_chat_prompt(tokenizer, system, user):
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def generate(model, tokenizer, device, prompt_text):
    enc = tokenizer(prompt_text, return_tensors="pt", truncation=False).to(device)
    out = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=[tokenizer.eos_token_id, QWEN_IM_END],
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    gen = out[0, enc["input_ids"].shape[1]:]
    return gen.tolist()


def main():
    mgr = GRPOModelManager(MODEL_CONFIG)
    model = mgr.policy_model
    tok = mgr.tokenizer
    device = mgr.policy_device

    ds = GRPOMGSMDataset(DATASET_CONFIG, split="test")
    item = ds[0]
    prompt = build_chat_prompt(tok, item["system"], item["user"])
    print(f"Prompt (first 120 chars): {prompt[:120]!r}\n")

    results = {}

    # --- step 0: base, LoRA disabled ---
    model.disable_adapter_layers()
    model.eval()
    results["base"] = generate(model, tok, device, prompt)
    print(f"[base ] {len(results['base'])} tokens, first 12: {results['base'][:12]}")

    # --- each checkpoint as a named adapter ---
    for step in STEPS_TO_TEST:
        ckpt = CHECKPOINT_DIR / f"checkpoint-{step}"
        name = f"ckpt_{step}"
        if name not in model.peft_config:
            model.load_adapter(str(ckpt), adapter_name=name, is_trainable=False)
        model.set_adapter(name)
        model.enable_adapter_layers()
        model.eval()
        active = getattr(model, "active_adapter", None)
        results[name] = generate(model, tok, device, prompt)
        print(f"[{name}] active_adapter={active!r}  "
              f"{len(results[name])} tokens, first 12: {results[name][:12]}")

    print()
    seqs = list(results.values())
    all_same = all(s == seqs[0] for s in seqs)
    if all_same:
        print(">>> FAIL: all generations identical. Adapter switch NOT working.")
        print(">>> Do NOT launch the full eval yet. Check PEFT version / API.")
        sys.exit(1)
    else:
        # Report which pairs differ.
        b, c20 = results["base"], results.get(f"ckpt_{STEPS_TO_TEST[0]}")
        print(">>> PASS: generations differ across checkpoints.")
        print(f">>>   base vs ckpt_{STEPS_TO_TEST[0]} identical? {b == c20}")
        if len(STEPS_TO_TEST) > 1:
            c_last = results[f"ckpt_{STEPS_TO_TEST[-1]}"]
            print(f">>>   ckpt_{STEPS_TO_TEST[0]} vs ckpt_{STEPS_TO_TEST[-1]} "
                  f"identical? {c20 == c_last}")
        print(">>> Safe to launch the full eval re-run.")


if __name__ == "__main__":
    main()