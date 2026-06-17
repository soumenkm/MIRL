#!/usr/bin/env python3
"""
diagnose_peft_state.py

The verify run showed active_adapter switches correctly but generations are
identical, even vs the LoRA-disabled base. That means the adapter delta is
zero at forward time in every case. This script interrogates the LIVE model
to find out exactly why, by inspecting the actual LoRA layer objects.

It answers:
  A. What adapters exist, what is active, and what does PEFT report?
  B. After disable_adapter_layers(), is the layer .disable_adapters flag set?
     After enable_adapter_layers(), is it cleared?
  C. After load_adapter("ckpt_20"), are the in-memory lora_B weights for the
     ACTIVE adapter actually nonzero? (If zero -> weights didn't bind to the
     active adapter; name/key mismatch.)
  D. Direct forward test: same input_ids through (base-disabled), (ckpt_20),
     (ckpt_360) -> compare the logits row, not just argmax tokens. Even a tiny
     applied LoRA changes logits in float; identical logits = truly no delta.
  E. peft / transformers / torch versions.

Run with the same env/config as verify_adapter.py.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import GRPOModelManager          # noqa: E402

MODEL_CONFIG = {
    "model_path": "models/Qwen2.5-7B-Instruct",
    "checkpoint_dir": Path("exp3/outputs/checkpoints"),
    # add any other keys your GRPOModelManager requires
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
CKPT_DIR = Path("exp3/outputs/checkpoints")
STEPS = [20, 360]


def find_lora_layers(model, limit=3):
    """Return up to `limit` modules that look like PEFT LoRA layers."""
    found = []
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            found.append((name, mod))
            if len(found) >= limit:
                break
    return found


def report_versions():
    print("=" * 70)
    print("E. Versions")
    print("=" * 70)
    import peft, transformers
    print(f"  peft        = {peft.__version__}")
    print(f"  transformers= {transformers.__version__}")
    print(f"  torch       = {torch.__version__}")


def lora_B_sum(layer, adapter_name):
    """Sum |lora_B| for a given adapter name on a layer (0 if absent)."""
    B = layer.lora_B
    # lora_B is a ModuleDict keyed by adapter name
    if adapter_name in B:
        w = B[adapter_name].weight
        return float(w.abs().sum().item())
    return None


def disabled_flag(layer):
    # PEFT versions differ: attribute is usually `disable_adapters`
    for attr in ("disable_adapters", "_disable_adapters", "merged"):
        if hasattr(layer, attr):
            return attr, getattr(layer, attr)
    return None, None


def main():
    report_versions()

    mgr = GRPOModelManager(MODEL_CONFIG)
    model = mgr.policy_model
    model.eval()
    device = mgr.policy_device

    print("\n" + "=" * 70)
    print("A. Adapter inventory (before loading checkpoints)")
    print("=" * 70)
    print(f"  type(model)        = {type(model).__name__}")
    print(f"  peft_config keys   = {list(model.peft_config.keys())}")
    print(f"  active_adapter     = {getattr(model, 'active_adapter', None)!r}")
    layers = find_lora_layers(model)
    print(f"  sample LoRA layers : {[n for n, _ in layers]}")
    for n, l in layers[:1]:
        print(f"    {n}: lora_A keys={list(l.lora_A.keys())}, "
              f"lora_B keys={list(l.lora_B.keys())}")

    name0, l0 = layers[0]

    print("\n" + "=" * 70)
    print("B. disable / enable flag behaviour on a real layer")
    print("=" * 70)
    model.disable_adapter_layers()
    attr, val = disabled_flag(l0)
    print(f"  after disable_adapter_layers(): {attr}={val}")
    model.enable_adapter_layers()
    attr, val = disabled_flag(l0)
    print(f"  after enable_adapter_layers() : {attr}={val}")

    print("\n" + "=" * 70)
    print("C. load_adapter -> are the ACTIVE adapter's lora_B nonzero?")
    print("=" * 70)
    for step in STEPS:
        name = f"ckpt_{step}"
        ckpt = CKPT_DIR / f"checkpoint-{step}"
        if name not in model.peft_config:
            model.load_adapter(str(ckpt), adapter_name=name, is_trainable=False)
        model.set_adapter(name)
        model.enable_adapter_layers()
        # inspect the same layer
        bsum = lora_B_sum(l0, name)
        attr, val = disabled_flag(l0)
        # what does the layer think its active adapters are?
        layer_active = getattr(l0, "active_adapter", getattr(l0, "active_adapters", None))
        print(f"  [{name}] model.active={model.active_adapter!r} "
              f"layer.active={layer_active!r} {attr}={val} "
              f"sum|lora_B[{name}]|={bsum}")
        if bsum is not None and bsum == 0.0:
            print(f"         >>> lora_B for {name} is ZERO in memory -> the "
                  f"safetensors weights did not bind to this adapter name.")

    print("\n" + "=" * 70)
    print("D. Direct logits comparison (the decisive test)")
    print("=" * 70)
    ids = torch.tensor([[785, 3491, 374, 25]], device=device)  # arbitrary tokens

    @torch.no_grad()
    def last_logits():
        return model(ids).logits[0, -1].float().cpu()

    model.disable_adapter_layers(); model.eval()
    lo_base = last_logits()

    outs = {"base": lo_base}
    for step in STEPS:
        name = f"ckpt_{step}"
        model.set_adapter(name)
        model.enable_adapter_layers(); model.eval()
        outs[name] = last_logits()

    import itertools
    keys = list(outs.keys())
    for a, b in itertools.combinations(keys, 2):
        same = torch.allclose(outs[a], outs[b], atol=1e-6, rtol=0)
        maxdiff = (outs[a] - outs[b]).abs().max().item()
        print(f"  {a:>9} vs {b:<9} identical={same}  max|Δlogit|={maxdiff:.3e}")

    print()
    print("  If base vs ckpt_* max|Δlogit| == 0 -> adapter truly not applied")
    print("    (enable flag ignored at forward, OR lora_B zero from C).")
    print("  If ckpt_20 vs ckpt_360 == 0 but both differ from base -> the")
    print("    same adapter is applied for every name (set_adapter not")
    print("    rebinding weights).")


if __name__ == "__main__":
    main()