#!/usr/bin/env python3
"""
debug_eval_pipeline.py

Diagnose why per-checkpoint eval accuracy is byte-identical across all
GRPO checkpoints. Run this ON THE CLUSTER where the npz / checkpoint /
score files live.

It answers, in order:
  Q1. Are the generated CoTs actually DIFFERENT between checkpoints?
      (If identical -> eval never really swapped the adapter, despite the
       log saying it did. Bug is in eval.py adapter activation.)
  Q2. Are the LoRA adapter weight files on disk actually DIFFERENT between
      checkpoints? (If identical -> training saved the same weights every
      time, or eval loaded the wrong dir.)
  Q3. Do the score files differ, and does the scorer collapse distinct CoTs
      to identical accuracy? (If CoTs differ but scores are identical ->
      bug is in the scorer / caching.)

Nothing here loads the big activation arrays fully; we only touch the
light-weight token/text fields and hash the adapter files.

Usage:
    python debug_eval_pipeline.py \
        --acts_dir   exp3/outputs/acts \
        --ckpt_dir   exp3/outputs/checkpoints \
        --scores_dir exp3/outputs/scores_gpt
(All three have sensible defaults below; edit CONFIG if your paths differ.)
"""

import sys
import json
import hashlib
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# CONFIG — edit these if your layout differs.
# ----------------------------------------------------------------------
CONFIG = {
    "acts_dir":   Path("exp3/outputs/acts"),
    "ckpt_dir":   Path("exp3/outputs/checkpoints"),
    "scores_dir": Path("exp3/outputs/scores_gpt"),
    # Which steps to compare. We deliberately pick a few spread-out ones
    # plus adjacent ones, rather than all 26, to keep output readable.
    "compare_steps": [0, 20, 40, 200, 360],
    # The npz field that holds the generated continuation tokens.
    # eval.py wrote "cot_tokens"; we also try a couple of fallbacks.
    "cot_token_keys": ["cot_tokens", "completion_tokens", "gen_tokens"],
    # How many prompts to spot-check when comparing decoded text.
    "n_prompt_spotcheck": 3,
}


def _hash_array(arr) -> str:
    """Stable hash of a numpy array's raw bytes (shape + dtype + data)."""
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:16]


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _load_npz_lazy(path: Path):
    """Open npz without materialising big arrays. Returns the NpzFile."""
    return np.load(path, allow_pickle=True)


def _find_cot_key(npz, keys):
    for k in keys:
        if k in npz.files:
            return k
    return None


# ----------------------------------------------------------------------
# Q1: do the CoT token arrays differ between checkpoints?
# ----------------------------------------------------------------------
def check_cot_tokens(cfg):
    print("=" * 70)
    print("Q1. Are generated CoTs different across checkpoints?")
    print("=" * 70)
    acts_dir = cfg["acts_dir"]
    steps = cfg["compare_steps"]

    per_step_hashes = {}     # step -> overall hash of the cot token field
    per_step_perprompt = {}  # step -> list of per-row hashes (first N rows)
    cot_key_used = None
    list_field = None        # what npz field enumerates prompts/rows

    for step in steps:
        path = acts_dir / f"checkpoint_{step}.npz"
        if not path.exists():
            print(f"  [step {step}] MISSING: {path}")
            continue
        npz = _load_npz_lazy(path)
        if cot_key_used is None:
            cot_key_used = _find_cot_key(npz, cfg["cot_token_keys"])
            print(f"  npz fields present: {list(npz.files)}")
            print(f"  using CoT token field: {cot_key_used!r}")
        if cot_key_used is None:
            print("  !! Could not find any CoT token field. Inspect fields above.")
            return None

        cot = npz[cot_key_used]
        # cot may be an object array (ragged: one token list per prompt) or a
        # padded 2D array. Handle both.
        overall = _hash_array(cot) if cot.dtype != object else \
            hashlib.sha256(
                b"".join(np.asarray(x).tobytes() for x in cot)
            ).hexdigest()[:16]
        per_step_hashes[step] = overall

        # per-row hashes for the first few prompts
        rows = []
        n = min(cfg["n_prompt_spotcheck"], len(cot))
        for i in range(n):
            row = np.asarray(cot[i])
            rows.append(hashlib.sha256(row.tobytes()).hexdigest()[:12])
        per_step_perprompt[step] = rows
        print(f"  [step {step:>3}] rows={len(cot):>4}  overall_cot_hash={overall}")
        npz.close()

    print()
    distinct = set(per_step_hashes.values())
    if len(distinct) <= 1:
        print("  >>> VERDICT: CoT tokens are IDENTICAL across all checked steps.")
        print("  >>> The adapter was NOT effectively applied during generation,")
        print("  >>> even though eval.log claims it was activated.")
        print("  >>> Fix target: eval.py adapter activation (see Q2 to localise).")
    else:
        print(f"  >>> VERDICT: CoT tokens DIFFER ({len(distinct)} distinct hashes).")
        print("  >>> Generation is checkpoint-dependent; the bug is downstream")
        print("  >>> in scoring/plotting (see Q3).")

    print("\n  Per-prompt spot check (first rows; should vary if CoTs differ):")
    for step in steps:
        if step in per_step_perprompt:
            print(f"    step {step:>3}: {per_step_perprompt[step]}")

    return per_step_hashes


# ----------------------------------------------------------------------
# Q2: do the LoRA adapter weights on disk differ between checkpoints?
# ----------------------------------------------------------------------
def check_adapter_weights(cfg):
    print("\n" + "=" * 70)
    print("Q2. Are the LoRA adapter weights different between checkpoints?")
    print("=" * 70)
    ckpt_dir = cfg["ckpt_dir"]
    steps = [s for s in cfg["compare_steps"] if s != 0]  # 0 = base, no adapter

    weight_names = ["adapter_model.safetensors", "adapter_model.bin"]
    hashes = {}
    for step in steps:
        d = ckpt_dir / f"checkpoint-{step}"
        if not d.exists():
            print(f"  [step {step}] MISSING dir: {d}")
            continue
        wpath = None
        for wn in weight_names:
            if (d / wn).exists():
                wpath = d / wn
                break
        if wpath is None:
            print(f"  [step {step}] no adapter weight file in {d}")
            print(f"             dir contents: {[p.name for p in d.iterdir()]}")
            continue
        h = _hash_file(wpath)
        size = wpath.stat().st_size
        hashes[step] = h
        print(f"  [step {step:>3}] {wpath.name}  size={size:>12,}  sha={h}")

    print()
    distinct = set(hashes.values())
    if len(hashes) == 0:
        print("  >>> No adapter weights found — check ckpt_dir path / layout.")
    elif len(distinct) <= 1:
        print("  >>> VERDICT: adapter weight files are IDENTICAL across steps.")
        print("  >>> Either training re-saved the same weights (LoRA not")
        print("  >>> updating) or all checkpoints point to the same data.")
    else:
        print(f"  >>> VERDICT: adapter weights DIFFER ({len(distinct)} distinct).")
        print("  >>> Training did update the adapter and saved distinct copies.")
        print("  >>> So if Q1 said CoTs are identical, eval is loading these")
        print("  >>> distinct weights but NOT applying them at generation time.")
    return hashes


# ----------------------------------------------------------------------
# Q2b: sanity — is the adapter delta actually nonzero vs base?
# ----------------------------------------------------------------------
def check_adapter_norm(cfg):
    print("\n" + "=" * 70)
    print("Q2b. Are LoRA B-matrices nonzero? (zero B => adapter is a no-op)")
    print("=" * 70)
    try:
        from safetensors import safe_open
    except Exception as e:
        print(f"  (safetensors not importable: {e}; skipping)")
        return
    ckpt_dir = cfg["ckpt_dir"]
    steps = [s for s in cfg["compare_steps"] if s != 0]
    for step in steps[:2]:  # two is enough
        d = ckpt_dir / f"checkpoint-{step}"
        wpath = d / "adapter_model.safetensors"
        if not wpath.exists():
            continue
        total = 0.0
        n_b = 0
        with safe_open(str(wpath), framework="numpy") as f:
            for key in f.keys():
                if "lora_B" in key:
                    t = f.get_tensor(key)
                    total += float(np.abs(t).sum())
                    n_b += 1
        print(f"  [step {step:>3}] sum|lora_B| over {n_b} matrices = {total:.6e}")
        if total == 0.0:
            print("           >>> ALL lora_B ARE ZERO -> adapter does nothing.")
            print("           >>> RL never moved the adapter (or wrong file).")
        else:
            print("           >>> nonzero -> adapter has real, applied-able weights.")


# ----------------------------------------------------------------------
# Q3: do score files differ, and decoded-text spot check
# ----------------------------------------------------------------------
def check_score_files(cfg):
    print("\n" + "=" * 70)
    print("Q3. Score files: do per-prompt CoT texts / scores differ?")
    print("=" * 70)
    sdir = cfg["scores_dir"]
    steps = cfg["compare_steps"]

    def load_scores(step):
        p = sdir / f"accuracy_step_{step}.json"
        if not p.exists():
            return None, p
        with open(p) as f:
            return json.load(f), p

    # Compare per-language accuracy and a few cot_text fields.
    prev_texts = None
    for step in steps:
        data, p = load_scores(step)
        if data is None:
            print(f"  [step {step}] missing score file: {p}")
            continue
        pla = data.get("per_language_accuracy", {})
        recs = data.get("records", data.get("per_prompt", []))
        # Hash the concatenation of cot_texts to see if generations differ
        texts = [r.get("cot_text", "") for r in recs] if recs else []
        thash = hashlib.sha256(
            "\u241f".join(texts).encode("utf-8", "ignore")
        ).hexdigest()[:16] if texts else "NA"
        print(f"  [step {step:>3}] acc={ {k: round(v,3) for k,v in pla.items()} }")
        print(f"            n_records={len(recs)}  cot_text_concat_hash={thash}")
        if prev_texts is not None and texts and prev_texts == texts:
            print("            >>> cot_texts IDENTICAL to previous checked step.")
        prev_texts = texts if texts else prev_texts

    print()
    print("  Interpretation:")
    print("   - If cot_text hashes are identical across steps -> generations")
    print("     never changed (matches a Q1 'identical' verdict).")
    print("   - If cot_text hashes DIFFER but per_language_accuracy is identical")
    print("     -> the scorer/extraction is the culprit (or cached score files).")


def main():
    cfg = dict(CONFIG)
    # crude arg override: --key value
    args = sys.argv[1:]
    for i in range(0, len(args) - 1, 2):
        key = args[i].lstrip("-")
        if key in cfg:
            cfg[key] = Path(args[i + 1]) if "dir" in key else args[i + 1]

    print("CONFIG:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print()

    cot_hashes = check_cot_tokens(cfg)
    check_adapter_weights(cfg)
    check_adapter_norm(cfg)
    check_score_files(cfg)

    print("\n" + "=" * 70)
    print("SUMMARY DECISION TREE")
    print("=" * 70)
    print("  Q1 identical + Q2 differ + Q2b nonzero  -> eval loads adapter but")
    print("      does not apply it at generate() time. Fix eval.py activation.")
    print("  Q1 identical + Q2 identical             -> training saved same")
    print("      weights every checkpoint. Fix train.py checkpoint saving.")
    print("  Q1 differ   + Q3 acc identical          -> scorer/caching bug.")
    print("      Fix plot_accuracy_regex.py (delete cached scores_gpt/*.json).")


if __name__ == "__main__":
    main()