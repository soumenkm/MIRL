import numpy as np
d = np.load("exp2/outputs/acts/checkpoint_0.npz", allow_pickle=True)
print({k: d[k].shape if hasattr(d[k], "shape") else d[k] for k in d.files})

# Spot-check one prompt
print("\nFirst prompt:")
print(f"  lang:              {d['prompt_langs'][0]}")
print(f"  dataset index:     {d['prompt_indices'][0]}")
print(f"  CoT length:        {d['cot_lengths'][0]}")
print(f"  top50 shape:       {d['top50_lang_renorm'][0].shape}")
print(f"  top1  shape:       {d['top1_lang_idx'][0].shape}")
print(f"  full  shape:       {d['full_lang_sum'][0].shape}")
print(f"  hidden_mean shape: {d['hidden_mean'][0].shape}")

# Plot 1 sanity: top50 should sum to ~1.0 along the last axis (renormalized)
arr = d['top50_lang_renorm'][0].astype(np.float32)
print(f"\ntop50 row sums (should all be ~1.0 or 0.0):")
print(f"  layer 0, token 0:  {arr[0, 0].sum():.4f}")
print(f"  layer 14, token 0: {arr[14, 0].sum():.4f}")
print(f"  layer 27, token 0: {arr[27, 0].sum():.4f}")

# Plot 3 sanity: full_lang_sum should sum to <= 1.0 along the last axis
arr2 = d['full_lang_sum'][0].astype(np.float32)
print(f"\nfull_lang_sum row sums (should be <= 1.0, with missing mass = punct/special/other):")
print(f"  layer 0, token 0:  {arr2[0, 0].sum():.4f}")
print(f"  layer 27, token 0: {arr2[27, 0].sum():.4f}")

import numpy as np
d = np.load("exp2/outputs/acts/checkpoint_620.npz", allow_pickle=True)
langs = d['prompt_langs']
lens = d['cot_lengths']
for L in ["en", "bn", "te", "th", "ru", "ja", "zh"]:
    mask = langs == L
    print(f"  {L}: n={mask.sum()}, mean_cot_len={lens[mask].mean():.0f}, min={lens[mask].min()}, max={lens[mask].max()}")