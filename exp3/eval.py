"""Single-pass evaluation that produces all four plot quantities.

For each selected checkpoint and each eval prompt, this script:

  1. Greedy-generates the chain-of-thought continuation (up to max_new_tokens,
     stopping at EOS or when a "Final Answer: <num>" pattern appears).
  2. Runs ONE forward pass on (prompt + CoT) with output_hidden_states=True
     and slices off the prompt-position hidden states. Only the CoT-position
     hidden states are retained.
  3. From those hidden states, derives four small per-prompt quantities:

         Q1 (Plot 1) -- top50_lang_renorm: (L, C, 7) fp16
              Top-50 logit-lens probs grouped by language and renormalised
              over the 7 studied languages. Plot 1 averages over k (CoT
              positions) and p (prompts) at plot time.

         Q2 (Plot 2) -- top1_lang_idx: (L, C) int8
              Language label index (0..9) of the argmax token per cell.
              Plot 2 builds phase-averaged indicators at plot time.

         Q3 (Plot 3) -- full_lang_sum: (L, C, 7) fp16
              Full-vocab probabilities scatter-summed by language. NOT
              renormalised; the missing mass is in punct/special/other.
              Plot 3 computes p_target / (p_en + p_target) at plot time.

         Q4 (Plot 4) -- hidden_mean: (L, d) fp16
              Per-layer hidden state averaged over CoT positions. Plot 4
              stacks across prompts and computes CKA against English at
              plot time.

  4. Saves one .npz file per checkpoint containing all four quantities for
     all evaluated prompts, plus prompt metadata and token IDs.

The token-to-language map is loaded once from token_language_map.json (see
token_classifier.py). The map is keyed by token-id-as-string and the value
is one of: en, bn, te, th, ru, ja, zh, punct_num, special, other.

Storage: with the default config (8 checkpoints, ~250 prompts), output is
about 1.5 GB total. All four plot scripts read these .npz files and never
need to re-run the model.

IMPORTANT (checkpoint layout): the policy was trained with TRL's GRPOTrainer
(train.py). TRL/HF Trainer save checkpoints as ``checkpoint-<N>/`` directories
containing the LoRA adapter files DIRECTLY (adapter_config.json,
adapter_model.safetensors) -- NOT the old hand-rolled layout
(``step_<N>/adapter/`` + train_state.pt) that GRPOModelManager.load_checkpoint
expects. This script therefore discovers and loads checkpoints itself rather
than delegating to GRPOModelManager.load_checkpoint.

IMPORTANT (prompt format): the policy was trained on CHAT-TEMPLATED prompts
(system + user turns via tokenizer.apply_chat_template). Evaluation MUST use
the same format or the activations are not representative of the trained
policy (and generation may not terminate). This script builds the chat
prompt from the dataset's "system"/"user" fields, exactly like train.py.

Standards: see project coding standards. All paths are pathlib.Path objects.
Logging to file + console. tqdm on every iteration over data. No os module,
no argparse, no print, no top-level code outside main()/if __name__.
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm

from grpo_dataset import GRPOMGSMDataset
from models import GRPOModelManager


class EvalActivationsExtractor:
    """Runs greedy CoT generation + single-pass activation extraction.

    Produces one .npz per checkpoint in output_dir/acts/. The .npz contains
    per-prompt arrays for all four plot quantities plus metadata.
    """

    # The seven studied languages, fixed order. This order is the column
    # axis of all per-language stored arrays.
    LANG_LABELS = ("en", "bn", "te", "th", "ru", "ja", "zh")

    # All ten classifier label classes, fixed order. Indices 0..6 = languages,
    # 7 = punct_num, 8 = special, 9 = other. This order defines the integer
    # code stored in top1_lang_idx.
    ALL_LABELS = LANG_LABELS + ("punct_num", "special", "other")

    # Regex that signals the model has emitted a final answer; we stop
    # generation early once this pattern fires plus a newline. Mirrors the
    # extractor in rewards.py.
    FINAL_ANSWER_PATTERN = re.compile(r"Final Answer\s*:\s*[+-]?\d[\d,]*")

    # Checkpoint directory prefix written by TRL / HF Trainer.
    CHECKPOINT_PREFIX = "checkpoint-"

    def __init__(self, config: dict):
        self.log_dir = Path(config["log_dir"])
        self.output_dir = Path(config["output_dir"])
        self.data_dir = Path(config["data_dir"])
        self.acts_dir = self.output_dir / "acts"

        # Eval / generation knobs
        self.max_new_tokens = int(config["max_new_tokens"])
        self.top_k = int(config.get("top_k", 50))
        self.prompt_fraction = float(config.get("prompt_fraction", 1.0))
        self.num_checkpoints = int(config["num_checkpoints"])
        self.subset_seed = int(config.get("subset_seed", 42))

        # Whether to render prompts through the chat template. MUST match how
        # the policy was trained (train.py uses chat templating for
        # Qwen2.5-Instruct). Auto-disabled if the tokenizer has no template.
        self.use_chat_template = bool(config.get("use_chat_template", True))

        # Stop-early heuristic: how many tokens to allow after Final Answer
        # before forcing a stop. Mirrors GRPOModelManager.tokens_after_final_answer.
        self.tokens_after_final_answer = int(
            config.get("tokens_after_final_answer", 20)
        )

        if not (0.0 < self.prompt_fraction <= 1.0):
            raise ValueError(
                f"prompt_fraction must be in (0, 1], got {self.prompt_fraction}"
            )
        if self.num_checkpoints < 1:
            raise ValueError(f"num_checkpoints must be >= 1, got {self.num_checkpoints}")

        # Paths to sub-configs we will pass on to GRPOModelManager and
        # GRPOMGSMDataset. eval.py reuses both classes verbatim.
        self.model_config = dict(config["model_config"])
        self.dataset_config = dict(config["dataset_config"])

        # token-to-language map (built earlier by token_classifier.py).
        self.token_map_path = Path(config["token_language_map_path"])

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.acts_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        # Load token-language map; convert to a dense int8 array of shape
        # (vocab_size,) whose value is the ALL_LABELS index 0..9. The dense
        # array is the fast path for the scatter-sum operations below.
        self.token_lang_codes = self._load_token_lang_codes()

        # Build model and dataset.
        self.logger.info("Initialising model manager (this also loads the policy)...")
        self.model_mgr = GRPOModelManager(self.model_config)
        self.policy_device = self.model_mgr.policy_device
        self.tokenizer = self.model_mgr.tokenizer
        self.policy_model = self.model_mgr.policy_model

        if self.use_chat_template and self.tokenizer.chat_template is None:
            self.logger.warning(
                "use_chat_template=True but tokenizer has no chat_template; "
                "falling back to raw prompts."
            )
            self.use_chat_template = False
        self.logger.info(f"Prompt formatting: chat_template={self.use_chat_template}")

        # Checkpoint directory (TRL layout: checkpoint-<N>/ with adapter files).
        self.checkpoint_dir = Path(self.model_config["checkpoint_dir"])

        # Align token_lang_codes to the model's lm_head output size, which
        # may be larger than the tokenizer's named vocabulary. Qwen2.5 (and
        # many Llama-family models) pad lm_head.out_features up to a
        # multiple of 64 for efficient matmul kernels. Concretely for
        # Qwen2.5-7B-Instruct:
        #
        #   tokenizer.vocab_size      = 151643  (canonical named tokens)
        #   len(tokenizer)            = 151665  (includes added special ids)
        #   lm_head.out_features      = 152064  (padded to multiple of 64)
        #
        # token_classifier.py iterates over len(tokenizer), so its map
        # covers 151665 entries. The logit-lens probs vector V we will
        # softmax has length 152064. To scatter-sum by language we need a
        # codes array of length exactly V. We extend the existing codes
        # with 'other' for the padding rows: those token ids are unused
        # vocabulary slots, so any (tiny) probability mass landing there
        # safely falls into the bucket that Plot 3 / Plot 1 already drop.
        lm_head_out = int(self._lm_head_out_features())
        codes_len = int(len(self.token_lang_codes))
        if lm_head_out > codes_len:
            other_code = self.ALL_LABELS.index("other")
            pad = np.full(lm_head_out - codes_len, other_code, dtype=np.int8)
            self.token_lang_codes = np.concatenate([self.token_lang_codes, pad])
            self.logger.info(
                f"Padded token_lang_codes {codes_len} -> {lm_head_out} "
                f"(lm_head out_features); pad rows -> 'other'."
            )
        elif lm_head_out < codes_len:
            # Should not happen with current Qwen2.5 release, but truncate
            # rather than risk a silent shape mismatch.
            self.token_lang_codes = self.token_lang_codes[:lm_head_out]
            self.logger.warning(
                f"Truncated token_lang_codes {codes_len} -> {lm_head_out}."
            )
        else:
            self.logger.info(
                f"token_lang_codes already matches lm_head out_features ({lm_head_out})."
            )

        self.logger.info("Initialising eval dataset (test split)...")
        self.test_dataset = GRPOMGSMDataset(self.dataset_config, split="test")
        self.languages = list(self.test_dataset.languages)

        # Build the list of (prompt_idx, lang) pairs for the eval run,
        # optionally subset by prompt_fraction.
        self.eval_indices = self._build_eval_indices()
        self.logger.info(
            f"Evaluating {len(self.eval_indices)} prompts "
            f"(fraction={self.prompt_fraction}, langs={self.languages})"
        )

        # Resolve which checkpoint steps to evaluate.
        self.checkpoint_steps = self._resolve_checkpoint_steps()
        self.logger.info(
            f"Will evaluate {len(self.checkpoint_steps)} checkpoints: "
            f"{self.checkpoint_steps}"
        )

        # Cache the final RMSNorm and lm_head weights for logit-lens
        # projection. Doing this outside the inner loop avoids re-resolving
        # the modules every time.
        base = self.policy_model.get_base_model()
        self.final_norm = base.model.norm
        self.lm_head = base.lm_head

    def _lm_head_out_features(self) -> int:
        """Out-features of the policy lm_head, == logit-lens vocab size V.

        Called during __init__ before self.lm_head is cached, so we resolve
        the lm_head module fresh from the base model. lm_head is an
        nn.Linear so .out_features is canonical; .weight.shape[0] is the
        fallback if a custom layer is used.
        """
        base = self.policy_model.get_base_model()
        head = base.lm_head
        if hasattr(head, "out_features"):
            return int(head.out_features)
        return int(head.weight.shape[0])

    def _setup_logging(self):
        log_file = self.log_dir / "eval.log"
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fh = logging.FileHandler(log_file, mode="w")
            fh.setFormatter(fmt)
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            self.logger.addHandler(fh)
            self.logger.addHandler(sh)
        self.logger.info(f"Logging to {log_file}")

    # ------------------------------------------------------------------ #
    #  Setup helpers
    # ------------------------------------------------------------------ #

    def _load_token_lang_codes(self) -> np.ndarray:
        """Load token_language_map.json into a dense int8 array.

        Returns a (vocab_size,) array where entry t is the index in
        ALL_LABELS of the language label assigned to token id t.
        Tokens not in the map get the 'other' code.
        """
        self.logger.info(f"Loading token-language map from {self.token_map_path}")
        with self.token_map_path.open() as f:
            mapping = json.load(f)

        label_to_code = {label: i for i, label in enumerate(self.ALL_LABELS)}
        other_code = label_to_code["other"]

        max_id = max(int(k) for k in mapping.keys())
        codes = np.full(max_id + 1, other_code, dtype=np.int8)
        for k, v in mapping.items():
            codes[int(k)] = label_to_code.get(v, other_code)
        self.logger.info(
            f"Token-language map loaded: {len(mapping)} entries, "
            f"max token id = {max_id}"
        )
        return codes

    def _build_chat_prompt(self, item: dict) -> str:
        """Render the eval prompt EXACTLY as train.py did.

        Training used tokenizer.apply_chat_template with a system turn (the
        formatting instruction) and a user turn (few-shot + question), with
        add_generation_prompt=True. We reproduce that here so the activations
        reflect the trained policy's real behaviour and generation terminates
        on <|im_end|>. Falls back to the raw text prompt if chat templating
        is disabled/unavailable.
        """
        if self.use_chat_template:
            messages = [
                {"role": "system", "content": item["system"]},
                {"role": "user", "content": item["user"]},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return item["prompt"]

    def _build_eval_indices(self) -> list[tuple[int, str]]:
        """Pick a deterministic subset of eval prompts, balanced per-language.

        Returns a list of (dataset_index, language) tuples. If
        prompt_fraction < 1.0, we pick the first round(fraction * n_l) prompts
        of each language l (after a per-language shuffle seeded by
        subset_seed). This guarantees every language has representatives.
        """
        by_lang: dict[str, list[int]] = {l: [] for l in self.languages}
        for idx in range(len(self.test_dataset)):
            item = self.test_dataset[idx]
            lang = item["lang"]
            if lang in by_lang:
                by_lang[lang].append(idx)

        result: list[tuple[int, str]] = []
        rng = np.random.default_rng(self.subset_seed)
        for lang in self.languages:
            ids = list(by_lang[lang])
            rng.shuffle(ids)
            keep_n = max(1, int(round(self.prompt_fraction * len(ids))))
            for i in ids[:keep_n]:
                result.append((i, lang))
            self.logger.info(
                f"  lang={lang}: {len(ids)} total, {keep_n} kept"
            )
        return result

    def _resolve_checkpoint_steps(self) -> list[int]:
        """Choose checkpoints to evaluate via linspace over saved steps.

        Step 0 is reserved for the base model (LoRA disabled at inference
        time). The linspace is over [0, max_step] where max_step is the
        largest saved checkpoint on disk.

        Each chosen step is snapped to the nearest *actually saved*
        checkpoint, or to step 0 if it falls below the smallest saved one.

        Checkpoints are TRL/HF-style ``checkpoint-<N>`` directories.
        """
        saved_steps: list[int] = []
        if self.checkpoint_dir.exists():
            for child in self.checkpoint_dir.iterdir():
                if child.is_dir() and child.name.startswith(self.CHECKPOINT_PREFIX):
                    suffix = child.name[len(self.CHECKPOINT_PREFIX):]
                    try:
                        saved_steps.append(int(suffix))
                    except ValueError:
                        continue
        saved_steps.sort()
        if not saved_steps:
            self.logger.warning(
                f"No '{self.CHECKPOINT_PREFIX}<N>' checkpoints found in "
                f"{self.checkpoint_dir}. Will evaluate only the base model "
                f"(step 0)."
            )
            return [0]

        self.logger.info(
            f"Found {len(saved_steps)} saved checkpoints "
            f"(steps {saved_steps[0]}..{saved_steps[-1]})."
        )

        max_step = saved_steps[-1]
        targets = np.linspace(0, max_step, self.num_checkpoints).round().astype(int)
        chosen: list[int] = []
        for t in targets:
            if t <= 0:
                chosen.append(0)
            else:
                # Snap to nearest available saved step.
                snapped = min(saved_steps, key=lambda s: abs(s - int(t)))
                chosen.append(int(snapped))
        # Deduplicate while preserving order.
        seen = set()
        deduped: list[int] = []
        for s in chosen:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        return deduped

    # ------------------------------------------------------------------ #
    #  Checkpoint loading
    # ------------------------------------------------------------------ #

    def _checkpoint_path(self, step: int) -> Path:
        """Path to the TRL checkpoint dir for a given step."""
        return self.checkpoint_dir / f"{self.CHECKPOINT_PREFIX}{step}"

    def _activate_checkpoint(self, step: int):
        """Put the policy model into the state of checkpoint `step`.

        step == 0 -> base model, LoRA disabled (no adapter applied).
        step >  0 -> the saved adapter at <checkpoint_dir>/checkpoint-<N>/.

        IMPORTANT — why this is written with named adapter slots rather than
        rebuilding the PeftModel each step:

        The previous implementation called ``disable_adapter_layers()`` for
        step 0 and then, for each later step, unwrapped the base model with
        ``get_base_model()`` and built a brand-new ``PeftModel.from_pretrained``
        around it. That left the adapter layers in a *disabled* state on the
        shared base modules (the step-0 disable was never undone), so every
        subsequent generation ran through the bare base model. The result:
        byte-identical CoTs for every checkpoint, and a perfectly flat
        accuracy curve that looked like "RL had no effect" but was really the
        base model evaluated 26 times.

        The robust pattern is to keep ONE PeftModel for the whole run, load
        each checkpoint's weights into it as a named adapter once, and switch
        between them with ``set_adapter`` (which selects AND enables) /
        ``enable_adapter_layers`` / ``disable_adapter_layers``. We never touch
        the 7B base weights and we can never leak a disabled state forward,
        because every step>0 explicitly re-enables the layers.

        TRL/HF Trainer save the LoRA adapter directly inside
        ``checkpoint-<N>/`` (adapter_config.json + adapter_model.safetensors),
        with no ``adapter/`` subfolder and no train_state.pt.
        """
        if step == 0:
            self.logger.info("Activating base model (LoRA disabled)")
            self.policy_model.disable_adapter_layers()
            self.policy_model.eval()
            return

        ckpt = self._checkpoint_path(step)
        adapter_cfg = ckpt / "adapter_config.json"
        if not adapter_cfg.exists():
            raise FileNotFoundError(
                f"Adapter config not found at {adapter_cfg}. Expected a TRL "
                f"checkpoint directory '{ckpt}' containing adapter_config.json "
                f"and adapter_model.safetensors."
            )

        adapter_name = f"ckpt_{step}"
        self.logger.info(f"Activating checkpoint step={step} from {ckpt}")

        # Load this checkpoint's weights as a named adapter exactly once.
        # PeftModel tracks loaded adapters in .peft_config; reuse if present.
        if adapter_name not in self.policy_model.peft_config:
            self.policy_model.load_adapter(
                str(ckpt),
                adapter_name=adapter_name,
                is_trainable=False,
            )

        # set_adapter selects this adapter as active. Crucially we then
        # ENABLE the adapter layers — this undoes any earlier step-0 disable
        # and is the line whose absence caused the flat-curve bug.
        self.policy_model.set_adapter(adapter_name)
        self.policy_model.enable_adapter_layers()
        self.policy_model.eval()

        # Sanity assertion: confirm the active adapter is what we asked for.
        active = getattr(self.policy_model, "active_adapter", None)
        if active != adapter_name:
            self.logger.warning(
                f"Active adapter is {active!r}, expected {adapter_name!r}."
            )

        # Refresh cached module references (base is unchanged, but the
        # lm_head/norm handles are cheap to re-resolve and keeps us safe if a
        # future PEFT version reparents modules).
        base = self.policy_model.get_base_model()
        self.final_norm = base.model.norm
        self.lm_head = base.lm_head

    # ------------------------------------------------------------------ #
    #  Per-prompt processing
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _generate_cot(
        self, prompt_text: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy-generate the CoT continuation.

        Returns:
            prompt_ids:    (P,) int64 on policy_device — the encoded prompt
            cot_ids:       (C,) int64 on policy_device — generated tokens only

        Stops on:
          - EOS token (including Qwen's <|im_end|>), or
          - max_new_tokens reached, or
          - "Final Answer: <num>" detected, after which we allow
            tokens_after_final_answer more tokens, then stop.
        """
        self.policy_model.eval()
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=False,
        ).to(self.policy_device)
        prompt_ids = encoded["input_ids"][0]
        prompt_attn = encoded["attention_mask"]

        # We use HF generate() with a custom stopping criterion to honour
        # the Final-Answer-then-N-more-tokens rule. The detection is done by
        # decoding the running suffix and regex-matching, similar to the
        # FinalAnswerStoppingCriteria already used by GRPOModelManager.
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _FinalAnswerCriteria(StoppingCriteria):
            def __init__(self, tokenizer, prompt_len, pattern, allowance):
                super().__init__()
                self.tokenizer = tokenizer
                self.prompt_len = prompt_len
                self.pattern = pattern
                self.allowance = allowance
                self.match_at = None

            def __call__(self, input_ids, scores, **kwargs):
                generated = input_ids[0, self.prompt_len:]
                if generated.numel() == 0:
                    return False
                # Only re-decode every few tokens to keep this cheap. We
                # check on every 4th token after generation length 32.
                gl = int(generated.numel())
                if gl < 32 or (self.match_at is None and gl % 4 != 0):
                    return False
                if self.match_at is None:
                    text = self.tokenizer.decode(
                        generated, skip_special_tokens=True
                    )
                    if self.pattern.search(text):
                        self.match_at = gl
                if self.match_at is not None:
                    return gl >= self.match_at + self.allowance
                return False

        stopping = StoppingCriteriaList([
            _FinalAnswerCriteria(
                tokenizer=self.tokenizer,
                prompt_len=int(prompt_ids.numel()),
                pattern=self.FINAL_ANSWER_PATTERN,
                allowance=self.tokens_after_final_answer,
            )
        ])

        # EOS ids: include Qwen's <|im_end|> turn token alongside the default
        # eos so chat-formatted generations terminate naturally (mirrors
        # train.py's generation_kwargs).
        eos_ids = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(self.tokenizer.eos_token_id)
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if im_end is not None and im_end >= 0 and im_end != unk_id and im_end not in eos_ids:
            eos_ids.append(im_end)

        out = self.policy_model.generate(
            input_ids=prompt_ids.unsqueeze(0),
            attention_mask=prompt_attn,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_ids if eos_ids else self.tokenizer.eos_token_id,
            stopping_criteria=stopping,
            return_dict_in_generate=True,
            output_scores=False,
            output_hidden_states=False,
        )
        seq = out.sequences[0]  # shape (P + C,)
        cot_ids = seq[prompt_ids.numel():]
        return prompt_ids, cot_ids

    @torch.no_grad()
    def _forward_for_hidden_states(
        self, prompt_ids: torch.Tensor, cot_ids: torch.Tensor
    ) -> torch.Tensor:
        """Single forward pass over (prompt + cot) -> CoT-position hidden states.

        Returns:
            hidden: (L, C, d) tensor on policy_device, fp16 dtype as
                    returned by the model. L = num transformer layers
                    (excludes embedding).

        We feed prompt + CoT through the model in eval mode with
        output_hidden_states=True. HF returns a tuple of length L+1:
        index 0 is the embedding output, indices 1..L are the post-block
        residuals. We drop index 0 (embedding) and slice all the rest to
        keep only the CoT positions [prompt_len : prompt_len + C].
        """
        self.policy_model.eval()
        prompt_len = int(prompt_ids.numel())
        cot_len = int(cot_ids.numel())
        full = torch.cat([prompt_ids, cot_ids]).unsqueeze(0)

        outputs = self.policy_model(
            input_ids=full,
            output_hidden_states=True,
            use_cache=False,
        )
        # hidden_states is a tuple of (L+1) tensors, each (1, P+C, d).
        # Drop the embedding output (index 0), slice CoT positions.
        hs = outputs.hidden_states
        slc_start = prompt_len
        slc_end = prompt_len + cot_len
        stacked = torch.stack(
            [h[0, slc_start:slc_end, :] for h in hs[1:]], dim=0
        )  # (L, C, d)
        return stacked

    @torch.no_grad()
    def _logit_lens_probs(self, hidden_LCD: torch.Tensor) -> torch.Tensor:
        """Project hidden states through final-norm + lm_head, softmax.

        hidden_LCD: (L, C, d)
        Returns:    (L, C, V) on policy_device, fp16

        Applies the model's final RMSNorm at every layer (standard logit-
        lens practice; otherwise mid-layer scales are off and the softmax
        produces near-uniform garbage).
        """
        L, C, D = hidden_LCD.shape
        # final_norm and lm_head operate on the last dim. Flatten (L, C) to
        # one batch dim to keep the call shape clean.
        flat = hidden_LCD.reshape(L * C, D)
        normed = self.final_norm(flat)
        logits = self.lm_head(normed)  # (L*C, V)
        probs = torch.softmax(logits.float(), dim=-1).to(torch.float16)
        return probs.reshape(L, C, -1)

    # ------------------------------------------------------------------ #
    #  Derived quantities from per-token full-vocab probs
    # ------------------------------------------------------------------ #

    def _compute_q1_top50_renorm(self, probs_LCV: torch.Tensor) -> np.ndarray:
        """Plot 1 quantity: top-K-grouped renormalised per (L, C, 7).

        For each (i, k):
          - Take top-K=50 indices and their probabilities.
          - Look up each top-K token's language code (0..9).
          - Scatter-sum probs into 10 buckets, then keep only the 7
            language buckets and renormalise to sum to 1.
            (If those 7 sum to 0 -- e.g. all 50 are punctuation -- we leave
            them at 0; division by zero would produce NaN.)
        """
        L, C, V = probs_LCV.shape
        topk_vals, topk_idx = torch.topk(probs_LCV, k=self.top_k, dim=-1)
        # Move to CPU for the language lookup; the gather is the only place
        # we hit the python-level map.
        topk_idx_cpu = topk_idx.cpu().numpy()  # (L, C, K)
        topk_vals_cpu = topk_vals.cpu().numpy().astype(np.float32)  # (L, C, K)
        codes = self.token_lang_codes[topk_idx_cpu]  # (L, C, K) int8

        # Scatter-sum into 10 buckets per (L, C).
        buckets = np.zeros((L, C, len(self.ALL_LABELS)), dtype=np.float32)
        for c in range(len(self.ALL_LABELS)):
            mask = (codes == c).astype(np.float32)
            buckets[..., c] = (topk_vals_cpu * mask).sum(axis=-1)

        # Keep only the first 7 (the studied languages), renormalise.
        langs = buckets[..., : len(self.LANG_LABELS)]
        totals = langs.sum(axis=-1, keepdims=True)
        safe = np.where(totals > 0, totals, 1.0)
        renormed = (langs / safe).astype(np.float16)
        # Where totals was 0, zero out the result explicitly.
        zero_mask = (totals == 0).squeeze(-1)
        if zero_mask.any():
            renormed[zero_mask] = 0
        return renormed  # (L, C, 7) fp16

    def _compute_q2_top1_lang_idx(self, probs_LCV: torch.Tensor) -> np.ndarray:
        """Plot 2 quantity: language code of the argmax token per (L, C).

        Returns (L, C) int8 array of ALL_LABELS indices (0..9).
        """
        top1 = probs_LCV.argmax(dim=-1).cpu().numpy()  # (L, C)
        return self.token_lang_codes[top1].astype(np.int8)

    def _compute_q3_full_lang_sum(self, probs_LCV: torch.Tensor) -> np.ndarray:
        """Plot 3 quantity: full-vocab probs scatter-summed by language.

        Returns (L, C, 7) fp16. NOT renormalised; the 7 entries sum to <= 1
        with the remaining mass spread over punct/special/other (which
        Plot 3 ignores).

        We do the scatter-add on GPU for speed: for each of the 10 codes,
        select the columns of probs whose token id has that code, sum.
        Only the first 7 codes are kept in the output.
        """
        L, C, V = probs_LCV.shape
        device = probs_LCV.device
        codes_t = torch.from_numpy(self.token_lang_codes).to(device).long()
        # codes_t has shape (V,); we want a (V, 10) one-hot membership, then
        # probs_LCV @ membership = (L, C, 10).
        # one_hot is large (V * 10 fp16 ~ 3 MB for V=152K), but fine.
        n_buckets = len(self.ALL_LABELS)
        # Use bool then cast to fp16; matmul against fp16 probs.
        membership = torch.zeros((V, n_buckets), dtype=torch.float16, device=device)
        membership[torch.arange(V, device=device), codes_t] = 1.0
        bucket_sums = probs_LCV @ membership  # (L, C, 10) fp16
        return bucket_sums[..., : len(self.LANG_LABELS)].cpu().numpy().astype(np.float16)

    def _compute_q4_hidden_mean(self, hidden_LCD: torch.Tensor) -> np.ndarray:
        """Plot 4 quantity: per-layer hidden state averaged over CoT tokens.

        Returns (L, d) fp16 on CPU. Mean-centering is deferred to plot4
        because it depends on the full set of N prompts in a language.
        """
        return hidden_LCD.mean(dim=1).cpu().numpy().astype(np.float16)

    # ------------------------------------------------------------------ #
    #  Per-checkpoint processing
    # ------------------------------------------------------------------ #

    def _process_one_checkpoint(self, step: int):
        """Generate, forward, derive, and save one .npz for this checkpoint."""
        out_path = self.acts_dir / f"checkpoint_{step}.npz"
        if out_path.exists():
            self.logger.info(f"Skipping step={step}: {out_path} already exists.")
            return

        self._activate_checkpoint(step)

        n_prompts = len(self.eval_indices)
        prompt_indices = np.zeros(n_prompts, dtype=np.int64)
        prompt_langs = np.empty(n_prompts, dtype=object)
        cot_lengths = np.zeros(n_prompts, dtype=np.int32)
        cot_tokens_list: list[np.ndarray] = []
        top50_list: list[np.ndarray] = []
        top1_list: list[np.ndarray] = []
        full_list: list[np.ndarray] = []
        hidden_mean_all: list[np.ndarray] = []

        n_layers_seen = None
        hidden_dim_seen = None

        for i, (ds_idx, lang) in enumerate(tqdm(
            self.eval_indices,
            desc=f"step={step}",
            unit="prompt",
        )):
            item = self.test_dataset[ds_idx]
            prompt_text = self._build_chat_prompt(item)

            try:
                prompt_ids, cot_ids = self._generate_cot(prompt_text)
                if cot_ids.numel() == 0:
                    self.logger.warning(
                        f"prompt {ds_idx} ({lang}): empty CoT; skipping."
                    )
                    cot_lengths[i] = 0
                    prompt_indices[i] = ds_idx
                    prompt_langs[i] = lang
                    cot_tokens_list.append(np.zeros(0, dtype=np.int32))
                    # Placeholder zeros so list-aligned indexing still works.
                    if n_layers_seen is not None:
                        top50_list.append(np.zeros((n_layers_seen, 0, 7), dtype=np.float16))
                        top1_list.append(np.zeros((n_layers_seen, 0), dtype=np.int8))
                        full_list.append(np.zeros((n_layers_seen, 0, 7), dtype=np.float16))
                        hidden_mean_all.append(
                            np.zeros((n_layers_seen, hidden_dim_seen), dtype=np.float16)
                        )
                    else:
                        # We don't yet know L, d. Defer: fill later.
                        top50_list.append(None)
                        top1_list.append(None)
                        full_list.append(None)
                        hidden_mean_all.append(None)
                    continue

                hidden_LCD = self._forward_for_hidden_states(prompt_ids, cot_ids)
                if n_layers_seen is None:
                    n_layers_seen = int(hidden_LCD.shape[0])
                    hidden_dim_seen = int(hidden_LCD.shape[2])
                    self.logger.info(
                        f"L={n_layers_seen}, d={hidden_dim_seen}"
                    )

                probs_LCV = self._logit_lens_probs(hidden_LCD)
                top50_q1 = self._compute_q1_top50_renorm(probs_LCV)
                top1_q2 = self._compute_q2_top1_lang_idx(probs_LCV)
                full_q3 = self._compute_q3_full_lang_sum(probs_LCV)
                hidden_q4 = self._compute_q4_hidden_mean(hidden_LCD)

                del probs_LCV, hidden_LCD
                torch.cuda.empty_cache()

                prompt_indices[i] = ds_idx
                prompt_langs[i] = lang
                cot_lengths[i] = int(cot_ids.numel())
                cot_tokens_list.append(cot_ids.cpu().numpy().astype(np.int32))
                top50_list.append(top50_q1)
                top1_list.append(top1_q2)
                full_list.append(full_q3)
                hidden_mean_all.append(hidden_q4)

            except torch.cuda.OutOfMemoryError as e:
                self.logger.error(
                    f"OOM on prompt {ds_idx} ({lang}) at step={step}: {e}"
                )
                torch.cuda.empty_cache()
                cot_lengths[i] = 0
                prompt_indices[i] = ds_idx
                prompt_langs[i] = lang
                cot_tokens_list.append(np.zeros(0, dtype=np.int32))
                top50_list.append(None)
                top1_list.append(None)
                full_list.append(None)
                hidden_mean_all.append(None)

        # Backfill any None placeholders now that we know shapes.
        if n_layers_seen is None:
            self.logger.error(
                f"step={step}: no prompt produced any CoT; nothing to save."
            )
            return
        for i in range(n_prompts):
            if top50_list[i] is None:
                top50_list[i] = np.zeros((n_layers_seen, 0, 7), dtype=np.float16)
                top1_list[i] = np.zeros((n_layers_seen, 0), dtype=np.int8)
                full_list[i] = np.zeros((n_layers_seen, 0, 7), dtype=np.float16)
                hidden_mean_all[i] = np.zeros(
                    (n_layers_seen, hidden_dim_seen), dtype=np.float16
                )

        # hidden_mean has a fixed shape across prompts, so it can be a
        # single dense (N, L, d) array.
        hidden_mean_dense = np.stack(hidden_mean_all, axis=0)

        # Build 1-D object arrays via assignment. The seemingly-equivalent
        # `np.array(list_of_arrays, dtype=object)` does not reliably produce
        # a 1-D object array: if the inner arrays happen to share a leading
        # dim (here all are (L, C_i, 7) with the same L), NumPy attempts to
        # broadcast them into a regular ndarray and fails on the variable
        # middle dim with a confusing 'could not broadcast' error. The
        # empty-then-assign pattern below is the canonical workaround and
        # is robust whether or not inner shapes happen to align.
        def _to_object_array(lst: list) -> np.ndarray:
            arr = np.empty(len(lst), dtype=object)
            for j, item in enumerate(lst):
                arr[j] = item
            return arr

        self.logger.info(f"Saving {out_path} ...")
        np.savez_compressed(
            out_path,
            checkpoint_step=np.int64(step),
            n_layers=np.int64(n_layers_seen),
            hidden_dim=np.int64(hidden_dim_seen),
            lang_labels=np.array(self.LANG_LABELS, dtype=object),
            all_labels=np.array(self.ALL_LABELS, dtype=object),
            prompt_indices=prompt_indices,
            prompt_langs=prompt_langs,
            cot_lengths=cot_lengths,
            cot_tokens=_to_object_array(cot_tokens_list),
            top50_lang_renorm=_to_object_array(top50_list),
            top1_lang_idx=_to_object_array(top1_list),
            full_lang_sum=_to_object_array(full_list),
            hidden_mean=hidden_mean_dense,
        )
        size_mb = out_path.stat().st_size / (1024 ** 2)
        self.logger.info(f"Saved {out_path} ({size_mb:.1f} MB)")

    # ------------------------------------------------------------------ #
    #  Top-level entry
    # ------------------------------------------------------------------ #

    def extract(self):
        """Iterate checkpoints x prompts, save one .npz per checkpoint."""
        self.logger.info("=" * 80)
        self.logger.info(
            f"Starting extraction: {len(self.checkpoint_steps)} checkpoints "
            f"x {len(self.eval_indices)} prompts"
        )
        self.logger.info("=" * 80)
        for step in tqdm(self.checkpoint_steps, desc="checkpoints", unit="ckpt"):
            self._process_one_checkpoint(step)
        self.logger.info("Extraction complete.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main_logger = logging.getLogger("Main")

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected.")
    main_logger.info(
        f"GPUs available: {torch.cuda.device_count()} "
        f"({torch.cuda.get_device_name(0)})"
    )

    config = {
        # ---- Required path roots ----
        "log_dir":    Path("./exp3/logs"),
        "output_dir": Path("./exp3/outputs"),
        "data_dir":   Path("./exp3/data"),

        # ---- Eval / extraction knobs ----
        "max_new_tokens": 1024,        # upper bound; greedy may stop earlier
        "top_k": 50,                  # Plot 1 top-K from the notes
        "prompt_fraction": 1.0,       # 0 < frac <= 1; per-language subset
        "num_checkpoints": 30,         # linspace count over [0, max_saved_step]
        "subset_seed": 42,            # determinism for prompt_fraction < 1
        "tokens_after_final_answer": 20,

        # Render prompts via chat template to MATCH training (train.py). Must
        # be True for Qwen2.5-Instruct or activations won't represent the
        # trained policy and generation may not terminate.
        "use_chat_template": True,

        # ---- Token-language map produced by token_classifier.py ----
        "token_language_map_path": Path("./exp3/outputs/token_language_map.json"),

        # ---- Model config (passed to GRPOModelManager) ----
        # LoRA rank/alpha MUST match train.py (r=16, alpha=32). The adapter's
        # own adapter_config.json governs the loaded weights, but the scaffold
        # built by GRPOModelManager should agree to avoid any mismatch.
        "model_config": {
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
        },

        # ---- Dataset config (passed to GRPOMGSMDataset) ----
        "dataset_config": {
            "data_dir": Path("./exp3/data"),
            "log_dir":  Path("./exp3/logs"),
            "languages": ["en", "bn", "te", "th", "ru", "ja", "zh"],
            "num_few_shot": 1,
            "seed": 42,
        },
    }

    extractor = EvalActivationsExtractor(config)
    extractor.extract()


if __name__ == "__main__":
    main()