"""GRPO training for cross-lingual mathematical reasoning, built on TRL.

This file REPLACES the hand-rolled GRPO loop (see the previous ``train.py``)
with HuggingFace TRL's battle-tested ``GRPOTrainer``. The previous loop
manually implemented rollout collection, group-advantage normalization, the
clipped surrogate, and the KL penalty. A flat-accuracy curve across all
languages and steps is a classic symptom of a subtle bug in a hand-rolled
loop (log-prob/token misalignment, advantage sign, KL estimator, or a
mismatched generation config between rollout and scoring). Delegating the RL
machinery to TRL removes that entire class of bugs, and we adopt the
known-good hyperparameters from the reference GRPO notebook as the gold
standard.

Architecture preserved from the previous setup:
    - Policy is a LoRA (optionally QLoRA / 4-bit) adapter over an HF causal LM
      loaded from ``config["model_path"]``.
    - Accuracy reward: regex-first extraction with an OFFLINE Gemma judge
      fallback for responses where regex cannot extract an integer. The judge
      is a remote vLLM server in its own SLURM job, discovered via a
      connection file. Both are implemented in ``rewards.py`` (RewardManager /
      JudgeClient), which we reuse as-is.
    - Format reward: rewards a single well-formed ``Final Answer: <int>`` line
      with nothing trailing it. Also from ``rewards.py``.
    - Checkpoints saved every 20 steps; file + stream logging; tqdm progress
      (TRL drives its own progress bar over training steps).

Coding-standards compliance:
    - All logic lives in the ``GRPOTRLTrainer`` class; only ``main()`` is a
      standalone function, called from the ``__main__`` block.
    - All inputs come from a single ``config: dict``; no argparse.
    - All paths are ``pathlib.Path``; the ``os`` module is not used for paths.
    - Directories created in ``__init__`` via ``mkdir(parents, exist_ok)``.
    - GPU availability checked in ``main()`` before instantiation; GPU name +
      count logged at startup.
    - Model loaded from HuggingFace using ``config["model_path"]``.
"""

import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm
from datasets import Dataset as HFDataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import GRPOConfig, GRPOTrainer

from grpo_dataset import GRPOMGSMDataset
from rewards import RewardManager


class CheckpointEveryNStepsCallback(TrainerCallback):
    """Save a checkpoint every ``save_freq`` optimizer steps.

    TRL's ``GRPOConfig`` already supports ``save_steps``; this callback adds
    explicit logging so the cadence is visible in our own log file, matching
    the previous trainer's "checkpoint every 20 steps" behaviour. The actual
    write is handled by the HF Trainer via ``control.should_save``.
    """

    def __init__(self, save_freq: int, logger: logging.Logger):
        self.save_freq = save_freq
        self.logger = logger

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.save_freq == 0:
            self.logger.info(
                f"[checkpoint] global_step={state.global_step} "
                f"(every {self.save_freq} steps) -> saving."
            )
            control.should_save = True
        return control


class RewardLoggingCallback(TrainerCallback):
    """Mirror TRL's logged metrics into our file/stream logger each step."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        parts = [f"step={state.global_step}"]
        for key in sorted(logs):
            val = logs[key]
            if isinstance(val, float):
                parts.append(f"{key}={val:.4f}")
            else:
                parts.append(f"{key}={val}")
        self.logger.info("[trl] " + " ".join(parts))
        return control


class GRPOTRLTrainer:
    """GRPO trainer for cross-lingual math reasoning, wrapping ``trl.GRPOTrainer``.

    Responsibilities:
        - Load the policy model (LoRA / optional QLoRA) + tokenizer from HF.
        - Build the prompt/answer dataset in TRL's expected schema.
        - Wire up two reward functions (accuracy, format) that delegate to a
          shared ``RewardManager`` (regex -> offline Gemma judge fallback).
        - Configure and run ``trl.GRPOTrainer`` with gold-standard
          hyperparameters and checkpoint-every-N-steps.
    """

    def __init__(self, config: dict):
        self.config = config

        # ---- Required path roots (coding standards) ----
        self.log_dir = Path(config["log_dir"])
        self.output_dir = Path(config["output_dir"])
        self.data_dir = Path(config["data_dir"])

        # ---- Model ----
        self.model_path = Path(config["model_path"])
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))
        self.load_in_4bit = bool(config.get("load_in_4bit", False))
        self.max_seq_length = int(config.get("max_seq_length", 2048))

        # ---- LoRA ----
        self.lora_rank = int(config.get("lora_rank", 16))
        self.lora_alpha = int(config.get("lora_alpha", 32))
        self.lora_dropout = float(config.get("lora_dropout", 0.1))
        self.lora_target_modules = config.get(
            "lora_target_modules", ["q_proj", "v_proj"]
        )

        # ---- Checkpoint cadence ----
        self.checkpoint_save_freq = int(config.get("checkpoint_save_freq", 20))

        # Create the directory structure at runtime.
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        # Order matters: build the reward manager FIRST so we block on the
        # offline judge becoming healthy before spending GPU time loading the
        # policy. RewardManager.__init__ calls JudgeClient.discover() when the
        # accuracy signal needs the judge fallback.
        self.logger.info("Initializing reward manager (may block on judge discovery)...")
        self.reward_mgr = RewardManager(config["reward_config"])

        self.logger.info(f"Loading tokenizer + model from {self.model_path}")
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.peft_config = self._build_lora_config()

        self.logger.info("Building train/eval datasets...")
        self.train_dataset = self._build_hf_dataset(split="train")
        self.eval_dataset = self._build_hf_dataset(split="test")

        self.logger.info("Building GRPO trainer...")
        self.training_args = self._build_grpo_config()
        self.trainer = self._build_trainer()

    # ------------------------------------------------------------------ #
    #  Logging
    # ------------------------------------------------------------------ #

    def _setup_logging(self):
        log_file = self.log_dir / "train.log"
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
    #  Model / tokenizer loading (HuggingFace, from config["model_path"])
    # ------------------------------------------------------------------ #

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # GRPO generates completions; left padding keeps the prompt aligned.
        tokenizer.padding_side = "left"
        # TRL 1.x dropped max_prompt_length, so cap prompt length here to
        # keep long few-shot prompts within the model's context window.
        tokenizer.model_max_length = self.max_seq_length
        self.logger.info(
            f"Tokenizer loaded (vocab={tokenizer.vocab_size}, "
            f"pad_token={tokenizer.pad_token}, "
            f"model_max_length={tokenizer.model_max_length})"
        )
        return tokenizer

    def _get_quantization_config(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_use_double_quant=True,
        )

    def _load_model(self):
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if self.load_in_4bit:
            load_kwargs["quantization_config"] = self._get_quantization_config()
            self.logger.info("Loading policy with 4-bit (QLoRA) quantization.")
        else:
            load_kwargs["torch_dtype"] = self.dtype
            self.logger.info(f"Loading policy in {self.dtype} (no quantization).")

        model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            **load_kwargs,
        )
        n_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Model loaded: ~{n_params / 1e9:.2f}B parameters.")
        return model

    def _build_lora_config(self) -> LoraConfig:
        cfg = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.logger.info(
            f"LoRA config: r={self.lora_rank}, alpha={self.lora_alpha}, "
            f"dropout={self.lora_dropout}, targets={self.lora_target_modules}"
        )
        return cfg

    # ------------------------------------------------------------------ #
    #  Dataset (TRL schema: a "prompt" column + extra columns passed through
    #  to reward functions as kwargs)
    # ------------------------------------------------------------------ #

    def _build_hf_dataset(self, split: str) -> HFDataset:
        """Convert GRPOMGSMDataset into a TRL-compatible HF Dataset.

        TRL's GRPOTrainer expects a "prompt" column and forwards any other
        columns to the reward functions as keyword arguments. We pass
        ``answer_number`` and ``lang`` through so the reward functions can
        score correctly.

        Prompts are rendered through the tokenizer's CHAT TEMPLATE (system +
        user turns, with the generation prompt appended). Qwen2.5-Instruct is
        trained on the <|im_start|>...<|im_end|> chat format; feeding it raw
        completion-style text is why completions previously never terminated
        (clipped_ratio=1.0). The system turn carries the formatting
        instruction; the user turn carries the few-shot blocks + question.
        """
        base = GRPOMGSMDataset(self.config["dataset_config"], split=split)

        use_chat_template = bool(
            self.config.get("use_chat_template", True)
        ) and self.tokenizer.chat_template is not None

        prompts: list[str] = []
        answer_numbers: list[int] = []
        langs: list[str] = []
        for i in tqdm(range(len(base)), desc=f"Materialize {split}", unit="ex"):
            item = base[i]
            if use_chat_template:
                messages = [
                    {"role": "system", "content": item["system"]},
                    {"role": "user", "content": item["user"]},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = item["prompt"]
            prompts.append(prompt)
            answer_numbers.append(item["answer_number"])
            langs.append(item["lang"])

        ds = HFDataset.from_dict(
            {
                "prompt": prompts,
                "answer_number": answer_numbers,
                "lang": langs,
            }
        )
        self.logger.info(
            f"HF dataset[{split}]: {len(ds)} examples "
            f"(chat_template={'on' if use_chat_template else 'off'})."
        )
        return ds

    # ------------------------------------------------------------------ #
    #  Reward functions (TRL-compatible signatures)
    #
    #  TRL calls each reward function as:
    #      func(prompts, completions, **kwargs) -> list[float]
    #  where ``completions`` is a list of generated strings (one per sample)
    #  and the dataset's extra columns arrive in kwargs as lists aligned to
    #  ``completions`` (e.g. kwargs["answer_number"], kwargs["lang"]).
    #
    #  Both functions delegate to the shared RewardManager so the regex /
    #  offline-Gemma-judge logic stays in one place.
    # ------------------------------------------------------------------ #

    def _normalize_completions(self, completions) -> list[str]:
        """TRL may pass completions as plain strings or as chat-style lists.

        Our dataset uses plain-text prompts, so completions are usually
        strings. Guard against the conversational format just in case.
        """
        texts = []
        for c in completions:
            if isinstance(c, str):
                texts.append(c)
            elif isinstance(c, list) and c and isinstance(c[0], dict):
                texts.append(c[0].get("content", ""))
            else:
                texts.append(str(c))
        return texts

    def _accuracy_reward(self, prompts, completions, **kwargs) -> list[float]:
        """Accuracy reward: regex extraction first, offline Gemma judge fallback.

        Delegates to RewardManager, whose combined-accuracy path runs the
        robust regex extractor and only defers to the remote judge for
        responses where regex returns no clean integer.
        """
        texts = self._normalize_completions(completions)
        answer_numbers = list(kwargs["answer_number"])
        langs = list(kwargs.get("lang", [None] * len(texts)))

        results = self.reward_mgr.compute_rewards(
            response_texts=texts,
            answer_numbers=answer_numbers,
            langs=langs,
        )
        self.reward_mgr.log_reward_summary(results)
        # RewardManager already applies its accuracy weight; expose the
        # combined accuracy component as the scalar for this reward function.
        # When weight_accuracy > 0, results[i]["reward"] is weight * accuracy,
        # so divide back out to a clean 0/1-style signal and let GRPOConfig /
        # this function's own scale handle weighting.
        return [float(r["accuracy_regex"]) for r in results]

    def _format_reward(self, prompts, completions, **kwargs) -> list[float]:
        """Format reward: well-formed single 'Final Answer: <int>' line."""
        texts = self._normalize_completions(completions)
        return [
            float(self.reward_mgr._compute_format_reward(t)) for t in texts
        ]

    # ------------------------------------------------------------------ #
    #  GRPO config (gold-standard hyperparameters from the reference notebook)
    # ------------------------------------------------------------------ #

    def _build_generation_kwargs(self) -> dict:
        """Build generation_kwargs forwarded to GenerationConfig at sampling.

        Ensures completions end their turn instead of running to
        max_completion_length. For Qwen2.5-Instruct the turn-ending token is
        ``<|im_end|>`` (id 151645, not ``<|endoftext|>``), so we add it to
        eos_token_id. We deliberately avoid ``stop_strings`` because TRL's
        generation path does not pass a tokenizer to ``model.generate()`` and
        HF then raises; integer eos ids need no tokenizer. Respects an
        explicit ``generation_kwargs`` override from config.
        """
        override = self.config.get("generation_kwargs")
        if override is not None:
            self.logger.info(f"Using generation_kwargs from config: {override}")
            return override

        eos_ids = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(self.tokenizer.eos_token_id)

        # Add Qwen's turn-ending token <|im_end|> to eos_token_id. We rely on
        # eos_token_id (integer ids) rather than stop_strings: TRL's
        # generation path does NOT forward a tokenizer to model.generate(),
        # and HF's stop_strings feature requires one, raising otherwise. EOS
        # token ids need no tokenizer and terminate generation just as well.
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if im_end is not None and im_end >= 0 and im_end != unk_id:
            if im_end not in eos_ids:
                eos_ids.append(im_end)

        gen_kwargs: dict = {}
        if eos_ids:
            gen_kwargs["eos_token_id"] = eos_ids

        self.logger.info(
            f"generation_kwargs: eos_token_id={gen_kwargs.get('eos_token_id')}"
        )
        return gen_kwargs

    def _build_grpo_config(self) -> GRPOConfig:
        c = self.config
        args = GRPOConfig(
            output_dir=str(self.checkpoint_dir),

            # ---- Learning (gold-standard: conservative LR for reasoning) ----
            learning_rate=float(c.get("learning_rate", 5e-6)),
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=float(c.get("weight_decay", 0.1)),
            warmup_ratio=float(c.get("warmup_ratio", 0.1)),
            lr_scheduler_type=c.get("lr_scheduler_type", "cosine"),
            optim=c.get("optim", "adamw_torch"),
            max_grad_norm=float(c.get("max_grad_norm", 0.1)),

            # ---- Batch / accumulation ----
            per_device_train_batch_size=int(
                c.get("per_device_train_batch_size", 2)
            ),
            gradient_accumulation_steps=int(
                c.get("gradient_accumulation_steps", 8)
            ),

            # ---- GRPO generation ----
            # NOTE: TRL removed `max_prompt_length` in the 1.x line; prompt
            # truncation is handled by the tokenizer. Only completion length
            # is configured here.
            num_generations=int(c.get("num_generations", 8)),
            max_completion_length=int(c.get("max_completion_length", 1024)),
            temperature=float(c.get("temperature", 0.7)),
            top_p=float(c.get("top_p", 0.95)),
            beta=float(c.get("beta", 0.04)),

            # ---- Stop condition so completions TERMINATE rather than run to
            #      max_completion_length every time. Adds Qwen's <|im_end|>
            #      turn token to eos_token_id. Without this, clipped_ratio
            #      sits at 1.0 and the regex often never sees a "Final Answer:"
            #      line -> accuracy is capped and the training signal is weak.
            generation_kwargs=self._build_generation_kwargs(),

            # ---- Per-reward-function weights ----
            # Positional: index 0 -> accuracy reward, index 1 -> format reward
            # (must match the order in reward_funcs in _build_trainer). TRL
            # multiplies each reward function's output by its weight, then sums.
            # Accuracy is 0/1; format maxes at 1.0, so down-weighting format
            # keeps it a secondary nudge rather than rivaling accuracy.
            reward_weights=list(c.get("reward_weights", [1.0, 0.2])),

            # ---- Duration / logging ----
            num_train_epochs=float(c.get("num_train_epochs", 1)),
            max_steps=int(c.get("max_steps", -1)),
            logging_steps=int(c.get("logging_steps", 1)),

            # ---- Checkpointing: every N steps ----
            save_strategy="steps",
            save_steps=self.checkpoint_save_freq,
            save_total_limit=c.get("save_total_limit", None),

            report_to="none",
            logging_dir=str(self.log_dir),
            bf16=(self.dtype == torch.bfloat16),
            fp16=(self.dtype == torch.float16),
            gradient_checkpointing=bool(c.get("gradient_checkpointing", True)),
            seed=int(c.get("seed", 42)),
        )
        self.logger.info(
            "GRPO config: "
            f"lr={args.learning_rate}, "
            f"per_device_bs={args.per_device_train_batch_size}, "
            f"grad_accum={args.gradient_accumulation_steps}, "
            f"num_generations={args.num_generations}, "
            f"beta={args.beta}, max_grad_norm={args.max_grad_norm}, "
            f"max_steps={args.max_steps}, save_steps={args.save_steps}, "
            f"reward_weights={args.reward_weights}"
        )
        return args

    def _build_trainer(self) -> GRPOTrainer:
        callbacks = [
            CheckpointEveryNStepsCallback(self.checkpoint_save_freq, self.logger),
            RewardLoggingCallback(self.logger),
        ]
        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            # Order matters: this list must align with reward_weights in
            # GRPOConfig (index 0 = accuracy, index 1 = format).
            reward_funcs=[self._accuracy_reward, self._format_reward],
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            peft_config=self.peft_config,
            callbacks=callbacks,
        )
        self.logger.info("GRPOTrainer constructed with 2 reward functions.")
        return trainer

    # ------------------------------------------------------------------ #
    #  Train
    # ------------------------------------------------------------------ #

    def train(self):
        self.logger.info("=" * 80)
        self.logger.info("Starting GRPO training (TRL)")
        self.logger.info(f"  Train examples: {len(self.train_dataset)}")
        self.logger.info(f"  Checkpoints every {self.checkpoint_save_freq} steps")
        self.logger.info("=" * 80)

        train_result = self.trainer.train()

        # Persist the final adapter + metrics.
        final_dir = self.output_dir / "final_adapter"
        final_dir.mkdir(parents=True, exist_ok=True)
        self.trainer.save_model(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))

        metrics_path = self.output_dir / "training_metrics.json"
        with metrics_path.open("w") as f:
            json.dump(train_result.metrics, f, indent=2)

        self.logger.info(f"Final adapter saved to {final_dir}")
        self.logger.info("Training complete.")
        return train_result


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main_logger = logging.getLogger("Main")

    # GPU availability check BEFORE instantiating anything (coding standards).
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected.")

    num_gpus = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    main_logger.info(f"GPUs available: {num_gpus} ({gpu_name})")

    config = {
        # ---- Required path roots (per coding standards) ----
        "log_dir":    Path("./exp3/logs"),
        "output_dir": Path("./exp3/outputs"),
        "data_dir":   Path("./exp3/data"),

        # ---- Model (loaded from HuggingFace via model_path) ----
        "model_path":      Path("./models/Qwen2.5-7B-Instruct"),
        "dtype":           "bfloat16",
        "load_in_4bit":    False,           # full-precision LoRA (no QLoRA)
        "max_seq_length":  2048,

        # Render prompts via the tokenizer's chat template (system+user turns).
        # Strongly recommended for Instruct models so completions terminate.
        "use_chat_template": True,

        # ---- LoRA (gold-standard notebook defaults) ----
        "lora_rank":           16,
        "lora_alpha":          32,
        "lora_dropout":        0.1,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],

        # ---- GRPO hyperparameters (gold standard from the reference notebook) ----
        "learning_rate":               5e-5,
        "weight_decay":                0.1,
        "warmup_ratio":                0.1,
        "lr_scheduler_type":           "cosine",
        "optim":                       "adamw_torch",
        "max_grad_norm":               0.1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,    # effective batch = 16
        "num_generations":             8,    # G responses per prompt
        "max_prompt_length":           1024, # NOTE: unused by TRL 1.x (kept
                                              # for reference); prompt length
                                              # is bounded via max_seq_length
                                              # on the tokenizer instead.
        "max_completion_length":       1024,
        "temperature":                 0.7,
        "top_p":                       0.95,
        "beta":                        0.04, # KL penalty coefficient
        "gradient_checkpointing":      True,

        # Optional: set to a dict to fully override generation stop behaviour.
        # Leave as None to auto-build eos_token_id (adds Qwen's <|im_end|>).
        "generation_kwargs":           None,
        "seed":                        42,

        # ---- Per-reward-function weights (order matches reward_funcs) ----
        # [accuracy, format]. Accuracy is 0/1; format maxes at 1.0, so 0.2
        # keeps format a secondary signal. Set [1.0, 0.0] to disable format.
        "reward_weights": [1.0, 0.0],

        # ---- Duration / logging ----
        # Set max_steps for a fixed-length run (notebook uses a short demo of
        # 10; use 500+ for a real run). num_train_epochs is ignored when
        # max_steps > 0.
        "num_train_epochs": 1,
        "max_steps":        500,
        "logging_steps":    1,

        # ---- Checkpointing ----
        "checkpoint_save_freq": 20,   # save every 20 optimizer steps
        "save_total_limit":     None,

        # ---- Reward config (passed to RewardManager) ----
        # Combined accuracy = regex first, offline Gemma judge ONLY on regex
        # misses. weight_accuracy > 0 activates that combined path; the judge
        # client is built and discovered automatically.
        "reward_config": {
            "log_dir":                Path("./exp3/logs"),
            "log_max_response_chars": 500,

            "weight_accuracy":             1.0,   # regex -> judge fallback
            "weight_accuracy_regex":       0.0,
            "weight_accuracy_judge":       0.0,
            "weight_format":               0.0,   # format handled as its own
                                                  # TRL reward func instead
            "weight_language_consistency": 0.0,

            "judge_max_tokens":  8,
            "judge_temperature": 0.0,

            "judge_config": {
                "connection_file":         Path("./exp3/outputs/judge_connection.json"),
                "log_dir":                 Path("./exp3/logs"),
                "discovery_timeout":       1800,
                "discovery_poll_interval": 5.0,
                "request_timeout":         120,
                "max_retries":             3,
                "retry_backoff":           2.0,
                "max_concurrent_requests": 32,
            },
        },

        # ---- Dataset config (passed to GRPOMGSMDataset) ----
        "dataset_config": {
            "data_dir":     Path("./exp3/data"),
            "log_dir":      Path("./exp3/logs"),
            "languages":    ["en", "bn", "te", "th", "ru", "ja", "zh"],
            "num_few_shot": 1,
            "seed":         42,
        },
    }

    trainer = GRPOTRLTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()