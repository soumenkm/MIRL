"""Model management for GRPO training.

Handles:
    - Policy model with LoRA adapters (optionally QLoRA with 4-bit base)
    - Reference model (frozen, optionally 4-bit)
    - Rollout generation with stopping criteria (Final Answer + max_new_tokens)
    - Per-token log-probability computation for response tokens only
    - Checkpoint saving/loading compatible with TransformerLens
    - Detailed I/O logging for debugging
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from tqdm import tqdm


class FinalAnswerStoppingCriteria(StoppingCriteria):
    """Stop generation after 'Final Answer:' followed by a few extra tokens.

    Once the model produces a token sequence containing 'Final Answer:',
    we allow up to `max_tokens_after` additional tokens (to capture the
    numeric answer and a newline), then force stop.
    """

    def __init__(self, tokenizer, max_tokens_after: int = 10):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_tokens_after = max_tokens_after
        self.trigger_text = "Final Answer:"
        self._tokens_after_trigger = {}
        self._triggered = {}

    def reset(self, batch_size: int, prompt_len: int):
        """Reset state for a new generation call."""
        self._tokens_after_trigger = {i: 0 for i in range(batch_size)}
        self._triggered = {i: False for i in range(batch_size)}
        self._prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for seq_idx in range(input_ids.shape[0]):
            if self._triggered.get(seq_idx, False):
                self._tokens_after_trigger[seq_idx] += 1
            else:
                # Only decode the GENERATED tokens (after prompt)
                generated_ids = input_ids[seq_idx, self._prompt_len:]
                decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                if self.trigger_text in decoded:
                    self._triggered[seq_idx] = True
                    self._tokens_after_trigger[seq_idx] = 0

        all_done = all(
            self._triggered.get(i, False)
            and self._tokens_after_trigger.get(i, 0) >= self.max_tokens_after
            for i in range(input_ids.shape[0])
        )
        return all_done


class GRPOModelManager:
    """Manages policy and reference models for GRPO training."""

    def __init__(self, config: dict):
        self.model_name = config["model_name"]
        self.policy_device = torch.device(config["policy_device"])
        self.reference_device = torch.device(config["reference_device"])
        self.dtype = getattr(torch, config["dtype"])
        self.checkpoint_dir = Path(config["checkpoint_dir"])
        self.log_dir = Path(config["log_dir"])

        # LoRA config
        self.lora_rank = config["lora_rank"]
        self.lora_alpha = config["lora_alpha"]
        self.lora_dropout = config["lora_dropout"]
        self.lora_target_modules = config["lora_target_modules"]

        # Quantization config
        self.policy_load_in_4bit = config.get("policy_load_in_4bit", False)
        self.reference_load_in_4bit = config.get("reference_load_in_4bit", False)

        # Generation config
        self.max_new_tokens = config["max_new_tokens"]
        self.temperature = config["temperature"]
        self.top_p = config["top_p"]
        self.group_size = config["group_size"]
        self.tokens_after_final_answer = config["tokens_after_final_answer"]

        # Logging config
        self.log_max_response_chars = config.get("log_max_response_chars", 500)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        # Load everything
        self.tokenizer = self._load_tokenizer()
        self.policy_model = self._load_policy_model()
        self.reference_model = self._load_reference_model()
        self.stopping_criteria = self._build_stopping_criteria()

    def _setup_logging(self):
        log_file = self.log_dir / "models.log"
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            fh = logging.FileHandler(log_file, mode="w")
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(fh)
            self.logger.addHandler(sh)
        self.logger.info(f"Logging to {log_file}")

    def _load_tokenizer(self):
        self.logger.info(f"Loading tokenizer: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        self.logger.info(f"Vocab size: {tokenizer.vocab_size}, pad_token: {tokenizer.pad_token}")
        return tokenizer

    def _get_quantization_config(self) -> BitsAndBytesConfig:
        """Build 4-bit quantization config for QLoRA."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    def _load_base_model(self, device: torch.device, load_in_4bit: bool = False):
        """Load the base HuggingFace model onto a specific device.

        Args:
            device: Target device.
            load_in_4bit: If True, load with NF4 quantization (QLoRA).
        """
        self.logger.info(
            f"Loading base model: {self.model_name} -> {device} "
            f"(4bit={load_in_4bit})"
        )

        load_kwargs = {
            "trust_remote_code": True,
        }

        if load_in_4bit:
            load_kwargs["quantization_config"] = self._get_quantization_config()
            load_kwargs["device_map"] = {"": device}
        else:
            load_kwargs["torch_dtype"] = self.dtype
            load_kwargs["device_map"] = {"": device}

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs,
        )

        param_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        self.logger.info(
            f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params, "
            f"memory={param_bytes / 1e9:.2f} GB"
        )
        return model

    def _load_policy_model(self):
        """Load policy model with LoRA adapters on policy_device.

        If policy_load_in_4bit is True, uses QLoRA: 4-bit frozen base weights
        with full-precision LoRA adapter weights that receive gradients.
        """
        base_model = self._load_base_model(self.policy_device, self.policy_load_in_4bit)

        # Prepare for k-bit training if using QLoRA
        if self.policy_load_in_4bit:
            base_model = prepare_model_for_kbit_training(
                base_model,
                use_gradient_checkpointing=True,
            )
            self.logger.info("Model prepared for QLoRA (4-bit) training.")

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
            task_type="CAUSAL_LM",
            bias="none",
        )

        model = get_peft_model(base_model, lora_config)

        trainable, total = model.get_nb_trainable_parameters()
        self.logger.info(
            f"Policy model: trainable={trainable / 1e6:.2f}M / "
            f"total={total / 1e6:.2f}M ({100 * trainable / total:.2f}%)"
        )
        return model

    def _load_reference_model(self):
        """Load frozen reference model on reference_device."""
        model = self._load_base_model(self.reference_device, self.reference_load_in_4bit)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.logger.info(
            f"Reference model frozen on {self.reference_device} "
            f"(4bit={self.reference_load_in_4bit})."
        )
        return model

    def _build_stopping_criteria(self) -> StoppingCriteriaList:
        criteria = FinalAnswerStoppingCriteria(
            tokenizer=self.tokenizer,
            max_tokens_after=self.tokens_after_final_answer,
        )
        return StoppingCriteriaList([criteria])

    def _log_io(self, prompt: str, responses: list[str], log_probs: list[torch.Tensor]):
        """Log the input prompt and generated outputs for debugging.

        Writes full prompt and each response to the log file for inspection.
        """
        max_chars = self.log_max_response_chars
        separator = "-" * 80

        self.logger.info(separator)
        self.logger.info(f"[INPUT PROMPT] ({len(prompt)} chars)")
        self.logger.info(prompt)
        self.logger.info(separator)

        for g, (resp, lp) in enumerate(zip(responses, log_probs)):
            resp_display = resp[:max_chars] + ("..." if len(resp) > max_chars else "")
            avg_lp = lp.mean().item() if len(lp) > 0 else float("nan")
            self.logger.info(
                f"[OUTPUT {g}] tokens={len(lp)}, avg_log_prob={avg_lp:.4f}"
            )
            self.logger.info(resp_display)

        self.logger.info(separator)

    def tokenize_prompts(self, prompts: list[str]) -> dict:
        """Tokenize a list of prompt strings with left padding."""
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )
        return encoded

    @torch.no_grad()
    def generate_rollouts(self, prompts: list[str]) -> list[dict]:
        """Generate G rollout responses per prompt from the current policy.

        Args:
            prompts: List of prompt strings (batch).

        Returns:
            List of dicts, one per prompt, each containing:
                - "prompt_ids": tensor of prompt token ids [prompt_len]
                - "response_ids_list": list of G tensors, each [response_len_j]
                - "response_texts": list of G decoded response strings
                - "log_probs_list": list of G tensors, each [response_len_j]
                  (per-token log-probs under the current policy = pi_old)
        """
        self.policy_model.eval()

        results = []
        for prompt in tqdm(prompts, desc="Generating rollouts", leave=False):
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            ).to(self.policy_device)

            prompt_len = encoded["input_ids"].shape[1]
            prompt_ids = encoded["input_ids"].squeeze(0)

            # Expand for G samples
            expanded_input_ids = encoded["input_ids"].expand(self.group_size, -1)
            expanded_attention_mask = encoded["attention_mask"].expand(self.group_size, -1)

            # Reset stopping criteria state
            for criteria in self.stopping_criteria:
                if hasattr(criteria, "reset"):
                    criteria.reset(self.group_size, prompt_len)

            # Generate
            output_ids = self.policy_model.generate(
                input_ids=expanded_input_ids,
                attention_mask=expanded_attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            response_ids_list = []
            response_texts = []
            log_probs_list = []

            for g in range(self.group_size):
                full_seq = output_ids[g]
                response_ids = full_seq[prompt_len:]
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

                log_probs = self._compute_response_log_probs(
                    model=self.policy_model,
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    device=self.policy_device,
                )

                response_ids_list.append(response_ids.cpu())
                response_texts.append(response_text)
                log_probs_list.append(log_probs.cpu())

            # Log I/O for debugging
            self._log_io(prompt, response_texts, log_probs_list)

            results.append({
                "prompt_ids": prompt_ids.cpu(),
                "response_ids_list": response_ids_list,
                "response_texts": response_texts,
                "log_probs_list": log_probs_list,
            })

        self.policy_model.train()
        return results

    def _compute_response_log_probs(
        self,
        model,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute per-token log-probs for response tokens only.

        Concatenates [prompt_ids, response_ids], runs a forward pass, and
        extracts log-probs at positions corresponding to response tokens.

        Args:
            model: The model to compute log-probs from.
            prompt_ids: [prompt_len] tensor.
            response_ids: [response_len] tensor.
            device: Device the model is on.

        Returns:
            [response_len] tensor of per-token log-probs.
        """
        full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0).to(device)
        prompt_len = prompt_ids.shape[0]
        response_len = response_ids.shape[0]

        with torch.no_grad() if not model.training else torch.enable_grad():
            outputs = model(input_ids=full_ids)
            logits = outputs.logits

        response_logits = logits[0, prompt_len - 1: prompt_len - 1 + response_len, :]
        log_probs_all = F.log_softmax(response_logits, dim=-1)

        response_ids_device = response_ids.to(device)
        token_log_probs = log_probs_all.gather(
            dim=-1, index=response_ids_device.unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs

    def compute_policy_log_probs(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log-probs under the CURRENT policy (with gradients).

        Called during the gradient update phase, so gradients flow through
        the LoRA adapter weights (base weights are frozen in QLoRA mode).

        Args:
            prompt_ids: [prompt_len] tensor (cpu).
            response_ids: [response_len] tensor (cpu).

        Returns:
            [response_len] tensor of per-token log-probs (on policy_device, with grad).
        """
        self.policy_model.train()
        full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0).to(self.policy_device)
        prompt_len = prompt_ids.shape[0]
        response_len = response_ids.shape[0]

        outputs = self.policy_model(input_ids=full_ids)
        logits = outputs.logits

        response_logits = logits[0, prompt_len - 1: prompt_len - 1 + response_len, :]
        log_probs_all = F.log_softmax(response_logits, dim=-1)

        response_ids_device = response_ids.to(self.policy_device)
        token_log_probs = log_probs_all.gather(
            dim=-1, index=response_ids_device.unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs

    @torch.no_grad()
    def compute_reference_log_probs(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log-probs under the frozen reference model.

        Args:
            prompt_ids: [prompt_len] tensor (cpu).
            response_ids: [response_len] tensor (cpu).

        Returns:
            [response_len] tensor of per-token log-probs (cpu).
        """
        log_probs = self._compute_response_log_probs(
            model=self.reference_model,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            device=self.reference_device,
        )
        return log_probs.cpu()

    def save_checkpoint(self, step: int, optimizer, scheduler=None, extra_state: dict = None):
        """Save LoRA adapter weights, optimizer state, and training metadata.

        The LoRA weights are saved separately so they can later be merged with
        the base model and loaded into TransformerLens via merge_and_export().

        Args:
            step: Current gradient step number.
            optimizer: The optimizer whose state to save.
            scheduler: Optional LR scheduler.
            extra_state: Any additional state to save (e.g., current epoch, loss).
        """
        step_dir = self.checkpoint_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter weights
        adapter_dir = step_dir / "adapter"
        self.policy_model.save_pretrained(str(adapter_dir))
        self.tokenizer.save_pretrained(str(adapter_dir))

        # Save optimizer + scheduler + metadata
        train_state = {
            "step": step,
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if scheduler is not None:
            train_state["scheduler_state_dict"] = scheduler.state_dict()
        if extra_state is not None:
            train_state.update(extra_state)

        torch.save(train_state, step_dir / "train_state.pt")

        self.logger.info(f"Checkpoint saved: {step_dir}")

    def load_checkpoint(self, step: int, optimizer=None, scheduler=None) -> dict:
        """Load a checkpoint by step number.

        Args:
            step: The gradient step to load.
            optimizer: Optional optimizer to restore state into.
            scheduler: Optional scheduler to restore state into.

        Returns:
            The extra_state dict that was saved (or empty dict).
        """
        step_dir = self.checkpoint_dir / f"step_{step}"
        adapter_dir = step_dir / "adapter"

        if not step_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {step_dir}")

        self.policy_model = PeftModel.from_pretrained(
            self.policy_model.get_base_model(),
            str(adapter_dir),
            is_trainable=True,
        ).to(self.policy_device)

        train_state = torch.load(step_dir / "train_state.pt", map_location="cpu")
        if optimizer is not None and "optimizer_state_dict" in train_state:
            optimizer.load_state_dict(train_state["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in train_state:
            scheduler.load_state_dict(train_state["scheduler_state_dict"])

        self.logger.info(f"Checkpoint loaded: {step_dir} (step={train_state['step']})")
        return {k: v for k, v in train_state.items()
                if k not in ("optimizer_state_dict", "scheduler_state_dict")}

    def merge_and_export(self, step: int, export_dir: Path):
        """Merge LoRA weights into base model and export for TransformerLens.

        Always loads the base model in FULL PRECISION (no quantization) before
        merging, regardless of whether QLoRA was used during training. This
        ensures the exported model has clean full-precision weights for
        mechanistic interpretability analysis.

        Args:
            step: Checkpoint step to merge.
            export_dir: Directory to save the merged model.
        """
        step_dir = self.checkpoint_dir / f"step_{step}"
        adapter_dir = step_dir / "adapter"

        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_dir}")

        self.logger.info(f"Merging LoRA weights from step {step} (full precision)")

        # Always load base in full precision for clean export
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            device_map="cpu",
        )

        peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        merged_model = peft_model.merge_and_unload()

        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(export_dir))
        self.tokenizer.save_pretrained(str(export_dir))

        self.logger.info(f"Merged model exported to {export_dir}")


def main():
    """Sanity check: load models, generate a rollout, compute log-probs."""

    config = {
        # Model
        "model_name": "/home/speech-nlp-cse/23m2157/MS_Research/MIRL/models/Qwen2.5-7B-Instruct",
        "policy_device": "cuda:0",
        "reference_device": "cuda:0",
        "dtype": "float16",

        # LoRA
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],

        # Quantization (set both to True for QLoRA mode)
        "policy_load_in_4bit": True,
        "reference_load_in_4bit": True,

        # Generation
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95,
        "group_size": 4,
        "tokens_after_final_answer": 10,

        # Logging
        "log_max_response_chars": 500,

        # Checkpointing
        "checkpoint_dir": "./exp2/outputs/checkpoints",
        "log_dir": "./exp2/logs",
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logging.warning("No GPU available. Exiting.")
        return

    num_gpus = torch.cuda.device_count()
    logging.info(f"GPUs available: {num_gpus}")
    if num_gpus < 2:
        logging.warning("Need at least 2 GPUs. Adjusting to single GPU mode.")
        config["reference_device"] = "cuda:0"

    manager = GRPOModelManager(config)

    # Quick sanity check
    test_prompt = (
        "Solve the following math problem step by step.\n\n"
        "Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. "
        "Each can has 3 tennis balls. How many tennis balls does he have now?\n"
        "Step-by-step Answer:"
    )

    logging.info("Running sanity check rollout...")
    rollouts = manager.generate_rollouts([test_prompt])
    for g in range(config["group_size"]):
        text = rollouts[0]["response_texts"][g]
        lp = rollouts[0]["log_probs_list"][g]
        logging.info(f"  Rollout {g}: {len(lp)} tokens, text={text[:100]}...")

    logging.info("Sanity check passed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()