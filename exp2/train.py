"""GRPO training loop for cross-lingual collapse experiments.

Implements the exact GRPO formulation from the handwritten notes (pages 3-7):
    1. Rollout collection: Sample G responses per prompt from π_θ^old
    2. Reward computation: Get R^i_j via regex/judge
    3. Group advantage normalization: A^i_j = (R^i_j - μ^i) / σ^i
    4. Policy gradient update with clipped surrogate + KL penalty
    5. Checkpoint saving at configurable intervals
    6. Fresh rollout collection after each epoch (epoch = iteration)

Loss function (per prompt i):
    L^i_GRPO = (1/G) Σ_j (1/|y^i_j|) Σ_t [ min(r_{j,t} · A_j, g_{j,t}) - β · D_KL^t ]

Where:
    r_{j,t} = π_θ(t_t|t_{1:t-1}) / π_θ^old(t_t|t_{1:t-1})
    g_{j,t} = clip(A_j) = (1+ε)A_j if A_j ≥ 0, (1-ε)A_j otherwise
    s_t = π_ref(t_t|t_{1:t-1}) / π_θ(t_t|t_{1:t-1})
    D_KL^t = s_t - log(s_t) - 1

Batch loss: L = (1/B) Σ_i L^i_GRPO
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import logging
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from grpo_dataset import GRPOMGSMDataset
from models import GRPOModelManager
from rewards import RewardManager


class GRPOTrainer:
    """GRPO training loop following the handwritten notes formulation."""

    def __init__(self, config: dict):
        self.config = config
        self.log_dir = Path(config["log_dir"])
        self.output_dir = Path(config["output_dir"])

        # Training hyperparameters
        self.num_iterations = config["num_iterations"]
        self.batch_size = config["batch_size"]
        self.group_size = config["group_size"]
        self.epsilon = config["epsilon"]
        self.beta = config["beta"]
        self.learning_rate = config["learning_rate"]
        self.max_grad_norm = config.get("max_grad_norm", 1.0)

        # Checkpoint config
        self.checkpoint_save_freq = config["checkpoint_save_freq"]
        self.resume_from_step = config.get("resume_from_step", None)

        # Logging config
        self.log_every_n_steps = config.get("log_every_n_steps", 1)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        # Initialize components
        self.logger.info("Initializing model manager...")
        self.model_mgr = GRPOModelManager(config["model_config"])

        self.logger.info("Initializing reward manager...")
        self.reward_mgr = RewardManager(config["reward_config"])

        self.logger.info("Initializing dataset...")
        self.train_dataset = GRPOMGSMDataset(config["dataset_config"], split="train")
        self.test_dataset = GRPOMGSMDataset(config["dataset_config"], split="test")

        # Optimizer (only LoRA parameters)
        trainable_params = [
            p for p in self.model_mgr.policy_model.parameters() if p.requires_grad
        ]
        self.optimizer = AdamW(trainable_params, lr=self.learning_rate)

        # Scheduler
        total_steps = self._estimate_total_steps()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        self.global_step = 0

        # Resume if needed
        if self.resume_from_step is not None:
            self._resume_training()

        self.logger.info(f"Training config: iterations={self.num_iterations}, "
                         f"batch_size={self.batch_size}, group_size={self.group_size}, "
                         f"epsilon={self.epsilon}, beta={self.beta}, lr={self.learning_rate}")
        self.logger.info(f"Total training examples: {len(self.train_dataset)}")
        self.logger.info(f"Steps per iteration: {len(self.train_dataset) // self.batch_size}")
        self.logger.info(f"Estimated total steps: {total_steps}")
        self.logger.info(f"Checkpoint save frequency: every {self.checkpoint_save_freq} steps")

    def _setup_logging(self):
        log_file = self.log_dir / "train.log"
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            fh = logging.FileHandler(log_file, mode="a")
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(fh)
            self.logger.addHandler(sh)
        self.logger.info(f"Logging to {log_file}")

    def _estimate_total_steps(self) -> int:
        """Estimate total gradient steps across all iterations."""
        steps_per_iteration = len(self.train_dataset) // self.batch_size
        return steps_per_iteration * self.num_iterations

    def _resume_training(self):
        """Resume training from a saved checkpoint."""
        self.logger.info(f"Resuming from step {self.resume_from_step}")
        extra_state = self.model_mgr.load_checkpoint(
            step=self.resume_from_step,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.global_step = extra_state.get("step", self.resume_from_step)
        self.logger.info(f"Resumed at global_step={self.global_step}")

    def _shuffle_indices(self, n: int, seed: int) -> list[int]:
        """Get shuffled indices for an epoch."""
        rng = random.Random(seed)
        indices = list(range(n))
        rng.shuffle(indices)
        return indices

    def _collect_rollouts(self, batch_indices: list[int]) -> dict:
        """Collect rollouts for a batch of prompts.

        For each prompt, generates G responses and computes rewards.

        Args:
            batch_indices: Indices into self.train_dataset.

        Returns:
            Dictionary containing all rollout data needed for gradient update:
                - prompts: list of prompt strings
                - prompt_ids_list: list of prompt token tensors
                - response_ids_list: list of lists of response token tensors [B][G]
                - response_texts: list of lists of response strings [B][G]
                - old_log_probs_list: list of lists of log-prob tensors [B][G]
                - rewards: list of lists of reward floats [B][G]
                - advantages: list of lists of advantage floats [B][G]
                - answer_numbers: list of ground truth answers [B]
                - langs: list of language codes [B]
        """
        batch_items = [self.train_dataset[i] for i in batch_indices]
        prompts = [item["prompt"] for item in batch_items]
        answer_numbers = [item["answer_number"] for item in batch_items]
        langs = [item["lang"] for item in batch_items]

        # Generate G rollouts per prompt
        rollouts = self.model_mgr.generate_rollouts(prompts)

        # Compute rewards for all responses
        all_rewards = []
        all_response_texts = []
        for i, rollout in enumerate(rollouts):
            group_rewards = []
            for g in range(self.group_size):
                reward_result = self.reward_mgr.compute_rewards(
                    response_texts=[rollout["response_texts"][g]],
                    answer_numbers=[answer_numbers[i]],
                    langs=[langs[i]],
                )
                group_rewards.append(reward_result[0]["reward"])
            all_rewards.append(group_rewards)
            all_response_texts.append(rollout["response_texts"])

        # Compute group-normalized advantages (page 4 of notes)
        # μ^i = Σ_j R^i_j / G
        # σ^i = sqrt(Σ_j (R^i_j - μ^i)^2 / G)
        # A^i_j = (R^i_j - μ^i) / σ^i
        all_advantages = []
        for i, group_rewards in enumerate(all_rewards):
            rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32)
            mu = rewards_tensor.mean()
            sigma = rewards_tensor.std(correction=0)  # population std, not sample std

            if sigma < 1e-8:
                # All rewards identical -> zero advantage
                advantages = [0.0] * self.group_size
            else:
                advantages = ((rewards_tensor - mu) / sigma).tolist()

            all_advantages.append(advantages)

        return {
            "prompts": prompts,
            "prompt_ids_list": [r["prompt_ids"] for r in rollouts],
            "response_ids_list": [r["response_ids_list"] for r in rollouts],
            "response_texts": all_response_texts,
            "old_log_probs_list": [r["log_probs_list"] for r in rollouts],
            "rewards": all_rewards,
            "advantages": all_advantages,
            "answer_numbers": answer_numbers,
            "langs": langs,
        }

    def _compute_grpo_loss(self, rollout_data: dict) -> torch.Tensor:
        """Compute the GRPO loss for a minibatch (page 6 of notes).

        L = (1/B) Σ_i L^i_GRPO

        L^i_GRPO = (1/G) Σ_j (1/|y^i_j|) Σ_t [ min(r_{j,t} · A_j, g_{j,t}) - β · D_KL^t ]

        Where:
            r_{j,t} = π_θ(t_t|...) / π_θ^old(t_t|...)     (probability ratio)
            g_{j,t} = (1+ε)·A_j if A_j ≥ 0, (1-ε)·A_j otherwise  (clip function)
            s_t = π_ref(t_t|...) / π_θ(t_t|...)           (for KL)
            D_KL^t = s_t - log(s_t) - 1                   (per-token KL)

        Returns:
            Scalar loss tensor with gradients.
        """
        B = len(rollout_data["prompt_ids_list"])
        batch_loss = torch.tensor(0.0, device=self.model_mgr.policy_device, requires_grad=False)

        for i in range(B):
            prompt_ids = rollout_data["prompt_ids_list"][i]
            prompt_loss = torch.tensor(0.0, device=self.model_mgr.policy_device)

            for j in range(self.group_size):
                response_ids = rollout_data["response_ids_list"][i][j]
                old_log_probs = rollout_data["old_log_probs_list"][i][j]
                advantage = rollout_data["advantages"][i][j]
                response_len = response_ids.shape[0]

                if response_len == 0:
                    continue

                # A^i_{j,t} = A^i_j for all t (page 5 of notes)
                A_jt = advantage

                # π_θ current log-probs (WITH gradients)
                current_log_probs = self.model_mgr.compute_policy_log_probs(
                    prompt_ids, response_ids
                )  # [response_len], on policy_device, with grad

                # π_θ^old log-probs (stored from rollout, no grad)
                old_log_probs_device = old_log_probs.to(self.model_mgr.policy_device)

                # π_ref log-probs (from frozen reference model)
                ref_log_probs = self.model_mgr.compute_reference_log_probs(
                    prompt_ids, response_ids
                ).to(self.model_mgr.policy_device)  # [response_len]

                # Per-token probability ratio: r_{j,t} = π_θ / π_θ^old
                # In log space: log(r) = log(π_θ) - log(π_θ^old)
                log_ratio = current_log_probs - old_log_probs_device
                ratio = torch.exp(log_ratio)  # r_{j,t}

                # Clipped objective (page 5-6 of notes):
                # g_{j,t} = (1+ε)·A_j if A_j ≥ 0, else (1-ε)·A_j
                if A_jt >= 0:
                    clipped_value = (1 + self.epsilon) * A_jt
                else:
                    clipped_value = (1 - self.epsilon) * A_jt

                # min(r_{j,t} · A_j, g_{j,t})
                surrogate = torch.min(ratio * A_jt, torch.tensor(clipped_value, device=ratio.device))

                # KL divergence (page 6 of notes):
                # s_t = π_ref(t_t|...) / π_θ(t_t|...)
                # D_KL^t = s_t - log(s_t) - 1
                log_s = ref_log_probs - current_log_probs  # log(s_t)
                s_t = torch.exp(log_s)
                kl_per_token = s_t - log_s - 1  # D_KL^t

                # Per-response loss: (1/|y^i_j|) Σ_t [ surrogate - β · D_KL^t ]
                per_token_objective = surrogate - self.beta * kl_per_token
                response_loss = per_token_objective.sum() / response_len

                prompt_loss = prompt_loss + response_loss

            # Average over group: (1/G) Σ_j
            prompt_loss = prompt_loss / self.group_size
            batch_loss = batch_loss + prompt_loss

        # Average over batch: (1/B) Σ_i
        batch_loss = batch_loss / B

        # We want to MAXIMIZE the objective, so MINIMIZE the negative
        return -batch_loss

    def _gradient_step(self, rollout_data: dict) -> dict:
        """Perform one gradient update step.

        Args:
            rollout_data: Output of _collect_rollouts().

        Returns:
            Dictionary of metrics for logging.
        """
        self.optimizer.zero_grad()
        loss = self._compute_grpo_loss(rollout_data)
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model_mgr.policy_model.parameters() if p.requires_grad],
            self.max_grad_norm,
        )

        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        # Compute metrics
        all_rewards_flat = [r for group in rollout_data["rewards"] for r in group]
        all_advantages_flat = [a for group in rollout_data["advantages"] for a in group]

        metrics = {
            "step": self.global_step,
            "loss": loss.item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr": self.scheduler.get_last_lr()[0],
            "avg_reward": sum(all_rewards_flat) / len(all_rewards_flat),
            "avg_advantage": sum(all_advantages_flat) / len(all_advantages_flat),
            "max_reward": max(all_rewards_flat),
            "min_reward": min(all_rewards_flat),
            "num_correct": sum(1 for r in all_rewards_flat if r > 0),
            "num_total": len(all_rewards_flat),
        }

        return metrics

    def _log_metrics(self, metrics: dict, iteration: int):
        """Log training metrics."""
        self.logger.info(
            f"[iter={iteration} step={metrics['step']}] "
            f"loss={metrics['loss']:.4f} "
            f"reward={metrics['avg_reward']:.3f} "
            f"correct={metrics['num_correct']}/{metrics['num_total']} "
            f"grad_norm={metrics['grad_norm']:.4f} "
            f"lr={metrics['lr']:.2e}"
        )

    def _save_metrics_json(self, all_metrics: list[dict]):
        """Save all collected metrics to a JSON file."""
        metrics_path = self.output_dir / "training_metrics.json"
        with metrics_path.open("w") as f:
            json.dump(all_metrics, f, indent=2)

    def _evaluate(self, iteration: int):
        """Run evaluation on the test set and log results."""
        self.logger.info(f"Running evaluation at iteration {iteration}...")

        # Evaluate on a subset per language
        eval_results = {}
        for lang in self.train_dataset.languages:
            lang_items = [
                self.test_dataset[i] for i in range(len(self.test_dataset))
                if self.test_dataset[i]["lang"] == lang
            ]

            if not lang_items:
                continue

            prompts = [item["prompt"] for item in lang_items]
            answer_numbers = [item["answer_number"] for item in lang_items]

            # Generate single response (greedy) for evaluation
            original_group_size = self.model_mgr.group_size
            self.model_mgr.group_size = 1
            rollouts = self.model_mgr.generate_rollouts(prompts)
            self.model_mgr.group_size = original_group_size

            correct = 0
            for i, rollout in enumerate(rollouts):
                extracted = self.reward_mgr.extract_final_answer(rollout["response_texts"][0])
                if extracted is not None and extracted == answer_numbers[i]:
                    correct += 1

            accuracy = correct / len(lang_items)
            eval_results[lang] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": len(lang_items),
            }
            self.logger.info(f"  eval/{lang}: accuracy={accuracy:.3f} ({correct}/{len(lang_items)})")

        # Save eval results
        eval_path = self.output_dir / f"eval_iter_{iteration}.json"
        with eval_path.open("w") as f:
            json.dump(eval_results, f, indent=2)

        return eval_results

    def train(self):
        """Main GRPO training loop.

        Outer loop: iterations (each = one epoch over all N training examples)
            - Collect fresh rollouts from π_θ^old
            - Inner loop: gradient steps over minibatches
            - At end of iteration: π_θ^old ← π_θ (implicit, since rollouts are re-collected)

        From page 7 of notes:
            N = 200 (per lang), G = 4, B = batch_size
            Steps per iteration = N_total / B
            Checkpoint every checkpoint_save_freq steps
        """
        all_metrics = []
        total_examples = len(self.train_dataset)
        steps_per_iteration = total_examples // self.batch_size

        self.logger.info("=" * 80)
        self.logger.info("Starting GRPO training")
        self.logger.info(f"  Total examples: {total_examples}")
        self.logger.info(f"  Steps per iteration (epoch): {steps_per_iteration}")
        self.logger.info(f"  Num iterations: {self.num_iterations}")
        self.logger.info(f"  Total gradient steps: {steps_per_iteration * self.num_iterations}")
        self.logger.info("=" * 80)

        # Determine starting iteration if resuming
        start_iteration = self.global_step // steps_per_iteration if self.global_step > 0 else 0

        for iteration in range(start_iteration, self.num_iterations):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Iteration {iteration + 1}/{self.num_iterations} "
                             f"(collecting fresh rollouts)")
            self.logger.info(f"{'='*80}")

            # Shuffle training examples for this iteration
            indices = self._shuffle_indices(total_examples, seed=iteration)

            # Create minibatches
            num_batches = total_examples // self.batch_size
            batches = [
                indices[b * self.batch_size: (b + 1) * self.batch_size]
                for b in range(num_batches)
            ]

            # Progress bar for this iteration
            pbar = tqdm(
                batches,
                desc=f"Iter {iteration + 1}/{self.num_iterations}",
                total=num_batches,
            )

            for batch_indices in pbar:
                # Step 1: Collect rollouts from current policy (= π_θ^old for this batch)
                rollout_data = self._collect_rollouts(batch_indices)

                # Step 2: Compute GRPO loss and update
                metrics = self._gradient_step(rollout_data)

                # Update progress bar
                pbar.set_postfix({
                    "loss": metrics["loss"],
                    "reward": metrics["avg_reward"],
                    "correct": metrics["num_correct"],
                    "total": metrics["num_total"],
                    "step": metrics["step"],
                })

                # Log metrics
                if self.global_step % self.log_every_n_steps == 0:
                    self._log_metrics(metrics, iteration + 1)

                all_metrics.append(metrics)

                # Save checkpoint
                if self.global_step % self.checkpoint_save_freq == 0:
                    self.model_mgr.save_checkpoint(
                        step=self.global_step,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        extra_state={
                            "iteration": iteration + 1,
                            "avg_reward": metrics["avg_reward"],
                            "loss": metrics["loss"],
                        },
                    )
                    self._save_metrics_json(all_metrics)

            pbar.close()

            # End of iteration: evaluate
            self._evaluate(iteration + 1)

            # Log iteration summary
            iter_metrics = [m for m in all_metrics if m["step"] > (iteration * steps_per_iteration)]
            if iter_metrics:
                avg_loss = sum(m["loss"] for m in iter_metrics) / len(iter_metrics)
                avg_reward = sum(m["avg_reward"] for m in iter_metrics) / len(iter_metrics)
                total_correct = sum(m["num_correct"] for m in iter_metrics)
                total_total = sum(m["num_total"] for m in iter_metrics)
                self.logger.info(
                    f"Iteration {iteration + 1} summary: "
                    f"avg_loss={avg_loss:.4f}, avg_reward={avg_reward:.3f}, "
                    f"correct={total_correct}/{total_total} "
                    f"({100*total_correct/total_total:.1f}%)"
                )

        # Final checkpoint
        self.model_mgr.save_checkpoint(
            step=self.global_step,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            extra_state={"iteration": self.num_iterations, "final": True},
        )
        self._save_metrics_json(all_metrics)

        self.logger.info("Training complete.")


def main():
    config = {
        # ---- Training hyperparameters ----
        "num_iterations": 16,       # Number of epochs/iterations (page 7: 500 steps / 31 steps_per_iter ≈ 16)
        "batch_size": 8,            # B in notes (page 7)
        "group_size": 4,            # G in notes (page 7)
        "epsilon": 0.2,             # ε for clipping (page 5)
        "beta": 0.04,               # β for KL penalty (page 6)
        "learning_rate": 1e-5,
        "max_grad_norm": 1.0,

        # ---- Checkpoint and logging ----
        "checkpoint_save_freq": 10,  # Save every N gradient steps (page 7: 10/20/30)
        "resume_from_step": None,    # Set to step number to resume
        "log_every_n_steps": 1,
        "log_dir": "./exp2/logs",
        "output_dir": "./exp2/outputs",

        # ---- Model config (passed to GRPOModelManager) ----
        "model_config": {
            "model_name": "./models/Qwen2.5-7B-Instruct",
            "policy_device": "cuda:0",
            "reference_device": "cuda:0",
            "dtype": "float16",

            # LoRA
            "lora_rank": 64,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],

            # Quantization
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
        },

        # ---- Reward config (passed to RewardManager) ----
        "reward_config": {
            "judge_model_name": "./models/gemma-4-31b-it",
            "judge_device": "cuda:0",
            "judge_dtype": "float16",
            "judge_load_in_4bit": True,
            "judge_max_new_tokens": 32,
            "log_dir": "./exp2/logs",
            "log_max_response_chars": 500,

            # Phase 1: accuracy only via regex
            "weight_accuracy_regex": 0.0,
            "weight_accuracy_judge": 1.0,
            "weight_format": 0.0,
            "weight_language_consistency": 0.0,
        },

        # ---- Dataset config (passed to GRPOMGSMDataset) ----
        "dataset_config": {
            "data_dir": "./exp2/data",
            "log_dir": "./exp2/logs",
            "languages": ["en", "bn", "te", "th", "ru", "ja", "zh"],
            "num_few_shot": 1,
            "seed": 42,
        },
    }

    # GPU check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logging.error("No GPU available. GRPO training requires GPU. Exiting.")
        return

    num_gpus = torch.cuda.device_count()
    logging.info(f"GPUs available: {num_gpus}")

    if num_gpus < 2:
        logging.warning("Only 1 GPU available. Putting ref model on cuda:0 alongside policy.")
        config["model_config"]["reference_device"] = "cuda:0"
        config["reward_config"]["judge_device"] = "cuda:0"

    # Ensure group_size is consistent
    config["model_config"]["group_size"] = config["group_size"]

    trainer = GRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()