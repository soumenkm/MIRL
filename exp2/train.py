"""GRPO training loop for cross-lingual collapse experiments.

Implements the exact GRPO formulation from the handwritten notes (pages 3-7):
    1. Rollout collection: Sample G responses per prompt from π_θ^old
    2. Reward computation: Get R^i_j via regex / remote vLLM judge
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

Phase 1 architecture:
    - Policy + reference live in this SLURM job (this file).
    - Judge LLM lives in a SEPARATE SLURM job (see judge_server.py).
    - This file's RewardManager talks to the judge over HTTP via a
      connection file written by the judge job. RewardManager's __init__
      blocks until the judge is healthy, so it is fine to submit both
      jobs at once — this trainer will wait.
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
from gpu_monitor import GPUMonitor
from models import GRPOModelManager
from rewards import RewardManager


class GRPOTrainer:
    """GRPO training loop following the handwritten notes formulation."""

    def __init__(self, config: dict):
        self.config = config
        self.log_dir = Path(config["log_dir"])
        self.output_dir = Path(config["output_dir"])
        self.data_dir = Path(config["data_dir"])

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
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        # Initialize components.
        # Order matters: reward manager comes BEFORE the model manager so the
        # trainer blocks on judge discovery first. If we built the policy/ref
        # models first and the judge never came up, we would have wasted the
        # ~minute of GPU loading time before failing.

        # Start GPU monitor BEFORE any heavy loading so the 'init' phase
        # captures the memory cost of bringing models up. The monitor is a
        # daemon thread inside this process; it dies with the trainer.
        self.logger.info("Starting GPU monitor...")
        self.gpu_monitor = GPUMonitor(config["monitor_config"])
        self.gpu_monitor.set_phase("init", step=0)
        self.gpu_monitor.start()

        self.logger.info("Initializing reward manager (will block on judge discovery)...")
        self.reward_mgr = RewardManager(config["reward_config"])

        self.logger.info("Initializing model manager...")
        self.model_mgr = GRPOModelManager(config["model_config"])

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

        self.logger.info(
            f"Training config: iterations={self.num_iterations}, "
            f"batch_size={self.batch_size}, group_size={self.group_size}, "
            f"epsilon={self.epsilon}, beta={self.beta}, lr={self.learning_rate}"
        )
        self.logger.info(f"Total training examples: {len(self.train_dataset)}")
        self.logger.info(
            f"Steps per iteration: {len(self.train_dataset) // self.batch_size}"
        )
        self.logger.info(f"Estimated total steps: {total_steps}")
        self.logger.info(
            f"Checkpoint save frequency: every {self.checkpoint_save_freq} steps"
        )

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

        Reward computation is BATCHED across the entire B*G grid in a single
        compute_rewards() call. The previous version called compute_rewards
        per-(prompt, group-index), defeating the JudgeClient's thread-pool
        concurrency: B=8 G=4 used to be 32 sequential single-prompt HTTP
        calls; now it is one call with 32 prompts processed concurrently.
        """
        batch_items = [self.train_dataset[i] for i in batch_indices]
        prompts = [item["prompt"] for item in batch_items]
        answer_numbers = [item["answer_number"] for item in batch_items]
        langs = [item["lang"] for item in batch_items]

        # Generate G rollouts per prompt
        rollouts = self.model_mgr.generate_rollouts(prompts)

        # Flatten (prompt_i, group_j) -> single list for batched reward computation.
        flat_response_texts: list[str] = []
        flat_answer_numbers: list[int] = []
        flat_langs: list[str] = []
        for i, rollout in enumerate(rollouts):
            for g in range(self.group_size):
                flat_response_texts.append(rollout["response_texts"][g])
                flat_answer_numbers.append(answer_numbers[i])
                flat_langs.append(langs[i])

        # ONE batched call -> all (B*G) judge requests issued concurrently.
        flat_results = self.reward_mgr.compute_rewards(
            response_texts=flat_response_texts,
            answer_numbers=flat_answer_numbers,
            langs=flat_langs,
        )

        # Reshape flat results back to [B][G]
        all_rewards: list[list[float]] = []
        all_response_texts: list[list[str]] = []
        for i, rollout in enumerate(rollouts):
            group_rewards = [
                flat_results[i * self.group_size + g]["reward"]
                for g in range(self.group_size)
            ]
            all_rewards.append(group_rewards)
            all_response_texts.append(rollout["response_texts"])

        # Aggregate stats for this batch
        self.reward_mgr.log_reward_summary(flat_results)

        # Compute group-normalized advantages (page 4 of notes)
        # μ^i = Σ_j R^i_j / G
        # σ^i = sqrt(Σ_j (R^i_j - μ^i)^2 / G)
        # A^i_j = (R^i_j - μ^i) / σ^i
        all_advantages = []
        for group_rewards in all_rewards:
            rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32)
            mu = rewards_tensor.mean()
            sigma = rewards_tensor.std(correction=0)  # population std

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

    def _compute_and_backward_grpo_loss(self, rollout_data: dict) -> float:
        """Compute the GRPO loss and run backward pass with memory-efficient
        gradient accumulation across the B*G inner loop (page 6 of notes).

        L = (1/B) Σ_i (1/G) Σ_j (1/|y^i_j|) Σ_t [ min(r_{j,t} · A_j, g_{j,t})
                                                  - β · D_KL^t ]

        By linearity of differentiation:
            ∇L = Σ_{i,j} ∇L_ij / (B*G)

        So instead of building the full (B*G)-graph and calling backward once
        (which keeps activations from all 32 forwards alive simultaneously,
        ~48 GB on Qwen2.5-7B fp16 → OOM on 80 GB A100), we backward each
        sub-loss immediately. PyTorch's .backward() accumulates into .grad;
        the autograd graph from each sub-forward is freed before the next
        forward starts, so peak activation memory drops from B*G * per_fwd to
        just per_fwd (~1.5 GB).

        This is mathematically identical to the previous implementation —
        same gradients, same trained weights modulo float-summation order
        (which is at noise level). One optimizer.step() per call still =
        one GRPO step. Checkpoint-every-N-steps semantics unchanged.

        Returns:
            float — the (scalar) total loss value, for logging only.
        """
        B = len(rollout_data["prompt_ids_list"])
        G = self.group_size
        scale = 1.0 / (B * G)

        total_loss_value = 0.0  # for logging only — no autograd needed

        for i in range(B):
            prompt_ids = rollout_data["prompt_ids_list"][i]

            for j in range(G):
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
                log_ratio = current_log_probs - old_log_probs_device
                ratio = torch.exp(log_ratio)

                # Clipped objective (page 5-6 of notes)
                if A_jt >= 0:
                    clipped_value = (1 + self.epsilon) * A_jt
                else:
                    clipped_value = (1 - self.epsilon) * A_jt

                surrogate = torch.min(
                    ratio * A_jt,
                    torch.tensor(clipped_value, device=ratio.device),
                )

                # KL divergence (page 6 of notes):
                # s_t = π_ref / π_θ ; D_KL^t = s_t - log(s_t) - 1
                log_s = ref_log_probs - current_log_probs
                s_t = torch.exp(log_s)
                kl_per_token = s_t - log_s - 1

                # Per-response loss: (1/|y^i_j|) Σ_t [ surrogate - β · D_KL^t ]
                per_token_objective = surrogate - self.beta * kl_per_token
                response_loss = per_token_objective.sum() / response_len

                # Negate (we MAXIMIZE the objective ⇒ MINIMIZE its negative)
                # and scale by 1/(B*G) so gradients sum to ∇L (not ∇(B*G·L)).
                sub_loss = -response_loss * scale

                # CRITICAL: backward NOW. This frees the autograd graph for
                # this single (i, j) sub-forward before the next iteration
                # builds a new one. Activation memory peak: 1 forward, not 32.
                sub_loss.backward()

                # Accumulate the value for logging. .item() detaches from
                # the graph and returns a Python float, so this does not
                # extend the autograd graph or hold tensor memory.
                total_loss_value += sub_loss.item()

        return total_loss_value

    def _gradient_step(self, rollout_data: dict) -> dict:
        """Perform one gradient update step (= one GRPO step).

        Order:
          1. zero_grad
          2. _compute_and_backward_grpo_loss: forward + backward each of the
             B*G sub-losses sequentially, accumulating gradients into .grad.
             Memory-efficient: peak activations = 1 forward, not B*G.
          3. clip_grad_norm on the accumulated gradient
          4. optimizer.step + scheduler.step
          5. global_step += 1   <-- this IS what defines a GRPO step.

        The checkpoint cadence (every checkpoint_save_freq global_steps) is
        completely unaffected by the gradient-accumulation refactor.

        Returns:
            Dictionary of metrics for logging.
        """
        self.optimizer.zero_grad()

        # Forward + backward for each (i, j) sub-loss, accumulating .grad.
        # Returns the scalar loss value (Python float) for logging.
        loss_value = self._compute_and_backward_grpo_loss(rollout_data)

        # Gradient clipping operates on the ALREADY-accumulated .grad, so
        # this clips the gradient of the full GRPO loss — same as before.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model_mgr.policy_model.parameters() if p.requires_grad],
            self.max_grad_norm,
        )

        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        # Compute metrics
        all_rewards_flat = [r for group in rollout_data["rewards"] for r in group]
        all_advantages_flat = [
            a for group in rollout_data["advantages"] for a in group
        ]

        metrics = {
            "step": self.global_step,
            "loss": loss_value,
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
        """Run evaluation on the test set and log results.

        Eval uses regex-based answer extraction (deterministic), NOT the
        judge LLM. Judge agreement is a training-time signal; evaluation
        accuracy is the ground-truth metric we track across iterations.
        """
        self.logger.info(f"Running evaluation at iteration {iteration}...")

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
                extracted = self.reward_mgr.extract_final_answer(
                    rollout["response_texts"][0]
                )
                if extracted is not None and extracted == answer_numbers[i]:
                    correct += 1

            accuracy = correct / len(lang_items)
            eval_results[lang] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": len(lang_items),
            }
            self.logger.info(
                f"  eval/{lang}: accuracy={accuracy:.3f} "
                f"({correct}/{len(lang_items)})"
            )

        eval_path = self.output_dir / f"eval_iter_{iteration}.json"
        with eval_path.open("w") as f:
            json.dump(eval_results, f, indent=2)
        return eval_results

    def train(self):
        """Main GRPO training loop.

        Outer loop: iterations (each = one epoch over all N training examples)
            - Collect fresh rollouts from π_θ^old
            - Inner loop: gradient steps over minibatches
            - At end of iteration: π_θ^old ← π_θ (implicit, since rollouts
              are re-collected next iteration)

        From page 7 of notes:
            N = 200 (per lang), G = 4, B = batch_size
            Steps per iteration = N_total / B
            Checkpoint every checkpoint_save_freq steps

        Wrapped in try/finally so the GPU monitor flushes its CSV and joins
        cleanly even if training crashes (the CSV around the moment of an
        OOM is exactly what we want to inspect).
        """
        try:
            self._train_loop()
        finally:
            self.gpu_monitor.set_phase("shutdown", step=self.global_step)
            self.gpu_monitor.stop()

    def _train_loop(self):
        """Inner training loop (separated from train() for monitor cleanup)."""
        all_metrics = []
        total_examples = len(self.train_dataset)
        steps_per_iteration = total_examples // self.batch_size

        self.logger.info("=" * 80)
        self.logger.info("Starting GRPO training")
        self.logger.info(f"  Total examples: {total_examples}")
        self.logger.info(f"  Steps per iteration (epoch): {steps_per_iteration}")
        self.logger.info(f"  Num iterations: {self.num_iterations}")
        self.logger.info(
            f"  Total gradient steps: {steps_per_iteration * self.num_iterations}"
        )
        self.logger.info("=" * 80)

        # Determine starting iteration if resuming
        start_iteration = (
            self.global_step // steps_per_iteration if self.global_step > 0 else 0
        )

        for iteration in range(start_iteration, self.num_iterations):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(
                f"Iteration {iteration + 1}/{self.num_iterations} "
                f"(collecting fresh rollouts)"
            )
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
                unit="batch",
                total=num_batches,
            )

            for batch_indices in pbar:
                # Step 1: Collect rollouts from current policy (= π_θ^old for this batch)
                # 'rollout' covers BOTH policy generation and the remote
                # judge call. Generation dominates GPU memory; the judge is
                # over HTTP and uses no local GPU.
                self.gpu_monitor.set_phase("rollout", step=self.global_step)
                rollout_data = self._collect_rollouts(batch_indices)

                # Step 2: Compute GRPO loss and update
                # 'backward' covers the B*G sub-forwards-with-grad, the per-
                # sub-loss backward calls, gradient clipping, and the
                # optimizer step. With Solution A (gradient accumulation),
                # peak memory here should be ~30 GB instead of 77.5 GB.
                self.gpu_monitor.set_phase("backward", step=self.global_step)
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
                    self.gpu_monitor.set_phase("checkpoint", step=self.global_step)
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
            self.gpu_monitor.set_phase("eval", step=self.global_step)
            self._evaluate(iteration + 1)

            # Log iteration summary
            iter_metrics = [
                m for m in all_metrics
                if m["step"] > (iteration * steps_per_iteration)
            ]
            if iter_metrics:
                avg_loss = sum(m["loss"] for m in iter_metrics) / len(iter_metrics)
                avg_reward = (
                    sum(m["avg_reward"] for m in iter_metrics) / len(iter_metrics)
                )
                total_correct = sum(m["num_correct"] for m in iter_metrics)
                total_total = sum(m["num_total"] for m in iter_metrics)
                self.logger.info(
                    f"Iteration {iteration + 1} summary: "
                    f"avg_loss={avg_loss:.4f}, avg_reward={avg_reward:.3f}, "
                    f"correct={total_correct}/{total_total} "
                    f"({100 * total_correct / total_total:.1f}%)"
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main_logger = logging.getLogger("Main")

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected.")

    num_gpus = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    main_logger.info(f"GPUs available: {num_gpus} ({gpu_name})")

    config = {
        # ---- Required path roots (per coding standards) ----
        "log_dir":    Path("./exp2/logs"),
        "output_dir": Path("./exp2/outputs"),
        "data_dir":   Path("./exp2/data"),

        # ---- Training hyperparameters ----
        "num_iterations": 4,        # 500 grad steps 
        "batch_size": 8,             # B in notes (page 7)
        "group_size": 4,             # G in notes (page 7)
        "epsilon": 0.2,              # ε for clipping (page 5)
        "beta": 0.04,                # β for KL penalty (page 6)
        "learning_rate": 1e-5,
        "max_grad_norm": 1.0,

        # ---- Checkpoint and logging ----
        "checkpoint_save_freq": 20,  # save every N gradient steps
        "resume_from_step": None,
        "log_every_n_steps": 1,

        # ---- Model config (passed to GRPOModelManager) ----
        "model_config": {
            "model_name": Path("./models/Qwen2.5-7B-Instruct"),
            "policy_device": "cuda:0",
            "reference_device": "cuda:0",
            "dtype": "float16",

            # LoRA
            "lora_rank": 64,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],

            # Quantization
            "policy_load_in_4bit": False,
            "reference_load_in_4bit": False,

            # Generation
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95,
            "group_size": 4,
            "tokens_after_final_answer": 20,

            # Logging
            "log_max_response_chars": 500,

            # Checkpointing
            "checkpoint_dir": Path("./exp2/outputs/checkpoints"),
            "log_dir":        Path("./exp2/logs"),
        },

        # ---- Reward config (passed to RewardManager) ----
        # The judge is now a remote vLLM server (its own SLURM job).
        # RewardManager.__init__ blocks on JudgeClient.discover() until the
        # judge advertises ready=True via the connection file.
        "reward_config": {
            "log_dir":                Path("./exp2/logs"),
            "log_max_response_chars": 500,

            # Phase 1: judge-only accuracy
            "weight_accuracy":       1.0,
            "weight_accuracy_regex": 0.0,
            "weight_accuracy_judge": 0.0,
            "weight_format":               0.0,
            "weight_language_consistency": 0.0,

            # Judge generation knobs
            "judge_max_tokens":  8,
            "judge_temperature": 0.0,

            # Sub-config consumed by JudgeClient
            "judge_config": {
                "connection_file":    Path("./exp2/outputs/judge_connection.json"),
                "log_dir":            Path("./exp2/logs"),
                "discovery_timeout":         1800,   # 30 min for judge to come up
                "discovery_poll_interval":   5.0,
                "request_timeout":           120,
                "max_retries":               3,
                "retry_backoff":             2.0,
                "max_concurrent_requests":   32,     # B*G upper bound for B=8 G=4
            },
        },

        # ---- Dataset config (passed to GRPOMGSMDataset) ----
        "dataset_config": {
            "data_dir": Path("./exp2/data"),
            "log_dir":  Path("./exp2/logs"),
            "languages": ["en", "bn", "te", "th", "ru", "ja", "zh"],
            "num_few_shot": 1,
            "seed": 42,
        },

        # ---- GPU monitor config (passed to GPUMonitor) ----
        # Daemon thread sampling memory + utilization to CSV. CSV is
        # truncated at start of every run -> rerunning train.py wipes the
        # previous run's data.
        "monitor_config": {
            "csv_path":   Path("./exp2/logs/gpu_monitor.csv"),
            "log_dir":    Path("./exp2/logs"),
            "interval_s": 2.0,
            "gpu_ids":    None,   # None = all visible GPUs
        },
    }

    # Single-GPU fallback for the trainer side. The judge is on a different
    # node entirely, so its placement is unaffected.
    if num_gpus < 2:
        main_logger.warning(
            "Only 1 GPU on trainer node. Putting ref model on cuda:0 alongside policy."
        )
        config["model_config"]["reference_device"] = "cuda:0"

    # Ensure group_size is consistent
    config["model_config"]["group_size"] = config["group_size"]

    trainer = GRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()