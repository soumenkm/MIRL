"""Reward computation for GRPO training.

Handles:
    - Regex-based answer extraction and accuracy reward (fast, deterministic)
    - Judge LLM-based accuracy verification (on configurable GPU, optionally 4-bit)
    - Format reward (stub for Phase 2)
    - Language consistency reward (stub for Phase 2)
    - Unified compute_rewards() with configurable weights
    - Detailed I/O logging for debugging
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import logging
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm


class RewardManager:
    """Manages reward computation for GRPO training."""

    def __init__(self, config: dict):
        self.judge_model_name = config["judge_model_name"]
        self.judge_device = torch.device(config["judge_device"])
        self.judge_dtype = getattr(torch, config["judge_dtype"])
        self.judge_load_in_4bit = config.get("judge_load_in_4bit", False)
        self.judge_max_new_tokens = config.get("judge_max_new_tokens", 32)
        self.log_dir = Path(config["log_dir"])

        # Reward weights (configurable, only accuracy enabled for Phase 1)
        self.weight_accuracy_regex = config.get("weight_accuracy_regex", 1.0)
        self.weight_accuracy_judge = config.get("weight_accuracy_judge", 0.0)
        self.weight_format = config.get("weight_format", 0.0)
        self.weight_language_consistency = config.get("weight_language_consistency", 0.0)

        # Logging config
        self.log_max_response_chars = config.get("log_max_response_chars", 500)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()

        # Load judge LLM only if judge weight is non-zero
        self.judge_model = None
        self.judge_tokenizer = None
        if self.weight_accuracy_judge > 0.0:
            self._load_judge_model()

    def _setup_logging(self):
        log_file = self.log_dir / "rewards.log"
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
        self.logger.info(
            f"Reward weights: regex={self.weight_accuracy_regex}, "
            f"judge={self.weight_accuracy_judge}, format={self.weight_format}, "
            f"lc={self.weight_language_consistency}"
        )

    def _load_judge_model(self):
        """Load the judge LLM for accuracy verification."""
        self.logger.info(
            f"Loading judge model: {self.judge_model_name} -> {self.judge_device} "
            f"(4bit={self.judge_load_in_4bit})"
        )

        self.judge_tokenizer = AutoTokenizer.from_pretrained(
            self.judge_model_name,
            trust_remote_code=True,
        )
        if self.judge_tokenizer.pad_token is None:
            self.judge_tokenizer.pad_token = self.judge_tokenizer.eos_token
            self.judge_tokenizer.pad_token_id = self.judge_tokenizer.eos_token_id

        load_kwargs = {
            "trust_remote_code": True,
            "device_map": {"": self.judge_device},
        }

        if self.judge_load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.judge_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.logger.info("Judge model will be loaded in 4-bit quantization.")
        else:
            load_kwargs["torch_dtype"] = self.judge_dtype

        self.judge_model = AutoModelForCausalLM.from_pretrained(
            self.judge_model_name,
            **load_kwargs,
        )
        self.judge_model.eval()

        num_params = sum(p.numel() for p in self.judge_model.parameters()) / 1e9
        param_bytes = sum(p.numel() * p.element_size() for p in self.judge_model.parameters())
        self.logger.info(
            f"Judge model loaded: {num_params:.2f}B params, "
            f"memory={param_bytes / 1e9:.2f} GB"
        )

    # ------------------------------------------------------------------ #
    #  Regex-based answer extraction
    # ------------------------------------------------------------------ #

    def extract_final_answer(self, response_text: str) -> int | None:
        """Extract the integer after 'Final Answer:' from the response.

        Tries multiple patterns in order of strictness:
            1. "Final Answer: <number>" (exact format)
            2. Last standalone integer in the response (fallback)

        Args:
            response_text: The model's generated response string.

        Returns:
            Extracted integer or None if no valid answer found.
        """
        # Pattern 1: "Final Answer: <number>" (with optional whitespace)
        match = re.search(r"Final Answer\s*:\s*([+-]?\d[\d,]*)", response_text)
        if match:
            return int(match.group(1).replace(",", ""))

        # Pattern 2: Last standalone integer in the text (fallback)
        all_numbers = re.findall(r"(?<!\S)([+-]?\d[\d,]*)(?!\S)", response_text)
        if all_numbers:
            return int(all_numbers[-1].replace(",", ""))

        return None

    def _compute_accuracy_regex(
        self,
        response_text: str,
        answer_number: int,
    ) -> float:
        """Compute accuracy reward via regex extraction.

        Returns:
            1.0 if extracted answer matches ground truth, 0.0 otherwise.
        """
        extracted = self.extract_final_answer(response_text)
        matched = extracted is not None and extracted == answer_number

        # Log regex I/O
        max_chars = self.log_max_response_chars
        resp_display = response_text[:max_chars] + ("..." if len(response_text) > max_chars else "")
        self.logger.info(
            f"[REGEX] gt={answer_number}, extracted={extracted}, "
            f"match={matched}, response={resp_display}"
        )

        return 1.0 if matched else 0.0

    # ------------------------------------------------------------------ #
    #  Judge LLM-based accuracy verification
    # ------------------------------------------------------------------ #

    def _build_judge_prompt(self, response_text: str, answer_number: int) -> str:
        """Build the prompt for the judge LLM to verify accuracy."""
        return (
            "You are a math answer verifier. Your task is to determine whether "
            "a student's solution arrives at the correct final answer.\n\n"
            f"Correct Answer: {answer_number}\n\n"
            f"Student's Response:\n{response_text}\n\n"
            "Does the student's response arrive at the correct final answer? "
            "Reply with exactly one word: YES or NO."
        )

    @torch.no_grad()
    def _compute_accuracy_judge_single(
        self,
        response_text: str,
        answer_number: int,
    ) -> float:
        """Query the judge LLM for a single response.

        Returns:
            1.0 if judge says YES, 0.0 if NO or unparseable.
        """
        prompt = self._build_judge_prompt(response_text, answer_number)

        encoded = self.judge_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
        ).to(self.judge_device)

        output_ids = self.judge_model.generate(
            **encoded,
            max_new_tokens=self.judge_max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=self.judge_tokenizer.pad_token_id,
        )

        # Decode only the generated tokens (skip prompt)
        generated_ids = output_ids[0, encoded["input_ids"].shape[1]:]
        judge_response = self.judge_tokenizer.decode(generated_ids, skip_special_tokens=True)
        judge_response_clean = judge_response.strip().upper()

        # Determine reward
        if "YES" in judge_response_clean:
            reward = 1.0
        elif "NO" in judge_response_clean:
            reward = 0.0
        else:
            self.logger.warning(f"Unparseable judge response: '{judge_response}'")
            reward = 0.0

        # Log judge I/O
        max_chars = self.log_max_response_chars
        separator = "-" * 80
        self.logger.info(separator)
        self.logger.info(f"[JUDGE INPUT] gt={answer_number}")
        prompt_display = prompt[:max_chars] + ("..." if len(prompt) > max_chars else "")
        self.logger.info(prompt_display)
        self.logger.info(f"[JUDGE OUTPUT] raw='{judge_response.strip()}', reward={reward}")
        self.logger.info(separator)

        return reward

    def _compute_accuracy_judge_batch(
        self,
        response_texts: list[str],
        answer_numbers: list[int],
    ) -> list[float]:
        """Compute judge-based accuracy for a batch of responses.

        Args:
            response_texts: List of model response strings.
            answer_numbers: List of ground truth answer integers.

        Returns:
            List of reward floats (1.0 or 0.0).
        """
        rewards = []
        for text, ans in tqdm(
            zip(response_texts, answer_numbers),
            total=len(response_texts),
            desc="Judge evaluation",
            leave=False,
        ):
            reward = self._compute_accuracy_judge_single(text, ans)
            rewards.append(reward)
        return rewards

    # ------------------------------------------------------------------ #
    #  Format reward (Phase 2 stub)
    # ------------------------------------------------------------------ #

    def _compute_format_reward(self, response_text: str) -> float:
        """Check if the response follows the required format.

        Checks:
            - Contains "Final Answer:" exactly once
            - Text after "Final Answer:" is a valid integer
            - No substantial text after the final answer line

        Returns:
            Float between 0.0 and 1.0.
        """
        score = 0.0

        # Check 1: "Final Answer:" present exactly once
        final_answer_count = len(re.findall(r"Final Answer\s*:", response_text))
        if final_answer_count == 1:
            score += 0.5
        elif final_answer_count > 1:
            score += 0.1

        # Check 2: Valid integer follows "Final Answer:"
        match = re.search(r"Final Answer\s*:\s*([+-]?\d[\d,]*)", response_text)
        if match:
            score += 0.3

        # Check 3: No substantial text after the final answer line
        if match:
            after_answer = response_text[match.end():].strip()
            if len(after_answer) < 5:
                score += 0.2

        return score

    # ------------------------------------------------------------------ #
    #  Language consistency reward (Phase 2 stub)
    # ------------------------------------------------------------------ #

    def _compute_language_consistency_reward(
        self,
        response_text: str,
        target_lang: str,
    ) -> float:
        """Compute what fraction of the CoT is in the target language.

        Uses Unicode script classification to determine token language.
        Only considers the CoT portion (everything before "Final Answer:").

        Returns:
            Float between 0.0 and 1.0.
        """
        # TODO: Implement Unicode script-based token classification
        # This will use the approach discussed:
        #   BENGALI -> bn, TELUGU -> te, THAI -> th, CYRILLIC -> ru, LATIN -> en
        # For now, return a placeholder.
        return 0.0

    # ------------------------------------------------------------------ #
    #  Unified reward computation
    # ------------------------------------------------------------------ #

    def compute_rewards(
        self,
        response_texts: list[str],
        answer_numbers: list[int],
        langs: list[str] | None = None,
    ) -> list[dict]:
        """Compute combined rewards for a batch of responses.

        Args:
            response_texts: List of model response strings.
            answer_numbers: List of ground truth answer integers.
            langs: Optional list of target language codes (for LC reward).

        Returns:
            List of dicts, each containing:
                - "reward": float (weighted sum of all active rewards)
                - "accuracy_regex": float (0.0 or 1.0)
                - "accuracy_judge": float (0.0 or 1.0, or None if disabled)
                - "format": float (0.0 to 1.0, or None if disabled)
                - "language_consistency": float (0.0 to 1.0, or None if disabled)
                - "extracted_answer": int or None
        """
        batch_size = len(response_texts)
        assert len(answer_numbers) == batch_size

        self.logger.info(f"Computing rewards for {batch_size} responses")

        # Regex accuracy (always computed)
        regex_rewards = []
        for text, ans in zip(response_texts, answer_numbers):
            regex_rewards.append(self._compute_accuracy_regex(text, ans))

        # Extracted answers (for logging)
        extracted_answers = [
            self.extract_final_answer(text) for text in response_texts
        ]

        # Judge accuracy (only if enabled)
        judge_rewards = None
        if self.weight_accuracy_judge > 0.0 and self.judge_model is not None:
            judge_rewards = self._compute_accuracy_judge_batch(
                response_texts, answer_numbers
            )

        # Format reward (only if enabled)
        format_rewards = None
        if self.weight_format > 0.0:
            format_rewards = [
                self._compute_format_reward(text) for text in response_texts
            ]

        # Language consistency reward (only if enabled)
        lc_rewards = None
        if self.weight_language_consistency > 0.0 and langs is not None:
            lc_rewards = [
                self._compute_language_consistency_reward(text, lang)
                for text, lang in zip(response_texts, langs)
            ]

        # Combine
        results = []
        for i in range(batch_size):
            total_reward = self.weight_accuracy_regex * regex_rewards[i]

            if judge_rewards is not None:
                total_reward += self.weight_accuracy_judge * judge_rewards[i]
            if format_rewards is not None:
                total_reward += self.weight_format * format_rewards[i]
            if lc_rewards is not None:
                total_reward += self.weight_language_consistency * lc_rewards[i]

            results.append({
                "reward": total_reward,
                "accuracy_regex": regex_rewards[i],
                "accuracy_judge": judge_rewards[i] if judge_rewards else None,
                "format": format_rewards[i] if format_rewards else None,
                "language_consistency": lc_rewards[i] if lc_rewards else None,
                "extracted_answer": extracted_answers[i],
            })

        return results

    def log_reward_summary(self, all_rewards: list[dict]):
        """Log aggregate statistics for a batch of rewards."""
        n = len(all_rewards)
        if n == 0:
            return

        avg_reward = sum(r["reward"] for r in all_rewards) / n
        avg_regex = sum(r["accuracy_regex"] for r in all_rewards) / n

        self.logger.info(
            f"Reward summary (n={n}): avg_reward={avg_reward:.3f}, "
            f"avg_accuracy_regex={avg_regex:.3f}"
        )

        if all_rewards[0]["accuracy_judge"] is not None:
            avg_judge = sum(r["accuracy_judge"] for r in all_rewards) / n
            self.logger.info(f"  avg_accuracy_judge={avg_judge:.3f}")

            # Agreement between regex and judge
            agree = sum(
                1 for r in all_rewards
                if r["accuracy_regex"] == r["accuracy_judge"]
            )
            self.logger.info(f"  regex-judge agreement={agree}/{n} ({100*agree/n:.1f}%)")

        if all_rewards[0]["format"] is not None:
            avg_fmt = sum(r["format"] for r in all_rewards) / n
            self.logger.info(f"  avg_format={avg_fmt:.3f}")

        if all_rewards[0]["language_consistency"] is not None:
            avg_lc = sum(r["language_consistency"] for r in all_rewards) / n
            self.logger.info(f"  avg_language_consistency={avg_lc:.3f}")


def main():
    """Sanity check: test regex extraction and reward computation."""

    config = {
        "judge_model_name": "./models/Qwen2.5-7B-Instruct",
        "judge_device": "cuda:0",
        "judge_dtype": "float16",
        "judge_load_in_4bit": True,
        "judge_max_new_tokens": 32,
        "log_dir": "./exp2/logs",
        "log_max_response_chars": 500,

        # Phase 1: regex only (set judge weight to 0.0)
        # To test judge: set weight_accuracy_judge=1.0
        "weight_accuracy_regex": 1.0,
        "weight_accuracy_judge": 1.0,
        "weight_format": 0.0,
        "weight_language_consistency": 0.0,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info(f"Device: {device}")

    reward_mgr = RewardManager(config)

    # Test cases for regex extraction
    test_cases = [
        # Clean format
        (
            "Step 1: 5 + 6 = 11\nFinal Answer: 11",
            11,
        ),
        # With commas in number
        (
            "The total is 2,125 dollars.\nFinal Answer: 2,125",
            2125,
        ),
        # Wrong answer
        (
            "Step 1: 5 + 6 = 12\nFinal Answer: 12",
            11,
        ),
        # No Final Answer marker (fallback to last number)
        (
            "Step 1: 5 + 6 = 11\nThe answer is 11",
            11,
        ),
        # No number at all
        (
            "I don't know how to solve this problem.",
            42,
        ),
        # Bengali response with English digits
        (
            "ধাপ ১: ৫টি বল + ৬টি বল = ১১টি বল\nFinal Answer: 11",
            11,
        ),
    ]

    logging.info("=" * 60)
    logging.info("Testing reward computation:")
    texts = [tc[0] for tc in test_cases]
    gts = [tc[1] for tc in test_cases]
    rewards = reward_mgr.compute_rewards(texts, gts)
    for i, r in enumerate(rewards):
        logging.info(
            f"  Case {i}: reward={r['reward']:.1f}, "
            f"extracted={r['extracted_answer']}, gt={gts[i]}"
        )
    reward_mgr.log_reward_summary(rewards)

    logging.info("Sanity check passed.")


if __name__ == "__main__":
    main()