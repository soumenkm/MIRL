"""Reward computation for GRPO training.

Phase 1 architecture: the judge LLM runs as a separate vLLM server in its own
SLURM job (see ``judge_server.py``). The trainer discovers the server via a
connection file (JSON written by the judge once it is healthy) and queries
the OpenAI-compatible ``/v1/chat/completions`` endpoint over HTTP.

Why HTTP and not a local model:
    The judge model (Gemma 4 31B) does not fit on the same node as the
    policy + reference + optimizer state. Co-locating the judge would force
    aggressive 4-bit quantization and still risk OOM. Running the judge on
    a separate node with its own GPUs and TP=2 is both simpler and faster
    (vLLM continuous batching dominates HF ``generate``).

This file provides:
    - JudgeClient:    HTTP client for the remote vLLM judge. Discovers the
                      server via the connection file, retries on transient
                      failures, batches via thread-pool concurrency.
    - RewardManager:  Computes regex / judge / format / language-consistency
                      rewards and combines them with configurable weights.

The trainer node has no internet, but the judge node lives on the cluster's
private network — internal HTTP between compute nodes is unaffected.
"""

import json
import logging
import re
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from tqdm import tqdm


class JudgeClient:
    """HTTP client for a remote vLLM judge server.

    Responsibilities:
        - Wait for the judge's connection file to report ready=True
        - Read the served model name + URL from the connection file
        - Issue chat-completion requests with retries on transient failures
        - Run a batch of prompts concurrently (vLLM's continuous batcher
          merges them server-side; client-side concurrency is just to keep
          the server's queue full).
    """

    def __init__(self, config: dict):
        self.connection_file = Path(config["connection_file"])
        self.discovery_timeout = float(config.get("discovery_timeout", 1800))
        self.discovery_poll_interval = float(
            config.get("discovery_poll_interval", 5.0)
        )
        self.request_timeout = float(config.get("request_timeout", 120))
        self.max_retries = int(config.get("max_retries", 3))
        self.retry_backoff = float(config.get("retry_backoff", 2.0))
        self.max_concurrent_requests = int(
            config.get("max_concurrent_requests", 32)
        )
        self.log_dir = Path(config["log_dir"])

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()

        # Populated by _discover()
        self.url: str | None = None
        self.served_model_name: str | None = None

    def _setup_logging(self):
        log_file = self.log_dir / "judge_client.log"
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
        self.logger.info(f"JudgeClient logging to {log_file}")

    # ------------------------------------------------------------------ #
    #  Discovery
    # ------------------------------------------------------------------ #

    def discover(self) -> None:
        """Block until the judge advertises ready=True; cache url + model."""
        self.logger.info(
            f"Discovering judge via {self.connection_file} "
            f"(timeout={self.discovery_timeout}s)"
        )

        start = time.time()
        last_state = None
        while time.time() - start < self.discovery_timeout:
            info = self._read_connection_file()
            if info is not None:
                state = (info.get("ready"), info.get("ip"), info.get("port"))
                if state != last_state:
                    self.logger.info(
                        f"Connection file: ready={info.get('ready')}, "
                        f"ip={info.get('ip')}, port={info.get('port')}"
                    )
                    last_state = state
                if info.get("ready"):
                    # vLLM's served model name is the full model path
                    # (the value passed to ``vllm serve <model>``).
                    self.url = info["url"].rstrip("/")
                    self.served_model_name = info["model_path"]
                    self.logger.info(
                        f"Judge ready: url={self.url} "
                        f"model={self.served_model_name}"
                    )
                    self._verify_health()
                    return
            time.sleep(self.discovery_poll_interval)

        raise RuntimeError(
            f"Judge did not become ready within {self.discovery_timeout}s. "
            f"Check the judge SLURM job and {self.connection_file}."
        )

    def _read_connection_file(self) -> dict | None:
        if not self.connection_file.exists():
            return None
        try:
            with self.connection_file.open() as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            # File mid-write; try again next poll.
            return None

    def _verify_health(self) -> None:
        """Sanity-check the /health endpoint before declaring success."""
        health_url = f"{self.url}/health"
        try:
            with urllib.request.urlopen(health_url, timeout=10) as resp:
                if resp.status == 200:
                    self.logger.info(f"Health check OK: {health_url}")
                    return
                raise RuntimeError(f"Health returned status {resp.status}")
        except (urllib.error.URLError, socket.timeout, OSError) as e:
            raise RuntimeError(
                f"Judge advertised ready but {health_url} unreachable: {e}"
            ) from e

    # ------------------------------------------------------------------ #
    #  Single-request (with retries)
    # ------------------------------------------------------------------ #

    def chat(
        self,
        prompt: str,
        max_tokens: int = 8,
        temperature: float = 0.0,
    ) -> str:
        """Issue a single chat-completion request. Returns generated text."""
        if self.url is None:
            raise RuntimeError("JudgeClient not discovered. Call .discover() first.")

        endpoint = f"{self.url}/v1/chat/completions"
        payload = {
            "model": self.served_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        body = json.dumps(payload).encode("utf-8")

        last_err = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    endpoint,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(
                    req, timeout=self.request_timeout
                ) as resp:
                    raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                return data["choices"][0]["message"]["content"]
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                socket.timeout,
                json.JSONDecodeError,
                KeyError,
                IndexError,
                OSError,
            ) as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    sleep_s = self.retry_backoff * (2 ** attempt)
                    self.logger.warning(
                        f"Judge request failed (attempt {attempt+1}/"
                        f"{self.max_retries}): {e}. Retrying in {sleep_s:.1f}s."
                    )
                    time.sleep(sleep_s)

        raise RuntimeError(
            f"Judge request failed after {self.max_retries} attempts: {last_err}"
        )

    # ------------------------------------------------------------------ #
    #  Batch (concurrent)
    # ------------------------------------------------------------------ #

    def chat_batch(
        self,
        prompts: list[str],
        max_tokens: int = 8,
        temperature: float = 0.0,
        desc: str = "Judge batch",
    ) -> list[str]:
        """Issue many requests concurrently. Order of results matches prompts."""
        if not prompts:
            return []

        results: list[str | None] = [None] * len(prompts)
        n_workers = min(self.max_concurrent_requests, len(prompts))

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(self.chat, p, max_tokens, temperature): i
                for i, p in enumerate(prompts)
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=desc,
                unit="req",
                leave=False,
            ):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    self.logger.error(f"Prompt {idx} failed permanently: {e}")
                    results[idx] = ""  # caller treats empty as unparseable

        return [r if r is not None else "" for r in results]


class RewardManager:
    """Manages reward computation for GRPO training.

    Reward components:
        - accuracy_regex:        deterministic, fast, weight 1.0 by default
        - accuracy_judge:        remote vLLM judge over HTTP
        - format:                stub (Phase 2)
        - language_consistency:  stub (Phase 2)

    The judge is OPTIONAL. It is only initialized when
    ``weight_accuracy_judge > 0``. If enabled, ``judge_config`` must be present
    in the config dict and contain a ``connection_file`` path written by
    ``judge_server.py``.
    """

    def __init__(self, config: dict):
        self.log_dir = Path(config["log_dir"])
        self.log_max_response_chars = int(config.get("log_max_response_chars", 500))

        # Reward weights
        self.weight_accuracy_regex = float(config.get("weight_accuracy_regex", 1.0))
        self.weight_accuracy_judge = float(config.get("weight_accuracy_judge", 0.0))
        self.weight_format = float(config.get("weight_format", 0.0))
        self.weight_language_consistency = float(
            config.get("weight_language_consistency", 0.0)
        )

        # Judge generation knobs
        self.judge_max_tokens = int(config.get("judge_max_tokens", 8))
        self.judge_temperature = float(config.get("judge_temperature", 0.0))

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()

        # Lazy: only build a JudgeClient if we actually need it.
        self.judge_client: JudgeClient | None = None
        if self.weight_accuracy_judge > 0.0:
            judge_cfg = config.get("judge_config")
            if judge_cfg is None:
                raise ValueError(
                    "weight_accuracy_judge > 0 but config['judge_config'] missing."
                )
            self.judge_client = JudgeClient(judge_cfg)
            self.judge_client.discover()

    def _setup_logging(self):
        log_file = self.log_dir / "rewards.log"
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
        self.logger.info(f"Rewards logging to {log_file}")
        self.logger.info(
            f"Reward weights: regex={self.weight_accuracy_regex}, "
            f"judge={self.weight_accuracy_judge}, "
            f"format={self.weight_format}, "
            f"lc={self.weight_language_consistency}"
        )

    # ------------------------------------------------------------------ #
    #  Regex-based answer extraction
    # ------------------------------------------------------------------ #

    def extract_final_answer(self, response_text: str) -> int | None:
        """Extract the integer after 'Final Answer:' from the response."""
        match = re.search(r"Final Answer\s*:\s*([+-]?\d[\d,]*)", response_text)
        if match:
            return int(match.group(1).replace(",", ""))
        all_numbers = re.findall(r"(?<!\S)([+-]?\d[\d,]*)(?!\S)", response_text)
        if all_numbers:
            return int(all_numbers[-1].replace(",", ""))
        return None

    def _compute_accuracy_regex(
        self,
        response_text: str,
        answer_number: int,
    ) -> float:
        extracted = self.extract_final_answer(response_text)
        matched = extracted is not None and extracted == answer_number

        max_chars = self.log_max_response_chars
        resp_display = response_text[:max_chars] + (
            "..." if len(response_text) > max_chars else ""
        )
        self.logger.info(
            f"[REGEX] gt={answer_number}, extracted={extracted}, "
            f"match={matched}, response={resp_display}"
        )
        return 1.0 if matched else 0.0

    # ------------------------------------------------------------------ #
    #  Judge LLM-based accuracy verification (remote vLLM)
    # ------------------------------------------------------------------ #

    def _build_judge_prompt(self, response_text: str, answer_number: int) -> str:
        return (
            "You are a math answer verifier. Your task is to determine whether "
            "a student's solution arrives at the correct final answer.\n\n"
            f"Correct Answer: {answer_number}\n\n"
            f"Student's Response:\n{response_text}\n\n"
            "Does the student's response arrive at the correct final answer? "
            "Reply with exactly one word: YES or NO."
        )

    @staticmethod
    def _parse_yes_no(judge_response: str) -> float:
        """Parse the judge's first whitespace-delimited token.

        Avoids substring traps:
          - 'YES' inside 'YESTERDAY' must NOT count as YES
          - 'No, the answer is correct' must NOT count as NO
          - 'Yes.' / 'YES!' / ' yes ' all count as YES.
        """
        if not judge_response:
            return 0.0
        # Take leading non-whitespace word, strip trailing punctuation.
        tokens = judge_response.strip().split()
        if not tokens:
            return 0.0
        first = re.sub(r"[^A-Za-z]", "", tokens[0]).upper()
        if first == "YES":
            return 1.0
        if first == "NO":
            return 0.0
        return 0.0  # unparseable -> conservative no-credit

    def _compute_accuracy_judge_batch(
        self,
        response_texts: list[str],
        answer_numbers: list[int],
    ) -> list[float]:
        assert self.judge_client is not None

        prompts = [
            self._build_judge_prompt(t, a)
            for t, a in zip(response_texts, answer_numbers)
        ]
        raw_responses = self.judge_client.chat_batch(
            prompts,
            max_tokens=self.judge_max_tokens,
            temperature=self.judge_temperature,
            desc="Judge evaluation",
        )

        rewards = []
        max_chars = self.log_max_response_chars
        separator = "-" * 80
        for i, (prompt, raw, gt) in enumerate(
            zip(prompts, raw_responses, answer_numbers)
        ):
            reward = self._parse_yes_no(raw)
            self.logger.info(separator)
            self.logger.info(f"[JUDGE INPUT #{i}] gt={gt}")
            prompt_disp = prompt[:max_chars] + (
                "..." if len(prompt) > max_chars else ""
            )
            self.logger.info(prompt_disp)
            self.logger.info(
                f"[JUDGE OUTPUT #{i}] raw='{raw.strip()}' reward={reward}"
            )
            if reward == 0.0 and raw.strip() and not raw.strip().upper().startswith(
                ("NO", "YES")
            ):
                self.logger.warning(f"  unparseable response: '{raw.strip()}'")
            rewards.append(reward)
        self.logger.info(separator)
        return rewards

    # ------------------------------------------------------------------ #
    #  Format reward (Phase 2 stub)
    # ------------------------------------------------------------------ #

    def _compute_format_reward(self, response_text: str) -> float:
        score = 0.0
        final_answer_count = len(re.findall(r"Final Answer\s*:", response_text))
        if final_answer_count == 1:
            score += 0.5
        elif final_answer_count > 1:
            score += 0.1

        match = re.search(r"Final Answer\s*:\s*([+-]?\d[\d,]*)", response_text)
        if match:
            score += 0.3
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
        # TODO(Phase 2): Unicode script-based token classification.
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
        batch_size = len(response_texts)
        assert len(answer_numbers) == batch_size

        self.logger.info(f"Computing rewards for {batch_size} responses")

        # Regex accuracy (always computed)
        regex_rewards = [
            self._compute_accuracy_regex(t, a)
            for t, a in zip(response_texts, answer_numbers)
        ]
        extracted_answers = [
            self.extract_final_answer(t) for t in response_texts
        ]

        # Judge accuracy
        judge_rewards = None
        if self.weight_accuracy_judge > 0.0 and self.judge_client is not None:
            judge_rewards = self._compute_accuracy_judge_batch(
                response_texts, answer_numbers
            )

        # Format reward
        format_rewards = None
        if self.weight_format > 0.0:
            format_rewards = [self._compute_format_reward(t) for t in response_texts]

        # Language consistency
        lc_rewards = None
        if self.weight_language_consistency > 0.0 and langs is not None:
            lc_rewards = [
                self._compute_language_consistency_reward(t, l)
                for t, l in zip(response_texts, langs)
            ]

        results = []
        for i in range(batch_size):
            total = self.weight_accuracy_regex * regex_rewards[i]
            if judge_rewards is not None:
                total += self.weight_accuracy_judge * judge_rewards[i]
            if format_rewards is not None:
                total += self.weight_format * format_rewards[i]
            if lc_rewards is not None:
                total += self.weight_language_consistency * lc_rewards[i]

            results.append({
                "reward": total,
                "accuracy_regex": regex_rewards[i],
                "accuracy_judge": judge_rewards[i] if judge_rewards else None,
                "format": format_rewards[i] if format_rewards else None,
                "language_consistency": lc_rewards[i] if lc_rewards else None,
                "extracted_answer": extracted_answers[i],
            })
        return results

    def log_reward_summary(self, all_rewards: list[dict]):
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
            agree = sum(
                1 for r in all_rewards
                if r["accuracy_regex"] == r["accuracy_judge"]
            )
            self.logger.info(
                f"  regex-judge agreement={agree}/{n} ({100 * agree / n:.1f}%)"
            )

        if all_rewards[0]["format"] is not None:
            avg_fmt = sum(r["format"] for r in all_rewards) / n
            self.logger.info(f"  avg_format={avg_fmt:.3f}")

        if all_rewards[0]["language_consistency"] is not None:
            avg_lc = sum(r["language_consistency"] for r in all_rewards) / n
            self.logger.info(f"  avg_language_consistency={avg_lc:.3f}")


def main():
    """Sanity check against a live remote judge.

    Pre-requisite: ``judge_server.py`` running in a separate SLURM job and
    ``judge_connection.json`` advertising ready=True.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main_logger = logging.getLogger("Main")
    main_logger.info(
        f"GPUs available: {torch.cuda.device_count()} "
        f"({torch.cuda.get_device_name(0)})"
    )

    config = {
        "log_dir": Path("./exp2/logs"),
        "log_max_response_chars": 500,

        "weight_accuracy_regex": 1.0,
        "weight_accuracy_judge": 1.0,
        "weight_format": 0.0,
        "weight_language_consistency": 0.0,

        "judge_max_tokens": 8,
        "judge_temperature": 0.0,

        "judge_config": {
            "connection_file": Path("./exp2/outputs/judge_connection.json"),
            "log_dir": Path("./exp2/logs"),
            "discovery_timeout": 1800,
            "discovery_poll_interval": 5.0,
            "request_timeout": 120,
            "max_retries": 3,
            "retry_backoff": 2.0,
            "max_concurrent_requests": 32,
        },
    }

    reward_mgr = RewardManager(config)

    test_cases = [
        ("Step 1: 5 + 6 = 11\nFinal Answer: 11", 11),
        ("The total is 2,125 dollars.\nFinal Answer: 2,125", 2125),
        ("Step 1: 5 + 6 = 12\nFinal Answer: 12", 11),
        ("Step 1: 5 + 6 = 11\nThe answer is 11", 11),
        ("I don't know how to solve this problem.", 42),
        ("ধাপ ১: ৫টি বল + ৬টি বল = ১১টি বল\nFinal Answer: 11", 11),
    ]

    main_logger.info("=" * 60)
    main_logger.info("Testing reward computation against remote judge:")
    texts = [tc[0] for tc in test_cases]
    gts = [tc[1] for tc in test_cases]
    rewards = reward_mgr.compute_rewards(texts, gts)
    for i, r in enumerate(rewards):
        main_logger.info(
            f"  Case {i}: reward={r['reward']:.1f}, "
            f"regex={r['accuracy_regex']}, judge={r['accuracy_judge']}, "
            f"extracted={r['extracted_answer']}, gt={gts[i]}"
        )
    reward_mgr.log_reward_summary(rewards)
    main_logger.info("Sanity check complete.")


if __name__ == "__main__":
    main()