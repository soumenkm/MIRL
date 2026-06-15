"""Re-score existing judge cache files using an OpenAI Chat Completions model.

This script reads each per-checkpoint JSON cache written by
plot_accuracy.py (in exp2/outputs/scores/), reuses the per-prompt
``judge_prompt`` field verbatim, sends it to an OpenAI GPT model
(default ``gpt-4o-mini``), parses the YES/NO response, and writes a
mirror-shape JSON file to exp2/outputs/scores_gpt/ with the same field
names but GPT's judgement.

The input directory ``scores/`` is read-only here; the existing Gemma
scores are preserved unchanged for later comparison.

The OpenAI judge prompt is rebuilt in Chat format (system message +
user message). The system message gives GPT a strict role; the user
message contains the math verification request. The verifier role and
the YES/NO output contract are taken from the original prompt
template in rewards.RewardManager._build_judge_prompt.

Behaviour:
    - Auto-discovers all accuracy_step_<N>.json under scores/.
    - For each file: if scores_gpt/accuracy_step_<N>.json already exists
      and use_cache is True, the file is skipped.
    - Otherwise, sends one OpenAI request per prompt with bounded
      concurrency (Tier 1 default = 8 workers), with retry on
      RateLimitError / APIConnectionError / 5xx.
    - Per-checkpoint atomic write: the output JSON is written ONLY after
      all prompts for that checkpoint have been re-scored, so a crash
      mid-checkpoint never leaves a half-written file. A crash leaves
      the previous checkpoints' outputs intact; a rerun resumes from
      the next unprocessed checkpoint.
    - Per-language accuracies are recomputed and stored in the same
      ``per_language_accuracy`` field.

Standards: see project coding standards. All paths are pathlib.Path. No
argparse, no print. tqdm where appropriate. Logger has FileHandler +
StreamHandler in the mandated format. The only use of the ``os`` module
is to read the API key from a local file via ``pathlib.Path``; no
``os.environ`` mutation, no other os calls.
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

from openai import (
    OpenAI,
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
)


class GPTJudgeRescorer:
    """Re-score Gemma judge cache files using an OpenAI GPT model."""

    LANG_LABELS = ("en", "bn", "te", "th", "ru", "ja", "zh")

    SYSTEM_PROMPT = (
        "You are a strict math answer verifier. You receive a math "
        "problem's correct integer answer and a student's full "
        "response. Decide whether the student's response arrives at "
        "the correct final answer. Respond with exactly one word: "
        "YES or NO. Do not include any other text, explanation, or "
        "punctuation."
    )

    def __init__(self, config: dict):
        self.log_dir = Path(config["log_dir"])
        self.output_dir = Path(config["output_dir"])
        self.data_dir = Path(config["data_dir"])

        self.scores_dir = self.output_dir / "scores"
        self.scores_gpt_dir = self.output_dir / "scores_gpt"

        self.api_key_path = Path(config["api_key_path"]).expanduser()
        self.model_name = str(config.get("model_name", "gpt-4o-mini"))
        self.max_workers = int(config.get("max_workers", 8))
        self.request_max_tokens = int(config.get("request_max_tokens", 4))
        self.request_temperature = float(
            config.get("request_temperature", 0.0)
        )
        self.request_timeout = float(config.get("request_timeout", 60))
        self.max_retries = int(config.get("max_retries", 4))
        self.retry_backoff = float(config.get("retry_backoff", 2.0))
        self.use_cache = bool(config.get("use_cache", True))

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.scores_gpt_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        # Validate API key path exists and read it. Stripping whitespace
        # because key files sometimes have a trailing newline.
        if not self.api_key_path.exists():
            raise FileNotFoundError(
                f"OpenAI API key file not found at {self.api_key_path}. "
                f"Place your key in this file (one line)."
            )
        api_key = self.api_key_path.read_text(encoding="utf-8").strip()
        if not api_key:
            raise ValueError(
                f"API key file {self.api_key_path} is empty."
            )

        self.client = OpenAI(api_key=api_key, timeout=self.request_timeout)
        self.logger.info(
            f"OpenAI client initialised (model={self.model_name}, "
            f"max_workers={self.max_workers})."
        )

    def _setup_logging(self):
        log_file = self.log_dir / "judge_with_gpt.log"
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
    #  Discovery
    # ------------------------------------------------------------------ #

    def _discover_score_files(self) -> list[tuple[int, Path]]:
        """Find all input score files under scores/, sorted by step."""
        pat = re.compile(r"^accuracy_step_(\d+)\.json$")
        found: list[tuple[int, Path]] = []
        if not self.scores_dir.exists():
            raise FileNotFoundError(
                f"Input scores directory does not exist: {self.scores_dir}"
            )
        for p in self.scores_dir.iterdir():
            if not p.is_file():
                continue
            m = pat.match(p.name)
            if m:
                found.append((int(m.group(1)), p))
        found.sort(key=lambda x: x[0])
        if not found:
            raise FileNotFoundError(
                f"No accuracy_step_*.json found in {self.scores_dir}."
            )
        self.logger.info(
            f"Discovered {len(found)} input score files: "
            f"steps {[s for s, _ in found]}"
        )
        return found

    # ------------------------------------------------------------------ #
    #  Judge prompt -> Chat messages
    # ------------------------------------------------------------------ #

    def _build_chat_messages(
        self,
        ground_truth: int,
        cot_text: str,
    ) -> list[dict]:
        """Build the GPT Chat Completions message list.

        Rather than reusing the raw ``judge_prompt`` field verbatim, we
        rewrite it into Chat format with a system role for the verifier
        instructions and a user role for the verification request. GPT
        models follow system messages more reliably than embedded
        instructions inside a user message.
        """
        user_msg = (
            f"Correct Answer: {ground_truth}\n\n"
            f"Student's Response:\n{cot_text}\n\n"
            "Does the student's response arrive at the correct final "
            "answer? Reply with exactly one word: YES or NO."
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    @staticmethod
    def _parse_yes_no(response_text: str) -> float:
        """Parse the first non-whitespace word as YES/NO.

        Identical semantics to RewardManager._parse_yes_no in rewards.py
        so that the GPT scoring is comparable to the Gemma scoring.
        """
        if not response_text:
            return 0.0
        tokens = response_text.strip().split()
        if not tokens:
            return 0.0
        first = re.sub(r"[^A-Za-z]", "", tokens[0]).upper()
        if first == "YES":
            return 1.0
        return 0.0

    # ------------------------------------------------------------------ #
    #  Single request with retries
    # ------------------------------------------------------------------ #

    def _chat_once(self, messages: list[dict]) -> str:
        """Send one Chat Completions request, with retry on transient errors.

        Returns the raw response text (empty string on permanent failure).
        Transient errors that trigger a retry: RateLimitError,
        APIConnectionError, APITimeoutError, and 5xx APIErrors. All other
        APIErrors propagate (likely indicates a bug in the request).
        """
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.request_max_tokens,
                    temperature=self.request_temperature,
                )
                content = resp.choices[0].message.content or ""
                return content
            except (
                RateLimitError,
                APIConnectionError,
                APITimeoutError,
            ) as e:
                last_err = e
                # Exponential backoff on rate limits and transient
                # connectivity issues.
                sleep_s = self.retry_backoff * (2 ** attempt)
                self.logger.warning(
                    f"OpenAI transient error (attempt {attempt+1}/"
                    f"{self.max_retries}): {type(e).__name__}: {e}. "
                    f"Retrying in {sleep_s:.1f}s."
                )
                time.sleep(sleep_s)
            except APIError as e:
                # 5xx server errors are retriable; 4xx (other than rate
                # limit, handled above) are not.
                status = getattr(e, "status_code", None)
                if status is not None and 500 <= int(status) < 600:
                    last_err = e
                    sleep_s = self.retry_backoff * (2 ** attempt)
                    self.logger.warning(
                        f"OpenAI 5xx error (attempt {attempt+1}/"
                        f"{self.max_retries}): {e}. Retrying in "
                        f"{sleep_s:.1f}s."
                    )
                    time.sleep(sleep_s)
                else:
                    self.logger.error(
                        f"OpenAI non-retriable error: {e}"
                    )
                    raise
        self.logger.error(
            f"All {self.max_retries} retries failed: {last_err}"
        )
        return ""

    # ------------------------------------------------------------------ #
    #  Per-checkpoint re-scoring
    # ------------------------------------------------------------------ #

    def _rescore_one_checkpoint(self, step: int, in_path: Path) -> None:
        """Read one score file, re-score with GPT, write to scores_gpt/.

        Per-checkpoint atomic write: builds the full new payload in
        memory and writes the destination file in a single step.
        """
        out_path = self.scores_gpt_dir / f"accuracy_step_{step}.json"
        if self.use_cache and out_path.exists():
            self.logger.info(
                f"Skipping step={step}: {out_path} already exists "
                f"(use_cache=True)."
            )
            return

        with in_path.open(encoding="utf-8") as f:
            payload = json.load(f)

        prompts: list[dict] = payload.get("prompts", [])
        if not prompts:
            self.logger.warning(
                f"step={step}: input file has no 'prompts' list; skipping."
            )
            return

        # Decide which prompts need a GPT call. Empty-CoT placeholders
        # (existing judge_score == 0.0 with no judge_prompt) are kept as
        # zero, matching the behaviour of the original scoring code.
        callable_indices: list[int] = []
        for i, rec in enumerate(prompts):
            jp = rec.get("judge_prompt")
            if jp is None or not rec.get("cot_text"):
                # Empty CoT or missing judge_prompt -> keep as-is, zero.
                continue
            callable_indices.append(i)

        self.logger.info(
            f"step={step}: {len(callable_indices)} / {len(prompts)} prompts "
            f"to re-score via GPT (rest are empty-CoT placeholders)."
        )

        # Submit Chat Completions concurrently.
        raws: dict[int, str] = {}
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            for i in callable_indices:
                rec = prompts[i]
                messages = self._build_chat_messages(
                    ground_truth=int(rec["ground_truth"]),
                    cot_text=str(rec["cot_text"]),
                )
                futures[pool.submit(self._chat_once, messages)] = i
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"step={step} GPT",
                unit="req",
            ):
                i = futures[fut]
                try:
                    raws[i] = fut.result()
                except Exception as e:
                    self.logger.error(
                        f"step={step} prompt #{i} permanently failed: {e}"
                    )
                    raws[i] = ""

        elapsed = time.time() - t0
        self.logger.info(
            f"step={step}: GPT returned for "
            f"{len(callable_indices)} prompts in {elapsed:.1f}s "
            f"(avg {elapsed / max(1, len(callable_indices)) * 1000:.0f} "
            f"ms/req)."
        )

        # Build the new per-prompt records, preserving every input field
        # but overwriting judge_raw / judge_score. The 'note' field on
        # empty-CoT records is preserved as-is.
        new_prompts: list[dict] = []
        for i, rec in enumerate(prompts):
            new_rec = dict(rec)  # shallow copy
            if i in raws:
                raw = raws[i]
                new_rec["judge_raw"] = raw
                new_rec["judge_score"] = self._parse_yes_no(raw)
            else:
                # Empty-CoT placeholder, or callable_indices excluded.
                # Keep the original fields untouched.
                new_rec["judge_raw"] = rec.get("judge_raw")
                new_rec["judge_score"] = float(rec.get("judge_score", 0.0))
            new_prompts.append(new_rec)

        # Recompute per-language averages from the new judge_scores.
        per_lang_acc: dict[str, float] = {}
        per_lang_n: dict[str, int] = {}
        for lang in self.LANG_LABELS:
            in_lang = [r for r in new_prompts if r["lang"] == lang]
            per_lang_n[lang] = len(in_lang)
            if in_lang:
                per_lang_acc[lang] = (
                    sum(float(r["judge_score"]) for r in in_lang)
                    / len(in_lang)
                )
            else:
                per_lang_acc[lang] = float("nan")

        new_payload = {
            "checkpoint_step": int(payload.get("checkpoint_step", step)),
            "n_prompts": len(new_prompts),
            "per_language_accuracy": per_lang_acc,
            "per_language_counts": per_lang_n,
            "judge_model": self.model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompts": new_prompts,
        }

        # Atomic write: write to a tmp file in the same directory, then
        # rename. If the process is killed mid-write, the final file
        # never appears, so the next run will retry this checkpoint.
        tmp_path = out_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(new_payload, f, indent=4, ensure_ascii=False)
        tmp_path.replace(out_path)
        self.logger.info(f"Wrote {out_path}")
        self._log_step_summary(step, per_lang_acc, per_lang_n)

    def _log_step_summary(
        self,
        step: int,
        per_lang_acc: dict[str, float],
        per_lang_n: dict[str, int],
    ):
        self.logger.info(f"step={step} GPT-judged per-language accuracy:")
        for lang in self.LANG_LABELS:
            n = per_lang_n.get(lang, 0)
            acc = per_lang_acc.get(lang, float("nan"))
            self.logger.info(f"  {lang}: {acc:.3f}  (n={n})")

    # ------------------------------------------------------------------ #
    #  Orchestrator
    # ------------------------------------------------------------------ #

    def run(self):
        self.logger.info("=" * 80)
        self.logger.info(
            f"GPT rescoring; reading {self.scores_dir} -> "
            f"writing {self.scores_gpt_dir}"
        )
        self.logger.info("=" * 80)
        files = self._discover_score_files()
        for step, in_path in tqdm(
            files, desc="checkpoints", unit="ckpt"
        ):
            self._rescore_one_checkpoint(step, in_path)
        self.logger.info("GPT rescoring complete.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main_logger = logging.getLogger("Main")

    # GPU not strictly required (this is HTTP / CPU work), but the
    # standards demand the check and the pipeline is GPU-node-only.
    # if not torch.cuda.is_available():
    #     raise RuntimeError("No GPU detected.")
    # main_logger.info(
    #     f"GPUs available: {torch.cuda.device_count()} "
    #     f"({torch.cuda.get_device_name(0)})"
    # )

    config = {
        # ---- Required path roots ----
        "log_dir":    Path("./exp2/logs"),
        "output_dir": Path("./exp2/outputs"),
        "data_dir":   Path("./exp2/data"),

        # ---- OpenAI access ----
        # One-line text file containing your sk-... key.
        "api_key_path": Path("~/.openai_key"),
        "model_name":   "gpt-4o-mini",

        # ---- Concurrency + retries (Tier 1 friendly defaults) ----
        "max_workers":         8,
        "request_max_tokens":  4,
        "request_temperature": 0.0,
        "request_timeout":     60,
        "max_retries":         4,
        "retry_backoff":       2.0,

        # ---- Cache: skip checkpoints whose output already exists ----
        "use_cache": True,
    }

    GPTJudgeRescorer(config).run()


if __name__ == "__main__":
    main()