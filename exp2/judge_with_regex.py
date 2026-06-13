"""Re-score existing judge cache files using a regex-first, GPT-fallback judge.

This script reads each per-checkpoint JSON cache written by
plot_accuracy.py (in exp2/outputs/scores/), and re-scores every prompt
using a two-stage judge:

    Stage 1 (regex): extract the student's final integer answer from
        ``cot_text`` using a robust set of patterns (primarily the
        ``Final Answer: <number>`` line mandated by the prompt template,
        with several fallbacks). If an integer is confidently extracted,
        the score is 1.0 iff it equals ``ground_truth``, else 0.0. No
        API call is made.

    Stage 2 (GPT fallback): only when the regex stage cannot confidently
        extract an answer, send the prompt to an OpenAI GPT model
        (default ``gpt-4o-mini``) and parse a YES/NO reply, exactly as in
        judge_with_gpt.py.

It writes a mirror-shape JSON file to exp2/outputs/scores_gpt/ with the
same field names as the input, but with judge_score populated by this
regex-first judge and per-language accuracies recomputed accordingly.
A ``judge_method`` field is added to each prompt record ("regex" or
"gpt") so the split between the two stages is auditable.

The input directory ``scores/`` is read-only here; the existing Gemma
scores are preserved unchanged for later comparison.

Behaviour:
    - Auto-discovers all accuracy_step_<N>.json under scores/.
    - For each file: if scores_gpt/accuracy_step_<N>.json already exists
      and use_cache is True, the file is skipped.
    - Regex-scored prompts make no network call. Only the residual set
      that regex cannot resolve is sent to OpenAI, with bounded
      concurrency and retry on transient errors.
    - Per-checkpoint atomic write: the output JSON is written ONLY after
      all prompts for that checkpoint have been re-scored.
    - Per-language accuracies are recomputed and stored in the same
      ``per_language_accuracy`` field.

Standards: see project coding standards. All paths are pathlib.Path. No
argparse, no print. tqdm where appropriate. Logger has FileHandler +
StreamHandler in the mandated format. The only use of the ``os`` module
is avoided entirely; the API key is read from a local file via
``pathlib.Path``.
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


class RegexFirstJudgeRescorer:
    """Re-score judge cache files with a regex-first, GPT-fallback judge."""

    LANG_LABELS = ("en", "bn", "te", "th", "ru", "ja", "zh")

    SYSTEM_PROMPT = (
        "You are a strict math answer verifier. You receive a math "
        "problem's correct integer answer and a student's full "
        "response. Decide whether the student's response arrives at "
        "the correct final answer. Respond with exactly one word: "
        "YES or NO. Do not include any other text, explanation, or "
        "punctuation."
    )

    # Ordered list of patterns used to locate the student's final answer
    # in the chain-of-thought. The first group that yields a parseable
    # integer wins. Patterns are tried in order of decreasing
    # reliability. All are case-insensitive and DOTALL-free (we match on
    # single lines where possible).
    #
    # A number is allowed to carry surrounding noise that we strip later:
    # thousands separators, a leading currency symbol, a trailing period,
    # or surrounding markup such as **...** or \boxed{...}.
    # The number may carry a decimal tail; we capture it so _clean_int
    # can see it and reject the value (MGSM answers are integers, so a
    # decimal output is wrong and must be deferred to the GPT fallback
    # rather than silently truncated).
    _NUM = r"[-+]?\$?\s*\d[\d,]*(?:\.\d+)?"

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
        self._compile_patterns()

        # Lazily-initialised OpenAI client: we only build it (and only
        # require the API key) if at least one prompt actually needs the
        # GPT fallback. This lets the script run entirely offline when
        # regex resolves every prompt.
        self._client: OpenAI | None = None

        # Running tallies for a final report of regex-vs-gpt split.
        self.n_regex_total = 0
        self.n_gpt_total = 0

    def _setup_logging(self):
        log_file = self.log_dir / "judge_with_regex.log"
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

    def _compile_patterns(self):
        """Compile the ordered answer-extraction patterns once."""
        n = self._NUM
        # Each entry: (name, compiled_regex). The capturing group (1)
        # holds the raw number-with-noise to be cleaned by _clean_int.
        # re.IGNORECASE makes "final answer" match any case; we keep
        # them line-oriented via [^\n]* where a trailing tail is allowed.
        self._patterns: list[tuple[str, re.Pattern]] = [
            (
                "final_answer_label",
                re.compile(
                    rf"final\s*answer\s*[:\-]?\s*\**\s*\\?boxed?\{{?\s*({n})",
                    re.IGNORECASE,
                ),
            ),
            (
                "boxed",
                re.compile(rf"\\boxed\{{\s*({n})\s*\}}", re.IGNORECASE),
            ),
            (
                "answer_is",
                re.compile(
                    rf"(?:the\s+)?answer\s+is\s*[:\-]?\s*\**\s*({n})",
                    re.IGNORECASE,
                ),
            ),
            (
                "answer_label",
                re.compile(
                    rf"\banswer\s*[:\-]\s*\**\s*({n})",
                    re.IGNORECASE,
                ),
            ),
        ]

    # ------------------------------------------------------------------ #
    #  OpenAI client (lazy)
    # ------------------------------------------------------------------ #

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key_path.exists():
                raise FileNotFoundError(
                    f"OpenAI API key file not found at {self.api_key_path}. "
                    f"Place your key in this file (one line). It is only "
                    f"needed for prompts the regex stage cannot resolve."
                )
            api_key = self.api_key_path.read_text(encoding="utf-8").strip()
            if not api_key:
                raise ValueError(
                    f"API key file {self.api_key_path} is empty."
                )
            self._client = OpenAI(
                api_key=api_key, timeout=self.request_timeout
            )
            self.logger.info(
                f"OpenAI client initialised for GPT fallback "
                f"(model={self.model_name}, max_workers={self.max_workers})."
            )
        return self._client

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
    #  Regex stage
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clean_int(raw: str) -> int | None:
        """Turn a raw matched number (with noise) into a clean int.

        Strips currency symbols, whitespace, and thousands separators.
        Returns None if what remains is not a pure integer (we do not
        accept decimals here, because all MGSM ground-truth answers are
        integers; a decimal extraction is treated as a regex miss so the
        GPT fallback can adjudicate).
        """
        if raw is None:
            return None
        s = raw.strip()
        s = s.replace("$", "").replace(" ", "")
        s = s.replace(",", "")
        # Reject anything that still has a decimal point or stray chars.
        if not re.fullmatch(r"[-+]?\d+", s):
            return None
        try:
            return int(s)
        except ValueError:
            return None

    def _extract_answer(self, cot_text: str) -> int | None:
        """Extract the student's final integer answer from the CoT.

        Strategy: the prompt template mandates a ``Final Answer:`` line,
        so we prioritise the LAST match of the strongest label pattern
        (the model sometimes emits the line more than once, e.g. the
        duplicated 'Final Answer: 120' in the sample; the last one is the
        model's settled answer). We then fall back to weaker patterns,
        again preferring the last occurrence. Returns None if no pattern
        yields a clean integer, signalling that the GPT fallback should
        adjudicate.
        """
        if not cot_text:
            return None
        for _name, pat in self._patterns:
            matches = list(pat.finditer(cot_text))
            if not matches:
                continue
            # Prefer the last occurrence: later answers supersede earlier
            # working values, and a duplicated final-answer line settles
            # on the same value anyway.
            for m in reversed(matches):
                val = self._clean_int(m.group(1))
                if val is not None:
                    return val
        return None

    # ------------------------------------------------------------------ #
    #  GPT fallback (mirrors judge_with_gpt.py)
    # ------------------------------------------------------------------ #

    def _build_chat_messages(
        self,
        ground_truth: int,
        cot_text: str,
    ) -> list[dict]:
        """Build the GPT Chat Completions message list for the fallback."""
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
        """Parse the first non-whitespace word as YES/NO."""
        if not response_text:
            return 0.0
        tokens = response_text.strip().split()
        if not tokens:
            return 0.0
        first = re.sub(r"[^A-Za-z]", "", tokens[0]).upper()
        if first == "YES":
            return 1.0
        return 0.0

    def _chat_once(self, messages: list[dict]) -> str:
        """Send one Chat Completions request, with retry on transient errors."""
        client = self._get_client()
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
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
                sleep_s = self.retry_backoff * (2 ** attempt)
                self.logger.warning(
                    f"OpenAI transient error (attempt {attempt+1}/"
                    f"{self.max_retries}): {type(e).__name__}: {e}. "
                    f"Retrying in {sleep_s:.1f}s."
                )
                time.sleep(sleep_s)
            except APIError as e:
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
        """Read one score file, re-score regex-first, write to scores_gpt/."""
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

        # Stage 1: regex. Decide, per prompt, whether regex resolves it.
        # Empty-CoT placeholders (no cot_text) are kept as zero without
        # any call, matching the original scoring behaviour.
        regex_scores: dict[int, float] = {}
        gpt_indices: list[int] = []
        for i, rec in enumerate(prompts):
            cot = rec.get("cot_text")
            if not cot:
                # Empty CoT -> failed generation -> score 0, method regex
                # (no API call, deterministic).
                regex_scores[i] = 0.0
                continue
            try:
                gt = int(rec["ground_truth"])
            except (KeyError, TypeError, ValueError):
                # No usable ground truth: defer to GPT.
                gt = None
            extracted = self._extract_answer(str(cot))
            if extracted is not None and gt is not None:
                regex_scores[i] = 1.0 if extracted == gt else 0.0
            else:
                gpt_indices.append(i)

        n_regex = len(regex_scores)
        n_gpt = len(gpt_indices)
        self.n_regex_total += n_regex
        self.n_gpt_total += n_gpt
        self.logger.info(
            f"step={step}: regex resolved {n_regex} / {len(prompts)} "
            f"prompts; {n_gpt} deferred to GPT fallback."
        )

        # Stage 2: GPT fallback for the unresolved set, concurrently.
        raws: dict[int, str] = {}
        if gpt_indices:
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {}
                for i in gpt_indices:
                    rec = prompts[i]
                    messages = self._build_chat_messages(
                        ground_truth=int(rec["ground_truth"]),
                        cot_text=str(rec["cot_text"]),
                    )
                    futures[pool.submit(self._chat_once, messages)] = i
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"step={step} GPT-fallback",
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
                f"step={step}: GPT fallback returned for {n_gpt} prompts "
                f"in {elapsed:.1f}s "
                f"(avg {elapsed / max(1, n_gpt) * 1000:.0f} ms/req)."
            )

        # Build new per-prompt records, preserving every input field but
        # overwriting judge_raw / judge_score and adding judge_method.
        new_prompts: list[dict] = []
        for i, rec in enumerate(prompts):
            new_rec = dict(rec)  # shallow copy
            if i in regex_scores:
                new_rec["judge_score"] = float(regex_scores[i])
                # Make the regex verdict auditable. judge_raw records the
                # extracted integer (or None for empty-CoT placeholders).
                if not rec.get("cot_text"):
                    new_rec["judge_raw"] = rec.get("judge_raw")
                else:
                    new_rec["judge_raw"] = (
                        f"regex_extracted="
                        f"{self._extract_answer(str(rec['cot_text']))}"
                    )
                new_rec["judge_method"] = "regex"
            else:
                raw = raws.get(i, "")
                new_rec["judge_raw"] = raw
                new_rec["judge_score"] = self._parse_yes_no(raw)
                new_rec["judge_method"] = "gpt"
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
            "judge_model": f"regex+{self.model_name}",
            "n_regex": n_regex,
            "n_gpt_fallback": n_gpt,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompts": new_prompts,
        }

        # Atomic write: tmp file in the same directory, then rename.
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
        self.logger.info(f"step={step} regex-first per-language accuracy:")
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
            f"Regex-first rescoring; reading {self.scores_dir} -> "
            f"writing {self.scores_gpt_dir}"
        )
        self.logger.info("=" * 80)
        files = self._discover_score_files()
        for step, in_path in tqdm(
            files, desc="checkpoints", unit="ckpt"
        ):
            self._rescore_one_checkpoint(step, in_path)
        total = self.n_regex_total + self.n_gpt_total
        frac = self.n_regex_total / max(1, total)
        self.logger.info(
            f"Regex-first rescoring complete. Across all checkpoints: "
            f"{self.n_regex_total} regex-resolved, {self.n_gpt_total} "
            f"GPT-fallback ({frac:.1%} resolved without an API call)."
        )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main_logger = logging.getLogger("Main")

    # GPU not strictly required (this is regex / HTTP / CPU work), but the
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

        # ---- OpenAI access (used only for the regex-fallback set) ----
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

    RegexFirstJudgeRescorer(config).run()


if __name__ == "__main__":
    main()