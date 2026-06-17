"""Score eval checkpoints with a regex-first / GPT-fallback judge, then plot.

This single script REPLACES the old two-step flow:
    1. plot_accuracy.py        -> produced scores/accuracy_step_<N>.json via
                                  the remote vLLM (Gemma) judge, then plotted.
    2. judge_with_gpt.py       -> re-scored those score files with a
                                  regex-first / GPT-fallback judge into
                                  scores_gpt/.

Here both happen in one pass. For each checkpoint:

  1. Read the saved CoT token IDs from acts/checkpoint_<step>.npz (produced by
     eval.py; no regeneration here) and decode them with the policy tokenizer.
  2. Stage 1 (regex): extract the student's final integer answer from the CoT
     using robust patterns (primarily the mandated "Final Answer: <number>"
     line, with fallbacks). If a clean integer is extracted, score 1.0 iff it
     equals ground_truth, else 0.0 -- no API call.
  3. Stage 2 (GPT fallback): only for prompts regex cannot resolve, query an
     OpenAI model (default gpt-4o-mini) and parse YES/NO.
  4. Write scores_gpt/accuracy_step_<step>.json in the canonical format (one
     record per prompt with judge_method "regex"/"gpt", judge_raw, judge_score;
     plus per-language accuracies and the regex/GPT split counts).
  5. After all checkpoints: plot per-language accuracy curves and write a JSON
     summary.

The per-prompt ``judge_method`` field marks whether the score came from the
regex stage or the GPT fallback; ``judge_raw`` carries ``regex_extracted=<N>``
for regex hits and the raw GPT reply for fallbacks, so the split is auditable.

Caching: scores_gpt/accuracy_step_<step>.json is the cache. With use_cache
True, a checkpoint whose score file already exists is loaded, not rescored
(no API calls). Set use_cache False to force a full rescore.

Standards: see project coding standards. All paths are pathlib.Path. No os,
no argparse, no print. tqdm where appropriate. Logger has FileHandler +
StreamHandler in the mandated format.
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from openai import (
    OpenAI,
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
)

from grpo_dataset import GRPOMGSMDataset


class PlotAccuracyRegexFirst:
    """Score eval checkpoints (regex-first, GPT fallback) and plot curves."""

    LANG_LABELS = ("en", "bn", "te", "th", "ru", "ja", "zh")

    LANG_DISPLAY = {
        "en": "English",
        "bn": "Bengali",
        "te": "Telugu",
        "th": "Thai",
        "ru": "Russian",
        "ja": "Japanese",
        "zh": "Chinese",
    }

    SYSTEM_PROMPT = (
        "You are a strict math answer verifier. You receive a math "
        "problem's correct integer answer and a student's full "
        "response. Decide whether the student's response arrives at "
        "the correct final answer. Respond with exactly one word: "
        "YES or NO. Do not include any other text, explanation, or "
        "punctuation."
    )

    # A number token: optional sign, optional leading currency, digits with
    # optional thousands separators, and an optional decimal tail. The
    # decimal tail is captured so _clean_int can REJECT it (MGSM answers are
    # integers; a decimal output is wrong and must defer to the GPT fallback
    # rather than be silently truncated).
    _NUM = r"[-+]?\$?\s*\d[\d,]*(?:\.\d+)?"

    def __init__(self, config: dict):
        self.log_dir = Path(config["log_dir"])
        self.output_dir = Path(config["output_dir"])
        self.data_dir = Path(config["data_dir"])

        self.acts_dir = self.output_dir / "acts"
        # Output score files mirror judge_with_gpt.py's location/format.
        self.scores_dir = self.output_dir / "scores_gpt"
        self.plots_dir = self.output_dir / "plots" / "plot_accuracy_regex"

        self.tokenizer_path = Path(config["tokenizer_path"])
        self.dataset_config = dict(config["dataset_config"])

        # OpenAI / GPT-fallback knobs
        self.api_key_path = Path(config["api_key_path"]).expanduser()
        self.model_name = str(config.get("model_name", "gpt-4o-mini"))
        self.max_workers = int(config.get("max_workers", 8))
        self.request_max_tokens = int(config.get("request_max_tokens", 4))
        self.request_temperature = float(config.get("request_temperature", 0.0))
        self.request_timeout = float(config.get("request_timeout", 60))
        self.max_retries = int(config.get("max_retries", 4))
        self.retry_backoff = float(config.get("retry_backoff", 2.0))

        # Figure styling
        self.dpi = int(config.get("dpi", 200))
        self.fig_width = float(config.get("fig_width", 7.0))
        self.fig_height = float(config.get("fig_height", 4.5))
        self.use_cache = bool(config.get("use_cache", True))
        # Store the prompt in score-file records the SAME way eval.py fed it
        # to the model (chat-templated for Instruct models), so the audit
        # field is internally consistent. Must match eval.py's setting.
        self.use_chat_template = bool(config.get("use_chat_template", True))

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.scores_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()
        self._compile_patterns()

        # Lazily-initialised; only built if actually needed.
        self.tokenizer = None
        self._client: OpenAI | None = None
        self.test_dataset: GRPOMGSMDataset | None = None
        self.dataset_index: dict[int, dict] | None = None

        # Running tallies for the final regex-vs-GPT report.
        self.n_regex_total = 0
        self.n_gpt_total = 0

        # Aggregated outputs populated by aggregate().
        self.checkpoint_steps: list[int] = []
        self.per_step_per_lang_acc: dict[int, dict[str, float]] = {}

    def _setup_logging(self):
        log_file = self.log_dir / "plot_accuracy_regex.log"
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
        """Compile the ordered answer-extraction patterns once.

        Group(1) holds the raw number-with-noise, cleaned by _clean_int.
        Patterns are ordered most-reliable-first: the mandated
        "Final Answer:" line, then \\boxed{}, "the answer is", "Answer:".
        """
        n = self._NUM
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
                re.compile(rf"\banswer\s*[:\-]\s*\**\s*({n})", re.IGNORECASE),
            ),
        ]

    # ------------------------------------------------------------------ #
    #  Lazy resources
    # ------------------------------------------------------------------ #

    def _ensure_tokenizer(self):
        if self.tokenizer is not None:
            return
        self.logger.info(f"Loading tokenizer from {self.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.tokenizer_path),
            trust_remote_code=True,
        )
        if self.use_chat_template and self.tokenizer.chat_template is None:
            self.logger.warning(
                "use_chat_template=True but tokenizer has no chat_template; "
                "storing raw prompts in score records instead."
            )
            self.use_chat_template = False

    def _prompt_for_record(self, meta: dict) -> str:
        """Reconstruct the prompt EXACTLY as eval.py fed it to the model.

        eval.py renders system+user turns through apply_chat_template with
        add_generation_prompt=True. We reproduce that here so the score-file
        ``prompt`` audit field matches what was actually generated from.
        Falls back to the raw prompt if chat templating is disabled/absent.
        Note: this field is for auditing only; scoring uses ``cot_text``.
        """
        if self.use_chat_template:
            messages = [
                {"role": "system", "content": meta["system"]},
                {"role": "user", "content": meta["user"]},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return meta["prompt"]

    def _ensure_dataset(self):
        """Load test split and index it by dataset row id for fast lookup."""
        if self.dataset_index is not None:
            return
        self.logger.info("Loading test dataset...")
        self.test_dataset = GRPOMGSMDataset(self.dataset_config, split="test")
        self.dataset_index = {}
        for idx in range(len(self.test_dataset)):
            item = self.test_dataset[idx]
            self.dataset_index[idx] = {
                "lang": item["lang"],
                "question_id": item["question_id"],
                "answer_number": int(item["answer_number"]),
                "prompt": item["prompt"],   # raw text (kept as fallback)
                "system": item["system"],   # for chat-templated reconstruction
                "user": item["user"],
            }
        self.logger.info(
            f"Test dataset indexed: {len(self.dataset_index)} examples."
        )

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
                raise ValueError(f"API key file {self.api_key_path} is empty.")
            self._client = OpenAI(api_key=api_key, timeout=self.request_timeout)
            self.logger.info(
                f"OpenAI client initialised for GPT fallback "
                f"(model={self.model_name}, max_workers={self.max_workers})."
            )
        return self._client

    # ------------------------------------------------------------------ #
    #  Discovery
    # ------------------------------------------------------------------ #

    def _discover_checkpoints(self) -> list[tuple[int, Path]]:
        """Find every checkpoint_<step>.npz under acts/, sorted by step."""
        pat = re.compile(r"^checkpoint_(\d+)\.npz$")
        found: list[tuple[int, Path]] = []
        if not self.acts_dir.exists():
            raise FileNotFoundError(
                f"Activations directory does not exist: {self.acts_dir}. "
                f"Run eval.py first."
            )
        for p in self.acts_dir.iterdir():
            if not p.is_file():
                continue
            m = pat.match(p.name)
            if m:
                found.append((int(m.group(1)), p))
        found.sort(key=lambda x: x[0])
        if not found:
            raise FileNotFoundError(
                f"No checkpoint_*.npz found in {self.acts_dir}. Run eval.py first."
            )
        self.logger.info(
            f"Discovered {len(found)} checkpoints: {[s for s, _ in found]}"
        )
        return found

    # ------------------------------------------------------------------ #
    #  Regex stage
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clean_int(raw: str | None) -> int | None:
        """Turn a raw matched number (with noise) into a clean int, or None.

        Strips currency symbols, whitespace, and thousands separators.
        Returns None for decimals or non-integers, so the caller treats it as
        a regex miss (GPT fallback adjudicates).
        """
        if raw is None:
            return None
        s = raw.strip().replace("$", "").replace(" ", "").replace(",", "")
        if not re.fullmatch(r"[-+]?\d+", s):
            return None
        try:
            return int(s)
        except ValueError:
            return None

    def _extract_answer(self, cot_text: str) -> int | None:
        """Extract the student's final integer answer from the CoT, or None.

        Tries patterns in order; within each, prefers the LAST occurrence
        (later answers supersede earlier working values, and a duplicated
        final-answer line settles on the same value).
        """
        if not cot_text:
            return None
        for _name, pat in self._patterns:
            matches = list(pat.finditer(cot_text))
            if not matches:
                continue
            for m in reversed(matches):
                val = self._clean_int(m.group(1))
                if val is not None:
                    return val
        return None

    # ------------------------------------------------------------------ #
    #  GPT fallback
    # ------------------------------------------------------------------ #

    def _build_chat_messages(self, ground_truth: int, cot_text: str) -> list[dict]:
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
        """Parse the first non-whitespace word as YES/NO -> 1.0/0.0."""
        if not response_text:
            return 0.0
        tokens = response_text.strip().split()
        if not tokens:
            return 0.0
        first = re.sub(r"[^A-Za-z]", "", tokens[0]).upper()
        return 1.0 if first == "YES" else 0.0

    def _chat_once(self, messages: list[dict]) -> str:
        """Send one Chat Completions request, retrying on transient errors."""
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
                return resp.choices[0].message.content or ""
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
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
                        f"{self.max_retries}): {e}. Retrying in {sleep_s:.1f}s."
                    )
                    time.sleep(sleep_s)
                else:
                    self.logger.error(f"OpenAI non-retriable error: {e}")
                    raise
        self.logger.error(f"All {self.max_retries} retries failed: {last_err}")
        return ""

    # ------------------------------------------------------------------ #
    #  Per-checkpoint scoring (regex-first, GPT fallback)
    # ------------------------------------------------------------------ #

    def _score_one_checkpoint(self, step: int, npz_path: Path) -> dict:
        """Compute (or load) per-language accuracy for one checkpoint.

        Output payload matches the canonical scores_gpt format.
        """
        out_path = self.scores_dir / f"accuracy_step_{step}.json"
        if self.use_cache and out_path.exists():
            self.logger.info(f"Loading cached scores: {out_path}")
            with out_path.open(encoding="utf-8") as f:
                return json.load(f)

        self._ensure_tokenizer()
        self._ensure_dataset()

        self.logger.info(f"Scoring step={step} from {npz_path}")
        with np.load(npz_path, allow_pickle=True) as d:
            prompt_indices = d["prompt_indices"]
            prompt_langs = d["prompt_langs"]
            cot_lengths = d["cot_lengths"]
            cot_tokens = d["cot_tokens"]

        n = len(prompt_indices)

        # Build per-prompt records first (decode CoT, attach metadata).
        records: list[dict] = []
        for i in tqdm(range(n), desc=f"step={step} decode", unit="prompt"):
            ds_idx = int(prompt_indices[i])
            lang = str(prompt_langs[i])
            c_len = int(cot_lengths[i])

            meta = self.dataset_index.get(ds_idx)
            if meta is None:
                self.logger.warning(
                    f"Dataset idx {ds_idx} missing from test split; skipping."
                )
                continue

            if c_len == 0:
                cot_text = ""
            else:
                tok_ids = cot_tokens[i].tolist()
                cot_text = self.tokenizer.decode(
                    tok_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

            records.append({
                "dataset_idx": ds_idx,
                "lang": lang,
                "question_id": meta["question_id"],
                "ground_truth": meta["answer_number"],
                "prompt": self._prompt_for_record(meta),
                "cot_text": cot_text,
                "judge_prompt": None,   # filled for GPT-fallback records below
                "judge_raw": None,
                "judge_score": None,
                "judge_method": None,
            })

        # Stage 1: regex. Resolve what we can; collect misses for GPT.
        gpt_indices: list[int] = []
        for i, rec in enumerate(records):
            cot = rec["cot_text"]
            if not cot:
                # Empty CoT -> failed generation -> wrong, no API call.
                rec["judge_score"] = 0.0
                rec["judge_raw"] = "regex_extracted=None (empty-CoT)"
                rec["judge_method"] = "regex"
                continue
            try:
                gt = int(rec["ground_truth"])
            except (KeyError, TypeError, ValueError):
                gt = None
            extracted = self._extract_answer(str(cot))
            if extracted is not None and gt is not None:
                rec["judge_score"] = 1.0 if extracted == gt else 0.0
                rec["judge_raw"] = f"regex_extracted={extracted}"
                rec["judge_method"] = "regex"
            else:
                gpt_indices.append(i)

        n_gpt = len(gpt_indices)
        n_regex = len(records) - n_gpt
        self.n_regex_total += n_regex
        self.n_gpt_total += n_gpt
        self.logger.info(
            f"step={step}: regex resolved {n_regex}/{len(records)}; "
            f"{n_gpt} deferred to GPT fallback."
        )

        # Stage 2: GPT fallback for the unresolved set, concurrently.
        if gpt_indices:
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {}
                for i in gpt_indices:
                    rec = records[i]
                    messages = self._build_chat_messages(
                        ground_truth=int(rec["ground_truth"]),
                        cot_text=str(rec["cot_text"]),
                    )
                    # Store the judge prompt (user turn) for auditability.
                    rec["judge_prompt"] = messages[-1]["content"]
                    futures[pool.submit(self._chat_once, messages)] = i
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"step={step} GPT-fallback",
                    unit="req",
                ):
                    i = futures[fut]
                    try:
                        raw = fut.result()
                    except Exception as e:
                        self.logger.error(
                            f"step={step} prompt #{i} permanently failed: {e}"
                        )
                        raw = ""
                    records[i]["judge_raw"] = raw
                    records[i]["judge_score"] = self._parse_yes_no(raw)
                    records[i]["judge_method"] = "gpt"
            elapsed = time.time() - t0
            self.logger.info(
                f"step={step}: GPT fallback returned for {n_gpt} prompts "
                f"in {elapsed:.1f}s "
                f"(avg {elapsed / max(1, n_gpt) * 1000:.0f} ms/req)."
            )

        # Per-language averages.
        per_lang_acc: dict[str, float] = {}
        per_lang_n: dict[str, int] = {}
        for lang in self.LANG_LABELS:
            in_lang = [r for r in records if r["lang"] == lang]
            per_lang_n[lang] = len(in_lang)
            if in_lang:
                per_lang_acc[lang] = float(
                    np.mean([float(r["judge_score"]) for r in in_lang])
                )
            else:
                per_lang_acc[lang] = float("nan")

        payload = {
            "checkpoint_step": int(step),
            "n_prompts": len(records),
            "per_language_accuracy": per_lang_acc,
            "per_language_counts": per_lang_n,
            "judge_model": f"regex+{self.model_name}",
            "n_regex": n_regex,
            "n_gpt_fallback": n_gpt,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompts": records,
        }

        # Atomic write: tmp then rename.
        tmp_path = out_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
        tmp_path.replace(out_path)
        self.logger.info(f"Wrote {out_path}")

        self._log_step_summary(step, per_lang_acc, per_lang_n)
        return payload

    def _log_step_summary(self, step, per_lang_acc, per_lang_n):
        self.logger.info(f"step={step} regex-first per-language accuracy:")
        for lang in self.LANG_LABELS:
            n = per_lang_n.get(lang, 0)
            acc = per_lang_acc.get(lang, float("nan"))
            self.logger.info(f"  {lang}: {acc:.3f}  (n={n})")

    # ------------------------------------------------------------------ #
    #  Aggregation + plotting
    # ------------------------------------------------------------------ #

    def aggregate(self):
        ckpts = self._discover_checkpoints()
        for step, npz_path in tqdm(ckpts, desc="checkpoints", unit="ckpt"):
            payload = self._score_one_checkpoint(step, npz_path)
            self.checkpoint_steps.append(step)
            self.per_step_per_lang_acc[step] = payload["per_language_accuracy"]
        self.logger.info(
            f"Aggregated accuracies for {len(self.checkpoint_steps)} checkpoints."
        )

    def plot(self):
        if not self.checkpoint_steps:
            raise RuntimeError("Call aggregate() before plot().")

        steps = np.asarray(self.checkpoint_steps, dtype=float)
        n_lang = len(self.LANG_LABELS)
        n_ckpt = len(steps)
        Y = np.full((n_lang, n_ckpt), np.nan, dtype=np.float64)
        for j, step in enumerate(self.checkpoint_steps):
            per_lang = self.per_step_per_lang_acc[step]
            for i, lang in enumerate(self.LANG_LABELS):
                Y[i, j] = per_lang.get(lang, float("nan"))

        palette = plt.get_cmap("tab10")
        colors = {lang: palette(i) for i, lang in enumerate(self.LANG_LABELS)}

        fig, ax = plt.subplots(
            figsize=(self.fig_width, self.fig_height), dpi=self.dpi
        )
        for i, lang in enumerate(self.LANG_LABELS):
            y = Y[i, :]
            mask = ~np.isnan(y)
            if not mask.any():
                continue
            ax.plot(
                steps[mask],
                y[mask],
                marker="o",
                markersize=4,
                linewidth=1.5,
                color=colors[lang],
                label=self.LANG_DISPLAY[lang],
            )

        ax.set_xlabel("GRPO step")
        ax.set_ylabel("Accuracy (regex-first; GPT fallback YES = 1)")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(
            title="Prompt language",
            fontsize=8,
            title_fontsize=8,
            loc="best",
            framealpha=0.9,
        )
        ax.set_title("Eval accuracy across GRPO checkpoints (regex-first judge)")

        total = self.n_regex_total + self.n_gpt_total
        frac = self.n_regex_total / max(1, total)
        caption = (
            f"One line per language. Accuracy = mean correct rate. Answers "
            f"extracted by regex where possible ({frac:.0%} of prompts); the "
            f"remainder adjudicated by {self.model_name}. CoTs were produced "
            f"by eval.py (greedy; re-used here, no regeneration)."
        )
        fig.subplots_adjust(bottom=0.24)
        fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=8, wrap=True)

        out_path = self.plots_dir / "plot_accuracy.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Wrote {out_path}")

    def write_summary(self):
        summary = {
            "checkpoint_steps": self.checkpoint_steps,
            "n_checkpoints": len(self.checkpoint_steps),
            "languages": list(self.LANG_LABELS),
            "per_step_per_language_accuracy": {
                str(step): self.per_step_per_lang_acc[step]
                for step in self.checkpoint_steps
            },
            "judge_model": f"regex+{self.model_name}",
            "n_regex_total": self.n_regex_total,
            "n_gpt_fallback_total": self.n_gpt_total,
            "notes": (
                "Accuracy at each (checkpoint, language) is the mean correct "
                "rate. Final answers are extracted by regex from the CoT where "
                "possible; prompts regex cannot resolve are sent to an OpenAI "
                "model for a YES/NO verdict. Per-prompt judge_method records "
                "which stage decided each score. CoTs are decoded from "
                "cot_tokens in acts/checkpoint_<step>.npz; no regeneration."
            ),
        }
        out = self.plots_dir / "plot_accuracy_data_summary.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        self.logger.info(f"Wrote {out}")

    # ------------------------------------------------------------------ #
    #  Orchestrator
    # ------------------------------------------------------------------ #

    def run(self):
        self.logger.info("=" * 80)
        self.logger.info(
            f"plot_accuracy_regex starting; reading {self.acts_dir}; "
            f"scores -> {self.scores_dir}"
        )
        self.logger.info("=" * 80)
        self.aggregate()
        self.plot()
        self.write_summary()
        total = self.n_regex_total + self.n_gpt_total
        frac = self.n_regex_total / max(1, total)
        self.logger.info(
            f"Complete. Across all checkpoints: {self.n_regex_total} "
            f"regex-resolved, {self.n_gpt_total} GPT-fallback "
            f"({frac:.1%} resolved without an API call)."
        )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main_logger = logging.getLogger("Main")

    # GPU not strictly required (tokenizer decode + regex + HTTP), but the
    # pipeline is GPU-node-only and the standards expect the check.
    # if not torch.cuda.is_available():
    #     raise RuntimeError("No GPU detected.")
    # main_logger.info(
    #     f"GPUs available: {torch.cuda.device_count()} "
    #     f"({torch.cuda.get_device_name(0)})"
    # )

    config = {
        # ---- Required path roots ----
        "log_dir":    Path("./exp3/logs"),
        "output_dir": Path("./exp3/outputs"),
        "data_dir":   Path("./exp3/data"),

        # ---- Tokenizer (must match the one eval.py used) ----
        "tokenizer_path": Path("./models/Qwen2.5-7B-Instruct"),

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

        # ---- Figure styling ----
        "dpi": 200,
        "fig_width": 7.0,
        "fig_height": 4.5,

        # ---- Cache: load existing scores_gpt/accuracy_step_<N>.json ----
        "use_cache": True,

        # Store the prompt audit field chat-templated, matching eval.py.
        "use_chat_template": True,

        # ---- Test dataset (must match the one eval.py used) ----
        "dataset_config": {
            "data_dir":     Path("./exp3/data"),
            "log_dir":      Path("./exp3/logs"),
            "languages":    ["en", "bn", "te", "th", "ru", "ja", "zh"],
            "num_few_shot": 1,
            "seed":         42,
        },
    }

    PlotAccuracyRegexFirst(config).run()


if __name__ == "__main__":
    main()