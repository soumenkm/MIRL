"""Eval accuracy across checkpoints, judged by the remote vLLM judge.

This script reuses CoT tokens already produced by eval.py and saved in
exp2/outputs/acts/checkpoint_<step>.npz. No regeneration happens here.
For each (checkpoint, prompt) pair we:

  1. Decode the saved CoT token IDs back to text with the policy tokenizer.
  2. Build a judge prompt with the same format used during training
     (rewards.RewardManager._build_judge_prompt).
  3. Send the prompt to the live remote vLLM judge over HTTP
     (rewards.JudgeClient).
  4. Parse the judge's first non-whitespace word as YES/NO -> 1.0 / 0.0.

Per-checkpoint outputs (cache):
    exp2/outputs/scores/accuracy_step_<step>.json   # indent=4
        Full debug payload: every prompt, every CoT, every judge raw output,
        every judge score, plus per-language averages.

Plotting output:
    exp2/outputs/plots/plot_accuracy/
        plot_accuracy.png                 # one line plot, 7 curves
        plot_accuracy_data_summary.json   # checkpoint steps + accuracies

Caching:
    The per-checkpoint .json is the cache. Reruns are idempotent: if a
    checkpoint already has its accuracy_step_<step>.json, we skip the judge
    round for that step and load the cached per-language averages. This
    means editing plot styling and re-running is free of judge calls.

Standards: see project coding standards. All paths are pathlib.Path. No
os, no argparse, no print. tqdm where appropriate. Logger has FileHandler
+ StreamHandler in mandated format.
"""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from grpo_dataset import GRPOMGSMDataset
from rewards import JudgeClient, RewardManager
from transformers import AutoTokenizer


class PlotAccuracyAcrossCheckpoints:
    """Judge eval prompts at each checkpoint and plot accuracy curves."""

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

    def __init__(self, config: dict):
        self.log_dir = Path(config["log_dir"])
        self.output_dir = Path(config["output_dir"])
        self.data_dir = Path(config["data_dir"])

        self.acts_dir = self.output_dir / "acts"
        self.scores_dir = self.output_dir / "scores"
        self.plots_dir = self.output_dir / "plots" / "plot_accuracy"

        self.tokenizer_path = Path(config["tokenizer_path"])
        self.judge_config = dict(config["judge_config"])
        self.dataset_config = dict(config["dataset_config"])

        self.dpi = int(config.get("dpi", 200))
        self.fig_width = float(config.get("fig_width", 7.0))
        self.fig_height = float(config.get("fig_height", 4.5))
        self.use_cache = bool(config.get("use_cache", True))

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.scores_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        # Loaded lazily in _ensure_judge / _ensure_tokenizer because they
        # are not needed if every checkpoint's score cache already exists.
        self.tokenizer = None
        self.judge_client: JudgeClient | None = None
        self.judge_prompt_builder: RewardManager | None = None
        self.test_dataset: GRPOMGSMDataset | None = None
        self.dataset_index: dict[int, dict] | None = None

        # Aggregated outputs populated by run().
        self.checkpoint_steps: list[int] = []
        self.per_step_per_lang_acc: dict[int, dict[str, float]] = {}

    def _setup_logging(self):
        log_file = self.log_dir / "plot_accuracy.log"
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
    #  Lazy resource setup
    # ------------------------------------------------------------------ #

    def _ensure_tokenizer(self):
        """Load the policy tokenizer the first time it is needed.

        Only required to decode saved cot_tokens -> text. No model weights
        are loaded; the judge runs on a separate node.
        """
        if self.tokenizer is not None:
            return
        self.logger.info(f"Loading tokenizer from {self.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.tokenizer_path),
            trust_remote_code=True,
        )

    def _ensure_judge(self):
        """Connect to the remote judge the first time it is needed.

        Skipped entirely on cache-only reruns (when every checkpoint has a
        cached accuracy_step_*.json file).
        """
        if self.judge_client is not None:
            return
        self.logger.info("Connecting to remote judge...")
        self.judge_client = JudgeClient(self.judge_config)
        self.judge_client.discover()

        # We borrow the judge prompt format and YES/NO parser from
        # RewardManager so this script's accuracy definition is bit-exact
        # with the one used during training. We instantiate with judge
        # weight = 0 to avoid building another JudgeClient inside it.
        self.judge_prompt_builder = RewardManager({
            "log_dir": self.log_dir,
            "weight_accuracy_regex": 1.0,
            "weight_accuracy_judge": 0.0,
            "weight_format": 0.0,
            "weight_language_consistency": 0.0,
        })

    def _ensure_dataset(self):
        """Load test split and index it by dataset row id for fast lookup."""
        if self.dataset_index is not None:
            return
        self.logger.info("Loading test dataset...")
        self.test_dataset = GRPOMGSMDataset(self.dataset_config, split="test")
        # __getitem__ on GRPOMGSMDataset rebuilds the prompt deterministically
        # given the same (seed, idx). We materialize all examples once so we
        # can look up prompt / answer_number / lang by dataset row id.
        self.dataset_index = {}
        for idx in range(len(self.test_dataset)):
            item = self.test_dataset[idx]
            self.dataset_index[idx] = {
                "lang": item["lang"],
                "question_id": item["question_id"],
                "answer_number": int(item["answer_number"]),
                "prompt": item["prompt"],
            }
        self.logger.info(
            f"Test dataset indexed: {len(self.dataset_index)} examples."
        )

    # ------------------------------------------------------------------ #
    #  Discovery
    # ------------------------------------------------------------------ #

    def _discover_checkpoints(self) -> list[tuple[int, Path]]:
        """Find every checkpoint_<step>.npz under acts/, sorted by step."""
        pat = re.compile(r"^checkpoint_(\d+)\.npz$")
        found: list[tuple[int, Path]] = []
        for p in self.acts_dir.iterdir():
            if not p.is_file():
                continue
            m = pat.match(p.name)
            if m:
                found.append((int(m.group(1)), p))
        found.sort(key=lambda x: x[0])
        if not found:
            raise FileNotFoundError(
                f"No checkpoint_*.npz found in {self.acts_dir}. "
                f"Run eval.py first."
            )
        self.logger.info(
            f"Discovered {len(found)} checkpoints: "
            f"{[step for step, _ in found]}"
        )
        return found

    # ------------------------------------------------------------------ #
    #  Per-checkpoint scoring (uses cache)
    # ------------------------------------------------------------------ #

    def _score_one_checkpoint(self, step: int, npz_path: Path) -> dict:
        """Compute (or load) per-language accuracy for this checkpoint.

        Returns:
            {
                "checkpoint_step": int,
                "n_prompts": int,
                "per_language_accuracy": {lang: float, ...},
                "per_language_counts":   {lang: int,   ...},   # n eval'd
                "judge_model": str,
                "timestamp": ISO datetime,
                "prompts": [ per-prompt debug records ... ],   # large
            }
        """
        cache_path = self.scores_dir / f"accuracy_step_{step}.json"
        if self.use_cache and cache_path.exists():
            self.logger.info(f"Loading cached scores: {cache_path}")
            with cache_path.open() as f:
                return json.load(f)

        # Build the per-prompt records by joining (eval npz) with (test dataset).
        self._ensure_tokenizer()
        self._ensure_dataset()
        self._ensure_judge()

        self.logger.info(f"Scoring step={step} from {npz_path}")
        with np.load(npz_path, allow_pickle=True) as d:
            prompt_indices = d["prompt_indices"]
            prompt_langs = d["prompt_langs"]
            cot_lengths = d["cot_lengths"]
            cot_tokens = d["cot_tokens"]

        n = len(prompt_indices)
        records: list[dict] = []
        judge_prompts: list[str] = []
        judge_targets: list[int] = []      # ground-truth answer per record
        record_to_judge_idx: list[int | None] = []  # index in judge_prompts
                                                    # or None if skipped

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
                # Empty CoT (eval.py placeholder); count as wrong without
                # bothering the judge.
                records.append({
                    "dataset_idx": ds_idx,
                    "lang": lang,
                    "question_id": meta["question_id"],
                    "ground_truth": meta["answer_number"],
                    "prompt": meta["prompt"],
                    "cot_text": "",
                    "judge_prompt": None,
                    "judge_raw": None,
                    "judge_score": 0.0,
                    "note": "empty-CoT (skipped judge)",
                })
                record_to_judge_idx.append(None)
                continue

            tok_ids = cot_tokens[i].tolist()
            cot_text = self.tokenizer.decode(
                tok_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            jp = self.judge_prompt_builder._build_judge_prompt(
                cot_text, meta["answer_number"]
            )
            records.append({
                "dataset_idx": ds_idx,
                "lang": lang,
                "question_id": meta["question_id"],
                "ground_truth": meta["answer_number"],
                "prompt": meta["prompt"],
                "cot_text": cot_text,
                "judge_prompt": jp,
                "judge_raw": None,        # filled after batch
                "judge_score": None,      # filled after batch
            })
            record_to_judge_idx.append(len(judge_prompts))
            judge_prompts.append(jp)
            judge_targets.append(meta["answer_number"])

        # Single concurrent batch -- vLLM continuous batcher will fan out
        # at the server side; client concurrency keeps the queue full.
        self.logger.info(
            f"step={step}: sending {len(judge_prompts)} prompts to judge..."
        )
        t0 = time.time()
        raws = self.judge_client.chat_batch(
            judge_prompts,
            max_tokens=8,
            temperature=0.0,
            desc=f"step={step} judge",
        )
        self.logger.info(
            f"step={step}: judge returned in {time.time() - t0:.1f}s"
        )

        # Fill in judge_raw / judge_score on the records.
        for rec, jidx in zip(records, record_to_judge_idx):
            if jidx is None:
                continue
            raw = raws[jidx]
            score = RewardManager._parse_yes_no(raw)
            rec["judge_raw"] = raw
            rec["judge_score"] = float(score)

        # Per-language averages over the records we attempted.
        per_lang_acc: dict[str, float] = {}
        per_lang_n: dict[str, int] = {}
        for lang in self.LANG_LABELS:
            in_lang = [r for r in records if r["lang"] == lang]
            per_lang_n[lang] = len(in_lang)
            if in_lang:
                per_lang_acc[lang] = float(
                    np.mean([r["judge_score"] for r in in_lang])
                )
            else:
                per_lang_acc[lang] = float("nan")

        payload = {
            "checkpoint_step": int(step),
            "n_prompts": len(records),
            "per_language_accuracy": per_lang_acc,
            "per_language_counts": per_lang_n,
            "judge_model": self.judge_client.served_model_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "prompts": records,
        }

        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
        self.logger.info(f"Wrote {cache_path}")

        self._log_step_summary(step, per_lang_acc, per_lang_n)
        return payload

    def _log_step_summary(
        self,
        step: int,
        per_lang_acc: dict[str, float],
        per_lang_n: dict[str, int],
    ):
        self.logger.info(f"step={step} per-language accuracy:")
        for lang in self.LANG_LABELS:
            n = per_lang_n.get(lang, 0)
            acc = per_lang_acc.get(lang, float("nan"))
            self.logger.info(
                f"  {lang}: {acc:.3f}  (n={n})"
            )

    # ------------------------------------------------------------------ #
    #  Aggregation + plotting
    # ------------------------------------------------------------------ #

    def aggregate(self):
        """Score every discovered checkpoint (or load its cache)."""
        ckpts = self._discover_checkpoints()
        for step, npz_path in ckpts:
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
        # Stack into (n_lang, n_ckpt) for easy plotting.
        n_lang = len(self.LANG_LABELS)
        n_ckpt = len(steps)
        Y = np.full((n_lang, n_ckpt), np.nan, dtype=np.float64)
        for j, step in enumerate(self.checkpoint_steps):
            per_lang = self.per_step_per_lang_acc[step]
            for i, lang in enumerate(self.LANG_LABELS):
                v = per_lang.get(lang, float("nan"))
                Y[i, j] = v

        palette = plt.get_cmap("tab10")
        colors = {lang: palette(i) for i, lang in enumerate(self.LANG_LABELS)}

        fig, ax = plt.subplots(
            figsize=(self.fig_width, self.fig_height),
            dpi=self.dpi,
        )
        for i, lang in enumerate(self.LANG_LABELS):
            # Drop NaN points (languages with no prompts at some step).
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
        ax.set_ylabel("Accuracy (judge: YES = 1, otherwise 0)")
        ax.set_ylim(0.0, 1.0)
        # ax.set_xticks(steps)
        # ax.set_xticklabels([str(int(s)) for s in steps], fontsize=8)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(
            title="Prompt language",
            fontsize=8,
            title_fontsize=8,
            loc="best",
            framealpha=0.9,
        )
        ax.set_title("Eval accuracy across GRPO checkpoints")

        caption = (
            f"One line per language. Accuracy = mean YES rate from the remote "
            f"vLLM judge over all eval prompts of that language. CoTs were "
            f"produced by eval.py via greedy decoding (re-used here; no "
            f"regeneration). Judge model: "
            f"{self._judge_model_for_caption()}."
        )
        fig.subplots_adjust(bottom=0.24)
        fig.text(
            0.5,
            0.02,
            caption,
            ha="center",
            va="bottom",
            fontsize=8,
            wrap=True,
        )

        out_path = self.plots_dir / "plot_accuracy.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Wrote {out_path}")

    def _judge_model_for_caption(self) -> str:
        """Get the judge model name (from any cache file or live client)."""
        if self.judge_client is not None and self.judge_client.served_model_name:
            return Path(self.judge_client.served_model_name).name
        # Fallback: pull from any cache file.
        for step in self.checkpoint_steps:
            cache_path = self.scores_dir / f"accuracy_step_{step}.json"
            if cache_path.exists():
                with cache_path.open() as f:
                    return Path(json.load(f).get("judge_model", "unknown")).name
        return "unknown"

    def write_summary(self):
        """Small JSON summary alongside the figure."""
        summary = {
            "checkpoint_steps": self.checkpoint_steps,
            "n_checkpoints": len(self.checkpoint_steps),
            "languages": list(self.LANG_LABELS),
            "per_step_per_language_accuracy": {
                str(step): self.per_step_per_lang_acc[step]
                for step in self.checkpoint_steps
            },
            "judge_model": self._judge_model_for_caption(),
            "notes": (
                "Accuracy at each (checkpoint, language) is the mean YES "
                "rate from the remote vLLM judge over the eval prompts of "
                "that language. CoTs are decoded from cot_tokens stored "
                "in exp2/outputs/acts/checkpoint_<step>.npz. No "
                "regeneration -- this script only decodes + judges."
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
            f"plot_accuracy starting; reading from {self.acts_dir}; "
            f"cache at {self.scores_dir}"
        )
        self.logger.info("=" * 80)
        self.aggregate()
        self.plot()
        self.write_summary()
        self.logger.info("plot_accuracy complete.")


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
        "log_dir":    Path("./exp2/logs"),
        "output_dir": Path("./exp2/outputs"),
        "data_dir":   Path("./exp2/data"),

        # ---- Tokenizer (must match the one eval.py used) ----
        "tokenizer_path": Path("./models/Qwen2.5-7B-Instruct"),

        # ---- Figure styling ----
        "dpi": 200,
        "fig_width": 7.0,
        "fig_height": 4.5,
        "use_cache": False, 

        # ---- Remote judge (must match the live judge_server.py job) ----
        "judge_config": {
            "connection_file": Path("./exp2/outputs/judge_connection.json"),
            "log_dir":          Path("./exp2/logs"),
            "discovery_timeout": 1800,
            "discovery_poll_interval": 5.0,
            "request_timeout":   120,
            "max_retries":       3,
            "retry_backoff":     2.0,
            "max_concurrent_requests": 32,
        },

        # ---- Test dataset (must match the one eval.py used) ----
        "dataset_config": {
            "data_dir":     Path("./exp2/data"),
            "log_dir":      Path("./exp2/logs"),
            "languages":    ["en", "bn", "te", "th", "ru", "ja", "zh"],
            "num_few_shot": 1,
            "seed":         42,
        },
    }

    PlotAccuracyAcrossCheckpoints(config).run()


if __name__ == "__main__":
    main()