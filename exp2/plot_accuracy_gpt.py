"""Plot eval accuracy across GRPO checkpoints from GPT-judged score files.

This script reads the JSON files produced by ``judge_with_gpt.py``
(located in exp2/outputs/scores_gpt/) and renders a single accuracy
line plot with one curve per prompt language. It does NOT call any
judge, decode any tokens, or load any model -- the per-prompt scores
are already finalised in the input JSON files.

Inputs:
    exp2/outputs/scores_gpt/accuracy_step_<N>.json   (from judge_with_gpt.py)

Outputs:
    exp2/outputs/plots/plot_accuracy_gpt/
        plot_accuracy_gpt.png
        plot_accuracy_gpt_data_summary.json

Standards: see project coding standards. All paths are pathlib.Path.
No os, no argparse, no print. tqdm where appropriate. Logger has
FileHandler + StreamHandler in mandated format.
"""

import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


class PlotAccuracyGPT:
    """Render the accuracy line plot directly from the scores_gpt JSON files."""

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

        # Configurable so you can flip between scores/ and scores_gpt/
        # with a one-line change if ever needed; defaults to scores_gpt/.
        self.scores_dir = Path(
            config.get(
                "scores_dir",
                self.output_dir / "scores_gpt",
            )
        )
        self.plots_dir = self.output_dir / "plots" / "plot_accuracy_gpt"

        self.dpi = int(config.get("dpi", 200))
        self.fig_width = float(config.get("fig_width", 7.0))
        self.fig_height = float(config.get("fig_height", 4.5))

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        self.checkpoint_steps: list[int] = []
        self.per_step_per_lang_acc: dict[int, dict[str, float]] = {}
        self.judge_model: str = "unknown"

    def _setup_logging(self):
        log_file = self.log_dir / "plot_accuracy_gpt.log"
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
    #  Discovery + load
    # ------------------------------------------------------------------ #

    def _discover_score_files(self) -> list[tuple[int, Path]]:
        pat = re.compile(r"^accuracy_step_(\d+)\.json$")
        found: list[tuple[int, Path]] = []
        if not self.scores_dir.exists():
            raise FileNotFoundError(
                f"Scores directory does not exist: {self.scores_dir}. "
                f"Run judge_with_gpt.py first."
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
                f"No accuracy_step_*.json found in {self.scores_dir}. "
                f"Run judge_with_gpt.py first."
            )
        self.logger.info(
            f"Discovered {len(found)} score files: "
            f"steps {[s for s, _ in found]}"
        )
        return found

    def load(self):
        """Read all per-checkpoint JSON files and stash the per-lang acc."""
        files = self._discover_score_files()
        judge_models_seen: set[str] = set()
        for step, path in tqdm(
            files, desc="loading score files", unit="file"
        ):
            with path.open(encoding="utf-8") as f:
                payload = json.load(f)
            per_lang = payload.get("per_language_accuracy") or {}
            # Coerce JSON's "NaN"-as-null back into float NaN, and any
            # numerics into floats.
            cleaned: dict[str, float] = {}
            for lang in self.LANG_LABELS:
                v = per_lang.get(lang)
                if v is None:
                    cleaned[lang] = float("nan")
                else:
                    cleaned[lang] = float(v)
            self.per_step_per_lang_acc[step] = cleaned
            self.checkpoint_steps.append(step)
            jm = payload.get("judge_model")
            if jm:
                judge_models_seen.add(str(jm))

        # Sanity: warn if score files were judged by multiple different
        # models. That should never happen if judge_with_gpt.py was run
        # in one batch, but it's a useful check.
        if len(judge_models_seen) == 1:
            self.judge_model = next(iter(judge_models_seen))
        elif len(judge_models_seen) > 1:
            self.logger.warning(
                f"Score files were judged by multiple models: "
                f"{sorted(judge_models_seen)}. Caption will list all."
            )
            self.judge_model = ", ".join(sorted(judge_models_seen))
        self.logger.info(
            f"Loaded {len(self.checkpoint_steps)} score files. "
            f"Judge model: {self.judge_model}"
        )

    # ------------------------------------------------------------------ #
    #  Plot
    # ------------------------------------------------------------------ #

    def plot(self):
        if not self.checkpoint_steps:
            raise RuntimeError("Call load() before plot().")

        steps = np.asarray(self.checkpoint_steps, dtype=float)
        n_lang = len(self.LANG_LABELS)
        n_ckpt = len(steps)
        Y = np.full((n_lang, n_ckpt), np.nan, dtype=np.float64)
        for j, step in enumerate(self.checkpoint_steps):
            per_lang = self.per_step_per_lang_acc[step]
            for i, lang in enumerate(self.LANG_LABELS):
                Y[i, j] = per_lang.get(lang, float("nan"))

        palette = plt.get_cmap("tab10")
        colors = {
            lang: palette(i) for i, lang in enumerate(self.LANG_LABELS)
        }

        fig, ax = plt.subplots(
            figsize=(self.fig_width, self.fig_height),
            dpi=self.dpi,
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
        ax.set_ylabel("Accuracy (judge: YES = 1, otherwise 0)")
        ax.set_ylim(0.0, 1.0)
        # Let matplotlib's MaxNLocator pick clean x-ticks rather than
        # forcing one per checkpoint; with 30 checkpoints the manual
        # labels overlap.
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(
            title="Prompt language",
            fontsize=8,
            title_fontsize=8,
            loc="best",
            framealpha=0.9,
        )
        judge_short = Path(self.judge_model).name if self.judge_model else "unknown"
        ax.set_title(f"Eval accuracy across GRPO checkpoints ({judge_short})")

        caption = (
            f"One line per language. Accuracy = mean YES rate from the "
            f"judge over all eval prompts of that language. "
            f"CoTs were produced by eval.py via greedy decoding (re-used "
            f"here; no regeneration). Judge model: {judge_short}."
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

        out_path = self.plots_dir / "plot_accuracy_gpt.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Wrote {out_path}")

    # ------------------------------------------------------------------ #
    #  Summary
    # ------------------------------------------------------------------ #

    def write_summary(self):
        summary = {
            "checkpoint_steps": self.checkpoint_steps,
            "n_checkpoints": len(self.checkpoint_steps),
            "languages": list(self.LANG_LABELS),
            "per_step_per_language_accuracy": {
                str(step): self.per_step_per_lang_acc[step]
                for step in self.checkpoint_steps
            },
            "judge_model": self.judge_model,
            "scores_dir": str(self.scores_dir),
            "notes": (
                "Accuracy at each (checkpoint, language) is the mean YES "
                "rate from the OpenAI GPT judge over the eval prompts of "
                "that language. Source: per-checkpoint JSON files "
                "produced by judge_with_gpt.py. This script is purely a "
                "plotting wrapper; it does not call the judge."
            ),
        }
        out = self.plots_dir / "plot_accuracy_gpt_data_summary.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        self.logger.info(f"Wrote {out}")

    # ------------------------------------------------------------------ #
    #  Orchestrator
    # ------------------------------------------------------------------ #

    def run(self):
        self.logger.info("=" * 80)
        self.logger.info(
            f"plot_accuracy_gpt starting; reading from {self.scores_dir}"
        )
        self.logger.info("=" * 80)
        self.load()
        self.plot()
        self.write_summary()
        self.logger.info("plot_accuracy_gpt complete.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main_logger = logging.getLogger("Main")

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

        # ---- Source directory of GPT-judged scores ----
        # Defaults to <output_dir>/scores_gpt; override here to point
        # the same plotter at the Gemma scores/ directory.
        "scores_dir": Path("./exp2/outputs/scores_gpt"),

        # ---- Figure styling ----
        "dpi": 200,
        "fig_width": 7.0,
        "fig_height": 4.5,
    }

    PlotAccuracyGPT(config).run()


if __name__ == "__main__":
    main()