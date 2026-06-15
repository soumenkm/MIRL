"""Plot 1 -- CoT Language Probability.

Renders two figure families from the .npz files produced by eval.py.

Heatmaps (one per (prompt-language, target-language) pair, 7 x 7 = 49):
    Q[j, i, P, T] = average over prompts p of language P, then over the
                    CoT tokens of those prompts, of the renormalized
                    top-50 logit-lens probability that layer i at
                    checkpoint j thinks "in language T".
    Y-axis: layer index (0 = first transformer block, at bottom).
    X-axis: GRPO step (linear).
    Color:  Q[j, i, P, T] in [0, 1] with shared colormap viridis.

Line plots (one per prompt language, 7 total):
    Y-axis: layer-averaged Q,  (1/L) sum_i Q[j, i, P, T], for each T.
    X-axis: GRPO step (linear).
    One curve per target language T, plotted on the same axes.

Inputs:
    exp2/outputs/acts/checkpoint_*.npz      (produced by eval.py)

Outputs:
    exp2/outputs/plots/plot1/
        plot1_heatmap_prompt-<P>_target-<T>.png   x 49
        plot1_lineplot_prompt-<P>.png             x 7
        plot1_data_summary.json                   (metadata: ckpt steps,
                                                   prompt counts, etc.)

Behaviour:
    - Auto-discovers all checkpoint_*.npz files in acts/. Running this
      after the full eval finishes simply produces denser plots without
      any config edit.
    - Skips a (prompt, target) heatmap entirely if no prompts exist for
      that prompt language (e.g. very small smoke tests).
    - Tolerates variable CoT length per prompt -- arrays are stored as
      object dtype in eval.py and averaged per prompt before stacking.

Standards: see project coding standards. All paths are pathlib.Path.
Logging to file + console. tqdm where appropriate. No os, no argparse,
no print, no top-level code outside main() / if __name__.
"""

import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


class Plot1CoTLanguageProbability:
    """Aggregate eval .npz files into Plot 1 figures."""

    # Studied languages. Must match the LANG_LABELS axis stored by eval.py.
    LANG_LABELS = ("en", "bn", "te", "th", "ru", "ja", "zh")

    # Human-readable language names for figure titles.
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

        # eval.py writes activations to <output_dir>/acts/. plot1 reads
        # from there and writes to <output_dir>/plots/plot1/.
        self.acts_dir = self.output_dir / "acts"
        self.plots_dir = self.output_dir / "plots" / "plot1"

        self.top_k = int(config.get("top_k", 50))
        self.dpi = int(config.get("dpi", 200))
        self.fig_width = float(config.get("fig_width", 6.0))
        self.fig_height = float(config.get("fig_height", 4.5))
        self.cmap = str(config.get("cmap", "viridis"))

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        # State populated by load() + aggregate().
        self.checkpoint_steps: list[int] = []
        self.n_layers: int | None = None
        self.lang_to_n_prompts: dict[str, int] = {}
        # Q has shape (n_ckpt, L, 7, 7): Q[j, i, p, t]
        self.Q: np.ndarray | None = None

    def _setup_logging(self):
        log_file = self.log_dir / "plot1.log"
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

    def _discover_checkpoints(self) -> list[Path]:
        """Find every checkpoint_<step>.npz in acts_dir, sorted by step.

        The auto-discovery is what makes plot1.py re-runnable as soon as
        eval.py writes new files -- no config change needed.
        """
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
                f"No checkpoint_*.npz files found in {self.acts_dir}. "
                f"Run eval.py first."
            )
        self.logger.info(
            f"Discovered {len(found)} checkpoints: "
            f"{[step for step, _ in found]}"
        )
        return [path for _, path in found]

    # ------------------------------------------------------------------ #
    #  Aggregation
    # ------------------------------------------------------------------ #

    def aggregate(self):
        """Build the Q tensor of shape (n_ckpt, L, 7, 7).

        Per checkpoint .npz:
          1. Load top50_lang_renorm (object array, len N, each (L, C_p, 7)).
          2. For each prompt p (lang P): average over its CoT axis ->
             (L, 7).
          3. For each prompt language P (rows in the 7-axis-2): average
             the per-prompt (L, 7) vectors across prompts of that language
             -> a (L, 7) row in Q[j, :, P, :].
          4. The 7 columns of that row are the 7 target-language values.

        Empty-CoT prompts (cot_length == 0) are skipped so they do not
        bias the per-language mean toward zero.
        """
        npz_paths = self._discover_checkpoints()
        n_ckpt = len(npz_paths)
        L_langs = len(self.LANG_LABELS)

        steps: list[int] = []
        Q: np.ndarray | None = None
        n_layers_seen: int | None = None
        lang_to_n_prompts: dict[str, int] = {l: 0 for l in self.LANG_LABELS}

        for j, path in enumerate(tqdm(
            npz_paths, desc="aggregating checkpoints", unit="ckpt"
        )):
            with np.load(path, allow_pickle=True) as d:
                step = int(d["checkpoint_step"])
                n_layers = int(d["n_layers"])
                lang_labels = list(d["lang_labels"])
                prompt_langs = d["prompt_langs"]
                cot_lengths = d["cot_lengths"]
                top50 = d["top50_lang_renorm"]

            if list(lang_labels) != list(self.LANG_LABELS):
                raise ValueError(
                    f"{path.name} has lang_labels={lang_labels}, "
                    f"expected {self.LANG_LABELS}"
                )

            if n_layers_seen is None:
                n_layers_seen = n_layers
                Q = np.zeros((n_ckpt, n_layers, L_langs, L_langs), dtype=np.float64)
            elif n_layers != n_layers_seen:
                raise ValueError(
                    f"{path.name} has n_layers={n_layers}, expected "
                    f"{n_layers_seen} from earlier file."
                )

            steps.append(step)

            # Aggregate per prompt language.
            for p_idx, P in enumerate(self.LANG_LABELS):
                mask = (prompt_langs == P) & (cot_lengths > 0)
                if not mask.any():
                    # Nothing to aggregate for this language at this
                    # checkpoint. Leave Q[j, :, p_idx, :] at zero and
                    # remember we did so for the summary.
                    continue
                # Stack per-prompt CoT-averages: (n_kept, L, 7).
                per_prompt_means = []
                for q_idx in np.where(mask)[0]:
                    arr = top50[q_idx].astype(np.float32)  # (L, C_p, 7) fp16->fp32
                    if arr.shape[1] == 0:
                        continue
                    per_prompt_means.append(arr.mean(axis=1))  # (L, 7)
                if not per_prompt_means:
                    continue
                stacked = np.stack(per_prompt_means, axis=0)  # (n, L, 7)
                Q[j, :, p_idx, :] = stacked.mean(axis=0)  # (L, 7)
                # Track per-language counts using the largest seen across
                # checkpoints (typically all checkpoints have the same N
                # because the eval set is fixed).
                lang_to_n_prompts[P] = max(
                    lang_to_n_prompts[P], int(stacked.shape[0])
                )

        self.checkpoint_steps = steps
        self.n_layers = n_layers_seen
        self.Q = Q
        self.lang_to_n_prompts = lang_to_n_prompts

        self.logger.info(
            f"Aggregated Q shape={Q.shape} "
            f"(checkpoints={n_ckpt}, layers={n_layers_seen}, langs={L_langs})"
        )
        for P in self.LANG_LABELS:
            self.logger.info(
                f"  prompt lang {P}: {lang_to_n_prompts[P]} prompts averaged"
            )

    # ------------------------------------------------------------------ #
    #  Plotting helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_lang(s: str) -> str:
        """File-safe representation of a language code (no slashes/spaces)."""
        return re.sub(r"[^A-Za-z0-9_-]+", "_", s)

    def _make_x_axis(self):
        """Linear-in-step x-axis: real GRPO step number for each checkpoint.

        Returns the numpy array of checkpoint step values. If only one
        checkpoint is available (smoke test), the axis still works but
        will of course be visually trivial.
        """
        return np.asarray(self.checkpoint_steps, dtype=float)

    def _caption(self, prompt_lang: str, target_lang: str | None) -> str:
        """Build the caption block placed under each figure axis."""
        n = self.lang_to_n_prompts.get(prompt_lang, 0)
        parts = [
            f"Prompt lang = {self.LANG_DISPLAY[prompt_lang]} (N={n}).",
            f"Top-K = {self.top_k}.",
            "Renormalized over the 7 studied languages.",
            "Averaged over all CoT tokens, then over prompts.",
        ]
        if target_lang is not None:
            parts.insert(1, f"Target lang = {self.LANG_DISPLAY[target_lang]}.")
        return " ".join(parts)

    # ------------------------------------------------------------------ #
    #  Heatmaps (one per (prompt-lang, target-lang) pair)
    # ------------------------------------------------------------------ #

    def plot_heatmaps(self):
        """49 heatmaps: Q[:, :, P, T] over (checkpoint, layer) for each (P, T)."""
        if self.Q is None:
            raise RuntimeError("Call aggregate() before plot_heatmaps().")

        steps = self._make_x_axis()
        layers = np.arange(self.n_layers)

        # Treat the x-axis as a categorical index for imshow; the values
        # are placed at the actual step positions so the visual spacing
        # is linear-in-step. imshow alone would treat columns as equally
        # spaced; we use pcolormesh on (step, layer) edges instead, which
        # is the natural choice for a non-uniformly-sampled x.
        # Build cell edges: midpoints between adjacent steps, with the
        # outer edges extrapolated by half a step.
        if len(steps) == 1:
            half = 0.5
            x_edges = np.array([steps[0] - half, steps[0] + half])
        else:
            mids = (steps[:-1] + steps[1:]) / 2.0
            left = steps[0] - (mids[0] - steps[0])
            right = steps[-1] + (steps[-1] - mids[-1])
            x_edges = np.concatenate([[left], mids, [right]])
        y_edges = np.arange(self.n_layers + 1) - 0.5  # layer i centered at i

        n_pairs = len(self.LANG_LABELS) ** 2
        with tqdm(total=n_pairs, desc="heatmaps", unit="fig") as bar:
            for p_idx, P in enumerate(self.LANG_LABELS):
                if self.lang_to_n_prompts.get(P, 0) == 0:
                    self.logger.warning(
                        f"Skipping heatmaps for prompt lang={P}: no prompts."
                    )
                    bar.update(len(self.LANG_LABELS))
                    continue
                for t_idx, T in enumerate(self.LANG_LABELS):
                    # Q[:, :, P, T] has shape (n_ckpt, L). For pcolormesh
                    # the array is laid out as (rows, cols) = (Y, X), so
                    # we want (L, n_ckpt).
                    grid = self.Q[:, :, p_idx, t_idx].T  # (L, n_ckpt)

                    fig, ax = plt.subplots(
                        figsize=(self.fig_width, self.fig_height),
                        dpi=self.dpi,
                    )
                    mesh = ax.pcolormesh(
                        x_edges,
                        y_edges,
                        grid,
                        cmap=self.cmap,
                        vmin=0.0,
                        vmax=1.0,
                        shading="flat",
                    )
                    ax.set_xlabel("GRPO step")
                    ax.set_ylabel("Layer (0 = first transformer block)")
                    # ax.set_xticks(steps)
                    # ax.set_xticklabels(
                    #     [str(int(s)) for s in steps],
                    #     rotation=0,
                    #     fontsize=8,
                    # )
                    ax.set_ylim(-0.5, self.n_layers - 0.5)
                    # Y ticks every 4 layers for readability with L=28.
                    ax.set_yticks(np.arange(0, self.n_layers, 4))
                    ax.set_title(
                        f"Plot 1: CoT Language Probability\n"
                        f"prompt = {self.LANG_DISPLAY[P]}, "
                        f"target = {self.LANG_DISPLAY[T]}"
                    )
                    cbar = fig.colorbar(mesh, ax=ax)
                    cbar.set_label("Q (renormalized prob.)")
                    # Place the caption under the axes. tight_layout gets
                    # us most of the way; we leave a fixed margin for the
                    # caption text below.
                    fig.subplots_adjust(bottom=0.28)
                    fig.text(
                        0.5,
                        0.02,
                        self._caption(P, T),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        wrap=True,
                    )

                    out = self.plots_dir / (
                        f"plot1_heatmap_prompt-{self._safe_lang(P)}_"
                        f"target-{self._safe_lang(T)}.png"
                    )
                    fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
                    plt.close(fig)
                    bar.update(1)

        self.logger.info(
            f"Wrote heatmaps to {self.plots_dir} (49 files expected)."
        )

    # ------------------------------------------------------------------ #
    #  Line plots (one per prompt-lang, 7 target curves each)
    # ------------------------------------------------------------------ #

    def plot_lineplots(self):
        """7 line plots: layer-averaged Q[j, :, P, T] vs step, one curve per T."""
        if self.Q is None:
            raise RuntimeError("Call aggregate() before plot_lineplots().")

        steps = self._make_x_axis()
        # Layer-average Q[j, :, P, T] over layer axis -> (n_ckpt, 7, 7).
        Q_layer_avg = self.Q.mean(axis=1)  # (n_ckpt, 7, 7)

        # Use a consistent palette: tab10 has 10 distinguishable colours;
        # we take the first 7 for our 7 target languages.
        palette = plt.get_cmap("tab10")
        colors = {lang: palette(i) for i, lang in enumerate(self.LANG_LABELS)}

        for p_idx, P in enumerate(tqdm(
            self.LANG_LABELS, desc="lineplots", unit="fig"
        )):
            if self.lang_to_n_prompts.get(P, 0) == 0:
                self.logger.warning(
                    f"Skipping lineplot for prompt lang={P}: no prompts."
                )
                continue

            fig, ax = plt.subplots(
                figsize=(self.fig_width, self.fig_height),
                dpi=self.dpi,
            )
            for t_idx, T in enumerate(self.LANG_LABELS):
                ax.plot(
                    steps,
                    Q_layer_avg[:, p_idx, t_idx],
                    marker="o",
                    markersize=4,
                    linewidth=1.5,
                    color=colors[T],
                    label=self.LANG_DISPLAY[T],
                )
            ax.set_xlabel("GRPO step")
            ax.set_ylabel(
                "Layer-averaged Q  (mean over layers 0..%d)" % (self.n_layers - 1)
            )
            ax.set_ylim(0.0, 1.0)
            # ax.set_xticks(steps)
            # ax.set_xticklabels([str(int(s)) for s in steps], fontsize=8)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            ax.legend(
                title="Target language",
                fontsize=8,
                title_fontsize=8,
                loc="best",
                framealpha=0.9,
            )
            ax.set_title(
                f"Plot 1: Layer-averaged CoT Language Probability\n"
                f"prompt = {self.LANG_DISPLAY[P]}"
            )
            fig.subplots_adjust(bottom=0.24)
            fig.text(
                0.5,
                0.02,
                self._caption(P, target_lang=None),
                ha="center",
                va="bottom",
                fontsize=8,
                wrap=True,
            )

            out = self.plots_dir / (
                f"plot1_lineplot_prompt-{self._safe_lang(P)}.png"
            )
            fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

        self.logger.info(
            f"Wrote line plots to {self.plots_dir} (7 files expected)."
        )

    # ------------------------------------------------------------------ #
    #  Summary metadata
    # ------------------------------------------------------------------ #

    def write_summary(self):
        """Small JSON beside the figures so they are interpretable later.

        Captures: which checkpoints went in, how many prompts per language
        contributed, the layer count, and the cmap/top-K used.
        """
        if self.Q is None:
            raise RuntimeError("Call aggregate() before write_summary().")
        summary = {
            "checkpoint_steps": self.checkpoint_steps,
            "n_checkpoints": len(self.checkpoint_steps),
            "n_layers": int(self.n_layers),
            "languages": list(self.LANG_LABELS),
            "prompts_per_language": {
                k: int(v) for k, v in self.lang_to_n_prompts.items()
            },
            "top_k": self.top_k,
            "cmap": self.cmap,
            "notes": (
                "Q[j, i, P, T] = mean over prompts of language P, then "
                "over CoT tokens, of the renormalized top-K logit-lens "
                "probability that layer i at checkpoint j predicts a "
                "token of language T. Renormalization is over the 7 "
                "studied languages (punct/special/other dropped)."
            ),
        }
        out = self.plots_dir / "plot1_data_summary.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Wrote summary to {out}")

    # ------------------------------------------------------------------ #
    #  Orchestrator
    # ------------------------------------------------------------------ #

    def run(self):
        self.logger.info("=" * 80)
        self.logger.info(f"Plot 1 starting; reading from {self.acts_dir}")
        self.logger.info("=" * 80)
        self.aggregate()
        self.plot_heatmaps()
        self.plot_lineplots()
        self.write_summary()
        self.logger.info("Plot 1 complete.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main_logger = logging.getLogger("Main")

    # Plotting is CPU-only. We keep the GPU check for standards
    # compliance since the rest of the pipeline assumes a GPU
    # environment and we do not want plot1.py to be silently runnable
    # on a node without GPUs (a sign that something is misconfigured).
    # if not torch.cuda.is_available():
    #     raise RuntimeError("No GPU detected.")
    # main_logger.info(
    #     f"GPUs available: {torch.cuda.device_count()} "
    #     f"({torch.cuda.get_device_name(0)})"
    # )

    config = {
        # ---- Required path roots per coding standards ----
        "log_dir":    Path("./exp2/logs"),
        "output_dir": Path("./exp2/outputs"),
        "data_dir":   Path("./exp2/data"),

        # ---- Figure styling ----
        "dpi": 200,
        "fig_width": 6.0,
        "fig_height": 4.5,
        "cmap": "viridis",

        # ---- Plot 1 quantity knob (must match eval.py's top_k) ----
        "top_k": 50,
    }

    Plot1CoTLanguageProbability(config).run()


if __name__ == "__main__":
    main()