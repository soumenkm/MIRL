"""Plot 3 -- Language Probability Ratio.

For each (checkpoint j, prompt p of language P, CoT token k, layer i),
eval.py stored the full-vocab logit-lens probability mass per language
in ``full_lang_sum`` as a 7-vector (one entry per studied language) of
shape (L, C_p, 7). The 7 values per cell sum to <= 1; the deficit is
the probability mass on punct / special / other tokens (which this plot
ignores).

The ratio of interest, per (layer i, CoT token k):

    rho_en (i, k)  = full_lang_sum[i, k, idx_en]
    rho_tgt(i, k)  = full_lang_sum[i, k, idx_tgt]
    r^k_{j, i}(tgt) = rho_tgt / (rho_en + rho_tgt)

If rho_en + rho_tgt == 0 (the model committed no mass to either language
at that cell, e.g. all mass on punctuation) the ratio is NaN and is
excluded from the downstream means. Per-(layer, checkpoint, target)
averages use nanmean.

We aggregate by averaging r over the CoT axis k and over the prompts of
language P:

    R_{j, i}(P, tgt) = mean_{p in lang P} ( nanmean_k  r^k_{p, j, i}(tgt) )

English is excluded from the target list because R(en) = 0.5 by
construction, which carries no signal. The 6 non-English studied
languages are the targets: {bn, te, th, ru, ja, zh}.

Outputs (auto-discovered checkpoint set in acts/):

    exp2/outputs/plots/plot3/
        plot3_heatmap_prompt-<P>_target-<T>.png    x 42  (7 prompts x 6 targets)
        plot3_lineplot_prompt-<P>.png              x 7   (6 curves each,
                                                          layer-averaged)
        plot3_data_summary.json

Standards: see project coding standards. All paths are pathlib.Path. No
os, no argparse, no print. tqdm where appropriate. Logger has FileHandler
+ StreamHandler in mandated format.
"""

import json
import logging
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


class Plot3LanguageProbabilityRatio:
    """Aggregate eval .npz files into Plot 3 figures."""

    # The seven studied languages in the order used by full_lang_sum's
    # last axis (must match LANG_LABELS in eval.py).
    LANG_LABELS = ("en", "bn", "te", "th", "ru", "ja", "zh")

    # English is the denominator's reference: R(en) = 0.5 by construction,
    # so we exclude it from the target list. These six are what we plot.
    TARGET_LABELS = ("bn", "te", "th", "ru", "ja", "zh")

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
        self.plots_dir = self.output_dir / "plots" / "plot3"

        self.dpi = int(config.get("dpi", 200))
        self.fig_width = float(config.get("fig_width", 6.0))
        self.fig_height = float(config.get("fig_height", 4.5))
        self.cmap = str(config.get("cmap", "viridis"))

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        # Pre-compute the integer indices for English and the six targets
        # into full_lang_sum's last axis. Faster than looking up by name
        # in the inner loop.
        self.en_idx = self.LANG_LABELS.index("en")
        self.target_indices = [
            self.LANG_LABELS.index(t) for t in self.TARGET_LABELS
        ]

        # State populated by aggregate().
        self.checkpoint_steps: list[int] = []
        self.n_layers: int | None = None
        self.lang_to_n_prompts: dict[str, int] = {}
        # R has shape (n_ckpt, L, 7 prompt-langs, 6 target-langs).
        self.R: np.ndarray | None = None

    def _setup_logging(self):
        log_file = self.log_dir / "plot3.log"
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
        """Build R[j, i, P_idx, T_idx] of shape (n_ckpt, L, 7, 6).

        For each prompt of language P, we read its (L, C_p, 7) slice of
        full_lang_sum, isolate rho_en and rho_tgt as (L, C_p) arrays, form
        the ratio with NaN at 0/0 cells, then nanmean over the CoT axis k
        to get a (L,) per-prompt-per-target vector. We average those vectors
        across prompts of language P to get R[j, :, P, T].
        """
        npz_paths = self._discover_checkpoints()
        n_ckpt = len(npz_paths)
        n_p = len(self.LANG_LABELS)
        n_t = len(self.TARGET_LABELS)

        steps: list[int] = []
        R: np.ndarray | None = None
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
                full_lang_sum = d["full_lang_sum"]  # object array of (L, C_p, 7) fp16

            if list(lang_labels) != list(self.LANG_LABELS):
                raise ValueError(
                    f"{path.name} has lang_labels={lang_labels}, "
                    f"expected {self.LANG_LABELS}"
                )

            if n_layers_seen is None:
                n_layers_seen = n_layers
                R = np.zeros(
                    (n_ckpt, n_layers, n_p, n_t), dtype=np.float64
                )
            elif n_layers != n_layers_seen:
                raise ValueError(
                    f"{path.name} has n_layers={n_layers}, expected "
                    f"{n_layers_seen} from an earlier file."
                )

            steps.append(step)

            for p_idx, P in enumerate(self.LANG_LABELS):
                mask = (prompt_langs == P) & (cot_lengths > 0)
                if not mask.any():
                    continue
                per_prompt: list[np.ndarray] = []  # each (L, 6 targets)
                for q_idx in np.where(mask)[0]:
                    arr = full_lang_sum[q_idx]  # (L, C_p, 7) fp16
                    if arr.shape[1] == 0:
                        continue
                    arr_f = arr.astype(np.float32, copy=False)
                    rho_en = arr_f[..., self.en_idx]   # (L, C_p)

                    # Compute the ratio for each target in one batched op:
                    # rho_tgt has shape (L, C_p, 6).
                    rho_tgt_all = arr_f[..., self.target_indices]
                    # broadcast rho_en to (L, C_p, 1)
                    denom = rho_en[..., None] + rho_tgt_all
                    # 0/0 -> nan; handled by np.errstate / nanmean below.
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = np.where(
                            denom > 0,
                            rho_tgt_all / denom,
                            np.nan,
                        )  # (L, C_p, 6)
                    # Average over CoT axis k, ignoring NaNs. Suppress the
                    # "Mean of empty slice" warning that nanmean emits when
                    # an entire (layer, target) column is NaN -- we want
                    # that to surface as NaN in R.
                    with warnings.catch_warnings():
                        warnings.simplefilter(
                            "ignore", category=RuntimeWarning
                        )
                        per_prompt_LT = np.nanmean(ratio, axis=1)  # (L, 6)
                    per_prompt.append(per_prompt_LT)

                if not per_prompt:
                    continue
                stacked = np.stack(per_prompt, axis=0)  # (n_kept, L, 6)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    R[j, :, p_idx, :] = np.nanmean(stacked, axis=0)  # (L, 6)
                lang_to_n_prompts[P] = max(
                    lang_to_n_prompts[P], int(stacked.shape[0])
                )

        self.checkpoint_steps = steps
        self.n_layers = n_layers_seen
        self.R = R
        self.lang_to_n_prompts = lang_to_n_prompts

        self.logger.info(
            f"Aggregated R shape={R.shape} "
            f"(checkpoints={n_ckpt}, layers={n_layers_seen}, "
            f"prompt-langs={n_p}, target-langs={n_t})"
        )
        for P in self.LANG_LABELS:
            self.logger.info(
                f"  prompt lang {P}: {lang_to_n_prompts[P]} prompts averaged"
            )

        # Sanity-check the NaN fraction so the user knows if a lot of cells
        # were degenerate.
        nan_frac = float(np.isnan(R).mean())
        if nan_frac > 0:
            self.logger.info(
                f"R contains {nan_frac:.2%} NaN entries "
                f"(target+en mass was zero at those cells, excluded from mean)."
            )

    # ------------------------------------------------------------------ #
    #  Plotting helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_lang(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", s)

    def _caption(self, prompt_lang: str, target_lang: str | None) -> str:
        n = self.lang_to_n_prompts.get(prompt_lang, 0)
        parts = [
            f"Prompt lang = {self.LANG_DISPLAY[prompt_lang]} (N={n}).",
            "R = p_target / (p_en + p_target) from the full-vocab "
            "logit-lens distribution; punctuation, special, and other "
            "tokens are not included in either sum.",
            "Cells where p_en + p_target = 0 are treated as NaN and "
            "excluded from the means.",
            "Averaged over all CoT tokens, then over prompts.",
        ]
        if target_lang is not None:
            parts.insert(
                1, f"Target lang = {self.LANG_DISPLAY[target_lang]}."
            )
        return " ".join(parts)

    def _make_x_edges(self, steps: np.ndarray) -> np.ndarray:
        """Cell edges for pcolormesh so columns are linear in step.

        With non-uniform checkpoint spacing, pcolormesh needs explicit edges
        rather than letting imshow pretend the columns are equally spaced.
        """
        if len(steps) == 1:
            half = 0.5
            return np.array([steps[0] - half, steps[0] + half])
        mids = (steps[:-1] + steps[1:]) / 2.0
        left = steps[0] - (mids[0] - steps[0])
        right = steps[-1] + (steps[-1] - mids[-1])
        return np.concatenate([[left], mids, [right]])

    # ------------------------------------------------------------------ #
    #  Heatmaps (42 figures: 7 prompt-langs x 6 non-English targets)
    # ------------------------------------------------------------------ #

    def plot_heatmaps(self):
        if self.R is None:
            raise RuntimeError("Call aggregate() before plot_heatmaps().")

        steps = np.asarray(self.checkpoint_steps, dtype=float)
        x_edges = self._make_x_edges(steps)
        y_edges = np.arange(self.n_layers + 1) - 0.5

        n_pairs = len(self.LANG_LABELS) * len(self.TARGET_LABELS)
        with tqdm(total=n_pairs, desc="heatmaps", unit="fig") as bar:
            for p_idx, P in enumerate(self.LANG_LABELS):
                if self.lang_to_n_prompts.get(P, 0) == 0:
                    self.logger.warning(
                        f"Skipping heatmaps for prompt lang={P}: no prompts."
                    )
                    bar.update(len(self.TARGET_LABELS))
                    continue
                for t_idx, T in enumerate(self.TARGET_LABELS):
                    # R[:, :, p_idx, t_idx] has shape (n_ckpt, L). For
                    # pcolormesh layout (rows, cols) = (Y, X), we want
                    # (L, n_ckpt).
                    grid = self.R[:, :, p_idx, t_idx].T  # (L, n_ckpt)

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
                    ax.set_ylim(-0.5, self.n_layers - 0.5)
                    ax.set_yticks(np.arange(0, self.n_layers, 4))
                    # Let matplotlib pick clean x-ticks automatically.
                    ax.set_title(
                        f"Plot 3: Language Probability Ratio\n"
                        f"prompt = {self.LANG_DISPLAY[P]}, "
                        f"target = {self.LANG_DISPLAY[T]}"
                    )
                    cbar = fig.colorbar(mesh, ax=ax)
                    cbar.set_label(
                        "R = p_tgt / (p_en + p_tgt)"
                    )
                    fig.subplots_adjust(bottom=0.30)
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
                        f"plot3_heatmap_prompt-{self._safe_lang(P)}_"
                        f"target-{self._safe_lang(T)}.png"
                    )
                    fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
                    plt.close(fig)
                    bar.update(1)

        self.logger.info(
            f"Wrote heatmaps to {self.plots_dir} (42 files expected)."
        )

    # ------------------------------------------------------------------ #
    #  Line plots (7 figures: layer-averaged R, 6 curves each)
    # ------------------------------------------------------------------ #

    def plot_lineplots(self):
        if self.R is None:
            raise RuntimeError("Call aggregate() before plot_lineplots().")

        steps = np.asarray(self.checkpoint_steps, dtype=float)
        # Layer-average using nanmean so NaN cells don't drag the curves.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            R_layer_avg = np.nanmean(self.R, axis=1)  # (n_ckpt, 7, 6)

        # Use the same palette as plots 1 / 2 so target colours are
        # consistent across the figure set. We index by the target's
        # position in LANG_LABELS so e.g. Bengali stays the same colour
        # in all four plots.
        palette = plt.get_cmap("tab10")
        colors = {
            T: palette(self.LANG_LABELS.index(T)) for T in self.TARGET_LABELS
        }

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
            for t_idx, T in enumerate(self.TARGET_LABELS):
                ax.plot(
                    steps,
                    R_layer_avg[:, p_idx, t_idx],
                    marker="o",
                    markersize=4,
                    linewidth=1.5,
                    color=colors[T],
                    label=self.LANG_DISPLAY[T],
                )
            ax.set_xlabel("GRPO step")
            ax.set_ylabel(
                "Layer-averaged R  (mean over layers 0..%d)" % (self.n_layers - 1)
            )
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            ax.legend(
                title="Target language",
                fontsize=8,
                title_fontsize=8,
                loc="best",
                framealpha=0.9,
            )
            ax.set_title(
                f"Plot 3: Layer-averaged Language Probability Ratio\n"
                f"prompt = {self.LANG_DISPLAY[P]}"
            )
            fig.subplots_adjust(bottom=0.26)
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
                f"plot3_lineplot_prompt-{self._safe_lang(P)}.png"
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
        if self.R is None:
            raise RuntimeError("Call aggregate() before write_summary().")
        nan_frac = float(np.isnan(self.R).mean())
        summary = {
            "checkpoint_steps": self.checkpoint_steps,
            "n_checkpoints": len(self.checkpoint_steps),
            "n_layers": int(self.n_layers),
            "prompt_languages": list(self.LANG_LABELS),
            "target_languages": list(self.TARGET_LABELS),
            "english_index": int(self.en_idx),
            "prompts_per_language": {
                k: int(v) for k, v in self.lang_to_n_prompts.items()
            },
            "cmap": self.cmap,
            "nan_fraction_in_R": nan_frac,
            "notes": (
                "R[j, i, P_idx, T_idx] = mean over prompts of language P, "
                "then over CoT tokens, of p_target / (p_en + p_target). "
                "Probabilities come from full-vocab logit-lens summed by "
                "language (punct / special / other dropped). Cells where "
                "p_en + p_target = 0 are NaN and excluded from the means. "
                "English is excluded from the target list because "
                "R(en) = 0.5 by construction."
            ),
        }
        out = self.plots_dir / "plot3_data_summary.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        self.logger.info(f"Wrote summary to {out}")

    # ------------------------------------------------------------------ #
    #  Orchestrator
    # ------------------------------------------------------------------ #

    def run(self):
        self.logger.info("=" * 80)
        self.logger.info(f"Plot 3 starting; reading from {self.acts_dir}")
        self.logger.info("=" * 80)
        self.aggregate()
        self.plot_heatmaps()
        self.plot_lineplots()
        self.write_summary()
        self.logger.info("Plot 3 complete.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main_logger = logging.getLogger("Main")

    # Plotting is CPU-only. The GPU check matches the rest of the pipeline.
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

        # ---- Figure styling ----
        "dpi": 200,
        "fig_width": 6.0,
        "fig_height": 4.5,
        "cmap": "viridis",
    }

    Plot3LanguageProbabilityRatio(config).run()


if __name__ == "__main__":
    main()