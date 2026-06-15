"""Plot 2 -- Fraction of Layers in each Phase + Per-Layer Detail.

For each (checkpoint j, prompt p, CoT token k, layer i), eval.py stored the
top-1 (greedy) logit-lens prediction's language label as an integer code
0..9 in ``top1_lang_idx`` (indices 0..6 are the seven studied languages,
7=punct_num, 8=special, 9=other).

This script produces, per prompt language P, a single figure with a 3 x 3
grid of subplots:

    Top row    -- one subplot per individual layer in config["first_layers"]
                  (default: the first three layers [0, 1, 2]), with the
                  y-axis "Fraction of CoT tokens".
    Middle row -- one subplot per phase (early / mid / late), with the
                  y-axis "Fraction of layers".
    Bottom row -- one subplot per individual layer in config["last_layers"]
                  (default: the last three layers of the late phase), with
                  the y-axis "Fraction of CoT tokens".

First-row / bottom-row math. For a single layer i and target T:

    g^k_{j, i}(T) = 1[ top1[i, k] == T ]

averaged over CoT tokens k and prompts of language P. For a single
layer this is the fraction of CoT TOKENS at that layer whose top-1 is
in T (the "fraction of layers" denominator collapses to 1 -- it would
be misleading to call this the same thing as the middle row, hence the
distinct y-axis label).

Middle-row math. For phase phi (a range of layers) and target T:

    f^k_{j, phi}(T) = (1 / |phi|) * sum_{i in phi}  1[ top1[i, k] == T ]

averaged over CoT tokens k and prompts of language P. This is the
fraction of LAYERS in the phase whose top-1 is in T (averaged over the
remaining axes). The y-axis ranges over [0, 1].

Punct/special/other top-1 predictions contribute 0 toward every
language's indicator in every row. The 7 per-target values per cell
therefore need not sum to 1; the deficit equals the punct/special/other
share.

Inputs
------
    exp2/outputs/acts/checkpoint_*.npz   (produced by eval.py)

Outputs
-------
    exp2/outputs/plots/plot2/
        plot2_lineplot_prompt-<P>.png    x 7
        plot2_data_summary.json

Behaviour
---------
    - Auto-discovers all checkpoint_*.npz files in acts/ on every run.
    - Phase boundaries, the individual layers shown in the TOP row
      (config["first_layers"]), AND the individual layers shown in the
      BOTTOM row (config["last_layers"]) come from config. Phase
      boundaries are inclusive on both ends.
    - Skips a figure if no prompts of that language exist at any checkpoint.

Standards: see project coding standards. All paths are pathlib.Path. No
os, no argparse, no print. tqdm where appropriate. Logger has FileHandler
+ StreamHandler in mandated format.
"""

import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


class Plot2FractionOfLayersInPhase:
    """Aggregate eval .npz files into Plot 2 figures."""

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

    # Order matters: this is the order of the phase subplots in the middle
    # row of each figure.
    PHASE_NAMES = ("early", "mid", "late")

    def __init__(self, config: dict):
        self.log_dir = Path(config["log_dir"])
        self.output_dir = Path(config["output_dir"])
        self.data_dir = Path(config["data_dir"])

        self.acts_dir = self.output_dir / "acts"
        self.plots_dir = self.output_dir / "plots" / "plot2"

        # Phase boundaries: dict of phase_name -> (start, end_inclusive).
        # Validated against the actual L observed in the first .npz so the
        # script complains loudly if the model architecture changes.
        boundaries = dict(config["phase_boundaries"])
        for name in self.PHASE_NAMES:
            if name not in boundaries:
                raise ValueError(
                    f"phase_boundaries['{name}'] missing in config."
                )
            lo, hi = boundaries[name]
            if not (isinstance(lo, int) and isinstance(hi, int) and lo <= hi):
                raise ValueError(
                    f"phase_boundaries['{name}'] = {boundaries[name]} is "
                    f"not a valid (lo, hi_inclusive) tuple of ints."
                )
        self.phase_boundaries = {
            name: (int(boundaries[name][0]), int(boundaries[name][1]))
            for name in self.PHASE_NAMES
        }

        # Individual layers shown in the TOP row. If absent in config,
        # default to the first three transformer layers [0, 1, 2].
        # Validated against actual L below in _validate_phase_boundaries.
        if "first_layers" in config:
            fl = list(config["first_layers"])
            if not fl:
                raise ValueError("first_layers must be non-empty.")
            self.first_layers = [int(x) for x in fl]
        else:
            self.first_layers = [0, 1, 2]

        # Individual layers shown in the BOTTOM row. If absent in config,
        # default to the last three layers of the late phase (inclusive).
        # Validated against actual L below in _validate_phase_boundaries.
        if "last_layers" in config:
            ll = list(config["last_layers"])
            if not ll:
                raise ValueError("last_layers must be non-empty.")
            self.last_layers = [int(x) for x in ll]
        else:
            late_hi = self.phase_boundaries["late"][1]
            late_lo = self.phase_boundaries["late"][0]
            # Take the last three layers of the late phase, or fewer if
            # the late phase is shorter than 3.
            start = max(late_lo, late_hi - 2)
            self.last_layers = list(range(start, late_hi + 1))

        self.dpi = int(config.get("dpi", 200))
        # Width grows with the 3 subplot columns; height grows with 3 rows.
        self.fig_width = float(config.get("fig_width", 14.0))
        self.fig_height = float(config.get("fig_height", 13.0))

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        # State populated by aggregate().
        self.checkpoint_steps: list[int] = []
        self.n_layers: int | None = None
        self.lang_to_n_prompts: dict[str, int] = {}
        # F[j, P, T, phi]       -- phase fractions, phi in (early, mid, late).
        # G_first[j, P, T, idx] -- per-individual-layer token fractions for
        #                          the TOP row, idx into self.first_layers.
        # G_last[j, P, T, idx]  -- per-individual-layer token fractions for
        #                          the BOTTOM row, idx into self.last_layers.
        self.F: np.ndarray | None = None
        self.G_first: np.ndarray | None = None
        self.G_last: np.ndarray | None = None

    def _setup_logging(self):
        log_file = self.log_dir / "plot2.log"
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
        """Find every checkpoint_<step>.npz under acts/, sorted by step.

        Auto-discovery: re-running this script after eval.py writes new
        checkpoints needs no config change.
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

    def _validate_phase_boundaries(self, n_layers: int):
        """Check phases + first_layers + last_layers fit in [0, n_layers-1]."""
        for name, (lo, hi) in self.phase_boundaries.items():
            if lo < 0 or hi >= n_layers:
                raise ValueError(
                    f"Phase '{name}' boundary ({lo}, {hi}) is outside "
                    f"the valid layer range [0, {n_layers - 1}] for this "
                    f"model. Edit phase_boundaries in the config."
                )
        for tag, layers in (
            ("first_layers", self.first_layers),
            ("last_layers", self.last_layers),
        ):
            for L in layers:
                if L < 0 or L >= n_layers:
                    raise ValueError(
                        f"{tag} value {L} is outside the valid layer "
                        f"range [0, {n_layers - 1}]. Edit {tag} in the "
                        f"config."
                    )
        sizes = {
            name: (hi - lo + 1) for name, (lo, hi) in self.phase_boundaries.items()
        }
        self.logger.info(
            f"Phase boundaries: early={self.phase_boundaries['early']} "
            f"(|early|={sizes['early']}), "
            f"mid={self.phase_boundaries['mid']} "
            f"(|mid|={sizes['mid']}), "
            f"late={self.phase_boundaries['late']} "
            f"(|late|={sizes['late']})"
        )
        self.logger.info(f"first_layers (top row): {self.first_layers}")
        self.logger.info(f"last_layers (bottom row): {self.last_layers}")

    # ------------------------------------------------------------------ #
    #  Aggregation
    # ------------------------------------------------------------------ #

    def aggregate(self):
        """Build F (per-phase) and G_first / G_last (per-layer) tensors.

        Layouts:
            F[j, p_idx, t_idx, phi_idx]      shape (n_ckpt, 7, 7, 3)
                j       index into checkpoint_steps
                p_idx   prompt-language index in LANG_LABELS
                t_idx   target-language index in LANG_LABELS
                phi_idx 0=early, 1=mid, 2=late
                Value: fraction of LAYERS in the phase whose top-1 is in
                       target language T, averaged over CoT tokens and
                       prompts of language P.

            G_first[j, p_idx, t_idx, idx]    shape (n_ckpt, 7, 7, |first_layers|)
                idx     index into self.first_layers
                Value: fraction of CoT TOKENS at that single layer whose
                       top-1 is in target language T, averaged over prompts
                       of language P.

            G_last[j, p_idx, t_idx, idx]     shape (n_ckpt, 7, 7, |last_layers|)
                idx     index into self.last_layers
                Value: as G_first, for the bottom-row layers.
        """
        npz_paths = self._discover_checkpoints()
        n_ckpt = len(npz_paths)
        n_langs = len(self.LANG_LABELS)
        n_phases = len(self.PHASE_NAMES)
        n_first = len(self.first_layers)
        n_last = len(self.last_layers)

        steps: list[int] = []
        F: np.ndarray | None = None
        G_first: np.ndarray | None = None
        G_last: np.ndarray | None = None
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
                top1 = d["top1_lang_idx"]  # object array of (L, C_p) int8

            if list(lang_labels) != list(self.LANG_LABELS):
                raise ValueError(
                    f"{path.name} has lang_labels={lang_labels}, "
                    f"expected {self.LANG_LABELS}"
                )

            if n_layers_seen is None:
                n_layers_seen = n_layers
                self._validate_phase_boundaries(n_layers_seen)
                F = np.zeros(
                    (n_ckpt, n_langs, n_langs, n_phases), dtype=np.float64
                )
                G_first = np.zeros(
                    (n_ckpt, n_langs, n_langs, n_first), dtype=np.float64
                )
                G_last = np.zeros(
                    (n_ckpt, n_langs, n_langs, n_last), dtype=np.float64
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
                # For every prompt of this language, compute the per-phase
                # and per-individual-layer indicator averages, then average
                # across prompts.
                per_prompt_F: list[np.ndarray] = []      # each (7, 3 phases)
                per_prompt_Gf: list[np.ndarray] = []     # each (7, |first|)
                per_prompt_Gl: list[np.ndarray] = []     # each (7, |last|)
                for q_idx in np.where(mask)[0]:
                    arr = top1[q_idx]  # (L, C_p) int8
                    if arr.shape[1] == 0:
                        continue
                    out_F = np.zeros((n_langs, n_phases), dtype=np.float64)
                    out_Gf = np.zeros((n_langs, n_first), dtype=np.float64)
                    out_Gl = np.zeros((n_langs, n_last), dtype=np.float64)
                    for t_idx in range(n_langs):
                        indicator = (arr == t_idx)  # (L, C_p) bool
                        # ---- middle-row: per-phase mean over (layers, tokens)
                        for phi_idx, phi_name in enumerate(self.PHASE_NAMES):
                            lo, hi = self.phase_boundaries[phi_name]
                            phase_slice = indicator[lo:hi + 1, :]
                            out_F[t_idx, phi_idx] = float(phase_slice.mean())
                        # ---- top-row: per-single-layer mean over tokens
                        for idx, layer_i in enumerate(self.first_layers):
                            out_Gf[t_idx, idx] = float(
                                indicator[layer_i, :].mean()
                            )
                        # ---- bottom-row: per-single-layer mean over tokens
                        for idx, layer_i in enumerate(self.last_layers):
                            out_Gl[t_idx, idx] = float(
                                indicator[layer_i, :].mean()
                            )
                    per_prompt_F.append(out_F)
                    per_prompt_Gf.append(out_Gf)
                    per_prompt_Gl.append(out_Gl)

                if not per_prompt_F:
                    continue
                stacked_F = np.stack(per_prompt_F, axis=0)   # (n_kept, 7, 3)
                stacked_Gf = np.stack(per_prompt_Gf, axis=0)  # (n_kept, 7, |first|)
                stacked_Gl = np.stack(per_prompt_Gl, axis=0)  # (n_kept, 7, |last|)
                F[j, p_idx, :, :] = stacked_F.mean(axis=0)
                G_first[j, p_idx, :, :] = stacked_Gf.mean(axis=0)
                G_last[j, p_idx, :, :] = stacked_Gl.mean(axis=0)
                lang_to_n_prompts[P] = max(
                    lang_to_n_prompts[P], int(stacked_F.shape[0])
                )

        self.checkpoint_steps = steps
        self.n_layers = n_layers_seen
        self.F = F
        self.G_first = G_first
        self.G_last = G_last
        self.lang_to_n_prompts = lang_to_n_prompts

        self.logger.info(
            f"Aggregated F shape={F.shape}, G_first shape={G_first.shape}, "
            f"G_last shape={G_last.shape} "
            f"(checkpoints={n_ckpt}, prompt-langs={n_langs}, "
            f"target-langs={n_langs}, phases={n_phases}, "
            f"first-layers={n_first}, last-layers={n_last})"
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
        return re.sub(r"[^A-Za-z0-9_-]+", "_", s)

    def _phase_subtitle(self, phase_name: str) -> str:
        lo, hi = self.phase_boundaries[phase_name]
        pretty = {"early": "Early", "mid": "Mid", "late": "Late"}[phase_name]
        return f"{pretty} phase  (layers {lo}\u2013{hi})"

    def _caption(self, prompt_lang: str) -> str:
        n = self.lang_to_n_prompts.get(prompt_lang, 0)
        return (
            f"Top row: fraction of CoT tokens at the given individual "
            f"layer whose top-1 (greedy) logit-lens prediction belongs to "
            f"each target language. Middle row: fraction of layers in the "
            f"given phase whose top-1 prediction belongs to each target "
            f"language. Bottom row: fraction of CoT tokens at the given "
            f"individual layer (as the top row). All rows are averaged "
            f"over the {n} evaluation prompts in "
            f"{self.LANG_DISPLAY[prompt_lang]} and over the CoT tokens "
            f"generated for each prompt. Tokens whose top-1 prediction is "
            f"punctuation, a special token, or outside the seven studied "
            f"languages contribute zero to every target; the seven curve "
            f"values per checkpoint therefore need not sum to one."
        )

    # ------------------------------------------------------------------ #
    #  Plot: one figure per prompt-lang with a 3 x 3 subplot grid:
    #     row 0 = three layers   (self.first_layers)   over self.G_first
    #     row 1 = three phases   (early / mid / late)  over self.F
    #     row 2 = three layers   (self.last_layers)    over self.G_last
    # ------------------------------------------------------------------ #

    def plot_lineplots(self):
        if self.F is None or self.G_first is None or self.G_last is None:
            raise RuntimeError("Call aggregate() before plot_lineplots().")

        steps = np.asarray(self.checkpoint_steps, dtype=float)
        palette = plt.get_cmap("tab10")
        colors = {lang: palette(i) for i, lang in enumerate(self.LANG_LABELS)}

        n_cols = max(
            len(self.first_layers), len(self.PHASE_NAMES), len(self.last_layers)
        )
        if not (len(self.first_layers) == len(self.PHASE_NAMES)
                == len(self.last_layers)):
            self.logger.warning(
                f"Row lengths differ: first_layers={len(self.first_layers)}, "
                f"phases={len(self.PHASE_NAMES)}, "
                f"last_layers={len(self.last_layers)}. Using n_cols={n_cols}; "
                f"spare cells will be hidden."
            )

        for p_idx, P in enumerate(tqdm(
            self.LANG_LABELS, desc="lineplot figures", unit="fig"
        )):
            if self.lang_to_n_prompts.get(P, 0) == 0:
                self.logger.warning(
                    f"Skipping lineplot for prompt lang={P}: no prompts."
                )
                continue

            # sharey is set per-row: rows 0 and 2 (token fraction) share
            # one scale, row 1 (layer fraction) is its own scale because
            # the y-label semantics differ. We link manually within each
            # row below; the top and bottom token rows are additionally
            # linked to each other for easy comparison.
            fig, axes = plt.subplots(
                3,
                n_cols,
                figsize=(self.fig_width, self.fig_height),
                dpi=self.dpi,
                sharey=False,   # we link manually below
                squeeze=False,
            )
            # Share y within each row by linking the right cols to col 0.
            for row in range(3):
                for c in range(1, n_cols):
                    axes[row, c].sharey(axes[row, 0])
            # Link the two token-fraction rows (0 and 2) to one another so
            # the first-layer and last-layer panels are directly comparable.
            axes[2, 0].sharey(axes[0, 0])

            # ---- Top row: per-individual-layer (first_layers) --------- #
            for idx, layer_i in enumerate(self.first_layers):
                ax = axes[0, idx]
                for t_idx, T in enumerate(self.LANG_LABELS):
                    ax.plot(
                        steps,
                        self.G_first[:, p_idx, t_idx, idx],
                        marker="o",
                        markersize=3.5,
                        linewidth=1.4,
                        color=colors[T],
                        label=(
                            self.LANG_DISPLAY[T] if idx == 0 else None
                        ),
                    )
                ax.set_xlabel("GRPO step")
                ax.set_ylim(0.0, 1.0)
                ax.set_title(f"Layer {layer_i}", fontsize=10)
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
                if idx == 0:
                    ax.set_ylabel("Fraction of CoT tokens")
                if idx > 0:
                    for tl in ax.get_yticklabels():
                        tl.set_visible(False)

            # Hide spare top-row cells if first_layers is shorter.
            for c in range(len(self.first_layers), n_cols):
                axes[0, c].set_visible(False)

            # ---- Middle row: per-phase -------------------------------- #
            for phi_idx, phase_name in enumerate(self.PHASE_NAMES):
                ax = axes[1, phi_idx]
                for t_idx, T in enumerate(self.LANG_LABELS):
                    ax.plot(
                        steps,
                        self.F[:, p_idx, t_idx, phi_idx],
                        marker="o",
                        markersize=3.5,
                        linewidth=1.4,
                        color=colors[T],
                    )
                ax.set_xlabel("GRPO step")
                ax.set_ylim(0.0, 1.0)
                ax.set_title(self._phase_subtitle(phase_name), fontsize=10)
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
                if phi_idx == 0:
                    ax.set_ylabel("Fraction of layers")
                if phi_idx > 0:
                    for tl in ax.get_yticklabels():
                        tl.set_visible(False)

            # Hide spare middle-row cells if PHASE_NAMES is shorter.
            for c in range(len(self.PHASE_NAMES), n_cols):
                axes[1, c].set_visible(False)

            # ---- Bottom row: per-individual-layer (last_layers) ------- #
            for idx, layer_i in enumerate(self.last_layers):
                ax = axes[2, idx]
                for t_idx, T in enumerate(self.LANG_LABELS):
                    ax.plot(
                        steps,
                        self.G_last[:, p_idx, t_idx, idx],
                        marker="o",
                        markersize=3.5,
                        linewidth=1.4,
                        color=colors[T],
                    )
                ax.set_xlabel("GRPO step")
                ax.set_ylim(0.0, 1.0)
                ax.set_title(f"Layer {layer_i}", fontsize=10)
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
                if idx == 0:
                    ax.set_ylabel("Fraction of CoT tokens")
                if idx > 0:
                    for tl in ax.get_yticklabels():
                        tl.set_visible(False)

            # Hide spare bottom-row cells if last_layers is shorter.
            for c in range(len(self.last_layers), n_cols):
                axes[2, c].set_visible(False)

            # Shared legend on the right; placed once, outside the axes.
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                title="Target language",
                fontsize=8,
                title_fontsize=8,
                loc="center right",
                bbox_to_anchor=(0.995, 0.5),
                framealpha=0.9,
            )

            fig.suptitle(
                f"Plot 2: Fraction of Layers Predicting Target Language\n"
                f"prompt = {self.LANG_DISPLAY[P]}",
                fontsize=12,
            )

            # Leave room on the right for the legend and at the bottom for
            # the caption block. With 3 rows we need more vertical space.
            fig.subplots_adjust(
                left=0.06,
                right=0.86,
                bottom=0.12,
                top=0.91,
                wspace=0.12,
                hspace=0.45,
            )
            fig.text(
                0.46,
                0.02,
                self._caption(P),
                ha="center",
                va="bottom",
                fontsize=8,
                wrap=True,
            )

            out = self.plots_dir / (
                f"plot2_lineplot_prompt-{self._safe_lang(P)}.png"
            )
            fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

        self.logger.info(
            f"Wrote lineplots to {self.plots_dir} "
            f"({len(self.LANG_LABELS)} files expected)."
        )

    # ------------------------------------------------------------------ #
    #  Summary metadata
    # ------------------------------------------------------------------ #

    def write_summary(self):
        if self.F is None or self.G_first is None or self.G_last is None:
            raise RuntimeError("Call aggregate() before write_summary().")
        summary = {
            "checkpoint_steps": self.checkpoint_steps,
            "n_checkpoints": len(self.checkpoint_steps),
            "n_layers": int(self.n_layers),
            "languages": list(self.LANG_LABELS),
            "phase_boundaries": {
                name: list(self.phase_boundaries[name])
                for name in self.PHASE_NAMES
            },
            "phase_sizes": {
                name: (self.phase_boundaries[name][1] -
                       self.phase_boundaries[name][0] + 1)
                for name in self.PHASE_NAMES
            },
            "first_layers": list(self.first_layers),
            "last_layers": list(self.last_layers),
            "prompts_per_language": {
                k: int(v) for k, v in self.lang_to_n_prompts.items()
            },
            "notes": (
                "F[j, P, T, phi] = mean over prompts of language P, "
                "then over CoT tokens, then over the layers in phase phi, "
                "of the indicator that the top-1 logit-lens predicted "
                "token belongs to target language T at checkpoint j "
                "(middle row). G_first[j, P, T, idx] = mean over prompts "
                "of language P, then over CoT tokens, of the same "
                "indicator at the single layer first_layers[idx] (top "
                "row). G_last[j, P, T, idx] is the same at the single "
                "layer last_layers[idx] (bottom row). Punctuation, special "
                "tokens, and tokens outside the seven studied languages "
                "contribute zero to every target in F, G_first, and G_last."
            ),
        }
        out = self.plots_dir / "plot2_data_summary.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        self.logger.info(f"Wrote summary to {out}")

    # ------------------------------------------------------------------ #
    #  Orchestrator
    # ------------------------------------------------------------------ #

    def run(self):
        self.logger.info("=" * 80)
        self.logger.info(f"Plot 2 starting; reading from {self.acts_dir}")
        self.logger.info("=" * 80)
        self.aggregate()
        self.plot_lineplots()
        self.write_summary()
        self.logger.info("Plot 2 complete.")


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
        # ---- Required path roots per coding standards ----
        "log_dir":    Path("./exp2/logs"),
        "output_dir": Path("./exp2/outputs"),
        "data_dir":   Path("./exp2/data"),

        # ---- Figure styling (taller because we now have 3 rows) ----
        "dpi": 200,
        "fig_width": 14.0,
        "fig_height": 13.0,

        # ---- Phase boundaries (inclusive on both ends) ----
        # Defaults are the proportional mapping of L=32 ranges onto
        # Qwen2.5-7B's L=28: early ~37%, mid ~37%, late ~25% of layers.
        # Override here if you want different phase splits.
        "phase_boundaries": {
            "early": (0, 10),    # 11 layers
            "mid":   (11, 20),   # 10 layers
            "late":  (21, 27),   # 7 layers
        },

        # ---- Individual layers shown in the TOP row ----
        # Each becomes its own subplot in row 0 of every figure.
        # Default below = the first three transformer layers.
        "first_layers": [0, 1, 2],

        # ---- Individual layers shown in the BOTTOM row ----
        # Each becomes its own subplot in row 2 of every figure.
        # Default below = the last three layers of the late phase.
        "last_layers": [25, 26, 27],
    }

    Plot2FractionOfLayersInPhase(config).run()


if __name__ == "__main__":
    main()