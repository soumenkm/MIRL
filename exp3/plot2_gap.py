"""Plot 2 (gap version) -- English-minus-Target margin across the layer stack.

For each (checkpoint j, prompt p, CoT token k, layer i), eval.py stored the
top-1 (greedy) logit-lens prediction's language label as an integer code
0..9 in ``top1_lang_idx`` (indices 0..6 are the seven studied languages,
7=punct_num, 8=special, 9=other).

This script is a re-cut of the original Plot 2. Instead of plotting the raw
per-target fraction, it plots the GAP (margin) between English and each
target language:

    gap = (English fraction)  -  (target fraction)

at the same aggregation granularity as before. By construction the English
curve is identically zero (English minus itself). A positive gap means
English wins more top-1 cells than the target language at that granularity;
a negative gap means the target language wins more than English (this is
allowed and is the interesting regime for a strong target language).

Per prompt language P, the figure is a 3 x 3 grid of subplots, grouped by
row:

    Row 1  -- individual layers 0, 1, 2             (token-level gap)
    Row 2  -- phases early / mid / late             (layer-level gap)
    Row 3  -- the three layers in last_layers       (token-level gap)

Each row has three columns; with the defaults that is three layers, three
phases, and three layers respectively.

Underlying quantities (before taking the gap), identical to the original
Plot 2:

Layer/phase math. For phase phi (a range of layers) and target T:

    f^k_{j, phi}(T) = (1 / |phi|) * sum_{i in phi}  1[ top1[i, k] == T ]

For a single layer i and target T:

    g^k_{j, i}(T) = 1[ top1[i, k] == T ]

both averaged over CoT tokens k and prompts of language P. The gap shown
in row r for target T at checkpoint j is then  value(en) - value(T)  for
the appropriate phase- or layer-level value.

Punct/special/other top-1 predictions contribute 0 toward every language's
indicator, so an individual gap is just (en share) - (T share) at that
granularity and lies in [-1, 1].

Inputs
------
    exp2/outputs/acts/checkpoint_*.npz   (produced by eval.py)

Outputs
-------
    exp2/outputs/plots/plot2_gap/
        plot2_gap_lineplot_prompt-<P>.png    x 7
        plot2_gap_data_summary.json

Behaviour
---------
    - Auto-discovers all checkpoint_*.npz files in acts/ on every run.
    - Phase boundaries, the individual layers shown in the first three
      rows (``first_layers``), and those in the last three rows
      (``last_layers``) all come from config. Phase boundaries are
      inclusive on both ends.
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


class Plot2GapAcrossStack:
    """Aggregate eval .npz files into 9-row English-minus-target gap figures."""

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

    # Order matters: this is the order of the phase rows in each figure.
    PHASE_NAMES = ("early", "mid", "late")

    # English is the reference; its index in LANG_LABELS.
    EN_IDX = 0

    def __init__(self, config: dict):
        self.log_dir = Path(config["log_dir"])
        self.output_dir = Path(config["output_dir"])
        self.data_dir = Path(config["data_dir"])

        self.acts_dir = self.output_dir / "acts"
        self.plots_dir = self.output_dir / "plots" / "plot2_gap"

        # Phase boundaries: dict of phase_name -> (start, end_inclusive).
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

        # First three rows: individual layers near the input. Defaults to
        # the first three transformer layers [0, 1, 2].
        if "first_layers" in config:
            fl = list(config["first_layers"])
            if not fl:
                raise ValueError("first_layers must be non-empty.")
            self.first_layers = [int(x) for x in fl]
        else:
            self.first_layers = [0, 1, 2]

        # Last three rows: individual layers near the output. Defaults to
        # the last three layers of the late phase (inclusive).
        if "last_layers" in config:
            ll = list(config["last_layers"])
            if not ll:
                raise ValueError("last_layers must be non-empty.")
            self.last_layers = [int(x) for x in ll]
        else:
            late_hi = self.phase_boundaries["late"][1]
            late_lo = self.phase_boundaries["late"][0]
            start = max(late_lo, late_hi - 2)
            self.last_layers = list(range(start, late_hi + 1))

        self.dpi = int(config.get("dpi", 200))
        # 3 columns x 3 rows -> wide-ish landscape figure.
        self.fig_width = float(config.get("fig_width", 14.0))
        self.fig_height = float(config.get("fig_height", 11.0))

        # If True, y-limits are a symmetric [-m, m] computed from the data
        # per figure; if a float is given, that fixed symmetric limit is
        # used instead. Default: symmetric auto.
        self.ylim_mode = config.get("ylim", "symmetric_auto")

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        # State populated by aggregate().
        self.checkpoint_steps: list[int] = []
        self.n_layers: int | None = None
        self.lang_to_n_prompts: dict[str, int] = {}
        # Raw (pre-gap) fractions:
        #   F[j, P, T, phi]       phase fractions (early, mid, late)
        #   G_first[j, P, T, idx] per-layer token fractions for first_layers
        #   G_last[j, P, T, idx]  per-layer token fractions for last_layers
        self.F: np.ndarray | None = None
        self.G_first: np.ndarray | None = None
        self.G_last: np.ndarray | None = None

    def _setup_logging(self):
        log_file = self.log_dir / "plot2_gap.log"
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

    def _validate_layers(self, n_layers: int):
        """Check phases, first_layers, and last_layers fit in [0, n_layers-1]."""
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
                        f"{tag} value {L} is outside the valid layer range "
                        f"[0, {n_layers - 1}]. Edit {tag} in the config."
                    )
        sizes = {
            name: (hi - lo + 1)
            for name, (lo, hi) in self.phase_boundaries.items()
        }
        self.logger.info(
            f"Phase boundaries: early={self.phase_boundaries['early']} "
            f"(|early|={sizes['early']}), "
            f"mid={self.phase_boundaries['mid']} (|mid|={sizes['mid']}), "
            f"late={self.phase_boundaries['late']} (|late|={sizes['late']})"
        )
        self.logger.info(f"first_layers (rows 1-3): {self.first_layers}")
        self.logger.info(f"last_layers (rows 7-9): {self.last_layers}")

    # ------------------------------------------------------------------ #
    #  Aggregation
    # ------------------------------------------------------------------ #

    def aggregate(self):
        """Build F (per-phase) and G_first / G_last (per-layer) fraction tensors.

        These hold the RAW per-target fractions (not yet gapped). The gap
        is taken at plot time as F[..., en, ...] - F[..., T, ...] so the
        summary JSON can also store the raw fractions if desired.
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
                self._validate_layers(n_layers_seen)
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
                per_prompt_F: list[np.ndarray] = []      # (7 targets, 3 phases)
                per_prompt_Gf: list[np.ndarray] = []     # (7 targets, n_first)
                per_prompt_Gl: list[np.ndarray] = []     # (7 targets, n_last)
                for q_idx in np.where(mask)[0]:
                    arr = top1[q_idx]  # (L, C_p) int8
                    if arr.shape[1] == 0:
                        continue
                    out_F = np.zeros((n_langs, n_phases), dtype=np.float64)
                    out_Gf = np.zeros((n_langs, n_first), dtype=np.float64)
                    out_Gl = np.zeros((n_langs, n_last), dtype=np.float64)
                    for t_idx in range(n_langs):
                        indicator = (arr == t_idx)  # (L, C_p) bool
                        # phase means over (layers in phase, tokens)
                        for phi_idx, phi_name in enumerate(self.PHASE_NAMES):
                            lo, hi = self.phase_boundaries[phi_name]
                            out_F[t_idx, phi_idx] = float(
                                indicator[lo:hi + 1, :].mean()
                            )
                        # first-layer means over tokens
                        for idx, layer_i in enumerate(self.first_layers):
                            out_Gf[t_idx, idx] = float(
                                indicator[layer_i, :].mean()
                            )
                        # last-layer means over tokens
                        for idx, layer_i in enumerate(self.last_layers):
                            out_Gl[t_idx, idx] = float(
                                indicator[layer_i, :].mean()
                            )
                    per_prompt_F.append(out_F)
                    per_prompt_Gf.append(out_Gf)
                    per_prompt_Gl.append(out_Gl)

                if not per_prompt_F:
                    continue
                F[j, p_idx, :, :] = np.stack(per_prompt_F, 0).mean(axis=0)
                G_first[j, p_idx, :, :] = np.stack(per_prompt_Gf, 0).mean(axis=0)
                G_last[j, p_idx, :, :] = np.stack(per_prompt_Gl, 0).mean(axis=0)
                lang_to_n_prompts[P] = max(
                    lang_to_n_prompts[P], int(len(per_prompt_F))
                )

        self.checkpoint_steps = steps
        self.n_layers = n_layers_seen
        self.F = F
        self.G_first = G_first
        self.G_last = G_last
        self.lang_to_n_prompts = lang_to_n_prompts

        self.logger.info(
            f"Aggregated F shape={F.shape}, G_first shape={G_first.shape}, "
            f"G_last shape={G_last.shape} (checkpoints={n_ckpt}, "
            f"prompt-langs={n_langs}, target-langs={n_langs}, "
            f"phases={n_phases}, first_layers={n_first}, "
            f"last_layers={n_last})"
        )
        for P in self.LANG_LABELS:
            self.logger.info(
                f"  prompt lang {P}: {lang_to_n_prompts[P]} prompts averaged"
            )

    # ------------------------------------------------------------------ #
    #  Gap helpers
    # ------------------------------------------------------------------ #

    def _gap(self, tensor: np.ndarray, p_idx: int, col_idx: int) -> np.ndarray:
        """Return (n_ckpt, 7) gap = en_fraction - target_fraction.

        tensor has shape (n_ckpt, P, T, COL). For the given prompt index
        p_idx and column col_idx, broadcast-subtract every target from the
        English reference. English's own column comes out exactly zero.
        """
        en = tensor[:, p_idx, self.EN_IDX, col_idx]          # (n_ckpt,)
        all_t = tensor[:, p_idx, :, col_idx]                 # (n_ckpt, 7)
        return en[:, None] - all_t                           # (n_ckpt, 7)

    def _symmetric_limit(self, gaps: list[np.ndarray]) -> float:
        """Pick a symmetric y-limit covering all gap data with a margin."""
        m = 0.0
        for g in gaps:
            if g.size:
                m = max(m, float(np.nanmax(np.abs(g))))
        if m <= 0.0:
            return 0.05
        # round up to a tidy value with ~12% headroom
        return float(np.ceil((m * 1.12) * 20.0) / 20.0)

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
            f"Each curve is the gap  (English fraction) \u2212 (target "
            f"fraction)  in the top-1 (greedy) logit-lens prediction, at "
            f"the granularity named in each panel's title. Rows 1 and 3 "
            f"use single layers (token-level fraction); row 2 uses phases "
            f"(layer-level fraction). A positive value means English wins "
            f"more top-1 cells than the target; a negative value means the "
            f"target wins more than English. The English curve is "
            f"identically zero by construction. Averaged over the {n} "
            f"evaluation prompts in {self.LANG_DISPLAY[prompt_lang]} and "
            f"over their CoT tokens. Punctuation / special / out-of-class "
            f"top-1 predictions contribute zero to every language, so each "
            f"gap lies in [\u22121, 1]."
        )

    # ------------------------------------------------------------------ #
    #  Plot: one figure per prompt-lang with a 3 x 3 grid of subplots.
    #     row 0 : individual layers first_layers       (token-level gap)
    #     row 1 : phases early / mid / late            (layer-level gap)
    #     row 2 : individual layers last_layers         (token-level gap)
    #  Each row has three columns; with the defaults that is 3 layers,
    #  3 phases, 3 layers respectively.
    # ------------------------------------------------------------------ #

    def plot_lineplots(self):
        if self.F is None or self.G_first is None or self.G_last is None:
            raise RuntimeError("Call aggregate() before plot_lineplots().")

        steps = np.asarray(self.checkpoint_steps, dtype=float)
        palette = plt.get_cmap("tab10")
        colors = {lang: palette(i) for i, lang in enumerate(self.LANG_LABELS)}

        # Build a 3-row plan. Each row is (kind, list_of_(col_idx, title)).
        # The three rows need not have equal column counts; n_cols is the
        # max and any spare cells are hidden.
        row0 = [
            ("first", idx, f"Layer {layer_i}")
            for idx, layer_i in enumerate(self.first_layers)
        ]
        row1 = [
            ("phase", phi_idx, self._phase_subtitle(phase_name))
            for phi_idx, phase_name in enumerate(self.PHASE_NAMES)
        ]
        row2 = [
            ("last", idx, f"Layer {layer_i}")
            for idx, layer_i in enumerate(self.last_layers)
        ]
        rows = [row0, row1, row2]
        row_ylabels = [
            "en \u2212 target  (token-level)",
            "en \u2212 target  (layer-level)",
            "en \u2212 target  (token-level)",
        ]
        n_rows = 3
        n_cols = max(len(r) for r in rows)

        def _gap_for(kind: str, p_idx: int, col_idx: int) -> np.ndarray:
            if kind == "first":
                return self._gap(self.G_first, p_idx, col_idx)
            if kind == "phase":
                return self._gap(self.F, p_idx, col_idx)
            return self._gap(self.G_last, p_idx, col_idx)  # "last"

        for p_idx, P in enumerate(tqdm(
            self.LANG_LABELS, desc="gap lineplot figures", unit="fig"
        )):
            if self.lang_to_n_prompts.get(P, 0) == 0:
                self.logger.warning(
                    f"Skipping gap lineplot for prompt lang={P}: no prompts."
                )
                continue

            # Pre-compute every cell's gap matrix (n_ckpt, 7) so we can pick
            # a shared symmetric y-limit across the whole figure.
            cell_gaps: dict[tuple[int, int], np.ndarray] = {}
            for r, row in enumerate(rows):
                for c, (kind, col_idx, _title) in enumerate(row):
                    cell_gaps[(r, c)] = _gap_for(kind, p_idx, col_idx)

            if self.ylim_mode == "symmetric_auto":
                ymax = self._symmetric_limit(list(cell_gaps.values()))
            else:
                ymax = float(self.ylim_mode)
            ylo, yhi = -ymax, ymax

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(self.fig_width, self.fig_height),
                dpi=self.dpi,
                sharex=True,
                sharey=True,
                squeeze=False,
            )

            first_legend_cell = True
            for r, row in enumerate(rows):
                for c in range(n_cols):
                    ax = axes[r, c]
                    if c >= len(row):
                        ax.set_visible(False)
                        continue
                    _kind, _col_idx, title = row[c]
                    g = cell_gaps[(r, c)]  # (n_ckpt, 7)
                    for t_idx, T in enumerate(self.LANG_LABELS):
                        ax.plot(
                            steps,
                            g[:, t_idx],
                            marker="o",
                            markersize=3.0,
                            linewidth=1.3,
                            color=colors[T],
                            label=(
                                self.LANG_DISPLAY[T]
                                if first_legend_cell else None
                            ),
                            zorder=2 if T != "en" else 1,
                        )
                    # Zero reference: where English sits and the sign flips.
                    ax.axhline(
                        0.0, color="black", linewidth=0.8, alpha=0.45,
                        zorder=0,
                    )
                    ax.set_ylim(ylo, yhi)
                    ax.set_title(title, fontsize=10)
                    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
                    if r == n_rows - 1:
                        ax.set_xlabel("GRPO step")
                    if c == 0:
                        ax.set_ylabel(row_ylabels[r], fontsize=9)
                    first_legend_cell = False

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
                f"Plot 2 (gap): English \u2212 Target top-1 margin across "
                f"the stack\nprompt = {self.LANG_DISPLAY[P]}",
                fontsize=12,
            )

            fig.subplots_adjust(
                left=0.07,
                right=0.86,
                bottom=0.14,
                top=0.90,
                wspace=0.12,
                hspace=0.30,
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
                f"plot2_gap_lineplot_prompt-{self._safe_lang(P)}.png"
            )
            fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

        self.logger.info(
            f"Wrote gap lineplots to {self.plots_dir} "
            f"({len(self.LANG_LABELS)} files expected)."
        )

    # ------------------------------------------------------------------ #
    #  Summary metadata
    # ------------------------------------------------------------------ #

    def write_summary(self):
        if self.F is None or self.G_first is None or self.G_last is None:
            raise RuntimeError("Call aggregate() before write_summary().")

        # Store the gap tensors (en - target) for every prompt language so
        # the figures are reproducible from the JSON without the .npz.
        gap_F = (
            self.F[:, :, self.EN_IDX:self.EN_IDX + 1, :] - self.F
        )  # (n_ckpt, P, T, phases)
        gap_Gf = (
            self.G_first[:, :, self.EN_IDX:self.EN_IDX + 1, :] - self.G_first
        )
        gap_Gl = (
            self.G_last[:, :, self.EN_IDX:self.EN_IDX + 1, :] - self.G_last
        )

        def _per_lang(gap: np.ndarray, cols: list) -> dict:
            # gap shape (n_ckpt, P, T, COL) -> nested dict keyed by prompt,
            # then target, then column label -> list over checkpoints.
            out = {}
            for p_idx, P in enumerate(self.LANG_LABELS):
                out[P] = {}
                for t_idx, T in enumerate(self.LANG_LABELS):
                    out[P][T] = {
                        str(cols[c]): [
                            float(gap[j, p_idx, t_idx, c])
                            for j in range(gap.shape[0])
                        ]
                        for c in range(gap.shape[3])
                    }
            return out

        summary = {
            "checkpoint_steps": self.checkpoint_steps,
            "n_checkpoints": len(self.checkpoint_steps),
            "n_layers": int(self.n_layers),
            "languages": list(self.LANG_LABELS),
            "reference_language": "en",
            "quantity": "gap = English_fraction - target_fraction",
            "phase_boundaries": {
                name: list(self.phase_boundaries[name])
                for name in self.PHASE_NAMES
            },
            "first_layers": list(self.first_layers),
            "last_layers": list(self.last_layers),
            "row_order": (
                [f"layer_{L}" for L in self.first_layers]
                + list(self.PHASE_NAMES)
                + [f"layer_{L}" for L in self.last_layers]
            ),
            "prompts_per_language": {
                k: int(v) for k, v in self.lang_to_n_prompts.items()
            },
            "gap_phase": _per_lang(gap_F, list(self.PHASE_NAMES)),
            "gap_first_layers": _per_lang(gap_Gf, list(self.first_layers)),
            "gap_last_layers": _per_lang(gap_Gl, list(self.last_layers)),
            "notes": (
                "Each value is (English fraction) - (target fraction) of "
                "top-1 logit-lens predictions, averaged over prompts of the "
                "given prompt language and over CoT tokens. For phases the "
                "fraction is also averaged over the layers in the phase "
                "(layer-level gap); for individual layers it is a single "
                "layer (token-level gap). The English-vs-English gap is "
                "identically zero. Punctuation / special / out-of-class "
                "top-1 predictions contribute zero to every language, so "
                "each gap lies in [-1, 1]."
            ),
        }
        out = self.plots_dir / "plot2_gap_data_summary.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        self.logger.info(f"Wrote summary to {out}")

    # ------------------------------------------------------------------ #
    #  Orchestrator
    # ------------------------------------------------------------------ #

    def run(self):
        self.logger.info("=" * 80)
        self.logger.info(f"Plot 2 (gap) starting; reading from {self.acts_dir}")
        self.logger.info("=" * 80)
        self.aggregate()
        self.plot_lineplots()
        self.write_summary()
        self.logger.info("Plot 2 (gap) complete.")


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

        # ---- Figure styling (3 x 3 grid -> wide landscape) ----
        "dpi": 200,
        "fig_width": 14.0,
        "fig_height": 11.0,

        # Symmetric y-limit: "symmetric_auto" picks [-m, m] from the data;
        # or set a float (e.g. 0.6) for a fixed symmetric limit.
        "ylim": "symmetric_auto",

        # ---- Phase boundaries (inclusive on both ends) ----
        "phase_boundaries": {
            "early": (0, 10),    # 11 layers
            "mid":   (11, 20),   # 10 layers
            "late":  (21, 27),   # 7 layers
        },

        # ---- First three rows: individual layers near the input ----
        "first_layers": [0, 1, 2],

        # ---- Last three rows: individual layers near the output ----
        "last_layers": [25, 26, 27],
    }

    Plot2GapAcrossStack(config).run()


if __name__ == "__main__":
    main()