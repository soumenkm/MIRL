"""Plot 4 -- CKA Similarity.

Two analyses derived from the per-prompt, per-layer mean-CoT hidden
states stored by eval.py as ``hidden_mean`` of shape (N, L, d) per
checkpoint:

  1. CROSS-LINGUAL CKA.
     For each (checkpoint j, layer i, target language T):
       X_en : (N_en, d) -- English prompts' mean-CoT activations at layer i
       X_T  : (N_T,  d) -- target lang prompts' mean-CoT activations at layer i
       CKA[j, i, T] = LinearCKA(X_en, X_T) using mean-centered columns
     T includes en for sanity (CKA(en, en) = 1.0 everywhere). 7 targets.

  2. DRIFT CKA (representation drift relative to the base model).
     For each (checkpoint j > 0, layer i, prompt language P):
       X_0  : prompts of language P at checkpoint 0   (the base model)
       X_j  : prompts of language P at checkpoint j
       D[j, i, P] = LinearCKA(X_0, X_j)
     j = 0 column is trivially 1.0 (a matrix vs itself) -- another sanity.
     7 prompt languages, all included.

Linear CKA implementation uses the (N, N) Gram-matrix form (HSIC-based)
which is mathematically identical to the (d, d) cross-covariance form
but is ~70x faster when N << d (we have N ~= 50 vs d = 3584):

      CKA(X, Y)  =  || vec(K_x)^T vec(K_y) ||^2
                    / ( || vec(K_x) ||^2 * || vec(K_y) ||^2 )

where K_x = X_c X_c^T  and  X_c is X with column means subtracted. The
identity follows from trace(A B) = trace(B A) applied to the standard
Kornblith form. Mean-centering is performed on column means (i.e.,
across the N prompts) as in the original paper.

Outputs:
    exp2/outputs/plots/plot4/
        plot4_heatmap_target-<T>.png            x 7   (cross-lingual)
        plot4_lineplot.png                       x 1   (cross-lingual, 7 curves)
        plot4_drift_heatmap_lang-<P>.png         x 7   (drift)
        plot4_drift_lineplot.png                 x 1   (drift, 7 curves)
        plot4_data_summary.json

The script auto-discovers all checkpoint_*.npz files in acts/, so a
rerun after eval.py writes new checkpoints just produces denser plots
without any config edit.

Standards: see project coding standards. All paths are pathlib.Path. No
os, no argparse, no print. tqdm where appropriate. Logger has
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


class Plot4CKASimilarity:
    """Aggregate eval .npz files into Plot 4 figures (CKA)."""

    LANG_LABELS = ("en", "bn", "te", "th", "ru", "ja", "zh")

    # Cross-lingual target list: English is included as the sanity reference
    # (CKA(en, en) = 1.0 everywhere). Other 6 are the comparisons of interest.
    TARGET_LABELS = LANG_LABELS

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
        self.plots_dir = self.output_dir / "plots" / "plot4"

        self.dpi = int(config.get("dpi", 200))
        self.fig_width = float(config.get("fig_width", 6.0))
        self.fig_height = float(config.get("fig_height", 4.5))
        self.cmap = str(config.get("cmap", "viridis"))

        # Numerical stability for the CKA denominator. Below this value the
        # CKA is reported as NaN rather than risk an unstable divide.
        self.cka_eps = float(config.get("cka_eps", 1e-8))

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        # State populated by aggregate().
        self.checkpoint_steps: list[int] = []
        self.n_layers: int | None = None
        self.hidden_dim: int | None = None
        self.lang_to_n_prompts: dict[str, int] = {}
        # CKA[j, i, T_idx]   -- cross-lingual; shape (n_ckpt, L, 7)
        # D[j, i, P_idx]     -- drift relative to step 0; shape (n_ckpt, L, 7)
        self.CKA: np.ndarray | None = None
        self.D: np.ndarray | None = None

    def _setup_logging(self):
        log_file = self.log_dir / "plot4.log"
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
    #  Linear CKA (Gram-matrix form, mean-centered columns)
    # ------------------------------------------------------------------ #

    def _linear_cka(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Mean-centered Linear CKA between two (N, d) matrices.

        Uses the (N, N) Gram-matrix form, which is mathematically
        identical to Kornblith et al.'s (d, d) cross-covariance form
        but ~70x faster when N << d.

        Returns CKA in [0, 1]; NaN if either matrix is constant.

        Pre-conditions: X and Y must have the same number of rows N
        (i.e., same number of prompts) -- this is how Kornblith's CKA
        is defined. We do NOT subsample to the smaller N here; the
        caller is responsible for ensuring matching N (typically all
        languages have the same eval set size).
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"_linear_cka: X has {X.shape[0]} rows, Y has "
                f"{Y.shape[0]}; CKA requires matching N."
            )
        if X.shape[0] < 2:
            return float("nan")
        # Cast to float32 for the math: hidden_mean is fp16 and we need
        # at least fp32 for stable Frobenius inner products.
        Xc = X.astype(np.float32, copy=False)
        Yc = Y.astype(np.float32, copy=False)
        Xc = Xc - Xc.mean(axis=0, keepdims=True)
        Yc = Yc - Yc.mean(axis=0, keepdims=True)
        Kx = Xc @ Xc.T   # (N, N)
        Ky = Yc @ Yc.T   # (N, N)
        num = float((Kx * Ky).sum())
        den_sq = float((Kx * Kx).sum()) * float((Ky * Ky).sum())
        if den_sq <= self.cka_eps:
            return float("nan")
        return num / float(np.sqrt(den_sq))

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
                f"No checkpoint_*.npz files found in {self.acts_dir}. "
                f"Run eval.py first."
            )
        self.logger.info(
            f"Discovered {len(found)} checkpoints: "
            f"{[step for step, _ in found]}"
        )
        return found

    # ------------------------------------------------------------------ #
    #  Per-checkpoint matrices
    # ------------------------------------------------------------------ #

    def _load_per_lang_matrices(
        self, npz_path: Path
    ) -> tuple[dict[str, np.ndarray], int, int]:
        """Load hidden_mean from one .npz, split by language.

        Returns:
            per_lang : dict mapping lang code -> (N_l, L, d) fp32 array of
                       valid (non-empty-CoT) prompts in that language
            n_layers : int
            hidden_dim : int
        """
        with np.load(npz_path, allow_pickle=True) as d:
            n_layers = int(d["n_layers"])
            hidden_dim = int(d["hidden_dim"])
            lang_labels = list(d["lang_labels"])
            prompt_langs = d["prompt_langs"]
            cot_lengths = d["cot_lengths"]
            hidden_mean = d["hidden_mean"]  # (N, L, d) fp16, dense
        if list(lang_labels) != list(self.LANG_LABELS):
            raise ValueError(
                f"{npz_path.name} has lang_labels={lang_labels}, "
                f"expected {self.LANG_LABELS}"
            )
        per_lang: dict[str, np.ndarray] = {}
        for lang in self.LANG_LABELS:
            mask = (prompt_langs == lang) & (cot_lengths > 0)
            if not mask.any():
                continue
            # Cast to fp32 once for downstream CKA; this is the bulk of
            # memory but it is needed for stable matrix products.
            per_lang[lang] = hidden_mean[mask].astype(np.float32, copy=False)
        return per_lang, n_layers, hidden_dim

    # ------------------------------------------------------------------ #
    #  Aggregation
    # ------------------------------------------------------------------ #

    def aggregate(self):
        """Build CKA and D tensors.

        CKA shape: (n_ckpt, L, 7 target-langs)
            For each checkpoint j and layer i, computes
            LinearCKA(X_en, X_T) for each T in TARGET_LABELS. T includes
            English (CKA(en, en) = 1.0 everywhere -- a sanity check that
            the CKA implementation is correct).

        D shape: (n_ckpt, L, 7 prompt-langs)
            For each checkpoint j and layer i, computes
            LinearCKA(X_P at step 0, X_P at step j). At step 0 this is
            trivially 1.0 (matrix vs itself) -- a sanity that step 0 is
            anchored correctly.
        """
        npz_paths = self._discover_checkpoints()
        n_ckpt = len(npz_paths)
        n_lang = len(self.LANG_LABELS)

        # Load step 0 first so we have the drift reference in RAM. If the
        # smallest discovered step is not 0, we still pick the smallest
        # one (typically still the base model in eval.py's convention).
        base_step, base_path = npz_paths[0]
        if base_step != 0:
            self.logger.warning(
                f"Smallest discovered checkpoint is step={base_step}, "
                f"not 0. Drift CKA will reference step={base_step}."
            )
        self.logger.info(f"Loading drift reference: step={base_step}")
        base_per_lang, n_layers_base, hidden_dim_base = (
            self._load_per_lang_matrices(base_path)
        )
        self.n_layers = n_layers_base
        self.hidden_dim = hidden_dim_base

        CKA = np.full((n_ckpt, n_layers_base, n_lang), np.nan, dtype=np.float64)
        D = np.full((n_ckpt, n_layers_base, n_lang), np.nan, dtype=np.float64)
        steps: list[int] = []
        lang_to_n_prompts: dict[str, int] = {l: 0 for l in self.LANG_LABELS}

        for j, (step, path) in enumerate(tqdm(
            npz_paths, desc="aggregating checkpoints", unit="ckpt"
        )):
            steps.append(step)
            if step == base_step:
                per_lang = base_per_lang
            else:
                per_lang, n_layers_j, _ = self._load_per_lang_matrices(path)
                if n_layers_j != self.n_layers:
                    raise ValueError(
                        f"{path.name} has n_layers={n_layers_j}, "
                        f"expected {self.n_layers}."
                    )

            # Record N per language for the summary.
            for lang in self.LANG_LABELS:
                if lang in per_lang:
                    lang_to_n_prompts[lang] = max(
                        lang_to_n_prompts[lang], int(per_lang[lang].shape[0])
                    )

            # ---- Cross-lingual CKA: vs English at this checkpoint ----
            if "en" in per_lang:
                X_en_LND = per_lang["en"]  # (N_en, L, d)
                for t_idx, T in enumerate(self.TARGET_LABELS):
                    if T not in per_lang:
                        continue
                    X_T_LND = per_lang[T]
                    # Match N by truncating to the smaller of the two. CKA
                    # requires matching N; if a language has fewer eval
                    # prompts than English (e.g. partial eval) we use the
                    # first N_min prompts of each.
                    n_min = min(X_en_LND.shape[0], X_T_LND.shape[0])
                    if n_min < 2:
                        continue
                    X_en = X_en_LND[:n_min]
                    X_T = X_T_LND[:n_min]
                    for i in range(self.n_layers):
                        CKA[j, i, t_idx] = self._linear_cka(
                            X_en[:, i, :], X_T[:, i, :]
                        )

            # ---- Drift CKA: vs same-language step-0 representation ----
            for p_idx, P in enumerate(self.LANG_LABELS):
                if P not in per_lang or P not in base_per_lang:
                    continue
                X_now = per_lang[P]
                X_base = base_per_lang[P]
                n_min = min(X_now.shape[0], X_base.shape[0])
                if n_min < 2:
                    continue
                X_now = X_now[:n_min]
                X_base = X_base[:n_min]
                for i in range(self.n_layers):
                    D[j, i, p_idx] = self._linear_cka(
                        X_base[:, i, :], X_now[:, i, :]
                    )

        self.checkpoint_steps = steps
        self.CKA = CKA
        self.D = D
        self.lang_to_n_prompts = lang_to_n_prompts

        self.logger.info(
            f"Aggregated CKA shape={CKA.shape}, D shape={D.shape} "
            f"(checkpoints={n_ckpt}, layers={self.n_layers}, "
            f"languages={n_lang})"
        )
        for lang in self.LANG_LABELS:
            self.logger.info(
                f"  lang {lang}: max N across checkpoints = "
                f"{lang_to_n_prompts[lang]}"
            )
        self._log_sanity_checks()

    def _log_sanity_checks(self):
        """Surface the two implementation sanity values in the log.

        - CKA(en, en) should be ~1.0 at every (j, i).
        - D[step 0, :, P] should be ~1.0 at every layer.
        """
        if self.CKA is None or self.D is None:
            return
        en_idx = self.LANG_LABELS.index("en")
        en_en_vals = self.CKA[:, :, en_idx]
        finite_en = en_en_vals[np.isfinite(en_en_vals)]
        if finite_en.size:
            self.logger.info(
                f"Sanity: CKA(en, en) range "
                f"[min={finite_en.min():.6f}, max={finite_en.max():.6f}] "
                f"-- expected near 1.0 everywhere."
            )
        step0_drift = self.D[0, :, :]
        finite_d = step0_drift[np.isfinite(step0_drift)]
        if finite_d.size:
            self.logger.info(
                f"Sanity: D[step={self.checkpoint_steps[0]}, :, :] range "
                f"[min={finite_d.min():.6f}, max={finite_d.max():.6f}] "
                f"-- expected near 1.0 everywhere."
            )

    # ------------------------------------------------------------------ #
    #  Plotting helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_lang(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", s)

    def _make_x_edges(self, steps: np.ndarray) -> np.ndarray:
        """pcolormesh column edges so columns are linear in step."""
        if len(steps) == 1:
            half = 0.5
            return np.array([steps[0] - half, steps[0] + half])
        mids = (steps[:-1] + steps[1:]) / 2.0
        left = steps[0] - (mids[0] - steps[0])
        right = steps[-1] + (steps[-1] - mids[-1])
        return np.concatenate([[left], mids, [right]])

    def _caption_crosslingual(self, target_lang: str) -> str:
        n_en = self.lang_to_n_prompts.get("en", 0)
        n_t = self.lang_to_n_prompts.get(target_lang, 0)
        parts = [
            f"CKA(X_en, X_T) per (layer, GRPO step).",
            f"X_en = English prompts' mean-CoT activations (N={n_en}); "
            f"X_T = {self.LANG_DISPLAY[target_lang]} prompts' mean-CoT "
            f"activations (N={n_t}).",
            "Linear CKA with column-mean-centered activations, computed "
            "via the (N, N) Gram-matrix form.",
        ]
        if target_lang == "en":
            parts.append(
                "Target = English serves as a sanity check: CKA(X, X) is "
                "1.0 by construction, so this heatmap should be a "
                "constant bright color across all (layer, step) cells."
            )
        return " ".join(parts)

    def _caption_drift(self, prompt_lang: str) -> str:
        n = self.lang_to_n_prompts.get(prompt_lang, 0)
        base_step = self.checkpoint_steps[0]
        parts = [
            f"Drift CKA: similarity between {self.LANG_DISPLAY[prompt_lang]} "
            f"prompts' mean-CoT activations at step j vs. those same "
            f"prompts at step {base_step} (the base model), per layer "
            f"(N={n}).",
            "Linear CKA with column-mean-centered activations.",
            f"Column at step={base_step} is 1.0 by construction; lower "
            f"values at later steps indicate larger representation drift.",
        ]
        return " ".join(parts)

    def _caption_lineplot_crosslingual(self) -> str:
        n_en = self.lang_to_n_prompts.get("en", 0)
        return (
            "Layer-averaged cross-lingual CKA: CKA(X_en, X_T) averaged "
            "over all transformer layers, plotted vs GRPO step, one "
            "curve per target language. The English curve is constant at "
            f"1.0 by construction (sanity). X_en uses N={n_en} English "
            "prompts; per-target N is shown in the data summary JSON."
        )

    def _caption_lineplot_drift(self) -> str:
        base_step = self.checkpoint_steps[0]
        return (
            f"Layer-averaged representation drift: CKA(X at step "
            f"{base_step}, X at step j) averaged over all layers, "
            f"plotted vs GRPO step, one curve per prompt language. All "
            f"curves are 1.0 at step={base_step} by construction "
            f"(sanity). Lower values at later steps indicate larger "
            f"representation drift."
        )

    # ------------------------------------------------------------------ #
    #  Cross-lingual heatmaps (7 figures)
    # ------------------------------------------------------------------ #

    def plot_crosslingual_heatmaps(self):
        if self.CKA is None:
            raise RuntimeError("Call aggregate() before plotting.")
        steps = np.asarray(self.checkpoint_steps, dtype=float)
        x_edges = self._make_x_edges(steps)
        y_edges = np.arange(self.n_layers + 1) - 0.5

        for t_idx, T in enumerate(tqdm(
            self.TARGET_LABELS, desc="cross-ling heatmaps", unit="fig"
        )):
            grid = self.CKA[:, :, t_idx].T  # (L, n_ckpt)
            fig, ax = plt.subplots(
                figsize=(self.fig_width, self.fig_height), dpi=self.dpi,
            )
            mesh = ax.pcolormesh(
                x_edges, y_edges, grid,
                cmap=self.cmap, vmin=0.0, vmax=1.0, shading="flat",
            )
            ax.set_xlabel("GRPO step")
            ax.set_ylabel("Layer (0 = first transformer block)")
            ax.set_ylim(-0.5, self.n_layers - 0.5)
            ax.set_yticks(np.arange(0, self.n_layers, 4))
            ax.set_title(
                f"Plot 4: Cross-lingual CKA Similarity\n"
                f"X_en vs X_{T} (target = {self.LANG_DISPLAY[T]})"
            )
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label("Linear CKA")
            fig.subplots_adjust(bottom=0.30)
            fig.text(
                0.5, 0.02, self._caption_crosslingual(T),
                ha="center", va="bottom", fontsize=8, wrap=True,
            )
            out = self.plots_dir / (
                f"plot4_heatmap_target-{self._safe_lang(T)}.png"
            )
            fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

        self.logger.info(
            f"Wrote cross-lingual heatmaps to {self.plots_dir} "
            f"({len(self.TARGET_LABELS)} files)."
        )

    # ------------------------------------------------------------------ #
    #  Cross-lingual line plot (1 figure, 7 curves)
    # ------------------------------------------------------------------ #

    def plot_crosslingual_lineplot(self):
        if self.CKA is None:
            raise RuntimeError("Call aggregate() before plotting.")
        steps = np.asarray(self.checkpoint_steps, dtype=float)
        # Layer-average via nanmean so any NaN layers don't drag curves.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cka_layer_avg = np.nanmean(self.CKA, axis=1)  # (n_ckpt, 7)

        palette = plt.get_cmap("tab10")
        colors = {
            T: palette(self.LANG_LABELS.index(T)) for T in self.TARGET_LABELS
        }

        fig, ax = plt.subplots(
            figsize=(self.fig_width + 1.5, self.fig_height), dpi=self.dpi,
        )
        for t_idx, T in enumerate(self.TARGET_LABELS):
            ax.plot(
                steps, cka_layer_avg[:, t_idx],
                marker="o", markersize=4, linewidth=1.5,
                color=colors[T], label=self.LANG_DISPLAY[T],
            )
        ax.set_xlabel("GRPO step")
        ax.set_ylabel(
            "Layer-averaged CKA  (mean over layers 0..%d)" % (self.n_layers - 1)
        )
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(
            title="Target language vs English",
            fontsize=8, title_fontsize=8, loc="best", framealpha=0.9,
        )
        ax.set_title(
            "Plot 4: Layer-averaged Cross-lingual CKA Similarity"
        )
        fig.subplots_adjust(bottom=0.24)
        fig.text(
            0.5, 0.02, self._caption_lineplot_crosslingual(),
            ha="center", va="bottom", fontsize=8, wrap=True,
        )
        out = self.plots_dir / "plot4_lineplot.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Wrote {out}")

    # ------------------------------------------------------------------ #
    #  Drift heatmaps (7 figures)
    # ------------------------------------------------------------------ #

    def plot_drift_heatmaps(self):
        if self.D is None:
            raise RuntimeError("Call aggregate() before plotting.")
        steps = np.asarray(self.checkpoint_steps, dtype=float)
        x_edges = self._make_x_edges(steps)
        y_edges = np.arange(self.n_layers + 1) - 0.5

        for p_idx, P in enumerate(tqdm(
            self.LANG_LABELS, desc="drift heatmaps", unit="fig"
        )):
            grid = self.D[:, :, p_idx].T  # (L, n_ckpt)
            fig, ax = plt.subplots(
                figsize=(self.fig_width, self.fig_height), dpi=self.dpi,
            )
            mesh = ax.pcolormesh(
                x_edges, y_edges, grid,
                cmap=self.cmap, vmin=0.0, vmax=1.0, shading="flat",
            )
            ax.set_xlabel("GRPO step")
            ax.set_ylabel("Layer (0 = first transformer block)")
            ax.set_ylim(-0.5, self.n_layers - 0.5)
            ax.set_yticks(np.arange(0, self.n_layers, 4))
            base_step = self.checkpoint_steps[0]
            ax.set_title(
                f"Plot 4: Representation Drift (CKA vs. step {base_step})\n"
                f"lang = {self.LANG_DISPLAY[P]}"
            )
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label("Linear CKA")
            fig.subplots_adjust(bottom=0.30)
            fig.text(
                0.5, 0.02, self._caption_drift(P),
                ha="center", va="bottom", fontsize=8, wrap=True,
            )
            out = self.plots_dir / (
                f"plot4_drift_heatmap_lang-{self._safe_lang(P)}.png"
            )
            fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

        self.logger.info(
            f"Wrote drift heatmaps to {self.plots_dir} "
            f"({len(self.LANG_LABELS)} files)."
        )

    # ------------------------------------------------------------------ #
    #  Drift line plot (1 figure, 7 curves)
    # ------------------------------------------------------------------ #

    def plot_drift_lineplot(self):
        if self.D is None:
            raise RuntimeError("Call aggregate() before plotting.")
        steps = np.asarray(self.checkpoint_steps, dtype=float)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            d_layer_avg = np.nanmean(self.D, axis=1)  # (n_ckpt, 7)

        palette = plt.get_cmap("tab10")
        colors = {
            P: palette(self.LANG_LABELS.index(P)) for P in self.LANG_LABELS
        }

        fig, ax = plt.subplots(
            figsize=(self.fig_width + 1.5, self.fig_height), dpi=self.dpi,
        )
        for p_idx, P in enumerate(self.LANG_LABELS):
            ax.plot(
                steps, d_layer_avg[:, p_idx],
                marker="o", markersize=4, linewidth=1.5,
                color=colors[P], label=self.LANG_DISPLAY[P],
            )
        ax.set_xlabel("GRPO step")
        ax.set_ylabel(
            "Layer-averaged CKA  (mean over layers 0..%d)" % (self.n_layers - 1)
        )
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(
            title="Prompt language", fontsize=8, title_fontsize=8,
            loc="best", framealpha=0.9,
        )
        base_step = self.checkpoint_steps[0]
        ax.set_title(
            f"Plot 4: Layer-averaged Representation Drift "
            f"(vs. step {base_step})"
        )
        fig.subplots_adjust(bottom=0.24)
        fig.text(
            0.5, 0.02, self._caption_lineplot_drift(),
            ha="center", va="bottom", fontsize=8, wrap=True,
        )
        out = self.plots_dir / "plot4_drift_lineplot.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Wrote {out}")

    # ------------------------------------------------------------------ #
    #  Summary metadata
    # ------------------------------------------------------------------ #

    def write_summary(self):
        if self.CKA is None or self.D is None:
            raise RuntimeError("Call aggregate() before write_summary().")
        en_idx = self.LANG_LABELS.index("en")
        en_en_vals = self.CKA[:, :, en_idx]
        en_en_finite = en_en_vals[np.isfinite(en_en_vals)]
        step0_drift = self.D[0, :, :]
        step0_drift_finite = step0_drift[np.isfinite(step0_drift)]

        summary = {
            "checkpoint_steps": self.checkpoint_steps,
            "n_checkpoints": len(self.checkpoint_steps),
            "n_layers": int(self.n_layers),
            "hidden_dim": int(self.hidden_dim),
            "languages": list(self.LANG_LABELS),
            "target_languages_cross_lingual": list(self.TARGET_LABELS),
            "prompts_per_language": {
                k: int(v) for k, v in self.lang_to_n_prompts.items()
            },
            "cmap": self.cmap,
            "cka_method": (
                "Linear CKA via (N, N) Gram-matrix form on "
                "column-mean-centered activations; mathematically "
                "identical to Kornblith et al. 2019."
            ),
            "sanity_checks": {
                "cka_en_en_min": (
                    float(en_en_finite.min()) if en_en_finite.size else None
                ),
                "cka_en_en_max": (
                    float(en_en_finite.max()) if en_en_finite.size else None
                ),
                "drift_step0_min": (
                    float(step0_drift_finite.min())
                    if step0_drift_finite.size else None
                ),
                "drift_step0_max": (
                    float(step0_drift_finite.max())
                    if step0_drift_finite.size else None
                ),
            },
            "notes": (
                "Cross-lingual CKA[j, i, T] = LinearCKA between English "
                "and target-T mean-CoT activations at layer i of "
                "checkpoint j. Drift D[j, i, P] = LinearCKA between "
                "language P's activations at the first discovered "
                "checkpoint (base model) and at checkpoint j. Both use "
                "the (N, N) Gram form for speed and mean-center column "
                "by column as in Kornblith et al."
            ),
        }
        out = self.plots_dir / "plot4_data_summary.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        self.logger.info(f"Wrote summary to {out}")

    # ------------------------------------------------------------------ #
    #  Orchestrator
    # ------------------------------------------------------------------ #

    def run(self):
        self.logger.info("=" * 80)
        self.logger.info(f"Plot 4 starting; reading from {self.acts_dir}")
        self.logger.info("=" * 80)
        self.aggregate()
        self.plot_crosslingual_heatmaps()
        self.plot_crosslingual_lineplot()
        self.plot_drift_heatmaps()
        self.plot_drift_lineplot()
        self.write_summary()
        self.logger.info("Plot 4 complete.")


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

        # ---- Figure styling ----
        "dpi": 200,
        "fig_width": 6.0,
        "fig_height": 4.5,
        "cmap": "viridis",

        # ---- CKA numerical knobs ----
        "cka_eps": 1e-8,
    }

    Plot4CKASimilarity(config).run()


if __name__ == "__main__":
    main()