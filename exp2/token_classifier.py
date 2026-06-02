"""Unicode-script-based token-to-language classification.

For each token ID in the policy tokenizer's vocabulary, decode it to its
string form and assign a language label by majority Unicode script. This
is Method 1 from the project notes (Hyperpolyglot LLMs style) and is the
simpler of the two planned classifiers.

For this project's seven languages (en, bn, te, th, ru, ja, zh) the scripts
are essentially non-overlapping, so Unicode script gives a clean
deterministic mapping with effectively 100% accuracy. The one ambiguity is
between Japanese kanji and Chinese — both use HAN. The convention used here
matches Wendler et al. (EMNLP 2023):

    - Token contains any HIRAGANA or KATAKANA characters -> ja
      (Chinese never uses these, so they unambiguously mark Japanese.)
    - Token is pure HAN (no kana) -> zh
      (Japanese text routinely mixes kana with kanji; pure-HAN tokens are
      far more likely to come from Chinese contexts. Acknowledged
      limitation: pure-HAN tokens that appear in a Japanese sentence will
      be labelled zh by this method. Method 2 -- langid.py -- would
      partially address this, but for sentence-level inference it is not
      a major issue.)

Label set (10 buckets):

    Languages:      {en, bn, te, th, ru, ja, zh}
    Non-language:   {punct_num, special, other}

    - punct_num: tokens consisting purely of digits, punctuation, or
      whitespace (Unicode script class COMMON). These appear in every
      language's text, so attributing them to any single language would
      be wrong.
    - special: model-internal tokens like <|im_start|>, <|endoftext|>,
      <pad>. Detected via tokenizer.all_special_ids plus a heuristic for
      angle-bracket-wrapped names.
    - other: byte-fallback tokens (<0xE0>, etc.), empty decodes, mixed
      tokens dominated by a script outside the seven studied.

Outputs:

    output_dir/token_language_map.json     -- {token_id: label, ...}
    output_dir/token_language_counts.json  -- {label: count, ...}

The plotting code consumes these JSON files; no need to rerun
classification per plot.
"""

import json
import logging
import unicodedata
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


class TokenLanguageClassifier:
    """Build a token_id -> language_label mapping for the policy tokenizer."""

    # Mapping from Unicode script names (as returned by unicodedata) to
    # language labels. HAN is handled specially in _classify_string because
    # of the ja/zh ambiguity.
    SCRIPT_TO_LANG = {
        "LATIN":    "en",
        "BENGALI":  "bn",
        "TELUGU":   "te",
        "THAI":     "th",
        "CYRILLIC": "ru",
        # HIRAGANA, KATAKANA, HAN: see _classify_string()
    }

    # Unicode "general category" prefixes that indicate punctuation,
    # numbers, whitespace, or symbols. These don't carry language signal.
    # 'C' covers Cc (control chars like \t, \n) and Cf (format chars like
    # zero-width joiner) which behave like whitespace/format for our
    # purposes -- they appear in every language's text. We deliberately
    # do NOT include 'L' (letters) or 'M' (combining marks, which attach
    # to letters and carry the same script identity).
    PUNCT_NUM_CATEGORIES = ("P", "N", "Z", "S", "C")

    def __init__(self, config: dict):
        self.tokenizer_path = Path(config["tokenizer_path"])
        self.output_dir = Path(config["output_dir"])
        self.log_dir = Path(config["log_dir"])
        self.data_dir = Path(config["data_dir"])

        self.languages = list(config.get(
            "languages", ["en", "bn", "te", "th", "ru", "ja", "zh"]
        ))
        # Full label set: languages + three non-language buckets.
        self.all_labels = self.languages + ["punct_num", "special", "other"]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        self.logger.info(f"Loading tokenizer from {self.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.tokenizer_path),
            trust_remote_code=True,
        )
        self.vocab_size = len(self.tokenizer)
        self.logger.info(f"Vocab size: {self.vocab_size}")

        # Cache the set of special-token IDs once; lookup must be O(1).
        self.special_token_ids = set(self.tokenizer.all_special_ids)
        self.logger.info(
            f"Special token IDs from tokenizer: {len(self.special_token_ids)}"
        )

    def _setup_logging(self):
        log_file = self.log_dir / "token_classifier.log"
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
    #  Single-character helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _script_of_char(ch: str) -> str:
        """Return the Unicode script name of a single character.

        unicodedata.name returns names like 'BENGALI LETTER KA',
        'LATIN SMALL LETTER A', 'CJK UNIFIED IDEOGRAPH-4E2D',
        'HIRAGANA LETTER A', etc. The first word is (almost always) the
        script. For 'CJK UNIFIED IDEOGRAPH-...' the script is HAN, which
        we detect from the 'CJK' prefix.

        Returns an empty string for characters with no name (e.g. control
        characters, surrogates).
        """
        try:
            name = unicodedata.name(ch)
        except ValueError:
            return ""
        if name.startswith("CJK UNIFIED IDEOGRAPH") or name.startswith("CJK COMPATIBILITY IDEOGRAPH"):
            return "HAN"
        # First word is the script for the vast majority of named chars.
        return name.split(" ", 1)[0]

    @staticmethod
    def _is_punct_num(ch: str) -> bool:
        """True if the character is punctuation, number, whitespace, or symbol.

        Uses Unicode general category prefixes: P*, N*, Z*, S*.
        """
        return unicodedata.category(ch)[0] in TokenLanguageClassifier.PUNCT_NUM_CATEGORIES

    # ------------------------------------------------------------------ #
    #  Token-string -> label
    # ------------------------------------------------------------------ #

    def _classify_string(self, s: str) -> str:
        """Classify a decoded token string into a language label.

        Algorithm:
          1. Empty string -> 'other'.
          2. If every char is punct/num/space/symbol -> 'punct_num'.
          3. Otherwise, walk all "letter-class" chars and tally scripts.
             - Any HIRAGANA or KATAKANA present -> 'ja' (unambiguous).
             - Pure HAN with no kana -> 'zh' (project convention).
             - Otherwise majority script -> language via SCRIPT_TO_LANG.
             - Script not in SCRIPT_TO_LANG (e.g. ARABIC, HEBREW) -> 'other'.
          4. Ties broken by SCRIPT_TO_LANG ordering: HAN > kana logic above
             handles the only ambiguity that matters for this language set.
        """
        if not s:
            return "other"

        # Strip the BPE leading-space marker before classifying. Qwen2.5
        # uses 'Ġ' (U+0120) like GPT-2/Llama; we should not let it bias
        # toward LATIN for non-Latin tokens whose first char is the
        # space-marker.
        s_classify = s.lstrip("\u0120").lstrip("\u2581")  # Ġ and ▁

        if not s_classify:
            # Token was pure leading-space marker.
            return "punct_num"

        # Check for pure punctuation/numbers/whitespace.
        if all(self._is_punct_num(ch) for ch in s_classify):
            return "punct_num"

        # Tally scripts across the non-punct/num chars only. Including
        # punctuation in the tally would let a Bengali word with a period
        # at the end count one vote for COMMON.
        script_counts: Counter[str] = Counter()
        has_hiragana = False
        has_katakana = False

        for ch in s_classify:
            if self._is_punct_num(ch):
                continue
            script = self._script_of_char(ch)
            if not script:
                continue
            if script == "HIRAGANA":
                has_hiragana = True
            elif script == "KATAKANA":
                has_katakana = True
            script_counts[script] += 1

        if not script_counts:
            # No classifiable letter chars (e.g. only control chars).
            return "other"

        # Japanese disambiguation: any kana present -> ja, regardless of
        # how much HAN the token also contains.
        if has_hiragana or has_katakana:
            return "ja"

        # Otherwise majority script. ties broken by Counter's insertion
        # order (which mirrors first-occurrence in the string).
        majority_script, _ = script_counts.most_common(1)[0]

        if majority_script == "HAN":
            return "zh"
        if majority_script in self.SCRIPT_TO_LANG:
            return self.SCRIPT_TO_LANG[majority_script]
        return "other"

    def _classify_token_id(self, token_id: int) -> str:
        """Classify a token ID via decode + string classification.

        Order:
          - special_token_ids fast path -> 'special'
          - decode to a UTF-8 string (this is the critical step: for
            byte-level BPE tokenizers like Qwen2.5 / GPT-2 / Llama,
            convert_ids_to_tokens returns the mojibake byte-level form
            where every non-ASCII byte is a Latin-block placeholder. So
            a Bengali token like 'আমি' would come back as 'à¦Ĩà¦®à¦¿'
            and get misclassified as English. tokenizer.decode reverses
            this encoding and returns the actual character.)
          - empty decode -> 'other' (byte-fallback tokens, etc.)
          - heuristic: angle-bracket-wrapped chat templates -> 'special'
            (catches tokenizer-specific markers not in all_special_ids)
          - otherwise hand off to _classify_string
        """
        if token_id in self.special_token_ids:
            return "special"

        # decode() reverses the byte-level BPE encoding. For a single
        # token there is no "subword merging" concern (it's a single
        # piece). Leading spaces become real ' ' characters, which
        # _classify_string handles correctly via the punct/num path.
        try:
            token_str = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            token_str = ""

        if not token_str:
            return "other"

        # Heuristic for chat-template specials not declared in
        # all_special_ids (e.g. <|im_start|>, <|endoftext|>).
        stripped = token_str.strip()
        if stripped.startswith("<|") and stripped.endswith("|>"):
            return "special"
        if (
            stripped.startswith("<")
            and stripped.endswith(">")
            and len(stripped) > 2
        ):
            inner = stripped[1:-1]
            if inner.startswith("0x"):
                return "other"
            return "special"

        return self._classify_string(token_str)

    # ------------------------------------------------------------------ #
    #  Build + persist
    # ------------------------------------------------------------------ #

    def build(self) -> dict[int, str]:
        """Classify every token in the vocab, save mapping + counts to JSON."""
        self.logger.info(f"Classifying {self.vocab_size} tokens...")

        mapping: dict[int, str] = {}
        counts: Counter[str] = Counter()

        for token_id in tqdm(
            range(self.vocab_size),
            desc="Classifying tokens",
            unit="tok",
        ):
            label = self._classify_token_id(token_id)
            mapping[token_id] = label
            counts[label] += 1

        # Save mapping. JSON keys must be strings, so int->str on the way out.
        map_path = self.output_dir / "token_language_map.json"
        with map_path.open("w", encoding="utf-8") as f:
            json.dump(
                {str(k): v for k, v in mapping.items()},
                f,
                ensure_ascii=False,
            )
        self.logger.info(f"Saved token-language map to {map_path}")

        # Save counts in a stable ordering (the all_labels order, not the
        # Counter's most_common ordering, so downstream code knows what
        # labels to expect).
        counts_dict = {label: counts.get(label, 0) for label in self.all_labels}
        counts_path = self.output_dir / "token_language_counts.json"
        with counts_path.open("w", encoding="utf-8") as f:
            json.dump(counts_dict, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved label counts to {counts_path}")

        self._log_summary(counts_dict)
        return mapping

    def _log_summary(self, counts: dict):
        total = sum(counts.values())
        self.logger.info("=" * 60)
        self.logger.info(f"Token-language classification summary (vocab={total}):")
        for label in self.all_labels:
            n = counts.get(label, 0)
            pct = 100.0 * n / total if total else 0.0
            self.logger.info(f"  {label:>10s}: {n:>7d}  ({pct:5.2f}%)")
        self.logger.info("=" * 60)

        # Quick sanity: low-resource languages with very few dedicated
        # tokens are exactly the cross-lingual collapse risk that this
        # project is built to study.
        for lang in self.languages:
            n = counts.get(lang, 0)
            if n < 100:
                self.logger.warning(
                    f"Language '{lang}' has only {n} dedicated tokens in "
                    f"the vocab -- expect heavy fragmentation in {lang} text."
                )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main_logger = logging.getLogger("Main")

    # Token classification is CPU-only (just tokenizer.decode + Unicode
    # category lookups). A GPU is not required, but we keep the check for
    # standards compliance since the rest of the pipeline assumes a GPU
    # environment.
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected.")
    main_logger.info(
        f"GPUs available: {torch.cuda.device_count()} "
        f"({torch.cuda.get_device_name(0)})"
    )

    config = {
        # Required path roots per coding standards.
        "log_dir":    Path("./exp2/logs"),
        "output_dir": Path("./exp2/outputs"),
        "data_dir":   Path("./exp2/data"),

        # Must match the tokenizer used by the trained policy. The
        # token-language map is keyed by token ID, and IDs only make sense
        # in the context of a specific tokenizer.
        "tokenizer_path": Path("./models/Qwen2.5-7B-Instruct"),

        # The seven languages studied. Tokens not falling into any of
        # these (or into punct_num / special / other) go to 'other'.
        "languages": ["en", "bn", "te", "th", "ru", "ja", "zh"],
    }

    classifier = TokenLanguageClassifier(config)
    classifier.build()


if __name__ == "__main__":
    main()