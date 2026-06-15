"""Prepare GRPO train/test splits from MGSM data.

Input:
    train.json: 8 few-shot exemplars per language
    test.json: 250 evaluation problems per language

Output:
    grpo_train.json: 200 examples per language (first 200 from test.json)
    grpo_test.json: 58 examples per language (remaining 50 from test.json + 8 from train.json)

Fields kept: question, answer_number, question_id
    - question_id is a parallel identifier across languages (e.g., question_id=0 in en
      corresponds to the same math problem as question_id=0 in bn).
    - For grpo_test.json, the 8 exemplars get question_ids "exemplar_0" .. "exemplar_7"
      to distinguish them from the 50 test problems.
"""

import json
import logging
from pathlib import Path

from tqdm import tqdm


class GRPOSplitPreparer:
    """Splits MGSM data into GRPO training and testing sets."""

    def __init__(self, config: dict):
        self.data_dir = Path(config["data_dir"])
        self.log_dir = Path(config["log_dir"])
        self.train_size = config["train_size"]
        self.languages = config["languages"]

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        log_file = self.log_dir / "prepare_grpo_splits.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode="w"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Logging to {log_file}")

    def _load_json(self, filename: str) -> dict:
        """Load a JSON file from the data directory."""
        filepath = self.data_dir / filename
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.logger.info(f"Loaded {filepath}")
        return data

    def _save_json(self, data: dict, filename: str):
        """Save dictionary to a JSON file in the data directory."""
        filepath = self.data_dir / filename
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Saved {filepath} ({filepath.stat().st_size / 1024:.1f} KB)")

    def _extract_fields(self, example: dict, question_id: str) -> dict:
        """Keep only the required fields."""
        return {
            "question_id": question_id,
            "question": example["question"],
            "answer_number": example["answer_number"],
        }

    def run(self):
        """Load MGSM data, split, and save GRPO files."""
        mgsm_test = self._load_json("test.json")
        mgsm_train = self._load_json("train.json")

        grpo_train = {}
        grpo_test = {}

        for lang in tqdm(self.languages, desc="Preparing splits"):
            test_examples = mgsm_test[lang]
            train_examples = mgsm_train[lang]

            total_test = len(test_examples)
            assert total_test == 250, f"Expected 250 test examples for {lang}, got {total_test}"
            assert len(train_examples) == 8, f"Expected 8 train examples for {lang}, got {len(train_examples)}"

            # Split 250 -> first 200 for GRPO train, remaining 50 for GRPO test
            grpo_train_examples = [
                self._extract_fields(ex, question_id=f"test_{i}")
                for i, ex in enumerate(test_examples[:self.train_size])
            ]

            # Remaining 50 from test + 8 exemplars = 58 for GRPO test
            grpo_test_examples = [
                self._extract_fields(ex, question_id=f"test_{i}")
                for i, ex in enumerate(test_examples[self.train_size:], start=self.train_size)
            ]
            grpo_test_exemplars = [
                self._extract_fields(ex, question_id=f"exemplar_{i}")
                for i, ex in enumerate(train_examples)
            ]
            grpo_test_examples = grpo_test_examples + grpo_test_exemplars

            grpo_train[lang] = grpo_train_examples
            grpo_test[lang] = grpo_test_examples

            self.logger.info(
                f"  {lang}: grpo_train={len(grpo_train_examples)}, "
                f"grpo_test={len(grpo_test_examples)}"
            )

        self._save_json(grpo_train, "grpo_train.json")
        self._save_json(grpo_test, "grpo_test.json")

        self.logger.info("Done.")


def main():
    config = {
        "languages": ["en", "bn", "te", "th", "ru", "zh", "ja"],
        "data_dir": "./exp2/data",
        "log_dir": "./exp2/logs",
        "train_size": 200,
    }

    preparer = GRPOSplitPreparer(config)
    preparer.run()


if __name__ == "__main__":
    main()