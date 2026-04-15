"""Download MGSM dataset from GitHub and save as JSON files.

Source: github.com/google-research/url-nlp/mgsm

MGSM structure:
    - TSV files (mgsm_{lang}.tsv): 250 test problems per language (question TAB answer_number)
    - exemplars.py: 8 few-shot exemplars per language (question + CoT answer)

Output:
    train.json: 8 few-shot exemplars per language (from exemplars.py)
    test.json: 250 evaluation problems per language (from TSV files)

Format:
    {
        "en": [{"question": ..., "answer": ..., "answer_number": ...}, ...],
        "bn": [...],
        ...
    }
"""

import csv
import json
import logging
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


class MGSMDownloader:
    """Downloads MGSM dataset from GitHub and saves as structured JSON."""

    REPO_URL = "https://github.com/google-research/url-nlp.git"

    def __init__(self, config: dict):
        self.languages = config["languages"]
        self.data_dir = Path(config["data_dir"])
        self.log_dir = Path(config["log_dir"])
        self.clone_dir = Path(config["clone_dir"])

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

    def _setup_logging(self):
        log_file = self.log_dir / "download_mgsm.log"
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

    def _clone_repo(self):
        """Sparse-clone only the mgsm folder from the GitHub repo."""
        if (self.clone_dir / "mgsm").exists():
            self.logger.info(f"Repo already cloned at {self.clone_dir}")
            return

        self.clone_dir.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Cloning {self.REPO_URL} into {self.clone_dir}")

        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             self.REPO_URL, str(self.clone_dir)],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "mgsm"],
            cwd=str(self.clone_dir),
            check=True, capture_output=True, text=True,
        )
        self.logger.info("Clone complete.")

    def _load_exemplars(self) -> dict:
        """Load few-shot exemplars from exemplars.py (8 per language).

        Returns:
            Dictionary mapping language code to list of exemplar dicts.
        """
        exemplars_path = self.clone_dir / "mgsm" / "exemplars.py"
        self.logger.info(f"Loading exemplars from {exemplars_path}")

        # Import the exemplars module dynamically
        sys.path.insert(0, str(self.clone_dir / "mgsm"))
        from exemplars import MGSM_EXEMPLARS, EXEMPLAR_NUMBER_ANSWERS, EXEMPLAR_EQUATION_SOLUTIONS
        sys.path.pop(0)

        train_data = {}
        for lang in tqdm(self.languages, desc="Loading exemplars"):
            if lang not in MGSM_EXEMPLARS:
                self.logger.warning(f"Language {lang} not found in exemplars, skipping.")
                continue

            examples = []
            for idx_str in sorted(MGSM_EXEMPLARS[lang].keys(), key=int):
                idx = int(idx_str) - 1  # 1-indexed to 0-indexed
                ex = MGSM_EXEMPLARS[lang][idx_str]
                examples.append({
                    "question": ex["q"],
                    "answer": ex["a"],
                    "answer_number": EXEMPLAR_NUMBER_ANSWERS[idx],
                    "equation_solution": EXEMPLAR_EQUATION_SOLUTIONS[idx],
                })

            train_data[lang] = examples
            self.logger.info(f"  {lang}: {len(examples)} exemplars")

        return train_data

    def _load_test_tsv(self) -> dict:
        """Load test problems from TSV files (250 per language).

        TSV format: question TAB answer_number

        Returns:
            Dictionary mapping language code to list of test dicts.
        """
        test_data = {}
        for lang in tqdm(self.languages, desc="Loading test TSVs"):
            tsv_path = self.clone_dir / "mgsm" / f"mgsm_{lang}.tsv"
            if not tsv_path.exists():
                self.logger.warning(f"TSV not found: {tsv_path}, skipping.")
                continue

            examples = []
            with tsv_path.open("r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if len(row) >= 2:
                        question = row[0].strip()
                        answer_number = int(row[1].strip().replace(",", ""))
                        examples.append({
                            "question": question,
                            "answer": None,
                            "answer_number": answer_number,
                        })

            test_data[lang] = examples
            self.logger.info(f"  {lang}: {len(examples)} problems")

        return test_data

    def _save_json(self, data: dict, filename: str):
        """Save dictionary to a JSON file in the data directory."""
        filepath = self.data_dir / filename
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Saved {filepath} ({filepath.stat().st_size / 1024:.1f} KB)")

    def run(self):
        """Clone repo, extract data, and save as JSON files."""
        self.logger.info(f"Languages: {self.languages}")
        self.logger.info(f"Data directory: {self.data_dir}")

        self._clone_repo()

        # 8 few-shot exemplars per language -> train.json
        train_data = self._load_exemplars()
        self._save_json(train_data, "train.json")

        # 250 test problems per language -> test.json
        test_data = self._load_test_tsv()
        self._save_json(test_data, "test.json")

        self.logger.info("Summary:")
        for lang in self.languages:
            n_train = len(train_data.get(lang, []))
            n_test = len(test_data.get(lang, []))
            self.logger.info(f"  {lang}: train={n_train}, test={n_test}")

        self.logger.info("Done.")


def main():
    config = {
        "languages": ["en", "bn", "te", "th", "ru", "zh", "ja"],
        "data_dir": "./exp2/data",
        "log_dir": "./exp2/logs",
        "clone_dir": "./exp2/data/url-nlp",
    }

    downloader = MGSMDownloader(config)
    downloader.run()


if __name__ == "__main__":
    main()