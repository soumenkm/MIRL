"""GRPO training dataset for cross-lingual collapse experiments.

Prompt structure (for each example):
    Solve the following math problem step by step, show your reasoning and then give the
    final answer after "Final Answer: " [Always in English]

    Question: <few-shot question in target lang>
    Step-by-step Answer: <few-shot CoT answer in target lang>
    Final Answer: <answer number in English digits>

    ... (repeat few-shot block N times)

    Question: <actual question in target lang>
    Step-by-step Answer:

Notes:
    - Instruction line + field headers ("Question:", "Step-by-step Answer:", "Final Answer:")
      are always in English.
    - Only the actual question text and the step-by-step answer text are in the target language.
    - Few-shot exemplars come from grpo_test.json (the 8 items with question_id starting
      with "exemplar_"), which contain the full CoT answer.
    - The main question comes from either grpo_train.json (for training) or
      grpo_test.json (for evaluation).
"""

import json
import logging
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class GRPOMGSMDataset(Dataset):
    """MGSM-based dataset for GRPO training with few-shot CoT prompting."""

    INSTRUCTION = (
        "You are a careful mathematical reasoner. Solve the problem by writing out your "
        "reasoning step by step, then conclude with the final numeric answer.\n"
        "\n"
        "Formatting rules (follow strictly):\n"
        "- Write your reasoning in the same language as the question.\n"
        "- Use clear arithmetic steps. Show every calculation.\n"
        '- End your response with a line of exactly this form: "Final Answer: <number>"\n'
        "- The final answer must be a single integer written in English digits (e.g., 42), "
        "with no units, no words, no commas, and no trailing punctuation.\n"
        "- Do not write anything after the final answer line."
    )

    def __init__(self, config: dict, split: str):
        """Initialise the dataset.

        Args:
            config: Configuration dictionary (see main() in this file for schema).
            split: Either "train" or "test". Determines which question source to use.
        """
        self.split = split
        self.data_dir = Path(config["data_dir"])
        self.log_dir = Path(config["log_dir"])
        self.languages = config["languages"]
        self.num_few_shot = config["num_few_shot"]
        self.seed = config["seed"]

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()

        # Load the full JSON files (both needed: few-shot pool from train,
        # eval questions from either)
        self.grpo_train = self._load_json("grpo_train.json")
        self.grpo_test = self._load_json("grpo_test.json")

        # Build the few-shot pool: keep only exemplar_* entries from grpo_test
        # (these are the 8 exemplars per language with full CoT answers)
        self.few_shot_pool = self._build_few_shot_pool()

        # Build the question list for the requested split
        self.examples = self._build_examples()

        self.logger.info(
            f"Dataset initialised (split={split}, total_examples={len(self.examples)})"
        )

    def _setup_logging(self):
        log_file = self.log_dir / f"grpo_dataset_{self.split}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode="w"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{self.split}]")
        self.logger.info(f"Logging to {log_file}")

    def _load_json(self, filename: str) -> dict:
        filepath = self.data_dir / filename
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.logger.info(f"Loaded {filepath}")
        return data

    def _build_few_shot_pool(self) -> dict:
        """Extract only the `exemplar_*` entries as the few-shot pool.

        Returns:
            {lang: [{"question_id": "exemplar_0", "question": ..., "answer_number": ...,
                     "answer": ...}, ...]}
            Note: the "answer" (step-by-step) field needs to come from the original
            train.json, not grpo_test.json (which dropped it). So we re-load train.json
            to recover the CoT step-by-step answers.
        """
        # grpo_test.json contains exemplars but only with answer_number, not the CoT.
        # We need to pull the CoT text from the original train.json.
        original_train = self._load_json("train.json")

        pool = {}
        for lang in self.languages:
            lang_pool = []
            for i, ex in enumerate(original_train[lang]):
                lang_pool.append({
                    "question_id": f"exemplar_{i}",
                    "question": ex["question"],
                    "answer": ex["answer"],           # step-by-step CoT in target lang
                    "answer_number": ex["answer_number"],
                })
            pool[lang] = lang_pool
            self.logger.info(f"  Few-shot pool[{lang}]: {len(lang_pool)} exemplars")

        assert all(len(pool[l]) >= self.num_few_shot for l in self.languages), (
            f"num_few_shot={self.num_few_shot} exceeds available exemplars"
        )
        return pool

    def _build_examples(self) -> list:
        """Build a flat list of (lang, example) tuples for the requested split."""
        if self.split == "train":
            source = self.grpo_train
        elif self.split == "test":
            # For test split, use only the 50 held-out problems (skip exemplars
            # which are already used as few-shot context)
            source = {
                lang: [ex for ex in self.grpo_test[lang]
                       if ex["question_id"].startswith("test_")]
                for lang in self.languages
            }
        else:
            raise ValueError(f"Unknown split: {self.split}")

        examples = []
        for lang in self.languages:
            for ex in source[lang]:
                examples.append({
                    "lang": lang,
                    "question_id": ex["question_id"],
                    "question": ex["question"],
                    "answer_number": ex["answer_number"],
                })

        self.logger.info(f"Total examples across {len(self.languages)} languages: {len(examples)}")
        return examples
    
    def _strip_field_prefix(self, text: str) -> str:
        """Remove the leading 'Question: ' or 'Step-by-Step Answer: ' prefix.

        The MGSM exemplars embed these prefixes in the data itself (in both
        English and target language). We strip everything up to and including
        the first colon, so our prompt template can add its own uniform English
        headers without duplication.

        Examples:
            "Question: Roger has 5 tennis balls..."  -> "Roger has 5 tennis balls..."
            "প্রশ্ন: রজারের 5টি টেনিস বল আছে..."       -> "রজারের 5টি টেনিস বল আছে..."
            "Step-by-Step Answer: Roger started..."  -> "Roger started..."
            "ধাপে ধাপে উত্তর: রজারের প্রথমে 5টি..."     -> "রজারের প্রথমে 5টি..."
        """
        if ":" in text:
            return text.split(":", 1)[1].strip()
        return text.strip()

    def _format_few_shot_block(self, exemplar: dict) -> str:
        """Format a single few-shot exemplar block."""
        question = self._strip_field_prefix(exemplar["question"])
        answer = self._strip_field_prefix(exemplar["answer"])
        return (
            f"Question: {question}\n"
            f"Step-by-step Answer: {answer}\n"
            f"Final Answer: {exemplar['answer_number']}"
        )

    def _build_prompt(self, lang: str, question: str, rng: random.Random) -> str:
        """Construct the full prompt with instruction + few-shot + target question."""
        # Sample `num_few_shot` exemplars from the pool for this language
        exemplars = rng.sample(self.few_shot_pool[lang], self.num_few_shot)
        few_shot_str = "\n\n".join(self._format_few_shot_block(ex) for ex in exemplars)

        question = self._strip_field_prefix(question)

        prompt = (
            f"{self.INSTRUCTION}\n\n"
            f"{few_shot_str}\n\n"
            f"Question: {question}\n"
            f"Step-by-step Answer: "
        )
        return prompt

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        """Return a single training item.

        Each __getitem__ call samples fresh few-shot exemplars (seeded by idx for
        reproducibility within an epoch).
        """
        ex = self.examples[idx]
        # Per-item RNG so few-shot sampling is deterministic given (seed, idx)
        rng = random.Random(self.seed + idx)
        prompt = self._build_prompt(ex["lang"], ex["question"], rng)

        return {
            "prompt": prompt,
            "question_id": ex["question_id"],
            "lang": ex["lang"],
            "answer_number": ex["answer_number"],
        }

    def preview(self, idx: int = 0):
        """Pretty-print a single example for sanity checking."""
        item = self[idx]
        print("=" * 80)
        print(f"lang={item['lang']}  question_id={item['question_id']}  "
              f"answer_number={item['answer_number']}")
        print("-" * 80)
        print(item["prompt"])
        print("=" * 80)


def main():
    config = {
        "data_dir": "./exp2/data",
        "log_dir": "./exp2/logs",
        "languages": ["en", "bn", "te", "th", "ru", "ja", "zh"],
        "num_few_shot": 1,
        "seed": 42,
    }

    # Sanity check on both splits
    for split in ["train", "test"]:
        dataset = GRPOMGSMDataset(config, split=split)
        print(f"\n===== Split: {split} | Total: {len(dataset)} =====")
        dataset.preview(idx=0)
        dataset.preview(idx=len(dataset) // 2)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    main()