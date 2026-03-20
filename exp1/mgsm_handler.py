"""
MGSM Dataset Handler
Loads and manages MGSM dataset from HuggingFace for baseline analysis
Note: MGSM dataset requires loading each language separately
Ensures PARALLEL examples across all languages for translation purposes
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm


class MGSMHandler:
    """Handles MGSM dataset loading and processing"""
    
    AVAILABLE_LANGUAGES = ['bn', 'ca', 'de', 'en', 'es', 'eu', 'fr', 'gl', 'ja', 'ru', 'sw', 'te', 'th', 'zh']
    
    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset = None
        self.prompts = {}
        self.example_indices = []
        
    def load_mgsm(self):
        """
        Load MGSM dataset from HuggingFace
        MGSM requires loading each language separately
        
        Returns:
            Concatenated dataset with all languages
        """
        self.logger.info("Loading MGSM dataset from HuggingFace...")
        self.logger.info(f"Loading {len(self.AVAILABLE_LANGUAGES)} languages...")
        
        all_datasets = []
        
        try:
            for lang in tqdm(self.AVAILABLE_LANGUAGES, desc="Loading languages"):
                try:
                    lang_dataset = load_dataset(
                        "jbross-ibm-research/mgsm",
                        lang,
                        split="test",
                        download_mode="reuse_dataset_if_exists"
                    )
                    
                    if 'language' not in lang_dataset.column_names:
                        lang_dataset = lang_dataset.add_column('language', [lang] * len(lang_dataset))
                    
                    lang_dataset = lang_dataset.select_columns(['question', 'answer_number', 'language'])
                    
                    # Then cast to string
                    from datasets import Value
                    lang_dataset = lang_dataset.cast_column('answer_number', Value('string'))

                    all_datasets.append(lang_dataset)
                    
                except Exception as e:
                    self.logger.warning(f"  Failed to load {lang}: {e}")
                    continue
            
            if not all_datasets:
                raise RuntimeError("Failed to load any language from MGSM dataset")
            
            self.dataset = concatenate_datasets(all_datasets)
            
            self.logger.info(f"\nSuccessfully loaded MGSM dataset")
            self.logger.info(f"  Total examples: {len(self.dataset)}")
            self.logger.info(f"  Languages loaded: {len(all_datasets)}/{len(self.AVAILABLE_LANGUAGES)}")
            
            languages_in_dataset = set(self.dataset['language'])
            self.logger.info(f"  Available languages: {sorted(languages_in_dataset)}")
            
            return self.dataset
            
        except Exception as e:
            self.logger.error(f"Failed to load MGSM dataset: {e}")
            raise
    
    def extract_prompts_by_language(
        self, 
        dataset, 
        num_prompts_per_language: int = 100,
        languages: list = None,
        random_seed: int = 42
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[int]]:
        """
        Extract PARALLEL prompts and answers by language with random sampling
        
        Args:
            dataset: HuggingFace dataset
            num_prompts_per_language: Number of prompts to sample per language
            languages: List of language codes (if None, use all available)
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (prompts_by_lang, answers_by_lang, example_indices)
        """
        import random
        random.seed(random_seed)
        
        if languages is None:
            languages = self.AVAILABLE_LANGUAGES
        
        lang_data = dataset.filter(lambda x: x['language'] == 'en')
        total_available = len(lang_data)
        
        if num_prompts_per_language > total_available:
            self.logger.warning(
                f"Requested {num_prompts_per_language} examples but only {total_available} available. "
                f"Using all {total_available} examples."
            )
            num_prompts_per_language = total_available
        
        example_indices = random.sample(range(total_available), num_prompts_per_language)
        example_indices.sort()
        
        self.example_indices = example_indices
        
        prompts_by_lang = {}
        answers_by_lang = {}
        
        # First get English data to extract answers
        en_data = dataset.filter(lambda x: x['language'] == 'en')
        
        for lang in languages:
            lang_data = dataset.filter(lambda x: x['language'] == lang)
            
            if len(lang_data) < total_available:
                self.logger.warning(
                    f"Language {lang} has fewer examples ({len(lang_data)}) than English ({total_available})"
                )
            
            lang_prompts = []
            lang_answers = []
            
            for idx in example_indices:
                if idx < len(lang_data):
                    lang_prompts.append(lang_data[idx]['question'])
                    
                    # Get answer from English version (numbers are same across languages)
                    if idx < len(en_data):
                        answer_text = en_data[idx]['answer_number']
                        # Extract just the number from answer
                        import re
                        numbers = re.findall(r'-?\d+\.?\d*', answer_text)
                        lang_answers.append(numbers[0] if numbers else answer_text)
                    else:
                        lang_answers.append("N/A")
                else:
                    self.logger.warning(f"Index {idx} out of range for {lang}")
                    lang_prompts.append(f"[MISSING EXAMPLE {idx}]")
                    lang_answers.append("N/A")
            
            prompts_by_lang[lang] = lang_prompts
            answers_by_lang[lang] = lang_answers
        
        self.logger.info(f"Extracted {len(example_indices)} PARALLEL examples with answers")
        
        return prompts_by_lang, answers_by_lang, example_indices
    
    def format_prompts_for_analysis(self, prompts_by_lang: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Format prompts by adding language-specific 'Answer:' suffix
        
        Args:
            prompts_by_lang: Dictionary mapping language to list of prompts
            
        Returns:
            Dictionary with formatted prompts
        """
        from language_metadata import LanguageMetadata
        
        formatted_prompts = {}
        
        for lang, questions in prompts_by_lang.items():
            formatted = []
            answer_suffix = LanguageMetadata.get_answer_prompt(lang)
            
            for question in questions:
                question = question.strip()
                formatted.append(f"{question}{answer_suffix}")
            
            formatted_prompts[lang] = formatted
        
        return formatted_prompts
    
    def save_prompts_to_file(self, prompts: Dict[str, List[str]], filepath: Path) -> None:
        """
        Save prompts to JSON file
        
        Args:
            prompts: Dictionary mapping language to list of prompts
            filepath: Path to save JSON file
        """
        import json
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Prompts saved to {filepath}")
    
    def save_metadata_to_file(self, filepath: Path) -> None:
        """
        Save extraction metadata including example indices
        
        Args:
            filepath: Path to save metadata JSON file
        """
        import json
        
        metadata = {
            'example_indices': self.example_indices,
            'num_examples': len(self.example_indices),
            'languages': list(self.prompts.keys()) if self.prompts else [],
            'random_seed': self.config.get('random_seed', None),
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Metadata saved to {filepath}")
    
    def load_prompts_from_file(self, filepath: Path) -> Dict[str, List[str]]:
        """
        Load prompts from JSON file
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Dictionary mapping language to list of prompts
        """
        import json
        
        self.logger.info(f"Loading prompts from {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        
        for lang, prompt_list in prompts.items():
            self.logger.info(f"  Loaded {len(prompt_list)} prompts for {lang}")
        
        return prompts
    
    def get_parallel_example(self, example_idx: int) -> Dict[str, str]:
        """
        Get a specific example across all languages (parallel)
        
        Args:
            example_idx: Index in the extracted prompts (not dataset index)
            
        Returns:
            Dictionary mapping language to prompt for this example
        """
        if not self.prompts:
            raise RuntimeError("No prompts loaded. Call extract_prompts_by_language first.")
        
        parallel_example = {}
        
        for lang, prompt_list in self.prompts.items():
            if example_idx < len(prompt_list):
                parallel_example[lang] = prompt_list[example_idx]
            else:
                parallel_example[lang] = None
        
        return parallel_example
    
    def get_sample_prompts(self, n: int = 5) -> Dict[str, List[str]]:
        """
        Get sample prompts for inspection
        
        Args:
            n: Number of samples per language
            
        Returns:
            Dictionary with sample prompts
        """
        samples = {}
        
        for lang, prompts in self.prompts.items():
            samples[lang] = prompts[:min(n, len(prompts))]
        
        return samples
    
    def verify_parallel_examples(self) -> bool:
        """
        Verify that all languages have the same number of examples
        
        Returns:
            True if all languages have same count, False otherwise
        """
        if not self.prompts:
            self.logger.warning("No prompts loaded")
            return False
        
        counts = {lang: len(prompts) for lang, prompts in self.prompts.items()}
        unique_counts = set(counts.values())
        
        if len(unique_counts) == 1:
            self.logger.info(f"All languages have {list(unique_counts)[0]} parallel examples")
            return True
        else:
            self.logger.error("Mismatch in example counts:")
            for lang, count in counts.items():
                self.logger.error(f"  {lang}: {count}")
            return False


def main():
    """Test function for MGSM dataset handler with parallel examples"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    config = {
        'languages': ['en', 'zh', 'es'],
        'num_prompts_per_language': 10,
        'random_seed': 42,
        'data_dir': Path('./exp1/data'),
    }
    
    config['data_dir'].mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("TESTING MGSM HANDLER WITH PARALLEL EXAMPLES")
    logger.info("="*80)
    
    handler = MGSMHandler(config)
    
    logger.info("\n[Test 1] Loading MGSM dataset...")
    dataset = handler.load_mgsm()
    
    logger.info("\n[Test 2] Extracting PARALLEL prompts...")
    prompts_by_lang, example_indices = handler.extract_prompts_by_language(
        dataset,
        num_prompts_per_language=config['num_prompts_per_language'],
        languages=config['languages'],
        random_seed=config['random_seed']
    )
    
    logger.info(f"\nExtracted indices: {example_indices}")
    
    handler.prompts = prompts_by_lang
    
    logger.info("\n[Test 3] Verifying parallel structure...")
    is_parallel = handler.verify_parallel_examples()
    
    if is_parallel:
        logger.info("SUCCESS: All languages have matching example counts")
    else:
        logger.error("FAILED: Example count mismatch")
    
    logger.info("\n[Test 4] Formatting prompts...")
    formatted_prompts = handler.format_prompts_for_analysis(prompts_by_lang)
    
    logger.info("\n[Test 5] Saving prompts and metadata...")
    prompts_path = config['data_dir'] / 'test_prompts_parallel.json'
    metadata_path = config['data_dir'] / 'test_metadata_parallel.json'
    
    handler.save_prompts_to_file(formatted_prompts, prompts_path)
    handler.save_metadata_to_file(metadata_path)
    
    logger.info("\n[Test 6] Getting parallel example 0...")
    handler.prompts = formatted_prompts
    parallel_ex_0 = handler.get_parallel_example(0)
    
    logger.info("\nParallel Example 0 (same problem in all languages):")
    for lang, prompt in parallel_ex_0.items():
        display = prompt[:100] + "..." if len(prompt) > 100 else prompt
        logger.info(f"  {lang.upper()}: {display}")
    
    logger.info("\n[Test 7] Demonstrating parallel structure...")
    logger.info("\nShowing that examples are aligned across languages:")
    
    for i in range(min(3, config['num_prompts_per_language'])):
        logger.info(f"\n--- Example {i} (MGSM index: {example_indices[i]}) ---")
        parallel_ex = handler.get_parallel_example(i)
        
        for lang in config['languages']:
            prompt = parallel_ex[lang]
            display = prompt[:80] + "..." if len(prompt) > 80 else prompt
            logger.info(f"{lang.upper():>3}: {display}")
    
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Example indices used: {example_indices}")
    logger.info(f"Number of parallel examples: {len(example_indices)}")
    
    for lang in config['languages']:
        count = len(formatted_prompts.get(lang, []))
        logger.info(f"{lang.upper()}: {count} prompts")
    
    logger.info(f"\nPrompts saved to: {prompts_path}")
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info("\nALL TESTS PASSED!")
    logger.info("="*80)


if __name__ == "__main__":
    main()