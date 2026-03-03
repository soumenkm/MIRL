"""
Main script for Experiment 1: Baseline logit lens analysis
Analyzes multilingual prompts and saves layer-wise activations to disk
"""

import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import logging
from pathlib import Path
import torch
from tqdm import tqdm
import json
from datetime import datetime

from model_manager import ModelManager
from language_detector import LanguageDetector
from logit_lens_analyzer import LogitLensAnalyzer
from mgsm_handler import MGSMHandler


def setup_logging(log_dir: Path):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """Main execution function"""
    
    config = {
        'model_name': 'Qwen/Qwen2.5-3B-Instruct',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dtype': 'bfloat16',
        'languages': ['en', 'zh', 'bn', 'te', 'th', 'es', 'fr', 'de', 'ja', 'ru'],
        'num_examples_per_language': 10,
        'random_seed': 42,
        'num_layers': 36,
        'top_k_tokens': 50,
        'acts_dir': Path('./exp1/acts_qwen2p5-3b-inst'),
        'data_dir': Path('./exp1/data'),
        'log_dir': Path('./exp1/logs'),
        'output_dir': Path('./exp1/outputs'),
    }
    
    for dir_path in [config['acts_dir'], config['data_dir'], 
                     config['log_dir'], config['output_dir']]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(config['log_dir'])
    
    logger.info("="*80)
    logger.info("EXPERIMENT 1: BASELINE LOGIT LENS ANALYSIS")
    logger.info("="*80)
    logger.info(f"Model: {config['model_name']}")
    logger.info(f"Languages: {config['languages']}")
    logger.info(f"Examples per language: {config['num_examples_per_language']}")
    logger.info(f"Device: {config['device']}")
    logger.info(f"Activation directory: {config['acts_dir']}")
    logger.info("="*80)
    
    if not torch.cuda.is_available():
        logger.warning("GPU not available! Analysis will be slow on CPU.")
    else:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Loading MGSM Dataset")
    logger.info("="*80)
    
    mgsm_handler = MGSMHandler(config)
    
    logger.info("Loading MGSM dataset from HuggingFace...")
    mgsm_dataset = mgsm_handler.load_mgsm()
    logger.info(f"Loaded {len(mgsm_dataset)} total examples")
    
    logger.info(f"\nExtracting {config['num_examples_per_language']} PARALLEL prompts per language...")
    prompts_by_lang, answers_by_lang, example_indices = mgsm_handler.extract_prompts_by_language(
        mgsm_dataset,
        num_prompts_per_language=config['num_examples_per_language'],
        languages=config['languages'],
        random_seed=config['random_seed']
    )
    
    logger.info(f"\nParallel example indices: {example_indices[:10]}{'...' if len(example_indices) > 10 else ''}")
    
    logger.info("\nDataset statistics:")
    for lang, prompts in prompts_by_lang.items():
        logger.info(f"  {lang.upper()}: {len(prompts)} prompts")
    
    logger.info("\nGround truth answers (first 3 examples):")
    for i in range(min(3, len(example_indices))):
        logger.info(f"  Example {i}: {answers_by_lang['en'][i]}")
    
    logger.info("\nFormatting prompts with answer markers...")
    formatted_prompts = mgsm_handler.format_prompts_for_analysis(prompts_by_lang)
    
    logger.info("\nSample prompts (first example from each language):")
    for lang in config['languages']:
        if lang in formatted_prompts and formatted_prompts[lang]:
            sample = formatted_prompts[lang][0]
            display = sample[:100] + "..." if len(sample) > 100 else sample
            logger.info(f"  {lang.upper()}: {display}")
    
    data_file = config['data_dir'] / 'prompts_and_answers.json'
    combined_data = {
        'prompts': formatted_prompts,
        'answers': answers_by_lang
    }
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved prompts and answers to {data_file}")
    
    metadata_prompts_file = config['data_dir'] / 'prompts_metadata.json'
    prompts_metadata = {
        'example_indices': example_indices,
        'num_examples': len(example_indices),
        'languages': config['languages'],
        'random_seed': config['random_seed'],
        'dataset': 'MGSM',
        'data_file': str(data_file),
    }
    with open(metadata_prompts_file, 'w') as f:
        json.dump(prompts_metadata, f, indent=2)
    logger.info(f"Saved prompts metadata to {metadata_prompts_file}")
    
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Loading Model")
    logger.info("="*80)
    
    model_manager = ModelManager(config)
    model_manager.load_model()
    
    config['num_layers'] = model_manager.model.cfg.n_layers
    logger.info(f"Model loaded with {config['num_layers']} layers")
    
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Initializing Language Detector")
    logger.info("="*80)
    
    language_detector = LanguageDetector(config, model_manager)
    language_detector.initialize_language_tokens()
    
    logger.info("Language detector initialized")
    
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Initializing Logit Lens Analyzer")
    logger.info("="*80)
    
    analyzer = LogitLensAnalyzer(config, model_manager, language_detector)
    logger.info("Logit lens analyzer initialized")
    
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Running Logit Lens Analysis")
    logger.info("="*80)
    logger.info(f"Total languages to process: {len(config['languages'])}")
    logger.info(f"Examples per language: {config['num_examples_per_language']}")
    logger.info(f"Total examples: {len(config['languages']) * config['num_examples_per_language']}")
    logger.info("")
    
    logger.info("Processing all languages...")
    results = analyzer.analyze_prompts_by_language(formatted_prompts)
    
    logger.info("\nAnalysis complete for all languages:")
    for lang in results.keys():
        logger.info(f"  {lang.upper()}: Activations saved to {config['acts_dir'] / lang}")
    
    logger.info("\n" + "="*80)
    logger.info("STEP 6: Saving Experiment Metadata")
    logger.info("="*80)
    
    metadata = {
        'experiment': 'baseline_logit_lens',
        'timestamp': datetime.now().isoformat(),
        'model': config['model_name'],
        'num_layers': config['num_layers'],
        'languages': config['languages'],
        'num_examples_per_language': config['num_examples_per_language'],
        'example_indices': example_indices,
        'random_seed': config['random_seed'],
        'dataset': 'MGSM',
        'activation_directory': str(config['acts_dir']),
        'data_file': str(data_file),
        'prompts_metadata_file': str(metadata_prompts_file),
    }
    
    metadata_file = config['acts_dir'] / 'experiment_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved experiment metadata to {metadata_file}")
    
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETE!")
    logger.info("="*80)
    logger.info(f"Total languages processed: {len(results)}")
    logger.info(f"Examples per language: {config['num_examples_per_language']}")
    logger.info(f"Total activations saved: {len(results) * config['num_examples_per_language']}")
    logger.info(f"\nParallel structure verified:")
    logger.info(f"  All languages use same {len(example_indices)} example indices")
    logger.info(f"  Example 0 in all languages = MGSM index {example_indices[0]}")
    logger.info(f"  Example 1 in all languages = MGSM index {example_indices[1]}")
    logger.info(f"\nResults directory structure:")
    for lang in results.keys():
        logger.info(f"  {config['acts_dir']}/{lang}/")
        logger.info(f"    - residuals.pt")
        logger.info(f"    - logits.pt")
        logger.info(f"    - lang_probs.pt")
        logger.info(f"    - metadata.json")
    logger.info(f"\nAll activations saved to: {config['acts_dir']}")
    logger.info(f"Data (prompts + answers) saved to: {data_file}")
    logger.info(f"Prompts metadata saved to: {metadata_prompts_file}")
    logger.info("="*80)
    
    return results


if __name__ == "__main__":
    main()