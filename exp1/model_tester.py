"""
Model Testing Script
Tests model generation on multilingual MGSM examples
"""

import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import logging
from pathlib import Path
import torch
import json
from datetime import datetime
from typing import Dict, List
import random

from model_manager import ModelManager
from mgsm_handler import MGSMHandler


def test_model_generation(config: dict):
    """
    Test model generation on random MGSM examples
    
    Args:
        config: Configuration dictionary containing:
            - model_name: HuggingFace model name
            - languages: List of language codes to test
            - num_examples_per_lang: K - number of examples per language
            - max_new_tokens: T - maximum tokens to generate
            - random_seed: Random seed for reproducibility
            - output_file: Output JSON file path
    """
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("MODEL GENERATION TEST")
    logger.info("="*80)
    logger.info(f"Model: {config['model_name']}")
    logger.info(f"Languages: {config['languages']}")
    logger.info(f"Examples per language (K): {config['num_examples_per_lang']}")
    logger.info(f"Max new tokens (T): {config['max_new_tokens']}")
    logger.info("="*80)
    
    # Step 1: Load MGSM dataset
    logger.info("\n[Step 1] Loading MGSM dataset...")
    mgsm_handler = MGSMHandler(config)
    dataset = mgsm_handler.load_mgsm()
    
    # Step 2: Extract parallel examples
    logger.info(f"\n[Step 2] Extracting {config['num_examples_per_lang']} examples per language...")
    prompts_by_lang, answers_by_lang, example_indices = mgsm_handler.extract_prompts_by_language(
        dataset,
        num_prompts_per_language=config['num_examples_per_lang'],
        languages=config['languages'],
        random_seed=config['random_seed']
    )
    
    # Format prompts
    formatted_prompts = mgsm_handler.format_prompts_for_analysis(prompts_by_lang)
    
    # Step 3: Load model
    logger.info("\n[Step 3] Loading model...")
    model_manager = ModelManager(config)
    model_manager.load_model()
    tokenizer = model_manager.get_tokenizer()
    
    # Step 4: Run generation tests
    logger.info(f"\n[Step 4] Running generation on {len(config['languages']) * config['num_examples_per_lang']} examples...")
    
    results = []
    
    for lang in config['languages']:
        logger.info(f"\n  Testing {lang.upper()}...")
        
        for idx in range(config['num_examples_per_lang']):
            input_text = formatted_prompts[lang][idx]
            input_en = formatted_prompts['en'][idx]
            ground_truth = answers_by_lang['en'][idx]
            
            logger.info(f"    Example {idx + 1}/{config['num_examples_per_lang']}")
            
            # Tokenize input
            input_tokens = model_manager.tokenize(input_text)
            input_length = input_tokens.shape[1]
            
            # Generate tokens
            with torch.no_grad():
                output_tokens = model_manager.model.generate(
                    input_tokens,
                    max_new_tokens=config['max_new_tokens'],
                    temperature=0.0,  # Greedy decoding (temperature=0)
                    stop_at_eos=True
                )
            
            # Extract generated tokens (excluding input)
            generated_tokens = output_tokens[0, input_length:]
            
            # Decode first generated token
            if len(generated_tokens) > 0:
                first_token_id = generated_tokens[0].item()
                first_token = tokenizer.decode([first_token_id])
            else:
                first_token = "<no generation>"
                first_token_id = None
            
            # Decode full output
            full_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Store result
            result = {
                'language': lang,
                'example_index': idx,
                'mgsm_index': example_indices[idx],
                'input': input_text,
                'input_en': input_en,
                'output': full_output,
                'ground_truth': ground_truth,
                'first_token': first_token,
                'first_token_id': first_token_id,
                'num_tokens_generated': len(generated_tokens),
            }
            
            results.append(result)
            
            # Log preview
            logger.info(f"      Input: {input_text[:60]}...")
            logger.info(f"      First token: '{first_token}'")
            logger.info(f"      Output: {full_output[:60]}...")
            logger.info(f"      Ground truth: {ground_truth}")
    
    # Step 5: Save results
    logger.info(f"\n[Step 5] Saving results to {config['output_file']}...")
    
    output_data = {
        'metadata': {
            'model': config['model_name'],
            'timestamp': datetime.now().isoformat(),
            'languages': config['languages'],
            'num_examples_per_lang': config['num_examples_per_lang'],
            'max_new_tokens': config['max_new_tokens'],
            'random_seed': config['random_seed'],
            'total_examples': len(results),
        },
        'results': results
    }
    
    output_path = Path(config['output_file'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ Results saved to {output_path}")
    
    # Step 6: Summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    for lang in config['languages']:
        lang_results = [r for r in results if r['language'] == lang]
        logger.info(f"\n{lang.upper()}:")
        logger.info(f"  Examples tested: {len(lang_results)}")
        
        # Check first tokens
        first_tokens = [r['first_token'] for r in lang_results]
        unique_first_tokens = set(first_tokens)
        logger.info(f"  Unique first tokens: {unique_first_tokens}")
        
        # Show one example
        if lang_results:
            example = lang_results[0]
            logger.info(f"  Sample output: {example['output'][:80]}...")
    
    logger.info("\n" + "="*80)
    logger.info(f"✅ Test complete! Results saved to {config['output_file']}")
    logger.info("="*80)
    
    return results


def main():
    """Main function with config dictionary"""
    
    config = {
        # Model settings
        'model_name': 'Qwen/Qwen2.5-3B-Instruct',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dtype': 'bfloat16',
        'num_layers': 36,
        
        # Test parameters
        'languages': ['en', 'zh', 'bn', 'te', 'th', 'es', 'fr', 'de', 'ja', 'ru'],
        'num_examples_per_lang': 10,  # K parameter
        'max_new_tokens': 20,  # T parameter
        'random_seed': 42,
        
        # Output
        'output_file': './exp1/outputs/qwen2p5-3b-inst_test_results.json',
    }
    
    test_model_generation(config)


if __name__ == "__main__":
    main()