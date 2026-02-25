"""
Logit Lens Analyzer
Computes and saves layer-wise activations and language probabilities at last token position
Language-agnostic: works with any set of languages specified in config
"""

import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
import numpy as np


class LogitLensAnalyzer:
    """
    Analyzes layer-wise representations using logit lens
    Saves activations for later analysis
    """
    
    def __init__(self, config: dict, model_manager, language_detector):
        """
        Args:
            config: Configuration dictionary (must contain 'languages' key)
            model_manager: ModelManager instance
            language_detector: LanguageDetector instance
        """
        self.config = config
        self.model_manager = model_manager
        self.language_detector = language_detector
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Get languages from config
        self.languages = config['languages']
        self.num_languages = len(self.languages)
        
        self.logger.info(f"Initialized analyzer for languages: {self.languages}")
        
        # Create activations directory
        self.acts_dir = config['acts_dir']
        self.acts_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_prompts_by_language(
        self, 
        prompts: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Analyze prompts for all languages and save activations
        
        Args:
            prompts: Dictionary mapping language to list of prompts
            
        Returns:
            Dictionary mapping language to dict of tensors:
                'residuals': [N, L, d_model]
                'logits': [N, L, vocab_size]
                'lang_probs': [N, L, num_languages]
        """
        results = {}
        
        for lang in self.languages:
            if lang not in prompts or not prompts[lang]:
                self.logger.warning(f"No prompts for {lang}, skipping")
                continue
            
            self.logger.info(f"\nAnalyzing {lang.upper()} prompts...")
            
            lang_results = self._analyze_single_language(
                prompts[lang],
                lang
            )
            
            results[lang] = lang_results
            
            # Save to disk
            self._save_language_results(lang, lang_results, prompts[lang])
        
        return results
    
    def _analyze_single_language(
        self,
        prompts: List[str],
        lang: str
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze all prompts for a single language
        
        Args:
            prompts: List of prompts
            lang: Language code
            
        Returns:
            Dictionary with tensors:
                'residuals': [N, L, d_model]
                'logits': [N, L, vocab_size]
                'lang_probs': [N, L, num_languages]
        """
        N = len(prompts)
        L = self.config['num_layers']
        d_model = self.model_manager.model.cfg.d_model
        vocab_size = self.model_manager.model.cfg.d_vocab
        num_languages = len(self.config['languages'])
        
        # Initialize storage tensors
        all_residuals = torch.zeros(N, L, d_model)
        all_logits = torch.zeros(N, L, vocab_size)
        all_lang_probs = torch.zeros(N, L, num_languages)
        
        # Process each prompt with progress bar
        for i, prompt in enumerate(tqdm(
            prompts,
            desc=f"  {lang.upper()}",
            leave=False,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )):
            residuals, logits, lang_probs = self._analyze_single_prompt(prompt)
            
            # Convert to torch tensors if they're numpy arrays
            if isinstance(residuals, np.ndarray):
                residuals = torch.from_numpy(residuals)
            if isinstance(logits, np.ndarray):
                logits = torch.from_numpy(logits)
            if isinstance(lang_probs, np.ndarray):
                lang_probs = torch.from_numpy(lang_probs)
            
            all_residuals[i] = residuals
            all_logits[i] = logits
            all_lang_probs[i] = lang_probs
        
        return {
            'residuals': all_residuals,
            'logits': all_logits,
            'lang_probs': all_lang_probs,
            'language': lang,
            'num_examples': N
        }
    
    def _analyze_single_prompt(
        self,
        prompt: str
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Analyze a single prompt at last token position across all layers
        
        Args:
            prompt: Input text
            
        Returns:
            residuals: [L, d_model] - residual streams at each layer
            logits: [L, vocab_size] - logits at each layer
            lang_probs: [L, num_languages] - language probabilities at each layer
        """
        # Tokenize
        tokens = self.model_manager.tokenize(prompt)
        
        num_layers = self.config['num_layers']
        d_model = self.model_manager.model.cfg.d_model
        vocab_size = self.model_manager.model.cfg.d_vocab
        
        # Initialize storage
        residuals = torch.zeros(num_layers, d_model)
        logits_all = torch.zeros(num_layers, vocab_size)
        lang_probs = np.zeros((num_layers, self.num_languages))
        
        # Get activations at each layer
        with torch.no_grad():
            _, cache = self.model_manager.run_with_cache(tokens)
            
            for layer_idx in range(num_layers):
                # Extract residual stream at this layer (last token position)
                resid_key = f'blocks.{layer_idx}.hook_resid_post'
                resid = cache[resid_key][0, -1, :].cpu()  # [d_model]
                
                # Apply logit lens
                layer_logits = self._apply_logit_lens(resid.unsqueeze(0).unsqueeze(0))
                layer_logits = layer_logits[0, -1, :].cpu()  # [vocab_size]
                
                # Get language probabilities for all configured languages
                lang_dist = self.language_detector.get_language_distribution(
                    layer_logits.unsqueeze(0).unsqueeze(0),
                    position=-1
                )
                
                # Store probabilities for each configured language
                for lang_idx, lang_code in enumerate(self.languages):
                    lang_probs[layer_idx, lang_idx] = lang_dist.get(lang_code, 0.0)
                
                # Store activations
                residuals[layer_idx] = resid
                logits_all[layer_idx] = layer_logits
        
        return residuals, logits_all, lang_probs
    
    def _apply_logit_lens(self, residual_stream: torch.Tensor) -> torch.Tensor:
        """
        Apply logit lens: LayerNorm -> Unembed
        
        Args:
            residual_stream: [batch_size, seq_len, d_model]
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        model = self.model_manager.model
        
        # Move to model device
        device = next(model.parameters()).device
        residual_stream = residual_stream.to(device)
        
        # Apply final layer norm
        normalized = model.ln_final(residual_stream)
        
        # Apply unembedding
        logits = model.unembed(normalized)
        
        return logits
    
    def _save_language_results(
        self,
        lang: str,
        results: Dict[str, torch.Tensor],
        prompts: List[str]
    ) -> None:
        """
        Save results for a single language
        
        Args:
            lang: Language code
            results: Dictionary with residuals, logits, lang_probs tensors
            prompts: List of prompts used
        """
        # Create language directory
        lang_dir = self.acts_dir / lang
        lang_dir.mkdir(exist_ok=True)
        
        # Save tensors
        torch.save(results['residuals'], lang_dir / 'residuals.pt')
        torch.save(results['logits'], lang_dir / 'logits.pt')
        torch.save(results['lang_probs'], lang_dir / 'lang_probs.pt')
        
        # Save metadata
        metadata = {
            'language': lang,
            'num_examples': len(prompts),
            'num_layers': self.config['num_layers'],
            'd_model': self.model_manager.model.cfg.d_model,
            'vocab_size': self.model_manager.model.cfg.d_vocab,
            'model_name': self.config['model_name'],
            'languages_analyzed': self.languages,  # Save which languages were analyzed
            'num_languages': self.num_languages,
            'prompts': prompts[:10],  # Save first 10 as samples
            'shapes': {
                'residuals': list(results['residuals'].shape),
                'logits': list(results['logits'].shape),
                'lang_probs': list(results['lang_probs'].shape)
            }
        }
        
        with open(lang_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"✓ Saved {lang} results to {lang_dir}")
        self.logger.info(f"  Residuals: {results['residuals'].shape}")
        self.logger.info(f"  Logits: {results['logits'].shape}")
        self.logger.info(f"  Lang probs: {results['lang_probs'].shape}")
    
    def load_language_results(self, lang: str) -> Tuple[Dict[str, torch.Tensor], dict]:
        """
        Load saved results for a language
        
        Args:
            lang: Language code
            
        Returns:
            results: Dictionary with residuals, logits, lang_probs tensors
            metadata: Metadata dictionary
        """
        lang_dir = self.acts_dir / lang
        
        if not lang_dir.exists():
            raise FileNotFoundError(f"No saved results for {lang} at {lang_dir}")
        
        self.logger.info(f"Loading {lang} results from {lang_dir}")
        
        results = {
            'residuals': torch.load(lang_dir / 'residuals.pt'),
            'logits': torch.load(lang_dir / 'logits.pt'),
            'lang_probs': torch.load(lang_dir / 'lang_probs.pt')
        }
        
        # Load metadata
        with open(lang_dir / 'metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.logger.info(f"  Loaded {metadata['num_examples']} examples")
        self.logger.info(f"  Languages in data: {metadata['languages_analyzed']}")
        
        return results, metadata
    
    def compute_averaged_lang_probs(
        self,
        results: Dict[str, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and std of language probabilities across examples
        
        Args:
            results: Dictionary with 'lang_probs' tensor [N, L, num_languages]
            
        Returns:
            mean_probs: [L, num_languages] - mean language probability per layer
            std_probs: [L, num_languages] - std language probability per layer
        """
        lang_probs = results['lang_probs'].numpy()  # [N, L, num_languages]
        
        mean_probs = np.mean(lang_probs, axis=0)  # [L, num_languages]
        std_probs = np.std(lang_probs, axis=0)    # [L, num_languages]
        
        return mean_probs, std_probs
    
    def debug_top_k_tokens(
        self,
        prompt: str,
        layers_to_debug: list = [0, 15, 31],
        k: int = 10
    ) -> None:
        """
        Debug method: Print top-k tokens at specific layers
        
        Args:
            prompt: Input text to analyze
            layers_to_debug: Which layers to inspect
            k: How many top tokens to show
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"DEBUG: Top-{k} tokens analysis")
        self.logger.info(f"Prompt: {prompt}")
        self.logger.info(f"{'='*80}")
        
        # Tokenize
        tokens = self.model_manager.tokenize(prompt)
        tokenizer = self.model_manager.get_tokenizer()
        
        self.logger.info(f"\nTokenized input: {tokenizer.decode(tokens[0])}")
        self.logger.info(f"Number of tokens: {tokens.shape[1]}")
        
        # Run forward pass
        with torch.no_grad():
            _, cache = self.model_manager.run_with_cache(tokens)
            
            for layer_idx in layers_to_debug:
                if layer_idx >= self.config['num_layers']:
                    continue
                
                self.logger.info(f"\n{'-'*80}")
                self.logger.info(f"Layer {layer_idx} (last token position)")
                self.logger.info(f"{'-'*80}")
                
                # Extract residual stream at this layer (last token position)
                resid_key = f'blocks.{layer_idx}.hook_resid_post'
                resid = cache[resid_key][0, -1, :].cpu()  # [d_model]
                
                # Apply logit lens
                layer_logits = self._apply_logit_lens(resid.unsqueeze(0).unsqueeze(0))
                layer_logits = layer_logits[0, -1, :].cpu()  # [vocab_size]
                
                # Get probabilities
                probs = torch.softmax(layer_logits, dim=-1)
                
                # Get top-k
                top_probs, top_indices = torch.topk(probs, k=k)
                
                # Print top-k tokens with their language classification
                self.logger.info(f"\nTop-{k} predicted tokens:")
                self.logger.info(f"{'Rank':<6} {'Token':<20} {'Prob':<10} {'Language':<10}")
                self.logger.info(f"{'-'*50}")
                
                for rank, (prob, token_id) in enumerate(zip(top_probs, top_indices), 1):
                    token_id_int = int(token_id.item())
                    prob_float = float(prob.item())
                    
                    # Decode token
                    token_text = tokenizer.decode([token_id_int])
                    # Clean up display (replace newlines, tabs)
                    token_text_display = token_text.replace('\n', '\\n').replace('\t', '\\t')
                    if len(token_text_display) > 18:
                        token_text_display = token_text_display[:15] + '...'
                    
                    # Find language
                    token_lang = 'other'
                    for lang, token_set in self.language_detector.language_tokens.items():
                        if token_id_int in token_set:
                            token_lang = lang
                            break
                    
                    self.logger.info(
                        f"{rank:<6} '{token_text_display}':<20 {prob_float:<10.4f} {token_lang:<10}"
                    )
                
                # Aggregate by language
                lang_probs = self.language_detector.get_language_distribution(
                    layer_logits.unsqueeze(0).unsqueeze(0),
                    position=-1
                )
                
                self.logger.info(f"\nAggregated language probabilities:")
                for lang in self.config['languages'] + ['other']:
                    prob = lang_probs.get(lang, 0.0)
                    self.logger.info(f"  {lang.upper():<10}: {prob:.4f} ({prob*100:.2f}%)")
        
        self.logger.info(f"\n{'='*80}\n")


def main():
    """Test function for logit lens analyzer with debugging"""
    
    import logging
    from model_manager import ModelManager
    from language_detector import LanguageDetector
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Check GPU
    if not torch.cuda.is_available():
        logger.warning("GPU not available! This will be very slow.")
    
    # Configuration
    config = {
        'model_name': 'Qwen/Qwen3-4B',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dtype': 'bfloat16',
        'languages': ['en', 'zh', 'sw'],
        'num_layers': 32,
        'top_k_tokens': 50,
        'acts_dir': Path('./exp1/acts_qwen3-4b'),
        'data_dir': Path('./exp1/data'),
    }
    
    logger.info("="*80)
    logger.info("TESTING LOGIT LENS ANALYZER - DEBUG MODE")
    logger.info(f"Configured languages: {config['languages']}")
    logger.info("="*80)
    
    # Test prompts
    test_prompts = {
        'en': "What is 2 + 2? Answer:",
        'zh': "2加2等于多少？答案：",
        'sw': "2 + 2 ni nini? Jibu:",
    }
    
    # Initialize components
    logger.info("\n[1/3] Loading model...")
    model_manager = ModelManager(config)
    model_manager.load_model()
    
    logger.info("\n[2/3] Initializing language detector...")
    language_detector = LanguageDetector(config, model_manager)
    language_detector.initialize_language_tokens()
    
    logger.info("\n[3/3] Running debug analysis...")
    analyzer = LogitLensAnalyzer(config, model_manager, language_detector)
    
    # Debug each language prompt
    for lang, prompt in test_prompts.items():
        logger.info(f"\n{'#'*80}")
        logger.info(f"ANALYZING {lang.upper()} PROMPT")
        logger.info(f"{'#'*80}")
        
        analyzer.debug_top_k_tokens(
            prompt=prompt,
            layers_to_debug=[0, 18, 35],  # Early, middle, late layers
            k=15  # Show top-15 tokens
        )
    
    logger.info("\n" + "="*80)
    logger.info("✅ Debug analysis complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()

