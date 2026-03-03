"""
Visualizer for Experiment 1: Baseline Logit Lens Analysis
Generates publication-quality figures from saved activations
"""

import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import logging
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional
import json
import random


class BaselineVisualizer:
    """Generates figures from saved baseline activations"""
    
    LANGUAGE_NAMES = {
        'en': 'English',
        'zh': 'Chinese',
        'bn': 'Bengali',
        'te': 'Telugu',
        'th': 'Thai',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'ja': 'Japanese',
        'ru': 'Russian',
        'hi': 'Hindi',
        'ko': 'Korean',
    }
    
    LANGUAGE_COLORS = {
        'en': '#1f77b4',
        'zh': '#ff7f0e',
        'bn': '#2ca02c',
        'te': '#d62728',
        'th': '#9467bd',
        'es': '#8c564b',
        'fr': '#e377c2',
        'de': '#7f7f7f',
        'ja': '#bcbd22',
        'ru': '#17becf',
        'hi': '#e7ba52',
        'ko': '#ad494a',
    }
    
    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.acts_dir = Path(config['acts_dir'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_short_name = self._get_model_short_name()
        
        self.languages_order = None
        self.lang_to_idx = None
        
        self._load_language_mapping()
        self._load_tokenizer()
        
    def _get_model_short_name(self) -> str:
        """Extract short model name from full model path"""
        full_name = self.config['model_name']
        if '/' in full_name:
            return full_name.split('/')[-1].lower().replace('-', '_')
        return full_name.lower().replace('-', '_')
    
    def _load_language_mapping(self):
        """Load language order from metadata to map dimensions"""
        metadata_file = self.acts_dir / 'experiment_metadata.json'
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            self.languages_order = metadata.get('languages', [])
        
        if not self.languages_order:
            any_lang_dir = next((d for d in self.acts_dir.iterdir() if d.is_dir()), None)
            if any_lang_dir:
                lang_metadata_file = any_lang_dir / 'metadata.json'
                
                if lang_metadata_file.exists():
                    with open(lang_metadata_file, 'r') as f:
                        lang_metadata = json.load(f)
                    self.languages_order = lang_metadata.get('languages_analyzed', [])
        
        if not self.languages_order:
            raise RuntimeError("Could not find language order in metadata")
        
        self.lang_to_idx = {lang: idx for idx, lang in enumerate(self.languages_order)}
        
        self.logger.info(f"Language order from metadata: {self.languages_order}")
        self.logger.info(f"Language to index mapping: {self.lang_to_idx}")
    
    def _load_tokenizer(self):
        """Load tokenizer for decoding predicted tokens"""
        from transformers import AutoTokenizer
        
        self.logger.info(f"Loading tokenizer for {self.config['model_name']}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'],
            trust_remote_code=True
        )
        self.logger.info("Tokenizer loaded successfully")
    
    def load_language_data(self, lang: str) -> Dict[str, torch.Tensor]:
        """
        Load saved activations for a language
        
        Args:
            lang: Language code
            
        Returns:
            Dictionary with residuals, logits, lang_probs tensors
        """
        lang_dir = self.acts_dir / lang
        
        if not lang_dir.exists():
            raise FileNotFoundError(f"No data found for language {lang} at {lang_dir}")
        
        data = {
            'residuals': torch.load(lang_dir / 'residuals.pt', map_location='cpu'),
            'logits': torch.load(lang_dir / 'logits.pt', map_location='cpu'),
            'lang_probs': torch.load(lang_dir / 'lang_probs.pt', map_location='cpu'),
        }
        
        with open(lang_dir / 'metadata.json', 'r') as f:
            data['metadata'] = json.load(f)
        
        return data
    
    def load_prompts_and_answers(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Load prompts and answers that were used in analysis"""
        data_file = Path(self.config['data_dir']) / 'prompts_and_answers.json'
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data['prompts'], data['answers']
    
    def get_num_examples(self, lang: str = 'en') -> int:
        """Get number of examples available"""
        data = self.load_language_data(lang)
        return data['lang_probs'].shape[0]
    
    def decode_predicted_token(self, lang: str, example_idx: int) -> str:
        """
        Decode predicted token from logits at last layer
        
        Args:
            lang: Language code
            example_idx: Example index
            
        Returns:
            Decoded predicted token as string
        """
        data = self.load_language_data(lang)
        logits = data['logits']
        
        if example_idx >= logits.shape[0]:
            return "N/A"
        
        last_layer_logits = logits[example_idx, -1, :]
        predicted_token_id = torch.argmax(last_layer_logits).item()
        
        predicted_token = self.tokenizer.decode([predicted_token_id])
        
        return predicted_token
    
    def plot_average_language_trajectory(
        self,
        target_lang: str = 'zh',
        languages_to_plot: Optional[List[str]] = None,
        highlight_langs: Optional[List[str]] = None,
        save_name: Optional[str] = None
    ):
        """
        Plot average language probability trajectory across layers
        
        Args:
            target_lang: Target language (determines which saved data to load)
            languages_to_plot: Which languages to show (default: all)
            highlight_langs: Languages to emphasize (default: [target_lang, 'en'])
            save_name: Custom save name (default: auto-generated)
        """
        if languages_to_plot is None:
            languages_to_plot = self.languages_order
        
        if highlight_langs is None:
            highlight_langs = [target_lang, 'en']
        
        if target_lang not in highlight_langs:
            highlight_langs = [target_lang] + highlight_langs
        
        self.logger.info(f"Generating average trajectory plot for {target_lang}")
        self.logger.info(f"Languages to plot: {languages_to_plot}")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        data = self.load_language_data(target_lang)
        lang_probs = data['lang_probs']
        
        avg_probs = lang_probs.mean(dim=0)
        
        for lang in languages_to_plot:
            if lang not in self.lang_to_idx:
                self.logger.warning(f"Language {lang} not in saved data, skipping")
                continue
            
            lang_idx = self.lang_to_idx[lang]
            lang_prob_trajectory = avg_probs[:, lang_idx].numpy()
            
            layers = np.arange(len(lang_prob_trajectory))
            
            if lang in highlight_langs:
                ax.plot(
                    layers, 
                    lang_prob_trajectory * 100,
                    color=self.LANGUAGE_COLORS.get(lang, '#000000'),
                    linewidth=3.0,
                    label=self.LANGUAGE_NAMES.get(lang, lang.upper()),
                    alpha=0.9,
                    zorder=10
                )
            else:
                ax.plot(
                    layers, 
                    lang_prob_trajectory * 100,
                    color=self.LANGUAGE_COLORS.get(lang, '#000000'),
                    linewidth=1.5,
                    label=self.LANGUAGE_NAMES.get(lang, lang.upper()),
                    alpha=0.4,
                    zorder=1
                )
        
        ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
        ax.set_ylabel('Language Probability (%)', fontsize=14, fontweight='bold')
        ax.set_title(
            f'Cross-Lingual Collapse: {self.LANGUAGE_NAMES.get(target_lang, target_lang)} Input (Average)',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, len(lang_prob_trajectory) - 1)
        ax.set_ylim(0, 100)
        
        legend = ax.legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize=11,
            frameon=True,
            fancybox=True,
            shadow=True
        )
        
        for line, text in zip(legend.get_lines(), legend.get_texts()):
            lang_name = text.get_text()
            if lang_name in [self.LANGUAGE_NAMES.get(l, l) for l in highlight_langs]:
                line.set_linewidth(3.0)
                line.set_alpha(0.9)
                text.set_fontweight('bold')
        
        plt.tight_layout()
        
        if save_name is None:
            save_name = f'fig1_avg_{target_lang}_{self.model_short_name}.png'
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved figure to {save_path}")
        
        plt.close()
    
    def plot_single_example_trajectory(
        self,
        target_lang: str = 'zh',
        example_idx: int = 0,
        languages_to_plot: Optional[List[str]] = None,
        highlight_langs: Optional[List[str]] = None,
        save_name: Optional[str] = None,
        show_prompts: bool = True
    ):
        """
        Plot single example language trajectory with prediction and ground truth
        
        Args:
            target_lang: Target language to show
            example_idx: Which example to visualize
            languages_to_plot: Which languages to show (default: all)
            highlight_langs: Languages to emphasize (default: [target_lang, 'en'])
            save_name: Custom save name
            show_prompts: Whether to show prompt text boxes
        """
        if languages_to_plot is None:
            languages_to_plot = self.languages_order
        
        if highlight_langs is None:
            highlight_langs = [target_lang, 'en']
        
        if target_lang not in highlight_langs:
            highlight_langs = [target_lang] + highlight_langs
        
        self.logger.info(f"Generating single example plot for {target_lang}, example {example_idx}")
        
        prompts, answers = self.load_prompts_and_answers()
        
        english_prompt = prompts.get('en', [])[example_idx] if example_idx < len(prompts.get('en', [])) else "N/A"
        ground_truth = answers.get('en', [])[example_idx] if example_idx < len(answers.get('en', [])) else "N/A"
        
        predicted_token = self.decode_predicted_token(target_lang, example_idx)
        
        if show_prompts:
            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 0.8, 0.8, 5], hspace=0.3)
            
            ax_english = fig.add_subplot(gs[0])
            ax_english.axis('off')
            ax_english.text(
                0.02, 0.5,
                f"Prompt (English parallel): {english_prompt[:250]}...",
                fontsize=10,
                verticalalignment='center',
                wrap=True,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3)
            )
            
            ax_gt = fig.add_subplot(gs[1])
            ax_gt.axis('off')
            ax_gt.text(
                0.02, 0.5,
                f"Ground Truth Answer: {ground_truth}",
                fontsize=12,
                verticalalignment='center',
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.4)
            )
            
            ax_pred = fig.add_subplot(gs[2])
            ax_pred.axis('off')
            
            predicted_clean = predicted_token.strip()
            ground_truth_clean = str(ground_truth).strip()
            
            match = "CORRECT" if predicted_clean == ground_truth_clean else "WRONG"
            color = 'lightgreen' if match == "CORRECT" else 'lightcoral'
            symbol = "✓" if match == "CORRECT" else "✗"
            
            ax_pred.text(
                0.02, 0.5,
                f"Predicted Token (last layer): '{predicted_token}'  {symbol} {match}",
                fontsize=12,
                verticalalignment='center',
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.4)
            )
            
            ax = fig.add_subplot(gs[3])
        else:
            fig, ax = plt.subplots(figsize=(12, 7))
        
        data = self.load_language_data(target_lang)
        lang_probs = data['lang_probs']
        
        if example_idx >= lang_probs.shape[0]:
            raise ValueError(f"Example {example_idx} not available (only {lang_probs.shape[0]} examples)")
        
        example_probs = lang_probs[example_idx]
        
        for lang in languages_to_plot:
            if lang not in self.lang_to_idx:
                self.logger.warning(f"Language {lang} not in saved data, skipping")
                continue
            
            lang_idx = self.lang_to_idx[lang]
            lang_prob_trajectory = example_probs[:, lang_idx].numpy()
            
            layers = np.arange(len(lang_prob_trajectory))
            
            if lang in highlight_langs:
                ax.plot(
                    layers,
                    lang_prob_trajectory * 100,
                    color=self.LANGUAGE_COLORS.get(lang, '#000000'),
                    linewidth=3.0,
                    label=self.LANGUAGE_NAMES.get(lang, lang.upper()),
                    alpha=0.9,
                    zorder=10
                )
            else:
                ax.plot(
                    layers,
                    lang_prob_trajectory * 100,
                    color=self.LANGUAGE_COLORS.get(lang, '#000000'),
                    linewidth=1.5,
                    label=self.LANGUAGE_NAMES.get(lang, lang.upper()),
                    alpha=0.4,
                    zorder=1
                )
        
        ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
        ax.set_ylabel('Language Probability (%)', fontsize=14, fontweight='bold')
        
        title = f'{self.LANGUAGE_NAMES.get(target_lang, target_lang)} Input - Example {example_idx}'
        ax.set_title(
            title,
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, len(lang_prob_trajectory) - 1)
        ax.set_ylim(0, 100)
        
        legend = ax.legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize=11,
            frameon=True,
            fancybox=True,
            shadow=True
        )
        
        for line, text in zip(legend.get_lines(), legend.get_texts()):
            lang_name = text.get_text()
            if lang_name in [self.LANGUAGE_NAMES.get(l, l) for l in highlight_langs]:
                line.set_linewidth(3.0)
                line.set_alpha(0.9)
                text.set_fontweight('bold')
        
        if save_name is None:
            save_name = f'fig1_ex{example_idx}_{target_lang}_{self.model_short_name}.png'
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved figure to {save_path}")
        
        plt.close()
    
    def plot_multiple_examples(
        self,
        target_lang: str = 'zh',
        num_examples: int = 3,
        example_indices: Optional[List[int]] = None,
        languages_to_plot: Optional[List[str]] = None,
        highlight_langs: Optional[List[str]] = None,
        random_seed: int = 42
    ):
        """
        Generate plots for multiple examples
        
        Args:
            target_lang: Target language
            num_examples: Number of examples to plot
            example_indices: Specific indices to plot (if None, random selection)
            languages_to_plot: Which languages to show
            highlight_langs: Languages to emphasize
            random_seed: Random seed for example selection
        """
        if example_indices is None:
            random.seed(random_seed)
            total_examples = self.get_num_examples(target_lang)
            example_indices = random.sample(range(total_examples), min(num_examples, total_examples))
        
        self.logger.info(f"Generating plots for {len(example_indices)} examples: {example_indices}")
        
        for idx in example_indices:
            self.plot_single_example_trajectory(
                target_lang=target_lang,
                example_idx=idx,
                languages_to_plot=languages_to_plot,
                highlight_langs=highlight_langs
            )
    
    def plot_all_languages_grid(
        self,
        languages_to_plot: Optional[List[str]] = None,
        save_name: Optional[str] = None
    ):
        """
        Plot grid showing all languages' trajectories
        
        Args:
            languages_to_plot: Languages to include in grid
            save_name: Custom save name
        """
        if languages_to_plot is None:
            languages_to_plot = self.languages_order
        
        self.logger.info(f"Generating grid plot for languages: {languages_to_plot}")
        
        n_langs = len(languages_to_plot)
        n_cols = 3
        n_rows = (n_langs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, lang in enumerate(languages_to_plot):
            ax = axes[idx]
            
            data = self.load_language_data(lang)
            lang_probs = data['lang_probs']
            
            avg_probs = lang_probs.mean(dim=0)
            
            for i, other_lang in enumerate(self.languages_order):
                if other_lang not in languages_to_plot:
                    continue
                
                lang_prob_trajectory = avg_probs[:, i].numpy()
                layers = np.arange(len(lang_prob_trajectory))
                
                if other_lang == lang or other_lang == 'en':
                    ax.plot(
                        layers,
                        lang_prob_trajectory * 100,
                        color=self.LANGUAGE_COLORS.get(other_lang, '#000000'),
                        linewidth=2.5,
                        label=self.LANGUAGE_NAMES.get(other_lang, other_lang.upper()),
                        alpha=0.9
                    )
                else:
                    ax.plot(
                        layers,
                        lang_prob_trajectory * 100,
                        color=self.LANGUAGE_COLORS.get(other_lang, '#000000'),
                        linewidth=1.0,
                        alpha=0.3
                    )
            
            ax.set_title(f'{self.LANGUAGE_NAMES.get(lang, lang.upper())} Input', fontweight='bold')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Probability (%)')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(0, 100)
            ax.legend(fontsize=8)
        
        for idx in range(n_langs, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle(
            f'Cross-Lingual Collapse Across Languages ({self.model_short_name.upper()})',
            fontsize=18,
            fontweight='bold',
            y=0.995
        )
        
        plt.tight_layout()
        
        if save_name is None:
            save_name = f'fig1_grid_all_{self.model_short_name}.png'
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved figure to {save_path}")
        
        plt.close()


def main():
    """Generate all baseline figures"""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    config = {
        'model_name': 'Qwen/Qwen2.5-3B-Instruct',
        'languages': ['en', 'zh', 'bn', 'te', 'th', 'es', 'fr', 'de', 'ja', 'ru'],
        'acts_dir': Path('./exp1/acts_qwen2p5-3b-inst'),
        'data_dir': Path('./exp1/data'),
        'output_dir': Path('./exp1/outputs'),
    }
    
    logger.info("="*80)
    logger.info("GENERATING BASELINE FIGURES")
    logger.info("="*80)
    
    visualizer = BaselineVisualizer(config)
    
    logger.info("\n[1/5] Generating average trajectory for Chinese...")
    visualizer.plot_average_language_trajectory(target_lang='zh')
    
    logger.info("\n[2/5] Generating 3 random Chinese examples...")
    visualizer.plot_multiple_examples(target_lang='zh', num_examples=3, random_seed=42)
    
    logger.info("\n[3/5] Generating average trajectory for Spanish (subset of languages)...")
    visualizer.plot_average_language_trajectory(
        target_lang='es',
        languages_to_plot=['en', 'es', 'fr', 'de'],
        save_name=f'fig1_avg_es_subset_{visualizer.model_short_name}.png'
    )
    
    logger.info("\n[4/5] Generating specific example for Russian...")
    visualizer.plot_single_example_trajectory(target_lang='ru', example_idx=0)
    
    logger.info("\n[5/5] Generating grid of all languages...")
    visualizer.plot_all_languages_grid()
    
    logger.info("\n" + "="*80)
    logger.info("FIGURE GENERATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"Figures saved to: {config['output_dir']}")
    logger.info("="*80)


if __name__ == "__main__":
    main()