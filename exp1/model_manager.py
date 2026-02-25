"""
Model management for TransformerLens
Handles model loading and forward passes
"""

import torch
import logging
from transformer_lens import HookedTransformer
from typing import Dict, Tuple


class ModelManager:
    """Manages model loading and forward passes with caching"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        
    def load_model(self) -> None:
        """Load the base model using TransformerLens"""
        self.logger.info(f"Loading model: {self.config['model_name']}")
        
        try:
            # Convert dtype string to torch dtype
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16
            }
            dtype = dtype_map.get(self.config['dtype'], torch.bfloat16)
            
            self.model = HookedTransformer.from_pretrained(
                self.config['model_name'],
                device=self.config['device'],
                dtype=dtype
            )
            
            self.logger.info(f"✓ Model loaded successfully")
            self.logger.info(f"  Layers: {self.model.cfg.n_layers}")
            self.logger.info(f"  Hidden size: {self.model.cfg.d_model}")
            self.logger.info(f"  Vocab size: {self.model.cfg.d_vocab}")
            
            # Verify layer count
            if self.model.cfg.n_layers != self.config['num_layers']:
                self.logger.warning(
                    f"Model has {self.model.cfg.n_layers} layers, "
                    f"config expects {self.config['num_layers']}. "
                    f"Updating config."
                )
                self.config['num_layers'] = self.model.cfg.n_layers
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def run_with_cache(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Run forward pass and return logits + activation cache
        
        Args:
            tokens: Input token IDs [batch_size, seq_len]
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            cache: Dictionary of cached activations
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(tokens)
        
        return logits, cache
    
    def tokenize(self, text: str) -> torch.Tensor:
        """
        Tokenize input text
        
        Args:
            text: Input string
            
        Returns:
            tokens: Token IDs [1, seq_len]
        """
        return self.model.to_tokens(text)
    
    def get_tokenizer(self):
        """Get the underlying tokenizer"""
        return self.model.tokenizer
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.model.cfg.d_vocab