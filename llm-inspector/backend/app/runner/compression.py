"""
Prompt Compression Module using LLMLingua principles.
Provides lossless-like compression to save token budget while maintaining semantics.

Reference: Jiang et al. (2023) LLMLingua
"""
import re
import math
from typing import List
from app.core.logging import get_logger

logger = get_logger(__name__)

class PromptCompressor:
    def __init__(self, target_ratio: float = 0.8):
        """
        Initialize the prompt compressor.
        
        Args:
            target_ratio: Target compression ratio (0.8 = keep 80% of tokens)
        """
        self.target_ratio = target_ratio
        
        # Stop words to potentially drop if aggressive compression needed
        self.stop_words = {
            "a", "an", "the", "and", "or", "but", "if", "because", 
            "as", "until", "while", "of", "at", "by", "for", "with", 
            "about", "against", "between", "into", "through", "during", 
            "before", "after", "above", "below", "to", "from", "up", "down",
            "in", "out", "on", "off", "over", "under", "again", "further", 
            "then", "once", "here", "there", "when", "where", "why", "how"
        }

    def compress(self, text: str) -> str:
        """
        Compress the input text heuristically.
        This is a lightweight approximation of LLMLingua that runs locally
        without needing a heavy transformer model.
        """
        if not text or len(text) < 100:
            return text  # Don't compress short prompts
            
        original_length = len(text)
        
        # 1. Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # 2. Remove filler phrases commonly used in prompts
        filler_phrases = [
            "Please ensure that you",
            "I would like you to",
            "Can you please",
            "It is very important that",
            "Make sure to",
            "Your task is to",
        ]
        for phrase in filler_phrases:
            text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
            
        # 3. If still too long and aggressive compression is enabled
        words = text.split()
        if len(words) > 50:
            # Simple Information Bottleneck: keep nouns, verbs, adjectives, drop some stop words
            compressed_words = []
            for w in words:
                # Keep words with punctuation (usually important)
                if not w.isalnum():
                    compressed_words.append(w)
                    continue
                    
                # Drop some stop words probabilistically or if they are very common
                if w.lower() in self.stop_words:
                    # Keep some stop words to maintain readability
                    if hash(w) % 3 == 0: 
                        compressed_words.append(w)
                else:
                    compressed_words.append(w)
                    
            text = " ".join(compressed_words)
            
        compressed_length = len(text)
        ratio = compressed_length / original_length if original_length > 0 else 1.0
        
        if ratio < 0.95:
            logger.info(f"Prompt compressed. Ratio: {ratio:.2f} (Length: {original_length} -> {compressed_length})")
            
        return text.strip()

# Global instance
compressor = PromptCompressor()
