"""Rule-based memory compressor inspired by Steno notation. This module provides
compression techniques to reduce memory size and token usage by converting
verbose text into compact, structured representations. Supports two output
formats: - Human-readable: Compressed but still legible (e.g., "usr logged in")
- Machine-compact: Extremely dense, optimized for token efficiency
"""

import re
import json
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum


class CompressionFormat(Enum):
    """Output format for compressed data."""
    HUMAN_READABLE = "human"      # Short but legible
    MACHINE_COMPACT = "compact"   # Token-minimized representation


class RuleBasedCompressor:
    """Rule-based text compressor using Steno-like abbreviation rules."""

    # Common word substitutions (human-readable)
    HUMAN_RULES = {
        r'\bthe\b': '',
        r'\ba\b': '',
        r'\ban\b': '',
        r'\band\b': '&',
        r'\bto\b': '2',
        r'\bfor\b': '4',
        r'\bbe\b': 'b',
        r'\bby\b': 'x',
        r'\bwith\b': 'w/',
        r'\bwithout\b': 'w/o',
        r'\bfrom\b': 'fm',
        r'\bthat\b': 'tht',
        r'\bthis\b': 'ths',
        r'\bhave\b': 'hav',
        r'\bwill\b': 'wil',
        r'\bnot\b': '!',
        r'\byou\b': 'u',
        r'\bare\b': 'r',
        r'\bplease\b': 'pls',
        r'\bthanks\b': 'thx',
    }

    # Ultra-compact rules for machine format
    MACHINE_RULES = {
        r'\bthe\b': 'T',
        r'\ba\b': 'A',
        r'\ban\b': 'AN',
        r'\band\b': '&',
        r'\bto\b': '2',
        r'\bfor\b': '4',
        r'\bof\b': 'O',
        r'\bin\b': 'N',
        r'\bon\b': 'ON',
        r'\bat\b': '@',
        r'\bwith\b': 'W',
        r'\bwithout\b': 'WO',
        r'\bfrom\b': 'FM',
        r'\bhave\b': 'HV',
        r'\bwill\b': 'WL',
        r'\bnot\b': '!',
        r'\byou\b': 'U',
        r'\bare\b': 'R',
        r'\bis\b': '=',
        r'\bwas\b': 'WS',
        r'\bwere\b': 'WR',
        r'\bhas\b': 'HZ',
        r'\bhad\b': 'HD',
        r'\bbut\b': 'B',
        r'\bor\b': '|',
        r'\bso\b': 'SO',
        r'\bif\b': '?',
        r'\bthen\b': '>',
        r'\belse\b': '<',
    }

    # Punctuation and whitespace normalization
    WHITESPACE_RULES = [
        (r'\s+', ' '),                     # Collapse spaces
        (r'^\s+|\s+$', ''),               # Trim edges
        (r'\s*([.,!?;:])\s*', r'\1 '),    # Fix punctuation spacing
    ]

    def __init__(self, compression_format: CompressionFormat = CompressionFormat.HUMAN_READABLE):
        """Initialize compressor with specified output format.
        
        Args:
            compression_format: HUMAN_READABLE or MACHINE_COMPACT
        """
        self.format = compression_format
        self._compiled_rules = self._compile_rules()

    def _compile_rules(self) -> List[Tuple[re.Pattern, str]]:
        """Compile regex rules for active format."""
        rules_dict = self.HUMAN_RULES if self.format == CompressionFormat.HUMAN_READABLE else self.MACHINE_RULES
        compiled = []
        for pattern, replacement in rules_dict.items():
            compiled.append((re.compile(pattern, re.IGNORECASE), replacement))
        return compiled

    def compress(self, text: str) -> str:
        """Compress input text using rule-based transformations.
        
        Args:
            text: Original text to compress
            
        Returns:
            Compressed string
        """
        if not text or not isinstance(text, str):
            return ""

        result = text

        # Apply substitution rules
        for pattern, replacement in self._compiled_rules:
            result = pattern.sub(replacement, result)

        # Apply whitespace cleanup
        for pattern, replacement in self.WHITESPACE_RULES:
            result = re.sub(pattern, replacement, result)

        # Machine-compact: remove vowels from remaining words (extreme)
        if self.format == CompressionFormat.MACHINE_COMPACT:
            result = self._strip_vowels(result)

        return result.strip()

    def _strip_vowels(self, text: str) -> str:
        """Remove vowels from words (keeping short words intact)."""
        words = text.split()
        processed = []
        for word in words:
            if len(word) <= 3:
                processed.append(word)
            else:
                # Remove a, e, i, o, u (case-insensitive)
                vowel_free = re.sub(r'[aeiouAEIOU]', '', word)
                processed.append(vowel_free if vowel_free else word[0])
        return ' '.join(processed)

    def compress_with_metadata(self, text: str) -> Dict[str, Any]:
        """Compress and return metadata including size savings.
        
        Args:
            text: Original text
            
        Returns:
            Dictionary with compressed text, original size, compressed size, and ratio
        """
        original_size = len(text.encode('utf-8'))
        compressed = self.compress(text)
        compressed_size = len(compressed.encode('utf-8'))
        return {
            'original': text,
            'compressed': compressed,
            'original_bytes': original_size,
            'compressed_bytes': compressed_size,
            'savings_bytes': original_size - compressed_size,
            'compression_ratio': compressed_size / original_size if original_size > 0 else 1.0,
            'format': self.format.value
        }

    def batch_compress(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Compress multiple texts.
        
        Args:
            texts: List of strings to compress
            
        Returns:
            List of compression metadata dictionaries
        """
        return [self.compress_with_metadata(t) for t in texts]


class StenoMemoryEncoder:
    """High-level encoder for storing memory items in compressed format."""
    
    def __init__(self, default_format: CompressionFormat = CompressionFormat.HUMAN_READABLE):
        """Initialize encoder with default compression format."""
        self.default_format = default_format
        self.compressors = {
            CompressionFormat.HUMAN_READABLE: RuleBasedCompressor(CompressionFormat.HUMAN_READABLE),
            CompressionFormat.MACHINE_COMPACT: RuleBasedCompressor(CompressionFormat.MACHINE_COMPACT)
        }

    def encode(
        self,
        memory_item: Dict[str, Any],
        compression_format: Optional[CompressionFormat] = None
    ) -> Dict[str, Any]:
        """Encode a memory item with compression.
        
        Args:
            memory_item: Dict with 'content' field and optional metadata
            compression_format: Override default format
            
        Returns:
            Compressed memory item with '_compressed' flag
        """
        if 'content' not in memory_item:
            return memory_item
        fmt = compression_format or self.default_format
        compressor = self.compressors[fmt]
        compressed_content = compressor.compress(memory_item['content'])
        result = memory_item.copy()
        result['content'] = compressed_content
        result['_compressed'] = True
        result['_compression_format'] = fmt.value
        return result

    def encode_batch(
        self,
        memory_items: List[Dict[str, Any]],
        compression_format: Optional[CompressionFormat] = None
    ) -> List[Dict[str, Any]]:
        """Encode multiple memory items."""
        return [self.encode(item, compression_format) for item in memory_items]

    def estimate_savings(self, memory_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate compression savings without modifying items.
        
        Args:
            memory_items: List of memory items with 'content' field
            
        Returns:
            Aggregate savings statistics
        """
        total_original = 0
        total_compressed_human = 0
        total_compressed_machine = 0
        for item in memory_items:
            content = item.get('content', '')
            total_original += len(content.encode('utf-8'))
            human_compressed = self.compressors[CompressionFormat.HUMAN_READABLE].compress(content)
            total_compressed_human += len(human_compressed.encode('utf-8'))
            machine_compressed = self.compressors[CompressionFormat.MACHINE_COMPACT].compress(content)
            total_compressed_machine += len(machine_compressed.encode('utf-8'))

        return {
            'total_original_bytes': total_original,
            'human_readable': {
                'compressed_bytes': total_compressed_human,
                'savings_bytes': total_original - total_compressed_human,
                'ratio': total_compressed_human / total_original if total_original > 0 else 1.0
            },
            'machine_compact': {
                'compressed_bytes': total_compressed_machine,
                'savings_bytes': total_original - total_compressed_machine,
                'ratio': total_compressed_machine / total_original if total_original > 0 else 1.0
            }
        }


# Convenience functions
def quick_compress(text: str, compact: bool = False) -> str:
    """Quick one-off compression.
    
    Args:
        text: Text to compress
        compact: Use machine-compact format if True
        
    Returns:
        Compressed string
    """
    fmt = CompressionFormat.MACHINE_COMPACT if compact else CompressionFormat.HUMAN_READABLE
    compressor = RuleBasedCompressor(fmt)
    return compressor.compress(text)
