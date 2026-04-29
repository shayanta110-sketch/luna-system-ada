#!/usr/bin/env python3
"""
Translation Layer Tool for Ada
Provides offline bidirectional Persian-English translation capabilities.
Integrates translation logic from the original project.
"""

import argparse
import sys
import json
from typing import Dict, Any, Optional

class TranslationLayer:
    """Handles Persian-English translations offline."""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the translation layer.
        
        Args:
            data_path: Path to translation data files (if any)
        """
        self.data_path = data_path
        # Simplified in-memory dictionary for demo purposes
        # In production, this would load from offline files
        self.en_fa_dict = {
            "hello": "سلام",
            "goodbye": "خداحافظ",
            "thank you": "متشکرم",
            "yes": "بله",
            "no": "نه",
            "please": "لطفا",
            "how are you": "چطور هستید",
            "good morning": "صبح بخیر",
            "good night": "شب بخیر",
            "friend": "دوست"
        }
        self.fa_en_dict = {v: k for k, v in self.en_fa_dict.items()}
    
    def translate(self, text: str, source: str = "auto", target: str = "auto") -> str:
        """
        Translate text between Persian and English.
        
        Args:
            text: The text to translate
            source: Source language ('en', 'fa', or 'auto')
            target: Target language ('en', 'fa', or 'auto')
            
        Returns:
            Translated text
        """
        if not text or not text.strip():
            return ""
        
        text_lower = text.strip().lower()
        
        # Auto-detect source language
        if source == "auto":
            # Simple detection: if any character is Persian/Arabic
            if any('\u0600' <= c <= '\u06FF' for c in text):
                source = "fa"
            else:
                source = "en"
        
        # Determine target if auto
        if target == "auto":
            target = "fa" if source == "en" else "en"
        
        # Perform translation
        if source == "en" and target == "fa":
            return self.en_fa_dict.get(text_lower, f"[untranslated: {text}]")
        elif source == "fa" and target == "en":
            return self.fa_en_dict.get(text_lower, f"[untranslated: {text}]")
        else:
            return text
    
    def translate_batch(self, texts: list, source: str = "auto", target: str = "auto") -> list:
        """
        Translate multiple texts.
        
        Args:
            texts: List of texts to translate
            source: Source language
            target: Target language
            
        Returns:
            List of translated texts
        """
        return [self.translate(t, source, target) for t in texts]


def main():
    """Command-line interface for the translation tool."""
    parser = argparse.ArgumentParser(
        description="Offline bidirectional Persian-English translation tool"
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to translate"
    )
    parser.add_argument(
        "--source", "-s",
        default="auto",
        choices=["en", "fa", "auto"],
        help="Source language (default: auto)"
    )
    parser.add_argument(
        "--target", "-t",
        default="auto",
        choices=["en", "fa", "auto"],
        help="Target language (default: auto)"
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Read multiple lines from stdin"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    
    args = parser.parse_args()
    
    translator = TranslationLayer()
    
    # Handle batch input
    if args.batch:
        texts = [line.strip() for line in sys.stdin if line.strip()]
        results = translator.translate_batch(texts, args.source, args.target)
        if args.json:
            output = [{"input": t, "output": r} for t, r in zip(texts, results)]
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            for inp, out in zip(texts, results):
                print(f"{inp} -> {out}")
    # Handle single text input
    elif args.text:
        result = translator.translate(args.text, args.source, args.target)
        if args.json:
            output = {"input": args.text, "output": result}
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            print(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
