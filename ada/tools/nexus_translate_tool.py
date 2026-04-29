#!/usr/bin/env python3
"""
Offline text translation tool for Persian ↔ English using nexus-translate.
Requires: pip install nexus-translate
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    from nexus_translate import Translator
except ImportError:
    print("Error: nexus-translate not installed. Run: pip install nexus-translate")
    sys.exit(1)


class OfflineTranslator:
    """Wrapper for nexus-translate with LangChain tool compatibility."""
    
    def __init__(self, source_lang: str, target_lang: str):
        """
        Initialize translator.
        
        Args:
            source_lang: Source language code ('en' or 'fa')
            target_lang: Target language code ('en' or 'fa')
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self._translator = Translator(source_lang=source_lang, target_lang=target_lang)
    
    def translate(self, text: str) -> str:
        """
        Translate text from source to target language.
        
        Args:
            text: Input text to translate
            
        Returns:
            Translated text
        """
        if not text or not text.strip():
            return ""
        return self._translator.translate(text)


def translate_fa_to_en(text: str) -> str:
    """
    LangChain tool function to translate Persian (fa) to English (en).
    
    Args:
        text: Persian text to translate
        
    Returns:
        English translation
    """
    translator = OfflineTranslator(source_lang="fa", target_lang="en")
    return translator.translate(text)


def translate_en_to_fa(text: str) -> str:
    """
    LangChain tool function to translate English (en) to Persian (fa).
    
    Args:
        text: English text to translate
        
    Returns:
        Persian translation
    """
    translator = OfflineTranslator(source_lang="en", target_lang="fa")
    return translator.translate(text)


def main():
    parser = argparse.ArgumentParser(
        description="Offline translation between Persian (fa) and English (en)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -t en "سلام دنیا"
  %(prog)s -t fa "Hello world"
  %(prog)s -i input.txt -o output.txt -t en
        """
    )
    
    parser.add_argument("-t", "--target", required=True, choices=["en", "fa"],
                        help="Target language: 'en' for English, 'fa' for Persian")
    parser.add_argument("-i", "--input", type=Path,
                        help="Input file path (if not provided, reads from stdin)")
    parser.add_argument("-o", "--output", type=Path,
                        help="Output file path (if not provided, writes to stdout)")
    parser.add_argument("text", nargs="?",
                        help="Text to translate (if not provided, reads from input file or stdin)")
    
    args = parser.parse_args()
    
    # Read input text
    if args.text:
        text = args.text
    elif args.input:
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading input file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Read from stdin
        text = sys.stdin.read()
        if not text:
            print("Error: No text provided", file=sys.stderr)
            sys.exit(1)
    
    if not text.strip():
        print("Error: Empty input text", file=sys.stderr)
        sys.exit(1)
    
    # Initialize translator
    try:
        # Determine source language based on target
        source = "fa" if args.target == "en" else "en"
        translator = Translator(source_lang=source, target_lang=args.target)
    except Exception as e:
        print(f"Error initializing translator: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Perform translation
    try:
        translated = translator.translate(text)
    except Exception as e:
        print(f"Translation error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Write output
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(translated)
            print(f"Translation saved to {args.output}", file=sys.stderr)
        except Exception as e:
            print(f"Error writing output file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(translated)


if __name__ == "__main__":
    main()
