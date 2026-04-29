"""
Offline translator using OmniTranslate GGUF model via llama-cpp-python.
Supports Persian-English translation with language detection and model lifecycle management.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
from langdetect import detect
from llama_cpp import Llama


class OfflineTranslator:
    """Handles offline Persian-English translation using a GGUF model."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,
        n_threads: int = 4,
        verbose: bool = False
    ):
        """
        Initialize the translator with model path and parameters.

        Args:
            model_path: Path to the GGUF model file.
            n_ctx: Context window size.
            n_threads: Number of CPU threads to use.
            verbose: Enable verbose logging.
        """
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.verbose = verbose
        self._model: Optional[Llama] = None
        self.logger = logging.getLogger(__name__)

    def load_model(self) -> None:
        """Load the GGUF model into memory."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        if self._model is None:
            self.logger.info(f"Loading model from {self.model_path}")
            self._model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=self.verbose
            )
            self.logger.info("Model loaded successfully")

    def unload_model(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            self.logger.info("Unloading model")
            del self._model
            self._model = None
            self.logger.info("Model unloaded")

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.

        Args:
            text: Input string to analyze.

        Returns:
            Language code ('fa' for Persian, 'en' for English).
        """
        if not text or not text.strip():
            return "unknown"

        try:
            lang = detect(text)
            # Map to supported languages
            if lang == "fa":
                return "fa"
            elif lang == "en":
                return "en"
            else:
                self.logger.warning(f"Detected unsupported language: {lang}")
                return "unknown"
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return "unknown"

    def translate(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: str = "en"
    ) -> str:
        """
        Translate text between Persian and English.

        Args:
            text: Input text to translate.
            source_lang: Source language ('fa' or 'en'). If None, auto-detect.
            target_lang: Target language ('fa' or 'en'). Default is 'en'.

        Returns:
            Translated text.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If unsupported languages are specified.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not text or not text.strip():
            return ""

        # Determine source language
        if source_lang is None:
            source_lang = self.detect_language(text)
            if source_lang == "unknown":
                raise ValueError("Could not detect source language")

        # Validate languages
        if source_lang not in ("fa", "en"):
            raise ValueError(f"Unsupported source language: {source_lang}")
        if target_lang not in ("fa", "en"):
            raise ValueError(f"Unsupported target language: {target_lang}")

        if source_lang == target_lang:
            return text

        # Build prompt based on translation direction
        if source_lang == "fa" and target_lang == "en":
            prompt = (
                f"Translate the following Persian text to English. "
                f"Output only the translation without any extra text.\n\n"
                f"Persian: {text}\nEnglish:"
            )
        elif source_lang == "en" and target_lang == "fa":
            prompt = (
                f"Translate the following English text to Persian. "
                f"Output only the translation without any extra text.\n\n"
                f"English: {text}\nPersian:"
            )
        else:
            raise ValueError(f"Unsupported translation pair: {source_lang} -> {target_lang}")

        # Run inference
        self.logger.debug(f"Translating: {text[:50]}...")
        response = self._model(
            prompt,
            max_tokens=512,
            temperature=0.1,
            top_p=0.95,
            echo=False,
            stop=["\n\n", "\nPersian:", "\nEnglish:"]
        )

        translated = response["choices"][0]["text"].strip()
        self.logger.debug(f"Translation result: {translated[:50]}...")
        return translated

    def __enter__(self):
        """Context manager entry: loads model."""
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: unloads model."""
        self.unload_model()
