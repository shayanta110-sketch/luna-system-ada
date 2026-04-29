#!/usr/bin/env python3
"""
Nexus Vision Tool - Image processing utility for OCR and VLM operations.
Supports optical character recognition (OCR) and vision-language model (VLM) inference.
"""

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional


class NexusVisionProcessor:
    """Handles image processing using OCR and VLM models."""

    def __init__(self, model_type: str = "ocr", model_name: str = "easyocr"):
        """
        Initialize the vision processor.

        Args:
            model_type: Type of model to use ('ocr' or 'vlm')
            model_name: Specific model name (e.g., 'easyocr', 'tesseract', 'blip', 'llava')
        """
        self.model_type = model_type
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the specified model."""
        if self.model_type == "ocr":
            if self.model_name == "easyocr":
                try:
                    import easyocr
                    self.model = easyocr.Reader(['en'])
                except ImportError:
                    raise ImportError("easyocr not installed. Run: pip install easyocr")
            elif self.model_name == "tesseract":
                try:
                    import pytesseract
                    from PIL import Image
                    self.model = "tesseract"
                except ImportError:
                    raise ImportError("pytesseract not installed. Run: pip install pytesseract pillow")
            else:
                raise ValueError(f"Unsupported OCR model: {self.model_name}")

        elif self.model_type == "vlm":
            if self.model_name == "blip":
                try:
                    from transformers import BlipProcessor, BlipForConditionalGeneration
                    from PIL import Image
                    self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                    self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                except ImportError:
                    raise ImportError("transformers not installed. Run: pip install transformers torch pillow")
            elif self.model_name == "llava":
                try:
                    from transformers import LlavaProcessor, LlavaForConditionalGeneration
                    self.processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
                    self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
                except ImportError:
                    raise ImportError("transformers and torch required for LLaVA")
            else:
                raise ValueError(f"Unsupported VLM model: {self.model_name}")

    def process_image(self, image_path: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an image for OCR or VLM inference.

        Args:
            image_path: Path to the image file
            prompt: Optional text prompt for VLM models

        Returns:
            Dictionary containing processing results
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self.model_type == "ocr":
            return self._perform_ocr(image_path)
        elif self.model_type == "vlm":
            return self._perform_vlm(image_path, prompt)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _perform_ocr(self, image_path: str) -> Dict[str, Any]:
        """Perform OCR on the image."""
        if self.model_name == "easyocr":
            result = self.model.readtext(image_path)
            text = "\n".join([item[1] for item in result])
            return {
                "success": True,
                "text": text,
                "details": result
            }
        elif self.model_name == "tesseract":
            import pytesseract
            from PIL import Image
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return {
                "success": True,
                "text": text,
                "details": {"confidence": "N/A"}
            }

    def _perform_vlm(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Perform VLM inference on the image."""
        from PIL import Image
        image = Image.open(image_path)

        if prompt is None:
            prompt = "Describe this image in detail."

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        output = self.model.generate(**inputs, max_new_tokens=200)
        description = self.processor.decode(output[0], skip_special_tokens=True)

        return {
            "success": True,
            "description": description,
            "prompt": prompt
        }


def main():
    parser = argparse.ArgumentParser(description="Nexus Vision Tool - OCR and VLM image processing")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--type", "-t", choices=["ocr", "vlm"], default="ocr",
                        help="Processing type: OCR or VLM (default: ocr)")
    parser.add_argument("--model", "-m", default="easyocr",
                        help="Model name (easyocr, tesseract, blip, llava)")
    parser.add_argument("--prompt", "-p", help="Prompt for VLM models")
    parser.add_argument("--output", "-o", help="Output file for results (JSON format)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    try:
        processor = NexusVisionProcessor(model_type=args.type, model_name=args.model)
        result = processor.process_image(args.image, args.prompt)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            if args.verbose:
                print(f"Results saved to {args.output}")
        else:
            if args.type == "ocr":
                print(result.get("text", ""))
            else:
                print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
