"""Vision Router Tool for Ada - Multi-model image processing with GLM-OCR and Gemma-4-E2B support."""

import os
import base64
import json
import asyncio
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import aiohttp
import requests
from PIL import Image


@dataclass
class VisionResult:
    """Result from vision processing."""
    text: str
    model_used: str
    raw_response: Optional[Dict] = None
    confidence: Optional[float] = None


class VisionRouter:
    """Router for image processing using GLM-OCR, Gemma-4-E2B, or fallback models."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize VisionRouter with configuration."""
        self.config = config or {}
        self.default_model = self.config.get("default_model", "glm-ocr")
        self.models = {
            "glm-ocr": {
                "enabled": self.config.get("glm_enabled", True),
                "endpoint": self.config.get("glm_endpoint", "https://api.example.com/glm-ocr"),
                "api_key": self.config.get("glm_api_key", os.environ.get("GLM_API_KEY")),
                "timeout": self.config.get("glm_timeout", 60)
            },
            "gemma-4-e2b": {
                "enabled": self.config.get("gemma_enabled", True),
                "endpoint": self.config.get("gemma_endpoint", "https://api.e2b.dev/v1/gemma-4"),
                "api_key": self.config.get("gemma_api_key", os.environ.get("GEMMA_API_KEY")),
                "timeout": self.config.get("gemma_timeout", 90)
            },
            "local": {
                "enabled": self.config.get("local_enabled", False),
                "model_path": self.config.get("local_model_path", "models/vision_model.pt"),
                "device": self.config.get("device", "cpu")
            }
        }

    async def process_image(self, image_input: Union[str, Path, bytes, Image.Image],
                           model_name: Optional[str] = None,
                           prompt: str = "Describe this image in detail",
                           **kwargs) -> VisionResult:
        """Process an image with the specified model."""
        model = model_name or self.default_model
        if model not in self.models:
            raise ValueError(f"Unknown model: {model}. Available: {list(self.models.keys())}")

        if not self.models[model]["enabled"]:
            raise RuntimeError(f"Model {model} is not enabled in configuration")

        # Convert image to base64 if needed
        image_data = self._prepare_image_data(image_input)

        if model == "glm-ocr":
            return await self._call_glm_ocr(image_data, prompt, **kwargs)
        elif model == "gemma-4-e2b":
            return await self._call_gemma(image_data, prompt, **kwargs)
        elif model == "local":
            return await self._call_local_model(image_data, prompt, **kwargs)
        else:
            raise RuntimeError(f"No handler for model {model}")

    def _prepare_image_data(self, image_input: Union[str, Path, bytes, Image.Image]) -> str:
        """Convert various image input types to base64 string."""
        if isinstance(image_input, (str, Path)):
            with open(image_input, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(image_input, bytes):
            return base64.b64encode(image_input).decode("utf-8")
        elif isinstance(image_input, Image.Image):
            import io
            buffer = io.BytesIO()
            image_input.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

    async def _call_glm_ocr(self, image_b64: str, prompt: str, **kwargs) -> VisionResult:
        """Call GLM-OCR API."""
        config = self.models["glm-ocr"]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config['api_key']}"
        }
        payload = {
            "image": image_b64,
            "prompt": prompt,
            **kwargs
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(config["endpoint"],
                                       json=payload,
                                       headers=headers,
                                       timeout=aiohttp.ClientTimeout(total=config["timeout"])) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        text = data.get("text", data.get("response", ""))
                        return VisionResult(
                            text=text,
                            model_used="glm-ocr",
                            raw_response=data,
                            confidence=data.get("confidence")
                        )
                    else:
                        error_text = await resp.text()
                        raise RuntimeError(f"GLM-OCR API error {resp.status}: {error_text}")
            except asyncio.TimeoutError:
                raise RuntimeError("GLM-OCR API timeout")

    async def _call_gemma(self, image_b64: str, prompt: str, **kwargs) -> VisionResult:
        """Call Gemma-4-E2B API."""
        config = self.models["gemma-4-e2b"]
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": config["api_key"]
        }
        payload = {
            "image": image_b64,
            "instruction": prompt,
            "max_tokens": kwargs.get("max_tokens", 500),
            "temperature": kwargs.get("temperature", 0.7)
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(config["endpoint"],
                                       json=payload,
                                       headers=headers,
                                       timeout=aiohttp.ClientTimeout(total=config["timeout"])) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        text = data.get("generated_text", data.get("output", ""))
                        return VisionResult(
                            text=text,
                            model_used="gemma-4-e2b",
                            raw_response=data,
                            confidence=None
                        )
                    else:
                        error_text = await resp.text()
                        raise RuntimeError(f"Gemma API error {resp.status}: {error_text}")
            except asyncio.TimeoutError:
                raise RuntimeError("Gemma API timeout")

    async def _call_local_model(self, image_b64: str, prompt: str, **kwargs) -> VisionResult:
        """Call local vision model (synchronous wrapper)."""
        # Placeholder for local model loading
        # In production, implement with torch, transformers, etc.
        import io
        from PIL import Image as PILImage
        import base64

        # Decode image
        image_bytes = base64.b64decode(image_b64)
        image = PILImage.open(io.BytesIO(image_bytes))

        # Simulate local processing (replace with actual model inference)
        # Example: from transformers import pipeline
        # pipe = pipeline("image-to-text", model=self.models["local"]["model_path"])
        # result = pipe(image, prompt=prompt)

        result_text = f"[Local model placeholder] Image size: {image.size}, prompt: {prompt}"
        return VisionResult(
            text=result_text,
            model_used="local",
            raw_response={"size": image.size, "format": image.format},
            confidence=0.8  # placeholder
        )

    def extract_text_from_image(self, image_path: Union[str, Path]) -> str:
        """Synchronous convenience method for text extraction."""
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.process_image(image_path))
                result = future.result()
        else:
            result = loop.run_until_complete(self.process_image(image_path))

        return result.text

    def batch_process(self, image_paths: List[Union[str, Path]], model_name: Optional[str] = None) -> List[str]:
        """Process multiple images synchronously."""
        results = []
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def process_all():
            tasks = [self.process_image(path, model_name) for path in image_paths]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results_async = loop.run_until_complete(process_all())
        for res in results_async:
            if isinstance(res, Exception):
                results.append(f"Error: {str(res)}")
            else:
                results.append(res.text)
        return results


# Ada Tool Interface
def get_tool_definition() -> Dict[str, Any]:
    """Return tool definition for Ada integration."""
    return {
        "name": "vision_router",
        "description": "Process images using GLM-OCR or Gemma-4-E2B models for text extraction, OCR, and image understanding",
        "actions": [
            {
                "name": "analyze_image",
                "description": "Analyze an image and return text description or extracted content",
                "parameters": {
                    "image_source": {
                        "type": "string",
                        "description": "Path to image file or URL"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use (glm-ocr, gemma-4-e2b, local)",
                        "default": "glm-ocr"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Optional custom prompt for image understanding",
                        "default": "Describe this image in detail including any text visible"
                    }
                }
            }
        ]
    }


class AdaVisionTool:
    """Main tool class for Ada integration."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.config = json.load(f)
        self.router = VisionRouter(self.config)

    async def run(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool action."""
        if action == "analyze_image":
            image_source = parameters["image_source"]
            model = parameters.get("model", "glm-ocr")
            prompt = parameters.get("prompt", "Describe this image in detail including any text visible")

            # Handle URLs
            if image_source.startswith(("http://", "https://")):
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_source) as resp:
                        if resp.status == 200:
                            image_bytes = await resp.read()
                            result = await self.router.process_image(image_bytes, model, prompt)
                        else:
                            raise RuntimeError(f"Failed to fetch image: HTTP {resp.status}")
            else:
                result = await self.router.process_image(image_source, model, prompt)

            return {
                "success": True,
                "result": result.text,
                "model": result.model_used,
                "metadata": {"confidence": result.confidence}
            }
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}"
            }


# Example usage
def main():
    """Example CLI usage."""
    import sys
    if len(sys.argv) < 2:
        print("Usage: python vision_router.py <image_path> [model]")
        sys.exit(1)

    tool = AdaVisionTool()
    import asyncio
    result = asyncio.run(tool.run("analyze_image", {
        "image_source": sys.argv[1],
        "model": sys.argv[2] if len(sys.argv) > 2 else "glm-ocr"
    }))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()