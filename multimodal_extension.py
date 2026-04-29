"""Multimodal extension with fallback OCR (pytesseract) and optional VLM models."""
import os
import warnings
from typing import Optional, Dict, Any, Union

# Lightweight OCR
try:
    import pytesseract
    from PIL import Image
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    warnings.warn("pytesseract not installed. OCR fallback will fail.")

# Heavy VLM models (only loaded when requested)
_VLM_MODELS = {
    "gemma-4-e2b": None,
    "glm-ocr": None
}

def _load_vlm_model(model_name: str):
    """Load a VLM model on demand with n_gpu_layers=0 due to 2GB VRAM limitation."""
    if _VLM_MODELS.get(model_name) is not None:
        return _VLM_MODELS[model_name]

    try:
        from llama_cpp import Llama
        model_path = os.getenv(f"{model_name.upper().replace('-', '_')}_PATH", f"./models/{model_name}.gguf")
        model = Llama(
            model_path=model_path,
            n_gpu_layers=0,      # Force CPU only due to 2GB VRAM limit
            n_ctx=2048,
            verbose=False
        )
        _VLM_MODELS[model_name] = model
        return model
    except Exception as e:
        warnings.warn(f"Failed to load {model_name}: {e}")
        return None

def run_ocr(image: Union[str, Image.Image]) -> str:
    """Run fast OCR using pytesseract (fallback)."""
    if not PYTESSERACT_AVAILABLE:
        raise RuntimeError("pytesseract is required for OCR fallback. Install with: pip install pytesseract pillow")
    if isinstance(image, str):
        image = Image.open(image)
    return pytesseract.image_to_string(image)

def run_vision(
    image: Union[str, Image.Image],
    prompt: Optional[str] = None,
    model: str = "ocr"  # 'ocr' (fast), 'gemma-4-e2b', 'glm-ocr'
) -> Dict[str, Any]:
    """
    Run vision task.

    Args:
        image: Path to image or PIL Image.
        prompt: Optional text prompt (ignored for OCR fallback).
        model: 'ocr' (default, fast pytesseract), 'gemma-4-e2b', or 'glm-ocr'.

    Returns:
        Dictionary with 'text' and 'model_used' keys.
    """
    if model == "ocr":
        text = run_ocr(image)
        return {"text": text, "model_used": "pytesseract-ocr"}

    if model not in _VLM_MODELS:
        raise ValueError(f"Unknown VLM model: {model}. Choose from {list(_VLM_MODELS.keys())}")

    vlm = _load_vlm_model(model)
    if vlm is None:
        # Fallback to OCR if VLM fails to load
        warnings.warn(f"{model} failed to load. Falling back to OCR.")
        text = run_ocr(image)
        return {"text": text, "model_used": "pytesseract-ocr (fallback)"}

    # Convert image to base64 or path for VLM (simplified)
    if isinstance(image, str):
        img_path = image
    else:
        # Save PIL Image temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            img_path = tmp.name

    # Simplified VLM inference (adapt to actual API)
    try:
        # Assuming the model expects "image" and "prompt" (placeholder)
        response = vlm.create_chat_completion(
            messages=[
                {"role": "user", "content": f"[Image: {img_path}]\n{prompt or 'Describe this image in detail.'}"}
            ]
        )
        text = response["choices"][0]["message"]["content"]
    except Exception as e:
        warnings.warn(f"VLM inference failed: {e}. Falling back to OCR.")
        text = run_ocr(image)
        model = "pytesseract-ocr (fallback)"
    finally:
        # Clean up temp file if created
        if not isinstance(image, str) and 'img_path' in locals():
            try:
                os.unlink(img_path)
            except:
                pass

    return {"text": text, "model_used": model}

# Convenience alias
multimodal_vision = run_vision
