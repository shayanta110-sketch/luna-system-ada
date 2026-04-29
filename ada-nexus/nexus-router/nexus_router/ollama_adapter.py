import subprocess
import json
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class OllamaAdapter:
    """Adapter to communicate with Ollama CLI for local model management."""

    def __init__(self, ollama_bin: str = "ollama"):
        self.ollama_bin = ollama_bin

    def _run_ollama_command(self, *args: str) -> str:
        """Run an ollama CLI command and return stdout."""
        cmd = [self.ollama_bin] + list(args)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Ollama command failed: {' '.join(cmd)} - {e.stderr}")
            raise RuntimeError(f"Failed to run ollama: {e.stderr}")

    def fetch_local_models(self) -> List[Dict[str, Any]]:
        """Fetch all locally available Ollama models with metadata."""
        output = self._run_ollama_command("list")
        return self._parse_model_list(output)

    def _parse_model_list(self, raw_output: str) -> List[Dict[str, Any]]:
        """Parse `ollama list` output into structured model data."""
        models = []
        lines = raw_output.strip().split('\n')
        if len(lines) < 2:
            return models
        # Skip header line (NAME, ID, SIZE, MODIFIED)
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            model_name = parts[0]
            model_id = parts[1]
            size_str = parts[2]
            modified = ' '.join(parts[3:])

            metadata = self._fetch_model_metadata(model_name)
            model_info = {
                "name": model_name,
                "id": model_id,
                "size": self._parse_size(size_str),
                "size_display": size_str,
                "modified": modified,
                **metadata
            }
            models.append(model_info)
        return models

    def _fetch_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """Fetch detailed metadata for a specific model using `ollama show`."""
        try:
            output = self._run_ollama_command("show", model_name)
            return self._parse_show_output(output, model_name)
        except RuntimeError:
            logger.warning(f"Could not fetch metadata for {model_name}")
            return self._fallback_metadata(model_name)

    def _parse_show_output(self, output: str, model_name: str) -> Dict[str, Any]:
        """Parse `ollama show` output to extract parameters, type, capabilities, quality."""
        metadata = {
            "parameters": None,
            "type": "unknown",
            "capabilities": [],
            "quality_score": 0.0,
            "model_family": None,
            "quantization": None
        }

        # Extract parameter count (e.g., "7B", "13B", "70B")
        param_match = re.search(r'(\d+(?:\.\d+)?)\s*[BGT]', output, re.IGNORECASE)
        if param_match:
            metadata["parameters"] = param_match.group(1) + "B"

        # Detect model type from name or output
        name_lower = model_name.lower()
        if "llama" in name_lower:
            metadata["type"] = "llama"
            metadata["model_family"] = "Llama"
        elif "mistral" in name_lower:
            metadata["type"] = "mistral"
            metadata["model_family"] = "Mistral"
        elif "phi" in name_lower:
            metadata["type"] = "phi"
            metadata["model_family"] = "Phi"
        elif "gemma" in name_lower:
            metadata["type"] = "gemma"
            metadata["model_family"] = "Gemma"
        elif "qwen" in name_lower:
            metadata["type"] = "qwen"
            metadata["model_family"] = "Qwen"
        elif "mixtral" in name_lower:
            metadata["type"] = "mixtral"
            metadata["model_family"] = "Mixtral"

        # Extract quantization (e.g., q4_0, q5_K_M)
        quant_match = re.search(r'(q\d+_[0-9A-Z_]+)', output, re.IGNORECASE)
        if quant_match:
            metadata["quantization"] = quant_match.group(1)

        # Infer capabilities based on model type and parameters
        metadata["capabilities"] = self._infer_capabilities(metadata["type"], metadata["parameters"])

        # Compute quality score heuristic
        metadata["quality_score"] = self._compute_quality_score(metadata)

        return metadata

    def _infer_capabilities(self, model_type: str, param_str: Optional[str]) -> List[str]:
        """Infer model capabilities from type and parameter size."""
        caps = ["text-generation"]
        if model_type in ["llama", "mistral", "mixtral"]:
            caps.append("chat")
            caps.append("instruction-following")
        elif model_type == "phi":
            caps.append("code-generation")
            caps.append("reasoning")
        elif model_type == "qwen":
            caps.append("multilingual")
            caps.append("code")

        if param_str:
            try:
                param_num = float(param_str.rstrip('BGT'))
                if param_num >= 13:
                    caps.append("complex-reasoning")
                if param_num >= 30:
                    caps.append("advanced-rag")
            except ValueError:
                pass
        return caps

    def _compute_quality_score(self, metadata: Dict[str, Any]) -> float:
        """Compute a heuristic quality score (0-10) based on parameters, quantization, type."""
        score = 5.0  # baseline

        param_str = metadata.get("parameters")
        if param_str:
            try:
                params = float(param_str.rstrip('BGT'))
                if params >= 70:
                    score += 3.0
                elif params >= 30:
                    score += 2.0
                elif params >= 13:
                    score += 1.5
                elif params >= 7:
                    score += 0.5
            except ValueError:
                pass

        # Quantization quality adjustment
        quant = metadata.get("quantization", "")
        if quant:
            if "q8" in quant or "q6" in quant:
                score += 1.0
            elif "q5" in quant:
                score += 0.5
            elif "q4" in quant:
                score -= 0.5
            elif "q2" in quant:
                score -= 1.5

        # Model family bonus
        family_bonus = {
            "llama": 1.0,
            "mistral": 1.0,
            "qwen": 0.5,
            "phi": 0.5,
            "gemma": 0.5
        }
        model_type = metadata.get("type", "")
        score += family_bonus.get(model_type, 0.0)

        return round(max(0.0, min(10.0, score)), 1)

    def _parse_size(self, size_str: str) -> int:
        """Convert size string like '4.2GB' to bytes."""
        match = re.match(r'([\d.]+)\s*([A-Za-z]+)', size_str)
        if not match:
            return 0
        value, unit = float(match.group(1)), match.group(2).upper()
        multipliers = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4}
        return int(value * multipliers.get(unit, 1))

    def _fallback_metadata(self, model_name: str) -> Dict[str, Any]:
        """Provide minimal metadata when show command fails."""
        return {
            "parameters": None,
            "type": "unknown",
            "capabilities": ["text-generation"],
            "quality_score": 3.0,
            "model_family": None,
            "quantization": None
        }

    def get_model_zoo_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Replace original model_zoo functionality by returning structured model list."""
        models = self.fetch_local_models()
        return {
            "models": models,
            "total_count": len(models),
            "source": "ollama-local"
        }

    def get_model_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific model's metadata by name."""
        models = self.fetch_local_models()
        for model in models:
            if model["name"] == name:
                return model
        return None
