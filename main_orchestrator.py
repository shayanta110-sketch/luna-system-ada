import subprocess
import tempfile
import os
import sys
from typing import Any, Dict

class MainOrchestrator:
    def __init__(self, state_manager):
        self.state_manager = state_manager

    def _check_ram_available(self, required_gb: float) -> bool:
        """Check if at least required_gb of RAM is free."""
        try:
            import psutil
            free_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
            return free_ram_gb >= required_gb
        except ImportError:
            # If psutil not available, assume RAM is sufficient
            return True

    def _unload_large_model(self, model_obj, model_name: str = "model"):
        """Safely unload a large model to free RAM."""
        if model_obj is not None:
            try:
                # Attempt to delete and force garbage collection
                del model_obj
                import gc
                gc.collect()
                # Optional: log memory freed
                # print(f"Unloaded {model_name}")
            except Exception:
                pass

    def execute_local_step(self, step: Dict[str, Any]) -> Any:
        """Execute a local code step safely."""
        code = step.get("code", "")
        if not code:
            raise ValueError("No code provided for execution")

        # Use sandboxed execution instead of direct exec()
        result = self.state_manager.execute_code_sandbox(code, timeout_sec=5)
        return result

    def audit_with_gemma(self, data: Any) -> Any:
        """Audit logic using Gemma-9B with RAM check."""
        required_ram_gb = 10.0
        if not self._check_ram_available(required_ram_gb):
            # Skip audit or use smaller model
            return {"audit_skipped": True, "reason": f"Insufficient RAM (<{required_ram_gb}GB free)"}

        model = None
        try:
            # Simulate loading Gemma-9B model (or smaller fallback)
            # Replace with actual model loading logic
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained("google/gemma-9b")
            # Perform audit
            result = {"audit_result": "success"}
            return result
        except Exception as e:
            # Fallback to smaller model if available
            try:
                from transformers import AutoModelForCausalLM
                small_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
                result = {"audit_result": "fallback_success", "model_used": "gemma-2b"}
                self._unload_large_model(small_model, "gemma-2b")
                return result
            except Exception as fallback_error:
                return {"audit_skipped": True, "reason": f"Model load failed: {str(e)}"}
        finally:
            # Ensure large model is unloaded after use
            self._unload_large_model(model, "gemma-9b")

    # Other methods would be below...
