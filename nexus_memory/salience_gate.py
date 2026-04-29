"""
Salience Gate Module

Evaluates conversation importance using three operational modes:
- LIGHTWEIGHT: Rule-based pattern matching for fast, offline scoring.
- LOCAL: Loads a local LLM via llama-cpp for inference.
- ONLINE: Proxies requests to DeepSeek API for high-quality evaluation.
"""

import os
import re
import json
import requests
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


class SalienceMode(Enum):
    LIGHTWEIGHT = "lightweight"
    LOCAL = "local"
    ONLINE = "online"


@dataclass
class SalienceResult:
    score: float  # 0.0 to 1.0
    rationale: str
    mode_used: SalienceMode
    tokens_evaluated: Optional[int] = None


class SalienceGate:
    def __init__(self, mode: SalienceMode = SalienceMode.LIGHTWEIGHT):
        self.mode = mode
        self._local_model = None
        self._local_model_path = os.getenv("LLAMA_CPP_MODEL_PATH", "")
        self._deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self._deepseek_endpoint = os.getenv("DEEPSEEK_ENDPOINT", "https://api.deepseek.com/v1/chat/completions")

        if mode == SalienceMode.LOCAL:
            self._init_local_model()

    def _init_local_model(self):
        """Initialize llama-cpp model if in LOCAL mode."""
        try:
            from llama_cpp import Llama
            if not os.path.exists(self._local_model_path):
                raise FileNotFoundError(f"Local model not found at {self._local_model_path}")
            self._local_model = Llama(model_path=self._local_model_path, verbose=False)
        except ImportError:
            raise RuntimeError("llama-cpp-python not installed. Run: pip install llama-cpp-python")
        except Exception as e:
            raise RuntimeError(f"Failed to load local model: {e}")

    def evaluate(self, conversation: List[Dict[str, str]]) -> SalienceResult:
        """Main entry point to evaluate conversation salience."""
        if self.mode == SalienceMode.LIGHTWEIGHT:
            return self._evaluate_lightweight(conversation)
        elif self.mode == SalienceMode.LOCAL:
            return self._evaluate_local(conversation)
        elif self.mode == SalienceMode.ONLINE:
            return self._evaluate_online(conversation)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _evaluate_lightweight(self, conversation: List[Dict[str, str]]) -> SalienceResult:
        """Rule-based scoring using patterns and heuristics."""
        text = " ".join([msg.get("content", "") for msg in conversation])
        score = 0.0

        # Importance indicators
        high_importance_patterns = [
            r"\b(important|critical|urgent|asap|priority)\b",
            r"\b(decision|decide|choose|agree|disagree)\b",
            r"\b(problem|issue|bug|error|crash|fail)\b",
            r"\b(plan|schedule|deadline|deliverable)\b",
            r"\?.*\?.*\?"  # multiple questions
        ]
        medium_importance_patterns = [
            r"\b(help|assist|support|guide)\b",
            r"\b(explain|clarify|understand|confused)\b",
            r"\b(update|progress|status|change)\b"
        ]

        for pattern in high_importance_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.25
        for pattern in medium_importance_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.1

        # Length heuristic (longer convos slightly more important)
        word_count = len(text.split())
        score += min(0.2, word_count / 500.0)

        # Cap at 1.0
        score = min(1.0, score)

        rationale = f"Lightweight rule-based scoring. Detected {int(score*100)}% salience."
        return SalienceResult(score=score, rationale=rationale, mode_used=SalienceMode.LIGHTWEIGHT)

    def _evaluate_local(self, conversation: List[Dict[str, str]]) -> SalienceResult:
        """Use local llama-cpp model to evaluate salience."""
        if self._local_model is None:
            raise RuntimeError("Local model not initialized. Check model path.")

        prompt = self._build_salience_prompt(conversation)
        try:
            output = self._local_model(prompt, max_tokens=100, temperature=0.2, echo=False)
            generated = output["choices"][0]["text"].strip()
            score, rationale = self._parse_local_response(generated)
            tokens = output["usage"]["total_tokens"]
            return SalienceResult(score=score, rationale=rationale, mode_used=SalienceMode.LOCAL, tokens_evaluated=tokens)
        except Exception as e:
            raise RuntimeError(f"Local inference failed: {e}")

    def _evaluate_online(self, conversation: List[Dict[str, str]]) -> SalienceResult:
        """Proxy to DeepSeek API for high-quality evaluation."""
        if not self._deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")

        prompt = self._build_salience_prompt(conversation)
        headers = {
            "Authorization": f"Bearer {self._deepseek_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a salience evaluator. Rate conversation importance (0.0 to 1.0) and provide a short rationale."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 150
        }

        try:
            response = requests.post(self._deepseek_endpoint, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            generated = data["choices"][0]["message"]["content"].strip()
            score, rationale = self._parse_online_response(generated)
            tokens = data.get("usage", {}).get("total_tokens", None)
            return SalienceResult(score=score, rationale=rationale, mode_used=SalienceMode.ONLINE, tokens_evaluated=tokens)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"DeepSeek API request failed: {e}")

    def _build_salience_prompt(self, conversation: List[Dict[str, str]]) -> str:
        """Construct prompt for LLM-based evaluation."""
        transcript = ""
        for msg in conversation:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            transcript += f"{role.upper()}: {content}\n"
        prompt = f"""Evaluate the importance (salience) of this conversation on a scale from 0.0 (trivial) to 1.0 (critical).

Conversation:
{transcript}

Respond in exactly two lines:
SCORE: <float 0.0-1.0>
RATIONALE: <short reason>
"""
        return prompt

    def _parse_local_response(self, response_text: str) -> tuple[float, str]:
        """Extract score and rationale from local model output."""
        score_match = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", response_text, re.IGNORECASE)
        rationale_match = re.search(r"RATIONALE:\s*(.+)", response_text, re.IGNORECASE | re.DOTALL)
        score = float(score_match.group(1)) if score_match else 0.5
        rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided."
        return min(1.0, max(0.0, score)), rationale

    def _parse_online_response(self, response_text: str) -> tuple[float, str]:
        """Extract score and rationale from DeepSeek output."""
        # Similar parsing but more robust
        score_match = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", response_text, re.IGNORECASE)
        if not score_match:
            score_match = re.search(r"([0-9]\.[0-9])", response_text)
        rationale_match = re.search(r"RATIONALE:\s*(.+)", response_text, re.IGNORECASE | re.DOTALL)
        if not rationale_match and len(response_text) < 200:
            rationale = response_text
        else:
            rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided."
        score = float(score_match.group(1)) if score_match else 0.5
        return min(1.0, max(0.0, score)), rationale

    @classmethod
    def from_env(cls) -> "SalienceGate":
        """Factory method to create SalienceGate based on env variable SALIENCE_MODE."""
        mode_str = os.getenv("SALIENCE_MODE", "lightweight").upper()
        try:
            mode = SalienceMode[mode_str]
        except KeyError:
            mode = SalienceMode.LIGHTWEIGHT
        return cls(mode=mode)
