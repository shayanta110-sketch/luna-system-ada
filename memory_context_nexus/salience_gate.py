"""
Salience Gate Module based on Memlayer concept.

Filters incoming information to separate important content from noise
using three operational modes: LOCAL, ONLINE, and LIGHTWEIGHT.

Modes:
- LOCAL: Uses a local pre-trained ML model for salience scoring
- ONLINE: Calls an external API (e.g., Hugging Face, OpenAI) for high-accuracy scoring
- LIGHTWEIGHT: Uses keyword-based heuristics and simple NLP for minimal resource usage
"""

import os
import json
import hashlib
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SalienceMode(Enum):
    """Operational modes for the salience gate."""
    LOCAL = "local"
    ONLINE = "online"
    LIGHTWEIGHT = "lightweight"


class SalienceScorer(ABC):
    """Abstract base class for salience scoring strategies."""
    
    @abstractmethod
    def score(self, text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute salience score for given text.
        
        Args:
            text: Input text to evaluate
            context: Optional context (e.g., user ID, session metadata)
            
        Returns:
            Float score between 0.0 (noise) and 1.0 (highly important)
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the scorer is ready to use."""
        pass


class LocalMLScorer(SalienceScorer):
    """Local ML model-based salience scorer.
    
    Uses a pre-trained transformer model (e.g., BERT, DistilBERT)
    fine-tuned for importance/relevance ranking.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize local ML scorer.
        
        Args:
            model_path: Path to local model directory. If None, uses default.
        """
        self.model = None
        self.tokenizer = None
        self.model_path = model_path or os.getenv("SALIENCE_MODEL_PATH", "models/salience")
        self._load_model()
    
    def _load_model(self):
        """Attempt to load the ML model."""
        try:
            # Defer heavy imports to avoid dependency issues
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            if os.path.exists(self.model_path):
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                logger.info(f"Loaded local salience model from {self.model_path}")
            else:
                logger.warning(f"Model path {self.model_path} not found. Using fallback.")
                self.model = None
        except ImportError:
            logger.warning("transformers not installed. Local ML scorer unavailable.")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            self.model = None
    
    def score(self, text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Compute salience using local transformer model."""
        if not self.is_available():
            # Fallback to simple length-based heuristic
            return min(1.0, len(text) / 1000.0)
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model(**inputs)
            logits = outputs.logits
            probability = torch.softmax(logits, dim=-1)[0][1].item()  # Binary classification
            return probability
        except Exception as e:
            logger.error(f"Local scoring failed: {e}")
            return 0.5
    
    def is_available(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.tokenizer is not None


class OnlineAPIScorer(SalienceScorer):
    """Online API-based salience scorer.
    
    Calls external APIs (Hugging Face Inference, OpenAI, Cohere) for
    high-accuracy salience scoring.
    """
    
    def __init__(self, api_type: str = "huggingface", api_key: Optional[str] = None):
        """
        Initialize online API scorer.
        
        Args:
            api_type: Type of API ("huggingface", "openai", "cohere")
            api_key: API key (falls back to env variable)
        """
        self.api_type = api_type
        self.api_key = api_key or self._get_api_key(api_type)
        self.cache = {}  # Simple cache to reduce API calls
        
    def _get_api_key(self, api_type: str) -> Optional[str]:
        """Retrieve API key from environment."""
        env_map = {
            "huggingface": "HF_API_TOKEN",
            "openai": "OPENAI_API_KEY",
            "cohere": "COHERE_API_KEY"
        }
        return os.getenv(env_map.get(api_type, ""))
    
    def score(self, text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Compute salience via external API."""
        if not self.is_available():
            logger.warning("Online API not available")
            return 0.5
        
        # Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        try:
            score = self._call_api(text)
            self.cache[text_hash] = score
            return score
        except Exception as e:
            logger.error(f"Online API scoring failed: {e}")
            return 0.5
    
    def _call_api(self, text: str) -> float:
        """Execute the actual API call."""
        import requests
        
        if self.api_type == "huggingface":
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.post(
                "https://api-inference.huggingface.co/models/facebook/bart-large-mnli",
                headers=headers,
                json={"inputs": f"This text is important: {text}", "parameters": {"candidate_labels": ["important", "unimportant"]}}
            )
            result = response.json()
            # Extract importance score
            scores = result.get("scores", [0.5, 0.5])
            return scores[0] if scores else 0.5
            
        elif self.api_type == "openai":
            import openai
            openai.api_key = self.api_key
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"Rate the importance of this text from 0 to 1 (0=noise, 1=critical):\n\n{text}\n\nScore:",
                max_tokens=10,
                temperature=0
            )
            try:
                return float(response.choices[0].text.strip())
            except:
                return 0.5
                
        else:
            raise ValueError(f"Unsupported API type: {self.api_type}")
    
    def is_available(self) -> bool:
        """Check if API key is configured."""
        return self.api_key is not None


class LightweightScorer(SalienceScorer):
    """Keyword-based lightweight salience scorer.
    
    Uses heuristic rules, keyword matching, and simple NLP (TF-IDF, stopwords)
    for minimal resource usage. Suitable for edge devices or high-throughput needs.
    """
    
    def __init__(self, keyword_file: Optional[str] = None):
        """
        Initialize lightweight scorer.
        
        Args:
            keyword_file: Path to JSON file with importance keywords
        """
        self.importance_keywords = self._load_keywords(keyword_file)
        self.stopwords = set(["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"])
    
    def _load_keywords(self, keyword_file: Optional[str]) -> Dict[str, float]:
        """Load keyword weights from file or use defaults."""
        default_keywords = {
            "urgent": 0.9, "critical": 1.0, "important": 0.8,
            "asap": 0.9, "action required": 0.9, "deadline": 0.8,
            "alert": 0.85, "error": 0.7, "failed": 0.7,
            "success": 0.6, "complete": 0.6, "update": 0.5,
            "new": 0.4, "change": 0.4, "info": 0.3
        }
        
        if keyword_file and os.path.exists(keyword_file):
            try:
                with open(keyword_file, 'r') as f:
                    user_keywords = json.load(f)
                    default_keywords.update(user_keywords)
            except Exception as e:
                logger.warning(f"Could not load keyword file: {e}")
        
        return default_keywords
    
    def score(self, text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Compute salience using keyword matching and heuristics."""
        text_lower = text.lower()
        
        # Factor 1: Keyword matching
        max_keyword_score = 0.0
        for keyword, weight in self.importance_keywords.items():
            if keyword in text_lower:
                max_keyword_score = max(max_keyword_score, weight)
        
        # Factor 2: Length normalization (neither too short nor too long)
        length = len(text)
        if length < 20:
            length_factor = 0.2
        elif length < 100:
            length_factor = 0.5
        elif length < 500:
            length_factor = 0.8
        else:
            length_factor = 0.6
        
        # Factor 3: Unique word ratio (less stopwords = more informative)
        words = text_lower.split()
        if words:
            content_words = [w for w in words if w not in self.stopwords]
            uniqueness = len(set(content_words)) / max(1, len(content_words))
        else:
            uniqueness = 0
        
        # Combine factors
        salience = (max_keyword_score * 0.6) + (length_factor * 0.2) + (uniqueness * 0.2)
        return min(1.0, salience)
    
    def is_available(self) -> bool:
        """Lightweight mode is always available."""
        return True


class SalienceGate:
    """Main salience gate interface.
    
    Filters information based on salience scores, routing important
    content to memory and discarding noise.
    """
    
    def __init__(self, mode: SalienceMode = SalienceMode.LIGHTWEIGHT, threshold: float = 0.5):
        """
        Initialize salience gate.
        
        Args:
            mode: Operational mode (LOCAL, ONLINE, LIGHTWEIGHT)
            threshold: Minimum score to consider as salient (0.0-1.0)
        """
        self.mode = mode
        self.threshold = threshold
        self.scorer = self._create_scorer()
        self.stats = {"processed": 0, "passed": 0, "blocked": 0}
        
    def _create_scorer(self) -> SalienceScorer:
        """Factory method to create appropriate scorer."""
        if self.mode == SalienceMode.LOCAL:
            return LocalMLScorer()
        elif self.mode == SalienceMode.ONLINE:
            return OnlineAPIScorer()
        else:  # LIGHTWEIGHT
            return LightweightScorer()
    
    def evaluate(self, text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute salience score for input text.
        
        Args:
            text: Input text to evaluate
            context: Optional context (e.g., user, session)
            
        Returns:
            Salience score between 0.0 and 1.0
        """
        if not text or not text.strip():
            return 0.0
        
        score = self.scorer.score(text, context)
        self.stats["processed"] += 1
        
        if score >= self.threshold:
            self.stats["passed"] += 1
        else:
            self.stats["blocked"] += 1
        
        return score
    
    def is_salient(self, text: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determine if text should be retained.
        
        Args:
            text: Input text to evaluate
            context: Optional context
            
        Returns:
            True if salience score >= threshold
        """
        return self.evaluate(text, context) >= self.threshold
    
    def filter_batch(self, texts: List[str], context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Filter multiple texts, returning only salient ones.
        
        Args:
            texts: List of text strings
            context: Optional context
            
        Returns:
            List of salient texts
        """
        return [t for t in texts if self.is_salient(t, context)]
    
    def get_stats(self) -> Dict[str, Any]:
        """Return processing statistics."""
        return {
            "mode": self.mode.value,
            "threshold": self.threshold,
            **self.stats,
            "pass_rate": self.stats["passed"] / max(1, self.stats["processed"])
        }
    
    def set_threshold(self, threshold: float):
        """Adjust salience threshold dynamically."""
        if 0.0 <= threshold <= 1.0:
            self.threshold = threshold
            logger.info(f"Salience threshold updated to {threshold}")
        else:
            raise ValueError("Threshold must be between 0.0 and 1.0")
    
    def switch_mode(self, mode: SalienceMode):
        """Switch operational mode at runtime."""
        self.mode = mode
        self.scorer = self._create_scorer()
        logger.info(f"Switched to {mode.value} mode")


# Convenience function for quick integration
def create_salience_gate(
    mode: str = "lightweight", 
    threshold: float = 0.5,
    api_type: Optional[str] = None,
    model_path: Optional[str] = None
) -> SalienceGate:
    """
    Factory function to create a configured salience gate.
    
    Args:
        mode: "local", "online", or "lightweight"
        threshold: Minimum salience score (0.0-1.0)
        api_type: For online mode ("huggingface", "openai", "cohere")
        model_path: For local mode (path to model)
        
    Returns:
        Configured SalienceGate instance
    """
    mode_map = {
        "local": SalienceMode.LOCAL,
        "online": SalienceMode.ONLINE,
        "lightweight": SalienceMode.LIGHTWEIGHT
    }
    
    selected_mode = mode_map.get(mode.lower(), SalienceMode.LIGHTWEIGHT)
    gate = SalienceGate(mode=selected_mode, threshold=threshold)
    
    # Override scorer configuration if needed
    if selected_mode == SalienceMode.LOCAL and model_path:
        gate.scorer = LocalMLScorer(model_path)
    elif selected_mode == SalienceMode.ONLINE and api_type:
        gate.scorer = OnlineAPIScorer(api_type)
    
    return gate