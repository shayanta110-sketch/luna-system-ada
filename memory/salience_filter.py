"""Salience filtering module for intelligent memory retention.

Implements salience-based filtering to identify and store only important
information from memory streams, reducing noise and preserving critical
context for downstream reasoning.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class SalienceSignal(Enum):
    """Types of salience signals that can influence memory importance."""
    RECENCY = "recency"
    FREQUENCY = "frequency"
    RECENCY_IN_DIALOG = "recency_in_dialog"
    USER_ACTION = "user_action"
    TASK_RELEVANCE = "task_relevance"
    EMOTIONAL_INTENSITY = "emotional_intensity"
    NOVELTY = "novelty"
    CONTRADICTION = "contradiction"
    QUESTION = "question"
    ANSWER = "answer"
    COMMAND = "command"
    ENTITY_MENTION = "entity_mention"


@dataclass
class SalienceConfig:
    """Configuration for salience scoring and filtering."""
    recency_decay: float = 0.95
    frequency_boost_max: float = 0.3
    user_action_weight: float = 0.4
    task_relevance_weight: float = 0.5
    novelty_threshold: float = 0.2
    contradiction_boost: float = 0.25
    question_boost: float = 0.15
    answer_boost: float = 0.2
    command_boost: float = 0.35
    entity_mention_boost: float = 0.1
    min_overall_score: float = 0.0
    max_overall_score: float = 1.0
    default_pass_threshold: float = 0.4


@dataclass
class SalienceResult:
    """Result of salience scoring for a memory entry."""
    score: float
    passed: bool
    signals_used: Dict[str, float]
    reason: Optional[str] = None


class SalienceGate:
    """Gatekeeper that identifies and filters important memory entries.
    
    The SalienceGate evaluates incoming memory entries across multiple
    salience signals and decides which entries should be stored in
    long-term memory and which can be discarded or compressed.
    """
    
    def __init__(self, config: Optional[SalienceConfig] = None):
        """Initialize the salience gate with optional configuration.
        
        Args:
            config: Configuration parameters for scoring and filtering.
        """
        self.config = config or SalienceConfig()
        self._entry_history: List[Dict[str, Any]] = []
        self._frequency_counter: Dict[str, int] = {}
        self._recent_entities: Dict[str, float] = {}  # entity -> last timestamp
        
    def score_memory_entry(
        self,
        entry: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SalienceResult:
        """Compute salience score for a single memory entry.
        
        Args:
            entry: Memory entry dict with fields like 'content', 'type',
                  'timestamp', 'entities', 'user_action', etc.
            context: Optional contextual information (current task, user goals).
            
        Returns:
            SalienceResult with computed score and pass decision.
        """
        signals = {}
        
        # 1. Recency (time-based decay)
        recency_score = self._compute_recency(entry)
        signals[SalienceSignal.RECENCY.value] = recency_score
        
        # 2. Frequency (how often similar content appears)
        freq_score = self._compute_frequency(entry)
        signals[SalienceSignal.FREQUENCY.value] = freq_score
        
        # 3. User action (if entry is a result of explicit user action)
        user_action_score = self._compute_user_action_boost(entry)
        signals[SalienceSignal.USER_ACTION.value] = user_action_score
        
        # 4. Task relevance (match with current task/goals)
        task_score = self._compute_task_relevance(entry, context)
        signals[SalienceSignal.TASK_RELEVANCE.value] = task_score
        
        # 5. Novelty (how unexpected/new this information is)
        novelty_score = self._compute_novelty(entry)
        signals[SalienceSignal.NOVELTY.value] = novelty_score
        
        # 6. Contradiction (if entry contradicts existing memory)
        contradiction_score = self._compute_contradiction(entry)
        signals[SalienceSignal.CONTRADICTION.value] = contradiction_score
        
        # 7. Dialog act based boosts
        dialog_scores = self._compute_dialog_boosts(entry)
        signals.update(dialog_scores)
        
        # 8. Entity mention boost
        entity_score = self._compute_entity_boost(entry)
        signals[SalienceSignal.ENTITY_MENTION.value] = entity_score
        
        # Combine all signals into overall score
        overall = self._combine_signals(signals)
        
        # Clamp to valid range
        overall = max(self.config.min_overall_score, 
                     min(self.config.max_overall_score, overall))
        
        passed = overall >= self.config.default_pass_threshold
        
        return SalienceResult(
            score=overall,
            passed=passed,
            signals_used=signals,
            reason=self._generate_reason(overall, signals, passed)
        )
    
    def filter_memories(
        self,
        entries: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], SalienceResult]]:
        """Filter a list of memory entries, returning only important ones.
        
        Args:
            entries: List of memory entries to evaluate.
            context: Optional contextual information.
            
        Returns:
            List of (entry, result) tuples for entries that passed the gate.
        """
        results = []
        for entry in entries:
            result = self.score_memory_entry(entry, context)
            if result.passed:
                results.append((entry, result))
        return results
    
    def update_after_filtering(
        self,
        passed_entries: List[Tuple[Dict[str, Any], SalienceResult]]
    ) -> None:
        """Update internal state after filtering (frequency counters, etc.).
        
        Args:
            passed_entries: List of (entry, result) pairs that were stored.
        """
        for entry, _ in passed_entries:
            self._update_frequency(entry)
            self._entry_history.append(entry)
            self._update_entity_tracking(entry)
    
    def _compute_recency(self, entry: Dict[str, Any]) -> float:
        """Compute recency score based on timestamp decay."""
        timestamp = entry.get("timestamp", 0.0)
        if not self._entry_history:
            return 1.0
        
        # Compare with most recent entry
        latest_ts = max(e.get("timestamp", 0.0) for e in self._entry_history)
        if timestamp >= latest_ts:
            return 1.0
        
        time_diff = latest_ts - timestamp
        decay = self.config.recency_decay ** (time_diff / 60.0)  # per-minute decay
        return decay
    
    def _compute_frequency(self, entry: Dict[str, Any]) -> float:
        """Compute frequency-based salience boost."""
        content_hash = self._get_content_hash(entry)
        count = self._frequency_counter.get(content_hash, 0)
        # Logarithmic scaling to avoid runaway boosts
        boost = math.log1p(count) / math.log1p(10) if count > 0 else 0
        return min(boost * self.config.frequency_boost_max, self.config.frequency_boost_max)
    
    def _compute_user_action_boost(self, entry: Dict[str, Any]) -> float:
        """Boost for entries caused by explicit user actions."""
        if entry.get("user_action", False):
            return self.config.user_action_weight
        return 0.0
    
    def _compute_task_relevance(
        self,
        entry: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Compute relevance to current task."""
        if not context or "task_keywords" not in context:
            return 0.0
        
        task_keywords = set(context.get("task_keywords", []))
        entry_text = entry.get("content", "").lower()
        
        # Simple keyword matching
        matches = sum(1 for kw in task_keywords if kw.lower() in entry_text)
        if not task_keywords:
            return 0.0
        
        relevance = matches / len(task_keywords)
        return relevance * self.config.task_relevance_weight
    
    def _compute_novelty(self, entry: Dict[str, Any]) -> float:
        """Compute novelty score based on similarity to past entries."""
        if not self._entry_history:
            return 1.0
        
        # Simple Jaccard similarity on words (simplified)
        entry_words = set(entry.get("content", "").lower().split())
        if not entry_words:
            return 0.5
        
        max_similarity = 0.0
        for past_entry in self._entry_history[-20:]:  # Check recent history
            past_words = set(past_entry.get("content", "").lower().split())
            if not past_words:
                continue
            intersection = len(entry_words & past_words)
            union = len(entry_words | past_words)
            similarity = intersection / union if union > 0 else 0
            max_similarity = max(max_similarity, similarity)
        
        novelty = 1.0 - max_similarity
        # If novelty is low (high similarity), apply boost if above threshold?
        # Actually novelty itself is the signal: higher novelty = more salient
        if novelty < self.config.novelty_threshold:
            return 0.0
        return min(novelty, 1.0)
    
    def _compute_contradiction(self, entry: Dict[str, Any]) -> float:
        """Boost for entries that contradict existing memory."""
        if not self._entry_history:
            return 0.0
        
        # Simple contradiction heuristic: same entity but opposite sentiment
        entity = entry.get("entity")
        if not entity:
            return 0.0
        
        # Check if we have stored past info about this entity
        for past in self._entry_history[-30:]:
            if past.get("entity") == entity:
                if past.get("sentiment") and entry.get("sentiment"):
                    if past["sentiment"] != entry["sentiment"]:
                        return self.config.contradiction_boost
        return 0.0
    
    def _compute_dialog_boosts(self, entry: Dict[str, Any]) -> Dict[str, float]:
        """Compute dialog-specific boosts (questions, answers, commands)."""
        boosts = {}
        entry_type = entry.get("type", "").lower()
        
        if entry_type == "question":
            boosts[SalienceSignal.QUESTION.value] = self.config.question_boost
        elif entry_type == "answer":
            boosts[SalienceSignal.ANSWER.value] = self.config.answer_boost
        elif entry_type == "command":
            boosts[SalienceSignal.COMMAND.value] = self.config.command_boost
        else:
            boosts[SalienceSignal.QUESTION.value] = 0.0
            boosts[SalienceSignal.ANSWER.value] = 0.0
            boosts[SalienceSignal.COMMAND.value] = 0.0
        
        return boosts
    
    def _compute_entity_boost(self, entry: Dict[str, Any]) -> float:
        """Boost for entries mentioning important entities."""
        entities = entry.get("entities", [])
        if not entities:
            return 0.0
        
        # Track entity recency
        now = entry.get("timestamp", 0.0)
        boost = 0.0
        for ent in entities:
            last_seen = self._recent_entities.get(ent, now - 1e6)
            recency = now - last_seen
            # Entities not seen recently get small boost
            if recency > 3600:  # More than 1 hour
                boost += 0.05
            self._recent_entities[ent] = now
        
        return min(boost, self.config.entity_mention_boost)
    
    def _combine_signals(self, signals: Dict[str, float]) -> float:
        """Combine individual salience signals into overall score."""
        # Weighted sum + baseline
        weights = {
            SalienceSignal.RECENCY.value: 0.20,
            SalienceSignal.FREQUENCY.value: 0.10,
            SalienceSignal.USER_ACTION.value: 0.15,
            SalienceSignal.TASK_RELEVANCE.value: 0.25,
            SalienceSignal.NOVELTY.value: 0.10,
            SalienceSignal.CONTRADICTION.value: 0.05,
            SalienceSignal.QUESTION.value: 0.04,
            SalienceSignal.ANSWER.value: 0.04,
            SalienceSignal.COMMAND.value: 0.04,
            SalienceSignal.ENTITY_MENTION.value: 0.03,
        }
        
        total = 0.0
        for signal, value in signals.items():
            weight = weights.get(signal, 0.0)
            total += value * weight
        
        return total
    
    def _generate_reason(
        self,
        score: float,
        signals: Dict[str, float],
        passed: bool
    ) -> str:
        """Generate human-readable reason for salience decision."""
        if passed:
            top_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join([f"{sig}={val:.2f}" for sig, val in top_signals if val > 0])
            return f"Passed (score={score:.2f}) due to {top_str if top_str else 'baseline'}"
        else:
            return f"Failed (score={score:.2f}) below threshold {self.config.default_pass_threshold}"
    
    def _get_content_hash(self, entry: Dict[str, Any]) -> str:
        """Generate a simple hash for frequency counting."""
        content = entry.get("content", "")
        return content[:100]  # Truncate for simplicity
    
    def _update_frequency(self, entry: Dict[str, Any]) -> None:
        """Update frequency counter for an entry."""
        content_hash = self._get_content_hash(entry)
        self._frequency_counter[content_hash] = self._frequency_counter.get(content_hash, 0) + 1
    
    def _update_entity_tracking(self, entry: Dict[str, Any]) -> None:
        """Update entity tracking information."""
        entities = entry.get("entities", [])
        timestamp = entry.get("timestamp", 0.0)
        for ent in entities:
            self._recent_entities[ent] = timestamp
    
    def reset(self) -> None:
        """Reset internal state (history, frequency counters, etc.)."""
        self._entry_history.clear()
        self._frequency_counter.clear()
        self._recent_entities.clear()


def quick_filter(
    entry: Dict[str, Any],
    threshold: float = 0.4,
    context: Optional[Dict[str, Any]] = None
) -> bool:
    """Convenience function to quickly filter a single memory entry.
    
    Args:
        entry: Memory entry to evaluate.
        threshold: Minimum salience score to pass.
        context: Optional context.
        
    Returns:
        True if entry should be kept, False otherwise.
    """
    gate = SalienceGate(SalienceConfig(default_pass_threshold=threshold))
    result = gate.score_memory_entry(entry, context)
    return result.passed
