"""Salience filtering module for intelligent memory retention. Implements
salience-based filtering to identify and store only important information from
memory streams, reducing noise and preserving critical context for downstream
reasoning.
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple
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
    recency_half_life_seconds: float = 60.0   # Half-life for recency decay (seconds)
    frequency_boost_max: float = 0.3
    user_action_weight: float = 0.4
    task_relevance_weight: float = 0.5
    novelty_threshold: float = 0.2
    novelty_boost_factor: float = 2.0          # Boost multiplier for novelty (new info)
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
    """Gatekeeper that identifies and filters important memory entries."""

    def __init__(self, config: Optional[SalienceConfig] = None):
        self.config = config or SalienceConfig()
        self._entry_history: List[Dict[str, Any]] = []
        self._frequency_counter: Dict[str, int] = {}
        self._recent_entities: Dict[str, float] = {}

    def score_memory_entry(
        self,
        entry: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SalienceResult:
        """Compute salience score for a single memory entry."""
        signals = {}

        signals[SalienceSignal.RECENCY.value] = self._compute_recency(entry)
        signals[SalienceSignal.FREQUENCY.value] = self._compute_frequency(entry)
        signals[SalienceSignal.USER_ACTION.value] = self._compute_user_action_boost(entry)
        signals[SalienceSignal.TASK_RELEVANCE.value] = self._compute_task_relevance(entry, context)
        signals[SalienceSignal.NOVELTY.value] = self._compute_novelty(entry)
        signals[SalienceSignal.CONTRADICTION.value] = self._compute_contradiction_boost(entry)
        dialog_scores = self._compute_dialog_boosts(entry)
        signals.update(dialog_scores)
        signals[SalienceSignal.ENTITY_MENTION.value] = self._compute_entity_boost(entry)

        overall = self._combine_signals(signals)
        overall = max(self.config.min_overall_score, min(self.config.max_overall_score, overall))
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
        """Filter a list of memory entries, returning only important ones."""
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
        """Update internal state after filtering."""
        for entry, _ in passed_entries:
            self._update_frequency(entry)
            self._entry_history.append(entry)
            self._update_entity_tracking(entry)

    def _compute_recency(self, entry: Dict[str, Any]) -> float:
        """Compute recency score based on exponential decay across all previous entries."""
        timestamp = entry.get("timestamp", time.time())
        # Use half-life formula: decay = 0.5^(dt / half_life)
        dt = time.time() - timestamp
        decay = 0.5 ** (dt / self.config.recency_half_life_seconds)
        return decay

    def _compute_frequency(self, entry: Dict[str, Any]) -> float:
        """Compute frequency-based salience boost."""
        content_hash = self._get_content_hash(entry)
        count = self._frequency_counter.get(content_hash, 0)
        # Boost scaling: log1p(count) / log1p(max_count)
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
        matches = sum(1 for kw in task_keywords if kw.lower() in entry_text)
        if not task_keywords:
            return 0.0
        relevance = matches / len(task_keywords)
        return relevance * self.config.task_relevance_weight

    def _compute_novelty(self, entry: Dict[str, Any]) -> float:
        """Compute novelty score based on similarity to past entries.
        Higher novelty = more salient (boost applied).
        """
        if not self._entry_history:
            return 1.0

        entry_words = set(entry.get("content", "").lower().split())
        if not entry_words:
            return 0.5

        max_similarity = 0.0
        for past_entry in self._entry_history[-20:]:
            past_words = set(past_entry.get("content", "").lower().split())
            if not past_words:
                continue
            intersection = len(entry_words & past_words)
            union = len(entry_words | past_words)
            similarity = intersection / union if union > 0 else 0
            max_similarity = max(max_similarity, similarity)

        novelty = 1.0 - max_similarity
        if novelty < self.config.novelty_threshold:
            return 0.0
        # Boost novel entries
        boosted = novelty * self.config.novelty_boost_factor
        return min(boosted, 1.0)

    def _compute_contradiction_boost(self, entry: Dict[str, Any]) -> float:
        """Boost for entries that contradict existing memory."""
        if not self._entry_history:
            return 0.0
        # Contradiction detection placeholder.
        # In practice, can use entity tracking or semantic similarity.
        return 0.0

    def _compute_dialog_boosts(self, entry: Dict[str, Any]) -> Dict[str, float]:
        """Compute dialog-specific boosts."""
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

        now = entry.get("timestamp", time.time())
        boost = 0.0
        for ent in entities:
            last_seen = self._recent_entities.get(ent, now - 3600)
            recency = now - last_seen
            if recency > 3600:          # More than 1 hour
                boost += 0.05
            self._recent_entities[ent] = now
        return min(boost, self.config.entity_mention_boost)

    def _combine_signals(self, signals: Dict[str, float]) -> float:
        """Combine individual salience signals into a normalized overall score."""
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
        total_weight = sum(weights.values())
        weighted_sum = 0.0
        for signal, value in signals.items():
            weight = weights.get(signal, 0.0)
            weighted_sum += value * weight
        # Normalize to [0,1] accounting for total weight
        # Since total_weight = 1.0, normalization is effectively identity
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _generate_reason(self, score: float, signals: Dict[str, float], passed: bool) -> str:
        """Generate human-readable reason for salience decision."""
        if passed:
            top_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join(f"{sig}={val:.2f}" for sig, val in top_signals if val > 0)
            return f"Passed (score={score:.2f}) due to {top_str if top_str else 'baseline'}"
        else:
            return f"Failed (score={score:.2f}) below threshold {self.config.default_pass_threshold}"

    def _get_content_hash(self, entry: Dict[str, Any]) -> str:
        """Generate a simple hash for frequency counting."""
        content = entry.get("content", "")
        # Use a fixed-length prefix as a simple fingerprint
        return content[:100]

    def _update_frequency(self, entry: Dict[str, Any]) -> None:
        """Update frequency counter for an entry."""
        content_hash = self._get_content_hash(entry)
        self._frequency_counter[content_hash] = self._frequency_counter.get(content_hash, 0) + 1

    def _update_entity_tracking(self, entry: Dict[str, Any]) -> None:
        """Update entity tracking information."""
        entities = entry.get("entities", [])
        timestamp = entry.get("timestamp", time.time())
        for ent in entities:
            self._recent_entities[ent] = timestamp

    def reset(self) -> None:
        """Reset internal state."""
        self._entry_history.clear()
        self._frequency_counter.clear()
        self._recent_entities.clear()


def quick_filter(
    entry: Dict[str, Any],
    threshold: float = 0.4,
    context: Optional[Dict[str, Any]] = None
) -> bool:
    """Convenience function to quickly filter a single memory entry."""
    gate = SalienceGate(SalienceConfig(default_pass_threshold=threshold))
    result = gate.score_memory_entry(entry, context)
    return result.passed
