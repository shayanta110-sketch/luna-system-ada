"""Token Budget module with slot-based budget enforcement and compression strategies.

Inspired by context-engine: allocate token budgets across system prompt, conversation history,
and documents. Includes exponential decay scoring, deduplication, and query-aware compression.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math


class CompressionStrategy(Enum):
    """Supported compression strategies for token budget."""
    TRUNCATE = "truncate"      # Simply truncate to budget
    SENTENCE = "sentence"      # Keep whole sentences
    EXTRACTIVE = "extractive"  # Keep most important sentences based on scoring


@dataclass
class BudgetSlot:
    """Represents a token budget slot for a specific content type."""
    name: str
    allocated_tokens: int
    used_tokens: int = 0
    compression_strategy: CompressionStrategy = CompressionStrategy.TRUNCATE


@dataclass
class ContentItem:
    """Represents a piece of content with scoring metadata."""
    text: str
    content_type: str  # 'system', 'conversation', 'document'
    timestamp: float  # Unix timestamp for decay
    score: float = 1.0
    hash_id: Optional[str] = None  # For deduplication


class TokenBudget:
    """Manages token budgets across multiple slots with compression and scoring."""

    def __init__(self, total_budget: int, system_slot: int, conversation_slot: int, document_slot: int):
        """
        Initialize token budget with slot allocations.

        Args:
            total_budget: Total token limit
            system_slot: Tokens allocated to system prompt
            conversation_slot: Tokens allocated to conversation history
            document_slot: Tokens allocated to documents
        """
        self.total_budget = total_budget
        self.slots = {
            'system': BudgetSlot('system', system_slot),
            'conversation': BudgetSlot('conversation', conversation_slot),
            'document': BudgetSlot('document', document_slot)
        }
        self.items: List[ContentItem] = []
        self.decay_factor = 0.95  # Exponential decay per time unit
        self.similarity_threshold = 0.85  # For deduplication

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token). Override with actual tokenizer."""
        return len(text) // 4

    def add_item(self, text: str, content_type: str, timestamp: float, item_hash: Optional[str] = None):
        """Add a content item with optional hash for deduplication."""
        # Deduplication check
        if item_hash and self._is_duplicate(item_hash):
            return

        item = ContentItem(
            text=text,
            content_type=content_type,
            timestamp=timestamp,
            hash_id=item_hash
        )
        self.items.append(item)

    def _is_duplicate(self, item_hash: str) -> bool:
        """Check if item with same hash already exists."""
        return any(item.hash_id == item_hash for item in self.items if item.hash_id)

    def _apply_exponential_decay(self, item: ContentItem, current_time: float) -> float:
        """Apply exponential decay scoring based on timestamp."""
        age = current_time - item.timestamp
        decay = math.exp(-self.decay_factor * age) if age >= 0 else 1.0
        return item.score * decay

    def _score_sentence(self, sentence: str, query: Optional[str] = None) -> float:
        """Score a sentence based on relevance to query and length."""
        score = 1.0
        if query:
            # Simple keyword matching (can be replaced with embeddings)
            query_terms = set(query.lower().split())
            sentence_terms = set(sentence.lower().split())
            intersection = query_terms.intersection(sentence_terms)
            if query_terms:
                score += len(intersection) / len(query_terms)
        # Penalize very short or very long sentences
        words = sentence.split()
        if len(words) < 3:
            score *= 0.5
        elif len(words) > 50:
            score *= 0.8
        return score

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences (basic regex)."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _compress_truncate(self, text: str, budget: int) -> str:
        """Simple truncation to token budget."""
        tokens = self.estimate_tokens(text)
        if tokens <= budget:
            return text
        # Rough truncation by character proportion
        target_chars = budget * 4
        return text[:target_chars] + "..."

    def _compress_sentence(self, text: str, budget: int) -> str:
        """Keep whole sentences until token budget is reached."""
        sentences = self._split_sentences(text)
        result = []
        current_tokens = 0
        for sent in sentences:
            sent_tokens = self.estimate_tokens(sent)
            if current_tokens + sent_tokens <= budget:
                result.append(sent)
                current_tokens += sent_tokens
            else:
                break
        return " ".join(result) + ("..." if result else "")

    def _compress_extractive(self, text: str, budget: int, query: Optional[str] = None) -> str:
        """Extract most important sentences based on scoring."""
        sentences = self._split_sentences(text)
        if not sentences:
            return ""

        # Score each sentence
        scored = [(self._score_sentence(sent, query), sent) for sent in sentences]
        scored.sort(reverse=True)  # Highest score first

        # Select sentences until budget
        result = []
        current_tokens = 0
        for score, sent in scored:
            sent_tokens = self.estimate_tokens(sent)
            if current_tokens + sent_tokens <= budget:
                result.append(sent)
                current_tokens += sent_tokens
            else:
                break

        # Restore original order for coherence
        original_order = [s for s in sentences if s in result]
        return " ".join(original_order) + ("..." if len(original_order) < len(sentences) else "")

    def compress_slot(self, slot_name: str, texts: List[str], query: Optional[str] = None, current_time: Optional[float] = None) -> str:
        """
        Compress all items in a slot to fit allocated budget with scoring.

        Args:
            slot_name: 'system', 'conversation', or 'document'
            texts: List of text items in the slot
            query: Optional query for extractive compression
            current_time: Current timestamp for decay scoring

        Returns:
            Compressed text fitting within slot budget
        """
        if slot_name not in self.slots:
            raise ValueError(f"Invalid slot: {slot_name}")

        slot = self.slots[slot_name]
        budget = slot.allocated_tokens

        if not texts:
            return ""

        # Score and sort items by exponential decay if timestamps available
        if current_time is not None:
            # Assume items are stored with timestamps; here we use provided
            scored_items = []
            for text in texts:
                # Find matching item or create temporary
                item = next((i for i in self.items if i.text == text), None)
                if item:
                    score = self._apply_exponential_decay(item, current_time)
                else:
                    score = 1.0
                scored_items.append((score, text))
            scored_items.sort(reverse=True)  # Higher score = more important
            ordered_texts = [text for _, text in scored_items]
        else:
            ordered_texts = texts

        # Combine all texts
        combined = " ".join(ordered_texts)
        estimated = self.estimate_tokens(combined)

        if estimated <= budget:
            return combined

        # Apply compression strategy
        if slot.compression_strategy == CompressionStrategy.TRUNCATE:
            return self._compress_truncate(combined, budget)
        elif slot.compression_strategy == CompressionStrategy.SENTENCE:
            return self._compress_sentence(combined, budget)
        elif slot.compression_strategy == CompressionStrategy.EXTRACTIVE:
            return self._compress_extractive(combined, budget, query)
        else:
            return self._compress_truncate(combined, budget)

    def get_budget_report(self) -> Dict[str, Any]:
        """Return budget usage report."""
        report = {}
        for name, slot in self.slots.items():
            report[name] = {
                'allocated': slot.allocated_tokens,
                'used': slot.used_tokens,
                'strategy': slot.compression_strategy.value
            }
        report['total_budget'] = self.total_budget
        return report

    def update_slot_strategy(self, slot_name: str, strategy: CompressionStrategy):
        """Update compression strategy for a slot."""
        if slot_name in self.slots:
            self.slots[slot_name].compression_strategy = strategy
