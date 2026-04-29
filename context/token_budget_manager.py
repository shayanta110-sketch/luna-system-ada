# context/token_budget_manager.py
"""
TokenBudgetManager for Ada - Manages token budgets for models with small context windows.
Implements exponential decay, importance scoring, and query-aware compression.
"""

import math
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class TokenItem:
    """Represents a chunk of text with its metadata."""
    content: str
    token_count: int
    importance_score: float = 0.0
    access_count: int = 0
    last_accessed: float = 0.0
    tags: List[str] = field(default_factory=list)


class TokenBudgetManager:
    """
    Manages token budget for small-context models using exponential decay,
    importance scoring, query-aware compression, and retrieval reranking.
    """

    def __init__(self, max_tokens: int = 4096, decay_rate: float = 0.1):
        """
        Initialize TokenBudgetManager.

        Args:
            max_tokens: Maximum allowed tokens in budget
            decay_rate: Rate at which importance decays over time (0.1 = slow decay)
        """
        self.max_tokens = max_tokens
        self.decay_rate = decay_rate
        self.items: List[TokenItem] = []
        self.global_step: int = 1  # Start at 1 to avoid zero-step conflict

    def add_item(self, content: str, token_count: int, tags: List[str] = None) -> None:
        """Add a new token item with initial importance."""
        tags = tags or []
        importance = self._compute_initial_importance(content, tags)
        item = TokenItem(
            content=content,
            token_count=token_count,
            importance_score=importance,
            tags=tags,
            last_accessed=self.global_step  # Set initial last_accessed to current step
        )
        self.items.append(item)
        self._enforce_budget()

    def _compute_initial_importance(self, content: str, tags: List[str]) -> float:
        """
        Compute initial importance based on content length and tags.

        Args:
            content: The text content
            tags: List of tags associated with the content

        Returns:
            Initial importance score (range: 0.0 to 1.0)
        """
        length_factor = min(1.0, len(content) / 1000.0)
        tag_bonus = min(0.3, 0.1 * len(tags)) if tags else 0
        # Base importance: 0.3 + length_factor (0-0.7) + tag_bonus (0-0.3)
        return 0.3 + length_factor * 0.4 + tag_bonus  # Max = 0.3 + 0.4 + 0.3 = 1.0

    def _exponential_decay(self, item: TokenItem, current_step: int) -> float:
        """
        Apply exponential decay to item's importance based on time since last access.

        Args:
            item: TokenItem to compute decay for
            current_step: Current global step number

        Returns:
            Decayed importance score
        """
        time_since_access = current_step - item.last_accessed
        if time_since_access <= 0:
            return item.importance_score
        decay = math.exp(-self.decay_rate * time_since_access)
        return item.importance_score * decay

    def update_access(self, index: int) -> None:
        """Update access statistics for an item."""
        if 0 <= index < len(self.items):
            item = self.items[index]
            item.access_count += 1
            item.last_accessed = self.global_step
            # After each access, importance_score should be recomputed based on current decay
            # to avoid stale values in future decay calculations.
            item.importance_score = self._exponential_decay(item, self.global_step)
            self.global_step += 1

    def _enforce_budget(self) -> None:
        """Remove lowest importance items if total token count exceeds budget."""
        # First, update all importance scores with current step
        for item in self.items:
            item.importance_score = self._exponential_decay(item, self.global_step)

        total_tokens = sum(item.token_count for item in self.items)
        while total_tokens > self.max_tokens and self.items:
            # Find the least important item
            min_idx = min(range(len(self.items)), key=lambda i: self.items[i].importance_score)
            removed = self.items.pop(min_idx)
            total_tokens -= removed.token_count

    def query_aware_compression(self, query: str, target_tokens: int) -> str:
        """
        Compress items based on query relevance and importance.

        Args:
            query: The user query to compute relevance against
            target_tokens: Maximum tokens allowed in compressed context

        Returns:
            A concatenated compressed context string
        """
        if not self.items:
            return ""

        # Update all importance scores to current step values
        for item in self.items:
            item.importance_score = self._exponential_decay(item, self.global_step)

        query_terms = set(query.lower().split())
        scored_items = []

        for item in self.items:
            content_terms = set(item.content.lower().split())
            overlap = len(query_terms & content_terms)
            relevance = overlap / max(len(query_terms), 1)
            combined_score = 0.7 * item.importance_score + 0.3 * relevance
            scored_items.append((combined_score, item))

        # Sort by combined score descending
        scored_items.sort(key=lambda x: x[0], reverse=True)

        selected = []
        total_tokens = 0

        for score, item in scored_items:
            if total_tokens + item.token_count <= target_tokens:
                selected.append(item.content)
                total_tokens += item.token_count

        return " ".join(selected)

    def get_context(self, query: str = "", max_tokens: Optional[int] = None) -> str:
        """
        Get the current context as a string, optionally compressed.

        Args:
            query: Query for relevance-based compression (if empty, returns all items)
            max_tokens: Maximum tokens allowed (defaults to self.max_tokens)

        Returns:
            Context string suitable for model input
        """
        budget = max_tokens if max_tokens is not None else self.max_tokens

        if query:
            return self.query_aware_compression(query, budget)
        else:
            # Simple concatenation without compression
            total_tokens = sum(item.token_count for item in self.items)
            if total_tokens <= budget:
                return " ".join(item.content for item in self.items)
            else:
                # Fall back to relevance-based compression with empty query
                return self.query_aware_compression("", budget)

    def get_items(self) -> List[Dict[str, Any]]:
        """
        Get list of items as dictionaries for inspection.

        Returns:
            List of item dictionaries
        """
        return [
            {
                "content": item.content[:100] + "..." if len(item.content) > 100 else item.content,
                "token_count": item.token_count,
                "importance_score": item.importance_score,
                "access_count": item.access_count,
                "tags": item.tags
            }
            for item in self.items
        ]

    def clear(self) -> None:
        """Clear all items from the manager."""
        self.items.clear()
        self.global_step = 1

    def get_memory_usage_mb(self) -> float:
        """
        Estimate memory usage of stored items in MB.

        Returns:
            Estimated memory usage in MB
        """
        total_chars = sum(len(item.content) for item in self.items)
        # Rough estimate: 1 char = 2 bytes in Python string
        bytes_used = total_chars * 2
        for item in self.items:
            bytes_used += sum(len(tag) for tag in item.tags) * 2
        return bytes_used / (1024 * 1024)


# Example usage
if __name__ == "__main__":
    # Example: For your Core i7-3770 with 16GB RAM and limited VRAM
    # Small context window for lightweight models (1024 tokens)
    manager = TokenBudgetManager(max_tokens=1024, decay_rate=0.05)

    # Add some items
    manager.add_item("System prompt: You are a helpful assistant.", token_count=15)
    manager.add_item("User memory: User prefers short answers.", token_count=10, tags=["user_preference"])
    manager.add_item("Conversation: Q: What is AI? A: Artificial Intelligence.", token_count=20)

    # Simulate access pattern
    manager.update_access(0)  # System prompt accessed
    manager.update_access(1)  # User preference accessed

    # Query-aware compression
    context = manager.get_context(query="What is AI?", max_tokens=30)
    print(f"Compressed context: {context}")
    print(f"Memory usage: {manager.get_memory_usage_mb():.2f} MB")
    print(f"Items: {manager.get_items()}")
