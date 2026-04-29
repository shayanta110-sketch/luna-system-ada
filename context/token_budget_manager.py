import math
import heapq
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


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
        self.max_tokens = max_tokens
        self.decay_rate = decay_rate
        self.items: List[TokenItem] = []
        self.global_step = 0

    def add_item(self, content: str, token_count: int, tags: List[str] = None) -> None:
        """Add a new token item with initial importance."""
        importance = self._compute_initial_importance(content, tags or [])
        item = TokenItem(
            content=content,
            token_count=token_count,
            importance_score=importance,
            tags=tags or []
        )
        self.items.append(item)
        self._enforce_budget()

    def _compute_initial_importance(self, content: str, tags: List[str]) -> float:
        """Compute initial importance based on content length and tags."""
        length_factor = min(1.0, len(content) / 1000)
        tag_bonus = 0.2 * len(tags) if tags else 0
        return 0.5 + length_factor * 0.3 + tag_bonus

    def _exponential_decay(self, item: TokenItem) -> float:
        """Apply exponential decay to item's importance."""
        time_since_access = self.global_step - item.last_accessed
        decay = math.exp(-self.decay_rate * time_since_access)
        return item.importance_score * decay

    def update_access(self, index: int) -> None:
        """Update access statistics for an item."""
        if 0 <= index < len(self.items):
            self.items[index].access_count += 1
            self.items[index].last_accessed = self.global_step
            self.global_step += 1

    def _enforce_budget(self) -> None:
        """Remove lowest importance items if total token count exceeds budget."""
        total_tokens = sum(item.token_count for item in self.items)
        while total_tokens > self.max_tokens and self.items:
            # Apply decay and find lowest effective importance
            for item in self.items:
                item.importance_score = self._exponential_decay(item)

            # Remove the least important item
            min_idx = min(range(len(self.items)), key=lambda i: self.items[i].importance_score)
            removed = self.items.pop(min_idx)
            total_tokens -= removed.token_count

    def query_aware_compression(self, query: str, target_tokens: int) -> str:
        """
        Compress items based on query relevance and importance.
        Returns a concatenated compressed context string.
        """
        if not self.items:
            return ""

        # Score each item based on query semantic relevance (simplified TF-IDF style)
        query_terms = set(query.lower().split())
        scored_items = []
        for item in self.items:
            content_terms = set(item.content.lower().split())
            overlap = len(query_terms & content_terms)
            relevance = overlap / max(len(query_terms), 1)
            combined_score = 0.7 * item.importance_score + 0.3 * relevance
            scored_items.append((combined_score, item))

        # Sort by combined score and select items until token limit
        scored_items.sort(key=lambda x: x[0], reverse=True)
        selected = []
        total = 0
        for score, item in scored_items:
            if total + item.token_count <= target_tokens:
                selected.append(item.content)
                total += item.token_count
            else:
                # Partial inclusion if important
                remaining = target_tokens - total
                if remaining > 10 and score > 0.5:
                    truncated = item.content[:remaining * 4]  # rough char to token
                    selected.append(truncated)
                    break
        return "\n\n".join(selected)

    def retrieve_and_rerank(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k items by relevance and rerank using importance."""
        query_terms = set(query.lower().split())
        candidates = []
        for item in self.items:
            content_terms = set(item.content.lower().split())
            overlap = len(query_terms & content_terms)
            relevance = overlap / max(len(query_terms), 1)
            effective_importance = self._exponential_decay(item)
            final_score = 0.5 * relevance + 0.5 * effective_importance
            candidates.append((final_score, item))

        # Sort by final score and return top_k
        candidates.sort(key=lambda x: x[0], reverse=True)
        result = []
        for score, item in candidates[:top_k]:
            result.append((item.content, score))
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Return current budget statistics."""
        total_tokens = sum(item.token_count for item in self.items)
        avg_importance = np.mean([item.importance_score for item in self.items]) if self.items else 0
        return {
            "total_items": len(self.items),
            "total_tokens": total_tokens,
            "max_tokens": self.max_tokens,
            "avg_importance": float(avg_importance),
            "global_step": self.global_step
        }

    def clear(self) -> None:
        """Clear all items."""
        self.items.clear()
        self.global_step = 0
