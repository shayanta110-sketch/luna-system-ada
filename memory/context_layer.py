"""Memory & Context Layer - Unified orchestration layer for memory management.
Integrates salience filter, rule-based compressor, token budget manager, and
incremental processor into a single cohesive system. Provides clean API for Ada
(short-term memory) and Nexus (long-term memory/graph) components.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time


class MemoryType(Enum):
    """Types of memory contexts."""
    WORKING = "working"      # Immediate conversation context
    EPISODIC = "episodic"    # Session-specific memory
    SEMANTIC = "semantic"    # Long-term knowledge graph


@dataclass
class MemoryEntry:
    """Single memory entry with metadata."""
    content: str
    memory_type: MemoryType
    timestamp: float
    salience_score: float = 0.0
    compressed_version: Optional[str] = None
    token_count: int = 0


@dataclass
class ContextWindow:
    """Result of context assembly."""
    text: str
    token_count: int
    entries_included: List[MemoryEntry]
    budget_remaining: int


class SalienceFilter:
    """Filters and scores memory entries by relevance."""
    def __init__(self, recency_weight: float = 0.4, frequency_weight: float = 0.3, relevance_weight: float = 0.3):
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.relevance_weight = relevance_weight

    def compute_salience(self, entry: MemoryEntry, current_context: str) -> float:
        """Compute salience score based on recency, frequency, and relevance."""
        recency = 1.0 / (1.0 + (time.time() - entry.timestamp)) if entry.timestamp else 0.5
        relevance = self._compute_relevance(entry.content, current_context)
        # Frequency would require tracking history
        frequency = 0.5
        score = (self.recency_weight * recency +
                 self.frequency_weight * frequency +
                 self.relevance_weight * relevance)
        entry.salience_score = score
        return score

    def filter_top_k(self, entries: List[MemoryEntry], k: int, current_context: str) -> List[MemoryEntry]:
        """Return top-k most salient entries."""
        for entry in entries:
            self.compute_salience(entry, current_context)
        sorted_entries = sorted(entries, key=lambda e: e.salience_score, reverse=True)
        return sorted_entries[:k]

    def _compute_relevance(self, content: str, context: str) -> float:
        """Simple relevance based on keyword overlap."""
        content_words = set(content.lower().split())
        context_words = set(context.lower().split())
        if not content_words:
            return 0.0
        overlap = len(content_words & context_words) / len(content_words)
        return overlap


class RuleBasedCompressor:
    """Compresses memory entries using rule-based techniques."""
    def __init__(self, max_chars: int = 500):
        self.max_chars = max_chars

    def compress(self, entry: MemoryEntry) -> str:
        """Compress a single memory entry."""
        content = entry.content
        if len(content) <= self.max_chars:
            entry.compressed_version = content
            return content

        # Rule 1: Remove filler words
        filler_words = {'um', 'uh', 'like', 'actually', 'basically', 'literally', 'so', 'well'}
        words = content.split()
        filtered = [w for w in words if w.lower() not in filler_words]

        # Rule 2: Extract first sentence or key phrases
        if len(' '.join(filtered)) > self.max_chars:
            compressed = ' '.join(filtered)[:self.max_chars] + "..."
        else:
            compressed = ' '.join(filtered)

        entry.compressed_version = compressed
        return compressed

    def compress_batch(self, entries: List[MemoryEntry]) -> List[MemoryEntry]:
        """Compress multiple entries in-place."""
        for entry in entries:
            self.compress(entry)
        return entries


class TokenBudgetManager:
    """Manages token budgets for context windows."""
    def __init__(self, total_budget: int = 4096, reserved_for_response: int = 1024):
        self.total_budget = total_budget
        self.reserved_for_response = reserved_for_response
        self.available_for_memory = total_budget - reserved_for_response

    def count_tokens(self, text: str) -> int:
        """
        Rough token estimation (4 chars ~1 token for English).
        WARNING: This is a rough estimate and may not match the tokenizer of
        any specific LLM. For production use, consider using a proper tokenizer
        (e.g., tiktoken for OpenAI models, or the model's native tokenizer).
        """
        return len(text) // 4

    def fit_to_budget(
        self,
        entries: List[MemoryEntry],
        current_context: str
    ) -> Tuple[List[MemoryEntry], int]:
        """Select entries that fit within available budget, sorted by importance."""
        # Sort entries by salience score (if available) or compressed version length
        entries_to_sort = [
            (e, len(e.compressed_version) if e.compressed_version else len(e.content))
            for e in entries
        ]
        # Lower token count and higher salience priority
        entries_sorted = sorted(
            entries,
            key=lambda e: (
                # Prioritize entries with higher salience scores
                -e.salience_score if e.salience_score != 0.0 else 0,
                # Then prioritize shorter entries
                len(e.compressed_version) if e.compressed_version else len(e.content)
            )
        )

        selected = []
        total_tokens = self.count_tokens(current_context)

        for entry in entries_sorted:
            content_to_use = entry.compressed_version if entry.compressed_version else entry.content
            entry_tokens = self.count_tokens(content_to_use)
            if total_tokens + entry_tokens <= self.available_for_memory:
                selected.append(entry)
                total_tokens += entry_tokens
                entry.token_count = entry_tokens

        remaining = self.available_for_memory - total_tokens
        return selected, remaining


class IncrementalProcessor:
    """Processes memories incrementally for streaming updates."""
    def __init__(self, buffer_size: int = 10):
        self.buffer = []
        self.buffer_size = buffer_size
        self.processed_count = 0

    def add_memory(self, entry: MemoryEntry):
        """Add a memory to the incremental buffer."""
        self.buffer.append(entry)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> List[MemoryEntry]:
        """Process and return all buffered memories."""
        processed = self.buffer.copy()
        self.buffer.clear()
        self.processed_count += len(processed)
        return processed

    def get_pending_count(self) -> int:
        """Number of memories waiting to be processed."""
        return len(self.buffer)


class MemoryContextLayer:
    """Unified orchestration layer for memory management.
    Provides clean API for Ada (short-term/working memory) and Nexus
    (long-term/semantic memory) to consume.
    """

    def __init__(
        self,
        total_token_budget: int = 4096,
        top_k_salient: int = 20,
        compression_max_chars: int = 500
    ):
        """Initialize the memory context layer.
        
        Args:
            total_token_budget: Total token limit for context window
            top_k_salient: Number of most salient memories to consider
            compression_max_chars: Max characters before compression
        """
        self.salience_filter = SalienceFilter()
        self.compressor = RuleBasedCompressor(max_chars=compression_max_chars)
        self.token_manager = TokenBudgetManager(total_budget=total_token_budget)
        self.processor = IncrementalProcessor()

        # Memory stores
        self.working_memory: List[MemoryEntry] = []   # Ada (short-term)
        self.semantic_memory: List[MemoryEntry] = []  # Nexus (long-term)
        self.episodic_memory: List[MemoryEntry] = []  # Session-specific

        self.top_k_salient = top_k_salient

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        timestamp: float = None
    ):
        """Add a new memory entry to the appropriate store.
        
        Args:
            content: Memory content text
            memory_type: Type of memory (WORKING, EPISODIC, SEMANTIC)
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()

        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            timestamp=timestamp,
            token_count=self.token_manager.count_tokens(content)
        )

        # Route to appropriate store
        if memory_type == MemoryType.WORKING:
            self.working_memory.append(entry)
        elif memory_type == MemoryType.EPISODIC:
            self.episodic_memory.append(entry)
        else:
            self.semantic_memory.append(entry)

        # Process incrementally
        self.processor.add_memory(entry)

    def assemble_context(
        self,
        current_query: str,
        memory_types: List[MemoryType] = None
    ) -> ContextWindow:
        """Assemble optimal context window for the current query.
        
        Args:
            current_query: The user's current input/query
            memory_types: Which memory types to include (defaults to all)
            
        Returns:
            ContextWindow containing assembled text and metadata
        """
        if memory_types is None:
            memory_types = [MemoryType.WORKING, MemoryType.EPISODIC, MemoryType.SEMANTIC]

        # Collect all candidate memories
        candidates = []
        if MemoryType.WORKING in memory_types:
            candidates.extend(self.working_memory)
        if MemoryType.EPISODIC in memory_types:
            candidates.extend(self.episodic_memory)
        if MemoryType.SEMANTIC in memory_types:
            candidates.extend(self.semantic_memory)

        if not candidates:
            return ContextWindow(
                text=current_query,
                token_count=self.token_manager.count_tokens(current_query),
                entries_included=[],
                budget_remaining=self.token_manager.available_for_memory
            )

        # Step 1: Filter by salience (top-k)
        salient_entries = self.salience_filter.filter_top_k(
            candidates, self.top_k_salient, current_query
        )

        # Step 2: Compress entries
        compressed_entries = self.compressor.compress_batch(salient_entries)

        # Step 3: Fit to token budget (this now sorts by salience and length)
        selected_entries, remaining_budget = self.token_manager.fit_to_budget(
            compressed_entries, current_query
        )

        # Step 4: Assemble final text
        assembled_parts = [current_query]
        for entry in selected_entries:
            content_to_use = entry.compressed_version if entry.compressed_version else entry.content
            prefix = f"[Memory {entry.memory_type.value}]: "
            assembled_parts.append(prefix + content_to_use)

        assembled_text = "\n\n".join(assembled_parts)
        token_count = self.token_manager.count_tokens(assembled_text)

        return ContextWindow(
            text=assembled_text,
            token_count=token_count,
            entries_included=selected_entries,
            budget_remaining=remaining_budget
        )

    # --- Public API Methods ---
    def get_working_memory(self) -> List[MemoryEntry]:
        """Get all working memory entries (for Ada)."""
        return self.working_memory.copy()

    def get_semantic_memory(self) -> List[MemoryEntry]:
        """Get all semantic memory entries (for Nexus)."""
        return self.semantic_memory.copy()

    def clear_working_memory(self):
        """Clear working memory (new session)."""
        self.working_memory.clear()

    def clear_episodic_memory(self):
        """Clear episodic memory."""
        self.episodic_memory.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return current memory statistics."""
        return {
            "working_memory_count": len(self.working_memory),
            "episodic_memory_count": len(self.episodic_memory),
            "semantic_memory_count": len(self.semantic_memory),
            "pending_incremental": self.processor.get_pending_count(),
            "total_processed": self.processor.processed_count,
            "token_budget_total": self.token_manager.total_budget,
            "token_budget_available": self.token_manager.available_for_memory
        }
