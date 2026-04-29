"""Structured Memory module with RLM Engine integration for external context storage and retrieval."""

import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ContextChunk:
    """Represents a piece of context stored in the external engine."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())


class RLMEngineClient:
    """Mock client for RLM Engine. Replace with actual RLM Engine SDK."""

    def __init__(self, endpoint: str = "http://localhost:8080"):
        self.endpoint = endpoint
        self._storage: Dict[str, ContextChunk] = {}

    async def store_context(self, key: str, context: ContextChunk) -> bool:
        """Store context chunk in external engine."""
        self._storage[key] = context
        return True

    async def retrieve_context(self, key: str) -> Optional[ContextChunk]:
        """Retrieve context chunk from external engine."""
        return self._storage.get(key)

    async def search_context(self, query: str, limit: int = 5) -> List[ContextChunk]:
        """Search context chunks by query (simple mock)."""
        # Mock implementation: return all chunks containing query
        results = [chunk for chunk in self._storage.values() if query.lower() in chunk.content.lower()]
        return results[:limit]

    async def delete_context(self, key: str) -> bool:
        """Delete context chunk from external engine."""
        if key in self._storage:
            del self._storage[key]
            return True
        return False


class StructuredMemory:
    """
    Structured Memory with RLM Engine integration for external context storage.
    Enables recursive context processing and dynamic exploration.
    """

    def __init__(self, rlm_endpoint: str = "http://localhost:8080"):
        self.rlm_client = RLMEngineClient(endpoint=rlm_endpoint)
        self.local_cache: Dict[str, ContextChunk] = {}
        self.recursion_depth: int = 0

    async def store(self, key: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store context in external RLM Engine."""
        chunk = ContextChunk(
            chunk_id=key,
            content=content,
            metadata=metadata or {}
        )
        success = await self.rlm_client.store_context(key, chunk)
        if success:
            self.local_cache[key] = chunk
        return key

    async def retrieve(self, key: str) -> Optional[str]:
        """Retrieve context from external RLM Engine."""
        chunk = await self.rlm_client.retrieve_context(key)
        return chunk.content if chunk else None

    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search external context storage."""
        chunks = await self.rlm_client.search_context(query, limit)
        return [
            {"chunk_id": c.chunk_id, "content": c.content, "metadata": c.metadata}
            for c in chunks
        ]

    async def recursive_process(self, initial_key: str, processor: callable, depth: int = 0) -> Any:
        """
        Recursively process context with external engine calls.
        Enables dynamic context exploration across multiple levels.
        """
        self.recursion_depth = max(self.recursion_depth, depth)
        context = await self.retrieve(initial_key)
        if not context:
            return None

        result = await processor(context, depth)

        # Explore related contexts (dynamic exploration)
        related_keys = await self._explore_related(initial_key, context)
        nested_results = []
        for related_key in related_keys:
            nested = await self.recursive_process(related_key, processor, depth + 1)
            if nested is not None:
                nested_results.append(nested)

        return {"result": result, "related": nested_results, "depth": depth}

    async def _explore_related(self, key: str, content: str) -> List[str]:
        """Dynamically explore related contexts using search."""
        # Extract keywords or use content to find related contexts
        keywords = content.split()[:5]  # Simple keyword extraction
        query = " ".join(keywords)
        related = await self.search(query, limit=3)
        return [item["chunk_id"] for item in related if item["chunk_id"] != key]

    async def delete(self, key: str) -> bool:
        """Delete context from external storage and local cache."""
        success = await self.rlm_client.delete_context(key)
        if success and key in self.local_cache:
            del self.local_cache[key]
        return success

    async def clear_cache(self) -> None:
        """Clear local cache only (external storage remains)."""
        self.local_cache.clear()

    def get_recursion_depth(self) -> int:
        """Return maximum recursion depth achieved."""
        return self.recursion_depth