"""
RLM Engine for recursive context processing of very long contexts (100k+ tokens).

This module provides an engine that uses external Python storage and recursive
exploration to handle contexts that exceed typical token limits. It supports
LiteLLM integration and parallel recursive calls.
"""

import asyncio
import hashlib
import json
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from dataclasses import dataclass, field
from pathlib import Path
import aiofiles
import aiofiles.os


@dataclass
class ContextChunk:
    """Represents a chunk of context stored externally."""
    chunk_id: str
    content: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)  # References to sub-chunk IDs
    parent_id: Optional[str] = None


class ExternalStorage:
    """External Python storage for context chunks using filesystem."""
    
    def __init__(self, storage_path: str = "./rlm_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.storage_path / "index.json"
        self._index: Dict[str, Dict] = {}
        self._load_index()
    
    def _load_index(self):
        """Load or create the storage index."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                self._index = json.load(f)
        else:
            self._index = {}
    
    async def _save_index(self):
        """Save the storage index asynchronously."""
        async with aiofiles.open(self.index_path, 'w') as f:
            await f.write(json.dumps(self._index, indent=2))
    
    async def store_chunk(self, chunk: ContextChunk) -> str:
        """Store a context chunk and return its ID."""
        chunk_path = self.storage_path / f"{chunk.chunk_id}.json"
        data = {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "token_count": chunk.token_count,
            "metadata": chunk.metadata,
            "children": chunk.children,
            "parent_id": chunk.parent_id
        }
        async with aiofiles.open(chunk_path, 'w') as f:
            await f.write(json.dumps(data, indent=2))
        
        self._index[chunk.chunk_id] = {
            "path": str(chunk_path),
            "token_count": chunk.token_count,
            "parent_id": chunk.parent_id
        }
        await self._save_index()
        return chunk.chunk_id
    
    async def retrieve_chunk(self, chunk_id: str) -> Optional[ContextChunk]:
        """Retrieve a context chunk by ID."""
        if chunk_id not in self._index:
            return None
        chunk_path = Path(self._index[chunk_id]["path"])
        if not chunk_path.exists():
            return None
        async with aiofiles.open(chunk_path, 'r') as f:
            data = json.loads(await f.read())
        return ContextChunk(
            chunk_id=data["chunk_id"],
            content=data["content"],
            token_count=data["token_count"],
            metadata=data.get("metadata", {}),
            children=data.get("children", []),
            parent_id=data.get("parent_id")
        )
    
    async def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a context chunk."""
        if chunk_id not in self._index:
            return False
        chunk_path = Path(self._index[chunk_id]["path"])
        if chunk_path.exists():
            await aiofiles.os.remove(chunk_path)
        del self._index[chunk_id]
        await self._save_index()
        return True
    
    async def list_chunks(self) -> List[str]:
        """List all chunk IDs."""
        return list(self._index.keys())


class RecursiveContextProcessor:
    """Handles recursive exploration and processing of context."""
    
    def __init__(
        self,
        storage: ExternalStorage,
        max_chunk_tokens: int = 8000,
        recursive_depth_limit: int = 5,
        parallel_limit: int = 10
    ):
        self.storage = storage
        self.max_chunk_tokens = max_chunk_tokens
        self.recursive_depth_limit = recursive_depth_limit
        self.parallel_limit = parallel_limit
    
    def _estimate_tokens(self, text: str) -> int:
        """Simple token estimator (4 chars per token approximation)."""
        return len(text) // 4
    
    async def _split_text(self, text: str, chunk_id_prefix: str, depth: int = 0) -> List[str]:
        """Split text into manageable chunks recursively."""
        token_estimate = self._estimate_tokens(text)
        if token_estimate <= self.max_chunk_tokens or depth >= self.recursive_depth_limit:
            return [text]
        
        # Split at paragraph boundaries or sentences
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = self._estimate_tokens(line)
            if current_tokens + line_tokens > self.max_chunk_tokens and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_tokens = line_tokens
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        # If still too big, recursively split further
        if len(chunks) == 1 and len(chunks[0]) == len(text):
            # Fallback to character-based splitting
            chunk_size = self.max_chunk_tokens * 4
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        return chunks
    
    async def store_recursive(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        depth: int = 0
    ) -> str:
        """Store text recursively, creating a tree of chunks."""
        chunk_id = hashlib.md5(f"{parent_id or 'root'}:{depth}:{text[:100]}".encode()).hexdigest()[:16]
        
        if depth >= self.recursive_depth_limit:
            chunk = ContextChunk(
                chunk_id=chunk_id,
                content=text,
                token_count=self._estimate_tokens(text),
                metadata=metadata or {},
                parent_id=parent_id
            )
            await self.storage.store_chunk(chunk)
            return chunk_id
        
        sub_texts = await self._split_text(text, chunk_id, depth)
        
        if len(sub_texts) == 1:
            chunk = ContextChunk(
                chunk_id=chunk_id,
                content=text,
                token_count=self._estimate_tokens(text),
                metadata=metadata or {},
                parent_id=parent_id
            )
            await self.storage.store_chunk(chunk)
            return chunk_id
        
        child_ids = []
        for i, sub_text in enumerate(sub_texts):
            child_metadata = {"part": i, "total_parts": len(sub_texts), **(metadata or {})}
            child_id = await self.store_recursive(sub_text, child_metadata, chunk_id, depth + 1)
            child_ids.append(child_id)
        
        # Store this as a container chunk
        chunk = ContextChunk(
            chunk_id=chunk_id,
            content="",  # Container chunk
            token_count=0,
            metadata={"container": True, "child_count": len(child_ids), **(metadata or {})},
            children=child_ids,
            parent_id=parent_id
        )
        await self.storage.store_chunk(chunk)
        return chunk_id
    
    async def reconstruct_context(self, chunk_id: str) -> str:
        """Reconstruct full context from a chunk and its children recursively."""
        chunk = await self.storage.retrieve_chunk(chunk_id)
        if not chunk:
            return ""
        
        if chunk.children:
            # Parallel reconstruction of children
            child_texts = await asyncio.gather(*[
                self.reconstruct_context(child_id) for child_id in chunk.children
            ])
            return "\n".join(child_texts)
        else:
            return chunk.content
    
    async def process_recursive(
        self,
        chunk_id: str,
        process_func: Callable[[str, Dict[str, Any]], Awaitable[str]]
    ) -> str:
        """Apply a processing function recursively to the context."""
        chunk = await self.storage.retrieve_chunk(chunk_id)
        if not chunk:
            return ""
        
        if chunk.children:
            # Process children in parallel
            processed_children = await asyncio.gather(*[
                self.process_recursive(child_id, process_func) for child_id in chunk.children
            ])
            combined = "\n".join(processed_children)
            result = await process_func(combined, chunk.metadata)
            return result
        else:
            result = await process_func(chunk.content, chunk.metadata)
            return result


class RLMEngine:
    """
    Recursive Language Model Engine for handling very long contexts.
    
    Supports LiteLLM integration and parallel recursive calls for efficient
    processing of contexts exceeding 100k tokens.
    """
    
    def __init__(
        self,
        storage_path: str = "./rlm_storage",
        max_chunk_tokens: int = 8000,
        recursive_depth_limit: int = 5,
        parallel_limit: int = 10,
        litellm_model: Optional[str] = None,
        litellm_api_key: Optional[str] = None,
        litellm_api_base: Optional[str] = None
    ):
        self.storage = ExternalStorage(storage_path)
        self.processor = RecursiveContextProcessor(
            storage=self.storage,
            max_chunk_tokens=max_chunk_tokens,
            recursive_depth_limit=recursive_depth_limit,
            parallel_limit=parallel_limit
        )
        self.litellm_model = litellm_model
        self.litellm_api_key = litellm_api_key
        self.litellm_api_base = litellm_api_base
        self._litellm_available = False
        
        if litellm_model:
            try:
                import litellm
                self.litellm = litellm
                self._litellm_available = True
            except ImportError:
                raise ImportError("LiteLLM not installed. Install with: pip install litellm")
    
    async def store_context(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a long context and return its root chunk ID."""
        return await self.processor.store_recursive(text, metadata)
    
    async def retrieve_context(self, chunk_id: str) -> str:
        """Retrieve and reconstruct the full context from a chunk ID."""
        return await self.processor.reconstruct_context(chunk_id)
    
    async def query_recursive(
        self,
        chunk_id: str,
        query: str,
        custom_prompt_template: Optional[str] = None
    ) -> str:
        """
        Query the context recursively, using LiteLLM to process chunks.
        
        Args:
            chunk_id: Root chunk ID to start from
            query: User query to answer from the context
            custom_prompt_template: Optional custom prompt template with {context} and {query} placeholders
        """
        if not self._litellm_available:
            raise RuntimeError("LiteLLM not configured. Provide litellm_model in constructor.")
        
        async def process_chunk(content: str, metadata: Dict) -> str:
            # Build prompt
            if custom_prompt_template:
                prompt = custom_prompt_template.format(context=content, query=query)
            else:
                prompt = f"""Based on the following context, answer the query.

Context:
{content}

Query: {query}

Answer:"""
            
            # Call LiteLLM
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.litellm.completion(
                    model=self.litellm_model,
                    messages=[{"role": "user", "content": prompt}],
                    api_key=self.litellm_api_key,
                    api_base=self.litellm_api_base
                )
            )
            return response.choices[0].message.content
        
        return await self.processor.process_recursive(chunk_id, process_chunk)
    
    async def query_parallel(
        self,
        chunk_ids: List[str],
        query: str,
        custom_prompt_template: Optional[str] = None
    ) -> List[str]:
        """
        Query multiple context trees in parallel.
        
        Args:
            chunk_ids: List of root chunk IDs to query
            query: Same query to apply to all contexts
            custom_prompt_template: Optional custom prompt template
        
        Returns:
            List of answers for each chunk ID in the same order
        """
        tasks = [
            self.query_recursive(chunk_id, query, custom_prompt_template)
            for chunk_id in chunk_ids
        ]
        # Limit concurrency
        semaphore = asyncio.Semaphore(self.processor.parallel_limit)
        
        async def bounded_task(task):
            async with semaphore:
                return await task
        
        return await asyncio.gather(*[bounded_task(task) for task in tasks])
    
    async def delete_context(self, chunk_id: str) -> bool:
        """Delete an entire context tree recursively."""
        chunk = await self.storage.retrieve_chunk(chunk_id)
        if not chunk:
            return False
        
        # Delete children first
        for child_id in chunk.children:
            await self.delete_context(child_id)
        
        # Delete this chunk
        return await self.storage.delete_chunk(chunk_id)
    
    async def get_context_stats(self, chunk_id: str) -> Dict[str, Any]:
        """Get statistics about a context tree."""
        chunk = await self.storage.retrieve_chunk(chunk_id)
        if not chunk:
            return {}
        
        stats = {
            "chunk_id": chunk_id,
            "depth": 0,
            "total_chunks": 1,
            "total_tokens": chunk.token_count
        }
        
        for child_id in chunk.children:
            child_stats = await self.get_context_stats(child_id)
            if child_stats:
                stats["total_chunks"] += child_stats.get("total_chunks", 0)
                stats["total_tokens"] += child_stats.get("total_tokens", 0)
                stats["depth"] = max(stats["depth"], 1 + child_stats.get("depth", 0))
        
        return stats
