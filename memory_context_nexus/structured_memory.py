"""Structured Memory module based on PRISM principles.

Implements incremental processing with typed hierarchical schema,
key-value caching for reducing redundant computations, and automatic
capacity management to prevent memory overflow. Supports chunking of
large inputs and hierarchical memory updates.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from collections import OrderedDict
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryNode:
    """Typed node in the hierarchical memory structure."""
    key: str
    value: Any
    node_type: str  # e.g., 'entity', 'relation', 'attribute', 'context'
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)  # child node keys
    parent: Optional[str] = None


class StructuredMemory:
    """PRISM-based structured memory with caching and capacity management."""

    def __init__(self, max_capacity: int = 10000, cache_size: int = 1000):
        """
        Initialize structured memory.

        Args:
            max_capacity: Maximum number of memory nodes before overflow protection
            cache_size: Size of LRU cache for reducing redundant computations
        """
        self.max_capacity = max_capacity
        self.cache_size = cache_size
        self.nodes: Dict[str, MemoryNode] = {}
        self.hierarchy: Dict[str, List[str]] = {}  # parent_key -> [child_keys]
        self.cache: OrderedDict = OrderedDict()  # LRU cache for computed results
        self.access_counter = 0

    def store(self, key: str, value: Any, node_type: str = 'entity',
              parent: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
              chunk_large: bool = True, chunk_size: int = 1000) -> str:
        """
        Store a value in memory with automatic chunking for large inputs.

        Args:
            key: Unique identifier for the memory node
            value: Value to store (can be large)
            node_type: Type classification
            parent: Parent node key for hierarchical organization
            metadata: Additional metadata
            chunk_large: Whether to chunk large values
            chunk_size: Size threshold for chunking

        Returns:
            Storage key (original key or chunked key prefix)
        """
        self._check_capacity()

        if chunk_large and self._is_large(value, chunk_size):
            return self._store_chunked(key, value, node_type, parent, metadata)

        node = MemoryNode(
            key=key,
            value=value,
            node_type=node_type,
            metadata=metadata or {},
            parent=parent
        )
        self.nodes[key] = node

        if parent:
            if parent not in self.hierarchy:
                self.hierarchy[parent] = []
            if key not in self.hierarchy[parent]:
                self.hierarchy[parent].append(key)
            node.parent = parent

        self._invalidate_cache_prefix(key)
        return key

    def _is_large(self, value: Any, threshold: int) -> bool:
        """Determine if a value is large enough to require chunking."""
        try:
            size = len(str(value))
            return size > threshold
        except:
            return False

    def _store_chunked(self, key: str, value: Any, node_type: str,
                       parent: Optional[str], metadata: Optional[Dict]) -> str:
        """Store large value as chunks."""
        chunks = self._chunk_data(value, self.cache_size // 10)
        chunk_keys = []
        for idx, chunk in enumerate(chunks):
            chunk_key = f"{key}_chunk_{idx}"
            chunk_node = MemoryNode(
                key=chunk_key,
                value=chunk,
                node_type=f"{node_type}_chunk",
                metadata={**metadata, "chunk_index": idx, "parent_chunked": key},
                parent=key
            )
            self.nodes[chunk_key] = chunk_node
            chunk_keys.append(chunk_key)

        # Store manifest node
        manifest_node = MemoryNode(
            key=key,
            value={"chunks": chunk_keys, "total_chunks": len(chunks)},
            node_type=node_type,
            metadata=metadata or {},
            parent=parent
        )
        self.nodes[key] = manifest_node

        if parent:
            if parent not in self.hierarchy:
                self.hierarchy[parent] = []
            if key not in self.hierarchy[parent]:
                self.hierarchy[parent].append(key)

        return key

    def _chunk_data(self, data: Any, max_chunk_size: int) -> List[Any]:
        """Split large data into chunks."""
        data_str = json.dumps(data) if not isinstance(data, str) else data
        chunks = []
        for i in range(0, len(data_str), max_chunk_size):
            chunk = data_str[i:i + max_chunk_size]
            try:
                chunks.append(json.loads(chunk) if not isinstance(data, str) else chunk)
            except:
                chunks.append(chunk)
        return chunks

    def retrieve(self, key: str, use_cache: bool = True) -> Optional[Any]:
        """
        Retrieve a value from memory with caching support.

        Args:
            key: Node key to retrieve
            use_cache: Whether to use the LRU cache

        Returns:
            Stored value or None if not found
        """
        if use_cache and key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        if key not in self.nodes:
            return None

        node = self.nodes[key]
        if node.node_type.endswith('_chunk'):
            # Reconstruct from chunks
            manifest_key = node.parent if node.parent else key.split('_chunk')[0]
            return self._reconstruct_chunked(manifest_key)

        if isinstance(node.value, dict) and "chunks" in node.value:
            return self._reconstruct_chunked(key)

        result = node.value
        if use_cache:
            self._add_to_cache(key, result)
        return result

    def _reconstruct_chunked(self, manifest_key: str) -> Any:
        """Reconstruct chunked data from manifest."""
        if manifest_key not in self.nodes:
            return None
        manifest = self.nodes[manifest_key]
        if not isinstance(manifest.value, dict) or "chunks" not in manifest.value:
            return manifest.value

        chunks = []
        for chunk_key in manifest.value["chunks"]:
            if chunk_key in self.nodes:
                chunks.append(self.nodes[chunk_key].value)

        if all(isinstance(c, str) for c in chunks):
            full_str = "".join(chunks)
            try:
                return json.loads(full_str)
            except:
                return full_str
        return chunks

    def retrieve_hierarchical(self, root_key: str, depth: int = -1) -> Dict[str, Any]:
        """
        Retrieve a subtree of memory nodes.

        Args:
            root_key: Root node key
            depth: Maximum depth (-1 for unlimited)

        Returns:
            Hierarchical dictionary of memory nodes
        """
        if root_key not in self.nodes:
            return {}

        cache_key = f"hier_{root_key}_{depth}"
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]

        def build_tree(key: str, current_depth: int) -> Dict:
            if current_depth == 0:
                return {"node": self._serialize_node(self.nodes[key]), "children": {}}

            node = self.nodes[key]
            children = self.hierarchy.get(key, [])
            result = {
                "node": self._serialize_node(node),
                "children": {}
            }
            for child_key in children:
                result["children"][child_key] = build_tree(child_key, current_depth - 1)
            return result

        tree = build_tree(root_key, depth)
        self._add_to_cache(cache_key, tree)
        return tree

    def update(self, key: str, value: Any, incremental: bool = True) -> bool:
        """
        Update memory incrementally (PRISM-style).

        Args:
            key: Node key to update
            value: New value or delta
            incremental: If True, merge with existing; if False, replace

        Returns:
            True if update succeeded
        """
        if key not in self.nodes:
            return False

        node = self.nodes[key]
        if incremental:
            if isinstance(node.value, dict) and isinstance(value, dict):
                node.value.update(value)
            elif isinstance(node.value, list) and isinstance(value, list):
                node.value.extend(value)
            else:
                node.value = value
        else:
            node.value = value

        node.timestamp = datetime.now()
        self._invalidate_cache_prefix(key)
        return True

    def _check_capacity(self):
        """Prevent memory overflow by pruning old nodes."""
        if len(self.nodes) >= self.max_capacity:
            # Remove oldest nodes (based on timestamp)
            sorted_nodes = sorted(self.nodes.items(),
                                  key=lambda x: x[1].timestamp)
            to_remove = len(self.nodes) - int(self.max_capacity * 0.8)
            for i in range(to_remove):
                key, _ = sorted_nodes[i]
                # Skip root nodes or protected nodes
                if self.nodes[key].metadata.get("protected", False):
                    continue
                del self.nodes[key]
                if key in self.hierarchy:
                    del self.hierarchy[key]

            logger.info(f"Pruned {to_remove} nodes due to capacity limit")

    def _add_to_cache(self, key: str, value: Any):
        """Add to LRU cache with size management."""
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        self.cache[key] = value

    def _invalidate_cache_prefix(self, key_prefix: str):
        """Invalidate cache entries with given key prefix."""
        keys_to_remove = [k for k in self.cache if k.startswith(key_prefix) or
                          (isinstance(k, str) and k.startswith(f"hier_{key_prefix}"))]
        for k in keys_to_remove:
            del self.cache[k]

    def _serialize_node(self, node: MemoryNode) -> Dict:
        """Serialize node for hierarchical retrieval."""
        return {
            "key": node.key,
            "value": node.value,
            "node_type": node.node_type,
            "timestamp": node.timestamp.isoformat(),
            "metadata": node.metadata,
            "children": node.children,
            "parent": node.parent
        }

    def delete(self, key: str, recursive: bool = False) -> bool:
        """
        Delete a memory node and optionally its children.

        Args:
            key: Node key to delete
            recursive: Whether to delete all child nodes

        Returns:
            True if deletion succeeded
        """
        if key not in self.nodes:
            return False

        if recursive and key in self.hierarchy:
            for child_key in self.hierarchy[key]:
                self.delete(child_key, recursive=True)

        del self.nodes[key]
        if key in self.hierarchy:
            del self.hierarchy[key]

        # Remove from parent's children list
        parent_key = self.nodes.get(key, MemoryNode(key, None, "")).parent
        if parent_key and parent_key in self.hierarchy:
            if key in self.hierarchy[parent_key]:
                self.hierarchy[parent_key].remove(key)

        self._invalidate_cache_prefix(key)
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Return memory statistics."""
        return {
            "total_nodes": len(self.nodes),
            "hierarchy_depth": len(self.hierarchy),
            "cache_size": len(self.cache),
            "max_capacity": self.max_capacity,
            "cache_limit": self.cache_size,
            "node_types": {}
        }

    def clear(self):
        """Clear all memory and cache."""
        self.nodes.clear()
        self.hierarchy.clear()
        self.cache.clear()
