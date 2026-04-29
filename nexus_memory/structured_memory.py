"""Hierarchical structured memory module with token-budget-aware retrieval.

This module provides a tree-based memory structure where each node contains
semantic information, supports hierarchical organization, and allows efficient
subtree operations with token budget constraints.
"""

import json
import uuid
from typing import Dict, List, Optional, Any, Tuple, Iterator
from datetime import datetime
from dataclasses import dataclass, field, asdict


@dataclass
class MemoryNode:
    """Node in the hierarchical memory tree.

    Attributes:
        node_id: Unique identifier for the node.
        content: Text content of the memory.
        metadata: Arbitrary key-value metadata.
        children: Dictionary mapping child node IDs to child nodes.
        parent_id: ID of the parent node, None for root.
        created_at: Timestamp of node creation.
        updated_at: Timestamp of last update.
    """

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: Dict[str, 'MemoryNode'] = field(default_factory=dict)
    parent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            'node_id': self.node_id,
            'content': self.content,
            'metadata': self.metadata,
            'parent_id': self.parent_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'children': {cid: child.to_dict() for cid, child in self.children.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNode':
        """Create node from dictionary."""
        node = cls(
            node_id=data['node_id'],
            content=data['content'],
            metadata=data['metadata'],
            parent_id=data['parent_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at'])
        )
        for cid, child_data in data.get('children', {}).items():
            child = cls.from_dict(child_data)
            child.parent_id = node.node_id
            node.children[cid] = child
        return node

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp to current time."""
        self.updated_at = datetime.now()


class StructuredMemory:
    """Hierarchical memory structure with token-aware retrieval."""

    def __init__(self, root_node: Optional[MemoryNode] = None):
        """Initialize the structured memory.

        Args:
            root_node: Optional root node. If not provided, creates empty root.
        """
        self.root = root_node if root_node else MemoryNode(content="Root", metadata={"is_root": True})
        self._node_cache: Dict[str, MemoryNode] = {self.root.node_id: self.root}
        self._update_cache(self.root)

    def _update_cache(self, node: MemoryNode) -> None:
        """Recursively update cache with node and its children."""
        self._node_cache[node.node_id] = node
        for child in node.children.values():
            self._update_cache(child)

    def _invalidate_cache(self) -> None:
        """Rebuild cache from root."""
        self._node_cache = {}
        self._update_cache(self.root)

    def set(self, path: List[str], content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Set or update a memory node at the given path.

        Args:
            path: List of node IDs or special strings like ':root' or ':new'.
                  If ':new' is used, a new node is created.
            content: Text content for the node.
            metadata: Optional metadata for the node.

        Returns:
            Node ID of the set/updated node.

        Raises:
            ValueError: If path is invalid or target node not found.
        """
        if not path:
            raise ValueError("Path cannot be empty")

        # Navigate to parent
        parent = self.root
        if len(path) > 1:
            for node_id in path[:-1]:
                if node_id not in self._node_cache:
                    raise ValueError(f"Node {node_id} not found in memory")
                parent = self._node_cache[node_id]

        target_id = path[-1]
        if target_id == ':new':
            # Create new node
            node = MemoryNode(content=content, metadata=metadata or {}, parent_id=parent.node_id)
            parent.children[node.node_id] = node
            self._node_cache[node.node_id] = node
            node.update_timestamp()
            return node.node_id
        else:
            # Update existing node
            if target_id not in self._node_cache:
                raise ValueError(f"Node {target_id} not found")
            node = self._node_cache[target_id]
            node.content = content
            if metadata:
                node.metadata.update(metadata)
            node.update_timestamp()
            return node.node_id

    def get(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieve a node by its ID.

        Args:
            node_id: Unique identifier of the node.

        Returns:
            MemoryNode if found, None otherwise.
        """
        return self._node_cache.get(node_id)

    def update(self, node_id: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update content and/or metadata of an existing node.

        Args:
            node_id: ID of the node to update.
            content: New content (if provided).
            metadata: Metadata to merge (if provided).

        Returns:
            True if node was found and updated, False otherwise.
        """
        node = self._node_cache.get(node_id)
        if not node:
            return False

        if content is not None:
            node.content = content
        if metadata is not None:
            node.metadata.update(metadata)
        node.update_timestamp()
        return True

    def delete(self, node_id: str) -> bool:
        """Delete a node and all its descendants.

        Args:
            node_id: ID of the node to delete.

        Returns:
            True if node was deleted, False if not found.
        """
        node = self._node_cache.get(node_id)
        if not node or node.parent_id is None:
            return False

        parent = self._node_cache.get(node.parent_id)
        if parent and node_id in parent.children:
            del parent.children[node_id]
            self._invalidate_cache()
            return True
        return False

    def get_subtree(self, node_id: str) -> MemoryNode:
        """Retrieve a subtree rooted at the given node.

        Args:
            node_id: Root ID of the subtree to retrieve.

        Returns:
            Copy of the subtree root node with its children.

        Raises:
            ValueError: If node_id not found.
        """
        node = self._node_cache.get(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")
        return self._deep_copy(node)

    def _deep_copy(self, node: MemoryNode) -> MemoryNode:
        """Create a deep copy of a node and its subtree."""
        copy_node = MemoryNode(
            node_id=node.node_id,
            content=node.content,
            metadata=node.metadata.copy(),
            parent_id=node.parent_id,
            created_at=node.created_at,
            updated_at=node.updated_at
        )
        for child_id, child in node.children.items():
            child_copy = self._deep_copy(child)
            child_copy.parent_id = copy_node.node_id
            copy_node.children[child_id] = child_copy
        return copy_node

    def retrieve_with_token_budget(self, node_id: str, token_budget: int, token_counter: callable) -> List[MemoryNode]:
        """Retrieve nodes from subtree respecting token budget.

        Args:
            node_id: Root ID to start retrieval from.
            token_budget: Maximum total tokens allowed.
            token_counter: Function that takes content string and returns token count.

        Returns:
            List of nodes (excluding root if root's content exceeds budget).
        """
        root_node = self._node_cache.get(node_id)
        if not root_node:
            return []

        collected_nodes = []
        remaining_budget = token_budget

        def traverse(node: MemoryNode) -> None:
            nonlocal remaining_budget
            if remaining_budget <= 0:
                return

            node_tokens = token_counter(node.content)
            if node_tokens <= remaining_budget:
                collected_nodes.append(node)
                remaining_budget -= node_tokens
                # Traverse children in order
                for child in node.children.values():
                    traverse(child)
            else:
                # Partial node not included
                pass

        traverse(root_node)
        return collected_nodes

    def search(self, query: str, node_id: Optional[str] = None) -> List[Tuple[MemoryNode, float]]:
        """Simple substring search with relevance scoring.

        Args:
            query: Search string.
            node_id: Optional root to search within. If None, search entire memory.

        Returns:
            List of (node, score) tuples sorted by score descending.
            Score is number of query occurrences normalized by content length.
        """
        start_node = self._node_cache.get(node_id) if node_id else self.root
        if not start_node:
            return []

        results = []
        query_lower = query.lower()

        def search_node(node: MemoryNode):
            count = node.content.lower().count(query_lower)
            if count > 0:
                score = count / (len(node.content) + 1)  # Normalized frequency
                results.append((node, score))
            for child in node.children.values():
                search_node(child)

        search_node(start_node)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def validate(self) -> Dict[str, Any]:
        """Validate the memory structure for consistency.

        Returns:
            Dictionary with validation results: 'valid' (bool), 'errors' (list),
            'total_nodes' (int), 'orphans' (list).
        """
        errors = []
        orphans = []
        total_nodes = 0

        def validate_node(node: MemoryNode, path: str):
            nonlocal total_nodes
            total_nodes += 1
            # Check parent link consistency
            if node.parent_id is not None and node.parent_id not in self._node_cache:
                errors.append(f"Node {node.node_id} has missing parent {node.parent_id}")
                orphans.append(node.node_id)
            # Check child references
            for child_id, child in node.children.items():
                if child.parent_id != node.node_id:
                    errors.append(f"Child {child_id} of node {node.node_id} has wrong parent {child.parent_id}")
                validate_node(child, f"{path}/{node.node_id}")

        validate_node(self.root, "")

        # Check cache consistency
        if len(self._node_cache) != total_nodes:
            errors.append(f"Cache size ({len(self._node_cache)}) != actual nodes ({total_nodes})")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'total_nodes': total_nodes,
            'orphans': orphans
        }

    def save(self, filepath: str) -> None:
        """Save memory to JSON file.

        Args:
            filepath: Path to save the memory data.
        """
        data = self.root.to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> 'StructuredMemory':
        """Load memory from JSON file.

        Args:
            filepath: Path to load memory data from.

        Returns:
            StructuredMemory instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            json.JSONDecodeError: If file is malformed.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        root_node = MemoryNode.from_dict(data)
        return cls(root_node)

    def __len__(self) -> int:
        """Return total number of nodes in memory."""
        return len(self._node_cache)

    def __contains__(self, node_id: str) -> bool:
        """Check if node ID exists."""
        return node_id in self._node_cache

    def get_all_nodes(self) -> Iterator[MemoryNode]:
        """Iterate over all nodes in the memory."""
        for node in self._node_cache.values():
            yield node