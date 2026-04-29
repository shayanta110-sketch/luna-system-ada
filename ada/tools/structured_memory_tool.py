"""LangChain tools for structured memory operations in ADA.

This module provides tools for interacting with a hierarchical memory store,
enabling agents to set, retrieve, navigate, and persist structured memory data.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class MemoryNode(BaseModel):
    """Represents a node in the hierarchical memory structure."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    key: str
    value: Any
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StructuredMemoryStore:
    """In-memory store with hierarchical structure and persistence."""

    def __init__(self):
        self.nodes: Dict[str, MemoryNode] = {}
        self.root_id: Optional[str] = None

    def set(self, key: str, value: Any, parent_key: Optional[str] = None) -> str:
        """Set a memory value at the given key, optionally under a parent."""
        # Check if key already exists
        existing_id = self._find_id_by_key(key)
        if existing_id:
            node = self.nodes[existing_id]
            node.value = value
            if parent_key:
                parent_id = self._find_id_by_key(parent_key)
                if parent_id and node.parent_id != parent_id:
                    # Remove from old parent
                    old_parent = self.nodes.get(node.parent_id)
                    if old_parent and existing_id in old_parent.children_ids:
                        old_parent.children_ids.remove(existing_id)
                    # Add to new parent
                    node.parent_id = parent_id
                    if parent_id not in self.nodes:
                        self.nodes[parent_id] = MemoryNode(
                            key=parent_key,
                            value=None,
                            id=parent_id
                        )
                    self.nodes[parent_id].children_ids.append(existing_id)
            return existing_id

        # Create new node
        parent_id = None
        if parent_key:
            parent_id = self._find_id_by_key(parent_key)
            if not parent_id:
                # Create parent node if it doesn't exist
                parent_node = MemoryNode(key=parent_key, value=None)
                self.nodes[parent_node.id] = parent_node
                parent_id = parent_node.id

        node = MemoryNode(key=key, value=value, parent_id=parent_id)
        self.nodes[node.id] = node

        if parent_id:
            self.nodes[parent_id].children_ids.append(node.id)
        elif not self.root_id:
            self.root_id = node.id

        return node.id

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a memory value by key."""
        node_id = self._find_id_by_key(key)
        if node_id:
            return self.nodes[node_id].value
        return None

    def get_subtree(self, key: str) -> Dict[str, Any]:
        """Get entire subtree rooted at key as nested dictionary."""
        node_id = self._find_id_by_key(key)
        if not node_id:
            return {}
        return self._build_subtree(node_id)

    def list_keys(self, parent_key: Optional[str] = None) -> List[str]:
        """List all keys at a given level (direct children of parent)."""
        if parent_key:
            parent_id = self._find_id_by_key(parent_key)
            if not parent_id:
                return []
            return [self.nodes[cid].key for cid in self.nodes[parent_id].children_ids]
        else:
            # Return all top-level keys (no parent or root)
            return [node.key for node in self.nodes.values() if node.parent_id is None]

    def save(self, filepath: str) -> None:
        """Save entire memory structure to JSON file."""
        data = {
            "nodes": {
                nid: {
                    "id": node.id,
                    "key": node.key,
                    "value": node.value,
                    "parent_id": node.parent_id,
                    "children_ids": node.children_ids,
                    "metadata": node.metadata
                }
                for nid, node in self.nodes.items()
            },
            "root_id": self.root_id
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """Load memory structure from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.nodes = {}
        for nid, node_data in data["nodes"].items():
            self.nodes[nid] = MemoryNode(
                id=node_data["id"],
                key=node_data["key"],
                value=node_data["value"],
                parent_id=node_data["parent_id"],
                children_ids=node_data["children_ids"],
                metadata=node_data.get("metadata", {})
            )
        self.root_id = data.get("root_id")

    def _find_id_by_key(self, key: str) -> Optional[str]:
        """Helper to find node ID by key (first occurrence)."""
        for nid, node in self.nodes.items():
            if node.key == key:
                return nid
        return None

    def _build_subtree(self, node_id: str) -> Dict[str, Any]:
        """Recursively build nested dict from node."""
        node = self.nodes[node_id]
        result = {node.key: node.value} if node.value is not None else {node.key: {}}
        for child_id in node.children_ids:
            child_subtree = self._build_subtree(child_id)
            # Merge child subtree
            if node.key in result and isinstance(result[node.key], dict):
                result[node.key].update(child_subtree)
            else:
                result[node.key] = child_subtree
        return result


# LangChain Tool Schemas
class SetMemoryInput(BaseModel):
    key: str = Field(description="Key to store the memory under")
    value: Any = Field(description="Value to store (can be string, number, dict, list, etc.)")
    parent_key: Optional[str] = Field(default=None, description="Optional parent key for hierarchical organization")


class GetMemoryInput(BaseModel):
    key: str = Field(description="Key to retrieve")


class GetSubtreeInput(BaseModel):
    key: str = Field(description="Root key of subtree to retrieve")


class ListKeysInput(BaseModel):
    parent_key: Optional[str] = Field(default=None, description="Parent key to list children of (None for top-level)")


class SaveMemoryInput(BaseModel):
    filepath: str = Field(description="File path to save memory to (JSON format)")


class LoadMemoryInput(BaseModel):
    filepath: str = Field(description="File path to load memory from (JSON format)")


class SetMemoryTool(BaseTool):
    """Tool to set a value in structured memory."""
    name: str = "set_memory"
    description: str = "Store a value at a given key, optionally under a parent key for hierarchy."
    args_schema: Type[BaseModel] = SetMemoryInput
    store: StructuredMemoryStore

    def _run(self, key: str, value: Any, parent_key: Optional[str] = None) -> str:
        node_id = self.store.set(key, value, parent_key)
        return f"Memory stored with ID: {node_id}"

    async def _arun(self, key: str, value: Any, parent_key: Optional[str] = None) -> str:
        return self._run(key, value, parent_key)


class GetMemoryTool(BaseTool):
    """Tool to retrieve a value from structured memory."""
    name: str = "get_memory"
    description: str = "Retrieve a value from memory by key."
    args_schema: Type[BaseModel] = GetMemoryInput
    store: StructuredMemoryStore

    def _run(self, key: str) -> str:
        value = self.store.get(key)
        if value is None:
            return f"No memory found for key: {key}"
        return json.dumps(value, indent=2, default=str)

    async def _arun(self, key: str) -> str:
        return self._run(key)


class GetSubtreeTool(BaseTool):
    """Tool to retrieve an entire hierarchical subtree."""
    name: str = "get_subtree"
    description: str = "Retrieve entire nested structure rooted at a given key."
    args_schema: Type[BaseModel] = GetSubtreeInput
    store: StructuredMemoryStore

    def _run(self, key: str) -> str:
        subtree = self.store.get_subtree(key)
        if not subtree:
            return f"No subtree found for key: {key}"
        return json.dumps(subtree, indent=2, default=str)

    async def _arun(self, key: str) -> str:
        return self._run(key)


class ListKeysTool(BaseTool):
    """Tool to list keys at a specific level."""
    name: str = "list_memory_keys"
    description: str = "List all keys directly under a parent key (or top-level if no parent)."
    args_schema: Type[BaseModel] = ListKeysInput
    store: StructuredMemoryStore

    def _run(self, parent_key: Optional[str] = None) -> str:
        keys = self.store.list_keys(parent_key)
        if not keys:
            return "No keys found."
        return "Available keys:\n- " + "\n- ".join(keys)

    async def _arun(self, parent_key: Optional[str] = None) -> str:
        return self._run(parent_key)


class SaveMemoryTool(BaseTool):
    """Tool to save memory to disk."""
    name: str = "save_memory"
    description: str = "Save the entire memory structure to a JSON file."
    args_schema: Type[BaseModel] = SaveMemoryInput
    store: StructuredMemoryStore

    def _run(self, filepath: str) -> str:
        try:
            self.store.save(filepath)
            return f"Memory saved successfully to {filepath}"
        except Exception as e:
            return f"Error saving memory: {str(e)}"

    async def _arun(self, filepath: str) -> str:
        return self._run(filepath)


class LoadMemoryTool(BaseTool):
    """Tool to load memory from disk."""
    name: str = "load_memory"
    description: str = "Load a previously saved memory structure from a JSON file."
    args_schema: Type[BaseModel] = LoadMemoryInput
    store: StructuredMemoryStore

    def _run(self, filepath: str) -> str:
        try:
            self.store.load(filepath)
            return f"Memory loaded successfully from {filepath}"
        except Exception as e:
            return f"Error loading memory: {str(e)}"

    async def _arun(self, filepath: str) -> str:
        return self._run(filepath)


# Factory function to create tool instances
def create_memory_tools() -> List[BaseTool]:
    """Create and return a list of all memory tools sharing a common store."""
    store = StructuredMemoryStore()
    return [
        SetMemoryTool(store=store),
        GetMemoryTool(store=store),
        GetSubtreeTool(store=store),
        ListKeysTool(store=store),
        SaveMemoryTool(store=store),
        LoadMemoryTool(store=store),
    ]
