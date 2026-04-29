"""
Incremental Processor Module

Inspired by PRISM architecture for long-running, multi-step tasks.
Provides incremental processing with hierarchical memory management,
structured key-value cache, and methods for chunking large inputs.
"""

from typing import Dict, Any, List, Optional, Callable, Union
from collections import OrderedDict
import json
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum


class MemoryLevel(Enum):
    """Hierarchical memory levels."""
    L1_CORE = "l1_core"       # Small, fast, working memory
    L2_TASK = "l2_task"       # Task-specific memory
    L3_LONG_TERM = "l3_long_term"  # Persistent, slower memory
    L4_ARCHIVE = "l4_archive"      # Compressed, batch storage


@dataclass
class MemoryEntry:
    """Single entry in the structured memory cache."""
    key: str
    value: Any
    level: MemoryLevel
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def touch(self):
        """Update access time and count."""
        self.timestamp = time.time()
        self.access_count += 1


class StructuredMemoryCache:
    """
    Key-value memory cache with hierarchical levels and eviction policies.
    
    Features:
    - Multi-level memory (L1, L2, L3, L4)
    - LRU-based eviction per level
    - Promotable entries (L1 <- L2 <- L3)
    - TTL support
    """
    
    def __init__(self, capacity_mapping: Optional[Dict[MemoryLevel, int]] = None):
        """
        Initialize cache with capacities per level.
        
        Args:
            capacity_mapping: Dict mapping MemoryLevel to max entries.
                             Default: L1=100, L2=1000, L3=10000, L4=unlimited
        """
        self.capacities = capacity_mapping or {
            MemoryLevel.L1_CORE: 100,
            MemoryLevel.L2_TASK: 1000,
            MemoryLevel.L3_LONG_TERM: 10000,
            MemoryLevel.L4_ARCHIVE: None  # Unlimited
        }
        self._store: Dict[MemoryLevel, OrderedDict[str, MemoryEntry]] = {
            level: OrderedDict() for level in self.capacities
        }
        
    def set(self, key: str, value: Any, level: MemoryLevel = MemoryLevel.L2_TASK,
            ttl: Optional[float] = None, metadata: Optional[Dict] = None) -> None:
        """
        Store a value in the cache at specified level.
        
        Args:
            key: Unique identifier
            value: Data to store
            level: Memory level
            ttl: Time-to-live in seconds (optional)
            metadata: Additional metadata
        """
        if ttl:
            metadata = metadata or {}
            metadata['ttl'] = ttl
            metadata['expires_at'] = time.time() + ttl
            
        entry = MemoryEntry(key=key, value=value, level=level, metadata=metadata or {})
        self._evict_if_needed(level)
        self._store[level][key] = entry
        
    def get(self, key: str, level: Optional[MemoryLevel] = None) -> Optional[Any]:
        """
        Retrieve a value from cache.
        
        Args:
            key: Entry key
            level: Specific level to search, or None for all levels (L1->L2->L3->L4)
            
        Returns:
            Value if found and not expired, else None
        """
        if level:
            return self._get_from_level(key, level)
        
        # Search from fastest to slowest
        for lvl in [MemoryLevel.L1_CORE, MemoryLevel.L2_TASK,
                    MemoryLevel.L3_LONG_TERM, MemoryLevel.L4_ARCHIVE]:
            if lvl in self._store:
                val = self._get_from_level(key, lvl)
                if val is not None:
                    return val
        return None
    
    def _get_from_level(self, key: str, level: MemoryLevel) -> Optional[Any]:
        """Internal method to get from specific level and update access."""
        if level not in self._store:
            return None
            
        store = self._store[level]
        if key not in store:
            return None
            
        entry = store[key]
        
        # Check expiration
        if 'expires_at' in entry.metadata and entry.metadata['expires_at'] < time.time():
            del store[key]
            return None
            
        entry.touch()
        # Move to end for LRU
        store.move_to_end(key)
        return entry.value
    
    def promote(self, key: str, from_level: MemoryLevel, to_level: MemoryLevel) -> bool:
        """
        Promote an entry from one level to a faster level.
        
        Returns:
            True if promotion succeeded, False otherwise
        """
        if from_level not in self._store or to_level not in self._store:
            return False
            
        entry = self._store[from_level].get(key)
        if not entry:
            return False
            
        self._evict_if_needed(to_level)
        self._store[to_level][key] = entry
        entry.level = to_level
        del self._store[from_level][key]
        return True
    
    def _evict_if_needed(self, level: MemoryLevel) -> None:
        """Evict oldest entries if capacity exceeded."""
        capacity = self.capacities.get(level)
        if capacity is None:
            return
            
        store = self._store[level]
        while len(store) >= capacity:
            # Remove oldest (first item in OrderedDict)
            key, _ = store.popitem(last=False)
            
    def clear_level(self, level: MemoryLevel) -> None:
        """Clear all entries at a specific memory level."""
        if level in self._store:
            self._store[level].clear()
            
    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            level.name: len(store)
            for level, store in self._store.items()
        }


class HierarchicalMemoryManager:
    """
    Manages memory across hierarchical levels with automatic demotion/promotion.
    
    Implements PRISM-like strategies:
    - Frequent access promotion to faster memory
    - Aging and demotion for cold data
    - Consolidation from L4 to archival storage
    """
    
    def __init__(self, cache: StructuredMemoryCache):
        self.cache = cache
        self.access_threshold_promote = 5  # Accesses needed to promote from L3 to L2
        self.idle_seconds_demote = 300     # 5 minutes idle to demote from L1 to L2
        
    def process_access(self, key: str) -> None:
        """
        Update memory hierarchy based on access pattern.
        Should be called whenever data is accessed.
        """
        # Find entry across levels
        for lvl in [MemoryLevel.L1_CORE, MemoryLevel.L2_TASK, MemoryLevel.L3_LONG_TERM]:
            if lvl in self.cache._store and key in self.cache._store[lvl]:
                entry = self.cache._store[lvl][key]
                
                # Promotion logic: frequently accessed L3 -> L2, L2 -> L1
                if lvl == MemoryLevel.L3_LONG_TERM and entry.access_count >= self.access_threshold_promote:
                    self.cache.promote(key, MemoryLevel.L3_LONG_TERM, MemoryLevel.L2_TASK)
                elif lvl == MemoryLevel.L2_TASK and entry.access_count >= self.access_threshold_promote * 2:
                    self.cache.promote(key, MemoryLevel.L2_TASK, MemoryLevel.L1_CORE)
                break
    
    def age_memory(self) -> None:
        """
        Age all memory entries and demote cold data.
        Should be called periodically (e.g., every 100 steps).
        """
        now = time.time()
        
        # Demote from L1 to L2 if idle too long
        for key, entry in list(self.cache._store.get(MemoryLevel.L1_CORE, {}).items()):
            if now - entry.timestamp > self.idle_seconds_demote:
                if self.cache.promote(key, MemoryLevel.L1_CORE, MemoryLevel.L2_TASK):
                    # It's a demote, so we actually move down
                    self.cache._store[MemoryLevel.L2_TASK][key] = entry
                    del self.cache._store[MemoryLevel.L1_CORE][key]
                    entry.level = MemoryLevel.L2_TASK
                    
        # Demote from L2 to L3 if idle for longer
        for key, entry in list(self.cache._store.get(MemoryLevel.L2_TASK, {}).items()):
            if now - entry.timestamp > self.idle_seconds_demote * 2:
                self.cache._store[MemoryLevel.L3_LONG_TERM][key] = entry
                del self.cache._store[MemoryLevel.L2_TASK][key]
                entry.level = MemoryLevel.L3_LONG_TERM


class IncrementalProcessor:
    """
    Main class for incremental processing of large inputs over multi-step tasks.
    
    Features:
    - Chunk large inputs into manageable pieces
    - Maintain state across processing steps
    - Checkpoint and resume capability
    - Integration with hierarchical memory
    """
    
    def __init__(self, chunk_size: int = 1000, cache_capacity: Optional[Dict[MemoryLevel, int]] = None):
        """
        Args:
            chunk_size: Default number of items per chunk
            cache_capacity: Memory capacity configuration
        """
        self.chunk_size = chunk_size
        self.cache = StructuredMemoryCache(cache_capacity)
        self.memory_manager = HierarchicalMemoryManager(self.cache)
        self.checkpoint_id = None
        self.current_step = 0
        
    def chunk_data(self, data: List[Any], chunk_size: Optional[int] = None) -> List[List[Any]]:
        """Split large data into chunks for incremental processing."""
        size = chunk_size or self.chunk_size
        return [data[i:i+size] for i in range(0, len(data), size)]
    
    def process_chunk(self, chunk: List[Any], processor_fn: Callable[[List[Any], Dict[str, Any]], Dict[str, Any]],
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a single chunk with access to cached memory.
        
        Args:
            chunk: Data chunk to process
            processor_fn: Function that takes (chunk, memory_context) and returns results
            context: Additional processing context
            
        Returns:
            Processing results for this chunk
        """
        # Build memory context from cache
        memory_context = self._build_memory_context()
        
        # Process chunk
        result = processor_fn(chunk, memory_context)
        
        # Store important results in memory
        if 'memory_updates' in result:
            for key, value in result['memory_updates'].items():
                level = result.get('memory_level', MemoryLevel.L2_TASK)
                self.cache.set(key, value, level=level)
                
        # Track access patterns
        for key in memory_context.get('accessed_keys', []):
            self.memory_manager.process_access(key)
            
        self.current_step += 1
        return result
    
    def process_incremental(self, data: List[Any], processor_fn: Callable,
                           chunk_size: Optional[int] = None,
                           on_chunk_complete: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Process entire dataset incrementally by chunks.
        
        Args:
            data: Full input data
            processor_fn: Function to process each chunk
            chunk_size: Override default chunk size
            on_chunk_complete: Callback after each chunk (receives chunk_index, results)
            
        Returns:
            List of results for all chunks
        """
        chunks = self.chunk_data(data, chunk_size)
        results = []
        
        for idx, chunk in enumerate(chunks):
            result = self.process_chunk(chunk, processor_fn, {'chunk_index': idx})
            results.append(result)
            
            if on_chunk_complete:
                on_chunk_complete(idx, result)
                
            # Age memory after each chunk to prevent bloat
            if idx % 10 == 0:
                self.memory_manager.age_memory()
                
        return results
    
    def _build_memory_context(self) -> Dict[str, Any]:
        """Build context dictionary from cache for processor function."""
        context = {
            'l1': {},
            'l2': {},
            'l3': {},
            'accessed_keys': []
        }
        
        # Extract from L1 (core memory)
        for key, entry in self.cache._store.get(MemoryLevel.L1_CORE, {}).items():
            context['l1'][key] = entry.value
            context['accessed_keys'].append(key)
            
        # Extract from L2 (task memory)
        for key, entry in self.cache._store.get(MemoryLevel.L2_TASK, {}).items():
            context['l2'][key] = entry.value
            
        # Extract from L3 (long-term)
        for key, entry in self.cache._store.get(MemoryLevel.L3_LONG_TERM, {}).items():
            context['l3'][key] = entry.value
            
        return context
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save current processing state to disk."""
        checkpoint = {
            'current_step': self.current_step,
            'chunk_size': self.chunk_size,
            'memory': {}
        }
        
        # Serialize cache (only L2 and L3 for checkpoint)
        for level in [MemoryLevel.L2_TASK, MemoryLevel.L3_LONG_TERM]:
            checkpoint['memory'][level.value] = {
                key: {
                    'value': entry.value,
                    'metadata': entry.metadata
                }
                for key, entry in self.cache._store.get(level, {}).items()
            }
            
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
            
    def load_checkpoint(self, filepath: str) -> None:
        """Restore processing state from checkpoint."""
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
            
        self.current_step = checkpoint['current_step']
        self.chunk_size = checkpoint['chunk_size']
        
        for level_name, entries in checkpoint['memory'].items():
            level = MemoryLevel(level_name)
            for key, data in entries.items():
                self.cache.set(key, data['value'], level=level, metadata=data['metadata'])


def create_hash_key(data: Any) -> str:
    """Utility to create a hash key for any data."""
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
