"""Configuration management for Memory Context Nexus layer."""

import os
from pathlib import Path
from typing import Literal


class MemoryContextNexusConfig:
    """Centralized configuration for Memory Context Nexus."""

    def __init__(self, base_dir: str | Path = None):
        if base_dir is None:
            base_dir = Path.home() / ".memory_context_nexus"
        self.base_dir = Path(base_dir)
        self._ensure_directories()

        # Storage paths
        self.chromadb_path = self.base_dir / "chromadb"
        self.networkx_storage_path = self.base_dir / "networkx_graphs"
        self.hash_chain_archive_dir = self.base_dir / "hash_chains"

        # Token budget limits
        self.token_budget_total = int(os.getenv("MCN_TOKEN_BUDGET_TOTAL", "8192"))
        self.token_budget_per_module = int(os.getenv("MCN_TOKEN_BUDGET_PER_MODULE", "2048"))
        self.token_reserve = int(os.getenv("MCN_TOKEN_RESERVE", "512"))

        # Operational modes for salience gate
        self.salience_mode: Literal["conservative", "balanced", "aggressive"] = os.getenv("MCN_SALIENCE_MODE", "balanced")
        self.salience_threshold = float(os.getenv("MCN_SALIENCE_THRESHOLD", "0.6"))

        # Compression settings
        self.compression_enabled = os.getenv("MCN_COMPRESSION_ENABLED", "true").lower() == "true"
        self.compression_algorithm = os.getenv("MCN_COMPRESSION_ALGORITHM", "lz4")
        self.compression_ratio_target = float(os.getenv("MCN_COMPRESSION_RATIO", "0.7"))
        self.max_compressed_chunk_size = int(os.getenv("MCN_MAX_COMPRESSED_CHUNK", "4096"))

        # Hash chain archive settings
        self.hash_chain_max_length = int(os.getenv("MCN_HASH_CHAIN_MAX_LENGTH", "10000"))
        self.hash_chain_retention_days = int(os.getenv("MCN_HASH_CHAIN_RETENTION", "30"))
        self.hash_archive_format = os.getenv("MCN_HASH_ARCHIVE_FORMAT", "json")

    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self):
        """Export configuration as dictionary."""
        return {
            "base_dir": str(self.base_dir),
            "chromadb_path": str(self.chromadb_path),
            "networkx_storage_path": str(self.networkx_storage_path),
            "hash_chain_archive_dir": str(self.hash_chain_archive_dir),
            "token_budget_total": self.token_budget_total,
            "token_budget_per_module": self.token_budget_per_module,
            "token_reserve": self.token_reserve,
            "salience_mode": self.salience_mode,
            "salience_threshold": self.salience_threshold,
            "compression_enabled": self.compression_enabled,
            "compression_algorithm": self.compression_algorithm,
            "compression_ratio_target": self.compression_ratio_target,
            "max_compressed_chunk_size": self.max_compressed_chunk_size,
            "hash_chain_max_length": self.hash_chain_max_length,
            "hash_chain_retention_days": self.hash_chain_retention_days,
            "hash_archive_format": self.hash_archive_format,
        }


# Singleton instance
_default_config = None


def get_config() -> MemoryContextNexusConfig:
    """Get or create the default configuration instance."""
    global _default_config
    if _default_config is None:
        _default_config = MemoryContextNexusConfig()
    return _default_config
