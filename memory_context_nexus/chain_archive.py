"""
hash-chained archive module inspired by COMB: three-link architecture
(temporal, semantic, social) with tamper-evident hash chain.

Implements three tiers:
- Active (context window): in-memory recent entries
- Daily Staging (append-only JSONL): on-disk buffer before finalization
- Chain Archive (daily documents with hash links): permanent, tamper-evident

Includes BM25 full-text search.
"""

import json
import hashlib
import os
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterable
from collections import deque
import re

# BM25 implementation for full-text search
import math
from collections import Counter
from functools import lru_cache


class BM25:
    """Okapi BM25 ranking algorithm for full-text search."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_lengths = []
        self.avgdl = 0.0
        self.idf = {}
        self.doc_freqs = []
        self.N = 0
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase, split on non-alphanumeric."""
        text = text.lower()
        return re.findall(r'\w+', text)
    
    def fit(self, documents: List[str]):
        """Build BM25 index from list of document strings."""
        self.corpus = documents
        self.N = len(documents)
        self.doc_lengths = [len(self.tokenize(doc)) for doc in documents]
        self.avgdl = sum(self.doc_lengths) / self.N if self.N else 0
        
        # Compute document frequencies
        self.doc_freqs = []
        term_doc_counts = {}
        
        for doc in documents:
            tokens = set(self.tokenize(doc))
            freq = Counter()
            for token in tokens:
                freq[token] += 1
                term_doc_counts[token] = term_doc_counts.get(token, 0) + 1
            self.doc_freqs.append(freq)
        
        # Compute IDF
        for term, df in term_doc_counts.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
    
    def score(self, query: str) -> List[Tuple[int, float]]:
        """Return list of (doc_index, score) for query."""
        query_tokens = self.tokenize(query)
        scores = [0.0] * self.N
        
        for i, doc_freq in enumerate(self.doc_freqs):
            doc_len = self.doc_lengths[i]
            for token in query_tokens:
                if token in doc_freq:
                    tf = doc_freq[token]
                    idf = self.idf.get(token, 0)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    scores[i] += idf * (numerator / denominator)
        
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in sorted_scores if score > 0]


class ChainEntry:
    """Represents a single entry in the hash chain."""
    
    def __init__(self, data: Dict[str, Any], prev_hash: str = ""):
        self.data = data
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.prev_hash = prev_hash
        self.hash = self.compute_hash()
    
    def compute_hash(self) -> str:
        """Compute SHA-256 hash of entry content + previous hash + timestamp."""
        content = json.dumps(self.data, sort_keys=True)
        block_string = f"{content}{self.prev_hash}{self.timestamp}"
        return hashlib.sha256(block_string.encode('utf-8')).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": self.data,
            "timestamp": self.timestamp,
            "prev_hash": self.prev_hash,
            "hash": self.hash
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ChainEntry':
        entry = cls(d['data'], d['prev_hash'])
        entry.timestamp = d['timestamp']
        entry.hash = d['hash']
        return entry


class ActiveWindow:
    """Tier 1: In-memory context window with size limit."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.entries: deque = deque(maxlen=max_size)
    
    def add(self, entry: ChainEntry) -> None:
        self.entries.append(entry)
    
    def get_all(self) -> List[ChainEntry]:
        return list(self.entries)
    
    def clear(self) -> List[ChainEntry]:
        old = self.get_all()
        self.entries.clear()
        return old


class DailyStaging:
    """Tier 2: Append-only JSONL buffer for daily entries before archiving."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.current_date = None
        self.current_file = None
    
    def _get_staging_file(self, date: datetime) -> Path:
        return self.base_dir / f"staging_{date.strftime('%Y%m%d')}.jsonl"
    
    def _rotate_if_needed(self):
        today = datetime.now(timezone.utc).date()
        if self.current_date != today:
            self.current_date = today
            self.current_file = self._get_staging_file(datetime.combine(today, datetime.min.time()))
    
    def append(self, entry: ChainEntry) -> None:
        self._rotate_if_needed()
        with open(self.current_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')
    
    def read_day(self, date: datetime) -> List[ChainEntry]:
        file_path = self._get_staging_file(date)
        if not file_path.exists():
            return []
        entries = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entries.append(ChainEntry.from_dict(json.loads(line)))
        return entries
    
    def finalize_day(self, date: datetime) -> List[ChainEntry]:
        """Read and remove staging file for given date (archive it)."""
        entries = self.read_day(date)
        if entries:
            file_path = self._get_staging_file(date)
            file_path.unlink()
        return entries


class ChainArchive:
    """Tier 3: Permanent daily documents with hash links (tamper-evident)."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_archive_file(self, date: datetime) -> Path:
        return self.base_dir / f"archive_{date.strftime('%Y%m%d')}.json"
    
    def save_day(self, date: datetime, entries: List[ChainEntry]) -> str:
        """Save entries for a day, linking them in a hash chain. Returns last hash."""
        if not entries:
            return ""
        
        # Build hash chain for the day
        prev_hash = self._get_previous_day_hash(date)
        daily_blocks = []
        
        for entry in entries:
            entry.prev_hash = prev_hash
            entry.hash = entry.compute_hash()
            daily_blocks.append(entry.to_dict())
            prev_hash = entry.hash
        
        archive_data = {
            "date": date.isoformat(),
            "entries": daily_blocks,
            "first_hash": daily_blocks[0]["hash"] if daily_blocks else "",
            "last_hash": daily_blocks[-1]["hash"] if daily_blocks else "",
            "prev_day_hash": self._get_previous_day_hash(date)  # link to previous day's last hash
        }
        
        file_path = self._get_archive_file(date)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(archive_data, f, indent=2)
        
        return archive_data["last_hash"]
    
    def _get_previous_day_hash(self, date: datetime) -> str:
        """Retrieve last hash from previous day's archive."""
        prev_date = date - timedelta(days=1)
        prev_file = self._get_archive_file(prev_date)
        if prev_file.exists():
            with open(prev_file, 'r', encoding='utf-8') as f:
                prev_data = json.load(f)
                return prev_data.get("last_hash", "")
        return ""
    
    def read_day(self, date: datetime) -> Optional[Dict[str, Any]]:
        file_path = self._get_archive_file(date)
        if not file_path.exists():
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def verify_chain(self, date: datetime) -> Tuple[bool, List[str]]:
        """Verify hash chain integrity for a given day and link to previous day."""
        archive = self.read_day(date)
        if not archive:
            return False, ["No archive found"]
        
        errors = []
        entries_data = archive["entries"]
        prev_hash = archive.get("prev_day_hash", "")
        
        # Verify each entry's hash
        for i, entry_dict in enumerate(entries_data):
            stored_hash = entry_dict["hash"]
            entry = ChainEntry.from_dict(entry_dict)
            computed_hash = entry.compute_hash()
            if stored_hash != computed_hash:
                errors.append(f"Entry {i}: hash mismatch (stored {stored_hash}, computed {computed_hash})")
            
            # Verify prev_hash linkage
            if i > 0 and entry.prev_hash != entries_data[i-1]["hash"]:
                errors.append(f"Entry {i}: broken link to previous entry")
            elif i == 0 and entry.prev_hash != prev_hash:
                errors.append(f"First entry: broken link to previous day (expected {prev_hash})")
        
        # Verify end hash matches stored last_hash
        if entries_data and archive["last_hash"] != entries_data[-1]["hash"]:
            errors.append("Last hash stored does not match last entry's hash")
        
        return len(errors) == 0, errors


class ThreeTierArchive:
    """Main orchestrator for three-link architecture (temporal, semantic, social).
    
    Temporal: time-based rotation (daily staging → archive)
    Semantic: BM25 full-text search across all entries
    Social: hash chain with tamper evidence (anyone can verify)
    """
    
    def __init__(self, base_dir: Path, active_window_size: int = 100):
        self.base_dir = Path(base_dir)
        self.active = ActiveWindow(active_window_size)
        self.staging = DailyStaging(self.base_dir / "staging")
        self.archive = ChainArchive(self.base_dir / "archive")
        self.bm25_index = None
        self._rebuild_search_index()
    
    def add_entry(self, content: Dict[str, Any]) -> ChainEntry:
        """Add a new entry (temporal link: goes to active, then staging)."""
        entry = ChainEntry(content)
        self.active.add(entry)
        self.staging.append(entry)
        # Update BM25 index incrementally or mark for rebuild
        self._mark_index_dirty()
        return entry
    
    def _mark_index_dirty(self):
        """Simple approach: rebuild on next search."""
        self.bm25_index = None
    
    def _rebuild_search_index(self):
        """Rebuild BM25 index from all archived entries."""
        documents = []
        self.all_entries = []  # store metadata for lookup
        
        # Walk through archive files
        archive_dir = self.base_dir / "archive"
        if archive_dir.exists():
            for archive_file in sorted(archive_dir.glob("archive_*.json")):
                with open(archive_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for entry_dict in data.get("entries", []):
                        content_str = json.dumps(entry_dict["data"])
                        documents.append(content_str)
                        self.all_entries.append(entry_dict)
        
        # Also include staging if needed (optional)
        staging_dir = self.base_dir / "staging"
        if staging_dir.exists():
            for staging_file in staging_dir.glob("staging_*.jsonl"):
                with open(staging_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            entry_dict = json.loads(line)
                            content_str = json.dumps(entry_dict["data"])
                            documents.append(content_str)
                            self.all_entries.append(entry_dict)
        
        if documents:
            self.bm25 = BM25()
            self.bm25.fit(documents)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """Semantic link: BM25 full-text search across all entries."""
        if self.bm25 is None:
            self._rebuild_search_index()
        
        if not hasattr(self, 'bm25') or self.bm25 is None:
            return []
        
        results = self.bm25.score(query)
        top_results = []
        for idx, score in results[:top_k]:
            if idx < len(self.all_entries):
                top_results.append((self.all_entries[idx], score))
        return top_results
    
    def finalize_day(self, date: datetime) -> bool:
        """Move staging entries for a specific day into chain archive."""
        entries = self.staging.finalize_day(date)
        if entries:
            self.archive.save_day(date, entries)
            self._mark_index_dirty()
            return True
        return False
    
    def rotate_active_to_staging(self):
        """Rotate active window contents to staging (e.g., on context overflow)."""
        entries = self.active.clear()
        for entry in entries:
            self.staging.append(entry)
        self._mark_index_dirty()
    
    def verify_full_chain(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Social link: verify entire chain for tamper evidence."""
        archive_dir = self.base_dir / "archive"
        if not archive_dir.exists():
            return {"valid": True, "errors": []}
        
        archive_files = sorted(archive_dir.glob("archive_*.json"))
        errors = []
        prev_last_hash = ""
        
        for archive_file in archive_files:
            # Extract date from filename
            date_str = archive_file.stem.split('_')[1]
            file_date = datetime.strptime(date_str, "%Y%m%d")
            
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue
            
            with open(archive_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Verify internal chain
            valid, day_errors = self.archive.verify_chain(file_date)
            if not valid:
                errors.extend([f"{file_date.date()}: {err}" for err in day_errors])
            
            # Verify cross-day link
            if prev_last_hash and data.get("prev_day_hash") != prev_last_hash:
                errors.append(f"{file_date.date()}: cross-day hash mismatch (expected {prev_last_hash})")
            
            prev_last_hash = data.get("last_hash", "")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return stats about the archive."""
        archive_dir = self.base_dir / "archive"
        staging_dir = self.base_dir / "staging"
        
        archive_files = list(archive_dir.glob("archive_*.json")) if archive_dir.exists() else []
        staging_files = list(staging_dir.glob("staging_*.jsonl")) if staging_dir.exists() else []
        
        total_archived_entries = 0
        for af in archive_files:
            with open(af, 'r', encoding='utf-8') as f:
                data = json.load(f)
                total_archived_entries += len(data.get("entries", []))
        
        return {
            "active_window_size": len(self.active.entries),
            "active_max": self.active.max_size,
            "staging_days": len(staging_files),
            "archive_days": len(archive_files),
            "total_archived_entries": total_archived_entries,
            "base_directory": str(self.base_dir)
        }
