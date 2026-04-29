# nexus_memory/recursive_search_skill.py

import json
import re
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from functools import wraps
import hashlib

# ============================================================================
# Decorator for marking skill tools
# ============================================================================

def skill_tool(func: Callable) -> Callable:
    """Decorator to mark a function as a skill tool for registration."""
    func._is_skill_tool = True
    return func


# ============================================================================
# Data models
# ============================================================================

@dataclass
class SearchResult:
    """Represents a single search result."""
    content: str
    source: str
    relevance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecursiveQuery:
    """Represents a decomposed sub-query for recursive search."""
    original_query: str
    sub_query: str
    depth: int
    parent_results: List[SearchResult] = field(default_factory=list)


# ============================================================================
# Recursive Search Skill Class
# ============================================================================

class RecursiveSearchSkill:
    """
    Deep recursive search skill inspired by NanoSage.
    Implements query decomposition, relevance scoring, TOC generation,
    and result synthesis.
    """
    
    def __init__(self, max_depth: int = 3, min_relevance: float = 0.4, 
                 max_results_per_query: int = 10, embedding_model: Optional[Any] = None):
        """
        Initialize recursive search skill.
        
        Args:
            max_depth: Maximum recursion depth for query decomposition
            min_relevance: Minimum relevance score threshold (0.0 to 1.0)
            max_results_per_query: Maximum results to return per query
            embedding_model: Optional embedding model for semantic similarity
        """
        self.max_depth = max_depth
        self.min_relevance = min_relevance
        self.max_results_per_query = max_results_per_query
        self.embedding_model = embedding_model
        self._search_callback: Optional[Callable] = None
        
    def register_search_callback(self, callback: Callable) -> None:
        """
        Register a search function to use for retrieving results.
        
        Args:
            callback: Function that takes (query, metadata) and returns List[SearchResult]
        """
        self._search_callback = callback
    
    def _compute_relevance(self, query: str, content: str) -> float:
        """
        Compute relevance score between query and content.
        Uses keyword matching with TF-IDF style weighting.
        
        Args:
            query: Search query
            content: Result content
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        content_terms = set(re.findall(r'\b\w+\b', content.lower()))
        
        if not query_terms:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(query_terms.intersection(content_terms))
        union = len(query_terms.union(content_terms))
        
        base_score = intersection / union if union > 0 else 0.0
        
        # Boost score for exact phrase matches
        phrase_score = 0.0
        for term in query_terms:
            if len(term) > 3 and term in content.lower():
                phrase_score += 0.05
        
        return min(1.0, base_score + phrase_score)
    
    def _decompose_query(self, query: str, depth: int) -> List[str]:
        """
        Decompose query into sub-queries for deeper search.
        
        Args:
            query: Original query string
            depth: Current recursion depth
            
        Returns:
            List of sub-query strings
        """
        if depth >= self.max_depth:
            return []
        
        sub_queries = []
        
        # Identify key concepts (noun phrases)
        concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b[a-z]{4,}\b', query)
        
        # Generate sub-queries by focusing on individual concepts
        for concept in set(concepts):
            if len(concept) > 3:
                sub_queries.append(concept)
        
        # Add "how" and "why" variants for deeper understanding
        if "what" in query.lower():
            sub_queries.append(query.replace("what", "how", 1))
            sub_queries.append(query.replace("what", "why", 1))
        
        # Limit number of sub-queries
        return list(set(sub_queries))[:5]
    
    @skill_tool
    def recursive_search(self, query: str, depth: int = 0, 
                         context: Optional[List[SearchResult]] = None) -> List[SearchResult]:
        """
        Perform recursive search with query decomposition.
        
        Args:
            query: Search query string
            depth: Current recursion depth (internal use)
            context: Previous search results for context (internal use)
            
        Returns:
            List of search results with relevance scores
        """
        if self._search_callback is None:
            raise RuntimeError("No search callback registered. Call register_search_callback() first.")
        
        if depth > self.max_depth:
            return []
        
        context = context or []
        
        # Perform initial search
        results = self._search_callback(query, {"depth": depth})
        
        # Filter and score results
        scored_results = []
        for result in results[:self.max_results_per_query]:
            relevance = self._compute_relevance(query, result.content)
            if relevance >= self.min_relevance:
                result.relevance_score = relevance
                scored_results.append(result)
        
        # Sort by relevance
        scored_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Decompose query and recurse
        sub_queries = self._decompose_query(query, depth)
        for sub_query in sub_queries:
            sub_results = self.recursive_search(sub_query, depth + 1, scored_results)
            scored_results.extend(sub_results)
        
        # Remove duplicates based on source
        seen_sources = set()
        unique_results = []
        for result in scored_results:
            if result.source not in seen_sources:
                seen_sources.add(result.source)
                unique_results.append(result)
        
        return unique_results[:self.max_results_per_query]
    
    @skill_tool
    def generate_toc(self, results: List[SearchResult], 
                     max_items: int = 20) -> Dict[str, Any]:
        """
        Generate table of contents from search results.
        
        Args:
            results: List of search results
            max_items: Maximum number of TOC entries
            
        Returns:
            Dictionary containing TOC structure
        """
        toc = {
            "title": "Search Results Table of Contents",
            "sections": [],
            "total_entries": len(results),
            "summary": {}
        }
        
        # Group results by source domain or type
        groups = {}
        for result in results[:max_items]:
            source_type = result.metadata.get("type", "general")
            if source_type not in groups:
                groups[source_type] = []
            groups[source_type].append(result)
        
        # Build sections
        for group_name, group_results in groups.items():
            section = {
                "title": group_name.capitalize(),
                "entries": []
            }
            
            for idx, result in enumerate(group_results[:10]):
                entry = {
                    "index": idx + 1,
                    "title": result.metadata.get("title", "Untitled"),
                    "source": result.source,
                    "relevance": result.relevance_score,
                    "preview": result.content[:200] + "..." if len(result.content) > 200 else result.content
                }
                section["entries"].append(entry)
            
            toc["sections"].append(section)
        
        # Add summary statistics
        toc["summary"] = {
            "avg_relevance": sum(r.relevance_score for r in results) / len(results) if results else 0,
            "unique_sources": len(set(r.source for r in results)),
            "max_relevance": max((r.relevance_score for r in results), default=0)
        }
        
        return toc
    
    @skill_tool
    def synthesize_findings(self, query: str, results: List[SearchResult],
                           max_length: int = 2000) -> Dict[str, Any]:
        """
        Synthesize findings from search results into coherent summary.
        
        Args:
            query: Original search query
            results: List of search results
            max_length: Maximum length of synthesized text
            
        Returns:
            Dictionary containing synthesis results
        """
        if not results:
            return {
                "query": query,
                "findings": "No relevant results found.",
                "key_points": [],
                "confidence": 0.0,
                "sources_analyzed": 0
            }
        
        # Sort results by relevance
        sorted_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)
        top_results = sorted_results[:min(5, len(sorted_results))]
        
        # Extract key points from each result
        key_points = []
        all_content = []
        
        for result in top_results:
            # Extract sentences that contain query terms
            sentences = re.split(r'[.!?]+', result.content)
            relevant_sentences = []
            
            query_terms = set(re.findall(r'\b\w+\b', query.lower()))
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(term in sentence_lower for term in query_terms):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                key_points.extend(relevant_sentences[:3])
                all_content.append(" ".join(relevant_sentences))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_key_points = []
        for point in key_points:
            if point not in seen:
                seen.add(point)
                unique_key_points.append(point)
        
        # Generate synthesis
        synthesis = " ".join(all_content)
        if len(synthesis) > max_length:
            synthesis = synthesis[:max_length] + "..."
        
        # Calculate confidence based on result quality
        avg_relevance = sum(r.relevance_score for r in top_results) / len(top_results) if top_results else 0
        confidence = min(1.0, avg_relevance * 1.2)  # Boost confidence slightly
        
        return {
            "query": query,
            "findings": synthesis if synthesis else "Unable to synthesize findings from available results.",
            "key_points": unique_key_points[:10],
            "confidence": confidence,
            "sources_analyzed": len(top_results),
            "total_results": len(results)
        }


# ============================================================================
# SKILL_MD_TEMPLATE - Reference documentation
# ============================================================================

SKILL_MD_TEMPLATE = """# Recursive Search Skill

## Overview
This skill provides deep recursive search capabilities inspired by NanoSage. It enables query decomposition, relevance scoring, automatic table of contents generation, and intelligent result synthesis.

## Tools

### `recursive_search(query: str, depth: int = 0) -> List[SearchResult]`
Performs recursive search with query decomposition.
- **query**: Search query string
- **depth**: Current recursion depth (internal use)
- **Returns**: List of search results with relevance scores

### `generate_toc(results: List[SearchResult], max_items: int = 20) -> Dict[str, Any]`
Generates structured table of contents from search results.
- **results**: List of search results
- **max_items**: Maximum TOC entries
- **Returns**: TOC dictionary with sections and entries

### `synthesize_findings(query: str, results: List[SearchResult], max_length: int = 2000) -> Dict[str, Any]`
Synthesizes findings into coherent summary.
- **query**: Original search query
- **results**: List of search results
- **max_length**: Maximum synthesis length
- **Returns**: Synthesis dictionary with findings and key points

## Usage Example

```python
from nexus_memory.recursive_search_skill import RecursiveSearchSkill, skill_tool

# Initialize skill
skill = RecursiveSearchSkill(max_depth=3, min_relevance=0.4)

# Register search callback
def my_search_function(query, metadata):
    # Implement search logic
    return [SearchResult(content="...", source="...", relevance_score=0.0)]

skill.register_search_callback(my_search_function)

# Perform recursive search
results = skill.recursive_search("What is machine learning?")

# Generate TOC
toc = skill.generate_toc(results)

# Synthesize findings
summary = skill.synthesize_findings("What is machine learning?", results)
```

## Parameters
- `max_depth`: Maximum recursion depth (default: 3)
- `min_relevance`: Minimum relevance threshold (default: 0.4)
- `max_results_per_query`: Max results per query (default: 10)

## SearchResult Structure
- `content`: Text content of the result
- `source`: Source identifier (URL, document ID, etc.)
- `relevance_score`: Computed relevance score (0.0-1.0)
- `metadata`: Additional metadata dictionary

## Implementation Notes
- Uses Jaccard similarity for relevance scoring
- Query decomposition extracts key concepts and variants
- Results are deduplicated by source
- TOC groups results by source type
- Synthesis extracts relevant sentences containing query terms
"""


# ============================================================================
# Helper function to extract skill tools
# ============================================================================

def get_skill_tools(skill_instance: RecursiveSearchSkill) -> Dict[str, Callable]:
    """
    Extract all methods marked with @skill_tool decorator.
    
    Args:
        skill_instance: Instance of RecursiveSearchSkill
        
    Returns:
        Dictionary mapping tool names to bound methods
    """
    tools = {}
    for attr_name in dir(skill_instance):
        attr = getattr(skill_instance, attr_name)
        if callable(attr) and hasattr(attr, '_is_skill_tool'):
            tools[attr_name] = attr
    return tools
