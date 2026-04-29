"""Recursive Search module for deep topic exploration.

Inspired by NanoSage's approach: break complex queries into sub-queries,
perform relevance-driven recursive search, and generate structured reports.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Awaitable
from enum import Enum

logger = logging.getLogger(__name__)


class RelevanceLevel(Enum):
    """Relevance level for search results."""
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    IRRELEVANT = 0


@dataclass
class SearchResult:
    """Single search result with relevance scoring."""
    content: str
    source: str
    relevance: RelevanceLevel
    metadata: Dict[str, Any] = field(default_factory=dict)
    sub_queries_generated: List[str] = field(default_factory=list)


@dataclass
class SubQuery:
    """A sub-query generated from the original query."""
    text: str
    parent_query: str
    depth: int
    expected_relevance: RelevanceLevel


@dataclass
class RecursiveSearchReport:
    """Structured report from recursive search."""
    original_query: str
    total_results: int
    max_depth_reached: int
    results_by_depth: Dict[int, List[SearchResult]]
    sub_queries_executed: List[SubQuery]
    summary: str
    key_findings: List[str]
    relevance_distribution: Dict[str, int]


class RecursiveSearchEngine:
    """Main engine for recursive, relevance-driven search.
    
    Features:
    - Query decomposition into sub-queries
    - Depth-limited recursive exploration
    - Relevance scoring and pruning
    - Structured report generation
    """
    
    def __init__(
        self,
        search_function: Callable[[str], Awaitable[List[SearchResult]]],
        relevance_threshold: RelevanceLevel = RelevanceLevel.MEDIUM,
        max_depth: int = 3,
        max_breadth: int = 5,
        similarity_threshold: float = 0.6,
    ):
        """Initialize the recursive search engine.
        
        Args:
            search_function: Async function that takes a query string and returns List[SearchResult]
            relevance_threshold: Minimum relevance to continue recursion (default: MEDIUM)
            max_depth: Maximum recursion depth (default: 3)
            max_breadth: Maximum sub-queries per node (default: 5)
            similarity_threshold: Threshold for considering results similar (0-1)
        """
        self.search_function = search_function
        self.relevance_threshold = relevance_threshold
        self.max_depth = max_depth
        self.max_breadth = max_breadth
        self.similarity_threshold = similarity_threshold
        
    async def search(self, query: str, initial_depth: int = 0) -> RecursiveSearchReport:
        """Perform recursive search starting from the given query.
        
        Args:
            query: The original search query
            initial_depth: Starting depth (default 0)
            
        Returns:
            Structured report of search results
        """
        logger.info(f"Starting recursive search for: {query}")
        
        all_results: Dict[int, List[SearchResult]] = {}
        all_sub_queries: List[SubQuery] = []
        
        # Initial search
        initial_results = await self._search_with_relevance(query)
        all_results[initial_depth] = initial_results
        
        # Generate sub-queries from high-relevance results
        if initial_depth < self.max_depth:
            sub_queries = await self._generate_sub_queries(
                query, initial_results, initial_depth
            )
            all_sub_queries.extend(sub_queries)
            
            # Recursively search each sub-query
            for sub_query in sub_queries[:self.max_breadth]:
                if sub_query.expected_relevance.value >= self.relevance_threshold.value:
                    deeper_results = await self._recursive_search(
                        sub_query.text, 
                        sub_query.depth + 1
                    )
                    for depth, results in deeper_results.items():
                        if depth not in all_results:
                            all_results[depth] = []
                        all_results[depth].extend(results)
        
        # Generate report
        return self._generate_report(query, all_results, all_sub_queries)
    
    async def _recursive_search(
        self, query: str, current_depth: int
    ) -> Dict[int, List[SearchResult]]:
        """Internal recursive search method.
        
        Args:
            query: Query to search
            current_depth: Current recursion depth
            
        Returns:
            Dictionary mapping depth to list of results
        """
        if current_depth > self.max_depth:
            return {}
        
        results = await self._search_with_relevance(query)
        depth_results = {current_depth: results}
        
        if current_depth < self.max_depth:
            sub_queries = await self._generate_sub_queries(
                query, results, current_depth
            )
            
            for sub_query in sub_queries[:self.max_breadth]:
                if sub_query.expected_relevance.value >= self.relevance_threshold.value:
                    deeper = await self._recursive_search(sub_query.text, current_depth + 1)
                    for depth, res_list in deeper.items():
                        if depth not in depth_results:
                            depth_results[depth] = []
                        depth_results[depth].extend(res_list)
        
        return depth_results
    
    async def _search_with_relevance(self, query: str) -> List[SearchResult]:
        """Execute search and filter by relevance threshold.
        
        Args:
            query: Search query string
            
        Returns:
            Filtered list of search results
        """
        results = await self.search_function(query)
        return [
            r for r in results 
            if r.relevance.value >= self.relevance_threshold.value
        ]
    
    async def _generate_sub_queries(
        self, 
        parent_query: str, 
        results: List[SearchResult],
        depth: int
    ) -> List[SubQuery]:
        """Generate sub-queries from search results.
        
        This is a heuristic-based method. Override for custom decomposition.
        
        Args:
            parent_query: Original parent query
            results: Search results to analyze
            depth: Current depth
            
        Returns:
            List of generated sub-queries
        """
        sub_queries = []
        
        # Extract key terms from high-relevance results
        key_terms = set()
        for result in results:
            if result.relevance == RelevanceLevel.HIGH:
                # Simple term extraction - override for NLP-based extraction
                words = result.content.split()[:20]  # First 20 words
                key_terms.update([w.lower() for w in words if len(w) > 5])
        
        # Generate sub-query for each key term
        for term in list(key_terms)[:self.max_breadth]:
            sub_query_text = f"{parent_query} {term}"
            sub_queries.append(SubQuery(
                text=sub_query_text,
                parent_query=parent_query,
                depth=depth,
                expected_relevance=RelevanceLevel.MEDIUM
            ))
        
        # If no terms extracted, add a generic deeper query
        if not sub_queries and results:
            sub_queries.append(SubQuery(
                text=f"more details about {parent_query}",
                parent_query=parent_query,
                depth=depth,
                expected_relevance=RelevanceLevel.LOW
            ))
        
        return sub_queries
    
    def _generate_report(
        self,
        query: str,
        results_by_depth: Dict[int, List[SearchResult]],
        sub_queries: List[SubQuery]
    ) -> RecursiveSearchReport:
        """Generate structured report from search results.
        
        Args:
            query: Original query
            results_by_depth: Results organized by depth
            sub_queries: All executed sub-queries
            
        Returns:
            Structured report
        """
        total_results = sum(len(results) for results in results_by_depth.values())
        max_depth = max(results_by_depth.keys()) if results_by_depth else 0
        
        # Calculate relevance distribution
        relevance_counts = {level.name: 0 for level in RelevanceLevel}
        for results in results_by_depth.values():
            for result in results:
                relevance_counts[result.relevance.name] += 1
        
        # Extract key findings (unique content snippets)
        key_findings = []
        for depth, results in sorted(results_by_depth.items()):
            for result in results[:3]:  # Top 3 per depth
                snippet = result.content[:200]
                if snippet and snippet not in key_findings:
                    key_findings.append(snippet)
        
        # Generate summary
        summary = (
            f"Recursive search on '{query}' explored {len(results_by_depth)} depth levels "
            f"(max depth {max_depth}), found {total_results} relevant results "
            f"with {sub_queries} sub-queries."
        )
        
        return RecursiveSearchReport(
            original_query=query,
            total_results=total_results,
            max_depth_reached=max_depth,
            results_by_depth=results_by_depth,
            sub_queries_executed=sub_queries,
            summary=summary,
            key_findings=key_findings,
            relevance_distribution=relevance_counts
        )


class RelevanceScorer:
    """Utility for scoring result relevance to a query."""
    
    @staticmethod
    def simple_keyword_overlap(query: str, content: str) -> RelevanceLevel:
        """Score based on keyword overlap.
        
        Args:
            query: Search query
            content: Result content
            
        Returns:
            Relevance level
        """
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        overlap = len(query_words.intersection(content_words))
        total = len(query_words)
        
        if total == 0:
            return RelevanceLevel.LOW
        
        ratio = overlap / total
        if ratio >= 0.6:
            return RelevanceLevel.HIGH
        elif ratio >= 0.3:
            return RelevanceLevel.MEDIUM
        elif ratio >= 0.1:
            return RelevanceLevel.LOW
        else:
            return RelevanceLevel.IRRELEVANT
    
    @staticmethod
    def semantic_scorer(
        embedding_function: Callable[[str], List[float]],
        similarity_threshold: float = 0.7
    ) -> Callable[[str, str], RelevanceLevel]:
        """Create a scorer based on semantic similarity.
        
        Args:
            embedding_function: Function that returns vector embedding for text
            similarity_threshold: Threshold for high relevance (cosine similarity)
            
        Returns:
            Scoring function
        """
        def scorer(query: str, content: str) -> RelevanceLevel:
            query_emb = embedding_function(query)
            content_emb = embedding_function(content[:500])  # Limit length
            
            # Cosine similarity
            dot = sum(a*b for a,b in zip(query_emb, content_emb))
            norm_q = sum(a*a for a in query_emb)**0.5
            norm_c = sum(b*b for b in content_emb)**0.5
            
            if norm_q == 0 or norm_c == 0:
                return RelevanceLevel.LOW
            
            similarity = dot / (norm_q * norm_c)
            
            if similarity >= 0.7:
                return RelevanceLevel.HIGH
            elif similarity >= 0.4:
                return RelevanceLevel.MEDIUM
            elif similarity >= 0.2:
                return RelevanceLevel.LOW
            else:
                return RelevanceLevel.IRRELEVANT
        
        return scorer


# Example usage and mock search function for testing
async def mock_search_function(query: str) -> List[SearchResult]:
    """Mock search function for demonstration.
    
    Args:
        query: Search query
        
    Returns:
        Mock search results
    """
    # Simulated results
    mock_docs = [
        ("The quick brown fox jumps over the lazy dog", "doc1"),
        ("Machine learning is a subset of artificial intelligence", "doc2"),
        ("Deep learning uses neural networks with many layers", "doc3"),
        ("Natural language processing enables computers to understand text", "doc4"),
    ]
    
    results = []
    for content, source in mock_docs:
        relevance = RelevanceScorer.simple_keyword_overlap(query, content)
        if relevance != RelevanceLevel.IRRELEVANT:
            results.append(SearchResult(
                content=content,
                source=source,
                relevance=relevance
            ))
    
    return results


async def demo():
    """Demonstration of recursive search."""
    engine = RecursiveSearchEngine(
        search_function=mock_search_function,
        relevance_threshold=RelevanceLevel.MEDIUM,
        max_depth=2,
        max_breadth=3
    )
    
    report = await engine.search("machine learning")
    
    print(f"Summary: {report.summary}")
    print(f"Total results: {report.total_results}")
    print(f"Max depth: {report.max_depth_reached}")
    print(f"Key findings: {report.key_findings[:3]}")
    print(f"Relevance distribution: {report.relevance_distribution}")


if __name__ == "__main__":
    asyncio.run(demo())
