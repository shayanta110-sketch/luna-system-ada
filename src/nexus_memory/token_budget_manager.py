"""TokenBudgetManager for advanced context composition and ToT integration."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import json
import logging

logger = logging.getLogger(__name__)


class ContextFormat(Enum):
    """Supported context formats."""
    JSON = "json"
    MARKDOWN = "markdown"
    TEXT = "text"
    YAML = "yaml"
    TREE = "tree"


@dataclass
class ContextFragment:
    """A single piece of context with metadata."""
    content: str
    format: ContextFormat
    priority: int = 0  # higher = more important
    source: Optional[str] = None
    token_count: Optional[int] = None


@dataclass
class RenderedContext:
    """Result of context rendering."""
    content: str
    token_count: int
    fragments_used: List[ContextFragment]


class TokenBudgetManager:
    """Manages token budgets and context composition across multiple formats."""

    def __init__(self, max_tokens: int = 4096, reserve_ratio: float = 0.15):
        self.max_tokens = max_tokens
        self.reserve_tokens = int(max_tokens * reserve_ratio)
        self.useable_tokens = max_tokens - self.reserve_tokens
        self.fragments: List[ContextFragment] = []
        self._format_handlers = {
            ContextFormat.JSON: self._render_json,
            ContextFormat.MARKDOWN: self._render_markdown,
            ContextFormat.TEXT: self._render_text,
            ContextFormat.YAML: self._render_yaml,
            ContextFormat.TREE: self._render_tree,
        }

    def add_fragment(self, content: Union[str, Dict, List], format: ContextFormat,
                     priority: int = 0, source: Optional[str] = None):
        """Add a context fragment."""
        if isinstance(content, (dict, list)):
            content = json.dumps(content)
        fragment = ContextFragment(
            content=content,
            format=format,
            priority=priority,
            source=source
        )
        self.fragments.append(fragment)

    def compose_context(self, query: Optional[str] = None,
                       max_tokens: Optional[int] = None) -> RenderedContext:
        """Compose context from fragments, respecting token budget."""
        limit = max_tokens or self.useable_tokens
        sorted_frags = sorted(self.fragments, key=lambda f: f.priority, reverse=True)
        selected = []
        total_tokens = 0

        for frag in sorted_frags:
            tokens = self._estimate_tokens(frag.content)
            if total_tokens + tokens <= limit:
                frag.token_count = tokens
                selected.append(frag)
                total_tokens += tokens
            else:
                # Try to truncate high-priority fragment
                if frag.priority >= 5 and total_tokens < limit:
                    truncated = self._truncate_to_budget(frag.content, limit - total_tokens)
                    trunc_tokens = self._estimate_tokens(truncated)
                    frag.content = truncated
                    frag.token_count = trunc_tokens
                    selected.append(frag)
                    total_tokens += trunc_tokens
                break

        rendered_parts = []
        for frag in selected:
            rendered = self._format_handlers[frag.format](frag.content, frag.source)
            rendered_parts.append(rendered)

        final_content = "\n\n".join(rendered_parts)
        if query:
            final_content = f"Query: {query}\n\n{final_content}"

        return RenderedContext(
            content=final_content,
            token_count=total_tokens + self._estimate_tokens(query or ""),
            fragments_used=selected
        )

    def _render_json(self, content: str, source: Optional[str]) -> str:
        """Render JSON with optional source annotation."""
        try:
            parsed = json.loads(content)
            pretty = json.dumps(parsed, indent=2)
        except:
            pretty = content
        if source:
            return f"[JSON Source: {source}]\n{pretty}"
        return pretty

    def _render_markdown(self, content: str, source: Optional[str]) -> str:
        """Render markdown content."""
        if source:
            return f"[Markdown: {source}]\n{content}"
        return content

    def _render_text(self, content: str, source: Optional[str]) -> str:
        """Render plain text."""
        if source:
            return f"[Text: {source}]\n{content}"
        return content

    def _render_yaml(self, content: str, source: Optional[str]) -> str:
        """Render YAML (simplified as text)."""
        if source:
            return f"[YAML: {source}]\n{content}"
        return content

    def _render_tree(self, content: str, source: Optional[str]) -> str:
        """Render tree-structured content (for ToT)."""
        lines = content.split('\n')
        tree_str = '\n'.join(f"  {line}" for line in lines)
        if source:
            return f"[Tree of Thought: {source}]\n{tree_str}"
        return tree_str

    def _estimate_tokens(self, text: Optional[str]) -> int:
        """Rough token estimation (4 chars per token)."""
        if not text:
            return 0
        return len(text) // 4

    def _truncate_to_budget(self, content: str, budget: int) -> str:
        """Truncate content to fit token budget."""
        if not content:
            return content
        target_chars = budget * 4
        if len(content) <= target_chars:
            return content
        return content[:target_chars] + "... [truncated]"

    # Tree of Thought Integration
    def for_tot_node(self, node_content: str, priority: int = 10) -> str:
        """Create a context fragment specifically for a ToT node."""
        self.add_fragment(
            content=node_content,
            format=ContextFormat.TREE,
            priority=priority,
            source="ToT_node"
        )
        return self.compose_context().content

    def estimate_tot_budget(self, num_nodes: int, depth: int) -> Dict[str, int]:
        """Estimate budget allocation for Tree of Thought reasoning."""
        per_node_base = self.useable_tokens // (num_nodes * 2)
        return {
            "total_budget": self.max_tokens,
            "reserve": self.reserve_tokens,
            "useable": self.useable_tokens,
            "per_node": per_node_base,
            "estimated_depth_impact": depth * per_node_base // 4
        }

    def compress_context(self, target_ratio: float = 0.5) -> RenderedContext:
        """Compress current context by removing low-priority fragments."""
        original_tokens = sum(self._estimate_tokens(f.content) for f in self.fragments)
        target_tokens = int(original_tokens * target_ratio)
        self.fragments.sort(key=lambda f: f.priority, reverse=True)
        compressed_frags = []
        current = 0
        for f in self.fragments:
            ftokens = self._estimate_tokens(f.content)
            if current + ftokens <= target_tokens:
                compressed_frags.append(f)
                current += ftokens
        self.fragments = compressed_frags
        return self.compose_context()
