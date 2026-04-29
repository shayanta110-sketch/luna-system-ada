from enum import Enum
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import math


class BudgetPriority(Enum):
    """Priority levels for token budget allocation slots."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class TokenBudget:
    """Represents a token budget for a specific slot or component."""
    name: str
    priority: BudgetPriority
    max_tokens: int
    allocated_tokens: int = 0
    content: Optional[str] = None
    
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.allocated_tokens)
    
    def is_full(self) -> bool:
        return self.allocated_tokens >= self.max_tokens
    
    def add_content(self, content: str, token_count: int) -> bool:
        if self.allocated_tokens + token_count <= self.max_tokens:
            self.content = content
            self.allocated_tokens = token_count
            return True
        return False


class TokenBudgetManager:
    """Manages token budgets across multiple slots with priority-based allocation."""
    
    def __init__(self, total_budget: int):
        self.total_budget = total_budget
        self.budgets: List[TokenBudget] = []
        self.tokenizer: Optional[Callable[[str], int]] = None
        
    def add_budget_slot(self, name: str, priority: BudgetPriority, max_tokens: int) -> None:
        """Add a new budget slot."""
        budget = TokenBudget(name, priority, max_tokens)
        self.budgets.append(budget)
        self._reallocate_budgets()
    
    def set_tokenizer(self, tokenizer_func: Callable[[str], int]) -> None:
        """Set custom token counting function."""
        self.tokenizer = tokenizer_func
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using configured tokenizer or simple heuristic."""
        if self.tokenizer:
            return self.tokenizer(text)
        # Simple heuristic: split by whitespace and punctuation
        import re
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return len(tokens)
    
    def _reallocate_budgets(self) -> None:
        """Reallocate total budget based on priorities."""
        if not self.budgets:
            return
        
        sorted_budgets = sorted(self.budgets, key=lambda b: b.priority.value)
        total_requested = sum(b.max_tokens for b in self.budgets)
        
        if total_requested <= self.total_budget:
            return  # All can be fully allocated
        
        # Scale down lower priority budgets
        remaining_budget = self.total_budget
        for budget in sorted_budgets:
            if remaining_budget <= 0:
                budget.max_tokens = 0
                continue
            
            # Higher priority gets full requested if possible
            allocation = min(budget.max_tokens, remaining_budget)
            budget.max_tokens = allocation
            remaining_budget -= allocation
    
    def assign_to_slot(self, slot_name: str, content: str) -> bool:
        """Assign content to a specific budget slot."""
        token_count = self.count_tokens(content)
        for budget in self.budgets:
            if budget.name == slot_name:
                return budget.add_content(content, token_count)
        return False
    
    def compress_history(self, history: List[str], decay_factor: float = 0.9) -> List[str]:
        """Compress conversation history using exponential decay on older messages."""
        if not history:
            return []
        
        compressed = []
        for idx, message in enumerate(reversed(history)):
            # Older messages get higher decay (more compression)
            age_weight = decay_factor ** idx
            # Simulate compression by truncating based on weight
            max_len = int(len(message) * age_weight)
            if max_len > 0:
                truncated = message[:max_len] + ("..." if max_len < len(message) else "")
                compressed.insert(0, truncated)
            else:
                compressed.insert(0, "[compressed]")
        return compressed
    
    def extract_relevant_for_query(self, documents: List[str], query: str, top_k: int = 3) -> List[str]:
        """Extract most relevant document chunks for a query."""
        if not documents:
            return []
        
        # Simple TF-IDF like scoring (keyword overlap)
        query_terms = set(query.lower().split())
        scored = []
        for doc in documents:
            doc_lower = doc.lower()
            score = sum(1 for term in query_terms if term in doc_lower)
            scored.append((score, doc))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]
    
    def build_prompt(self, base_prompt: str, slot_contents: Dict[str, str]) -> str:
        """Build final prompt by combining base prompt and slot contents within budget."""
        prompt_parts = [base_prompt]
        
        # Add slot contents in priority order
        sorted_slots = sorted(self.budgets, key=lambda b: b.priority.value)
        for slot in sorted_slots:
            if slot.content and slot.name in slot_contents:
                prompt_parts.append(f"\n[{slot.name.upper()}]\n{slot.content}")
        
        full_prompt = "\n".join(prompt_parts)
        
        # Truncate if exceeding budget (crude but effective)
        token_count = self.count_tokens(full_prompt)
        if token_count > self.total_budget:
            # Simple truncation to approximate budget
            char_limit = int(len(full_prompt) * (self.total_budget / token_count))
            return full_prompt[:char_limit] + "...[truncated]"
        
        return full_prompt
    
    def get_usage_report(self) -> Dict[str, Any]:
        """Generate current token usage report."""
        total_allocated = sum(b.allocated_tokens for b in self.budgets)
        return {
            "total_budget": self.total_budget,
            "total_used": total_allocated,
            "remaining": self.total_budget - total_allocated,
            "slots": [
                {
                    "name": b.name,
                    "priority": b.priority.name,
                    "max_tokens": b.max_tokens,
                    "used_tokens": b.allocated_tokens,
                    "remaining_tokens": b.remaining()
                }
                for b in self.budgets
            ]
        }
