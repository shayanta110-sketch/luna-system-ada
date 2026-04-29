"""Tree of Thought (ToT) module for advanced reasoning.

This module implements a Tree of Thought reasoning structure that breaks
complex tasks into hierarchical subtasks, explores multiple solution paths,
and collaborates with TokenBudgetManager to stay within context limits.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid


class ThoughtStatus(Enum):
    """Status of a thought node in the tree."""
    PENDING = "pending"
    EXPLORING = "exploring"
    COMPLETED = "completed"
    PRUNED = "pruned"
    FAILED = "failed"


@dataclass
class ThoughtNode:
    """A single thought node in the Tree of Thought."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0
    score: float = 0.0
    status: ThoughtStatus = ThoughtStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_estimate: int = 0


class TreeOfThought:
    """Tree of Thought reasoning engine with token budget management."""
    
    def __init__(self, token_budget_manager=None, max_depth: int = 5,
                 max_branches: int = 3, exploration_strategy: str = "bfs"):
        """
        Initialize the Tree of Thought.
        
        Args:
            token_budget_manager: Instance of TokenBudgetManager for context tracking.
            max_depth: Maximum depth of the thought tree.
            max_branches: Maximum number of child branches per node.
            exploration_strategy: "bfs" or "dfs" for exploration order.
        """
        self.token_budget_manager = token_budget_manager
        self.max_depth = max_depth
        self.max_branches = max_branches
        self.exploration_strategy = exploration_strategy
        self.nodes: Dict[str, ThoughtNode] = {}
        self.root_id: Optional[str] = None
        self.current_node_id: Optional[str] = None
        
    def initialize_root(self, task_description: str) -> str:
        """Create the root thought node representing the main task."""
        root = ThoughtNode(
            content=task_description,
            depth=0,
            status=ThoughtStatus.PENDING
        )
        self.nodes[root.id] = root
        self.root_id = root.id
        self.current_node_id = root.id
        
        # Update token budget
        if self.token_budget_manager:
            root.token_estimate = self._estimate_tokens(task_description)
            self.token_budget_manager.consume_tokens(root.token_estimate)
        
        return root.id
    
    def add_thought(self, parent_id: str, thought_content: str,
                    score: float = 0.0) -> Optional[str]:
        """
        Add a new thought node under a parent node.
        
        Args:
            parent_id: ID of the parent node.
            thought_content: Content of the thought.
            score: Evaluation score for this thought path.
            
        Returns:
            ID of the new node, or None if budget exceeded or max branches reached.
        """
        if parent_id not in self.nodes:
            return None
            
        parent = self.nodes[parent_id]
        
        # Check branch limit
        if len(parent.children_ids) >= self.max_branches:
            return None
        
        # Check depth limit
        if parent.depth >= self.max_depth:
            return None
        
        # Check token budget
        token_estimate = self._estimate_tokens(thought_content)
        if self.token_budget_manager:
            if not self.token_budget_manager.can_consume(token_estimate):
                return None
        
        # Create new node
        new_node = ThoughtNode(
            content=thought_content,
            parent_id=parent_id,
            depth=parent.depth + 1,
            score=score,
            token_estimate=token_estimate
        )
        self.nodes[new_node.id] = new_node
        parent.children_ids.append(new_node.id)
        
        # Consume tokens
        if self.token_budget_manager:
            self.token_budget_manager.consume_tokens(token_estimate)
        
        return new_node.id
    
    def evaluate_thought(self, node_id: str, score: float) -> None:
        """Assign a score to a thought node."""
        if node_id in self.nodes:
            self.nodes[node_id].score = score
            self.nodes[node_id].status = ThoughtStatus.COMPLETED
    
    def prune_branch(self, node_id: str) -> None:
        """Prune a subtree from the given node."""
        if node_id not in self.nodes:
            return
            
        def _prune_recursive(nid: str):
            node = self.nodes[nid]
            node.status = ThoughtStatus.PRUNED
            for child_id in node.children_ids:
                _prune_recursive(child_id)
        
        _prune_recursive(node_id)
    
    def get_best_path(self) -> List[ThoughtNode]:
        """Retrieve the highest-scoring path from root to a leaf."""
        if not self.root_id:
            return []
        
        best_path = []
        best_score = -float('inf')
        
        def _dfs(node_id: str, current_path: List[ThoughtNode]):
            node = self.nodes[node_id]
            current_path.append(node)
            
            if not node.children_ids and node.score > best_score:
                # Leaf node
                nonlocal best_path, best_score
                best_path = current_path.copy()
                best_score = node.score
            else:
                for child_id in node.children_ids:
                    _dfs(child_id, current_path)
            
            current_path.pop()
        
        _dfs(self.root_id, [])
        return best_path
    
    def get_next_node_to_explore(self) -> Optional[str]:
        """Get the next node to explore based on exploration strategy."""
        if self.exploration_strategy == "bfs":
            return self._get_next_bfs()
        elif self.exploration_strategy == "dfs":
            return self._get_next_dfs()
        return None
    
    def _get_next_bfs(self) -> Optional[str]:
        """BFS exploration: process nodes level by level."""
        from collections import deque
        
        if not self.root_id:
            return None
        
        queue = deque([self.root_id])
        visited = set()
        
        while queue:
            node_id = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            node = self.nodes[node_id]
            
            if node.status == ThoughtStatus.PENDING:
                return node_id
            
            for child_id in node.children_ids:
                if child_id not in visited:
                    queue.append(child_id)
        
        return None
    
    def _get_next_dfs(self) -> Optional[str]:
        """DFS exploration: go deep first."""
        if not self.root_id:
            return None
        
        visited = set()
        
        def _dfs_search(node_id: str) -> Optional[str]:
            if node_id in visited:
                return None
            visited.add(node_id)
            node = self.nodes[node_id]
            
            if node.status == ThoughtStatus.PENDING:
                return node_id
            
            for child_id in node.children_ids:
                result = _dfs_search(child_id)
                if result:
                    return result
            
            return None
        
        return _dfs_search(self.root_id)
    
    def get_tree_summary(self) -> Dict[str, Any]:
        """Get a summary of the current tree state."""
        if not self.root_id:
            return {"error": "No tree initialized"}
        
        def _count_nodes(node_id: str) -> Dict:
            node = self.nodes[node_id]
            counts = {"total": 1, "completed": 1 if node.status == ThoughtStatus.COMPLETED else 0,
                      "pruned": 1 if node.status == ThoughtStatus.PRUNED else 0,
                      "pending": 1 if node.status == ThoughtStatus.PENDING else 0}
            for child_id in node.children_ids:
                child_counts = _count_nodes(child_id)
                counts["total"] += child_counts["total"]
                counts["completed"] += child_counts["completed"]
                counts["pruned"] += child_counts["pruned"]
                counts["pending"] += child_counts["pending"]
            return counts
        
        counts = _count_nodes(self.root_id)
        
        if self.token_budget_manager:
            budget_info = self.token_budget_manager.get_status()
        else:
            budget_info = {"available": "unlimited"}
        
        return {
            "total_nodes": counts["total"],
            "completed_nodes": counts["completed"],
            "pruned_nodes": counts["pruned"],
            "pending_nodes": counts["pending"],
            "max_depth": self.max_depth,
            "max_branches": self.max_branches,
            "exploration_strategy": self.exploration_strategy,
            "token_budget": budget_info
        }
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token approximate)."""
        return len(text) // 4
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize the entire tree to a dictionary."""
        tree_data = {
            "max_depth": self.max_depth,
            "max_branches": self.max_branches,
            "exploration_strategy": self.exploration_strategy,
            "root_id": self.root_id,
            "current_node_id": self.current_node_id,
            "nodes": {}
        }
        
        for node_id, node in self.nodes.items():
            tree_data["nodes"][node_id] = {
                "id": node.id,
                "content": node.content,
                "parent_id": node.parent_id,
                "children_ids": node.children_ids,
                "depth": node.depth,
                "score": node.score,
                "status": node.status.value,
                "metadata": node.metadata,
                "token_estimate": node.token_estimate
            }
        
        return tree_data
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any], token_budget_manager=None) -> "TreeOfThought":
        """Deserialize a tree from a dictionary."""
        tree = cls(
            token_budget_manager=token_budget_manager,
            max_depth=data["max_depth"],
            max_branches=data["max_branches"],
            exploration_strategy=data["exploration_strategy"]
        )
        tree.root_id = data["root_id"]
        tree.current_node_id = data["current_node_id"]
        
        for node_id, node_data in data["nodes"].items():
            node = ThoughtNode(
                id=node_data["id"],
                content=node_data["content"],
                parent_id=node_data["parent_id"],
                children_ids=node_data["children_ids"],
                depth=node_data["depth"],
                score=node_data["score"],
                status=ThoughtStatus(node_data["status"]),
                metadata=node_data["metadata"],
                token_estimate=node_data["token_estimate"]
            )
            tree.nodes[node_id] = node
        
        return tree
