"""
Tree of Thought (ToT) Engine for multi-step reasoning.
Supports BFS, DFS, and Beam Search strategies.
"""

import copy
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import deque
import heapq


@dataclass
class ThoughtNode:
    """Node in the Tree of Thought."""
    thought: str
    score: float = 0.0
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = field(default_factory=list)
    depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToTEngine:
    """
    Tree of Thought reasoning engine with multiple search strategies.
    """

    def __init__(
        self,
        llm_callable: Callable[[str], str],
        evaluator_callable: Optional[Callable[[str], float]] = None,
        max_depth: int = 3,
        num_thoughts_per_step: int = 3,
        beam_width: int = 3,
        temperature: float = 0.7,
        strategy: str = "bfs",
    ):
        """
        Initialize ToT Engine.

        Args:
            llm_callable: Function that takes a prompt and returns generated thought string.
            evaluator_callable: Function that takes a thought string and returns score (0-1). If None, LLM will be used for evaluation.
            max_depth: Maximum depth of reasoning tree.
            num_thoughts_per_step: Number of candidate thoughts to generate at each step.
            beam_width: Width for beam search (keeps top-k nodes).
            temperature: Sampling temperature for thought generation.
            strategy: Search strategy - "bfs", "dfs", or "beam".
        """
        self.llm = llm_callable
        self.evaluator = evaluator_callable
        self.max_depth = max_depth
        self.num_thoughts_per_step = num_thoughts_per_step
        self.beam_width = beam_width
        self.temperature = temperature
        self.strategy = strategy.lower()

        if self.strategy not in ["bfs", "dfs", "beam"]:
            raise ValueError(f"Unsupported strategy: {strategy}. Use 'bfs', 'dfs', or 'beam'.")

    def _generate_thoughts(
        self, prompt: str, parent_thought: Optional[str] = None, num: int = None
    ) -> List[str]:
        """
        Generate candidate thoughts based on current context.

        Args:
            prompt: Base problem prompt.
            parent_thought: Previous thought step (if any).
            num: Number of thoughts to generate.

        Returns:
            List of generated thought strings.
        """
        if num is None:
            num = self.num_thoughts_per_step

        if parent_thought:
            context = f"{prompt}\n\nPrevious reasoning: {parent_thought}\n\nNext thought(s):"
        else:
            context = f"{prompt}\n\nLet's think step by step. Possible next thoughts:"

        thoughts = []
        for _ in range(num):
            try:
                thought = self.llm(context)
                thoughts.append(thought.strip())
            except Exception as e:
                print(f"Error generating thought: {e}")
                thoughts.append(f"[ERROR] {e}")

        return thoughts

    def _evaluate_thought(self, thought: str, context: str = "") -> float:
        """
        Evaluate the quality/score of a thought.

        Args:
            thought: Thought string to evaluate.
            context: Additional context for evaluation.

        Returns:
            Score between 0 and 1.
        """
        if self.evaluator is not None:
            try:
                return float(self.evaluator(thought))
            except Exception:
                return 0.5

        # Default LLM-based evaluation
        eval_prompt = f"""Evaluate the following reasoning step on a scale of 0 to 1, where 0 is completely wrong/useless and 1 is perfect/relevant.
Reasoning step: "{thought}"
Score (just return a number between 0 and 1, nothing else):"""
        try:
            response = self.llm(eval_prompt)
            # Extract first float from response
            import re
            match = re.search(r"(\d+\.?\d*)", response)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            return 0.5
        except Exception:
            return 0.5

    def solve(
        self,
        problem: str,
        initial_thought: Optional[str] = None,
        max_thoughts: int = 20,
        return_tree: bool = False,
    ) -> Union[str, Tuple[str, ThoughtNode]]:
        """
        Solve problem using Tree of Thought reasoning.

        Args:
            problem: Problem description.
            initial_thought: Optional initial thought to start the tree.
            max_thoughts: Maximum total thoughts to generate.
            return_tree: If True, return (best_solution, root_node).

        Returns:
            Best solution string, or tuple with root node if return_tree=True.
        """
        if self.strategy == "bfs":
            result, root = self._bfs_search(problem, initial_thought, max_thoughts)
        elif self.strategy == "dfs":
            result, root = self._dfs_search(problem, initial_thought, max_thoughts)
        elif self.strategy == "beam":
            result, root = self._beam_search(problem, initial_thought, max_thoughts)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        if return_tree:
            return result, root
        return result

    def _bfs_search(
        self, problem: str, initial_thought: Optional[str], max_thoughts: int
    ) -> Tuple[str, ThoughtNode]:
        """BFS strategy for Tree of Thought."""
        root = ThoughtNode(thought=initial_thought or "Start", depth=0)
        queue = deque([root])
        best_solution = ""
        best_score = -1.0

        thoughts_generated = 0

        while queue and thoughts_generated < max_thoughts:
            level_size = len(queue)
            for _ in range(level_size):
                node = queue.popleft()

                if node.depth >= self.max_depth:
                    continue

                # Generate candidate thoughts
                candidate_thoughts = self._generate_thoughts(
                    problem, node.thought, self.num_thoughts_per_step
                )
                thoughts_generated += len(candidate_thoughts)

                for thought_str in candidate_thoughts:
                    score = self._evaluate_thought(thought_str, problem)
                    child = ThoughtNode(
                        thought=thought_str,
                        score=score,
                        parent=node,
                        depth=node.depth + 1,
                    )
                    node.children.append(child)

                    if score > best_score:
                        best_score = score
                        best_solution = self._reconstruct_path(child)

                    if child.depth < self.max_depth:
                        queue.append(child)

        return best_solution, root

    def _dfs_search(
        self, problem: str, initial_thought: Optional[str], max_thoughts: int
    ) -> Tuple[str, ThoughtNode]:
        """DFS strategy for Tree of Thought."""
        root = ThoughtNode(thought=initial_thought or "Start", depth=0)
        best_solution = ""
        best_score = -1.0
        thoughts_generated = 0

        def dfs_recursive(node: ThoughtNode) -> None:
            nonlocal best_solution, best_score, thoughts_generated
            if thoughts_generated >= max_thoughts:
                return

            if node.depth >= self.max_depth:
                return

            # Generate children
            candidate_thoughts = self._generate_thoughts(
                problem, node.thought, self.num_thoughts_per_step
            )
            thoughts_generated += len(candidate_thoughts)

            # Sort by score (descending) to explore promising paths first
            scored_children = []
            for thought_str in candidate_thoughts:
                score = self._evaluate_thought(thought_str, problem)
                child = ThoughtNode(
                    thought=thought_str,
                    score=score,
                    parent=node,
                    depth=node.depth + 1,
                )
                node.children.append(child)
                scored_children.append((score, child))

                if score > best_score:
                    best_score = score
                    best_solution = self._reconstruct_path(child)

            # Explore children in order of decreasing score
            scored_children.sort(key=lambda x: x[0], reverse=True)
            for _, child in scored_children:
                if thoughts_generated >= max_thoughts:
                    break
                dfs_recursive(child)

        dfs_recursive(root)
        return best_solution, root

    def _beam_search(
        self, problem: str, initial_thought: Optional[str], max_thoughts: int
    ) -> Tuple[str, ThoughtNode]:
        """Beam Search strategy for Tree of Thought."""
        root = ThoughtNode(thought=initial_thought or "Start", depth=0)
        root.score = self._evaluate_thought(root.thought, problem)

        # Beam: list of (negative_score, node) for min-heap (we want top scores)
        beam = [(-root.score, root)]
        best_solution = ""
        best_score = -1.0
        thoughts_generated = 0

        for depth in range(self.max_depth):
            if thoughts_generated >= max_thoughts:
                break

            candidates = []
            for _, node in beam:
                if node.depth != depth:
                    continue

                # Generate thoughts for current node
                candidate_thoughts = self._generate_thoughts(
                    problem, node.thought, self.num_thoughts_per_step
                )
                thoughts_generated += len(candidate_thoughts)

                for thought_str in candidate_thoughts:
                    score = self._evaluate_thought(thought_str, problem)
                    child = ThoughtNode(
                        thought=thought_str,
                        score=score,
                        parent=node,
                        depth=node.depth + 1,
                    )
                    node.children.append(child)
                    candidates.append((-score, child))

                    if score > best_score:
                        best_score = score
                        best_solution = self._reconstruct_path(child)

            # Keep top beam_width candidates
            heapq.heapify(candidates)
            beam = heapq.nsmallest(self.beam_width, candidates)

            if not beam:
                break

        return best_solution, root

    def _reconstruct_path(self, node: ThoughtNode) -> str:
        """Reconstruct reasoning path from root to given node."""
        path = []
        current = node
        while current:
            if current.thought != "Start":
                path.append(current.thought)
            current = current.parent
        return " -> ".join(reversed(path))

    def aggregate_results(
        self, root: ThoughtNode, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Aggregate best reasoning paths from tree.

        Args:
            root: Root node of reasoning tree.
            top_k: Number of top paths to return.

        Returns:
            List of (path_string, score) tuples.
        """
        all_paths = []

        def collect_paths(node: ThoughtNode, current_path: List[str]):
            current_path_copy = current_path.copy()
            if node.thought != "Start":
                current_path_copy.append(node.thought)

            if not node.children:
                # Leaf node
                path_str = " -> ".join(current_path_copy)
                all_paths.append((path_str, node.score))
            else:
                for child in node.children:
                    collect_paths(child, current_path_copy)

        collect_paths(root, [])
        all_paths.sort(key=lambda x: x[1], reverse=True)
        return all_paths[:top_k]
