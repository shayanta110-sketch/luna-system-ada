import json
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum


class ActionType(Enum):
    ANSWER = "ANSWER"
    EXPLORE = "EXPLORE"
    BACK = "BACK"


@dataclass
class ExplorationState:
    """Represents a node in the exploration tree."""
    action: ActionType
    context_snapshot: str
    reasoning: str
    depth: int
    parent_id: Optional[str] = None
    step_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])
    children: List[str] = field(default_factory=list)
    node_summary: Optional[str] = None


class RLMEngine:
    """
    Recursive Language Model Engine for handling arbitrarily long contexts.
    Uses LLM-driven decisions to explore relevant parts of external memory.
    """

    def __init__(self, llm_callable: Callable[[str], str], max_depth: int = 5, max_exploration_branches: int = 3):
        """
        Initialize RLM Engine.

        Args:
            llm_callable: Function that takes a prompt string and returns LLM response string.
            max_depth: Maximum recursion depth for exploration.
            max_exploration_branches: Maximum number of EXPLORE actions per node.
        """
        self.llm_call = llm_callable
        self.max_depth = max_depth
        self.max_branches = max_exploration_branches
        self.external_memory: Dict[str, Any] = {}
        self.step_history: Dict[str, ExplorationState] = {}
        self.root_step_id: Optional[str] = None
        self.node_summary_cache: Dict[str, str] = {}

    def ingest_long_text(self, text: str, memory_id: Optional[str] = None) -> str:
        """
        Store long text into external memory with hierarchical chunking.

        Args:
            text: The long text to ingest.
            memory_id: Optional identifier for the memory block.

        Returns:
            Memory reference ID for later retrieval.
        """
        if memory_id is None:
            memory_id = hashlib.md5(text.encode()).hexdigest()[:16]

        # Hierarchical chunking: large chunks, medium chunks, small chunks
        large_chunk_size = 2000
        medium_chunk_size = 500
        small_chunk_size = 100

        large_chunks = [text[i:i + large_chunk_size] for i in range(0, len(text), large_chunk_size)]
        
        # Generate summaries for each chunk level
        chunk_hierarchy = {
            "large": [],
            "medium": [],
            "small": []
        }
        
        for idx, chunk in enumerate(large_chunks):
            large_summary = self._generate_summary(chunk)
            chunk_hierarchy["large"].append({
                "index": idx,
                "content": chunk,
                "summary": large_summary
            })
            
            # Medium chunks from large chunk
            medium_chunks = [chunk[i:i + medium_chunk_size] for i in range(0, len(chunk), medium_chunk_size)]
            for midx, mchunk in enumerate(medium_chunks):
                medium_summary = self._generate_summary(mchunk)
                chunk_hierarchy["medium"].append({
                    "parent_large_idx": idx,
                    "index": midx,
                    "content": mchunk,
                    "summary": medium_summary
                })
                
                # Small chunks from medium chunk
                small_chunks = [mchunk[i:i + small_chunk_size] for i in range(0, len(mchunk), small_chunk_size)]
                for sidx, schunk in enumerate(small_chunks):
                    small_summary = self._generate_summary(schunk)
                    chunk_hierarchy["small"].append({
                        "parent_medium_idx": midx,
                        "parent_large_idx": idx,
                        "index": sidx,
                        "content": schunk,
                        "summary": small_summary
                    })
        
        # Store full hierarchical structure
        self.external_memory[memory_id] = {
            "full_text": text,
            "hierarchy": chunk_hierarchy,
            "total_length": len(text),
            "overall_summary": self._generate_summary(text[:1500])
        }
        
        # Build index for keyword lookup
        self._build_index(memory_id, text)
        
        return memory_id

    def _build_index(self, memory_id: str, text: str) -> None:
        """Simple keyword index for exploration guidance."""
        if "indices" not in self.external_memory:
            self.external_memory["indices"] = {}
        
        words = set(text.lower().split())
        keywords = [w for w in words if w.isalnum() and len(w) > 3]
        
        self.external_memory["indices"][memory_id] = keywords[:100]

    def _generate_summary(self, text_sample: str) -> str:
        """Generate a summary of a text sample using LLM with caching."""
        text_hash = hashlib.md5(text_sample.encode()).hexdigest()[:16]
        if text_hash in self.node_summary_cache:
            return self.node_summary_cache[text_hash]
        
        prompt = f"Summarize the following text in one short sentence (max 20 words):\n\n{text_sample}"
        try:
            summary = self.llm_call(prompt).strip()
            self.node_summary_cache[text_hash] = summary
            return summary
        except Exception:
            return "[Summary generation failed]"

    def _retrieve_relevant_chunks(self, query: str, memory_id: str, top_k: int = 3, granularity: str = "medium") -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks based on keyword overlap and granularity."""
        memory_block = self.external_memory.get(memory_id)
        if not memory_block:
            return []
        
        query_words = set(query.lower().split())
        hierarchy = memory_block["hierarchy"]
        chunks = hierarchy.get(granularity, [])
        
        scored_chunks = []
        for chunk_info in chunks:
            chunk_words = set(chunk_info["content"].lower().split())
            score = len(query_words.intersection(chunk_words))
            scored_chunks.append((score, chunk_info))
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk_info for _, chunk_info in scored_chunks[:top_k]]

    def process_context(self, query: str, initial_context: str = "", memory_refs: Optional[List[str]] = None) -> str:
        """
        Process a query by recursively exploring external memory as needed.

        Args:
            query: User query or task.
            initial_context: Initial context string (e.g., user instruction).
            memory_refs: List of memory IDs that contain relevant external context.

        Returns:
            Final answer from the LLM.
        """
        if memory_refs is None:
            memory_refs = [m for m in self.external_memory.keys() if m != "indices"]
        
        root_context = f"Initial Context: {initial_context}\nQuery: {query}\n\nAvailable external memory IDs: {memory_refs}"
        root_state = ExplorationState(
            action=ActionType.EXPLORE,
            context_snapshot=root_context,
            reasoning="Starting exploration",
            depth=0
        )
        self.root_step_id = root_state.step_id
        self.step_history[root_state.step_id] = root_state
        
        final_answer = self._explore(
            step_id=root_state.step_id,
            query=query,
            memory_refs=memory_refs,
            depth=0
        )
        
        return final_answer

    def _explore(self, step_id: str, query: str, memory_refs: List[str], depth: int) -> str:
        """
        Internal recursive exploration logic.
        Returns answer string when ANSWER action is taken or max depth reached.
        """
        if depth >= self.max_depth:
            return self._force_answer(query, step_id)
        
        current_state = self.step_history[step_id]
        current_context = current_state.context_snapshot
        
        decision_prompt = self._build_decision_prompt(
            query=query,
            current_context=current_context,
            memory_refs=memory_refs,
            depth=depth,
            history=self._get_exploration_history(step_id)
        )
        
        try:
            llm_response = self.llm_call(decision_prompt)
            decision = self._parse_decision(llm_response)
        except Exception as e:
            decision = {"action": ActionType.ANSWER, "reasoning": f"Error in LLM call: {e}"}
        
        action = decision.get("action", ActionType.ANSWER)
        reasoning = decision.get("reasoning", "No reasoning provided")
        
        if action == ActionType.ANSWER:
            answer_prompt = f"Based on the following context and query, provide a final answer:\n\nContext:\n{current_context}\n\nQuery: {query}\n\nAnswer:"
            final_answer = self.llm_call(answer_prompt)
            current_state.action = ActionType.ANSWER
            current_state.reasoning = reasoning
            return final_answer
        
        elif action == ActionType.BACK:
            current_state.action = ActionType.BACK
            current_state.reasoning = reasoning
            fallback_prompt = f"Given that backtracking is needed, provide best possible answer from current context:\n\n{current_context}\n\nQuery: {query}\n\nAnswer:"
            return self.llm_call(fallback_prompt)
        
        elif action == ActionType.EXPLORE:
            current_state.action = ActionType.EXPLORE
            current_state.reasoning = reasoning
            
            exploration_target = decision.get("target", "")
            granularity = decision.get("granularity", "medium")
            new_context = self._fetch_external_context(exploration_target, query, memory_refs, granularity)
            
            child_state = ExplorationState(
                action=ActionType.EXPLORE,
                context_snapshot=f"Exploring: {exploration_target}\nRetrieved context:\n{new_context}\n\nParent context:\n{current_context}",
                reasoning=f"Exploring target: {exploration_target}",
                depth=depth + 1,
                parent_id=step_id
            )
            
            # Cache node summary
            child_state.node_summary = self._generate_summary(new_context[:500])
            
            self.step_history[child_state.step_id] = child_state
            current_state.children.append(child_state.step_id)
            
            return self._explore(child_state.step_id, query, memory_refs, depth + 1)
        
        else:
            return self._force_answer(query, step_id)

    def _build_decision_prompt(self, query: str, current_context: str, memory_refs: List[str], depth: int, history: str) -> str:
        """Construct prompt for LLM to decide next action."""
        memory_summary = "Available external memory:\n"
        for mem_id in memory_refs:
            if mem_id in self.external_memory:
                summary = self.external_memory[mem_id].get("overall_summary", "No summary")
                memory_summary += f"- {mem_id}: {summary}\n"
                
                # Show chunk summaries for hierarchical retrieval
                hierarchy = self.external_memory[mem_id].get("hierarchy", {})
                if hierarchy.get("large"):
                    memory_summary += f"  Large chunks: {len(hierarchy['large'])} available\n"
                if hierarchy.get("medium"):
                    memory_summary += f"  Medium chunks: {len(hierarchy['medium'])} available\n"
        
        prompt = f"""You are controlling an exploration agent. Your task is to answer the query by recursively exploring external memory.

QUERY: {query}

CURRENT CONTEXT (already known):
{current_context[:1500]}

{memory_summary}

EXPLORATION HISTORY:
{history[:500]}

CURRENT DEPTH: {depth} (max depth: {self.max_depth})

You must choose ONE action:
- ANSWER: You have enough information to answer the query.
- EXPLORE: Need more information. Specify target (memory ID or keyword query) and granularity ("large", "medium", or "small").
- BACK: Current exploration is not helpful, backtrack.

Respond in JSON format:
{{"action": "ANSWER", "reasoning": "why you can answer"}}
{{"action": "EXPLORE", "target": "memory_id or keyword query", "granularity": "medium", "reasoning": "what you need to find"}}
{{"action": "BACK", "reasoning": "why this path is unhelpful"}}

Your response (JSON only):"""
        return prompt

    def _parse_decision(self, llm_output: str) -> Dict[str, Any]:
        """Parse LLM decision output into structured action."""
        try:
            start = llm_output.find('{')
            end = llm_output.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = llm_output[start:end]
                data = json.loads(json_str)
                action_str = data.get("action", "ANSWER").upper()
                if action_str == "ANSWER":
                    action = ActionType.ANSWER
                elif action_str == "EXPLORE":
                    action = ActionType.EXPLORE
                elif action_str == "BACK":
                    action = ActionType.BACK
                else:
                    action = ActionType.ANSWER
                return {
                    "action": action,
                    "reasoning": data.get("reasoning", ""),
                    "target": data.get("target", ""),
                    "granularity": data.get("granularity", "medium")
                }
        except Exception:
            pass
        
        return {"action": ActionType.ANSWER, "reasoning": "Parsing failed, defaulting to ANSWER"}

    def _fetch_external_context(self, target: str, query: str, memory_refs: List[str], granularity: str = "medium") -> str:
        """Fetch relevant context from external memory based on target and granularity."""
        if target in self.external_memory and target != "indices":
            mem = self.external_memory[target]
            hierarchy = mem.get("hierarchy", {})
            chunks = hierarchy.get(granularity, [])
            if chunks:
                # Return top chunks of specified granularity
                relevant_chunks = self._retrieve_relevant_chunks(query, target, top_k=2, granularity=granularity)
                if relevant_chunks:
                    return f"[Memory {target}]\nOverall Summary: {mem['overall_summary']}\n\nRelevant {granularity} chunks:\n" + "\n---\n".join([c['content'][:500] for c in relevant_chunks])
            return f"[Memory {target}]\nOverall Summary: {mem['overall_summary']}\nFirst large chunk preview: {hierarchy.get('large', [{}])[0].get('content', '')[:500]}"
        else:
            retrieved_parts = []
            for mem_id in memory_refs:
                chunks = self._retrieve_relevant_chunks(target, mem_id, top_k=2, granularity=granularity)
                if chunks:
                    for chunk in chunks:
                        retrieved_parts.append(f"From {mem_id} ({granularity} chunk):\n{chunk['content'][:500]}\nSummary: {chunk.get('summary', '')}")
            if retrieved_parts:
                return "\n\n".join(retrieved_parts)
            else:
                for mem_id in memory_refs:
                    mem = self.external_memory.get(mem_id)
                    if mem and mem.get("hierarchy", {}).get("large"):
                        return f"Fallback context from {mem_id}:\n{mem['hierarchy']['large'][0]['content'][:1000]}"
                return "No additional context found."

    def _get_exploration_history(self, current_step_id: str) -> str:
        """Build exploration history string from root to current step."""
        history_parts = []
        state = self.step_history.get(current_step_id)
        while state and state.parent_id:
            summary = state.node_summary if state.node_summary else state.reasoning[:100]
            history_parts.insert(0, f"Step {state.step_id}: {summary}")
            state = self.step_history.get(state.parent_id)
        if history_parts:
            return " -> ".join(history_parts)
        return "No previous exploration steps."

    def _force_answer(self, query: str, step_id: str) -> str:
        """Generate answer when max depth reached or error occurs."""
        current_state = self.step_history[step_id]
        forced_prompt = f"Provide the best possible answer to the query based only on the following context. You must answer even if incomplete.\n\nContext:\n{current_state.context_snapshot}\n\nQuery: {query}\n\nAnswer:"
        return self.llm_call(forced_prompt)