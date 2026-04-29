"""Multi-Agent System coordinator for managing specialized sub-agents.

This module provides a lightweight coordinator that handles intent detection,
task delegation, and sub-agent lifecycle management.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union
from uuid import uuid4

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Status of a sub-agent in the system."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"


class IntentCategory(str, Enum):
    """Pre-defined intent categories for task routing."""
    QUERY = "query"
    COMPUTE = "compute"
    TRANSFORM = "transform"
    ANALYZE = "analyze"
    MONITOR = "monitor"
    CUSTOM = "custom"


@dataclass
class Task:
    """Represents a task to be processed by a sub-agent."""
    id: str = field(default_factory=lambda: str(uuid4()))
    intent: Union[str, IntentCategory] = ""
    payload: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1 (highest) to 10 (lowest)
    created_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())


@dataclass
class TaskResult:
    """Result returned by a sub-agent after processing a task."""
    task_id: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0


class SubAgent(ABC):
    """Base abstract class for all specialized sub-agents."""

    def __init__(self, name: str, capabilities: List[Union[str, IntentCategory]]):
        self.name = name
        self.capabilities = capabilities
        self.status = AgentStatus.IDLE
        self.current_task_id: Optional[str] = None

    @abstractmethod
    async def process(self, task: Task) -> TaskResult:
        """Process a task and return the result.
        
        Args:
            task: The task to process
            
        Returns:
            TaskResult with processing outcome
        """
        pass

    def can_handle(self, intent: Union[str, IntentCategory]) -> bool:
        """Check if this agent can handle the given intent."""
        return intent in self.capabilities

    async def start(self) -> None:
        """Start the sub-agent (override for custom initialization)."""
        self.status = AgentStatus.IDLE
        logger.info(f"Sub-agent '{self.name}' started")

    async def stop(self) -> None:
        """Stop the sub-agent gracefully."""
        self.status = AgentStatus.STOPPED
        logger.info(f"Sub-agent '{self.name}' stopped")

    def get_status(self) -> Dict[str, Any]:
        """Return current status information."""
        return {
            "name": self.name,
            "status": self.status.value,
            "capabilities": [str(c) for c in self.capabilities],
            "current_task": self.current_task_id,
        }


class IntentDetector:
    """Detects intent from user input or task payload."""

    def __init__(self, custom_rules: Optional[Dict[str, Callable[[Any], bool]]] = None):
        self.custom_rules = custom_rules or {}

    async def detect(self, payload: Any) -> IntentCategory:
        """Detect the intent category from payload.
        
        Args:
            payload: The task payload to analyze
            
        Returns:
            Detected IntentCategory
        """
        if isinstance(payload, str):
            return self._detect_from_string(payload)
        elif isinstance(payload, dict):
            if "intent" in payload:
                intent_val = payload["intent"]
                if isinstance(intent_val, IntentCategory):
                    return intent_val
                if isinstance(intent_val, str):
                    try:
                        return IntentCategory(intent_val)
                    except ValueError:
                        pass
            return self._detect_from_dict(payload)
        else:
            # Default to CUSTOM for non-string, non-dict payloads
            return IntentCategory.CUSTOM

    def _detect_from_string(self, text: str) -> IntentCategory:
        """Detect intent from string input."""
        text_lower = text.lower()
        
        # Simple keyword-based detection
        if any(word in text_lower for word in ["what", "who", "where", "when", "why", "how", "explain", "describe"]):
            return IntentCategory.QUERY
        elif any(word in text_lower for word in ["calculate", "compute", "sum", "average", "count", "total"]):
            return IntentCategory.COMPUTE
        elif any(word in text_lower for word in ["convert", "format", "parse", "extract", "transform"]):
            return IntentCategory.TRANSFORM
        elif any(word in text_lower for word in ["analyze", "evaluate", "assess", "examine"]):
            return IntentCategory.ANALYZE
        elif any(word in text_lower for word in ["monitor", "watch", "track", "check"]):
            return IntentCategory.MONITOR
        else:
            return IntentCategory.CUSTOM

    def _detect_from_dict(self, data: Dict) -> IntentCategory:
        """Detect intent from dictionary payload."""
        # Check for explicit intent field
        if "intent" in data:
            intent = data["intent"]
            if isinstance(intent, IntentCategory):
                return intent
            if isinstance(intent, str):
                try:
                    return IntentCategory(intent)
                except ValueError:
                    pass
        
        # Check for operation type
        if "operation" in data:
            op = data["operation"]
            if op in ["query", "select", "find"]:
                return IntentCategory.QUERY
            if op in ["compute", "calculate"]:
                return IntentCategory.COMPUTE
        
        # Apply custom rules
        for category, rule in self.custom_rules.items():
            if isinstance(category, IntentCategory):
                if rule(data):
                    return category
        
        return IntentCategory.CUSTOM


class MultiAgentCoordinator:
    """Lightweight coordinator for managing multiple sub-agents."""

    def __init__(self, intent_detector: Optional[IntentDetector] = None):
        self.agents: Dict[str, SubAgent] = {}
        self.intent_detector = intent_detector or IntentDetector()
        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

    def register_agent(self, agent: SubAgent) -> None:
        """Register a new sub-agent with the coordinator.
        
        Args:
            agent: The SubAgent instance to register
        """
        if agent.name in self.agents:
            raise ValueError(f"Agent with name '{agent.name}' already registered")
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name} with capabilities {[str(c) for c in agent.capabilities]}")

    def unregister_agent(self, agent_name: str) -> None:
        """Unregister a sub-agent from the coordinator.
        
        Args:
            agent_name: Name of the agent to unregister
        """
        if agent_name in self.agents:
            del self.agents[agent_name]
            logger.info(f"Unregistered agent: {agent_name}")

    async def start(self) -> None:
        """Start the coordinator and all registered agents."""
        if self._running:
            logger.warning("Coordinator already running")
            return
        
        # Start all agents
        for agent in self.agents.values():
            await agent.start()
        
        # Start worker task
        self._running = True
        self._worker_task = asyncio.create_task(self._process_queue())
        logger.info("Multi-Agent Coordinator started")

    async def stop(self) -> None:
        """Stop the coordinator and all agents gracefully."""
        if not self._running:
            logger.warning("Coordinator not running")
            return
        
        self._running = False
        
        # Cancel worker task
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        # Stop all agents
        stop_tasks = [agent.stop() for agent in self.agents.values()]
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        logger.info("Multi-Agent Coordinator stopped")

    async def submit_task(self, payload: Any, intent: Optional[Union[str, IntentCategory]] = None,
                         priority: int = 5, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Submit a task to the coordinator.
        
        Args:
            payload: Task payload
            intent: Optional explicit intent (auto-detected if not provided)
            priority: Task priority (1-10, lower is higher priority)
            metadata: Additional metadata for the task
            
        Returns:
            Task ID
        """
        # Detect or use provided intent
        if intent is None:
            detected_intent = await self.intent_detector.detect(payload)
        else:
            detected_intent = intent if isinstance(intent, IntentCategory) else IntentCategory(intent)
        
        task = Task(
            intent=detected_intent,
            payload=payload,
            metadata=metadata or {},
            priority=priority
        )
        
        # Priority queue uses (priority, timestamp) for ordering
        # Lower priority number = higher priority
        await self._task_queue.put((priority, task.created_at, task))
        logger.debug(f"Task {task.id} submitted with intent {detected_intent.value}")
        
        return task.id

    async def _process_queue(self) -> None:
        """Background worker to process tasks from the queue."""
        while self._running:
            try:
                # Wait for a task
                priority, timestamp, task = await self._task_queue.get()
                
                # Find suitable agent
                agent = self._select_agent(task.intent)
                if agent is None:
                    logger.error(f"No agent available for intent: {task.intent}")
                    continue
                
                # Execute task
                if agent.status == AgentStatus.IDLE:
                    agent.status = AgentStatus.BUSY
                    agent.current_task_id = task.id
                    
                    try:
                        start_time = asyncio.get_event_loop().time()
                        result = await agent.process(task)
                        result.processing_time = asyncio.get_event_loop().time() - start_time
                        
                        if result.success:
                            logger.info(f"Task {task.id} completed successfully by {agent.name}")
                        else:
                            logger.error(f"Task {task.id} failed on {agent.name}: {result.error}")
                    except Exception as e:
                        logger.exception(f"Exception during task {task.id} processing: {e}")
                    finally:
                        agent.status = AgentStatus.IDLE
                        agent.current_task_id = None
                else:
                    # Agent busy, re-queue task
                    await self._task_queue.put((priority, timestamp, task))
                    await asyncio.sleep(0.01)  # Small delay to avoid tight loop
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in queue processor: {e}")
                await asyncio.sleep(0.1)

    def _select_agent(self, intent: Union[str, IntentCategory]) -> Optional[SubAgent]:
        """Select the best agent for the given intent.
        
        Currently uses first idle agent that can handle the intent.
        Can be overridden for more sophisticated routing.
        """
        for agent in self.agents.values():
            if agent.status == AgentStatus.IDLE and agent.can_handle(intent):
                return agent
        return None

    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered agents.
        
        Returns:
            Dictionary mapping agent names to their status dicts
        """
        return {name: agent.get_status() for name, agent in self.agents.items()}

    def get_queue_size(self) -> int:
        """Get the number of pending tasks in the queue."""
        return self._task_queue.qsize()
