import asyncio
import json
from typing import Dict, List, Any, Callable, Optional, Union, Awaitable
from enum import Enum
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class RoutingType(Enum):
    RULE_BASED = "rule_based"
    LLM_BASED = "llm_based"


@dataclass
class Agent:
    name: str
    description: str
    capabilities: List[str]
    handler: Callable[[Dict[str, Any]], Union[Dict[str, Any], Awaitable[Dict[str, Any]]]]
    
    def __post_init__(self):
        if not callable(self.handler):
            raise ValueError(f"Handler for agent {self.name} must be callable")


@dataclass
class Task:
    intent: str
    payload: Dict[str, Any]
    routing_hint: Optional[List[str]] = None
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteResult:
    selected_agents: List[Agent]
    reasoning: str


class MultiAgentSystem:
    def __init__(self, llm_router: Optional[Callable[[str, List[Agent]], str]] = None):
        """
        Initialize Multi-Agent System.
        
        Args:
            llm_router: Optional callable that takes intent and list of agents
                       and returns the name of the selected agent.
        """
        self.agents: Dict[str, Agent] = {}
        self.llm_router = llm_router
    
    def register_agent(self, name: str, description: str, capabilities: List[str],
                       handler: Callable[[Dict[str, Any]], Union[Dict[str, Any], Awaitable[Dict[str, Any]]]]) -> None:
        """
        Register a new agent in the system.
        
        Args:
            name: Unique agent identifier
            description: What the agent does
            capabilities: List of task types the agent can handle
            handler: Async or sync function that processes payload and returns result
        """
        if name in self.agents:
            raise ValueError(f"Agent {name} already registered")
        
        agent = Agent(
            name=name,
            description=description,
            capabilities=capabilities,
            handler=handler
        )
        self.agents[name] = agent
        logger.info(f"Registered agent: {name} with capabilities: {capabilities}")
    
    def route(self, task: Task, routing_type: RoutingType = RoutingType.RULE_BASED) -> RouteResult:
        """
        Route a task to appropriate agent(s) based on intent.
        
        Args:
            task: Task to route
            routing_type: Whether to use rule-based or LLM-based routing
        
        Returns:
            RouteResult containing selected agents and reasoning
        """
        if not self.agents:
            raise RuntimeError("No agents registered in the system")
        
        # If routing hint is provided, use it directly
        if task.routing_hint:
            selected = []
            for agent_name in task.routing_hint:
                if agent_name in self.agents:
                    selected.append(self.agents[agent_name])
                else:
                    logger.warning(f"Routing hint agent {agent_name} not found")
            if selected:
                return RouteResult(selected_agents=selected, reasoning="Using routing hint")
        
        if routing_type == RoutingType.RULE_BASED:
            return self._rule_based_routing(task.intent)
        elif routing_type == RoutingType.LLM_BASED:
            return self._llm_based_routing(task.intent)
        else:
            raise ValueError(f"Unknown routing type: {routing_type}")
    
    def _rule_based_routing(self, intent: str) -> RouteResult:
        """Simple rule-based routing using keyword matching on capabilities."""
        intent_lower = intent.lower()
        matched_agents = []
        
        for agent in self.agents.values():
            for capability in agent.capabilities:
                if capability.lower() in intent_lower or intent_lower in capability.lower():
                    matched_agents.append(agent)
                    break
        
        if not matched_agents:
            # Fallback: select first agent as default
            matched_agents = [list(self.agents.values())[0]]
            reasoning = f"No capability match found for intent '{intent}'. Using default agent."
        else:
            reasoning = f"Matched intent '{intent}' with agent capabilities"
        
        return RouteResult(selected_agents=matched_agents, reasoning=reasoning)
    
    def _llm_based_routing(self, intent: str) -> RouteResult:
        """LLM-based routing using provided router function."""
        if not self.llm_router:
            raise RuntimeError("LLM router function not provided for LLM-based routing")
        
        agents_list = list(self.agents.values())
        selected_names = self.llm_router(intent, agents_list)
        
        if isinstance(selected_names, str):
            selected_names = [selected_names]
        
        selected_agents = []
        for name in selected_names:
            if name in self.agents:
                selected_agents.append(self.agents[name])
            else:
                logger.warning(f"LLM selected unknown agent: {name}")
        
        if not selected_agents:
            selected_agents = [agents_list[0]]
            reasoning = "LLM returned no valid agents, using default"
        else:
            reasoning = f"LLM selected agents: {[a.name for a in selected_agents]}"
        
        return RouteResult(selected_agents=selected_agents, reasoning=reasoning)
    
    async def process(self, task: Task, routing_type: RoutingType = RoutingType.RULE_BASED) -> Dict[str, Any]:
        """
        Process a task by routing to agents and executing in specified mode.
        
        Args:
            task: Task to process
            routing_type: Type of routing to use
        
        Returns:
            Aggregated results from all executed agents
        """
        route_result = self.route(task, routing_type)
        logger.info(f"Routing result for intent '{task.intent}': {route_result.reasoning}")
        
        if task.execution_mode == ExecutionMode.SEQUENTIAL:
            results = await self._execute_sequential(route_result.selected_agents, task.payload, task.context)
        elif task.execution_mode == ExecutionMode.PARALLEL:
            results = await self._execute_parallel(route_result.selected_agents, task.payload, task.context)
        else:
            raise ValueError(f"Unknown execution mode: {task.execution_mode}")
        
        return self._aggregate_results(results, task.execution_mode)
    
    async def _execute_sequential(self, agents: List[Agent], payload: Dict[str, Any],
                                  context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute agents sequentially, passing accumulated context between them.
        
        Args:
            agents: List of agents to execute
            payload: Original task payload
            context: Shared context that gets updated after each agent
        
        Returns:
            List of results from each agent
        """
        results = []
        current_payload = payload.copy()
        current_context = context.copy()
        
        for agent in agents:
            try:
                if asyncio.iscoroutinefunction(agent.handler):
                    result = await agent.handler(current_payload)
                else:
                    result = agent.handler(current_payload)
                
                results.append({
                    "agent": agent.name,
                    "success": True,
                    "result": result,
                    "error": None
                })
                
                # Update context with result for next agents
                current_context[agent.name] = result
                # Optionally allow payload to be updated (if result contains 'updated_payload')
                if isinstance(result, dict) and "updated_payload" in result:
                    current_payload.update(result["updated_payload"])
                    
            except Exception as e:
                logger.error(f"Agent {agent.name} failed: {str(e)}")
                results.append({
                    "agent": agent.name,
                    "success": False,
                    "result": None,
                    "error": str(e)
                })
                # Stop execution on failure unless specified otherwise
                break
        
        return results
    
    async def _execute_parallel(self, agents: List[Agent], payload: Dict[str, Any],
                                context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute agents in parallel.
        
        Args:
            agents: List of agents to execute
            payload: Original task payload
            context: Shared context (read-only for parallel execution)
        
        Returns:
            List of results from each agent
        """
        async def run_agent(agent: Agent):
            try:
                if asyncio.iscoroutinefunction(agent.handler):
                    result = await agent.handler(payload)
                else:
                    result = agent.handler(payload)
                
                return {
                    "agent": agent.name,
                    "success": True,
                    "result": result,
                    "error": None
                }
            except Exception as e:
                logger.error(f"Agent {agent.name} failed: {str(e)}")
                return {
                    "agent": agent.name,
                    "success": False,
                    "result": None,
                    "error": str(e)
                }
        
        tasks = [run_agent(agent) for agent in agents]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results
    
    def _aggregate_results(self, results: List[Dict[str, Any]], mode: ExecutionMode) -> Dict[str, Any]:
        """
        Aggregate results from multiple agents into a single response.
        
        Args:
            results: List of agent execution results
            mode: Execution mode used (for logging purposes)
        
        Returns:
            Aggregated result dictionary
        """
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        aggregated = {
            "execution_mode": mode.value,
            "total_agents": len(results),
            "successful_agents": len(successful),
            "failed_agents": len(failed),
            "results": {},
            "errors": {},
            "all_success": len(failed) == 0
        }
        
        for result in results:
            if result["success"]:
                aggregated["results"][result["agent"]] = result["result"]
            else:
                aggregated["errors"][result["agent"]] = result["error"]
        
        # If only one agent, flatten for convenience
        if len(results) == 1 and results[0]["success"]:
            aggregated["primary_result"] = results[0]["result"]
        
        return aggregated
