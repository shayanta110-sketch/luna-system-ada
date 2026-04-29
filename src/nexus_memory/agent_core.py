"""
Nexus Memory Agent Core - Central orchestration for Ada agent with multi-agent, skill, RLM, and ToT capabilities.
"""
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from nexus_memory.multi_agent.orchestrator import MultiAgentOrchestrator
from nexus_memory.skill_system.skill_registry import SkillRegistry
from nexus_memory.rlm_engine.rlm_integration import RLMEngine
from nexus_memory.tree_of_thought.tot_executor import TreeOfThoughtExecutor


@dataclass
class AgentConfig:
    """Configuration for the Ada agent core."""
    enable_multi_agent: bool = True
    enable_skill_system: bool = True
    enable_rlm_engine: bool = True
    enable_tree_of_thought: bool = False  # Optional, disabled by default
    agent_name: str = "Ada"
    max_iterations: int = 10
    verbose: bool = False


class AgentCore:
    """
    Main agent core for Ada. Integrates multi-agent orchestration, skill system,
    RLM engine, and Tree of Thought as optional capabilities.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self._initialized = False
        
        # Capability components
        self.multi_agent: Optional[MultiAgentOrchestrator] = None
        self.skill_registry: Optional[SkillRegistry] = None
        self.rlm_engine: Optional[RLMEngine] = None
        self.tot_executor: Optional[TreeOfThoughtExecutor] = None
        
        # Tool exposure for Ada
        self.exposed_tools: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize all enabled subsystems."""
        if self._initialized:
            return

        # Multi-Agent System
        if self.config.enable_multi_agent:
            self.multi_agent = MultiAgentOrchestrator(
                agent_name=self.config.agent_name,
                max_iterations=self.config.max_iterations
            )
            await self.multi_agent.initialize()
            self._expose_tool("multi_agent.orchestrate", self.multi_agent.orchestrate)

        # Skill System
        if self.config.enable_skill_system:
            self.skill_registry = SkillRegistry()
            await self.skill_registry.load_skills()
            self._expose_tool("skill.execute", self.skill_registry.execute_skill)
            self._expose_tool("skill.list", self.skill_registry.list_skills)

        # RLM Engine
        if self.config.enable_rlm_engine:
            self.rlm_engine = RLMEngine()
            await self.rlm_engine.initialize()
            self._expose_tool("rlm.reason", self.rlm_engine.reason)
            self._expose_tool("rlm.reflect", self.rlm_engine.reflect)

        # Tree of Thought (optional)
        if self.config.enable_tree_of_thought:
            self.tot_executor = TreeOfThoughtExecutor(
                max_depth=self.config.max_iterations,
                verbose=self.config.verbose
            )
            await self.tot_executor.initialize()
            self._expose_tool("tot.solve", self.tot_executor.solve_problem)
            self._expose_tool("tot.evaluate", self.tot_executor.evaluate_paths)

        self._initialized = True

    def _expose_tool(self, tool_name: str, tool_func: Any) -> None:
        """Expose a capability as a tool for the Ada agent."""
        self.exposed_tools[tool_name] = tool_func
        if self.config.verbose:
            print(f"[AgentCore] Exposed tool: {tool_name}")

    async def run_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a task using available capabilities.
        
        Args:
            task: Task description or input
            context: Additional context for execution
            
        Returns:
            Result dictionary with outputs from the chosen capability
        """
        if not self._initialized:
            await self.initialize()

        context = context or {}
        result = {"task": task, "status": "pending", "output": None}

        # Prefer Tree of Thought if enabled for complex reasoning
        if self.config.enable_tree_of_thought and self.tot_executor:
            solution = await self.tot_executor.solve_problem(task, context)
            result["output"] = solution
            result["status"] = "completed"
            result["method"] = "tree_of_thought"
        
        # Otherwise use RLM engine for reasoning
        elif self.config.enable_rlm_engine and self.rlm_engine:
            reasoning = await self.rlm_engine.reason(task, context)
            # Apply skill if needed
            if self.skill_registry and reasoning.get("requires_skill"):
                skill_result = await self.skill_registry.execute_skill(
                    reasoning["skill_name"],
                    reasoning.get("skill_params", {})
                )
                result["output"] = skill_result
            else:
                result["output"] = reasoning
            result["status"] = "completed"
            result["method"] = "rlm"
        
        # Fall back to multi-agent orchestration
        elif self.config.enable_multi_agent and self.multi_agent:
            orchestration = await self.multi_agent.orchestrate(task, context)
            result["output"] = orchestration
            result["status"] = "completed"
            result["method"] = "multi_agent"
        else:
            result["status"] = "failed"
            result["error"] = "No capabilities enabled or initialized"

        return result

    def get_tools(self) -> Dict[str, Any]:
        """Return exposed tools for external agent integration."""
        return self.exposed_tools.copy()

    async def shutdown(self) -> None:
        """Gracefully shut down all subsystems."""
        if self.multi_agent:
            await self.multi_agent.shutdown()
        if self.rlm_engine:
            await self.rlm_engine.shutdown()
        if self.tot_executor:
            await self.tot_executor.shutdown()
        self._initialized = False


# Convenience function for quick agent instantiation
async def create_ada_agent(
    enable_tot: bool = False,
    verbose: bool = False
) -> AgentCore:
    """Factory function to create and initialize an Ada agent."""
    config = AgentConfig(
        enable_tree_of_thought=enable_tot,
        verbose=verbose
    )
    agent = AgentCore(config)
    await agent.initialize()
    return agent
