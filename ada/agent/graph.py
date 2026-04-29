from langgraph.graph import StateGraph, END
from typing import Dict, Any, Literal

class AgentState(Dict[str, Any]):
    """State schema for the LangGraph agent"""
    pass

def should_use_gpu(state: AgentState) -> Literal["gpu_node", "cpu_node"]:
    """Conditional router based on resource requirements"""
    resource_req = state.get("resource_requirement", "cpu")
    if resource_req == "gpu" and state.get("gpu_available", False):
        return "gpu_node"
    return "cpu_node"

def should_limit_memory(state: AgentState) -> Literal["high_memory", "low_memory"]:
    """Conditional router based on memory availability"""
    memory_needed = state.get("memory_mb", 0)
    memory_available = state.get("available_memory_mb", 8192)
    if memory_needed > memory_available:
        return "low_memory"
    return "high_memory"

def gpu_node(state: AgentState) -> AgentState:
    """Node for GPU-accelerated processing"""
    state["execution_device"] = "gpu"
    state["processing"] = "gpu_optimized"
    return state

def cpu_node(state: AgentState) -> AgentState:
    """Node for CPU-based processing"""
    state["execution_device"] = "cpu"
    state["processing"] = "cpu_optimized"
    return state

def high_memory_node(state: AgentState) -> AgentState:
    """Node for high-memory operations"""
    state["memory_mode"] = "high"
    return state

def low_memory_node(state: AgentState) -> AgentState:
    """Node for memory-constrained operations"""
    state["memory_mode"] = "low"
    state["chunked_processing"] = True
    return state

def build_agent_graph() -> StateGraph:
    """Build LangGraph with conditional resource-based routing"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("gpu_node", gpu_node)
    workflow.add_node("cpu_node", cpu_node)
    workflow.add_node("high_memory_node", high_memory_node)
    workflow.add_node("low_memory_node", low_memory_node)
    
    # Set entry point
    workflow.set_entry_point("gpu_node")
    
    # Add conditional edges from gpu_node
    workflow.add_conditional_edges(
        "gpu_node",
        should_limit_memory,
        {
            "high_memory": "high_memory_node",
            "low_memory": "low_memory_node"
        }
    )
    
    # Add conditional edge from cpu_node
    workflow.add_conditional_edges(
        "cpu_node",
        should_limit_memory,
        {
            "high_memory": "high_memory_node",
            "low_memory": "low_memory_node"
        }
    )
    
    # Add edges from memory nodes to END
    workflow.add_edge("high_memory_node", END)
    workflow.add_edge("low_memory_node", END)
    
    return workflow.compile()

# Optional: Example usage
if __name__ == "__main__":
    graph = build_agent_graph()
    sample_state = {
        "resource_requirement": "gpu",
        "gpu_available": True,
        "memory_mb": 10000,
        "available_memory_mb": 8192
    }
    result = graph.invoke(sample_state)
    print(result)