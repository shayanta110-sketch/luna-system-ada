#!/usr/bin/env python3
"""
Nexus Router Tool with LangChain Tools

Provides three LangChain tools for Ada to check system resources,
recommend models, and verify model compatibility using nexus-router.
"""

import argparse
import json
import sys
import subprocess
from typing import Optional, Dict, Any, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


def call_nexus_router(task: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Call nexus-router to get the best Ollama model for the given task.

    Args:
        task: Description of the task to route.
        config_path: Optional path to nexus-router configuration file.

    Returns:
        Dictionary containing the routing decision.
    """
    cmd = ["nexus-router", "route"]
    if config_path:
        cmd.extend(["--config", config_path])
    cmd.extend(["--task", task])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout.strip()
        # Assume nexus-router returns JSON output
        return json.loads(output) if output else {"model": None, "error": "No output from router"}
    except subprocess.CalledProcessError as e:
        return {
            "error": f"nexus-router failed with code {e.returncode}",
            "stderr": e.stderr,
            "stdout": e.stdout
        }
    except FileNotFoundError:
        return {"error": "nexus-router command not found. Is it installed and in PATH?"}
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse router output as JSON: {e}", "raw_output": output}


# LangChain Tool Input Schemas
class SystemResourcesInput(BaseModel):
    """Input for checking system resources."""
    pass


class RecommendModelInput(BaseModel):
    """Input for recommending a model based on task."""
    task_description: str = Field(description="Description of the task to find a suitable model for")


class CanRunModelInput(BaseModel):
    """Input for checking if a model can run on current system."""
    model_name: str = Field(description="Name of the Ollama model to check")


class CheckSystemResourcesTool(BaseTool):
    """Tool to check available system resources (RAM, CPU, GPU)."""
    name: str = "check_system_resources"
    description: str = "Check current system resources including RAM, CPU cores, and GPU availability. Returns a JSON with resource info."
    args_schema: Type[BaseModel] = SystemResourcesInput

    def _run(self, **kwargs) -> str:
        """Run system resource check."""
        try:
            import psutil
            import torch
            
            resources = {
                "ram_gb": psutil.virtual_memory().total / (1024**3),
                "ram_available_gb": psutil.virtual_memory().available / (1024**3),
                "cpu_cores": psutil.cpu_count(logical=True),
                "cpu_usage_percent": psutil.cpu_percent(interval=0.1),
                "gpu_available": torch.cuda.is_available() if torch.cuda.is_available() else False,
            }
            if resources["gpu_available"]:
                resources["gpu_count"] = torch.cuda.device_count()
                resources["gpu_memory_gb"] = [
                    torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    for i in range(torch.cuda.device_count())
                ]
            return json.dumps(resources, indent=2)
        except ImportError:
            return json.dumps({"error": "psutil or torch not installed"})

    async def _arun(self, **kwargs) -> str:
        """Async version."""
        return self._run(**kwargs)


class RecommendModelTool(BaseTool):
    """Tool to recommend the best Ollama model for a given task."""
    name: str = "recommend_model"
    description: str = "Given a task description, returns the best Ollama model to use based on nexus-router recommendations."
    args_schema: Type[BaseModel] = RecommendModelInput

    def __init__(self, config_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.config_path = config_path

    def _run(self, task_description: str, **kwargs) -> str:
        """Run model recommendation."""
        result = call_nexus_router(task_description, self.config_path)
        if "error" in result:
            return json.dumps({"error": result["error"]})
        return json.dumps({
            "recommended_model": result.get("model"),
            "confidence": result.get("confidence"),
            "reason": result.get("reason"),
            "task": task_description
        }, indent=2)

    async def _arun(self, task_description: str, **kwargs) -> str:
        """Async version."""
        return self._run(task_description, **kwargs)


class CanRunModelTool(BaseTool):
    """Tool to check if a specific Ollama model can run on current system resources."""
    name: str = "can_run_model"
    description: str = "Check if a specified Ollama model can run on the current system based on RAM, GPU, and CPU requirements."
    args_schema: Type[BaseModel] = CanRunModelInput

    def _run(self, model_name: str, **kwargs) -> str:
        """Check if model can run."""
        # Mock model requirements - in production, query nexus-router or a model registry
        # For now, use heuristics
        model_requirements = {
            "llama2:7b": {"min_ram_gb": 8, "recommended_ram_gb": 16},
            "llama2:13b": {"min_ram_gb": 16, "recommended_ram_gb": 32},
            "llama2:70b": {"min_ram_gb": 64, "recommended_ram_gb": 128},
            "mistral:7b": {"min_ram_gb": 8, "recommended_ram_gb": 16},
            "codellama:7b": {"min_ram_gb": 8, "recommended_ram_gb": 16},
            "phi:2.7b": {"min_ram_gb": 4, "recommended_ram_gb": 8},
        }
        
        try:
            import psutil
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            
            reqs = model_requirements.get(model_name, {"min_ram_gb": 4, "recommended_ram_gb": 8})
            can_run = available_ram_gb >= reqs["min_ram_gb"]
            
            return json.dumps({
                "model": model_name,
                "can_run": can_run,
                "available_ram_gb": round(available_ram_gb, 2),
                "min_ram_required_gb": reqs["min_ram_gb"],
                "recommended_ram_gb": reqs["recommended_ram_gb"],
                "warning": "Performance may be degraded" if available_ram_gb < reqs["recommended_ram_gb"] else None
            }, indent=2)
        except ImportError:
            return json.dumps({"error": "psutil not installed", "can_run": False})

    async def _arun(self, model_name: str, **kwargs) -> str:
        """Async version."""
        return self._run(model_name, **kwargs)


def get_nexus_tools(config_path: Optional[str] = None):
    """
    Return the three LangChain tools as a list.
    
    Args:
        config_path: Optional path to nexus-router config file.
        
    Returns:
        List of BaseTool instances.
    """
    return [
        CheckSystemResourcesTool(),
        RecommendModelTool(config_path=config_path),
        CanRunModelTool()
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Return best Ollama model for a given task using nexus-router"
    )
    parser.add_argument(
        "task",
        type=str,
        help="The task description to route"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to nexus-router configuration file"
    )
    parser.add_argument(
        "--pretty", "-p",
        action="store_true",
        help="Pretty-print JSON output"
    )

    args = parser.parse_args()

    result = call_nexus_router(args.task, args.config)

    if args.pretty:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result))

    if "error" in result and result["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
