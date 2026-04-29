"""
Skill System for standardized skill definition and loading.

Supports:
- Isolated sub-agent execution or direct tool exposure
- Skill discovery from filesystem, Python packages, or MCP servers
- Standardized skill interface
"""

import importlib
import inspect
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class SkillSpec:
    """Standard skill specification."""
    name: str
    description: str
    version: str = "1.0.0"
    author: str = ""
    tags: List[str] = field(default_factory=list)
    parameters: Optional[Dict[str, Any]] = None
    returns: Optional[Dict[str, Any]] = None
    execution_mode: str = "direct"  # "direct" or "subagent"
    entry_point: Optional[Union[str, Callable]] = None
    source: str = ""  # "filesystem", "package", "mcp"
    source_path: Optional[str] = None


class Skill:
    """Represents a loaded skill."""
    
    def __init__(self, spec: SkillSpec, executor: Optional[Callable] = None):
        self.spec = spec
        self._executor = executor
    
    async def execute(self, **kwargs) -> Any:
        """Execute the skill with given parameters."""
        if self.spec.execution_mode == "direct" and self._executor:
            if inspect.iscoroutinefunction(self._executor):
                return await self._executor(**kwargs)
            else:
                return self._executor(**kwargs)
        elif self.spec.execution_mode == "subagent":
            # Sub-agent execution mode (to be implemented)
            raise NotImplementedError("Sub-agent execution mode not yet implemented")
        else:
            raise RuntimeError(f"Unknown execution mode: {self.spec.execution_mode}")
    
    def to_tool_schema(self) -> Dict[str, Any]:
        """Convert skill to tool schema for agent exposure."""
        return {
            "type": "function",
            "function": {
                "name": self.spec.name,
                "description": self.spec.description,
                "parameters": self.spec.parameters or {"type": "object", "properties": {}}
            }
        }


class SkillDiscovery:
    """Discover skills from various sources."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.discovered_skills: Dict[str, SkillSpec] = {}
    
    def discover_from_filesystem(self, path: Union[str, Path]) -> List[SkillSpec]:
        """Discover skills from filesystem directory."""
        path = Path(path)
        skills = []
        
        # Look for skill.json or skill.yaml files
        for skill_file in path.glob("*skill.json"):
            try:
                with open(skill_file, 'r') as f:
                    spec_data = json.load(f)
                spec = self._dict_to_spec(spec_data)
                spec.source = "filesystem"
                spec.source_path = str(skill_file.parent)
                skills.append(spec)
            except Exception as e:
                logger.error(f"Failed to load skill from {skill_file}: {e}")
        
        # Also look for Python files with Skill class
        for py_file in path.glob("*_skill.py"):
            try:
                module_name = py_file.stem
                spec = self._load_skill_from_module(module_name, str(py_file.parent))
                if spec:
                    spec.source = "filesystem"
                    spec.source_path = str(py_file)
                    skills.append(spec)
            except Exception as e:
                logger.error(f"Failed to load skill from {py_file}: {e}")
        
        self.discovered_skills.update({s.name: s for s in skills})
        return skills
    
    def discover_from_package(self, package_name: str) -> List[SkillSpec]:
        """Discover skills from Python package."""
        skills = []
        try:
            module = importlib.import_module(package_name)
            # Look for SKILLS registry or skill classes
            if hasattr(module, 'SKILLS'):
                for skill_spec in module.SKILLS:
                    if isinstance(skill_spec, SkillSpec):
                        skill_spec.source = "package"
                        skill_spec.source_path = package_name
                        skills.append(skill_spec)
            
            # Also scan for Skill subclasses
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, Skill) and obj != Skill:
                    # Create spec from class
                    spec = self._class_to_spec(obj)
                    spec.source = "package"
                    spec.source_path = package_name
                    skills.append(spec)
        except Exception as e:
            logger.error(f"Failed to discover skills from package {package_name}: {e}")
        
        self.discovered_skills.update({s.name: s for s in skills})
        return skills
    
    def discover_from_mcp(self, server_endpoint: str, api_key: Optional[str] = None) -> List[SkillSpec]:
        """Discover skills from MCP server."""
        # Placeholder for MCP integration
        logger.warning("MCP discovery not yet fully implemented")
        return []
    
    def _dict_to_spec(self, data: Dict[str, Any]) -> SkillSpec:
        """Convert dictionary to SkillSpec."""
        return SkillSpec(
            name=data.get('name', ''),
            description=data.get('description', ''),
            version=data.get('version', '1.0.0'),
            author=data.get('author', ''),
            tags=data.get('tags', []),
            parameters=data.get('parameters'),
            returns=data.get('returns'),
            execution_mode=data.get('execution_mode', 'direct'),
            entry_point=data.get('entry_point'),
        )
    
    def _load_skill_from_module(self, module_name: str, module_path: str) -> Optional[SkillSpec]:
        """Load skill spec from Python module."""
        try:
            import sys
            if module_path not in sys.path:
                sys.path.insert(0, module_path)
            module = importlib.import_module(module_name)
            if hasattr(module, 'get_skill_spec'):
                return module.get_skill_spec()
            elif hasattr(module, 'SKILL_SPEC'):
                return module.SKILL_SPEC
        except Exception as e:
            logger.error(f"Failed to load module {module_name}: {e}")
        return None
    
    def _class_to_spec(self, skill_class: type) -> SkillSpec:
        """Convert Skill subclass to SkillSpec."""
        name = getattr(skill_class, '__skill_name__', skill_class.__name__)
        description = getattr(skill_class, '__skill_description__', skill_class.__doc__ or '')
        return SkillSpec(
            name=name,
            description=description,
            execution_mode="direct",
        )


class SkillLoader:
    """Load and manage skills."""
    
    def __init__(self, discovery: Optional[SkillDiscovery] = None):
        self.discovery = discovery or SkillDiscovery()
        self.loaded_skills: Dict[str, Skill] = {}
    
    def load_skill(self, spec: SkillSpec, executor: Optional[Callable] = None) -> Skill:
        """Load a skill from its specification."""
        if spec.execution_mode == "direct":
            if executor is None and spec.entry_point:
                executor = self._load_entry_point(spec.entry_point, spec.source_path)
            elif executor is None:
                raise ValueError(f"No executor provided for direct skill {spec.name}")
        
        skill = Skill(spec, executor)
        self.loaded_skills[spec.name] = skill
        return skill
    
    def load_all_discovered(self) -> List[Skill]:
        """Load all discovered skills."""
        skills = []
        for spec in self.discovery.discovered_skills.values():
            try:
                skill = self.load_skill(spec)
                skills.append(skill)
            except Exception as e:
                logger.error(f"Failed to load skill {spec.name}: {e}")
        return skills
    
    def _load_entry_point(self, entry_point: Union[str, Callable], source_path: Optional[str] = None) -> Callable:
        """Load entry point from string path or return callable."""
        if callable(entry_point):
            return entry_point
        
        # Parse module:function format
        if isinstance(entry_point, str) and ':' in entry_point:
            module_path, func_name = entry_point.split(':', 1)
            try:
                import sys
                if source_path and source_path not in sys.path:
                    sys.path.insert(0, source_path)
                module = importlib.import_module(module_path)
                return getattr(module, func_name)
            except Exception as e:
                raise ImportError(f"Failed to load entry point {entry_point}: {e}")
        
        raise ValueError(f"Invalid entry point: {entry_point}")
    
    def get_skill(self, name: str) -> Optional[Skill]:
        """Get loaded skill by name."""
        return self.loaded_skills.get(name)
    
    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get tool schemas for all loaded direct-mode skills."""
        return [
            skill.to_tool_schema() 
            for skill in self.loaded_skills.values() 
            if skill.spec.execution_mode == "direct"
        ]


class SkillSystem:
    """Main skill system orchestrator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.discovery = SkillDiscovery(config.get('discovery', {}))
        self.loader = SkillLoader(self.discovery)
        
        # Auto-discover from configured sources
        self._auto_discover()
    
    def _auto_discover(self):
        """Auto-discover skills from configured sources."""
        if 'filesystem_paths' in self.config:
            for path in self.config['filesystem_paths']:
                self.discovery.discover_from_filesystem(path)
        
        if 'packages' in self.config:
            for package in self.config['packages']:
                self.discovery.discover_from_package(package)
        
        if 'mcp_servers' in self.config:
            for server in self.config['mcp_servers']:
                self.discovery.discover_from_mcp(
                    server.get('endpoint'),
                    server.get('api_key')
                )
    
    def register_skill(self, spec: SkillSpec, executor: Optional[Callable] = None) -> Skill:
        """Register and load a new skill."""
        self.discovery.discovered_skills[spec.name] = spec
        return self.loader.load_skill(spec, executor)
    
    def get_skill(self, name: str) -> Optional[Skill]:
        """Get skill by name."""
        return self.loader.get_skill(name)
    
    async def execute_skill(self, name: str, **kwargs) -> Any:
        """Execute skill by name."""
        skill = self.get_skill(name)
        if not skill:
            # Try to load from discovered specs
            spec = self.discovery.discovered_skills.get(name)
            if spec:
                skill = self.loader.load_skill(spec)
            else:
                raise ValueError(f"Skill not found: {name}")
        return await skill.execute(**kwargs)
    
    def list_skills(self) -> List[Dict[str, Any]]:
        """List all available skills."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "version": spec.version,
                "execution_mode": spec.execution_mode,
                "source": spec.source
            }
            for spec in self.discovery.discovered_skills.values()
        ]