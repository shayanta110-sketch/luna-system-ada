from pathlib import Path
import sys
import yaml
import importlib.util
import subprocess
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from types import ModuleType


@dataclass
class SkillMetadata:
    name: str
    description: str
    version: str
    author: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    entry_point: Optional[str] = None


@dataclass
class Skill:
    path: Path
    metadata: SkillMetadata
    tools_module: Optional[ModuleType] = None
    tools: Dict[str, Callable] = field(default_factory=dict)


class SkillSystem:
    def __init__(self):
        self._skills: Dict[str, Skill] = {}

    def load_skill_from_folder(self, folder_path: str) -> Optional[Skill]:
        folder = Path(folder_path).resolve()
        if not folder.is_dir():
            raise ValueError(f"Folder does not exist: {folder_path}")

        skill_md_path = folder / "SKILL.md"
        if not skill_md_path.exists():
            raise FileNotFoundError(f"SKILL.md not found in {folder_path}")

        metadata = self._parse_skill_md(skill_md_path)
        if not metadata:
            return None

        tools_module = None
        tools = {}

        tools_py_path = folder / "tools.py"
        if tools_py_path.exists():
            tools_module = self._load_tools_module(tools_py_path, metadata.name)
            if tools_module:
                tools = self._extract_tools_from_module(tools_module)

        skill = Skill(
            path=folder,
            metadata=metadata,
            tools_module=tools_module,
            tools=tools
        )

        self._skills[metadata.name] = skill
        return skill

    def load_skills_from_directory(self, directory_path: str, recursive: bool = False) -> List[Skill]:
        base_dir = Path(directory_path).resolve()
        if not base_dir.is_dir():
            raise ValueError(f"Directory does not exist: {directory_path}")

        loaded = []
        if recursive:
            for skill_md in base_dir.rglob("SKILL.md"):
                folder = skill_md.parent
                try:
                    skill = self.load_skill_from_folder(str(folder))
                    if skill:
                        loaded.append(skill)
                except Exception as e:
                    print(f"Failed to load skill from {folder}: {e}", file=sys.stderr)
        else:
            for folder in base_dir.iterdir():
                if folder.is_dir() and (folder / "SKILL.md").exists():
                    try:
                        skill = self.load_skill_from_folder(str(folder))
                        if skill:
                            loaded.append(skill)
                    except Exception as e:
                        print(f"Failed to load skill from {folder}: {e}", file=sys.stderr)

        return loaded

    def get_skill(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def get_all_tools_for_agent(self) -> Dict[str, Callable]:
        all_tools = {}
        for skill in self._skills.values():
            all_tools.update(skill.tools)
        return all_tools

    def execute_skill_direct(self, skill_name: str, input_data: Any, timeout: int = 30) -> Dict[str, Any]:
        skill = self.get_skill(skill_name)
        if not skill:
            raise ValueError(f"Skill '{skill_name}' not found")

        entry_point = skill.metadata.entry_point
        if not entry_point:
            raise ValueError(f"Skill '{skill_name}' has no entry_point defined in metadata")

        if skill.tools_module and hasattr(skill.tools_module, entry_point):
            entry_func = getattr(skill.tools_module, entry_point)
            if callable(entry_func):
                try:
                    result = entry_func(input_data)
                    return {"success": True, "result": result, "skill": skill_name, "mode": "direct"}
                except Exception as e:
                    return {"success": False, "error": str(e), "skill": skill_name, "mode": "direct"}

        raise RuntimeError(f"No direct executable entry point found for skill '{skill_name}'")

    def execute_skill_as_subagent(self, skill_name: str, input_data: Any, timeout: int = 30) -> Dict[str, Any]:
        skill = self.get_skill(skill_name)
        if not skill:
            raise ValueError(f"Skill '{skill_name}' not found")

        # Try subprocess execution first for subagent mode
        script_path = skill.path / "run.py"
        if script_path.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(script_path), json.dumps(input_data)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(skill.path)
                )
                if result.returncode == 0:
                    output = json.loads(result.stdout) if result.stdout else {}
                    return {"success": True, "result": output, "skill": skill_name, "mode": "subprocess"}
                else:
                    return {"success": False, "error": result.stderr, "skill": skill_name, "mode": "subprocess"}
            except subprocess.TimeoutExpired:
                return {"success": False, "error": "Timeout", "skill": skill_name, "mode": "subprocess"}
            except Exception as e:
                return {"success": False, "error": str(e), "skill": skill_name, "mode": "subprocess"}

        # Fallback to direct execution if no run.py
        entry_point = skill.metadata.entry_point
        if entry_point and skill.tools_module and hasattr(skill.tools_module, entry_point):
            entry_func = getattr(skill.tools_module, entry_point)
            if callable(entry_func):
                try:
                    result = entry_func(input_data)
                    return {"success": True, "result": result, "skill": skill_name, "mode": "direct_fallback"}
                except Exception as e:
                    return {"success": False, "error": str(e), "skill": skill_name, "mode": "direct_fallback"}

        raise RuntimeError(f"No executable entry point found for skill '{skill_name}'")

    def _parse_skill_md(self, skill_md_path: Path) -> Optional[SkillMetadata]:
        content = skill_md_path.read_text(encoding="utf-8")

        if not content.startswith("---"):
            raise ValueError(f"SKILL.md missing YAML frontmatter: {skill_md_path}")

        parts = content.split("---", 2)
        if len(parts) < 2:
            raise ValueError(f"Invalid frontmatter format in {skill_md_path}")

        frontmatter_text = parts[1].strip()

        try:
            frontmatter = yaml.safe_load(frontmatter_text)
        except yaml.YAMLError as e:
            # Fallback to simple key-value parsing
            frontmatter = {}
            for line in frontmatter_text.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    frontmatter[key.strip()] = val.strip()

        if not frontmatter or "name" not in frontmatter:
            raise ValueError(f"SKILL.md missing 'name' field in frontmatter: {skill_md_path}")

        return SkillMetadata(
            name=frontmatter.get("name"),
            description=frontmatter.get("description", ""),
            version=frontmatter.get("version", "0.1.0"),
            author=frontmatter.get("author"),
            dependencies=frontmatter.get("dependencies", []),
            tags=frontmatter.get("tags", []),
            entry_point=frontmatter.get("entry_point")
        )

    def _load_tools_module(self, tools_py_path: Path, skill_name: str) -> Optional[ModuleType]:
        try:
            spec = importlib.util.spec_from_file_location(f"skill_{skill_name}_tools", tools_py_path)
            if not spec or not spec.loader:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"Failed to load tools.py for skill '{skill_name}': {e}", file=sys.stderr)
            return None

    def _extract_tools_from_module(self, module: ModuleType) -> Dict[str, Callable]:
        tools = {}
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            attr = getattr(module, attr_name)
            if callable(attr):
                tools[attr_name] = attr
        return tools