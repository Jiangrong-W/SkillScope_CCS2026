from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .analysis import SkillProfile
from .bundle import SkillBundle
from .graph import ActionGraph, UnifiedExecutionGraph


@dataclass(slots=True)
class InstalledSkill:
    install_id: str
    bundle: SkillBundle
    profile: SkillProfile
    instruction_graph: ActionGraph
    code_graphs: list[ActionGraph] = field(default_factory=list)
    ueg: UnifiedExecutionGraph | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
