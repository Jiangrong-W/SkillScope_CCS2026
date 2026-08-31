from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SkillArtifact:
    artifact_id: str
    relative_path: str
    absolute_path: str
    kind: str
    role: str
    size_bytes: int
    text_preview: str = ""
    is_binary: bool = False


@dataclass(slots=True)
class SkillBundle:
    bundle_id: str
    root_path: str
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_files: list[SkillArtifact] = field(default_factory=list)
    instruction_files: list[SkillArtifact] = field(default_factory=list)
    script_files: list[SkillArtifact] = field(default_factory=list)
    resource_files: list[SkillArtifact] = field(default_factory=list)
    artifacts: list[SkillArtifact] = field(default_factory=list)
