from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .bundle import SkillBundle
from .graph import ActionGraph, UnifiedExecutionGraph


@dataclass(slots=True)
class SkillProfile:
    name: str
    description: str
    use_when: str
    summary: str
    declared_capabilities: list[str] = field(default_factory=list)
    declared_outputs: list[str] = field(default_factory=list)
    declared_data_scope: list[str] = field(default_factory=list)
    declared_execution_scope: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateAction:
    candidate_id: str
    node_id: str
    layer: str
    summary: str
    source_file: str | None
    risk_tags: list[str]
    reason: str
    confidence: float
    classification_label: str = "suspicious"
    upstream_action_chain: list[str] = field(default_factory=list)
    downstream_action_chain: list[str] = field(default_factory=list)
    predicate_context: list[str] = field(default_factory=list)
    context_node_ids: list[str] = field(default_factory=list)
    retained_due_to_ambiguity: bool = False
    ambiguity_kind: str | None = None
    classifier_input: dict[str, Any] = field(default_factory=dict)
    classifier_strategy: str = "unknown"
    privilege_type: str | None = None
    privilege_relevant: bool = True


@dataclass(slots=True)
class CandidateExtractionResult:
    bundle: SkillBundle
    profile: SkillProfile
    instruction_graph: ActionGraph
    code_graphs: list[ActionGraph]
    ueg: UnifiedExecutionGraph
    candidates: list[CandidateAction]
