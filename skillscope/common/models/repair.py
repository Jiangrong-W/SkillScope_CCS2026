from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RepairItem:
    repair_id: str
    overreach_id: str
    node_id: str
    layer: str
    repair_type: str
    target_files: list[str] = field(default_factory=list)
    rationale: str = ""
    overreach_summary: str = ""
    guard_condition: str | None = None
    allowed_task_summaries: list[str] = field(default_factory=list)
    blocked_task_summaries: list[str] = field(default_factory=list)
    source_file: str | None = None
    source_start_line: int | None = None
    source_end_line: int | None = None
    source_start_column: int | None = None
    source_end_column: int | None = None
    raw_text: str | None = None
    descriptor_ids: list[str] = field(default_factory=list)
    allowed_cluster_keys: list[str] = field(default_factory=list)
    blocked_cluster_keys: list[str] = field(default_factory=list)
    generated_files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def candidate_id(self) -> str:
        return self.overreach_id

    @property
    def candidate_summary(self) -> str:
        return self.overreach_summary


@dataclass(slots=True)
class RepairPlan:
    skill_id: str
    items: list[RepairItem] = field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RepairValidationReport:
    patched_skill_id: str
    overreach_count: int = 0
    remaining_overreach_count: int = 0
    repaired_overreach_ids: list[str] = field(default_factory=list)
    remaining_overreach_ids: list[str] = field(default_factory=list)
    task_count: int = 0
    decision_count: int = 0
    successful_task_count: int = 0
    completed_replay_pairs: int = 0
    unnecessary_count: int = 0
    overprivileged_count: int = 0
    output_equivalent_count: int = 0
    core_preserved_count: int = 0
    goal_satisfied_count: int = 0
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def candidate_count(self) -> int:
        return self.overreach_count

    @property
    def remaining_candidate_count(self) -> int:
        return self.remaining_overreach_count

    @property
    def repair_succeeded(self) -> bool:
        """Require completed safety and utility evidence for an actual repair."""
        verdicts = self.metadata.get("after_repair_final_verdicts", [])
        return bool(
            self.overreach_count > 0
            and self.task_count > 0
            and self.decision_count >= self.task_count
            and self.remaining_overreach_count == 0
            and self.completed_replay_pairs == self.decision_count
            and self.successful_task_count == self.decision_count
            and self.core_preserved_count == self.decision_count
            and self.goal_satisfied_count == self.decision_count
            and self.overprivileged_count == 0
            and self.metadata.get("validation_mode") == "dynamic"
            and self.metadata.get("after_repair_inconclusive_count") == 0
            and isinstance(verdicts, list)
            and len(verdicts) == self.decision_count
            and all(
                isinstance(verdict, dict)
                and verdict.get("label") == "not_overprivileged"
                and verdict.get("core_preserved") is True
                and verdict.get("goal_satisfied") is True
                and verdict.get("projection_integrity_complete") is True
                for verdict in verdicts
            )
        )


@dataclass(slots=True)
class RepairOutcome:
    plan: RepairPlan | None = None
    patched_bundle_path: str | None = None
    validation: RepairValidationReport | None = None
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
