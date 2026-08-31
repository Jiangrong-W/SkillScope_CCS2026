from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ResourceFixture:
    """A deterministic resource required to exercise one candidate-reaching task."""

    fixture_id: str
    fixture_type: str
    target: str
    content: str | None = None
    source: str | None = None
    required: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LegitimateActionChain:
    chain_id: str
    candidate_id: str
    node_ids: list[str] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    reaches_candidate: bool = False
    candidate_position: int | None = None
    predicate_context: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# Keep the legacy name import-compatible for existing callers.
CandidateReachableActionChain = LegitimateActionChain


@dataclass(slots=True)
class TaskSpec:
    task_id: str
    candidate_id: str
    prompt: str
    chain_node_ids: list[str] = field(default_factory=list)
    chain_summaries: list[str] = field(default_factory=list)
    task_summary: str = ""
    generation_strategy: str = "fallback"
    fixtures: list[ResourceFixture] = field(default_factory=list)
    expected_candidate_node_id: str | None = None
    trigger_required: bool = True
    generation_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TaskTriggerEvidence:
    candidate_id: str
    task_id: str
    triggered: bool
    candidate_node_id: str
    executed_node_ids: list[str] = field(default_factory=list)
    required_chain_node_ids: list[str] = field(default_factory=list)
    observed_chain_node_ids: list[str] = field(default_factory=list)
    trigger_strategy: str = "execution_trace"
    reason: str = ""
    execution_run_id: str | None = None
    uncertainty_flags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AblationPlan:
    candidate_id: str
    node_id: str
    layer: str
    strategy: str
    source_file: str | None = None
    source_start_line: int | None = None
    source_end_line: int | None = None
    source_start_column: int | None = None
    source_end_column: int | None = None
    operation_type: str | None = None
    raw_text: str | None = None
    bypass_successor_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExecutionEvent:
    event_type: str
    summary: str
    node_id: str | None = None
    layer: str | None = None
    object_ref: str | None = None
    arguments_summary: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionRecord:
    run_id: str
    mode: str
    prompt: str
    trace: list[ExecutionEvent] = field(default_factory=list)
    raw_trace: list[dict[str, Any]] = field(default_factory=list)
    final_output: str = ""
    stdout: str = ""
    stderr: str = ""
    bundle_root: str = ""
    status: str = "not_run"
    notes: list[str] = field(default_factory=list)
    executed_node_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReplayPairRecord:
    candidate_id: str
    task_id: str
    ablation: AblationPlan
    original: ExecutionRecord
    replay: ExecutionRecord


@dataclass(slots=True)
class NecessityDecision:
    candidate_id: str
    task_id: str
    label: str
    reason: str
    trace_equivalent: bool = False
    output_equivalent: bool = False
    core_preserved: bool = False
    goal_satisfied: bool = False
    executed_in_original: bool = False
    confidence: float = 0.0
    judge_strategy: str = "fallback"
    necessity_basis: list[str] = field(default_factory=list)
    policy_violation_type: str | None = None
    task_boundary_explanation: str = ""
    uncertainty_flags: list[str] = field(default_factory=list)
    judge_input: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AuthorizationDecision:
    """Authorization of the complete action tuple for one task context."""

    candidate_id: str
    task_id: str
    label: str
    reason: str
    operation_authorized: bool | None = None
    source_authorized: bool | None = None
    destination_authorized: bool | None = None
    side_effect_authorized: bool | None = None
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    uncertainty_flags: list[str] = field(default_factory=list)
    judge_strategy: str = "fallback"
    judge_input: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_components(
        cls,
        *,
        candidate_id: str,
        task_id: str,
        operation_authorized: bool | None,
        source_authorized: bool | None,
        destination_authorized: bool | None,
        side_effect_authorized: bool | None,
        reason: str,
        confidence: float = 0.0,
        evidence: list[str] | None = None,
        uncertainty_flags: list[str] | None = None,
        judge_strategy: str = "fallback",
        judge_input: dict[str, Any] | None = None,
    ) -> "AuthorizationDecision":
        components = (
            operation_authorized,
            source_authorized,
            destination_authorized,
            side_effect_authorized,
        )
        if any(component is False for component in components):
            label = "unauthorized"
        elif all(component is True for component in components):
            label = "authorized"
        else:
            label = "inconclusive"
        return cls(
            candidate_id=candidate_id,
            task_id=task_id,
            label=label,
            reason=reason,
            operation_authorized=operation_authorized,
            source_authorized=source_authorized,
            destination_authorized=destination_authorized,
            side_effect_authorized=side_effect_authorized,
            confidence=confidence,
            evidence=list(evidence or []),
            uncertainty_flags=list(uncertainty_flags or []),
            judge_strategy=judge_strategy,
            judge_input=dict(judge_input or {}),
        )


@dataclass(slots=True)
class FinalVerdict:
    """Final verdict for a privilege-relevant candidate action."""

    candidate_id: str
    task_id: str
    label: str
    authorization_label: str
    necessity_label: str
    reason: str
    overprivilege_reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0
    uncertainty_flags: list[str] = field(default_factory=list)
    privilege_type: str | None = None
    privilege_relevant: bool = True

    @property
    def is_overprivileged(self) -> bool:
        return self.label == "overprivileged"

    @classmethod
    def combine(
        cls,
        authorization: AuthorizationDecision,
        necessity: NecessityDecision,
        *,
        privilege_relevant: bool = True,
        privilege_type: str | None = None,
    ) -> "FinalVerdict":
        if authorization.candidate_id != necessity.candidate_id or authorization.task_id != necessity.task_id:
            raise ValueError("Authorization and necessity decisions must describe the same candidate-task pair.")

        reasons: list[str] = []
        if authorization.label == "unauthorized":
            reasons.append("unauthorized")
        if necessity.label == "unnecessary":
            reasons.append("unnecessary")

        if not privilege_relevant:
            label = "not_overprivileged"
            reasons = []
        elif reasons:
            label = "overprivileged"
        elif authorization.label == "authorized" and necessity.label == "necessary":
            label = "not_overprivileged"
        else:
            label = "inconclusive"

        uncertainty_flags = sorted(
            set(authorization.uncertainty_flags + necessity.uncertainty_flags)
        )
        if not privilege_relevant:
            reason = (
                "The action is outside the configured privilege-relevant "
                "taxonomy and therefore does not satisfy type(a) in T."
            )
        elif label == "overprivileged":
            reason = "The action is over-privileged because it is " + " and ".join(reasons) + "."
        elif label == "not_overprivileged":
            reason = "The action is both authorized by the task boundary and necessary for task completion."
        else:
            reason = "The final verdict is inconclusive because authorization or necessity is unresolved."
        return cls(
            candidate_id=authorization.candidate_id,
            task_id=authorization.task_id,
            label=label,
            authorization_label=authorization.label,
            necessity_label=necessity.label,
            reason=reason,
            overprivilege_reasons=reasons,
            confidence=min(authorization.confidence, necessity.confidence),
            uncertainty_flags=uncertainty_flags,
            privilege_type=privilege_type,
            privilege_relevant=privilege_relevant,
        )


@dataclass(slots=True)
class ActionTaskDescriptor:
    """Fixed-schema task context consumed by Module 3.

    ``operation`` through ``side_effect`` retain the candidate/action facts for
    backwards-compatible diagnostics.  The ``requested_*`` fields are the
    actual clustering and guard inputs: they describe authority conveyed by the
    concrete user task, rather than copying behavior observed in the Skill.
    ``material_action_instances`` preserves every distinct realized action that
    the candidate produced so a compound instruction cannot be reduced to its
    final tool event during repair synthesis.
    """

    descriptor_id: str
    candidate_id: str
    task_id: str
    intent: str
    operation: str
    object: str
    scope: str
    destination: str
    side_effect: str
    final_verdict: str
    requested_operation: str = "unspecified"
    requested_object: str = "unspecified"
    requested_scope: str = "unspecified"
    requested_destination: str = "none"
    explicit_side_effect_requested: bool | None = None
    material_action_instances: list[dict[str, Any]] = field(default_factory=list)
    normalized_slots: dict[str, str] = field(default_factory=dict)
    cluster_key: str = ""
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ValidationResult:
    bundle_id: str
    legitimate_chains: list[LegitimateActionChain] = field(default_factory=list)
    tasks: list[TaskSpec] = field(default_factory=list)
    trigger_evidence: list[TaskTriggerEvidence] = field(default_factory=list)
    replay_pairs: list[ReplayPairRecord] = field(default_factory=list)
    decisions: list[NecessityDecision] = field(default_factory=list)
    authorization_decisions: list[AuthorizationDecision] = field(default_factory=list)
    final_verdicts: list[FinalVerdict] = field(default_factory=list)
    descriptors: list[ActionTaskDescriptor] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
