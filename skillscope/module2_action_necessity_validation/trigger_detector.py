from __future__ import annotations

from skillscope.common.models import (
    CandidateAction,
    ExecutionRecord,
    TaskSpec,
    TaskTriggerEvidence,
    UnifiedExecutionGraph,
)
from skillscope.common.privilege import MATERIAL_EVENT_TYPES


class TaskTriggerDetector:
    """Verify that the original run materially executed the candidate action."""

    MATERIAL_EVENT_TYPES = MATERIAL_EVENT_TYPES

    def detect(
        self,
        *,
        candidate: CandidateAction,
        task: TaskSpec,
        record: ExecutionRecord,
        ueg: UnifiedExecutionGraph | None = None,
    ) -> TaskTriggerEvidence:
        expected_node_id = task.expected_candidate_node_id or candidate.node_id
        uncertainty_flags: list[str] = []
        if expected_node_id != candidate.node_id:
            uncertainty_flags.append("task_expected_candidate_node_mismatch")
        if record.status != "completed":
            uncertainty_flags.append("original_execution_not_completed")

        candidate_seen = candidate.node_id in record.executed_node_ids
        material_events = self._related_material_events(
            record=record,
            candidate_node_id=candidate.node_id,
        )
        material_action_seen = bool(material_events)
        required_chain_node_ids = self._required_observable_path(
            task=task,
            ueg=ueg,
        )
        required_chain_node_id_set = set(required_chain_node_ids)
        observed_chain_node_ids = [
            node_id
            for node_id in record.executed_node_ids
            if node_id in required_chain_node_id_set
        ]
        chain_verified = (
            self._is_subsequence(
                required_chain_node_ids,
                record.executed_node_ids,
            )
            if required_chain_node_ids
            else ueg is None
        )
        if ueg is None:
            uncertainty_flags.append(
                "candidate_reaching_action_chain_not_supplied"
            )
        elif not chain_verified:
            uncertainty_flags.append(
                "candidate_materialized_outside_intended_action_chain"
            )
        triggered = (
            expected_node_id == candidate.node_id
            and material_action_seen
            and chain_verified
        )
        if candidate_seen and not material_action_seen:
            uncertainty_flags.append(
                "candidate_selected_without_material_action_evidence"
            )
        if triggered:
            event_types = sorted(
                {
                    str(event.get("event_type") or "")
                    for event in material_events
                }
            )
            trace_subject = (
                "The completed original execution trace"
                if record.status == "completed"
                else (
                    "Although the original execution ended with status "
                    f"{record.status!r} after the candidate action, its trace"
                )
            )
            reason = (
                f"{trace_subject} contains material action event(s) "
                f"{event_types} tied to candidate node {candidate.node_id}, "
                "and the observed action order covers every runtime-observable "
                "node in the candidate-reaching task chain. Overall task "
                "completion is evaluated separately from whether the action "
                "was realized."
            )
        elif material_action_seen and not chain_verified:
            reason = (
                f"Candidate node {candidate.node_id} produced material action "
                "evidence, but the original execution did not follow the "
                "runtime-observable candidate-reaching action chain synthesized "
                "for this task; "
                "ablation replay is therefore not justified."
            )
        elif candidate_seen:
            reason = (
                f"Candidate node {candidate.node_id} was selected or traversed, but "
                "the trace contains no candidate-linked material side effect; "
                "ablation replay is therefore not justified."
            )
        else:
            reason = (
                f"The original execution trace contains no material action event "
                f"tied to candidate node {candidate.node_id}; ablation replay is "
                "therefore not justified."
            )

        return TaskTriggerEvidence(
            candidate_id=candidate.candidate_id,
            task_id=task.task_id,
            triggered=triggered,
            candidate_node_id=candidate.node_id,
            executed_node_ids=list(record.executed_node_ids),
            required_chain_node_ids=required_chain_node_ids,
            observed_chain_node_ids=observed_chain_node_ids,
            trigger_strategy=(
                "original_candidate_linked_material_action_trace"
            ),
            reason=reason,
            execution_run_id=record.run_id,
            uncertainty_flags=uncertainty_flags,
        )

    def _related_material_events(
        self,
        *,
        record: ExecutionRecord,
        candidate_node_id: str,
    ) -> list[dict[str, object]]:
        payloads: list[dict[str, object]] = []
        if record.raw_trace:
            payloads.extend(
                dict(payload)
                for payload in record.raw_trace
                if isinstance(payload, dict)
            )
        else:
            payloads.extend(
                {
                    "event_type": event.event_type,
                    "node_id": event.node_id,
                    "attributes": event.attributes,
                }
                for event in record.trace
            )

        related: list[dict[str, object]] = []
        for payload in payloads:
            attributes = payload.get("attributes")
            material_operation = (
                attributes.get("material_operation")
                if isinstance(attributes, dict)
                else None
            )
            if (
                str(payload.get("event_type") or "")
                not in self.MATERIAL_EVENT_TYPES
                and str(material_operation or "")
                not in self.MATERIAL_EVENT_TYPES
            ):
                continue
            instruction_node_id = (
                attributes.get("instruction_node_id")
                if isinstance(attributes, dict)
                else None
            )
            if (
                payload.get("node_id") == candidate_node_id
                or instruction_node_id == candidate_node_id
            ):
                related.append(payload)
        return related

    def _required_observable_path(
        self,
        *,
        task: TaskSpec,
        ueg: UnifiedExecutionGraph | None,
    ) -> list[str]:
        if ueg is None:
            return []
        return [
            node_id
            for node_id in task.chain_node_ids
            if (
                (node := ueg.node_by_id(node_id)) is not None
                and (
                    (
                        node.layer == "instruction"
                        and node.node_type not in {"ENTRY", "EXIT"}
                    )
                    or (
                        node.layer == "code"
                        and node.node_type == "CODE_ACTION"
                        and node.source_range is not None
                    )
                    or node_id == task.expected_candidate_node_id
                )
            )
        ]

    def _is_subsequence(
        self,
        expected: list[str],
        observed: list[str],
    ) -> bool:
        if not expected:
            return False
        iterator = iter(observed)
        return all(any(value == expected_id for value in iterator) for expected_id in expected)
