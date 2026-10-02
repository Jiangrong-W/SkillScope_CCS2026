from __future__ import annotations

from skillscope.common.models import (
    CandidateAction,
    ExecutionRecord,
    TaskSpec,
    TaskTriggerEvidence,
    UnifiedExecutionGraph,
)
from skillscope.common.privilege import MATERIAL_EVENT_TYPES, material_event_matches_operation


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
            candidate_operation=(
                node.operation_type
                if ueg is not None and (node := ueg.node_by_id(candidate.node_id)) is not None
                else None
            ),
        )
        material_action_seen = bool(material_events)
        required_chain_node_ids = self._required_observable_path(
            task=task,
            ueg=ueg,
            candidate_node_id=candidate.node_id,
        )
        chain_verified, observed_chain_node_ids, chain_uncertainty = self._verify_observable_prefix(
            task=task,
            record=record,
            ueg=ueg,
            required_node_ids=required_chain_node_ids,
            candidate_node_id=candidate.node_id,
        )
        uncertainty_flags.extend(chain_uncertainty)
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
                "and observed control steps and physical action order cover "
                "the candidate-reaching task prefix. Later task completion "
                "steps do not establish whether this action was triggered. Overall task "
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
        candidate_operation: str | None = None,
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
            if str(payload.get("event_type") or "") in {"call", "return", "line"}:
                continue
            if isinstance(attributes, dict) and attributes.get("ablated") is True:
                continue
            if candidate_operation is not None and not material_event_matches_operation(
                candidate_operation,
                str(payload.get("event_type") or ""),
                attributes if isinstance(attributes, dict) else None,
            ):
                continue
            instruction_node_id = (
                attributes.get("instruction_node_id")
                if isinstance(attributes, dict)
                else None
            )
            if (
                payload.get("node_id") == candidate_node_id
                or (
                    instruction_node_id == candidate_node_id
                    and candidate_operation is not None
                )
            ):
                related.append(payload)
        return related

    def _required_observable_path(
        self,
        *,
        task: TaskSpec,
        ueg: UnifiedExecutionGraph | None,
        candidate_node_id: str | None = None,
    ) -> list[str]:
        if ueg is None:
            return []
        candidate_id = task.expected_candidate_node_id or candidate_node_id
        if candidate_id not in task.chain_node_ids:
            return []
        prefix = task.chain_node_ids[:task.chain_node_ids.index(candidate_id) + 1]
        return [
            node_id
            for node_id in prefix
            if (
                (node := ueg.node_by_id(node_id)) is not None
                and (
                    (
                        node.layer == "instruction"
                        and node.node_type not in {"ENTRY", "EXIT"}
                    )
                    or (
                        node.layer == "code"
                        and (
                            (node.node_type == "CODE_ACTION" and material_event_matches_operation(node.operation_type, node.operation_type))
                            or self._is_branch_checkpoint(ueg, node_id)
                        )
                    )
                    or node_id == task.expected_candidate_node_id
                )
            )
        ]

    def _verify_observable_prefix(
        self,
        *,
        task: TaskSpec,
        record: ExecutionRecord,
        ueg: UnifiedExecutionGraph | None,
        required_node_ids: list[str],
        candidate_node_id: str,
    ) -> tuple[bool, list[str], list[str]]:
        if ueg is None:
            return True, [], []
        if not required_node_ids:
            return False, [], []
        required = {node_id: ueg.node_by_id(node_id) for node_id in required_node_ids}
        instruction_ids = [node_id for node_id, node in required.items() if node is not None and node.layer == "instruction"]
        code_ids = [node_id for node_id, node in required.items() if node is not None and node.layer == "code"]
        checkpoint_ids = {
            node_id for node_id in code_ids
            if (node := required[node_id]) is not None
            and not material_event_matches_operation(node.operation_type, node.operation_type)
        }
        material_code_ids = [node_id for node_id in code_ids if node_id not in checkpoint_ids]
        payloads = record.raw_trace or [
            {"event_type": event.event_type, "node_id": event.node_id, "attributes": event.attributes}
            for event in record.trace
        ]
        observed: list[str] = []
        controls: list[str] = []
        material_code: list[str] = []
        ordered_code: list[str] = []
        unordered_code: set[str] = set()
        observed_checkpoints: set[str] = set()
        uncertainty: list[str] = []
        has_instruction_trace = any(payload.get("event_type") == "instruction_step_start" for payload in payloads)

        for payload in payloads:
            attributes = payload.get("attributes")
            if not isinstance(attributes, dict):
                attributes = {}
            event_type = str(payload.get("event_type") or "")
            node_id = payload.get("node_id")
            node = required.get(node_id)
            if event_type == "line":
                for checkpoint_id in checkpoint_ids:
                    checkpoint = required[checkpoint_id]
                    if (
                        checkpoint is not None
                        and checkpoint.source_range is not None
                        and attributes.get("source_file") == checkpoint.source_file
                        and attributes.get("line_number") == checkpoint.source_range.start_line
                    ):
                        callers = self._prefix_callers(ueg, task, checkpoint_id, candidate_node_id)
                        caller = attributes.get("instruction_node_id")
                        if callers and caller not in callers:
                            continue
                        observed_checkpoints.add(checkpoint_id)
                        ordered_code.append(checkpoint_id)
                        observed.append(checkpoint_id)
                continue
            if node is None:
                continue
            if node.layer == "instruction" and event_type == "instruction_step_start":
                controls.append(node.node_id)
                observed.append(node.node_id)
                continue
            if not material_event_matches_operation(node.operation_type, event_type, attributes):
                continue
            if node.layer == "code":
                callers = self._prefix_callers(ueg, task, node.node_id, candidate_node_id)
                caller = attributes.get("instruction_node_id")
                if callers and isinstance(caller, str) and caller not in callers:
                    continue
                if callers and not isinstance(caller, str):
                    if has_instruction_trace:
                        continue
                    uncertainty.append("legacy_material_action_call_origin_unavailable")
                material_code.append(node.node_id)
                if attributes.get("temporal_order_observed") is False:
                    unordered_code.add(node.node_id)
                else:
                    ordered_code.append(node.node_id)
                    observed.append(node.node_id)
            else:
                controls.append(node.node_id)
                observed.append(node.node_id)

        # Legacy/imported records may have instruction IDs without control
        # events. They can establish the selected control prefix, while code
        # actions always require their own physical evidence and ordering.
        if not has_instruction_trace:
            controls = [node_id for node_id in record.executed_node_ids if node_id in instruction_ids]
            if instruction_ids:
                uncertainty.append("legacy_control_prefix_order_from_executed_node_ids")
            observed = [*controls, *ordered_code]

        control_verified = self._is_subsequence(instruction_ids, controls) if instruction_ids else True
        material_verified = set(material_code_ids).issubset(material_code) and checkpoint_ids.issubset(observed_checkpoints)
        expected_ordered_code = [node_id for node_id in code_ids if node_id not in unordered_code]
        material_order_verified = self._is_subsequence(expected_ordered_code, ordered_code) if expected_ordered_code else True
        ordered_prefix = [node_id for node_id in required_node_ids if node_id not in unordered_code]
        cross_layer_order_verified = (
            self._is_subsequence(ordered_prefix, observed)
            if has_instruction_trace and ordered_prefix
            else True
        )
        if unordered_code:
            uncertainty.append("material_action_temporal_order_unavailable")
        return (
            control_verified and material_verified and material_order_verified and cross_layer_order_verified,
            observed + [node_id for node_id in required_node_ids if node_id in unordered_code],
            uncertainty,
        )

    def _is_branch_checkpoint(self, ueg: UnifiedExecutionGraph, node_id: str) -> bool:
        """A distinct branch-entry source line observes control, not a call order."""
        node = ueg.node_by_id(node_id)
        if node is None or node.layer != "code" or node.source_range is None:
            return False
        for edge in ueg.edges:
            if edge.target != node_id or edge.edge_type not in {"CONDITIONAL_TRUE", "CONDITIONAL_FALSE"}:
                continue
            predicate = ueg.node_by_id(edge.source)
            if (
                predicate is not None
                and predicate.node_type == "CODE_PREDICATE"
                and predicate.source_file == node.source_file
                and predicate.source_range is not None
                and node.source_range.start_line > predicate.source_range.end_line
            ):
                return True
        return False

    def _prefix_callers(self, ueg: UnifiedExecutionGraph, task: TaskSpec, code_node_id: str, candidate_node_id: str) -> set[str]:
        """Find the represented instruction invocation that reaches this code action."""
        candidate_id = task.expected_candidate_node_id or candidate_node_id
        if candidate_id not in task.chain_node_ids:
            return set()
        prefix = task.chain_node_ids[:task.chain_node_ids.index(candidate_id) + 1]
        matches: list[str] = []
        for instruction_id in prefix:
            node = ueg.node_by_id(instruction_id)
            if node is None or node.layer != "instruction":
                continue
            entries = [edge.target for edge in ueg.edges if edge.source == instruction_id and edge.edge_type == "CALLS"]
            for entry in entries:
                pending = [entry]
                visited: set[str] = set()
                while pending:
                    current = pending.pop()
                    if current in visited:
                        continue
                    visited.add(current)
                    if current == code_node_id:
                        matches.append(instruction_id)
                        break
                    pending.extend(
                        edge.target for edge in ueg.edges
                        if edge.source == current
                        and (target := ueg.node_by_id(edge.target)) is not None
                        and target.layer == "code"
                    )
        return {matches[-1]} if matches else set()

    def _is_subsequence(
        self,
        expected: list[str],
        observed: list[str],
    ) -> bool:
        if not expected:
            return False
        iterator = iter(observed)
        return all(any(value == expected_id for value in iterator) for expected_id in expected)
