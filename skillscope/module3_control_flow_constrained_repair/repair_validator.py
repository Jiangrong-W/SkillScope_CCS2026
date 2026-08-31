from __future__ import annotations

import hashlib
import inspect
import json
import re
from pathlib import Path
from typing import Any

from skillscope.common.models import (
    CandidateAction,
    CandidateExtractionResult,
    ExecutionRecord,
    FinalVerdict,
    RepairItem,
    RepairPlan,
    RepairValidationReport,
    ResourceFixture,
    TaskSpec,
)
from skillscope.common.sandbox import SandboxedSkillAgent
from skillscope.module1_candidate_extraction import CandidateExtractionService
from skillscope.module2_action_necessity_validation.output_comparator import (
    OutputComparator,
)
from skillscope.module2_action_necessity_validation.service import (
    ActionNecessityValidationRun,
)

from .descriptor_clustering import DescriptorClusterer
from .code_rewriter import PrivilegeSemanticAnalyzer


class RepairValidator:
    """Replay every repaired descriptor context and recompute validation metrics."""

    def __init__(
        self,
        *,
        candidate_service: CandidateExtractionService,
        sandboxed_agent: SandboxedSkillAgent,
        output_comparator: OutputComparator | None = None,
        clusterer: DescriptorClusterer | None = None,
    ) -> None:
        self.candidate_service = candidate_service
        self.sandboxed_agent = sandboxed_agent
        self.output_comparator = output_comparator or OutputComparator()
        self.clusterer = clusterer or DescriptorClusterer()
        self.semantic_analyzer = PrivilegeSemanticAnalyzer()

    def validate(
        self,
        *,
        original_run: ActionNecessityValidationRun,
        repair_plan: RepairPlan,
        patched_bundle_root: Path,
    ) -> tuple[
        RepairValidationReport,
        CandidateExtractionResult,
        list[ExecutionRecord],
    ]:
        patched_analysis = self.candidate_service.run(patched_bundle_root)
        public_entrypoint_status = self._public_entrypoint_status(
            patched_bundle_root=patched_bundle_root,
            repair_plan=repair_plan,
        )
        projection_integrity_status = self._projection_integrity_status(
            patched_bundle_root=patched_bundle_root,
            patched_analysis=patched_analysis,
            repair_plan=repair_plan,
            public_entrypoint_status=public_entrypoint_status,
        )
        integrity_by_candidate = {
            str(entry["candidate_id"]): entry
            for entry in projection_integrity_status["candidates"]
        }
        contexts, context_skips, skipped_candidate_ids = self._validation_contexts(
            original_run=original_run,
            repair_plan=repair_plan,
        )
        tasks_by_pair = {
            (task.candidate_id, task.task_id): task
            for task in original_run.validation.tasks
        }
        tasks_by_id = {task.task_id: task for task in original_run.validation.tasks}
        original_records = self._original_records(original_run)
        candidates_by_id = {
            candidate.candidate_id: candidate
            for candidate in original_run.analysis.candidates
        }
        items_by_candidate = {
            str(item.metadata.get("source_candidate_id") or ""): item
            for item in repair_plan.items
        }

        execution_records: list[ExecutionRecord] = []
        after_verdicts: list[dict[str, Any]] = []
        successful_task_count = 0
        output_equivalent_count = 0
        core_preserved_count = 0
        goal_satisfied_count = 0
        patched_installed_skill = self.sandboxed_agent.install_skill(
            patched_bundle_root
        )

        for context_index, context in enumerate(contexts, start=1):
            candidate_id = context["candidate_id"]
            task_id = context["task_id"]
            candidate = candidates_by_id.get(candidate_id)
            item = items_by_candidate.get(candidate_id)
            task = tasks_by_pair.get((candidate_id, task_id)) or tasks_by_id.get(
                task_id
            )
            if candidate is None or item is None or task is None:
                skipped_candidate_ids.add(candidate_id)
                context_skips.append(
                    f"{candidate_id}/{task_id}: missing candidate, repair item, "
                    "or representative task."
                )
                continue

            fixtures = self._fixtures_without_routing_oracle(task)
            record = self._execute_patched_task(
                patched_installed_skill=patched_installed_skill,
                task=task,
                candidate=candidate,
                context_index=context_index,
                fixtures=fixtures,
            )
            execution_records.append(record)
            if record.status == "completed":
                successful_task_count += 1

            original_record = original_records.get((candidate_id, task_id))
            output_equivalent = self._output_equivalent(
                task=task,
                original_record=original_record,
                patched_record=record,
            )
            if output_equivalent:
                output_equivalent_count += 1
            core_preserved = self._core_preserved(
                task=task,
                candidate=candidate,
                item=item,
                record=record,
                patched_analysis=patched_analysis,
                projection_integrity_entry=integrity_by_candidate.get(
                    candidate_id
                ),
            )
            if core_preserved:
                core_preserved_count += 1
            # CorePres and GoalSat are independent obligations.  In the
            # absence of grounded original/patched result evidence, preserving
            # the task flow alone cannot establish that the user goal remains
            # satisfied.
            goal_satisfied = bool(
                record.status == "completed" and output_equivalent
            )
            if goal_satisfied:
                goal_satisfied_count += 1

            action_executed, execution_strategy = self._action_executed(
                record=record,
                candidate=candidate,
                item=item,
                patched_analysis=patched_analysis,
                projection_integrity_entry=integrity_by_candidate.get(candidate_id),
            )
            after_verdicts.append(
                self._after_repair_verdict(
                    context=context,
                    action_executed=action_executed,
                    action_execution_strategy=execution_strategy,
                    core_preserved=core_preserved,
                    goal_satisfied=goal_satisfied,
                    output_equivalent=output_equivalent,
                    record=record,
                )
            )

        original_confirmed_candidate_ids = sorted(
            candidate_id for candidate_id in items_by_candidate if candidate_id
        )
        for verdict in after_verdicts:
            integrity_complete = bool(
                integrity_by_candidate.get(str(verdict["candidate_id"]), {}).get(
                    "complete"
                )
            )
            verdict["projection_integrity_complete"] = integrity_complete
            if not integrity_complete and verdict["label"] != "overprivileged":
                verdict["label"] = "inconclusive"
                verdict["overprivilege_reasons"] = sorted(
                    {
                        *verdict["overprivilege_reasons"],
                        "projection_integrity_incomplete",
                    }
                )
        overprivileged = [
            verdict
            for verdict in after_verdicts
            if verdict["label"] == "overprivileged"
        ]
        inconclusive = [
            verdict for verdict in after_verdicts if verdict["label"] == "inconclusive"
        ]
        verdicts_by_candidate = {
            candidate_id: [
                verdict
                for verdict in after_verdicts
                if verdict["candidate_id"] == candidate_id
            ]
            for candidate_id in original_confirmed_candidate_ids
        }
        repaired_candidate_ids = sorted(
            candidate_id
            for candidate_id in original_confirmed_candidate_ids
            if candidate_id not in skipped_candidate_ids
            and bool(integrity_by_candidate.get(candidate_id, {}).get("complete"))
            and verdicts_by_candidate[candidate_id]
            and all(
                verdict["label"] == "not_overprivileged"
                and verdict["core_preserved"]
                and verdict["goal_satisfied"]
                for verdict in verdicts_by_candidate[candidate_id]
            )
        )
        remaining_candidate_ids = sorted(
            set(original_confirmed_candidate_ids) - set(repaired_candidate_ids)
        )
        remaining_overreach_ids = [
            self._overreach_id(candidate_id) for candidate_id in remaining_candidate_ids
        ]
        repaired_overreach_ids = [
            self._overreach_id(candidate_id) for candidate_id in repaired_candidate_ids
        ]
        unnecessary_count = sum(
            verdict["label"] == "overprivileged"
            and verdict["necessity_label"] == "unnecessary"
            for verdict in after_verdicts
        )
        targeted_signature_status = self._target_signature_status(
            original_run=original_run,
            patched_analysis=patched_analysis,
            repair_plan=repair_plan,
        )
        validation_mode = str(
            original_run.validation.metadata.get("validation_mode") or "dynamic"
        )
        report = RepairValidationReport(
            patched_skill_id=patched_analysis.bundle.bundle_id,
            overreach_count=len(original_confirmed_candidate_ids),
            remaining_overreach_count=len(remaining_candidate_ids),
            repaired_overreach_ids=repaired_overreach_ids,
            remaining_overreach_ids=remaining_overreach_ids,
            task_count=len(
                {
                    (verdict["candidate_id"], verdict["task_id"])
                    for verdict in after_verdicts
                }
            ),
            decision_count=len(after_verdicts),
            successful_task_count=successful_task_count,
            completed_replay_pairs=successful_task_count,
            unnecessary_count=unnecessary_count,
            overprivileged_count=len(overprivileged),
            output_equivalent_count=output_equivalent_count,
            core_preserved_count=core_preserved_count,
            goal_satisfied_count=goal_satisfied_count,
            notes=[
                "Repair validation reran every patched descriptor context, "
                "and a repair is successful only when every corresponding "
                "context preserves the prompt-required core flow and still "
                "satisfies the user's goal.",
                "After-repair verdicts count an action as over-privileged only "
                "when an unauthorized or unnecessary context still executes it.",
                (
                    "All repaired descriptor contexts produced conclusive "
                    "after-repair verdicts."
                    if not inconclusive
                    else f"{len(inconclusive)} repaired descriptor context(s) "
                    "remained inconclusive."
                ),
            ],
            metadata={
                "validator_strategy": (
                    "prompt_driven_patched_context_replay_plus_after_final_verdict"
                ),
                "routing_contract": (
                    "normal SKILLSCOPE_USER_PROMPT propagation; no expected "
                    "descriptor-cluster fixture"
                ),
                "public_entrypoint_status": public_entrypoint_status,
                "projection_integrity_status": projection_integrity_status,
                "validation_mode": validation_mode,
                "after_repair_final_verdicts": after_verdicts,
                "after_repair_inconclusive_count": len(inconclusive),
                "repaired_candidate_contract": (
                    "every_context_not_overprivileged_and_core_preserved_and_"
                    "goal_satisfied_and_projection_integrity_complete"
                ),
                "context_skips": context_skips,
                "skipped_candidate_ids": sorted(skipped_candidate_ids),
                "target_signature_status": targeted_signature_status,
                "metric_contract": {
                    "overprivileged": (
                        "unauthorized_or_unnecessary_and_action_still_executed"
                    ),
                    "core_preserved": (
                        "patched run completed and retained the representative "
                        "task core chain"
                    ),
                    "goal_satisfied": (
                        "patched run completed and grounded task-result evidence "
                        "remained semantically aligned with the user goal"
                    ),
                },
            },
        )
        return report, patched_analysis, execution_records

    def _validation_contexts(
        self,
        *,
        original_run: ActionNecessityValidationRun,
        repair_plan: RepairPlan,
    ) -> tuple[list[dict[str, Any]], list[str], set[str]]:
        verdicts_by_pair: dict[tuple[str, str], FinalVerdict] = {
            (verdict.candidate_id, verdict.task_id): verdict
            for verdict in original_run.validation.final_verdicts
        }
        items_by_candidate = {
            str(item.metadata.get("source_candidate_id") or ""): item
            for item in repair_plan.items
        }
        contexts_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
        skips: list[str] = []
        skipped_candidate_ids: set[str] = set()
        seen_descriptor_ids_by_candidate: dict[str, set[str]] = {}
        for descriptor in sorted(
            original_run.validation.descriptors,
            key=lambda value: value.descriptor_id,
        ):
            item = items_by_candidate.get(descriptor.candidate_id)
            if item is None or descriptor.descriptor_id not in item.descriptor_ids:
                continue
            seen_descriptor_ids_by_candidate.setdefault(
                descriptor.candidate_id, set()
            ).add(descriptor.descriptor_id)
            verdict = verdicts_by_pair.get(
                (descriptor.candidate_id, descriptor.task_id)
            )
            if verdict is None:
                skipped_candidate_ids.add(descriptor.candidate_id)
                skips.append(
                    f"{descriptor.descriptor_id}: missing final verdict during "
                    "repair validation."
                )
                continue
            normalized = self.clusterer.normalize(descriptor, final_verdict=verdict)
            if normalized.cluster_key in item.blocked_cluster_keys:
                disposition = "blocked"
            elif normalized.cluster_key in item.allowed_cluster_keys:
                disposition = "allowed"
            else:
                skipped_candidate_ids.add(descriptor.candidate_id)
                skips.append(
                    f"{descriptor.descriptor_id}: normalized cluster was not "
                    "present in the repair item's allowed/blocked contract."
                )
                continue
            key = (
                descriptor.candidate_id,
                descriptor.task_id,
                normalized.cluster_key,
            )
            context = contexts_by_key.setdefault(
                key,
                {
                    "candidate_id": descriptor.candidate_id,
                    "task_id": descriptor.task_id,
                    "cluster_key": normalized.cluster_key,
                    "disposition": disposition,
                    "descriptor_ids": [],
                    "authorization_label": verdict.authorization_label,
                    "necessity_label": verdict.necessity_label,
                    "original_final_verdict": verdict.label,
                    "original_overprivilege_reasons": list(
                        verdict.overprivilege_reasons
                    ),
                },
            )
            context["descriptor_ids"].append(descriptor.descriptor_id)
        contexts = list(contexts_by_key.values())
        for context in contexts:
            context["descriptor_ids"] = sorted(set(context["descriptor_ids"]))
        for candidate_id, item in items_by_candidate.items():
            if not candidate_id:
                continue
            missing_descriptor_ids = sorted(
                set(item.descriptor_ids)
                - seen_descriptor_ids_by_candidate.get(candidate_id, set())
            )
            if not item.descriptor_ids:
                skipped_candidate_ids.add(candidate_id)
                skips.append(
                    f"{candidate_id}: repair item has no descriptor context to "
                    "validate."
                )
            elif missing_descriptor_ids:
                skipped_candidate_ids.add(candidate_id)
                skips.append(
                    f"{candidate_id}: missing descriptor context(s): "
                    + ", ".join(missing_descriptor_ids)
                )
        contexts.sort(
            key=lambda value: (
                value["candidate_id"],
                value["task_id"],
                value["cluster_key"],
            )
        )
        return contexts, skips, skipped_candidate_ids

    def _execute_patched_task(
        self,
        *,
        patched_installed_skill: object,
        task: TaskSpec,
        candidate: CandidateAction,
        context_index: int,
        fixtures: list[ResourceFixture],
    ) -> ExecutionRecord:
        kwargs: dict[str, Any] = {
            "installed_skill": patched_installed_skill,
            "prompt": task.prompt,
            "run_id": (
                f"repair-validate:{candidate.candidate_id}:{task.task_id}:"
                f"context-{context_index:04d}"
            ),
            "mode": "repair_validation",
        }
        supports_fixtures = True
        try:
            signature = inspect.signature(self.sandboxed_agent.execute_installed_skill)
            supports_fixtures = "fixtures" in signature.parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        except (TypeError, ValueError):
            pass
        if supports_fixtures:
            kwargs["fixtures"] = fixtures
        try:
            return self.sandboxed_agent.execute_installed_skill(**kwargs)
        except Exception as exc:
            return ExecutionRecord(
                run_id=str(kwargs["run_id"]),
                mode="repair_validation",
                prompt=task.prompt,
                status="failed",
                bundle_root=str(
                    getattr(
                        getattr(patched_installed_skill, "bundle", None),
                        "root_path",
                        "",
                    )
                ),
                notes=[f"patched_execution_error={exc}"],
                metadata={"execution_strategy": "failed_patched_descriptor_context"},
            )

    def _fixtures_without_routing_oracle(self, task: TaskSpec) -> list[ResourceFixture]:
        return [
            fixture
            for fixture in task.fixtures
            if not (
                fixture.fixture_type == "env"
                and fixture.target == "SKILLSCOPE_TASK_CLUSTER"
            )
        ]

    def _original_records(
        self, original_run: ActionNecessityValidationRun
    ) -> dict[tuple[str, str], ExecutionRecord]:
        records: dict[tuple[str, str], ExecutionRecord] = {}
        for pair in original_run.validation.replay_pairs:
            records.setdefault((pair.candidate_id, pair.task_id), pair.original)
        return records

    def _output_equivalent(
        self,
        *,
        task: TaskSpec,
        original_record: ExecutionRecord | None,
        patched_record: ExecutionRecord,
    ) -> bool:
        if (
            patched_record.status != "completed"
            or original_record is None
            or not original_record.final_output
            or not patched_record.final_output
            or original_record.metadata.get("final_output_grounded") is not True
            or patched_record.metadata.get("final_output_grounded") is not True
        ):
            return False
        original_result_fingerprints = self._direct_result_fingerprints(
            original_record
        )
        patched_result_fingerprints = self._direct_result_fingerprints(
            patched_record
        )
        if not original_result_fingerprints or not patched_result_fingerprints:
            return False
        if not (
            self._final_output_matches_direct_result(original_record)
            and self._final_output_matches_direct_result(patched_record)
        ):
            return False
        comparison = self.output_comparator.compare(
            prompt=task.prompt,
            original_output=original_record.final_output,
            replay_output=patched_record.final_output,
            task_summary=task.task_summary,
            original_output_grounded=True,
            replay_output_grounded=True,
        )
        return comparison.get("equivalent") is True

    @staticmethod
    def _has_direct_result_evidence(record: ExecutionRecord) -> bool:
        evidence_refs = record.metadata.get("final_output_evidence_refs")
        return bool(
            isinstance(evidence_refs, list)
            and any(
                isinstance(value, str)
                and (
                    value == "execution.stdout"
                    or value.startswith("execution.tool_output.")
                )
                for value in evidence_refs
            )
        )

    @staticmethod
    def _direct_result_fingerprints(
        record: ExecutionRecord,
    ) -> tuple[str, ...]:
        values = record.metadata.get("final_output_direct_result_sha256")
        if not isinstance(values, list) or not values:
            return ()
        normalized: list[str] = []
        for value in values:
            if not isinstance(value, str) or len(value) != 64:
                return ()
            lowered = value.casefold()
            if any(character not in "0123456789abcdef" for character in lowered):
                return ()
            normalized.append(lowered)
        return tuple(sorted(normalized))

    @classmethod
    def _final_output_matches_direct_result(
        cls,
        record: ExecutionRecord,
    ) -> bool:
        if not cls._has_direct_result_evidence(record):
            return False
        normalized_output = cls._normalize_result_text(record.final_output)
        if not normalized_output:
            return False
        output_fingerprint = hashlib.sha256(
            normalized_output.encode("utf-8")
        ).hexdigest()
        return output_fingerprint in cls._direct_result_fingerprints(record)

    @staticmethod
    def _normalize_result_text(value: str) -> str:
        return value.replace("\r\n", "\n").replace("\r", "\n").strip()

    def _core_preserved(
        self,
        *,
        task: TaskSpec,
        candidate: CandidateAction,
        item: RepairItem,
        record: ExecutionRecord,
        patched_analysis: CandidateExtractionResult,
        projection_integrity_entry: dict[str, object] | None = None,
    ) -> bool:
        if record.status != "completed":
            return False
        expected_candidate_id = task.expected_candidate_node_id or candidate.node_id
        covered_code_node_ids = {
            str(value) for value in item.metadata.get("covered_code_node_ids", [])
        }
        excluded_node_ids = {
            candidate.node_id,
            expected_candidate_id,
            *covered_code_node_ids,
        }
        core_node_ids = [
            node_id
            for node_id in task.chain_node_ids
            if node_id not in excluded_node_ids
        ]
        if core_node_ids and set(core_node_ids).issubset(set(record.executed_node_ids)):
            return True

        candidate_position = None
        if expected_candidate_id in task.chain_node_ids:
            candidate_position = task.chain_node_ids.index(expected_candidate_id)
        covered_code_summaries = {
            self._normalize(str(value))
            for value in item.metadata.get("covered_code_summaries", [])
        }
        core_summaries = [
            summary
            for index, summary in enumerate(task.chain_summaries)
            if index != candidate_position
            and self._normalize(summary) != self._normalize(candidate.summary)
            and self._normalize(summary) not in covered_code_summaries
            and not self._is_boundary_summary(summary)
        ]
        if not core_summaries:
            return True

        observed_summaries = [event.summary for event in record.trace if event.summary]
        observed_summaries.extend(
            str(event.get("summary") or "")
            for event in record.raw_trace
            if event.get("summary")
        )
        for node_id in record.executed_node_ids:
            node = patched_analysis.ueg.node_by_id(node_id)
            if node is not None and node.summary:
                observed_summaries.append(node.summary)
        return all(
            self._semantically_contains(observed_summaries, summary)
            or self._projected_invocation_preserved(
                summary=summary,
                candidate=candidate,
                item=item,
                record=record,
                patched_analysis=patched_analysis,
                projection_integrity_entry=projection_integrity_entry,
            )
            for summary in core_summaries
        )

    def _projected_invocation_preserved(
        self,
        *,
        summary: str,
        candidate: CandidateAction,
        item: RepairItem,
        record: ExecutionRecord,
        patched_analysis: CandidateExtractionResult,
        projection_integrity_entry: dict[str, object] | None = None,
    ) -> bool:
        """Match an original script invocation to its executed repair unit.

        Code projection deliberately replaces an instruction that invoked the
        original source file with a guarded allowed unit or a neutralized safe
        unit.  That filename change is not itself a loss of task semantics.  We
        accept the structural substitution only when the original invocation
        is the summary being checked, the projected unit exists in the patched
        graph, and runtime evidence shows that unit actually executed.  Other
        core-chain summaries must still match independently.
        """

        source_file = str(
            item.metadata.get("dispatch_source_file")
            or item.source_file
            or candidate.source_file
            or ""
        )
        if not (
            isinstance(projection_integrity_entry, dict)
            and projection_integrity_entry.get("complete") is True
        ):
            return False

        normalized_summary = self._normalize(summary)
        normalized_source = self._normalize(source_file)
        summary_tokens = set(normalized_summary.split())
        invocation_verbs = {"run", "invoke", "execute", "call"}
        if not normalized_source or not invocation_verbs & summary_tokens:
            return False

        template = item.metadata.get("instruction_invocation_template")
        template_matches = False
        expected_tool_name = ""
        if isinstance(template, dict):
            prefix_tokens = template.get("prefix_tokens")
            source_token = str(template.get("source_token") or "")
            suffix_tokens = template.get("suffix_tokens")
            invocation_prefix = [
                str(token).strip()
                for token in prefix_tokens
                if str(token).strip()
            ] if isinstance(prefix_tokens, list) else []
            suffix = Path(source_file).suffix.casefold()
            language_tokens = {
                ".py": {"python", "python3", "script"},
                ".sh": {"shell", "sh", "bash", "script"},
                ".js": {"javascript", "node", "nodejs", "script"},
                ".ts": {"typescript", "node", "nodejs", "script"},
            }.get(suffix, {"script"})
            expected_tool_name = {
                ".py": "python_script",
                ".sh": "shell_script",
                ".js": "node_script",
                ".ts": "node_script",
            }.get(suffix, "")
            accepted_prefixes = {
                ".py": {"python", "python3"},
                ".sh": {"sh", "bash"},
                ".js": {"node", "nodejs"},
                ".ts": {"node", "nodejs"},
            }.get(suffix, set())
            template_matches = bool(
                self._normalized_relative_path(source_token)
                == self._normalized_relative_path(source_file)
                and isinstance(suffix_tokens, list)
                and not suffix_tokens
                and len(invocation_prefix) == 1
                and invocation_prefix[0] in accepted_prefixes
                and language_tokens & summary_tokens
                and expected_tool_name
            )
        # A filename mention is not enough: the projected runtime currently
        # preserves only argument-free script invocations. Require the full
        # recorded invocation template so arguments cannot be silently lost.
        if not template_matches:
            return False

        projected_units = self._projected_execution_units(item)
        graph_units = {
            str(node.source_file or "")
            for node in patched_analysis.ueg.nodes
            if node.source_file
        }
        represented_units = {
            unit
            for unit in projected_units
            if unit in graph_units
            or Path(unit).name in {Path(graph_unit).name for graph_unit in graph_units}
        }
        if not represented_units:
            return False

        return self._observed_projected_tool_invocation(
            record=record,
            projected_units=represented_units,
            expected_tool_name=expected_tool_name,
        )

    def _observed_projected_tool_invocation(
        self,
        *,
        record: ExecutionRecord,
        projected_units: set[str],
        expected_tool_name: str,
    ) -> bool:
        """Require a concrete top-level tool call to one projected unit.

        Internal coverage events and filename mentions are insufficient.  The
        invocation envelope is preserved only when the traced runtime records
        the matching script tool and exact generated unit as a tool call.
        """

        normalized_units = {
            self._normalized_relative_path(unit)
            for unit in projected_units
            if unit
        }
        if not normalized_units or not expected_tool_name:
            return False

        events: list[dict[str, Any]] = list(record.raw_trace)
        events.extend(
            {
                "event_type": event.event_type,
                "object_ref": event.object_ref,
                "attributes": event.attributes,
            }
            for event in record.trace
        )
        for event in events:
            if event.get("event_type") != "tool_call_start":
                continue
            attributes = event.get("attributes")
            if not isinstance(attributes, dict):
                continue
            if str(attributes.get("tool_name") or "") != expected_tool_name:
                continue
            reference = str(
                event.get("object_ref")
                or event.get("target")
                or attributes.get("target")
                or ""
            )
            normalized_reference = self._normalized_relative_path(reference)
            if any(
                normalized_reference == unit
                or normalized_reference.endswith("/" + unit)
                for unit in normalized_units
            ):
                return True
        return False

    def _action_executed(
        self,
        *,
        record: ExecutionRecord,
        candidate: CandidateAction,
        item: RepairItem,
        patched_analysis: CandidateExtractionResult,
        projection_integrity_entry: dict[str, object] | None = None,
    ) -> tuple[bool | None, str]:
        projected_units = self._projected_execution_units(item)
        observed_units = self._observed_projected_execution_units(
            record=record,
            projected_units=projected_units,
        )
        manifest = item.metadata.get("source_variant_manifest")
        if isinstance(manifest, list) and manifest:
            observed_variants = [
                variant
                for variant in manifest
                if isinstance(variant, dict)
                and str(variant.get("relative_path") or "")
                and str(variant["relative_path"]) in observed_units
            ]
            if len(observed_variants) > 1:
                return (
                    None,
                    "multiple_task_conditioned_execution_units_observed",
                )
            if len(observed_variants) == 1:
                allowed_ids = {
                    str(value)
                    for value in observed_variants[0].get("allowed_repair_ids", [])
                }
                effective_repair_ids = {item.repair_id}
                manifest_repair_ids = {
                    str(value)
                    for variant in manifest
                    if isinstance(variant, dict)
                    for value in variant.get("allowed_repair_ids", [])
                }
                if item.repair_id not in manifest_repair_ids:
                    effective_repair_ids = {
                        str(value)
                        for value in item.metadata.get(
                            "covered_by_code_projection_repair_ids", []
                        )
                    } or effective_repair_ids
                if allowed_ids & effective_repair_ids:
                    return (
                        True,
                        "observed_independent_allowed_execution_unit",
                    )
                return False, "observed_independent_blocked_execution_unit"

        safe_unit = str(item.metadata.get("safe_execution_unit") or "")
        allowed_unit = str(item.metadata.get("allowed_execution_unit") or "")
        observed_safe = safe_unit in observed_units
        observed_allowed = allowed_unit in observed_units
        if observed_safe and observed_allowed:
            return None, "conflicting_allowed_and_safe_execution_units_observed"
        if observed_safe:
            expected_semantics = item.metadata.get("safe_expected_privilege_semantics")
            observed_semantics = self.semantic_analyzer.semantics_for_graph_nodes(
                list(patched_analysis.ueg.nodes),
                source_file=safe_unit,
            )
            direct_semantics_proven = bool(
                item.metadata.get("safe_semantic_proof_complete") is True
                and isinstance(expected_semantics, list)
                and observed_semantics == expected_semantics
            )
            if direct_semantics_proven:
                return False, "observed_safe_execution_unit"

            if (
                isinstance(projection_integrity_entry, dict)
                and projection_integrity_entry.get("complete") is True
                and bool(
                    projection_integrity_entry.get("safe_semantics_valid")
                    or projection_integrity_entry.get("variant_semantics_valid")
                )
            ):
                return (
                    False,
                    "observed_independently_validated_safe_execution_unit",
                )

            projection_strategy = str(
                item.metadata.get("instruction_projection_strategy") or ""
            )
            cross_layer_evidence_field = {
                "covered_by_semantically_equivalent_code_projection": (
                    "cross_layer_equivalence_evidence_valid"
                ),
                "covered_by_causally_complete_code_projection": (
                    "cross_layer_causal_coverage_evidence_valid"
                ),
            }.get(projection_strategy)
            if (
                cross_layer_evidence_field
                and isinstance(projection_integrity_entry, dict)
                and projection_integrity_entry.get("complete") is True
                and projection_integrity_entry.get(cross_layer_evidence_field) is True
            ):
                return (
                    False,
                    "observed_cross_layer_semantically_validated_safe_unit",
                )
            return (
                None,
                "safe_execution_unit_semantic_integrity_unproven",
            )
        if observed_allowed:
            return True, "observed_allowed_execution_unit"

        patched_candidate_node_ids = {
            patched_candidate.node_id
            for patched_candidate in patched_analysis.candidates
            if patched_candidate.layer == candidate.layer
            and self._normalize(patched_candidate.summary)
            == self._normalize(candidate.summary)
        }
        if patched_candidate_node_ids & set(record.executed_node_ids):
            return True, "observed_matching_patched_candidate_node"

        if record.status == "completed" and patched_candidate_node_ids:
            return (
                False,
                "completed_trace_omits_matching_patched_candidate_node",
            )
        return None, "action_execution_not_observable"

    def _observed_projected_execution_units(
        self,
        *,
        record: ExecutionRecord,
        projected_units: set[str],
    ) -> set[str]:
        """Resolve execution units only from structured runtime evidence.

        User-visible output, stdout, stderr, summaries, and free-form notes can
        mention a filename without executing it.  They are intentionally not
        evidence here.  Runtime tool/trace path fields are matched against the
        finite set of projected units; basename matches remain conservative
        when multiple units share a basename.
        """

        normalized_units = {
            unit: self._normalized_relative_path(unit)
            for unit in projected_units
            if unit
        }
        names_to_units: dict[str, set[str]] = {}
        for unit, normalized in normalized_units.items():
            names_to_units.setdefault(Path(normalized).name, set()).add(unit)

        references: list[str] = []

        def collect_mapping(value: object) -> None:
            if not isinstance(value, dict):
                return
            for key in (
                "object_ref",
                "target",
                "source_file",
                "script_relative_path",
                "executed_script",
                "execution_unit",
            ):
                reference = value.get(key)
                if isinstance(reference, str) and reference:
                    references.append(reference)
            attributes = value.get("attributes")
            if isinstance(attributes, dict):
                collect_mapping(attributes)

        for event in record.raw_trace:
            collect_mapping(event)
        for event in record.trace:
            if event.object_ref:
                references.append(event.object_ref)
            collect_mapping(event.attributes)
        collect_mapping(record.metadata)

        observed: set[str] = set()
        for reference in references:
            normalized_reference = self._normalized_relative_path(reference)
            exact = {
                unit
                for unit, normalized in normalized_units.items()
                if normalized_reference == normalized
                or normalized_reference.endswith("/" + normalized)
            }
            if exact:
                observed.update(exact)
                continue
            observed.update(names_to_units.get(Path(normalized_reference).name, set()))
        return observed

    def _projected_execution_units(self, item: RepairItem) -> set[str]:
        manifest = item.metadata.get("source_variant_manifest")
        if isinstance(manifest, list):
            units = {
                str(value.get("relative_path") or "")
                for value in manifest
                if isinstance(value, dict)
            } - {""}
            if units:
                return units
        return {
            str(item.metadata.get(key) or "")
            for key in ("allowed_execution_unit", "safe_execution_unit")
        } - {""}

    def _after_repair_verdict(
        self,
        *,
        context: dict[str, Any],
        action_executed: bool | None,
        action_execution_strategy: str,
        core_preserved: bool,
        goal_satisfied: bool,
        output_equivalent: bool,
        record: ExecutionRecord,
    ) -> dict[str, Any]:
        blocked = context["disposition"] == "blocked"
        if action_executed is None or record.status != "completed":
            label = "inconclusive"
            reasons = [
                "patched_action_execution_unresolved",
                *(["core_flow_not_preserved"] if not core_preserved else []),
                *(["goal_not_satisfied"] if not goal_satisfied else []),
            ]
        elif blocked and action_executed:
            label = "overprivileged"
            reasons = sorted(
                set(
                    context["original_overprivilege_reasons"]
                    or [
                        reason
                        for reason, active in (
                            (
                                "unauthorized",
                                context["authorization_label"] == "unauthorized",
                            ),
                            (
                                "unnecessary",
                                context["necessity_label"] == "unnecessary",
                            ),
                        )
                        if active
                    ]
                )
            )
        elif not core_preserved or not goal_satisfied:
            label = "inconclusive"
            reasons = [
                reason
                for reason, active in (
                    ("core_flow_not_preserved", not core_preserved),
                    ("goal_not_satisfied", not goal_satisfied),
                )
                if active
            ]
        elif not blocked and not action_executed:
            label = "inconclusive"
            reasons = ["authorized_necessary_action_not_executed"]
        else:
            label = "not_overprivileged"
            reasons = []
        return {
            "candidate_id": context["candidate_id"],
            "task_id": context["task_id"],
            "descriptor_ids": list(context["descriptor_ids"]),
            "cluster_key": context["cluster_key"],
            "cluster_disposition": context["disposition"],
            "label": label,
            "authorization_label": context["authorization_label"],
            "necessity_label": context["necessity_label"],
            "action_executed": action_executed,
            "action_execution_strategy": action_execution_strategy,
            "core_preserved": core_preserved,
            "goal_satisfied": goal_satisfied,
            "output_equivalent": output_equivalent,
            "execution_status": record.status,
            "execution_run_id": record.run_id,
            "overprivilege_reasons": reasons,
        }

    def _target_signature_status(
        self,
        *,
        original_run: ActionNecessityValidationRun,
        patched_analysis: CandidateExtractionResult,
        repair_plan: RepairPlan,
    ) -> dict[str, object]:
        repaired_candidate_ids = {
            str(item.metadata.get("source_candidate_id") or "")
            for item in repair_plan.items
        }
        target_signatures = {
            self._overreach_signature(
                candidate.layer, candidate.source_file, candidate.summary
            )
            for candidate in original_run.analysis.candidates
            if candidate.candidate_id in repaired_candidate_ids
        }
        patched_signatures = {
            self._overreach_signature(
                candidate.layer, candidate.source_file, candidate.summary
            )
            for candidate in patched_analysis.candidates
        }
        return {
            "target_signature_count": len(target_signatures),
            "remaining_target_signatures": sorted(
                target_signatures & patched_signatures
            ),
            "note": (
                "Static candidate presence is diagnostic only because a guarded "
                "allowed branch intentionally remains in the patched bundle."
            ),
        }

    def _projection_integrity_status(
        self,
        *,
        patched_bundle_root: Path,
        patched_analysis: CandidateExtractionResult,
        repair_plan: RepairPlan,
        public_entrypoint_status: dict[str, object],
    ) -> dict[str, object]:
        """Verify that projected artifacts and semantic dispatch are complete.

        Replay evidence alone cannot prove that a repair is safely deployable.
        A candidate is therefore ineligible for the repaired set unless its
        public entrypoint is safe-only, its execution units exist, and the
        instruction/UEG dispatch reaches both units.  This also prevents a
        partially written or overwritten same-source projection from being
        reported as successful.
        """

        root = patched_bundle_root.resolve()
        public_by_repair = {
            str(entry.get("repair_id") or ""): entry
            for entry in public_entrypoint_status.get("entrypoints", [])
            if isinstance(entry, dict)
        }
        instruction_texts: list[str] = []
        for artifact in patched_analysis.bundle.instruction_files:
            target = (root / artifact.relative_path).resolve()
            if (target != root and root not in target.parents) or not target.is_file():
                continue
            instruction_texts.append(target.read_text(encoding="utf-8"))
        instruction_text = "\n".join(instruction_texts)

        graph_source_files = {
            self._normalized_relative_path(str(node.source_file or ""))
            for node in patched_analysis.ueg.nodes
            if node.source_file
        }
        called_resources = {
            self._normalized_relative_path(
                str(edge.attributes.get("invoked_resource") or "")
            )
            for edge in patched_analysis.ueg.edges
            if edge.edge_type == "CALLS" and edge.attributes.get("invoked_resource")
        }

        items_by_source: dict[str, list[RepairItem]] = {}
        items_by_repair_id = {item.repair_id: item for item in repair_plan.items}
        for item in repair_plan.items:
            if (
                item.source_file
                and item.repair_type == "REORGANIZE_CODE_AND_ADD_DISPATCH"
            ):
                items_by_source.setdefault(item.source_file, []).append(item)

        candidates: list[dict[str, object]] = []
        integrity_by_repair_id: dict[str, dict[str, object]] = {}
        ordered_items = sorted(
            repair_plan.items,
            key=lambda item: (
                item.repair_type != "REORGANIZE_CODE_AND_ADD_DISPATCH",
                item.repair_id,
            ),
        )
        for item in ordered_items:
            candidate_id = self._candidate_id_for_item(item)
            entry: dict[str, object] = {
                "candidate_id": candidate_id,
                "repair_id": item.repair_id,
                "repair_type": item.repair_type,
                "complete": False,
            }
            if item.repair_type == "GUARD_INSTRUCTION_TASK_CONDITIONED":
                projection_strategy = str(
                    item.metadata.get("instruction_projection_strategy") or ""
                )
                if projection_strategy == (
                    "unresolved_cross_layer_projection_conflict"
                ):
                    entry.update(
                        {
                            "instruction_projection_strategy": (projection_strategy),
                            "complete": False,
                            "reason": (
                                "cross-layer invocation reachability did not "
                                "prove equivalent action and guard semantics"
                            ),
                        }
                    )
                    candidates.append(entry)
                    integrity_by_repair_id[item.repair_id] = entry
                    continue
                covered_by = [
                    str(value)
                    for value in item.metadata.get(
                        "covered_by_code_projection_repair_ids", []
                    )
                ]
                if covered_by:
                    if projection_strategy == (
                        "covered_by_semantically_equivalent_code_projection"
                    ):
                        evidence_valid, evidence_reason = (
                            self._cross_layer_equivalence_evidence_valid(
                                item=item,
                                covered_repair_ids=covered_by,
                                items_by_repair_id=items_by_repair_id,
                            )
                        )
                        evidence_field = "cross_layer_equivalence_evidence_valid"
                        completion_reason = (
                            "instruction invocation is enforced by complete "
                            "semantically equivalent concrete code projection(s)"
                        )
                    elif projection_strategy == (
                        "covered_by_causally_complete_code_projection"
                    ):
                        evidence_valid, evidence_reason = (
                            self._cross_layer_causal_coverage_evidence_valid(
                                item=item,
                                covered_repair_ids=covered_by,
                                items_by_repair_id=items_by_repair_id,
                                integrity_by_repair_id=(integrity_by_repair_id),
                            )
                        )
                        evidence_field = "cross_layer_causal_coverage_evidence_valid"
                        completion_reason = (
                            "instruction action is enforced by complete concrete "
                            "code projections that neutralize every realized "
                            "material action"
                        )
                    else:
                        evidence_valid = False
                        evidence_reason = (
                            "cross-layer code coverage strategy is unsupported"
                        )
                        evidence_field = "cross_layer_coverage_evidence_valid"
                        completion_reason = ""
                    covered_entries = [
                        integrity_by_repair_id.get(repair_id)
                        for repair_id in covered_by
                    ]
                    complete = bool(
                        evidence_valid
                        and covered_entries
                        and all(
                            covered_entry is not None
                            and bool(covered_entry.get("complete"))
                            for covered_entry in covered_entries
                        )
                    )
                    entry.update(
                        {
                            "covered_by_code_projection_repair_ids": (covered_by),
                            "instruction_projection_strategy": (projection_strategy),
                            evidence_field: evidence_valid,
                            "complete": complete,
                            "reason": (
                                completion_reason
                                if complete
                                else evidence_reason
                                or "covering code projection is incomplete"
                            ),
                        }
                    )
                    candidates.append(entry)
                    integrity_by_repair_id[item.repair_id] = entry
                    continue
                if projection_strategy in {
                    "covered_by_semantically_equivalent_code_projection",
                    "covered_by_causally_complete_code_projection",
                }:
                    entry.update(
                        {
                            "instruction_projection_strategy": (projection_strategy),
                            "complete": False,
                            "reason": (
                                "cross-layer coverage strategy named no concrete "
                                "covering code repair"
                            ),
                        }
                    )
                    candidates.append(entry)
                    integrity_by_repair_id[item.repair_id] = entry
                    continue
                guard = str(item.guard_condition or "")
                guard_present = bool(guard and guard in instruction_text)
                entry.update(
                    {
                        "instruction_guard_present": guard_present,
                        "complete": guard_present,
                        "reason": (
                            "instruction guard is present in the projected "
                            "Skill instructions"
                            if guard_present
                            else "projected instruction guard is missing"
                        ),
                    }
                )
                candidates.append(entry)
                integrity_by_repair_id[item.repair_id] = entry
                continue

            if item.repair_type != "REORGANIZE_CODE_AND_ADD_DISPATCH":
                entry["reason"] = "unsupported repair type"
                candidates.append(entry)
                integrity_by_repair_id[item.repair_id] = entry
                continue

            source_file = str(item.source_file or "")
            peer_items = items_by_source.get(source_file, [item])
            manifest = item.metadata.get("source_variant_manifest")
            if isinstance(manifest, list) and manifest:
                entry.update(
                    self._composite_projection_integrity(
                        root=root,
                        item=item,
                        peer_items=peer_items,
                        manifest=manifest,
                        instruction_text=instruction_text,
                        graph_source_files=graph_source_files,
                        called_resources=called_resources,
                        public_safe_only=bool(
                            public_by_repair.get(item.repair_id, {}).get("safe_only")
                        ),
                    )
                )
                candidates.append(entry)
                integrity_by_repair_id[item.repair_id] = entry
                continue

            allowed_unit = str(item.metadata.get("allowed_execution_unit") or "")
            safe_unit = str(item.metadata.get("safe_execution_unit") or "")
            public_entry = public_by_repair.get(item.repair_id, {})
            public_safe_only = bool(public_entry.get("safe_only"))
            allowed_path = self._rooted_file(root, allowed_unit)
            safe_path = self._rooted_file(root, safe_unit)
            units_exist = allowed_path is not None and safe_path is not None
            expected_allowed_hash = str(item.metadata.get("allowed_unit_sha256") or "")
            expected_safe_hash = str(item.metadata.get("safe_unit_sha256") or "")
            unit_hashes_valid = bool(
                allowed_path is not None
                and safe_path is not None
                and expected_allowed_hash
                and expected_safe_hash
                and self._file_sha256(allowed_path) == expected_allowed_hash
                and self._file_sha256(safe_path) == expected_safe_hash
            )
            normalized_units = {
                self._normalized_relative_path(allowed_unit),
                self._normalized_relative_path(safe_unit),
            } - {""}
            graph_units_present = bool(
                len(normalized_units) == 2
                and normalized_units.issubset(graph_source_files)
            )
            graph_dispatch_complete = bool(
                len(normalized_units) == 2
                and normalized_units.issubset(called_resources)
            )
            dispatch_block = str(item.metadata.get("instruction_dispatch_block") or "")
            instruction_dispatch_complete = bool(
                dispatch_block
                and dispatch_block in instruction_text
                and allowed_unit in instruction_text
                and safe_unit in instruction_text
            )

            peer_repair_ids = {peer.repair_id for peer in peer_items}
            neutralized_ids = {
                str(value) for value in item.metadata.get("neutralized_repair_ids", [])
            }
            neutralization_complete = peer_repair_ids.issubset(neutralized_ids)
            safe_targets_absent = False
            safe_semantics_valid = False
            safe_semantic_status: dict[str, object] = {
                "complete": False,
                "reason": "safe or original execution unit was missing",
            }
            if safe_path is not None:
                safe_source = safe_path.read_text(encoding="utf-8")
                raw_targets = [
                    str(peer.raw_text or "").strip()
                    for peer in peer_items
                    if str(peer.raw_text or "").strip()
                ]
                safe_targets_absent = all(
                    raw_target not in safe_source for raw_target in raw_targets
                )
                if not raw_targets:
                    safe_targets_absent = neutralization_complete
            if allowed_path is not None and safe_path is not None:
                safe_semantic_status = self.semantic_analyzer.compare_projection(
                    original_source=allowed_path.read_text(encoding="utf-8"),
                    projected_source=safe_path.read_text(encoding="utf-8"),
                    suffix=allowed_path.suffix,
                    filename=safe_path.name,
                    blocked_items=peer_items,
                )
                safe_semantics_valid = bool(safe_semantic_status["complete"])

            expected_original_hash = str(
                item.metadata.get("original_source_sha256") or ""
            )
            allowed_semantics_preserved = bool(
                allowed_path is not None
                and expected_allowed_hash
                and expected_original_hash
                and expected_allowed_hash == expected_original_hash
                and self._file_sha256(allowed_path) == expected_allowed_hash
            )

            complete = all(
                (
                    public_safe_only,
                    units_exist,
                    unit_hashes_valid,
                    graph_units_present,
                    graph_dispatch_complete,
                    instruction_dispatch_complete,
                    neutralization_complete,
                    safe_semantics_valid,
                    allowed_semantics_preserved,
                )
            )
            entry.update(
                {
                    "source_file": source_file,
                    "allowed_execution_unit": allowed_unit,
                    "safe_execution_unit": safe_unit,
                    "public_entrypoint_safe_only": public_safe_only,
                    "units_exist": units_exist,
                    "unit_hashes_valid": unit_hashes_valid,
                    "graph_units_present": graph_units_present,
                    "graph_dispatch_complete": graph_dispatch_complete,
                    "instruction_dispatch_complete": (instruction_dispatch_complete),
                    "neutralization_complete": neutralization_complete,
                    "safe_targets_absent": safe_targets_absent,
                    "safe_semantics_valid": safe_semantics_valid,
                    "safe_semantic_reason": str(
                        safe_semantic_status.get("reason") or ""
                    ),
                    "safe_unexpected_privilege_semantics": list(
                        safe_semantic_status.get("unexpected_privilege_semantics", [])
                    ),
                    "allowed_semantics_preserved": (allowed_semantics_preserved),
                    "complete": complete,
                    "reason": (
                        "safe-only entrypoint, execution units, neutralization, "
                        "and semantic instruction/graph dispatch are complete"
                        if complete
                        else "one or more projection integrity checks failed"
                    ),
                }
            )
            candidates.append(entry)
            integrity_by_repair_id[item.repair_id] = entry

        return {
            "all_complete": bool(candidates)
            and all(bool(entry["complete"]) for entry in candidates),
            "candidates": candidates,
        }

    def _cross_layer_equivalence_evidence_valid(
        self,
        *,
        item: RepairItem,
        covered_repair_ids: list[str],
        items_by_repair_id: dict[str, RepairItem],
    ) -> tuple[bool, str]:
        """Recompute explicit instruction/code equivalence evidence.

        A source filename in an instruction summary proves reachability only.
        Sharing the code projection requires the projector's complete evidence
        record, and every recorded value is compared with the live repair plan
        rather than trusting a boolean coverage flag.
        """

        evidence = item.metadata.get("cross_layer_equivalence_evidence")
        if not isinstance(evidence, dict):
            return False, "cross-layer semantic equivalence evidence is missing"
        required_fields = {
            "evidence_version",
            "instruction_candidate_semantics",
            "instruction_guard_condition",
            "instruction_allowed_cluster_keys",
            "instruction_blocked_cluster_keys",
            "covered_code_items",
        }
        if not required_fields.issubset(evidence):
            return False, "cross-layer semantic equivalence evidence is incomplete"
        if evidence.get("evidence_version") != 1:
            return False, "cross-layer equivalence evidence version is unsupported"

        instruction_semantics = item.metadata.get("candidate_semantics")
        if (
            not isinstance(instruction_semantics, dict)
            or evidence.get("instruction_candidate_semantics") != instruction_semantics
            or evidence.get("instruction_guard_condition") != item.guard_condition
            or evidence.get("instruction_allowed_cluster_keys")
            != item.allowed_cluster_keys
            or evidence.get("instruction_blocked_cluster_keys")
            != item.blocked_cluster_keys
        ):
            return False, "instruction-side equivalence evidence is stale"

        covered_entries = evidence.get("covered_code_items")
        if not isinstance(covered_entries, list) or any(
            not isinstance(value, dict) for value in covered_entries
        ):
            return False, "covered code equivalence evidence is malformed"
        by_repair_id = {
            str(value.get("repair_id") or ""): value for value in covered_entries
        }
        if (
            "" in by_repair_id
            or set(by_repair_id) != set(covered_repair_ids)
            or len(by_repair_id) != len(covered_entries)
        ):
            return False, "covered code repair IDs do not match the evidence"

        for repair_id in covered_repair_ids:
            code_item = items_by_repair_id.get(repair_id)
            recorded = by_repair_id.get(repair_id)
            if (
                code_item is None
                or code_item.layer != "code"
                or code_item.repair_type != "REORGANIZE_CODE_AND_ADD_DISPATCH"
                or recorded is None
            ):
                return False, "a covered code repair is absent or not concrete"
            code_semantics = code_item.metadata.get("candidate_semantics")
            if (
                not isinstance(code_semantics, dict)
                or recorded.get("candidate_semantics") != code_semantics
                or recorded.get("guard_condition") != code_item.guard_condition
                or recorded.get("allowed_cluster_keys")
                != code_item.allowed_cluster_keys
                or recorded.get("blocked_cluster_keys")
                != code_item.blocked_cluster_keys
            ):
                return False, "covered code equivalence evidence is stale"

            # Recompute the actual semantic boundary; matching an evidence
            # record to each side is not enough if the two sides differ.
            if (
                instruction_semantics != code_semantics
                or item.guard_condition != code_item.guard_condition
                or item.allowed_cluster_keys != code_item.allowed_cluster_keys
                or item.blocked_cluster_keys != code_item.blocked_cluster_keys
            ):
                return False, "instruction and code projection semantics differ"
        return True, ""

    def _cross_layer_causal_coverage_evidence_valid(
        self,
        *,
        item: RepairItem,
        covered_repair_ids: list[str],
        items_by_repair_id: dict[str, RepairItem],
        integrity_by_repair_id: dict[str, dict[str, object]],
    ) -> tuple[bool, str]:
        """Recompute complete instruction-to-code material-action coverage.

        An instruction such as ``Run script`` can be over-privileged because a
        concrete action realized inside that script is unauthorized.  This is a
        valid cross-layer coverage relation only when every material action
        observed for the instruction verdict maps to a concrete code repair,
        and each mapped code projection independently passes safe-unit semantic
        validation.  Filenames and projector-supplied booleans are not proof.
        """

        evidence = item.metadata.get("cross_layer_causal_coverage_evidence")
        if not isinstance(evidence, dict):
            return False, "cross-layer causal coverage evidence is missing"
        required_fields = {
            "evidence_version",
            "instruction_final_verdicts",
            "instruction_material_actions",
            "material_action_to_repair_ids",
            "covered_code_items",
        }
        if not required_fields.issubset(evidence):
            return False, "cross-layer causal coverage evidence is incomplete"
        if evidence.get("evidence_version") != 2:
            return False, "cross-layer causal evidence version is unsupported"
        if len(covered_repair_ids) != len(set(covered_repair_ids)):
            return False, "covered code repair IDs contain duplicates"

        source_verdicts = item.metadata.get("source_final_verdicts")
        if not isinstance(source_verdicts, list) or any(
            not isinstance(value, dict) for value in source_verdicts
        ):
            return False, "instruction final verdict evidence is unavailable"
        confirmed_verdicts = [
            dict(value)
            for value in source_verdicts
            if value.get("label") == "overprivileged"
        ]
        if not confirmed_verdicts:
            return False, "instruction has no confirmed over-privileged verdict"
        if any(
            value.get("authorization_label") != "unauthorized"
            or value.get("necessity_label") == "unnecessary"
            or "unnecessary" in (value.get("overprivilege_reasons") or [])
            for value in confirmed_verdicts
        ):
            return (
                False,
                "instruction verdict is not exclusively grounded in an "
                "unauthorized realized action",
            )
        if evidence.get("instruction_final_verdicts") != confirmed_verdicts:
            return False, "instruction final-verdict evidence is stale"

        live_instruction_actions = self._material_actions_for_item(
            item,
            overprivileged_only=True,
        )
        if not live_instruction_actions:
            return False, "instruction has no grounded material actions"
        recorded_instruction_actions = self._material_action_set_from_evidence(
            evidence.get("instruction_material_actions")
        )
        if recorded_instruction_actions is None:
            return False, "instruction material-action evidence is malformed"
        if recorded_instruction_actions != live_instruction_actions:
            return False, "instruction material-action evidence is stale"

        covered_entries = evidence.get("covered_code_items")
        if not isinstance(covered_entries, list) or any(
            not isinstance(value, dict) for value in covered_entries
        ):
            return False, "covered code causal evidence is malformed"
        covered_by_id = {
            str(value.get("repair_id") or ""): value for value in covered_entries
        }
        if (
            "" in covered_by_id
            or len(covered_by_id) != len(covered_entries)
            or set(covered_by_id) != set(covered_repair_ids)
        ):
            return False, "covered code repair IDs do not match causal evidence"

        live_code_actions: dict[str, set[tuple[tuple[str, str], ...]]] = {}
        for repair_id in covered_repair_ids:
            code_item = items_by_repair_id.get(repair_id)
            recorded = covered_by_id.get(repair_id)
            integrity = integrity_by_repair_id.get(repair_id)
            if (
                code_item is None
                or code_item.layer != "code"
                or code_item.repair_type != "REORGANIZE_CODE_AND_ADD_DISPATCH"
                or recorded is None
                or integrity is None
            ):
                return False, "a covered code repair is absent or not concrete"
            if not bool(integrity.get("complete")) or not bool(
                integrity.get("safe_semantics_valid")
                or integrity.get("variant_semantics_valid")
            ):
                return (
                    False,
                    "a covered code projection has not independently proven its "
                    "safe-unit semantics",
                )

            actions = self._material_actions_for_item(
                code_item,
                overprivileged_only=False,
            )
            recorded_actions = self._material_action_set_from_evidence(
                recorded.get("material_actions")
            )
            if not actions or recorded_actions != actions:
                return False, "covered code material-action evidence is stale"

            live_neutralized = list(
                code_item.metadata.get("neutralized_repair_ids") or []
            )
            live_safe_unit = str(code_item.metadata.get("safe_execution_unit") or "")
            live_safe_hash = str(code_item.metadata.get("safe_unit_sha256") or "")
            live_dispatch_source = code_item.metadata.get("dispatch_source_file")
            live_invocation_template = code_item.metadata.get(
                "instruction_invocation_template"
            )
            live_dispatch_block = code_item.metadata.get("instruction_dispatch_block")
            live_routing_policy = code_item.metadata.get("instruction_routing_policy")
            live_entrypoint_policy = code_item.metadata.get("public_entrypoint_policy")
            if (
                repair_id not in {str(value) for value in live_neutralized}
                or not live_safe_unit
                or not live_safe_hash
                or code_item.metadata.get("safe_semantic_proof_complete") is not True
                or recorded.get("neutralized_repair_ids") != live_neutralized
                or recorded.get("safe_execution_unit") != live_safe_unit
                or recorded.get("safe_unit_sha256") != live_safe_hash
                or recorded.get("safe_semantic_proof_complete") is not True
                or recorded.get("dispatch_source_file") != live_dispatch_source
                or recorded.get("instruction_invocation_template")
                != live_invocation_template
                or recorded.get("instruction_dispatch_block") != live_dispatch_block
                or recorded.get("instruction_routing_policy") != live_routing_policy
                or recorded.get("public_entrypoint_policy") != live_entrypoint_policy
            ):
                return False, "covered code neutralization evidence is stale"
            live_code_actions[repair_id] = actions

        action_mapping = evidence.get("material_action_to_repair_ids")
        if not isinstance(action_mapping, list) or any(
            not isinstance(value, dict) for value in action_mapping
        ):
            return False, "material-action coverage mapping is malformed"
        recomputed_mapping: dict[
            tuple[tuple[str, str], ...], tuple[str, tuple[str, ...]]
        ] = {}
        for action in live_instruction_actions:
            matching_repairs = tuple(
                sorted(
                    repair_id
                    for repair_id, actions in live_code_actions.items()
                    if action in actions
                )
            )
            coverage_kind = "code_action_neutralization"
            if not matching_repairs:
                matching_repairs = tuple(
                    sorted(
                        repair_id
                        for repair_id in covered_repair_ids
                        if self._invocation_envelope_is_rewritten(
                            action=action,
                            code_item=items_by_repair_id[repair_id],
                        )
                    )
                )
                coverage_kind = "instruction_dispatch_rewrite"
            if not matching_repairs:
                return False, "an instruction material action is not neutralized"
            recomputed_mapping[action] = (coverage_kind, matching_repairs)

        recorded_mapping: dict[
            tuple[tuple[str, str], ...], tuple[str, tuple[str, ...]]
        ] = {}
        for mapping_entry in action_mapping:
            action = self._canonical_material_action(
                mapping_entry.get("material_action")
            )
            repair_ids = mapping_entry.get("repair_ids")
            if (
                action is None
                or action in recorded_mapping
                or not isinstance(repair_ids, list)
                or any(not isinstance(value, str) for value in repair_ids)
                or len(repair_ids) != len(set(repair_ids))
            ):
                return False, "material-action coverage mapping is malformed"
            coverage_kind = mapping_entry.get("coverage_kind")
            if coverage_kind not in {
                "code_action_neutralization",
                "instruction_dispatch_rewrite",
            }:
                return False, "material-action coverage kind is unsupported"
            recorded_mapping[action] = (
                str(coverage_kind),
                tuple(sorted(repair_ids)),
            )
        if recorded_mapping != recomputed_mapping:
            return False, "material-action coverage mapping is stale or incomplete"
        mapped_repair_ids = {
            repair_id
            for _, repair_ids in recomputed_mapping.values()
            for repair_id in repair_ids
        }
        if mapped_repair_ids != set(covered_repair_ids):
            return False, "causal evidence contains an unused code repair"
        return True, ""

    def _invocation_envelope_is_rewritten(
        self,
        *,
        action: tuple[tuple[str, str], ...],
        code_item: RepairItem,
    ) -> bool:
        """Recompute deny-by-default dispatch coverage for one script call."""

        action_fields = dict(action)
        if action_fields.get("operation") not in {
            "exec_command",
            "execute",
            "invoke",
            "run",
            "call",
        }:
            return False

        def normalize(value: object) -> str:
            return " ".join(str(value or "").casefold().split())

        source_file = normalize(code_item.source_file)
        dispatch_source = normalize(code_item.metadata.get("dispatch_source_file"))
        template = code_item.metadata.get("instruction_invocation_template")
        if (
            not source_file
            or dispatch_source != source_file
            or not isinstance(template, dict)
        ):
            return False
        source_token = normalize(template.get("source_token"))
        prefix_tokens = template.get("prefix_tokens")
        suffix_tokens = template.get("suffix_tokens")
        if (
            source_token != source_file
            or not isinstance(prefix_tokens, list)
            or any(not isinstance(token, str) for token in prefix_tokens)
            or not isinstance(suffix_tokens, list)
            or any(not isinstance(token, str) for token in suffix_tokens)
        ):
            return False
        object_tokens = {
            normalize(token.strip("'\"`.,;:()[]{}"))
            for token in action_fields.get("object", "").split()
        }
        if source_token not in object_tokens:
            return False

        dispatch_block = str(
            code_item.metadata.get("instruction_dispatch_block") or ""
        ).strip()
        allowed_unit = str(code_item.metadata.get("allowed_execution_unit") or "")
        safe_unit = str(code_item.metadata.get("safe_execution_unit") or "")
        return bool(
            dispatch_block
            and allowed_unit
            and safe_unit
            and allowed_unit in dispatch_block
            and safe_unit in dispatch_block
            and code_item.metadata.get("instruction_routing_policy")
            == "semantic_agent_selection_with_safe_default"
            and code_item.metadata.get("public_entrypoint_policy")
            == "safe_only_instruction_layer_selects_allowed_unit"
            and code_item.metadata.get("safe_semantic_proof_complete") is True
            and code_item.metadata.get("safe_unit_sha256")
        )

    def _material_actions_for_item(
        self,
        item: RepairItem,
        *,
        overprivileged_only: bool,
    ) -> set[tuple[tuple[str, str], ...]]:
        descriptor_contexts = item.metadata.get("descriptor_contexts")
        if not isinstance(descriptor_contexts, list):
            return set()
        actions: set[tuple[tuple[str, str], ...]] = set()
        for descriptor in descriptor_contexts:
            if not isinstance(descriptor, dict):
                continue
            if (
                overprivileged_only
                and descriptor.get("final_verdict") != "overprivileged"
            ):
                continue
            material_actions = descriptor.get("material_action_instances")
            if not isinstance(material_actions, list):
                continue
            for raw_action in material_actions:
                action = self._canonical_material_action(raw_action)
                if action is not None:
                    actions.add(action)
        return actions

    def _material_action_set_from_evidence(
        self,
        value: object,
    ) -> set[tuple[tuple[str, str], ...]] | None:
        if not isinstance(value, list):
            return None
        actions: list[tuple[tuple[str, str], ...]] = []
        for raw_action in value:
            action = self._canonical_material_action(raw_action)
            if action is None:
                return None
            actions.append(action)
        if len(actions) != len(set(actions)):
            return None
        return set(actions)

    def _canonical_material_action(
        self,
        value: object,
    ) -> tuple[tuple[str, str], ...] | None:
        if not isinstance(value, dict):
            return None
        action = tuple(
            (
                field_name,
                " ".join(str(value.get(field_name) or "").casefold().split()),
            )
            for field_name in (
                "operation",
                "object",
                "source",
                "scope",
                "destination",
                "side_effect",
            )
        )
        return action if action[0][1] else None

    def _composite_projection_integrity(
        self,
        *,
        root: Path,
        item: RepairItem,
        peer_items: list[RepairItem],
        manifest: list[object],
        instruction_text: str,
        graph_source_files: set[str],
        called_resources: set[str],
        public_safe_only: bool,
    ) -> dict[str, object]:
        """Validate every independently guarded same-source action variant."""

        variants = [value for value in manifest if isinstance(value, dict)]
        peer_ids = sorted(peer.repair_id for peer in peer_items)
        peer_id_set = set(peer_ids)
        expected_subsets = {
            frozenset(
                peer_ids[index] for index in range(len(peer_ids)) if mask & (1 << index)
            )
            for mask in range(1 << len(peer_ids))
        }
        observed_subsets: set[frozenset[str]] = set()
        paths: list[str] = []
        paths_unique = True
        path_hashes_valid = True
        variant_contracts_valid = len(variants) == len(manifest)
        neutralization_contracts_valid = True
        all_variants_complete = True
        safe_variants: list[dict[str, object]] = []
        all_allowed_variants: list[dict[str, object]] = []
        singleton_variant: dict[str, object] | None = None
        actual_hashes: dict[str, str] = {}

        for variant in variants:
            allowed_ids = {
                str(value) for value in variant.get("allowed_repair_ids", [])
            }
            blocked_ids = {
                str(value) for value in variant.get("blocked_repair_ids", [])
            }
            neutralized_ids = {
                str(value) for value in variant.get("neutralized_repair_ids", [])
            }
            unlocalized_ids = {
                str(value) for value in variant.get("unlocalized_repair_ids", [])
            }
            observed_subsets.add(frozenset(allowed_ids))
            contract_valid = (
                allowed_ids <= peer_id_set
                and blocked_ids <= peer_id_set
                and allowed_ids.isdisjoint(blocked_ids)
                and allowed_ids | blocked_ids == peer_id_set
            )
            variant_contracts_valid = variant_contracts_valid and contract_valid
            variant_complete = variant.get("complete") is True
            all_variants_complete = all_variants_complete and variant_complete
            neutralization_contracts_valid = (
                neutralization_contracts_valid
                and neutralized_ids == blocked_ids
                and not unlocalized_ids
            )

            relative_path = str(variant.get("relative_path") or "")
            if not relative_path or relative_path in paths:
                paths_unique = False
            paths.append(relative_path)
            target = self._rooted_file(root, relative_path)
            expected_hash = str(variant.get("sha256") or "")
            if target is None or not expected_hash:
                path_hashes_valid = False
            else:
                actual_hash = self._file_sha256(target)
                actual_hashes[relative_path] = actual_hash
                path_hashes_valid = path_hashes_valid and actual_hash == expected_hash

            if variant.get("safe_default") is True:
                safe_variants.append(variant)
            if variant.get("all_allowed") is True:
                all_allowed_variants.append(variant)
            if allowed_ids == {item.repair_id}:
                singleton_variant = variant

        lattice_complete = observed_subsets == expected_subsets
        safe_variant_valid = bool(
            len(safe_variants) == 1
            and not safe_variants[0].get("allowed_repair_ids")
            and set(safe_variants[0].get("blocked_repair_ids", [])) == peer_id_set
        )
        all_allowed_variant_valid = bool(
            len(all_allowed_variants) == 1
            and set(all_allowed_variants[0].get("allowed_repair_ids", []))
            == peer_id_set
            and not all_allowed_variants[0].get("blocked_repair_ids")
        )
        singleton_path = (
            str(singleton_variant.get("relative_path") or "")
            if singleton_variant is not None
            else ""
        )
        safe_path = (
            str(safe_variants[0].get("relative_path") or "")
            if len(safe_variants) == 1
            else ""
        )
        candidate_units_valid = bool(
            singleton_path
            and singleton_path == str(item.metadata.get("allowed_execution_unit") or "")
            and safe_path == str(item.metadata.get("safe_execution_unit") or "")
            and actual_hashes.get(singleton_path)
            == str(item.metadata.get("allowed_unit_sha256") or "")
            and actual_hashes.get(safe_path)
            == str(item.metadata.get("safe_unit_sha256") or "")
            and {
                str(variant.get("relative_path") or "")
                for variant in variants
                if item.repair_id
                in {str(value) for value in variant.get("allowed_repair_ids", [])}
            }
            == {
                str(value) for value in item.metadata.get("allowed_execution_units", [])
            }
            and {
                str(variant.get("relative_path") or "")
                for variant in variants
                if item.repair_id
                in {str(value) for value in variant.get("blocked_repair_ids", [])}
            }
            == {
                str(value) for value in item.metadata.get("blocked_execution_units", [])
            }
        )
        generated_files_complete = set(item.generated_files) == set(paths)

        normalized_paths = {
            self._normalized_relative_path(value) for value in paths
        } - {""}
        graph_units_present = bool(
            len(normalized_paths) == len(paths)
            and normalized_paths.issubset(graph_source_files)
        )
        graph_dispatch_complete = bool(
            normalized_paths and normalized_paths.issubset(called_resources)
        )
        dispatch_block = str(item.metadata.get("instruction_dispatch_block") or "")
        guard_specs = item.metadata.get("composite_guard_specs")
        guards_by_repair_id = (
            {
                str(value.get("repair_id") or ""): str(
                    value.get("guard_condition") or ""
                )
                for value in guard_specs
                if isinstance(value, dict)
            }
            if isinstance(guard_specs, list)
            else {}
        )
        expected_guards = {
            peer.repair_id: str(peer.guard_condition or "") for peer in peer_items
        }
        guard_specs_valid = bool(
            guards_by_repair_id == expected_guards
            and all(value for value in expected_guards.values())
        )
        instruction_dispatch_complete = bool(
            dispatch_block
            and dispatch_block in instruction_text
            and all(path in dispatch_block for path in paths)
            and guard_specs_valid
            and all(guard in dispatch_block for guard in expected_guards.values())
        )

        manifest_payload = {
            "source_file": str(item.source_file or ""),
            "repair_order": list(item.metadata.get("source_variant_repair_order", [])),
            "variants": variants,
        }
        observed_manifest_hash = hashlib.sha256(
            json.dumps(
                manifest_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        manifest_hash_valid = bool(
            item.metadata.get("source_variant_manifest_sha256")
            and observed_manifest_hash
            == item.metadata.get("source_variant_manifest_sha256")
            and list(item.metadata.get("source_variant_repair_order", [])) == peer_ids
        )

        original_hash = str(item.metadata.get("original_source_sha256") or "")
        full_allowed_path = (
            str(all_allowed_variants[0].get("relative_path") or "")
            if len(all_allowed_variants) == 1
            else ""
        )
        all_allowed_semantics_preserved = bool(
            original_hash
            and full_allowed_path
            and actual_hashes.get(full_allowed_path) == original_hash
        )

        # Rebuild every projected source variant independently.  Manifest
        # booleans and hashes establish provenance, but do not prove that an
        # LLM has not replaced a blocked action with an equivalent API or a new
        # command.  Each variant must statically equal the original all-allowed
        # action graph minus exactly its declared blocked actions.
        variant_semantics_valid = False
        semantic_variant_status: list[dict[str, object]] = []
        full_allowed_target = self._rooted_file(root, full_allowed_path)
        peer_by_id = {peer.repair_id: peer for peer in peer_items}
        if full_allowed_target is not None:
            original_source = full_allowed_target.read_text(encoding="utf-8")
            semantic_variant_status = []
            for variant in variants:
                relative_path = str(variant.get("relative_path") or "")
                target = self._rooted_file(root, relative_path)
                blocked_ids = [
                    str(value) for value in variant.get("blocked_repair_ids", [])
                ]
                blocked_items = [
                    peer_by_id[repair_id]
                    for repair_id in blocked_ids
                    if repair_id in peer_by_id
                ]
                if target is None or len(blocked_items) != len(blocked_ids):
                    status: dict[str, object] = {
                        "complete": False,
                        "reason": (
                            "variant file or grounded blocked repair item was missing"
                        ),
                    }
                else:
                    status = self.semantic_analyzer.compare_projection(
                        original_source=original_source,
                        projected_source=target.read_text(encoding="utf-8"),
                        suffix=full_allowed_target.suffix,
                        filename=target.name,
                        blocked_items=blocked_items,
                    )
                semantic_variant_status.append(
                    {
                        "relative_path": relative_path,
                        "complete": bool(status["complete"]),
                        "reason": str(status.get("reason") or ""),
                        "unexpected_privilege_semantics": list(
                            status.get("unexpected_privilege_semantics", [])
                        ),
                    }
                )
            variant_semantics_valid = bool(semantic_variant_status) and all(
                bool(value["complete"]) for value in semantic_variant_status
            )

        safe_targets_absent = False
        safe_target = self._rooted_file(root, safe_path)
        if safe_target is not None:
            safe_source = safe_target.read_text(encoding="utf-8")
            raw_targets = [
                str(peer.raw_text or "").strip()
                for peer in peer_items
                if str(peer.raw_text or "").strip()
            ]
            safe_targets_absent = bool(raw_targets) and all(
                raw_target not in safe_source for raw_target in raw_targets
            )
            if not raw_targets:
                safe_targets_absent = neutralization_contracts_valid

        complete = all(
            (
                public_safe_only,
                paths_unique,
                path_hashes_valid,
                variant_contracts_valid,
                neutralization_contracts_valid,
                all_variants_complete,
                item.metadata.get("composite_projection_complete") is True,
                lattice_complete,
                safe_variant_valid,
                all_allowed_variant_valid,
                candidate_units_valid,
                generated_files_complete,
                graph_units_present,
                graph_dispatch_complete,
                instruction_dispatch_complete,
                guard_specs_valid,
                manifest_hash_valid,
                all_allowed_semantics_preserved,
                variant_semantics_valid,
            )
        )
        return {
            "source_file": str(item.source_file or ""),
            "public_entrypoint_safe_only": public_safe_only,
            "variant_count": len(variants),
            "expected_variant_count": 1 << len(peer_ids),
            "variant_paths_unique": paths_unique,
            "variant_hashes_valid": path_hashes_valid,
            "variant_contracts_valid": variant_contracts_valid,
            "neutralization_complete": neutralization_contracts_valid,
            "all_variants_complete": all_variants_complete,
            "privilege_lattice_complete": lattice_complete,
            "safe_variant_valid": safe_variant_valid,
            "all_allowed_variant_valid": all_allowed_variant_valid,
            "candidate_independent_unit_valid": candidate_units_valid,
            "generated_files_complete": generated_files_complete,
            "graph_units_present": graph_units_present,
            "graph_dispatch_complete": graph_dispatch_complete,
            "instruction_dispatch_complete": instruction_dispatch_complete,
            "independent_guard_specs_valid": guard_specs_valid,
            "manifest_hash_valid": manifest_hash_valid,
            "allowed_semantics_preserved": (all_allowed_semantics_preserved),
            "safe_targets_absent": safe_targets_absent,
            "variant_semantics_valid": variant_semantics_valid,
            "semantic_variant_status": semantic_variant_status,
            "complete": complete,
            "reason": (
                "every action has an independent semantic condition; the "
                "complete privilege lattice, source hashes, safe default, and "
                "instruction/graph dispatch all match"
                if complete
                else "one or more independent projection integrity checks failed"
            ),
        }

    def _candidate_id_for_item(self, item: RepairItem) -> str:
        source_candidate_id = str(item.metadata.get("source_candidate_id") or "")
        if source_candidate_id:
            return source_candidate_id
        if item.overreach_id.startswith("overreach-"):
            return "candidate-" + item.overreach_id[len("overreach-") :]
        if item.overreach_id.startswith("overreach_"):
            return "candidate_" + item.overreach_id[len("overreach_") :]
        return item.overreach_id

    def _rooted_file(self, root: Path, relative_path: str) -> Path | None:
        if not relative_path:
            return None
        target = (root / relative_path).resolve()
        if (target != root and root not in target.parents) or not target.is_file():
            return None
        return target

    def _normalized_relative_path(self, value: str) -> str:
        return value.replace("\\", "/").removeprefix("./")

    def _file_sha256(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _public_entrypoint_status(
        self,
        *,
        patched_bundle_root: Path,
        repair_plan: RepairPlan,
    ) -> dict[str, object]:
        root = patched_bundle_root.resolve()
        entries: list[dict[str, object]] = []
        for item in repair_plan.items:
            if item.repair_type != "REORGANIZE_CODE_AND_ADD_DISPATCH":
                continue
            source_file = item.source_file
            allowed_unit = str(item.metadata.get("allowed_execution_unit") or "")
            safe_unit = str(item.metadata.get("safe_execution_unit") or "")
            entry: dict[str, object] = {
                "repair_id": item.repair_id,
                "source_file": source_file,
                "allowed_execution_unit": allowed_unit,
                "safe_execution_unit": safe_unit,
                "safe_only": False,
            }
            if not source_file or not allowed_unit or not safe_unit:
                entry["reason"] = "missing projected execution-unit metadata"
                entries.append(entry)
                continue
            source_path = (root / source_file).resolve()
            if (
                source_path != root and root not in source_path.parents
            ) or not source_path.is_file():
                entry["reason"] = "public entrypoint was missing or escaped root"
                entries.append(entry)
                continue
            source = source_path.read_text(encoding="utf-8")
            safe_name = Path(safe_unit).name
            allowed_name = Path(allowed_unit).name
            expected_entrypoint_hash = str(
                item.metadata.get("public_entrypoint_sha256") or ""
            )
            entrypoint_hash_valid = bool(
                expected_entrypoint_hash
                and self._file_sha256(source_path) == expected_entrypoint_hash
            )
            forbidden_routing = (
                "SKILLSCOPE_TASK_CLUSTER",
                "SKILLSCOPE_USER_PROMPT",
            )
            safe_only = (
                safe_name in source
                and allowed_name not in source
                and all(marker not in source for marker in forbidden_routing)
                and entrypoint_hash_valid
            )
            entry["safe_only"] = safe_only
            entry["entrypoint_hash_valid"] = entrypoint_hash_valid
            entry["reason"] = (
                "public entrypoint matches the projected safe-only artifact, "
                "references only the safe unit, and uses no routing environment "
                "input"
                if safe_only
                else "public entrypoint did not satisfy the safe-only contract"
            )
            entries.append(entry)
        return {
            "all_safe_only": all(bool(entry["safe_only"]) for entry in entries),
            "entrypoints": entries,
        }

    def _semantically_contains(self, haystack: list[str], needle: str) -> bool:
        normalized_needle = self._normalize(needle)
        if not normalized_needle:
            return True
        needle_tokens = set(normalized_needle.split())
        for entry in haystack:
            normalized_entry = self._normalize(entry)
            if (
                normalized_needle == normalized_entry
                or normalized_needle in normalized_entry
            ):
                return True
            if (
                needle_tokens
                and len(needle_tokens & set(normalized_entry.split()))
                / len(needle_tokens)
                >= 0.8
            ):
                return True
        return False

    def _is_boundary_summary(self, summary: str) -> bool:
        normalized = self._normalize(summary)
        return normalized.startswith(
            ("enter ", "exit ", "return from ", "start ", "end ")
        )

    def _normalize(self, value: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.casefold())).strip()

    def _overreach_signature(
        self, layer: str, source_file: str | None, summary: str
    ) -> str:
        return f"{layer}:{source_file or '-'}:{summary}"

    def _overreach_id(self, candidate_id: str) -> str:
        if candidate_id.startswith("candidate-"):
            return "overreach-" + candidate_id[len("candidate-") :]
        if candidate_id.startswith("candidate_"):
            return "overreach_" + candidate_id[len("candidate_") :]
        return f"overreach-{candidate_id}"
