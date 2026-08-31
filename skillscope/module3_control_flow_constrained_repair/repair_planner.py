from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from skillscope.common.llm import (
    DisabledLLMClient,
    JSONResponseContract,
    MAX_VALIDATED_LLM_ATTEMPTS,
    PromptAssetLoader,
    StructuredLLMClient,
    complete_validated_json,
)
from skillscope.common.models import (
    ActionTaskDescriptor,
    CandidateAction,
    CandidateExtractionResult,
    FinalVerdict,
    RepairItem,
    RepairPlan,
    ValidationResult,
)

from .descriptor_clustering import DescriptorCluster, DescriptorClusterer
from .naming import overreach_id_from_candidate_id
from .overreach_pruner import OverreachPruner
from .task_conditioned_guarder import TaskConditionedGuarder


class RepairPlanner:
    """Plan only from final action-task verdicts and fixed descriptors."""

    _PLANNER_OUTPUT_KEYS = {
        "repair_type",
        "descriptor_ids",
        "allowed_cluster_keys",
        "blocked_cluster_keys",
        "rationale",
        "guard_condition",
        "notes",
    }

    def __init__(
        self,
        *,
        llm_client: StructuredLLMClient | None = None,
        prompt_loader: PromptAssetLoader | None = None,
        prompt_asset: str = "prompts/module3_repair_planning.md",
        pruner: OverreachPruner | None = None,
        guarder: TaskConditionedGuarder | None = None,
        clusterer: DescriptorClusterer | None = None,
    ) -> None:
        self.llm_client = llm_client or DisabledLLMClient()
        self.prompt_loader = prompt_loader
        self.prompt_asset = prompt_asset
        # Retained for API compatibility. Permanent pruning is intentionally
        # excluded from the Module 3 planning policy.
        self.pruner = pruner or OverreachPruner()
        self.guarder = guarder or TaskConditionedGuarder()
        self.clusterer = clusterer or DescriptorClusterer(
            llm_client=self.llm_client,
            prompt_loader=self.prompt_loader,
        )

    def plan(
        self,
        *,
        analysis: CandidateExtractionResult,
        validation: ValidationResult,
    ) -> RepairPlan:
        descriptors_by_candidate: dict[str, list[ActionTaskDescriptor]] = defaultdict(
            list
        )
        verdicts_by_candidate: dict[str, list[FinalVerdict]] = defaultdict(list)
        for descriptor in validation.descriptors:
            descriptors_by_candidate[descriptor.candidate_id].append(descriptor)
        for verdict in validation.final_verdicts:
            verdicts_by_candidate[verdict.candidate_id].append(verdict)

        task_summary_by_id = {
            task.task_id: (task.task_summary or task.prompt)
            for task in validation.tasks
        }
        instruction_file = (
            analysis.bundle.instruction_files[0].relative_path
            if analysis.bundle.instruction_files
            else None
        )
        items: list[RepairItem] = []
        skipped: list[str] = []
        normalization_skips: list[str] = []
        normalized_descriptor_count = 0
        cluster_count = 0

        for candidate in sorted(
            analysis.candidates, key=lambda value: value.candidate_id
        ):
            overreach_id = overreach_id_from_candidate_id(candidate.candidate_id)
            candidate_descriptors = descriptors_by_candidate.get(
                candidate.candidate_id, []
            )
            candidate_verdicts = verdicts_by_candidate.get(
                candidate.candidate_id, []
            )
            if not candidate_verdicts:
                skipped.append(
                    f"{overreach_id}: no final action-task verdict was available; "
                    "candidate extraction or necessity evidence alone cannot "
                    "confirm over-privilege."
                )
                continue
            if not candidate_descriptors:
                skipped.append(
                    f"{overreach_id}: final verdicts existed but no fixed-schema "
                    "action-task descriptor was available."
                )
                continue

            normalized_descriptors, clusters, descriptor_skips = self.clusterer.cluster(
                candidate_descriptors,
                final_verdicts=candidate_verdicts,
            )
            normalized_descriptor_count += len(normalized_descriptors)
            cluster_count += len(clusters)
            normalization_skips.extend(
                f"{overreach_id}: {reason}" for reason in descriptor_skips
            )
            blocked_clusters = [
                cluster for cluster in clusters if cluster.disposition == "blocked"
            ]
            if not blocked_clusters:
                inconclusive_count = sum(
                    cluster.disposition == "inconclusive" for cluster in clusters
                )
                skipped.append(
                    f"{overreach_id}: no descriptor cluster had a confirmed "
                    "over-privileged final verdict "
                    f"(inconclusive_clusters={inconclusive_count})."
                )
                continue

            node = analysis.ueg.node_by_id(candidate.node_id)
            item = self.guarder.build_item(
                candidate=candidate,
                node=node,
                profile=analysis.profile,
                descriptors=normalized_descriptors,
                clusters=clusters,
                instruction_file=instruction_file,
                task_summary_by_id=task_summary_by_id,
            )
            item.metadata["source_final_verdicts"] = [
                {
                    "candidate_id": verdict.candidate_id,
                    "task_id": verdict.task_id,
                    "label": verdict.label,
                    "authorization_label": verdict.authorization_label,
                    "necessity_label": verdict.necessity_label,
                    "overprivilege_reasons": list(
                        verdict.overprivilege_reasons
                    ),
                    "privilege_type": verdict.privilege_type,
                    "privilege_relevant": verdict.privilege_relevant,
                }
                for verdict in sorted(
                    candidate_verdicts,
                    key=lambda value: value.task_id,
                )
            ]
            self._annotate_with_llm_if_valid(
                analysis=analysis,
                candidate=candidate,
                node=node,
                item=item,
                clusters=clusters,
            )
            items.append(item)

        confirmed_overreach_ids = sorted(item.overreach_id for item in items)
        summary = (
            f"Planned {len(items)} guard-first repair items from confirmed "
            "final action-task verdicts."
            if items
            else "No repair item was planned because no descriptor-backed final "
            "verdict confirmed over-privilege."
        )
        return RepairPlan(
            skill_id=analysis.bundle.bundle_id,
            items=items,
            summary=summary,
            metadata={
                "planned_item_count": len(items),
                "candidate_count": len(analysis.candidates),
                "confirmed_overreach_count": len(items),
                "confirmed_overreach_ids": confirmed_overreach_ids,
                "skipped_overreaches": skipped,
                "descriptor_normalization_skips": normalization_skips,
                "normalized_descriptor_count": normalized_descriptor_count,
                "descriptor_cluster_count": cluster_count,
                "planning_strategy": (
                    "final_verdict_descriptor_clusters_guard_first"
                ),
                "permanent_pruning_enabled": False,
            },
        )

    def _annotate_with_llm_if_valid(
        self,
        *,
        analysis: CandidateExtractionResult,
        candidate: CandidateAction,
        node: object | None,
        item: RepairItem,
        clusters: list[DescriptorCluster],
    ) -> None:
        item.metadata["planner_strategy"] = "deterministic_guard_first"
        if self.prompt_loader is None:
            return

        payload = {
            "skill_profile": self._skill_profile_payload(analysis),
            "overreach": {
                "overreach_id": item.overreach_id,
                "source_candidate_id": candidate.candidate_id,
                "node_id": candidate.node_id,
                "layer": candidate.layer,
                "summary": candidate.summary,
                "source_file": candidate.source_file,
                "risk_tags": list(candidate.risk_tags),
                "reason": candidate.reason,
            },
            "overreach_node": self._node_payload(node),
            "descriptor_ids": list(item.descriptor_ids),
            "descriptor_clusters": [
                cluster.as_payload()
                for cluster in sorted(
                    clusters, key=lambda value: value.cluster_key
                )
            ],
            "required_repair_type": item.repair_type,
            "required_guard_condition": item.guard_condition,
            "allowed_cluster_keys": list(item.allowed_cluster_keys),
            "blocked_cluster_keys": list(item.blocked_cluster_keys),
            "policy": {
                "confirmation_source": "final_verdicts_plus_descriptors_only",
                "allowed_context": "authorized_and_necessary",
                "blocked_context": "unauthorized_or_unnecessary",
                "permanent_pruning": "forbidden",
            },
        }
        try:
            response = complete_validated_json(
                self.llm_client,
                system_prompt=self.prompt_loader.load(self.prompt_asset),
                user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
                schema_name="module3_repair_planning",
                contract=JSONResponseContract(
                    required_fields=tuple(sorted(self._PLANNER_OUTPUT_KEYS)),
                    non_empty_string_fields=(
                        "repair_type",
                        "rationale",
                        "guard_condition",
                    ),
                    enum_fields={
                        "repair_type": {item.repair_type.lower()},
                    },
                    evidence_field="descriptor_ids",
                    grounded_evidence_ids=set(item.descriptor_ids),
                    consistency_checks=(
                        lambda value: self._validate_llm_response(value, item),
                    ),
                ),
                max_attempts=MAX_VALIDATED_LLM_ATTEMPTS,
            )
        except RuntimeError as exc:
            item.metadata["planner_llm_validation_error"] = str(exc)
            return

        item.rationale = str(response["rationale"]).strip()
        item.metadata["planner_strategy"] = "llm_strictly_validated"
        item.metadata["planner_notes"] = list(response["notes"])

    def _validate_llm_response(
        self,
        response: dict[str, Any],
        item: RepairItem,
    ) -> str | None:
        if set(response) != self._PLANNER_OUTPUT_KEYS:
            return (
                "planner response keys did not exactly match the required schema"
            )
        repair_type = response.get("repair_type")
        if (
            not isinstance(repair_type, str)
            or repair_type.strip().upper() != item.repair_type.upper()
        ):
            return "planner attempted to change the required guard-first repair type"
        if not self._is_exact_string_list(
            response.get("descriptor_ids"), item.descriptor_ids
        ):
            return "planner descriptor_ids did not exactly match validated evidence"
        if not self._is_exact_string_list(
            response.get("allowed_cluster_keys"), item.allowed_cluster_keys
        ):
            return "planner allowed_cluster_keys did not exactly match validated evidence"
        if not self._is_exact_string_list(
            response.get("blocked_cluster_keys"), item.blocked_cluster_keys
        ):
            return "planner blocked_cluster_keys did not exactly match validated evidence"
        rationale = response.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            return "planner rationale must be a non-empty string"
        if response.get("guard_condition") != item.guard_condition:
            return "planner attempted to broaden or replace the deterministic guard"
        notes = response.get("notes")
        if not isinstance(notes, list) or any(
            not isinstance(note, str) for note in notes
        ):
            return "planner notes must be a list of strings"
        return None

    def _is_exact_string_list(
        self, value: object, expected: list[str]
    ) -> bool:
        return (
            isinstance(value, list)
            and all(isinstance(entry, str) for entry in value)
            and value == expected
        )

    def _skill_profile_payload(
        self, analysis: CandidateExtractionResult
    ) -> dict[str, Any]:
        return {
            "name": analysis.profile.name,
            "description": analysis.profile.description,
            "use_when": analysis.profile.use_when,
            "summary": analysis.profile.summary,
            "declared_capabilities": analysis.profile.declared_capabilities,
            "declared_outputs": analysis.profile.declared_outputs,
        }

    def _node_payload(self, node: object | None) -> dict[str, Any] | None:
        if node is None:
            return None
        source_range = getattr(node, "source_range", None)
        return {
            "node_id": getattr(node, "node_id", None),
            "layer": getattr(node, "layer", None),
            "node_type": getattr(node, "node_type", None),
            "summary": getattr(node, "summary", None),
            "source_file": getattr(node, "source_file", None),
            "source_range": (
                {
                    "start_line": source_range.start_line,
                    "end_line": source_range.end_line,
                    "start_column": source_range.start_column,
                    "end_column": source_range.end_column,
                }
                if source_range is not None
                else None
            ),
            "raw_text": getattr(node, "raw_text", None),
            "operation_type": getattr(node, "operation_type", None),
            "risk_tags": list(getattr(node, "risk_tags", []) or []),
        }
