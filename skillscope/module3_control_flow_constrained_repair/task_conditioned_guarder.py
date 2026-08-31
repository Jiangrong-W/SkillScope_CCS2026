from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, ClassVar

from skillscope.common.llm import (
    MAX_VALIDATED_LLM_ATTEMPTS,
    DisabledLLMClient,
    JSONResponseContract,
    PromptAssetLoader,
    StructuredLLMClient,
    complete_validated_json,
)
from skillscope.common.models import (
    ActionTaskDescriptor,
    CandidateAction,
    RepairItem,
    SkillProfile,
    UEGNode,
)

from .descriptor_clustering import DescriptorCluster
from .naming import overreach_id_from_candidate_id


class TaskConditionedGuarder:
    """Build guard-first repairs from normalized descriptor clusters."""

    _GUARD_OUTPUT_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "descriptor_ids",
            "allowed_cluster_keys",
            "blocked_cluster_keys",
            "evidence_refs",
            "deny_by_default",
            "canonical_guard_sha256",
            "rationale",
        }
    )

    def __init__(
        self,
        *,
        llm_client: StructuredLLMClient | None = None,
        prompt_loader: PromptAssetLoader | None = None,
        prompt_asset: str = "prompts/module3_task_conditioned_guard.md",
    ) -> None:
        self.llm_client = llm_client or DisabledLLMClient()
        self.prompt_loader = prompt_loader
        self.prompt_asset = prompt_asset

    def build_item(
        self,
        *,
        candidate: CandidateAction,
        node: UEGNode | None,
        profile: SkillProfile,
        descriptors: list[ActionTaskDescriptor],
        clusters: list[DescriptorCluster],
        instruction_file: str | None,
        task_summary_by_id: dict[str, str] | None = None,
    ) -> RepairItem:
        allowed_clusters = sorted(
            (cluster for cluster in clusters if cluster.disposition == "allowed"),
            key=lambda cluster: cluster.cluster_key,
        )
        blocked_clusters = sorted(
            (cluster for cluster in clusters if cluster.disposition == "blocked"),
            key=lambda cluster: cluster.cluster_key,
        )
        guard_payload = self._synthesize_guard(
            candidate=candidate,
            allowed_clusters=allowed_clusters,
            blocked_clusters=blocked_clusters,
        )

        source_start_line = (
            node.source_range.start_line
            if node is not None and node.source_range is not None
            else None
        )
        source_end_line = (
            node.source_range.end_line
            if node is not None and node.source_range is not None
            else None
        )
        source_start_column = (
            (
                node.source_range.start_column
                if node.source_range is not None
                and node.source_range.start_column is not None
                else self._optional_int(node.attributes.get("col_offset"))
            )
            if node is not None
            else None
        )
        source_end_column = (
            (
                node.source_range.end_column
                if node.source_range is not None
                and node.source_range.end_column is not None
                else self._optional_int(
                    node.attributes.get("end_col_offset")
                )
            )
            if node is not None
            else None
        )
        target_files = [path for path in [candidate.source_file, instruction_file] if path]
        task_summary_by_id = task_summary_by_id or {}
        allowed_task_ids = {
            task_id for cluster in allowed_clusters for task_id in cluster.task_ids
        }
        blocked_task_ids = {
            task_id for cluster in blocked_clusters for task_id in cluster.task_ids
        }

        if candidate.layer == "instruction":
            repair_type = "GUARD_INSTRUCTION_TASK_CONDITIONED"
            projection_metadata = {
                "guard_instruction_text": guard_payload["guard_instruction_text"],
            }
        else:
            repair_type = "REORGANIZE_CODE_AND_ADD_DISPATCH"
            projection_metadata = {
                "dispatch_instruction_text": guard_payload[
                    "dispatch_instruction_text"
                ],
            }

        cluster_payloads = [cluster.as_payload() for cluster in clusters]
        descriptor_payloads = [
            self._descriptor_payload(descriptor)
            for descriptor in sorted(
                descriptors, key=lambda value: value.descriptor_id
            )
        ]
        descriptor_ids = sorted(
            {
                descriptor_id
                for cluster in allowed_clusters + blocked_clusters
                for descriptor_id in cluster.descriptor_ids
            }
        )
        return RepairItem(
            repair_id=f"repair-{overreach_id_from_candidate_id(candidate.candidate_id)}",
            overreach_id=overreach_id_from_candidate_id(candidate.candidate_id),
            node_id=candidate.node_id,
            layer=candidate.layer,
            repair_type=repair_type,
            target_files=sorted(set(target_files)),
            rationale=(
                "A final action-task verdict confirmed over-privilege in at least "
                "one descriptor cluster. The action is therefore retained behind "
                "a deny-by-default, task-conditioned guard instead of being "
                "permanently removed from a finite representative task sample."
            ),
            overreach_summary=candidate.summary,
            guard_condition=guard_payload["guard_condition"],
            allowed_task_summaries=self._task_summaries(
                allowed_task_ids, task_summary_by_id
            ),
            blocked_task_summaries=self._task_summaries(
                blocked_task_ids, task_summary_by_id
            ),
            source_file=node.source_file if node is not None else candidate.source_file,
            source_start_line=source_start_line,
            source_end_line=source_end_line,
            source_start_column=source_start_column,
            source_end_column=source_end_column,
            raw_text=node.raw_text if node is not None else None,
            descriptor_ids=descriptor_ids,
            allowed_cluster_keys=[
                cluster.cluster_key for cluster in allowed_clusters
            ],
            blocked_cluster_keys=[
                cluster.cluster_key for cluster in blocked_clusters
            ],
            metadata={
                "repair_strategy": "guard_first_descriptor_cluster_control",
                "guard_policy": "allow_only_authorized_and_necessary_clusters",
                "source_candidate_id": candidate.candidate_id,
                "candidate_semantics": {
                    "summary": candidate.summary,
                    "operation_type": (
                        node.operation_type if node is not None else None
                    ),
                    "object_ref": (
                        node.object_ref if node is not None else None
                    ),
                },
                "skill_profile": self._profile_payload(profile),
                "descriptor_contexts": descriptor_payloads,
                "descriptor_clusters": cluster_payloads,
                "allowed_descriptor_clusters": [
                    cluster.as_payload() for cluster in allowed_clusters
                ],
                "blocked_descriptor_clusters": [
                    cluster.as_payload() for cluster in blocked_clusters
                ],
                "guard_synthesis_strategy": guard_payload[
                    "guard_synthesis_strategy"
                ],
                "guard_llm_validation_error": guard_payload.get(
                    "guard_llm_validation_error"
                ),
                "canonical_guard_sha256": guard_payload.get(
                    "canonical_guard_sha256"
                ),
                "routing_layer": "instruction",
            }
            | projection_metadata,
        )

    def _optional_int(self, value: object) -> int | None:
        return value if isinstance(value, int) else None

    def _synthesize_guard(
        self,
        *,
        candidate: CandidateAction,
        allowed_clusters: list[DescriptorCluster],
        blocked_clusters: list[DescriptorCluster],
    ) -> dict[str, Any]:
        fallback = self._synthesize_guard_deterministically(
            candidate=candidate,
            allowed_clusters=allowed_clusters,
            blocked_clusters=blocked_clusters,
        )
        if self.prompt_loader is None:
            return fallback

        descriptor_ids = sorted(
            {
                descriptor_id
                for cluster in allowed_clusters + blocked_clusters
                for descriptor_id in cluster.descriptor_ids
            }
        )
        allowed_cluster_keys = [
            cluster.cluster_key for cluster in allowed_clusters
        ]
        blocked_cluster_keys = [
            cluster.cluster_key for cluster in blocked_clusters
        ]
        evidence_refs = descriptor_ids + allowed_cluster_keys + blocked_cluster_keys
        allowed_clauses = [
            self._semantic_clause(cluster) for cluster in allowed_clusters
        ]
        blocked_clauses = [
            self._semantic_clause(cluster) for cluster in blocked_clusters
        ]
        payload = {
            "candidate": {
                "candidate_id": candidate.candidate_id,
                "summary": candidate.summary,
                "layer": candidate.layer,
                "source_file": candidate.source_file,
                "risk_tags": list(candidate.risk_tags),
            },
            "descriptor_ids": descriptor_ids,
            "allowed_cluster_keys": allowed_cluster_keys,
            "blocked_cluster_keys": blocked_cluster_keys,
            "required_evidence_refs": evidence_refs,
            "allowed_descriptor_clusters": [
                cluster.as_payload() for cluster in allowed_clusters
            ],
            "blocked_descriptor_clusters": [
                cluster.as_payload() for cluster in blocked_clusters
            ],
            "required_allowed_semantic_clauses": allowed_clauses,
            "required_blocked_semantic_clauses": blocked_clauses,
            "required_guard_condition": fallback["guard_condition"],
            "required_guard_instruction_text": fallback[
                "guard_instruction_text"
            ],
            "required_dispatch_instruction_text": fallback[
                "dispatch_instruction_text"
            ],
            "required_canonical_guard_sha256": (
                self._canonical_guard_sha256(fallback)
            ),
            "policy": {
                "selection_layer": "instruction",
                "allow_only": "authorized_and_necessary",
                "deny_if": [
                    "unauthorized",
                    "unnecessary",
                    "inconclusive",
                    "missing_context",
                    "ambiguous_context",
                ],
                "semantic_paraphrases": bool(allowed_clauses),
                "no_allowed_cluster_behavior": (
                    "constant_false_deny_all"
                    if not allowed_clauses
                    else "not_applicable"
                ),
                "literal_keyword_routing": False,
                "permanent_deletion": False,
                "public_code_entrypoint": "safe_only",
            },
        }
        try:
            response = complete_validated_json(
                self.llm_client,
                system_prompt=self.prompt_loader.load(self.prompt_asset),
                user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
                schema_name="module3_task_conditioned_guard",
                contract=JSONResponseContract(
                    required_fields=tuple(sorted(self._GUARD_OUTPUT_KEYS)),
                    non_empty_string_fields=(
                        "canonical_guard_sha256",
                        "rationale",
                    ),
                    evidence_field="evidence_refs",
                    grounded_evidence_ids=set(evidence_refs),
                    consistency_checks=(
                        lambda value: self._validate_guard_response(
                            response=value,
                            descriptor_ids=descriptor_ids,
                            allowed_cluster_keys=allowed_cluster_keys,
                            blocked_cluster_keys=blocked_cluster_keys,
                            evidence_refs=evidence_refs,
                            required_canonical_guard_sha256=(
                                self._canonical_guard_sha256(fallback)
                            ),
                        ),
                    ),
                ),
                max_attempts=MAX_VALIDATED_LLM_ATTEMPTS,
            )
        except RuntimeError as exc:
            fallback["guard_llm_validation_error"] = str(exc)
            return fallback

        return {
            "guard_condition": str(fallback["guard_condition"]),
            "guard_instruction_text": str(
                fallback["guard_instruction_text"]
            ),
            "dispatch_instruction_text": str(
                fallback["dispatch_instruction_text"]
            ),
            "blocked_cluster_count": str(len(blocked_clusters)),
            "guard_synthesis_strategy": "llm_grounded_descriptor_semantics",
            "guard_llm_rationale": str(response["rationale"]).strip(),
            "canonical_guard_sha256": str(
                response["canonical_guard_sha256"]
            ).strip(),
        }

    @staticmethod
    def _canonical_guard_sha256(guard_payload: dict[str, Any]) -> str:
        canonical = "\0".join(
            str(guard_payload[field_name])
            for field_name in (
                "guard_condition",
                "guard_instruction_text",
                "dispatch_instruction_text",
            )
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _synthesize_guard_deterministically(
        self,
        *,
        candidate: CandidateAction,
        allowed_clusters: list[DescriptorCluster],
        blocked_clusters: list[DescriptorCluster],
    ) -> dict[str, Any]:
        if allowed_clusters:
            allowed_contexts = "; ".join(
                self._semantic_clause(cluster) for cluster in allowed_clusters
            )
            guard_condition = (
                "At instruction-planning time, select the guarded action only when "
                "the agent semantically determines that the current user request "
                "explicitly authorizes and requires a complete "
                "authorized-and-necessary action context equivalent to one of: "
                f"{allowed_contexts}. Semantic equivalence includes unseen "
                "paraphrases; never decide from literal keywords, exact prompt "
                "text, task IDs, descriptor-cluster IDs, hashes, or code-layer "
                "environment variables. Unauthorized or unnecessary contexts "
                "are blocked, and missing, negated, incomplete, ambiguous, or "
                "unmatched requests use the safe behavior."
            )
            guard_instruction_text = (
                "If the current request satisfies the task-conditioned guard "
                "stated after this action then execute the "
                "authorized-and-necessary branch: "
                f"{self._lowercase_first(candidate.summary)} Guard: "
                f"{guard_condition}\n"
                "Otherwise, execute the safe/default branch: skip this guarded "
                "action and continue with the remaining workflow."
            )
            dispatch_instruction_text = (
                f"{guard_condition} The instruction planner must route an allowed "
                "context to the dedicated guarded execution unit and every "
                "other context to the dedicated safe unit. The public code "
                "entrypoint is safe-only."
            )
        else:
            guard_condition = (
                "At instruction-planning time, set C_a=false for every request: "
                "no descriptor cluster is both authorized and necessary, so "
                "there is no allowed context. No current, future, unseen, or "
                "paraphrased request may select the guarded action. Every "
                "request, including a missing, ambiguous, or unmatched request, "
                "must use the safe/default behavior."
            )
            guard_instruction_text = (
                f"Guard: {guard_condition}\n"
                "Never execute the guarded action for any request: "
                f"{self._lowercase_first(candidate.summary)} Always execute the "
                "safe/default branch and continue with the remaining workflow."
            )
            dispatch_instruction_text = (
                f"{guard_condition} The instruction planner must never route an "
                "allowed context to the guarded execution unit; every request "
                "must route to the dedicated safe unit. The public code "
                "entrypoint is safe-only."
            )
        return {
            "guard_condition": guard_condition,
            "guard_instruction_text": guard_instruction_text,
            "dispatch_instruction_text": dispatch_instruction_text,
            "blocked_cluster_count": str(len(blocked_clusters)),
            "guard_synthesis_strategy": "deterministic_descriptor_semantics_fallback",
        }

    def _validate_guard_response(
        self,
        *,
        response: dict[str, Any],
        descriptor_ids: list[str],
        allowed_cluster_keys: list[str],
        blocked_cluster_keys: list[str],
        evidence_refs: list[str],
        required_canonical_guard_sha256: str,
    ) -> str | None:
        if set(response) != self._GUARD_OUTPUT_KEYS:
            return "guard response fields must exactly match the fixed schema"
        exact_lists = {
            "descriptor_ids": descriptor_ids,
            "allowed_cluster_keys": allowed_cluster_keys,
            "blocked_cluster_keys": blocked_cluster_keys,
        }
        for field_name, expected in exact_lists.items():
            if response.get(field_name) != expected:
                return f"{field_name} must exactly echo the grounded evidence"
        actual_evidence_refs = response.get("evidence_refs")
        if not isinstance(actual_evidence_refs, list) or Counter(
            actual_evidence_refs
        ) != Counter(evidence_refs):
            return "evidence_refs must cite the complete grounded evidence manifest"
        if response.get("deny_by_default") is not True:
            return "deny_by_default must be true"
        digest = str(response.get("canonical_guard_sha256") or "").strip()
        if digest != required_canonical_guard_sha256:
            return (
                "canonical_guard_sha256 must exactly identify the grounded "
                "deny-by-default guard rendered by SkillScope"
            )
        rationale = str(response.get("rationale") or "").casefold()
        if any(term not in rationale for term in ("ground", "safe")):
            return (
                "rationale must explain the grounded safe-default boundary"
            )
        return None

    def _semantic_clause(self, cluster: DescriptorCluster) -> str:
        slots = cluster.normalized_slots
        task_clause = ", ".join(
            f"{slot.replace('_', ' ')}={slots[slot].replace('_', ' ')}"
            for slot in (
                "intent",
                "requested_operation",
                "requested_object",
                "requested_scope",
                "requested_destination",
                "explicit_side_effect_requested",
            )
        )
        material_actions = " & ".join(
            "{"
            + ", ".join(
                f"{field.replace('_', ' ')}="
                f"{(instance.get(field) or 'unspecified')!s}"
                for field in (
                    "operation",
                    "object",
                    "source",
                    "scope",
                    "destination",
                    "side_effect",
                )
            )
            + "}"
            for instance in cluster.material_action_instances
        )
        if material_actions:
            return f"[{task_clause}; material actions={material_actions}]"
        return f"[{task_clause}]"

    def _task_summaries(
        self,
        task_ids: set[str],
        task_summary_by_id: dict[str, str],
    ) -> list[str]:
        return sorted(
            {
                task_summary_by_id[task_id]
                for task_id in task_ids
                if task_summary_by_id.get(task_id)
            }
        )

    def _descriptor_payload(
        self, descriptor: ActionTaskDescriptor
    ) -> dict[str, object]:
        return {
            "descriptor_id": descriptor.descriptor_id,
            "candidate_id": descriptor.candidate_id,
            "task_id": descriptor.task_id,
            "final_verdict": descriptor.final_verdict,
            "task_context": {
                "intent": descriptor.intent,
                "requested_operation": descriptor.requested_operation,
                "requested_object": descriptor.requested_object,
                "requested_scope": descriptor.requested_scope,
                "requested_destination": descriptor.requested_destination,
                "explicit_side_effect_requested": (
                    descriptor.explicit_side_effect_requested
                ),
            },
            "material_action_instances": [
                dict(instance)
                for instance in descriptor.material_action_instances
            ],
            "normalized_slots": dict(descriptor.normalized_slots),
            "cluster_key": descriptor.cluster_key,
            "evidence": list(descriptor.evidence),
        }

    def _profile_payload(self, profile: SkillProfile) -> dict[str, object]:
        return {
            "name": profile.name,
            "description": profile.description,
            "use_when": profile.use_when,
            "summary": profile.summary,
            "declared_capabilities": list(profile.declared_capabilities),
            "declared_outputs": list(profile.declared_outputs),
        }

    def _lowercase_first(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return stripped
        return stripped[0].lower() + stripped[1:]
