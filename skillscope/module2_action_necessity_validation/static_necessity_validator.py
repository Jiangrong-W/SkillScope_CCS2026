from __future__ import annotations

from pathlib import Path
from typing import Any

from skillscope.common.llm import (
    DisabledLLMClient,
    MAX_VALIDATED_LLM_ATTEMPTS,
    PromptAssetLoader,
    StructuredLLMClient,
)
from skillscope.common.models import (
    CandidateAction,
    CandidateExtractionResult,
    NecessityDecision,
    TaskSpec,
)

from .validated_llm import (
    ValidatedLLMCaller,
    bounded_confidence,
    strict_bool,
    string_list,
)


class StaticNecessityValidator:
    """Predict CorePres and GoalSat without claiming that a replay occurred."""

    RESPONSE_KEY_ORDER = [
        "label",
        "reason",
        "confidence",
        "would_execute_under_prompt",
        "predicted_core_preserved_if_removed",
        "predicted_goal_satisfied_if_removed",
        "necessity_basis",
        "evidence_refs",
        "task_boundary_explanation",
        "uncertainty_flags",
    ]
    RESPONSE_KEYS = {
        *RESPONSE_KEY_ORDER,
    }
    ALLOWED_EVIDENCE_REFS = {
        "skill_profile",
        "skill_context.instruction_files",
        "skill_context.script_files",
        "user_task.prompt",
        "user_task.task_summary",
        "user_task.chain_node_ids",
        "user_task.chain_summaries",
        "user_task.fixtures",
        "target_action.node_id",
        "target_action.summary",
        "target_action.operation_type",
        "target_action.object_ref",
        "target_action.upstream_action_chain",
        "target_action.downstream_action_chain",
        "target_action.predicate_context",
        "target_node.raw_text",
        "target_node.source_excerpt",
        "judgment_policy.mode",
    }
    REQUIRED_EVIDENCE_REFS = {
        "user_task.prompt",
        "user_task.chain_node_ids",
        "target_action.node_id",
        "target_action.upstream_action_chain",
        "target_action.downstream_action_chain",
        "judgment_policy.mode",
    }

    def __init__(
        self,
        *,
        llm_client: StructuredLLMClient | None = None,
        prompt_loader: PromptAssetLoader | None = None,
        prompt_asset: str = "prompts/module2_static_necessity_reasoning.md",
        max_llm_attempts: int = MAX_VALIDATED_LLM_ATTEMPTS,
    ) -> None:
        self.llm_client = llm_client or DisabledLLMClient()
        self.prompt_loader = prompt_loader
        self.prompt_asset = prompt_asset
        self.llm_caller = ValidatedLLMCaller(
            llm_client=self.llm_client,
            prompt_loader=self.prompt_loader,
            max_attempts=max_llm_attempts,
        )

    def judge(
        self,
        *,
        analysis: CandidateExtractionResult,
        candidate: CandidateAction,
        task: TaskSpec,
    ) -> NecessityDecision:
        payload = self._build_payload(analysis=analysis, candidate=candidate, task=task)
        result = self.llm_caller.complete(
            prompt_asset=self.prompt_asset,
            payload=payload,
            schema_name="module2_static_corepres_goalsat_reasoning",
            validator=self._validate_llm_response,
        )
        if result.payload is not None:
            return self._decision_from_llm(payload, result.payload, result.attempts)
        return self._judge_with_fallback(payload, result.validation_errors)

    def _build_payload(
        self,
        *,
        analysis: CandidateExtractionResult,
        candidate: CandidateAction,
        task: TaskSpec,
    ) -> dict[str, Any]:
        node = analysis.ueg.node_by_id(candidate.node_id)
        instruction_context = [
            {
                "relative_path": artifact.relative_path,
                "text_preview": artifact.text_preview,
            }
            for artifact in analysis.bundle.instruction_files[:3]
        ]
        return {
            "skill_profile": {
                "name": analysis.profile.name,
                "description": analysis.profile.description,
                "use_when": analysis.profile.use_when,
                "summary": analysis.profile.summary,
                "declared_capabilities": analysis.profile.declared_capabilities,
                "declared_outputs": analysis.profile.declared_outputs,
                "declared_data_scope": analysis.profile.declared_data_scope,
                "declared_execution_scope": analysis.profile.declared_execution_scope,
            },
            "skill_context": {
                "instruction_files": instruction_context,
                "script_files": [
                    artifact.relative_path for artifact in analysis.bundle.script_files
                ],
            },
            "user_task": {
                "task_id": task.task_id,
                "prompt": task.prompt,
                "task_summary": task.task_summary,
                "chain_node_ids": task.chain_node_ids,
                "chain_summaries": task.chain_summaries,
                "fixtures": [
                    {
                        "fixture_type": fixture.fixture_type,
                        "target": fixture.target,
                        "required": fixture.required,
                    }
                    for fixture in task.fixtures
                ],
            },
            "target_action": {
                "candidate_id": candidate.candidate_id,
                "node_id": candidate.node_id,
                "layer": candidate.layer,
                "summary": candidate.summary,
                "source_file": candidate.source_file,
                "risk_tags": candidate.risk_tags,
                "operation_type": node.operation_type if node is not None else None,
                "object_ref": node.object_ref if node is not None else None,
                "reason": candidate.reason,
                "upstream_action_chain": candidate.upstream_action_chain,
                "downstream_action_chain": candidate.downstream_action_chain,
                "predicate_context": candidate.predicate_context,
            },
            "target_node": (
                {
                    "node_id": node.node_id,
                    "layer": node.layer,
                    "node_type": node.node_type,
                    "summary": node.summary,
                    "raw_text": node.raw_text,
                    "source_file": node.source_file,
                    "operation_type": node.operation_type,
                    "object_ref": node.object_ref,
                    "risk_tags": node.risk_tags,
                    "attributes": node.attributes,
                    "source_excerpt": self._source_excerpt(
                        analysis,
                        node.source_file,
                        node.source_range.start_line if node.source_range is not None else None,
                        node.source_range.end_line if node.source_range is not None else None,
                    ),
                }
                if node is not None
                else None
            ),
            "judgment_policy": {
                "mode": "static_prediction_only",
                "dynamic_replay_available": False,
                "unnecessary_iff": (
                    "the action would execute and removal is predicted to preserve "
                    "both core task behavior and user-goal satisfaction"
                ),
                "no_dynamic_claims": True,
            },
            "response_contract": {
                "required_keys": list(self.RESPONSE_KEY_ORDER),
                "uncertainty_flags_required": True,
                "empty_uncertainty_flags_if_none": True,
                "uncertainty_flags_semantics": (
                    "list only material unresolved uncertainty that forces an "
                    "inconclusive label; do not list generic static-mode caveats"
                ),
                "required_evidence_refs": sorted(self.REQUIRED_EVIDENCE_REFS),
            },
        }

    def _validate_llm_response(
        self,
        response: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, list[str]]:
        errors: list[str] = []
        if set(response) != self.RESPONSE_KEYS:
            errors.append(
                "response_keys_must_exactly_match_static_necessity_schema"
            )
        label = response.get("label")
        if not isinstance(label, str) or label.strip().lower() not in {
            "necessary",
            "unnecessary",
            "inconclusive",
        }:
            errors.append("label_must_be_valid_necessity_label")
            normalized_label = "inconclusive"
        else:
            normalized_label = label.strip().lower()
        reason = response.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append("reason_must_be_nonempty_string")
            normalized_reason = ""
        else:
            normalized_reason = reason.strip()
        would_execute = strict_bool(
            response.get("would_execute_under_prompt"),
            "would_execute_under_prompt",
            errors,
        )
        predicted_core = strict_bool(
            response.get("predicted_core_preserved_if_removed"),
            "predicted_core_preserved_if_removed",
            errors,
        )
        predicted_goal = strict_bool(
            response.get("predicted_goal_satisfied_if_removed"),
            "predicted_goal_satisfied_if_removed",
            errors,
        )
        confidence = bounded_confidence(response.get("confidence"), errors)
        evidence = string_list(
            response.get("necessity_basis"),
            "necessity_basis",
            errors,
            required=True,
        )
        evidence_refs = string_list(
            response.get("evidence_refs"),
            "evidence_refs",
            errors,
            required=True,
        )
        if set(evidence_refs) - self.ALLOWED_EVIDENCE_REFS:
            errors.append("evidence_refs_contain_unsupported_static_assertions")
        if self.REQUIRED_EVIDENCE_REFS - set(evidence_refs):
            errors.append("evidence_refs_missing_required_static_context")
        if "uncertainty_flags" not in response:
            errors.append("uncertainty_flags_field_required_even_when_empty")
        uncertainty_flags = string_list(
            response.get("uncertainty_flags"),
            "uncertainty_flags",
            errors,
        )
        boundary = response.get("task_boundary_explanation")
        if not isinstance(boundary, str) or not boundary.strip():
            errors.append("task_boundary_explanation_must_be_nonempty_string")
            normalized_boundary = ""
        else:
            normalized_boundary = boundary.strip()

        expected_label = self._label_for_prediction(
            would_execute=bool(would_execute),
            predicted_core=bool(predicted_core),
            predicted_goal=bool(predicted_goal),
            uncertainty_flags=uncertainty_flags,
        )
        if normalized_label != expected_label:
            errors.append("label_conflicts_with_static_corepres_goalsat_contract")
        if errors:
            return None, errors
        return {
            "label": normalized_label,
            "reason": normalized_reason,
            "would_execute": would_execute,
            "predicted_core": predicted_core,
            "predicted_goal": predicted_goal,
            "confidence": confidence,
            "necessity_basis": evidence,
            "evidence_refs": evidence_refs,
            "uncertainty_flags": uncertainty_flags,
            "task_boundary_explanation": normalized_boundary,
        }, []

    def _decision_from_llm(
        self,
        payload: dict[str, Any],
        response: dict[str, Any],
        attempts: int,
    ) -> NecessityDecision:
        predicted_core = bool(response["predicted_core"])
        predicted_goal = bool(response["predicted_goal"])
        return NecessityDecision(
            candidate_id=payload["target_action"]["candidate_id"],
            task_id=payload["user_task"]["task_id"],
            label=response["label"],
            reason=response["reason"],
            # These fields represent observed replay equivalence and must remain false
            # in static mode. Predictions live in CorePres/GoalSat.
            trace_equivalent=False,
            output_equivalent=False,
            core_preserved=predicted_core,
            goal_satisfied=predicted_goal,
            executed_in_original=False,
            confidence=float(response["confidence"]),
            judge_strategy="static_llm_corepres_goalsat_prediction",
            necessity_basis=[
                *response["necessity_basis"],
                *[
                    f"evidence_ref:{evidence_ref}"
                    for evidence_ref in response["evidence_refs"]
                ],
                f"Validated after {attempts} LLM attempt(s); no dynamic run was performed.",
            ],
            task_boundary_explanation=response["task_boundary_explanation"],
            uncertainty_flags=response["uncertainty_flags"],
            judge_input=payload,
        )

    def _judge_with_fallback(
        self,
        payload: dict[str, Any],
        validation_errors: list[str],
    ) -> NecessityDecision:
        uncertainty_flags = ["missing_or_invalid_llm_for_static_reasoning"]
        if validation_errors:
            uncertainty_flags.append("bounded_llm_validation_exhausted")
        return NecessityDecision(
            candidate_id=payload["target_action"]["candidate_id"],
            task_id=payload["user_task"]["task_id"],
            label="inconclusive",
            reason=(
                "Static CorePres and GoalSat predictions could not be validated. "
                "No original execution or ablation replay was performed."
            ),
            trace_equivalent=False,
            output_equivalent=False,
            core_preserved=False,
            goal_satisfied=False,
            executed_in_original=False,
            confidence=0.0,
            judge_strategy="static_fallback_no_dynamic_evidence",
            necessity_basis=[],
            task_boundary_explanation=(
                "The evidence is limited to the task and code/instruction context."
            ),
            uncertainty_flags=uncertainty_flags,
            judge_input={
                **payload,
                "llm_validation_errors": list(validation_errors),
            },
        )

    def _label_for_prediction(
        self,
        *,
        would_execute: bool,
        predicted_core: bool,
        predicted_goal: bool,
        uncertainty_flags: list[str],
    ) -> str:
        if uncertainty_flags or not would_execute:
            return "inconclusive"
        if predicted_core and predicted_goal:
            return "unnecessary"
        return "necessary"

    def _source_excerpt(
        self,
        analysis: CandidateExtractionResult,
        source_file: str | None,
        start_line: int | None,
        end_line: int | None,
    ) -> str | None:
        if source_file is None:
            return None
        target_path = Path(analysis.bundle.root_path) / source_file
        if not target_path.exists():
            return None
        try:
            lines = target_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return None
        if start_line is None or end_line is None:
            return "\n".join(lines[:20])[:1200]
        start_index = max(start_line - 3, 0)
        end_index = min(end_line + 2, len(lines))
        return "\n".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(
                lines[start_index:end_index],
                start=start_index + 1,
            )
        )[:1500]
