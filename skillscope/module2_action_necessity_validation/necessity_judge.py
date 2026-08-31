from __future__ import annotations

from typing import Any

from skillscope.common.llm import (
    DisabledLLMClient,
    MAX_VALIDATED_LLM_ATTEMPTS,
    PromptAssetLoader,
    StructuredLLMClient,
)
from skillscope.common.models import (
    CandidateAction,
    NecessityDecision,
    ReplayPairRecord,
    TaskSpec,
    TaskTriggerEvidence,
)

from .output_comparator import OutputComparator
from .trace_normalizer import TraceNormalizer
from .validated_llm import (
    ValidatedLLMCaller,
    bounded_confidence,
    strict_bool,
    string_list,
)


class NecessityJudge:
    """Judge action necessity from CorePres(a, t) and GoalSat(t \\ a)."""

    MATERIAL_EVENT_TYPES = {
        "network_send",
        "exec_command",
        "file_read",
        "file_write",
        "file_access",
        "delete",
        "read_env",
        "collect_identifier",
    }
    MATERIAL_UNCERTAINTY_FLAGS = {
        "candidate_removal_unverified",
        "candidate_still_executed_in_replay",
        "material_goal_ambiguity",
        "missing_original_output_evidence",
        "missing_replay_output_evidence",
        "insufficient_goal_evidence",
        "ungrounded_original_final_output",
        "ungrounded_replay_final_output",
    }
    RESPONSE_KEYS = {
        "label",
        "executed_in_original",
        "core_preserved",
        "goal_satisfied",
        "confidence",
        "reason",
        "evidence",
        "evidence_refs",
        "uncertainty_flags",
    }
    ALLOWED_EVIDENCE_REFS = {
        "task.prompt",
        "task.chain_node_ids",
        "task.chain_summaries",
        "trigger_evidence.triggered",
        "trigger_evidence.reason",
        "replay_pair.original_status",
        "replay_pair.replay_status",
        "replay_pair.original_trace",
        "replay_pair.replay_trace",
        "replay_pair.original_output",
        "replay_pair.replay_output",
        "replay_pair.original_output_grounded",
        "replay_pair.replay_output_grounded",
        "replay_pair.candidate_absent_in_replay",
        "replay_pair.candidate_absence_verified",
        "replay_pair.goal_evidence_sufficient",
        "replay_pair.contract_uncertainty_flags",
        "replay_pair.core_preserved_fallback",
        "replay_pair.goal_satisfied_fallback",
        "replay_pair.output_comparison_fallback",
        "replay_pair.ablation",
    }
    REQUIRED_EVIDENCE_REFS = {
        "task.prompt",
        "trigger_evidence.triggered",
        "replay_pair.original_status",
        "replay_pair.replay_status",
        "replay_pair.original_trace",
        "replay_pair.replay_trace",
        "replay_pair.original_output",
        "replay_pair.replay_output",
        "replay_pair.original_output_grounded",
        "replay_pair.replay_output_grounded",
        "replay_pair.candidate_absent_in_replay",
        "replay_pair.candidate_absence_verified",
        "replay_pair.goal_evidence_sufficient",
    }

    def __init__(
        self,
        llm_client: StructuredLLMClient | None = None,
        prompt_loader: PromptAssetLoader | None = None,
        trace_normalizer: TraceNormalizer | None = None,
        output_comparator: OutputComparator | None = None,
        prompt_asset: str = "prompts/module2_necessity_judge.md",
        max_llm_attempts: int = MAX_VALIDATED_LLM_ATTEMPTS,
    ) -> None:
        self.llm_client = llm_client or DisabledLLMClient()
        self._deterministic_development_fallback = isinstance(
            self.llm_client,
            DisabledLLMClient,
        )
        self.prompt_loader = prompt_loader
        self.trace_normalizer = trace_normalizer or TraceNormalizer()
        self.output_comparator = output_comparator or OutputComparator()
        self.prompt_asset = prompt_asset
        self.llm_caller = ValidatedLLMCaller(
            llm_client=self.llm_client,
            prompt_loader=self.prompt_loader,
            max_attempts=max_llm_attempts,
        )

    def judge(
        self,
        candidate: CandidateAction,
        task: TaskSpec,
        replay_pair: ReplayPairRecord,
        trigger_evidence: TaskTriggerEvidence | None = None,
    ) -> NecessityDecision:
        payload = self._build_payload(
            candidate=candidate,
            task=task,
            replay_pair=replay_pair,
            trigger_evidence=trigger_evidence,
        )
        result = self.llm_caller.complete(
            prompt_asset=self.prompt_asset,
            payload=payload,
            schema_name="module2_dynamic_corepres_goalsat_judge",
            validator=lambda response: self._validate_llm_response(response, payload),
        )
        if result.payload is not None:
            return self._decision_from_llm(payload, result.payload, result.attempts)
        if not self._deterministic_development_fallback:
            return self._llm_failure_decision(
                payload=payload,
                validation_errors=result.validation_errors,
                attempts=result.attempts,
            )
        return self._judge_with_fallback(payload, [])

    def _build_payload(
        self,
        *,
        candidate: CandidateAction,
        task: TaskSpec,
        replay_pair: ReplayPairRecord,
        trigger_evidence: TaskTriggerEvidence | None,
    ) -> dict[str, Any]:
        normalized_original = self.trace_normalizer.normalize(replay_pair.original)
        normalized_replay = self.trace_normalizer.normalize(replay_pair.replay)
        output_comparison = self.output_comparator.compare(
            prompt=task.prompt,
            original_output=replay_pair.original.final_output,
            replay_output=replay_pair.replay.final_output,
            task_summary=task.task_summary,
            original_output_grounded=self._final_output_grounded(
                replay_pair.original
            ),
            replay_output_grounded=self._final_output_grounded(
                replay_pair.replay
            ),
        )
        core_preserved = self._core_preserved(
            task=task,
            normalized_original=normalized_original,
            normalized_replay=normalized_replay,
            candidate=candidate,
            ablation=replay_pair.ablation,
        )
        observed_trigger = (
            trigger_evidence.triggered
            if trigger_evidence is not None
            else (
                replay_pair.original.status == "completed"
                and candidate.node_id in replay_pair.original.executed_node_ids
            )
        )
        candidate_removal = self._candidate_removal_evidence(
            candidate=candidate,
            replay_pair=replay_pair,
        )
        contract_uncertainty_flags = list(
            output_comparison.get("uncertainty_flags", [])
        )
        if not candidate_removal["verified"]:
            contract_uncertainty_flags.append("candidate_removal_unverified")
        if candidate_removal["present"]:
            contract_uncertainty_flags.append(
                "candidate_still_executed_in_replay"
            )
        contract_uncertainty_flags = sorted(set(contract_uncertainty_flags))
        return {
            "candidate": {
                "candidate_id": candidate.candidate_id,
                "node_id": candidate.node_id,
                "layer": candidate.layer,
                "summary": candidate.summary,
                "risk_tags": candidate.risk_tags,
            },
            "task": {
                "task_id": task.task_id,
                "prompt": task.prompt,
                "task_summary": task.task_summary,
                "chain_node_ids": task.chain_node_ids,
                "chain_summaries": task.chain_summaries,
            },
            "trigger_evidence": {
                "triggered": observed_trigger,
                "strategy": (
                    trigger_evidence.trigger_strategy
                    if trigger_evidence is not None
                    else "derived_exact_node_trace"
                ),
                "reason": trigger_evidence.reason if trigger_evidence is not None else "",
            },
            "replay_pair": {
                "original_status": replay_pair.original.status,
                "replay_status": replay_pair.replay.status,
                "original_trace": normalized_original,
                "replay_trace": normalized_replay,
                "original_output": replay_pair.original.final_output,
                "replay_output": replay_pair.replay.final_output,
                "original_output_grounded": self._final_output_grounded(
                    replay_pair.original
                ),
                "replay_output_grounded": self._final_output_grounded(
                    replay_pair.replay
                ),
                "candidate_absent_in_replay": candidate_removal["absent"],
                "candidate_absence_verified": candidate_removal["verified"],
                "candidate_removal_evidence": candidate_removal,
                "goal_evidence_sufficient": bool(
                    output_comparison["evidence_sufficient"]
                ),
                "contract_uncertainty_flags": contract_uncertainty_flags,
                "core_preserved_fallback": core_preserved,
                "goal_satisfied_fallback": bool(output_comparison["equivalent"]),
                "output_comparison_fallback": output_comparison,
                "ablation": {
                    "strategy": replay_pair.ablation.strategy,
                    "node_id": replay_pair.ablation.node_id,
                    "notes": replay_pair.ablation.notes,
                },
            },
            "decision_contract": {
                "required_evidence_refs": sorted(
                    self.REQUIRED_EVIDENCE_REFS
                ),
                "required_uncertainty_flags": contract_uncertainty_flags,
                "unnecessary_iff": (
                    "original and replay completed, candidate triggered, "
                    "the replay proves the candidate absent, goal evidence is "
                    "sufficient, no material uncertainty remains, "
                    "core_preserved=true, and goal_satisfied=true"
                ),
                "necessary_if": (
                    "a completed and verified candidate removal breaks either "
                    "core preservation or goal satisfaction"
                ),
                "otherwise": "inconclusive",
            },
        }

    def _validate_llm_response(
        self,
        response: dict[str, Any],
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, list[str]]:
        errors: list[str] = []
        if set(response) != self.RESPONSE_KEYS:
            errors.append("response_keys_must_exactly_match_necessity_schema")
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

        executed = strict_bool(
            response.get("executed_in_original"),
            "executed_in_original",
            errors,
        )
        core_preserved = strict_bool(
            response.get("core_preserved"),
            "core_preserved",
            errors,
        )
        goal_satisfied = strict_bool(
            response.get("goal_satisfied"),
            "goal_satisfied",
            errors,
        )
        confidence = bounded_confidence(response.get("confidence"), errors)
        evidence = string_list(response.get("evidence"), "evidence", errors, required=True)
        evidence_refs = string_list(
            response.get("evidence_refs"),
            "evidence_refs",
            errors,
            required=True,
        )
        unsupported_refs = sorted(
            set(evidence_refs) - self.ALLOWED_EVIDENCE_REFS
        )
        if unsupported_refs:
            errors.append("evidence_refs_contain_unsupported_assertions")
        missing_refs = sorted(
            self.REQUIRED_EVIDENCE_REFS - set(evidence_refs)
        )
        if missing_refs:
            errors.append("evidence_refs_missing_required_runtime_evidence")
        uncertainty_flags = string_list(
            response.get("uncertainty_flags", []),
            "uncertainty_flags",
            errors,
        )
        contract_uncertainty_flags = list(
            payload["replay_pair"]["contract_uncertainty_flags"]
        )
        missing_contract_flags = sorted(
            set(contract_uncertainty_flags) - set(uncertainty_flags)
        )
        if missing_contract_flags:
            errors.append(
                "uncertainty_flags_omit_grounded_contract_uncertainty"
            )

        observed_trigger = bool(payload["trigger_evidence"]["triggered"])
        runs_completed = (
            payload["replay_pair"]["original_status"] == "completed"
            and payload["replay_pair"]["replay_status"] == "completed"
        )
        if executed is not None and executed != observed_trigger:
            errors.append("executed_in_original_conflicts_with_observed_trigger")
        expected_label = self._label_for_contract(
            triggered=observed_trigger,
            runs_completed=runs_completed,
            candidate_absent=bool(
                payload["replay_pair"]["candidate_absent_in_replay"]
            ),
            candidate_absence_verified=bool(
                payload["replay_pair"]["candidate_absence_verified"]
            ),
            goal_evidence_sufficient=bool(
                payload["replay_pair"]["goal_evidence_sufficient"]
            ),
            material_uncertainty=self._has_material_uncertainty(
                [*contract_uncertainty_flags, *uncertainty_flags]
            ),
            core_preserved=bool(core_preserved),
            goal_satisfied=bool(goal_satisfied),
        )
        if normalized_label != expected_label:
            errors.append("label_conflicts_with_corepres_goalsat_contract")
        if (
            self._has_material_uncertainty(uncertainty_flags)
            and normalized_label != "inconclusive"
        ):
            errors.append("material_uncertainty_precludes_necessity_label")

        if errors:
            return None, errors
        return {
            "label": normalized_label,
            "reason": normalized_reason,
            "executed_in_original": executed,
            "core_preserved": core_preserved,
            "goal_satisfied": goal_satisfied,
            "confidence": confidence,
            "evidence": evidence,
            "evidence_refs": evidence_refs,
            "uncertainty_flags": uncertainty_flags,
        }, []

    def _decision_from_llm(
        self,
        payload: dict[str, Any],
        response: dict[str, Any],
        attempts: int,
    ) -> NecessityDecision:
        core_preserved = bool(response["core_preserved"])
        goal_satisfied = bool(response["goal_satisfied"])
        uncertainty_flags = sorted(
            set(
                response["uncertainty_flags"]
                + payload["replay_pair"]["contract_uncertainty_flags"]
            )
        )
        return NecessityDecision(
            candidate_id=payload["candidate"]["candidate_id"],
            task_id=payload["task"]["task_id"],
            label=response["label"],
            reason=response["reason"],
            trace_equivalent=core_preserved,
            output_equivalent=goal_satisfied,
            core_preserved=core_preserved,
            goal_satisfied=goal_satisfied,
            executed_in_original=bool(response["executed_in_original"]),
            confidence=float(response["confidence"]),
            judge_strategy="dynamic_llm_corepres_goalsat",
            necessity_basis=[
                *response["evidence"],
                *[
                    f"evidence_ref:{evidence_ref}"
                    for evidence_ref in response["evidence_refs"]
                ],
                f"Validated after {attempts} LLM attempt(s).",
            ],
            uncertainty_flags=uncertainty_flags,
            judge_input=payload,
        )

    def _judge_with_fallback(
        self,
        payload: dict[str, Any],
        validation_errors: list[str],
    ) -> NecessityDecision:
        original_status = str(payload["replay_pair"]["original_status"] or "")
        replay_status = str(payload["replay_pair"]["replay_status"] or "")
        triggered = bool(payload["trigger_evidence"]["triggered"])
        runs_completed = original_status == "completed" and replay_status == "completed"
        core_preserved = bool(payload["replay_pair"]["core_preserved_fallback"])
        goal_satisfied = bool(payload["replay_pair"]["goal_satisfied_fallback"])
        candidate_absent = bool(
            payload["replay_pair"]["candidate_absent_in_replay"]
        )
        candidate_absence_verified = bool(
            payload["replay_pair"]["candidate_absence_verified"]
        )
        goal_evidence_sufficient = bool(
            payload["replay_pair"]["goal_evidence_sufficient"]
        )
        contract_uncertainty_flags = list(
            payload["replay_pair"]["contract_uncertainty_flags"]
        )
        llm_reported_material_uncertainty = (
            "material_uncertainty_precludes_necessity_label"
            in validation_errors
        )
        if llm_reported_material_uncertainty:
            contract_uncertainty_flags.append(
                "material_llm_judgment_uncertainty"
            )
        label = self._label_for_contract(
            triggered=triggered,
            runs_completed=runs_completed,
            candidate_absent=candidate_absent,
            candidate_absence_verified=candidate_absence_verified,
            goal_evidence_sufficient=goal_evidence_sufficient,
            material_uncertainty=self._has_material_uncertainty(
                contract_uncertainty_flags
            ),
            core_preserved=core_preserved,
            goal_satisfied=goal_satisfied,
        )
        uncertainty_flags: list[str] = list(contract_uncertainty_flags)
        if not runs_completed:
            uncertainty_flags.append("incomplete_original_or_replay_execution")
        if not triggered:
            uncertainty_flags.append("candidate_not_triggered_in_original")
        if validation_errors:
            uncertainty_flags.append("llm_unavailable_or_invalid_after_bounded_retry")
        uncertainty_flags.append(
            "deterministic_development_fallback_no_llm_configured"
        )
        uncertainty_flags = sorted(set(uncertainty_flags))

        if label == "unnecessary":
            reason = (
                "The candidate triggered in the original run, while its ablation "
                "preserved the core task flow and still satisfied the user goal."
            )
            confidence = 0.84
        elif label == "necessary":
            broken = []
            if not core_preserved:
                broken.append("core task flow")
            if not goal_satisfied:
                broken.append("user goal")
            reason = (
                "The candidate triggered and removing it failed to preserve "
                + " and ".join(broken)
                + "."
            )
            confidence = 0.72
        else:
            reason = (
                "Necessity is inconclusive because the completed trigger, "
                "candidate-removal postcondition, or goal evidence contract "
                "could not be established without material uncertainty."
            )
            confidence = 0.0

        basis = [
            f"CorePres={core_preserved}",
            f"GoalSat={goal_satisfied}",
            f"Triggered={triggered}",
            f"CandidateAbsentInReplay={candidate_absent}",
            f"CandidateAbsenceVerified={candidate_absence_verified}",
            f"GoalEvidenceSufficient={goal_evidence_sufficient}",
            f"OriginalStatus={original_status}",
            f"ReplayStatus={replay_status}",
        ]
        return NecessityDecision(
            candidate_id=payload["candidate"]["candidate_id"],
            task_id=payload["task"]["task_id"],
            label=label,
            reason=reason,
            trace_equivalent=core_preserved,
            output_equivalent=goal_satisfied,
            core_preserved=core_preserved,
            goal_satisfied=goal_satisfied,
            executed_in_original=triggered,
            confidence=confidence,
            judge_strategy="dynamic_deterministic_development_fallback_corepres_goalsat",
            necessity_basis=basis,
            uncertainty_flags=uncertainty_flags,
            judge_input=payload,
        )

    def _llm_failure_decision(
        self,
        *,
        payload: dict[str, Any],
        validation_errors: list[str],
        attempts: int,
    ) -> NecessityDecision:
        """Keep configured-LLM exhaustion unresolved, never heuristic-positive."""

        original_status = str(payload["replay_pair"]["original_status"] or "")
        replay_status = str(payload["replay_pair"]["replay_status"] or "")
        triggered = bool(payload["trigger_evidence"]["triggered"])
        core_preserved = bool(
            payload["replay_pair"]["core_preserved_fallback"]
        )
        goal_satisfied = bool(
            payload["replay_pair"]["goal_satisfied_fallback"]
        )
        candidate_absent = bool(
            payload["replay_pair"]["candidate_absent_in_replay"]
        )
        candidate_absence_verified = bool(
            payload["replay_pair"]["candidate_absence_verified"]
        )
        goal_evidence_sufficient = bool(
            payload["replay_pair"]["goal_evidence_sufficient"]
        )
        diagnostic_label = self._label_for_contract(
            triggered=triggered,
            runs_completed=(
                original_status == "completed" and replay_status == "completed"
            ),
            candidate_absent=candidate_absent,
            candidate_absence_verified=candidate_absence_verified,
            goal_evidence_sufficient=goal_evidence_sufficient,
            material_uncertainty=self._has_material_uncertainty(
                payload["replay_pair"]["contract_uncertainty_flags"]
            ),
            core_preserved=core_preserved,
            goal_satisfied=goal_satisfied,
        )
        basis = [
            f"DiagnosticFallbackLabel={diagnostic_label}",
            f"DiagnosticCorePres={core_preserved}",
            f"DiagnosticGoalSat={goal_satisfied}",
            f"Triggered={triggered}",
            f"CandidateAbsentInReplay={candidate_absent}",
            f"CandidateAbsenceVerified={candidate_absence_verified}",
            f"GoalEvidenceSufficient={goal_evidence_sufficient}",
            f"ConfiguredLLMCalls={attempts}",
            "Diagnostic fallback values were not used for the final verdict.",
        ]
        if validation_errors:
            basis.append(
                "Validator diagnostics: " + "; ".join(validation_errors)
            )
        uncertainty_flags = [
            *payload["replay_pair"]["contract_uncertainty_flags"],
            "configured_llm_necessity_judgment_unresolved",
            "llm_unavailable_or_invalid_after_bounded_retry",
        ]
        if "material_uncertainty_precludes_necessity_label" in validation_errors:
            uncertainty_flags.append("material_llm_judgment_uncertainty")
        judge_input = dict(payload)
        judge_input["llm_validation_errors"] = list(validation_errors)
        judge_input["diagnostic_fallback"] = {
            "label": diagnostic_label,
            "core_preserved": core_preserved,
            "goal_satisfied": goal_satisfied,
            "not_used_for_final_verdict": True,
        }
        return NecessityDecision(
            candidate_id=payload["candidate"]["candidate_id"],
            task_id=payload["task"]["task_id"],
            label="inconclusive",
            reason=(
                "Necessity is unresolved because the configured LLM did not "
                "return a schema-valid, evidence-consistent judgment. "
                "Deterministic CorePres and GoalSat estimates are diagnostic only."
            ),
            trace_equivalent=core_preserved,
            output_equivalent=goal_satisfied,
            core_preserved=core_preserved,
            goal_satisfied=goal_satisfied,
            executed_in_original=triggered,
            confidence=0.0,
            judge_strategy="dynamic_llm_corepres_goalsat_unresolved",
            necessity_basis=basis,
            uncertainty_flags=sorted(set(uncertainty_flags)),
            judge_input=judge_input,
        )

    def _label_for_contract(
        self,
        *,
        triggered: bool,
        runs_completed: bool,
        candidate_absent: bool,
        candidate_absence_verified: bool,
        goal_evidence_sufficient: bool,
        material_uncertainty: bool,
        core_preserved: bool,
        goal_satisfied: bool,
    ) -> str:
        if (
            not runs_completed
            or not triggered
            or not candidate_absent
            or not candidate_absence_verified
            or not goal_evidence_sufficient
            or material_uncertainty
        ):
            return "inconclusive"
        if core_preserved and goal_satisfied:
            return "unnecessary"
        return "necessary"

    def _candidate_removal_evidence(
        self,
        *,
        candidate: CandidateAction,
        replay_pair: ReplayPairRecord,
    ) -> dict[str, object]:
        metadata = replay_pair.replay.metadata
        ablation_applied = metadata.get("ablation_applied") is True
        fingerprint_absent = (
            metadata.get("ablated_candidate_fingerprint_absent") is True
        )
        present = self._candidate_present_in_replay(
            candidate=candidate,
            replay_pair=replay_pair,
            fingerprint_absent=fingerprint_absent,
        )
        verified = ablation_applied and fingerprint_absent
        return {
            "absent": verified and not present,
            "present": present,
            "verified": verified,
            "ablation_applied": ablation_applied,
            "fingerprint_absent_from_replay_graph": fingerprint_absent,
        }

    def _candidate_present_in_replay(
        self,
        *,
        candidate: CandidateAction,
        replay_pair: ReplayPairRecord,
        fingerprint_absent: bool,
    ) -> bool:
        record = replay_pair.replay
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
                    "summary": event.summary,
                    "node_id": event.node_id,
                    "object_ref": event.object_ref,
                    "attributes": event.attributes,
                }
                for event in record.trace
            )

        for payload in payloads:
            attributes = payload.get("attributes")
            if not isinstance(attributes, dict):
                attributes = {}
            event_type = str(payload.get("event_type") or "")
            material_operation = str(
                attributes.get("material_operation") or ""
            )
            material = (
                event_type in self.MATERIAL_EVENT_TYPES
                or material_operation in self.MATERIAL_EVENT_TYPES
            )
            same_summary = str(payload.get("summary") or "") == candidate.summary
            identities = {
                value
                for value in (
                    payload.get("node_id"),
                    attributes.get("instruction_node_id"),
                )
                if isinstance(value, str) and value
            }
            if identities:
                if candidate.node_id not in identities:
                    continue
                if material or same_summary:
                    return True
                continue
            if material and self._strong_candidate_fingerprint_matches(
                observed_operation=material_operation or event_type,
                source_file=attributes.get("source_file"),
                line_number=attributes.get("line_number"),
                ablation=replay_pair.ablation,
            ):
                return True

        return (
            not fingerprint_absent
            and candidate.node_id in record.executed_node_ids
        )

    def _strong_candidate_fingerprint_matches(
        self,
        *,
        observed_operation: str,
        source_file: object,
        line_number: object,
        ablation: object,
    ) -> bool:
        expected_operation = getattr(ablation, "operation_type", None)
        expected_source = getattr(ablation, "source_file", None)
        start_line = getattr(ablation, "source_start_line", None)
        end_line = getattr(ablation, "source_end_line", None)
        if not (
            isinstance(expected_operation, str)
            and expected_operation
            and isinstance(source_file, str)
            and source_file
            and isinstance(expected_source, str)
            and expected_source
            and isinstance(line_number, int)
            and isinstance(start_line, int)
            and isinstance(end_line, int)
        ):
            return False
        normalized_source = source_file.replace("\\", "/")
        normalized_expected = expected_source.replace("\\", "/")
        source_matches = (
            normalized_source == normalized_expected
            or normalized_source.endswith(f"/{normalized_expected}")
        )
        return (
            source_matches
            and start_line <= line_number <= end_line
            and self._operation_matches(
                observed=observed_operation,
                expected=expected_operation,
            )
        )

    def _operation_matches(
        self,
        *,
        observed: str,
        expected: str | None,
    ) -> bool:
        if not expected:
            return True
        aliases = {
            "send": "network_send",
            "sync": "network_send",
            "upload": "network_send",
            "post": "network_send",
            "execute": "exec_command",
            "run": "exec_command",
            "read": "file_access",
            "write": "file_access",
            "file_read": "file_access",
            "file_write": "file_access",
        }
        normalized_observed = aliases.get(observed.lower(), observed.lower())
        normalized_expected = aliases.get(expected.lower(), expected.lower())
        return normalized_observed == normalized_expected

    def _has_material_uncertainty(self, flags: list[str]) -> bool:
        return any(
            flag in self.MATERIAL_UNCERTAINTY_FLAGS
            or flag.startswith("material_")
            for flag in flags
        )

    def _final_output_grounded(self, record: object) -> bool:
        metadata = getattr(record, "metadata", {})
        if not isinstance(metadata, dict):
            return True
        value = metadata.get("final_output_grounded")
        # Legacy/imported records predate this field; preserve compatibility.
        # Runtime-produced records always set it explicitly.
        return True if value is None else value is True

    def _core_preserved(
        self,
        *,
        task: TaskSpec,
        normalized_original: list[dict[str, object]],
        normalized_replay: list[dict[str, object]],
        candidate: CandidateAction,
        ablation: object,
    ) -> bool:
        original_core = [
            self._event_signature(item)
            for item in normalized_original
            if not self._is_candidate_trace_item(
                item,
                candidate,
                ablation,
            )
        ]
        replay_core = [self._event_signature(item) for item in normalized_replay]
        if original_core == replay_core:
            return True

        candidate_position = None
        if task.expected_candidate_node_id in task.chain_node_ids:
            candidate_position = task.chain_node_ids.index(task.expected_candidate_node_id)
        task_core = [
            summary
            for index, summary in enumerate(task.chain_summaries)
            if index != candidate_position
        ]
        if not task_core:
            return False
        original_summaries = [
            str(item["summary"])
            for item in normalized_original
            if item.get("summary")
            and not self._is_candidate_trace_item(
                item,
                candidate,
                ablation,
            )
        ]
        replay_summaries = [
            str(item["summary"])
            for item in normalized_replay
            if item.get("summary")
        ]
        return self._contains_subsequence(original_summaries, task_core) and self._contains_subsequence(
            replay_summaries,
            task_core,
        )

    def _is_candidate_trace_item(
        self,
        item: dict[str, object],
        candidate: CandidateAction,
        ablation: object,
    ) -> bool:
        identities = {
            value
            for value in (
                item.get("node_id"),
                item.get("instruction_node_id"),
            )
            if isinstance(value, str) and value
        }
        if identities:
            return candidate.node_id in identities
        observed_operation = str(
            item.get("material_operation")
            or item.get("event_type")
            or ""
        )
        return self._strong_candidate_fingerprint_matches(
            observed_operation=observed_operation,
            source_file=item.get("source_file"),
            line_number=item.get("line_number"),
            ablation=ablation,
        )

    def _event_signature(
        self,
        item: dict[str, object],
    ) -> tuple[object, ...]:
        return (
            item.get("event_type"),
            item.get("summary"),
            item.get("object_ref"),
            item.get("arguments_summary"),
        )

    def _contains_subsequence(
        self,
        haystack: list[str | None],
        needle: list[str],
    ) -> bool:
        if not needle:
            return True
        pointer = 0
        for item in haystack:
            if item == needle[pointer]:
                pointer += 1
                if pointer == len(needle):
                    return True
        return False
