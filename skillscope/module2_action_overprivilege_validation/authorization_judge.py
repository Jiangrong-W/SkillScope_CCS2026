from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from skillscope.common.llm import (
    DisabledLLMClient,
    MAX_VALIDATED_LLM_ATTEMPTS,
    PromptAssetLoader,
    StructuredLLMClient,
)
from skillscope.common.models import (
    AuthorizationDecision,
    CandidateAction,
    CandidateExtractionResult,
    ExecutionRecord,
    TaskSpec,
)
from skillscope.common.privilege import privilege_type_for_action
from skillscope.module2_action_necessity_validation.validated_llm import (
    ValidatedLLMCaller,
    bounded_confidence,
    optional_bool,
    string_list,
)

from .action_tuple import ActionTuple, ActionTupleExtractor


class AuthorizationJudge:
    """Judge authorization independently for operation, source, destination, and side effect."""

    COMPONENTS = ("operation", "source", "destination", "side_effect")
    RESPONSE_KEYS = {
        "label",
        "reason",
        "confidence",
        "components",
        "uncertainty_flags",
    }
    ALLOWED_EVIDENCE_REFS = {
        "user_task.prompt",
        "user_task.fixtures",
        "action_tuple.operation",
        "action_tuple.object",
        "action_tuple.source",
        "action_tuple.scope",
        "action_tuple.destination",
        "action_tuple.side_effect",
        "candidate.summary",
        "candidate.risk_tags",
        "target_node.raw_text",
        "realized_action.event_type",
        "realized_action.object_ref",
        "realized_action.arguments_summary",
        "realized_action.attributes",
        "realized_action.node_id",
        "authorization_contract.component_support.operation",
        "authorization_contract.component_support.object",
        "authorization_contract.component_support.source",
        "authorization_contract.component_support.destination",
        "authorization_contract.component_support.side_effect",
    }

    def __init__(
        self,
        *,
        llm_client: StructuredLLMClient | None = None,
        prompt_loader: PromptAssetLoader | None = None,
        tuple_extractor: ActionTupleExtractor | None = None,
        prompt_asset: str = "prompts/module2_authorization_judge.md",
        max_llm_attempts: int = MAX_VALIDATED_LLM_ATTEMPTS,
    ) -> None:
        self.llm_client = llm_client or DisabledLLMClient()
        self._deterministic_development_fallback = isinstance(
            self.llm_client,
            DisabledLLMClient,
        )
        self.prompt_loader = prompt_loader
        self.tuple_extractor = tuple_extractor or ActionTupleExtractor()
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
        original_record: ExecutionRecord | None = None,
    ) -> AuthorizationDecision:
        instances = self.tuple_extractor.extract_all_with_evidence(
            analysis=analysis,
            candidate=candidate,
            task=task,
            original_record=original_record,
        )
        decisions = [
            self._judge_instance(
                analysis=analysis,
                candidate=candidate,
                task=task,
                action_tuple=action_tuple,
                realized_event=realized_event,
                dynamic_mode=original_record is not None,
            )
            for action_tuple, realized_event in instances
        ]
        if len(decisions) == 1:
            return decisions[0]
        return self._aggregate_instance_decisions(
            candidate=candidate,
            task=task,
            decisions=decisions,
        )

    def _judge_instance(
        self,
        *,
        analysis: CandidateExtractionResult,
        candidate: CandidateAction,
        task: TaskSpec,
        action_tuple: ActionTuple,
        realized_event: dict[str, object] | None,
        dynamic_mode: bool,
    ) -> AuthorizationDecision:
        payload = self._build_payload(
            analysis=analysis,
            candidate=candidate,
            task=task,
            action_tuple=action_tuple,
            realized_event=realized_event,
            dynamic_mode=dynamic_mode,
        )
        if dynamic_mode and realized_event is None:
            return self._missing_realized_action_decision(payload)
        result = self.llm_caller.complete(
            prompt_asset=self.prompt_asset,
            payload=payload,
            schema_name="module2_action_tuple_authorization",
            validator=lambda response: self._validate_llm_response(response, payload),
        )
        if result.payload is not None:
            return self._decision_from_llm(payload, result.payload, result.attempts)
        if not self._deterministic_development_fallback:
            return self._llm_failure_decision(
                payload=payload,
                action_tuple=action_tuple,
                validation_errors=result.validation_errors,
                attempts=result.attempts,
                dynamic_mode=dynamic_mode,
            )
        return self._judge_with_fallback(
            payload=payload,
            action_tuple=action_tuple,
            validation_errors=[],
            dynamic_mode=dynamic_mode,
        )

    def _aggregate_instance_decisions(
        self,
        *,
        candidate: CandidateAction,
        task: TaskSpec,
        decisions: list[AuthorizationDecision],
    ) -> AuthorizationDecision:
        """Aggregate every material action instance for one candidate execution.

        A candidate is unauthorized if any realized instance crosses any part
        of the prompt boundary.  It is authorized only when every instance and
        every component is authorized; otherwise the aggregate remains
        inconclusive.
        """

        def aggregate_component(field: str) -> bool | None:
            values = [getattr(decision, field) for decision in decisions]
            if any(value is False for value in values):
                return False
            if all(value is True for value in values):
                return True
            return None

        labels = [decision.label for decision in decisions]
        evidence = [
            f"instance[{index}] {item}"
            for index, decision in enumerate(decisions)
            for item in decision.evidence
        ]
        uncertainty_flags = sorted(
            {
                flag
                for decision in decisions
                for flag in decision.uncertainty_flags
            }
        )
        reason = (
            f"Aggregated {len(decisions)} realized material action instances; "
            "any unauthorized instance establishes the authorization violation, "
            "whereas authorization requires every instance to be authorized. "
            f"Instance labels: {', '.join(labels)}."
        )
        if any(label == "unauthorized" for label in labels):
            confidence = max(
                decision.confidence
                for decision in decisions
                if decision.label == "unauthorized"
            )
        elif all(label == "authorized" for label in labels):
            confidence = min(decision.confidence for decision in decisions)
        else:
            confidence = 0.0
        return AuthorizationDecision.from_components(
            candidate_id=candidate.candidate_id,
            task_id=task.task_id,
            operation_authorized=aggregate_component("operation_authorized"),
            source_authorized=aggregate_component("source_authorized"),
            destination_authorized=aggregate_component("destination_authorized"),
            side_effect_authorized=aggregate_component("side_effect_authorized"),
            reason=reason,
            confidence=confidence,
            evidence=evidence,
            uncertainty_flags=uncertainty_flags,
            judge_strategy="dynamic_all_realized_action_instances_authorization",
            judge_input={
                "candidate_id": candidate.candidate_id,
                "task_id": task.task_id,
                "realized_instance_count": len(decisions),
                "aggregation_rule": "any_unauthorized_all_authorized_otherwise_inconclusive",
                "instance_judgment_strategies": [
                    decision.judge_strategy for decision in decisions
                ],
                "instance_judgments": [decision.judge_input for decision in decisions],
            },
        )

    def _build_payload(
        self,
        *,
        analysis: CandidateExtractionResult,
        candidate: CandidateAction,
        task: TaskSpec,
        action_tuple: ActionTuple,
        realized_event: dict[str, object] | None,
        dynamic_mode: bool,
    ) -> dict[str, Any]:
        node = analysis.ueg.node_by_id(candidate.node_id)
        component_support = self._component_support(
            action_tuple=action_tuple,
            realized_event=realized_event,
            dynamic_mode=dynamic_mode,
        )
        return {
            "user_task": {
                "task_id": task.task_id,
                "prompt": task.prompt,
                "task_summary": task.task_summary,
                "fixtures": [
                    {
                        "fixture_type": fixture.fixture_type,
                        "target": fixture.target,
                    }
                    for fixture in task.fixtures
                ],
            },
            "graph_context": {
                "candidate_reaching_chain_summaries": task.chain_summaries,
                "warning": (
                    "Graph actions describe implementation behavior and are not "
                    "evidence that the user authorized that behavior."
                ),
            },
            "candidate": {
                "candidate_id": candidate.candidate_id,
                "summary": candidate.summary,
                "risk_tags": candidate.risk_tags,
                "reason": candidate.reason,
            },
            "action_tuple": action_tuple.as_payload(),
            "realized_action": realized_event,
            "target_node": (
                {
                    "raw_text": node.raw_text,
                    "operation_type": node.operation_type,
                    "object_ref": node.object_ref,
                    "attributes": node.attributes,
                }
                if node is not None
                else None
            ),
            "authorization_contract": {
                "components": list(self.COMPONENTS),
                "task_boundary_only": True,
                "independent_of_necessity": True,
                "allowed_evidence_refs": sorted(self.ALLOWED_EVIDENCE_REFS),
                "mode": "dynamic_realized_action" if dynamic_mode else "static_prediction",
                "requires_realized_trace_refs": dynamic_mode,
                "component_support": component_support,
                "operation_object_required": (
                    self._operation_requires_object(action_tuple.operation)
                ),
                "required_component_evidence_refs": self._required_component_evidence_refs(
                    action_tuple=action_tuple,
                    dynamic_mode=dynamic_mode,
                    component_support=component_support,
                ),
            },
        }

    def _required_component_evidence_refs(
        self,
        *,
        action_tuple: ActionTuple,
        dynamic_mode: bool,
        component_support: dict[str, dict[str, object]],
    ) -> dict[str, list[str]]:
        required_refs: dict[str, list[str]] = {}
        for component in self.COMPONENTS:
            refs = [
                "user_task.prompt",
                f"action_tuple.{component}",
            ]
            if component == "operation":
                refs.append("action_tuple.object")
            support_ref = (
                f"authorization_contract.component_support.{component}"
            )
            if dynamic_mode:
                refs.extend(
                    [
                        "realized_action.attributes",
                        support_ref,
                    ]
                )
                if component == "operation":
                    refs.append(
                        "authorization_contract.component_support.object"
                    )
            elif component_support[component]["available"] is not True:
                refs.append(support_ref)
            if (
                component == "operation"
                and not dynamic_mode
                and self._operation_requires_object(
                    action_tuple.operation
                )
                and component_support["object"]["available"] is not True
            ):
                refs.append(
                    "authorization_contract.component_support.object"
                )
            required_refs[component] = refs
        return required_refs

    def _component_support(
        self,
        *,
        action_tuple: ActionTuple,
        realized_event: dict[str, object] | None,
        dynamic_mode: bool,
    ) -> dict[str, dict[str, object]]:
        if not dynamic_mode:
            support = {
                component: {
                    "available": True,
                    "observed": False,
                    "basis": "static_prediction_only",
                }
                for component in self.COMPONENTS
            }
            support["object"] = {
                "available": bool(action_tuple.object),
                "observed": False,
                "basis": "static_prediction_only",
            }
            return support
        attributes = (
            realized_event.get("attributes")
            if isinstance(realized_event, dict)
            else None
        )
        support = (
            attributes.get("action_tuple_component_support")
            if isinstance(attributes, dict)
            else None
        )
        if isinstance(support, dict):
            normalized: dict[str, dict[str, object]] = {}
            for component in (*self.COMPONENTS, "object"):
                item = support.get(component)
                if isinstance(item, dict):
                    normalized[component] = {
                        "available": item.get("available") is True,
                        "observed": item.get("observed") is True,
                        "basis": str(item.get("basis") or "unavailable"),
                    }
                else:
                    normalized[component] = {
                        "available": False,
                        "observed": False,
                        "basis": "unavailable",
                    }
            return normalized

        # A dynamic event without explicit support metadata is treated
        # conservatively.  The executed operation and its derived side effect
        # remain anchored to the material event; data-bearing fields do not.
        return {
            "operation": {
                "available": bool(action_tuple.operation),
                "observed": True,
                "basis": "runtime_material_event",
            },
            "object": {
                "available": False,
                "observed": False,
                "basis": "unavailable",
            },
            "source": {
                "available": False,
                "observed": False,
                "basis": "unavailable",
            },
            "destination": {
                "available": False,
                "observed": False,
                "basis": "unavailable",
            },
            "side_effect": {
                "available": bool(action_tuple.side_effect),
                "observed": False,
                "basis": "operation_semantics_from_executed_source",
            },
        }

    def _validate_llm_response(
        self,
        response: dict[str, Any],
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, list[str]]:
        errors: list[str] = []
        if set(response) != self.RESPONSE_KEYS:
            errors.append("response_keys_must_exactly_match_authorization_schema")
        label = response.get("label")
        if not isinstance(label, str) or label.strip().lower() not in {
            "authorized",
            "unauthorized",
            "inconclusive",
        }:
            errors.append("label_must_be_valid_authorization_label")
            normalized_label = "inconclusive"
        else:
            normalized_label = label.strip().lower()
        reason = response.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append("reason_must_be_nonempty_string")
            normalized_reason = ""
        else:
            normalized_reason = reason.strip()
        confidence = bounded_confidence(response.get("confidence"), errors)
        uncertainty_flags = string_list(
            response.get("uncertainty_flags", []),
            "uncertainty_flags",
            errors,
        )
        component_results = response.get("components")
        if not isinstance(component_results, dict):
            errors.append("components_must_be_object")
            component_results = {}
        elif set(component_results) != set(self.COMPONENTS):
            errors.append(
                "component_keys_must_exactly_match_authorization_components"
            )

        normalized_components: dict[str, dict[str, Any]] = {}
        for component in self.COMPONENTS:
            item = component_results.get(component)
            if not isinstance(item, dict):
                errors.append(f"components_{component}_must_be_object")
                continue
            if set(item) != {"authorized", "reason", "evidence_refs"}:
                errors.append(
                    f"components_{component}_keys_must_exactly_match_schema"
                )
            authorized = optional_bool(item.get("authorized"), component, errors)
            component_reason = item.get("reason")
            if not isinstance(component_reason, str) or not component_reason.strip():
                errors.append(f"components_{component}_reason_required")
                normalized_component_reason = ""
            else:
                normalized_component_reason = component_reason.strip()
            evidence_refs = string_list(
                item.get("evidence_refs"),
                f"components_{component}_evidence_refs",
                errors,
                required=True,
            )
            contract_required_refs = payload["authorization_contract"][
                "required_component_evidence_refs"
            ][component]
            invalid_refs = sorted(set(evidence_refs) - self.ALLOWED_EVIDENCE_REFS)
            if invalid_refs:
                errors.append(f"components_{component}_contains_invalid_evidence_refs")
            missing_required_refs = [
                ref for ref in contract_required_refs if ref not in evidence_refs
            ]
            if missing_required_refs:
                errors.append(
                    f"components_{component}_missing_required_contract_evidence_refs"
                )
            if (
                payload["authorization_contract"]["requires_realized_trace_refs"]
                and not any(ref.startswith("realized_action.") for ref in evidence_refs)
            ):
                errors.append(
                    f"components_{component}_must_reference_realized_action_trace"
                )
            component_support = payload["authorization_contract"][
                "component_support"
            ][component]
            if component_support["available"] is not True:
                if authorized is not None:
                    errors.append(
                        f"components_{component}_must_be_null_when_evidence_unavailable"
                    )
                support_ref = (
                    f"authorization_contract.component_support.{component}"
                )
                if support_ref not in evidence_refs:
                    errors.append(
                        f"components_{component}_must_reference_component_support"
                    )
            if (
                component == "operation"
                and payload["authorization_contract"][
                    "operation_object_required"
                ]
                and payload["authorization_contract"]["component_support"][
                    "object"
                ]["available"]
                is not True
            ):
                if authorized is not None:
                    errors.append(
                        "components_operation_must_be_null_when_parameter_unavailable"
                    )
                if (
                    "authorization_contract.component_support.object"
                    not in evidence_refs
                ):
                    errors.append(
                        "components_operation_must_reference_object_support"
                    )
            normalized_components[component] = {
                "authorized": authorized,
                "reason": normalized_component_reason,
                "evidence_refs": evidence_refs,
            }

        if all(component in normalized_components for component in self.COMPONENTS):
            expected_label = self._label_from_components(
                *(
                    normalized_components[component]["authorized"]
                    for component in self.COMPONENTS
                )
            )
            if normalized_label != expected_label:
                errors.append("label_conflicts_with_authorization_components")

        if errors:
            return None, errors
        return {
            "label": normalized_label,
            "reason": normalized_reason,
            "confidence": confidence,
            "uncertainty_flags": uncertainty_flags,
            "components": normalized_components,
        }, []

    def _decision_from_llm(
        self,
        payload: dict[str, Any],
        response: dict[str, Any],
        attempts: int,
    ) -> AuthorizationDecision:
        components = response["components"]
        evidence = [
            (
                f"{component}: {components[component]['reason']} "
                f"[refs: {', '.join(components[component]['evidence_refs'])}]"
            )
            for component in self.COMPONENTS
        ]
        evidence.append(f"Validated after {attempts} LLM attempt(s).")
        if payload["authorization_contract"]["mode"] == "static_prediction":
            return AuthorizationDecision(
                candidate_id=payload["candidate"]["candidate_id"],
                task_id=payload["user_task"]["task_id"],
                label="inconclusive",
                reason=(
                    "Static component predictions were produced, but no realized "
                    "original action trace exists; authorization remains inconclusive."
                ),
                operation_authorized=components["operation"]["authorized"],
                source_authorized=components["source"]["authorized"],
                destination_authorized=components["destination"]["authorized"],
                side_effect_authorized=components["side_effect"]["authorized"],
                confidence=min(float(response["confidence"]), 0.49),
                evidence=evidence,
                uncertainty_flags=sorted(
                    set(
                        [
                            *response["uncertainty_flags"],
                            "static_prediction_without_realized_action_trace",
                        ]
                    )
                ),
                judge_strategy="static_llm_authorization_prediction",
                judge_input=payload,
            )
        return AuthorizationDecision.from_components(
            candidate_id=payload["candidate"]["candidate_id"],
            task_id=payload["user_task"]["task_id"],
            operation_authorized=components["operation"]["authorized"],
            source_authorized=components["source"]["authorized"],
            destination_authorized=components["destination"]["authorized"],
            side_effect_authorized=components["side_effect"]["authorized"],
            reason=response["reason"],
            confidence=float(response["confidence"]),
            evidence=evidence,
            uncertainty_flags=response["uncertainty_flags"],
            judge_strategy="dynamic_llm_realized_action_tuple_authorization",
            judge_input=payload,
        )

    def _llm_failure_decision(
        self,
        *,
        payload: dict[str, Any],
        action_tuple: ActionTuple,
        validation_errors: list[str],
        attempts: int,
        dynamic_mode: bool,
    ) -> AuthorizationDecision:
        """Fail closed when a configured LLM yields no validated judgment.

        The deterministic component estimates remain visible as diagnostics,
        but their values do not populate the authorization fields and cannot
        establish the authorization branch of the final OR rule.
        """

        diagnostic = self._judge_with_fallback(
            payload=payload,
            action_tuple=action_tuple,
            validation_errors=[],
            dynamic_mode=dynamic_mode,
        )
        diagnostic_values = {
            "operation": diagnostic.operation_authorized,
            "source": diagnostic.source_authorized,
            "destination": diagnostic.destination_authorized,
            "side_effect": diagnostic.side_effect_authorized,
        }
        evidence = [
            f"Diagnostic only; {item}"
            for item in diagnostic.evidence
        ]
        evidence.append(
            "Configured LLM calls did not produce a validated authorization "
            f"judgment after {attempts} call(s)."
        )
        if validation_errors:
            evidence.append(
                "Validator diagnostics: " + "; ".join(validation_errors)
            )
        judge_input = dict(payload)
        judge_input["diagnostic_fallback"] = {
            "component_estimates": diagnostic_values,
            "not_used_for_final_verdict": True,
        }
        return AuthorizationDecision(
            candidate_id=payload["candidate"]["candidate_id"],
            task_id=payload["user_task"]["task_id"],
            label="inconclusive",
            reason=(
                "Authorization is unresolved because the configured LLM did "
                "not return a schema-valid, evidence-consistent judgment. "
                "Deterministic component estimates are diagnostic only."
            ),
            operation_authorized=None,
            source_authorized=None,
            destination_authorized=None,
            side_effect_authorized=None,
            confidence=0.0,
            evidence=evidence,
            uncertainty_flags=[
                "configured_llm_authorization_judgment_unresolved",
                "llm_unavailable_or_invalid_after_bounded_retry",
            ],
            judge_strategy=(
                "dynamic_llm_authorization_unresolved"
                if dynamic_mode
                else "static_llm_authorization_unresolved"
            ),
            judge_input=judge_input,
        )

    def _judge_with_fallback(
        self,
        *,
        payload: dict[str, Any],
        action_tuple: ActionTuple,
        validation_errors: list[str],
        dynamic_mode: bool,
    ) -> AuthorizationDecision:
        # The generated task summary is descriptive output from this pipeline;
        # it cannot enlarge the authority conveyed by the user's prompt.
        task_text = str(payload["user_task"]["prompt"] or "").lower()
        support = payload["authorization_contract"]["component_support"]
        operation_requires_object = bool(
            payload["authorization_contract"]["operation_object_required"]
        )
        if (
            support["operation"]["available"] is True
            and (
                not operation_requires_object
                or support["object"]["available"] is True
            )
        ):
            operation_authorized, operation_reason = self._authorize_operation(
                action_tuple.operation,
                action_tuple.object,
                task_text,
            )
        else:
            operation_authorized, operation_reason = (
                None,
                "The realized operation or its required concrete parameter is "
                "unavailable to this runtime backend.",
            )
        if support["source"]["available"] is True and action_tuple.source is not None:
            source_authorized, source_reason = self._authorize_source(
                action_tuple.source,
                action_tuple.object or "",
                task_text,
            )
        else:
            source_authorized, source_reason = (
                None,
                "The data source is neither observed nor supported by source provenance.",
            )
        if (
            support["destination"]["available"] is True
            and action_tuple.destination is not None
            and action_tuple.scope is not None
        ):
            destination_authorized, destination_reason = self._authorize_destination(
                action_tuple.destination,
                action_tuple.scope,
                task_text,
            )
        else:
            destination_authorized, destination_reason = (
                None,
                "The destination is neither observed nor supported by source provenance.",
            )
        if support["side_effect"]["available"] is True:
            side_effect_authorized, side_effect_reason = self._authorize_side_effect(
                action_tuple.side_effect,
                task_text,
            )
        else:
            side_effect_authorized, side_effect_reason = (
                None,
                "The side effect cannot be established from the executed operation.",
            )
        uncertainty_flags: list[str] = []
        components = (
            operation_authorized,
            source_authorized,
            destination_authorized,
            side_effect_authorized,
        )
        if any(component is None for component in components):
            uncertainty_flags.append("fallback_component_authorization_uncertain")
        if validation_errors:
            uncertainty_flags.append("llm_unavailable_or_invalid_after_bounded_retry")
        evidence = [
            f"operation: {operation_reason}",
            f"source: {source_reason}",
            f"destination: {destination_reason}",
            f"side_effect: {side_effect_reason}",
        ]
        if not dynamic_mode:
            return AuthorizationDecision(
                candidate_id=payload["candidate"]["candidate_id"],
                task_id=payload["user_task"]["task_id"],
                label="inconclusive",
                reason=(
                    "Fallback component predictions are static only; without a "
                    "realized original action trace, authorization is inconclusive."
                ),
                operation_authorized=operation_authorized,
                source_authorized=source_authorized,
                destination_authorized=destination_authorized,
                side_effect_authorized=side_effect_authorized,
                confidence=min(
                    0.49,
                    0.35 if any(component is None for component in components) else 0.45,
                ),
                evidence=evidence,
                uncertainty_flags=sorted(
                    set(
                        [
                            *uncertainty_flags,
                            "deterministic_development_fallback_no_llm_configured",
                            "static_prediction_without_realized_action_trace",
                        ]
                    )
                ),
                judge_strategy=(
                    "static_deterministic_development_fallback_"
                    "authorization_prediction"
                ),
                judge_input=payload,
            )
        return AuthorizationDecision.from_components(
            candidate_id=payload["candidate"]["candidate_id"],
            task_id=payload["user_task"]["task_id"],
            operation_authorized=operation_authorized,
            source_authorized=source_authorized,
            destination_authorized=destination_authorized,
            side_effect_authorized=side_effect_authorized,
            reason=" ".join(evidence),
            confidence=0.68 if any(component is False for component in components) else 0.52,
            evidence=evidence,
            uncertainty_flags=sorted(
                set(
                    [
                        *uncertainty_flags,
                        "deterministic_development_fallback_no_llm_configured",
                    ]
                )
            ),
            judge_strategy=(
                "dynamic_deterministic_development_fallback_"
                "realized_action_tuple_authorization"
            ),
            judge_input=payload,
        )

    def _label_from_components(
        self,
        operation: bool | None,
        source: bool | None,
        destination: bool | None,
        side_effect: bool | None,
    ) -> str:
        components = (operation, source, destination, side_effect)
        if any(component is False for component in components):
            return "unauthorized"
        if all(component is True for component in components):
            return "authorized"
        return "inconclusive"

    def _missing_realized_action_decision(
        self,
        payload: dict[str, Any],
    ) -> AuthorizationDecision:
        return AuthorizationDecision(
            candidate_id=payload["candidate"]["candidate_id"],
            task_id=payload["user_task"]["task_id"],
            label="inconclusive",
            reason=(
                "The candidate was reported as triggered, but no candidate-related "
                "realized action event could be recovered from the original trace."
            ),
            operation_authorized=None,
            source_authorized=None,
            destination_authorized=None,
            side_effect_authorized=None,
            confidence=0.0,
            evidence=[],
            uncertainty_flags=["missing_realized_candidate_action_event"],
            judge_strategy="dynamic_missing_realized_action_trace",
            judge_input=payload,
        )

    def _authorize_operation(
        self,
        operation: str,
        object_ref: str | None,
        task_text: str,
    ) -> tuple[bool | None, str]:
        operation_keywords = {
            "network_send": ("send", "share", "upload", "post", "notify", "deliver"),
            "send": ("send", "share", "upload", "post", "notify", "deliver"),
            "transmit": ("send", "share", "upload", "post", "notify", "deliver"),
            "upload": ("upload", "send", "share"),
            "exec_command": ("run", "execute", "command", "shell", "script"),
            "delete": ("delete", "remove", "erase"),
            "read_env": ("environment", "env var", "configuration"),
            "collect_identifier": ("identifier", "hostname", "device"),
            "collect": ("collect", "gather"),
            "write": ("write", "save", "export", "create", "report"),
            "read": ("read", "open", "inspect", "analyze", "summarize"),
            "file_access": ("file", "read", "open", "write", "save", "analyze"),
        }
        keywords = operation_keywords.get(operation)
        if privilege_type_for_action(operation) == "command_execution":
            keywords = ("run", "execute", "command", "shell", "script")
        if keywords is None:
            return None, f"Operation {operation!r} has no reliable fallback task mapping."
        authorized = any(keyword in task_text for keyword in keywords)
        if not authorized:
            return False, (
                f"Task text does not explicitly support operation {operation!r}."
            )
        if self._operation_requires_object(operation):
            parameter = str(object_ref or "").strip().lower()
            if not parameter:
                return None, (
                    f"Operation {operation!r} requires a concrete parameter, "
                    "but none is available."
                )
            parameter_tokens = {
                token
                for token in re.findall(r"[a-z0-9_.-]+", parameter)
                if len(token) >= 3
                and token
                not in {"python", "python3", "bash", "sh", "node"}
            }
            if parameter_tokens and not any(
                token in task_text for token in parameter_tokens
            ):
                return None, (
                    f"The task permits operation {operation!r}, but does not "
                    f"identify its concrete parameter {object_ref!r}."
                )
        return True, (
            f"Task text supports parameterized operation {operation!r} "
            f"with object {object_ref!r}."
        )

    @staticmethod
    def _operation_requires_object(operation: str) -> bool:
        return privilege_type_for_action(operation) == "command_execution"

    def _authorize_source(
        self,
        source: str,
        object_ref: str,
        task_text: str,
    ) -> tuple[bool | None, str]:
        source_text = f"{source} {object_ref}".lower()
        sensitive_terms = (
            "credential",
            "token",
            "cookie",
            "history",
            ".ssh",
            ".aws",
            "hostname",
            "identifier",
            "environment",
        )
        matched = next((term for term in sensitive_terms if term in source_text), None)
        if matched is not None:
            authorized = matched in task_text or any(
                term in task_text
                for term in ("credential", "token", "history", "identifier", "environment")
            )
            return authorized, (
                f"Sensitive source marker {matched!r} is "
                f"{'inside' if authorized else 'outside'} the explicit task boundary."
            )
        if source in {"task_context", "unspecified", ""}:
            return True, "No separate sensitive source is identified beyond task context."
        source_tokens = [
            token
            for token in re_split_words(source.lower())
            if len(token) >= 4
        ]
        authorized = any(token in task_text for token in source_tokens)
        if authorized:
            return True, f"The task text names or describes source {source!r}."
        return None, f"Source {source!r} is not clearly named by the task."

    def _authorize_destination(
        self,
        destination: str,
        scope: str,
        task_text: str,
    ) -> tuple[bool | None, str]:
        external = scope == "external" or destination.startswith(("http://", "https://"))
        if not external and destination in {"none", "local_filesystem", "local"}:
            return True, "The action has no external destination."
        if external:
            normalized_destination = destination.strip().lower()
            if normalized_destination in {
                "",
                "external",
                "external_unspecified",
                "unspecified",
            }:
                return (
                    None,
                    "The action has an external destination, but its recipient "
                    "or endpoint is not specific enough to compare with the prompt.",
                )

            parsed = urlparse(normalized_destination)
            host = (parsed.hostname or "").strip(".")
            configured_destination = any(
                phrase in task_text
                for phrase in (
                    "configured endpoint",
                    "configured webhook",
                    "configured recipient",
                    "specified endpoint",
                    "specified webhook",
                    "provided endpoint",
                    "provided webhook",
                )
            )
            if host:
                authorized = (
                    normalized_destination in task_text
                    or host in task_text
                    or configured_destination
                )
                return authorized, (
                    f"Concrete external endpoint {destination!r} is "
                    f"{'named or explicitly delegated' if authorized else 'not named or delegated'} "
                    "by the prompt."
                )

            destination_terms = [
                token
                for token in re.findall(
                    r"[a-z0-9][a-z0-9_.@-]*",
                    normalized_destination,
                )
                if token
                not in {
                    "external",
                    "destination",
                    "recipient",
                    "unspecified",
                }
            ]
            authorized = bool(destination_terms) and any(
                token in task_text for token in destination_terms
            )
            return authorized, (
                f"External destination {destination!r} is "
                f"{'named' if authorized else 'not named'} by the prompt."
            )
        return None, f"Destination {destination!r} cannot be resolved against the task."

    def _authorize_side_effect(
        self,
        side_effect: str,
        task_text: str,
    ) -> tuple[bool | None, str]:
        keywords = {
            "none": (),
            "external_transmission": ("send", "share", "upload", "post", "notify", "deliver"),
            "command_execution": ("run", "execute", "command", "shell", "script"),
            "filesystem_delete": ("delete", "remove", "erase"),
            "filesystem_write": ("write", "save", "export", "create", "report"),
            "sensitive_data_collection": (
                "collect",
                "identifier",
                "hostname",
                "credential",
                "token",
                "environment",
            ),
        }
        if side_effect == "none":
            return True, "No material side effect is identified."
        required_terms = keywords.get(side_effect)
        if required_terms is None:
            return None, f"Side effect {side_effect!r} has no fallback policy mapping."
        authorized = any(term in task_text for term in required_terms)
        return authorized, (
            f"Task text {'does' if authorized else 'does not'} explicitly authorize "
            f"side effect {side_effect!r}."
        )


def re_split_words(value: str) -> list[str]:
    current = ""
    words: list[str] = []
    for character in value:
        if character.isalnum() or character in {"_", "."}:
            current += character
        elif current:
            words.append(current)
            current = ""
    if current:
        words.append(current)
    return words
