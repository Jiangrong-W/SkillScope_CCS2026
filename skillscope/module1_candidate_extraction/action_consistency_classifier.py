from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import json

from skillscope.common.llm import (
    DisabledLLMClient,
    JSONResponseContract,
    MAX_VALIDATED_LLM_ATTEMPTS,
    PromptAssetLoader,
    StructuredLLMClient,
    complete_validated_json,
)
from skillscope.common.models import SkillProfile, UEGNode


@dataclass(slots=True)
class ActionConsistencyInput:
    declared_skill_profile: dict[str, Any]
    normalized_action_summary: str
    upstream_action_chain: list[str]
    downstream_action_chain: list[str] = field(default_factory=list)
    predicate_context: list[str] = field(default_factory=list)
    action_context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConsistencyAssessment:
    suspicious: bool
    label: str
    reason: str
    confidence: float
    input_payload: dict[str, Any] = field(default_factory=dict)
    strategy: str = "unknown"
    ambiguity_kind: str | None = None


class ActionConsistencyClassifier:
    RESPONSE_KEYS = {
        "label",
        "ambiguity_kind",
        "confidence",
        "reason",
        "evidence_refs",
    }
    ALLOWED_EVIDENCE_REFS = {
        "declared_skill_profile.summary",
        "declared_skill_profile.declared_capabilities",
        "declared_skill_profile.declared_outputs",
        "declared_skill_profile.declared_data_scope",
        "declared_skill_profile.declared_execution_scope",
        "normalized_action_summary",
        "upstream_action_chain",
        "downstream_action_chain",
        "predicate_context",
        "action_context.operation_type",
        "action_context.object_ref",
        "action_context.risk_tags",
        "action_context.provenance",
        "action_context.source_file",
        "action_context.source_range",
        "action_context.context_node_ids",
    }
    REQUIRED_EVIDENCE_REFS = {
        "declared_skill_profile.summary",
        "normalized_action_summary",
        "upstream_action_chain",
        "downstream_action_chain",
        "predicate_context",
        "action_context.operation_type",
        "action_context.object_ref",
        "action_context.provenance",
    }

    def __init__(
        self,
        llm_client: StructuredLLMClient | None = None,
        prompt_loader: PromptAssetLoader | None = None,
        prompt_asset: str = "prompts/module1_action_consistency_classification.md",
    ) -> None:
        self.llm_client = llm_client or DisabledLLMClient()
        self.prompt_loader = prompt_loader
        self.prompt_asset = prompt_asset

    def build_input(
        self,
        profile: SkillProfile,
        node: UEGNode,
        upstream_action_chain: list[str],
        downstream_action_chain: list[str] | None = None,
        predicate_context: list[str] | None = None,
        context_node_ids: list[str] | None = None,
    ) -> ActionConsistencyInput:
        profile_payload = {
            "name": profile.name,
            "description": profile.description,
            "use_when": profile.use_when,
            "summary": profile.summary,
            "declared_capabilities": profile.declared_capabilities,
            "declared_outputs": profile.declared_outputs,
            "declared_data_scope": profile.declared_data_scope,
            "declared_execution_scope": profile.declared_execution_scope,
        }
        action_context = {
            "node_id": node.node_id,
            "layer": node.layer,
            "node_type": node.node_type,
            "raw_text": node.raw_text,
            "operation_type": node.operation_type,
            "object_ref": node.object_ref,
            "risk_tags": node.risk_tags,
            "source_file": node.source_file,
            "source_range": {
                "start_line": node.source_range.start_line,
                "end_line": node.source_range.end_line,
            }
            if node.source_range is not None
            else None,
            "provenance": node.attributes.get("provenance"),
            "context_node_ids": list(context_node_ids or []),
            "prompt_asset": self.prompt_asset,
        }
        return ActionConsistencyInput(
            declared_skill_profile=profile_payload,
            normalized_action_summary=node.summary,
            upstream_action_chain=upstream_action_chain,
            downstream_action_chain=list(downstream_action_chain or []),
            predicate_context=list(predicate_context or []),
            action_context=action_context,
        )

    def classify(
        self,
        profile: SkillProfile,
        node: UEGNode,
        upstream_action_chain: list[str],
        downstream_action_chain: list[str] | None = None,
        predicate_context: list[str] | None = None,
        context_node_ids: list[str] | None = None,
    ) -> ConsistencyAssessment:
        payload = self.build_input(
            profile,
            node,
            upstream_action_chain,
            downstream_action_chain,
            predicate_context,
            context_node_ids,
        )
        llm_assessment = self._classify_with_llm(payload)
        if llm_assessment is not None:
            return llm_assessment
        return self._classify_heuristically(payload)

    def _classify_with_llm(self, payload: ActionConsistencyInput) -> ConsistencyAssessment | None:
        if self.prompt_loader is None:
            return None
        evidence_payload = self._to_dict(payload)
        grounded_evidence_ids = self._grounded_evidence_ids(
            evidence_payload
        )
        request_payload = {
            **evidence_payload,
            "classification_contract": {
                "allowed_evidence_refs": sorted(grounded_evidence_ids),
                "required_evidence_refs": sorted(
                    self.REQUIRED_EVIDENCE_REFS
                ),
                "response_keys": sorted(self.RESPONSE_KEYS),
            },
        }
        try:
            response = complete_validated_json(
                self.llm_client,
                system_prompt=self.prompt_loader.load(self.prompt_asset),
                user_prompt=json.dumps(
                    request_payload,
                    ensure_ascii=False,
                    indent=2,
                ),
                schema_name="action_consistency_classification",
                contract=JSONResponseContract(
                    required_fields=tuple(sorted(self.RESPONSE_KEYS)),
                    non_empty_string_fields=("reason",),
                    enum_fields={
                        "label": {"related", "suspicious", "ambiguous"},
                        "ambiguity_kind": {
                            "none",
                            "task_ambiguous",
                            "analysis_uncertain",
                            "underspecified",
                        },
                    },
                    evidence_field="evidence_refs",
                    grounded_evidence_ids=grounded_evidence_ids,
                    consistency_checks=(
                        self._validate_classifier_response,
                    ),
                ),
                max_attempts=MAX_VALIDATED_LLM_ATTEMPTS,
            )
        except RuntimeError:
            return None

        label = str(response.get("label") or "").strip().lower()
        if label not in {"related", "suspicious", "ambiguous"}:
            return None

        reason = str(response.get("reason") or "").strip()
        confidence = float(response.get("confidence") or 0.0)
        return ConsistencyAssessment(
            suspicious=label == "suspicious",
            label=label,
            reason=reason or "The configured classifier did not supply a reason.",
            confidence=max(0.0, min(1.0, confidence)),
            input_payload=self._to_dict(payload),
            strategy="llm_validated",
            ambiguity_kind=(
                None
                if response.get("ambiguity_kind") == "none"
                else str(response.get("ambiguity_kind") or "")
            ),
        )

    def _grounded_evidence_ids(
        self,
        payload: dict[str, Any],
    ) -> set[str]:
        """Enumerate exact JSON paths that exist in the supplied evidence.

        Models sometimes cite a concrete provenance leaf such as
        ``action_context.provenance.origin`` rather than its parent object.
        Both are grounded when that exact path exists; fabricated descendants
        remain outside this finite set and are rejected.
        """

        output: set[str] = set()

        def visit(value: object, prefix: str) -> None:
            output.add(prefix)
            if not isinstance(value, dict):
                return
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                visit(child, child_prefix)

        for key, value in payload.items():
            visit(value, str(key))
        return output

    def _classify_heuristically(self, payload: ActionConsistencyInput) -> ConsistencyAssessment:
        profile = payload.declared_skill_profile
        profile_text = " ".join(
            [
                str(profile.get("summary") or ""),
                " ".join(profile.get("declared_capabilities") or []),
                " ".join(profile.get("declared_execution_scope") or []),
                " ".join(profile.get("declared_data_scope") or []),
                " ".join(profile.get("declared_outputs") or []),
            ]
        ).lower()
        summary = payload.normalized_action_summary.lower()
        upstream = " ".join(payload.upstream_action_chain).lower()
        downstream = " ".join(payload.downstream_action_chain).lower()
        predicates = " ".join(payload.predicate_context).lower()
        risk_tags = set(payload.action_context.get("risk_tags") or [])
        raw_text = str(payload.action_context.get("raw_text") or "").lower()
        operation_type = str(payload.action_context.get("operation_type") or "").lower()
        object_ref = str(payload.action_context.get("object_ref") or "").strip()
        explicit_context_guard = any(
            phrase in predicates
            for phrase in (
                "user request",
                "user asks",
                "explicitly request",
                "consent",
                "external sync",
                "用户请求",
                "明确要求",
                "用户同意",
            )
        )
        vague_context_guard = any(
            phrase in predicates
            for phrase in (
                "if needed",
                "when appropriate",
                "as necessary",
                "if configured",
                "必要时",
                "适当时",
                "如已配置",
            )
        )

        if "analysis_gap" in risk_tags or operation_type in {"parse_error", "unparsed_script"}:
            return self._assessment(
                suspicious=False,
                label="ambiguous",
                reason=(
                    "The action cannot be fully normalized from the available parser evidence, "
                    "so it is retained for downstream validation rather than screened out."
                ),
                confidence=0.5,
                payload=payload,
                ambiguity_kind="analysis_uncertain",
            )

        if risk_tags and vague_context_guard:
            return self._assessment(
                suspicious=False,
                label="ambiguous",
                reason=(
                    "The potentially sensitive action is guarded only by a vague condition, "
                    "so the bidirectional context does not establish whether it is authorized."
                ),
                confidence=0.53,
                payload=payload,
                ambiguity_kind="task_ambiguous",
            )

        if (
            "network" in risk_tags
            and explicit_context_guard
            and not any(
                word in profile_text
                for word in ("network", "telegram", "api", "upload", "send", "share")
            )
        ):
            return self._assessment(
                suspicious=False,
                label="ambiguous",
                reason=(
                    "The predicate suggests a user-conditioned external action, but the declared "
                    "profile does not establish that external scope; the action is retained."
                ),
                confidence=0.56,
                payload=payload,
                ambiguity_kind="task_ambiguous",
            )

        if "network" in risk_tags and not any(word in profile_text for word in ("network", "telegram", "api", "upload", "send", "share")):
            return self._assessment(
                suspicious=True,
                label="suspicious",
                reason="The action reaches an external network boundary but the declared skill profile does not mention network activity.",
                confidence=0.86,
                payload=payload,
            )

        if "command_execution" in risk_tags and not any(word in profile_text for word in ("command", "shell", "script", "run")):
            return self._assessment(
                suspicious=True,
                label="suspicious",
                reason="The action invokes commands, which appears broader than the declared skill functionality.",
                confidence=0.78,
                payload=payload,
            )

        if "sensitive_collection" in risk_tags and not any(word in profile_text for word in ("diagnostic", "system", "monitor", "debug", "logs")):
            return self._assessment(
                suspicious=True,
                label="suspicious",
                reason="The action collects sensitive environment details without matching declared diagnostic capabilities.",
                confidence=0.82,
                payload=payload,
            )

        bidirectional_context = f"{upstream} {downstream}"
        if (
            "network" in risk_tags
            and any(
                word in bidirectional_context
                for word in ("local", "report", "graph", "summary", "analysis")
            )
            and not explicit_context_guard
        ):
            return self._assessment(
                suspicious=True,
                label="suspicious",
                reason=(
                    "The predecessor and successor chains describe a local task flow, but the "
                    "current action introduces an unguarded external transmission step."
                ),
                confidence=0.88,
                payload=payload,
            )

        if "fixed" in raw_text and any(word in raw_text for word in ("telegram", "recipient", "webhook")):
            return self._assessment(
                suspicious=True,
                label="suspicious",
                reason="The action sends data to a fixed external destination, which looks like an auxiliary behavior detached from the declared local purpose.",
                confidence=0.9,
                payload=payload,
            )

        if risk_tags and not object_ref:
            return self._assessment(
                suspicious=False,
                label="ambiguous",
                reason=(
                    "The action crosses a potentially sensitive privilege boundary, but the "
                    "operated object is underspecified in the available evidence."
                ),
                confidence=0.52,
                payload=payload,
                ambiguity_kind="underspecified",
            )

        if any(token in raw_text.split() for token in ("it", "this", "that", "them")) and risk_tags:
            return self._assessment(
                suspicious=False,
                label="ambiguous",
                reason=(
                    "The action uses an unresolved object reference and remains context-dependent; "
                    "it is retained for downstream task-conditioned validation."
                ),
                confidence=0.54,
                payload=payload,
                ambiguity_kind="underspecified",
            )

        if any(word in summary for word in ("send", "upload", "post", "telegram", "webhook")) and not any(
            word in profile_text for word in ("send", "share", "telegram", "webhook", "upload", "network")
        ):
            return self._assessment(
                suspicious=True,
                label="suspicious",
                reason="The normalized action summary describes external transmission that is not supported by the declared skill profile.",
                confidence=0.84,
                payload=payload,
            )

        if (
            operation_type in {"call", "shell_action", "javascript_action", "typescript_action", "instruction_step"}
            and not object_ref
            and len((raw_text or summary).split()) <= 4
        ):
            return self._assessment(
                suspicious=False,
                label="ambiguous",
                reason=(
                    "The normalized action is too underspecified to establish semantic alignment "
                    "from the profile and bidirectional graph context."
                ),
                confidence=0.45,
                payload=payload,
                ambiguity_kind="underspecified",
            )

        return self._assessment(
            suspicious=False,
            label="related",
            reason=(
                "The action is consistent with the current understanding of the skill profile, "
                "its bidirectional action context, and surrounding predicates."
            ),
            confidence=0.58,
            payload=payload,
        )

    def _assessment(
        self,
        *,
        suspicious: bool,
        label: str,
        reason: str,
        confidence: float,
        payload: ActionConsistencyInput,
        ambiguity_kind: str | None = None,
    ) -> ConsistencyAssessment:
        return ConsistencyAssessment(
            suspicious=suspicious,
            label=label,
            reason=reason,
            confidence=confidence,
            input_payload=self._to_dict(payload),
            strategy="heuristic_fallback",
            ambiguity_kind=ambiguity_kind,
        )

    def _to_dict(self, payload: ActionConsistencyInput) -> dict[str, Any]:
        return {
            "declared_skill_profile": payload.declared_skill_profile,
            "normalized_action_summary": payload.normalized_action_summary,
            "upstream_action_chain": payload.upstream_action_chain,
            "downstream_action_chain": payload.downstream_action_chain,
            "predicate_context": payload.predicate_context,
            "action_context": payload.action_context,
        }

    def _validate_confidence(self, response: dict[str, Any]) -> str | None:
        confidence = response.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return "field 'confidence' must be a number between 0 and 1"
        if not 0.0 <= float(confidence) <= 1.0:
            return "field 'confidence' must be a number between 0 and 1"
        return None

    def _validate_classifier_response(
        self,
        response: dict[str, Any],
    ) -> str | None:
        if set(response) != self.RESPONSE_KEYS:
            return "classifier response keys must exactly match the schema"
        label = response.get("label")
        ambiguity_kind = response.get("ambiguity_kind")
        if label == "ambiguous" and ambiguity_kind == "none":
            return "ambiguous responses must identify an ambiguity_kind"
        if label != "ambiguous" and ambiguity_kind != "none":
            return "non-ambiguous responses must use ambiguity_kind=none"
        confidence_error = self._validate_confidence(response)
        if confidence_error is not None:
            return confidence_error
        evidence_refs = response.get("evidence_refs")
        if not isinstance(evidence_refs, list):
            return "evidence_refs must be a list"
        supplied_refs = {str(reference) for reference in evidence_refs}
        missing_required = {
            required
            for required in self.REQUIRED_EVIDENCE_REFS
            if required not in supplied_refs
            and not any(
                reference.startswith(f"{required}.")
                for reference in supplied_refs
            )
        }
        if missing_required:
            return "classifier response is missing required grounded evidence"
        return None
