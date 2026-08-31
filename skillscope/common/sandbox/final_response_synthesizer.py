from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from skillscope.common.llm import (
    DisabledLLMClient,
    JSONResponseContract,
    MAX_VALIDATED_LLM_ATTEMPTS,
    PromptAssetLoader,
    ResponseValidationError,
    StructuredLLMClient,
    complete_validated_json,
)
from skillscope.common.models import ExecutionEvent


@dataclass(slots=True)
class FinalResponseSynthesis:
    final_output: str
    grounded: bool
    strategy: str
    rationale: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    direct_result_sha256: list[str] = field(default_factory=list)
    uncertainty_flags: list[str] = field(default_factory=list)
    attempts: int = 0
    validation_errors: list[str] = field(default_factory=list)


class FinalResponseSynthesizer:
    """Produce the tested agent's user-visible response from runtime facts."""

    NO_DIRECT_RESULT_OUTPUT = (
        "The execution produced no direct tool or stdout result that can "
        "ground a task-result response."
    )

    RESPONSE_KEYS = {
        "final_output",
        "rationale",
        "evidence_refs",
        "uncertainty_flags",
    }

    def __init__(
        self,
        *,
        llm_client: StructuredLLMClient | None = None,
        prompt_loader: PromptAssetLoader | None = None,
        prompt_asset: str = "prompts/module2_final_response_synthesis.md",
        max_llm_attempts: int = MAX_VALIDATED_LLM_ATTEMPTS,
    ) -> None:
        self.llm_client = llm_client or DisabledLLMClient()
        self.prompt_loader = prompt_loader
        self.prompt_asset = prompt_asset
        self.max_llm_attempts = min(
            MAX_VALIDATED_LLM_ATTEMPTS,
            max(1, max_llm_attempts),
        )
        self._development_fallback = isinstance(
            self.llm_client,
            DisabledLLMClient,
        )

    def synthesize(
        self,
        *,
        prompt: str,
        status: str,
        trace: list[ExecutionEvent],
        selected_instruction_node_ids: list[str],
        stdout: str,
        stderr: str,
        direct_outputs: list[dict[str, str]],
        telemetry_fallback: str,
    ) -> FinalResponseSynthesis:
        payload = self._build_payload(
            prompt=prompt,
            status=status,
            trace=trace,
            selected_instruction_node_ids=selected_instruction_node_ids,
            stdout=stdout,
            stderr=stderr,
            direct_outputs=direct_outputs,
        )
        if self._development_fallback:
            return self._fallback(
                payload=payload,
                direct_outputs=direct_outputs,
                telemetry_fallback=telemetry_fallback,
            )
        if self.prompt_loader is None:
            return self._unresolved(
                attempts=0,
                errors=["missing_prompt_loader"],
            )
        try:
            system_prompt = self.prompt_loader.load(self.prompt_asset)
        except OSError as exc:
            return self._unresolved(
                attempts=0,
                errors=[f"prompt_load_error:{type(exc).__name__}"],
            )

        latest_errors: list[str] = []

        def consistency_check(response: dict[str, Any]) -> str | None:
            nonlocal latest_errors
            latest_errors = self._validate_response(response, payload)
            return "; ".join(latest_errors) if latest_errors else None

        counting_client = _CountingLLMClient(self.llm_client)
        try:
            response = complete_validated_json(
                counting_client,
                system_prompt=system_prompt,
                user_prompt=json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                schema_name="module2_tested_agent_final_response",
                contract=JSONResponseContract(
                    consistency_checks=(consistency_check,),
                ),
                max_attempts=self.max_llm_attempts,
            )
        except (ResponseValidationError, RuntimeError) as exc:
            return self._unresolved(
                attempts=counting_client.call_count,
                errors=(
                    latest_errors
                    or [f"final_response_llm_error:{type(exc).__name__}"]
                ),
            )

        final_errors = self._validate_response(response, payload)
        if final_errors:
            return self._unresolved(
                attempts=counting_client.call_count,
                errors=final_errors,
            )
        has_direct_result_text = bool(
            payload["response_contract"]["has_direct_result_text"]
        )
        return FinalResponseSynthesis(
            final_output=str(response["final_output"]).strip(),
            grounded=has_direct_result_text,
            strategy=(
                "llm_validated_grounded_final_response"
                if has_direct_result_text
                else "llm_validated_no_direct_result_response"
            ),
            rationale=str(response["rationale"]).strip(),
            evidence_refs=[str(item) for item in response["evidence_refs"]],
            direct_result_sha256=self._direct_result_sha256(payload),
            uncertainty_flags=[
                str(item) for item in response["uncertainty_flags"]
            ],
            attempts=counting_client.call_count,
        )

    def _build_payload(
        self,
        *,
        prompt: str,
        status: str,
        trace: list[ExecutionEvent],
        selected_instruction_node_ids: list[str],
        stdout: str,
        stderr: str,
        direct_outputs: list[dict[str, str]],
    ) -> dict[str, Any]:
        evidence: dict[str, Any] = {
            "user_task.prompt": prompt,
            "execution.status": status,
            "execution.selected_instruction_node_ids": list(
                selected_instruction_node_ids
            ),
        }
        if stdout.strip():
            evidence["execution.stdout"] = stdout.strip()
        if stderr.strip():
            evidence["execution.stderr"] = stderr.strip()
        for index, output in enumerate(direct_outputs, start=1):
            evidence[f"execution.tool_output.{index:04d}"] = dict(output)
        for index, event in enumerate(trace[-200:], start=1):
            evidence[f"execution.trace.{index:04d}"] = {
                "event_type": event.event_type,
                "summary": event.summary,
                "node_id": event.node_id,
                "layer": event.layer,
                "object_ref": event.object_ref,
                "arguments_summary": event.arguments_summary,
                "attributes": event.attributes,
            }
        allowed_refs = sorted(evidence)
        runtime_refs = [
            ref
            for ref in allowed_refs
            if ref not in {"user_task.prompt", "execution.status"}
        ]
        direct_result_refs = [
            ref
            for ref in allowed_refs
            if ref == "execution.stdout"
            or ref.startswith("execution.tool_output.")
        ]
        has_direct_result_text = bool(direct_outputs or stdout.strip())
        return {
            "user_task": {"prompt": prompt},
            "observed_runtime_evidence": evidence,
            "response_contract": {
                "allowed_evidence_refs": allowed_refs,
                "required_evidence_refs": [
                    "user_task.prompt",
                    "execution.status",
                ],
                "runtime_evidence_refs": runtime_refs,
                "direct_result_evidence_refs": direct_result_refs,
                "has_direct_result_text": has_direct_result_text,
                "required_uncertainty_flag_without_direct_result": (
                    "no_direct_tool_result_text"
                ),
                "required_final_output_without_direct_result": (
                    self.NO_DIRECT_RESULT_OUTPUT
                ),
                "max_final_output_characters": 8000,
            },
        }

    def _validate_response(
        self,
        response: dict[str, Any],
        payload: dict[str, Any],
    ) -> list[str]:
        errors: list[str] = []
        if set(response) != self.RESPONSE_KEYS:
            errors.append("response_keys_must_exactly_match_final_response_schema")
        final_output = response.get("final_output")
        if not isinstance(final_output, str) or not final_output.strip():
            errors.append("final_output_must_be_nonempty_string")
        elif len(final_output) > int(
            payload["response_contract"]["max_final_output_characters"]
        ):
            errors.append("final_output_exceeds_character_limit")
        rationale = response.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            errors.append("rationale_must_be_nonempty_string")

        evidence_refs = response.get("evidence_refs")
        if (
            not isinstance(evidence_refs, list)
            or not evidence_refs
            or any(not isinstance(item, str) or not item for item in evidence_refs)
        ):
            errors.append("evidence_refs_must_be_nonempty_string_array")
            normalized_refs: set[str] = set()
        else:
            normalized_refs = set(evidence_refs)
        allowed_refs = set(
            payload["response_contract"]["allowed_evidence_refs"]
        )
        if normalized_refs - allowed_refs:
            errors.append("evidence_refs_contain_unsupported_runtime_claims")
        required_refs = set(
            payload["response_contract"]["required_evidence_refs"]
        )
        if not required_refs.issubset(normalized_refs):
            errors.append("evidence_refs_missing_prompt_or_execution_status")
        runtime_refs = set(
            payload["response_contract"]["runtime_evidence_refs"]
        )
        if runtime_refs and not normalized_refs.intersection(runtime_refs):
            errors.append("evidence_refs_missing_observed_runtime_fact")
        direct_result_refs = set(
            payload["response_contract"]["direct_result_evidence_refs"]
        )
        cited_direct_result_refs = normalized_refs.intersection(
            direct_result_refs
        )
        if direct_result_refs and not cited_direct_result_refs:
            errors.append("evidence_refs_missing_direct_result_text")
        elif cited_direct_result_refs and isinstance(final_output, str):
            direct_result_texts = self._cited_direct_result_texts(
                payload=payload,
                evidence_refs=cited_direct_result_refs,
            )
            normalized_output = self._normalize_result_text(final_output)
            if normalized_output not in direct_result_texts:
                errors.append("final_output_must_equal_cited_direct_result_text")

        uncertainty_flags = response.get("uncertainty_flags")
        if not isinstance(uncertainty_flags, list) or any(
            not isinstance(item, str) or not item for item in uncertainty_flags
        ):
            errors.append("uncertainty_flags_must_be_string_array")
            normalized_flags: set[str] = set()
        else:
            normalized_flags = set(uncertainty_flags)
        if not payload["response_contract"]["has_direct_result_text"]:
            required_flag = str(
                payload["response_contract"][
                    "required_uncertainty_flag_without_direct_result"
                ]
            )
            if required_flag not in normalized_flags:
                errors.append("missing_no_direct_result_uncertainty_flag")
            required_output = str(
                payload["response_contract"][
                    "required_final_output_without_direct_result"
                ]
            )
            if final_output != required_output:
                errors.append(
                    "final_output_without_direct_result_must_use_fixed_abstention"
                )
        return errors

    @classmethod
    def _cited_direct_result_texts(
        cls,
        *,
        payload: dict[str, Any],
        evidence_refs: set[str],
    ) -> list[str]:
        evidence = payload.get("observed_runtime_evidence")
        if not isinstance(evidence, dict):
            return []
        result_texts: list[str] = []
        for evidence_ref in evidence_refs:
            value = evidence.get(evidence_ref)
            if evidence_ref == "execution.stdout":
                result_text = value if isinstance(value, str) else ""
            elif isinstance(value, dict):
                output = value.get("output")
                result_text = output if isinstance(output, str) else ""
            else:
                result_text = ""
            normalized = cls._normalize_result_text(result_text)
            if normalized:
                result_texts.append(normalized)
        return result_texts

    @staticmethod
    def _normalize_result_text(value: str) -> str:
        """Normalize transport line endings, preserving content whitespace."""

        return value.replace("\r\n", "\n").replace("\r", "\n").strip()

    @classmethod
    def _direct_result_sha256(cls, payload: dict[str, Any]) -> list[str]:
        contract = payload.get("response_contract")
        if not isinstance(contract, dict):
            return []
        refs = contract.get("direct_result_evidence_refs")
        if not isinstance(refs, list):
            return []
        result_texts = cls._cited_direct_result_texts(
            payload=payload,
            evidence_refs={str(ref) for ref in refs},
        )
        return sorted(
            hashlib.sha256(text.encode("utf-8")).hexdigest()
            for text in result_texts
        )

    def _fallback(
        self,
        *,
        payload: dict[str, Any],
        direct_outputs: list[dict[str, str]],
        telemetry_fallback: str,
    ) -> FinalResponseSynthesis:
        if direct_outputs:
            latest = direct_outputs[-1]
            evidence_ref = str(latest["evidence_ref"])
            return FinalResponseSynthesis(
                final_output=str(latest["output"]),
                grounded=True,
                strategy="deterministic_development_direct_output",
                rationale=(
                    "No LLM was configured; the user-visible response is the "
                    "direct observed tool result."
                ),
                evidence_refs=["user_task.prompt", "execution.status", evidence_ref],
                direct_result_sha256=self._direct_result_sha256(payload),
                uncertainty_flags=[
                    "llm_not_configured_development_fallback"
                ],
            )
        return FinalResponseSynthesis(
            final_output=telemetry_fallback,
            grounded=False,
            strategy="deterministic_development_telemetry_fallback",
            rationale=(
                "No LLM or direct tool result was available; this text is a "
                "diagnostic telemetry summary, not task-level output evidence."
            ),
            evidence_refs=[],
            uncertainty_flags=[
                "llm_not_configured_development_fallback",
                "telemetry_is_not_task_level_output",
            ],
        )

    def _unresolved(
        self,
        *,
        attempts: int,
        errors: list[str],
    ) -> FinalResponseSynthesis:
        return FinalResponseSynthesis(
            final_output="",
            grounded=False,
            strategy="configured_llm_final_response_unresolved",
            rationale=(
                "The configured LLM did not produce a schema-valid, "
                "evidence-grounded final response."
            ),
            uncertainty_flags=[
                "configured_llm_final_response_unresolved",
                "final_output_ungrounded",
            ],
            attempts=attempts,
            validation_errors=list(errors),
        )


class _CountingLLMClient:
    def __init__(self, delegate: StructuredLLMClient) -> None:
        self.delegate = delegate
        self.call_count = 0

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
    ) -> dict[str, Any]:
        self.call_count += 1
        return self.delegate.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name=schema_name,
        )
