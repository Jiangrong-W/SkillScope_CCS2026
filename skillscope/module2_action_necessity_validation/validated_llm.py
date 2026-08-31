from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from skillscope.common.llm import (
    JSONResponseContract,
    MAX_VALIDATED_LLM_ATTEMPTS,
    PromptAssetLoader,
    ResponseValidationError,
    StructuredLLMClient,
    complete_validated_json,
)


ResponseValidator = Callable[[dict[str, Any]], tuple[dict[str, Any] | None, list[str]]]


@dataclass(slots=True)
class ValidatedLLMResult:
    payload: dict[str, Any] | None
    attempts: int
    validation_errors: list[str] = field(default_factory=list)


class ValidatedLLMCaller:
    """Module adapter over the shared bounded JSON/evidence validator."""

    def __init__(
        self,
        *,
        llm_client: StructuredLLMClient,
        prompt_loader: PromptAssetLoader | None,
        max_attempts: int = MAX_VALIDATED_LLM_ATTEMPTS,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_loader = prompt_loader
        self.max_attempts = min(
            MAX_VALIDATED_LLM_ATTEMPTS,
            max(1, max_attempts),
        )

    def complete(
        self,
        *,
        prompt_asset: str,
        payload: dict[str, Any],
        schema_name: str,
        validator: ResponseValidator,
    ) -> ValidatedLLMResult:
        if self.prompt_loader is None:
            return ValidatedLLMResult(payload=None, attempts=0, validation_errors=["missing_prompt_loader"])
        try:
            system_prompt = self.prompt_loader.load(prompt_asset)
        except OSError as exc:
            return ValidatedLLMResult(
                payload=None,
                attempts=0,
                validation_errors=[f"prompt_load_error:{type(exc).__name__}"],
            )

        latest_errors: list[str] = []

        def consistency_check(response: dict[str, Any]) -> str | None:
            nonlocal latest_errors
            if not isinstance(response, dict):
                latest_errors = ["response_must_be_json_object"]
            else:
                _, latest_errors = validator(response)
            return "; ".join(latest_errors) if latest_errors else None

        counting_client = _CountingLLMClient(self.llm_client)
        contract = JSONResponseContract(
            consistency_checks=(consistency_check,),
        )
        try:
            response = complete_validated_json(
                counting_client,
                system_prompt=system_prompt,
                user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
                schema_name=schema_name,
                contract=contract,
                max_attempts=self.max_attempts,
            )
        except (ResponseValidationError, RuntimeError) as exc:
            errors = latest_errors or [f"validated_llm_error:{type(exc).__name__}"]
            return ValidatedLLMResult(
                payload=None,
                attempts=counting_client.call_count,
                validation_errors=errors,
            )

        normalized, final_errors = validator(response)
        if normalized is None or final_errors:
            return ValidatedLLMResult(
                payload=None,
                attempts=counting_client.call_count,
                validation_errors=final_errors or ["response_failed_validation"],
            )
        return ValidatedLLMResult(
            payload=normalized,
            attempts=counting_client.call_count,
            validation_errors=[],
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


def strict_bool(value: Any, field_name: str, errors: list[str]) -> bool | None:
    if isinstance(value, bool):
        return value
    errors.append(f"{field_name}_must_be_boolean")
    return None


def optional_bool(value: Any, field_name: str, errors: list[str]) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    errors.append(f"{field_name}_must_be_boolean_or_null")
    return None


def bounded_confidence(value: Any, errors: list[str]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append("confidence_must_be_number")
        return 0.0
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        errors.append("confidence_must_be_between_zero_and_one")
        return max(0.0, min(1.0, confidence))
    return confidence


def string_list(value: Any, field_name: str, errors: list[str], *, required: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        errors.append(f"{field_name}_must_be_string_array")
        return []
    normalized = [item.strip() for item in value if item.strip()]
    if required and not normalized:
        errors.append(f"{field_name}_must_contain_grounded_evidence")
    return normalized
