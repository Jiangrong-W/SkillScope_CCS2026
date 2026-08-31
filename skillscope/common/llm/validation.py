from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .client import DisabledLLMClient, StructuredLLMClient


# Keep the bounded validator-loop default in one place so every LLM-assisted
# stage follows the same contract.
MAX_VALIDATED_LLM_ATTEMPTS = 5


class ResponseValidationError(RuntimeError):
    pass


@dataclass(slots=True)
class JSONResponseContract:
    """Small dependency-free schema/evidence contract for LLM judges."""

    required_fields: tuple[str, ...] = ()
    non_empty_string_fields: tuple[str, ...] = ()
    enum_fields: dict[str, set[str]] = field(default_factory=dict)
    evidence_field: str | None = None
    grounded_evidence_ids: set[str] = field(default_factory=set)
    consistency_checks: tuple[Callable[[dict[str, Any]], str | None], ...] = ()

    def validate(self, response: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        for field_name in self.required_fields:
            if field_name not in response:
                errors.append(f"missing required field {field_name!r}")
        for field_name in self.non_empty_string_fields:
            value = response.get(field_name)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"field {field_name!r} must be a non-empty string")
        for field_name, allowed in self.enum_fields.items():
            value = str(response.get(field_name) or "").strip().lower()
            if value not in allowed:
                errors.append(
                    f"field {field_name!r} must be one of {sorted(allowed)!r}; received {value!r}"
                )

        if self.evidence_field is not None:
            citations = response.get(self.evidence_field)
            if not isinstance(citations, list) or not citations:
                errors.append(f"field {self.evidence_field!r} must contain grounded evidence identifiers")
            else:
                unsupported = [
                    str(citation)
                    for citation in citations
                    if str(citation) not in self.grounded_evidence_ids
                ]
                if unsupported:
                    errors.append(f"unsupported evidence identifiers: {unsupported!r}")

        for check in self.consistency_checks:
            error = check(response)
            if error:
                errors.append(error)
        return errors


def complete_validated_json(
    client: StructuredLLMClient,
    *,
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    contract: JSONResponseContract,
    max_attempts: int = MAX_VALIDATED_LLM_ATTEMPTS,
) -> dict[str, Any]:
    """Retry invalid output while keeping the same grounded user evidence."""

    # Enforce the retry bound as an API invariant, including for
    # callers that inject a custom value larger than the production default.
    attempts = min(MAX_VALIDATED_LLM_ATTEMPTS, max(1, max_attempts))
    if isinstance(client, DisabledLLMClient):
        # Disabled mode is a deliberate configuration state, not a transient
        # endpoint failure.  Let the stage's explicit development fallback run
        # immediately instead of issuing five identical disabled calls.
        raise RuntimeError(
            "No LLM client is configured. Set SKILLSCOPE_LLM_PROVIDER, "
            "SKILLSCOPE_LLM_API_KEY, SKILLSCOPE_LLM_BASE_URL, and "
            "SKILLSCOPE_LLM_MODEL in the environment or project .env."
        )
    validation_errors: list[str] = []
    for attempt in range(1, attempts + 1):
        retry_suffix = ""
        if validation_errors:
            retry_suffix = (
                "\n\nYour previous response failed the automated validator for these reasons:\n- "
                + "\n- ".join(validation_errors)
                + "\nReturn a corrected JSON object using only the unchanged evidence in the user message."
            )
        try:
            response = client.complete_json(
                system_prompt=system_prompt + retry_suffix,
                user_prompt=user_prompt,
                schema_name=schema_name,
            )
        except RuntimeError as exc:
            validation_errors = [
                f"endpoint_or_json_error:{type(exc).__name__}:{exc}"
            ]
            if attempt == attempts:
                break
            continue
        validation_errors = contract.validate(response)
        if not validation_errors:
            return response
        if attempt == attempts:
            break
    raise ResponseValidationError(
        f"{schema_name} failed validation after {attempts} attempt(s): "
        + "; ".join(validation_errors)
    )
