from __future__ import annotations

from collections.abc import Iterable


# The privilege-relevant action taxonomy. Graph construction, runtime tracing, trigger detection,
# and the final verdict all normalize through this shared mapping.
PRIVILEGE_RELEVANT_ACTION_TYPES = frozenset(
    {
        "sensitive_data_access",
        "external_data_transmission",
        "command_execution",
        "persistent_state_modification",
    }
)

OPERATION_PRIVILEGE_TYPES: dict[str, str] = {
    "read": "sensitive_data_access",
    "file_read": "sensitive_data_access",
    "file_access": "sensitive_data_access",
    "read_env": "sensitive_data_access",
    "collect": "sensitive_data_access",
    "collect_identifier": "sensitive_data_access",
    "credential_read": "sensitive_data_access",
    "network_send": "external_data_transmission",
    "send": "external_data_transmission",
    "transmit": "external_data_transmission",
    "upload": "external_data_transmission",
    "post": "external_data_transmission",
    "share": "external_data_transmission",
    "exec_command": "command_execution",
    "execute": "command_execution",
    "invoke": "command_execution",
    "run": "command_execution",
    "command_execution": "command_execution",
    "process_spawn": "command_execution",
    "shell_exec": "command_execution",
    "write": "persistent_state_modification",
    "create": "persistent_state_modification",
    "file_write": "persistent_state_modification",
    "delete": "persistent_state_modification",
    "state_write": "persistent_state_modification",
    "persistent_state_modification": "persistent_state_modification",
}

RISK_TAG_PRIVILEGE_TYPES: dict[str, str] = {
    "sensitive_collection": "sensitive_data_access",
    "sensitive_data": "sensitive_data_access",
    "credential_access": "sensitive_data_access",
    "file_access": "sensitive_data_access",
    "network": "external_data_transmission",
    "external_transmission": "external_data_transmission",
    "command_execution": "command_execution",
    "persistent_state": "persistent_state_modification",
    "file_write": "persistent_state_modification",
    "deletion": "persistent_state_modification",
}

MATERIAL_EVENT_TYPES = frozenset(OPERATION_PRIVILEGE_TYPES)


def material_event_matches_operation(
    operation_type: str | None,
    event_type: str | None,
    attributes: dict[str, object] | None = None,
) -> bool:
    """Match the candidate's operation to physical evidence of that operation.

    A Python call/line/return trace and a tool's parent instruction ID are
    provenance, not evidence that every side effect of the child was an action
    of its caller.  Aliases are normalized by actual operation, rather than by
    the broader privilege family (e.g. a delete is not a file write).
    """
    evidence = attributes or {}
    if evidence.get("ablated") is True:
        return False
    actual = str(event_type or "").strip().casefold()
    if actual in {"call", "return", "line", "instruction_step_start", "instruction_step_end"}:
        return False
    if actual not in MATERIAL_EVENT_TYPES:
        actual = str(evidence.get("material_operation") or "").strip().casefold()
    aliases = {
        "read": "file_read", "file_access": "file_access",
        "credential_read": "file_read", "sensitive_data_access": "file_read",
        "send": "network_send", "transmit": "network_send",
        "upload": "network_send", "post": "network_send", "share": "network_send",
        "execute": "exec_command", "invoke": "exec_command", "run": "exec_command",
        "command_execution": "exec_command", "process_spawn": "exec_command", "shell_exec": "exec_command",
        "write": "file_write", "create": "file_write", "state_write": "file_write",
        "persistent_state_modification": "file_write",
    }
    expected = str(operation_type or "").strip().casefold()
    expected = aliases.get(expected, expected)
    actual = aliases.get(actual, actual)
    if expected == "file_access":
        return actual in {"file_read", "file_write", "file_access"}
    if expected == "collect":
        return actual in {"file_read", "read_env", "collect_identifier"}
    return expected in MATERIAL_EVENT_TYPES and expected == actual


def privilege_type_for_action(
    operation_type: str | None,
    risk_tags: Iterable[str] = (),
) -> str | None:
    """Return the canonical privilege type, if one is evidenced."""

    normalized_operation = str(operation_type or "").strip().casefold()
    privilege_type = OPERATION_PRIVILEGE_TYPES.get(normalized_operation)
    if privilege_type is not None:
        return privilege_type
    for raw_tag in risk_tags:
        privilege_type = RISK_TAG_PRIVILEGE_TYPES.get(
            str(raw_tag).strip().casefold()
        )
        if privilege_type is not None:
            return privilege_type
    return None


def is_privilege_relevant_action(
    operation_type: str | None,
    risk_tags: Iterable[str] = (),
) -> bool:
    return (
        privilege_type_for_action(operation_type, risk_tags)
        in PRIVILEGE_RELEVANT_ACTION_TYPES
    )
