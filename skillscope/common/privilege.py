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
    "call": "command_execution",
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
