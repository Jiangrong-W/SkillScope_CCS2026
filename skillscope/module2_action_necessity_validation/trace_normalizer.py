from __future__ import annotations

from typing import Any

from skillscope.common.models import ExecutionRecord


class TraceNormalizer:
    ABSTRACT_EVENT_TYPES = {
        "instruction_step_start",
        "instruction_step_end",
        "instruction_tools_selected",
        "execution_plan_selected",
        "tool_resolution",
        "tool_call_start",
        "tool_call_end",
        "tool_call_error",
        "tool_observation",
        "file_read",
        "file_write",
        "file_access",
        "delete",
        "network_send",
        "exec_command",
        "read_env",
        "collect_identifier",
        "call",
        "return",
        "exception",
        "script_start",
        "script_end",
        "script_error",
    }

    def normalize(self, record: ExecutionRecord) -> list[dict[str, object]]:
        normalized: list[dict[str, object]] = []
        source_events = record.raw_trace or [self._event_to_payload(event) for event in record.trace]
        for event in source_events:
            event_type = str(event.get("event_type") or "")
            attributes = event.get("attributes")
            if not isinstance(attributes, dict):
                attributes = {}
            grounded_code_action = (
                event.get("layer") == "code"
                and event.get("node_id") is not None
                and attributes.get("material_operation") is not None
            )
            if (
                event_type not in self.ABSTRACT_EVENT_TYPES
                and not grounded_code_action
            ):
                continue
            normalized.append(
                {
                    "event_type": event_type,
                    "summary": self._string_or_none(event.get("summary")),
                    "object_ref": self._string_or_none(event.get("object_ref")),
                    "arguments_summary": self._string_or_none(event.get("arguments_summary")),
                    "node_id": self._string_or_none(event.get("node_id")),
                    "instruction_node_id": self._string_or_none(
                        attributes.get("instruction_node_id")
                    ),
                    "material_operation": self._string_or_none(
                        attributes.get("material_operation")
                    ),
                    "source_file": self._string_or_none(
                        attributes.get("source_file")
                    ),
                    "line_number": (
                        attributes.get("line_number")
                        if isinstance(attributes.get("line_number"), int)
                        else None
                    ),
                    "temporal_order_observed": (
                        attributes.get("temporal_order_observed")
                        if isinstance(
                            attributes.get("temporal_order_observed"), bool
                        )
                        else None
                    ),
                    "execution_count_observed": (
                        attributes.get("execution_count_observed")
                        if isinstance(
                            attributes.get("execution_count_observed"), bool
                        )
                        else None
                    ),
                    "ordering_evidence": self._string_or_none(
                        attributes.get("ordering_evidence")
                    ),
                }
            )
        return normalized

    def _event_to_payload(self, event: Any) -> dict[str, Any]:
        return {
            "event_type": getattr(event, "event_type", None),
            "summary": getattr(event, "summary", None),
            "node_id": getattr(event, "node_id", None),
            "layer": getattr(event, "layer", None),
            "object_ref": getattr(event, "object_ref", None),
            "arguments_summary": getattr(event, "arguments_summary", None),
            "attributes": getattr(event, "attributes", {}),
        }

    def _string_or_none(self, value: Any) -> str | None:
        if value is None:
            return None
        return str(value)
