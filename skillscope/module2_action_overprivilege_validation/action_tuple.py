from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from skillscope.common.models import (
    CandidateAction,
    CandidateExtractionResult,
    ExecutionRecord,
    TaskSpec,
    UEGNode,
)
from skillscope.common.privilege import MATERIAL_EVENT_TYPES


URL_RE = re.compile(r"https?://[^\s'\"`)>\]]+", re.IGNORECASE)
PATH_RE = re.compile(
    r"(?:\./|/)?[A-Za-z0-9_.~-]+(?:/[A-Za-z0-9_.~-]+)+|"
    r"[A-Za-z0-9_.-]+\.(?:log|txt|json|csv|md|yaml|yml)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ActionTuple:
    intent: str
    operation: str
    object: str | None
    source: str | None
    scope: str | None
    destination: str | None
    side_effect: str

    def as_payload(self) -> dict[str, str | None]:
        return asdict(self)


class ActionTupleExtractor:
    """Normalize the complete action tuple used by the authorization judgment."""

    EXTERNAL_OPERATIONS = {"network_send", "send", "transmit", "upload", "post", "share"}
    REALIZED_EVENT_TYPES = MATERIAL_EVENT_TYPES

    def extract(
        self,
        *,
        analysis: CandidateExtractionResult,
        candidate: CandidateAction,
        task: TaskSpec,
        original_record: ExecutionRecord | None = None,
    ) -> ActionTuple:
        action_tuple, _ = self.extract_with_evidence(
            analysis=analysis,
            candidate=candidate,
            task=task,
            original_record=original_record,
        )
        return action_tuple

    def extract_with_evidence(
        self,
        *,
        analysis: CandidateExtractionResult,
        candidate: CandidateAction,
        task: TaskSpec,
        original_record: ExecutionRecord | None = None,
    ) -> tuple[ActionTuple, dict[str, object] | None]:
        instances = self.extract_all_with_evidence(
            analysis=analysis,
            candidate=candidate,
            task=task,
            original_record=original_record,
        )
        # Preserve the legacy single-instance API for callers that only need a
        # representative tuple.  Authorization uses the complete API below.
        return instances[-1]

    def extract_all_with_evidence(
        self,
        *,
        analysis: CandidateExtractionResult,
        candidate: CandidateAction,
        task: TaskSpec,
        original_record: ExecutionRecord | None = None,
    ) -> list[tuple[ActionTuple, dict[str, object] | None]]:
        """Return every material action instance realized by the candidate.

        One instruction node can invoke more than one privilege-relevant tool
        action (for example, a file read followed by a network send).  Keeping
        all instances prevents a later event from hiding an earlier one during
        task-scope authorization.
        """
        node = analysis.ueg.node_by_id(candidate.node_id)
        operation = self._operation(node, candidate)
        destination = self._destination(node, operation)
        source = self._source(analysis, node, operation)
        object_ref = self._object(node, source, operation)
        scope = self._scope(node, operation, destination, source)
        side_effect = self._side_effect(operation, node, destination)
        static_tuple = ActionTuple(
            intent=task.task_summary or task.prompt,
            operation=operation,
            object=object_ref,
            source=source,
            scope=scope,
            destination=destination,
            side_effect=side_effect,
        )
        if original_record is None:
            return [(static_tuple, None)]

        realized_events = self._realized_events(original_record, candidate.node_id)
        if not realized_events:
            return [(static_tuple, None)]
        return [
            (
                self._tuple_from_realized_event(
                    static_tuple=static_tuple,
                    node=node,
                    realized_event=realized_event,
                ),
                realized_event,
            )
            for realized_event in realized_events
        ]

    def _tuple_from_realized_event(
        self,
        *,
        static_tuple: ActionTuple,
        node: UEGNode | None,
        realized_event: dict[str, object],
    ) -> ActionTuple:
        attributes = realized_event.get("attributes")
        if not isinstance(attributes, dict):
            attributes = {}
            realized_event["attributes"] = attributes
        realized_operation = str(realized_event.get("event_type") or "").strip().lower()
        if realized_operation not in self.REALIZED_EVENT_TYPES:
            realized_operation = static_tuple.operation
        realized_object_ref = str(realized_event.get("object_ref") or "").strip()
        arguments_summary = str(realized_event.get("arguments_summary") or "").strip()

        object_observed = self._runtime_value_observed(
            attributes=attributes,
            key="object_value_observed",
            default=bool(realized_object_ref),
        )
        arguments_observed = self._runtime_value_observed(
            attributes=attributes,
            key="arguments_value_observed",
            default=bool(arguments_summary),
        )
        static_object_supported = self._static_object_supported(
            node=node,
            value=static_tuple.object,
        )
        static_source_supported = self._static_source_supported(
            node=node,
            value=static_tuple.source,
        )
        static_destination_supported = self._static_destination_supported(
            node=node,
            operation=realized_operation,
            value=static_tuple.destination,
        )

        destination_basis = "not_applicable"
        realized_destination: str | None
        if realized_operation in self.EXTERNAL_OPERATIONS:
            if object_observed and realized_object_ref:
                realized_destination = realized_object_ref
                destination_basis = "runtime_observed"
            elif static_destination_supported:
                realized_destination = static_tuple.destination
                destination_basis = "static_source_provenance"
            else:
                realized_destination = None
                destination_basis = "unavailable"
        elif realized_operation in {"file_write", "delete"}:
            if object_observed and realized_object_ref:
                realized_destination = realized_object_ref
                destination_basis = "runtime_observed"
            elif static_destination_supported:
                realized_destination = static_tuple.destination
                destination_basis = "static_source_provenance"
            else:
                realized_destination = None
                destination_basis = "unavailable"
        else:
            realized_destination = "none"

        source_basis = "unavailable"
        realized_source: str | None
        if realized_operation in {
            "file_read",
            "read_env",
            "collect_identifier",
        }:
            if object_observed and realized_object_ref:
                realized_source = realized_object_ref
                source_basis = "runtime_observed"
            elif static_source_supported:
                realized_source = static_tuple.source
                source_basis = "static_source_provenance"
            else:
                realized_source = None
        elif realized_operation == "network_send" and arguments_observed and arguments_summary:
            realized_source = arguments_summary[:500]
            source_basis = "runtime_observed"
        elif static_source_supported:
            realized_source = static_tuple.source
            source_basis = "static_source_provenance"
        elif static_tuple.source == "task_context":
            realized_source = "task_context"
            source_basis = "not_applicable"
        else:
            realized_source = None

        object_basis = "unavailable"
        realized_object: str | None
        if (
            realized_operation == "exec_command"
            and arguments_observed
            and arguments_summary
        ):
            realized_object = arguments_summary[:500]
            object_basis = "runtime_observed"
        elif realized_operation in {
            "file_read",
            "read_env",
            "collect_identifier",
            "file_write",
            "delete",
        } and object_observed and realized_object_ref:
            realized_object = realized_object_ref
            object_basis = "runtime_observed"
        elif realized_operation == "network_send" and arguments_observed and arguments_summary:
            realized_object = arguments_summary[:500]
            object_basis = "runtime_observed"
        elif static_object_supported:
            realized_object = static_tuple.object
            object_basis = "static_source_provenance"
        else:
            realized_object = None

        realized_scope = self._scope(
            node,
            realized_operation,
            realized_destination or "unobserved",
            realized_source or "unobserved",
        )
        realized_side_effect = self._side_effect(
            realized_operation,
            node,
            realized_destination or "unobserved",
        )
        support = {
            "operation": {
                "available": True,
                "observed": True,
                "basis": "runtime_material_event_plus_source_mapping",
            },
            "object": {
                "available": realized_object is not None,
                "observed": object_basis == "runtime_observed",
                "basis": object_basis,
            },
            "source": {
                "available": realized_source is not None,
                "observed": source_basis == "runtime_observed",
                "basis": source_basis,
            },
            "destination": {
                "available": realized_destination is not None,
                "observed": destination_basis == "runtime_observed",
                "basis": destination_basis,
            },
            "side_effect": {
                "available": True,
                "observed": False,
                "basis": "operation_semantics_from_executed_source",
            },
        }
        attributes["action_tuple_component_support"] = support
        attributes["action_tuple_unobserved_fields"] = sorted(
            component
            for component, evidence in support.items()
            if not evidence["observed"]
        )
        return ActionTuple(
            intent=static_tuple.intent,
            operation=realized_operation,
            object=realized_object,
            source=realized_source,
            scope=realized_scope,
            destination=realized_destination,
            side_effect=realized_side_effect,
        )

    @staticmethod
    def _runtime_value_observed(
        *,
        attributes: dict[str, object],
        key: str,
        default: bool,
    ) -> bool:
        if key in attributes:
            return attributes.get(key) is True
        return default

    def _static_object_supported(
        self,
        *,
        node: UEGNode | None,
        value: str | None,
    ) -> bool:
        if not value or node is None:
            return False
        if node.layer == "instruction":
            return True
        return self._looks_like_concrete_source_value(value)

    def _static_source_supported(
        self,
        *,
        node: UEGNode | None,
        value: str | None,
    ) -> bool:
        if not value or value in {"task_context", "unspecified"}:
            return False
        if node is not None:
            for key in ("source", "source_path", "input", "input_path", "data_source"):
                attribute_value = node.attributes.get(key)
                if attribute_value is not None and str(attribute_value).strip() == value:
                    return True
        # A non-generic predecessor/object identity is source provenance even
        # when its runtime value is unavailable.  It remains explicitly marked
        # as unobserved in the support metadata above.
        return bool(value.strip())

    def _static_destination_supported(
        self,
        *,
        node: UEGNode | None,
        operation: str,
        value: str | None,
    ) -> bool:
        if not value or value in {
            "external",
            "external_unspecified",
            "local_filesystem",
            "unspecified",
        }:
            return False
        if node is not None:
            for key in ("destination", "endpoint", "url", "recipient", "target"):
                attribute_value = node.attributes.get(key)
                if attribute_value is not None and str(attribute_value).strip() == value:
                    return True
        if operation in self.EXTERNAL_OPERATIONS:
            return URL_RE.search(value) is not None
        if operation in {"file_write", "delete"}:
            return PATH_RE.search(value) is not None
        return value == "none"

    @staticmethod
    def _looks_like_concrete_source_value(value: str) -> bool:
        stripped = value.strip()
        if not stripped:
            return False
        if URL_RE.search(stripped) or PATH_RE.search(stripped):
            return True
        if (
            len(stripped) >= 2
            and stripped[0] in {"'", '"'}
            and stripped[-1] == stripped[0]
        ):
            return True
        return stripped in {"uuid.getnode", "platform.node"}

    def _realized_events(
        self,
        record: ExecutionRecord,
        candidate_node_id: str,
    ) -> list[dict[str, object]]:
        payloads: list[dict[str, object]] = []
        if record.raw_trace:
            payloads.extend(
                payload for payload in record.raw_trace if isinstance(payload, dict)
            )
        else:
            payloads.extend(
                {
                    "event_type": event.event_type,
                    "summary": event.summary,
                    "node_id": event.node_id,
                    "layer": event.layer,
                    "object_ref": event.object_ref,
                    "arguments_summary": event.arguments_summary,
                    "attributes": event.attributes,
                }
                for event in record.trace
            )
        related = [
            payload
            for payload in payloads
            if (
                payload.get("node_id") == candidate_node_id
                or (
                    isinstance(payload.get("attributes"), dict)
                    and payload["attributes"].get("instruction_node_id")
                    == candidate_node_id
                )
            )
        ]
        material = [
            payload
            for payload in related
            if (
                str(payload.get("event_type") or "")
                in self.REALIZED_EVENT_TYPES
                or (
                    isinstance(payload.get("attributes"), dict)
                    and str(
                        payload["attributes"].get("material_operation") or ""
                    )
                    in self.REALIZED_EVENT_TYPES
                )
            )
        ]
        # Planning, instruction traversal, and tool bookkeeping are not realized
        # action evidence. Returning non-material events here would let values
        # such as ``SKILL.md`` overwrite the candidate's actual destination.
        realized: list[dict[str, object]] = []
        for selected in material:
            attributes = selected.get("attributes")
            event_type = str(selected.get("event_type") or "")
            if (
                event_type not in self.REALIZED_EVENT_TYPES
                and isinstance(attributes, dict)
            ):
                event_type = str(attributes.get("material_operation") or event_type)
            realized.append(
                {
                    "event_type": event_type,
                    "summary": str(selected.get("summary") or ""),
                    "node_id": str(selected.get("node_id") or ""),
                    "layer": str(selected.get("layer") or ""),
                    "object_ref": str(selected.get("object_ref") or ""),
                    "arguments_summary": str(selected.get("arguments_summary") or ""),
                    "attributes": (
                        dict(attributes) if isinstance(attributes, dict) else {}
                    ),
                }
            )
        return realized

    def _realized_event(
        self,
        record: ExecutionRecord,
        candidate_node_id: str,
    ) -> dict[str, object] | None:
        """Backward-compatible representative material event."""
        realized = self._realized_events(record, candidate_node_id)
        return realized[-1] if realized else None

    def _operation(self, node: UEGNode | None, candidate: CandidateAction) -> str:
        raw = str(getattr(node, "operation_type", "") or "").strip().lower()
        if raw and raw not in {"instruction_step", "call"}:
            return raw
        text = " ".join(
            [
                candidate.summary,
                str(getattr(node, "raw_text", "") or ""),
                str(getattr(node, "attributes", {}).get("call_name", "") if node else ""),
            ]
        ).lower()
        aliases = (
            ("network_send", ("send", "upload", "post", "webhook", "telegram", "requests.")),
            ("exec_command", ("subprocess", "os.system", "shell", "execute command")),
            ("delete", ("delete", "remove", "erase", "unlink")),
            ("write", ("write", "save", "export", "store")),
            ("read", ("read", "load", "open", "inspect")),
            ("collect", ("collect", "gather", "identifier")),
            ("analyze", ("analyze", "summarize", "review")),
        )
        for normalized, keywords in aliases:
            if any(keyword in text for keyword in keywords):
                return normalized
        return raw or "unspecified"

    def _destination(self, node: UEGNode | None, operation: str) -> str:
        if node is None:
            return "none"
        attributes = node.attributes
        for key in ("destination", "endpoint", "url", "recipient", "target"):
            value = attributes.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        text = " ".join(
            [str(node.object_ref or ""), str(node.raw_text or ""), node.summary]
        )
        url_match = URL_RE.search(text)
        if url_match:
            return url_match.group(0)
        if operation in self.EXTERNAL_OPERATIONS:
            return str(node.object_ref or "external_unspecified")
        if operation in {"write", "delete"}:
            path_match = PATH_RE.search(text)
            return path_match.group(0) if path_match else str(node.object_ref or "local_filesystem")
        return "none"

    def _source(
        self,
        analysis: CandidateExtractionResult,
        node: UEGNode | None,
        operation: str,
    ) -> str:
        if node is None:
            return "unspecified"
        for key in ("source", "source_path", "input", "input_path", "data_source"):
            value = node.attributes.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        text = " ".join([str(node.object_ref or ""), str(node.raw_text or ""), node.summary])
        if operation in {"read", "file_access", "read_env", "collect", "collect_identifier"}:
            path_match = PATH_RE.search(text)
            return path_match.group(0) if path_match else str(node.object_ref or text[:160])
        data_predecessors: list[str] = []
        for predecessor_id in analysis.ueg.predecessor_ids(node.node_id, {"DATA_DEP"}):
            predecessor = analysis.ueg.node_by_id(predecessor_id)
            if predecessor is None:
                continue
            value = predecessor.object_ref or predecessor.summary
            if value and value not in data_predecessors:
                data_predecessors.append(value)
        if data_predecessors:
            return "; ".join(data_predecessors[:4])
        return "task_context"

    def _object(self, node: UEGNode | None, source: str, operation: str) -> str:
        if node is None:
            return source
        if node.object_ref:
            return str(node.object_ref)
        if operation in {"read", "file_access", "read_env", "collect", "collect_identifier"}:
            return source
        call_name = node.attributes.get("call_name")
        if call_name:
            return str(call_name)
        return node.summary

    def _scope(
        self,
        node: UEGNode | None,
        operation: str,
        destination: str,
        source: str,
    ) -> str:
        joined = " ".join(
            [
                operation,
                destination,
                source,
                str(getattr(node, "raw_text", "") or ""),
            ]
        ).lower()
        if operation in self.EXTERNAL_OPERATIONS or URL_RE.search(joined):
            return "external"
        if any(token in joined for token in ("/etc/", "/var/", "~/.", ".ssh", ".aws")):
            return "system_or_user_global"
        if operation in {"read", "write", "delete", "file_access"}:
            return "local"
        return "task"

    def _side_effect(
        self,
        operation: str,
        node: UEGNode | None,
        destination: str,
    ) -> str:
        if operation in self.EXTERNAL_OPERATIONS or destination.startswith(("http://", "https://")):
            return "external_transmission"
        if operation == "exec_command":
            return "command_execution"
        if operation == "delete":
            return "filesystem_delete"
        if operation in {"write", "file_write"}:
            return "filesystem_write"
        if operation in {"read_env", "collect_identifier", "collect"}:
            return "sensitive_data_collection"
        risk_tags = {tag.lower() for tag in getattr(node, "risk_tags", [])}
        if "sensitive_collection" in risk_tags:
            return "sensitive_data_collection"
        return "none"
