from __future__ import annotations

import json
import re
import shlex
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skillscope.common.llm import PromptAssetLoader, StructuredLLMClient
from skillscope.common.models import ExecutionEvent, UEGNode, UnifiedExecutionGraph

from .final_response_synthesizer import FinalResponseSynthesizer
from .policy import SandboxPolicyError, resolve_within
from .tooling import (
    ToolDispatcher,
    ToolInvocationRequest,
    ToolInvocationResult,
    _shell_command_tokens,
)


@dataclass(slots=True)
class AgentRuntimeExecutionOutcome:
    trace: list[ExecutionEvent] = field(default_factory=list)
    raw_trace: list[dict[str, Any]] = field(default_factory=list)
    executed_node_ids: list[str] = field(default_factory=list)
    final_output: str = ""
    stdout: str = ""
    stderr: str = ""
    status: str = "completed"
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InstructionActionMaterialization:
    event_type: str
    summary: str
    object_ref: str
    arguments_summary: str | None
    attributes: dict[str, Any] = field(default_factory=dict)
    note: str | None = None


class InstructionActionMaterializer:
    """Safely realize explicit instruction-only actions inside the sandbox copy."""

    _ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    _NETWORK_DESTINATION_PREFIXES = (
        "http://",
        "https://",
        "mailto:",
        "slack:",
        "telegram:",
        "webhook:",
    )

    def materialize(
        self,
        *,
        node: UEGNode,
        sandbox_root: Path,
    ) -> InstructionActionMaterialization | None:
        if node.layer != "instruction" or node.node_type != "INSTR_ACTION":
            return None
        operation = str(node.operation_type or "").strip().lower()
        if not operation:
            return None

        fixtures = self._load_fixtures(sandbox_root)
        if operation in {"network_send", "send", "post", "upload", "notify", "transmit"}:
            return self._materialize_network_send(
                node=node,
                fixtures=fixtures,
            )
        if operation in {"read_env", "environment_read", "getenv"}:
            return self._materialize_environment_read(
                node=node,
                fixtures=fixtures,
            )
        if operation in {"file_read", "read"}:
            return self._materialize_file_read(node=node, sandbox_root=sandbox_root)
        if operation in {"file_write", "write", "create", "save"}:
            return self._materialize_file_write(node=node, sandbox_root=sandbox_root)
        if operation == "file_access":
            access_mode = str(
                node.attributes.get("access_mode")
                or node.attributes.get("mode")
                or ""
            ).lower()
            if access_mode in {"r", "read", "readonly"}:
                return self._materialize_file_read(
                    node=node,
                    sandbox_root=sandbox_root,
                )
            if access_mode in {"w", "write", "append", "create"}:
                return self._materialize_file_write(
                    node=node,
                    sandbox_root=sandbox_root,
                )
            return None
        if operation in {"delete", "remove", "unlink"}:
            return self._materialize_delete(node=node, sandbox_root=sandbox_root)
        if operation in {"collect_identifier", "identifier_read"}:
            return self._materialize_identifier_read(node)
        return None

    def _materialize_network_send(
        self,
        *,
        node: UEGNode,
        fixtures: list[dict[str, Any]],
    ) -> InstructionActionMaterialization | None:
        destination = self._attribute_text(
            node,
            "destination",
            "endpoint",
            "url",
            "recipient",
        )
        if destination is None and self._looks_like_network_destination(node.object_ref):
            destination = str(node.object_ref)
        payload_ref = self._attribute_value(
            node,
            "payload",
            "data",
            "content",
            "source",
            "object",
        )
        if payload_ref is None and node.object_ref and node.object_ref != destination:
            payload_ref = node.object_ref
        if destination is None or payload_ref is None:
            return None
        fixture = self._matching_fixture(
            fixtures,
            fixture_type="api",
            target=destination,
        )
        if fixture is None:
            return None
        return InstructionActionMaterialization(
            event_type="network_send",
            summary=f"Mocked instruction network send to {destination}",
            object_ref=destination,
            arguments_summary=self._safe_summary(payload_ref),
            attributes={
                "mocked": True,
                "fixture_id": fixture.get("fixture_id"),
                "destination": destination,
                "payload_ref": self._safe_summary(payload_ref),
            },
            note=(
                "Materialized an instruction-only network action through an "
                "exact API fixture; no real network request was made."
            ),
        )

    def _materialize_environment_read(
        self,
        *,
        node: UEGNode,
        fixtures: list[dict[str, Any]],
    ) -> InstructionActionMaterialization | None:
        key = self._attribute_text(node, "env_key", "key", "source") or node.object_ref
        if not isinstance(key, str) or self._ENV_KEY_RE.fullmatch(key) is None:
            return None
        fixture = self._matching_fixture(fixtures, fixture_type="env", target=key)
        if fixture is None:
            return None
        return InstructionActionMaterialization(
            event_type="read_env",
            summary=f"Read instruction environment fixture {key}",
            object_ref=key,
            arguments_summary=None,
            attributes={
                "mocked": True,
                "fixture_id": fixture.get("fixture_id"),
                "source": key,
            },
            note=(
                "Materialized an instruction-only environment read from an "
                "explicit fixture rather than the host environment."
            ),
        )

    def _materialize_file_read(
        self,
        *,
        node: UEGNode,
        sandbox_root: Path,
    ) -> InstructionActionMaterialization | None:
        target = self._file_target(node)
        confined = self._confined_file(sandbox_root, target)
        if confined is None or not confined.is_file():
            return None
        try:
            with confined.open("rb") as handle:
                observed = handle.read(4096)
        except OSError:
            return None
        return InstructionActionMaterialization(
            event_type="file_read",
            summary=f"Read sandbox file for instruction {target}",
            object_ref=target,
            arguments_summary=None,
            attributes={
                "bytes_observed": len(observed),
                "sandbox_local": True,
            },
            note="Materialized an instruction-only file read inside the isolated sandbox copy.",
        )

    def _materialize_file_write(
        self,
        *,
        node: UEGNode,
        sandbox_root: Path,
    ) -> InstructionActionMaterialization | None:
        target = self._file_target(node)
        content = self._attribute_value(node, "content", "data", "payload", "value")
        confined = self._confined_file(sandbox_root, target)
        if confined is None or content is None:
            return None
        try:
            confined.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                confined.write_bytes(content)
                bytes_written = len(content)
            else:
                rendered = str(content)
                confined.write_text(rendered, encoding="utf-8")
                bytes_written = len(rendered.encode("utf-8"))
        except OSError:
            return None
        return InstructionActionMaterialization(
            event_type="file_write",
            summary=f"Write sandbox file for instruction {target}",
            object_ref=target,
            arguments_summary=f"bytes={bytes_written}",
            attributes={
                "bytes_written": bytes_written,
                "sandbox_local": True,
            },
            note="Materialized an instruction-only file write inside the isolated sandbox copy.",
        )

    def _materialize_delete(
        self,
        *,
        node: UEGNode,
        sandbox_root: Path,
    ) -> InstructionActionMaterialization | None:
        target = self._file_target(node)
        confined = self._confined_file(sandbox_root, target)
        if confined is None or not confined.is_file():
            return None
        try:
            confined.unlink()
        except OSError:
            return None
        return InstructionActionMaterialization(
            event_type="delete",
            summary=f"Delete sandbox file for instruction {target}",
            object_ref=target,
            arguments_summary=None,
            attributes={"sandbox_local": True},
            note=(
                "Materialized an instruction-only delete against the isolated "
                "sandbox copy; host files were not reachable."
            ),
        )

    def _materialize_identifier_read(
        self,
        node: UEGNode,
    ) -> InstructionActionMaterialization | None:
        identifier = self._attribute_text(node, "identifier", "source") or node.object_ref
        if not isinstance(identifier, str) or identifier.lower() not in {
            "hardware_id",
            "hostname",
            "machine_id",
            "platform.node",
            "uuid",
            "uuid.getnode",
        }:
            return None
        return InstructionActionMaterialization(
            event_type="collect_identifier",
            summary=f"Read deterministic sandbox identifier {identifier}",
            object_ref=identifier,
            arguments_summary=None,
            attributes={"mocked": True, "sandbox_value": "sandbox-host"},
            note=(
                "Materialized an instruction-only identifier read with a "
                "deterministic sandbox value."
            ),
        )

    def _file_target(self, node: UEGNode) -> str | None:
        return self._attribute_text(node, "path", "target", "source", "destination") or (
            node.object_ref if isinstance(node.object_ref, str) else None
        )

    def _confined_file(self, sandbox_root: Path, target: str | None) -> Path | None:
        if not target or self._looks_like_network_destination(target):
            return None
        try:
            confined = resolve_within(sandbox_root, target)
        except (OSError, SandboxPolicyError):
            return None
        if confined == sandbox_root.resolve():
            return None
        return confined

    def _load_fixtures(self, sandbox_root: Path) -> list[dict[str, Any]]:
        manifest_path = sandbox_root / ".skillscope" / "fixtures.json"
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return []
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def _matching_fixture(
        self,
        fixtures: list[dict[str, Any]],
        *,
        fixture_type: str,
        target: str,
    ) -> dict[str, Any] | None:
        return next(
            (
                fixture
                for fixture in fixtures
                if fixture.get("fixture_type") == fixture_type
                and str(fixture.get("target") or "") == target
            ),
            None,
        )

    def _attribute_text(self, node: UEGNode, *keys: str) -> str | None:
        value = self._attribute_value(node, *keys)
        if not isinstance(value, str) or not value.strip():
            return None
        return value.strip()

    def _attribute_value(self, node: UEGNode, *keys: str) -> Any:
        for key in keys:
            if key in node.attributes and node.attributes[key] is not None:
                return node.attributes[key]
        return None

    def _looks_like_network_destination(self, value: object) -> bool:
        return isinstance(value, str) and value.lower().startswith(
            self._NETWORK_DESTINATION_PREFIXES
        )

    def _safe_summary(self, value: object) -> str:
        try:
            return str(value)[:200]
        except Exception:
            return f"<{type(value).__name__}>"


class RuntimeTraceRecorder:
    _MATERIAL_OPERATION_TYPES = {
        "network_send",
        "exec_command",
        "file_read",
        "file_write",
        "file_access",
        "delete",
        "read_env",
        "collect_identifier",
    }

    def __init__(self, ueg: UnifiedExecutionGraph) -> None:
        self.ueg = ueg
        self.trace: list[ExecutionEvent] = []
        self.raw_trace: list[dict[str, Any]] = []
        self.executed_node_ids: list[str] = []
        self._executed_node_id_set: set[str] = set()

    def append(
        self,
        *,
        event_type: str,
        summary: str,
        node_id: str | None = None,
        layer: str | None = None,
        object_ref: str | None = None,
        arguments_summary: str | None = None,
        attributes: dict[str, Any] | None = None,
        mark_executed: bool = True,
    ) -> None:
        payload = {
            "event_type": event_type,
            "summary": summary,
            "node_id": node_id,
            "layer": layer,
            "object_ref": object_ref,
            "arguments_summary": arguments_summary,
            "attributes": attributes or {},
        }
        self.raw_trace.append(payload)
        self.trace.append(
            ExecutionEvent(
                event_type=event_type,
                summary=summary,
                node_id=node_id,
                layer=layer,
                object_ref=object_ref,
                arguments_summary=arguments_summary,
                attributes=attributes or {},
            )
        )
        if node_id is not None and mark_executed:
            self.mark_node(node_id)

    def mark_node(self, node_id: str) -> None:
        if node_id not in self._executed_node_id_set:
            self.executed_node_ids.append(node_id)
            self._executed_node_id_set.add(node_id)

    def append_tool_trace(
        self,
        *,
        script_path: str,
        tool_name: str,
        tool_call_id: str,
        instruction_node_id: str | None,
        trace_events: list[dict[str, Any]],
    ) -> None:
        for event_payload in trace_events:
            payload = dict(event_payload)
            attributes = payload.get("attributes") or {}
            if not isinstance(attributes, dict):
                attributes = {}
            attributes = dict(attributes)
            attributes["tool_name"] = tool_name
            attributes["tool_call_id"] = tool_call_id
            if instruction_node_id is not None:
                attributes["instruction_node_id"] = instruction_node_id
            payload["attributes"] = attributes
            matched_nodes = self._match_code_nodes(script_path, payload)
            if not matched_nodes:
                self._append_code_event(payload=payload, node=None, attributes=attributes)
                continue

            runtime_event_type = str(payload.get("event_type") or "sandbox_event")
            for node in matched_nodes:
                self.mark_node(node.node_id)
                matched_attributes = dict(attributes)
                if runtime_event_type in {
                    "source_command_execution",
                    "source_coverage_execution",
                }:
                    matched_attributes.pop("executed_column_spans", None)
                    matched_attributes.pop("source_line_indents", None)
                    event_type = node.operation_type or runtime_event_type
                    if node.operation_type in self._MATERIAL_OPERATION_TYPES:
                        matched_attributes["material_operation"] = node.operation_type
                    elif node.operation_type is not None:
                        matched_attributes["mapped_operation"] = node.operation_type
                    matched_attributes["runtime_event_type"] = runtime_event_type
                    if runtime_event_type == "source_command_execution":
                        matched_attributes["tuple_evidence"] = (
                            "bash_xtrace_command_plus_static_source_semantics"
                        )
                        matched_attributes["arguments_value_observed"] = True
                        matched_attributes["object_value_observed"] = False
                    else:
                        matched_attributes["tuple_evidence"] = (
                            "v8_covered_source_span_plus_static_source_semantics"
                        )
                        matched_attributes["arguments_value_observed"] = False
                        matched_attributes["object_value_observed"] = False
                    if node.source_range is not None:
                        matched_attributes["line_number"] = (
                            node.source_range.start_line
                        )
                        start_column = node.source_range.start_column
                        end_column = node.source_range.end_column
                        if not isinstance(start_column, int):
                            start_column = node.attributes.get("col_offset")
                        if not isinstance(end_column, int):
                            end_column = node.attributes.get("end_col_offset")
                        if isinstance(start_column, int):
                            matched_attributes["source_start_column"] = start_column
                        if isinstance(end_column, int):
                            matched_attributes["source_end_column"] = end_column
                elif runtime_event_type == "line":
                    if node.operation_type is not None:
                        matched_attributes["mapped_operation"] = node.operation_type
                    matched_attributes["runtime_event_type"] = runtime_event_type
                    event_type = runtime_event_type
                else:
                    if node.operation_type is not None:
                        matched_attributes["material_operation"] = node.operation_type
                    event_type = runtime_event_type
                matched_payload = dict(payload)
                matched_payload["event_type"] = event_type
                if runtime_event_type in {
                    "source_command_execution",
                    "source_coverage_execution",
                }:
                    matched_payload["summary"] = (
                        f"Execute {event_type} at "
                        f"{node.source_file}:"
                        f"{node.source_range.start_line if node.source_range else '?'}"
                    )
                matched_payload["object_ref"] = self._realized_object_ref(
                    node=node,
                    payload=payload,
                )
                matched_payload["arguments_summary"] = (
                    self._realized_arguments_summary(
                        node=node,
                        payload=payload,
                    )
                )
                self._append_code_event(
                    payload=matched_payload,
                    node=node,
                    attributes=matched_attributes,
                )

    def _match_code_node_id(self, script_path: str, event_payload: dict[str, object]) -> str | None:
        matched_nodes = self._match_code_nodes(script_path, event_payload)
        return matched_nodes[0].node_id if matched_nodes else None

    def _match_code_nodes(
        self,
        script_path: str,
        event_payload: dict[str, object],
    ) -> list[UEGNode]:
        event_type = str(event_payload.get("event_type") or "")
        summary = str(event_payload.get("summary") or "")
        object_ref = event_payload.get("object_ref")
        attributes = event_payload.get("attributes") or {}
        if not isinstance(attributes, dict):
            attributes = {}
        if isinstance(attributes.get("source_file"), str):
            source_file = str(attributes["source_file"])
        elif isinstance(object_ref, str) and object_ref.endswith(
            (".py", ".sh", ".js", ".mjs", ".cjs", ".ts", ".mts", ".cts")
        ):
            source_file = object_ref
        else:
            source_file = script_path
        line_number = attributes.get("line_number")
        if not isinstance(line_number, int):
            line_number = None

        # Runtime actions are only linked to UEG actions when there is concrete
        # source-location evidence. A V8 snapshot carries precise spans for
        # the whole source file; all other events require one source line.
        if line_number is None and event_type != "source_coverage_execution":
            return []

        normalized_source_file = self._normalize_source_file(source_file)
        candidates: list[UEGNode] = []
        for node in self.ueg.nodes:
            if (
                node.layer != "code"
                or node.source_file is None
                or self._normalize_source_file(node.source_file) != normalized_source_file
                or node.source_range is None
            ):
                continue
            if (
                line_number is not None
                and not (
                    node.source_range.start_line
                    <= line_number
                    <= node.source_range.end_line
                )
            ):
                continue
            candidates.append(node)

        if event_type == "source_command_execution":
            return self._match_shell_command_node(candidates, attributes)
        if event_type == "source_coverage_execution":
            return self._match_node_coverage_nodes(candidates, attributes)
        if event_type == "line":
            return [
                node
                for node in candidates
                if node.node_type == "CODE_ACTION"
                and node.operation_type not in self._MATERIAL_OPERATION_TYPES
            ]

        compatible_operations = {
            "file_read": {"file_read", "file_access"},
            "file_write": {"file_write", "file_access"},
        }.get(event_type, {event_type})
        matched: list[UEGNode] = []
        for node in candidates:
            if event_type in {"call", "return"} and node.summary == summary:
                matched.append(node)
            elif node.operation_type in compatible_operations:
                matched.append(node)
        return matched

    def _match_shell_command_node(
        self,
        candidates: list[UEGNode],
        attributes: dict[str, Any],
    ) -> list[UEGNode]:
        runtime_tokens = attributes.get("runtime_command_tokens")
        if not isinstance(runtime_tokens, list) or not all(
            isinstance(token, str) for token in runtime_tokens
        ):
            return []
        matches = [
            node
            for node in candidates
            if node.node_type == "CODE_ACTION"
            and isinstance(node.raw_text, str)
            and _shell_command_tokens(node.raw_text) == runtime_tokens
        ]
        # Bash xtrace does not expose source columns. If equal command text
        # identifies multiple actions on one source line, attributing the
        # event to any one of them would be a guess.
        return matches if len(matches) == 1 else []

    def _match_node_coverage_nodes(
        self,
        candidates: list[UEGNode],
        attributes: dict[str, Any],
    ) -> list[UEGNode]:
        spans_by_line = attributes.get("executed_column_spans")
        if not isinstance(spans_by_line, dict):
            return []
        indents_by_line = attributes.get("source_line_indents")
        if not isinstance(indents_by_line, dict):
            indents_by_line = {}
        output: list[UEGNode] = []
        action_candidates = [
            node for node in candidates if node.node_type == "CODE_ACTION"
        ]
        for node in action_candidates:
            source_range = node.source_range
            if source_range is None or source_range.start_line != source_range.end_line:
                continue
            start_column = source_range.start_column
            end_column = source_range.end_column
            if not isinstance(start_column, int):
                start_column = node.attributes.get("col_offset")
            if not isinstance(end_column, int):
                end_column = node.attributes.get("end_col_offset")
            line_indent = indents_by_line.get(str(source_range.start_line), 0)
            if not isinstance(line_indent, int) or line_indent < 0:
                return []
            if not isinstance(start_column, int) or not isinstance(end_column, int):
                # A whole-line action is accepted only if it is the sole
                # action on the line and V8 covers its entire raw-text span.
                same_line_actions = [
                    candidate
                    for candidate in action_candidates
                    if candidate.source_range is not None
                    and candidate.source_range.start_line
                    == source_range.start_line
                ]
                if len(same_line_actions) != 1 or not isinstance(node.raw_text, str):
                    continue
                start_column = len(node.raw_text) - len(node.raw_text.lstrip())
                end_column = start_column + len(node.raw_text.strip())
            start_column += line_indent
            end_column += line_indent
            raw_spans = spans_by_line.get(str(source_range.start_line))
            if not isinstance(raw_spans, list):
                continue
            spans = [
                (span[0], span[1])
                for span in raw_spans
                if (
                    isinstance(span, list)
                    and len(span) == 2
                    and isinstance(span[0], int)
                    and isinstance(span[1], int)
                )
            ]
            if self._column_range_is_covered(start_column, end_column, spans):
                output.append(node)
        return output

    @staticmethod
    def _column_range_is_covered(
        start_column: int,
        end_column: int,
        spans: list[tuple[int, int]],
    ) -> bool:
        if start_column < 0 or start_column >= end_column:
            return False
        cursor = start_column
        for span_start, span_end in sorted(spans):
            if span_end <= cursor:
                continue
            if span_start > cursor:
                return False
            cursor = max(cursor, span_end)
            if cursor >= end_column:
                return True
        return False

    @staticmethod
    def _realized_object_ref(
        *,
        node: UEGNode,
        payload: dict[str, object],
    ) -> str | None:
        if isinstance(node.object_ref, str) and node.object_ref:
            return node.object_ref
        call_name = node.attributes.get("call_name")
        if isinstance(call_name, str) and call_name:
            return call_name
        runtime_tokens = (
            payload.get("attributes", {}).get("runtime_command_tokens")
            if isinstance(payload.get("attributes"), dict)
            else None
        )
        if isinstance(runtime_tokens, list) and runtime_tokens and isinstance(
            runtime_tokens[0], str
        ):
            return runtime_tokens[0]
        return None

    @staticmethod
    def _realized_arguments_summary(
        *,
        node: UEGNode,
        payload: dict[str, object],
    ) -> str | None:
        attributes = payload.get("attributes")
        runtime_command = (
            attributes.get("runtime_command")
            if isinstance(attributes, dict)
            else None
        )
        if isinstance(runtime_command, str) and runtime_command:
            return runtime_command[:500]
        arguments = node.attributes.get("arguments")
        if isinstance(arguments, str) and arguments:
            return arguments[:500]
        if isinstance(node.raw_text, str) and node.raw_text:
            return node.raw_text[:500]
        return None

    def _append_code_event(
        self,
        *,
        payload: dict[str, object],
        node: UEGNode | None,
        attributes: dict[str, Any],
    ) -> None:
        event_type = str(payload.get("event_type") or "sandbox_event")
        summary = str(payload.get("summary") or "")
        node_id = node.node_id if node is not None else None
        raw_event = {
            "event_type": event_type,
            "summary": summary,
            "node_id": node_id,
            "layer": "code",
            "object_ref": payload.get("object_ref"),
            "arguments_summary": payload.get("arguments_summary"),
            "attributes": attributes,
        }
        self.raw_trace.append(raw_event)
        self.trace.append(
            ExecutionEvent(
                event_type=event_type,
                summary=summary,
                node_id=node_id,
                layer="code",
                object_ref=payload.get("object_ref"),
                arguments_summary=payload.get("arguments_summary"),
                attributes=attributes,
            )
        )

    @staticmethod
    def _normalize_source_file(source_file: str) -> str:
        return Path(source_file).as_posix()


class TracedSkillAgentRuntime:
    def __init__(
        self,
        tool_dispatcher: ToolDispatcher,
        instruction_action_materializer: InstructionActionMaterializer | None = None,
        llm_client: StructuredLLMClient | None = None,
        prompt_loader: PromptAssetLoader | None = None,
        final_response_synthesizer: FinalResponseSynthesizer | None = None,
    ) -> None:
        self.tool_dispatcher = tool_dispatcher
        self.instruction_action_materializer = (
            instruction_action_materializer or InstructionActionMaterializer()
        )
        self.final_response_synthesizer = (
            final_response_synthesizer
            or FinalResponseSynthesizer(
                llm_client=llm_client,
                prompt_loader=prompt_loader,
            )
        )

    def execute(
        self,
        *,
        source_bundle_root: Path,
        sandbox_root: Path,
        ueg: UnifiedExecutionGraph,
        prompt: str,
        run_id: str,
        mode: str,
        instruction_node_ids: list[str],
    ) -> AgentRuntimeExecutionOutcome:
        recorder = RuntimeTraceRecorder(ueg)
        notes = [
            "Executed with a traced agent runtime that resolves tools from a registry and records unified tool-call telemetry.",
        ]
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        final_output = ""
        direct_outputs: list[dict[str, str]] = []
        status = "completed"
        instruction_materialization_count = 0

        recorder.append(
            event_type="agent_run_start",
            summary="Start sandboxed skill execution",
            layer="agent",
            object_ref=str(source_bundle_root),
            arguments_summary=prompt[:200] or None,
            attributes={
                "mode": mode,
                "run_id": run_id,
                "instruction_count": len(instruction_node_ids),
                "registered_tools": self.tool_dispatcher.available_tool_names(),
            },
        )
        recorder.append(
            event_type="execution_plan_selected",
            summary="Select instruction execution plan",
            layer="agent",
            object_ref=str(source_bundle_root),
            arguments_summary=", ".join(instruction_node_ids[:12]) or None,
            attributes={
                "selected_instruction_node_ids": instruction_node_ids,
                "registered_tools": self.tool_dispatcher.available_tool_names(),
            },
        )

        for node_id in instruction_node_ids:
            node = ueg.node_by_id(node_id)
            if node is None:
                continue
            self._append_instruction_step(recorder, node=node, phase="start")

            tool_requests = self._build_tool_requests_for_node(
                node=node,
                sandbox_root=sandbox_root,
                prompt=prompt,
                run_id=run_id,
            )
            if tool_requests:
                recorder.append(
                    event_type="instruction_tools_selected",
                    summary=f"Select {len(tool_requests)} tool call(s) for {node.summary}",
                    node_id=node.node_id,
                    layer="agent",
                    object_ref=node.source_file,
                    arguments_summary=", ".join(request.tool_name for request in tool_requests),
                    attributes={
                        "instruction_node_id": node.node_id,
                        "tool_targets": [request.target for request in tool_requests],
                    },
                    mark_executed=False,
                )

            for request in tool_requests:
                recorder.append(
                    event_type="tool_resolution",
                    summary=f"Resolve {request.tool_name} for {request.target}",
                    node_id=node.node_id,
                    layer="agent",
                    object_ref=request.target,
                    attributes={
                        "tool_name": request.tool_name,
                        "tool_call_id": request.tool_call_id,
                        "instruction_node_id": node.node_id,
                    },
                    mark_executed=False,
                )
                recorder.append(
                    event_type="tool_call_start",
                    summary=f"Invoke {request.tool_name} on {request.target}",
                    node_id=node.node_id,
                    layer="agent",
                    object_ref=request.target,
                    arguments_summary=prompt[:200] or None,
                    attributes={
                        "tool_name": request.tool_name,
                        "tool_call_id": request.tool_call_id,
                        "instruction_node_id": node.node_id,
                    },
                )
                tool_result = self._invoke_tool(request)
                notes.extend(tool_result.notes)
                if tool_result.error:
                    notes.append(tool_result.error)
                if tool_result.status != "completed":
                    status = "failed"
                    recorder.append(
                        event_type="tool_call_error",
                        summary=f"{request.tool_name} failed on {request.target}",
                        node_id=node.node_id,
                        layer="agent",
                        object_ref=request.target,
                        arguments_summary=tool_result.error,
                        attributes={
                            "tool_name": request.tool_name,
                            "tool_call_id": request.tool_call_id,
                            "instruction_node_id": node.node_id,
                            "status": tool_result.status,
                        },
                    )
                if tool_result.final_output:
                    final_output = tool_result.final_output
                    if self._is_direct_result_text(
                        tool_result=tool_result,
                        target=request.target,
                    ):
                        direct_outputs.append(
                            {
                                "evidence_ref": (
                                    "execution.tool_output."
                                    f"{len(direct_outputs) + 1:04d}"
                                ),
                                "tool_name": request.tool_name,
                                "target": request.target,
                                "output": tool_result.final_output,
                            }
                        )
                if tool_result.stdout:
                    stdout_chunks.append(tool_result.stdout)
                if tool_result.stderr:
                    stderr_chunks.append(tool_result.stderr)

                self._append_bundled_command_materialization(
                    recorder=recorder,
                    node=node,
                    request=request,
                    tool_result=tool_result,
                )
                recorder.append_tool_trace(
                    script_path=request.target,
                    tool_name=request.tool_name,
                    tool_call_id=request.tool_call_id,
                    instruction_node_id=node.node_id,
                    trace_events=tool_result.trace_events,
                )
                recorder.append(
                    event_type="tool_observation",
                    summary=f"Observe {request.tool_name} result from {request.target}",
                    node_id=node.node_id,
                    layer="agent",
                    object_ref=request.target,
                    arguments_summary=tool_result.final_output[:200] or None,
                    attributes={
                        "tool_name": request.tool_name,
                        "tool_call_id": request.tool_call_id,
                        "instruction_node_id": node.node_id,
                        "status": tool_result.status,
                    },
                )
                recorder.append(
                    event_type="tool_call_end",
                    summary=f"Finish {request.tool_name} on {request.target}",
                    node_id=node.node_id,
                    layer="agent",
                    object_ref=request.target,
                    arguments_summary=tool_result.final_output[:200] or None,
                    attributes={
                        "tool_name": request.tool_name,
                        "tool_call_id": request.tool_call_id,
                        "instruction_node_id": node.node_id,
                        "status": tool_result.status,
                    },
                )

            if not tool_requests:
                materialization = self.instruction_action_materializer.materialize(
                    node=node,
                    sandbox_root=sandbox_root,
                )
                if materialization is not None:
                    instruction_materialization_count += 1
                    source_line = (
                        node.source_range.start_line
                        if node.source_range is not None
                        else None
                    )
                    recorder.append(
                        event_type=materialization.event_type,
                        summary=materialization.summary,
                        node_id=node.node_id,
                        layer="instruction",
                        object_ref=materialization.object_ref,
                        arguments_summary=materialization.arguments_summary,
                        attributes={
                            **materialization.attributes,
                            "instruction_node_id": node.node_id,
                            "instruction_materialization": "sandbox_local",
                            "material_operation": materialization.event_type,
                            "static_operation": node.operation_type,
                            "source_file": node.source_file,
                            "line_number": source_line,
                        },
                    )
                    if materialization.note:
                        notes.append(materialization.note)

            self._append_instruction_step(recorder, node=node, phase="end")

        stdout = "\n".join(
            chunk for chunk in stdout_chunks if chunk
        ).strip()
        stderr = "\n".join(
            chunk for chunk in stderr_chunks if chunk
        ).strip()
        telemetry_fallback = final_output or self._fallback_output(recorder.trace)
        final_response = self.final_response_synthesizer.synthesize(
            prompt=prompt,
            status=status,
            trace=recorder.trace,
            selected_instruction_node_ids=instruction_node_ids,
            stdout=stdout,
            stderr=stderr,
            direct_outputs=direct_outputs,
            telemetry_fallback=telemetry_fallback,
        )
        final_output = final_response.final_output
        if not final_response.grounded:
            notes.append(
                "The final user-visible response is not grounded task-output "
                "evidence and cannot establish GoalSat."
            )

        recorder.append(
            event_type="agent_run_end",
            summary="End sandboxed skill execution",
            layer="agent",
            object_ref=str(source_bundle_root),
            arguments_summary=final_output[:200] or None,
            attributes={
                "status": status,
                "executed_node_count": len(recorder.executed_node_ids),
                "instruction_materialization_count": instruction_materialization_count,
                "tool_call_count": sum(1 for event in recorder.trace if event.event_type == "tool_call_start"),
                "final_output_grounded": final_response.grounded,
                "final_output_strategy": final_response.strategy,
            },
        )

        return AgentRuntimeExecutionOutcome(
            trace=recorder.trace,
            raw_trace=recorder.raw_trace,
            executed_node_ids=recorder.executed_node_ids,
            final_output=final_output,
            stdout=stdout,
            stderr=stderr,
            status=status,
            notes=notes,
            metadata={
                "execution_strategy": "sandboxed_skill_agent",
                "runtime_type": "traced_agent_runtime",
                "runtime_class": self.__class__.__name__,
                "tool_dispatcher_strategy": "registry_resolved",
                "registered_tools": self.tool_dispatcher.available_tool_names(),
                "bundle_source_root": str(source_bundle_root),
                "bundle_execution_mode": "full_skill_bundle",
                "instruction_materialization_count": instruction_materialization_count,
                "final_output_grounded": final_response.grounded,
                "final_output_strategy": final_response.strategy,
                "final_output_rationale": final_response.rationale,
                "final_output_evidence_refs": final_response.evidence_refs,
                "final_output_direct_result_sha256": (
                    final_response.direct_result_sha256
                ),
                "final_output_uncertainty_flags": (
                    final_response.uncertainty_flags
                ),
                "final_output_llm_attempts": final_response.attempts,
                "final_output_validation_errors": (
                    final_response.validation_errors
                ),
            },
        )

    def _append_instruction_step(
        self,
        recorder: RuntimeTraceRecorder,
        *,
        node: UEGNode,
        phase: str,
    ) -> None:
        recorder.append(
            event_type=f"instruction_step_{phase}",
            summary=node.summary,
            node_id=node.node_id,
            layer=node.layer,
            object_ref=node.source_file,
            arguments_summary=node.raw_text,
            attributes={"risk_tags": node.risk_tags, "phase": phase},
            mark_executed=phase == "end",
        )

    def _fallback_output(self, trace: list[ExecutionEvent]) -> str:
        meaningful = [
            event.summary
            for event in trace
            if event.event_type
            not in {
                "call",
                "return",
                "line",
                "instruction_step_start",
                "instruction_step_end",
                "agent_run_start",
                "agent_run_end",
                "tool_call_start",
                "tool_call_end",
            }
            and event.summary
        ]
        if meaningful:
            return meaningful[-1]
        if trace:
            return trace[-1].summary
        return "Sandboxed skill execution completed."

    def _is_direct_result_text(
        self,
        *,
        tool_result: ToolInvocationResult,
        target: str,
    ) -> bool:
        output = tool_result.final_output.strip()
        if not output:
            return False
        if tool_result.stdout.strip():
            return True
        return output != f"Executed {target}"

    def _append_bundled_command_materialization(
        self,
        *,
        recorder: RuntimeTraceRecorder,
        node: UEGNode,
        request: ToolInvocationRequest,
        tool_result: ToolInvocationResult,
    ) -> None:
        if request.metadata.get("command_execution_mode") != "instrumented_bundled_script":
            return
        explicit_command = request.metadata.get("explicit_command")
        if not isinstance(explicit_command, str) or not explicit_command.strip():
            return
        process_started = any(
            str(event.get("event_type") or "") == "script_start"
            for event in tool_result.trace_events
            if isinstance(event, dict)
        )
        if not process_started:
            return
        try:
            command_tokens = shlex.split(explicit_command, posix=True)
        except ValueError:
            return
        if not command_tokens:
            return
        recorder.append(
            event_type="exec_command",
            summary=f"Execute explicit bundled command {command_tokens[0]}",
            node_id=node.node_id,
            layer="instruction",
            object_ref=command_tokens[0],
            arguments_summary=explicit_command[:500],
            attributes={
                "instruction_node_id": node.node_id,
                "material_operation": "exec_command",
                "runtime_evidence": "instrumented_script_start",
                "runtime_command": explicit_command[:2000],
                "runtime_command_tokens": command_tokens,
                "tool_name": request.tool_name,
                "tool_call_id": request.tool_call_id,
                "script_relative_path": request.target,
            },
        )

    def _build_tool_requests_for_node(
        self,
        *,
        node: UEGNode,
        sandbox_root: Path,
        prompt: str,
        run_id: str,
    ) -> list[ToolInvocationRequest]:
        requests: list[ToolInvocationRequest] = []
        invoked_scripts = [
            str(target)
            for target in node.attributes.get("invoked_scripts", [])
            if str(target).strip()
        ]
        command_invocations = list(
            dict.fromkeys(
                str(command).strip()
                for command in node.attributes.get("command_invocations", [])
                if str(command).strip()
            )
        )

        script_commands: dict[str, str] = {}
        unmatched_commands: list[str] = []
        for command_text in command_invocations:
            matched_target = next(
                (
                    target
                    for target in invoked_scripts
                    if self._command_references_script(command_text, target)
                ),
                None,
            )
            if matched_target is None:
                unmatched_commands.append(command_text)
            else:
                script_commands.setdefault(matched_target, command_text)

        for ordinal, target in enumerate(invoked_scripts, start=1):
            tool_name = self.tool_dispatcher.resolve_tool_name(str(target))
            explicit_command = script_commands.get(target)
            requests.append(
                ToolInvocationRequest(
                    tool_call_id=f"{run_id}:{node.node_id}:{tool_name}:{ordinal:03d}:{uuid.uuid4().hex[:8]}",
                    tool_name=tool_name,
                    target=str(target),
                    sandbox_root=str(sandbox_root),
                    prompt=prompt,
                    instruction_node_id=node.node_id,
                    arguments={"target": str(target)},
                    metadata={
                        "instruction_summary": node.summary,
                        "explicit_command": explicit_command,
                        "command_execution_mode": (
                            "instrumented_bundled_script"
                            if explicit_command is not None
                            else "referenced_bundled_script"
                        ),
                    },
                )
            )
        for ordinal, command_text in enumerate(unmatched_commands, start=1):
            requests.append(
                ToolInvocationRequest(
                    tool_call_id=(
                        f"{run_id}:{node.node_id}:inline_command:"
                        f"{ordinal:03d}:{uuid.uuid4().hex[:8]}"
                    ),
                    tool_name="inline_command",
                    target=f"inline-command-{ordinal:03d}",
                    sandbox_root=str(sandbox_root),
                    prompt=prompt,
                    instruction_node_id=node.node_id,
                    arguments={"command": command_text},
                    metadata={
                        "instruction_summary": node.summary,
                        "explicit_command": command_text,
                        "command_execution_mode": "sandboxed_inline_command",
                    },
                )
            )
        return requests

    @staticmethod
    def _command_references_script(command_text: str, target: str) -> bool:
        try:
            tokens = shlex.split(command_text, posix=True)
        except ValueError:
            return False
        normalized_target = Path(target).as_posix()
        target_name = Path(target).name
        executable = tokens[0]
        if (
            Path(executable).as_posix() == normalized_target
            or Path(executable).name == target_name
        ):
            return True
        if Path(executable).name not in {
            "bash",
            "deno",
            "node",
            "npx",
            "python",
            "python3",
            "sh",
            "ts-node",
        }:
            return False
        for token in tokens[1:]:
            if token.startswith("-"):
                continue
            return (
                Path(token).as_posix() == normalized_target
                or Path(token).name == target_name
            )
        return False

    def _invoke_tool(self, request: ToolInvocationRequest) -> ToolInvocationResult:
        try:
            return self.tool_dispatcher.invoke(request)
        except Exception as exc:
            return ToolInvocationResult(
                status="failed",
                final_output="",
                stdout="",
                stderr="",
                trace_events=[],
                notes=["The tool dispatcher raised an exception before producing a tool result."],
                error=str(exc),
                metadata={"tool_name": request.tool_name, "target": request.target},
            )
