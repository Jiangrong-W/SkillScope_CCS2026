from __future__ import annotations

import ast
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from skillscope.common.models import ActionGraph, SkillBundle, SourceRange, UEGEdge, UEGNode


JS_CALL_RE = re.compile(
    r"(?<![\w$])([A-Za-z_$][\w$]*(?:(?:\?|\!)?\.[A-Za-z_$][\w$]*)*)\s*\("
)
JS_IF_RE = re.compile(r"^\s*if\s*\((.+?)\)\s*(.*)$")
JS_ASSIGN_RE = re.compile(
    r"(?:^|[;{}]\s*)(?:const|let|var)\s+([A-Za-z_$][\w$]*)"
    r"(?:\s*:\s*[^=;]+)?\s*="
)
JS_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_$][\w$]*\b")
JS_RESERVED_CALLS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "function",
    "return",
    "typeof",
}


@dataclass(slots=True)
class _PythonScopeState:
    previous_node_id: str | None
    first_node_id: str | None = None
    variable_producers: dict[str, str] = field(default_factory=dict)
    parameter_nodes: dict[str, str] = field(default_factory=dict)
    pending_control_edges: list[tuple[str, str]] = field(default_factory=list)


PythonFunctionDef = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True, slots=True)
class _ShellCommandSegment:
    text: str
    start_column: int
    end_column: int
    operator_before: str | None = None
    operator_after: str | None = None


class CodeGraphBuilder:
    def build(self, bundle: SkillBundle) -> list[ActionGraph]:
        graphs: list[ActionGraph] = []
        root = Path(bundle.root_path)

        for artifact in bundle.script_files:
            file_path = root / artifact.relative_path
            if file_path.suffix.lower() == ".py":
                graphs.append(self._build_python_graph(bundle.bundle_id, artifact.relative_path, file_path))
            elif file_path.suffix.lower() == ".sh":
                graphs.append(self._build_shell_graph(bundle.bundle_id, artifact.relative_path, file_path))
            elif file_path.suffix.lower() in {".js", ".ts"}:
                graphs.append(
                    self._build_javascript_typescript_graph(
                        bundle.bundle_id,
                        artifact.relative_path,
                        file_path,
                    )
                )
            else:
                graphs.append(
                    self._build_unparsed_script_graph(
                        bundle.bundle_id,
                        artifact.relative_path,
                        file_path,
                    )
                )

        return graphs

    def _build_python_graph(self, skill_id: str, relative_path: str, file_path: Path) -> ActionGraph:
        graph = ActionGraph(graph_id=f"{skill_id}:code:{relative_path}", layer="code")
        entry_id = f"{graph.graph_id}:entry"
        return_id = f"{graph.graph_id}:return"
        graph.metadata["entry_node_id"] = entry_id
        graph.metadata["return_node_id"] = return_id
        graph.metadata["relative_path"] = relative_path
        graph.metadata["parser_metadata"] = {
            "language": "python",
            "parser": "python_ast",
            "parser_kind": "ast",
            "status": "parsed",
            "ast_available": True,
        }
        graph.metadata["construction_strategy"] = {
            "ast_parsing": "python_ast",
            "data_flow": "lightweight_intra_procedural",
            "control_flow": "conservative_structured_cfg",
            "preserves": [
                "function_calls",
                "data_propagation",
                "data_merge",
                "return_flow",
                "conditional_branches",
                "exception_handlers",
                "loops",
                "context_managers",
                "match_cases",
                "async_function_boundaries",
            ],
        }

        graph.add_node(UEGNode(node_id=entry_id, layer="code", node_type="CODE_ENTRY", summary=f"Enter {relative_path}", source_file=relative_path))

        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            graph.metadata["parser_metadata"]["status"] = "parse_error"
            graph.metadata["parser_metadata"]["error_type"] = type(exc).__name__
            node_id = f"{graph.graph_id}:parse-error"
            graph.add_node(
                UEGNode(
                    node_id=node_id,
                    layer="code",
                    node_type="CODE_ACTION",
                    summary=f"Unable to parse {relative_path}",
                    source_file=relative_path,
                    operation_type="parse_error",
                    risk_tags=["analysis_gap"],
                )
            )
            graph.add_edge(UEGEdge(source=entry_id, target=node_id, edge_type="SEQUENTIAL"))
            graph.add_node(UEGNode(node_id=return_id, layer="code", node_type="CODE_RETURN", summary=f"Return from {relative_path}", source_file=relative_path))
            graph.add_edge(UEGEdge(source=node_id, target=return_id, edge_type="SEQUENTIAL"))
            self._finalize_graph_metadata(graph)
            return graph

        local_functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        graph.metadata["local_functions"] = sorted(local_functions)
        state = _PythonScopeState(previous_node_id=entry_id)
        call_counter = 0

        for statement in tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                call_counter = self._emit_python_definition_expressions(
                    graph=graph,
                    function_def=statement,
                    relative_path=relative_path,
                    state=state,
                    local_functions=local_functions,
                    call_counter=call_counter,
                )
                continue
            call_counter = self._process_python_statement(
                graph=graph,
                statement=statement,
                relative_path=relative_path,
                state=state,
                local_functions=local_functions,
                call_counter=call_counter,
            )

        graph.metadata["function_boundaries"] = {}
        for function_name, function_def in local_functions.items():
            function_state = _PythonScopeState(previous_node_id=None)
            first_counter = call_counter
            call_counter = self._emit_parameter_nodes(
                graph=graph,
                function_name=function_name,
                function_def=function_def,
                relative_path=relative_path,
                state=function_state,
                call_counter=call_counter,
            )
            for statement in function_def.body:
                call_counter = self._process_python_statement(
                    graph=graph,
                    statement=statement,
                    relative_path=relative_path,
                    state=function_state,
                    local_functions=local_functions,
                    call_counter=call_counter,
                    scope_name=function_name,
                )
            function_exits = self._python_control_exits(function_state)
            if (
                call_counter > first_counter
                and function_state.first_node_id is not None
                and function_exits
            ):
                graph.metadata["function_boundaries"][function_name] = {
                    "first_node_id": function_state.first_node_id,
                    "last_node_id": function_exits[0][0],
                    "last_node_ids": [
                        node_id for node_id, _edge_type in function_exits
                    ],
                    "parameter_nodes": dict(function_state.parameter_nodes),
                }

        graph.add_node(UEGNode(node_id=return_id, layer="code", node_type="CODE_RETURN", summary=f"Return from {relative_path}", source_file=relative_path))
        for source_id, edge_type in self._python_control_exits(state):
            graph.add_edge(
                UEGEdge(source=source_id, target=return_id, edge_type=edge_type)
            )
        self._add_local_call_edges(graph)
        self._finalize_graph_metadata(graph)
        return graph

    def _build_shell_graph(self, skill_id: str, relative_path: str, file_path: Path) -> ActionGraph:
        graph = ActionGraph(graph_id=f"{skill_id}:code:{relative_path}", layer="code")
        entry_id = f"{graph.graph_id}:entry"
        return_id = f"{graph.graph_id}:return"
        graph.metadata["entry_node_id"] = entry_id
        graph.metadata["return_node_id"] = return_id
        graph.metadata["relative_path"] = relative_path
        graph.metadata["parser_metadata"] = {
            "language": "shell",
            "parser": "stdlib_top_level_command_scanner",
            "parser_kind": "lexical",
            "status": "parsed_approximately",
            "ast_available": False,
        }
        graph.metadata["construction_strategy"] = {
            "ast_parsing": "unavailable",
            "lexical_parsing": "quote_aware_top_level_shell_segments",
            "data_flow": "command_segment_flow",
        }

        graph.add_node(UEGNode(node_id=entry_id, layer="code", node_type="CODE_ENTRY", summary=f"Enter {relative_path}", source_file=relative_path))

        previous_id = entry_id
        for line_number, raw_line in enumerate(
            file_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            for segment_index, segment in enumerate(
                self._shell_command_segments(raw_line),
                start=1,
            ):
                node_id = (
                    f"{graph.graph_id}:line-{line_number:03d}:"
                    f"segment-{segment_index:03d}"
                )
                operation_type, risk_tags = self._classify_shell_line(
                    segment.text
                )
                graph.add_node(
                    UEGNode(
                        node_id=node_id,
                        layer="code",
                        node_type="CODE_ACTION",
                        summary=f"Shell command: {segment.text[:80]}",
                        source_file=relative_path,
                        source_range=SourceRange(
                            start_line=line_number,
                            end_line=line_number,
                            start_column=segment.start_column,
                            end_column=segment.end_column,
                        ),
                        raw_text=segment.text,
                        operation_type=operation_type,
                        object_ref=self._shell_object_ref(segment.text),
                        risk_tags=risk_tags,
                        attributes={
                            "shell_segment_index": segment_index,
                            "shell_operator_before": segment.operator_before,
                            "shell_operator_after": segment.operator_after,
                            "col_offset": segment.start_column,
                            "end_col_offset": segment.end_column,
                            "parser": "quote_aware_top_level_shell_segments",
                        },
                    )
                )
                edge_type = self._shell_edge_type(
                    segment.operator_before
                )
                graph.add_edge(
                    UEGEdge(
                        source=previous_id,
                        target=node_id,
                        edge_type=edge_type,
                        attributes={
                            "shell_operator": segment.operator_before,
                        },
                    )
                )
                previous_id = node_id

        graph.add_node(UEGNode(node_id=return_id, layer="code", node_type="CODE_RETURN", summary=f"Return from {relative_path}", source_file=relative_path))
        graph.add_edge(UEGEdge(source=previous_id, target=return_id, edge_type="SEQUENTIAL"))
        self._finalize_graph_metadata(graph)
        return graph

    def _shell_command_segments(
        self,
        raw_line: str,
    ) -> list[_ShellCommandSegment]:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            return []

        segments: list[_ShellCommandSegment] = []
        segment_start = 0
        operator_before: str | None = None
        quote: str | None = None
        escaped = False
        paren_depth = 0
        brace_depth = 0
        test_depth = 0
        index = 0

        def append_segment(end_column: int, operator_after: str | None) -> None:
            nonlocal segment_start, operator_before
            start_column = segment_start
            while (
                start_column < end_column
                and raw_line[start_column].isspace()
            ):
                start_column += 1
            trimmed_end = end_column
            while (
                trimmed_end > start_column
                and raw_line[trimmed_end - 1].isspace()
            ):
                trimmed_end -= 1
            if start_column < trimmed_end:
                segments.append(
                    _ShellCommandSegment(
                        text=raw_line[start_column:trimmed_end],
                        start_column=start_column,
                        end_column=trimmed_end,
                        operator_before=operator_before,
                        operator_after=operator_after,
                    )
                )
            operator_before = operator_after

        while index < len(raw_line):
            char = raw_line[index]
            if escaped:
                escaped = False
                index += 1
                continue
            if quote == "'":
                if char == "'":
                    quote = None
                index += 1
                continue
            if quote in {'"', "`"}:
                if char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                index += 1
                continue
            if char == "\\":
                escaped = True
                index += 1
                continue
            if char in {"'", '"', "`"}:
                quote = char
                index += 1
                continue
            if raw_line.startswith("[[", index):
                test_depth += 1
                index += 2
                continue
            if test_depth and raw_line.startswith("]]", index):
                test_depth -= 1
                index += 2
                continue
            if char == "(":
                paren_depth += 1
                index += 1
                continue
            if char == ")" and paren_depth:
                paren_depth -= 1
                index += 1
                continue
            if char == "{":
                brace_depth += 1
                index += 1
                continue
            if char == "}" and brace_depth:
                brace_depth -= 1
                index += 1
                continue
            if paren_depth or brace_depth or test_depth:
                index += 1
                continue
            if char == "#" and (
                index == segment_start
                or index == 0
                or raw_line[index - 1].isspace()
            ):
                append_segment(index, None)
                return segments

            operator = self._shell_operator_at(raw_line, index)
            if operator is None:
                index += 1
                continue
            append_segment(index, operator)
            index += len(operator)
            segment_start = index

        append_segment(len(raw_line), None)
        return segments

    def _shell_operator_at(self, line: str, index: int) -> str | None:
        for operator in (";;&", "&&", "||", "|&", ";;", ";&", ";", "|", "&"):
            if not line.startswith(operator, index):
                continue
            if operator == "&" and (
                (index + 1 < len(line) and line[index + 1] == ">")
                or (index > 0 and line[index - 1] == ">")
            ):
                return None
            return operator
        return None

    def _shell_edge_type(self, operator_before: str | None) -> str:
        if operator_before == "&&":
            return "CONDITIONAL_TRUE"
        if operator_before == "||":
            return "CONDITIONAL_FALSE"
        return "SEQUENTIAL"

    def _build_javascript_typescript_graph(
        self,
        skill_id: str,
        relative_path: str,
        file_path: Path,
    ) -> ActionGraph:
        suffix = file_path.suffix.lower()
        language = "typescript" if suffix == ".ts" else "javascript"
        graph = ActionGraph(graph_id=f"{skill_id}:code:{relative_path}", layer="code")
        entry_id = f"{graph.graph_id}:entry"
        return_id = f"{graph.graph_id}:return"
        graph.metadata["entry_node_id"] = entry_id
        graph.metadata["return_node_id"] = return_id
        graph.metadata["relative_path"] = relative_path
        graph.metadata["parser_metadata"] = {
            "language": language,
            "parser": "stdlib_javascript_typescript_scanner",
            "parser_kind": "conservative_lexical",
            "status": "parsed_approximately",
            "ast_available": False,
            "limitations": [
                "No JavaScript/TypeScript AST dependency is bundled.",
                "Control and data dependencies are conservative lexical approximations.",
            ],
        }
        graph.metadata["construction_strategy"] = {
            "ast_parsing": "unavailable",
            "lexical_parsing": "javascript_typescript_call_and_statement_scan",
            "data_flow": "lightweight_identifier_propagation",
            "control_flow": "immediate_if_guard_recovery",
            "preserves": [
                "function_calls",
                "risk-bearing_property_access",
                "immediate_predicate_guards",
                "approximate_data_dependencies",
                "return_flow",
            ],
        }
        graph.add_node(
            UEGNode(
                node_id=entry_id,
                layer="code",
                node_type="CODE_ENTRY",
                summary=f"Enter {relative_path}",
                source_file=relative_path,
                object_ref=relative_path,
            )
        )

        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            graph.metadata["parser_metadata"]["status"] = "read_error"
            graph.metadata["parser_metadata"]["error_type"] = type(exc).__name__
            node_id = f"{graph.graph_id}:read-error"
            graph.add_node(
                UEGNode(
                    node_id=node_id,
                    layer="code",
                    node_type="CODE_ACTION",
                    summary=f"Unable to inspect {relative_path}",
                    source_file=relative_path,
                    operation_type="parse_error",
                    object_ref=relative_path,
                    risk_tags=["analysis_gap"],
                )
            )
            graph.add_edge(UEGEdge(source=entry_id, target=node_id, edge_type="SEQUENTIAL"))
            graph.add_node(
                UEGNode(
                    node_id=return_id,
                    layer="code",
                    node_type="CODE_RETURN",
                    summary=f"Return from {relative_path}",
                    source_file=relative_path,
                    object_ref=relative_path,
                )
            )
            graph.add_edge(UEGEdge(source=node_id, target=return_id, edge_type="SEQUENTIAL"))
            self._finalize_graph_metadata(graph)
            return graph

        previous_id = entry_id
        pending_predicate_id: str | None = None
        variable_producers: dict[str, str] = {}
        node_counter = 0

        for line_number, raw_line in enumerate(lines, start=1):
            line = self._strip_javascript_comments(raw_line).strip()
            if not line or line in {"{", "}", "};"}:
                continue
            if language == "typescript" and re.match(
                r"^(?:interface|type|enum|namespace|declare)\b",
                line,
            ):
                continue

            if_match = JS_IF_RE.match(line)
            if if_match:
                condition = if_match.group(1).strip()
                node_counter += 1
                predicate_id = f"{graph.graph_id}:predicate-{node_counter:03d}"
                graph.add_node(
                    UEGNode(
                        node_id=predicate_id,
                        layer="code",
                        node_type="CODE_PREDICATE",
                        summary=f"If {condition[:140]}",
                        source_file=relative_path,
                        source_range=SourceRange(
                            start_line=line_number,
                            end_line=line_number,
                        ),
                        raw_text=condition,
                        operation_type="predicate",
                        object_ref=condition[:160],
                        attributes={"parser_fidelity": "conservative_lexical"},
                    )
                )
                graph.add_edge(
                    UEGEdge(
                        source=previous_id,
                        target=predicate_id,
                        edge_type="SEQUENTIAL",
                    )
                )
                previous_id = predicate_id
                pending_predicate_id = predicate_id

            emitted_ids: list[str] = []
            for call_match in JS_CALL_RE.finditer(line):
                callee = call_match.group(1)
                if callee.lower() in JS_RESERVED_CALLS:
                    continue
                if re.search(r"\bfunction\s*$", line[: call_match.start()]):
                    continue

                arguments, call_end = self._javascript_call_extent(
                    line,
                    call_match.end() - 1,
                )
                call_start = call_match.start()
                call_text = line[call_start:call_end]
                operation_type, risk_tags = self._classify_javascript_action(
                    callee,
                    line,
                )
                object_ref = self._javascript_object_ref(
                    callee,
                    arguments,
                    operation_type,
                )
                node_counter += 1
                node_id = f"{graph.graph_id}:call-{node_counter:03d}"
                graph.add_node(
                    UEGNode(
                        node_id=node_id,
                        layer="code",
                        node_type="CODE_ACTION",
                        summary=f"Call {callee}",
                        source_file=relative_path,
                        source_range=SourceRange(
                            start_line=line_number,
                            end_line=line_number,
                        ),
                        raw_text=call_text[:500],
                        operation_type=operation_type,
                        object_ref=object_ref,
                        risk_tags=risk_tags,
                        attributes={
                            "call_name": callee,
                            "arguments": arguments,
                            "col_offset": call_start,
                            "end_col_offset": call_end,
                            "parser_fidelity": "conservative_lexical",
                        },
                    )
                )
                incoming_type = (
                    "CONDITIONAL_TRUE"
                    if pending_predicate_id is not None and not emitted_ids
                    else "SEQUENTIAL"
                )
                graph.add_edge(
                    UEGEdge(
                        source=previous_id,
                        target=node_id,
                        edge_type=incoming_type,
                    )
                )
                self._add_javascript_data_edges(
                    graph,
                    node_id,
                    line,
                    variable_producers,
                )
                previous_id = node_id
                emitted_ids.append(node_id)

            if not emitted_ids:
                operation_type, risk_tags = self._classify_javascript_statement(line)
                if operation_type is not None:
                    node_counter += 1
                    node_id = f"{graph.graph_id}:statement-{node_counter:03d}"
                    object_ref = self._javascript_statement_object_ref(
                        line,
                        operation_type,
                    )
                    graph.add_node(
                        UEGNode(
                            node_id=node_id,
                            layer="code",
                            node_type="CODE_ACTION",
                            summary=self._javascript_statement_summary(
                                line,
                                operation_type,
                            ),
                            source_file=relative_path,
                            source_range=SourceRange(
                                start_line=line_number,
                                end_line=line_number,
                            ),
                            raw_text=line[:500],
                            operation_type=operation_type,
                            object_ref=object_ref,
                            risk_tags=risk_tags,
                            attributes={"parser_fidelity": "conservative_lexical"},
                        )
                    )
                    incoming_type = (
                        "CONDITIONAL_TRUE"
                        if pending_predicate_id is not None
                        else "SEQUENTIAL"
                    )
                    graph.add_edge(
                        UEGEdge(
                            source=previous_id,
                            target=node_id,
                            edge_type=incoming_type,
                        )
                    )
                    self._add_javascript_data_edges(
                        graph,
                        node_id,
                        line,
                        variable_producers,
                    )
                    previous_id = node_id
                    emitted_ids.append(node_id)

            assignment_match = JS_ASSIGN_RE.search(line)
            if assignment_match and emitted_ids:
                variable_producers[assignment_match.group(1)] = emitted_ids[-1]
            if emitted_ids:
                pending_predicate_id = None

        graph.add_node(
            UEGNode(
                node_id=return_id,
                layer="code",
                node_type="CODE_RETURN",
                summary=f"Return from {relative_path}",
                source_file=relative_path,
                object_ref=relative_path,
            )
        )
        graph.add_edge(
            UEGEdge(
                source=previous_id,
                target=return_id,
                edge_type="SEQUENTIAL",
            )
        )
        self._finalize_graph_metadata(graph)
        return graph

    def _build_unparsed_script_graph(
        self,
        skill_id: str,
        relative_path: str,
        file_path: Path,
    ) -> ActionGraph:
        graph = ActionGraph(graph_id=f"{skill_id}:code:{relative_path}", layer="code")
        entry_id = f"{graph.graph_id}:entry"
        return_id = f"{graph.graph_id}:return"
        graph.metadata["entry_node_id"] = entry_id
        graph.metadata["return_node_id"] = return_id
        graph.metadata["relative_path"] = relative_path
        graph.metadata["parser_metadata"] = {
            "language": file_path.suffix.lower().lstrip(".") or "unknown",
            "parser": "none",
            "parser_kind": "unsupported",
            "status": "unsupported_but_retained",
            "ast_available": False,
        }
        graph.metadata["construction_strategy"] = {
            "ast_parsing": "unavailable",
            "preserves": ["analysis_gap", "source_provenance", "return_flow"],
        }
        graph.add_node(
            UEGNode(
                node_id=entry_id,
                layer="code",
                node_type="CODE_ENTRY",
                summary=f"Enter {relative_path}",
                source_file=relative_path,
                object_ref=relative_path,
            )
        )
        try:
            preview = file_path.read_text(encoding="utf-8")[:500]
        except (OSError, UnicodeDecodeError):
            preview = ""
        action_id = f"{graph.graph_id}:unparsed"
        graph.add_node(
            UEGNode(
                node_id=action_id,
                layer="code",
                node_type="CODE_ACTION",
                summary=f"Unparsed script {relative_path}",
                source_file=relative_path,
                raw_text=preview,
                operation_type="unparsed_script",
                object_ref=relative_path,
                risk_tags=["analysis_gap"],
            )
        )
        graph.add_edge(
            UEGEdge(source=entry_id, target=action_id, edge_type="SEQUENTIAL")
        )
        graph.add_node(
            UEGNode(
                node_id=return_id,
                layer="code",
                node_type="CODE_RETURN",
                summary=f"Return from {relative_path}",
                source_file=relative_path,
                object_ref=relative_path,
            )
        )
        graph.add_edge(
            UEGEdge(source=action_id, target=return_id, edge_type="SEQUENTIAL")
        )
        self._finalize_graph_metadata(graph)
        return graph

    def _call_name(self, func: ast.AST) -> str:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parts: list[str] = []
            current: ast.AST | None = func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            elif isinstance(current, ast.Call):
                called_base = self._call_name(current.func)
                if called_base != "call":
                    parts.append(called_base)
            return ".".join(reversed(parts))
        return "call"

    def _classify_call(self, func_name: str, call: ast.Call) -> tuple[str, list[str]]:
        lowered = func_name.lower()
        arg_text = " ".join(ast.unparse(arg) for arg in call.args) if call.args else ""
        keyword_text = " ".join(
            f"{keyword.arg}={self._safe_ast_unparse(keyword.value)}"
            for keyword in call.keywords
            if keyword.arg
        )
        combined = f"{lowered} {arg_text} {keyword_text}".lower()

        if any(token in lowered for token in ("requests.", "httpx.", "urllib.", "socket.", "webhook", "telegram")):
            return "network_send", ["network"]
        if lowered in {
            "subprocess.run",
            "subprocess.popen",
            "subprocess.call",
            "subprocess.check_call",
            "subprocess.check_output",
            "os.system",
            "os.popen",
        }:
            return "exec_command", ["command_execution"]
        if lowered in {
            "os.remove",
            "os.unlink",
            "os.rmdir",
            "os.removedirs",
            "shutil.rmtree",
            "pathlib.path.unlink",
            "pathlib.path.rmdir",
        } or lowered.endswith((".unlink", ".rmdir")):
            return "delete", ["persistent_state", "deletion"]
        if lowered in {
            "os.rename",
            "os.renames",
            "os.replace",
            "os.mkdir",
            "os.makedirs",
            "os.chmod",
            "os.chown",
            "shutil.move",
            "shutil.copy",
            "shutil.copy2",
            "shutil.copyfile",
            "shutil.copytree",
            "pathlib.path.mkdir",
            "pathlib.path.touch",
            "pathlib.path.chmod",
            "pathlib.path.write_text",
            "pathlib.path.write_bytes",
        } or lowered.endswith(
            (".write_text", ".write_bytes", ".mkdir", ".touch", ".chmod")
        ):
            return "file_write", ["persistent_state", "file_write"]
        if lowered in {"open", "pathlib.path.open"}:
            mode = ""
            if len(call.args) > 1:
                mode = self._safe_ast_unparse(call.args[1]).strip("'\"")
            for keyword in call.keywords:
                if keyword.arg == "mode":
                    mode = self._safe_ast_unparse(keyword.value).strip("'\"")
            if any(marker in mode for marker in ("w", "a", "x", "+")):
                return "file_write", ["persistent_state", "file_write"]
            risk_tags = ["file_access"]
            if any(keyword in combined for keyword in ("history", ".ssh", ".aws", "token", "cookie")):
                risk_tags.append("sensitive_collection")
            return "file_access", risk_tags
        if lowered in {
            "pathlib.path.read_text",
        } or lowered.endswith((".read_text", ".read_bytes")):
            risk_tags = ["file_access"]
            if any(keyword in combined for keyword in ("history", ".ssh", ".aws", "token", "cookie")):
                risk_tags.append("sensitive_collection")
            return "file_access", risk_tags
        if lowered in {"os.getenv", "os.environ.get"}:
            return "read_env", ["sensitive_collection"]
        if lowered in {"uuid.getnode", "platform.node"}:
            return "collect_identifier", ["sensitive_collection"]
        return "call", []

    def _process_python_statement(
        self,
        *,
        graph: ActionGraph,
        statement: ast.stmt,
        relative_path: str,
        state: _PythonScopeState,
        local_functions: dict[str, PythonFunctionDef],
        call_counter: int,
        scope_name: str = "module",
    ) -> int:
        if isinstance(statement, ast.Assign):
            previous_counter = call_counter
            call_counter = self._emit_calls_from_expression(
                graph=graph,
                expression=statement.value,
                relative_path=relative_path,
                state=state,
                call_counter=call_counter,
                local_functions=local_functions,
                scope_name=scope_name,
            )
            referenced_names = self._extract_load_names(statement.value)
            producer_ids = self._referenced_producer_ids(referenced_names, state)
            if call_counter > previous_counter and state.previous_node_id is not None:
                producer_ids.add(state.previous_node_id)
            if self._should_materialize_merge_node(statement.value, producer_ids):
                call_counter, merge_node_id = self._emit_merge_node(
                    graph=graph,
                    expression=statement.value,
                    relative_path=relative_path,
                    state=state,
                    call_counter=call_counter,
                    producer_ids=producer_ids,
                    scope_name=scope_name,
                    summary="Merge assigned data",
                    attributes={"referenced_names": sorted(referenced_names), "kind": "assignment_merge"},
                )
                producer_ids = {merge_node_id}
            preferred_producer = (
                sorted(producer_ids)[-1] if producer_ids else None
            )
            for target in statement.targets:
                call_counter = self._emit_calls_from_expression(
                    graph=graph,
                    expression=target,
                    relative_path=relative_path,
                    state=state,
                    call_counter=call_counter,
                    local_functions=local_functions,
                    scope_name=scope_name,
                )
            if preferred_producer is not None:
                for target in statement.targets:
                    for variable_name in self._extract_store_names(target):
                        state.variable_producers[variable_name] = preferred_producer
            return call_counter

        if isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
            value = statement.value
            previous_counter = call_counter
            call_counter = self._emit_calls_from_expression(
                graph=graph,
                expression=statement.target,
                relative_path=relative_path,
                state=state,
                call_counter=call_counter,
                local_functions=local_functions,
                scope_name=scope_name,
            )
            call_counter = self._emit_calls_from_expression(
                graph=graph,
                expression=value,
                relative_path=relative_path,
                state=state,
                call_counter=call_counter,
                local_functions=local_functions,
                scope_name=scope_name,
            )
            referenced_names = self._extract_load_names(value)
            producer_ids = self._referenced_producer_ids(referenced_names, state)
            if call_counter > previous_counter and state.previous_node_id is not None:
                producer_ids.add(state.previous_node_id)
            if producer_ids:
                preferred_producer = sorted(producer_ids)[-1]
                for variable_name in self._extract_store_names(statement.target):
                    state.variable_producers[variable_name] = preferred_producer
            return call_counter

        if isinstance(statement, ast.Expr):
            return self._emit_calls_from_expression(
                graph=graph,
                expression=statement.value,
                relative_path=relative_path,
                state=state,
                call_counter=call_counter,
                local_functions=local_functions,
                scope_name=scope_name,
            )

        if isinstance(statement, ast.Return):
            previous_counter = call_counter
            call_counter = self._emit_calls_from_expression(
                graph=graph,
                expression=statement.value,
                relative_path=relative_path,
                state=state,
                call_counter=call_counter,
                local_functions=local_functions,
                scope_name=scope_name,
            )
            referenced = self._extract_load_names(statement.value)
            producer_ids = self._referenced_producer_ids(referenced, state)
            if call_counter > previous_counter and state.previous_node_id is not None:
                producer_ids.add(state.previous_node_id)
            if self._should_materialize_merge_node(statement.value, producer_ids):
                call_counter, merge_node_id = self._emit_merge_node(
                    graph=graph,
                    expression=statement.value,
                    relative_path=relative_path,
                    state=state,
                    call_counter=call_counter,
                    producer_ids=producer_ids,
                    scope_name=scope_name,
                    summary="Merge return data",
                    attributes={"referenced_names": sorted(referenced), "kind": "return_merge"},
                )
                producer_ids = {merge_node_id}
            node_id = f"{graph.graph_id}:return-value-{call_counter + 1:03d}"
            graph.add_node(
                UEGNode(
                    node_id=node_id,
                    layer="code",
                    node_type="CODE_ACTION",
                    summary="Return value",
                    source_file=relative_path,
                    source_range=SourceRange(start_line=getattr(statement, "lineno", 0), end_line=getattr(statement, "end_lineno", getattr(statement, "lineno", 0))),
                    raw_text=ast.unparse(statement.value) if statement.value is not None else None,
                    operation_type="return_value",
                    object_ref=ast.unparse(statement.value)[:160] if statement.value is not None else "None",
                    attributes={
                        "scope_name": scope_name,
                        "referenced_names": referenced,
                        **self._python_ast_position_attributes(statement.value),
                    },
                )
            )
            self._add_python_control_edges(
                graph=graph,
                state=state,
                target_node_id=node_id,
            )
            if producer_ids:
                for producer_id in sorted(producer_ids):
                    graph.add_edge(UEGEdge(source=producer_id, target=node_id, edge_type="DATA_DEP", attributes={"kind": "return_dependency"}))
            else:
                for name in referenced:
                    producer_id = state.variable_producers.get(name)
                    if producer_id is not None:
                        graph.add_edge(UEGEdge(source=producer_id, target=node_id, edge_type="DATA_DEP", attributes={"variable": name}))
            if state.first_node_id is None:
                state.first_node_id = node_id
            state.previous_node_id = node_id
            return call_counter + 1

        if isinstance(statement, ast.If):
            call_counter = self._emit_calls_from_expression(
                graph=graph,
                expression=statement.test,
                relative_path=relative_path,
                state=state,
                call_counter=call_counter,
                local_functions=local_functions,
                scope_name=scope_name,
            )

            call_counter += 1
            predicate_id = f"{graph.graph_id}:predicate-{call_counter:03d}"
            condition_text = self._safe_ast_unparse(statement.test)
            referenced_names = self._extract_load_names(statement.test)
            graph.add_node(
                UEGNode(
                    node_id=predicate_id,
                    layer="code",
                    node_type="CODE_PREDICATE",
                    summary=f"If {condition_text[:140]}",
                    source_file=relative_path,
                    source_range=SourceRange(
                        start_line=getattr(
                            statement.test,
                            "lineno",
                            getattr(statement, "lineno", 0),
                        ),
                        end_line=getattr(
                            statement.test,
                            "end_lineno",
                            getattr(statement, "lineno", 0),
                        ),
                    ),
                    raw_text=condition_text,
                    operation_type="predicate",
                    object_ref=condition_text[:160],
                    attributes={
                        "scope_name": scope_name,
                        "predicate_kind": "if",
                        "referenced_names": sorted(referenced_names),
                        **self._python_ast_position_attributes(statement.test),
                    },
                )
            )
            self._add_python_control_edges(
                graph=graph,
                state=state,
                target_node_id=predicate_id,
            )
            for producer_id in sorted(
                self._referenced_producer_ids(referenced_names, state)
            ):
                graph.add_edge(
                    UEGEdge(
                        source=producer_id,
                        target=predicate_id,
                        edge_type="DATA_DEP",
                        attributes={"kind": "predicate_dependency"},
                    )
                )
            if state.first_node_id is None:
                state.first_node_id = predicate_id
            state.previous_node_id = predicate_id

            true_state = _PythonScopeState(
                previous_node_id=None,
                first_node_id=state.first_node_id,
                variable_producers=dict(state.variable_producers),
                parameter_nodes=dict(state.parameter_nodes),
                pending_control_edges=[
                    (predicate_id, "CONDITIONAL_TRUE")
                ],
            )
            for nested in statement.body:
                call_counter = self._process_python_statement(
                    graph=graph,
                    statement=nested,
                    relative_path=relative_path,
                    state=true_state,
                    local_functions=local_functions,
                    call_counter=call_counter,
                    scope_name=scope_name,
                )

            false_state = _PythonScopeState(
                previous_node_id=None,
                first_node_id=state.first_node_id,
                variable_producers=dict(state.variable_producers),
                parameter_nodes=dict(state.parameter_nodes),
                pending_control_edges=[
                    (predicate_id, "CONDITIONAL_FALSE")
                ],
            )
            for nested in statement.orelse:
                call_counter = self._process_python_statement(
                    graph=graph,
                    statement=nested,
                    relative_path=relative_path,
                    state=false_state,
                    local_functions=local_functions,
                    call_counter=call_counter,
                    scope_name=scope_name,
                )

            state.previous_node_id = None
            state.pending_control_edges = [
                *self._python_control_exits(true_state),
                *self._python_control_exits(false_state),
            ]
            state.variable_producers = self._merge_python_branch_producers(
                original=state.variable_producers,
                true_branch=true_state.variable_producers,
                false_branch=false_state.variable_producers,
            )
            return call_counter

        if isinstance(statement, (ast.Try, ast.TryStar)):
            return self._process_python_try_statement(
                graph=graph,
                statement=statement,
                relative_path=relative_path,
                state=state,
                local_functions=local_functions,
                call_counter=call_counter,
                scope_name=scope_name,
            )

        if isinstance(statement, (ast.For, ast.AsyncFor)):
            return self._process_python_for_statement(
                graph=graph,
                statement=statement,
                relative_path=relative_path,
                state=state,
                local_functions=local_functions,
                call_counter=call_counter,
                scope_name=scope_name,
            )

        if isinstance(statement, ast.While):
            return self._process_python_while_statement(
                graph=graph,
                statement=statement,
                relative_path=relative_path,
                state=state,
                local_functions=local_functions,
                call_counter=call_counter,
                scope_name=scope_name,
            )

        if isinstance(statement, (ast.With, ast.AsyncWith)):
            return self._process_python_with_statement(
                graph=graph,
                statement=statement,
                relative_path=relative_path,
                state=state,
                local_functions=local_functions,
                call_counter=call_counter,
                scope_name=scope_name,
            )

        if isinstance(statement, ast.Match):
            return self._process_python_match_statement(
                graph=graph,
                statement=statement,
                relative_path=relative_path,
                state=state,
                local_functions=local_functions,
                call_counter=call_counter,
                scope_name=scope_name,
            )

        return self._process_python_fallback_statement(
            graph=graph,
            statement=statement,
            relative_path=relative_path,
            state=state,
            local_functions=local_functions,
            call_counter=call_counter,
            scope_name=scope_name,
        )

    def _python_branch_state(
        self,
        state: _PythonScopeState,
        pending_control_edges: list[tuple[str, str]],
    ) -> _PythonScopeState:
        return _PythonScopeState(
            previous_node_id=None,
            first_node_id=state.first_node_id,
            variable_producers=dict(state.variable_producers),
            parameter_nodes=dict(state.parameter_nodes),
            pending_control_edges=list(pending_control_edges),
        )

    def _process_python_block(
        self,
        *,
        graph: ActionGraph,
        statements: list[ast.stmt],
        relative_path: str,
        state: _PythonScopeState,
        local_functions: dict[str, PythonFunctionDef],
        call_counter: int,
        scope_name: str,
    ) -> int:
        for nested in statements:
            call_counter = self._process_python_statement(
                graph=graph,
                statement=nested,
                relative_path=relative_path,
                state=state,
                local_functions=local_functions,
                call_counter=call_counter,
                scope_name=scope_name,
            )
        return call_counter

    def _emit_python_predicate_node(
        self,
        *,
        graph: ActionGraph,
        source_node: ast.AST,
        expression: ast.AST | None,
        raw_text: str,
        summary: str,
        predicate_kind: str,
        relative_path: str,
        state: _PythonScopeState,
        call_counter: int,
        scope_name: str,
    ) -> tuple[int, str]:
        call_counter += 1
        predicate_id = f"{graph.graph_id}:predicate-{call_counter:03d}"
        position_node = expression or source_node
        referenced_names = self._extract_load_names(expression)
        graph.add_node(
            UEGNode(
                node_id=predicate_id,
                layer="code",
                node_type="CODE_PREDICATE",
                summary=summary[:180],
                source_file=relative_path,
                source_range=SourceRange(
                    start_line=getattr(
                        position_node,
                        "lineno",
                        getattr(source_node, "lineno", 0),
                    ),
                    end_line=getattr(
                        position_node,
                        "end_lineno",
                        getattr(source_node, "end_lineno", 0),
                    ),
                ),
                raw_text=raw_text,
                operation_type="predicate",
                object_ref=raw_text[:160],
                attributes={
                    "scope_name": scope_name,
                    "predicate_kind": predicate_kind,
                    "referenced_names": sorted(referenced_names),
                    **self._python_ast_position_attributes(position_node),
                },
            )
        )
        self._add_python_control_edges(
            graph=graph,
            state=state,
            target_node_id=predicate_id,
        )
        for producer_id in sorted(
            self._referenced_producer_ids(referenced_names, state)
        ):
            graph.add_edge(
                UEGEdge(
                    source=producer_id,
                    target=predicate_id,
                    edge_type="DATA_DEP",
                    attributes={"kind": "predicate_dependency"},
                )
            )
        if state.first_node_id is None:
            state.first_node_id = predicate_id
        state.previous_node_id = predicate_id
        return call_counter, predicate_id

    def _set_python_join_state(
        self,
        *,
        state: _PythonScopeState,
        original_producers: dict[str, str],
        branch_states: list[_PythonScopeState],
    ) -> None:
        state.previous_node_id = None
        state.pending_control_edges = [
            exit_edge
            for branch_state in branch_states
            for exit_edge in self._python_control_exits(branch_state)
        ]
        state.variable_producers = self._merge_python_path_producers(
            original=original_producers,
            branches=branch_states,
        )

    def _merge_python_path_producers(
        self,
        *,
        original: dict[str, str],
        branches: list[_PythonScopeState],
    ) -> dict[str, str]:
        if not branches:
            return dict(original)
        merged: dict[str, str] = {}
        variable_names = set(original)
        for branch in branches:
            variable_names.update(branch.variable_producers)
        for variable_name in sorted(variable_names):
            values = {
                branch.variable_producers.get(variable_name)
                for branch in branches
            }
            if len(values) == 1:
                producer_id = next(iter(values))
                if producer_id is not None:
                    merged[variable_name] = producer_id
        return merged

    def _add_python_loop_back_edges(
        self,
        *,
        graph: ActionGraph,
        state: _PythonScopeState,
        loop_target_id: str,
    ) -> None:
        seen: set[str] = set()
        for source_id, _edge_type in self._python_control_exits(state):
            if source_id == loop_target_id or source_id in seen:
                continue
            seen.add(source_id)
            graph.add_edge(
                UEGEdge(
                    source=source_id,
                    target=loop_target_id,
                    edge_type="SEQUENTIAL",
                    attributes={"kind": "loop_back"},
                )
            )

    def _process_python_try_statement(
        self,
        *,
        graph: ActionGraph,
        statement: ast.Try | ast.TryStar,
        relative_path: str,
        state: _PythonScopeState,
        local_functions: dict[str, PythonFunctionDef],
        call_counter: int,
        scope_name: str,
    ) -> int:
        original_producers = dict(state.variable_producers)
        call_counter, try_predicate_id = self._emit_python_predicate_node(
            graph=graph,
            source_node=statement,
            expression=None,
            raw_text="try",
            summary="Enter try block",
            predicate_kind=(
                "try_star_dispatch"
                if isinstance(statement, ast.TryStar)
                else "try_dispatch"
            ),
            relative_path=relative_path,
            state=state,
            call_counter=call_counter,
            scope_name=scope_name,
        )

        normal_state = self._python_branch_state(
            state,
            [(try_predicate_id, "CONDITIONAL_TRUE")],
        )
        call_counter = self._process_python_block(
            graph=graph,
            statements=statement.body,
            relative_path=relative_path,
            state=normal_state,
            local_functions=local_functions,
            call_counter=call_counter,
            scope_name=scope_name,
        )
        normal_completion = normal_state
        if statement.orelse:
            else_state = self._python_branch_state(
                normal_state,
                self._python_control_exits(normal_state),
            )
            call_counter = self._process_python_block(
                graph=graph,
                statements=statement.orelse,
                relative_path=relative_path,
                state=else_state,
                local_functions=local_functions,
                call_counter=call_counter,
                scope_name=scope_name,
            )
            normal_completion = else_state

        handler_states: list[_PythonScopeState] = []
        handler_dispatch_state = self._python_branch_state(
            state,
            [(try_predicate_id, "CONDITIONAL_FALSE")],
        )
        for handler_index, handler in enumerate(statement.handlers, start=1):
            call_counter = self._emit_calls_from_expression(
                graph=graph,
                expression=handler.type,
                relative_path=relative_path,
                state=handler_dispatch_state,
                call_counter=call_counter,
                local_functions=local_functions,
                scope_name=scope_name,
            )
            handler_type = (
                self._safe_ast_unparse(handler.type)
                if handler.type is not None
                else "BaseException"
            )
            handler_text = (
                f"except {handler_type} as {handler.name}"
                if handler.name
                else f"except {handler_type}"
            )
            call_counter, handler_predicate_id = (
                self._emit_python_predicate_node(
                    graph=graph,
                    source_node=handler,
                    expression=handler.type,
                    raw_text=handler_text,
                    summary=f"Match exception handler {handler_type}",
                    predicate_kind=f"except_handler_{handler_index}",
                    relative_path=relative_path,
                    state=handler_dispatch_state,
                    call_counter=call_counter,
                    scope_name=scope_name,
                )
            )
            handler_state = self._python_branch_state(
                handler_dispatch_state,
                [(handler_predicate_id, "CONDITIONAL_TRUE")],
            )
            if handler.name:
                handler_state.variable_producers[handler.name] = (
                    handler_predicate_id
                )
            call_counter = self._process_python_block(
                graph=graph,
                statements=handler.body,
                relative_path=relative_path,
                state=handler_state,
                local_functions=local_functions,
                call_counter=call_counter,
                scope_name=scope_name,
            )
            handler_states.append(handler_state)
            handler_dispatch_state = self._python_branch_state(
                handler_dispatch_state,
                [(handler_predicate_id, "CONDITIONAL_FALSE")],
            )

        completed_states = [normal_completion, *handler_states]
        if statement.finalbody:
            final_input_states = [
                *completed_states,
                handler_dispatch_state,
            ]
            final_state = self._python_branch_state(
                state,
                [
                    exit_edge
                    for completed_state in final_input_states
                    for exit_edge in self._python_control_exits(completed_state)
                ],
            )
            final_state.variable_producers = self._merge_python_path_producers(
                original=original_producers,
                branches=final_input_states,
            )
            call_counter = self._process_python_block(
                graph=graph,
                statements=statement.finalbody,
                relative_path=relative_path,
                state=final_state,
                local_functions=local_functions,
                call_counter=call_counter,
                scope_name=scope_name,
            )
            self._set_python_join_state(
                state=state,
                original_producers=original_producers,
                branch_states=[final_state],
            )
            return call_counter

        self._set_python_join_state(
            state=state,
            original_producers=original_producers,
            branch_states=completed_states,
        )
        return call_counter

    def _process_python_for_statement(
        self,
        *,
        graph: ActionGraph,
        statement: ast.For | ast.AsyncFor,
        relative_path: str,
        state: _PythonScopeState,
        local_functions: dict[str, PythonFunctionDef],
        call_counter: int,
        scope_name: str,
    ) -> int:
        call_counter = self._emit_calls_from_expression(
            graph=graph,
            expression=statement.iter,
            relative_path=relative_path,
            state=state,
            call_counter=call_counter,
            local_functions=local_functions,
            scope_name=scope_name,
        )
        loop_text = (
            f"{self._safe_ast_unparse(statement.target)} in "
            f"{self._safe_ast_unparse(statement.iter)}"
        )
        call_counter, predicate_id = self._emit_python_predicate_node(
            graph=graph,
            source_node=statement,
            expression=statement.iter,
            raw_text=loop_text,
            summary=(
                f"Async for {loop_text[:140]}"
                if isinstance(statement, ast.AsyncFor)
                else f"For {loop_text[:140]}"
            ),
            predicate_kind=(
                "async_for_iteration"
                if isinstance(statement, ast.AsyncFor)
                else "for_iteration"
            ),
            relative_path=relative_path,
            state=state,
            call_counter=call_counter,
            scope_name=scope_name,
        )
        original_producers = dict(state.variable_producers)
        body_state = self._python_branch_state(
            state,
            [(predicate_id, "CONDITIONAL_TRUE")],
        )
        call_counter = self._emit_calls_from_expression(
            graph=graph,
            expression=statement.target,
            relative_path=relative_path,
            state=body_state,
            call_counter=call_counter,
            local_functions=local_functions,
            scope_name=scope_name,
        )
        for variable_name in self._extract_store_names(statement.target):
            body_state.variable_producers[variable_name] = predicate_id
        call_counter = self._process_python_block(
            graph=graph,
            statements=statement.body,
            relative_path=relative_path,
            state=body_state,
            local_functions=local_functions,
            call_counter=call_counter,
            scope_name=scope_name,
        )
        self._add_python_loop_back_edges(
            graph=graph,
            state=body_state,
            loop_target_id=predicate_id,
        )

        exhausted_state = self._python_branch_state(
            state,
            [(predicate_id, "CONDITIONAL_FALSE")],
        )
        call_counter = self._process_python_block(
            graph=graph,
            statements=statement.orelse,
            relative_path=relative_path,
            state=exhausted_state,
            local_functions=local_functions,
            call_counter=call_counter,
            scope_name=scope_name,
        )
        self._set_python_join_state(
            state=state,
            original_producers=original_producers,
            branch_states=[exhausted_state],
        )
        return call_counter

    def _process_python_while_statement(
        self,
        *,
        graph: ActionGraph,
        statement: ast.While,
        relative_path: str,
        state: _PythonScopeState,
        local_functions: dict[str, PythonFunctionDef],
        call_counter: int,
        scope_name: str,
    ) -> int:
        nodes_before_test = len(graph.nodes)
        call_counter = self._emit_calls_from_expression(
            graph=graph,
            expression=statement.test,
            relative_path=relative_path,
            state=state,
            call_counter=call_counter,
            local_functions=local_functions,
            scope_name=scope_name,
        )
        test_entry_id = (
            graph.nodes[nodes_before_test].node_id
            if len(graph.nodes) > nodes_before_test
            else None
        )
        test_text = self._safe_ast_unparse(statement.test)
        call_counter, predicate_id = self._emit_python_predicate_node(
            graph=graph,
            source_node=statement,
            expression=statement.test,
            raw_text=f"while {test_text}",
            summary=f"While {test_text[:140]}",
            predicate_kind="while_iteration",
            relative_path=relative_path,
            state=state,
            call_counter=call_counter,
            scope_name=scope_name,
        )
        original_producers = dict(state.variable_producers)
        body_state = self._python_branch_state(
            state,
            [(predicate_id, "CONDITIONAL_TRUE")],
        )
        call_counter = self._process_python_block(
            graph=graph,
            statements=statement.body,
            relative_path=relative_path,
            state=body_state,
            local_functions=local_functions,
            call_counter=call_counter,
            scope_name=scope_name,
        )
        self._add_python_loop_back_edges(
            graph=graph,
            state=body_state,
            loop_target_id=test_entry_id or predicate_id,
        )

        exhausted_state = self._python_branch_state(
            state,
            [(predicate_id, "CONDITIONAL_FALSE")],
        )
        call_counter = self._process_python_block(
            graph=graph,
            statements=statement.orelse,
            relative_path=relative_path,
            state=exhausted_state,
            local_functions=local_functions,
            call_counter=call_counter,
            scope_name=scope_name,
        )
        self._set_python_join_state(
            state=state,
            original_producers=original_producers,
            branch_states=[exhausted_state],
        )
        return call_counter

    def _process_python_with_statement(
        self,
        *,
        graph: ActionGraph,
        statement: ast.With | ast.AsyncWith,
        relative_path: str,
        state: _PythonScopeState,
        local_functions: dict[str, PythonFunctionDef],
        call_counter: int,
        scope_name: str,
    ) -> int:
        for item in statement.items:
            call_counter = self._emit_calls_from_expression(
                graph=graph,
                expression=item.context_expr,
                relative_path=relative_path,
                state=state,
                call_counter=call_counter,
                local_functions=local_functions,
                scope_name=scope_name,
            )
        context_text = ", ".join(
            self._safe_ast_unparse(item.context_expr)
            for item in statement.items
        )
        context_guard_text = (
            f"async with {context_text}"
            if isinstance(statement, ast.AsyncWith)
            else f"with {context_text}"
        )
        call_counter, predicate_id = self._emit_python_predicate_node(
            graph=graph,
            source_node=statement,
            expression=None,
            raw_text=context_guard_text,
            summary=(
                f"Enter async context {context_text[:130]}"
                if isinstance(statement, ast.AsyncWith)
                else f"Enter context {context_text[:140]}"
            ),
            predicate_kind=(
                "async_with_entry"
                if isinstance(statement, ast.AsyncWith)
                else "with_entry"
            ),
            relative_path=relative_path,
            state=state,
            call_counter=call_counter,
            scope_name=scope_name,
        )
        original_producers = dict(state.variable_producers)
        body_state = self._python_branch_state(
            state,
            [(predicate_id, "CONDITIONAL_TRUE")],
        )
        for item in statement.items:
            if item.optional_vars is None:
                continue
            call_counter = self._emit_calls_from_expression(
                graph=graph,
                expression=item.optional_vars,
                relative_path=relative_path,
                state=body_state,
                call_counter=call_counter,
                local_functions=local_functions,
                scope_name=scope_name,
            )
            for variable_name in self._extract_store_names(item.optional_vars):
                body_state.variable_producers[variable_name] = predicate_id
        call_counter = self._process_python_block(
            graph=graph,
            statements=statement.body,
            relative_path=relative_path,
            state=body_state,
            local_functions=local_functions,
            call_counter=call_counter,
            scope_name=scope_name,
        )
        bypass_state = self._python_branch_state(
            state,
            [(predicate_id, "CONDITIONAL_FALSE")],
        )
        self._set_python_join_state(
            state=state,
            original_producers=original_producers,
            branch_states=[body_state, bypass_state],
        )
        return call_counter

    def _process_python_match_statement(
        self,
        *,
        graph: ActionGraph,
        statement: ast.Match,
        relative_path: str,
        state: _PythonScopeState,
        local_functions: dict[str, PythonFunctionDef],
        call_counter: int,
        scope_name: str,
    ) -> int:
        call_counter = self._emit_calls_from_expression(
            graph=graph,
            expression=statement.subject,
            relative_path=relative_path,
            state=state,
            call_counter=call_counter,
            local_functions=local_functions,
            scope_name=scope_name,
        )
        subject_text = self._safe_ast_unparse(statement.subject)
        call_counter, match_predicate_id = self._emit_python_predicate_node(
            graph=graph,
            source_node=statement,
            expression=statement.subject,
            raw_text=f"match {subject_text}",
            summary=f"Match {subject_text[:140]}",
            predicate_kind="match_dispatch",
            relative_path=relative_path,
            state=state,
            call_counter=call_counter,
            scope_name=scope_name,
        )
        original_producers = dict(state.variable_producers)
        dispatch_state = self._python_branch_state(
            state,
            [(match_predicate_id, "SEQUENTIAL")],
        )
        case_states: list[_PythonScopeState] = []

        for case_index, case in enumerate(statement.cases, start=1):
            call_counter = self._emit_calls_from_expression(
                graph=graph,
                expression=case.guard,
                relative_path=relative_path,
                state=dispatch_state,
                call_counter=call_counter,
                local_functions=local_functions,
                scope_name=scope_name,
            )
            pattern_text = self._safe_ast_unparse(case.pattern)
            guard_text = (
                f" if {self._safe_ast_unparse(case.guard)}"
                if case.guard is not None
                else ""
            )
            case_text = f"case {pattern_text}{guard_text}"
            call_counter, case_predicate_id = self._emit_python_predicate_node(
                graph=graph,
                source_node=case.pattern,
                expression=case.guard or statement.subject,
                raw_text=case_text,
                summary=f"Match {case_text[:140]}",
                predicate_kind=f"match_case_{case_index}",
                relative_path=relative_path,
                state=dispatch_state,
                call_counter=call_counter,
                scope_name=scope_name,
            )
            case_state = self._python_branch_state(
                dispatch_state,
                [(case_predicate_id, "CONDITIONAL_TRUE")],
            )
            for variable_name in self._extract_match_capture_names(case.pattern):
                case_state.variable_producers[variable_name] = case_predicate_id
            call_counter = self._process_python_block(
                graph=graph,
                statements=case.body,
                relative_path=relative_path,
                state=case_state,
                local_functions=local_functions,
                call_counter=call_counter,
                scope_name=scope_name,
            )
            case_states.append(case_state)
            dispatch_state = self._python_branch_state(
                dispatch_state,
                [(case_predicate_id, "CONDITIONAL_FALSE")],
            )

        self._set_python_join_state(
            state=state,
            original_producers=original_producers,
            branch_states=[*case_states, dispatch_state],
        )
        return call_counter

    def _process_python_fallback_statement(
        self,
        *,
        graph: ActionGraph,
        statement: ast.stmt,
        relative_path: str,
        state: _PythonScopeState,
        local_functions: dict[str, PythonFunctionDef],
        call_counter: int,
        scope_name: str,
    ) -> int:
        expression_fields: list[ast.AST] = []
        nested_blocks: list[list[ast.stmt]] = []
        for _field_name, value in ast.iter_fields(statement):
            if isinstance(value, ast.stmt):
                nested_blocks.append([value])
            elif isinstance(value, list):
                statements = [item for item in value if isinstance(item, ast.stmt)]
                if statements:
                    nested_blocks.append(statements)
                expression_fields.extend(
                    item
                    for item in value
                    if isinstance(item, ast.AST)
                    and not isinstance(item, ast.stmt)
                )
            elif isinstance(value, ast.AST):
                expression_fields.append(value)

        for expression in expression_fields:
            call_counter = self._emit_calls_from_expression(
                graph=graph,
                expression=expression,
                relative_path=relative_path,
                state=state,
                call_counter=call_counter,
                local_functions=local_functions,
                scope_name=scope_name,
            )

        if not nested_blocks:
            return call_counter

        statement_name = type(statement).__name__
        call_counter, predicate_id = self._emit_python_predicate_node(
            graph=graph,
            source_node=statement,
            expression=None,
            raw_text=statement_name,
            summary=f"Conservatively enter {statement_name}",
            predicate_kind=f"conservative_{statement_name.lower()}",
            relative_path=relative_path,
            state=state,
            call_counter=call_counter,
            scope_name=scope_name,
        )
        original_producers = dict(state.variable_producers)
        branch_states: list[_PythonScopeState] = []
        for block_index, block in enumerate(nested_blocks):
            edge_type = (
                "CONDITIONAL_TRUE"
                if block_index == 0
                else "CONDITIONAL_FALSE"
            )
            branch_state = self._python_branch_state(
                state,
                [(predicate_id, edge_type)],
            )
            call_counter = self._process_python_block(
                graph=graph,
                statements=block,
                relative_path=relative_path,
                state=branch_state,
                local_functions=local_functions,
                call_counter=call_counter,
                scope_name=scope_name,
            )
            branch_states.append(branch_state)
        branch_states.append(
            self._python_branch_state(
                state,
                [(predicate_id, "CONDITIONAL_FALSE")],
            )
        )
        self._set_python_join_state(
            state=state,
            original_producers=original_producers,
            branch_states=branch_states,
        )
        return call_counter

    def _add_python_control_edges(
        self,
        *,
        graph: ActionGraph,
        state: _PythonScopeState,
        target_node_id: str,
    ) -> None:
        incoming = (
            list(state.pending_control_edges)
            if state.pending_control_edges
            else (
                [(state.previous_node_id, "SEQUENTIAL")]
                if state.previous_node_id is not None
                else []
            )
        )
        seen: set[tuple[str, str]] = set()
        for source_id, edge_type in incoming:
            edge_key = (source_id, edge_type)
            if edge_key in seen:
                continue
            seen.add(edge_key)
            graph.add_edge(
                UEGEdge(
                    source=source_id,
                    target=target_node_id,
                    edge_type=edge_type,
                )
            )
        state.pending_control_edges.clear()
        state.previous_node_id = target_node_id

    def _python_control_exits(
        self,
        state: _PythonScopeState,
    ) -> list[tuple[str, str]]:
        if state.pending_control_edges:
            return list(state.pending_control_edges)
        if state.previous_node_id is not None:
            return [(state.previous_node_id, "SEQUENTIAL")]
        return []

    def _merge_python_branch_producers(
        self,
        *,
        original: dict[str, str],
        true_branch: dict[str, str],
        false_branch: dict[str, str],
    ) -> dict[str, str]:
        merged: dict[str, str] = {}
        for variable_name in sorted(
            set(original) | set(true_branch) | set(false_branch)
        ):
            true_producer = true_branch.get(variable_name)
            false_producer = false_branch.get(variable_name)
            if (
                true_producer is not None
                and true_producer == false_producer
            ):
                merged[variable_name] = true_producer
        return merged

    def _emit_python_definition_expressions(
        self,
        *,
        graph: ActionGraph,
        function_def: PythonFunctionDef,
        relative_path: str,
        state: _PythonScopeState,
        local_functions: dict[str, PythonFunctionDef],
        call_counter: int,
    ) -> int:
        arguments = [
            *function_def.args.posonlyargs,
            *function_def.args.args,
            *function_def.args.kwonlyargs,
        ]
        if function_def.args.vararg is not None:
            arguments.append(function_def.args.vararg)
        if function_def.args.kwarg is not None:
            arguments.append(function_def.args.kwarg)
        expressions: list[ast.AST] = [
            *function_def.decorator_list,
            *function_def.args.defaults,
            *(
                default
                for default in function_def.args.kw_defaults
                if default is not None
            ),
            *(
                argument.annotation
                for argument in arguments
                if argument.annotation is not None
            ),
        ]
        if function_def.returns is not None:
            expressions.append(function_def.returns)
        expressions.extend(getattr(function_def, "type_params", []))
        for expression in expressions:
            call_counter = self._emit_calls_from_expression(
                graph=graph,
                expression=expression,
                relative_path=relative_path,
                state=state,
                call_counter=call_counter,
                local_functions=local_functions,
                scope_name="module",
            )
        return call_counter

    def _emit_parameter_nodes(
        self,
        *,
        graph: ActionGraph,
        function_name: str,
        function_def: PythonFunctionDef,
        relative_path: str,
        state: _PythonScopeState,
        call_counter: int,
    ) -> int:
        arguments = [
            *function_def.args.posonlyargs,
            *function_def.args.args,
            *function_def.args.kwonlyargs,
        ]
        if function_def.args.vararg is not None:
            arguments.append(function_def.args.vararg)
        if function_def.args.kwarg is not None:
            arguments.append(function_def.args.kwarg)
        for argument in arguments:
            call_counter += 1
            node_id = f"{graph.graph_id}:param-{call_counter:03d}"
            graph.add_node(
                UEGNode(
                    node_id=node_id,
                    layer="code",
                    node_type="CODE_ACTION",
                    summary=f"Parameter {function_name}.{argument.arg}",
                    source_file=relative_path,
                    source_range=SourceRange(
                        start_line=getattr(argument, "lineno", getattr(function_def, "lineno", 0)),
                        end_line=getattr(argument, "end_lineno", getattr(function_def, "lineno", 0)),
                    ),
                    operation_type="parameter_input",
                    object_ref=argument.arg,
                    attributes={
                        "scope_name": function_name,
                        "parameter_name": argument.arg,
                        **self._python_ast_position_attributes(argument),
                    },
                )
            )
            self._add_python_control_edges(
                graph=graph,
                state=state,
                target_node_id=node_id,
            )
            if state.first_node_id is None:
                state.first_node_id = node_id
            state.previous_node_id = node_id
            state.variable_producers[argument.arg] = node_id
            state.parameter_nodes[argument.arg] = node_id
        return call_counter

    def _emit_calls_from_expression(
        self,
        *,
        graph: ActionGraph,
        expression: ast.AST | None,
        relative_path: str,
        state: _PythonScopeState,
        call_counter: int,
        local_functions: dict[str, PythonFunctionDef],
        scope_name: str,
    ) -> int:
        if expression is None:
            return call_counter

        call_nodes = self._python_calls_in_evaluation_order(expression)
        for call in call_nodes:
            func_name = self._call_name(call.func)
            call_counter, merge_node_ids = self._emit_argument_merge_nodes(
                graph=graph,
                call=call,
                func_name=func_name,
                relative_path=relative_path,
                state=state,
                call_counter=call_counter,
                scope_name=scope_name,
            )
            call_counter += 1
            operation_type, risk_tags = self._classify_call(func_name, call)
            referenced_names = sorted(self._extract_load_names(call))
            node_id = f"{graph.graph_id}:call-{call_counter:03d}"
            local_callee = func_name if func_name in local_functions else None
            graph.add_node(
                UEGNode(
                    node_id=node_id,
                    layer="code",
                    node_type="CODE_ACTION",
                    summary=f"Call {func_name}",
                    source_file=relative_path,
                    source_range=SourceRange(start_line=getattr(call, "lineno", 0), end_line=getattr(call, "end_lineno", getattr(call, "lineno", 0))),
                    raw_text=ast.unparse(call),
                    operation_type=operation_type,
                    object_ref=self._python_call_object_ref(
                        func_name,
                        call,
                        operation_type,
                    ),
                    risk_tags=risk_tags,
                    attributes={
                        "call_name": func_name,
                        "referenced_names": referenced_names,
                        "argument_referenced_names": [sorted(self._extract_load_names(argument)) for argument in call.args],
                        "scope_name": scope_name,
                        "local_callee": local_callee,
                        **self._python_ast_position_attributes(call),
                    },
                )
            )
            self._add_python_control_edges(
                graph=graph,
                state=state,
                target_node_id=node_id,
            )
            for name in referenced_names:
                producer_id = state.variable_producers.get(name)
                if producer_id is not None:
                    graph.add_edge(UEGEdge(source=producer_id, target=node_id, edge_type="DATA_DEP", attributes={"variable": name}))
            for merge_node_id in merge_node_ids:
                graph.add_edge(UEGEdge(source=merge_node_id, target=node_id, edge_type="DATA_DEP", attributes={"kind": "argument_merge"}))
            if state.first_node_id is None:
                state.first_node_id = node_id
            state.previous_node_id = node_id
        return call_counter

    @staticmethod
    def _python_calls_in_evaluation_order(expression: ast.AST) -> list[ast.Call]:
        """Return nested calls before the calls that consume their results."""

        ordered_calls: list[ast.Call] = []

        class EvaluationOrderVisitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:
                self.visit(node.func)
                for argument in node.args:
                    self.visit(argument)
                for keyword in node.keywords:
                    self.visit(keyword.value)
                ordered_calls.append(node)

        EvaluationOrderVisitor().visit(expression)
        return ordered_calls

    def _emit_argument_merge_nodes(
        self,
        *,
        graph: ActionGraph,
        call: ast.Call,
        func_name: str,
        relative_path: str,
        state: _PythonScopeState,
        call_counter: int,
        scope_name: str,
    ) -> tuple[int, list[str]]:
        merge_node_ids: list[str] = []
        argument_expressions = list(call.args) + [keyword.value for keyword in call.keywords]
        for expression in argument_expressions:
            referenced_names = self._extract_load_names(expression)
            producer_ids = self._referenced_producer_ids(referenced_names, state)
            if not self._should_materialize_merge_node(expression, producer_ids):
                continue
            call_counter, merge_node_id = self._emit_merge_node(
                graph=graph,
                expression=expression,
                relative_path=relative_path,
                state=state,
                call_counter=call_counter,
                producer_ids=producer_ids,
                scope_name=scope_name,
                summary=f"Merge arguments for {func_name}",
                attributes={"referenced_names": sorted(referenced_names), "kind": "call_argument_merge"},
            )
            merge_node_ids.append(merge_node_id)
        return call_counter, merge_node_ids

    def _emit_merge_node(
        self,
        *,
        graph: ActionGraph,
        expression: ast.AST | None,
        relative_path: str,
        state: _PythonScopeState,
        call_counter: int,
        producer_ids: set[str],
        scope_name: str,
        summary: str,
        attributes: dict[str, object],
    ) -> tuple[int, str]:
        call_counter += 1
        node_id = f"{graph.graph_id}:merge-{call_counter:03d}"
        graph.add_node(
            UEGNode(
                node_id=node_id,
                layer="code",
                node_type="CODE_ACTION",
                summary=summary,
                source_file=relative_path,
                source_range=SourceRange(
                    start_line=getattr(expression, "lineno", 0),
                    end_line=getattr(expression, "end_lineno", getattr(expression, "lineno", 0)),
                )
                if expression is not None
                else None,
                raw_text=ast.unparse(expression) if expression is not None else None,
                operation_type="data_merge",
                object_ref=ast.unparse(expression)[:160] if expression is not None else None,
                attributes={
                    "scope_name": scope_name,
                    **attributes,
                    **self._python_ast_position_attributes(expression),
                },
            )
        )
        self._add_python_control_edges(
            graph=graph,
            state=state,
            target_node_id=node_id,
        )
        for producer_id in sorted(producer_ids):
            graph.add_edge(UEGEdge(source=producer_id, target=node_id, edge_type="DATA_DEP", attributes={"kind": "merge_input"}))
        if state.first_node_id is None:
            state.first_node_id = node_id
        state.previous_node_id = node_id
        return call_counter, node_id

    def _extract_store_names(self, target: ast.AST) -> set[str]:
        names: set[str] = set()
        for node in ast.walk(target):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
        return names

    def _extract_match_capture_names(self, pattern: ast.pattern) -> set[str]:
        names: set[str] = set()
        for node in ast.walk(pattern):
            if isinstance(node, ast.MatchAs) and node.name:
                names.add(node.name)
            elif isinstance(node, ast.MatchStar) and node.name:
                names.add(node.name)
            elif isinstance(node, ast.MatchMapping) and node.rest:
                names.add(node.rest)
        return names

    def _extract_load_names(self, expression: ast.AST | None) -> set[str]:
        if expression is None:
            return set()
        names: set[str] = set()
        for node in ast.walk(expression):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                names.add(node.id)
        return names

    def _referenced_producer_ids(self, referenced_names: set[str], state: _PythonScopeState) -> set[str]:
        producer_ids: set[str] = set()
        for referenced_name in referenced_names:
            producer_id = state.variable_producers.get(referenced_name)
            if producer_id is not None:
                producer_ids.add(producer_id)
        return producer_ids

    def _should_materialize_merge_node(self, expression: ast.AST | None, producer_ids: set[str]) -> bool:
        if expression is None or not producer_ids:
            return False
        if isinstance(expression, ast.Call):
            return False
        if len(producer_ids) > 1:
            return True
        return isinstance(expression, (ast.Dict, ast.List, ast.Tuple, ast.Set, ast.JoinedStr, ast.BinOp, ast.BoolOp))

    def _add_local_call_edges(self, graph: ActionGraph) -> None:
        function_boundaries = graph.metadata.get("function_boundaries", {})
        if not function_boundaries:
            return

        sequential_successors = {
            edge.source: edge.target
            for edge in graph.edges
            if edge.edge_type == "SEQUENTIAL"
        }

        for node in graph.nodes:
            local_callee = node.attributes.get("local_callee")
            if not local_callee:
                continue
            boundary = function_boundaries.get(local_callee)
            if not boundary:
                continue
            graph.add_edge(UEGEdge(source=node.node_id, target=boundary["first_node_id"], edge_type="CALLS_LOCAL"))
            parameter_nodes = list((boundary.get("parameter_nodes") or {}).values())
            argument_groups = node.attributes.get("argument_referenced_names") or []
            for index, parameter_node_id in enumerate(parameter_nodes):
                if index >= len(argument_groups):
                    break
                for referenced_name in argument_groups[index]:
                    for producer_id in (
                        edge.source
                        for edge in graph.edges
                        if edge.target == node.node_id and edge.edge_type == "DATA_DEP" and edge.attributes.get("variable") == referenced_name
                    ):
                        graph.add_edge(
                            UEGEdge(
                                source=producer_id,
                                target=parameter_node_id,
                                edge_type="DATA_DEP",
                                attributes={"kind": "parameter_binding", "variable": referenced_name},
                            )
                        )
            successor_id = sequential_successors.get(node.node_id)
            if successor_id:
                for last_node_id in boundary.get(
                    "last_node_ids",
                    [boundary["last_node_id"]],
                ):
                    graph.add_edge(
                        UEGEdge(
                            source=last_node_id,
                            target=successor_id,
                            edge_type="RETURNS_LOCAL",
                        )
                    )

    def _classify_shell_line(self, line: str) -> tuple[str, list[str]]:
        lowered = line.lower()
        if any(word in lowered for word in ("curl ", "wget ", "http://", "https://", "telegram")):
            return "network_send", ["network"]
        try:
            words = shlex.split(line, posix=True)
        except ValueError:
            words = line.split()
        wrappers = {"env", "sudo", "command", "builtin", "nohup"}
        while words:
            if re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*=.*",
                words[0],
            ):
                words.pop(0)
                continue
            if Path(words[0]).name.casefold() in wrappers:
                words.pop(0)
                while words and words[0].startswith("-"):
                    words.pop(0)
                continue
            break
        command = Path(words[0]).name.casefold() if words else ""
        arguments = [word.casefold() for word in words[1:]]
        persistent_commands = {
            "rm",
            "mv",
            "cp",
            "mkdir",
            "rmdir",
            "touch",
            "chmod",
            "chown",
            "chgrp",
            "ln",
            "install",
            "truncate",
            "tee",
            "dd",
            "crontab",
            "systemctl",
            "launchctl",
            "defaults",
        }
        if command in persistent_commands or re.search(
            r"(?:^|[^<])>>?",
            line,
        ):
            return "file_write", ["persistent_state", "file_write"]
        if command == "git" and tuple(arguments[:1]) in {
            ("add",),
            ("apply",),
            ("am",),
            ("branch",),
            ("checkout",),
            ("cherry-pick",),
            ("clean",),
            ("commit",),
            ("merge",),
            ("mv",),
            ("pull",),
            ("push",),
            ("rebase",),
            ("reset",),
            ("restore",),
            ("revert",),
            ("rm",),
            ("switch",),
            ("tag",),
        }:
            return "file_write", ["persistent_state", "file_write"]
        if command in {
            "bash",
            "sh",
            "zsh",
            "fish",
            "python",
            "python3",
            "node",
            "deno",
            "osascript",
            "open",
            "eval",
            "exec",
            "git",
            "aws",
            "gcloud",
            "az",
            "kubectl",
            "docker",
            "podman",
        }:
            return "exec_command", ["command_execution"]
        if command in {"cat", "head", "tail", "less", "more", "grep"} or any(
            word in lowered
            for word in ("history", ".ssh", ".aws", "cookie", "token")
        ):
            return "file_access", ["file_access", "sensitive_collection"]
        if command in {"", ":", "true", "false"}:
            return "shell_action", []
        # Every remaining top-level shell segment invokes an executable or a
        # stateful shell primitive.  Preserve its concrete command identity,
        # but normalize it to the command-execution category so it
        # cannot silently fall outside the formal privilege taxonomy.
        return "exec_command", ["command_execution"]

    def _python_call_object_ref(
        self,
        func_name: str,
        call: ast.Call,
        operation_type: str,
    ) -> str:
        keyword_values = {
            keyword.arg: self._safe_ast_unparse(keyword.value)
            for keyword in call.keywords
            if keyword.arg
        }
        positional = [self._safe_ast_unparse(argument) for argument in call.args]
        if operation_type == "network_send":
            for keyword in ("json", "data", "files", "body", "payload", "content"):
                if keyword_values.get(keyword):
                    return keyword_values[keyword][:160]
            if len(positional) > 1:
                return positional[1][:160]
            if positional:
                return positional[0][:160]
        if operation_type in {
            "exec_command",
            "file_access",
            "file_write",
            "delete",
            "read_env",
            "collect_identifier",
        } and positional:
            return positional[0][:160]
        return func_name[:160]

    def _safe_ast_unparse(self, node: ast.AST) -> str:
        try:
            return ast.unparse(node)
        except (AttributeError, ValueError):
            return type(node).__name__

    def _python_ast_position_attributes(
        self,
        node: ast.AST | None,
    ) -> dict[str, int]:
        if node is None:
            return {}
        attributes: dict[str, int] = {}
        col_offset = getattr(node, "col_offset", None)
        end_col_offset = getattr(node, "end_col_offset", None)
        if isinstance(col_offset, int):
            attributes["col_offset"] = col_offset
        if isinstance(end_col_offset, int):
            attributes["end_col_offset"] = end_col_offset
        return attributes

    def _strip_javascript_comments(self, line: str) -> str:
        quote: str | None = None
        escaped = False
        index = 0
        while index < len(line):
            character = line[index]
            if escaped:
                escaped = False
                index += 1
                continue
            if character == "\\" and quote is not None:
                escaped = True
                index += 1
                continue
            if character in {"'", '"', "`"}:
                if quote is None:
                    quote = character
                elif quote == character:
                    quote = None
                index += 1
                continue
            if quote is None and line[index : index + 2] == "//":
                return line[:index]
            index += 1
        return line

    def _javascript_call_arguments(self, line: str, opening_index: int) -> str:
        arguments, _end_index = self._javascript_call_extent(
            line,
            opening_index,
        )
        return arguments

    def _javascript_call_extent(
        self,
        line: str,
        opening_index: int,
    ) -> tuple[str, int]:
        depth = 0
        quote: str | None = None
        escaped = False
        for index in range(opening_index, len(line)):
            character = line[index]
            if escaped:
                escaped = False
                continue
            if character == "\\" and quote is not None:
                escaped = True
                continue
            if character in {"'", '"', "`"}:
                if quote is None:
                    quote = character
                elif quote == character:
                    quote = None
                continue
            if quote is not None:
                continue
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    return (
                        line[opening_index + 1 : index].strip(),
                        index + 1,
                    )
        return line[opening_index + 1 :].strip(), len(line)

    def _split_javascript_arguments(self, arguments: str) -> list[str]:
        parts: list[str] = []
        start = 0
        depth = 0
        quote: str | None = None
        escaped = False
        for index, character in enumerate(arguments):
            if escaped:
                escaped = False
                continue
            if character == "\\" and quote is not None:
                escaped = True
                continue
            if character in {"'", '"', "`"}:
                if quote is None:
                    quote = character
                elif quote == character:
                    quote = None
                continue
            if quote is not None:
                continue
            if character in "([{":
                depth += 1
            elif character in ")]}":
                depth = max(0, depth - 1)
            elif character == "," and depth == 0:
                parts.append(arguments[start:index].strip())
                start = index + 1
        parts.append(arguments[start:].strip())
        return [part for part in parts if part]

    def _classify_javascript_action(
        self,
        callee: str,
        line: str,
    ) -> tuple[str, list[str]]:
        lowered_callee = callee.lower()
        lowered_line = line.lower()
        if any(
            token in lowered_callee
            for token in (
                "fetch",
                "axios.",
                "http.request",
                "https.request",
                "xmlhttprequest",
                "websocket",
                "telegram",
                "webhook",
            )
        ) or (
            lowered_callee.rsplit(".", 1)[-1] in {"send", "post", "put", "upload"}
            and "http" in lowered_line
        ):
            return "network_send", ["network"]
        if any(
            token in lowered_callee
            for token in (
                "child_process",
                "execsync",
                "execfile",
                ".exec",
                ".spawn",
                "deno.command",
                "bun.$",
            )
        ):
            return "exec_command", ["command_execution"]
        if "process.env" in lowered_line or "deno.env" in lowered_line:
            return "read_env", ["sensitive_collection"]
        if any(
            token in lowered_callee
            for token in (
                "writefile",
                "appendfile",
                "writetextfile",
                "writeall",
                "createwritestream",
                ".truncate",
                ".chmod",
                ".chown",
                ".mkdir",
                ".rename",
                ".copyfile",
                ".symlink",
                ".link",
            )
        ):
            return "file_write", ["persistent_state", "file_write"]
        if any(
            token in lowered_callee
            for token in (
                ".unlink",
                ".rmdir",
                ".remove",
                "fs.rm",
                "fs.promises.rm",
                "deno.remove",
            )
        ):
            return "delete", ["persistent_state", "deletion"]
        if any(
            token in lowered_callee
            for token in (
                "fs.",
                "readfile",
                "readtextfile",
            )
        ):
            risk_tags = ["file_access"]
            if any(
                token in lowered_line
                for token in ("history", ".ssh", ".aws", "token", "cookie", "credential")
            ):
                risk_tags.append("sensitive_collection")
            return "file_access", risk_tags
        return "call", []

    def _classify_javascript_statement(
        self,
        line: str,
    ) -> tuple[str | None, list[str]]:
        lowered = line.lower()
        if "process.env" in lowered or "deno.env" in lowered:
            return "read_env", ["sensitive_collection"]
        if any(
            token in lowered
            for token in ("history", ".ssh", ".aws", "cookie", "credential", "token")
        ):
            return "file_access", ["file_access", "sensitive_collection"]
        if JS_ASSIGN_RE.search(line):
            return "data_assignment", []
        if re.match(r"^return\b", line):
            return "return_value", []
        return None, []

    def _javascript_object_ref(
        self,
        callee: str,
        arguments: str,
        operation_type: str,
    ) -> str:
        parts = self._split_javascript_arguments(arguments)
        if operation_type == "network_send":
            if len(parts) > 1:
                return parts[1][:160]
            if parts:
                return parts[0][:160]
        if operation_type in {
            "exec_command",
            "file_access",
            "file_write",
            "delete",
            "read_env",
        } and parts:
            return parts[0][:160]
        return callee[:160]

    def _javascript_statement_object_ref(
        self,
        line: str,
        operation_type: str,
    ) -> str:
        if operation_type == "read_env":
            match = re.search(
                r"(?:process|Deno)\.env(?:\.get)?(?:\[['\"]([^'\"]+)['\"]\]|\.([A-Za-z_$][\w$]*)|\(\s*['\"]([^'\"]+)['\"])",
                line,
                flags=re.IGNORECASE,
            )
            if match:
                return next(
                    (group for group in match.groups() if group),
                    "environment",
                )
            return "environment"
        assignment_match = JS_ASSIGN_RE.search(line)
        if assignment_match:
            return assignment_match.group(1)
        if operation_type == "return_value":
            return re.sub(r"^return\s+", "", line).rstrip(";")[:160]
        return line[:160]

    def _javascript_statement_summary(
        self,
        line: str,
        operation_type: str,
    ) -> str:
        labels = {
            "read_env": "Read environment value",
            "data_assignment": "Assign JavaScript/TypeScript data",
            "return_value": "Return JavaScript/TypeScript value",
            "file_access": "Access potentially sensitive file data",
            "delete": "Delete JavaScript/TypeScript object",
        }
        return labels.get(operation_type, f"Execute statement: {line[:100]}")

    def _add_javascript_data_edges(
        self,
        graph: ActionGraph,
        node_id: str,
        line: str,
        variable_producers: dict[str, str],
    ) -> None:
        assignment_match = JS_ASSIGN_RE.search(line)
        assignment_target = assignment_match.group(1) if assignment_match else None
        for identifier in sorted(set(JS_IDENTIFIER_RE.findall(line))):
            if identifier == assignment_target:
                continue
            producer_id = variable_producers.get(identifier)
            if producer_id is None or producer_id == node_id:
                continue
            graph.add_edge(
                UEGEdge(
                    source=producer_id,
                    target=node_id,
                    edge_type="DATA_DEP",
                    attributes={"variable": identifier},
                )
            )

    def _shell_object_ref(self, line: str) -> str:
        url_match = re.search(r"https?://[^\s'\";]+", line)
        if url_match:
            return url_match.group(0)[:160]
        sensitive_match = re.search(
            r"(?:~?/)?(?:\.[A-Za-z0-9_.-]+|[A-Za-z0-9_.-]+)/(?:[^\s'\";]+)",
            line,
        )
        if sensitive_match:
            return sensitive_match.group(0)[:160]
        try:
            parts = shlex.split(line)
        except ValueError:
            parts = line.split()
        if len(parts) > 1:
            return parts[-1][:160]
        return parts[0][:160] if parts else line[:160]

    def _finalize_graph_metadata(self, graph: ActionGraph) -> None:
        parser_metadata = graph.metadata.get("parser_metadata")
        if not isinstance(parser_metadata, dict):
            parser_metadata = {
                "language": "unknown",
                "parser": "unknown",
                "parser_kind": "unknown",
                "status": "unknown",
            }
            graph.metadata["parser_metadata"] = parser_metadata

        graph.metadata["edge_semantics"] = {
            "control": [
                "SEQUENTIAL",
                "CONDITIONAL_TRUE",
                "CONDITIONAL_FALSE",
                "SEMANTIC_DEP",
            ],
            "data": ["DATA_DEP"],
            "call": ["CALLS", "CALLS_LOCAL"],
            "return": ["RETURNS_TO", "RETURNS_LOCAL"],
        }
        for node in graph.nodes:
            if node.object_ref is None:
                node.object_ref = self._fallback_code_object_ref(node)
            provenance = node.attributes.get("provenance")
            if not isinstance(provenance, dict):
                source_range = (
                    {
                        "start_line": node.source_range.start_line,
                        "end_line": node.source_range.end_line,
                    }
                    if node.source_range is not None
                    else None
                )
                node.attributes["provenance"] = {
                    "origin": (
                        "synthetic_code_boundary"
                        if node.node_type in {"CODE_ENTRY", "CODE_RETURN"}
                        else "code"
                    ),
                    "source_file": node.source_file,
                    "source_range": source_range,
                    "parser": parser_metadata.get("parser"),
                    "parser_kind": parser_metadata.get("parser_kind"),
                    "parser_status": parser_metadata.get("status"),
                }
            node.attributes.setdefault(
                "parser_metadata",
                {
                    "language": parser_metadata.get("language"),
                    "parser": parser_metadata.get("parser"),
                    "status": parser_metadata.get("status"),
                },
            )

        semantic_family_by_type = {
            "SEQUENTIAL": "control",
            "CONDITIONAL_TRUE": "control",
            "CONDITIONAL_FALSE": "control",
            "SEMANTIC_DEP": "control",
            "DATA_DEP": "data",
            "CALLS": "call",
            "CALLS_LOCAL": "call",
            "RETURNS_TO": "return",
            "RETURNS_LOCAL": "return",
        }
        for edge in graph.edges:
            semantic_family = semantic_family_by_type.get(edge.edge_type)
            if semantic_family is not None:
                edge.attributes.setdefault("semantic_family", semantic_family)

    def _fallback_code_object_ref(self, node: UEGNode) -> str | None:
        for key in (
            "parameter_name",
            "call_name",
            "variable",
            "kind",
        ):
            value = node.attributes.get(key)
            if value:
                return str(value)[:160]
        if node.raw_text:
            return " ".join(node.raw_text.split())[:160]
        if node.source_file:
            return node.source_file
        return None
