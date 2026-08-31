from __future__ import annotations

import ast
import re
import shutil
import tempfile
from pathlib import Path

from skillscope.common.models import AblationPlan, CandidateAction, UnifiedExecutionGraph


class CandidateAblation:
    ALLOWED_EDGE_TYPES = {
        "SEQUENTIAL",
        "CONDITIONAL_TRUE",
        "CONDITIONAL_FALSE",
        "SEMANTIC_DEP",
        "CALLS",
        "CALLS_LOCAL",
        "RETURNS_TO",
        "RETURNS_LOCAL",
    }

    def build_replay_variant(self, candidate: CandidateAction, ueg: UnifiedExecutionGraph) -> AblationPlan:
        bypass_successors = ueg.successor_ids(candidate.node_id, self.ALLOWED_EDGE_TYPES)
        strategy = "delete_instruction_from_bundle" if candidate.layer == "instruction" else "delete_or_neutralize_code_in_bundle"
        node = ueg.node_by_id(candidate.node_id)
        notes = [
            "Replay run removes or disables the current candidate directly in a copied skill bundle before re-executing the full skill.",
        ]
        if bypass_successors:
            notes.append("The replayed bundle preserves downstream flow after the target action is removed.")
        return AblationPlan(
            candidate_id=candidate.candidate_id,
            node_id=candidate.node_id,
            layer=candidate.layer,
            strategy=strategy,
            source_file=node.source_file if node is not None else candidate.source_file,
            source_start_line=node.source_range.start_line if node is not None and node.source_range is not None else None,
            source_end_line=node.source_range.end_line if node is not None and node.source_range is not None else None,
            source_start_column=(
                (
                    node.source_range.start_column
                    if node.source_range is not None
                    and node.source_range.start_column is not None
                    else self._optional_int(node.attributes.get("col_offset"))
                )
                if node is not None
                else None
            ),
            source_end_column=(
                (
                    node.source_range.end_column
                    if node.source_range is not None
                    and node.source_range.end_column is not None
                    else self._optional_int(
                        node.attributes.get("end_col_offset")
                    )
                )
                if node is not None
                else None
            ),
            operation_type=node.operation_type if node is not None else None,
            raw_text=node.raw_text if node is not None else None,
            bypass_successor_ids=bypass_successors,
            notes=notes,
        )

    def _optional_int(self, value: object) -> int | None:
        return value if isinstance(value, int) else None

    def materialize_replay_bundle(self, bundle_root: Path, ablation: AblationPlan) -> Path:
        temp_root = Path(tempfile.mkdtemp(prefix="skillscope-replay-bundle-"))
        replay_root = temp_root / "skill"
        shutil.copytree(bundle_root, replay_root, dirs_exist_ok=True)
        self._apply_ablation(replay_root, ablation)
        return replay_root

    def _apply_ablation(self, replay_root: Path, ablation: AblationPlan) -> None:
        if not ablation.source_file:
            raise ValueError(f"Ablation {ablation.candidate_id} does not have a source file.")
        target_path = replay_root / ablation.source_file
        if not target_path.exists():
            raise FileNotFoundError(f"Ablation target does not exist: {target_path}")

        original_text = target_path.read_text(encoding="utf-8")
        lines = original_text.splitlines(keepends=True)

        if ablation.layer == "instruction":
            updated_text = self._remove_instruction_text(lines, ablation)
        else:
            updated_text = self._neutralize_code_text(lines, ablation)

        if updated_text == original_text:
            raise ValueError(
                "Candidate ablation did not change the copied bundle at "
                f"{ablation.source_file!r}."
            )
        target_path.write_text(updated_text, encoding="utf-8")

    def _remove_instruction_text(self, lines: list[str], ablation: AblationPlan) -> str:
        start, end = self._line_slice(lines, ablation)
        if start is not None and end is not None:
            raw_text = (ablation.raw_text or "").strip()
            selected_text = "".join(lines[start:end])
            if raw_text and raw_text in selected_text:
                if self._instruction_span_is_only_action(
                    selected_text=selected_text,
                    raw_text=raw_text,
                ):
                    del lines[start:end]
                else:
                    marker = (
                        f"<!-- SkillScope replay ablation for "
                        f"{ablation.candidate_id} -->"
                    )
                    lines[start:end] = [
                        selected_text.replace(raw_text, marker, 1)
                    ]
                return "".join(lines)
            del lines[start:end]
            return "".join(lines)
        if ablation.raw_text:
            updated_lines = [line for line in lines if ablation.raw_text.strip() not in line.strip()]
            return "".join(updated_lines)
        return "".join(lines)

    def _instruction_span_is_only_action(
        self,
        *,
        selected_text: str,
        raw_text: str,
    ) -> bool:
        stripped = selected_text.strip()
        markdown_body = re.sub(
            r"^(?:[-*+]|\d+[.)])\s+",
            "",
            stripped,
            count=1,
        ).strip()
        return stripped == raw_text or markdown_body == raw_text

    def _neutralize_code_text(self, lines: list[str], ablation: AblationPlan) -> str:
        suffix = Path(ablation.source_file or "").suffix.lower()
        if suffix == ".py":
            ast_rewrite = self._neutralize_python_code("".join(lines), ablation)
            if ast_rewrite is not None:
                return ast_rewrite
        if suffix in {".js", ".mjs", ".cjs", ".ts", ".mts", ".cts"}:
            lexical_rewrite = self._neutralize_javascript_expression(
                lines,
                ablation,
            )
            if lexical_rewrite is not None:
                return lexical_rewrite
        if suffix == ".sh":
            segment_rewrite = self._neutralize_shell_segment(
                lines,
                ablation,
            )
            if segment_rewrite is not None:
                return segment_rewrite
        start, end = self._line_slice(lines, ablation)
        if start is None or end is None:
            return "".join(lines)

        target_lines = lines[start:end]
        indent = self._indentation_for(target_lines)
        if suffix == ".sh":
            replacement = self._neutralize_shell_statement(
                target_lines,
                indent=indent,
                candidate_id=ablation.candidate_id,
            )
        elif suffix in {".js", ".mjs", ".cjs", ".ts", ".mts", ".cts"}:
            replacement = self._neutralize_javascript_statement(
                target_lines,
                indent=indent,
                candidate_id=ablation.candidate_id,
            )
        elif suffix == ".py":
            replacement = (
                f"{indent}pass  # SkillScope replay ablation for "
                f"{ablation.candidate_id}\n"
            )
        else:
            raise ValueError(
                "No syntax-preserving ablation strategy exists for code file "
                f"{ablation.source_file!r}."
            )
        lines[start:end] = [replacement]
        return "".join(lines)

    def _neutralize_javascript_expression(
        self,
        lines: list[str],
        ablation: AblationPlan,
    ) -> str | None:
        start, end = self._line_slice(lines, ablation)
        if start is None or end is None:
            return None
        raw_text = (ablation.raw_text or "").strip()
        if end - start == 1:
            line = lines[start]
            start_column = ablation.source_start_column
            end_column = ablation.source_end_column
            if (
                isinstance(start_column, int)
                and isinstance(end_column, int)
                and 0 <= start_column < end_column <= len(line.rstrip("\r\n"))
            ):
                selected = line[start_column:end_column]
                if not raw_text or selected.strip() == raw_text:
                    replacement = self._javascript_expression_replacement(
                        raw_text or selected.strip()
                    )
                    lines[start] = (
                        line[:start_column]
                        + replacement
                        + line[end_column:]
                    )
                    return "".join(lines)

        selected_text = "".join(lines[start:end])
        if raw_text and selected_text.count(raw_text) == 1:
            replacement = self._javascript_expression_replacement(raw_text)
            lines[start:end] = [
                selected_text.replace(raw_text, replacement, 1)
            ]
            return "".join(lines)
        if raw_text:
            raise ValueError(
                "Cannot safely locate the grounded JavaScript/TypeScript "
                f"candidate expression {raw_text!r} without risking sibling "
                "or nested argument actions."
            )
        return None

    def _javascript_expression_replacement(self, raw_text: str) -> str:
        """Neutralize a call while retaining independently evaluated arguments.

        Replacing ``outer(inner())`` with ``undefined`` would also erase
        ``inner()``. A comma expression preserves JavaScript's left-to-right
        argument evaluation while discarding only the outer call. Constructs
        whose argument execution is conditional or whose spread semantics
        cannot be preserved this way fail closed.
        """

        if not raw_text:
            return "undefined"
        parsed = self._javascript_outer_call(raw_text)
        if parsed is None:
            if "(" in raw_text or ")" in raw_text:
                raise ValueError(
                    "Cannot safely parse the grounded JavaScript/TypeScript "
                    f"candidate call {raw_text!r}; refusing a line-level "
                    "ablation that could erase nested actions."
                )
            return "undefined"
        callee, arguments = parsed
        if "?." in callee:
            raise ValueError(
                "Cannot preserve optional-call argument evaluation without "
                "changing whether nested actions execute."
            )
        if re.fullmatch(
            r"[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*",
            callee,
        ) is None:
            raise ValueError(
                "Cannot safely neutralize a JavaScript/TypeScript call with "
                f"the complex callee {callee!r}."
            )

        preserved_arguments = arguments.rstrip()
        if "/*" in preserved_arguments or "*/" in preserved_arguments:
            raise ValueError(
                "Cannot safely preserve JavaScript/TypeScript call arguments "
                "that contain block comments."
            )
        if preserved_arguments.endswith(","):
            preserved_arguments = preserved_arguments[:-1].rstrip()
        if not preserved_arguments:
            return "undefined"
        if self._javascript_has_unquoted_spread(preserved_arguments):
            raise ValueError(
                "Cannot preserve JavaScript/TypeScript spread "
                "argument semantics while neutralizing the outer call."
            )
        return f"({preserved_arguments}, undefined)"

    def _javascript_outer_call(self, raw_text: str) -> tuple[str, str] | None:
        """Return the callee and arguments of one complete grounded call."""

        if not raw_text.endswith(")"):
            return None
        stack: list[int] = []
        final_opening: int | None = None
        quote: str | None = None
        escaped = False
        for index, character in enumerate(raw_text):
            if quote is not None:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
                continue
            if character in {'"', "'", "`"}:
                quote = character
                continue
            if character == "(":
                stack.append(index)
                continue
            if character != ")" or not stack:
                continue
            opening = stack.pop()
            if index == len(raw_text) - 1:
                final_opening = opening
        if quote is not None or stack or final_opening is None:
            return None
        callee = raw_text[:final_opening].strip()
        if not callee:
            return None
        return callee, raw_text[final_opening + 1 : -1]

    def _javascript_has_unquoted_spread(self, arguments: str) -> bool:
        """Conservatively reject unquoted spread syntax at any nesting depth."""

        quote: str | None = None
        escaped = False
        index = 0
        while index < len(arguments):
            character = arguments[index]
            if quote is not None:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
                index += 1
                continue
            if character in {'"', "'", "`"}:
                quote = character
                index += 1
                continue
            if arguments.startswith("...", index):
                return True
            index += 1
        return False

    def _neutralize_shell_segment(
        self,
        lines: list[str],
        ablation: AblationPlan,
    ) -> str | None:
        start, end = self._line_slice(lines, ablation)
        if start is None or end is None or end - start != 1:
            return None
        start_column = ablation.source_start_column
        end_column = ablation.source_end_column
        if not isinstance(start_column, int) or not isinstance(end_column, int):
            return None

        line = lines[start]
        line_body = line.rstrip("\r\n")
        line_ending = line[len(line_body) :]
        if not 0 <= start_column < end_column <= len(line_body):
            return None
        selected = line_body[start_column:end_column]
        raw_text = (ablation.raw_text or "").strip()
        if raw_text and selected.strip() != raw_text:
            return None

        following = line_body[end_column:].lstrip()
        replacement = self._shell_segment_replacement(
            selected,
            force_failure=following.startswith("||"),
        )
        lines[start] = (
            line_body[:start_column]
            + replacement
            + line_body[end_column:]
            + line_ending
        )
        return "".join(lines)

    def _shell_segment_replacement(
        self,
        segment: str,
        *,
        force_failure: bool,
    ) -> str:
        if force_failure:
            # Bypass an ablated left-hand side into its `||` successor.
            return "false"
        assignment = re.fullmatch(
            r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=.*",
            segment,
            flags=re.DOTALL,
        )
        if assignment is not None:
            return f"{assignment.group(1)}=''"
        return ":"

    def _neutralize_shell_statement(
        self,
        target_lines: list[str],
        *,
        indent: str,
        candidate_id: str,
    ) -> str:
        statement = "".join(target_lines).strip()
        assignment = re.match(
            r"^([A-Za-z_][A-Za-z0-9_]*)\s*=.*$",
            statement,
            flags=re.DOTALL,
        )
        if assignment is not None:
            return (
                f"{indent}{assignment.group(1)}='' "
                f"# SkillScope replay ablation for {candidate_id}\n"
            )
        return (
            f"{indent}: # SkillScope replay ablation for {candidate_id}\n"
        )

    def _neutralize_javascript_statement(
        self,
        target_lines: list[str],
        *,
        indent: str,
        candidate_id: str,
    ) -> str:
        statement = "".join(target_lines).strip()
        assignment = re.match(
            (
                r"^(const|let|var)\s+"
                r"([A-Za-z_$][A-Za-z0-9_$]*)\s*=.*$"
            ),
            statement,
            flags=re.DOTALL,
        )
        if assignment is not None:
            return (
                f"{indent}{assignment.group(1)} {assignment.group(2)} = "
                f"undefined; // SkillScope replay ablation for {candidate_id}\n"
            )
        if re.match(r"^return\b", statement):
            return (
                f"{indent}return undefined; "
                f"// SkillScope replay ablation for {candidate_id}\n"
            )
        return (
            f"{indent}; // SkillScope replay ablation for {candidate_id}\n"
        )

    def _line_slice(self, lines: list[str], ablation: AblationPlan) -> tuple[int | None, int | None]:
        start_line = ablation.source_start_line
        end_line = ablation.source_end_line
        if start_line is None or end_line is None:
            return None, None
        start = max(start_line - 1, 0)
        end = min(end_line, len(lines))
        if start >= end:
            return None, None
        return start, end

    def _indentation_for(self, lines: list[str]) -> str:
        for line in lines:
            stripped = line.lstrip()
            if stripped:
                return line[: len(line) - len(stripped)]
        return ""

    def _neutralize_python_code(self, source_text: str, ablation: AblationPlan) -> str | None:
        start_line = ablation.source_start_line
        end_line = ablation.source_end_line
        if start_line is None or end_line is None:
            return None
        try:
            tree = ast.parse(source_text)
        except SyntaxError:
            return None

        transformer = _PythonAblationTransformer(
            start_line=start_line,
            end_line=end_line,
            start_column=ablation.source_start_column,
            end_column=ablation.source_end_column,
            raw_text=ablation.raw_text,
        )
        updated_tree = transformer.visit(tree)
        if not transformer.replaced:
            return None
        ast.fix_missing_locations(updated_tree)
        return ast.unparse(updated_tree) + "\n"


class _PythonAblationTransformer(ast.NodeTransformer):
    def __init__(
        self,
        *,
        start_line: int,
        end_line: int,
        start_column: int | None,
        end_column: int | None,
        raw_text: str | None,
    ) -> None:
        self.start_line = start_line
        self.end_line = end_line
        self.start_column = start_column
        self.end_column = end_column
        self.raw_text = (raw_text or "").strip()
        self.replaced = False

    def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.stmt]:
        if isinstance(node.value, ast.Call) and self._matches_target_call(
            node.value
        ):
            self.replaced = True
            preserved = self._preserved_argument_statements(node.value, node)
            return [*preserved, ast.copy_location(ast.Pass(), node)]
        if self._contains_target_call(node.value):
            return self._replacement_statements(
                node=node,
                terminal=ast.Pass(),
            )
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        if not self._contains_target_call(node):
            return self.generic_visit(node)
        if self._direct_binop_target(node.value):
            return self.generic_visit(node)
        terminal: ast.stmt
        if any(self._contains_target_call(target) for target in node.targets):
            terminal = ast.Pass()
        else:
            terminal = ast.Assign(
                targets=node.targets,
                value=ast.Constant(value=None),
                type_comment=getattr(node, "type_comment", None),
            )
        return self._replacement_statements(node=node, terminal=terminal)

    def visit_AnnAssign(
        self,
        node: ast.AnnAssign,
    ) -> ast.AST | list[ast.stmt]:
        if not self._contains_target_call(node):
            return self.generic_visit(node)
        terminal: ast.stmt
        if self._contains_target_call(node.target):
            terminal = ast.Pass()
        else:
            terminal = ast.AnnAssign(
                target=node.target,
                annotation=node.annotation,
                value=ast.Constant(value=None),
                simple=node.simple,
            )
        return self._replacement_statements(node=node, terminal=terminal)

    def visit_Return(self, node: ast.Return) -> ast.AST | list[ast.stmt]:
        if isinstance(node.value, ast.Call) and self._matches_target_call(
            node.value
        ):
            self.replaced = True
            preserved = self._preserved_argument_statements(node.value, node)
            replacement = ast.copy_location(
                ast.Return(value=ast.Constant(value=None)),
                node,
            )
            return [*preserved, replacement]
        if node.value is not None and self._contains_target_call(node.value):
            return self._replacement_statements(
                node=node,
                terminal=ast.Return(value=ast.Constant(value=None)),
            )
        return self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        left_is_target = self._matches_target_call(node.left)
        right_is_target = self._matches_target_call(node.right)
        if left_is_target and not right_is_target:
            self.replaced = True
            assert isinstance(node.left, ast.Call)
            values = [
                *self._preserved_argument_values(node.left),
                self.visit(node.right),
            ]
            return ast.copy_location(
                self._sequence_value(values=values, selected_index=-1),
                node,
            )
        if right_is_target and not left_is_target:
            self.replaced = True
            assert isinstance(node.right, ast.Call)
            values = [
                self.visit(node.left),
                *self._preserved_argument_values(node.right),
            ]
            return ast.copy_location(
                self._sequence_value(values=values, selected_index=0),
                node,
            )
        if left_is_target and right_is_target:
            self.replaced = True
            assert isinstance(node.left, ast.Call)
            assert isinstance(node.right, ast.Call)
            values = [
                *self._preserved_argument_values(node.left),
                *self._preserved_argument_values(node.right),
                ast.Constant(value=None),
            ]
            return ast.copy_location(
                self._sequence_value(values=values, selected_index=-1),
                node,
            )
        return self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        retained = [
            value
            for value in node.values
            if not self._contains_target_call(value)
        ]
        if len(retained) == len(node.values):
            return self.generic_visit(node)
        self.replaced = True
        if not retained:
            return ast.copy_location(ast.Constant(value=False), node)
        visited = [self.visit(value) for value in retained]
        if len(visited) == 1:
            return ast.copy_location(visited[0], node)
        return ast.copy_location(ast.BoolOp(op=node.op, values=visited), node)

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        if self._contains_target_call(node):
            self.replaced = True
            values = [
                *self._preserved_call_values(node),
                ast.Constant(value=False),
            ]
            return ast.copy_location(
                self._sequence_value(values=values, selected_index=-1),
                node,
            )
        return self.generic_visit(node)

    def visit_Await(self, node: ast.Await) -> ast.AST:
        if self._contains_target_call(node):
            self.replaced = True
            values = [
                *self._preserved_call_values(node),
                ast.Constant(value=None),
            ]
            return ast.copy_location(
                self._sequence_value(values=values, selected_index=-1),
                node,
            )
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if self._matches_target_call(node):
            self.replaced = True
            preserved_values = self._preserved_argument_values(node)
            if not preserved_values:
                return ast.copy_location(ast.Constant(value=None), node)
            replacement = ast.Subscript(
                value=ast.Tuple(
                    elts=[*preserved_values, ast.Constant(value=None)],
                    ctx=ast.Load(),
                ),
                slice=ast.Constant(value=-1),
                ctx=ast.Load(),
            )
            return ast.copy_location(replacement, node)
        if self._contains_target_call(node):
            self.replaced = True
            values = [
                *self._preserved_call_values(node),
                ast.Constant(value=None),
            ]
            return ast.copy_location(
                self._sequence_value(values=values, selected_index=-1),
                node,
            )
        return self.generic_visit(node)

    def _replacement_statements(
        self,
        *,
        node: ast.stmt,
        terminal: ast.stmt,
    ) -> list[ast.stmt]:
        self.replaced = True
        preserved = [
            ast.copy_location(ast.Expr(value=value), node)
            for value in self._preserved_call_values(node)
        ]
        return [*preserved, ast.copy_location(terminal, node)]

    def _preserved_argument_statements(
        self,
        node: ast.Call,
        location: ast.AST,
    ) -> list[ast.stmt]:
        return [
            ast.copy_location(ast.Expr(value=value), location)
            for value in self._preserved_argument_values(node)
        ]

    def _preserved_argument_values(self, node: ast.Call) -> list[ast.expr]:
        values: list[ast.expr] = []
        if any(isinstance(item, ast.Call) for item in ast.walk(node.func)):
            visited_func = self.visit(node.func)
            if isinstance(visited_func, ast.expr):
                values.append(visited_func)
        for argument in node.args:
            argument_value = (
                argument.value if isinstance(argument, ast.Starred) else argument
            )
            visited_argument = self.visit(argument_value)
            if isinstance(visited_argument, ast.expr):
                values.append(visited_argument)
        for keyword in node.keywords:
            visited_keyword = self.visit(keyword.value)
            if isinstance(visited_keyword, ast.expr):
                values.append(visited_keyword)
        return values

    def _preserved_call_values(self, node: ast.AST) -> list[ast.expr]:
        values: list[ast.expr] = []

        def collect(current: ast.AST) -> None:
            if isinstance(current, ast.Call):
                if self._matches_target_call(current) or self._contains_target_call(
                    current
                ):
                    collect(current.func)
                    for argument in current.args:
                        collect(argument)
                    for keyword in current.keywords:
                        collect(keyword.value)
                    return
                values.append(current)
                return
            for child in ast.iter_child_nodes(current):
                collect(child)

        collect(node)
        return values

    def _sequence_value(
        self,
        *,
        values: list[ast.AST],
        selected_index: int,
    ) -> ast.expr:
        expressions = [value for value in values if isinstance(value, ast.expr)]
        if not expressions:
            return ast.Constant(value=None)
        if len(expressions) == 1:
            return expressions[0]
        return ast.Subscript(
            value=ast.Tuple(elts=expressions, ctx=ast.Load()),
            slice=ast.Constant(value=selected_index),
            ctx=ast.Load(),
        )

    def _direct_binop_target(self, node: ast.AST) -> bool:
        return isinstance(node, ast.BinOp) and (
            self._matches_target_call(node.left)
            or self._matches_target_call(node.right)
        )

    def _contains_target_call(self, node: ast.AST) -> bool:
        return any(
            self._matches_target_call(item)
            for item in ast.walk(node)
            if isinstance(item, ast.Call)
        )

    def _matches_target_call(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", lineno)
        if lineno is None or end_lineno is None:
            return False
        if not (
            lineno >= self.start_line
            and end_lineno <= self.end_line
        ):
            return False
        if (
            self.start_column is not None
            and self.end_column is not None
            and (
                getattr(node, "col_offset", None) != self.start_column
                or getattr(node, "end_col_offset", None) != self.end_column
            )
        ):
            return False
        if not self.raw_text:
            return lineno == self.start_line and end_lineno == self.end_line
        return ast.unparse(node).strip() == self.raw_text
