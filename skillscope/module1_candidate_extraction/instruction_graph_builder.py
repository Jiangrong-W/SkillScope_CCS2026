from __future__ import annotations

import re
import shlex
from pathlib import Path

from skillscope.common.models import ActionGraph, SkillBundle, SourceRange, UEGEdge, UEGNode

from .instruction_semantic_normalizer import (
    InstructionGraphEdgeSpec,
    InstructionGraphNodeSpec,
    InstructionSemanticNormalizer,
    MarkdownBlock,
)


LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+(.*)$")
HEADER_RE = re.compile(r"^(#+)\s+(.*)$")
COMMAND_RE = re.compile(
    r"`([^`]+)`|((?:python|python3|bash|sh|node|npx|deno|ts-node)\s+[^\s`]+)"
)
COMMAND_NAMES = frozenset(
    {
        "aws",
        "az",
        "bash",
        "cat",
        "chmod",
        "chown",
        "cp",
        "curl",
        "deno",
        "echo",
        "find",
        "gcloud",
        "git",
        "grep",
        "mkdir",
        "mv",
        "node",
        "npx",
        "open",
        "osascript",
        "printf",
        "python",
        "python3",
        "rm",
        "sed",
        "sh",
        "tar",
        "touch",
        "ts-node",
        "unzip",
        "wget",
    }
)
SCRIPT_INTERPRETERS = frozenset(
    {"bash", "deno", "node", "npx", "python", "python3", "sh", "ts-node"}
)


class InstructionGraphBuilder:
    def __init__(self, normalizer: InstructionSemanticNormalizer | None = None) -> None:
        self.normalizer = normalizer or InstructionSemanticNormalizer()

    def build(self, bundle: SkillBundle) -> ActionGraph:
        graph = ActionGraph(graph_id=f"{bundle.bundle_id}:instruction", layer="instruction")
        entry_id = f"{graph.graph_id}:entry"
        exit_id = f"{graph.graph_id}:exit"
        graph.metadata["entry_node_id"] = entry_id
        graph.metadata["exit_node_id"] = exit_id

        markdown_blocks, sections = self._collect_markdown_blocks(bundle)
        graph.metadata["sections"] = sections
        graph.metadata["markdown_blocks"] = [self._block_metadata(block) for block in markdown_blocks]
        graph.metadata["construction_strategy"] = {
            "structural_parsing": [
                "markdown_section_boundaries",
                "markdown_list_items",
                "markdown_paragraph_blocks",
                "markdown_code_fence_blocks",
            ],
            "semantic_normalization": {
                "component": "InstructionSemanticNormalizer",
                "mode": "llm_graph_synthesis",
                "prompt_asset": self.normalizer.prompt_asset,
            },
        }

        graph.add_node(
            UEGNode(
                node_id=entry_id,
                layer="instruction",
                node_type="ENTRY",
                summary="Instruction entry",
                object_ref=bundle.bundle_id,
                attributes={
                    "provenance": {
                        "origin": "synthetic",
                        "graph_id": graph.graph_id,
                    }
                },
            )
        )

        graph_spec = self.normalizer.build_instruction_graph_spec(
            skill_name=bundle.bundle_id,
            markdown_blocks=markdown_blocks,
        )
        graph.metadata["graph_synthesis"] = {
            "strategy": graph_spec.strategy,
            "notes": graph_spec.notes,
            "node_count": len(graph_spec.nodes),
            "edge_count": len(graph_spec.edges),
        }

        local_to_global: dict[str, str] = {}
        for index, node_spec in enumerate(graph_spec.nodes, start=1):
            node_id = self._add_instruction_node(
                graph=graph,
                bundle=bundle,
                node_spec=node_spec,
                markdown_blocks=markdown_blocks,
                ordinal=index,
            )
            local_to_global[node_spec.local_id] = node_id

        for edge_spec in graph_spec.edges:
            self._add_instruction_edge(graph, edge_spec, local_to_global)

        self._connect_entry_and_exit(graph, local_to_global)
        return graph

    def _collect_markdown_blocks(self, bundle: SkillBundle) -> tuple[list[MarkdownBlock], list[dict[str, object]]]:
        blocks: list[MarkdownBlock] = []
        sections: list[dict[str, object]] = []
        block_counter = 0

        for artifact in bundle.instruction_files:
            lines = Path(artifact.absolute_path).read_text(encoding="utf-8").splitlines()
            frontmatter_end = self._frontmatter_end_line(lines)
            current_section = "root"
            paragraph_lines: list[str] = []
            paragraph_start_line: int | None = None
            in_code_block = False
            code_block_lines: list[str] = []
            code_block_start_line: int | None = None

            def flush_paragraph(end_line: int) -> None:
                nonlocal block_counter, paragraph_lines, paragraph_start_line
                if not paragraph_lines or paragraph_start_line is None:
                    return
                block_counter += 1
                blocks.append(
                    MarkdownBlock(
                        block_id=f"block-{block_counter:04d}",
                        block_type="paragraph",
                        text=" ".join(line.strip() for line in paragraph_lines if line.strip()),
                        source_file=artifact.relative_path,
                        start_line=paragraph_start_line,
                        end_line=end_line,
                        section_title=current_section,
                    )
                )
                paragraph_lines = []
                paragraph_start_line = None

            def flush_code_block(end_line: int) -> None:
                nonlocal block_counter, code_block_lines, code_block_start_line
                if not code_block_lines or code_block_start_line is None:
                    return
                block_counter += 1
                blocks.append(
                    MarkdownBlock(
                        block_id=f"block-{block_counter:04d}",
                        block_type="code_fence",
                        text="\n".join(code_block_lines).strip(),
                        source_file=artifact.relative_path,
                        start_line=code_block_start_line,
                        end_line=end_line,
                        section_title=current_section,
                    )
                )
                code_block_lines = []
                code_block_start_line = None

            for line_number, raw_line in enumerate(lines, start=1):
                if line_number <= frontmatter_end:
                    continue
                stripped = raw_line.strip()

                if stripped.startswith("```"):
                    flush_paragraph(line_number - 1)
                    if in_code_block:
                        flush_code_block(line_number)
                        in_code_block = False
                    else:
                        in_code_block = True
                        code_block_start_line = line_number
                        code_block_lines = []
                    continue

                if in_code_block:
                    code_block_lines.append(raw_line)
                    continue

                header_match = HEADER_RE.match(stripped)
                if header_match:
                    flush_paragraph(line_number - 1)
                    current_section = header_match.group(2).strip()
                    sections.append({"title": current_section, "line": line_number, "file": artifact.relative_path})
                    block_counter += 1
                    blocks.append(
                        MarkdownBlock(
                            block_id=f"block-{block_counter:04d}",
                            block_type="header",
                            text=current_section,
                            source_file=artifact.relative_path,
                            start_line=line_number,
                            end_line=line_number,
                            section_title=current_section,
                            attributes={"level": len(header_match.group(1))},
                        )
                    )
                    continue

                list_match = LIST_RE.match(raw_line)
                if list_match:
                    flush_paragraph(line_number - 1)
                    block_counter += 1
                    blocks.append(
                        MarkdownBlock(
                            block_id=f"block-{block_counter:04d}",
                            block_type="list_item",
                            text=list_match.group(1).strip(),
                            source_file=artifact.relative_path,
                            start_line=line_number,
                            end_line=line_number,
                            section_title=current_section,
                        )
                    )
                    continue

                if not stripped or stripped == "---":
                    flush_paragraph(line_number - 1)
                    continue

                if paragraph_start_line is None:
                    paragraph_start_line = line_number
                paragraph_lines.append(raw_line)

            flush_paragraph(len(lines))
            if in_code_block:
                flush_code_block(len(lines))

        return blocks, sections

    def _add_instruction_node(
        self,
        *,
        graph: ActionGraph,
        bundle: SkillBundle,
        node_spec: InstructionGraphNodeSpec,
        markdown_blocks: list[MarkdownBlock],
        ordinal: int,
    ) -> str:
        node_id = f"{graph.graph_id}:{ordinal:03d}"
        referenced_blocks = [block for block in markdown_blocks if block.block_id in set(node_spec.block_ids)]
        source_file = referenced_blocks[0].source_file if referenced_blocks else (bundle.instruction_files[0].relative_path if bundle.instruction_files else None)
        start_line = min((block.start_line for block in referenced_blocks), default=1)
        end_line = max((block.end_line for block in referenced_blocks), default=start_line)
        raw_text = node_spec.raw_text.strip() or " ".join(block.text for block in referenced_blocks)
        invoked_scripts, command_invocations = self._extract_invocations(raw_text, bundle)
        operation_type = node_spec.operation_type or (
            "instruction_step" if node_spec.node_type == "INSTR_ACTION" else "predicate"
        )
        summary_operation_type, summary_object_ref = (
            self._infer_action_semantics(node_spec.summary)
        )
        inferred_operation_type, inferred_object_ref = (
            self._infer_action_semantics(raw_text)
        )
        if summary_operation_type != "instruction_step":
            operation_type = summary_operation_type
        elif (
            operation_type == "instruction_step"
            and inferred_operation_type != "instruction_step"
        ):
            operation_type = inferred_operation_type
        object_ref = (
            node_spec.object_ref
            or (invoked_scripts[0] if invoked_scripts else None)
            or summary_object_ref
            or inferred_object_ref
        )
        if node_spec.node_type == "INSTR_PREDICATE" and not object_ref:
            object_ref = node_spec.summary
        provenance = {
            "origin": "instruction",
            "source_file": source_file,
            "source_range": {
                "start_line": start_line,
                "end_line": end_line,
            },
            "block_ids": list(node_spec.block_ids),
        }
        attributes = dict(node_spec.attributes)
        attributes.update(
            {
                "referenced_block_ids": node_spec.block_ids,
                "referenced_sections": sorted(
                    {
                        block.section_title
                        for block in referenced_blocks
                        if block.section_title
                    }
                ),
                "invoked_scripts": invoked_scripts,
                "command_invocations": command_invocations,
                "normalization_strategy": graph.metadata[
                    "graph_synthesis"
                ]["strategy"],
                "prompt_asset": self.normalizer.prompt_asset,
                "provenance": provenance,
            }
        )

        graph.add_node(
            UEGNode(
                node_id=node_id,
                layer="instruction",
                node_type=node_spec.node_type,
                summary=node_spec.summary[:160],
                source_file=source_file,
                source_range=SourceRange(start_line=start_line, end_line=end_line),
                raw_text=raw_text,
                operation_type=operation_type,
                object_ref=object_ref,
                risk_tags=self._risk_tags(
                    raw_text,
                    has_explicit_command=bool(
                        command_invocations or invoked_scripts
                    ),
                ),
                attributes=attributes,
            )
        )
        return node_id

    def _add_instruction_edge(
        self,
        graph: ActionGraph,
        edge_spec: InstructionGraphEdgeSpec,
        local_to_global: dict[str, str],
    ) -> None:
        source_id = local_to_global.get(edge_spec.source_local_id)
        target_id = local_to_global.get(edge_spec.target_local_id)
        if not source_id or not target_id:
            return
        edge_attributes = dict(edge_spec.attributes)
        edge_attributes.setdefault("semantic_family", "control")
        graph.add_edge(
            UEGEdge(
                source=source_id,
                target=target_id,
                edge_type=edge_spec.edge_type,
                attributes=edge_attributes,
            )
        )

    def _connect_entry_and_exit(self, graph: ActionGraph, local_to_global: dict[str, str]) -> None:
        entry_id = graph.metadata["entry_node_id"]
        exit_id = graph.metadata["exit_node_id"]
        internal_node_ids = set(local_to_global.values())
        graph.add_node(
            UEGNode(
                node_id=exit_id,
                layer="instruction",
                node_type="EXIT",
                summary="Instruction exit",
                object_ref=graph.graph_id,
                attributes={
                    "provenance": {
                        "origin": "synthetic",
                        "graph_id": graph.graph_id,
                    }
                },
            )
        )
        if not internal_node_ids:
            graph.add_edge(
                UEGEdge(
                    source=entry_id,
                    target=exit_id,
                    edge_type="SEQUENTIAL",
                    attributes={"synthetic": True, "semantic_family": "control"},
                )
            )
            return

        for node in graph.nodes:
            if (
                node.node_id in internal_node_ids
                and node.node_type == "INSTR_PREDICATE"
                and node.attributes.get("implicit_false_exit") is True
            ):
                graph.add_edge(
                    UEGEdge(
                        source=node.node_id,
                        target=exit_id,
                        edge_type="CONDITIONAL_FALSE",
                        attributes={
                            "synthetic": True,
                            "semantic_family": "control",
                        },
                    )
                )

        incoming = {edge.target for edge in graph.edges if edge.source in internal_node_ids and edge.target in internal_node_ids}
        outgoing = {edge.source for edge in graph.edges if edge.source in internal_node_ids and edge.target in internal_node_ids}

        roots = [node.node_id for node in graph.nodes if node.node_id in internal_node_ids and node.node_id not in incoming]
        leaves = [node.node_id for node in graph.nodes if node.node_id in internal_node_ids and node.node_id not in outgoing]

        if not roots:
            roots = sorted(internal_node_ids)
        if not leaves:
            leaves = sorted(internal_node_ids)

        for root_id in roots:
            graph.add_edge(
                UEGEdge(
                    source=entry_id,
                    target=root_id,
                    edge_type="SEQUENTIAL",
                    attributes={"synthetic": True, "semantic_family": "control"},
                )
            )

        for leaf_id in leaves:
            graph.add_edge(
                UEGEdge(
                    source=leaf_id,
                    target=exit_id,
                    edge_type="SEQUENTIAL",
                    attributes={"synthetic": True, "semantic_family": "control"},
                )
            )

    def _block_metadata(self, block: MarkdownBlock) -> dict[str, object]:
        return {
            "block_id": block.block_id,
            "block_type": block.block_type,
            "text": block.text,
            "source_file": block.source_file,
            "start_line": block.start_line,
            "end_line": block.end_line,
            "section_title": block.section_title,
            "attributes": block.attributes,
        }

    def _extract_invocations(self, text: str, bundle: SkillBundle) -> tuple[list[str], list[str]]:
        invocations: list[str] = []
        invoked_scripts: list[str] = []

        for match in COMMAND_RE.finditer(text):
            command_text = (match.group(1) or match.group(2) or "").strip()
            if command_text and self._is_explicit_command(command_text, bundle):
                invocations.append(command_text)

        for script in bundle.script_files:
            if any(
                self._command_invokes_script(command, script.relative_path)
                for command in invocations
            ):
                invoked_scripts.append(script.relative_path)
                continue
            # Preserve explicit prose invocations that omit backticks or an
            # interpreter, while avoiding the old behavior of treating every
            # mere filename mention as an execution request.
            invocation_pattern = re.compile(
                r"\b(?:run|execute|invoke)\s+`?"
                + re.escape(script.relative_path)
                + r"`?(?=\s|[.,;:]|$)",
                flags=re.IGNORECASE,
            )
            if invocation_pattern.search(text):
                invoked_scripts.append(script.relative_path)

        return sorted(set(invoked_scripts)), invocations

    def _command_invokes_script(self, command: str, target: str) -> bool:
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError:
            return False
        if not tokens:
            return False
        normalized_target = Path(target).as_posix()
        target_name = Path(target).name
        executable = tokens[0]
        if (
            Path(executable).as_posix() == normalized_target
            or Path(executable).name == target_name
        ):
            return True
        if Path(executable).name not in SCRIPT_INTERPRETERS:
            return False
        for token in tokens[1:]:
            if token.startswith("-"):
                continue
            return (
                Path(token).as_posix() == normalized_target
                or Path(token).name == target_name
            )
        return False

    def _is_explicit_command(self, text: str, bundle: SkillBundle) -> bool:
        try:
            tokens = shlex.split(text, posix=True)
        except ValueError:
            return False
        if not tokens:
            return False
        executable = tokens[0]
        executable_name = Path(executable).name
        script_names = {
            Path(script.relative_path).name for script in bundle.script_files
        }
        if executable_name in COMMAND_NAMES or executable_name in script_names:
            return True
        if executable.startswith(("./", "../")):
            return True
        if len(tokens) > 1 and any(
            marker in text for marker in ("&&", "||", ";", "|", ">", "<")
        ):
            return True
        return False

    def _risk_tags(
        self,
        line: str,
        *,
        has_explicit_command: bool = False,
    ) -> list[str]:
        lowered = line.lower()
        risk_tags: list[str] = []
        if any(
            word in lowered
            for word in (
                "telegram",
                "webhook",
                "upload",
                "send",
                "post to",
                "externally",
                "发送",
                "上传",
                "外部",
            )
        ):
            risk_tags.append("network")
        if has_explicit_command or any(
            word in lowered
            for word in (
                "run ",
                "execute ",
                "invoke ",
                "shell",
                "command",
                "运行",
                "执行",
                "调用",
                "命令",
            )
        ):
            risk_tags.append("command_execution")
        if any(
            word in lowered
            for word in (
                "history",
                "identifier",
                "hostname",
                "token",
                "credential",
                ".ssh",
                ".aws",
                "历史记录",
                "令牌",
                "凭据",
            )
        ):
            risk_tags.append("sensitive_collection")
        return risk_tags

    def _frontmatter_end_line(self, lines: list[str]) -> int:
        if not lines or lines[0].lstrip("\ufeff").strip() != "---":
            return 0
        for index, line in enumerate(lines[1:], start=2):
            if line.strip() in {"---", "..."}:
                return index
        return 0

    def _infer_action_semantics(self, text: str) -> tuple[str, str | None]:
        return self.normalizer._infer_action_semantics(text)
