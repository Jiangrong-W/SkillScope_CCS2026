from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from skillscope.common.llm import (
    DisabledLLMClient,
    JSONResponseContract,
    MAX_VALIDATED_LLM_ATTEMPTS,
    PromptAssetLoader,
    StructuredLLMClient,
    complete_validated_json,
)


CONDITIONAL_RE = re.compile(r"^(if|when|unless)\b(.*?)(?:,\s*|\s+then\s+)(.+)$", re.IGNORECASE)
OTHERWISE_RE = re.compile(r"^otherwise\b[:,]?\s*(.+)$", re.IGNORECASE)

PREDICATE_COVERAGE_TERMS = (
    "if",
    "only after",
    "only before",
    "only if",
    "only when",
    "only with",
    "provided that",
    "subject to",
    "unless",
    "when",
    "如果",
    "除非",
    "当",
)

CONDITIONAL_AUTH_GUARD_RE = re.compile(
    r"\b(?:after|before|following|if|once|only|pending|prior\s+to|"
    r"provided|subject|unless|upon|when|with)\b"
    r"(?:(?![.;!?！？。；]).){0,96}?"
    r"\b(?:approv(?:al|e|ed|es|ing)|"
    r"authori[sz](?:ation|e|ed|es|ing)|"
    r"consent(?:ed|ing|s)?)\b",
    re.IGNORECASE,
)

OPERATION_FAMILY_TERMS: dict[str, tuple[str, ...]] = {
    "send": (
        "deliver",
        "exfiltrate",
        "forward",
        "leak",
        "post",
        "publish",
        "send",
        "share",
        "sync",
        "transmit",
        "upload",
        "上传",
        "发送",
        "同步",
    ),
    "read": (
        "access",
        "download",
        "inspect",
        "load",
        "open",
        "read",
        "retrieve",
        "访问",
        "加载",
        "打开",
        "获取",
        "读取",
    ),
    "write": (
        "append",
        "export",
        "modify",
        "save",
        "store",
        "update",
        "write",
        "保存",
        "修改",
        "导出",
        "写入",
    ),
    "execute": (
        "call",
        "eval",
        "execute",
        "install",
        "invoke",
        "launch",
        "run",
        "执行",
        "调用",
        "运行",
    ),
    "collect": (
        "collect",
        "fetch",
        "gather",
        "收集",
        "获取",
        "采集",
    ),
    "analyze": (
        "analyze",
        "assess",
        "review",
        "summarize",
        "分析",
        "审查",
        "总结",
        "评估",
    ),
    "delete": (
        "delete",
        "erase",
        "remove",
        "删除",
        "清除",
        "移除",
    ),
    "create": (
        "build",
        "create",
        "generate",
        "render",
        "创建",
        "构建",
        "渲染",
        "生成",
    ),
    "output": (
        "display",
        "present",
        "respond",
        "return",
        "show",
        "展示",
        "呈现",
        "返回",
        "回复",
    ),
}

OPERATION_TYPE_FAMILIES = {
    "analysis": "analyze",
    "command_execution": "execute",
    "external_data_transmission": "send",
    "file_read": "read",
    "file_write": "write",
    "network_send": "send",
    "read_env": "read",
    "respond": "output",
    "return": "output",
    "sensitive_data_access": "read",
    "user_output": "output",
}

SEMANTIC_COVERAGE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "after",
    "be",
    "before",
    "been",
    "being",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "of",
    "on",
    "once",
    "or",
    "otherwise",
    "please",
    "that",
    "the",
    "then",
    "these",
    "this",
    "those",
    "to",
    "unless",
    "upon",
    "via",
    "when",
    "while",
    "with",
}

RISK_OR_COMMAND_TOKENS = {
    "aws",
    "bash",
    "command",
    "cookie",
    "credential",
    "credentials",
    "curl",
    "endpoint",
    "environment",
    "external",
    "history",
    "identifier",
    "key",
    "network",
    "password",
    "private",
    "remote",
    "secret",
    "secrets",
    "shell",
    "ssh",
    "token",
    "wget",
    "凭据",
    "命令",
    "外部",
    "密码",
    "私有",
    "秘密",
    "网络",
    "远程",
    "令牌",
}


@dataclass(slots=True)
class MarkdownBlock:
    block_id: str
    block_type: str
    text: str
    source_file: str
    start_line: int
    end_line: int
    section_title: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InstructionGraphNodeSpec:
    local_id: str
    node_type: str
    summary: str
    raw_text: str
    block_ids: list[str] = field(default_factory=list)
    operation_type: str | None = None
    object_ref: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InstructionGraphEdgeSpec:
    source_local_id: str
    target_local_id: str
    edge_type: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InstructionGraphSpec:
    nodes: list[InstructionGraphNodeSpec] = field(default_factory=list)
    edges: list[InstructionGraphEdgeSpec] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    strategy: str = "llm"


class InstructionSemanticNormalizer:
    """LLM-first instruction graph synthesis from markdown blocks."""

    def __init__(
        self,
        *,
        llm_client: StructuredLLMClient | None = None,
        prompt_loader: PromptAssetLoader | None = None,
        prompt_asset: str = "prompts/module1_instruction_graph_normalization.md",
    ) -> None:
        self.llm_client = llm_client or DisabledLLMClient()
        self.prompt_loader = prompt_loader
        self.prompt_asset = prompt_asset
        self.last_validation_error: str | None = None

    def build_instruction_graph_spec(self, *, skill_name: str, markdown_blocks: list[MarkdownBlock]) -> InstructionGraphSpec:
        llm_spec = self._build_instruction_graph_with_llm(skill_name=skill_name, markdown_blocks=markdown_blocks)
        if llm_spec is not None:
            return llm_spec
        return self._build_fallback_spec(markdown_blocks)

    def _build_instruction_graph_with_llm(
        self,
        *,
        skill_name: str,
        markdown_blocks: list[MarkdownBlock],
    ) -> InstructionGraphSpec | None:
        if self.prompt_loader is None:
            return None
        self.last_validation_error = None

        required_atomic_actions = {
            block.block_id: [
                unit["text"]
                for unit in self._atomic_action_units(block.text)
            ]
            for block in markdown_blocks
            if block.block_type
            in {"list_item", "paragraph", "code_fence"}
            and block.text.strip()
        }
        required_predicate_spans = {
            block.block_id: self._required_predicate_spans(block.text)
            for block in markdown_blocks
            if block.block_type
            in {"list_item", "paragraph", "code_fence"}
            and block.text.strip()
        }
        actionable_block_ids = {
            block_id
            for block_id in set(required_atomic_actions).union(
                required_predicate_spans
            )
            if required_atomic_actions.get(block_id)
            or required_predicate_spans.get(block_id)
        }
        payload = {
            "skill_name": skill_name,
            "markdown_blocks": [self._block_to_dict(block) for block in markdown_blocks],
            "required_atomic_actions": required_atomic_actions,
            "required_predicate_spans": required_predicate_spans,
            "actionable_block_ids": sorted(actionable_block_ids),
        }
        block_text_by_id = {
            block.block_id: block.text for block in markdown_blocks
        }
        try:
            response = complete_validated_json(
                self.llm_client,
                system_prompt=self.prompt_loader.load(self.prompt_asset),
                user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
                schema_name="instruction_graph_normalization",
                contract=JSONResponseContract(
                    required_fields=("nodes", "edges", "notes"),
                    consistency_checks=(
                        lambda response: self._validate_graph_response_shape(
                            response,
                            block_text_by_id=block_text_by_id,
                            actionable_block_ids=actionable_block_ids,
                        ),
                    ),
                ),
                max_attempts=MAX_VALIDATED_LLM_ATTEMPTS,
            )
        except RuntimeError as exc:
            self.last_validation_error = str(exc)
            return None

        nodes_payload = response.get("nodes")
        edges_payload = response.get("edges")
        if not isinstance(nodes_payload, list) or not nodes_payload:
            return None
        if not isinstance(edges_payload, list):
            edges_payload = []

        node_specs: list[InstructionGraphNodeSpec] = []
        for item in nodes_payload:
            if not isinstance(item, dict):
                continue
            local_id = str(item.get("local_id") or "").strip()
            node_type = str(item.get("node_type") or "").strip().upper()
            summary = str(item.get("summary") or "").strip()
            raw_text = str(item.get("raw_text") or summary).strip()
            if not local_id or node_type not in {"INSTR_ACTION", "INSTR_PREDICATE"} or not summary:
                continue
            block_ids = item.get("block_ids")
            if not isinstance(block_ids, list):
                block_ids = []
            attributes = item.get("attributes")
            node_specs.append(
                InstructionGraphNodeSpec(
                    local_id=local_id,
                    node_type=node_type,
                    summary=summary,
                    raw_text=raw_text,
                    block_ids=[str(block_id) for block_id in block_ids],
                    operation_type=str(item.get("operation_type") or self._default_operation_type(node_type)),
                    object_ref=self._optional_text(
                        item.get("object_ref")
                        or (
                            attributes.get("object_ref")
                            if isinstance(attributes, dict)
                            else None
                        )
                    ),
                    attributes=attributes if isinstance(attributes, dict) else {},
                )
            )
        if not node_specs:
            return None

        edge_specs: list[InstructionGraphEdgeSpec] = []
        valid_node_ids = {node.local_id for node in node_specs}
        for item in edges_payload:
            if not isinstance(item, dict):
                continue
            source_local_id = str(item.get("source_local_id") or "").strip()
            target_local_id = str(item.get("target_local_id") or "").strip()
            edge_type = str(item.get("edge_type") or "").strip().upper()
            if not source_local_id or not target_local_id or edge_type not in {
                "SEQUENTIAL",
                "CONDITIONAL_TRUE",
                "CONDITIONAL_FALSE",
                "SEMANTIC_DEP",
            }:
                continue
            if source_local_id not in valid_node_ids or target_local_id not in valid_node_ids:
                continue
            attributes = item.get("attributes")
            edge_specs.append(
                InstructionGraphEdgeSpec(
                    source_local_id=source_local_id,
                    target_local_id=target_local_id,
                    edge_type=edge_type,
                    attributes=attributes if isinstance(attributes, dict) else {},
                )
            )

        notes = response.get("notes")
        return InstructionGraphSpec(
            nodes=node_specs,
            edges=edge_specs,
            notes=[str(note) for note in notes] if isinstance(notes, list) else [],
            strategy="llm",
        )

    def _build_fallback_spec(self, markdown_blocks: list[MarkdownBlock]) -> InstructionGraphSpec:
        nodes: list[InstructionGraphNodeSpec] = []
        edges: list[InstructionGraphEdgeSpec] = []
        pending_sources: list[tuple[str, str]] = []
        open_conditional: tuple[str, str] | None = None
        counter = 0

        def connect_pending(target_id: str) -> None:
            nonlocal pending_sources
            for source_id, edge_type in pending_sources:
                edges.append(
                    InstructionGraphEdgeSpec(
                        source_id,
                        target_id,
                        edge_type,
                    )
                )
            pending_sources = []

        def add_action(
            *,
            text: str,
            block: MarkdownBlock,
            branch_disposition: str | None = None,
        ) -> str:
            nonlocal counter
            counter += 1
            action_id = f"n{counter}"
            operation_type, object_ref = self._infer_action_semantics(text)
            attributes: dict[str, Any] = {
                "normalization_strategy": "heuristic_fallback"
            }
            if branch_disposition is not None:
                attributes["branch_disposition"] = branch_disposition
            nodes.append(
                InstructionGraphNodeSpec(
                    local_id=action_id,
                    node_type="INSTR_ACTION",
                    summary=self._normalize_text(text),
                    raw_text=text,
                    block_ids=[block.block_id],
                    operation_type=operation_type,
                    object_ref=object_ref,
                    attributes=attributes,
                )
            )
            return action_id

        def guarded_disposition(
            text: str,
            *,
            fallback: str,
        ) -> str:
            """Recover SkillScope's explicit deny-by-default branch markers.

            Projected instruction-only guards deliberately name both branches
            so the deterministic parser retains the same control-flow policy
            when LLM normalization is unavailable or rejected.
            """

            normalized = self._normalize_text(text).casefold()
            if "authorized-and-necessary branch" in normalized:
                return "allowed"
            if (
                "safe/default branch" in normalized
                or "safe default branch" in normalized
            ):
                return "safe"
            return fallback

        def close_conditional_without_otherwise() -> None:
            nonlocal open_conditional, pending_sources
            if open_conditional is None:
                return
            predicate_id, true_action_id = open_conditional
            pending_sources = [
                (true_action_id, "SEQUENTIAL"),
                (predicate_id, "CONDITIONAL_FALSE"),
            ]
            open_conditional = None

        for block in markdown_blocks:
            if block.block_type not in {"list_item", "paragraph", "code_fence"} or not block.text.strip():
                continue

            conditional = CONDITIONAL_RE.match(block.text.strip())
            otherwise = OTHERWISE_RE.match(block.text.strip())

            if otherwise and open_conditional is not None:
                predicate_id, true_action_id = open_conditional
                false_action_id = add_action(
                    text=otherwise.group(1).strip(),
                    block=block,
                    branch_disposition=guarded_disposition(
                        otherwise.group(1).strip(),
                        fallback="false",
                    ),
                )
                edges.append(
                    InstructionGraphEdgeSpec(
                        predicate_id,
                        false_action_id,
                        "CONDITIONAL_FALSE",
                    )
                )
                pending_sources = [
                    (true_action_id, "SEQUENTIAL"),
                    (false_action_id, "SEQUENTIAL"),
                ]
                open_conditional = None
                continue

            close_conditional_without_otherwise()

            if conditional:
                counter += 1
                predicate_id = f"n{counter}"
                predicate_text = f"{conditional.group(1)}{conditional.group(2)}".strip()
                nodes.append(
                    InstructionGraphNodeSpec(
                        local_id=predicate_id,
                        node_type="INSTR_PREDICATE",
                        summary=predicate_text,
                        raw_text=predicate_text,
                        block_ids=[block.block_id],
                        operation_type="predicate",
                        object_ref=predicate_text,
                        attributes={
                            "normalization_strategy": "heuristic_fallback"
                        },
                    )
                )
                connect_pending(predicate_id)
                action_text = conditional.group(3).strip()
                action_id = add_action(
                    text=action_text,
                    block=block,
                    branch_disposition=guarded_disposition(
                        action_text,
                        fallback="true",
                    ),
                )
                edges.append(
                    InstructionGraphEdgeSpec(
                        predicate_id,
                        action_id,
                        "CONDITIONAL_TRUE",
                    )
                )
                open_conditional = (predicate_id, action_id)
                continue

            if otherwise:
                action_id = add_action(
                    text=otherwise.group(1).strip(),
                    block=block,
                    branch_disposition="orphan_otherwise",
                )
                connect_pending(action_id)
                pending_sources = [(action_id, "SEQUENTIAL")]
                continue

            action_id = add_action(
                text=block.text.strip(),
                block=block,
            )
            connect_pending(action_id)
            pending_sources = [(action_id, "SEQUENTIAL")]

        if open_conditional is not None:
            predicate_id, _ = open_conditional
            for node in nodes:
                if node.local_id == predicate_id:
                    node.attributes["implicit_false_exit"] = True
                    break

        return InstructionGraphSpec(
            nodes=nodes,
            edges=edges,
            notes=[
                (
                    "Fallback instruction graph synthesis was used after the "
                    "LLM response failed validation: "
                    f"{self.last_validation_error}"
                    if self.last_validation_error
                    else (
                        "Fallback instruction graph synthesis was used because "
                        "no LLM client was available."
                    )
                )
            ],
            strategy="heuristic_fallback",
        )

    def _block_to_dict(self, block: MarkdownBlock) -> dict[str, Any]:
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

    def _default_operation_type(self, node_type: str) -> str:
        return "instruction_step" if node_type == "INSTR_ACTION" else "predicate"

    def _normalize_text(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.strip())
        normalized = re.sub(r"^(always|please|then)\s+", "", normalized, flags=re.IGNORECASE)
        return normalized

    def _infer_action_semantics(self, text: str) -> tuple[str, str | None]:
        normalized = self._normalize_text(text)
        lowered = self._coverage_text(normalized)
        operation_type = "instruction_step"
        matched_span: tuple[int, int] | None = None
        operation_matches: list[tuple[int, int, int, str]] = []
        for family_index, (operation, terms) in enumerate(
            OPERATION_FAMILY_TERMS.items()
        ):
            for term in terms:
                operation_matches.extend(
                    (start, family_index, end, operation)
                    for start, end in self._operation_term_spans(lowered, term)
                )
        if operation_matches:
            start, _, end, operation_type = min(operation_matches)
            matched_span = (start, end)

        object_ref: str | None = None
        inline_code = re.findall(r"`([^`]+)`", normalized)
        if inline_code:
            object_ref = inline_code[-1][:160]
        elif matched_span is not None:
            object_phrase = re.sub(
                r"^(?:the|a|an)\s+",
                "",
                normalized[matched_span[1] :].strip(),
                flags=re.IGNORECASE,
            )
            object_phrase = re.split(
                r"\s+(?:to|from|using|with|into|via|when|if)\s+|[.;]",
                object_phrase,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" `*_:.")
            if object_phrase:
                object_ref = " ".join(object_phrase.split()[:12])[:160]
        return operation_type, object_ref

    def _optional_text(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _validate_graph_response_shape(
        self,
        response: dict[str, Any],
        *,
        block_text_by_id: dict[str, str],
        actionable_block_ids: set[str],
    ) -> str | None:
        if set(response) != {"nodes", "edges", "notes"}:
            return "instruction graph response keys must exactly match the schema"
        nodes = response.get("nodes")
        edges = response.get("edges")
        if not isinstance(nodes, list) or not nodes:
            return "field 'nodes' must be a non-empty list"
        if not isinstance(edges, list):
            return "field 'edges' must be a list"
        notes = response.get("notes")
        if not isinstance(notes, list) or any(
            not isinstance(note, str) for note in notes
        ):
            return "field 'notes' must be a string array"

        node_keys = {
            "local_id",
            "node_type",
            "summary",
            "raw_text",
            "block_ids",
            "operation_type",
            "object_ref",
            "attributes",
        }
        local_ids: list[str] = []
        action_nodes_by_block_id: dict[str, list[dict[str, str | None]]] = {}
        action_node_ids_by_block_id: dict[str, set[str]] = {}
        predicate_raw_text_by_block_id: dict[str, list[str]] = {}
        predicate_node_ids_by_block_id: dict[str, set[str]] = {}
        cited_actionable_block_ids: set[str] = set()
        for node in nodes:
            if not isinstance(node, dict) or set(node) != node_keys:
                return "every instruction node must exactly match the node schema"
            local_id = str(node.get("local_id") or "").strip()
            node_type = str(node.get("node_type") or "").strip().upper()
            summary = node.get("summary")
            raw_text = node.get("raw_text")
            operation_type = node.get("operation_type")
            object_ref = node.get("object_ref")
            attributes = node.get("attributes")
            if not local_id:
                return "every instruction node must have a local_id"
            if node_type not in {"INSTR_ACTION", "INSTR_PREDICATE"}:
                return "instruction node_type is unsupported"
            if not isinstance(summary, str) or not summary.strip():
                return "every instruction node must have a non-empty summary"
            if not isinstance(raw_text, str) or not raw_text.strip():
                return "every instruction node must have grounded raw_text"
            if (
                not isinstance(operation_type, str)
                or not operation_type.strip()
            ):
                return "every instruction node must normalize operation_type"
            if object_ref is not None and not isinstance(object_ref, str):
                return "instruction object_ref must be a string or null"
            if not isinstance(attributes, dict):
                return "instruction node attributes must be an object"
            block_ids = node.get("block_ids")
            if (
                not isinstance(block_ids, list)
                or not block_ids
                or any(
                    not isinstance(block_id, str) or not block_id
                    for block_id in block_ids
                )
            ):
                return "every instruction node must cite source block_ids"
            if any(
                block_id not in block_text_by_id for block_id in block_ids
            ):
                return "instruction node cites an unsupported source block"
            normalized_raw = self._normalize_text(raw_text).casefold()
            if not any(
                normalized_raw
                in self._normalize_text(block_text_by_id[block_id]).casefold()
                for block_id in block_ids
            ):
                return "instruction raw_text is not grounded in its cited block"
            cited_actionable_block_ids.update(
                block_id
                for block_id in block_ids
                if block_id in actionable_block_ids
            )
            for block_id in block_ids:
                if normalized_raw in self._normalize_text(
                    block_text_by_id[block_id]
                ).casefold():
                    if node_type == "INSTR_PREDICATE":
                        predicate_raw_text_by_block_id.setdefault(
                            block_id, []
                        ).append(raw_text)
                        predicate_node_ids_by_block_id.setdefault(
                            block_id, set()
                        ).add(local_id)
                    else:
                        action_node_ids_by_block_id.setdefault(
                            block_id, set()
                        ).add(local_id)
                        action_nodes_by_block_id.setdefault(
                            block_id, []
                        ).append(
                            {
                                "summary": summary,
                                "operation_type": operation_type,
                                "object_ref": object_ref,
                            }
                        )
            semantic_error = self._validate_node_semantics(
                node_type=node_type,
                operation_type=operation_type,
            )
            if semantic_error is not None:
                return semantic_error
            local_ids.append(local_id)

        if len(local_ids) != len(set(local_ids)):
            return "instruction node local_ids must be unique"
        uncovered_blocks = [
            block_id
            for block_id in sorted(actionable_block_ids)
            if block_id not in cited_actionable_block_ids
        ]
        if uncovered_blocks:
            return (
                "instruction graph omitted actionable source blocks: "
                f"{uncovered_blocks}"
            )
        atomically_uncovered_blocks = [
            block_id
            for block_id in sorted(actionable_block_ids)
            if not self._atomic_actions_have_distinct_nodes(
                block_text_by_id[block_id],
                action_nodes_by_block_id.get(block_id, []),
            )
        ]
        if atomically_uncovered_blocks:
            return (
                "instruction graph did not map every atomic action to a "
                f"distinct grounded node: {atomically_uncovered_blocks}"
            )
        omitted_predicate_spans = [
            block_id
            for block_id in sorted(actionable_block_ids)
            if not self._predicate_spans_are_covered(
                block_text_by_id[block_id],
                predicate_raw_text_by_block_id.get(block_id, []),
            )
        ]
        if omitted_predicate_spans:
            return (
                "instruction graph omitted one or more predicate conditions "
                f"from source blocks: {omitted_predicate_spans}"
            )

        edge_keys = {
            "source_local_id",
            "target_local_id",
            "edge_type",
            "attributes",
        }
        valid_local_ids = set(local_ids)
        false_edges: set[tuple[str, str]] = set()
        for edge in edges:
            if not isinstance(edge, dict) or set(edge) != edge_keys:
                return "every instruction edge must exactly match the edge schema"
            if (
                edge.get("source_local_id") not in valid_local_ids
                or edge.get("target_local_id") not in valid_local_ids
            ):
                return "instruction edge references an unknown local node"
            normalized_edge_type = str(
                edge.get("edge_type") or ""
            ).strip().upper()
            if normalized_edge_type not in {
                "SEQUENTIAL",
                "CONDITIONAL_TRUE",
                "CONDITIONAL_FALSE",
                "SEMANTIC_DEP",
            }:
                return "instruction edge_type is unsupported"
            if not isinstance(edge.get("attributes"), dict):
                return "instruction edge attributes must be an object"
            if normalized_edge_type == "CONDITIONAL_FALSE":
                false_edges.add(
                    (
                        str(edge["source_local_id"]),
                        str(edge["target_local_id"]),
                    )
                )

        ordered_block_ids = list(block_text_by_id)
        block_position = {
            block_id: position
            for position, block_id in enumerate(ordered_block_ids)
        }
        structurally_uncovered_otherwise_blocks: list[str] = []
        for block_id in ordered_block_ids:
            if block_id not in actionable_block_ids or not self._anchor_spans(
                self._coverage_text(block_text_by_id[block_id]),
                ("otherwise", "否则"),
            ):
                continue
            allowed_predicate_sources = set(
                predicate_node_ids_by_block_id.get(block_id, set())
            )
            if not allowed_predicate_sources:
                for preceding_block_id in reversed(
                    ordered_block_ids[: block_position[block_id]]
                ):
                    preceding_predicates = predicate_node_ids_by_block_id.get(
                        preceding_block_id, set()
                    )
                    if preceding_predicates:
                        allowed_predicate_sources = set(preceding_predicates)
                        break
            otherwise_action_targets = action_node_ids_by_block_id.get(
                block_id, set()
            )
            if not any(
                source_id in allowed_predicate_sources
                and target_id in otherwise_action_targets
                for source_id, target_id in false_edges
            ):
                structurally_uncovered_otherwise_blocks.append(block_id)
        if structurally_uncovered_otherwise_blocks:
            return (
                "instruction graph did not represent otherwise branches with "
                "a controlling-predicate CONDITIONAL_FALSE edge to an action "
                "in the otherwise block; the controlling predicate must be "
                "same-block or from the nearest preceding grounded-predicate "
                "block: "
                f"{structurally_uncovered_otherwise_blocks}"
            )
        return None

    def _validate_node_semantics(
        self,
        *,
        node_type: str,
        operation_type: str,
    ) -> str | None:
        normalized_operation = operation_type.strip().casefold()
        if node_type == "INSTR_PREDICATE":
            if normalized_operation != "predicate":
                return "predicate nodes must use operation_type=predicate"
        return None

    def _coverage_mask(
        self,
        normalized_block: str,
        grounded_fragments: list[str],
    ) -> list[bool]:
        covered = [False] * len(normalized_block)
        for fragment in grounded_fragments:
            normalized_fragment = self._coverage_text(fragment)
            if not normalized_fragment:
                continue
            occurrence_starts: list[int] = []
            search_start = 0
            while True:
                index = normalized_block.find(
                    normalized_fragment,
                    search_start,
                )
                if index < 0:
                    break
                occurrence_starts.append(index)
                search_start = index + 1
            if not occurrence_starts:
                continue
            best_start = max(
                occurrence_starts,
                key=lambda start: sum(
                    not covered[offset]
                    for offset in range(
                        start,
                        min(start + len(normalized_fragment), len(covered)),
                    )
                ),
            )
            for offset in range(
                best_start,
                min(best_start + len(normalized_fragment), len(covered)),
            ):
                covered[offset] = True
        return covered

    def _atomic_actions_have_distinct_nodes(
        self,
        block_text: str,
        action_nodes: list[dict[str, str | None]],
    ) -> bool:
        action_units = self._atomic_action_units(block_text)
        if not action_units:
            return True
        if len(action_nodes) < len(action_units):
            return False

        candidate_nodes = [
            [
                node_index
                for node_index, node in enumerate(action_nodes)
                if self._action_unit_matches_node(unit, node)
            ]
            for unit in action_units
        ]
        if any(not candidates for candidates in candidate_nodes):
            return False
        requirement_order = sorted(
            range(len(action_units)),
            key=lambda index: len(candidate_nodes[index]),
        )

        def assign(position: int, used_nodes: set[int]) -> bool:
            if position == len(requirement_order):
                return True
            requirement_index = requirement_order[position]
            return any(
                assign(position + 1, used_nodes | {node_index})
                for node_index in candidate_nodes[requirement_index]
                if node_index not in used_nodes
            )

        return assign(0, set())

    def _atomic_action_units(
        self,
        block_text: str,
    ) -> list[dict[str, Any]]:
        normalized = self._coverage_text(block_text)
        occurrences = self._merge_execution_effect_occurrences(
            normalized,
            self._operation_occurrences(normalized),
        )
        unknown_risk_units = self._unknown_risk_action_units(
            normalized,
            known_operation_spans=[
                (occurrence["start"], occurrence["end"])
                for occurrence in occurrences
            ],
        )
        units: list[dict[str, Any]] = []
        for occurrence_index, occurrence in enumerate(occurrences):
            if not self._operation_occurrence_is_affirmative(
                normalized,
                occurrence["start"],
            ):
                continue
            next_start = (
                occurrences[occurrence_index + 1]["start"]
                if occurrence_index + 1 < len(occurrences)
                else len(normalized)
            )
            next_unknown_start = min(
                (
                    unit["_start"]
                    for unit in unknown_risk_units
                    if occurrence["end"] <= unit["_start"] < next_start
                ),
                default=next_start,
            )
            next_start = min(next_start, next_unknown_start)
            hard_boundary = re.search(
                r"[.;。；!?！？]",
                normalized[occurrence["end"] : next_start],
            )
            unit_end = next_start
            if hard_boundary is not None:
                unit_end = occurrence["end"] + hard_boundary.start()
            clause = normalized[occurrence["start"] : unit_end].strip(
                " ,:-"
            )
            tokens = self._semantic_tokens(clause)
            if clause and tokens:
                units.append(
                    {
                        "text": clause,
                        "families": set(occurrence["families"]),
                        "tokens": tokens,
                    }
                )

        units.extend(
            {
                key: value
                for key, value in unit.items()
                if not key.startswith("_")
            }
            for unit in unknown_risk_units
        )
        return units

    def _operation_occurrences(self, text: str) -> list[dict[str, Any]]:
        normalized = self._coverage_text(text)
        families_by_span: dict[tuple[int, int], set[str]] = {}
        for family, terms in OPERATION_FAMILY_TERMS.items():
            for term in terms:
                for span in self._operation_term_spans(normalized, term):
                    families_by_span.setdefault(span, set()).add(family)
        return [
            {
                "start": start,
                "end": end,
                "families": families,
            }
            for (start, end), families in sorted(families_by_span.items())
        ]

    def _merge_execution_effect_occurrences(
        self,
        normalized_text: str,
        occurrences: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for occurrence in occurrences:
            if merged:
                previous = merged[-1]
                bridge = normalized_text[
                    previous["end"] : occurrence["start"]
                ]
                is_execution_effect = (
                    "execute" in previous["families"]
                    and "create" in occurrence["families"]
                    and re.search(r"\bto\s*$", bridge) is not None
                    and re.search(
                        r"[;,。；，!?！？]|\.(?=\s|$)|"
                        r"\b(?:and|but|finally|next|otherwise|then)\b",
                        bridge,
                    )
                    is None
                )
                if is_execution_effect:
                    previous["end"] = occurrence["end"]
                    previous["families"] = set(
                        previous["families"]
                    ).union(occurrence["families"])
                    continue
            merged.append(
                {
                    "start": occurrence["start"],
                    "end": occurrence["end"],
                    "families": set(occurrence["families"]),
                }
            )
        return merged

    def _operation_occurrence_is_affirmative(
        self,
        normalized_text: str,
        operation_start: int,
    ) -> bool:
        sentence_start = max(
            normalized_text.rfind(delimiter, 0, operation_start)
            for delimiter in ".;。；!?！？"
        ) + 1
        prefix = normalized_text[sentence_start:operation_start]
        contrast_matches = list(
            re.finditer(r"\b(?:but|however|instead|yet)\b", prefix)
        )
        if contrast_matches:
            prefix = prefix[contrast_matches[-1].end() :]
        if re.search(
            r"(?:^|\b)(?:avoid|do\s+not|don't|never|no\s+longer|"
            r"skip|without)\b",
            prefix,
        ):
            return False
        return not bool(
            re.match(
                r"\s*(?:because|since|so\s+that|which|where)\b",
                prefix,
            )
        )

    def _unknown_risk_action_units(
        self,
        normalized_text: str,
        *,
        known_operation_spans: list[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        boundary_re = re.compile(
            r"[,.;，。；!?！？]+|"
            r"\b(?:after|alongside|and|before|finally|followed\s+by|"
            r"next|or|otherwise|prior\s+to|then|while)\b"
        )
        boundaries = [0]
        boundaries.extend(
            match.end() for match in boundary_re.finditer(normalized_text)
        )
        boundaries.append(len(normalized_text))
        units: list[dict[str, Any]] = []
        for start, end in zip(boundaries, boundaries[1:]):
            clause = normalized_text[start:end].strip(" ,:-")
            if not clause or self._negative_or_explanatory_clause(clause):
                continue
            clause_start = normalized_text.find(clause, start, end)
            if clause_start < 0 or not self._operation_occurrence_is_affirmative(
                normalized_text,
                clause_start,
            ):
                continue
            if any(
                start <= operation_start < end
                for operation_start, _ in known_operation_spans
            ):
                continue
            tokens = self._semantic_tokens(clause)
            if not self._contains_risk_or_command_token(tokens):
                continue
            families = self._risk_implied_operation_families(tokens)
            if not families or self._looks_like_condition_only(clause):
                continue
            units.append(
                {
                    "text": clause,
                    "families": families,
                    "tokens": tokens,
                    "_start": start,
                    "_end": end,
                }
            )
        return units

    def _action_unit_matches_node(
        self,
        unit: dict[str, Any],
        node: dict[str, str | None],
    ) -> bool:
        summary = str(node.get("summary") or "")
        object_ref = str(node.get("object_ref") or "")
        operation_type = str(node.get("operation_type") or "")
        node_text = self._coverage_text(f"{summary} {object_ref}")
        node_tokens = self._semantic_tokens(node_text)
        node_families = self._operation_families_in_text(summary)
        declared_family = self._declared_operation_family(operation_type)
        if declared_family is not None:
            node_families.add(declared_family)

        unit_families = set(unit["families"])
        if (
            declared_family is not None
            and unit_families
            and declared_family not in unit_families
        ):
            return False
        if unit_families and not unit_families.intersection(node_families):
            return False
        operation_terms = {
            term
            for terms in OPERATION_FAMILY_TERMS.values()
            for term in terms
        }
        distinctive_tokens = {
            token
            for token in set(unit["tokens"])
            if token not in operation_terms
            and token not in {"action", "step", "task", "workflow"}
        }
        risk_tokens = set(unit["tokens"]).intersection(
            RISK_OR_COMMAND_TOKENS
        )
        if risk_tokens and not risk_tokens.issubset(node_tokens):
            return False
        return not distinctive_tokens or bool(
            distinctive_tokens.intersection(node_tokens)
        )

    def _declared_operation_family(self, operation_type: str) -> str | None:
        normalized = operation_type.strip().casefold()
        if normalized in OPERATION_FAMILY_TERMS:
            return normalized
        return OPERATION_TYPE_FAMILIES.get(normalized)

    def _operation_families_in_text(self, text: str) -> set[str]:
        normalized = self._coverage_text(text)
        return {
            family
            for family, terms in OPERATION_FAMILY_TERMS.items()
            if any(
                self._operation_term_spans(normalized, term)
                for term in terms
            )
        }

    def _operation_term_spans(
        self,
        normalized_text: str,
        term: str,
    ) -> list[tuple[int, int]]:
        """Locate a verb lexeme while accepting ordinary English inflections."""

        normalized_term = term.casefold()
        if not normalized_term.isascii() or " " in normalized_term:
            return self._anchor_spans(normalized_text, (normalized_term,))

        forms = {normalized_term}
        if normalized_term.endswith("e"):
            forms.update(
                {
                    f"{normalized_term}s",
                    f"{normalized_term}d",
                    f"{normalized_term[:-1]}ing",
                }
            )
        elif (
            normalized_term.endswith("y")
            and len(normalized_term) > 1
            and normalized_term[-2] not in "aeiou"
        ):
            forms.update(
                {
                    f"{normalized_term[:-1]}ies",
                    f"{normalized_term[:-1]}ied",
                    f"{normalized_term}ing",
                }
            )
        elif normalized_term.endswith(("ch", "o", "s", "sh", "x", "z")):
            forms.update(
                {
                    f"{normalized_term}es",
                    f"{normalized_term}ed",
                    f"{normalized_term}ing",
                }
            )
        else:
            forms.update(
                {
                    f"{normalized_term}s",
                    f"{normalized_term}ed",
                    f"{normalized_term}ing",
                }
            )
        irregular_forms = {
            "build": {"built"},
            "run": {"ran", "running"},
            "send": {"sent"},
            "write": {"wrote", "written"},
        }
        forms.update(irregular_forms.get(normalized_term, set()))
        pattern = re.compile(
            rf"(?<![a-z0-9_])(?:{'|'.join(re.escape(form) for form in sorted(forms, key=len, reverse=True))})(?![a-z0-9_])"
        )
        return [
            (match.start(), match.end())
            for match in pattern.finditer(normalized_text)
            if not self._operation_match_is_path_component(
                normalized_text,
                match.start(),
                match.end(),
            )
        ]

    def _operation_match_is_path_component(
        self,
        normalized_text: str,
        start: int,
        end: int,
    ) -> bool:
        if start > 0 and normalized_text[start - 1] in {"/", "\\"}:
            return True
        return re.match(r"\.[a-z0-9]{1,8}\b", normalized_text[end:]) is not None

    def _semantic_tokens(self, text: str) -> set[str]:
        return {
            match.group(0)
            for match in re.finditer(r"\w+", self._coverage_text(text))
            if match.group(0) not in SEMANTIC_COVERAGE_STOPWORDS
        }

    def _negative_or_explanatory_clause(self, clause: str) -> bool:
        return bool(
            re.match(
                r"^(?:because|do not|don't|never|no |since|so that|which|where)\b",
                clause,
            )
        )

    def _looks_like_condition_only(self, clause: str) -> bool:
        normalized_clause = self._coverage_text(clause)
        return bool(
            re.match(
                r"^(?:if|only after|only before|only if|only when|only with|provided that|subject to|unless|when)\b",
                normalized_clause,
            )
            or CONDITIONAL_AUTH_GUARD_RE.match(normalized_clause)
        )

    def _contains_risk_or_command_token(self, tokens: set[str]) -> bool:
        return bool(tokens.intersection(RISK_OR_COMMAND_TOKENS))

    def _risk_implied_operation_families(
        self,
        tokens: set[str],
    ) -> set[str]:
        if tokens.intersection({"curl", "wget"}):
            return {"execute", "send"}
        if tokens.intersection(
            {"endpoint", "external", "network", "remote", "远程", "外部", "网络"}
        ):
            return {"send"}
        if tokens.intersection(
            {
                "aws",
                "cookie",
                "credential",
                "credentials",
                "environment",
                "history",
                "identifier",
                "key",
                "password",
                "private",
                "secret",
                "secrets",
                "ssh",
                "token",
                "凭据",
                "密码",
                "私有",
                "秘密",
                "令牌",
            }
        ):
            return {"collect", "read"}
        return set()

    def _predicate_spans_are_covered(
        self,
        block_text: str,
        grounded_fragments: list[str],
    ) -> bool:
        normalized_block = self._coverage_text(block_text)
        anchor_spans = self._anchor_spans(
            normalized_block,
            PREDICATE_COVERAGE_TERMS,
        )
        anchor_spans.extend(
            (match.start(), match.end())
            for match in CONDITIONAL_AUTH_GUARD_RE.finditer(normalized_block)
        )
        anchor_spans = sorted(set(anchor_spans))
        if not anchor_spans:
            return True

        covered = self._coverage_mask(
            normalized_block,
            grounded_fragments,
        )

        return all(
            all(covered[offset] for offset in range(start, end))
            for start, end in anchor_spans
        )

    def _required_predicate_spans(self, block_text: str) -> list[str]:
        normalized_block = self._coverage_text(block_text)
        spans = self._anchor_spans(
            normalized_block,
            PREDICATE_COVERAGE_TERMS,
        )
        spans.extend(
            (match.start(), match.end())
            for match in CONDITIONAL_AUTH_GUARD_RE.finditer(normalized_block)
        )
        return [
            normalized_block[start:end]
            for start, end in sorted(set(spans))
        ]

    def _anchor_spans(
        self,
        normalized_text: str,
        anchor_terms: tuple[str, ...],
    ) -> list[tuple[int, int]]:
        spans: set[tuple[int, int]] = set()
        for term in anchor_terms:
            escaped = re.escape(term.casefold())
            if term.isascii():
                pattern = re.compile(
                    rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])"
                )
            else:
                pattern = re.compile(escaped)
            spans.update(
                (match.start(), match.end())
                for match in pattern.finditer(normalized_text)
            )
        return sorted(spans)

    def _coverage_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", str(value).strip().casefold())
