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
    ResponseValidationError,
    StructuredLLMClient,
    complete_validated_json,
)
from skillscope.common.models import SkillProfile, UEGNode, UnifiedExecutionGraph


TOKEN_RE = re.compile(r"[a-z0-9_]+")
ALLOWED_EDGE_TYPES = {"SEQUENTIAL", "CONDITIONAL_TRUE", "CONDITIONAL_FALSE", "SEMANTIC_DEP"}


@dataclass(slots=True)
class InstructionExecutionPlan:
    node_ids: list[str] = field(default_factory=list)
    strategy: str = "fallback"
    notes: list[str] = field(default_factory=list)


class TaskConditionedInstructionPlanner:
    def __init__(
        self,
        *,
        llm_client: StructuredLLMClient | None = None,
        prompt_loader: PromptAssetLoader | None = None,
        prompt_asset: str = "prompts/module2_task_conditioned_execution.md",
        max_paths: int = 32,
        max_depth: int = 64,
    ) -> None:
        self.llm_client = llm_client or DisabledLLMClient()
        self.prompt_loader = prompt_loader
        self.prompt_asset = prompt_asset
        self.max_paths = max_paths
        self.max_depth = max_depth

    def plan(self, *, profile: SkillProfile, ueg: UnifiedExecutionGraph, prompt: str) -> InstructionExecutionPlan:
        llm_plan = self._plan_with_llm(profile=profile, ueg=ueg, prompt=prompt)
        if llm_plan is not None and llm_plan.node_ids:
            return llm_plan
        return self._plan_with_fallback(profile=profile, ueg=ueg, prompt=prompt)

    def _plan_with_llm(
        self,
        *,
        profile: SkillProfile,
        ueg: UnifiedExecutionGraph,
        prompt: str,
    ) -> InstructionExecutionPlan | None:
        if self.prompt_loader is None:
            return None

        instruction_nodes = [
            {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "summary": node.summary,
                "raw_text": node.raw_text,
                "risk_tags": node.risk_tags,
                "source_file": node.source_file,
                "source_range": (
                    {
                        "start_line": node.source_range.start_line,
                        "end_line": node.source_range.end_line,
                    }
                    if node.source_range is not None
                    else None
                ),
                "invoked_scripts": list(node.attributes.get("invoked_scripts", [])),
            }
            for node in ueg.nodes
            if node.layer == "instruction"
        ]
        instruction_edges = [
            {
                "source": edge.source,
                "target": edge.target,
                "edge_type": edge.edge_type,
            }
            for edge in ueg.edges
            if edge.edge_type in ALLOWED_EDGE_TYPES
            and self._node_is_instruction(ueg.node_by_id(edge.source))
            and self._node_is_instruction(ueg.node_by_id(edge.target))
        ]
        payload = {
            "user_prompt": prompt,
            "declared_skill_profile": {
                "name": profile.name,
                "description": profile.description,
                "use_when": profile.use_when,
                "summary": profile.summary,
                "declared_capabilities": profile.declared_capabilities,
                "declared_outputs": profile.declared_outputs,
            },
            "instruction_graph": {
                "nodes": instruction_nodes,
                "edges": instruction_edges,
            },
        }
        valid_paths = {
            tuple(
                node_id
                for node_id in path
                if self._node_is_instruction_action(ueg.node_by_id(node_id))
            )
            for path in self._enumerate_paths_from_all_entries(ueg)
        }
        valid_paths.discard(())
        valid_node_ids = {
            node.node_id
            for node in ueg.nodes
            if self._node_is_instruction_action(node)
        }
        guarded_allowed_node_ids = {
            node.node_id
            for node in ueg.nodes
            if self._guarded_branch_disposition(node) == "allowed"
        }
        guarded_safe_node_ids = {
            node.node_id
            for node in ueg.nodes
            if self._guarded_branch_disposition(node) == "safe"
        }
        contract = JSONResponseContract(
            required_fields=("execution_plan_node_ids", "notes"),
            consistency_checks=(
                lambda response: self._validate_llm_plan_response(
                    response,
                    valid_node_ids=valid_node_ids,
                    valid_paths=valid_paths,
                    guarded_allowed_node_ids=guarded_allowed_node_ids,
                    guarded_safe_node_ids=guarded_safe_node_ids,
                ),
            ),
        )
        try:
            response = complete_validated_json(
                self.llm_client,
                system_prompt=self.prompt_loader.load(self.prompt_asset),
                user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
                schema_name="module2_task_conditioned_execution",
                contract=contract,
                max_attempts=MAX_VALIDATED_LLM_ATTEMPTS,
            )
        except (ResponseValidationError, RuntimeError):
            return None

        node_ids = [str(node_id) for node_id in response["execution_plan_node_ids"]]
        notes = response.get("notes")
        return InstructionExecutionPlan(
            node_ids=node_ids,
            strategy="llm_validated_path",
            notes=[str(note) for note in notes] if isinstance(notes, list) else [],
        )

    def _validate_llm_plan_response(
        self,
        response: dict[str, Any],
        *,
        valid_node_ids: set[str],
        valid_paths: set[tuple[str, ...]],
        guarded_allowed_node_ids: set[str],
        guarded_safe_node_ids: set[str],
    ) -> str | None:
        if set(response) != {"execution_plan_node_ids", "notes"}:
            return (
                "response must contain exactly execution_plan_node_ids and notes"
            )
        node_ids = response.get("execution_plan_node_ids")
        if (
            not isinstance(node_ids, list)
            or not node_ids
            or any(not isinstance(node_id, str) or not node_id for node_id in node_ids)
        ):
            return "execution_plan_node_ids must be a non-empty string array"
        if len(node_ids) != len(set(node_ids)):
            return "execution_plan_node_ids must not contain duplicates"
        unsupported = [
            node_id for node_id in node_ids if node_id not in valid_node_ids
        ]
        if unsupported:
            return (
                "execution_plan_node_ids contains nodes absent from the grounded "
                f"instruction graph: {unsupported!r}"
            )
        if valid_paths and tuple(node_ids) not in valid_paths:
            return (
                "execution_plan_node_ids must exactly match one grounded "
                "instruction-graph path"
            )
        if (
            guarded_allowed_node_ids.intersection(node_ids)
            and guarded_safe_node_ids.intersection(node_ids)
        ):
            return (
                "execution_plan_node_ids cannot execute both guarded allowed "
                "and safe/default units"
            )
        notes = response.get("notes")
        if not isinstance(notes, list) or any(
            not isinstance(note, str) or not note.strip() for note in notes
        ):
            return "notes must be an array of non-empty strings"
        return None

    def _enumerate_paths_from_all_entries(
        self,
        ueg: UnifiedExecutionGraph,
    ) -> list[list[str]]:
        entries = [
            node
            for node in ueg.nodes
            if node.layer == "instruction" and node.node_type == "ENTRY"
        ]
        paths: list[list[str]] = []
        for entry in entries:
            for path in self._enumerate_paths(ueg, entry.node_id):
                if path not in paths:
                    paths.append(path)
        return paths

    def _plan_with_fallback(
        self,
        *,
        profile: SkillProfile,
        ueg: UnifiedExecutionGraph,
        prompt: str,
    ) -> InstructionExecutionPlan:
        entry_node = next(
            (
                node
                for node in ueg.nodes
                if node.layer == "instruction" and node.node_type == "ENTRY"
            ),
            None,
        )
        if entry_node is None:
            node_ids = self._fail_closed_full_plan(ueg)
            return InstructionExecutionPlan(
                node_ids=node_ids,
                strategy="fallback_full_plan",
                notes=[
                    "No instruction entry node was available. Guarded allowed "
                    "units were excluded from the fallback plan."
                ],
            )

        paths = self._enumerate_paths(ueg, entry_node.node_id)
        if not paths:
            node_ids = self._fail_closed_full_plan(ueg)
            return InstructionExecutionPlan(
                node_ids=node_ids,
                strategy="fallback_full_plan",
                notes=[
                    "No instruction path could be enumerated. Guarded allowed "
                    "units were excluded from the fallback plan."
                ],
            )

        guarded_allowed_node_ids = {
            node.node_id
            for node in ueg.nodes
            if self._guarded_branch_disposition(node) == "allowed"
        }
        guarded_safe_node_ids = {
            node.node_id
            for node in ueg.nodes
            if self._guarded_branch_disposition(node) == "safe"
        }
        safe_only_paths = [
            path
            for path in paths
            if guarded_safe_node_ids.intersection(path)
            and not guarded_allowed_node_ids.intersection(path)
        ]
        if guarded_allowed_node_ids and safe_only_paths:
            paths = safe_only_paths

        prompt_tokens = self._tokenize(" ".join([prompt, profile.summary, profile.description, profile.use_when]))
        scored_paths: list[tuple[float, list[str]]] = []
        for path in paths:
            score = self._score_path(ueg, path, prompt_tokens)
            scored_paths.append((score, path))
        scored_paths.sort(key=lambda item: (-item[0], len(item[1])))
        best_path = scored_paths[0][1]
        materialized = [
            node_id
            for node_id in best_path
            if self._node_is_instruction_action(ueg.node_by_id(node_id))
            and node_id not in guarded_allowed_node_ids
        ]
        if not materialized:
            materialized = self._fail_closed_full_plan(ueg)
        fail_closed = bool(guarded_allowed_node_ids)
        return InstructionExecutionPlan(
            node_ids=materialized,
            strategy=(
                "fallback_safe_guarded_path"
                if fail_closed
                else "fallback_path_scoring"
            ),
            notes=[
                (
                    "Semantic planning was unavailable, so guarded allowed "
                    "units were excluded and the safe/default path was selected."
                    if fail_closed
                    else (
                        "The task-conditioned execution plan was selected by "
                        "scoring instruction-layer paths against the user prompt."
                    )
                ),
            ],
        )

    def _enumerate_paths(self, ueg: UnifiedExecutionGraph, entry_node_id: str) -> list[list[str]]:
        output: list[list[str]] = []
        self._dfs(
            ueg=ueg,
            current_id=entry_node_id,
            current_path=[entry_node_id],
            visited={entry_node_id},
            output=output,
        )
        return output

    def _dfs(
        self,
        *,
        ueg: UnifiedExecutionGraph,
        current_id: str,
        current_path: list[str],
        visited: set[str],
        output: list[list[str]],
    ) -> None:
        if len(output) >= self.max_paths or len(current_path) > self.max_depth:
            return

        current_node = ueg.node_by_id(current_id)
        if current_node is None:
            return
        if current_node.layer == "instruction" and current_node.node_type == "EXIT":
            output.append(current_path)
            return

        successors = [
            successor_id
            for successor_id in ueg.successor_ids(current_id, ALLOWED_EDGE_TYPES)
            if self._node_is_instruction(ueg.node_by_id(successor_id))
        ]
        if not successors:
            output.append(current_path)
            return

        for successor_id in successors:
            if successor_id in visited:
                continue
            self._dfs(
                ueg=ueg,
                current_id=successor_id,
                current_path=current_path + [successor_id],
                visited=visited | {successor_id},
                output=output,
            )
            if len(output) >= self.max_paths:
                return

    def _score_path(self, ueg: UnifiedExecutionGraph, path: list[str], prompt_tokens: set[str]) -> float:
        score = 0.0
        for node_id in path:
            node = ueg.node_by_id(node_id)
            if node is None or node.node_type in {"ENTRY", "EXIT"}:
                continue
            node_tokens = self._tokenize(" ".join(filter(None, [node.summary, node.raw_text or ""])))
            overlap = len(prompt_tokens & node_tokens)
            score += 1.0 + overlap * 3.0
            if node.node_type == "INSTR_PREDICATE":
                score += 0.25
            if node.attributes.get("invoked_scripts"):
                score += 0.5
            disposition = self._guarded_branch_disposition(node)
            if disposition == "safe":
                score += 100.0
            elif disposition == "allowed":
                score -= 100.0
            raw_text = (node.raw_text or "").lower()
            if raw_text.startswith("always "):
                score += 0.4
        return score

    def _all_instruction_nodes(self, ueg: UnifiedExecutionGraph) -> list[str]:
        ordered: list[tuple[int, str]] = []
        for node in ueg.nodes:
            if not self._node_is_instruction_action(node):
                continue
            line_number = node.source_range.start_line if node.source_range is not None else 10**9
            ordered.append((line_number, node.node_id))
        return [node_id for _, node_id in sorted(ordered)]

    def _fail_closed_full_plan(
        self,
        ueg: UnifiedExecutionGraph,
    ) -> list[str]:
        return [
            node_id
            for node_id in self._all_instruction_nodes(ueg)
            if self._guarded_branch_disposition(
                ueg.node_by_id(node_id)
            )
            != "allowed"
        ]

    def _guarded_branch_disposition(
        self,
        node: UEGNode | None,
    ) -> str | None:
        if node is None or not self._node_is_instruction_action(node):
            return None
        explicit = str(node.attributes.get("branch_disposition") or "").lower()
        if explicit in {"allowed", "authorized", "authorized_and_necessary"}:
            return "allowed"
        if explicit in {"safe", "default_safe"}:
            return "safe"
        text = " ".join(
            filter(
                None,
                [
                    node.summary,
                    node.raw_text or "",
                    " ".join(
                        str(value)
                        for value in node.attributes.get(
                            "invoked_scripts",
                            [],
                        )
                    ),
                ],
            )
        ).lower()
        if (
            "__task_allowed" in text
            or "authorized-and-necessary branch" in text
        ):
            return "allowed"
        if (
            "__default_safe" in text
            or "safe/default" in text
            or "safe default" in text
        ):
            return "safe"
        return None

    def _tokenize(self, text: str) -> set[str]:
        return {match.group(0) for match in TOKEN_RE.finditer(text.lower())}

    def _node_is_instruction(self, node: UEGNode | None) -> bool:
        return node is not None and node.layer == "instruction"

    def _node_is_instruction_action(self, node: UEGNode | None) -> bool:
        return self._node_is_instruction(node) and node.node_type not in {"ENTRY", "EXIT"}
