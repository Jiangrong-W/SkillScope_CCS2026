from __future__ import annotations

from pathlib import Path

from skillscope.common.models import ActionGraph, SkillBundle, UEGEdge, UnifiedExecutionGraph


class UEGComposer:
    def compose(self, bundle: SkillBundle, instruction_graph: ActionGraph, code_graphs: list[ActionGraph]) -> UnifiedExecutionGraph:
        ueg = UnifiedExecutionGraph(skill_id=bundle.bundle_id)
        ueg.metadata["instruction_graph_id"] = instruction_graph.graph_id
        ueg.metadata["code_graph_ids"] = [graph.graph_id for graph in code_graphs]
        ueg.metadata["composition_strategy"] = {
            "cross_layer_links": ["instruction_action_to_code_entry", "code_return_to_downstream_instruction"],
            "bundle_id": bundle.bundle_id,
        }
        ueg.metadata["edge_semantics"] = {
            "control": ["SEQUENTIAL", "CONDITIONAL_TRUE", "CONDITIONAL_FALSE", "SEMANTIC_DEP"],
            "data": ["DATA_DEP"],
            "call": ["CALLS", "CALLS_LOCAL"],
            "return": ["RETURNS_TO", "RETURNS_LOCAL"],
        }

        for node in instruction_graph.nodes:
            self._ensure_node_evidence(node, instruction_graph.graph_id)
            ueg.add_node(node)
        for edge in instruction_graph.edges:
            self._ensure_edge_semantics(edge)
            ueg.add_edge(edge)

        for graph in code_graphs:
            for node in graph.nodes:
                self._ensure_node_evidence(node, graph.graph_id)
                ueg.add_node(node)
            for edge in graph.edges:
                self._ensure_edge_semantics(edge)
                ueg.add_edge(edge)

        entry_map = {graph.metadata.get("relative_path"): graph.metadata.get("entry_node_id") for graph in code_graphs}
        return_map = {graph.metadata.get("relative_path"): graph.metadata.get("return_node_id") for graph in code_graphs}
        downstream_map: dict[str, list[str]] = {}
        for edge in instruction_graph.edges:
            downstream_map.setdefault(edge.source, []).append(edge.target)

        for node in instruction_graph.nodes:
            invoked_scripts = list(node.attributes.get("invoked_scripts", []))
            for relative_path in dict.fromkeys(invoked_scripts):
                relative_path = self._resolve_invoked_path(relative_path, entry_map)
                entry_id = entry_map.get(relative_path)
                return_id = return_map.get(relative_path)
                if entry_id:
                    self._add_edge_once(
                        ueg,
                        UEGEdge(
                            source=node.node_id,
                            target=entry_id,
                            edge_type="CALLS",
                            attributes={
                                "semantic_family": "call",
                                "invoked_resource": relative_path,
                                "provenance": {
                                    "instruction_node_id": node.node_id,
                                    "source_file": node.source_file,
                                },
                            },
                        ),
                    )
                if return_id:
                    for downstream_id in downstream_map.get(node.node_id, []):
                        self._add_edge_once(
                            ueg,
                            UEGEdge(
                                source=return_id,
                                target=downstream_id,
                                edge_type="RETURNS_TO",
                                attributes={
                                    "semantic_family": "return",
                                    "invoked_resource": relative_path,
                                    "instruction_node_id": node.node_id,
                                },
                            ),
                        )

        return ueg

    def _resolve_invoked_path(
        self,
        relative_path: str,
        entry_map: dict[object, object],
    ) -> str:
        normalized = str(relative_path).replace("\\", "/").removeprefix("./")
        if normalized in entry_map:
            return normalized
        matches = [
            str(path)
            for path in entry_map
            if path and Path(str(path)).name == Path(normalized).name
        ]
        if len(matches) == 1:
            return matches[0]
        return normalized

    def _ensure_node_evidence(self, node, graph_id: str) -> None:
        if node.object_ref is None:
            for key in ("object_ref", "call_name", "parameter_name", "invoked_resource"):
                value = node.attributes.get(key)
                if value:
                    node.object_ref = str(value)[:160]
                    break
        if node.object_ref is None and node.raw_text:
            node.object_ref = " ".join(node.raw_text.split())[:160]
        if (
            node.object_ref is None
            and node.node_type
            not in {"ENTRY", "EXIT", "CODE_ENTRY", "CODE_RETURN"}
        ):
            node.object_ref = node.summary[:160]

        provenance = node.attributes.get("provenance")
        if isinstance(provenance, dict):
            return
        node.attributes["provenance"] = {
            "origin": node.layer,
            "graph_id": graph_id,
            "source_file": node.source_file,
            "source_range": (
                {
                    "start_line": node.source_range.start_line,
                    "end_line": node.source_range.end_line,
                }
                if node.source_range is not None
                else None
            ),
        }

    def _add_edge_once(self, ueg: UnifiedExecutionGraph, edge: UEGEdge) -> None:
        if any(
            existing.source == edge.source
            and existing.target == edge.target
            and existing.edge_type == edge.edge_type
            for existing in ueg.edges
        ):
            return
        ueg.add_edge(edge)

    def _ensure_edge_semantics(self, edge: UEGEdge) -> None:
        families = {
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
        semantic_family = families.get(edge.edge_type)
        if semantic_family is not None:
            edge.attributes.setdefault("semantic_family", semantic_family)
