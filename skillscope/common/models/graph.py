from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SourceRange:
    start_line: int
    end_line: int
    start_column: int | None = None
    end_column: int | None = None


@dataclass(slots=True)
class UEGNode:
    node_id: str
    layer: str
    node_type: str
    summary: str
    source_file: str | None = None
    source_range: SourceRange | None = None
    raw_text: str | None = None
    operation_type: str | None = None
    object_ref: str | None = None
    risk_tags: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UEGEdge:
    source: str
    target: str
    edge_type: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionGraph:
    graph_id: str
    layer: str
    nodes: list[UEGNode] = field(default_factory=list)
    edges: list[UEGEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: UEGNode) -> None:
        self.nodes.append(node)

    def add_edge(self, edge: UEGEdge) -> None:
        self.edges.append(edge)

    def node_by_id(self, node_id: str) -> UEGNode | None:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def predecessor_ids(self, node_id: str, edge_types: set[str] | None = None) -> list[str]:
        predecessors: list[str] = []
        for edge in self.edges:
            if edge.target != node_id:
                continue
            if edge_types is not None and edge.edge_type not in edge_types:
                continue
            predecessors.append(edge.source)
        return predecessors

    def successor_ids(self, node_id: str, edge_types: set[str] | None = None) -> list[str]:
        successors: list[str] = []
        for edge in self.edges:
            if edge.source != node_id:
                continue
            if edge_types is not None and edge.edge_type not in edge_types:
                continue
            successors.append(edge.target)
        return successors


@dataclass(slots=True)
class UnifiedExecutionGraph:
    skill_id: str
    nodes: list[UEGNode] = field(default_factory=list)
    edges: list[UEGEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: UEGNode) -> None:
        self.nodes.append(node)

    def add_edge(self, edge: UEGEdge) -> None:
        self.edges.append(edge)

    def node_by_id(self, node_id: str) -> UEGNode | None:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def predecessor_ids(self, node_id: str, edge_types: set[str] | None = None) -> list[str]:
        predecessors: list[str] = []
        for edge in self.edges:
            if edge.target != node_id:
                continue
            if edge_types is not None and edge.edge_type not in edge_types:
                continue
            predecessors.append(edge.source)
        return predecessors

    def successor_ids(self, node_id: str, edge_types: set[str] | None = None) -> list[str]:
        successors: list[str] = []
        for edge in self.edges:
            if edge.source != node_id:
                continue
            if edge_types is not None and edge.edge_type not in edge_types:
                continue
            successors.append(edge.target)
        return successors
