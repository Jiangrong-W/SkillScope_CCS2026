from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from inspect import Parameter, signature

from skillscope.common.models import CandidateAction, SkillProfile, UnifiedExecutionGraph
from skillscope.common.privilege import privilege_type_for_action

from .action_consistency_classifier import ActionConsistencyClassifier


@dataclass(slots=True)
class _ClassificationWorkItem:
    node_index: int
    node_id: str
    upstream_chain: list[str]
    downstream_chain: list[str]
    predicate_context: list[str]
    context_node_ids: list[str]


class CandidateExtractor:
    def __init__(
        self,
        classifier: ActionConsistencyClassifier | None = None,
        *,
        max_workers: int = 4,
    ) -> None:
        self.classifier = classifier or ActionConsistencyClassifier()
        self.max_workers = max(1, max_workers)

    def extract(self, profile: SkillProfile, ueg: UnifiedExecutionGraph) -> list[CandidateAction]:
        work_items: list[_ClassificationWorkItem] = []
        for node_index, node in enumerate(ueg.nodes):
            if self._is_non_action_node(node.node_type):
                continue
            upstream_chain, downstream_chain, predicate_context, context_node_ids = self._extract_graph_context(
                ueg,
                node.node_id,
            )
            work_items.append(
                _ClassificationWorkItem(
                    node_index=node_index,
                    node_id=node.node_id,
                    upstream_chain=upstream_chain,
                    downstream_chain=downstream_chain,
                    predicate_context=predicate_context,
                    context_node_ids=context_node_ids,
                )
            )

        if not work_items:
            ueg.metadata["action_consistency_classifications"] = []
            return []

        def classify_work_item(work_item: _ClassificationWorkItem):
            node = ueg.node_by_id(work_item.node_id)
            if node is None:
                return work_item, None, None
            assessment = self._classify_with_context(profile, node, work_item)
            return work_item, node, assessment

        if self.max_workers == 1 or len(work_items) == 1:
            assessed_items = [classify_work_item(work_item) for work_item in work_items]
        else:
            worker_count = min(self.max_workers, len(work_items))
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="skillscope-m1") as executor:
                assessed_items = list(executor.map(classify_work_item, work_items))

        candidates: list[CandidateAction] = []
        classification_audit: list[dict[str, object]] = []
        for work_item, node, assessment in assessed_items:
            if node is None or assessment is None:
                classification_audit.append(
                    {
                        "node_id": work_item.node_id,
                        "label": None,
                        "strategy": "missing_assessment",
                        "retained": False,
                        "retained_due_to_ambiguity": False,
                        "ambiguity_kind": None,
                        "privilege_type": None,
                    }
                )
                continue
            privilege_type = privilege_type_for_action(
                node.operation_type,
                node.risk_tags,
            )
            # A profile-compatible privilege-relevant action can still be
            # authorized/necessary for one concrete task and over-privileged
            # for another.  Module 1 therefore must not filter it merely for
            # being globally related to the Skill profile.
            profile_compatible_task_ambiguity = (
                assessment.label == "related" and privilege_type is not None
            )
            retained_due_to_ambiguity = (
                assessment.label == "ambiguous"
                or profile_compatible_task_ambiguity
            )
            ambiguity_kind = assessment.ambiguity_kind
            if profile_compatible_task_ambiguity:
                ambiguity_kind = "task_ambiguous"
            retained = bool(
                assessment.suspicious or retained_due_to_ambiguity
            )
            classification_audit.append(
                {
                    "node_id": work_item.node_id,
                    "label": assessment.label,
                    "strategy": assessment.strategy,
                    "retained": retained,
                    "retained_due_to_ambiguity": retained_due_to_ambiguity,
                    "ambiguity_kind": ambiguity_kind,
                    "privilege_type": privilege_type,
                }
            )
            if not retained:
                continue

            candidates.append(
                CandidateAction(
                    candidate_id=f"candidate-{len(candidates) + 1:03d}",
                    node_id=work_item.node_id,
                    layer=node.layer,
                    summary=node.summary,
                    source_file=node.source_file,
                    risk_tags=node.risk_tags,
                    reason=assessment.reason,
                    confidence=assessment.confidence,
                    classification_label=assessment.label,
                    upstream_action_chain=work_item.upstream_chain,
                    downstream_action_chain=work_item.downstream_chain,
                    predicate_context=work_item.predicate_context,
                    context_node_ids=work_item.context_node_ids,
                    retained_due_to_ambiguity=retained_due_to_ambiguity,
                    ambiguity_kind=ambiguity_kind,
                    classifier_input=assessment.input_payload,
                    classifier_strategy=assessment.strategy,
                    privilege_type=privilege_type,
                    privilege_relevant=privilege_type is not None,
                )
            )

        # Persist every classification, including profile-aligned actions that
        # are filtered out and therefore never become CandidateAction objects.
        # This makes module-level LLM validation auditable without changing the
        # high-recall candidate set consumed by downstream stages.
        ueg.metadata["action_consistency_classifications"] = (
            classification_audit
        )
        return candidates

    _CONTEXT_EDGE_TYPES = {
        "SEQUENTIAL",
        "CALLS",
        "CALLS_LOCAL",
        "DATA_DEP",
        "SEMANTIC_DEP",
        "RETURNS_TO",
        "RETURNS_LOCAL",
        "CONDITIONAL_TRUE",
        "CONDITIONAL_FALSE",
    }
    _STRUCTURAL_NODE_TYPES = {"ENTRY", "EXIT", "CODE_ENTRY", "CODE_RETURN"}

    def _classify_with_context(self, profile, node, work_item: _ClassificationWorkItem):
        classify = self.classifier.classify
        parameters = signature(classify).parameters.values()
        supports_extended_context = any(parameter.kind == Parameter.VAR_POSITIONAL for parameter in parameters) or len(
            tuple(parameters)
        ) >= 6
        if not supports_extended_context:
            return classify(profile, node, work_item.upstream_chain)
        return classify(
            profile,
            node,
            work_item.upstream_chain,
            work_item.downstream_chain,
            work_item.predicate_context,
            work_item.context_node_ids,
        )

    def _extract_graph_context(
        self,
        ueg: UnifiedExecutionGraph,
        node_id: str,
        *,
        action_limit: int = 6,
        traversal_limit: int = 64,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        upstream_nodes = self._traverse_context(
            ueg,
            node_id,
            direction="backward",
            action_limit=action_limit,
            traversal_limit=traversal_limit,
        )
        downstream_nodes = self._traverse_context(
            ueg,
            node_id,
            direction="forward",
            action_limit=action_limit,
            traversal_limit=traversal_limit,
        )

        upstream_actions = [item for item in upstream_nodes if not self._is_predicate_node(item[2].node_type)]
        downstream_actions = [item for item in downstream_nodes if not self._is_predicate_node(item[2].node_type)]
        predicate_nodes = [
            item
            for item in (*upstream_nodes, *downstream_nodes)
            if self._is_predicate_node(item[2].node_type)
        ]

        # Backward traversal is discovered nearest-first. Present it in execution
        # order (farthest-to-nearest), while keeping branch ordering deterministic.
        upstream_actions.sort(key=lambda item: (-item[0], item[1]))
        downstream_actions.sort(key=lambda item: (item[0], item[1]))
        predicate_nodes.sort(key=lambda item: (item[0], item[1]))

        upstream_chain = [item[2].summary for item in upstream_actions[:action_limit]]
        downstream_chain = [item[2].summary for item in downstream_actions[:action_limit]]

        predicate_context: list[str] = []
        context_node_ids: list[str] = []
        seen_predicates: set[str] = set()
        seen_context_ids: set[str] = set()
        for _, _, context_node in (*upstream_actions, *downstream_actions, *predicate_nodes):
            if context_node.node_id not in seen_context_ids:
                seen_context_ids.add(context_node.node_id)
                context_node_ids.append(context_node.node_id)
        for _, _, predicate_node in predicate_nodes:
            if predicate_node.node_id in seen_predicates:
                continue
            seen_predicates.add(predicate_node.node_id)
            predicate_context.append(predicate_node.summary)

        return upstream_chain, downstream_chain, predicate_context, context_node_ids

    def _traverse_context(
        self,
        ueg: UnifiedExecutionGraph,
        node_id: str,
        *,
        direction: str,
        action_limit: int,
        traversal_limit: int,
    ):
        node_order = {node.node_id: index for index, node in enumerate(ueg.nodes)}
        queue: list[tuple[str, int]] = [(node_id, 0)]
        visited: set[str] = {node_id}
        collected = []
        action_count = 0

        while queue and len(visited) <= traversal_limit:
            current_id, distance = queue.pop(0)
            adjacent_ids = (
                ueg.predecessor_ids(current_id, self._CONTEXT_EDGE_TYPES)
                if direction == "backward"
                else ueg.successor_ids(current_id, self._CONTEXT_EDGE_TYPES)
            )
            for adjacent_id in sorted(set(adjacent_ids), key=lambda item: node_order.get(item, len(node_order))):
                if adjacent_id in visited:
                    continue
                visited.add(adjacent_id)
                adjacent = ueg.node_by_id(adjacent_id)
                if adjacent is None:
                    continue
                next_distance = distance + 1
                queue.append((adjacent_id, next_distance))
                if adjacent.node_type in self._STRUCTURAL_NODE_TYPES:
                    continue
                collected.append((next_distance, node_order.get(adjacent_id, len(node_order)), adjacent))
                if not self._is_predicate_node(adjacent.node_type):
                    action_count += 1
            if action_count >= action_limit and not any(
                self._is_predicate_node(item[2].node_type) for item in collected
            ):
                # Continue just far enough to cross structural nodes and find a
                # nearby guard, but do not fan out through an unbounded graph.
                if queue and queue[0][1] > distance + 1:
                    break

        return collected

    def _extract_upstream_chain(self, ueg: UnifiedExecutionGraph, node_id: str, limit: int = 6) -> list[str]:
        upstream, _, _, _ = self._extract_graph_context(ueg, node_id, action_limit=limit)
        return upstream

    def _is_predicate_node(self, node_type: str) -> bool:
        return node_type.upper().endswith("PREDICATE")

    def _is_non_action_node(self, node_type: str) -> bool:
        return node_type in self._STRUCTURAL_NODE_TYPES or self._is_predicate_node(node_type)
