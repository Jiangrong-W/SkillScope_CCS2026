from __future__ import annotations

from typing import Any

from skillscope.common.models import CandidateAction, LegitimateActionChain, UEGNode, UnifiedExecutionGraph


class LegitimateChainExtractor:
    """Enumerate task paths that reach, rather than bypass, the candidate action."""

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
    NON_MATERIALIZED_NODE_TYPES = {"ENTRY", "EXIT", "CODE_ENTRY", "CODE_RETURN"}

    def __init__(self, max_chains: int = 64, max_depth: int = 96, max_suffixes_per_prefix: int = 4) -> None:
        self.max_chains = max(1, max_chains)
        self.max_depth = max(2, max_depth)
        self.max_suffixes_per_prefix = max(1, max_suffixes_per_prefix)
        self.last_coverage_report: dict[str, Any] = {}

    def extract(
        self,
        ueg: UnifiedExecutionGraph,
        candidate: CandidateAction,
        excluded_node_ids: set[str] | None = None,
    ) -> list[LegitimateActionChain]:
        self._coverage_flags = {
            "prefix_limit_reached": False,
            "suffix_limit_reached": False,
            "depth_limit_reached": False,
            "chain_limit_reached": False,
        }
        excluded = set(excluded_node_ids or set())
        excluded.discard(candidate.node_id)
        entries = [
            node.node_id
            for node in ueg.nodes
            if node.node_type == "ENTRY" and node.node_id not in excluded
        ]
        if not entries:
            entries = [
                node.node_id
                for node in ueg.nodes
                if node.node_type == "CODE_ENTRY" and node.node_id not in excluded
            ]

        prefixes: list[list[str]] = []
        for entry_id in entries:
            self._dfs_to_candidate(
                ueg=ueg,
                current_id=entry_id,
                candidate_node_id=candidate.node_id,
                excluded_node_ids=excluded,
                current_path=[entry_id],
                visited={entry_id},
                output=prefixes,
            )
            if len(prefixes) >= self.max_chains:
                break

        if not prefixes:
            # Module 2 instantiates tasks only from executable graph paths that
            # reach the candidate from an instruction entry. Module 1 context
            # and data-dependency edges are useful classification evidence, but
            # they cannot manufacture control reachability for an isolated or
            # uninvoked code action.
            self._finalize_coverage_report(
                candidate=candidate,
                prefix_count=0,
                chain_count=0,
            )
            return []

        chains: list[LegitimateActionChain] = []
        seen_materialized_paths: set[tuple[str, ...]] = set()
        for prefix in prefixes:
            suffixes = self._suffixes_from_candidate(ueg, candidate.node_id, excluded)
            if not suffixes:
                suffixes = [[candidate.node_id]]
            for suffix in suffixes[: self.max_suffixes_per_prefix]:
                full_path = prefix + suffix[1:]
                chain = self._materialize_chain(
                    ueg=ueg,
                    path_node_ids=full_path,
                    candidate=candidate,
                    chain_index=len(chains),
                    used_fallback=False,
                )
                path_key = tuple(chain.node_ids)
                if not chain.reaches_candidate or path_key in seen_materialized_paths:
                    continue
                chains.append(chain)
                seen_materialized_paths.add(path_key)
                if len(chains) >= self.max_chains:
                    self._coverage_flags["chain_limit_reached"] = True
                    self._finalize_coverage_report(
                        candidate=candidate,
                        prefix_count=len(prefixes),
                        chain_count=len(chains),
                    )
                    return chains
        self._finalize_coverage_report(
            candidate=candidate,
            prefix_count=len(prefixes),
            chain_count=len(chains),
        )
        return chains

    def _finalize_coverage_report(
        self,
        *,
        candidate: CandidateAction,
        prefix_count: int,
        chain_count: int,
    ) -> None:
        flags = dict(self._coverage_flags)
        self.last_coverage_report = {
            "candidate_id": candidate.candidate_id,
            "candidate_node_id": candidate.node_id,
            "prefix_count": prefix_count,
            "materialized_chain_count": chain_count,
            "max_chains": self.max_chains,
            "max_depth": self.max_depth,
            "max_suffixes_per_prefix": self.max_suffixes_per_prefix,
            **flags,
            "truncated": any(flags.values()),
        }

    def _dfs_to_candidate(
        self,
        *,
        ueg: UnifiedExecutionGraph,
        current_id: str,
        candidate_node_id: str,
        excluded_node_ids: set[str],
        current_path: list[str],
        visited: set[str],
        output: list[list[str]],
    ) -> None:
        if len(output) >= self.max_chains:
            self._coverage_flags["prefix_limit_reached"] = True
            return
        if len(current_path) > self.max_depth:
            self._coverage_flags["depth_limit_reached"] = True
            return
        if current_id == candidate_node_id:
            output.append(current_path)
            return
        for successor_id in ueg.successor_ids(current_id, self.ALLOWED_EDGE_TYPES):
            if successor_id in excluded_node_ids or successor_id in visited:
                continue
            self._dfs_to_candidate(
                ueg=ueg,
                current_id=successor_id,
                candidate_node_id=candidate_node_id,
                excluded_node_ids=excluded_node_ids,
                current_path=current_path + [successor_id],
                visited=visited | {successor_id},
                output=output,
            )
            if len(output) >= self.max_chains:
                self._coverage_flags["prefix_limit_reached"] = True
                return

    def _suffixes_from_candidate(
        self,
        ueg: UnifiedExecutionGraph,
        candidate_node_id: str,
        excluded_node_ids: set[str],
    ) -> list[list[str]]:
        output: list[list[str]] = []
        self._dfs_to_terminal(
            ueg=ueg,
            current_id=candidate_node_id,
            excluded_node_ids=excluded_node_ids,
            current_path=[candidate_node_id],
            visited={candidate_node_id},
            output=output,
        )
        return output

    def _dfs_to_terminal(
        self,
        *,
        ueg: UnifiedExecutionGraph,
        current_id: str,
        excluded_node_ids: set[str],
        current_path: list[str],
        visited: set[str],
        output: list[list[str]],
    ) -> None:
        if len(output) >= self.max_suffixes_per_prefix:
            self._coverage_flags["suffix_limit_reached"] = True
            return
        if len(current_path) > self.max_depth:
            self._coverage_flags["depth_limit_reached"] = True
            return
        node = ueg.node_by_id(current_id)
        if node is None:
            return
        successors = [
            node_id
            for node_id in ueg.successor_ids(current_id, self.ALLOWED_EDGE_TYPES)
            if node_id not in excluded_node_ids and node_id not in visited
        ]
        if node.node_type == "EXIT" or not successors:
            output.append(current_path)
            return
        for successor_id in successors:
            self._dfs_to_terminal(
                ueg=ueg,
                current_id=successor_id,
                excluded_node_ids=excluded_node_ids,
                current_path=current_path + [successor_id],
                visited=visited | {successor_id},
                output=output,
            )
            if len(output) >= self.max_suffixes_per_prefix:
                self._coverage_flags["suffix_limit_reached"] = True
                return

    def _materialize_chain(
        self,
        *,
        ueg: UnifiedExecutionGraph,
        path_node_ids: list[str],
        candidate: CandidateAction,
        chain_index: int,
        used_fallback: bool,
    ) -> LegitimateActionChain:
        node_ids: list[str] = []
        summaries: list[str] = []
        predicate_context: list[str] = []
        for node_id in path_node_ids:
            node = ueg.node_by_id(node_id)
            if node is None or node.node_type in self.NON_MATERIALIZED_NODE_TYPES:
                continue
            node_ids.append(node.node_id)
            summaries.append(node.summary)
            if node.node_type == "INSTR_PREDICATE":
                predicate_context.append(node.summary)

        reaches_candidate = candidate.node_id in node_ids
        candidate_position = node_ids.index(candidate.node_id) if reaches_candidate else None
        notes = ["Candidate-reaching traversal over the unified execution graph."]
        if used_fallback:
            notes.append("Graph reachability was incomplete; Module 1 context nodes were used as a conservative fallback.")
        if predicate_context:
            notes.append("The task path preserves instruction predicate context.")
        if any(self._is_code_node(ueg.node_by_id(node_id)) for node_id in path_node_ids):
            notes.append("The task path crosses into code-level execution.")
        return LegitimateActionChain(
            chain_id=f"{candidate.candidate_id}-chain-{chain_index + 1:03d}",
            candidate_id=candidate.candidate_id,
            node_ids=node_ids,
            summaries=summaries,
            reaches_candidate=reaches_candidate,
            candidate_position=candidate_position,
            predicate_context=predicate_context,
            notes=notes,
        )

    def _is_code_node(self, node: UEGNode | None) -> bool:
        return node is not None and node.layer == "code"
