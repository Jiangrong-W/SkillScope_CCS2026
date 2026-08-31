from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from skillscope.common.llm import (
    DisabledLLMClient,
    JSONResponseContract,
    MAX_VALIDATED_LLM_ATTEMPTS,
    PromptAssetLoader,
    StructuredLLMClient,
    complete_validated_json,
)
from skillscope.common.models import AblationPlan, RepairItem
from skillscope.common.privilege import privilege_type_for_action
from skillscope.module1_candidate_extraction.code_graph_builder import (
    CodeGraphBuilder,
)
from skillscope.module2_action_necessity_validation.candidate_ablation import (
    CandidateAblation,
)


class PrivilegeSemanticAnalyzer:
    """Prove that a projected unit removes actions, not only source text.

    The LLM projection path is allowed to reorganize code, but it must not
    replace a removed operation with an equivalent API (for example,
    ``requests.post`` with ``requests.put``), a different command launcher, or
    an opaque new call.  This analyzer reuses Module 1's language-specific
    graph construction and compares both privilege semantics and the complete
    executable-action multiset against ``original - blocked targets``.

    A multiset is intentional: two legitimate calls may have the same
    semantics.  Removing one target must reduce the corresponding count by one
    without deleting its legitimate sibling.
    """

    _SUPPORTED_SUFFIXES = {
        ".py",
        ".sh",
        ".js",
        ".mjs",
        ".cjs",
        ".ts",
        ".mts",
        ".cts",
    }
    _URL_RE = re.compile(r"https?://[^\s'\"`)>,;\]]+", re.IGNORECASE)

    def __init__(self) -> None:
        self.graph_builder = CodeGraphBuilder()

    def compare_projection(
        self,
        *,
        original_source: str,
        projected_source: str,
        suffix: str,
        filename: str,
        blocked_items: list[RepairItem],
    ) -> dict[str, object]:
        original = self._analyze(original_source, suffix, filename)
        projected = self._analyze(projected_source, suffix, filename)
        if original is None or projected is None:
            return {
                "complete": False,
                "analysis_complete": False,
                "reason": (
                    "the original or projected source could not be rebuilt "
                    "into a supported static action graph"
                ),
                "original_privilege_semantics": [],
                "target_privilege_semantics": [],
                "expected_privilege_semantics": [],
                "observed_privilege_semantics": [],
                "unexpected_privilege_semantics": [],
                "missing_privilege_semantics": [],
                "action_multiset_matches": False,
                "raw_targets_absent": False,
            }

        blocked_indices: set[int] = set()
        target_action_semantics: list[dict[str, str]] = []
        target_privilege_semantics: list[dict[str, str]] = []
        unmatched_targets: list[str] = []
        for item in blocked_items:
            matches = self._target_indices(original["actions"], item)
            if not matches:
                unmatched_targets.append(item.repair_id)
                continue
            grounded_matches = matches | self._dependent_receiver_call_indices(
                original["actions"],
                matches,
            )
            blocked_indices.update(grounded_matches)
            for index in sorted(grounded_matches):
                entry = original["actions"][index]
                target_action_semantics.append(dict(entry["action_semantics"]))
                privilege_semantics = entry.get("privilege_semantics")
                if isinstance(privilege_semantics, dict):
                    target_privilege_semantics.append(dict(privilege_semantics))

        expected_actions = [
            dict(entry["action_semantics"])
            for index, entry in enumerate(original["actions"])
            if index not in blocked_indices
        ]
        observed_actions = [
            dict(entry["action_semantics"]) for entry in projected["actions"]
        ]
        original_privilege = [
            dict(entry["privilege_semantics"])
            for entry in original["actions"]
            if isinstance(entry.get("privilege_semantics"), dict)
        ]
        expected_privilege = [
            dict(entry["privilege_semantics"])
            for index, entry in enumerate(original["actions"])
            if index not in blocked_indices
            and isinstance(entry.get("privilege_semantics"), dict)
        ]
        observed_privilege = [
            dict(entry["privilege_semantics"])
            for entry in projected["actions"]
            if isinstance(entry.get("privilege_semantics"), dict)
        ]

        expected_action_counter = self._counter(expected_actions)
        observed_action_counter = self._counter(observed_actions)
        expected_privilege_counter = self._counter(expected_privilege)
        observed_privilege_counter = self._counter(observed_privilege)
        unexpected_privilege = self._expand_counter(
            observed_privilege_counter - expected_privilege_counter
        )
        missing_privilege = self._expand_counter(
            expected_privilege_counter - observed_privilege_counter
        )

        # A target with no privilege signature can still be a grounded action
        # (e.g., a project-specific helper in a unit test).  Exact source-span
        # removal plus the full action-multiset comparison proves that it was
        # removed and that no replacement call was introduced.  In production,
        # recognized network/command/data/state actions additionally enter the
        # privilege-semantic comparison above.
        targets_localized = not unmatched_targets
        # Removing a target can also remove a dependent non-privileged call
        # (e.g., ``dangerous().format()``).  Such loss is checked later by
        # CorePres/GoalSat replay.  Safety requires that the model introduce no
        # executable call that is absent from the grounded remainder.
        action_multiset_matches = not bool(
            observed_action_counter - expected_action_counter
        )
        # Source text can legitimately remain in a function declaration or a
        # comment after its invocation is removed.  Absence is therefore
        # established at the executable-action level rather than by a global
        # substring search.
        raw_targets_absent = action_multiset_matches
        privilege_multiset_matches = (
            expected_privilege_counter == observed_privilege_counter
        )
        complete = bool(
            targets_localized
            and raw_targets_absent
            and action_multiset_matches
            and privilege_multiset_matches
        )
        return {
            "complete": complete,
            "analysis_complete": True,
            "reason": (
                "the projected action graph exactly matches the original "
                "graph with the blocked target actions removed"
                if complete
                else "the projected unit contains missing, residual, or new "
                "actions relative to the grounded static projection"
            ),
            "unmatched_target_repair_ids": unmatched_targets,
            "original_privilege_semantics": self._sorted_payloads(original_privilege),
            "target_action_semantics": self._sorted_payloads(target_action_semantics),
            "target_privilege_semantics": self._sorted_payloads(
                target_privilege_semantics
            ),
            "expected_privilege_semantics": self._sorted_payloads(expected_privilege),
            "observed_privilege_semantics": self._sorted_payloads(observed_privilege),
            "unexpected_privilege_semantics": unexpected_privilege,
            "missing_privilege_semantics": missing_privilege,
            "expected_action_semantics": self._sorted_payloads(expected_actions),
            "observed_action_semantics": self._sorted_payloads(observed_actions),
            "action_multiset_matches": action_multiset_matches,
            "privilege_multiset_matches": privilege_multiset_matches,
            "raw_targets_absent": raw_targets_absent,
        }

    def semantics_for_graph_nodes(
        self,
        nodes: list[object],
        *,
        source_file: str,
    ) -> list[dict[str, str]]:
        """Return privilege semantics for one already-built execution unit."""

        normalized_source = self._normalize_path(source_file)
        semantics = []
        for node in nodes:
            if (
                self._normalize_path(str(getattr(node, "source_file", "") or ""))
                != normalized_source
            ):
                continue
            payload = self._privilege_semantics(node)
            if payload is not None:
                semantics.append(payload)
        return self._sorted_payloads(semantics)

    def _analyze(
        self,
        source: str,
        suffix: str,
        filename: str,
    ) -> dict[str, object] | None:
        normalized_suffix = suffix.casefold()
        if normalized_suffix not in self._SUPPORTED_SUFFIXES:
            return None
        with tempfile.TemporaryDirectory(prefix="skillscope-safe-semantic-") as raw:
            source_path = Path(raw) / (Path(filename).name or f"unit{suffix}")
            source_path.write_text(source, encoding="utf-8")
            relative_path = source_path.name
            if normalized_suffix == ".py":
                graph = self.graph_builder._build_python_graph(
                    "safe-semantic", relative_path, source_path
                )
            elif normalized_suffix == ".sh":
                graph = self.graph_builder._build_shell_graph(
                    "safe-semantic", relative_path, source_path
                )
            else:
                graph = self.graph_builder._build_javascript_typescript_graph(
                    "safe-semantic", relative_path, source_path
                )
        parser_metadata = graph.metadata.get("parser_metadata")
        parser_status = (
            str(parser_metadata.get("status") or "")
            if isinstance(parser_metadata, dict)
            else ""
        )
        if parser_status in {"parse_error", "read_error", "unknown"}:
            return None
        actions: list[dict[str, object]] = []
        for node in graph.nodes:
            if str(node.node_type).upper() != "CODE_ACTION":
                continue
            operation = str(node.operation_type or "").strip().casefold()
            if operation in {"", "parse_error", "unparsed_script"}:
                continue
            privilege_semantics = self._privilege_semantics(node)
            raw_text = self._normalize_text(str(node.raw_text or ""))
            # Compare executable calls and every privilege-relevant action.
            # Data-merge/return bookkeeping can change shape when an exact
            # nested call is neutralized even though all remaining calls are
            # preserved.  The Shell ':' command is the deterministic no-op.
            if privilege_semantics is None and operation not in {
                "call",
                "shell_action",
            }:
                continue
            if operation == "shell_action" and raw_text == ":":
                continue
            actions.append(
                {
                    "node": node,
                    "action_semantics": self._action_semantics(node),
                    "privilege_semantics": privilege_semantics,
                }
            )
        return {"actions": actions, "parser_status": parser_status}

    def _target_indices(
        self,
        actions: list[dict[str, object]],
        item: RepairItem,
    ) -> set[int]:
        raw_target = self._normalize_text(str(item.raw_text or ""))
        exact_raw = {
            index
            for index, entry in enumerate(actions)
            if raw_target
            and self._normalize_text(str(getattr(entry["node"], "raw_text", "") or ""))
            == raw_target
        }
        start_line = item.source_start_line
        end_line = item.source_end_line
        ranged: set[int] = set()
        if isinstance(start_line, int) and isinstance(end_line, int):
            for index, entry in enumerate(actions):
                source_range = getattr(entry["node"], "source_range", None)
                if source_range is None:
                    continue
                if (
                    source_range.end_line < start_line
                    or source_range.start_line > end_line
                ):
                    continue
                if (
                    start_line == end_line == source_range.start_line
                    and isinstance(item.source_start_column, int)
                    and isinstance(item.source_end_column, int)
                    and isinstance(source_range.start_column, int)
                    and isinstance(source_range.end_column, int)
                    and (
                        source_range.end_column <= item.source_start_column
                        or source_range.start_column >= item.source_end_column
                    )
                ):
                    continue
                ranged.add(index)
        if exact_raw and ranged:
            intersection = exact_raw & ranged
            if intersection:
                return intersection
        if exact_raw:
            return exact_raw
        return ranged

    def _dependent_receiver_call_indices(
        self,
        actions: list[dict[str, object]],
        target_indices: set[int],
    ) -> set[int]:
        """Include calls that cannot execute once a receiver call is removed.

        In ``dangerous().safe()``, the graph contains both ``dangerous()`` and
        the outer ``dangerous().safe()`` calls.  Removing the receiver correctly
        removes both; treating the outer call as an unrelated missing privilege
        would reject the deterministic neutralization.  Argument/sibling calls
        such as ``safe(dangerous())`` are intentionally excluded because their
        normalized call expression does not begin with the target receiver.
        """

        target_expressions = {
            self._normalize_text(
                str(actions[index]["action_semantics"].get("expression") or "")
            )
            for index in target_indices
        }
        target_expressions.discard("")
        dependent: set[int] = set()
        for index, entry in enumerate(actions):
            if index in target_indices:
                continue
            expression = self._normalize_text(
                str(entry["action_semantics"].get("expression") or "")
            )
            if any(
                expression.startswith(f"{target}.")
                or expression.startswith(f"{target}[")
                for target in target_expressions
            ):
                dependent.add(index)
        return dependent

    def _action_semantics(self, node: object) -> dict[str, str]:
        operation = str(getattr(node, "operation_type", "") or "").strip().casefold()
        attributes = getattr(node, "attributes", {})
        call_name = (
            str(attributes.get("call_name") or "").strip().casefold()
            if isinstance(attributes, dict)
            else ""
        )
        object_ref = self._normalize_value(str(getattr(node, "object_ref", "") or ""))
        raw_text = self._normalize_text(str(getattr(node, "raw_text", "") or ""))
        # Shell nodes do not expose a call name.  Their normalized command text
        # is part of the executable identity, which prevents a replacement
        # command from being accepted as the removed target.
        executable_identity = call_name
        if not executable_identity and operation in {
            "shell_action",
            "exec_command",
            "network_send",
        }:
            executable_identity = raw_text
        return {
            "operation": operation,
            "executable_identity": executable_identity,
            "object": object_ref,
            "expression": raw_text,
        }

    def _privilege_semantics(self, node: object) -> dict[str, str] | None:
        operation = str(getattr(node, "operation_type", "") or "").strip().casefold()
        risk_tags = list(getattr(node, "risk_tags", []) or [])
        privilege_type = privilege_type_for_action(operation, risk_tags)
        if privilege_type is None:
            return None
        object_ref = self._normalize_value(str(getattr(node, "object_ref", "") or ""))
        raw_text = str(getattr(node, "raw_text", "") or "")
        joined = " ".join(
            [raw_text, object_ref, str(getattr(node, "summary", "") or "")]
        )
        url_match = self._URL_RE.search(joined)
        if privilege_type == "external_data_transmission":
            destination = (
                self._normalize_value(url_match.group(0))
                if url_match is not None
                else self._call_first_argument(raw_text)
                or object_ref
                or "external_unspecified"
            )
            side_effect = "external_transmission"
        elif privilege_type == "command_execution":
            destination = (
                object_ref
                or self._call_first_argument(raw_text)
                or "command_unspecified"
            )
            side_effect = "command_execution"
        elif privilege_type == "persistent_state_modification":
            destination = object_ref or "state_unspecified"
            side_effect = "persistent_state_modification"
        else:
            destination = "none"
            side_effect = "sensitive_data_access"
        return {
            "operation": operation,
            "privilege_type": privilege_type,
            "destination": destination,
            "object": object_ref,
            "side_effect": side_effect,
        }

    def _call_first_argument(self, raw_text: str) -> str:
        candidate = raw_text.strip().rstrip(";")
        try:
            expression = ast.parse(candidate, mode="eval").body
        except SyntaxError:
            expression = None
        if isinstance(expression, ast.Call) and expression.args:
            return self._normalize_value(ast.unparse(expression.args[0]))
        opening = candidate.find("(")
        closing = candidate.rfind(")")
        if opening < 0 or closing <= opening:
            return ""
        arguments = candidate[opening + 1 : closing]
        first = arguments.split(",", 1)[0]
        return self._normalize_value(first)

    def _counter(self, payloads: list[dict[str, str]]) -> Counter[str]:
        return Counter(self._payload_key(payload) for payload in payloads)

    def _expand_counter(self, counter: Counter[str]) -> list[dict[str, str]]:
        payloads: list[dict[str, str]] = []
        for encoded, count in sorted(counter.items()):
            payload = json.loads(encoded)
            payloads.extend(dict(payload) for _ in range(count))
        return payloads

    def _sorted_payloads(self, payloads: list[dict[str, str]]) -> list[dict[str, str]]:
        return [dict(payload) for payload in sorted(payloads, key=self._payload_key)]

    def _payload_key(self, payload: dict[str, str]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _normalize_text(self, value: str) -> str:
        return " ".join(value.strip().rstrip(";").split())

    def _normalize_value(self, value: str) -> str:
        return " ".join(value.strip().split())[:500]

    def _normalize_path(self, value: str) -> str:
        return value.replace("\\", "/").removeprefix("./")


class CodeRewriter:
    _PROJECTION_OUTPUT_KEYS = {
        "repair_id",
        "repair_type",
        "source_file",
        "source_start_line",
        "source_end_line",
        "descriptor_ids",
        "allowed_cluster_keys",
        "blocked_cluster_keys",
        "guard_condition",
        "file_outputs",
        "notes",
    }

    def __init__(
        self,
        *,
        llm_client: StructuredLLMClient | None = None,
        prompt_loader: PromptAssetLoader | None = None,
        prompt_asset: str = "prompts/module3_code_projection.md",
    ) -> None:
        self.llm_client = llm_client or DisabledLLMClient()
        self.prompt_loader = prompt_loader
        self.prompt_asset = prompt_asset
        self.semantic_analyzer = PrivilegeSemanticAnalyzer()

    def rewrite(self, *, patched_bundle_root: Path, item: RepairItem) -> list[str]:
        return self.rewrite_many(
            patched_bundle_root=patched_bundle_root,
            items=[item],
        )

    def rewrite_many(
        self,
        *,
        patched_bundle_root: Path,
        items: list[RepairItem],
    ) -> list[str]:
        """Project every confirmed action in one source file atomically.

        Rewriting items one at a time is unsafe: after the first item, the
        public source has already become a safe-only dispatcher, so a later
        item would copy that dispatcher into its allowed unit and overwrite
        the first item's variants.  A source group therefore receives an
        atomic set of task-conditioned variants spanning the independently
        allowed action subsets, plus one safe-only public entrypoint.  Every
        item receives the same grounded projection contract for subsequent
        instruction dispatch and repair validation.
        """

        repair_items = [
            item
            for item in items
            if item.repair_type == "REORGANIZE_CODE_AND_ADD_DISPATCH"
        ]
        if not repair_items:
            return []
        missing_source_items = [
            item for item in repair_items if item.source_file is None
        ]
        if missing_source_items:
            return [
                f"No code source file was available for {item.repair_id}."
                for item in missing_source_items
            ]

        source_files = {str(item.source_file) for item in repair_items}
        if len(source_files) != 1:
            raise ValueError(
                "CodeRewriter.rewrite_many requires one shared source_file"
            )
        source_file = next(iter(source_files))

        source_path = self._safe_output_path(patched_bundle_root, source_file)
        if source_path is None or not source_path.exists():
            return [
                f"Code source file was missing for {item.repair_id}: {source_file}."
                for item in repair_items
            ]

        if len(repair_items) > 1:
            return self._materialize_composite_fallback_variants(
                patched_bundle_root=patched_bundle_root,
                source_path=source_path,
                items=repair_items,
            )

        item = repair_items[0]
        llm_notes = self._rewrite_with_llm(
            patched_bundle_root=patched_bundle_root,
            source_path=source_path,
            item=item,
        )
        if llm_notes is not None:
            return llm_notes
        return self._materialize_fallback_variants(
            patched_bundle_root=patched_bundle_root,
            source_path=source_path,
            item=item,
        )

    def _materialize_composite_fallback_variants(
        self,
        *,
        patched_bundle_root: Path,
        source_path: Path,
        items: list[RepairItem],
    ) -> list[str]:
        """Materialize independently guarded variants for one source group.

        A shared ``original``/``safe`` pair is insufficient for two actions in
        the same source: selecting the allowed unit for action A would also
        restore action B.  Instead, enumerate the finite privilege lattice of
        the validated actions.  Each variant preserves exactly one subset of
        those actions and neutralizes the rest.  The instruction layer can
        therefore evaluate every :math:`C_a` independently and select exactly
        one matching execution unit, while the public entrypoint remains the
        all-blocked safe path.
        """

        ordered_items = sorted(items, key=lambda value: value.repair_id)
        source_text = source_path.read_text(encoding="utf-8")
        all_ids = {item.repair_id for item in ordered_items}
        original_source = self._ensure_trailing_newline(source_text)
        original_source_sha256 = hashlib.sha256(
            original_source.encode("utf-8")
        ).hexdigest()

        variant_cases: list[dict[str, Any]] = []
        projection_complete = True
        full_mask = (1 << len(ordered_items)) - 1
        for mask in range(full_mask + 1):
            allowed_items = [
                item for index, item in enumerate(ordered_items) if mask & (1 << index)
            ]
            blocked_items = [
                item
                for index, item in enumerate(ordered_items)
                if not mask & (1 << index)
            ]
            expected_blocked_ids = {item.repair_id for item in blocked_items}
            if blocked_items:
                variant_source, neutralized_ids, unlocalized_ids = (
                    self._neutralize_source_many(source_text, blocked_items)
                )
                localized = (
                    neutralized_ids == expected_blocked_ids
                    and not unlocalized_ids
                    and variant_source != source_text
                )
            else:
                variant_source = source_text
                neutralized_ids = set()
                unlocalized_ids = set()
                localized = True

            variant_path = self._composite_variant_path(
                source_path=source_path,
                mask=mask,
                full_mask=full_mask,
            )
            source_valid = self._source_is_valid(
                variant_source,
                source_path.suffix,
                variant_path.name,
            )
            semantic_status = self.semantic_analyzer.compare_projection(
                original_source=source_text,
                projected_source=variant_source,
                suffix=source_path.suffix,
                filename=variant_path.name,
                blocked_items=blocked_items,
            )
            variant_complete = bool(
                localized and source_valid and semantic_status["complete"]
            )
            if not variant_complete:
                projection_complete = False
                # Never broaden permission when a target cannot be localized.
                # The validator records this degraded variant as incomplete, so
                # it cannot make the corresponding repair successful.
                variant_source = self._conservative_safe_source(source_path)
                semantic_status = self.semantic_analyzer.compare_projection(
                    original_source=source_text,
                    projected_source=variant_source,
                    suffix=source_path.suffix,
                    filename=variant_path.name,
                    blocked_items=blocked_items,
                )
            normalized_source = self._ensure_trailing_newline(variant_source)
            variant_path.write_text(normalized_source, encoding="utf-8")
            if source_path.suffix == ".sh":
                self._copy_executable_mode(source_path, variant_path)
            relative_path = (
                variant_path.resolve()
                .relative_to(patched_bundle_root.resolve())
                .as_posix()
            )
            variant_cases.append(
                {
                    "mask": format(mask, f"0{len(ordered_items)}b"),
                    "relative_path": relative_path,
                    "allowed_repair_ids": sorted(
                        item.repair_id for item in allowed_items
                    ),
                    "blocked_repair_ids": sorted(expected_blocked_ids),
                    "neutralized_repair_ids": sorted(neutralized_ids),
                    "unlocalized_repair_ids": sorted(unlocalized_ids),
                    "sha256": hashlib.sha256(
                        normalized_source.encode("utf-8")
                    ).hexdigest(),
                    "complete": variant_complete,
                    "semantic_proof_complete": bool(
                        variant_complete and semantic_status["complete"]
                    ),
                    "original_privilege_semantics": list(
                        semantic_status.get("original_privilege_semantics", [])
                    ),
                    "target_privilege_semantics": list(
                        semantic_status.get("target_privilege_semantics", [])
                    ),
                    "expected_privilege_semantics": list(
                        semantic_status.get("expected_privilege_semantics", [])
                    ),
                    "observed_privilege_semantics": list(
                        semantic_status.get("observed_privilege_semantics", [])
                    ),
                    "unexpected_privilege_semantics": list(
                        semantic_status.get("unexpected_privilege_semantics", [])
                    ),
                    "safe_default": mask == 0,
                    "all_allowed": mask == full_mask,
                }
            )

        safe_case = next(case for case in variant_cases if case["safe_default"])
        safe_path = patched_bundle_root / str(safe_case["relative_path"])
        dispatcher = self._fallback_dispatcher_source(
            source_path=source_path,
            item=ordered_items[0],
            safe_path=safe_path,
        )
        source_path.write_text(
            self._ensure_trailing_newline(dispatcher), encoding="utf-8"
        )
        if source_path.suffix == ".sh":
            self._copy_executable_mode(safe_path, source_path)

        repair_ids = sorted(all_ids)
        manifest_payload = {
            "source_file": source_path.resolve()
            .relative_to(patched_bundle_root.resolve())
            .as_posix(),
            "repair_order": [item.repair_id for item in ordered_items],
            "variants": variant_cases,
        }
        manifest_sha256 = hashlib.sha256(
            json.dumps(
                manifest_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        candidate_ids = sorted(
            {
                str(item.metadata.get("source_candidate_id") or "")
                for item in ordered_items
            }
            - {""}
        )
        generated_files = [str(case["relative_path"]) for case in variant_cases]
        for item in ordered_items:
            singleton_case = next(
                case
                for case in variant_cases
                if case["allowed_repair_ids"] == [item.repair_id]
            )
            allowed_path = patched_bundle_root / str(singleton_case["relative_path"])
            self._record_projection(
                patched_bundle_root=patched_bundle_root,
                source_path=source_path,
                allowed_path=allowed_path,
                safe_path=safe_path,
                item=item,
                strategy=(
                    "composite_independent_task_conditioned_variants_safe_entrypoint"
                ),
                original_source_code=source_text,
            )
            item.generated_files[:] = list(generated_files)
            item.metadata.update(
                {
                    "composite_source_repair_ids": repair_ids,
                    "composite_source_candidate_ids": candidate_ids,
                    "composite_source_repair_count": len(items),
                    "neutralized_repair_ids": list(safe_case["neutralized_repair_ids"]),
                    "unlocalized_repair_ids": list(safe_case["unlocalized_repair_ids"]),
                    "composite_projection_degraded": (not projection_complete),
                    "composite_projection_complete": projection_complete,
                    "original_source_sha256": original_source_sha256,
                    "allowed_unit_sha256": singleton_case["sha256"],
                    "safe_unit_sha256": safe_case["sha256"],
                    "source_privilege_semantics": list(
                        safe_case.get("original_privilege_semantics", [])
                    ),
                    "safe_expected_privilege_semantics": list(
                        safe_case.get("expected_privilege_semantics", [])
                    ),
                    "safe_observed_privilege_semantics": list(
                        safe_case.get("observed_privilege_semantics", [])
                    ),
                    "safe_semantic_proof_complete": bool(
                        safe_case.get("semantic_proof_complete")
                    ),
                    "source_variant_manifest": variant_cases,
                    "source_variant_manifest_sha256": manifest_sha256,
                    "source_variant_repair_order": [
                        value.repair_id for value in ordered_items
                    ],
                    "allowed_execution_units": [
                        str(case["relative_path"])
                        for case in variant_cases
                        if item.repair_id in case["allowed_repair_ids"]
                    ],
                    "blocked_execution_units": [
                        str(case["relative_path"])
                        for case in variant_cases
                        if item.repair_id in case["blocked_repair_ids"]
                    ],
                    "composite_guard_specs": [
                        {
                            "repair_id": value.repair_id,
                            "candidate_id": str(
                                value.metadata.get("source_candidate_id") or ""
                            ),
                            "action_summary": value.overreach_summary
                            or value.raw_text
                            or value.node_id,
                            "guard_condition": str(value.guard_condition or ""),
                        }
                        for value in ordered_items
                    ],
                }
            )
            if not projection_complete:
                item.metadata["safe_projection_degraded"] = (
                    "At least one task-conditioned source variant could not "
                    "localize every blocked target. That variant uses a "
                    "conservative safe implementation and projection "
                    "integrity remains incomplete."
                )
            item.metadata["dispatch_contract"] = self._code_dispatch_contract(
                public_entrypoint=str(
                    item.metadata.get("dispatch_source_file") or item.source_file or ""
                ),
                allowed_execution_unit=str(
                    item.metadata.get("allowed_execution_unit") or ""
                ),
                safe_execution_unit=str(item.metadata.get("safe_execution_unit") or ""),
            )

        return [
            f"Created {len(variant_cases)} independently guarded execution "
            f"variants for {source_path.name}; each of the {len(items)} "
            "validated actions has its own semantic allow condition and the "
            "public entrypoint selects the all-blocked safe variant."
        ]

    def _composite_variant_path(
        self,
        *,
        source_path: Path,
        mask: int,
        full_mask: int,
    ) -> Path:
        if mask == 0:
            return source_path.with_name(
                f"{source_path.stem}__default_safe{source_path.suffix}"
            )
        if mask == full_mask:
            return source_path.with_name(
                f"{source_path.stem}__task_allowed{source_path.suffix}"
            )
        width = max(1, full_mask.bit_length())
        return source_path.with_name(
            f"{source_path.stem}__task_mask_{mask:0{width}b}{source_path.suffix}"
        )

    def _rewrite_with_llm(
        self,
        *,
        patched_bundle_root: Path,
        source_path: Path,
        item: RepairItem,
    ) -> list[str] | None:
        if self.prompt_loader is None:
            return None
        source_code = source_path.read_text(encoding="utf-8")
        if not self._source_span_is_valid(source_code, item):
            item.metadata["code_projection_llm_error"] = (
                "source target line range was missing or outside the grounded "
                "source file"
            )
            return None
        allowed_path, safe_path = self._variant_paths(source_path)
        resolved_root = patched_bundle_root.resolve()
        expected_paths = [
            source_path.resolve().relative_to(resolved_root).as_posix(),
            allowed_path.resolve().relative_to(resolved_root).as_posix(),
            safe_path.resolve().relative_to(resolved_root).as_posix(),
        ]
        dispatch_contract = self._code_dispatch_contract(
            public_entrypoint=expected_paths[0],
            allowed_execution_unit=expected_paths[1],
            safe_execution_unit=expected_paths[2],
        )
        payload = {
            "skill_profile": item.metadata.get("skill_profile") or {},
            "repair_item": self._item_payload(item),
            "source_file": item.source_file,
            "source_code": source_code,
            "required_output_paths": expected_paths,
            "dispatch_contract": dispatch_contract,
            "safe_unit_contract": {
                "blocked_source_text": item.raw_text,
                "blocked_source_start_line": item.source_start_line,
                "blocked_source_end_line": item.source_end_line,
                "must_not_contain_blocked_source_text": True,
                "preserve_all_other_action_semantics": True,
            },
        }
        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        try:
            response = complete_validated_json(
                self.llm_client,
                system_prompt=self.prompt_loader.load(self.prompt_asset),
                user_prompt=user_prompt,
                schema_name="module3_code_projection",
                contract=JSONResponseContract(
                    required_fields=tuple(sorted(self._PROJECTION_OUTPUT_KEYS)),
                    non_empty_string_fields=(
                        "repair_id",
                        "repair_type",
                        "source_file",
                        "guard_condition",
                    ),
                    enum_fields={
                        "repair_type": {item.repair_type.lower()},
                    },
                    evidence_field="descriptor_ids",
                    grounded_evidence_ids=set(item.descriptor_ids),
                    consistency_checks=(
                        lambda value: self._validate_projection_response(
                            response=value,
                            item=item,
                            expected_paths=expected_paths,
                            source_suffix=source_path.suffix,
                            original_source_code=source_code,
                        ),
                    ),
                ),
                max_attempts=MAX_VALIDATED_LLM_ATTEMPTS,
            )
        except RuntimeError as exc:
            item.metadata["code_projection_llm_error"] = str(exc)
            return None

        validated = self._validate_file_outputs(
            file_outputs=response["file_outputs"],
            expected_paths=expected_paths,
            source_suffix=source_path.suffix,
            item=item,
            original_source_code=source_code,
        )
        if validated is None:
            item.metadata["code_projection_llm_error"] = (
                "validated response could not be materialized safely"
            )
            return None
        # The LLM may reorganize execution units, but the public code entrypoint
        # is always replaced with the deterministic safe-only dispatcher.
        # High-dimensional task selection belongs to the instruction layer.
        validated[expected_paths[0]] = self._fallback_dispatcher_source(
            source_path=source_path,
            item=item,
            safe_path=safe_path,
        )
        if not self._source_is_valid(
            validated[expected_paths[0]],
            source_path.suffix,
            expected_paths[0],
        ):
            item.metadata["code_projection_llm_error"] = (
                "the deterministic safe-only public entrypoint failed source "
                "language validation"
            )
            return None

        for relative_path, content in validated.items():
            target_path = self._safe_output_path(patched_bundle_root, relative_path)
            if target_path is None:
                return None
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(
                self._ensure_trailing_newline(content), encoding="utf-8"
            )
            if source_path.suffix == ".sh":
                self._copy_executable_mode(source_path, target_path)

        self._record_projection(
            patched_bundle_root=patched_bundle_root,
            source_path=source_path,
            allowed_path=allowed_path,
            safe_path=safe_path,
            item=item,
            strategy="llm_units_deterministic_safe_entrypoint",
            original_source_code=source_code,
        )
        item.metadata["dispatch_contract"] = dispatch_contract
        return list(response["notes"])

    def _materialize_fallback_variants(
        self,
        *,
        patched_bundle_root: Path,
        source_path: Path,
        item: RepairItem,
    ) -> list[str]:
        allowed_path, safe_path = self._variant_paths(source_path)
        source_text = source_path.read_text(encoding="utf-8")
        shutil.copy2(source_path, allowed_path)
        shutil.copy2(source_path, safe_path)
        # Use the same canonical source representation as the validated LLM
        # path: the only permitted normalization is adding a missing final
        # newline.  This keeps integrity hashes comparable across strategies.
        allowed_path.write_text(
            self._ensure_trailing_newline(source_text), encoding="utf-8"
        )
        safe_text = self._neutralize_source(source_text, item)
        semantic_status = self.semantic_analyzer.compare_projection(
            original_source=source_text,
            projected_source=safe_text,
            suffix=source_path.suffix,
            filename=safe_path.name,
            blocked_items=[item],
        )
        if (
            safe_text == source_text
            or not self._source_is_valid(
                safe_text,
                source_path.suffix,
                safe_path.name,
            )
            or not semantic_status["complete"]
        ):
            safe_text = self._conservative_safe_source(source_path)
            semantic_status = self.semantic_analyzer.compare_projection(
                original_source=source_text,
                projected_source=safe_text,
                suffix=source_path.suffix,
                filename=safe_path.name,
                blocked_items=[item],
            )
            item.metadata["safe_projection_degraded"] = (
                "The target source span could not be localized and proved "
                "safe by static action-graph comparison; the safe unit uses a "
                "conservative completed-without-side-effect implementation."
            )
        if not semantic_status["complete"]:
            item.metadata["safe_projection_degraded"] = (
                "The conservative safe unit could not preserve every unrelated "
                "action while proving removal of the guarded privilege; repair "
                "integrity remains incomplete."
            )
        safe_path.write_text(self._ensure_trailing_newline(safe_text), encoding="utf-8")
        dispatcher = self._fallback_dispatcher_source(
            source_path=source_path,
            item=item,
            safe_path=safe_path,
        )
        source_path.write_text(
            self._ensure_trailing_newline(dispatcher), encoding="utf-8"
        )
        if source_path.suffix == ".sh":
            self._copy_executable_mode(allowed_path, source_path)
            self._copy_executable_mode(allowed_path, safe_path)
        self._record_projection(
            patched_bundle_root=patched_bundle_root,
            source_path=source_path,
            allowed_path=allowed_path,
            safe_path=safe_path,
            item=item,
            strategy="fallback_instruction_routed_safe_entrypoint",
            original_source_code=source_text,
        )
        item.metadata["dispatch_contract"] = self._code_dispatch_contract(
            public_entrypoint=str(
                item.metadata.get("dispatch_source_file") or item.source_file or ""
            ),
            allowed_execution_unit=str(
                item.metadata.get("allowed_execution_unit") or ""
            ),
            safe_execution_unit=str(item.metadata.get("safe_execution_unit") or ""),
        )
        return [
            f"Created a valid safe-only {source_path.suffix or 'script'} entrypoint "
            f"in {item.source_file} plus instruction-routed execution units "
            f"{item.generated_files[0]} and {item.generated_files[1]} for "
            f"{item.repair_id}."
        ]

    def _fallback_dispatcher_source(
        self,
        *,
        source_path: Path,
        item: RepairItem,
        safe_path: Path,
    ) -> str:
        if source_path.suffix == ".sh":
            return self._shell_dispatcher_source(safe_path=safe_path)
        if source_path.suffix == ".py":
            return self._python_dispatcher_source(safe_path=safe_path)
        if source_path.suffix.lower() in {
            ".js",
            ".mjs",
            ".cjs",
            ".ts",
            ".mts",
            ".cts",
        }:
            return self._node_dispatcher_source(safe_path=safe_path)
        # For an unsupported runtime, stay safe instead of writing a Python
        # dispatcher into a non-Python source file.
        item.metadata["unsupported_dispatch_suffix"] = source_path.suffix
        return self._conservative_safe_source(source_path)

    def _python_dispatcher_source(
        self,
        *,
        safe_path: Path,
    ) -> str:
        return "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import runpy",
                "from pathlib import Path",
                "",
                f"SAFE_EXECUTION_UNIT = {safe_path.name!r}",
                "",
                "def main() -> None:",
                "    runpy.run_path(str(Path(__file__).with_name("
                "SAFE_EXECUTION_UNIT)), "
                'run_name="__main__")',
                "",
                'if __name__ == "__main__":',
                "    main()",
            ]
        )

    def _shell_dispatcher_source(
        self,
        *,
        safe_path: Path,
    ) -> str:
        return "\n".join(
            [
                "#!/bin/sh",
                "set -eu",
                'SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)',
                f'exec /bin/sh "$SCRIPT_DIR/{safe_path.name}" "$@"',
            ]
        )

    def _node_dispatcher_source(
        self,
        *,
        safe_path: Path,
    ) -> str:
        """Return a CJS/ESM-compatible, safe-only Node entrypoint.

        Dynamic imports are valid in both CommonJS and ES modules. Resolving the
        safe unit from ``process.argv[1]`` also avoids relying on ``__dirname``
        (CJS-only) or ``import.meta`` (ESM-only). Any resolution or import error
        exits non-zero without falling back to the allowed unit.
        """

        return "\n".join(
            [
                '"use strict";',
                'Promise.all([import("node:path"), import("node:url")])',
                "  .then(([pathModule, urlModule]) => {",
                "    const entrypoint = pathModule.resolve(process.argv[1]);",
                (
                    "    const safePath = pathModule.join("
                    f"pathModule.dirname(entrypoint), {safe_path.name!r});"
                ),
                "    return import(urlModule.pathToFileURL(safePath).href);",
                "  })",
                "  .catch((error) => {",
                '    console.error("SkillScope safe entrypoint failed:", error);',
                "    process.exitCode = 1;",
                "  });",
            ]
        )

    def _code_dispatch_contract(
        self,
        *,
        public_entrypoint: str,
        allowed_execution_unit: str,
        safe_execution_unit: str,
    ) -> dict[str, object]:
        return {
            "selection_layer": "instruction",
            "public_entrypoint": public_entrypoint,
            "public_entrypoint_policy": "always_safe",
            "allowed_execution_unit": allowed_execution_unit,
            "safe_execution_unit": safe_execution_unit,
            "environment_routing": "forbidden",
            "default": "safe_execution_unit",
        }

    def _neutralize_source(self, source_text: str, item: RepairItem) -> str:
        if item.source_file and item.source_file.endswith(".py"):
            rewritten = self._neutralize_python(source_text, item)
            if rewritten is not None:
                return rewritten
        if Path(item.source_file or "").suffix.lower() in {
            ".js",
            ".mjs",
            ".cjs",
            ".ts",
            ".mts",
            ".cts",
        }:
            rewritten = self._neutralize_javascript_typescript(
                source_text,
                item,
            )
            if rewritten is not None:
                return rewritten
        return self._neutralize_by_lines(source_text, item)

    def _neutralize_source_many(
        self,
        source_text: str,
        items: list[RepairItem],
    ) -> tuple[str, set[str], set[str]]:
        """Neutralize a source group without invalidating later source spans."""

        suffix = Path(items[0].source_file or "").suffix.casefold()
        expected_ids = {item.repair_id for item in items}
        if suffix == ".py":
            rewritten, neutralized_ids = self._neutralize_python_many(
                source_text,
                items,
            )
            return (
                rewritten if rewritten is not None else source_text,
                neutralized_ids,
                expected_ids - neutralized_ids,
            )

        # Descending source order keeps every not-yet-processed line/column
        # coordinate stable, including multiple calls on the same source line.
        ordered = sorted(
            items,
            key=lambda item: (
                item.source_start_line or -1,
                item.source_start_column or -1,
                item.source_end_line or -1,
                item.source_end_column or -1,
            ),
            reverse=True,
        )
        rewritten = source_text
        neutralized_ids: set[str] = set()
        for item in ordered:
            updated = self._neutralize_source(rewritten, item)
            if updated != rewritten:
                neutralized_ids.add(item.repair_id)
                rewritten = updated
        return rewritten, neutralized_ids, expected_ids - neutralized_ids

    def _neutralize_python_many(
        self,
        source_text: str,
        items: list[RepairItem],
    ) -> tuple[str | None, set[str]]:
        if any(
            item.source_start_line is None or item.source_end_line is None
            for item in items
        ):
            return None, set()
        try:
            tree = ast.parse(source_text)
        except SyntaxError:
            return None, set()
        transformer = _PythonCompositeRepairTransformer(items)
        updated_tree = transformer.visit(tree)
        if not transformer.replaced:
            return None, set()
        ast.fix_missing_locations(updated_tree)
        return ast.unparse(updated_tree) + "\n", set(transformer.matched_ids)

    def _neutralize_javascript_typescript(
        self,
        source_text: str,
        item: RepairItem,
    ) -> str | None:
        """Reuse replay ablation's span-aware JavaScript neutralization.

        The replay transformer replaces the exact call expression with
        ``undefined`` when columns are available. Before using that shared
        fallback, this method preserves independently evaluated argument
        expressions in a removed call through a comma expression. Together,
        these paths retain same-line siblings and independent nested actions
        instead of deleting an entire source line.
        """

        if item.source_start_line is None or item.source_end_line is None:
            return None
        argument_preserving = self._neutralize_javascript_call_arguments(
            source_text=source_text,
            item=item,
        )
        if argument_preserving is not None:
            return argument_preserving
        ablation = AblationPlan(
            candidate_id=item.overreach_id,
            node_id=item.node_id,
            layer="code",
            strategy="neutralize_code_in_safe_execution_unit",
            source_file=item.source_file,
            source_start_line=item.source_start_line,
            source_end_line=item.source_end_line,
            source_start_column=item.source_start_column,
            source_end_column=item.source_end_column,
            raw_text=item.raw_text,
        )
        try:
            rewritten = CandidateAblation()._neutralize_code_text(
                source_text.splitlines(keepends=True),
                ablation,
            )
        except ValueError:
            return None
        if rewritten == source_text:
            return None
        return rewritten

    def _neutralize_javascript_call_arguments(
        self,
        *,
        source_text: str,
        item: RepairItem,
    ) -> str | None:
        raw_text = (item.raw_text or "").strip()
        argument_source = self._javascript_call_argument_source(raw_text)
        if argument_source is None or not argument_source.strip():
            return None
        arguments = argument_source.rstrip()
        if arguments.endswith(","):
            arguments = arguments[:-1].rstrip()
        if not arguments:
            return None
        replacement = f"({arguments}, undefined)"
        lines = source_text.splitlines(keepends=True)
        start = item.source_start_line - 1
        end = item.source_end_line
        if not (0 <= start < end <= len(lines)):
            return None
        selected_text = "".join(lines[start:end])
        start_column = item.source_start_column
        end_column = item.source_end_column
        if (
            end - start == 1
            and isinstance(start_column, int)
            and isinstance(end_column, int)
            and 0 <= start_column < end_column <= len(selected_text.rstrip("\r\n"))
            and selected_text[start_column:end_column].strip() == raw_text
        ):
            lines[start] = (
                selected_text[:start_column] + replacement + selected_text[end_column:]
            )
            return "".join(lines)
        if selected_text.count(raw_text) != 1:
            return None
        lines[start:end] = [selected_text.replace(raw_text, replacement, 1)]
        return "".join(lines)

    def _javascript_call_argument_source(self, raw_text: str) -> str | None:
        """Return the outer call's arguments for a simple grounded call span."""

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
        return raw_text[final_opening + 1 : -1]

    def _neutralize_python(self, source_text: str, item: RepairItem) -> str | None:
        if item.source_start_line is None or item.source_end_line is None:
            return None
        try:
            tree = ast.parse(source_text)
        except SyntaxError:
            return None
        transformer = _PythonRepairTransformer(
            start_line=item.source_start_line,
            end_line=item.source_end_line,
            start_column=item.source_start_column,
            end_column=item.source_end_column,
            raw_text=item.raw_text,
        )
        updated_tree = transformer.visit(tree)
        if not transformer.replaced:
            return None
        ast.fix_missing_locations(updated_tree)
        return ast.unparse(updated_tree) + "\n"

    def _neutralize_by_lines(self, source_text: str, item: RepairItem) -> str:
        lines = source_text.splitlines(keepends=True)
        if item.source_start_line is None or item.source_end_line is None:
            return self._neutralize_raw_text(source_text, item)
        start = max(item.source_start_line - 1, 0)
        end = min(item.source_end_line, len(lines))
        if start >= end:
            return self._neutralize_raw_text(source_text, item)
        indentation = self._indentation_for(lines[start:end])
        no_op = self._no_op_statement(item)
        lines[start:end] = [
            f"{indentation}{no_op}  # SkillScope repair {item.repair_id}\n"
        ]
        return "".join(lines)

    def _neutralize_raw_text(self, source_text: str, item: RepairItem) -> str:
        raw_text = (item.raw_text or "").strip()
        if not raw_text or raw_text not in source_text:
            return source_text
        return source_text.replace(
            raw_text,
            f"{self._no_op_statement(item)}  # SkillScope repair {item.repair_id}",
            1,
        )

    def _no_op_statement(self, item: RepairItem) -> str:
        suffix = Path(item.source_file or "").suffix.casefold()
        if suffix == ".sh":
            return ":"
        if suffix in {".js", ".mjs", ".cjs", ".ts", ".mts", ".cts"}:
            return "void 0;"
        return "pass"

    def _conservative_safe_source(self, source_path: Path) -> str:
        if source_path.suffix == ".sh":
            return "\n".join(
                [
                    "#!/bin/sh",
                    "set -eu",
                    "printf '%s\\n' 'SkillScope safe path completed without the "
                    "guarded side effect.'",
                ]
            )
        if source_path.suffix == ".py":
            return "\n".join(
                [
                    "from __future__ import annotations",
                    "",
                    "def main() -> None:",
                    '    print("SkillScope safe path completed without the guarded '
                    'side effect.")',
                    "",
                    'if __name__ == "__main__":',
                    "    main()",
                ]
            )
        if source_path.suffix.lower() in {
            ".js",
            ".mjs",
            ".cjs",
            ".ts",
            ".mts",
            ".cts",
        }:
            return "\n".join(
                [
                    '"use strict";',
                    'console.log("SkillScope safe path completed without the '
                    'guarded side effect.");',
                ]
            )
        return (
            "/* SkillScope safe path: guarded side effect disabled because the "
            "source span could not be localized. */\n"
        )

    def _validate_projection_response(
        self,
        *,
        response: dict[str, Any],
        item: RepairItem,
        expected_paths: list[str],
        source_suffix: str,
        original_source_code: str,
    ) -> str | None:
        if set(response) != self._PROJECTION_OUTPUT_KEYS:
            return "code projection fields must exactly match the fixed schema"
        expected_scalars: dict[str, object] = {
            "repair_id": item.repair_id,
            "repair_type": item.repair_type,
            "source_file": item.source_file,
            "source_start_line": item.source_start_line,
            "source_end_line": item.source_end_line,
            "guard_condition": item.guard_condition,
        }
        for field_name, expected in expected_scalars.items():
            if response.get(field_name) != expected:
                return (
                    f"code projection field {field_name!r} does not match "
                    "grounded source evidence"
                )
        for field_name, expected in (
            ("descriptor_ids", item.descriptor_ids),
            ("allowed_cluster_keys", item.allowed_cluster_keys),
            ("blocked_cluster_keys", item.blocked_cluster_keys),
        ):
            value = response.get(field_name)
            if (
                not isinstance(value, list)
                or any(not isinstance(entry, str) for entry in value)
                or value != expected
            ):
                return (
                    f"code projection field {field_name!r} must exactly echo "
                    "grounded evidence"
                )
        notes = response.get("notes")
        if not isinstance(notes, list) or any(
            not isinstance(note, str) for note in notes
        ):
            return "code projection notes must be a list of strings"
        file_outputs = response.get("file_outputs")
        if isinstance(file_outputs, list):
            outputs_by_path = {
                str(output.get("relative_path")): output.get("content")
                for output in file_outputs
                if isinstance(output, dict)
            }
            allowed_content = outputs_by_path.get(expected_paths[1])
            if isinstance(allowed_content, str) and self._ensure_trailing_newline(
                allowed_content
            ) != self._ensure_trailing_newline(original_source_code):
                return (
                    "allowed execution unit must copy source_code exactly; "
                    "do not rewrite its comments, whitespace, or behavior"
                )
            safe_content = outputs_by_path.get(expected_paths[2])
            raw_target = (item.raw_text or "").strip()
            if (
                isinstance(safe_content, str)
                and len(raw_target) >= 8
                and raw_target in safe_content
            ):
                return (
                    "safe execution unit still contains repair_item.raw_text; "
                    "remove or neutralize that exact blocked source action "
                    "while preserving the remaining source behavior"
                )
        if (
            self._validate_file_outputs(
                file_outputs=file_outputs,
                expected_paths=expected_paths,
                source_suffix=source_suffix,
                item=item,
                original_source_code=original_source_code,
            )
            is None
        ):
            return (
                "file_outputs did not match grounded target paths, source "
                "language, the exact original allowed unit, safe-only public "
                "entrypoint, or safe-unit constraints"
            )
        return None

    def _source_span_is_valid(self, source_code: str, item: RepairItem) -> bool:
        start = item.source_start_line
        end = item.source_end_line
        line_count = len(source_code.splitlines())
        return (
            isinstance(start, int)
            and isinstance(end, int)
            and 1 <= start <= end <= line_count
        )

    def _validate_file_outputs(
        self,
        *,
        file_outputs: object,
        expected_paths: list[str],
        source_suffix: str,
        item: RepairItem,
        original_source_code: str,
    ) -> dict[str, str] | None:
        if not isinstance(file_outputs, list) or len(file_outputs) != len(
            expected_paths
        ):
            return None
        materialized: dict[str, str] = {}
        for output in file_outputs:
            if not isinstance(output, dict) or set(output) != {
                "relative_path",
                "content",
            }:
                return None
            relative_path = output.get("relative_path")
            content = output.get("content")
            if (
                not isinstance(relative_path, str)
                or not isinstance(content, str)
                or not content.strip()
                or relative_path in materialized
            ):
                return None
            materialized[relative_path] = content
        if list(materialized) != expected_paths and set(materialized) != set(
            expected_paths
        ):
            return None

        dispatcher = materialized[expected_paths[0]]
        allowed_unit_name = Path(expected_paths[1]).name
        safe_unit_name = Path(expected_paths[2]).name
        safe_unit_references = {
            safe_unit_name,
            Path(safe_unit_name).stem,
        }
        if not any(reference in dispatcher for reference in safe_unit_references):
            return None
        allowed_unit_references = {
            allowed_unit_name,
            Path(allowed_unit_name).stem,
        }
        if any(
            reference in dispatcher
            for reference in allowed_unit_references
        ):
            return None
        if (
            "SKILLSCOPE_TASK_CLUSTER" in dispatcher
            or "SKILLSCOPE_USER_PROMPT" in dispatcher
        ):
            return None
        # The allowed unit is not a model-authored rewrite.  It must preserve
        # the complete original artifact byte-for-byte after the sole allowed
        # normalization of adding a missing final newline.  This prevents a
        # syntactically valid LLM projection from changing legitimate behavior.
        if self._ensure_trailing_newline(materialized[expected_paths[1]]) != (
            self._ensure_trailing_newline(original_source_code)
        ):
            return None
        raw_target = (item.raw_text or "").strip()
        if len(raw_target) >= 8 and (
            raw_target in dispatcher or raw_target in materialized[expected_paths[2]]
        ):
            return None
        for path, content in materialized.items():
            if not self._source_is_valid(content, source_suffix, path):
                return None
        semantic_status = self.semantic_analyzer.compare_projection(
            original_source=original_source_code,
            projected_source=materialized[expected_paths[2]],
            suffix=source_suffix,
            filename=expected_paths[2],
            blocked_items=[item],
        )
        if not semantic_status["complete"]:
            return None
        return materialized

    def _source_is_valid(self, content: str, suffix: str, filename: str) -> bool:
        if suffix == ".py":
            try:
                compile(content, filename, "exec")
            except SyntaxError:
                return False
        if suffix == ".sh":
            completed = subprocess.run(
                ["/bin/sh", "-n"],
                input=content,
                text=True,
                capture_output=True,
                check=False,
            )
            return completed.returncode == 0
        if suffix.lower() in {
            ".js",
            ".mjs",
            ".cjs",
            ".ts",
            ".mts",
            ".cts",
        }:
            return self._node_source_is_valid(
                content=content,
                suffix=suffix.lower(),
                filename=filename,
            )
        return True

    def _node_source_is_valid(
        self,
        *,
        content: str,
        suffix: str,
        filename: str,
    ) -> bool:
        node_binary = shutil.which("node")
        if node_binary is None:
            return False
        basename = Path(filename).name
        stem = Path(basename).stem or "projection"
        syntax_content = content
        syntax_suffix = suffix
        if suffix in {".ts", ".mts", ".cts"}:
            stripped = self._strip_typescript_for_validation(
                content=content,
                node_binary=node_binary,
            )
            if stripped is None:
                return False
            syntax_content = stripped
            syntax_suffix = {
                ".ts": ".js",
                ".mts": ".mjs",
                ".cts": ".cjs",
            }[suffix]
        try:
            with tempfile.TemporaryDirectory(prefix="skillscope-node-syntax-") as raw:
                source_path = Path(raw) / f"{stem}{syntax_suffix}"
                source_path.write_text(
                    self._ensure_trailing_newline(syntax_content),
                    encoding="utf-8",
                )
                completed = subprocess.run(
                    [node_binary, "--check", str(source_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
        except OSError:
            return False
        return completed.returncode == 0

    def _strip_typescript_for_validation(
        self,
        *,
        content: str,
        node_binary: str,
    ) -> str | None:
        """Strip erasable TypeScript syntax without executing the projection."""

        validation_script = " ".join(
            [
                'const fs = require("node:fs");',
                'const { stripTypeScriptTypes } = require("node:module");',
                'if (typeof stripTypeScriptTypes !== "function")',
                "  process.exit(2);",
                'const source = fs.readFileSync(0, "utf8");',
                "process.stdout.write(stripTypeScriptTypes(source,",
                '  { mode: "strip" }));',
            ]
        )
        try:
            completed = subprocess.run(
                [
                    node_binary,
                    "--no-warnings",
                    "--experimental-strip-types",
                    "-e",
                    validation_script,
                ],
                input=content,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout

    def _safe_output_path(self, root: Path, relative_path: str) -> Path | None:
        pure_path = PurePosixPath(relative_path)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            return None
        root = root.resolve()
        target = (root / Path(*pure_path.parts)).resolve()
        if target != root and root not in target.parents:
            return None
        return target

    def _record_projection(
        self,
        *,
        patched_bundle_root: Path,
        source_path: Path,
        allowed_path: Path,
        safe_path: Path,
        item: RepairItem,
        strategy: str,
        original_source_code: str,
    ) -> None:
        resolved_root = patched_bundle_root.resolve()
        item.generated_files[:] = [
            allowed_path.resolve().relative_to(resolved_root).as_posix(),
            safe_path.resolve().relative_to(resolved_root).as_posix(),
        ]
        item.metadata["dispatch_source_file"] = (
            source_path.resolve().relative_to(resolved_root).as_posix()
        )
        item.metadata["allowed_execution_unit"] = item.generated_files[0]
        item.metadata["safe_execution_unit"] = item.generated_files[1]
        item.metadata["public_entrypoint_policy"] = (
            "safe_only_instruction_layer_selects_allowed_unit"
        )
        item.metadata["code_projection_strategy"] = strategy
        item.metadata["original_source_sha256"] = hashlib.sha256(
            self._ensure_trailing_newline(original_source_code).encode("utf-8")
        ).hexdigest()
        item.metadata["public_entrypoint_sha256"] = self._file_sha256(source_path)
        item.metadata["allowed_unit_sha256"] = self._file_sha256(allowed_path)
        item.metadata["safe_unit_sha256"] = self._file_sha256(safe_path)
        semantic_status = self.semantic_analyzer.compare_projection(
            original_source=original_source_code,
            projected_source=safe_path.read_text(encoding="utf-8"),
            suffix=source_path.suffix,
            filename=safe_path.name,
            blocked_items=[item],
        )
        item.metadata["source_privilege_semantics"] = list(
            semantic_status.get("original_privilege_semantics", [])
        )
        item.metadata["target_privilege_semantics"] = list(
            semantic_status.get("target_privilege_semantics", [])
        )
        item.metadata["safe_expected_privilege_semantics"] = list(
            semantic_status.get("expected_privilege_semantics", [])
        )
        item.metadata["safe_observed_privilege_semantics"] = list(
            semantic_status.get("observed_privilege_semantics", [])
        )
        item.metadata["safe_unexpected_privilege_semantics"] = list(
            semantic_status.get("unexpected_privilege_semantics", [])
        )
        item.metadata["safe_semantic_proof_complete"] = bool(
            semantic_status["complete"]
        )
        item.metadata["safe_semantic_proof_reason"] = str(semantic_status["reason"])
        item.metadata.setdefault("composite_source_repair_ids", [item.repair_id])
        item.metadata.setdefault("composite_source_repair_count", 1)
        item.metadata.setdefault("neutralized_repair_ids", [item.repair_id])

    def _item_payload(self, item: RepairItem) -> dict[str, object]:
        return {
            "repair_id": item.repair_id,
            "overreach_id": item.overreach_id,
            "node_id": item.node_id,
            "layer": item.layer,
            "repair_type": item.repair_type,
            "overreach_summary": item.overreach_summary,
            "rationale": item.rationale,
            "guard_condition": item.guard_condition,
            "descriptor_ids": list(item.descriptor_ids),
            "allowed_cluster_keys": list(item.allowed_cluster_keys),
            "blocked_cluster_keys": list(item.blocked_cluster_keys),
            "descriptor_contexts": item.metadata.get("descriptor_contexts") or [],
            "descriptor_clusters": item.metadata.get("descriptor_clusters") or [],
            "source_file": item.source_file,
            "source_start_line": item.source_start_line,
            "source_end_line": item.source_end_line,
            "source_start_column": item.source_start_column,
            "source_end_column": item.source_end_column,
            "raw_text": item.raw_text,
        }

    def _variant_paths(self, source_path: Path) -> tuple[Path, Path]:
        stem = source_path.stem
        suffix = source_path.suffix
        return (
            source_path.with_name(f"{stem}__task_allowed{suffix}"),
            source_path.with_name(f"{stem}__default_safe{suffix}"),
        )

    def _copy_executable_mode(self, source: Path, target: Path) -> None:
        target.chmod(source.stat().st_mode)

    def _indentation_for(self, lines: list[str]) -> str:
        for line in lines:
            stripped = line.lstrip()
            if stripped:
                return line[: len(line) - len(stripped)]
        return ""

    def _ensure_trailing_newline(self, text: str) -> str:
        return text if text.endswith("\n") else text + "\n"

    def _file_sha256(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


class _PythonRepairTransformer(ast.NodeTransformer):
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
        self.raw_call_ast_dump: str | None = None
        if self.raw_text:
            try:
                raw_expression = ast.parse(self.raw_text, mode="eval").body
            except SyntaxError:
                raw_expression = None
            if isinstance(raw_expression, ast.Call):
                self.raw_call_ast_dump = ast.dump(
                    raw_expression,
                    include_attributes=False,
                )
        self.replaced = False

    def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.stmt]:
        if self._contains_target_call(node.value):
            return self._replacement_statements(
                node=node,
                terminal=ast.Pass(),
            )
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        if self._contains_target_call(node):
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
        return self.generic_visit(node)

    def visit_AnnAssign(
        self,
        node: ast.AnnAssign,
    ) -> ast.AST | list[ast.stmt]:
        if self._contains_target_call(node):
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
        return self.generic_visit(node)

    def visit_AugAssign(
        self,
        node: ast.AugAssign,
    ) -> ast.AST | list[ast.stmt]:
        if self._contains_target_call(node):
            return self._replacement_statements(
                node=node,
                terminal=ast.Pass(),
            )
        return self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> ast.AST | list[ast.stmt]:
        if node.value is not None and self._contains_target_call(node.value):
            return self._replacement_statements(
                node=node,
                terminal=ast.Return(value=ast.Constant(value=None)),
            )
        return self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> ast.AST | list[ast.stmt]:
        if self._contains_target_call(node):
            return self._replacement_statements(
                node=node,
                terminal=ast.Pass(),
            )
        return self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> ast.AST | list[ast.stmt]:
        if self._contains_target_call(node):
            return self._replacement_statements(
                node=node,
                terminal=ast.Pass(),
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
        if not (lineno >= self.start_line and end_lineno <= self.end_line):
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
        if self.raw_call_ast_dump is not None:
            return ast.dump(node, include_attributes=False) == (self.raw_call_ast_dump)
        return ast.unparse(node).strip() == self.raw_text


class _PythonCompositeRepairTransformer(_PythonRepairTransformer):
    """Apply all grounded Python call targets in one AST transformation."""

    def __init__(self, items: list[RepairItem]) -> None:
        first = items[0]
        assert first.source_start_line is not None
        assert first.source_end_line is not None
        super().__init__(
            start_line=first.source_start_line,
            end_line=first.source_end_line,
            start_column=first.source_start_column,
            end_column=first.source_end_column,
            raw_text=first.raw_text,
        )
        self._target_matchers: list[tuple[str, _PythonRepairTransformer]] = []
        for item in items:
            assert item.source_start_line is not None
            assert item.source_end_line is not None
            self._target_matchers.append(
                (
                    item.repair_id,
                    _PythonRepairTransformer(
                        start_line=item.source_start_line,
                        end_line=item.source_end_line,
                        start_column=item.source_start_column,
                        end_column=item.source_end_column,
                        raw_text=item.raw_text,
                    ),
                )
            )
        self.matched_ids: set[str] = set()

    def _matches_target_call(self, node: ast.AST) -> bool:
        matched = False
        for repair_id, matcher in self._target_matchers:
            if matcher._matches_target_call(node):
                self.matched_ids.add(repair_id)
                matched = True
        return matched
