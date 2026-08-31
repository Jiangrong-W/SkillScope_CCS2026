from __future__ import annotations

import shutil
from pathlib import Path

from skillscope.common.models import (
    RepairItem,
    RepairOutcome,
    RepairPlan,
    SkillBundle,
)

from .code_rewriter import CodeRewriter
from .instruction_rewriter import InstructionRewriter


class BundleProjector:
    def __init__(
        self,
        *,
        instruction_rewriter: InstructionRewriter | None = None,
        code_rewriter: CodeRewriter | None = None,
    ) -> None:
        self.instruction_rewriter = instruction_rewriter or InstructionRewriter()
        self.code_rewriter = code_rewriter or CodeRewriter()

    def project(
        self,
        *,
        bundle: SkillBundle,
        plan: RepairPlan,
        output_root: Path,
    ) -> RepairOutcome:
        patched_bundle_root = output_root.resolve()
        if patched_bundle_root.exists():
            shutil.rmtree(patched_bundle_root)
        shutil.copytree(bundle.root_path, patched_bundle_root, dirs_exist_ok=True)

        notes: list[str] = []
        code_items_by_source: dict[str, list[RepairItem]] = {}
        code_items_without_source: list[RepairItem] = []
        for item in plan.items:
            if item.layer == "code":
                if item.source_file:
                    code_items_by_source.setdefault(item.source_file, []).append(
                        item
                    )
                else:
                    code_items_without_source.append(item)

        for items in code_items_by_source.values():
            notes.extend(
                self.code_rewriter.rewrite_many(
                    patched_bundle_root=patched_bundle_root,
                    items=items,
                )
            )
        for item in code_items_without_source:
            notes.extend(
                self.code_rewriter.rewrite(
                    patched_bundle_root=patched_bundle_root,
                    item=item,
                )
            )

        # Code dispatch is written only after every code group has been
        # materialized, so every item has a complete allowed/safe-unit
        # contract.  An instruction candidate that merely invokes one of these
        # repaired sources is covered by that concrete projection; projecting a
        # second independent guard over the same line would overwrite the
        # executable dispatch and create two competing enforcement points.
        code_items = [
            item
            for item in plan.items
            if item.layer == "code"
            and item.repair_type == "REORGANIZE_CODE_AND_ADD_DISPATCH"
        ]
        code_dispatch_groups: dict[str, list[RepairItem]] = {}
        for item in code_items:
            group_key = str(item.source_file or item.repair_id)
            code_dispatch_groups.setdefault(group_key, []).append(item)
        for group in code_dispatch_groups.values():
            ordered_group = sorted(group, key=lambda value: value.repair_id)
            representative = ordered_group[0]
            notes.extend(
                self.instruction_rewriter.rewrite(
                    patched_bundle_root=patched_bundle_root,
                    item=representative,
                )
            )
            # A same-source group has one composable semantic dispatch block.
            # Record that exact block on every action so integrity and replay
            # validation remain action-specific without rewriting the source
            # instruction multiple times.
            for peer in ordered_group[1:]:
                for key in (
                    "instruction_dispatch_block",
                    "dispatch_instruction_text",
                    "instruction_routing_policy",
                    "non_semantic_planner_fallback",
                    "instruction_projection_strategy",
                    "instruction_projection_llm_error",
                ):
                    if key in representative.metadata:
                        peer.metadata[key] = representative.metadata[key]
                peer.metadata["instruction_projection_strategy"] = (
                    "shared_composite_source_dispatch"
                )
                notes.append(
                    f"Mapped {peer.repair_id} to the shared independently "
                    f"guarded dispatch for {peer.source_file}."
                )
        standalone_instruction_items: list[RepairItem] = []
        for item in plan.items:
            if item.layer != "instruction":
                continue
            related_items = self._related_code_items(item, code_items)
            covering_items = [
                code_item
                for code_item in related_items
                if self._projection_semantics_are_equivalent(
                    instruction_item=item,
                    code_item=code_item,
                )
            ]
            coverage_strategy = (
                "covered_by_semantically_equivalent_code_projection"
                if covering_items
                else None
            )
            coverage_evidence: dict[str, object] | None = None
            if covering_items:
                coverage_evidence = self._equivalence_evidence(
                    instruction_item=item,
                    covering_items=covering_items,
                )
            elif related_items:
                causal_coverage = self._causal_code_coverage(
                    instruction_item=item,
                    related_items=related_items,
                )
                if causal_coverage is not None:
                    covering_items, coverage_evidence = causal_coverage
                    coverage_strategy = (
                        "covered_by_causally_complete_code_projection"
                    )
            if covering_items:
                self._apply_code_coverage(
                    instruction_item=item,
                    covering_items=covering_items,
                    strategy=str(coverage_strategy),
                    evidence=coverage_evidence or {},
                )
                notes.append(
                    f"Mapped {item.repair_id} to evidence-complete code "
                    "projection(s) "
                    + ", ".join(
                        value.repair_id for value in covering_items
                    )
                    + "."
                )
                continue
            if related_items:
                # A filename-level invocation relation proves only that the
                # instruction can reach the script.  It does not prove that an
                # instruction action and one of the script's internal actions
                # share the same task authorization or necessity boundary.
                # Silently treating the code rewrite as coverage would drop an
                # independent guard.  Fail closed and leave this item for the
                # validator to report as unresolved instead of emitting a
                # competing rewrite over the same instruction span.
                item.metadata.update(
                    {
                        "instruction_projection_strategy": (
                            "unresolved_cross_layer_projection_conflict"
                        ),
                        "projection_conflict_reason": (
                            "A code projection targets an invoked script, but "
                            "candidate semantics, descriptor boundaries, and "
                            "guard conditions are not proven equivalent."
                        ),
                        "related_code_projection_repair_ids": [
                            value.repair_id for value in related_items
                        ],
                    }
                )
                notes.append(
                    f"Left {item.repair_id} unresolved because a filename-level "
                    "code relation did not prove equivalent action and guard "
                    "semantics."
                )
                continue
            standalone_instruction_items.append(item)

        # Instruction graph normalization may split one Markdown block into
        # multiple atomic nodes that share the same source range.  Project the
        # surviving instruction repairs as one ordered batch so same-range
        # fragments are composed from the original block instead of silently
        # overwriting one another.
        notes.extend(
            self.instruction_rewriter.rewrite_many(
                patched_bundle_root=patched_bundle_root,
                items=standalone_instruction_items,
            )
        )

        return RepairOutcome(
            plan=plan,
            patched_bundle_path=str(patched_bundle_root),
            notes=notes,
            metadata={
                "projected_item_count": len(plan.items),
                "projection_strategy": "localized_control_flow_projection",
                "code_projection_group_count": len(code_items_by_source),
            },
        )

    def _related_code_items(
        self,
        instruction_item: RepairItem,
        code_items: list[RepairItem],
    ) -> list[RepairItem]:
        instruction_text = " ".join(
            value
            for value in (
                instruction_item.raw_text,
                instruction_item.overreach_summary,
            )
            if value
        ).replace("\\", "/")
        matches: list[RepairItem] = []
        for code_item in code_items:
            source_file = str(code_item.source_file or "").replace("\\", "/")
            if not source_file:
                continue
            if (
                source_file in instruction_text
                or Path(source_file).name in instruction_text
            ):
                matches.append(code_item)
        return matches

    def _projection_semantics_are_equivalent(
        self,
        *,
        instruction_item: RepairItem,
        code_item: RepairItem,
    ) -> bool:
        """Require positive evidence before sharing a cross-layer guard.

        Mentioning a script path establishes reachability, not action identity.
        Shared projection is sound only when both repair items preserve the
        same normalized operation/object pair, descriptor-cluster boundary,
        and exact deny-by-default guard.
        """

        instruction_semantics = instruction_item.metadata.get(
            "candidate_semantics"
        )
        code_semantics = code_item.metadata.get("candidate_semantics")
        if not isinstance(instruction_semantics, dict) or not isinstance(
            code_semantics, dict
        ):
            return False

        for field_name in ("operation_type", "object_ref"):
            instruction_value = self._normalize_semantic_value(
                instruction_semantics.get(field_name)
            )
            code_value = self._normalize_semantic_value(
                code_semantics.get(field_name)
            )
            if (
                not instruction_value
                or not code_value
                or instruction_value != code_value
            ):
                return False

        if self._normalize_semantic_value(
            instruction_item.guard_condition
        ) != self._normalize_semantic_value(code_item.guard_condition):
            return False
        if set(instruction_item.allowed_cluster_keys) != set(
            code_item.allowed_cluster_keys
        ):
            return False
        if set(instruction_item.blocked_cluster_keys) != set(
            code_item.blocked_cluster_keys
        ):
            return False
        return True

    def _apply_code_coverage(
        self,
        *,
        instruction_item: RepairItem,
        covering_items: list[RepairItem],
        strategy: str,
        evidence: dict[str, object],
    ) -> None:
        representative = covering_items[0]
        instruction_item.generated_files[:] = list(
            representative.generated_files
        )
        for key in (
            "dispatch_source_file",
            "allowed_execution_unit",
            "safe_execution_unit",
            "public_entrypoint_policy",
            "code_projection_strategy",
            "dispatch_contract",
            "instruction_dispatch_block",
            "dispatch_instruction_text",
            "instruction_routing_policy",
            "non_semantic_planner_fallback",
            "source_variant_manifest",
            "source_variant_manifest_sha256",
            "source_variant_repair_order",
            "allowed_execution_units",
            "blocked_execution_units",
            "composite_guard_specs",
        ):
            if key in representative.metadata:
                instruction_item.metadata[key] = representative.metadata[key]
        instruction_item.metadata.update(
            {
                "instruction_projection_strategy": strategy,
                "covered_by_code_projection_repair_ids": [
                    value.repair_id for value in covering_items
                ],
                "covered_code_node_ids": [
                    value.node_id for value in covering_items
                ],
                "covered_code_summaries": [
                    value.overreach_summary for value in covering_items
                ],
            }
        )
        if strategy == "covered_by_semantically_equivalent_code_projection":
            instruction_item.metadata[
                "cross_layer_equivalence_evidence"
            ] = evidence
        else:
            instruction_item.metadata[
                "cross_layer_causal_coverage_evidence"
            ] = evidence

    def _equivalence_evidence(
        self,
        *,
        instruction_item: RepairItem,
        covering_items: list[RepairItem],
    ) -> dict[str, object]:
        return {
            "evidence_version": 1,
            "instruction_candidate_semantics": dict(
                instruction_item.metadata.get("candidate_semantics") or {}
            ),
            "instruction_guard_condition": instruction_item.guard_condition,
            "instruction_allowed_cluster_keys": list(
                instruction_item.allowed_cluster_keys
            ),
            "instruction_blocked_cluster_keys": list(
                instruction_item.blocked_cluster_keys
            ),
            "covered_code_items": [
                {
                    "repair_id": value.repair_id,
                    "candidate_semantics": dict(
                        value.metadata.get("candidate_semantics") or {}
                    ),
                    "guard_condition": value.guard_condition,
                    "allowed_cluster_keys": list(
                        value.allowed_cluster_keys
                    ),
                    "blocked_cluster_keys": list(
                        value.blocked_cluster_keys
                    ),
                }
                for value in covering_items
            ],
        }

    def _causal_code_coverage(
        self,
        *,
        instruction_item: RepairItem,
        related_items: list[RepairItem],
    ) -> tuple[list[RepairItem], dict[str, object]] | None:
        final_verdicts = instruction_item.metadata.get(
            "source_final_verdicts"
        )
        if not isinstance(final_verdicts, list):
            return None
        confirmed = [
            value
            for value in final_verdicts
            if isinstance(value, dict)
            and value.get("label") == "overprivileged"
        ]
        if not confirmed or any(
            value.get("authorization_label") != "unauthorized"
            or value.get("necessity_label") == "unnecessary"
            or "unnecessary" in (value.get("overprivilege_reasons") or [])
            for value in confirmed
        ):
            return None

        instruction_actions = self._material_actions(
            instruction_item,
            overprivileged_only=True,
        )
        if not instruction_actions:
            return None
        code_actions_by_repair = {
            item.repair_id: self._material_actions(
                item,
                overprivileged_only=False,
            )
            for item in related_items
            if item.metadata.get("safe_semantic_proof_complete") is True
            and item.repair_id
            in set(item.metadata.get("neutralized_repair_ids") or [])
            and item.metadata.get("safe_execution_unit")
            and item.metadata.get("safe_unit_sha256")
        }
        if not code_actions_by_repair:
            return None

        action_coverage: dict[
            tuple[tuple[str, str], ...], dict[str, object]
        ] = {}
        for action in instruction_actions:
            matching_repairs = sorted(
                repair_id
                for repair_id, code_actions in code_actions_by_repair.items()
                if action in code_actions
            )
            coverage_kind = "code_action_neutralization"
            if not matching_repairs:
                matching_repairs = sorted(
                    item.repair_id
                    for item in related_items
                    if item.repair_id in code_actions_by_repair
                    and self._invocation_envelope_is_rewritten(
                        action=action,
                        code_item=item,
                    )
                )
                coverage_kind = "instruction_dispatch_rewrite"
            if not matching_repairs:
                return None
            action_coverage[action] = {
                "repair_ids": matching_repairs,
                "coverage_kind": coverage_kind,
            }
        contributing_ids = {
            repair_id
            for coverage in action_coverage.values()
            for repair_id in coverage["repair_ids"]
        }
        contributing_items = [
            item
            for item in related_items
            if item.repair_id in contributing_ids
        ]
        evidence = {
            "evidence_version": 2,
            "instruction_final_verdicts": [dict(value) for value in confirmed],
            "instruction_material_actions": [
                dict(action) for action in instruction_actions
            ],
            "material_action_to_repair_ids": [
                {
                    "material_action": dict(action),
                    "repair_ids": coverage["repair_ids"],
                    "coverage_kind": coverage["coverage_kind"],
                }
                for action, coverage in sorted(action_coverage.items())
            ],
            "covered_code_items": [
                {
                    "repair_id": item.repair_id,
                    "material_actions": [
                        dict(action)
                        for action in code_actions_by_repair[item.repair_id]
                    ],
                    "neutralized_repair_ids": list(
                        item.metadata.get("neutralized_repair_ids") or []
                    ),
                    "safe_execution_unit": item.metadata.get(
                        "safe_execution_unit"
                    ),
                    "safe_unit_sha256": item.metadata.get(
                        "safe_unit_sha256"
                    ),
                    "safe_semantic_proof_complete": item.metadata.get(
                        "safe_semantic_proof_complete"
                    ),
                    "dispatch_source_file": item.metadata.get(
                        "dispatch_source_file"
                    ),
                    "instruction_invocation_template": dict(
                        item.metadata.get("instruction_invocation_template")
                        or {}
                    ),
                    "instruction_dispatch_block": item.metadata.get(
                        "instruction_dispatch_block"
                    ),
                    "instruction_routing_policy": item.metadata.get(
                        "instruction_routing_policy"
                    ),
                    "public_entrypoint_policy": item.metadata.get(
                        "public_entrypoint_policy"
                    ),
                }
                for item in contributing_items
            ],
        }
        return contributing_items, evidence

    def _invocation_envelope_is_rewritten(
        self,
        *,
        action: tuple[tuple[str, str], ...],
        code_item: RepairItem,
    ) -> bool:
        """Prove coverage for an instruction-to-script invocation envelope.

        A realized instruction candidate can contain both the command that
        enters a bundled script and the privilege-relevant actions executed
        inside that script.  The inner actions must be neutralized by exact
        code-action evidence above.  The invocation envelope is covered only
        when the original invocation has an exact, deny-by-default instruction
        dispatch projection to the code repair's safe/allowed units.
        """

        action_fields = dict(action)
        if action_fields.get("operation") not in {
            "exec_command",
            "execute",
            "invoke",
            "run",
            "call",
        }:
            return False
        source_file = self._normalize_semantic_value(code_item.source_file)
        dispatch_source = self._normalize_semantic_value(
            code_item.metadata.get("dispatch_source_file")
        )
        template = code_item.metadata.get("instruction_invocation_template")
        if (
            not source_file
            or dispatch_source != source_file
            or not isinstance(template, dict)
        ):
            return False
        source_token = self._normalize_semantic_value(
            template.get("source_token")
        )
        if source_token != source_file:
            return False
        action_object = action_fields.get("object", "")
        object_tokens = {
            self._normalize_semantic_value(token.strip("'\"`.,;:()[]{}"))
            for token in action_object.split()
        }
        if source_token not in object_tokens:
            return False
        return (
            bool(
                str(
                    code_item.metadata.get("instruction_dispatch_block") or ""
                ).strip()
            )
            and code_item.metadata.get("instruction_routing_policy")
            == "semantic_agent_selection_with_safe_default"
            and code_item.metadata.get("public_entrypoint_policy")
            == "safe_only_instruction_layer_selects_allowed_unit"
            and code_item.metadata.get("safe_semantic_proof_complete") is True
            and bool(code_item.metadata.get("safe_execution_unit"))
            and bool(code_item.metadata.get("safe_unit_sha256"))
        )

    def _material_actions(
        self,
        item: RepairItem,
        *,
        overprivileged_only: bool,
    ) -> set[tuple[tuple[str, str], ...]]:
        descriptor_contexts = item.metadata.get("descriptor_contexts")
        if not isinstance(descriptor_contexts, list):
            return set()
        action_fields = (
            "operation",
            "object",
            "source",
            "scope",
            "destination",
            "side_effect",
        )
        actions: set[tuple[tuple[str, str], ...]] = set()
        for descriptor in descriptor_contexts:
            if not isinstance(descriptor, dict):
                continue
            if (
                overprivileged_only
                and descriptor.get("final_verdict") != "overprivileged"
            ):
                continue
            material_actions = descriptor.get("material_action_instances")
            if not isinstance(material_actions, list):
                continue
            for action in material_actions:
                if not isinstance(action, dict):
                    continue
                normalized = tuple(
                    (
                        field_name,
                        self._normalize_semantic_value(
                            action.get(field_name)
                        ),
                    )
                    for field_name in action_fields
                )
                if normalized[0][1]:
                    actions.add(normalized)
        return actions

    def _normalize_semantic_value(self, value: object) -> str:
        return " ".join(str(value or "").casefold().split())
