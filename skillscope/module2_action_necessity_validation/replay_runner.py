from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from skillscope.common.models import (
    CandidateAction,
    CandidateExtractionResult,
    ExecutionRecord,
    ReplayPairRecord,
    TaskSpec,
    TaskTriggerEvidence,
)
from skillscope.common.sandbox import SandboxedSkillAgent

from .candidate_ablation import CandidateAblation
from .trigger_detector import TaskTriggerDetector


@dataclass(slots=True)
class TriggerGatedReplayOutcome:
    """An original run and, only after a verified trigger, its ablation replay."""

    original: ExecutionRecord
    trigger_evidence: TaskTriggerEvidence
    replay_pair: ReplayPairRecord | None = None


class ReplayRunner:
    def __init__(
        self,
        *,
        ablation_builder: CandidateAblation | None = None,
        sandboxed_agent: SandboxedSkillAgent | None = None,
        trigger_detector: TaskTriggerDetector | None = None,
        force_chain_instruction_plan: bool = False,
    ) -> None:
        self.ablation_builder = ablation_builder or CandidateAblation()
        self.sandboxed_agent = sandboxed_agent or SandboxedSkillAgent(Path.cwd())
        self.trigger_detector = trigger_detector or TaskTriggerDetector()
        # Production must let the tested agent select a path from the prompt.
        # The override exists only for narrowly controlled deterministic tests.
        self.force_chain_instruction_plan = force_chain_instruction_plan

    def run_trigger_then_replay(
        self,
        *,
        analysis: CandidateExtractionResult,
        task: TaskSpec,
        candidate: CandidateAction,
    ) -> TriggerGatedReplayOutcome:
        """Run the original first and build no ablation unless it triggers the candidate."""

        try:
            original_installed_skill = self.sandboxed_agent.install_skill(
                Path(analysis.bundle.root_path)
            )
            original_instruction_plan = (
                self._instruction_plan(
                    task=task,
                    installed_skill=original_installed_skill,
                )
                if self.force_chain_instruction_plan
                else None
            )
            original_record = self.sandboxed_agent.execute_installed_skill(
                installed_skill=original_installed_skill,
                prompt=task.prompt,
                run_id=f"{task.task_id}:original",
                mode="original",
                instruction_node_ids=original_instruction_plan,
                fixtures=task.fixtures,
            )
            original_record.metadata["replay_strategy"] = (
                "candidate_trigger_verification_before_ablation"
            )
        except Exception as exc:
            original_record = self._build_failed_record(
                task=task,
                run_id=f"{task.task_id}:original",
                mode="original",
                bundle_root=analysis.bundle.root_path,
                error=f"original_execution_error={exc}",
            )

        trigger_evidence = self.trigger_detector.detect(
            candidate=candidate,
            task=task,
            record=original_record,
            ueg=analysis.ueg,
        )
        if not trigger_evidence.triggered:
            return TriggerGatedReplayOutcome(
                original=original_record,
                trigger_evidence=trigger_evidence,
                replay_pair=None,
            )

        # The ablation is deliberately constructed only after exact trigger evidence.
        ablation = self.ablation_builder.build_replay_variant(candidate, analysis.ueg)
        replay_bundle_root: Path | None = None
        try:
            replay_bundle_root = self.ablation_builder.materialize_replay_bundle(
                Path(analysis.bundle.root_path),
                ablation,
            )
            replay_installed_skill = self.sandboxed_agent.install_skill(replay_bundle_root)
            replay_instruction_plan = (
                self._instruction_plan(
                    task=task,
                    installed_skill=replay_installed_skill,
                )
                if self.force_chain_instruction_plan
                else None
            )
            replay_record = self.sandboxed_agent.execute_installed_skill(
                installed_skill=replay_installed_skill,
                prompt=task.prompt,
                run_id=f"{task.task_id}:replay",
                mode="replay",
                instruction_node_ids=replay_instruction_plan,
                fixtures=task.fixtures,
            )
            replay_record.metadata["replay_strategy"] = "repackaged_skill_bundle_replay"
            replay_record.metadata["ablated_candidate_id"] = candidate.candidate_id
            replay_record.metadata["ablated_bundle_root"] = str(replay_bundle_root)
            replay_record.metadata["replay_installed_skill_id"] = replay_installed_skill.install_id
            replay_record.metadata["ablation_applied"] = True
            replay_record.metadata["ablated_candidate_fingerprint_absent"] = (
                self._candidate_fingerprint_absent(
                    candidate=candidate,
                    original_ueg=analysis.ueg,
                    replay_ueg=replay_installed_skill.ueg,
                )
            )
        except Exception as exc:
            replay_record = self._build_failed_record(
                task=task,
                run_id=f"{task.task_id}:replay",
                mode="replay",
                bundle_root=(
                    str(replay_bundle_root)
                    if replay_bundle_root is not None
                    else analysis.bundle.root_path
                ),
                error=f"ablation_replay_error={exc}",
            )
        finally:
            if replay_bundle_root is not None:
                shutil.rmtree(replay_bundle_root.parent, ignore_errors=True)

        replay_pair = ReplayPairRecord(
            candidate_id=candidate.candidate_id,
            task_id=task.task_id,
            ablation=ablation,
            original=original_record,
            replay=replay_record,
        )
        return TriggerGatedReplayOutcome(
            original=original_record,
            trigger_evidence=trigger_evidence,
            replay_pair=replay_pair,
        )

    def run_pair(
        self,
        *,
        analysis: CandidateExtractionResult,
        task: TaskSpec,
        candidate: CandidateAction,
    ) -> ReplayPairRecord:
        """Compatibility wrapper that refuses to invent an ablation pair when untriggered."""

        outcome = self.run_trigger_then_replay(
            analysis=analysis,
            task=task,
            candidate=candidate,
        )
        if outcome.replay_pair is None:
            raise RuntimeError(
                "The original task did not trigger the candidate; no ablation replay was run."
            )
        return outcome.replay_pair

    def _instruction_plan(
        self,
        *,
        task: TaskSpec,
        installed_skill: object,
    ) -> list[str] | None:
        ueg = getattr(installed_skill, "ueg", None)
        if ueg is None:
            return None
        instruction_node_ids = [
            node_id
            for node_id in task.chain_node_ids
            if (
                (node := ueg.node_by_id(node_id)) is not None
                and node.layer == "instruction"
                and node.node_type not in {"ENTRY", "EXIT"}
            )
        ]
        return instruction_node_ids or None

    def _candidate_fingerprint_absent(
        self,
        *,
        candidate: CandidateAction,
        original_ueg: object,
        replay_ueg: object,
    ) -> bool:
        original_lookup = getattr(original_ueg, "node_by_id", None)
        replay_nodes = getattr(replay_ueg, "nodes", None)
        if not callable(original_lookup) or not isinstance(replay_nodes, list):
            return False
        original_node = original_lookup(candidate.node_id)
        if original_node is None:
            return False

        original_raw = self._normalized_fingerprint_text(
            getattr(original_node, "raw_text", None)
        )
        for replay_node in replay_nodes:
            if getattr(replay_node, "layer", None) != candidate.layer:
                continue
            if getattr(replay_node, "source_file", None) != getattr(
                original_node,
                "source_file",
                None,
            ):
                continue
            replay_raw = self._normalized_fingerprint_text(
                getattr(replay_node, "raw_text", None)
            )
            if original_raw and replay_raw == original_raw:
                return False
            if not original_raw and (
                getattr(replay_node, "summary", None)
                == getattr(original_node, "summary", None)
                and getattr(replay_node, "operation_type", None)
                == getattr(original_node, "operation_type", None)
            ):
                return False
        return True

    def _normalized_fingerprint_text(self, value: object) -> str:
        if not isinstance(value, str):
            return ""
        return " ".join(value.split())

    def _build_failed_record(
        self,
        *,
        task: TaskSpec,
        run_id: str,
        mode: str,
        bundle_root: str,
        error: str,
    ) -> ExecutionRecord:
        return ExecutionRecord(
            run_id=run_id,
            mode=mode,
            prompt=task.prompt,
            trace=[],
            raw_trace=[],
            final_output="",
            stdout="",
            stderr="",
            bundle_root=bundle_root,
            status="failed",
            notes=[
                "Bundle execution failed before a comparable trace could be collected.",
                error,
            ],
            executed_node_ids=[],
            metadata={"execution_strategy": "failed_bundle_execution"},
        )
