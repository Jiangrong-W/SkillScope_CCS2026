from __future__ import annotations

from pathlib import Path

from skillscope.common.config import AppConfig
from skillscope.common.io import ensure_directory, write_json
from skillscope.module2_action_necessity_validation import ActionNecessityValidationService


def task_context_summary(validation) -> dict[str, dict[str, int]]:
    """Keep user-requested contexts separate from generated diagnostic tasks."""
    groups = {
        "explicit_user_prompt": {"task_count": 0, "triggered_task_count": 0,
                                 "final_verdict_count": 0, "overprivileged_count": 0,
                                 "inconclusive_count": 0, "not_overprivileged_count": 0},
        "representative": {"task_count": 0, "triggered_task_count": 0,
                           "final_verdict_count": 0, "overprivileged_count": 0,
                           "inconclusive_count": 0, "not_overprivileged_count": 0},
    }
    source_by_id = {}
    for task in validation.tasks:
        source = ("explicit_user_prompt" if "user" in task.generation_strategy
                  else "representative")
        source_by_id[task.task_id] = source
        groups[source]["task_count"] += 1
    for evidence in validation.trigger_evidence:
        if evidence.triggered and evidence.task_id in source_by_id:
            groups[source_by_id[evidence.task_id]]["triggered_task_count"] += 1
    for verdict in validation.final_verdicts:
        if verdict.task_id not in source_by_id:
            continue
        group = groups[source_by_id[verdict.task_id]]
        group["final_verdict_count"] += 1
        key = f"{verdict.label}_count"
        if key in group:
            group[key] += 1
    return groups


def coverage_summary(analysis, validation) -> dict[str, object]:
    candidates = {candidate.candidate_id for candidate in analysis.candidates}
    tasked = {task.candidate_id for task in validation.tasks}
    triggered = {e.candidate_id for e in validation.trigger_evidence if e.triggered}
    conclusive = {v.candidate_id for v in validation.final_verdicts
                  if v.label in {"overprivileged", "not_overprivileged"}}
    truncated = bool(validation.metadata.get("chain_enumeration_truncated"))
    return {
        "candidate_ids_without_tasks": sorted(candidates - tasked),
        "untriggered_candidate_ids": sorted(candidates - triggered),
        "unvalidated_candidate_ids": sorted(candidates - conclusive),
        "chain_enumeration_truncated": truncated,
        "coverage_incomplete": bool(
            candidates - conclusive or truncated
            or sum(e.triggered for e in validation.trigger_evidence) < len(validation.tasks)
        ),
    }


class ValidatePipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.service = ActionNecessityValidationService(config)

    def run(self, skill_root: Path, *, user_prompts: list[str] | None = None) -> dict[str, object]:
        run = self.service.validate(skill_root, user_prompts=user_prompts)
        output_dir = ensure_directory(self.config.artifact_dir_for(run.analysis.bundle.bundle_id, "validate"))

        write_json(output_dir / "bundle.json", run.analysis.bundle)
        write_json(output_dir / "profile.json", run.analysis.profile)
        write_json(output_dir / "instruction_graph.json", run.analysis.instruction_graph)
        write_json(output_dir / "code_graphs.json", run.analysis.code_graphs)
        write_json(output_dir / "ueg.json", run.analysis.ueg)
        write_json(output_dir / "candidates.json", run.analysis.candidates)
        write_json(output_dir / "legitimate_chains.json", run.validation.legitimate_chains)
        write_json(output_dir / "tasks.json", run.validation.tasks)
        write_json(output_dir / "trigger_evidence.json", run.validation.trigger_evidence)
        write_json(output_dir / "replay_pairs.json", run.validation.replay_pairs)
        write_json(output_dir / "necessity_decisions.json", run.validation.decisions)
        write_json(output_dir / "authorization_decisions.json", run.validation.authorization_decisions)
        write_json(output_dir / "final_verdicts.json", run.validation.final_verdicts)
        write_json(output_dir / "task_context_descriptors.json", run.validation.descriptors)
        write_json(output_dir / "validation_metadata.json", run.validation.metadata)
        write_json(output_dir / "original_execution_records.json",
                   [entry.get("record", entry) for entry in
                    run.validation.metadata.get("original_execution_records", [])])

        unnecessary_count = sum(1 for decision in run.validation.decisions if decision.label == "unnecessary")
        unauthorized_count = sum(
            1 for decision in run.validation.authorization_decisions if decision.label == "unauthorized"
        )
        overprivileged_count = sum(
            1 for verdict in run.validation.final_verdicts if verdict.label == "overprivileged"
        )
        triggered_count = sum(evidence.triggered for evidence in run.validation.trigger_evidence)
        final_count = len(run.validation.final_verdicts)
        inconclusive_count = sum(v.label == "inconclusive" for v in run.validation.final_verdicts)
        validation_status = (
            "static_prediction" if self.config.validation_mode != "dynamic"
            else "no_material_trigger" if not triggered_count
            else "incomplete_validation" if final_count < triggered_count
            else "overprivilege_reported" if overprivileged_count
            else "inconclusive" if inconclusive_count
            else "validated_triggered_contexts"
        )
        summary = {
            "skill_id": run.analysis.bundle.bundle_id,
            "candidate_count": len(run.analysis.candidates),
            "task_count": len(run.validation.tasks),
            "decision_count": len(run.validation.decisions),
            "unnecessary_count": unnecessary_count,
            "unauthorized_count": unauthorized_count,
            "overprivileged_count": overprivileged_count,
            "final_verdict_count": final_count,
            "validation_status": validation_status,
            **coverage_summary(run.analysis, run.validation),
            "untriggered_task_count": len(run.validation.tasks) - triggered_count,
            "task_context_summary": task_context_summary(run.validation),
            "inconclusive_final_count": sum(
                1 for verdict in run.validation.final_verdicts if verdict.label == "inconclusive"
            ),
            "triggered_task_count": sum(
                1 for evidence in run.validation.trigger_evidence if evidence.triggered
            ),
            "descriptor_count": len(run.validation.descriptors),
            "explicit_user_prompt_count": len(user_prompts or []),
            "validation_mode": self.config.validation_mode,
            "artifact_dir": str(output_dir),
            "llm_enabled": self.config.llm.enabled,
            "llm_debug_log_path": str(self.config.llm.debug_log_path) if self.config.llm.debug_enabled and self.config.llm.debug_log_path else None,
        }
        write_json(output_dir / "summary.json", summary)
        return summary
