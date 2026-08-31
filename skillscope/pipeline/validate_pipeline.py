from __future__ import annotations

from pathlib import Path

from skillscope.common.config import AppConfig
from skillscope.common.io import ensure_directory, write_json
from skillscope.module2_action_necessity_validation import ActionNecessityValidationService


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

        unnecessary_count = sum(1 for decision in run.validation.decisions if decision.label == "unnecessary")
        unauthorized_count = sum(
            1 for decision in run.validation.authorization_decisions if decision.label == "unauthorized"
        )
        overprivileged_count = sum(
            1 for verdict in run.validation.final_verdicts if verdict.label == "overprivileged"
        )
        summary = {
            "skill_id": run.analysis.bundle.bundle_id,
            "candidate_count": len(run.analysis.candidates),
            "task_count": len(run.validation.tasks),
            "decision_count": len(run.validation.decisions),
            "unnecessary_count": unnecessary_count,
            "unauthorized_count": unauthorized_count,
            "overprivileged_count": overprivileged_count,
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
