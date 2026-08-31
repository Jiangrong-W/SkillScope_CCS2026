from __future__ import annotations

from pathlib import Path

from skillscope.common.config import AppConfig
from skillscope.common.io import ensure_directory, write_json
from skillscope.module3_control_flow_constrained_repair import ControlFlowConstrainedRepairService


class RepairPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.service = ControlFlowConstrainedRepairService(config)

    def run(self, skill_root: Path, *, user_prompts: list[str] | None = None) -> dict[str, object]:
        if self.config.validation_mode != "dynamic":
            raise ValueError(
                "Repair requires dynamic replay validation; static validation "
                "is a predictive fallback for `skillscope validate` and cannot "
                "establish a successful repair."
            )
        run = self.service.repair(skill_root, user_prompts=user_prompts)
        output_dir = ensure_directory(self.config.artifact_dir_for(run.validation_run.analysis.bundle.bundle_id, "repair"))
        for legacy_name in ("original_candidates.json", "patched_candidates.json"):
            legacy_path = output_dir / legacy_name
            if legacy_path.exists():
                legacy_path.unlink()

        write_json(output_dir / "repair_plan.json", run.plan)
        write_json(output_dir / "repair_outcome.json", run.outcome)
        write_json(output_dir / "repair_validation_report.json", run.outcome.validation)
        write_json(output_dir / "original_overreaches.json", run.validation_run.analysis.candidates)
        write_json(output_dir / "original_necessity_decisions.json", run.validation_run.validation.decisions)
        write_json(
            output_dir / "original_authorization_decisions.json",
            run.validation_run.validation.authorization_decisions,
        )
        write_json(output_dir / "original_final_verdicts.json", run.validation_run.validation.final_verdicts)
        write_json(output_dir / "original_task_context_descriptors.json", run.validation_run.validation.descriptors)
        if run.patched_analysis is not None:
            write_json(output_dir / "patched_overreaches.json", run.patched_analysis.candidates)
        if run.patched_execution_records is not None:
            write_json(output_dir / "patched_execution_records.json", run.patched_execution_records)

        summary = {
            "skill_id": run.validation_run.analysis.bundle.bundle_id,
            "repair_item_count": len(run.plan.items),
            "patched_bundle_path": run.outcome.patched_bundle_path,
            "remaining_overreach_count": run.outcome.validation.remaining_overreach_count if run.outcome.validation else None,
            "overprivileged_count_after_repair": (
                run.outcome.validation.overprivileged_count if run.outcome.validation else None
            ),
            "core_preserved_count_after_repair": (
                run.outcome.validation.core_preserved_count if run.outcome.validation else None
            ),
            "goal_satisfied_count_after_repair": (
                run.outcome.validation.goal_satisfied_count if run.outcome.validation else None
            ),
            "repair_success": (
                run.outcome.validation.remaining_overreach_count == 0
                if run.outcome.validation
                else False
            ),
            "explicit_user_prompt_count": len(user_prompts or []),
            "validation_mode": self.config.validation_mode,
            "artifact_dir": str(output_dir),
            "llm_enabled": self.config.llm.enabled,
            "llm_debug_log_path": str(self.config.llm.debug_log_path) if self.config.llm.debug_enabled and self.config.llm.debug_log_path else None,
        }
        write_json(output_dir / "summary.json", summary)
        return summary
