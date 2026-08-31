from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from skillscope.common.config import AppConfig
from skillscope.common.llm import PromptAssetLoader, build_llm_client
from skillscope.common.models import CandidateExtractionResult, ExecutionRecord, RepairOutcome, RepairPlan
from skillscope.module2_action_necessity_validation.service import (
    ActionNecessityValidationRun,
    ActionNecessityValidationService,
)
from skillscope.module2_action_necessity_validation.output_comparator import OutputComparator

from .bundle_projector import BundleProjector
from .code_rewriter import CodeRewriter
from .instruction_rewriter import InstructionRewriter
from .overreach_pruner import OverreachPruner
from .repair_planner import RepairPlanner
from .repair_validator import RepairValidator
from .task_conditioned_guarder import TaskConditionedGuarder


@dataclass(slots=True)
class ControlFlowConstrainedRepairRun:
    validation_run: ActionNecessityValidationRun
    plan: RepairPlan
    outcome: RepairOutcome
    patched_analysis: CandidateExtractionResult | None = None
    patched_execution_records: list[ExecutionRecord] | None = None


class ControlFlowConstrainedRepairService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        llm_client = build_llm_client(config)
        prompt_loader = PromptAssetLoader(config.project_root)

        self.validation_service = ActionNecessityValidationService(config)
        self.repair_planner = RepairPlanner(
            llm_client=llm_client,
            prompt_loader=prompt_loader,
            pruner=OverreachPruner(),
            guarder=TaskConditionedGuarder(
                llm_client=llm_client,
                prompt_loader=prompt_loader,
            ),
        )
        self.bundle_projector = BundleProjector(
            instruction_rewriter=InstructionRewriter(
                llm_client=llm_client,
                prompt_loader=prompt_loader,
            ),
            code_rewriter=CodeRewriter(
                llm_client=llm_client,
                prompt_loader=prompt_loader,
            ),
        )
        self.repair_validator = RepairValidator(
            candidate_service=self.validation_service.candidate_service,
            sandboxed_agent=self.validation_service.replay_runner.sandboxed_agent,
            output_comparator=OutputComparator(),
        )

    def repair(
        self,
        skill_root: Path,
        user_prompts: list[str] | None = None,
    ) -> ControlFlowConstrainedRepairRun:
        validation_run = self.validation_service.validate(
            skill_root,
            user_prompts=user_prompts,
        )
        plan = self.repair_planner.plan(
            analysis=validation_run.analysis,
            validation=validation_run.validation,
        )

        patched_bundle_root = self.config.artifact_dir_for(validation_run.analysis.bundle.bundle_id, "repair") / "patched_bundle"
        outcome = self.bundle_projector.project(
            bundle=validation_run.analysis.bundle,
            plan=plan,
            output_root=patched_bundle_root,
        )
        validation_report, patched_analysis, patched_execution_records = self.repair_validator.validate(
            original_run=validation_run,
            repair_plan=plan,
            patched_bundle_root=Path(outcome.patched_bundle_path or patched_bundle_root),
        )
        outcome.validation = validation_report
        outcome.metadata["module3_strategy"] = "control_flow_constrained_repair"

        return ControlFlowConstrainedRepairRun(
            validation_run=validation_run,
            plan=plan,
            outcome=outcome,
            patched_analysis=patched_analysis,
            patched_execution_records=patched_execution_records,
        )
