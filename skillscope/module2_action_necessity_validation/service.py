from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from skillscope.common.config import AppConfig
from skillscope.common.llm import PromptAssetLoader, build_llm_client
from skillscope.common.models import (
    AuthorizationDecision,
    CandidateExtractionResult,
    FinalVerdict,
    LegitimateActionChain,
    NecessityDecision,
    TaskSpec,
    ValidationResult,
)
from skillscope.common.sandbox import SandboxedSkillAgent
from skillscope.module1_candidate_extraction import CandidateExtractionService
from skillscope.module2_action_overprivilege_validation import (
    ActionTaskDescriptorBuilder,
    ActionTupleExtractor,
    AuthorizationJudge,
)

from .candidate_ablation import CandidateAblation
from .fixture_builder import ResourceFixtureBuilder
from .legitimate_chain_extractor import LegitimateChainExtractor
from .necessity_judge import NecessityJudge
from .output_comparator import OutputComparator
from .prompt_instantiator import PromptInstantiator
from .replay_runner import ReplayRunner
from .static_necessity_validator import StaticNecessityValidator
from .trace_normalizer import TraceNormalizer
from .trigger_detector import TaskTriggerDetector


@dataclass(slots=True)
class ActionNecessityValidationRun:
    analysis: CandidateExtractionResult
    validation: ValidationResult


class ActionNecessityValidationService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        llm_client = build_llm_client(config)
        prompt_loader = PromptAssetLoader(config.project_root)
        sandboxed_agent = SandboxedSkillAgent(
            config,
            llm_client=llm_client,
            prompt_loader=prompt_loader,
        )
        tuple_extractor = ActionTupleExtractor()

        self.candidate_service = CandidateExtractionService(config)
        self.chain_extractor = LegitimateChainExtractor()
        self.prompt_instantiator = PromptInstantiator(
            llm_client=llm_client,
            prompt_loader=prompt_loader,
        )
        self.fixture_builder = ResourceFixtureBuilder()
        self.replay_runner = ReplayRunner(
            ablation_builder=CandidateAblation(),
            sandboxed_agent=sandboxed_agent,
            trigger_detector=TaskTriggerDetector(),
        )
        self.necessity_judge = NecessityJudge(
            llm_client=llm_client,
            prompt_loader=prompt_loader,
            trace_normalizer=TraceNormalizer(),
            output_comparator=OutputComparator(),
        )
        self.static_validator = StaticNecessityValidator(
            llm_client=llm_client,
            prompt_loader=prompt_loader,
        )
        self.authorization_judge = AuthorizationJudge(
            llm_client=llm_client,
            prompt_loader=prompt_loader,
            tuple_extractor=tuple_extractor,
        )
        self.descriptor_builder = ActionTaskDescriptorBuilder(
            tuple_extractor=tuple_extractor,
        )

    def validate(
        self,
        skill_root: Path,
        user_prompts: list[str] | None = None,
    ) -> ActionNecessityValidationRun:
        """Validate representative graph tasks plus any explicit Skill+prompt contexts."""

        analysis = self.candidate_service.run(skill_root)
        legitimate_chains: list[LegitimateActionChain] = []
        tasks: list[TaskSpec] = []
        trigger_evidence = []
        replay_pairs = []
        decisions = []
        authorization_decisions = []
        final_verdicts = []
        descriptors = []
        chain_coverage_reports: list[dict[str, object]] = []
        validation_mode = self.config.validation_mode
        explicit_prompts = self._normalize_user_prompts(user_prompts)

        for candidate in analysis.candidates:
            candidate_chains = self.chain_extractor.extract(
                analysis.ueg,
                candidate,
            )
            coverage_report = getattr(
                self.chain_extractor,
                "last_coverage_report",
                None,
            )
            if isinstance(coverage_report, dict) and coverage_report:
                chain_coverage_reports.append(dict(coverage_report))
            # Guard the validation invariant at the service boundary, even if a
            # custom extractor is injected.
            candidate_chains = [
                chain
                for chain in candidate_chains
                if (
                    chain.reaches_candidate
                    and chain.candidate_position is not None
                    and candidate.node_id in chain.node_ids
                )
            ]
            legitimate_chains.extend(candidate_chains)

            representative_tasks = self.prompt_instantiator.instantiate(
                analysis.profile,
                candidate_chains,
                analysis.ueg,
            )
            user_context_tasks = self.prompt_instantiator.instantiate_user_prompts(
                chains=candidate_chains,
                user_prompts=explicit_prompts,
            )
            candidate_tasks = representative_tasks + user_context_tasks

            for task in candidate_tasks:
                chain = self._chain_for_task(task, candidate_chains)
                if chain is None:
                    continue
                task.fixtures = self.fixture_builder.build(
                    analysis=analysis,
                    candidate=candidate,
                    chain=chain,
                    task=task,
                )
                tasks.append(task)

                if validation_mode == "static":
                    # Static mode may predict components, but without a realized
                    # original action instance its authorization label remains
                    # explicitly inconclusive.
                    authorization = self.authorization_judge.judge(
                        analysis=analysis,
                        candidate=candidate,
                        task=task,
                        original_record=None,
                    )
                    authorization_decisions.append(authorization)
                    necessity = self.static_validator.judge(
                        analysis=analysis,
                        candidate=candidate,
                        task=task,
                    )
                    decisions.append(necessity)
                    final_verdict = self._static_inconclusive_verdict(
                        authorization,
                        necessity,
                    )
                    final_verdicts.append(final_verdict)
                    descriptors.append(
                        self.descriptor_builder.build(
                            analysis=analysis,
                            candidate=candidate,
                            task=task,
                            authorization=authorization,
                            necessity=necessity,
                            final_verdict=final_verdict,
                        )
                    )
                    continue

                replay_outcome = self.replay_runner.run_trigger_then_replay(
                    analysis=analysis,
                    task=task,
                    candidate=candidate,
                )
                trigger_evidence.append(replay_outcome.trigger_evidence)
                if replay_outcome.replay_pair is None:
                    # No authorization, ablation verdict, combined verdict, or
                    # descriptor is emitted for a task that did not trigger.
                    continue

                replay_pair = replay_outcome.replay_pair
                replay_pairs.append(replay_pair)
                authorization = self.authorization_judge.judge(
                    analysis=analysis,
                    candidate=candidate,
                    task=task,
                    original_record=replay_outcome.original,
                )
                authorization_decisions.append(authorization)
                necessity = self.necessity_judge.judge(
                    candidate,
                    task,
                    replay_pair,
                    replay_outcome.trigger_evidence,
                )
                decisions.append(necessity)
                final_verdict = FinalVerdict.combine(
                    authorization,
                    necessity,
                    privilege_relevant=candidate.privilege_relevant,
                    privilege_type=candidate.privilege_type,
                )
                final_verdicts.append(final_verdict)
                descriptors.append(
                    self.descriptor_builder.build(
                        analysis=analysis,
                        candidate=candidate,
                        task=task,
                        authorization=authorization,
                        necessity=necessity,
                        final_verdict=final_verdict,
                        original_record=replay_outcome.original,
                    )
                )

        validation = ValidationResult(
            bundle_id=analysis.bundle.bundle_id,
            legitimate_chains=legitimate_chains,
            tasks=tasks,
            trigger_evidence=trigger_evidence,
            replay_pairs=replay_pairs,
            decisions=decisions,
            authorization_decisions=authorization_decisions,
            final_verdicts=final_verdicts,
            descriptors=descriptors,
            metadata={
                "module2_strategy": (
                    "trigger_gated_bundle_ablation_with_authorization"
                    if validation_mode == "dynamic"
                    else "static_corepres_goalsat_with_authorization"
                ),
                "execution_runner": (
                    "installed_skill_original_then_trigger_gated_replay"
                    if validation_mode == "dynamic"
                    else "not_used_static_mode"
                ),
                "tested_agent_execution": (
                    "prompt_conditioned_installed_skill_runtime"
                    if validation_mode == "dynamic"
                    else "static_prediction_only"
                ),
                "legitimate_chain_filter": "candidate_reaching_chain_required",
                "chain_coverage_reports": chain_coverage_reports,
                "chain_enumeration_truncated": any(
                    report.get("truncated") is True
                    for report in chain_coverage_reports
                ),
                "candidate_count": len(analysis.candidates),
                "validation_mode": validation_mode,
                "representative_task_count": sum(
                    task.generation_strategy
                    != "user_supplied_candidate_reaching_context"
                    for task in tasks
                ),
                "user_prompt_task_count": sum(
                    task.generation_strategy
                    == "user_supplied_candidate_reaching_context"
                    for task in tasks
                ),
                "original_trigger_count": sum(
                    evidence.triggered for evidence in trigger_evidence
                ),
                "ablation_replay_count": len(replay_pairs),
                "static_mode_has_dynamic_trigger_evidence": False,
                "static_mode_has_confirmed_final_verdicts": False,
            },
        )
        return ActionNecessityValidationRun(analysis=analysis, validation=validation)

    def _static_inconclusive_verdict(
        self,
        authorization: AuthorizationDecision,
        necessity: NecessityDecision,
    ) -> FinalVerdict:
        """Keep static predictions from masquerading as runtime evidence."""

        if (
            authorization.candidate_id != necessity.candidate_id
            or authorization.task_id != necessity.task_id
        ):
            raise ValueError(
                "Static authorization and necessity predictions must describe "
                "the same candidate-task pair."
            )
        return FinalVerdict(
            candidate_id=authorization.candidate_id,
            task_id=authorization.task_id,
            label="inconclusive",
            authorization_label=authorization.label,
            necessity_label=necessity.label,
            reason=(
                "Static component predictions are advisory only. A confirmed "
                "A confirmed verdict requires a candidate-triggering original "
                "trace and its candidate-neutralized replay."
            ),
            overprivilege_reasons=[],
            confidence=0.0,
            uncertainty_flags=sorted(
                set(
                    [
                        *authorization.uncertainty_flags,
                        *necessity.uncertainty_flags,
                        "static_prediction_without_original_replay_evidence",
                    ]
                )
            ),
        )

    def _normalize_user_prompts(self, user_prompts: list[str] | None) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for prompt in user_prompts or []:
            value = str(prompt).strip()
            if value and value not in seen:
                seen.add(value)
                normalized.append(value)
        return normalized

    def _chain_for_task(
        self,
        task: TaskSpec,
        chains: list[LegitimateActionChain],
    ) -> LegitimateActionChain | None:
        for chain in chains:
            if (
                chain.candidate_id == task.candidate_id
                and chain.node_ids == task.chain_node_ids
            ):
                return chain
        return None
