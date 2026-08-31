from __future__ import annotations

from pathlib import Path

from skillscope.common.config import AppConfig
from skillscope.common.llm import PromptAssetLoader, build_llm_client
from skillscope.common.models import CandidateExtractionResult

from .action_consistency_classifier import ActionConsistencyClassifier
from .bundle_loader import SkillBundleLoader
from .candidate_extractor import CandidateExtractor
from .code_graph_builder import CodeGraphBuilder
from .instruction_graph_builder import InstructionGraphBuilder
from .instruction_semantic_normalizer import InstructionSemanticNormalizer
from .skill_profile_extractor import SkillProfileExtractor
from .ueg_composer import UEGComposer


class CandidateExtractionService:
    def __init__(self, config: AppConfig) -> None:
        llm_client = build_llm_client(config)
        prompt_loader = PromptAssetLoader(config.project_root)
        self.bundle_loader = SkillBundleLoader(config)
        self.profile_extractor = SkillProfileExtractor()
        self.instruction_graph_builder = InstructionGraphBuilder(
            InstructionSemanticNormalizer(
                llm_client=llm_client,
                prompt_loader=prompt_loader,
            )
        )
        self.code_graph_builder = CodeGraphBuilder()
        self.ueg_composer = UEGComposer()
        self.candidate_extractor = CandidateExtractor(
            ActionConsistencyClassifier(
                llm_client=llm_client,
                prompt_loader=prompt_loader,
            ),
            max_workers=config.llm.max_concurrency,
        )

    def run(self, skill_root: Path) -> CandidateExtractionResult:
        bundle = self.bundle_loader.load(skill_root)
        profile = self.profile_extractor.extract(bundle)
        instruction_graph = self.instruction_graph_builder.build(bundle)
        code_graphs = self.code_graph_builder.build(bundle)
        ueg = self.ueg_composer.compose(bundle, instruction_graph, code_graphs)
        candidates = self.candidate_extractor.extract(profile, ueg)
        return CandidateExtractionResult(
            bundle=bundle,
            profile=profile,
            instruction_graph=instruction_graph,
            code_graphs=code_graphs,
            ueg=ueg,
            candidates=candidates,
        )
