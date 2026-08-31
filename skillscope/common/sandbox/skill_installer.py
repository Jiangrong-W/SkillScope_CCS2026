from __future__ import annotations

import hashlib
from pathlib import Path

from skillscope.common.config import AppConfig
from skillscope.common.llm import PromptAssetLoader, StructuredLLMClient
from skillscope.common.models import InstalledSkill
from skillscope.module1_candidate_extraction.bundle_loader import SkillBundleLoader
from skillscope.module1_candidate_extraction.code_graph_builder import CodeGraphBuilder
from skillscope.module1_candidate_extraction.instruction_graph_builder import InstructionGraphBuilder
from skillscope.module1_candidate_extraction.instruction_semantic_normalizer import InstructionSemanticNormalizer
from skillscope.module1_candidate_extraction.skill_profile_extractor import SkillProfileExtractor
from skillscope.module1_candidate_extraction.ueg_composer import UEGComposer


class SkillInstaller:
    def __init__(
        self,
        config: AppConfig,
        *,
        llm_client: StructuredLLMClient,
        prompt_loader: PromptAssetLoader,
    ) -> None:
        self.config = config
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

    def install(self, skill_root: Path) -> InstalledSkill:
        bundle = self.bundle_loader.load(skill_root)
        profile = self.profile_extractor.extract(bundle)
        instruction_graph = self.instruction_graph_builder.build(bundle)
        code_graphs = self.code_graph_builder.build(bundle)
        ueg = self.ueg_composer.compose(bundle, instruction_graph, code_graphs)
        install_hash = hashlib.sha1(str(skill_root.resolve()).encode("utf-8")).hexdigest()[:12]
        install_id = f"installed:{bundle.bundle_id}:{install_hash}"
        return InstalledSkill(
            install_id=install_id,
            bundle=bundle,
            profile=profile,
            instruction_graph=instruction_graph,
            code_graphs=code_graphs,
            ueg=ueg,
            metadata={
                "install_strategy": "bundle_compiled_into_agent_runtime",
                "installed_root": str(skill_root.resolve()),
                "instruction_node_count": len(instruction_graph.nodes),
                "code_graph_count": len(code_graphs),
                "ueg_node_count": len(ueg.nodes),
            },
        )
