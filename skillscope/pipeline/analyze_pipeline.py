from __future__ import annotations

from pathlib import Path

from skillscope.common.config import AppConfig
from skillscope.common.io import ensure_directory, write_json
from skillscope.common.models import CandidateExtractionResult
from skillscope.module1_candidate_extraction import CandidateExtractionService


class AnalyzePipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.service = CandidateExtractionService(config)

    def run(self, skill_root: Path) -> dict[str, object]:
        result: CandidateExtractionResult = self.service.run(skill_root)
        output_dir = ensure_directory(self.config.artifact_dir_for(result.bundle.bundle_id, "analyze"))

        write_json(output_dir / "bundle.json", result.bundle)
        write_json(output_dir / "profile.json", result.profile)
        write_json(output_dir / "instruction_graph.json", result.instruction_graph)
        write_json(output_dir / "code_graphs.json", result.code_graphs)
        write_json(output_dir / "ueg.json", result.ueg)
        write_json(output_dir / "candidates.json", result.candidates)

        summary = {
            "skill_id": result.bundle.bundle_id,
            "candidate_count": len(result.candidates),
            "ambiguous_candidate_count": sum(
                1 for candidate in result.candidates if candidate.retained_due_to_ambiguity
            ),
            "artifact_dir": str(output_dir),
            "llm_enabled": self.config.llm.enabled,
            "llm_debug_log_path": str(self.config.llm.debug_log_path) if self.config.llm.debug_enabled and self.config.llm.debug_log_path else None,
        }
        write_json(output_dir / "summary.json", summary)
        return summary
