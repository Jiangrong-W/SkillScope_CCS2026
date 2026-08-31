from __future__ import annotations

import base64
import re
from pathlib import Path

from skillscope.common.models import (
    CandidateAction,
    CandidateExtractionResult,
    LegitimateActionChain,
    ResourceFixture,
    TaskSpec,
)

from .mcp_fixture_tools import MCPBackedFixtureToolchain


RESOURCE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:\./)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.(?:log|txt|json|csv|md|yaml|yml)\b",
    re.IGNORECASE,
)


class ResourceFixtureBuilder:
    """Derive deterministic, bundle-local resources for a candidate-reaching task."""

    MATERIALIZABLE_TYPES = {"file", "text_file", "generated_file", "existing_file"}

    def build(
        self,
        *,
        analysis: CandidateExtractionResult,
        candidate: CandidateAction,
        chain: LegitimateActionChain,
        task: TaskSpec,
    ) -> list[ResourceFixture]:
        fixtures = list(task.fixtures)
        seen = {(fixture.fixture_type, fixture.target) for fixture in fixtures}
        relevant_text = " ".join(
            [
                task.prompt,
                task.task_summary,
                *chain.summaries,
                *chain.predicate_context,
                candidate.summary,
            ]
        )
        node_text = " ".join(
            filter(
                None,
                [
                    getattr(analysis.ueg.node_by_id(node_id), "raw_text", None)
                    for node_id in chain.node_ids
                ],
            )
        )
        combined_text = f"{relevant_text} {node_text}"
        artifact_by_path = {artifact.relative_path: artifact for artifact in analysis.bundle.artifacts}
        excluded_paths = {
            artifact.relative_path
            for artifact in analysis.bundle.instruction_files + analysis.bundle.script_files
        }

        for resource in analysis.bundle.resource_files:
            basename = Path(resource.relative_path).name
            if resource.relative_path not in combined_text and basename not in combined_text:
                continue
            key = ("existing_file", resource.relative_path)
            if key in seen:
                continue
            content, metadata = self._artifact_content(resource.absolute_path, resource.is_binary)
            fixtures.append(
                ResourceFixture(
                    fixture_id=self._fixture_id(task, len(fixtures)),
                    fixture_type="existing_file",
                    target=resource.relative_path,
                    content=content,
                    source=resource.absolute_path,
                    required=True,
                    metadata={
                        "derivation": "referenced_bundle_resource",
                        **MCPBackedFixtureToolchain.metadata_for(
                            "existing_file",
                            resource.relative_path,
                        ),
                        **metadata,
                    },
                )
            )
            seen.add(key)

        for raw_path in RESOURCE_PATH_RE.findall(combined_text):
            relative_path = raw_path.removeprefix("./")
            if relative_path in excluded_paths:
                continue
            existing_artifact = artifact_by_path.get(relative_path)
            fixture_type = "existing_file" if existing_artifact is not None else "generated_file"
            key = (fixture_type, relative_path)
            if key in seen:
                continue
            if existing_artifact is not None:
                content, content_metadata = self._artifact_content(
                    existing_artifact.absolute_path,
                    existing_artifact.is_binary,
                )
            else:
                content = self._default_content(
                    task=task,
                    candidate=candidate,
                    target=relative_path,
                )
                content_metadata = {}
            fixtures.append(
                ResourceFixture(
                    fixture_id=self._fixture_id(task, len(fixtures)),
                    fixture_type=fixture_type,
                    target=relative_path,
                    content=content,
                    source=existing_artifact.absolute_path if existing_artifact is not None else None,
                    required=True,
                    metadata={
                        "derivation": "candidate_reaching_path_reference",
                        **MCPBackedFixtureToolchain.metadata_for(
                            fixture_type,
                            relative_path,
                        ),
                        **content_metadata,
                    },
                )
            )
            seen.add(key)

        return fixtures

    def _artifact_content(
        self,
        absolute_path: str,
        is_binary: bool,
    ) -> tuple[str, dict[str, str]]:
        path = Path(absolute_path)
        try:
            if is_binary:
                return (
                    base64.b64encode(path.read_bytes()).decode("ascii"),
                    {"encoding": "base64"},
                )
            return path.read_text(encoding="utf-8"), {}
        except (OSError, UnicodeDecodeError):
            return "", {"fixture_read_error": "true"}

    def _fixture_id(self, task: TaskSpec, index: int) -> str:
        return f"{task.task_id}-fixture-{index + 1:03d}"

    def _default_content(self, *, task: TaskSpec, candidate: CandidateAction, target: str) -> str:
        suffix = Path(target).suffix.lower()
        if suffix == ".json":
            return (
                '{\n'
                f'  "task_id": "{task.task_id}",\n'
                f'  "candidate_id": "{candidate.candidate_id}",\n'
                '  "fixture": true\n'
                '}\n'
            )
        if suffix == ".csv":
            return f"task_id,candidate_id\n{task.task_id},{candidate.candidate_id}\n"
        return (
            f"SkillScope fixture for task {task.task_id}\n"
            f"Candidate under validation: {candidate.candidate_id}\n"
        )
