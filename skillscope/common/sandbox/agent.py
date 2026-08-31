from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from skillscope.common.config import AppConfig
from skillscope.common.llm import DisabledLLMClient, PromptAssetLoader, StructuredLLMClient
from skillscope.common.models import ExecutionRecord, InstalledSkill, ResourceFixture

from .agent_runtime import TracedSkillAgentRuntime
from .policy import SandboxPolicy, SandboxPolicyError
from .runtime import SandboxedPythonRunner, create_isolated_skill_copy
from .skill_installer import SkillInstaller
from .task_planner import TaskConditionedInstructionPlanner
from .tooling import (
    InlineCommandTool,
    NodeScriptTool,
    PythonScriptTool,
    ShellScriptTool,
    ToolDispatcher,
    ToolRegistry,
)


class _FixtureToolchain(Protocol):
    def materialize(
        self,
        *,
        sandbox_root: Path,
        fixture: ResourceFixture,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class _FixtureMaterializationSummary:
    requested_count: int
    materialized_count: int
    manifest_sha256: str
    optional_failures: tuple[str, ...] = ()


class FixtureMaterializationError(RuntimeError):
    """A required task fixture could not be created inside the sandbox."""


class SandboxedSkillAgent:
    def __init__(
        self,
        config_or_project_root: AppConfig | Path,
        *,
        llm_client: StructuredLLMClient | None = None,
        prompt_loader: PromptAssetLoader | None = None,
        fixture_toolchain: _FixtureToolchain | None = None,
    ) -> None:
        if isinstance(config_or_project_root, AppConfig):
            self.config = config_or_project_root
        else:
            self.config = AppConfig.from_project_root(Path(config_or_project_root))

        if self.config.sandbox_allow_network:
            raise SandboxPolicyError(
                "Real network access is not permitted during SkillScope "
                "execution. Use an explicit API fixture instead."
            )
        self.project_root = self.config.project_root
        self.llm_client = llm_client or DisabledLLMClient()
        self.prompt_loader = prompt_loader or PromptAssetLoader(self.project_root)
        self.sandbox_policy = SandboxPolicy(
            allow_network=self.config.sandbox_allow_network,
            require_os_isolation=self.config.sandbox_require_os_isolation,
            timeout_seconds=self.config.sandbox_timeout_seconds,
        )
        self.python_runner = SandboxedPythonRunner(self.project_root, policy=self.sandbox_policy)
        self.tool_dispatcher = ToolDispatcher(
            ToolRegistry(
                [
                    PythonScriptTool(self.python_runner),
                    ShellScriptTool(self.sandbox_policy),
                    NodeScriptTool(self.sandbox_policy),
                    InlineCommandTool(self.sandbox_policy),
                ]
            )
        )
        self.runtime = TracedSkillAgentRuntime(
            self.tool_dispatcher,
            llm_client=self.llm_client,
            prompt_loader=self.prompt_loader,
        )
        self.installer = SkillInstaller(
            self.config,
            llm_client=self.llm_client,
            prompt_loader=self.prompt_loader,
        )
        self._installed_skill_cache: dict[str, InstalledSkill] = {}
        self.execution_planner = TaskConditionedInstructionPlanner(
            llm_client=self.llm_client,
            prompt_loader=self.prompt_loader,
        )
        if fixture_toolchain is None:
            # Import lazily to avoid a package-initialization cycle: Module 2
            # depends on this sandbox agent, while its fixture toolchain is the
            # concrete materializer used by the agent at execution time.
            from skillscope.module2_action_necessity_validation.mcp_fixture_tools import (
                MCPBackedFixtureToolchain,
            )

            fixture_toolchain = MCPBackedFixtureToolchain()
        self.fixture_toolchain = fixture_toolchain

    def install_skill(self, skill_root: Path) -> InstalledSkill:
        resolved_root = str(skill_root.resolve())
        installed_skill = self._installed_skill_cache.get(resolved_root)
        if installed_skill is not None:
            return installed_skill
        installed_skill = self.installer.install(skill_root)
        self._installed_skill_cache[resolved_root] = installed_skill
        return installed_skill

    def execute(
        self,
        *,
        prompt: str,
        run_id: str,
        mode: str,
        skill_root: Path | None = None,
        installed_skill: InstalledSkill | None = None,
        instruction_node_ids: list[str] | None = None,
        fixtures: list[ResourceFixture] | None = None,
    ) -> ExecutionRecord:
        if installed_skill is None:
            if skill_root is None:
                raise ValueError("Either skill_root or installed_skill must be provided.")
            installed_skill = self.install_skill(skill_root)
        return self.execute_installed_skill(
            installed_skill=installed_skill,
            prompt=prompt,
            run_id=run_id,
            mode=mode,
            instruction_node_ids=instruction_node_ids,
            fixtures=fixtures,
        )

    def execute_installed_skill(
        self,
        *,
        installed_skill: InstalledSkill,
        prompt: str,
        run_id: str,
        mode: str,
        instruction_node_ids: list[str] | None = None,
        fixtures: list[ResourceFixture] | None = None,
    ) -> ExecutionRecord:
        if installed_skill.ueg is None:
            raise ValueError("Installed skill is missing a unified execution graph.")

        source_bundle_root = Path(installed_skill.bundle.root_path)
        sandbox_root = create_isolated_skill_copy(source_bundle_root)
        try:
            fixture_summary = self._materialize_fixtures(
                sandbox_root,
                fixtures or [],
            )
            plan = (
                None
                if instruction_node_ids is not None
                else self.execution_planner.plan(
                    profile=installed_skill.profile,
                    ueg=installed_skill.ueg,
                    prompt=prompt,
                )
            )
            instruction_plan = instruction_node_ids or (
                plan.node_ids
                if plan is not None
                else self._full_instruction_plan(installed_skill)
            )
            runtime_outcome = self.runtime.execute(
                source_bundle_root=source_bundle_root,
                sandbox_root=sandbox_root,
                ueg=installed_skill.ueg,
                prompt=prompt,
                run_id=run_id,
                mode=mode,
                instruction_node_ids=instruction_plan,
            )
        finally:
            shutil.rmtree(sandbox_root.parent, ignore_errors=True)

        # Runtime evidence is authoritative for security-sensitive execution
        # fields such as final_output_grounded.  Skill-controlled metadata must
        # not overwrite facts observed by the tested-agent runtime.
        metadata = dict(installed_skill.metadata)
        metadata.update(runtime_outcome.metadata)
        metadata.update(
            {
                "install_id": installed_skill.install_id,
                "installed_skill_root": installed_skill.bundle.root_path,
                "execution_subject": "installed_skill_bundle",
                "plan_strategy": plan.strategy if plan is not None else "explicit_override",
                "selected_instruction_node_ids": instruction_plan,
                "plan_notes": plan.notes if plan is not None else [],
                "fixture_count": fixture_summary.requested_count,
                "fixture_materialized_count": fixture_summary.materialized_count,
                "fixture_manifest_sha256": fixture_summary.manifest_sha256,
                "fixture_optional_failures": list(fixture_summary.optional_failures),
                "sandbox_backend": self.sandbox_policy.backend,
                "sandbox_network_allowed": self.sandbox_policy.allow_network,
            }
        )

        return ExecutionRecord(
            run_id=run_id,
            mode=mode,
            prompt=prompt,
            trace=runtime_outcome.trace,
            raw_trace=runtime_outcome.raw_trace,
            final_output=runtime_outcome.final_output,
            stdout=runtime_outcome.stdout,
            stderr=runtime_outcome.stderr,
            bundle_root=str(source_bundle_root),
            status=runtime_outcome.status,
            notes=runtime_outcome.notes,
            executed_node_ids=runtime_outcome.executed_node_ids,
            metadata=metadata,
        )

    def _full_instruction_plan(self, installed_skill: InstalledSkill) -> list[str]:
        if installed_skill.ueg is None:
            return []
        plan: list[tuple[int, str]] = []
        for node in installed_skill.ueg.nodes:
            if node.layer != "instruction" or node.node_type in {"ENTRY", "EXIT"}:
                continue
            line_number = node.source_range.start_line if node.source_range is not None else 10**9
            plan.append((line_number, node.node_id))
        return [node_id for _, node_id in sorted(plan)]

    def _materialize_fixtures(
        self,
        sandbox_root: Path,
        fixtures: list[ResourceFixture],
    ) -> _FixtureMaterializationSummary:
        from skillscope.module2_action_necessity_validation.mcp_fixture_tools import (
            fixture_manifest_payload,
        )

        # The manifest is a pure projection of the task fixtures.  In
        # particular, it excludes sandbox-specific artifact paths so original
        # and replay executions receive byte-identical fixture manifests.
        manifest = [fixture_manifest_payload(fixture) for fixture in fixtures]
        materialized_count = 0
        optional_failures: list[str] = []
        for fixture in fixtures:
            try:
                self.fixture_toolchain.materialize(
                    sandbox_root=sandbox_root,
                    fixture=fixture,
                )
            except Exception as exc:
                message = (
                    f"Fixture {fixture.fixture_id!r} "
                    f"({fixture.fixture_type} -> {fixture.target}) could not be "
                    f"materialized: {exc}"
                )
                if fixture.required:
                    raise FixtureMaterializationError(
                        f"Required {message[0].lower()}{message[1:]}"
                    ) from exc
                optional_failures.append(message)
            else:
                materialized_count += 1

        manifest_root = sandbox_root / ".skillscope"
        manifest_root.mkdir(parents=True, exist_ok=True)
        manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
        (manifest_root / "fixtures.json").write_text(manifest_text, encoding="utf-8")
        return _FixtureMaterializationSummary(
            requested_count=len(fixtures),
            materialized_count=materialized_count,
            manifest_sha256=hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
            optional_failures=tuple(optional_failures),
        )
