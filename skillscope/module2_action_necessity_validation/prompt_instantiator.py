from __future__ import annotations

import base64
import binascii
import re
from pathlib import PurePosixPath
from typing import Any

from skillscope.common.llm import (
    DisabledLLMClient,
    MAX_VALIDATED_LLM_ATTEMPTS,
    PromptAssetLoader,
    StructuredLLMClient,
)
from skillscope.common.models import (
    LegitimateActionChain,
    ResourceFixture,
    SkillProfile,
    TaskSpec,
    UnifiedExecutionGraph,
)

from .validated_llm import ValidatedLLMCaller, string_list
from .mcp_fixture_tools import MCPBackedFixtureToolchain


class PromptInstantiator:
    RESPONSE_KEYS = {
        "prompt",
        "task_summary",
        "candidate_trigger_rationale",
        "fixtures",
        "notes",
        "evidence_node_ids",
    }
    FIXTURE_KEYS = {
        "fixture_type",
        "target",
        "content",
        "required",
    }
    ALLOWED_FIXTURE_TYPES = {
        "api",
        "binary_file",
        "config",
        "document",
        "env",
        "file",
        "git",
        "image",
        "text_file",
        "generated_file",
        "existing_file",
    }
    FILE_FIXTURE_TYPES = {
        "binary_file",
        "config",
        "document",
        "file",
        "image",
        "text_file",
        "generated_file",
        "existing_file",
    }
    ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(
        self,
        llm_client: StructuredLLMClient | None = None,
        prompt_loader: PromptAssetLoader | None = None,
        prompt_asset: str = "prompts/module2_prompt_instantiation.md",
        max_llm_attempts: int = MAX_VALIDATED_LLM_ATTEMPTS,
    ) -> None:
        self.llm_client = llm_client or DisabledLLMClient()
        self.prompt_loader = prompt_loader
        self.prompt_asset = prompt_asset
        self.llm_caller = ValidatedLLMCaller(
            llm_client=self.llm_client,
            prompt_loader=self.prompt_loader,
            max_attempts=max_llm_attempts,
        )

    def instantiate(
        self,
        profile: SkillProfile,
        chains: list[LegitimateActionChain],
        ueg: UnifiedExecutionGraph,
    ) -> list[TaskSpec]:
        tasks: list[TaskSpec] = []
        for index, chain in enumerate(chains, start=1):
            if not chain.reaches_candidate or chain.candidate_position is None:
                continue
            tasks.append(self._instantiate_single(profile, chain, ueg, index))
        return tasks

    def instantiate_user_prompts(
        self,
        *,
        chains: list[LegitimateActionChain],
        user_prompts: list[str],
    ) -> list[TaskSpec]:
        """Bind caller-provided prompts to each candidate-reaching graph context."""

        normalized_prompts: list[str] = []
        seen: set[str] = set()
        for prompt in user_prompts:
            normalized = str(prompt).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            normalized_prompts.append(normalized)

        tasks: list[TaskSpec] = []
        for chain_ordinal, chain in enumerate(chains, start=1):
            if not chain.reaches_candidate or chain.candidate_position is None:
                continue
            for prompt_ordinal, prompt in enumerate(normalized_prompts, start=1):
                tasks.append(
                    TaskSpec(
                        task_id=(
                            f"{chain.candidate_id}-user-task-"
                            f"{chain_ordinal:03d}-{prompt_ordinal:03d}"
                        ),
                        candidate_id=chain.candidate_id,
                        prompt=prompt,
                        chain_node_ids=list(chain.node_ids),
                        chain_summaries=list(chain.summaries),
                        task_summary=prompt,
                        generation_strategy="user_supplied_candidate_reaching_context",
                        expected_candidate_node_id=chain.node_ids[chain.candidate_position],
                        trigger_required=True,
                        generation_notes=[
                            "Caller-provided user prompt retained verbatim.",
                            (
                                "Prompt is evaluated in a graph context that explicitly "
                                "contains the candidate; dynamic mode still requires an "
                                "observed original-run trigger."
                            ),
                        ],
                    )
                )
        return tasks

    def _instantiate_single(
        self,
        profile: SkillProfile,
        chain: LegitimateActionChain,
        ueg: UnifiedExecutionGraph,
        ordinal: int,
    ) -> TaskSpec:
        llm_task = self._instantiate_with_llm(profile, chain, ueg, ordinal)
        if llm_task is not None:
            return llm_task
        return self._instantiate_with_fallback(profile, chain, ordinal)

    def _instantiate_with_llm(
        self,
        profile: SkillProfile,
        chain: LegitimateActionChain,
        ueg: UnifiedExecutionGraph,
        ordinal: int,
    ) -> TaskSpec | None:
        chain_payload: list[dict[str, Any]] = []
        for index, node_id in enumerate(chain.node_ids):
            node = ueg.node_by_id(node_id)
            if node is None:
                continue
            chain_payload.append(
                {
                    "node_id": node.node_id,
                    "is_candidate": index == chain.candidate_position,
                    "layer": node.layer,
                    "node_type": node.node_type,
                    "summary": node.summary,
                    "raw_text": node.raw_text,
                    "operation_type": node.operation_type,
                    "object_ref": node.object_ref,
                    "source_file": node.source_file,
                    "risk_tags": node.risk_tags,
                    "attributes": node.attributes,
                }
            )
        payload = {
            "declared_skill_profile": {
                "name": profile.name,
                "description": profile.description,
                "use_when": profile.use_when,
                "summary": profile.summary,
                "declared_outputs": profile.declared_outputs,
                "declared_data_scope": profile.declared_data_scope,
                "declared_execution_scope": profile.declared_execution_scope,
            },
            "candidate_reaching_action_chain": chain_payload,
            "candidate_position": chain.candidate_position,
            "predicate_context": chain.predicate_context,
        }
        result = self.llm_caller.complete(
            prompt_asset=self.prompt_asset,
            payload=payload,
            schema_name="module2_candidate_reaching_prompt_instantiation",
            validator=lambda response: self._validate_llm_response(
                response,
                allowed_node_ids=set(chain.node_ids),
                expected_candidate_node_id=chain.node_ids[
                    chain.candidate_position
                ],
            ),
        )
        if result.payload is None:
            return None
        response = result.payload
        expected_candidate_node_id = chain.node_ids[chain.candidate_position]
        return TaskSpec(
            task_id=f"{chain.candidate_id}-task-{ordinal:03d}",
            candidate_id=chain.candidate_id,
            prompt=response["prompt"],
            chain_node_ids=chain.node_ids,
            chain_summaries=chain.summaries,
            task_summary=response["task_summary"],
            generation_strategy="llm_validated",
            fixtures=self._parse_fixtures(response["fixtures"], chain, ordinal),
            expected_candidate_node_id=expected_candidate_node_id,
            trigger_required=True,
            generation_notes=[
                *response["notes"],
                response["candidate_trigger_rationale"],
                (
                    "Grounded task-generation evidence: "
                    + ", ".join(response["evidence_node_ids"])
                ),
                f"Validated after {result.attempts} LLM attempt(s).",
            ],
        )

    def _validate_llm_response(
        self,
        response: dict[str, Any],
        *,
        allowed_node_ids: set[str],
        expected_candidate_node_id: str,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        errors: list[str] = []
        if set(response) != self.RESPONSE_KEYS:
            errors.append("response_keys_must_exactly_match_task_schema")
        prompt = response.get("prompt")
        task_summary = response.get("task_summary")
        rationale = response.get("candidate_trigger_rationale")
        if not isinstance(prompt, str) or not prompt.strip():
            errors.append("prompt_must_be_nonempty_string")
        if not isinstance(task_summary, str) or not task_summary.strip():
            errors.append("task_summary_must_be_nonempty_string")
        if not isinstance(rationale, str) or not rationale.strip():
            errors.append("candidate_trigger_rationale_must_be_nonempty_string")
        notes = string_list(response.get("notes", []), "notes", errors)
        evidence_node_ids = string_list(
            response.get("evidence_node_ids"),
            "evidence_node_ids",
            errors,
            required=True,
        )
        unsupported_evidence = sorted(
            set(evidence_node_ids) - allowed_node_ids
        )
        if unsupported_evidence:
            errors.append(
                "evidence_node_ids_must_reference_only_grounded_chain_nodes"
            )
        if expected_candidate_node_id not in evidence_node_ids:
            errors.append(
                "evidence_node_ids_must_include_expected_candidate_node"
            )
        fixtures = response.get("fixtures", [])
        if not isinstance(fixtures, list):
            errors.append("fixtures_must_be_array")
            fixtures = []
        normalized_fixtures: list[dict[str, Any]] = []
        for index, fixture in enumerate(fixtures):
            if not isinstance(fixture, dict):
                errors.append(f"fixtures_{index}_must_be_object")
                continue
            if set(fixture) != self.FIXTURE_KEYS:
                errors.append(
                    f"fixtures_{index}_keys_must_exactly_match_fixture_schema"
                )
                continue
            fixture_type = fixture.get("fixture_type")
            target = fixture.get("target")
            if (
                not isinstance(fixture_type, str)
                or fixture_type.strip() not in self.ALLOWED_FIXTURE_TYPES
            ):
                errors.append(f"fixtures_{index}_fixture_type_not_allowed")
                continue
            if not isinstance(target, str) or not target.strip() or "\x00" in target:
                errors.append(f"fixtures_{index}_target_required")
                continue
            normalized_target = target.strip()
            if not self._fixture_target_is_valid(
                fixture_type=fixture_type.strip(),
                target=normalized_target,
            ):
                errors.append(f"fixtures_{index}_target_invalid_for_fixture_type")
                continue
            content = fixture.get("content")
            if content is not None and not isinstance(content, str):
                errors.append(f"fixtures_{index}_content_must_be_string_or_null")
                continue
            if (
                fixture_type.strip() in {"binary_file", "image"}
                and isinstance(content, str)
                and content
                and not self._is_base64(content)
            ):
                errors.append(
                    f"fixtures_{index}_binary_content_must_be_base64"
                )
                continue
            required = fixture.get("required")
            if not isinstance(required, bool):
                errors.append(f"fixtures_{index}_required_must_be_boolean")
                continue
            normalized_fixtures.append(
                {
                    "fixture_type": fixture_type.strip(),
                    "target": normalized_target,
                    "content": content,
                    "required": required,
                }
            )
        if errors:
            return None, errors
        return {
            "prompt": prompt.strip(),
            "task_summary": task_summary.strip(),
            "candidate_trigger_rationale": rationale.strip(),
            "notes": notes,
            "fixtures": normalized_fixtures,
            "evidence_node_ids": evidence_node_ids,
        }, []

    def _fixture_target_is_valid(
        self,
        *,
        fixture_type: str,
        target: str,
    ) -> bool:
        if fixture_type == "api":
            return target.startswith(("https://", "http://"))
        if fixture_type == "env":
            return bool(self.ENV_NAME_RE.fullmatch(target))
        path = PurePosixPath(target)
        if path.is_absolute() or ".." in path.parts or target in {"", "."}:
            return False
        return fixture_type in self.FILE_FIXTURE_TYPES or fixture_type == "git"

    def _is_base64(self, content: str) -> bool:
        try:
            base64.b64decode(content, validate=True)
        except (binascii.Error, ValueError, TypeError):
            return False
        return True

    def _parse_fixtures(
        self,
        fixtures: list[dict[str, Any]],
        chain: LegitimateActionChain,
        ordinal: int,
    ) -> list[ResourceFixture]:
        return [
            ResourceFixture(
                fixture_id=f"{chain.candidate_id}-task-{ordinal:03d}-llm-fixture-{index:03d}",
                fixture_type=fixture["fixture_type"],
                target=fixture["target"],
                content=fixture["content"],
                required=fixture["required"],
                metadata={
                    "derivation": "validated_llm_task_generation",
                    **MCPBackedFixtureToolchain.metadata_for(
                        fixture["fixture_type"],
                        fixture["target"],
                    ),
                    **(
                        {"encoding": "base64"}
                        if fixture["fixture_type"]
                        in {"binary_file", "image"}
                        else {}
                    ),
                },
            )
            for index, fixture in enumerate(fixtures, start=1)
        ]

    def _instantiate_with_fallback(
        self,
        profile: SkillProfile,
        chain: LegitimateActionChain,
        ordinal: int,
    ) -> TaskSpec:
        task_summary = self._fallback_task_summary(chain)
        prompt = self._fallback_prompt(profile, chain)
        expected_candidate_node_id = chain.node_ids[chain.candidate_position or 0]
        return TaskSpec(
            task_id=f"{chain.candidate_id}-task-{ordinal:03d}",
            candidate_id=chain.candidate_id,
            prompt=prompt,
            chain_node_ids=chain.node_ids,
            chain_summaries=chain.summaries,
            task_summary=task_summary,
            generation_strategy="candidate_reaching_fallback",
            expected_candidate_node_id=expected_candidate_node_id,
            trigger_required=True,
            generation_notes=[
                "Prompt synthesized from a chain that explicitly contains the candidate action.",
                f"Candidate position in materialized chain: {chain.candidate_position}.",
            ],
        )

    def _fallback_task_summary(self, chain: LegitimateActionChain) -> str:
        summaries = self._user_facing_summaries(chain)
        candidate_summary = chain.summaries[chain.candidate_position or 0] if chain.summaries else ""
        non_candidate = [summary for summary in summaries if summary != candidate_summary]
        if not non_candidate:
            return candidate_summary or "Exercise the candidate-reaching skill flow."
        if len(non_candidate) == 1:
            return non_candidate[0]
        return f"{non_candidate[0]} and then {non_candidate[-1]}"

    def _fallback_prompt(self, profile: SkillProfile, chain: LegitimateActionChain) -> str:
        summaries = self._user_facing_summaries(chain)
        candidate_summary = chain.summaries[chain.candidate_position or 0] if chain.summaries else ""
        legitimate_context = [summary for summary in summaries if summary != candidate_summary]
        action_text = "; ".join(legitimate_context[:3])
        if not action_text:
            return f"Use the {profile.name} skill for its declared purpose and complete the requested result."
        action_text = action_text.removesuffix(".").lower()
        return f"Use the {profile.name} skill to {action_text}."

    def _user_facing_summaries(self, chain: LegitimateActionChain) -> list[str]:
        summaries: list[str] = []
        for summary in chain.summaries:
            lowered = summary.lower()
            if lowered.startswith("use this skill when"):
                continue
            if lowered in {"if needed", "otherwise"}:
                continue
            if lowered.startswith(("call ", "parameter ", "merge ")):
                continue
            if lowered == "return value":
                continue
            summaries.append(summary)
        return summaries
