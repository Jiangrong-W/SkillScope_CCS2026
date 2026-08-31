from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from skillscope.common.models import (
    ActionTaskDescriptor,
    AuthorizationDecision,
    CandidateAction,
    CandidateExtractionResult,
    ExecutionRecord,
    FinalVerdict,
    NecessityDecision,
    TaskSpec,
)

from .action_tuple import ActionTuple, ActionTupleExtractor


class ActionTaskDescriptorBuilder:
    """Build Module 3's fixed-schema action-task descriptor deterministically."""

    _OPERATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        (
            "transmit",
            re.compile(
                r"\b(?:send|sync|upload|post|share|transmit|forward|email|notify|publish|deliver)\w*\b",
                re.IGNORECASE,
            ),
        ),
        (
            "analyze",
            re.compile(
                r"\b(?:analy[sz]e|inspect|review|check|diagnose|audit)\w*\b",
                re.IGNORECASE,
            ),
        ),
        (
            "summarize",
            re.compile(r"\b(?:summari[sz]e|summarization)\w*\b", re.IGNORECASE),
        ),
        (
            "create",
            re.compile(
                r"\b(?:create|generate|produce|build|render|compose)\w*\b",
                re.IGNORECASE,
            ),
        ),
        (
            "read",
            re.compile(r"\b(?:read|load|open|access)\w*\b", re.IGNORECASE),
        ),
        (
            "write",
            re.compile(r"\b(?:write|save|export|store)\w*\b", re.IGNORECASE),
        ),
        (
            "execute",
            re.compile(r"\b(?:execute|run|invoke|launch)\w*\b", re.IGNORECASE),
        ),
        (
            "delete",
            re.compile(r"\b(?:delete|remove|erase|unlink)\w*\b", re.IGNORECASE),
        ),
        (
            "collect",
            re.compile(r"\b(?:collect|gather|enumerate)\w*\b", re.IGNORECASE),
        ),
    )
    _NEGATION_RE = re.compile(
        r"(?:\bdo\s+not\b|\bdon't\b|\bnever\b|\bwithout\b|\bmust\s+not\b|\bno\b)\s*$",
        re.IGNORECASE,
    )
    _URL_RE = re.compile(r"https?://[^\s'\"`)>,;]+", re.IGNORECASE)
    _EMAIL_RE = re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        re.IGNORECASE,
    )
    _PATH_RE = re.compile(
        r"(?:\./|/)?[A-Za-z0-9_.~-]+(?:/[A-Za-z0-9_.~-]+)+|"
        r"[A-Za-z0-9_.-]+\.(?:log|txt|json|csv|md|yaml|yml|pdf|docx|png|jpg|jpeg)",
        re.IGNORECASE,
    )
    _OBJECT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("credentials", re.compile(r"\b(?:credential|secret|token|password)s?\b", re.IGNORECASE)),
        ("system_profile", re.compile(r"\b(?:system|machine|host)\s+profile\b", re.IGNORECASE)),
        ("activity_log", re.compile(r"\b(?:activity|audit)\s+logs?\b", re.IGNORECASE)),
        ("repository_commits", re.compile(r"\b(?:repository|repo)?\s*commits?\b", re.IGNORECASE)),
        ("configuration", re.compile(r"\b(?:configuration|config)\b", re.IGNORECASE)),
        ("report", re.compile(r"\breports?\b", re.IGNORECASE)),
        ("summary", re.compile(r"\bsummar(?:y|ies)\b", re.IGNORECASE)),
        ("repository", re.compile(r"\b(?:repository|repo)\b", re.IGNORECASE)),
        ("document", re.compile(r"\bdocuments?\b", re.IGNORECASE)),
        ("image", re.compile(r"\b(?:image|picture|photo)s?\b", re.IGNORECASE)),
        ("log", re.compile(r"\blogs?\b", re.IGNORECASE)),
        ("file", re.compile(r"\bfiles?\b", re.IGNORECASE)),
        ("data", re.compile(r"\bdata\b", re.IGNORECASE)),
    )
    _DESTINATION_ALIASES: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("telegram", re.compile(r"\btelegram\b", re.IGNORECASE)),
        ("slack", re.compile(r"\bslack\b", re.IGNORECASE)),
        ("email", re.compile(r"\b(?:email|mailbox)\b", re.IGNORECASE)),
        ("webhook", re.compile(r"\bwebhook\b", re.IGNORECASE)),
        ("remote_endpoint", re.compile(r"\b(?:remote|external)\s+endpoint\b", re.IGNORECASE)),
    )

    def __init__(self, tuple_extractor: ActionTupleExtractor | None = None) -> None:
        self.tuple_extractor = tuple_extractor or ActionTupleExtractor()

    def build(
        self,
        *,
        analysis: CandidateExtractionResult,
        candidate: CandidateAction,
        task: TaskSpec,
        authorization: AuthorizationDecision,
        necessity: NecessityDecision,
        final_verdict: FinalVerdict,
        original_record: ExecutionRecord | None = None,
    ) -> ActionTaskDescriptor:
        action_instances = self.tuple_extractor.extract_all_with_evidence(
            analysis=analysis,
            candidate=candidate,
            task=task,
            original_record=original_record,
        )
        material_action_instances = self._material_action_instances(action_instances)
        action_facts = self._aggregate_action_facts(material_action_instances)
        requested_operation, positive_operations = self._requested_operation(
            task.prompt
        )
        requested_object = self._requested_object(task.prompt)
        requested_scope = self._requested_scope(
            task.prompt,
            positive_operations=positive_operations,
        )
        requested_destination = self._requested_destination(
            task.prompt,
            positive_operations=positive_operations,
        )
        explicit_side_effect_requested = self._explicit_side_effect_requested(
            task_prompt=task.prompt,
            authorization=authorization,
            material_action_instances=material_action_instances,
        )
        requested_side_effect = self._tri_state_token(
            explicit_side_effect_requested
        )
        raw_slots = {
            "intent": task.task_summary or task.prompt,
            "requested_operation": requested_operation,
            "requested_object": requested_object,
            "requested_scope": requested_scope,
            "requested_destination": requested_destination,
            "explicit_side_effect_requested": requested_side_effect,
        }
        normalized_slots = {
            key: self._normalize(value) for key, value in raw_slots.items()
        }
        cluster_key = "|".join(
            f"{key}={normalized_slots[key]}"
            for key in (
                "intent",
                "requested_operation",
                "requested_object",
                "requested_scope",
                "requested_destination",
                "explicit_side_effect_requested",
            )
        )
        return ActionTaskDescriptor(
            descriptor_id=f"{candidate.candidate_id}:{task.task_id}:descriptor",
            candidate_id=candidate.candidate_id,
            task_id=task.task_id,
            intent=task.task_summary or task.prompt,
            operation=action_facts["operation"],
            object=action_facts["object"],
            scope=action_facts["scope"],
            destination=action_facts["destination"],
            side_effect=action_facts["side_effect"],
            final_verdict=final_verdict.label,
            requested_operation=requested_operation,
            requested_object=requested_object,
            requested_scope=requested_scope,
            requested_destination=requested_destination,
            explicit_side_effect_requested=explicit_side_effect_requested,
            material_action_instances=material_action_instances,
            normalized_slots=normalized_slots,
            cluster_key=cluster_key,
            evidence=[
                f"Task prompt={task.prompt}",
                (
                    "Task-request slots="
                    + json.dumps(raw_slots, ensure_ascii=False, sort_keys=True)
                ),
                (
                    "Material candidate actions="
                    + json.dumps(
                        material_action_instances,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                ),
                f"Authorization={authorization.label}: {authorization.reason}",
                (
                    f"Necessity={necessity.label}; "
                    f"CorePres={necessity.core_preserved}; GoalSat={necessity.goal_satisfied}."
                ),
                final_verdict.reason,
            ],
        )

    def _material_action_instances(
        self,
        instances: list[tuple[ActionTuple, dict[str, object] | None]],
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()
        for action_tuple, realized_event in instances:
            payload = dict(action_tuple.as_payload())
            key = tuple(
                str(payload.get(field) or "")
                for field in (
                    "operation",
                    "object",
                    "source",
                    "scope",
                    "destination",
                    "side_effect",
                )
            )
            if key in seen:
                continue
            seen.add(key)
            payload["evidence_kind"] = (
                "realized_original_action"
                if realized_event is not None
                else "static_candidate_action"
            )
            output.append(payload)
        return output

    def _aggregate_action_facts(
        self,
        instances: list[dict[str, Any]],
    ) -> dict[str, str]:
        return {
            field: self._join_distinct(
                str(instance.get(field) or "unspecified")
                for instance in instances
            )
            for field in (
                "operation",
                "object",
                "scope",
                "destination",
                "side_effect",
            )
        }

    def _requested_operation(self, prompt: str) -> tuple[str, set[str]]:
        matches: list[tuple[int, str]] = []
        positive: set[str] = set()
        for canonical, pattern in self._OPERATION_PATTERNS:
            for match in pattern.finditer(prompt):
                prefix = prompt[max(0, match.start() - 28) : match.start()]
                negated = bool(self._NEGATION_RE.search(prefix))
                value = f"not_{canonical}" if negated else canonical
                matches.append((match.start(), value))
                if not negated:
                    positive.add(canonical)
        ordered = self._ordered_values(matches)
        return ("+".join(ordered) if ordered else "unspecified", positive)

    def _requested_object(self, prompt: str) -> str:
        matches: list[tuple[int, str]] = []
        for canonical, pattern in self._OBJECT_PATTERNS:
            for match in pattern.finditer(prompt):
                matches.append((match.start(), canonical))
        for match in self._PATH_RE.finditer(prompt):
            matches.append((match.start(), match.group(0)))
        ordered = self._ordered_values(matches)
        return "+".join(ordered) if ordered else "unspecified"

    def _requested_scope(
        self,
        prompt: str,
        *,
        positive_operations: set[str],
    ) -> str:
        lowered = prompt.casefold()
        if "transmit" in positive_operations or self._URL_RE.search(prompt) or self._EMAIL_RE.search(prompt):
            return "external"
        if re.search(
            r"\b(?:local(?:ly)?|local[- ]only|on[- ]device|within (?:this )?(?:workspace|repository|repo|project))\b",
            lowered,
        ):
            return "local"
        if re.search(r"\b(?:external|remote|third[- ]party|internet|webhook|telegram|slack)\b", lowered):
            return "external"
        if re.search(r"\b(?:system[- ]wide|global)\b", lowered):
            return "global"
        if re.search(r"\b(?:workspace|repository|repo|project)\b", lowered):
            return "workspace"
        return "unspecified"

    def _requested_destination(
        self,
        prompt: str,
        *,
        positive_operations: set[str],
    ) -> str:
        if "transmit" not in positive_operations:
            return "none"
        url_match = self._URL_RE.search(prompt)
        if url_match is not None:
            return url_match.group(0)
        email_match = self._EMAIL_RE.search(prompt)
        if email_match is not None:
            return email_match.group(0)
        for canonical, pattern in self._DESTINATION_ALIASES:
            if pattern.search(prompt):
                return canonical
        destination_match = re.search(
            r"\b(?:send|sync|upload|post|share|forward|deliver)\w*\b"
            r".{0,80}?\bto\s+(?:the\s+)?"
            r"([A-Za-z0-9_.:@/-]+(?:\s+[A-Za-z0-9_.:@/-]+){0,3})",
            prompt,
            flags=re.IGNORECASE,
        )
        if destination_match is not None:
            return destination_match.group(1).strip(" .,:;")
        return "external_unspecified"

    def _explicit_side_effect_requested(
        self,
        *,
        task_prompt: str,
        authorization: AuthorizationDecision,
        material_action_instances: list[dict[str, Any]],
    ) -> bool | None:
        del task_prompt, material_action_instances
        # This is the authorization judge's task-grounded side-effect component,
        # conservatively aggregated over every realized action instance.  It is
        # deliberately not inferred from the fact that the Skill performed the
        # side effect.
        return authorization.side_effect_authorized

    def _tri_state_token(self, value: bool | None) -> str:
        if value is True:
            return "true"
        if value is False:
            return "false"
        return "unresolved"

    def _ordered_values(self, matches: list[tuple[int, str]]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for _, value in sorted(matches, key=lambda item: item[0]):
            normalized = value.casefold().strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            output.append(value)
        return output

    def _join_distinct(self, values: Iterable[str]) -> str:
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = str(value).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            output.append(normalized)
        return " | ".join(output) if output else "unspecified"

    def _normalize(self, value: str) -> str:
        normalized = re.sub(r"\s+", " ", str(value).strip().lower())
        normalized = re.sub(r"[^a-z0-9_./: -]", "", normalized)
        return normalized[:240] or "unspecified"
