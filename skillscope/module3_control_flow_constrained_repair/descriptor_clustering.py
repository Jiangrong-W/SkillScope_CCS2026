from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, ClassVar
from urllib.parse import urlparse

from skillscope.common.llm import (
    MAX_VALIDATED_LLM_ATTEMPTS,
    DisabledLLMClient,
    JSONResponseContract,
    PromptAssetLoader,
    StructuredLLMClient,
    complete_validated_json,
)
from skillscope.common.models import ActionTaskDescriptor, FinalVerdict

DESCRIPTOR_SLOTS = (
    "intent",
    "requested_operation",
    "requested_object",
    "requested_scope",
    "requested_destination",
    "explicit_side_effect_requested",
)


_ALIASES: dict[str, dict[str, str]] = {
    "intent": {
        "analyse": "analysis",
        "analyze": "analysis",
        "inspect": "analysis",
        "notify": "notification",
        "report": "reporting",
        "summarise": "reporting",
        "summarize": "reporting",
        "share": "sharing",
    },
    "requested_operation": {
        "call": "execute",
        "deliver": "transmit",
        "email": "transmit",
        "execute command": "execute",
        "forward": "transmit",
        "invoke": "execute",
        "load": "read",
        "message": "transmit",
        "notify": "transmit",
        "open": "read",
        "post": "transmit",
        "publish": "transmit",
        "remove": "delete",
        "render": "create",
        "run": "execute",
        "save": "write",
        "send": "transmit",
        "share": "transmit",
        "shell": "execute",
        "sync": "transmit",
        "transmit": "transmit",
        "upload": "transmit",
    },
    "requested_object": {
        "artifact": "file",
        "artifacts": "file",
        "credential": "credential",
        "credentials": "credential",
        "digest": "report",
        "document": "file",
        "documents": "file",
        "file": "file",
        "files": "file",
        "information": "data",
        "notification": "message",
        "reports": "report",
        "secret": "credential",
        "secrets": "credential",
        "summary": "report",
        "token": "credential",
        "tokens": "credential",
    },
    "requested_scope": {
        "bundle": "workspace",
        "cross boundary": "external",
        "external": "external",
        "global": "global",
        "internet": "external",
        "local": "local",
        "local only": "local",
        "localhost": "local",
        "network": "external",
        "on device": "local",
        "project": "workspace",
        "remote": "external",
        "system wide": "global",
        "third party": "external",
    },
    "requested_destination": {
        "email address": "email",
        "external destination": "external",
        "filesystem": "local",
        "local file": "local",
        "local filesystem": "local",
        "none": "none",
        "standard output": "stdout",
        "third party": "external",
        "webhook endpoint": "webhook",
    },
    "explicit_side_effect_requested": {
        "true": "true",
        "false": "false",
        "unresolved": "unresolved",
    },
}


@dataclass(frozen=True, slots=True)
class DescriptorCluster:
    cluster_key: str
    normalized_slots: dict[str, str]
    descriptor_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    disposition: str
    material_action_instances: tuple[dict[str, Any], ...] = ()

    def as_payload(self) -> dict[str, object]:
        return {
            "cluster_key": self.cluster_key,
            "normalized_slots": dict(self.normalized_slots),
            "descriptor_ids": list(self.descriptor_ids),
            "task_ids": list(self.task_ids),
            "disposition": self.disposition,
            "material_action_instances": [
                dict(instance) for instance in self.material_action_instances
            ],
        }


class DescriptorClusterer:
    """Normalize fixed action-task slots and cluster them deterministically."""

    _RESPONSE_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"normalized_slots", "rationale", "evidence_refs"}
    )

    def __init__(
        self,
        *,
        llm_client: StructuredLLMClient | None = None,
        prompt_loader: PromptAssetLoader | None = None,
        prompt_asset: str = "prompts/module3_descriptor_normalization.md",
    ) -> None:
        self.llm_client = llm_client or DisabledLLMClient()
        self.prompt_loader = prompt_loader
        self.prompt_asset = prompt_asset

    def normalize(
        self,
        descriptor: ActionTaskDescriptor,
        *,
        final_verdict: FinalVerdict,
    ) -> ActionTaskDescriptor:
        deterministic_slots = {
            slot: self.normalize_slot(slot, self._descriptor_slot(descriptor, slot))
            for slot in DESCRIPTOR_SLOTS
        }
        normalized_slots, strategy, rationale = self._semantic_normalize(
            descriptor=descriptor,
            deterministic_slots=deterministic_slots,
        )
        cluster_key = self.cluster_key(normalized_slots)
        return replace(
            descriptor,
            final_verdict=final_verdict.label,
            normalized_slots=normalized_slots,
            cluster_key=cluster_key,
            evidence=[
                *descriptor.evidence,
                f"slot_normalization={strategy}",
                *([f"slot_normalization_rationale={rationale}"] if rationale else []),
            ],
        )

    def _semantic_normalize(
        self,
        *,
        descriptor: ActionTaskDescriptor,
        deterministic_slots: dict[str, str],
    ) -> tuple[dict[str, str], str, str]:
        if self.prompt_loader is None or isinstance(
            self.llm_client,
            DisabledLLMClient,
        ):
            return deterministic_slots, "deterministic_conservative", ""

        raw_slots = {
            slot: self._descriptor_slot(descriptor, slot)
            for slot in DESCRIPTOR_SLOTS
        }
        evidence_refs = {f"descriptor.{slot}" for slot in DESCRIPTOR_SLOTS}
        payload = {
            "descriptor_id": descriptor.descriptor_id,
            "raw_slots": raw_slots,
            "deterministic_conservative_normalization": deterministic_slots,
            "normalization_contract": {
                "slots": list(DESCRIPTOR_SLOTS),
                "allowed_evidence_refs": sorted(evidence_refs),
                "preserve_specific_objects_and_destinations": True,
                "do_not_infer_missing_authority": True,
            },
        }
        try:
            response = complete_validated_json(
                self.llm_client,
                system_prompt=self.prompt_loader.load(self.prompt_asset),
                user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
                schema_name="module3_descriptor_semantic_normalization",
                contract=JSONResponseContract(
                    required_fields=tuple(sorted(self._RESPONSE_KEYS)),
                    non_empty_string_fields=("rationale",),
                    evidence_field="evidence_refs",
                    grounded_evidence_ids=evidence_refs,
                    consistency_checks=(
                        lambda value: self._validate_semantic_response(
                            value,
                            raw_slots=raw_slots,
                            deterministic_slots=deterministic_slots,
                        ),
                    ),
                ),
                max_attempts=MAX_VALIDATED_LLM_ATTEMPTS,
            )
        except RuntimeError:
            # Failure must not create an ungrounded merge. Keeping the
            # deterministic representation is conservative because it may
            # split equivalent contexts but cannot broaden an allow cluster.
            return deterministic_slots, "deterministic_after_llm_failure", ""
        return (
            dict(response["normalized_slots"]),
            "llm_strictly_validated",
            str(response["rationale"]).strip(),
        )

    def _validate_semantic_response(
        self,
        response: dict[str, object],
        *,
        raw_slots: dict[str, str],
        deterministic_slots: dict[str, str],
    ) -> str | None:
        if set(response) != self._RESPONSE_KEYS:
            return "descriptor normalization response keys must exactly match the schema"
        slots = response.get("normalized_slots")
        if not isinstance(slots, dict) or set(slots) != set(DESCRIPTOR_SLOTS):
            return "normalized_slots must contain every descriptor slot exactly once"
        for slot in DESCRIPTOR_SLOTS:
            value = slots.get(slot)
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 160
                or re.fullmatch(r"[a-z0-9][a-z0-9_.:@-]*", value.strip()) is None
            ):
                return f"normalized slot {slot} must be a bounded canonical token"
            # The deterministic normalization is the auditable semantic
            # boundary.  The LLM may explain it, but may not turn missing
            # evidence into a concrete permission-bearing value or remap one
            # concrete operation/object/intent into another.  This also
            # preserves sentinel values such as ``unspecified``, ``none``, and
            # ``external_unspecified`` instead of hallucinating specificity.
            if value.strip() != deterministic_slots[slot]:
                return (
                    f"normalized slot {slot} must exactly preserve the "
                    "conservative grounded normalization"
                )

        # Concrete URL hosts and exact local/external scope are security
        # boundaries, not free-form synonyms. They cannot be generalized by
        # the model during semantic grouping.
        destination_host = self._destination_host(
            raw_slots["requested_destination"]
        )
        if (
            destination_host
            and slots["requested_destination"] != destination_host
        ):
            return "normalization must preserve the concrete destination host"
        if (
            deterministic_slots["requested_scope"]
            in {"local", "external", "global", "workspace"}
            and slots["requested_scope"]
            != deterministic_slots["requested_scope"]
        ):
            return "normalization must preserve the concrete execution scope"
        if (
            slots["explicit_side_effect_requested"]
            != deterministic_slots["explicit_side_effect_requested"]
        ):
            return (
                "normalization must preserve the task-grounded explicit "
                "side-effect-request value"
            )

        refs = response.get("evidence_refs")
        if not isinstance(refs, list) or set(refs) != {
            f"descriptor.{slot}" for slot in DESCRIPTOR_SLOTS
        }:
            return "evidence_refs must cite every supplied descriptor slot exactly once"
        return None

    def cluster(
        self,
        descriptors: list[ActionTaskDescriptor],
        *,
        final_verdicts: list[FinalVerdict],
    ) -> tuple[list[ActionTaskDescriptor], list[DescriptorCluster], list[str]]:
        verdicts_by_pair: dict[tuple[str, str], FinalVerdict] = {}
        ambiguous_pairs: set[tuple[str, str]] = set()
        for verdict in final_verdicts:
            pair = (verdict.candidate_id, verdict.task_id)
            existing = verdicts_by_pair.get(pair)
            if existing is not None and existing != verdict:
                ambiguous_pairs.add(pair)
                continue
            verdicts_by_pair[pair] = verdict

        normalized: list[ActionTaskDescriptor] = []
        skipped: list[str] = []
        dispositions: dict[str, list[str]] = {}
        slots_by_key: dict[str, dict[str, str]] = {}
        descriptor_ids_by_key: dict[str, list[str]] = {}
        task_ids_by_key: dict[str, list[str]] = {}
        material_actions_by_key: dict[str, dict[str, dict[str, Any]]] = {}

        for descriptor in sorted(descriptors, key=lambda value: value.descriptor_id):
            pair = (descriptor.candidate_id, descriptor.task_id)
            if pair in ambiguous_pairs:
                skipped.append(
                    f"{descriptor.descriptor_id}: conflicting final verdicts for "
                    f"{descriptor.candidate_id}/{descriptor.task_id}."
                )
                continue
            verdict = verdicts_by_pair.get(pair)
            if verdict is None:
                skipped.append(
                    f"{descriptor.descriptor_id}: no final verdict for "
                    f"{descriptor.candidate_id}/{descriptor.task_id}."
                )
                continue
            normalized_descriptor = self.normalize(
                descriptor,
                final_verdict=verdict,
            )
            normalized.append(normalized_descriptor)
            key = normalized_descriptor.cluster_key
            slots_by_key[key] = dict(normalized_descriptor.normalized_slots)
            descriptor_ids_by_key.setdefault(key, []).append(
                normalized_descriptor.descriptor_id
            )
            task_ids_by_key.setdefault(key, []).append(normalized_descriptor.task_id)
            material_actions = material_actions_by_key.setdefault(key, {})
            for action_instance in normalized_descriptor.material_action_instances:
                fingerprint = json.dumps(
                    action_instance,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                material_actions.setdefault(fingerprint, dict(action_instance))
            dispositions.setdefault(key, []).append(self.disposition(verdict))

        clusters: list[DescriptorCluster] = []
        for key in sorted(slots_by_key):
            member_dispositions = dispositions[key]
            if "blocked" in member_dispositions:
                disposition = "blocked"
            elif member_dispositions and all(
                value == "allowed" for value in member_dispositions
            ):
                disposition = "allowed"
            else:
                disposition = "inconclusive"
            clusters.append(
                DescriptorCluster(
                    cluster_key=key,
                    normalized_slots=slots_by_key[key],
                    descriptor_ids=tuple(sorted(set(descriptor_ids_by_key[key]))),
                    task_ids=tuple(sorted(set(task_ids_by_key[key]))),
                    disposition=disposition,
                    material_action_instances=tuple(
                        material_actions_by_key.get(key, {})[fingerprint]
                        for fingerprint in sorted(
                            material_actions_by_key.get(key, {})
                        )
                    ),
                )
            )
        return normalized, clusters, skipped

    def disposition(self, verdict: FinalVerdict) -> str:
        if (
            verdict.authorization_label == "unauthorized"
            or verdict.necessity_label == "unnecessary"
        ):
            return "blocked"
        if (
            verdict.label == "not_overprivileged"
            and verdict.authorization_label == "authorized"
            and verdict.necessity_label == "necessary"
        ):
            return "allowed"
        return "inconclusive"

    def normalize_slot(self, slot: str, value: object) -> str:
        normalized = self._normalize_text(value)
        if not normalized:
            return "unspecified"
        if slot == "requested_destination":
            host = self._destination_host(value)
            if host:
                normalized = host
        return _ALIASES.get(slot, {}).get(normalized, normalized.replace(" ", "_"))

    def _descriptor_slot(
        self,
        descriptor: ActionTaskDescriptor,
        slot: str,
    ) -> str:
        if slot == "explicit_side_effect_requested":
            value = descriptor.explicit_side_effect_requested
            if value is True:
                return "true"
            if value is False:
                return "false"
            return "unresolved"
        return str(getattr(descriptor, slot) or "unspecified")

    def cluster_key(
        self,
        normalized_slots: dict[str, str],
    ) -> str:
        payload = {
            slot: normalized_slots.get(slot, "unspecified")
            for slot in DESCRIPTOR_SLOTS
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
        return f"descriptor-cluster-v1-{digest}"

    def _normalize_text(self, value: str) -> str:
        text = unicodedata.normalize("NFKC", str(value or "")).casefold()
        text = re.sub(r"[_/\\|:+-]+", " ", text)
        text = re.sub(r"[^\w\s.@]", " ", text, flags=re.UNICODE)
        return re.sub(r"\s+", " ", text).strip(" .")

    def _destination_host(self, value: str) -> str | None:
        raw = str(value or "").strip()
        if "://" not in raw:
            return None
        parsed = urlparse(raw)
        host = (parsed.hostname or "").casefold().strip(".")
        if not host:
            return None
        return host.removeprefix("www.")
