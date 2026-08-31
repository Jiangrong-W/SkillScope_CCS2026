from __future__ import annotations

import copy
import json
import re
import shlex
from pathlib import Path
from typing import Any

from skillscope.common.llm import (
    DisabledLLMClient,
    JSONResponseContract,
    MAX_VALIDATED_LLM_ATTEMPTS,
    PromptAssetLoader,
    StructuredLLMClient,
    complete_validated_json,
)
from skillscope.common.models import RepairItem


class InstructionRewriter:
    _PROJECTION_OUTPUT_KEYS = {
        "repair_id",
        "repair_type",
        "instruction_file",
        "target_start_line",
        "target_end_line",
        "source_file",
        "source_start_line",
        "source_end_line",
        "descriptor_ids",
        "allowed_cluster_keys",
        "blocked_cluster_keys",
        "guard_condition",
        "updated_instruction_file",
        "notes",
    }

    def __init__(
        self,
        *,
        llm_client: StructuredLLMClient | None = None,
        prompt_loader: PromptAssetLoader | None = None,
        prompt_asset: str = "prompts/module3_instruction_projection.md",
    ) -> None:
        self.llm_client = llm_client or DisabledLLMClient()
        self.prompt_loader = prompt_loader
        self.prompt_asset = prompt_asset

    def rewrite(self, *, patched_bundle_root: Path, item: RepairItem) -> list[str]:
        if item.repair_type not in {
            "GUARD_INSTRUCTION_TASK_CONDITIONED",
            "REORGANIZE_CODE_AND_ADD_DISPATCH",
        }:
            return []

        target_file = self._resolve_instruction_file(patched_bundle_root, item)
        if target_file is None:
            return [f"No instruction file could be resolved for {item.repair_id}."]
        if item.repair_type == "REORGANIZE_CODE_AND_ADD_DISPATCH":
            self._capture_instruction_invocation(
                target_file=target_file,
                item=item,
            )
            self._record_instruction_dispatch(item)

        llm_notes = self._rewrite_with_llm(
            patched_bundle_root=patched_bundle_root,
            target_file=target_file,
            item=item,
        )
        if llm_notes is not None:
            return llm_notes
        return self._rewrite_with_fallback(target_file=target_file, item=item)

    def rewrite_many(
        self,
        *,
        patched_bundle_root: Path,
        items: list[RepairItem],
    ) -> list[str]:
        """Project instruction repairs without same-range overwrite races.

        Instruction normalization may recover several atomic actions from one
        Markdown block while retaining only the block-level source range.  A
        sequence of ordinary ``rewrite`` calls would therefore replace the
        same range repeatedly.  Group such repairs and project all grounded
        fragments from the unchanged block in one deterministic transaction.

        Different source ranges are processed bottom-up so line insertions at
        a later range cannot invalidate an earlier range's provenance.
        """

        if not items:
            return []
        grouped: dict[
            tuple[str, int | None, int | None, str], list[RepairItem]
        ] = {}
        for item in items:
            target_file = self._resolve_instruction_file(
                patched_bundle_root, item
            )
            if target_file is None:
                raise RuntimeError(
                    "instruction projection failed closed: no instruction "
                    f"file could be resolved for {item.repair_id}"
                )
            relative = (
                target_file.resolve()
                .relative_to(patched_bundle_root.resolve())
                .as_posix()
            )
            grouped.setdefault(
                (
                    relative,
                    item.source_start_line,
                    item.source_end_line,
                    (
                        ""
                        if item.source_start_line is not None
                        and item.source_end_line is not None
                        else item.repair_id
                    ),
                ),
                [],
            ).append(item)

        ordered_groups = sorted(
            grouped.items(),
            key=lambda value: (
                value[0][0],
                -(value[0][1] or -1),
                -(value[0][2] or -1),
                value[0][3],
                min(item.repair_id for item in value[1]),
            ),
        )
        target_files = {
            target_file
            for item in items
            if (
                target_file := self._resolve_instruction_file(
                    patched_bundle_root, item
                )
            )
            is not None
        }
        original_contents = {
            target_file: target_file.read_text(encoding="utf-8")
            for target_file in target_files
        }
        original_metadata = {
            item.repair_id: copy.deepcopy(item.metadata) for item in items
        }
        try:
            return self._apply_rewrite_groups(
                patched_bundle_root=patched_bundle_root,
                ordered_groups=ordered_groups,
            )
        except Exception:
            # A later, earlier-in-file range may fail after a lower range was
            # successfully projected.  Restore the entire instruction batch so
            # fail-closed never leaves a partially rewritten bundle or stale
            # projection metadata behind.
            for target_file, content in original_contents.items():
                target_file.write_text(content, encoding="utf-8")
            for item in items:
                item.metadata.clear()
                item.metadata.update(
                    copy.deepcopy(original_metadata[item.repair_id])
                )
            raise

    def _apply_rewrite_groups(
        self,
        *,
        patched_bundle_root: Path,
        ordered_groups: list[
            tuple[
                tuple[str, int | None, int | None, str],
                list[RepairItem],
            ]
        ],
    ) -> list[str]:
        notes: list[str] = []
        for (_, _, _, _), group in ordered_groups:
            ordered = sorted(group, key=lambda item: item.repair_id)
            if len(ordered) == 1:
                notes.extend(
                    self.rewrite(
                        patched_bundle_root=patched_bundle_root,
                        item=ordered[0],
                    )
                )
                continue
            notes.extend(
                self._rewrite_same_range_group(
                    patched_bundle_root=patched_bundle_root,
                    items=ordered,
                )
            )
        return notes

    def _rewrite_same_range_group(
        self,
        *,
        patched_bundle_root: Path,
        items: list[RepairItem],
    ) -> list[str]:
        if any(
            item.repair_type != "GUARD_INSTRUCTION_TASK_CONDITIONED"
            for item in items
        ):
            raise RuntimeError(
                "instruction projection failed closed: a shared source range "
                "contains non-composable repair types"
            )
        target_file = self._resolve_instruction_file(
            patched_bundle_root, items[0]
        )
        if target_file is None:
            raise RuntimeError(
                "instruction projection failed closed: no instruction file "
                "could be resolved for a same-range repair group"
            )
        if any(
            self._resolve_instruction_file(patched_bundle_root, item)
            != target_file
            for item in items[1:]
        ):
            raise RuntimeError(
                "instruction projection failed closed: same-range repairs "
                "resolved to different instruction files"
            )

        content = target_file.read_text(encoding="utf-8")
        instruction_file = (
            target_file.resolve()
            .relative_to(patched_bundle_root.resolve())
            .as_posix()
        )
        start = items[0].source_start_line
        end = items[0].source_end_line
        if (
            start is None
            or end is None
            or any(
                item.source_start_line != start
                or item.source_end_line != end
                for item in items[1:]
            )
        ):
            raise RuntimeError(
                "instruction projection failed closed: grouped repairs do not "
                "share one grounded source range"
            )
        updated, projections = self._project_guard_fragments(
            content=content,
            instruction_file=instruction_file,
            items=items,
            target_start_line=start,
            target_end_line=end,
        )
        target_file.write_text(
            self._ensure_trailing_newline(updated), encoding="utf-8"
        )
        group_ids = [item.repair_id for item in items]
        for item in items:
            projection = projections[item.repair_id]
            item.metadata.update(
                {
                    "instruction_projection_strategy": (
                        "fallback_composite_grounded_fragments"
                    ),
                    "instruction_projection_scope": "grounded_fragment",
                    "instruction_projection_fragment": projection[
                        "matched_text"
                    ],
                    "instruction_projection_group_repair_ids": group_ids,
                }
            )
        return [
            "Composed grounded instruction fragments in "
            f"{target_file.name} for {', '.join(group_ids)}."
        ]

    def _rewrite_with_llm(
        self,
        *,
        patched_bundle_root: Path,
        target_file: Path,
        item: RepairItem,
    ) -> list[str] | None:
        if self.prompt_loader is None:
            return None
        content = target_file.read_text(encoding="utf-8")
        instruction_file = (
            target_file.resolve()
            .relative_to(patched_bundle_root.resolve())
            .as_posix()
        )
        target_start_line, target_end_line = self._projection_target_lines(
            content=content,
            instruction_file=instruction_file,
            item=item,
        )
        required_replacement = self._required_guarded_replacement(item)
        required_updated_instruction_file = (
            self._deterministic_projected_content(
                content=content,
                item=item,
                target_start_line=target_start_line,
                target_end_line=target_end_line,
            )
        )
        payload = {
            "skill_profile": item.metadata.get("skill_profile") or {},
            "repair_item": self._item_payload(item),
            "instruction_file": instruction_file,
            "projection_target": {
                "start_line": target_start_line,
                "end_line": target_end_line,
            },
            "projection_contract": {
                "original_target_text": self._projection_target_text(
                    content=content,
                    target_start_line=target_start_line,
                    target_end_line=target_end_line,
                ),
                "required_guarded_replacement": (
                    required_replacement
                ),
                "required_updated_instruction_file": (
                    required_updated_instruction_file
                ),
                "must_replace_original_target": True,
                "must_preserve_content_outside_target": True,
            },
            "instruction_content": content,
        }
        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        try:
            response = complete_validated_json(
                self.llm_client,
                system_prompt=self.prompt_loader.load(self.prompt_asset),
                user_prompt=user_prompt,
                schema_name="module3_instruction_projection",
                contract=JSONResponseContract(
                    required_fields=tuple(
                        sorted(self._PROJECTION_OUTPUT_KEYS)
                    ),
                    non_empty_string_fields=(
                        "repair_id",
                        "repair_type",
                        "instruction_file",
                        "guard_condition",
                        "updated_instruction_file",
                    ),
                    enum_fields={
                        "repair_type": {item.repair_type.lower()},
                    },
                    evidence_field="descriptor_ids",
                    grounded_evidence_ids=set(item.descriptor_ids),
                    consistency_checks=(
                        lambda value: self._validate_projection_response(
                            response=value,
                            item=item,
                            instruction_file=instruction_file,
                            target_start_line=target_start_line,
                            target_end_line=target_end_line,
                            original_instruction_content=content,
                            required_updated_instruction_file=(
                                required_updated_instruction_file
                            ),
                        ),
                    ),
                ),
                max_attempts=MAX_VALIDATED_LLM_ATTEMPTS,
            )
        except RuntimeError as exc:
            item.metadata["instruction_projection_llm_error"] = str(exc)
            return None
        updated_instruction_file = str(response["updated_instruction_file"])
        target_file.write_text(self._ensure_trailing_newline(updated_instruction_file), encoding="utf-8")
        item.metadata["instruction_projection_strategy"] = (
            "llm_validated_grounded_retry"
        )
        return list(response["notes"])

    def _validate_projection_response(
        self,
        *,
        response: dict[str, Any],
        item: RepairItem,
        instruction_file: str,
        target_start_line: int,
        target_end_line: int,
        original_instruction_content: str,
        required_updated_instruction_file: str,
    ) -> str | None:
        if set(response) != self._PROJECTION_OUTPUT_KEYS:
            return "instruction projection fields must exactly match the fixed schema"
        expected_scalars: dict[str, object] = {
            "repair_id": item.repair_id,
            "repair_type": item.repair_type,
            "instruction_file": instruction_file,
            "target_start_line": target_start_line,
            "target_end_line": target_end_line,
            "source_file": item.source_file,
            "source_start_line": item.source_start_line,
            "source_end_line": item.source_end_line,
            "guard_condition": item.guard_condition,
        }
        for field_name, expected in expected_scalars.items():
            if response.get(field_name) != expected:
                return (
                    f"instruction projection field {field_name!r} does not "
                    "match grounded source evidence"
                )
        for field_name, expected in (
            ("descriptor_ids", item.descriptor_ids),
            ("allowed_cluster_keys", item.allowed_cluster_keys),
            ("blocked_cluster_keys", item.blocked_cluster_keys),
        ):
            value = response.get(field_name)
            if (
                not isinstance(value, list)
                or any(not isinstance(entry, str) for entry in value)
                or value != expected
            ):
                return (
                    f"instruction projection field {field_name!r} must "
                    "exactly echo grounded evidence"
                )
        updated = response.get("updated_instruction_file")
        if not isinstance(updated, str) or not updated.strip():
            return "updated_instruction_file must be a non-empty string"
        if self._ensure_trailing_newline(
            updated
        ) != self._ensure_trailing_newline(
            required_updated_instruction_file
        ):
            return (
                "updated instruction must exactly apply the grounded target "
                "replacement while preserving every byte outside the target"
            )
        if item.guard_condition and item.guard_condition not in updated:
            return "updated instruction omitted the deterministic guard"
        required_replacement = self._required_guarded_replacement(item)
        if (
            not required_replacement
            or required_replacement not in updated
        ):
            return (
                "updated instruction omitted the complete deterministic "
                "guarded replacement"
            )
        original_target_text = self._projection_target_text(
            content=original_instruction_content,
            target_start_line=target_start_line,
            target_end_line=target_end_line,
        )
        if (
            original_target_text
            and original_target_text != required_replacement
            and self._contains_line_block(
                content=updated,
                block=original_target_text,
            )
        ):
            return (
                "updated instruction retained the original unguarded "
                "projection target"
            )
        if item.repair_type == "REORGANIZE_CODE_AND_ADD_DISPATCH":
            required_block = str(
                item.metadata.get("instruction_dispatch_block") or ""
            )
            if not required_block or required_block not in updated:
                return (
                    "updated instruction omitted the grounded instruction-layer "
                    "allowed/safe dispatch block"
                )
            if (
                "SKILLSCOPE_TASK_CLUSTER" in updated
                or "SKILLSCOPE_USER_PROMPT" in updated
            ):
                return (
                    "instruction dispatch must use semantic agent planning, not "
                    "a code-layer environment routing oracle"
                )
        notes = response.get("notes")
        if not isinstance(notes, list) or any(
            not isinstance(note, str) for note in notes
        ):
            return "instruction projection notes must be a list of strings"
        return None

    def _projection_target_lines(
        self,
        *,
        content: str,
        instruction_file: str,
        item: RepairItem,
    ) -> tuple[int, int]:
        lines = content.splitlines()
        if (
            item.source_file == instruction_file
            and item.source_start_line is not None
            and item.source_end_line is not None
            and 1 <= item.source_start_line <= item.source_end_line <= len(lines)
        ):
            return item.source_start_line, item.source_end_line
        dispatch_script = item.metadata.get("dispatch_source_file")
        if isinstance(dispatch_script, str):
            for index, line in enumerate(lines, start=1):
                if dispatch_script in line:
                    return index, index
        raw_text = (item.raw_text or "").strip()
        if raw_text:
            for index, line in enumerate(lines, start=1):
                if raw_text in line:
                    return index, index
        append_line = len(lines) + 1
        return append_line, append_line

    def _projection_target_text(
        self,
        *,
        content: str,
        target_start_line: int,
        target_end_line: int,
    ) -> str:
        lines = content.splitlines()
        if not (
            1
            <= target_start_line
            <= target_end_line
            <= len(lines)
        ):
            return ""
        return "\n".join(
            lines[target_start_line - 1 : target_end_line]
        ).strip()

    def _required_guarded_replacement(self, item: RepairItem) -> str:
        return "".join(self._replacement_lines(item)).strip()

    def _deterministic_projected_content(
        self,
        *,
        content: str,
        item: RepairItem,
        target_start_line: int,
        target_end_line: int,
    ) -> str:
        """Apply only the grounded target replacement."""

        if item.repair_type == "GUARD_INSTRUCTION_TASK_CONDITIONED":
            instruction_file = str(item.source_file or "SKILL.md")
            updated, _ = self._project_guard_fragments(
                content=content,
                instruction_file=instruction_file,
                items=[item],
                target_start_line=target_start_line,
                target_end_line=target_end_line,
            )
            return updated

        lines = content.splitlines(keepends=True)
        replacement_lines = self._replacement_lines(item)
        if (
            1
            <= target_start_line
            <= target_end_line
            <= len(lines)
        ):
            lines[target_start_line - 1 : target_end_line] = replacement_lines
            return "".join(lines)
        if target_start_line == target_end_line == len(lines) + 1:
            prefix = "".join(lines)
            if prefix and not prefix.endswith("\n"):
                prefix += "\n"
            return prefix + "".join(replacement_lines)
        return content

    def _project_guard_fragments(
        self,
        *,
        content: str,
        instruction_file: str,
        items: list[RepairItem],
        target_start_line: int,
        target_end_line: int,
    ) -> tuple[str, dict[str, dict[str, object]]]:
        """Replace unique grounded action fragments inside one source range.

        The projection accepts only a simple Markdown bullet because splitting
        a prose paragraph, table row, code block, or numbered item can silently
        alter its surrounding semantics.  Unsupported or ambiguous shapes fail
        closed before any file is written.
        """

        lines = content.splitlines(keepends=True)
        if not (
            1
            <= target_start_line
            <= target_end_line
            <= len(lines)
        ):
            raise RuntimeError(
                "instruction projection failed closed: grounded source range "
                f"{instruction_file}:{target_start_line}-{target_end_line} "
                "is outside the current instruction file"
            )
        region_lines = lines[target_start_line - 1 : target_end_line]
        region = "".join(region_lines)
        matches: list[dict[str, object]] = []
        for item in items:
            match = self._grounded_fragment_match(region=region, item=item)
            matches.append(
                {
                    "repair_id": item.repair_id,
                    "item": item,
                    "start": match.start(),
                    "end": match.end(),
                    "matched_text": match.group(0),
                }
            )
        ordered_matches = sorted(
            matches,
            key=lambda value: (
                int(value["start"]),
                int(value["end"]),
                str(value["repair_id"]),
            ),
        )
        for previous, current in zip(
            ordered_matches, ordered_matches[1:]
        ):
            if int(current["start"]) < int(previous["end"]):
                raise RuntimeError(
                    "instruction projection failed closed: grounded fragments "
                    f"for {previous['repair_id']} and {current['repair_id']} "
                    "overlap in the same source range"
                )

        line_starts: list[int] = []
        offset = 0
        for line in region_lines:
            line_starts.append(offset)
            offset += len(line)
        matches_by_line: dict[int, list[dict[str, object]]] = {}
        for match in ordered_matches:
            start = int(match["start"])
            end = int(match["end"])
            line_index = max(
                index
                for index, line_start in enumerate(line_starts)
                if line_start <= start
            )
            line_end = line_starts[line_index] + len(
                region_lines[line_index].rstrip("\r\n")
            )
            if end > line_end:
                raise RuntimeError(
                    "instruction projection failed closed: a grounded action "
                    "fragment crosses Markdown lines"
                )
            localized = dict(match)
            localized["start"] = start - line_starts[line_index]
            localized["end"] = end - line_starts[line_index]
            matches_by_line.setdefault(line_index, []).append(localized)

        projected_lines: list[str] = []
        for line_index, line in enumerate(region_lines):
            line_matches = matches_by_line.get(line_index)
            if not line_matches:
                projected_lines.append(line)
                continue
            projected_lines.append(
                self._project_compound_bullet_line(
                    line=line,
                    matches=line_matches,
                )
            )
        lines[target_start_line - 1 : target_end_line] = [
            "".join(projected_lines)
        ]
        projections = {
            str(match["repair_id"]): {
                "matched_text": str(match["matched_text"]),
                "source_start": int(match["start"]),
                "source_end": int(match["end"]),
            }
            for match in ordered_matches
        }
        return "".join(lines), projections

    def _grounded_fragment_match(
        self,
        *,
        region: str,
        item: RepairItem,
    ) -> re.Match[str]:
        semantic_payload = item.metadata.get("candidate_semantics")
        semantic_summary = (
            semantic_payload.get("summary")
            if isinstance(semantic_payload, dict)
            else None
        )
        candidate_fragments: list[str] = []
        for value in (
            item.raw_text,
            semantic_summary,
            item.overreach_summary,
        ):
            fragment = str(value or "").strip()
            if fragment and fragment not in candidate_fragments:
                candidate_fragments.append(fragment)

        unique_matches: list[tuple[int, re.Match[str]]] = []
        ambiguous_fragments: list[str] = []
        for fragment in candidate_fragments:
            tokens = re.findall(r"\S+", fragment)
            if not tokens:
                continue
            pattern = re.compile(
                r"\s+".join(re.escape(token) for token in tokens),
                flags=re.IGNORECASE,
            )
            found = list(pattern.finditer(region))
            if len(found) > 1:
                ambiguous_fragments.append(fragment)
            elif len(found) == 1:
                unique_matches.append((len(found[0].group(0)), found[0]))
        if ambiguous_fragments:
            raise RuntimeError(
                "instruction projection failed closed: grounded fragment is "
                f"not unique for {item.repair_id}"
            )
        if not unique_matches:
            raise RuntimeError(
                "instruction projection failed closed: no grounded raw-text "
                f"or semantic fragment matched {item.repair_id}"
            )

        # Prefer the narrowest grounded fragment.  ``raw_text`` may cite the
        # complete Markdown block while the candidate summary identifies the
        # atomic action recovered from that block.
        unique_matches.sort(
            key=lambda value: (
                value[0],
                value[1].start(),
                value[1].end(),
            )
        )
        narrowest_length = unique_matches[0][0]
        narrowest = [
            match
            for length, match in unique_matches
            if length == narrowest_length
        ]
        spans = {(match.start(), match.end()) for match in narrowest}
        if len(spans) != 1:
            raise RuntimeError(
                "instruction projection failed closed: grounded semantic "
                f"fragments disagree for {item.repair_id}"
            )
        return narrowest[0]

    def _project_compound_bullet_line(
        self,
        *,
        line: str,
        matches: list[dict[str, object]],
    ) -> str:
        newline_match = re.search(r"(?:\r\n|\n|\r)$", line)
        newline = newline_match.group(0) if newline_match else "\n"
        body = line[: -len(newline)] if newline_match else line
        bullet = re.match(r"^(?P<marker>-\s+)", body)
        if bullet is None:
            raise RuntimeError(
                "instruction projection failed closed: partial atomic-action "
                "projection requires a top-level '-' Markdown bullet"
            )
        content_start = bullet.end()
        ordered = sorted(
            matches,
            key=lambda value: (
                int(value["start"]),
                int(value["end"]),
                str(value["repair_id"]),
            ),
        )
        if any(int(match["start"]) < content_start for match in ordered):
            raise RuntimeError(
                "instruction projection failed closed: grounded fragment "
                "overlaps Markdown structure"
            )

        output: list[str] = []
        cursor = content_start
        for index, match in enumerate(ordered):
            residual = body[cursor : int(match["start"])]
            cleaned = self._clean_surrounding_instruction(
                residual,
                strip_leading=index > 0,
                strip_trailing=True,
            )
            if cleaned:
                output.append(f"- {cleaned}{newline}")
            item = match["item"]
            assert isinstance(item, RepairItem)
            output.extend(self._replacement_lines(item))
            cursor = int(match["end"])
        trailing = self._clean_surrounding_instruction(
            body[cursor:],
            strip_leading=True,
            strip_trailing=False,
        )
        if trailing:
            output.append(f"- {trailing}{newline}")
        if not output:
            raise RuntimeError(
                "instruction projection failed closed: grounded projection "
                "produced an empty Markdown block"
            )
        return "".join(output)

    def _clean_surrounding_instruction(
        self,
        text: str,
        *,
        strip_leading: bool,
        strip_trailing: bool,
    ) -> str:
        value = text.strip()
        if strip_leading:
            value = re.sub(
                r"^(?:(?:,|;|:)\s*)?(?:(?:and\s+then|and|then)\b\s*)?",
                "",
                value,
                flags=re.IGNORECASE,
            ).strip()
        if strip_trailing:
            value = re.sub(
                r"\s*(?:(?:,|;|:)\s*)?(?:and\s+then|and|then)\s*$",
                "",
                value,
                flags=re.IGNORECASE,
            ).strip()
            value = re.sub(r"[,;:]\s*$", "", value).strip()
        if re.fullmatch(r"[.,;:!?\s]*", value):
            return ""
        return value

    def _contains_line_block(self, *, content: str, block: str) -> bool:
        content_lines = [
            self._normalize_projection_line(line)
            for line in content.splitlines()
        ]
        block_lines = [
            self._normalize_projection_line(line)
            for line in block.splitlines()
        ]
        if not block_lines or len(block_lines) > len(content_lines):
            return False
        block_size = len(block_lines)
        return any(
            content_lines[index : index + block_size] == block_lines
            for index in range(len(content_lines) - block_size + 1)
        )

    def _normalize_projection_line(self, line: str) -> str:
        return " ".join(line.split()).casefold()

    def _rewrite_with_fallback(self, *, target_file: Path, item: RepairItem) -> list[str]:
        lines = target_file.read_text(encoding="utf-8").splitlines(keepends=True)
        replacement_lines = self._replacement_lines(item)

        if item.repair_type == "REORGANIZE_CODE_AND_ADD_DISPATCH":
            updated = self._rewrite_dispatch_lines(lines, item, replacement_lines)
            target_file.write_text(updated, encoding="utf-8")
            item.metadata["instruction_projection_strategy"] = "fallback"
            return [f"Rewrote instruction dispatch in {target_file.name} for {item.repair_id}."]

        content = "".join(lines)
        instruction_file = str(item.source_file or target_file.name)
        target_start_line, target_end_line = self._projection_target_lines(
            content=content,
            instruction_file=instruction_file,
            item=item,
        )
        updated_text, projections = self._project_guard_fragments(
            content=content,
            instruction_file=instruction_file,
            items=[item],
            target_start_line=target_start_line,
            target_end_line=target_end_line,
        )

        target_file.write_text(updated_text, encoding="utf-8")
        item.metadata["instruction_projection_strategy"] = "fallback"
        item.metadata["instruction_projection_scope"] = "grounded_fragment"
        item.metadata["instruction_projection_fragment"] = projections[
            item.repair_id
        ]["matched_text"]
        return [
            f"Guarded instruction-level overreach in {target_file.name} "
            f"for {item.repair_id}."
        ]

    def _item_payload(self, item: RepairItem) -> dict[str, object]:
        return {
            "repair_id": item.repair_id,
            "overreach_id": item.overreach_id,
            "layer": item.layer,
            "repair_type": item.repair_type,
            "overreach_summary": item.overreach_summary,
            "rationale": item.rationale,
            "guard_condition": item.guard_condition,
            "descriptor_ids": list(item.descriptor_ids),
            "allowed_cluster_keys": list(item.allowed_cluster_keys),
            "blocked_cluster_keys": list(item.blocked_cluster_keys),
            "descriptor_contexts": item.metadata.get("descriptor_contexts") or [],
            "allowed_descriptor_clusters": item.metadata.get("allowed_descriptor_clusters") or [],
            "blocked_descriptor_clusters": item.metadata.get("blocked_descriptor_clusters") or [],
            "target_files": item.target_files,
            "source_file": item.source_file,
            "source_start_line": item.source_start_line,
            "source_end_line": item.source_end_line,
            "raw_text": item.raw_text,
            "generated_files": item.generated_files,
            "metadata": item.metadata,
        }

    def _replacement_lines(self, item: RepairItem) -> list[str]:
        if item.repair_type == "GUARD_INSTRUCTION_TASK_CONDITIONED":
            text = str(item.metadata.get("guard_instruction_text") or item.guard_condition or "")
            return [
                self._ensure_bullet_line(line)
                for line in text.splitlines()
                if line.strip()
            ]
        if item.repair_type == "REORGANIZE_CODE_AND_ADD_DISPATCH":
            return self._dispatch_replacement_lines(item)
        return []

    def _record_instruction_dispatch(self, item: RepairItem) -> None:
        replacement_lines = self._dispatch_replacement_lines(item)
        block = "".join(replacement_lines).strip()
        item.metadata["instruction_dispatch_block"] = block
        item.metadata["dispatch_instruction_text"] = block
        item.metadata["instruction_routing_policy"] = (
            "independent_semantic_action_conditions_with_safe_default"
            if item.metadata.get("source_variant_manifest")
            else "semantic_agent_selection_with_safe_default"
        )
        item.metadata["non_semantic_planner_fallback"] = "safe_unit_only"

    def _dispatch_replacement_lines(self, item: RepairItem) -> list[str]:
        composite_lines = self._composite_dispatch_replacement_lines(item)
        if composite_lines is not None:
            return composite_lines
        allowed_path, safe_path = self._execution_unit_paths(item)
        if not allowed_path or not safe_path:
            return [
                self._ensure_bullet_line(
                    str(item.guard_condition or "")
                    + " No grounded task-specific execution units were "
                    "available, so use only the safe public behavior."
                )
            ]
        allowed_command = self._execution_command(allowed_path, item=item)
        safe_command = self._execution_command(safe_path, item=item)
        guard = str(item.guard_condition or "").strip()
        return [
            self._ensure_bullet_line(
                "If and only if the current request meets the semantic "
                "authorized-and-necessary condition described in this "
                f"instruction, run `{allowed_command}`. The condition is: "
                f"{guard} The agent must make this semantic "
                "instruction-plan decision; do not translate it into a rigid "
                "keyword, exact-prompt, task-ID, cluster-ID, hash, or environment "
                "variable check."
            ),
            self._ensure_bullet_line(
                "Otherwise, including for a local-only, blocked, missing, "
                "incomplete, negated, ambiguous, or unmatched request, run "
                f"`{safe_command}`."
            ),
            self._ensure_bullet_line(
                "The original public code entrypoint is a safe-only compatibility "
                "path and never grants access to the guarded action."
            ),
        ]

    def _composite_dispatch_replacement_lines(
        self, item: RepairItem
    ) -> list[str] | None:
        variants = item.metadata.get("source_variant_manifest")
        guard_specs = item.metadata.get("composite_guard_specs")
        if not isinstance(variants, list) or not isinstance(guard_specs, list):
            return None
        if not variants or not guard_specs:
            return None
        if any(not isinstance(value, dict) for value in variants + guard_specs):
            return None

        ordered_specs = sorted(
            guard_specs,
            key=lambda value: str(value.get("repair_id") or ""),
        )
        labels_by_repair_id = {
            str(spec.get("repair_id") or ""): f"Action {index}"
            for index, spec in enumerate(ordered_specs, start=1)
        }
        lines = [
            self._ensure_bullet_line(
                "Evaluate the following task-conditioned action conditions "
                "independently before running this script. A condition holds "
                "only when the current request semantically authorizes and "
                "requires that action; treat missing, incomplete, negated, "
                "ambiguous, or unmatched context as false."
            )
        ]
        for index, spec in enumerate(ordered_specs, start=1):
            action_summary = str(
                spec.get("action_summary") or f"guarded action {index}"
            ).strip()
            guard_condition = str(spec.get("guard_condition") or "").strip()
            lines.append(
                self._ensure_bullet_line(
                    f"Action {index} ({action_summary}): {guard_condition}"
                )
            )

        lines.append(
            self._ensure_bullet_line(
                "After evaluating all conditions, select exactly one of the "
                "following execution units from the complete condition set and "
                "run it exactly once. Never execute multiple units for one "
                "request and never infer a condition from a filename, action "
                "number, task ID, cluster ID, hash, keyword, or environment "
                "variable."
            )
        )
        all_repair_ids = set(labels_by_repair_id)
        for variant in sorted(
            variants,
            key=lambda value: (
                len(value.get("allowed_repair_ids") or []),
                str(value.get("mask") or ""),
            ),
        ):
            allowed_ids = {
                str(value)
                for value in variant.get("allowed_repair_ids") or []
            }
            blocked_ids = all_repair_ids - allowed_ids
            if allowed_ids:
                holds = ", ".join(
                    labels_by_repair_id[value]
                    for value in sorted(allowed_ids)
                )
                does_not_hold = ", ".join(
                    labels_by_repair_id[value]
                    for value in sorted(blocked_ids)
                )
                condition_text = f"when {holds} hold"
                if blocked_ids:
                    condition_text += f" and {does_not_hold} do not hold"
            else:
                condition_text = "when none of the action conditions hold"
            relative_path = str(variant.get("relative_path") or "")
            if not relative_path:
                return None
            lines.append(
                self._ensure_bullet_line(
                    f"{condition_text}, run "
                    f"`{self._execution_command(relative_path, item=item)}`."
                )
            )

        safe_variants = [
            value for value in variants if value.get("safe_default") is True
        ]
        if len(safe_variants) != 1:
            return None
        safe_path = str(safe_variants[0].get("relative_path") or "")
        lines.append(
            self._ensure_bullet_line(
                "The original public code entrypoint is a safe-only "
                "compatibility path that runs "
                f"`{self._execution_command(safe_path, item=item)}` "
                "and never grants any guarded action."
            )
        )
        return lines

    def _execution_unit_paths(
        self, item: RepairItem
    ) -> tuple[str | None, str | None]:
        allowed = item.metadata.get("allowed_execution_unit")
        safe = item.metadata.get("safe_execution_unit")
        if isinstance(allowed, str) and isinstance(safe, str):
            return allowed, safe
        if len(item.generated_files) >= 2:
            return item.generated_files[0], item.generated_files[1]
        source_file = item.source_file
        if not source_file:
            return None, None
        source = Path(source_file)
        allowed_path = source.with_name(
            f"{source.stem}__task_allowed{source.suffix}"
        ).as_posix()
        safe_path = source.with_name(
            f"{source.stem}__default_safe{source.suffix}"
        ).as_posix()
        return allowed_path, safe_path

    def _capture_instruction_invocation(
        self,
        *,
        target_file: Path,
        item: RepairItem,
    ) -> None:
        """Preserve the original interpreter, flags, and script arguments.

        The code projector changes only the selected execution-unit path.  A
        surrounding instruction such as ``python3 report.py --format json``
        must therefore retain both the interpreter and every argument when it
        is rewritten to an allowed/safe unit.
        """

        source_file = str(
            item.metadata.get("dispatch_source_file")
            or item.source_file
            or ""
        ).strip()
        if not source_file:
            return
        content = target_file.read_text(encoding="utf-8")
        for command in re.findall(r"`([^`\n]+)`", content):
            try:
                tokens = shlex.split(command, posix=True)
            except ValueError:
                continue
            source_index = self._source_token_index(
                tokens=tokens,
                source_file=source_file,
            )
            if source_index is None:
                continue
            item.metadata["instruction_invocation_template"] = {
                "prefix_tokens": tokens[:source_index],
                "source_token": tokens[source_index],
                "suffix_tokens": tokens[source_index + 1 :],
            }
            return

    def _source_token_index(
        self,
        *,
        tokens: list[str],
        source_file: str,
    ) -> int | None:
        normalized_source = source_file.replace("\\", "/").removeprefix(
            "./"
        )
        source_name = Path(normalized_source).name
        exact_matches: list[int] = []
        basename_matches: list[int] = []
        for index, token in enumerate(tokens):
            normalized_token = token.replace("\\", "/").removeprefix("./")
            if (
                normalized_token == normalized_source
                or normalized_token.endswith("/" + normalized_source)
            ):
                exact_matches.append(index)
            elif Path(normalized_token).name == source_name:
                basename_matches.append(index)
        if len(exact_matches) == 1:
            return exact_matches[0]
        if not exact_matches and len(basename_matches) == 1:
            return basename_matches[0]
        return None

    def _execution_command(
        self,
        relative_path: str,
        *,
        item: RepairItem,
    ) -> str:
        template = item.metadata.get("instruction_invocation_template")
        if isinstance(template, dict):
            prefix = template.get("prefix_tokens")
            suffix_tokens = template.get("suffix_tokens")
            if (
                isinstance(prefix, list)
                and all(isinstance(token, str) for token in prefix)
                and isinstance(suffix_tokens, list)
                and all(isinstance(token, str) for token in suffix_tokens)
            ):
                return shlex.join(
                    [*prefix, relative_path, *suffix_tokens]
                )
        quoted_path = shlex.quote(relative_path)
        suffix = Path(relative_path).suffix.casefold()
        if suffix == ".sh":
            return f"sh {quoted_path}"
        if suffix == ".py":
            return f"python3 {quoted_path}"
        if suffix in {".js", ".mjs", ".cjs", ".ts", ".mts", ".cts"}:
            return f"node {quoted_path}"
        return quoted_path

    def _rewrite_dispatch_lines(self, lines: list[str], item: RepairItem, replacement_lines: list[str]) -> str:
        dispatch_script = item.metadata.get("dispatch_source_file")
        if isinstance(dispatch_script, str):
            for index, line in enumerate(lines):
                if dispatch_script in line:
                    lines[index : index + 1] = replacement_lines
                    return "".join(lines)
        if item.raw_text:
            updated = self._replace_by_raw_text("".join(lines), item, replacement_lines)
            if updated != "".join(lines):
                return updated
        append_text = "".join(lines)
        if append_text and not append_text.endswith("\n"):
            append_text += "\n"
        append_text += "".join(replacement_lines)
        return append_text

    def _replace_by_raw_text(self, content: str, item: RepairItem, replacement_lines: list[str]) -> str:
        raw_text = (item.raw_text or "").strip()
        if not raw_text:
            return content
        replacement = "".join(replacement_lines)
        replaced = False
        updated_lines: list[str] = []
        for line in content.splitlines(keepends=True):
            if raw_text in line and not replaced:
                if replacement:
                    updated_lines.append(replacement)
                replaced = True
            else:
                updated_lines.append(line)
        return "".join(updated_lines)

    def _line_slice(self, lines: list[str], item: RepairItem) -> tuple[int | None, int | None]:
        if item.source_start_line is None or item.source_end_line is None:
            return None, None
        start = max(item.source_start_line - 1, 0)
        end = min(item.source_end_line, len(lines))
        if start >= end:
            return None, None
        return start, end

    def _resolve_instruction_file(self, patched_bundle_root: Path, item: RepairItem) -> Path | None:
        for path in item.target_files:
            if path.endswith(".md"):
                target_file = self._safe_target_path(
                    patched_bundle_root, path
                )
                if target_file is not None and target_file.exists():
                    return target_file
        target_file = self._safe_target_path(
            patched_bundle_root, "SKILL.md"
        )
        if target_file is None:
            return None
        return target_file if target_file.exists() else None

    def _safe_target_path(
        self, root: Path, relative_path: str
    ) -> Path | None:
        root = root.resolve()
        target = (root / relative_path).resolve()
        if target != root and root not in target.parents:
            return None
        return target

    def _ensure_bullet_line(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return ""
        if stripped.startswith(("-", "*")) or stripped[:2].isdigit():
            return stripped + ("\n" if not stripped.endswith("\n") else "")
        return f"- {stripped}\n"

    def _ensure_trailing_newline(self, text: str) -> str:
        return text if text.endswith("\n") else text + "\n"
