from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from skillscope.common.config import AppConfig
from skillscope.common.models import SkillArtifact, SkillBundle


YAML_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class SkillBundleLoader:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def load(self, skill_root: Path) -> SkillBundle:
        root = skill_root.resolve()
        if not root.is_dir():
            raise ValueError(f"Skill bundle root is not a directory: {root}")
        artifacts: list[SkillArtifact] = []
        metadata_files: list[SkillArtifact] = []
        instruction_files: list[SkillArtifact] = []
        script_files: list[SkillArtifact] = []
        resource_files: list[SkillArtifact] = []
        json_metadata: dict[str, Any] = {}
        frontmatter_sources: list[dict[str, Any]] = []
        metadata_sources: list[dict[str, str]] = []

        for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
            self._assert_confined_artifact(root, file_path)
            if self._is_python_cache_artifact(root, file_path):
                continue
            artifact = self._build_artifact(root, file_path, len(artifacts))
            artifacts.append(artifact)

            if artifact.role == "metadata":
                metadata_files.append(artifact)
                loaded = self._load_json(file_path)
                if isinstance(loaded, dict):
                    json_metadata.update(loaded)
                    metadata_sources.append({"file": artifact.relative_path, "format": "json"})
            elif artifact.role == "instruction":
                instruction_files.append(artifact)
                frontmatter = self._load_frontmatter(file_path)
                if frontmatter:
                    frontmatter_sources.append({"file": artifact.relative_path, "data": frontmatter})
                    metadata_sources.append({"file": artifact.relative_path, "format": "yaml_frontmatter"})
            elif artifact.role == "script":
                script_files.append(artifact)
            else:
                resource_files.append(artifact)

        metadata: dict[str, Any] = dict(json_metadata)
        for source in frontmatter_sources:
            metadata.update(source["data"])
        metadata["_skillscope_bundle"] = {
            "metadata_sources": metadata_sources,
            "frontmatter_sources": frontmatter_sources,
            "instruction_files": [self._artifact_inventory_item(artifact) for artifact in instruction_files],
            "script_files": [self._artifact_inventory_item(artifact) for artifact in script_files],
            "resource_files": [self._artifact_inventory_item(artifact) for artifact in resource_files],
        }

        return SkillBundle(
            bundle_id=root.name,
            root_path=str(root),
            metadata=metadata,
            metadata_files=metadata_files,
            instruction_files=instruction_files,
            script_files=script_files,
            resource_files=resource_files,
            artifacts=artifacts,
        )

    def _is_python_cache_artifact(
        self,
        root: Path,
        file_path: Path,
    ) -> bool:
        """Return whether *file_path* is generated Python bytecode state.

        Executing a Skill can create these files inside its bundle.  They are
        runtime cache state rather than authored Skill artifacts and must not
        become script/resource inputs on a later analysis run.
        """

        relative_path = file_path.relative_to(root)
        return (
            any(part.casefold() == "__pycache__" for part in relative_path.parts[:-1])
            or file_path.suffix.casefold() in {".pyc", ".pyo"}
        )

    def _assert_confined_artifact(
        self,
        root: Path,
        file_path: Path,
    ) -> None:
        try:
            file_path.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "Skill bundle artifact resolves outside the bundle root: "
                f"{file_path}"
            ) from exc

    def _build_artifact(self, root: Path, file_path: Path, index: int) -> SkillArtifact:
        relative_path = file_path.relative_to(root).as_posix()
        role = self._infer_role(file_path)
        preview, is_binary = self._read_preview(file_path)
        return SkillArtifact(
            artifact_id=f"artifact-{index:04d}",
            relative_path=relative_path,
            absolute_path=str(file_path),
            kind=file_path.suffix.lower().lstrip(".") or "file",
            role=role,
            size_bytes=file_path.stat().st_size,
            text_preview=preview,
            is_binary=is_binary,
        )

    def _infer_role(self, file_path: Path) -> str:
        filename = file_path.name.lower()
        if filename in {name.lower() for name in self.config.metadata_filenames}:
            return "metadata"
        if filename in {name.lower() for name in self.config.instruction_filenames}:
            return "instruction"
        if file_path.suffix.lower() in self.config.code_extensions or "scripts" in file_path.parts:
            return "script"
        return "resource"

    def _read_preview(self, file_path: Path) -> tuple[str, bool]:
        try:
            text = file_path.read_text(encoding="utf-8")
            return text[: self.config.max_preview_chars], False
        except (OSError, UnicodeDecodeError):
            return "", True

    def _load_json(self, file_path: Path) -> object:
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _load_frontmatter(self, file_path: Path) -> dict[str, Any]:
        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return {}
        return parse_yaml_frontmatter(text)

    def _artifact_inventory_item(self, artifact: SkillArtifact) -> dict[str, Any]:
        return {
            "path": artifact.relative_path,
            "kind": artifact.kind,
            "role": artifact.role,
            "size_bytes": artifact.size_bytes,
            "is_binary": artifact.is_binary,
            "preview": artifact.text_preview,
        }


def parse_yaml_frontmatter(text: str) -> dict[str, Any]:
    """Parse the conservative YAML subset commonly used by SKILL.md.

    SkillScope deliberately stays dependency-free. Unsupported YAML constructs
    remain strings instead of being evaluated or silently discarded.
    """

    lines = text.lstrip("\ufeff").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    closing_index = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() in {"---", "..."}),
        None,
    )
    if closing_index is None:
        return {}
    parsed, _ = _parse_yaml_block(lines[1:closing_index], 0, 0)
    return parsed if isinstance(parsed, dict) else {}


def _parse_yaml_block(lines: list[str], start: int, indent: int) -> tuple[Any, int]:
    index = _next_yaml_content_line(lines, start)
    if index >= len(lines):
        return {}, index
    first_indent = _yaml_indent(lines[index])
    if first_indent < indent:
        return {}, index
    indent = first_indent
    if lines[index].lstrip().startswith("- "):
        return _parse_yaml_list(lines, index, indent)
    return _parse_yaml_mapping(lines, index, indent)


def _parse_yaml_mapping(lines: list[str], start: int, indent: int) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    index = start
    while index < len(lines):
        index = _next_yaml_content_line(lines, index)
        if index >= len(lines):
            break
        current_indent = _yaml_indent(lines[index])
        if current_indent < indent:
            break
        if current_indent > indent:
            index += 1
            continue
        content = _strip_yaml_comment(lines[index].strip())
        if content.startswith("- "):
            break
        entry = _split_yaml_mapping_entry(content)
        if entry is None:
            index += 1
            continue
        key, raw_value = entry
        if raw_value in {"|", "|-", "|+", ">", ">-", ">+"}:
            value, index = _parse_yaml_block_scalar(lines, index + 1, indent, folded=raw_value.startswith(">"))
            result[key] = value
            continue
        if raw_value:
            result[key] = _parse_yaml_scalar(raw_value)
            index += 1
            continue

        child_index = _next_yaml_content_line(lines, index + 1)
        if child_index < len(lines) and _yaml_indent(lines[child_index]) > indent:
            child, index = _parse_yaml_block(lines, child_index, _yaml_indent(lines[child_index]))
            result[key] = child
        else:
            result[key] = None
            index += 1
    return result, index


def _parse_yaml_list(lines: list[str], start: int, indent: int) -> tuple[list[Any], int]:
    result: list[Any] = []
    index = start
    while index < len(lines):
        index = _next_yaml_content_line(lines, index)
        if index >= len(lines):
            break
        current_indent = _yaml_indent(lines[index])
        if current_indent < indent:
            break
        if current_indent != indent or not lines[index].lstrip().startswith("- "):
            break
        content = _strip_yaml_comment(lines[index].lstrip()[2:].strip())
        if not content:
            child_index = _next_yaml_content_line(lines, index + 1)
            if child_index < len(lines) and _yaml_indent(lines[child_index]) > indent:
                child, index = _parse_yaml_block(lines, child_index, _yaml_indent(lines[child_index]))
                result.append(child)
            else:
                result.append(None)
                index += 1
            continue

        entry = _split_yaml_mapping_entry(content)
        if entry is None:
            result.append(_parse_yaml_scalar(content))
            index += 1
            continue

        key, raw_value = entry
        item: dict[str, Any] = {key: _parse_yaml_scalar(raw_value) if raw_value else None}
        child_index = _next_yaml_content_line(lines, index + 1)
        if child_index < len(lines) and _yaml_indent(lines[child_index]) > indent:
            child, index = _parse_yaml_block(lines, child_index, _yaml_indent(lines[child_index]))
            if raw_value:
                if isinstance(child, dict):
                    item.update(child)
            else:
                item[key] = child
        else:
            index += 1
        result.append(item)
    return result, index


def _parse_yaml_block_scalar(
    lines: list[str],
    start: int,
    parent_indent: int,
    *,
    folded: bool,
) -> tuple[str, int]:
    collected: list[str] = []
    index = start
    content_indent: int | None = None
    while index < len(lines):
        raw_line = lines[index]
        if not raw_line.strip():
            collected.append("")
            index += 1
            continue
        current_indent = _yaml_indent(raw_line)
        if current_indent <= parent_indent:
            break
        if content_indent is None:
            content_indent = current_indent
        collected.append(raw_line[min(content_indent, len(raw_line)) :])
        index += 1
    if folded:
        return " ".join(part.strip() for part in collected if part.strip()), index
    return "\n".join(collected).rstrip(), index


def _parse_yaml_scalar(raw_value: str) -> Any:
    value = _strip_yaml_comment(raw_value.strip())
    if not value:
        return ""
    if value[0:1] in {"'", '"'} and value[-1:] == value[0]:
        if value[0] == '"':
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        return value[1:-1].replace("''", "'")
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        return [_parse_yaml_scalar(part) for part in _split_yaml_inline(value[1:-1]) if part.strip()]
    if value.startswith("{") and value.endswith("}"):
        mapping: dict[str, Any] = {}
        for part in _split_yaml_inline(value[1:-1]):
            entry = _split_yaml_mapping_entry(part.strip())
            if entry is not None:
                mapping[entry[0]] = _parse_yaml_scalar(entry[1])
        return mapping
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _split_yaml_mapping_entry(content: str) -> tuple[str, str] | None:
    quote: str | None = None
    for index, character in enumerate(content):
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
        elif character == ":" and quote is None:
            key = content[:index].strip()
            if not key or not YAML_KEY_RE.fullmatch(key):
                return None
            return key, content[index + 1 :].strip()
    return None


def _split_yaml_inline(content: str) -> list[str]:
    parts: list[str] = []
    start = 0
    quote: str | None = None
    depth = 0
    for index, character in enumerate(content):
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
        elif quote is None:
            if character in "[{(":
                depth += 1
            elif character in "]})":
                depth = max(0, depth - 1)
            elif character == "," and depth == 0:
                parts.append(content[start:index].strip())
                start = index + 1
    parts.append(content[start:].strip())
    return parts


def _strip_yaml_comment(content: str) -> str:
    quote: str | None = None
    for index, character in enumerate(content):
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
        elif character == "#" and quote is None and (index == 0 or content[index - 1].isspace()):
            return content[:index].rstrip()
    return content


def _next_yaml_content_line(lines: list[str], start: int) -> int:
    index = start
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped and not stripped.startswith("#"):
            break
        index += 1
    return index


def _yaml_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))
