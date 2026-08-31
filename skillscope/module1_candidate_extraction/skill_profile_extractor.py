from __future__ import annotations

from typing import Any, Iterable

from skillscope.common.models import SkillBundle, SkillProfile


class SkillProfileExtractor:
    _NAME_KEYS = ("name", "title", "skill_name", "名称")
    _DESCRIPTION_KEYS = ("description", "summary", "purpose", "简介", "说明", "用途")
    _USE_WHEN_KEYS = (
        "use_when",
        "use-when",
        "when_to_use",
        "when-to-use",
        "triggers",
        "trigger",
        "适用场景",
        "何时使用",
    )
    _CAPABILITY_KEYS = (
        "capabilities",
        "capability",
        "features",
        "tools",
        "allowed-tools",
        "allowed_tools",
        "能力",
        "工具",
    )
    _OUTPUT_KEYS = ("outputs", "output", "produces", "artifacts", "输出", "产物")
    _DATA_SCOPE_KEYS = ("data_scope", "data-scope", "inputs", "input", "data", "数据范围", "输入")
    _EXECUTION_SCOPE_KEYS = (
        "execution_scope",
        "execution-scope",
        "permissions",
        "permission",
        "access",
        "执行范围",
        "权限",
    )
    _RESOURCE_KEYS = (
        "resources",
        "resource",
        "scripts",
        "assets",
        "templates",
        "references",
        "资源",
        "脚本",
    )

    def extract(self, bundle: SkillBundle) -> SkillProfile:
        metadata = bundle.metadata or {}
        bundle_inventory = metadata.get("_skillscope_bundle")
        if not isinstance(bundle_inventory, dict):
            bundle_inventory = {}

        instruction_summaries = [
            {
                "file": artifact.relative_path,
                "summary": self._summarize_instruction_preview(
                    self._without_yaml_frontmatter(artifact.text_preview)
                ),
            }
            for artifact in bundle.instruction_files
        ]
        instruction_summary = " ".join(
            item["summary"] for item in instruction_summaries if item["summary"]
        ).strip()

        resource_summaries = [
            {
                "file": artifact.relative_path,
                "kind": artifact.kind,
                "summary": self._summarize_resource_preview(artifact.text_preview),
            }
            for artifact in bundle.resource_files
        ]
        inventory_paths = [
            artifact.relative_path
            for artifact in (*bundle.script_files, *bundle.resource_files)
        ]
        declared_resource_values = self._collect_declared_values(metadata, self._RESOURCE_KEYS)
        resource_summary = self._resource_summary(
            inventory_paths,
            declared_resource_values,
            resource_summaries,
        )
        declared_resource_summary = (
            f"Declared resources: {', '.join(declared_resource_values[:12])}."
            if declared_resource_values
            else ""
        )

        name = self._first_metadata_text(metadata, self._NAME_KEYS) or bundle.bundle_id
        description = self._first_metadata_text(metadata, self._DESCRIPTION_KEYS)
        use_when = self._first_metadata_text(metadata, self._USE_WHEN_KEYS)
        summary = "\n".join(
            part
            for part in (
                description,
                use_when,
                instruction_summary,
                declared_resource_summary,
            )
            if part
        ).strip()

        declared_capabilities = self._collect_declared_values(metadata, self._CAPABILITY_KEYS)
        declared_outputs = self._collect_declared_values(metadata, self._OUTPUT_KEYS)
        declared_data_scope = self._collect_declared_values(metadata, self._DATA_SCOPE_KEYS)
        declared_execution_scope = self._collect_declared_values(
            metadata,
            self._EXECUTION_SCOPE_KEYS,
        )
        inference_text = " ".join(
            (
                summary,
                " ".join(declared_capabilities),
                " ".join(declared_outputs),
                " ".join(declared_data_scope),
                " ".join(declared_execution_scope),
            )
        )

        capabilities = self._deduplicate(
            (*declared_capabilities, *self._infer_capabilities(inference_text))
        )
        outputs = self._deduplicate(
            (*declared_outputs, *self._infer_outputs(inference_text))
        )
        data_scope = self._deduplicate(
            (*declared_data_scope, *self._infer_data_scope(inference_text))
        )
        execution_scope = self._deduplicate(
            (*declared_execution_scope, *self._infer_execution_scope(inference_text))
        )

        return SkillProfile(
            name=name,
            description=description,
            use_when=use_when,
            summary=summary,
            declared_capabilities=capabilities,
            declared_outputs=outputs,
            declared_data_scope=data_scope,
            declared_execution_scope=execution_scope,
            evidence={
                "metadata_name": name,
                "metadata_description": description,
                "metadata_use_when": use_when,
                "metadata_sources": bundle_inventory.get("metadata_sources", []),
                "frontmatter_sources": bundle_inventory.get("frontmatter_sources", []),
                "declared_profile_text": summary,
                "instruction_summary": instruction_summary,
                "instruction_summaries": instruction_summaries,
                "declared_resources": declared_resource_values,
                "resource_paths": inventory_paths,
                "resource_summary": resource_summary,
                "resource_summaries": resource_summaries,
            },
        )

    def _summarize_instruction_preview(self, text: str) -> str:
        lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("```"):
                continue
            if line.startswith(("-", "*")) or line[:2].isdigit() or line[:1].isdigit():
                lines.append(line.lstrip("-*0123456789. ").strip())
            elif len(lines) < 3:
                lines.append(line)
            if len(lines) >= 5:
                break
        return " ".join(lines)

    def _summarize_resource_preview(self, text: str) -> str:
        if not text:
            return ""
        lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("#", "//", "/*", "*")):
                continue
            lines.append(line[:160])
            if len(lines) >= 2:
                break
        return " ".join(lines)

    def _without_yaml_frontmatter(self, text: str) -> str:
        lines = text.lstrip("\ufeff").splitlines()
        if not lines or lines[0].strip() != "---":
            return text
        closing_index = next(
            (
                index
                for index, line in enumerate(lines[1:], start=1)
                if line.strip() in {"---", "..."}
            ),
            None,
        )
        if closing_index is None:
            return text
        return "\n".join(lines[closing_index + 1 :])

    def _resource_summary(
        self,
        inventory_paths: list[str],
        declared_resources: list[str],
        resource_summaries: list[dict[str, str]],
    ) -> str:
        parts: list[str] = []
        if declared_resources:
            parts.append(f"Declared resources: {', '.join(declared_resources[:12])}.")
        if inventory_paths:
            parts.append(f"Bundled scripts/resources: {', '.join(inventory_paths[:20])}.")
        preview_parts = [
            f"{item['file']}: {item['summary']}"
            for item in resource_summaries[:6]
            if item["summary"]
        ]
        if preview_parts:
            parts.append("Resource evidence: " + " ".join(preview_parts))
        return " ".join(parts)

    def _first_metadata_text(self, metadata: dict[str, Any], keys: Iterable[str]) -> str:
        normalized_keys = {self._normalize_key(key) for key in keys}
        for mapping in self._metadata_mappings(metadata):
            for key, value in mapping.items():
                if self._normalize_key(str(key)) not in normalized_keys:
                    continue
                values = self._flatten_values(value)
                if values:
                    return "; ".join(values)
        return ""

    def _collect_declared_values(
        self,
        metadata: dict[str, Any],
        keys: Iterable[str],
    ) -> list[str]:
        normalized_keys = {self._normalize_key(key) for key in keys}
        values: list[str] = []
        for mapping in self._metadata_mappings(metadata):
            for key, value in mapping.items():
                if self._normalize_key(str(key)) in normalized_keys:
                    values.extend(self._flatten_values(value))
        return self._deduplicate(values)

    def _metadata_mappings(self, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        mappings = [metadata]
        for key in ("metadata", "profile", "skill"):
            nested = metadata.get(key)
            if isinstance(nested, dict):
                mappings.append(nested)
        return mappings

    def _flatten_values(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        if isinstance(value, (bool, int, float)):
            return [str(value)]
        if isinstance(value, (list, tuple, set)):
            flattened: list[str] = []
            for item in value:
                flattened.extend(self._flatten_values(item))
            return flattened
        if isinstance(value, dict):
            flattened = []
            for key, item in value.items():
                item_values = self._flatten_values(item)
                if item_values:
                    flattened.extend(f"{key}: {item_value}" for item_value in item_values)
                else:
                    flattened.append(str(key))
            return flattened
        return [str(value)]

    def _normalize_key(self, key: str) -> str:
        return key.strip().lower().replace("-", "_").replace(" ", "_")

    def _infer_capabilities(self, text: str) -> list[str]:
        lowered = text.lower()
        capabilities: list[str] = []
        keyword_map = {
            "local_analysis": (
                "analyze",
                "analysis",
                "local",
                "diagnostic",
                "monitor",
                "inspect",
                "分析",
                "诊断",
                "监控",
                "检查",
                "本地",
            ),
            "reporting": (
                "report",
                "summary",
                "graph",
                "heatmap",
                "chart",
                "报告",
                "总结",
                "摘要",
                "图表",
                "可视化",
            ),
            "network": (
                "api",
                "telegram",
                "webhook",
                "upload",
                "share",
                "send",
                "network",
                "网络",
                "发送",
                "上传",
                "分享",
            ),
            "command": (
                "command",
                "shell",
                "script",
                "bash",
                "powershell",
                "命令",
                "脚本",
                "运行",
                "执行",
            ),
            "file_ops": (
                "file",
                "read",
                "write",
                "save",
                "document",
                "文件",
                "读取",
                "写入",
                "保存",
                "文档",
            ),
        }
        for capability, keywords in keyword_map.items():
            if any(keyword in lowered for keyword in keywords):
                capabilities.append(capability)
        return capabilities

    def _infer_outputs(self, text: str) -> list[str]:
        lowered = text.lower()
        keyword_map = {
            "report": ("report", "报告"),
            "summary": ("summary", "摘要", "总结"),
            "graph": ("graph", "图", "关系图"),
            "heatmap": ("heatmap", "热力图"),
            "recommendation": ("recommendation", "建议", "推荐"),
            "status": ("status", "状态"),
        }
        return [
            output
            for output, keywords in keyword_map.items()
            if any(keyword in lowered for keyword in keywords)
        ]

    def _infer_data_scope(self, text: str) -> list[str]:
        lowered = text.lower()
        keyword_map = {
            "local files": ("local files", "local file", "本地文件"),
            "system status": ("system status", "系统状态"),
            "gpu": ("gpu", "显卡"),
            "cpu": ("cpu", "处理器"),
            "memory": ("memory", "内存"),
            "logs": ("logs", "log file", "日志"),
            "documents": ("documents", "document", "文档"),
        }
        return [
            scope
            for scope, keywords in keyword_map.items()
            if any(keyword in lowered for keyword in keywords)
        ]

    def _infer_execution_scope(self, text: str) -> list[str]:
        lowered = text.lower()
        keyword_map = {
            "network": ("network", "网络", "api", "webhook"),
            "command": ("command", "命令"),
            "shell": ("shell", "bash", "powershell", "终端"),
            "local": ("local", "本地"),
            "analysis": ("analysis", "analyze", "分析"),
            "file": ("file", "文件"),
        }
        return [
            scope
            for scope, keywords in keyword_map.items()
            if any(keyword in lowered for keyword in keywords)
        ]

    def _deduplicate(self, values: Iterable[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = str(value).strip()
            if not normalized:
                continue
            identity = normalized.casefold()
            if identity in seen:
                continue
            seen.add(identity)
            result.append(normalized)
        return result
