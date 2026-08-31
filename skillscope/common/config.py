from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class LLMConfig:
    provider: str = "disabled"
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    max_concurrency: int = 4
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 4_000
    json_mode: bool = True
    debug_enabled: bool = False
    debug_log_path: Path | None = None
    debug_preview_chars: int = 500

    @property
    def enabled(self) -> bool:
        return self.provider != "disabled" and bool(self.api_key) and bool(self.base_url) and bool(self.model)


@dataclass(slots=True)
class AppConfig:
    project_root: Path
    artifact_root: Path
    llm: LLMConfig = field(default_factory=LLMConfig)
    validation_mode: str = "dynamic"
    sandbox_timeout_seconds: float = 60.0
    sandbox_require_os_isolation: bool = True
    sandbox_allow_network: bool = False
    max_preview_chars: int = 2_000
    max_instruction_lines: int = 200
    metadata_filenames: tuple[str, ...] = ("_meta.json", "meta.json", "skill.json")
    instruction_filenames: tuple[str, ...] = ("SKILL.md",)
    code_extensions: tuple[str, ...] = (".py", ".sh", ".js", ".ts")
    risky_network_keywords: tuple[str, ...] = ("http", "https", "telegram", "webhook", "socket", "upload")
    risky_command_keywords: tuple[str, ...] = ("subprocess", "os.system", "exec", "bash", "sh", "zsh")
    risky_sensitive_keywords: tuple[str, ...] = ("history", ".ssh", ".aws", "token", "credential", "cookie")
    default_notes: list[str] = field(default_factory=list)

    @classmethod
    def from_project_root(
        cls,
        project_root: Path,
        artifact_root: Path | None = None,
        validation_mode: str | None = None,
    ) -> "AppConfig":
        root = project_root.resolve()
        env_values = _load_env_settings(root / ".env")
        resolved_artifact_root = artifact_root.resolve() if artifact_root is not None else (root / "artifacts")
        debug_log_path_value = _env(
            env_values,
            "SKILLSCOPE_LLM_DEBUG_PATH",
            str(resolved_artifact_root / "_llm_debug" / "llm_calls.jsonl"),
        )
        resolved_validation_mode = _normalize_validation_mode(
            validation_mode if validation_mode is not None else _env(env_values, "SKILLSCOPE_VALIDATION_MODE", "dynamic")
        )
        return cls(
            project_root=root,
            artifact_root=resolved_artifact_root,
            llm=LLMConfig(
                provider=_env(env_values, "SKILLSCOPE_LLM_PROVIDER", "disabled"),
                api_key=_env(env_values, "SKILLSCOPE_LLM_API_KEY"),
                base_url=_env(env_values, "SKILLSCOPE_LLM_BASE_URL"),
                model=_env(env_values, "SKILLSCOPE_LLM_MODEL"),
                max_concurrency=_env_int(env_values, "SKILLSCOPE_LLM_MAX_CONCURRENCY", 4),
                timeout_seconds=_env_float(env_values, "SKILLSCOPE_LLM_TIMEOUT_SECONDS", 60.0),
                temperature=_env_float(env_values, "SKILLSCOPE_LLM_TEMPERATURE", 0.0),
                max_tokens=_env_int(env_values, "SKILLSCOPE_LLM_MAX_TOKENS", 4_000),
                json_mode=_env_bool(env_values, "SKILLSCOPE_LLM_JSON_MODE", True),
                debug_enabled=_env_bool(env_values, "SKILLSCOPE_LLM_DEBUG", False),
                debug_log_path=_resolve_path(root, debug_log_path_value) if debug_log_path_value is not None else None,
                debug_preview_chars=_env_int(env_values, "SKILLSCOPE_LLM_DEBUG_PREVIEW_CHARS", 500),
            ),
            validation_mode=resolved_validation_mode,
            sandbox_timeout_seconds=_env_float(env_values, "SKILLSCOPE_SANDBOX_TIMEOUT_SECONDS", 60.0),
            sandbox_require_os_isolation=_env_bool(
                env_values,
                "SKILLSCOPE_SANDBOX_REQUIRE_OS_ISOLATION",
                True,
            ),
            sandbox_allow_network=_env_bool(env_values, "SKILLSCOPE_SANDBOX_ALLOW_NETWORK", False),
        )

    def artifact_dir_for(self, skill_name: str, mode: str) -> Path:
        return self.artifact_root / skill_name / mode


def _load_env_settings(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _env(values: dict[str, str], key: str, default: str | None = None) -> str | None:
    if key in os.environ:
        return os.environ[key]
    return values.get(key, default)


def _env_bool(values: dict[str, str], key: str, default: bool) -> bool:
    value = _env(values, key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(values: dict[str, str], key: str, default: int) -> int:
    value = _env(values, key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(values: dict[str, str], key: str, default: float) -> float:
    value = _env(values, key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _normalize_validation_mode(value: str | None) -> str:
    normalized = (value or "dynamic").strip().lower()
    if normalized not in {"dynamic", "static"}:
        return "dynamic"
    return normalized
