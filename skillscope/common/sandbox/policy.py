from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path


class SandboxPolicyError(RuntimeError):
    """Raised when an execution request cannot be confined safely."""


@dataclass(slots=True)
class SandboxPolicy:
    """Fail-closed runtime policy used by replay and repair validation."""

    allow_network: bool = False
    require_os_isolation: bool = True
    timeout_seconds: float = 60.0
    readable_roots: list[Path] = field(default_factory=list)
    writable_roots: list[Path] = field(default_factory=list)

    @property
    def backend(self) -> str:
        if sys.platform == "darwin" and shutil.which("sandbox-exec"):
            return "macos-seatbelt"
        return "language-guards"

    @property
    def has_os_isolation(self) -> bool:
        return self.backend == "macos-seatbelt"


def resolve_within(root: Path, target: str | Path) -> Path:
    root_resolved = root.resolve()
    candidate = Path(target)
    if candidate.is_absolute():
        candidate_resolved = candidate.resolve()
    else:
        candidate_resolved = (root_resolved / candidate).resolve()
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise SandboxPolicyError(f"Sandbox path escapes the execution root: {target!s}") from exc
    return candidate_resolved


def seatbelt_profile(policy: SandboxPolicy) -> str:
    if not policy.has_os_isolation:
        if policy.require_os_isolation:
            raise SandboxPolicyError("No supported OS sandbox backend is available.")
        return ""

    readable = _deduplicate_roots(policy.readable_roots)
    writable = _deduplicate_roots(policy.writable_roots)
    clauses = ["(version 1)", "(allow default)"]
    if not policy.allow_network:
        clauses.append("(deny network*)")

    user_home = Path.home().resolve()
    if not any(_is_within(user_home, root) for root in readable):
        clauses.append(f'(deny file-read* (subpath "{_scheme_string(user_home)}"))')

    if writable:
        root_filters = [
            f'(require-not (subpath "{_scheme_string(root)}"))'
            for root in writable
        ]
        device_filters = [
            '(require-not (literal "/dev/null"))',
            '(require-not (literal "/dev/tty"))',
        ]
        filters = " ".join(root_filters + device_filters)
        clauses.append(f"(deny file-write* (require-all {filters}))")
    else:
        clauses.append("(deny file-write*)")
    return "".join(clauses)


def seatbelt_command(command: list[str], policy: SandboxPolicy) -> tuple[list[str], str]:
    profile = seatbelt_profile(policy)
    if not profile:
        return command, policy.backend
    sandbox_exec = shutil.which("sandbox-exec")
    if sandbox_exec is None:  # pragma: no cover - guarded by seatbelt_profile
        raise SandboxPolicyError("sandbox-exec disappeared while preparing the command.")
    return [sandbox_exec, "-p", profile, *command], policy.backend


def _deduplicate_roots(roots: list[Path]) -> list[Path]:
    output: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = root.resolve()
        value = str(resolved)
        if value in seen:
            continue
        seen.add(value)
        output.append(resolved)
    return output


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _scheme_string(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')
