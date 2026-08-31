from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .policy import SandboxPolicy, SandboxPolicyError, resolve_within, seatbelt_command


@dataclass(slots=True)
class ScriptExecutionRequest:
    sandbox_root: str
    script_relative_path: str
    prompt: str
    ablation: dict[str, Any] | None = None


@dataclass(slots=True)
class ScriptExecutionResult:
    status: str
    final_output: str = ""
    stdout: str = ""
    stderr: str = ""
    trace_events: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SandboxedPythonRunner:
    def __init__(self, project_root: Path, policy: SandboxPolicy | None = None) -> None:
        self.project_root = project_root
        self.policy = policy or SandboxPolicy()

    def run(self, request: ScriptExecutionRequest) -> ScriptExecutionResult:
        request_dir = Path(tempfile.mkdtemp(prefix="skillscope-python-run-"))
        request_path = request_dir / "request.json"
        response_path = request_dir / "response.json"
        try:
            sandbox_root = Path(request.sandbox_root).resolve()
            script_path = resolve_within(sandbox_root, request.script_relative_path)
            if not script_path.is_file():
                raise SandboxPolicyError(f"Python target does not exist inside the sandbox: {request.script_relative_path}")

            worker_copy = request_dir / "python_worker.py"
            shutil.copy2(Path(__file__).with_name("python_worker.py"), worker_copy)
            request_path.write_text(
                json.dumps(self._request_to_dict(request), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            runtime_policy = SandboxPolicy(
                allow_network=self.policy.allow_network,
                require_os_isolation=self.policy.require_os_isolation,
                timeout_seconds=self.policy.timeout_seconds,
                readable_roots=[sandbox_root, request_dir],
                writable_roots=[sandbox_root, request_dir],
            )
            # ``sys.executable`` points at the virtual-environment shim when
            # SkillScope itself runs from a venv.  The macOS Seatbelt profile
            # deliberately denies reads from the user's home directory, so
            # asking ``sandbox-exec`` to exec that shim is rejected before the
            # worker starts.  The worker is self-contained and only needs the
            # standard library, therefore execute the shim's canonical Python
            # binary instead.  This keeps the host project (including the
            # venv) outside the sandbox's readable roots without weakening the
            # file or network policy.
            python_binary = str(Path(sys.executable).resolve(strict=True))
            base_command = [
                python_binary,
                "-I",
                str(worker_copy),
                str(request_path),
                str(response_path),
            ]
            command, backend = seatbelt_command(base_command, runtime_policy)
            env = {
                "HOME": str(sandbox_root),
                "PATH": "/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUTF8": "1",
                "TMPDIR": str(sandbox_root),
            }
            process = subprocess.run(
                command,
                cwd=sandbox_root,
                capture_output=True,
                text=True,
                env=env,
                timeout=self.policy.timeout_seconds,
            )

            if response_path.exists():
                response = json.loads(response_path.read_text(encoding="utf-8"))
                result = ScriptExecutionResult(
                    status=str(response.get("status") or "failed"),
                    final_output=str(response.get("final_output") or ""),
                    stdout=str(response.get("stdout") or ""),
                    stderr=str(response.get("stderr") or ""),
                    trace_events=response.get("trace_events") or [],
                    notes=[str(note) for note in response.get("notes") or []],
                    error=response.get("error"),
                    metadata={
                        "sandbox_backend": backend,
                        "network_allowed": runtime_policy.allow_network,
                        "filesystem_root": str(sandbox_root),
                    },
                )
            else:
                result = ScriptExecutionResult(
                    status="failed",
                    stdout=process.stdout,
                    stderr=process.stderr,
                    notes=["The sandbox worker did not produce a response file."],
                    error=f"process_exit={process.returncode}",
                    metadata={"sandbox_backend": backend},
                )

            if process.stdout and process.stdout not in result.stdout:
                result.stdout = "\n".join(part for part in (result.stdout, process.stdout) if part)
            if process.stderr and process.stderr not in result.stderr:
                result.stderr = "\n".join(part for part in (result.stderr, process.stderr) if part)
            return result
        except subprocess.TimeoutExpired as exc:
            return ScriptExecutionResult(
                status="failed",
                notes=["Sandbox execution exceeded its configured time limit."],
                error=f"timeout_after={exc.timeout}",
                metadata={"sandbox_backend": self.policy.backend},
            )
        except SandboxPolicyError as exc:
            return ScriptExecutionResult(
                status="blocked",
                notes=["Execution was rejected by the fail-closed sandbox policy."],
                error=str(exc),
                metadata={"sandbox_backend": self.policy.backend},
            )
        finally:
            shutil.rmtree(request_dir, ignore_errors=True)

    def _request_to_dict(self, request: ScriptExecutionRequest) -> dict[str, Any]:
        return {
            "sandbox_root": request.sandbox_root,
            "script_relative_path": request.script_relative_path,
            "prompt": request.prompt,
            "ablation": request.ablation,
        }


def create_isolated_skill_copy(bundle_root: Path) -> Path:
    sandbox_root = Path(tempfile.mkdtemp(prefix="skillscope-sandbox-"))
    skill_root = sandbox_root / "skill"
    try:
        shutil.copytree(bundle_root, skill_root, dirs_exist_ok=True, symlinks=True)
        for path in skill_root.rglob("*"):
            if not path.is_symlink():
                continue
            try:
                path.resolve().relative_to(skill_root.resolve())
            except ValueError as exc:
                raise SandboxPolicyError(
                    f"Skill bundle contains a symlink that escapes the sandbox: {path.relative_to(skill_root)}"
                ) from exc
        return skill_root
    except Exception:
        shutil.rmtree(sandbox_root, ignore_errors=True)
        raise
