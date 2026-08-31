from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import tempfile
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .policy import SandboxPolicy, SandboxPolicyError, resolve_within, seatbelt_command
from .runtime import SandboxedPythonRunner, ScriptExecutionRequest


_SHELL_XTRACE_MARKER_RE = re.compile(
    r"\x1e(?P<line>[0-9]+)\x1f(?P<source>[^\x1d\r\n]*)\x1d"
    r"(?P<command>[^\r\n]*)"
)


@dataclass(slots=True)
class ToolInvocationRequest:
    tool_call_id: str
    tool_name: str
    target: str
    sandbox_root: str
    prompt: str
    instruction_node_id: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolInvocationResult:
    status: str
    final_output: str = ""
    stdout: str = ""
    stderr: str = ""
    trace_events: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentTool(Protocol):
    name: str
    supported_suffixes: tuple[str, ...]

    def invoke(self, request: ToolInvocationRequest) -> ToolInvocationResult:
        ...


class PythonScriptTool:
    name = "python_script"
    supported_suffixes = (".py",)

    def __init__(self, runner: SandboxedPythonRunner) -> None:
        self.runner = runner

    def invoke(self, request: ToolInvocationRequest) -> ToolInvocationResult:
        result = self.runner.run(
            ScriptExecutionRequest(
                sandbox_root=request.sandbox_root,
                script_relative_path=request.target,
                prompt=request.prompt,
                ablation=(
                    request.metadata.get("ablation")
                    if isinstance(request.metadata.get("ablation"), dict)
                    else None
                ),
            )
        )
        return ToolInvocationResult(
            status=result.status,
            final_output=result.final_output,
            stdout=result.stdout,
            stderr=result.stderr,
            trace_events=result.trace_events,
            notes=result.notes,
            error=result.error,
            metadata=result.metadata | {
                "tool_name": self.name,
                "target": request.target,
                "trace_granularity": "instrumented_python",
            },
        )


class ShellScriptTool:
    name = "shell_script"
    supported_suffixes = (".sh",)

    def __init__(self, policy: SandboxPolicy | None = None) -> None:
        self.policy = policy or SandboxPolicy()

    def invoke(self, request: ToolInvocationRequest) -> ToolInvocationResult:
        sandbox_root = Path(request.sandbox_root).resolve()
        bash_binary = shutil.which("bash")
        if bash_binary is None:
            return ToolInvocationResult(
                status="blocked",
                notes=["Bash is required to produce trustworthy source-line shell traces."],
                error="bash_runtime_unavailable",
                metadata={"tool_name": self.name, "target": request.target},
            )
        shell_trace = ""
        try:
            script_path = resolve_within(sandbox_root, request.target)
            if not script_path.is_file():
                raise SandboxPolicyError(f"Shell target does not exist inside the sandbox: {request.target}")
            runtime_policy = SandboxPolicy(
                allow_network=self.policy.allow_network,
                require_os_isolation=self.policy.require_os_isolation,
                timeout_seconds=self.policy.timeout_seconds,
                readable_roots=[sandbox_root],
                writable_roots=[sandbox_root],
            )
            _require_os_process_isolation(runtime_policy, "Shell")
            env = self._minimal_environment(sandbox_root, request.prompt)
            with tempfile.TemporaryFile(
                mode="w+b",
                prefix=".skillscope-shell-trace-",
                dir=sandbox_root,
            ) as trace_handle:
                trace_fd = trace_handle.fileno()
                env["BASH_XTRACEFD"] = str(trace_fd)
                # Bash 3.2 truncates long PS4 expansions, so emit the line
                # number before a basename-sized source identifier.
                env["PS4"] = "\x1e${LINENO}\x1f${BASH_SOURCE[0]##*/}\x1d"
                command, backend = seatbelt_command(
                    [bash_binary, "--noprofile", "--norc", "-x", str(script_path)],
                    runtime_policy,
                )
                process = subprocess.run(
                    command,
                    cwd=sandbox_root,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=self.policy.timeout_seconds,
                    pass_fds=(trace_fd,),
                )
                trace_handle.seek(0)
                shell_trace = trace_handle.read().decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired as exc:
            return ToolInvocationResult(
                status="failed",
                notes=["Shell execution exceeded its configured time limit."],
                error=f"timeout_after={exc.timeout}",
                metadata={"tool_name": self.name, "target": request.target, "sandbox_backend": self.policy.backend},
            )
        except SandboxPolicyError as exc:
            return ToolInvocationResult(
                status="blocked",
                notes=["Shell execution was rejected by the fail-closed sandbox policy."],
                error=str(exc),
                metadata={"tool_name": self.name, "target": request.target, "sandbox_backend": self.policy.backend},
            )

        status = "completed" if process.returncode == 0 else "failed"
        stdout = process.stdout or ""
        raw_stderr = process.stderr or ""
        trace_output = shell_trace or raw_stderr
        stderr = (
            raw_stderr
            if shell_trace
            else _without_shell_xtrace_lines(raw_stderr)
        )
        final_output = stdout.strip() or f"Executed {request.target}"
        trace_events = [
            _script_boundary_event(
                event_type="script_start",
                summary=f"Start shell script {request.target}",
                source_file=request.target,
                arguments_summary=request.prompt[:200] or None,
                attributes={"tool_name": self.name},
            ),
            *_shell_command_events(
                trace_output=trace_output,
                script_path=script_path,
                source_file=request.target,
            ),
            _script_boundary_event(
                event_type="script_end",
                summary=f"End shell script {request.target}",
                source_file=request.target,
                attributes={"status": status, "tool_name": self.name},
            ),
        ]
        notes = [
            "Shell execution was confined by the OS sandbox with network disabled and writes restricted to the copied skill root.",
            "Shell action traces retain the Bash xtrace command text and source line. Static actions are linked only when that command identifies one source segment unambiguously.",
        ]
        return ToolInvocationResult(
            status=status,
            final_output=final_output,
            stdout=stdout,
            stderr=stderr,
            trace_events=trace_events,
            notes=notes,
            error=None if status == "completed" else f"process_exit={process.returncode}",
            metadata={
                "tool_name": self.name,
                "target": request.target,
                "trace_granularity": "bash_xtrace_command",
                "executed_command_event_count": len(trace_events) - 2,
                "sandbox_backend": backend,
                "network_allowed": runtime_policy.allow_network,
                "filesystem_root": str(sandbox_root),
            },
        )

    def _minimal_environment(self, sandbox_root: Path, prompt: str) -> dict[str, str]:
        environment = {
            "HOME": str(sandbox_root),
            "LANG": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "SKILLSCOPE_USER_PROMPT": prompt,
            "TMPDIR": str(sandbox_root),
        }
        manifest_path = sandbox_root / ".skillscope" / "fixtures.json"
        if not manifest_path.exists():
            return environment
        try:
            fixtures = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return environment
        if not isinstance(fixtures, list):
            return environment
        for fixture in fixtures:
            if not isinstance(fixture, dict) or fixture.get("fixture_type") != "env":
                continue
            key = str(fixture.get("target") or "")
            value = fixture.get("content")
            if key and isinstance(value, str):
                environment[key] = value
        return environment


class InlineCommandTool:
    """Execute an explicit instruction command once inside the OS sandbox.

    This tool is intentionally selected by name rather than by a filename
    suffix.  The command originates from a concrete ``command_invocations``
    entry on an instruction node, and Bash xtrace supplies the material
    evidence for the command(s) that were actually attempted.
    """

    name = "inline_command"
    supported_suffixes: tuple[str, ...] = ()

    def __init__(self, policy: SandboxPolicy | None = None) -> None:
        self.policy = policy or SandboxPolicy()

    def invoke(self, request: ToolInvocationRequest) -> ToolInvocationResult:
        sandbox_root = Path(request.sandbox_root).resolve()
        command_text = request.arguments.get("command")
        if not isinstance(command_text, str):
            return self._blocked(request, "inline_command_missing")
        command_text = command_text.strip()
        if not command_text or len(command_text.encode("utf-8")) > 16 * 1024:
            return self._blocked(request, "inline_command_invalid_length")
        if "\x00" in command_text or "\r" in command_text or "\n" in command_text:
            return self._blocked(request, "inline_command_contains_control_character")

        bash_binary = shutil.which("bash")
        if bash_binary is None:
            return self._blocked(request, "bash_runtime_unavailable")

        shell_trace = ""
        try:
            runtime_policy = SandboxPolicy(
                allow_network=self.policy.allow_network,
                require_os_isolation=self.policy.require_os_isolation,
                timeout_seconds=self.policy.timeout_seconds,
                readable_roots=[sandbox_root],
                writable_roots=[sandbox_root],
            )
            _require_os_process_isolation(runtime_policy, "Inline command")
            environment = ShellScriptTool(self.policy)._minimal_environment(
                sandbox_root,
                request.prompt,
            )
            with tempfile.TemporaryFile(
                mode="w+b",
                prefix=".skillscope-inline-trace-",
                dir=sandbox_root,
            ) as trace_handle:
                trace_fd = trace_handle.fileno()
                environment["BASH_XTRACEFD"] = str(trace_fd)
                environment["PS4"] = "\x1e${LINENO}\x1f<inline>\x1d"
                command, backend = seatbelt_command(
                    [
                        bash_binary,
                        "--noprofile",
                        "--norc",
                        "-x",
                        "-c",
                        command_text,
                    ],
                    runtime_policy,
                )
                process = subprocess.run(
                    command,
                    cwd=sandbox_root,
                    capture_output=True,
                    text=True,
                    env=environment,
                    timeout=self.policy.timeout_seconds,
                    pass_fds=(trace_fd,),
                )
                trace_handle.seek(0)
                shell_trace = trace_handle.read().decode(
                    "utf-8",
                    errors="replace",
                )
        except subprocess.TimeoutExpired as exc:
            return ToolInvocationResult(
                status="failed",
                notes=["Inline command execution exceeded its configured time limit."],
                error=f"timeout_after={exc.timeout}",
                metadata={
                    "tool_name": self.name,
                    "target": request.target,
                    "sandbox_backend": self.policy.backend,
                },
            )
        except SandboxPolicyError as exc:
            return ToolInvocationResult(
                status="blocked",
                notes=["Inline command execution was rejected by the fail-closed sandbox policy."],
                error=str(exc),
                metadata={
                    "tool_name": self.name,
                    "target": request.target,
                    "sandbox_backend": self.policy.backend,
                },
            )

        status = "completed" if process.returncode == 0 else "failed"
        stdout = process.stdout or ""
        raw_stderr = process.stderr or ""
        trace_output = shell_trace or raw_stderr
        events = _inline_command_events(trace_output)
        for event in events:
            attributes = event.get("attributes")
            if isinstance(attributes, dict):
                attributes["instruction_command"] = command_text[:2000]
        return ToolInvocationResult(
            status=status,
            final_output=stdout.strip() or f"Executed {request.target}",
            stdout=stdout,
            stderr=(
                raw_stderr
                if shell_trace
                else _without_shell_xtrace_lines(raw_stderr)
            ),
            trace_events=events,
            notes=[
                "Executed the explicit instruction command once under the OS sandbox with network disabled and writes confined to the copied skill root.",
            ],
            error=None if status == "completed" else f"process_exit={process.returncode}",
            metadata={
                "tool_name": self.name,
                "target": request.target,
                "trace_granularity": "bash_xtrace_command",
                "executed_command_event_count": len(events),
                "sandbox_backend": backend,
                "network_allowed": runtime_policy.allow_network,
                "filesystem_root": str(sandbox_root),
            },
        )

    def _blocked(
        self,
        request: ToolInvocationRequest,
        error: str,
    ) -> ToolInvocationResult:
        return ToolInvocationResult(
            status="blocked",
            notes=["The explicit instruction command was rejected before execution."],
            error=error,
            metadata={"tool_name": self.name, "target": request.target},
        )


class NodeScriptTool:
    name = "node_script"
    supported_suffixes = (".js", ".mjs", ".cjs", ".ts", ".mts", ".cts")

    def __init__(self, policy: SandboxPolicy | None = None) -> None:
        self.policy = policy or SandboxPolicy()

    def invoke(self, request: ToolInvocationRequest) -> ToolInvocationResult:
        sandbox_root = Path(request.sandbox_root).resolve()
        node_binary = shutil.which("node")
        if node_binary is None:
            return ToolInvocationResult(
                status="blocked",
                notes=["Node.js is not installed, so the JavaScript/TypeScript fixture cannot run."],
                error="node_runtime_unavailable",
                metadata={"tool_name": self.name, "target": request.target},
            )
        coverage_dir: Path | None = None
        coverage_events: list[dict[str, Any]] = []
        try:
            script_path = resolve_within(sandbox_root, request.target)
            if not script_path.is_file():
                raise SandboxPolicyError(f"Node target does not exist inside the sandbox: {request.target}")
            source_text = script_path.read_text(encoding="utf-8")
            runtime_policy = SandboxPolicy(
                allow_network=self.policy.allow_network,
                require_os_isolation=self.policy.require_os_isolation,
                timeout_seconds=self.policy.timeout_seconds,
                readable_roots=[sandbox_root],
                writable_roots=[sandbox_root],
            )
            _require_os_process_isolation(runtime_policy, "Node")
            coverage_dir = Path(
                tempfile.mkdtemp(
                    prefix=".skillscope-node-coverage-",
                    dir=sandbox_root,
                )
            )
            command, backend = seatbelt_command([node_binary, str(script_path)], runtime_policy)
            environment = {
                "HOME": str(sandbox_root),
                "LANG": "C.UTF-8",
                "NODE_V8_COVERAGE": str(coverage_dir),
                "PATH": "/usr/bin:/bin",
                "SKILLSCOPE_USER_PROMPT": request.prompt,
                "TMPDIR": str(sandbox_root),
            }
            process = subprocess.run(
                command,
                cwd=sandbox_root,
                capture_output=True,
                text=True,
                env=environment,
                timeout=self.policy.timeout_seconds,
            )
            coverage_events = _node_coverage_events(
                coverage_dir=coverage_dir,
                script_path=script_path,
                source_file=request.target,
                source_text=source_text,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolInvocationResult(
                status="failed",
                notes=["Node execution exceeded its configured time limit."],
                error=f"timeout_after={exc.timeout}",
                metadata={"tool_name": self.name, "target": request.target, "sandbox_backend": self.policy.backend},
            )
        except SandboxPolicyError as exc:
            return ToolInvocationResult(
                status="blocked",
                notes=["Node execution was rejected by the fail-closed sandbox policy."],
                error=str(exc),
                metadata={"tool_name": self.name, "target": request.target, "sandbox_backend": self.policy.backend},
            )
        except (OSError, UnicodeDecodeError) as exc:
            return ToolInvocationResult(
                status="failed",
                notes=["Node execution coverage could not be prepared or decoded safely."],
                error=f"{type(exc).__name__}: {exc}",
                metadata={"tool_name": self.name, "target": request.target, "sandbox_backend": self.policy.backend},
            )
        finally:
            if coverage_dir is not None:
                _remove_runtime_directory(coverage_dir)

        status = "completed" if process.returncode == 0 else "failed"
        stdout = process.stdout or ""
        stderr = process.stderr or ""
        events = [
            _script_boundary_event(
                event_type="script_start",
                summary=f"Start node script {request.target}",
                source_file=request.target,
                arguments_summary=request.prompt[:200] or None,
                attributes={"tool_name": self.name},
            ),
            *coverage_events,
            _script_boundary_event(
                event_type="script_end",
                summary=f"End node script {request.target}",
                source_file=request.target,
                attributes={"status": status, "tool_name": self.name},
            ),
        ]
        return ToolInvocationResult(
            status=status,
            final_output=stdout.strip() or f"Executed {request.target}",
            stdout=stdout,
            stderr=stderr,
            trace_events=events,
            notes=[
                "Node execution was confined by the OS sandbox; network is disabled and writes are restricted to the copied skill root.",
                "Node action traces are derived from V8 block coverage for the executed entry script.",
            ],
            error=None if status == "completed" else f"process_exit={process.returncode}",
            metadata={
                "tool_name": self.name,
                "target": request.target,
                "trace_granularity": "v8_precise_block_coverage",
                "coverage_snapshot_count": len(coverage_events),
                "sandbox_backend": backend,
                "network_allowed": runtime_policy.allow_network,
                "filesystem_root": str(sandbox_root),
            },
        )


def _script_boundary_event(
    *,
    event_type: str,
    summary: str,
    source_file: str,
    arguments_summary: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "summary": summary,
        "object_ref": source_file,
        "arguments_summary": arguments_summary,
        "attributes": {
            "script_relative_path": source_file,
            "source_file": source_file,
            **(attributes or {}),
        },
    }


def _require_os_process_isolation(
    policy: SandboxPolicy,
    runtime_name: str,
) -> None:
    if not policy.has_os_isolation:
        raise SandboxPolicyError(
            f"{runtime_name} execution requires an OS sandbox that can deny "
            "network access and confine writes to the copied skill root."
        )


def _shell_command_events(
    *,
    trace_output: str,
    script_path: Path,
    source_file: str,
) -> list[dict[str, Any]]:
    expected_path = script_path.resolve()
    events: list[dict[str, Any]] = []
    for match in _SHELL_XTRACE_MARKER_RE.finditer(trace_output):
        traced_source = match.group("source")
        if traced_source != expected_path.name:
            continue
        line_number = int(match.group("line"))
        command_text = match.group("command").strip()
        command_tokens = _shell_command_tokens(command_text)
        if line_number <= 0 or not command_text or not command_tokens:
            continue
        events.append(
            {
                "event_type": "source_command_execution",
                "summary": f"Execute shell command at {source_file}:{line_number}",
                "object_ref": command_tokens[0],
                "arguments_summary": command_text[:500],
                "attributes": {
                    "source_file": source_file,
                    "line_number": line_number,
                    "runtime_command": command_text[:2000],
                    "runtime_command_tokens": command_tokens,
                    "runtime_evidence": "bash_xtrace",
                },
            }
        )
    return events


def _inline_command_events(trace_output: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for match in _SHELL_XTRACE_MARKER_RE.finditer(trace_output):
        if match.group("source") != "<inline>":
            continue
        command_text = match.group("command").strip()
        command_tokens = _shell_command_tokens(command_text)
        if not command_text or not command_tokens:
            continue
        events.append(
            {
                "event_type": "exec_command",
                "summary": f"Execute explicit instruction command {command_tokens[0]}",
                "object_ref": command_tokens[0],
                "arguments_summary": command_text[:500],
                "attributes": {
                    "material_operation": "exec_command",
                    "runtime_command": command_text[:2000],
                    "runtime_command_tokens": command_tokens,
                    "runtime_evidence": "bash_xtrace",
                    "inline_instruction_command": True,
                },
            }
        )
    return events


def _shell_command_tokens(command_text: str) -> list[str]:
    """Return a conservative command signature with redirections removed."""

    try:
        raw_tokens = shlex.split(command_text, posix=True)
    except ValueError:
        return []
    if not raw_tokens:
        return []

    tokens: list[str] = []
    skip_redirection_target = False
    for token in raw_tokens:
        if skip_redirection_target:
            skip_redirection_target = False
            continue
        if re.fullmatch(r"[0-9]*(?:>>?|<<?|<>|>&|<&)", token):
            skip_redirection_target = True
            continue
        if re.match(r"^[0-9]*(?:>>?|<<?|<>|>&|<&).+", token):
            continue
        tokens.append(token)
    return tokens


def _without_shell_xtrace_lines(stderr: str) -> str:
    return "".join(
        line
        for line in stderr.splitlines(keepends=True)
        if _SHELL_XTRACE_MARKER_RE.search(line) is None
    )


def _node_coverage_events(
    *,
    coverage_dir: Path,
    script_path: Path,
    source_file: str,
    source_text: str,
) -> list[dict[str, Any]]:
    coverage_ranges: list[tuple[int, int, int]] = []
    has_block_coverage = False
    for coverage_path in sorted(coverage_dir.glob("coverage-*.json")):
        try:
            if coverage_path.is_symlink():
                continue
            coverage_path.resolve().relative_to(coverage_dir.resolve())
            if coverage_path.stat().st_size > 32 * 1024 * 1024:
                continue
            payload = json.loads(coverage_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        results = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            continue
        for script_result in results:
            if not isinstance(script_result, dict) or not _coverage_url_matches_script(
                script_result.get("url"),
                script_path,
            ):
                continue
            functions = script_result.get("functions")
            if not isinstance(functions, list):
                continue
            for function in functions:
                if not isinstance(function, dict):
                    continue
                has_block_coverage = (
                    has_block_coverage or function.get("isBlockCoverage") is True
                )
                ranges = function.get("ranges")
                if not isinstance(ranges, list):
                    continue
                for range_payload in ranges:
                    if not isinstance(range_payload, dict):
                        continue
                    start_offset = range_payload.get("startOffset")
                    end_offset = range_payload.get("endOffset")
                    count = range_payload.get("count")
                    if (
                        isinstance(start_offset, int)
                        and isinstance(end_offset, int)
                        and isinstance(count, int)
                        and 0 <= start_offset < end_offset
                        and count >= 0
                    ):
                        coverage_ranges.append((start_offset, end_offset, count))

    # Function-only coverage cannot distinguish an untaken branch in a function
    # that otherwise ran, so it is not accepted as line execution evidence.
    if not has_block_coverage or not coverage_ranges:
        return []

    executed_spans = _flatten_executed_coverage_spans(
        source_text,
        coverage_ranges,
    )
    if not executed_spans:
        return []
    return [
        {
            "event_type": "source_coverage_execution",
            "summary": f"Observe precise V8 coverage for {source_file}",
            "object_ref": None,
            "arguments_summary": None,
            "attributes": {
                "source_file": source_file,
                "runtime_evidence": "v8_block_coverage",
                "temporal_order_observed": False,
                "execution_count_observed": False,
                "ordering_evidence": "source_provenance_only",
                "executed_column_spans": executed_spans,
                "source_line_indents": {
                    str(line_number): len(line) - len(line.lstrip())
                    for line_number, line in enumerate(
                        source_text.splitlines(),
                        start=1,
                    )
                },
            },
        }
    ]


def _flatten_executed_coverage_spans(
    source_text: str,
    coverage_ranges: list[tuple[int, int, int]],
) -> dict[str, list[list[int]]]:
    """Flatten nested V8 ranges into executed code-point columns per line."""

    source_length = _utf16_length(source_text)
    boundaries = {0, source_length}
    for range_start, range_end, _count in coverage_ranges:
        boundaries.add(max(0, min(source_length, range_start)))
        boundaries.add(max(0, min(source_length, range_end)))
    ordered = sorted(boundaries)

    spans_by_line: dict[str, list[list[int]]] = {}
    for start_offset, end_offset in zip(ordered, ordered[1:]):
        if start_offset >= end_offset:
            continue
        count = _innermost_coverage_count(start_offset, coverage_ranges)
        if count is None or count <= 0:
            continue
        for line_number, start_column, end_column in _utf16_span_to_line_columns(
            source_text,
            start_offset,
            end_offset,
        ):
            if start_column >= end_column:
                continue
            line_key = str(line_number)
            line_spans = spans_by_line.setdefault(line_key, [])
            if line_spans and line_spans[-1][1] == start_column:
                line_spans[-1][1] = end_column
            else:
                line_spans.append([start_column, end_column])
    return spans_by_line


def _utf16_span_to_line_columns(
    source_text: str,
    start_offset: int,
    end_offset: int,
) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    line_offset = 0
    for line_number, raw_line in enumerate(
        source_text.splitlines(keepends=True),
        start=1,
    ):
        line_text = raw_line.rstrip("\r\n")
        line_content_end = line_offset + _utf16_length(line_text)
        overlap_start = max(start_offset, line_offset)
        overlap_end = min(end_offset, line_content_end)
        if overlap_start < overlap_end:
            spans.append(
                (
                    line_number,
                    _codepoint_column_at_utf16_offset(
                        line_text,
                        overlap_start - line_offset,
                    ),
                    _codepoint_column_at_utf16_offset(
                        line_text,
                        overlap_end - line_offset,
                    ),
                )
            )
        line_offset += _utf16_length(raw_line)
        if line_offset >= end_offset:
            break
    return spans


def _codepoint_column_at_utf16_offset(text: str, offset: int) -> int:
    observed = 0
    for index, character in enumerate(text):
        if observed >= offset:
            return index
        observed += _utf16_length(character)
        if observed > offset:
            # A V8 range should not split a surrogate pair.  Returning the
            # preceding boundary is conservative if malformed data does.
            return index
    return len(text)


def _coverage_url_matches_script(value: object, script_path: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = urllib.parse.urlparse(value)
        if parsed.scheme == "file":
            candidate = Path(urllib.parse.unquote(parsed.path))
        elif not parsed.scheme:
            candidate = Path(value)
        else:
            return False
        return candidate.resolve() == script_path.resolve()
    except (OSError, RuntimeError, ValueError):
        return False


def _fully_covered_source_lines(
    source_text: str,
    coverage_ranges: list[tuple[int, int, int]],
) -> list[int]:
    executed_lines: list[int] = []
    line_offset = 0
    for line_number, raw_line in enumerate(source_text.splitlines(keepends=True), start=1):
        line_content = raw_line.rstrip("\r\n")
        leading_length = len(line_content) - len(line_content.lstrip())
        meaningful_content = line_content[leading_length:]
        if meaningful_content and not meaningful_content.startswith("//"):
            start_offset = line_offset + _utf16_length(line_content[:leading_length])
            end_offset = line_offset + _utf16_length(line_content)
            if _coverage_span_is_fully_executed(
                start_offset,
                end_offset,
                coverage_ranges,
            ):
                executed_lines.append(line_number)
        line_offset += _utf16_length(raw_line)
    return executed_lines


def _coverage_span_is_fully_executed(
    start_offset: int,
    end_offset: int,
    coverage_ranges: list[tuple[int, int, int]],
) -> bool:
    if start_offset >= end_offset:
        return False
    boundaries = {start_offset, end_offset}
    for range_start, range_end, _count in coverage_ranges:
        if range_end <= start_offset or range_start >= end_offset:
            continue
        boundaries.add(max(start_offset, range_start))
        boundaries.add(min(end_offset, range_end))
    ordered = sorted(boundaries)
    observed_segment = False
    for segment_start, segment_end in zip(ordered, ordered[1:]):
        if segment_start >= segment_end:
            continue
        count = _innermost_coverage_count(segment_start, coverage_ranges)
        if count is None or count <= 0:
            return False
        observed_segment = True
    return observed_segment


def _innermost_coverage_count(
    offset: int,
    coverage_ranges: list[tuple[int, int, int]],
) -> int | None:
    containing = [
        (range_end - range_start, count)
        for range_start, range_end, count in coverage_ranges
        if range_start <= offset < range_end
    ]
    if not containing:
        return None
    narrowest = min(length for length, _count in containing)
    return min(count for length, count in containing if length == narrowest)


def _utf16_length(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def _remove_runtime_directory(path: Path) -> None:
    try:
        if path.is_symlink() or not path.is_dir():
            path.unlink(missing_ok=True)
        else:
            shutil.rmtree(path)
    except OSError:
        pass


class ToolRegistry:
    def __init__(self, tools: list[AgentTool] | None = None) -> None:
        self._tools_by_name: dict[str, AgentTool] = {}
        self._tool_name_by_suffix: dict[str, str] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: AgentTool) -> None:
        self._tools_by_name[tool.name] = tool
        for suffix in tool.supported_suffixes:
            self._tool_name_by_suffix[suffix] = tool.name

    def get(self, tool_name: str) -> AgentTool:
        try:
            return self._tools_by_name[tool_name]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(f"No tool is registered under the name {tool_name!r}.") from exc

    def resolve_tool_name(self, target: str) -> str:
        suffix = Path(target).suffix.lower()
        try:
            return self._tool_name_by_suffix[suffix]
        except KeyError as exc:
            raise RuntimeError(f"No registered tool can execute target {target!r}.") from exc

    def available_tool_names(self) -> list[str]:
        return sorted(self._tools_by_name)


class ToolDispatcher:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def resolve_tool_name(self, target: str) -> str:
        return self.registry.resolve_tool_name(target)

    def invoke(self, request: ToolInvocationRequest) -> ToolInvocationResult:
        tool = self.registry.get(request.tool_name)
        return tool.invoke(request)

    def available_tool_names(self) -> list[str]:
        return self.registry.available_tool_names()
