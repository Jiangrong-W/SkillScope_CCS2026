from __future__ import annotations

import builtins
import contextlib
import io
import json
import os
import platform
import pathlib
import runpy
import socket
import subprocess
import sys
import traceback
import types
import urllib.request
import uuid
from pathlib import Path
from typing import Any


class _SandboxState:
    def __init__(
        self,
        sandbox_root: Path,
        script_path: Path,
        ablation: dict[str, Any] | None,
        fixtures: list[dict[str, Any]],
    ) -> None:
        self.sandbox_root = sandbox_root
        self.script_path = script_path
        self.ablation = ablation or {}
        self.trace_events: list[dict[str, Any]] = []
        self.stdout_capture = io.StringIO()
        self.stderr_capture = io.StringIO()
        self.notes: list[str] = []
        self.last_string_return: str = ""
        self.fixtures = fixtures

    def record(self, *, event_type: str, summary: str, object_ref: str | None = None, arguments_summary: str | None = None, attributes: dict[str, Any] | None = None) -> None:
        self.trace_events.append(
            {
                "event_type": event_type,
                "summary": summary,
                "object_ref": object_ref,
                "arguments_summary": arguments_summary,
                "attributes": attributes or {},
            }
        )

    def match_ablation(self, *, source_file: str | None, line_number: int | None, operation_type: str | None) -> bool:
        if not self.ablation:
            return False
        if self.ablation.get("layer") != "code":
            return False
        if source_file and self.ablation.get("source_file") and source_file != self.ablation.get("source_file"):
            return False
        start_line = self.ablation.get("source_start_line")
        end_line = self.ablation.get("source_end_line")
        if line_number is not None and start_line is not None and end_line is not None:
            if not (int(start_line) <= int(line_number) <= int(end_line)):
                return False
        if operation_type and self.ablation.get("operation_type") and operation_type != self.ablation.get("operation_type"):
            return False
        return True

    def api_fixture(self, url: str) -> dict[str, Any] | None:
        for fixture in self.fixtures:
            if fixture.get("fixture_type") == "api" and str(fixture.get("target") or "") == url:
                return fixture
        return None


def _relative_to_sandbox(path: str, sandbox_root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(sandbox_root.resolve()))
    except Exception:
        return path


def _call_site(state: _SandboxState) -> tuple[str | None, int | None]:
    frame = sys._getframe(2)
    sandbox_root = state.sandbox_root.resolve()
    while frame is not None:
        filename = frame.f_code.co_filename
        if filename and not filename.startswith("<"):
            frame_path = Path(filename)
            try:
                relative = frame_path.resolve().relative_to(sandbox_root)
                return relative.as_posix(), frame.f_lineno
            except Exception:
                pass
        frame = frame.f_back
    return None, None


def _fake_response(status_code: int = 200, text: str = "") -> types.SimpleNamespace:
    def as_json() -> Any:
        try:
            return json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return {"ok": True, "text": text}

    return types.SimpleNamespace(
        status_code=status_code,
        text=text,
        content=text.encode("utf-8"),
        json=as_json,
        raise_for_status=lambda: None,
    )


def _install_requests_stub(state: _SandboxState) -> None:
    def request(method: str, url: str, **kwargs: Any) -> types.SimpleNamespace:
        source_file, line_number = _call_site(state)
        if state.match_ablation(source_file=source_file, line_number=line_number, operation_type="network_send"):
            state.record(
                event_type="network_send",
                summary=f"Ablated network request {method.upper()} {url}",
                object_ref=url,
                arguments_summary=str(kwargs),
                attributes={"ablated": True, "source_file": source_file, "line_number": line_number},
            )
            return _fake_response()
        fixture = state.api_fixture(url)
        if fixture is not None:
            status_code = int((fixture.get("metadata") or {}).get("status_code") or 200)
            content = str(fixture.get("content") or "")
            state.record(
                event_type="network_send",
                summary=f"Mocked network request {method.upper()} {url}",
                object_ref=url,
                arguments_summary=str(kwargs),
                attributes={
                    "mocked": True,
                    "fixture_id": fixture.get("fixture_id"),
                    "source_file": source_file,
                    "line_number": line_number,
                },
            )
            return _fake_response(status_code=status_code, text=content)
        state.record(
            event_type="network_send",
            summary=f"Blocked network request {method.upper()} {url}",
            object_ref=url,
            arguments_summary=str(kwargs),
            attributes={"blocked": True, "source_file": source_file, "line_number": line_number},
        )
        return _fake_response()

    requests_module = types.ModuleType("requests")
    requests_module.post = lambda url, **kwargs: request("post", url, **kwargs)
    requests_module.get = lambda url, **kwargs: request("get", url, **kwargs)
    requests_module.request = request

    class Session:
        def request(self, method: str, url: str, **kwargs: Any) -> types.SimpleNamespace:
            return request(method, url, **kwargs)

    requests_module.Session = Session
    sys.modules["requests"] = requests_module


def _confined_path(state: _SandboxState, file: Any) -> Path:
    raw_path = Path(os.fspath(file))
    candidate = raw_path if raw_path.is_absolute() else state.sandbox_root / raw_path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(state.sandbox_root.resolve())
    except ValueError as exc:
        raise PermissionError(f"Sandbox file access denied outside execution root: {file!s}") from exc
    return resolved


def _install_runtime_hooks(state: _SandboxState) -> dict[str, Any]:
    originals: dict[str, Any] = {
        "open": builtins.open,
        "io_open": io.open,
        "os_open": os.open,
        "os_system": os.system,
        "os_popen": os.popen,
        "subprocess_run": subprocess.run,
        "subprocess_popen": subprocess.Popen,
        "os_getenv": os.getenv,
        "uuid_getnode": uuid.getnode,
        "platform_node": platform.node,
        "path_open": pathlib.Path.open,
        "path_read_text": pathlib.Path.read_text,
        "path_write_text": pathlib.Path.write_text,
        "socket_socket": socket.socket,
        "socket_create_connection": socket.create_connection,
        "urlopen": urllib.request.urlopen,
    }

    def instrumented_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any):
        if isinstance(file, int):
            return originals["open"](file, mode, *args, **kwargs)
        path_str = str(file)
        source_file, line_number = _call_site(state)
        event_type = "file_write" if any(flag in mode for flag in ("w", "a", "+")) else "file_read"
        if state.match_ablation(source_file=source_file, line_number=line_number, operation_type="file_access"):
            state.record(
                event_type=event_type,
                summary=f"Ablated file access {path_str}",
                object_ref=path_str,
                arguments_summary=mode,
                attributes={"ablated": True, "source_file": source_file, "line_number": line_number},
            )
            if "r" in mode and "b" not in mode:
                return io.StringIO("")
            if "r" in mode and "b" in mode:
                return io.BytesIO(b"")
            return io.StringIO()
        try:
            confined = _confined_path(state, file)
        except PermissionError:
            state.record(
                event_type=event_type,
                summary=f"Blocked file access outside sandbox {path_str}",
                object_ref=path_str,
                arguments_summary=mode,
                attributes={"blocked": True, "source_file": source_file, "line_number": line_number},
            )
            raise
        state.record(
            event_type=event_type,
            summary=f"File access {path_str}",
            object_ref=_relative_to_sandbox(str(confined), state.sandbox_root),
            arguments_summary=mode,
            attributes={"source_file": source_file, "line_number": line_number},
        )
        return originals["open"](confined, mode, *args, **kwargs)

    def instrumented_os_open(file: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if dir_fd is not None:
            raise PermissionError("dir_fd based file access is disabled inside the SkillScope sandbox.")
        access_mode = "w" if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND) else "r"
        confined = _confined_path(state, file)
        source_file, line_number = _call_site(state)
        state.record(
            event_type="file_write" if access_mode == "w" else "file_read",
            summary=f"os.open {file!s}",
            object_ref=_relative_to_sandbox(str(confined), state.sandbox_root),
            arguments_summary=str(flags),
            attributes={"source_file": source_file, "line_number": line_number},
        )
        return originals["os_open"](confined, flags, mode)

    def instrumented_path_open(self: pathlib.Path, mode: str = "r", *args: Any, **kwargs: Any):
        return instrumented_open(str(self), mode, *args, **kwargs)

    def instrumented_read_text(self: pathlib.Path, *args: Any, **kwargs: Any) -> str:
        with instrumented_open(str(self), "r", encoding=kwargs.get("encoding") or "utf-8") as handle:
            return handle.read()

    def instrumented_write_text(self: pathlib.Path, data: str, *args: Any, **kwargs: Any) -> int:
        with instrumented_open(str(self), "w", encoding=kwargs.get("encoding") or "utf-8") as handle:
            written = handle.write(data)
        return int(written)

    def instrumented_system(command: str) -> int:
        source_file, line_number = _call_site(state)
        ablated = state.match_ablation(source_file=source_file, line_number=line_number, operation_type="exec_command")
        state.record(
            event_type="exec_command",
            summary=("Ablated command execution" if ablated else "Blocked command execution"),
            object_ref=command,
            arguments_summary=command,
            attributes={"ablated": ablated, "blocked": not ablated, "source_file": source_file, "line_number": line_number},
        )
        return 0

    def instrumented_popen(command: str, mode: str = "r", buffering: int = -1):
        instrumented_system(command)
        return io.StringIO("")

    def instrumented_subprocess_run(*args: Any, **kwargs: Any):
        command = args[0] if args else kwargs.get("args")
        source_file, line_number = _call_site(state)
        ablated = state.match_ablation(source_file=source_file, line_number=line_number, operation_type="exec_command")
        state.record(
            event_type="exec_command",
            summary=("Ablated subprocess.run" if ablated else "Blocked subprocess.run"),
            object_ref=str(command),
            arguments_summary=str(kwargs),
            attributes={"ablated": ablated, "blocked": not ablated, "source_file": source_file, "line_number": line_number},
        )
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    class InstrumentedPopen:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            command = args[0] if args else kwargs.get("args")
            source_file, line_number = _call_site(state)
            ablated = state.match_ablation(source_file=source_file, line_number=line_number, operation_type="exec_command")
            state.record(
                event_type="exec_command",
                summary=("Ablated subprocess.Popen" if ablated else "Blocked subprocess.Popen"),
                object_ref=str(command),
                arguments_summary=str(kwargs),
                attributes={"ablated": ablated, "blocked": not ablated, "source_file": source_file, "line_number": line_number},
            )
            self.returncode = 0

        def communicate(self, input: Any = None, timeout: Any = None) -> tuple[str, str]:
            return "", ""

        def wait(self, timeout: Any = None) -> int:
            return 0

    def instrumented_getenv(key: str, default: Any = None) -> Any:
        source_file, line_number = _call_site(state)
        if state.match_ablation(source_file=source_file, line_number=line_number, operation_type="read_env"):
            state.record(
                event_type="read_env",
                summary=f"Ablated environment read {key}",
                object_ref=key,
                arguments_summary=None,
                attributes={"ablated": True, "source_file": source_file, "line_number": line_number},
            )
            return default
        state.record(
            event_type="read_env",
            summary=f"Environment read {key}",
            object_ref=key,
            arguments_summary=None,
            attributes={"source_file": source_file, "line_number": line_number},
        )
        return originals["os_getenv"](key, default)

    def instrumented_getnode() -> int:
        source_file, line_number = _call_site(state)
        ablated = state.match_ablation(source_file=source_file, line_number=line_number, operation_type="collect_identifier")
        state.record(
            event_type="collect_identifier",
            summary=("Ablated hardware identifier read" if ablated else "Hardware identifier read"),
            object_ref="uuid.getnode",
            arguments_summary=None,
            attributes={"ablated": ablated, "source_file": source_file, "line_number": line_number},
        )
        return 0 if ablated else 123456789

    def instrumented_platform_node() -> str:
        source_file, line_number = _call_site(state)
        ablated = state.match_ablation(source_file=source_file, line_number=line_number, operation_type="collect_identifier")
        state.record(
            event_type="collect_identifier",
            summary=("Ablated hostname read" if ablated else "Hostname read"),
            object_ref="platform.node",
            arguments_summary=None,
            attributes={"ablated": ablated, "source_file": source_file, "line_number": line_number},
        )
        return "sandbox-host" if not ablated else ""

    def blocked_socket(*args: Any, **kwargs: Any):
        source_file, line_number = _call_site(state)
        state.record(
            event_type="network_send",
            summary="Blocked raw socket creation",
            object_ref=str(args[0]) if args else None,
            arguments_summary=str(kwargs),
            attributes={"blocked": True, "source_file": source_file, "line_number": line_number},
        )
        raise PermissionError("Network access is disabled; use an explicit API fixture.")

    def instrumented_urlopen(url: Any, *args: Any, **kwargs: Any):
        raw_url = str(getattr(url, "full_url", url))
        fixture = state.api_fixture(raw_url)
        source_file, line_number = _call_site(state)
        if fixture is None:
            state.record(
                event_type="network_send",
                summary=f"Blocked URL request {raw_url}",
                object_ref=raw_url,
                arguments_summary=str(kwargs),
                attributes={"blocked": True, "source_file": source_file, "line_number": line_number},
            )
            raise PermissionError("Network access is disabled; use an explicit API fixture.")
        content = str(fixture.get("content") or "").encode("utf-8")
        state.record(
            event_type="network_send",
            summary=f"Mocked URL request {raw_url}",
            object_ref=raw_url,
            arguments_summary=str(kwargs),
            attributes={
                "mocked": True,
                "fixture_id": fixture.get("fixture_id"),
                "source_file": source_file,
                "line_number": line_number,
            },
        )
        response = io.BytesIO(content)
        response.status = int((fixture.get("metadata") or {}).get("status_code") or 200)
        return response

    builtins.open = instrumented_open
    io.open = instrumented_open
    os.open = instrumented_os_open
    pathlib.Path.open = instrumented_path_open
    pathlib.Path.read_text = instrumented_read_text
    pathlib.Path.write_text = instrumented_write_text
    os.system = instrumented_system
    os.popen = instrumented_popen
    subprocess.run = instrumented_subprocess_run
    subprocess.Popen = InstrumentedPopen
    os.getenv = instrumented_getenv
    uuid.getnode = instrumented_getnode
    platform.node = instrumented_platform_node
    socket.socket = blocked_socket
    socket.create_connection = blocked_socket
    urllib.request.urlopen = instrumented_urlopen
    _install_requests_stub(state)
    return originals


def _restore_runtime_hooks(originals: dict[str, Any]) -> None:
    builtins.open = originals["open"]
    io.open = originals["io_open"]
    os.open = originals["os_open"]
    pathlib.Path.open = originals["path_open"]
    pathlib.Path.read_text = originals["path_read_text"]
    pathlib.Path.write_text = originals["path_write_text"]
    os.system = originals["os_system"]
    os.popen = originals["os_popen"]
    subprocess.run = originals["subprocess_run"]
    subprocess.Popen = originals["subprocess_popen"]
    os.getenv = originals["os_getenv"]
    uuid.getnode = originals["uuid_getnode"]
    platform.node = originals["platform_node"]
    socket.socket = originals["socket_socket"]
    socket.create_connection = originals["socket_create_connection"]
    urllib.request.urlopen = originals["urlopen"]


def _trace_factory(state: _SandboxState):
    sandbox_root = state.sandbox_root.resolve()

    def tracer(frame: Any, event: str, arg: Any):
        filename_value = frame.f_code.co_filename
        if not filename_value or filename_value.startswith("<"):
            return tracer
        filename = Path(filename_value)
        try:
            relative = filename.resolve().relative_to(sandbox_root)
        except Exception:
            return tracer
        function_name = frame.f_code.co_name
        attributes = {
            "source_file": relative.as_posix(),
            "line_number": frame.f_lineno,
            "function_name": function_name,
        }
        if event == "call":
            state.record(
                event_type="call",
                summary=f"Call {function_name}",
                object_ref=relative.as_posix(),
                arguments_summary=None,
                attributes=attributes,
            )
        elif event == "line":
            state.record(
                event_type="line",
                summary=f"Line {relative.as_posix()}:{frame.f_lineno} in {function_name}",
                object_ref=relative.as_posix(),
                arguments_summary=None,
                attributes=attributes,
            )
        elif event == "return":
            return_value = arg if isinstance(arg, str) else None
            if return_value:
                state.last_string_return = return_value
            state.record(
                event_type="return",
                summary=f"Return {function_name}",
                object_ref=relative.as_posix(),
                arguments_summary=_safe_repr(arg),
                attributes=attributes,
            )
        elif event == "exception":
            exception_type, exception_value, _ = arg
            state.record(
                event_type="exception",
                summary=f"Exception in {function_name}",
                object_ref=relative.as_posix(),
                arguments_summary=_safe_repr(exception_value),
                attributes={
                    **attributes,
                    "exception_type": getattr(exception_type, "__name__", str(exception_type)),
                },
            )
        return tracer

    return tracer


def _safe_repr(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return repr(value)[:200]
    except Exception:
        return f"<unreprable:{type(value).__name__}>"


def _load_fixtures(sandbox_root: Path) -> list[dict[str, Any]]:
    manifest_path = sandbox_root / ".skillscope" / "fixtures.json"
    if not manifest_path.exists():
        return []
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def run_request(request_path: Path, response_path: Path) -> int:
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    sandbox_root = Path(payload["sandbox_root"]).resolve()
    script_relative_path = payload["script_relative_path"]
    script_path = (sandbox_root / script_relative_path).resolve()
    try:
        script_path.relative_to(sandbox_root)
    except ValueError:
        response_path.write_text(
            json.dumps(
                {
                    "status": "blocked",
                    "final_output": "",
                    "stdout": "",
                    "stderr": "",
                    "trace_events": [],
                    "notes": ["The requested script path escaped the sandbox root."],
                    "error": f"invalid_script_path={script_relative_path}",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 0
    prompt = str(payload.get("prompt") or "")
    fixtures = _load_fixtures(sandbox_root)
    state = _SandboxState(
        sandbox_root=sandbox_root,
        script_path=script_path,
        ablation=payload.get("ablation"),
        fixtures=fixtures,
    )

    originals = _install_runtime_hooks(state)
    tracer = _trace_factory(state)
    stdout_capture = state.stdout_capture
    stderr_capture = state.stderr_capture
    final_output = ""
    status = "completed"
    error: str | None = None
    original_prompt_env = os.environ.get("SKILLSCOPE_USER_PROMPT")
    original_fixture_env: dict[str, str | None] = {}
    original_sys_path = list(sys.path)

    try:
        os.chdir(sandbox_root)
        os.environ["SKILLSCOPE_USER_PROMPT"] = prompt
        for fixture in fixtures:
            if fixture.get("fixture_type") != "env":
                continue
            key = str(fixture.get("target") or "")
            if not key:
                continue
            original_fixture_env[key] = os.environ.get(key)
            os.environ[key] = str(fixture.get("content") or "")
        script_dir = str(script_path.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        state.record(
            event_type="script_start",
            summary=f"Start script {script_relative_path}",
            object_ref=script_relative_path,
            arguments_summary=prompt[:200] or None,
            attributes={
                "script_relative_path": script_relative_path,
                "source_file": script_relative_path,
            },
        )
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            sys.settrace(tracer)
            runpy.run_path(str(script_path), run_name="__main__")
    except Exception:
        status = "failed"
        error = traceback.format_exc()
        state.record(
            event_type="script_error",
            summary=f"Script failed {script_relative_path}",
            object_ref=script_relative_path,
            arguments_summary=None,
            attributes={"error": error, "source_file": script_relative_path},
        )
    finally:
        state.record(
            event_type="script_end",
            summary=f"End script {script_relative_path}",
            object_ref=script_relative_path,
            arguments_summary=None,
            attributes={"status": status, "source_file": script_relative_path},
        )
        sys.settrace(None)
        if original_prompt_env is None:
            os.environ.pop("SKILLSCOPE_USER_PROMPT", None)
        else:
            os.environ["SKILLSCOPE_USER_PROMPT"] = original_prompt_env
        for key, original_value in original_fixture_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
        sys.path[:] = original_sys_path
        _restore_runtime_hooks(originals)

    stdout_value = stdout_capture.getvalue()
    stderr_value = stderr_capture.getvalue()
    if stdout_value.strip():
        final_output = stdout_value.strip()
    elif state.last_string_return.strip():
        final_output = state.last_string_return.strip()
    else:
        final_output = f"Executed {script_relative_path}"

    response = {
        "status": status,
        "final_output": final_output,
        "stdout": stdout_value,
        "stderr": stderr_value,
        "trace_events": state.trace_events,
        "notes": state.notes,
        "error": error,
    }
    response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: python -m skillscope.common.sandbox.python_worker <request.json> <response.json>", file=sys.stderr)
        return 2
    return run_request(Path(args[0]), Path(args[1]))


if __name__ == "__main__":
    raise SystemExit(main())
