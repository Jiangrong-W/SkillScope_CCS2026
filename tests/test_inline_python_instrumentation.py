from __future__ import annotations

import hashlib
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path

from skillscope.common.sandbox import SandboxPolicy
from skillscope.common.sandbox.tooling import InlineCommandTool, ToolInvocationRequest


class InlinePythonInstrumentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="skillscope-inline-python-proof-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.python = str(Path(sys.executable).resolve())
        self.tool = InlineCommandTool(SandboxPolicy(require_os_isolation=True, timeout_seconds=10))

    def _command(self, body: str) -> str:
        return shlex.join([self.python, "-S", "-c", "exec(" + json.dumps(body) + ")"])

    def _request(self, command: str) -> ToolInvocationRequest:
        return ToolInvocationRequest(
            "inline-proof-call", "inline_command", "inline-command-001",
            str(self.root), "Compute from the local input.",
            instruction_node_id="inline-owner",
            arguments={"command": command},
            metadata={
                "instruction_source_file": "SKILL.md",
                "instruction_source_range": {"start_line": 7, "end_line": 7},
            },
        )

    def test_literal_body_has_complete_physical_io_and_original_provenance(self) -> None:
        (self.root / "input.json").write_text('{"value":7}', encoding="utf-8")
        body = (
            "import json\nfrom pathlib import Path\n"
            "data=json.loads(Path('input.json').read_text())\n"
            "result={'answer':data['value']*2}\n"
            "Path('result.json').write_text(json.dumps(result))\n"
            "print(json.dumps(result))\n"
        )
        command = self._command(body)
        result = self.tool.invoke(self._request(command))
        self.assertEqual(result.status, "completed", result.error)
        self.assertEqual(result.stdout, '{"answer": 14}\n')
        self.assertEqual(result.final_output, '{"answer": 14}')
        self.assertEqual(json.loads((self.root / "result.json").read_text()), {"answer": 14})
        events = [e for e in result.trace_events if e["event_type"] in {"exec_command", "script_start", "file_read", "file_write", "script_end"}]
        self.assertEqual([e["event_type"] for e in events], ["exec_command", "script_start", "file_read", "file_write", "script_end"])
        self.assertEqual(events[0]["object_ref"], self.python)
        self.assertEqual(events[0]["attributes"]["runtime_command"], command)
        self.assertEqual(events[0]["attributes"]["runtime_command_tokens"], shlex.split(command))
        physical = [e for e in events if e["event_type"] in {"file_read", "file_write"}]
        self.assertEqual([e["object_ref"] for e in physical], ["input.json", "result.json"])
        for event in physical:
            attributes = event["attributes"]
            self.assertEqual(attributes["runtime_evidence"], "python_instrumented_operation")
            self.assertTrue(attributes["execution_count_observed"])
            self.assertTrue(attributes["temporal_order_observed"])
            self.assertTrue(attributes["source_file"].startswith(".skillscope-inline-python-"))
            self.assertIsInstance(attributes["source_start_column"], int)
            self.assertEqual(attributes["instruction_node_id"], "inline-owner")
            self.assertEqual(attributes["instruction_source_file"], "SKILL.md")
            self.assertEqual(attributes["instruction_source_range"], {"start_line": 7, "end_line": 7})
            self.assertEqual(attributes["instruction_command_sha256"], hashlib.sha256(command.encode()).hexdigest())
            self.assertEqual(attributes["inline_python_body_sha256"], hashlib.sha256(body.encode()).hexdigest())
        self.assertEqual(events[-1]["attributes"]["status"], "completed")
        self.assertEqual(result.metadata["tool_name"], "inline_command")
        self.assertEqual(result.metadata["trace_granularity"], "instrumented_python")
        self.assertFalse(result.metadata["network_allowed"])
        self.assertEqual(list(self.root.glob(".skillscope-inline-python-*")), [])

    def test_quoted_python_semicolons_and_shell_metacharacters_are_literals(self) -> None:
        body = "x=1; print(x)\nprint('|; && $HOME `literal`')\n"
        result = self.tool.invoke(self._request(self._command(body)))
        self.assertEqual(result.status, "completed", result.error)
        self.assertEqual(result.stdout, "1\n|; && $HOME `literal`\n")
        self.assertTrue(result.metadata["inline_python_instrumented"])

    def test_empty_stdout_keeps_the_existing_inline_result_contract(self) -> None:
        result = self.tool.invoke(self._request(self._command("def produce():\n    return 'not stdout'\nproduce()\n")))
        self.assertEqual(result.status, "completed", result.error)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.final_output, "Executed inline-command-001")

    def test_error_retains_end_evidence_and_cleans_temporary_source(self) -> None:
        result = self.tool.invoke(self._request(self._command("raise ValueError('controlled failure')\n")))
        self.assertEqual(result.status, "failed")
        self.assertEqual(sum(e["event_type"] == "script_start" for e in result.trace_events), 1)
        end = next(e for e in result.trace_events if e["event_type"] == "script_end")
        self.assertEqual(end["attributes"]["status"], "failed")
        self.assertEqual(list(self.root.glob(".skillscope-inline-python-*")), [])

    def test_rejects_entrypoint_context_private_members_and_alias_bypasses(self) -> None:
        bodies = [
            "print(__file__)", "import sys\nprint(sys.argv)",
            "import os\nprint(os.getenv('HOME'))", "import pathlib\nprint(pathlib.sys.argv)",
            "from pathlib import sys as s\nprint('unused alias')",
            "import random\nprint(random._os.getcwd())",
            "from random import _os as operating\nprint('unused alias')",
            "from pathlib import *\nprint('star import')",
            "print(getattr(1, '__class__'))", "print('{0.__class__}'.format(1))",
            "from operator import methodcaller as m\nprint(m('stat'))",
            "from operator import attrgetter as a\nimport pathlib\nprint(a('sys.argv')(pathlib))",
            "import pathlib\nprint('{0.sys.argv}'.format(pathlib))",
            "import pathlib\nprint('{obj.sys.argv}'.format_map({'obj': pathlib}))",
            "from string import Formatter as f\nimport pathlib\nprint(f().vformat('{0.sys.argv}', [pathlib], {}))",
            "import string\nimport pathlib\nprint(string.Formatter().get_field('0.sys.argv', [pathlib], {}))",
            "exec('print(1)')", "eval('1+1')", "compile('1','x','eval')",
            "from local_module import value\nprint(value)",
        ]
        for body in bodies:
            with self.subTest(body=body):
                self.assertIsNone(self.tool._literal_python_payload(self._command(body), self.root))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_directory_enumeration_and_metadata_use_shell_fallback(self) -> None:
        for member in ["glob", "rglob", "iterdir", "stat", "lstat"]:
            body = f"from pathlib import Path\nprint(Path('.').{member}())\n"
            with self.subTest(member=member):
                self.assertIsNone(self.tool._literal_python_payload(self._command(body), self.root))

    def test_shell_expansion_composition_and_unknown_options_are_not_routed(self) -> None:
        normal = self._command("print('normal')\n")
        payload = "exec(" + json.dumps("print('$HOME')\n") + ")"
        expanded = self.python + " -S -c \"" + payload.replace('"', '\\"') + "\""
        commands = [
            expanded, normal + " && printf additional", normal + " > output.txt",
            normal.replace(" -S -c ", " -S -X utf8 -c "),
            normal.replace(self.python, Path(self.python).name, 1),
            normal.replace(self.python, "./" + Path(self.python).name, 1),
        ]
        for command in commands:
            with self.subTest(command=command):
                self.assertIsNone(self.tool._literal_python_payload(command, self.root))
        # The old inline shell tool remains available for an ordinary command.
        result = self.tool.invoke(self._request("printf shell-fallback"))
        self.assertEqual(result.stdout, "shell-fallback")
        self.assertEqual(result.metadata["trace_granularity"], "bash_xtrace_command")
        self.assertNotIn("inline_python_instrumented", result.metadata)

    def test_stdlib_shadowing_is_not_routed(self) -> None:
        for name in ["json.py", "json.pyc", "json.cpython-314-darwin.so", "json.pyd"]:
            shadow = self.root / name
            shadow.write_bytes(b"local module shadow")
            with self.subTest(name=name):
                self.assertIsNone(self.tool._literal_python_payload(self._command("import json\nprint(json.dumps(1))\n"), self.root))
            shadow.unlink()
        self.assertFalse(any(p.name.startswith(".skillscope-inline-python-") for p in self.root.iterdir()))


if __name__ == "__main__":
    unittest.main()
