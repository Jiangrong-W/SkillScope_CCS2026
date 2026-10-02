from __future__ import annotations

import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path

from skillscope.common.config import AppConfig
from skillscope.common.sandbox import (
    InlineCommandTool,
    SandboxedPythonRunner,
    SandboxPolicy,
    TracedSkillAgentRuntime,
)
from skillscope.common.sandbox.tooling import PythonScriptTool, ToolDispatcher, ToolRegistry
from skillscope.module1_candidate_extraction.bundle_loader import SkillBundleLoader
from skillscope.module1_candidate_extraction.code_graph_builder import CodeGraphBuilder
from skillscope.module1_candidate_extraction.instruction_graph_builder import InstructionGraphBuilder
from skillscope.module1_candidate_extraction.instruction_semantic_normalizer import (
    InstructionGraphEdgeSpec,
    InstructionGraphNodeSpec,
    InstructionGraphSpec,
    InstructionSemanticNormalizer,
)
from skillscope.module1_candidate_extraction.ueg_composer import UEGComposer


class _FixtureNormalizer(InstructionSemanticNormalizer):
    def __init__(self, factory=None) -> None:
        super().__init__()
        self.factory = factory

    def build_instruction_graph_spec(self, *, skill_name, markdown_blocks):
        if self.factory is not None:
            return self.factory(markdown_blocks)
        nodes = [
            InstructionGraphNodeSpec(
                f"n{i}", "INSTR_ACTION", "Execute the exact command",
                "Run this exact command once", [block.block_id], "execute",
            )
            for i, block in enumerate(markdown_blocks, start=1)
            if block.block_type != "header"
        ]
        return InstructionGraphSpec(
            nodes=nodes,
            edges=[InstructionGraphEdgeSpec(a.local_id, b.local_id, "SEQUENTIAL") for a, b in zip(nodes, nodes[1:])],
            strategy="offline_fixture",
        )


class SourceCommandBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="skillscope-source-command-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = Path(__file__).resolve().parents[1]
        self.config = AppConfig(project_root=self.project, artifact_root=self.root / "unused-artifacts")
        self.python = str(Path(sys.executable).resolve())
        self.policy = SandboxPolicy(require_os_isolation=True, timeout_seconds=10)

    def _command(self, code: str) -> str:
        return f"{shlex.quote(self.python)} -S -c {shlex.quote('exec(' + json.dumps(code) + ')')}"

    def _graph(self, source: str, normalizer=None):
        (self.root / "SKILL.md").write_text(source, encoding="utf-8")
        bundle = SkillBundleLoader(self.config).load(self.root)
        builder = InstructionGraphBuilder(normalizer or _FixtureNormalizer())
        instructions = builder.build(bundle)
        ueg = UEGComposer().compose(bundle, instructions, CodeGraphBuilder().build(bundle))
        return bundle, instructions, ueg

    def _runtime(self):
        return TracedSkillAgentRuntime(ToolDispatcher(ToolRegistry([
            InlineCommandTool(self.policy),
            PythonScriptTool(SandboxedPythonRunner(self.project, self.policy)),
        ])))

    def _run(self, ueg):
        return self._runtime().execute(
            source_bundle_root=self.root, sandbox_root=self.root, ueg=ueg,
            prompt="Perform the declared local computation.", run_id="source-command",
            mode="original",
            instruction_node_ids=[n.node_id for n in ueg.nodes if n.layer == "instruction" and n.node_type not in {"ENTRY", "EXIT"}],
        )

    def test_recovers_exact_quoted_command_from_truncated_execution_excerpt(self) -> None:
        command = self._command("from pathlib import Path\nPath('quoted.txt').write_text(\"Alice's quoted result\")\n")
        source = f"# Workflow\nRun this exact command once: `{command}`.\n"
        _, graph, ueg = self._graph(source)
        owner = next(n for n in graph.nodes if n.node_type == "INSTR_ACTION")
        self.assertEqual(owner.attributes["command_invocations"], [command])
        self.assertIn(f"`{command}`", owner.raw_text)
        outcome = self._run(ueg)
        self.assertEqual(outcome.status, "completed", outcome.stderr)
        self.assertEqual((self.root / "quoted.txt").read_text(), "Alice's quoted result")
        self.assertEqual(sum(e["event_type"] == "exec_command" for e in outcome.raw_trace), 1)

    def test_primary_and_additional_commands_execute_once_each(self) -> None:
        (self.root / "input.json").write_text('{"values":[2,3,5]}', encoding="utf-8")
        main = self._command(
            "import json\nfrom pathlib import Path\n"
            "data = json.loads(Path('input.json').read_text())\n"
            "result = {'total': sum(data['values'])}\n"
            "Path('result.json').write_text(json.dumps(result))\nprint(json.dumps(result))\n"
        )
        extra = self._command(
            "from pathlib import Path\np=Path('audit.log')\n"
            "p.write_text((p.read_text() if p.exists() else '') + 'once\\n')\n"
        )
        source = f"# Workflow\nRun this exact command once: `{main}`.\n\nRun this exact command once: `{extra}`.\n"
        _, graph, ueg = self._graph(source)
        self.assertEqual([c for n in graph.nodes for c in n.attributes.get("command_invocations", [])], [main, extra])
        outcome = self._run(ueg)
        self.assertEqual(outcome.status, "completed", outcome.stderr)
        self.assertEqual(json.loads((self.root / "result.json").read_text()), {"total": 10})
        self.assertEqual((self.root / "audit.log").read_text(), "once\n")
        self.assertEqual(sum(e["event_type"] == "exec_command" for e in outcome.raw_trace), 2)

    def test_internal_actions_and_predicates_do_not_inherit_enclosing_command(self) -> None:
        command = self._command("print('one execution')\n")
        source = f"# Workflow\nRun this exact command once: `{command}`.\n"

        def factory(blocks):
            block = next(b for b in blocks if b.block_type == "paragraph")
            nodes = [
                InstructionGraphNodeSpec("run", "INSTR_ACTION", "Execute the exact command", "Run this exact command once", [block.block_id], "execute"),
                InstructionGraphNodeSpec("read", "INSTR_ACTION", "Read the input", block.text, [block.block_id], "read"),
                InstructionGraphNodeSpec("condition", "INSTR_PREDICATE", "Input is ready", block.text, [block.block_id], "predicate"),
                InstructionGraphNodeSpec("write", "INSTR_ACTION", "Write the result", block.text, [block.block_id], "write"),
            ]
            return InstructionGraphSpec(nodes=nodes, strategy="offline_fixture")

        _, graph, ueg = self._graph(source, _FixtureNormalizer(factory))
        owners = [n for n in graph.nodes if n.attributes.get("command_invocations")]
        self.assertEqual(len(owners), 1)
        self.assertEqual(owners[0].summary, "Execute the exact command")
        outcome = self._run(ueg)
        self.assertEqual(outcome.stdout.strip(), "one execution")
        self.assertEqual(sum(e["event_type"] == "tool_call_start" for e in outcome.raw_trace), 1)

    def test_examples_references_negation_and_code_fences_are_not_commands(self) -> None:
        command = self._command("from pathlib import Path\nPath('must-not-exist.txt').write_text('bad')\n")
        source = (
            f"# Examples\nRun this exact command once: `{command}`.\n\n"
            f"# Workflow\nFor example, run `{command}`.\n\n"
            f"Do not run `{command}`.\n\n"
            f"For reference, use `{command}`.\n\n"
            f"The command `{command}` is displayed as text.\n\n"
            f"Run the following code example:\n\n```sh\n{command}\n```\n"
        )
        _, graph, ueg = self._graph(source)
        self.assertFalse(any(n.attributes.get("command_invocations") for n in graph.nodes))
        outcome = self._run(ueg)
        self.assertFalse(any(e["event_type"] == "tool_call_start" for e in outcome.raw_trace))
        self.assertFalse((self.root / "must-not-exist.txt").exists())

    def test_ambiguous_execution_owners_are_left_unbound(self) -> None:
        command = self._command("from pathlib import Path\nPath('ambiguous.txt').write_text('bad')\n")
        source = f"Run this exact command once: `{command}`.\n"

        def factory(blocks):
            block = blocks[0]
            return InstructionGraphSpec(nodes=[
                InstructionGraphNodeSpec(name, "INSTR_ACTION", "Execute the command", "Run this exact command once", [block.block_id], "execute")
                for name in ["first", "second"]
            ], strategy="offline_fixture")

        _, graph, ueg = self._graph(source, _FixtureNormalizer(factory))
        self.assertFalse(any(n.attributes.get("command_invocations") for n in graph.nodes))
        self.assertEqual(graph.metadata["unbound_explicit_invocations"][0]["reason"], "execution_node_assignment_ambiguous")
        self._run(ueg)
        self.assertFalse((self.root / "ambiguous.txt").exists())

    def test_negated_condition_keeps_execution_but_negated_action_does_not(self) -> None:
        for name in ["selected.py", "forbidden.py"]:
            (self.root / name).write_text("print('branch')\n", encoding="utf-8")
        source = (
            "When Action 1 hold and Action 2 do not hold, run `python3 selected.py`.\n\n"
            "When Action 1 do not hold, do not run `python3 forbidden.py`.\n\n"
            "Do not run `python3 forbidden.py` when Action 1 do not hold.\n"
        )
        _, graph, _ = self._graph(source)
        self.assertEqual(
            [cmd for node in graph.nodes for cmd in node.attributes.get("command_invocations", [])],
            ["python3 selected.py"],
        )
        self.assertEqual(
            [path for node in graph.nodes for path in node.attributes.get("invoked_scripts", [])],
            ["selected.py"],
        )

    def test_normalized_fabricated_command_cannot_replace_source_command(self) -> None:
        command = self._command("from pathlib import Path\nPath('source.txt').write_text('source')\n")
        source = f"Run this exact command once: `{command}`.\n"

        def factory(blocks):
            return InstructionGraphSpec(nodes=[InstructionGraphNodeSpec(
                "run", "INSTR_ACTION", "Execute the command", "Run `printf forged > forged.txt`.",
                [blocks[0].block_id], "execute",
            )], strategy="offline_fixture")

        _, graph, ueg = self._graph(source, _FixtureNormalizer(factory))
        self.assertEqual([c for n in graph.nodes for c in n.attributes.get("command_invocations", [])], [command])
        self._run(ueg)
        self.assertEqual((self.root / "source.txt").read_text(), "source")
        self.assertFalse((self.root / "forged.txt").exists())

    def test_versioned_python_is_recognized_without_unknown_executable_fallback(self) -> None:
        bundle, _, _ = self._graph("Run `python3.14 -S -c 'print(1)'`.\n")
        builder = InstructionGraphBuilder()
        self.assertTrue(builder._is_explicit_command("/some/runtime/python3.14 -S -c 'print(1)'", bundle))
        self.assertFalse(builder._is_explicit_command("python-unrecognized -c 'print(1);print(2)'", bundle))
        self.assertFalse(builder._is_explicit_command("'python3.14", bundle))

    def test_versioned_interpreter_for_bundled_script_dispatches_once(self) -> None:
        (self.root / "helper.py").write_text(
            "from pathlib import Path\np=Path('script-count.txt')\n"
            "p.write_text((p.read_text() if p.exists() else '') + 'once\\n')\nprint('script result')\n",
            encoding="utf-8",
        )
        command = f"{shlex.quote(self.python)} -S helper.py"
        _, graph, ueg = self._graph(f"Run this exact command once: `{command}`.\n")
        owner = next(n for n in graph.nodes if n.attributes.get("command_invocations"))
        self.assertEqual(owner.attributes["invoked_scripts"], ["helper.py"])
        requests = self._runtime()._build_tool_requests_for_node(
            node=owner, sandbox_root=self.root, prompt="Run helper", run_id="versioned-script",
        )
        self.assertEqual([r.tool_name for r in requests], ["python_script"])
        outcome = self._run(ueg)
        self.assertEqual(outcome.status, "completed", outcome.stderr)
        self.assertEqual((self.root / "script-count.txt").read_text(), "once\n")
        self.assertEqual(sum(e["event_type"] == "tool_call_start" for e in outcome.raw_trace), 1)


if __name__ == "__main__":
    unittest.main()
