from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skillscope.common.config import AppConfig
from skillscope.common.llm import DisabledLLMClient, PromptAssetLoader
from skillscope.common.models import (
    ActionGraph,
    CandidateAction,
    CandidateExtractionResult,
    SourceRange,
    TaskSpec,
    UEGEdge,
    UEGNode,
)
from skillscope.common.sandbox import SandboxedSkillAgent, SkillInstaller
from skillscope.module1_candidate_extraction.bundle_loader import SkillBundleLoader
from skillscope.module1_candidate_extraction.code_graph_builder import CodeGraphBuilder
from skillscope.module1_candidate_extraction.skill_profile_extractor import SkillProfileExtractor
from skillscope.module1_candidate_extraction.ueg_composer import UEGComposer
from skillscope.module2_action_necessity_validation.replay_runner import ReplayRunner
from skillscope.module2_action_necessity_validation.trigger_detector import TaskTriggerDetector


class InstalledGraphIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="skillscope-graph-identity-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "skill"
        self.root.mkdir()
        (self.root / "SKILL.md").write_text(
            "- Verify the supplied input.\n"
            "- Continue only if the supplied input is ready.\n"
            "- Run `python3 helper.py`.\n",
            encoding="utf-8",
        )
        (self.root / "helper.py").write_text(
            "from pathlib import Path\n"
            "text = Path('input.txt').read_text()\n"
            "Path('result.txt').write_text(text.upper())\n"
            "print(text.upper())\n",
            encoding="utf-8",
        )
        (self.root / "input.txt").write_text("local input", encoding="utf-8")
        self.config = AppConfig(
            project_root=Path(__file__).resolve().parents[1],
            artifact_root=Path(self.temp.name) / "artifacts",
            sandbox_timeout_seconds=10,
        )
        self.client = DisabledLLMClient()
        self.prompts = PromptAssetLoader(self.config.project_root)

    def _analysis(self, *, swapped: bool = False) -> CandidateExtractionResult:
        bundle = SkillBundleLoader(self.config).load(self.root)
        first = "skill:instruction:002" if swapped else "skill:instruction:001"
        second = "skill:instruction:001" if swapped else "skill:instruction:002"
        check = UEGNode(
            first, "instruction", "INSTR_ACTION", "Verify supplied input",
            source_file="SKILL.md", source_range=SourceRange(1, 1),
            raw_text="Verify the supplied input.", operation_type="verify",
        )
        predicate = UEGNode(
            second, "instruction", "INSTR_PREDICATE", "Expected input is ready",
            source_file="SKILL.md", source_range=SourceRange(2, 2),
            raw_text="only if the supplied input is ready", operation_type="predicate",
        )
        helper = UEGNode(
            "skill:instruction:003", "instruction", "INSTR_ACTION", "Execute bundled helper",
            source_file="SKILL.md", source_range=SourceRange(3, 3),
            raw_text="Run `python3 helper.py`.", operation_type="exec_command",
            attributes={
                "invoked_scripts": ["helper.py"],
                "command_invocations": ["python3 helper.py"],
            },
        )
        instructions = ActionGraph(
            "skill:instruction", "instruction", [check, predicate, helper],
            [UEGEdge(first, second, "SEQUENTIAL"), UEGEdge(second, helper.node_id, "CONDITIONAL_TRUE")],
        )
        code = CodeGraphBuilder().build(bundle)
        ueg = UEGComposer().compose(bundle, instructions, code)
        return CandidateExtractionResult(
            bundle, SkillProfileExtractor().extract(bundle), instructions, code, ueg, [],
        )

    def _installer(self) -> SkillInstaller:
        return SkillInstaller(
            self.config, llm_client=self.client, prompt_loader=self.prompts,
        )

    def _agent(self) -> SandboxedSkillAgent:
        return SandboxedSkillAgent(
            self.config, llm_client=self.client, prompt_loader=self.prompts,
        )

    def _candidate_task(self, analysis, *, operation="file_access"):
        node = next(n for n in analysis.ueg.nodes if n.operation_type == operation)
        candidate = CandidateAction(
            "candidate", node.node_id, node.layer, node.summary,
            node.source_file, node.risk_tags, "offline regression", 1.0,
        )
        prefix = [n.node_id for n in analysis.instruction_graph.nodes]
        read = next(n for n in analysis.ueg.nodes if n.operation_type == "file_access")
        if node != read:
            prefix.append(read.node_id)
        task = TaskSpec(
            "task", candidate.candidate_id, "Uppercase the supplied local input.",
            chain_node_ids=[*prefix, node.node_id],
            expected_candidate_node_id=node.node_id,
        )
        return candidate, task

    def test_reuses_exact_analysis_without_graph_build_or_llm_calls(self) -> None:
        analysis = self._analysis()
        installer = self._installer()
        with (
            patch.object(installer.instruction_graph_builder, "build") as build_instruction,
            patch.object(installer.code_graph_builder, "build") as build_code,
            patch.object(installer.ueg_composer, "compose") as compose,
            patch.object(self.client, "complete_json") as llm,
        ):
            installed = installer.install(self.root, analysis=analysis)
        build_instruction.assert_not_called()
        build_code.assert_not_called()
        compose.assert_not_called()
        llm.assert_not_called()
        self.assertEqual(installed.instruction_graph, analysis.instruction_graph)
        self.assertEqual(installed.code_graphs, analysis.code_graphs)
        self.assertEqual(installed.ueg, analysis.ueg)
        self.assertEqual(installed.metadata["graph_source"], "provided_analysis")
        self.assertIsNot(installed.ueg, analysis.ueg)
        node = installed.instruction_graph.nodes[0]
        self.assertIs(node, installed.ueg.node_by_id(node.node_id))
        node.summary = "runtime-local mutation"
        self.assertNotEqual(node.summary, analysis.instruction_graph.nodes[0].summary)

    def test_omitted_analysis_still_builds_and_binds_a_fresh_graph(self) -> None:
        analysis = self._analysis(swapped=True)
        installer = self._installer()
        with patch.object(
            installer.instruction_graph_builder, "build", return_value=analysis.instruction_graph,
        ) as build:
            installed = installer.install(self.root)
        build.assert_called_once()
        self.assertEqual(installed.metadata["graph_source"], "fresh_bundle_analysis")
        callers = [edge.source for edge in installed.ueg.edges if edge.edge_type == "CALLS"]
        self.assertEqual(callers, ["skill:instruction:003"])

    def test_analysis_for_another_root_is_rejected(self) -> None:
        analysis = self._analysis()
        patched = Path(self.temp.name) / "patched"
        patched.mkdir()
        (patched / "SKILL.md").write_text("- Write a new result.\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different Skill bundle root"):
            self._installer().install(patched, analysis=analysis)

    def test_modified_bundle_is_not_installed_with_stale_analysis(self) -> None:
        analysis = self._analysis()
        (self.root / "helper.py").write_text("print('changed task')\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed after"):
            self._installer().install(self.root, analysis=analysis)

    def test_explicit_analysis_cannot_reuse_cached_independent_node_meanings(self) -> None:
        analysis = self._analysis()
        other = self._analysis(swapped=True)
        agent = self._agent()
        with patch.object(
            agent.installer.instruction_graph_builder, "build", return_value=other.instruction_graph,
        ) as build:
            independent = agent.install_skill(self.root)
            reused = agent.install_skill(self.root, analysis=analysis)
            newer = agent.install_skill(self.root, analysis=other)
        build.assert_called_once()
        self.assertEqual(independent.ueg.node_by_id("skill:instruction:001").summary, "Expected input is ready")
        self.assertEqual(reused.ueg.node_by_id("skill:instruction:001").summary, "Verify supplied input")
        self.assertEqual(newer.ueg.node_by_id("skill:instruction:001").summary, "Expected input is ready")
        self.assertIsNot(reused, independent)
        self.assertIsNot(newer, independent)

    def test_physical_code_trace_uses_analysis_ids_and_strict_prefix(self) -> None:
        analysis = self._analysis()
        agent = self._agent()
        with patch.object(agent.installer.instruction_graph_builder, "build") as build:
            installed = agent.install_skill(self.root, analysis=analysis)
        build.assert_not_called()
        record = agent.execute_installed_skill(
            installed_skill=installed, prompt="Uppercase the local input.",
            run_id="original", mode="original",
            instruction_node_ids=[n.node_id for n in analysis.instruction_graph.nodes],
        )
        self.assertEqual(record.status, "completed", record.stderr)
        self.assertIn("LOCAL INPUT", record.stdout)
        self.assertFalse((self.root / "result.txt").exists())
        candidate, task = self._candidate_task(analysis, operation="file_write")
        detector = TaskTriggerDetector()
        self.assertTrue(detector.detect(candidate=candidate, task=task, record=record, ueg=analysis.ueg).triggered)
        material = [p for p in record.raw_trace if p["event_type"] in {"file_read", "file_write"}]
        self.assertEqual([p["event_type"] for p in material], ["file_read", "file_write"])
        self.assertEqual(material[1]["node_id"], candidate.node_id)
        self.assertEqual(material[1]["attributes"]["instruction_node_id"], "skill:instruction:003")

        reversed_controls = copy.deepcopy(record)
        starts = [p for p in reversed_controls.raw_trace if p["event_type"] == "instruction_step_start"]
        starts[0]["node_id"], starts[1]["node_id"] = starts[1]["node_id"], starts[0]["node_id"]
        self.assertFalse(detector.detect(candidate=candidate, task=task, record=reversed_controls, ueg=analysis.ueg).triggered)

        wrong_caller = copy.deepcopy(record)
        next(p for p in wrong_caller.raw_trace if p["event_type"] == "file_write")["attributes"]["instruction_node_id"] = "skill:instruction:001"
        self.assertFalse(detector.detect(candidate=candidate, task=task, record=wrong_caller, ueg=analysis.ueg).triggered)

        selected_only = copy.deepcopy(record)
        selected_only.raw_trace = [p for p in selected_only.raw_trace if p["event_type"] != "file_write"]
        self.assertFalse(detector.detect(candidate=candidate, task=task, record=selected_only, ueg=analysis.ueg).triggered)

        reversed_material = copy.deepcopy(record)
        positions = [i for i, p in enumerate(reversed_material.raw_trace) if p["event_type"] in {"file_read", "file_write"}]
        a, b = positions
        reversed_material.raw_trace[a], reversed_material.raw_trace[b] = reversed_material.raw_trace[b], reversed_material.raw_trace[a]
        self.assertFalse(detector.detect(candidate=candidate, task=task, record=reversed_material, ueg=analysis.ueg).triggered)

    def test_replay_reuses_original_analysis_but_rebuilds_ablated_bundle(self) -> None:
        analysis = self._analysis()
        candidate, task = self._candidate_task(analysis)
        agent = self._agent()
        runner = ReplayRunner(sandboxed_agent=agent, force_chain_instruction_plan=True)
        with patch.object(
            agent.installer.instruction_graph_builder, "build",
            wraps=agent.installer.instruction_graph_builder.build,
        ) as build:
            outcome = runner.run_trigger_then_replay(analysis=analysis, task=task, candidate=candidate)
        self.assertTrue(outcome.trigger_evidence.triggered, outcome.trigger_evidence.reason)
        self.assertIsNotNone(outcome.replay_pair)
        self.assertEqual(outcome.original.metadata["graph_source"], "provided_analysis")
        self.assertEqual(outcome.replay_pair.replay.metadata["graph_source"], "fresh_bundle_analysis")
        self.assertTrue(outcome.replay_pair.replay.metadata["ablated_candidate_fingerprint_absent"])
        build.assert_called_once()
        self.assertNotEqual(build.call_args.args[0].root_path, str(self.root))

    def test_instruction_file_read_keeps_its_semantic_node_identity(self) -> None:
        skill_file = self.root / "SKILL.md"
        skill_file.write_text(
            skill_file.read_text(encoding="utf-8") + "- Read private_input.txt.\n",
            encoding="utf-8",
        )
        (self.root / "private_input.txt").write_text("synthetic private input", encoding="utf-8")
        analysis = self._analysis()
        read = UEGNode(
            "skill:instruction:004", "instruction", "INSTR_ACTION", "Read private input",
            source_file="SKILL.md", source_range=SourceRange(4, 4),
            raw_text="Read private_input.txt.", operation_type="file_read",
            object_ref="private_input.txt",
        )
        edge = UEGEdge("skill:instruction:003", read.node_id, "SEQUENTIAL")
        analysis.instruction_graph.nodes.append(read)
        analysis.instruction_graph.edges.append(edge)
        analysis.ueg.nodes.append(read)
        analysis.ueg.edges.append(edge)
        agent = self._agent()
        with patch.object(agent.installer.instruction_graph_builder, "build") as build:
            installed = agent.install_skill(self.root, analysis=analysis)
        build.assert_not_called()
        record = agent.execute_installed_skill(
            installed_skill=installed, prompt="Compute locally.", run_id="private-read",
            mode="original",
            instruction_node_ids=[n.node_id for n in analysis.instruction_graph.nodes],
        )
        physical = next(p for p in record.raw_trace if p["event_type"] == "file_read" and p["object_ref"] == "private_input.txt")
        self.assertEqual(physical["node_id"], read.node_id)
        candidate = CandidateAction(
            "private-read", read.node_id, read.layer, read.summary,
            read.source_file, [], "offline regression", 1.0,
        )
        task = TaskSpec(
            "private-read-task", candidate.candidate_id, "Compute locally.",
            chain_node_ids=[n.node_id for n in analysis.instruction_graph.nodes],
            expected_candidate_node_id=read.node_id,
        )
        detector = TaskTriggerDetector()
        self.assertTrue(detector.detect(candidate=candidate, task=task, record=record, ueg=analysis.ueg).triggered)
        # A real read labeled as the execute node is insufficient for this candidate.
        wrong_identity = copy.deepcopy(record)
        physical = next(p for p in wrong_identity.raw_trace if p["event_type"] == "file_read" and p["object_ref"] == "private_input.txt")
        physical["node_id"] = "skill:instruction:003"
        physical["attributes"]["instruction_node_id"] = "skill:instruction:003"
        self.assertFalse(detector.detect(candidate=candidate, task=task, record=wrong_identity, ueg=analysis.ueg).triggered)


if __name__ == "__main__":
    unittest.main()
