from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from skillscope.common.config import AppConfig
from skillscope.common.models import (
    ActionGraph, CandidateExtractionResult, RepairItem, RepairPlan,
    SkillBundle, SkillProfile, UnifiedExecutionGraph, ValidationResult,
)
from skillscope.module1_candidate_extraction.bundle_loader import SkillBundleLoader
from skillscope.module2_action_necessity_validation.service import ActionNecessityValidationRun
from skillscope.module3_control_flow_constrained_repair.bundle_projector import BundleProjector
from skillscope.module3_control_flow_constrained_repair.instruction_rewriter import InstructionRewriter
from skillscope.module3_control_flow_constrained_repair.repair_validator import RepairValidator


class MarkdownDispatchTests(unittest.TestCase):
    def _item(self, source_file: str = "process.py") -> RepairItem:
        source = Path(source_file)
        return RepairItem(
            repair_id="repair-receipt", overreach_id="overreach-receipt", node_id="code-receipt",
            layer="code", repair_type="REORGANIZE_CODE_AND_ADD_DISPATCH",
            target_files=["SKILL.md", source_file], source_file=source_file,
            source_start_line=4, source_end_line=4, raw_text="receipt.write_text('processed')",
            guard_condition="Create the additional receipt only when explicitly authorized and required.",
            metadata={
                "dispatch_source_file": source_file,
                "allowed_execution_unit": source.with_name(source.stem + "__task_allowed.py").as_posix(),
                "safe_execution_unit": source.with_name(source.stem + "__default_safe.py").as_posix(),
            },
        )

    def test_explicit_local_script_links_project_unique_dispatch_and_preserve_neighbors(self) -> None:
        sources = [
            ("Use [process.py](process.py) to read the supplied input and produce result.json. Run it once using Python.\n", "process.py"),
            ("1. Run [report helper](scripts/process.py) using Python to create the report.\n", "scripts/process.py"),
            ("Execute [report helper](./scripts/process.py) with Python3.\n", "scripts/process.py"),
            ("Use [helper](scripts/process.py) to create the report. Execute the linked script with Python 3.14.\n", "scripts/process.py"),
        ]
        for body, path in sources:
            with self.subTest(body=body), tempfile.TemporaryDirectory(prefix="skillscope-linked-dispatch-") as raw:
                root = Path(raw)
                target = root / "SKILL.md"
                original = "# Local processing\n\n" + body + "\nReturn the requested result without changing its format.\n"
                target.write_text(original, encoding="utf-8")
                item = self._item(path)
                InstructionRewriter().rewrite(patched_bundle_root=root, item=item)
                projected = target.read_text(encoding="utf-8")
                self.assertEqual(item.metadata["instruction_invocation_template"]["prefix_tokens"], ["python3"])
                self.assertEqual(item.metadata["instruction_invocation_template"]["suffix_tokens"], [])
                self.assertEqual(item.metadata["instruction_invocation_span"]["line_number"], 3)
                self.assertNotIn(item.metadata["instruction_invocation_span"]["matched_text"], projected)
                self.assertIn("the execution unit selected by the task-conditioned dispatch below", projected)
                self.assertIn(item.guard_condition, projected)
                self.assertIn("python3 " + item.metadata["safe_execution_unit"], projected)
                self.assertTrue(projected.startswith("# Local processing\n\n"))
                self.assertTrue(projected.endswith("Return the requested result without changing its format.\n"))
                if body.startswith("1."):
                    self.assertIn("1. Run the execution unit selected", projected)

    def test_reference_negated_structural_and_ambiguous_links_fail_closed(self) -> None:
        cases = [
            "The [helper](process.py) documents the calculation.\n",
            "Use [helper](process.py) as a reference.\n",
            "Use [helper](process.py) to read the input. Python is available.\n",
            "Do not run [helper](process.py) using Python.\n",
            "Do not ever run [helper](process.py) using Python.\n",
            "Never execute [helper](process.py) with Python.\n",
            "The literal text is `Run [helper](process.py) using Python`.\n",
            "The literal text is ``Run [helper](process.py) using Python``.\n",
            "Run [helper](process.py) using Node.\n",
            "An example: Run [helper](process.py) using Python.\n",
            "Run [first](process.py) and [second](other.py) using Python.\n",
            "Run [first](process.py) and [second](process.py) using Python.\n",
            "Use [helper](process.py) with [input](input.json). Run it using Python.\n",
            "Use [helper](process.py) or [remote](https://example.invalid/helper.py). Run it using Python.\n",
            "Run [helper](different/process.py) using Python.\n",
            "Run [helper](../process.py) using Python.\n",
            "Run [helper](/process.py) using Python.\n",
            "Run [helper](https://example.invalid/process.py) using Python.\n",
            "Run ![helper](process.py) using Python.\n",
            "Run [helper](process.py) and `python3 other.py`.\n",
            "Run [helper](process.py) using Python.\n\nRun [helper](process.py) using Python.\n",
            "# Run [helper](process.py) using Python\n",
            "---\ndescription: Run [helper](process.py) using Python\n---\n",
            "```text\nRun [helper](process.py) using Python.\n```\n",
        ]
        for body in cases:
            with self.subTest(body=body), tempfile.TemporaryDirectory(prefix="skillscope-linked-refusal-") as raw:
                root = Path(raw)
                target = root / "SKILL.md"
                source = body if body.startswith("---") else "# Workflow\n\n" + body
                target.write_text(source, encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "failed closed"):
                    InstructionRewriter().rewrite(patched_bundle_root=root, item=self._item())
                self.assertEqual(target.read_text(encoding="utf-8"), source)

    def test_backtick_interpreter_flags_and_arguments_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory(prefix="skillscope-linked-legacy-") as raw:
            root = Path(raw)
            target = root / "SKILL.md"
            target.write_text(
                "---\ndescription: [one](process.py) and [two](other.py) are documented helpers\n---\n\n"
                "Run `python3 -S process.py --style concise` and retain the report format.\n",
                encoding="utf-8",
            )
            item = self._item()
            InstructionRewriter().rewrite(patched_bundle_root=root, item=item)
            self.assertEqual(item.metadata["instruction_invocation_template"]["prefix_tokens"], ["python3", "-S"])
            self.assertEqual(item.metadata["instruction_invocation_template"]["suffix_tokens"], ["--style", "concise"])
            projected = target.read_text(encoding="utf-8")
            self.assertIn("python3 -S process__default_safe.py --style concise", projected)
            self.assertIn("and retain the report format.", projected)

    def test_linked_entrypoint_projects_safe_unit_preserving_result_and_optional_receipt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="skillscope-linked-bundle-") as raw:
            workspace = Path(raw)
            source_root = workspace / "source"
            source_root.mkdir()
            instructions = (
                "# Local total\n\n"
                "Use [processor](process.py) to read the input and produce result.json. Run it once using Python.\n"
            )
            (source_root / "SKILL.md").write_text(instructions, encoding="utf-8")
            (source_root / "input.json").write_text("[1, 2, 3]", encoding="utf-8")
            code = (
                "from pathlib import Path\nimport json\n"
                "data = json.loads(Path('input.json').read_text(encoding='utf-8'))\n"
                "result = {'total': sum(data)}\n"
                "Path('receipt.txt').write_text('processed', encoding='utf-8')\n"
                "Path('result.json').write_text(json.dumps(result), encoding='utf-8')\n"
                "print(json.dumps(result))\n"
            )
            (source_root / "process.py").write_text(code, encoding="utf-8")
            call = next(node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and node.func.attr == "write_text"
                        and "receipt.txt" in ast.get_source_segment(code, node))
            item = self._item()
            item.source_start_line, item.source_end_line = call.lineno, call.end_lineno
            item.source_start_column, item.source_end_column = call.col_offset, call.end_col_offset
            item.raw_text = ast.get_source_segment(code, call)
            item.metadata = {
                "source_candidate_id": "candidate-receipt",
                "candidate_semantics": {"operation_type": "file_write", "object_ref": "receipt.txt"},
            }
            config = AppConfig(project_root=Path(__file__).resolve().parents[1], artifact_root=workspace / "artifacts")
            bundle = SkillBundleLoader(config).load(source_root)
            patched = workspace / "patched"
            BundleProjector().project(bundle=bundle, plan=RepairPlan(skill_id=bundle.bundle_id, items=[item]), output_root=patched)
            for relative in ["process.py", item.metadata["safe_execution_unit"]]:
                completed = subprocess.run([sys.executable, "-I", "-S", str(patched / relative)],
                                           cwd=patched, capture_output=True, text=True, check=True)
                self.assertEqual(json.loads(completed.stdout), {"total": 6})
                self.assertEqual(json.loads((patched / "result.json").read_text()), {"total": 6})
                self.assertFalse((patched / "receipt.txt").exists())
            completed = subprocess.run([sys.executable, "-I", "-S", str(patched / item.metadata["allowed_execution_unit"])],
                                       cwd=patched, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(completed.stdout), {"total": 6})
            self.assertEqual((patched / "receipt.txt").read_text(), "processed")
            self.assertEqual((source_root / "SKILL.md").read_text(), instructions)

    def test_validator_installs_the_fresh_patched_graph_instead_of_original_graph(self) -> None:
        with tempfile.TemporaryDirectory(prefix="skillscope-patched-analysis-") as raw:
            workspace = Path(raw)
            original_root, patched_root = workspace / "original", workspace / "patched"
            original_root.mkdir(); patched_root.mkdir()
            def analysis(root: Path) -> CandidateExtractionResult:
                return CandidateExtractionResult(
                    bundle=SkillBundle(bundle_id=root.name, root_path=str(root)),
                    profile=SkillProfile(name=root.name, description="", use_when="", summary=""),
                    instruction_graph=ActionGraph(graph_id=root.name + "-instructions", layer="instruction"),
                    code_graphs=[], ueg=UnifiedExecutionGraph(skill_id=root.name), candidates=[],
                )
            original_analysis, patched_analysis = analysis(original_root), analysis(patched_root)
            candidate_service = Mock()
            candidate_service.run.return_value = patched_analysis
            agent = Mock()
            agent.install_skill.return_value = SimpleNamespace(bundle=patched_analysis.bundle)
            validator = RepairValidator(candidate_service=candidate_service, sandboxed_agent=agent)
            validation = ValidationResult(bundle_id=original_analysis.bundle.bundle_id)
            original = ActionNecessityValidationRun(analysis=original_analysis, validation=validation)
            report, observed_analysis, records = validator.validate(
                original_run=original, repair_plan=RepairPlan(skill_id=original_analysis.bundle.bundle_id),
                patched_bundle_root=patched_root,
            )
            candidate_service.run.assert_called_once_with(patched_root)
            agent.install_skill.assert_called_once_with(patched_root, analysis=patched_analysis)
            self.assertIs(observed_analysis, patched_analysis)
            self.assertIsNot(observed_analysis, original_analysis)
            self.assertEqual(records, [])
            self.assertFalse(report.repair_succeeded)


if __name__ == "__main__":
    unittest.main()
