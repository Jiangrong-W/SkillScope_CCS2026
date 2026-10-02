from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from skillscope.common.models import ExecutionRecord, RepairItem, SourceRange, ExecutionEvent, UEGNode, UnifiedExecutionGraph
from skillscope.module3_control_flow_constrained_repair.repair_validator import RepairValidator


class ProjectedCoreProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = "for n in range(3):\n    print(n)\n"
        (self.root / "compute.py").write_text(self.source)
        (self.root / "SKILL.md").write_text("# Task\nUse [compute.py](compute.py) to produce the result. Run it once using Python.\n")
        self.item = RepairItem(
            repair_id="repair", overreach_id="overreach", node_id="removed", layer="code",
            repair_type="REORGANIZE_CODE_AND_ADD_DISPATCH", source_file="compute.py",
            metadata={"dispatch_source_file": "compute.py", "safe_semantic_proof_complete": True,
                      "original_source_sha256": hashlib.sha256(self.source.encode()).hexdigest(),
                      "safe_execution_unit": "compute__default_safe.py",
                      "allowed_execution_unit": "compute__task_allowed.py"},
        )
        self.instruction = UEGNode(
            "instruction", "instruction", "INSTR_ACTION", "Execute compute.py to produce the result",
            source_file="SKILL.md", source_range=SourceRange(2, 2),
            raw_text="Use [compute.py](compute.py) to produce the result", operation_type="execute",
        )
        self.predicate_node = UEGNode(
            "original-predicate", "code", "CODE_PREDICATE", "For n in range(3)",
            source_file="compute.py", source_range=SourceRange(1, 1), raw_text="n in range(3)",
            operation_type="predicate", attributes={"scope_name": "<module>", "predicate_kind": "for_iteration"},
        )
        self.projected = copy.deepcopy(self.predicate_node)
        self.projected.node_id = "projected-predicate"
        self.projected.source_file = "compute__default_safe.py"
        self.original = SimpleNamespace(bundle=SimpleNamespace(root_path=str(self.root)),
                                        ueg=UnifiedExecutionGraph("original", [self.instruction, self.predicate_node]))
        self.patched = SimpleNamespace(ueg=UnifiedExecutionGraph("patched", [self.projected]))
        self.validator = RepairValidator(candidate_service=None, sandboxed_agent=None)
        self.integrity = {key: True for key in ("complete", "unit_hashes_valid", "safe_semantics_valid", "neutralization_complete")}

    def invocation(self):
        return self.validator._grounded_original_invocation(
            original_analysis=self.original, node_id="instruction", item=self.item,
            expected_tokens=["python3", "compute.py"],
        )

    def record(self, source_file, counts=(2,)):
        events = []
        for call_number, count in enumerate(counts):
            for kind in ("tool_call_start", "script_start", *("line",) * count, "script_end", "tool_call_end"):
                events.append(ExecutionEvent(kind, kind, object_ref=source_file, attributes={
                    "tool_name": "python_script", "tool_call_id": f"physical-call-{call_number}",
                    "source_file": source_file, "script_relative_path": source_file,
                    "line_number": 1, "status": "completed", "function_name": "<module>",
                }))
        return ExecutionRecord(run_id=source_file, mode="repair_validation", prompt="Compute the result", status="completed", trace=events)

    def predicate(self, original=None, patched=None, integrity=None):
        return self.validator._projected_predicate_preserved(
            node_id=self.predicate_node.node_id, item=self.item, original_analysis=self.original,
            original_record=original or self.record("compute.py"), patched_analysis=self.patched,
            patched_record=patched or self.record("compute__default_safe.py"),
            integrity=self.integrity if integrity is None else integrity,
        )

    def projected_invocation(self, summary="Use compute.py to produce the result", original=True):
        self.item.metadata.setdefault("instruction_invocation_template", {
            "prefix_tokens": ["python3"], "source_token": "compute.py", "suffix_tokens": [],
        })
        return self.validator._projected_invocation_preserved(
            summary=summary, candidate=SimpleNamespace(source_file="compute.py"), item=self.item,
            record=self.record("compute__default_safe.py"), patched_analysis=self.patched,
            projection_integrity_entry=self.integrity,
            original_analysis=self.original if original else None,
            original_instruction_node_id=self.instruction.node_id,
        )

    def test_markdown_invocation_requires_the_actual_source_span_and_template(self):
        self.assertTrue(self.invocation())
        self.instruction.source_range = SourceRange(1, 1)
        self.assertFalse(self.invocation())
        self.instruction.source_range = SourceRange(2, 2)
        self.instruction.raw_text = "Run another.py"
        self.assertFalse(self.invocation())

    def test_reference_negation_ambiguity_and_source_escape_fail_closed(self):
        for text in (
            "# Task\nSee [compute.py](compute.py) for implementation details.\n",
            "# Task\nDo not run [compute.py](compute.py) using Python.\n",
            "# Task\nRun [compute.py](compute.py) using Python.\nRun [compute.py](compute.py) using Python.\n",
        ):
            (self.root / "SKILL.md").write_text(text)
            with self.subTest(text=text):
                self.assertFalse(self.invocation())
        self.instruction.source_file = "../SKILL.md"
        self.assertFalse(self.invocation())

    def test_real_bound_header_visits_cover_the_identical_projected_predicate(self):
        record = self.record("compute__default_safe.py")
        self.assertTrue(self.predicate(patched=record))
        self.assertEqual(record.metadata["projected_predicate_coverage_evidence"][0]["tool_bound_header_visits"], 2)

    def test_graph_presence_alone_or_summary_mentions_do_not_establish_execution(self):
        record = self.record("compute__default_safe.py")
        record.trace = [e for e in record.trace if e.event_type != "line"]
        record.executed_node_ids = [self.projected.node_id]
        self.assertFalse(self.predicate(patched=record))

    def test_boundaries_tool_owner_and_observed_visits_must_match(self):
        for kind in ("tool_call_start", "script_start", "script_end", "line"):
            record = self.record("compute__default_safe.py")
            record.trace.remove(next(e for e in record.trace if e.event_type == kind))
            with self.subTest(kind=kind):
                self.assertFalse(self.predicate(patched=record))
        for field, value in (("tool_call_id", "other-call"), ("source_file", "other.py"), ("line_number", 2), ("function_name", "other"), ("tool_name", "shell_script")):
            record = self.record("compute__default_safe.py")
            for event in record.trace:
                if event.event_type == "line":
                    event.attributes[field] = value
            with self.subTest(field=field):
                self.assertFalse(self.predicate(patched=record))

    def test_static_semantics_original_hash_and_projection_integrity_are_required(self):
        for field, value in (("raw_text", "n in range(4)"), ("source_file", "another.py"), ("operation_type", "call")):
            old = getattr(self.projected, field)
            setattr(self.projected, field, value)
            with self.subTest(field=field):
                self.assertFalse(self.predicate())
            setattr(self.projected, field, old)
        for key in self.integrity:
            self.assertFalse(self.predicate(integrity={**self.integrity, key: False}))
        (self.root / "compute.py").write_text(self.source + "print('changed')\n")
        self.assertFalse(self.predicate())

    def test_neighboring_instruction_on_the_same_line_cannot_borrow_the_command(self):
        self.instruction.raw_text = "to produce the result"
        self.assertFalse(self.invocation())

    def test_instruction_columns_must_contain_the_matched_invocation(self):
        text = (self.root / "SKILL.md").read_text().splitlines()[1]
        start = text.index("[compute.py]")
        end = start + len("[compute.py](compute.py)")
        self.instruction.source_range = SourceRange(2, 2, start, end)
        self.assertTrue(self.invocation())
        for columns in ((start + 1, end), (start, end - 1)):
            self.instruction.source_range = SourceRange(2, 2, *columns)
            with self.subTest(columns=columns):
                self.assertFalse(self.invocation())

    def test_all_four_boundaries_must_be_unique_completed_and_from_the_same_tool_source(self):
        source = "compute__default_safe.py"
        for kind in ("tool_call_start", "script_start", "script_end", "tool_call_end"):
            for mutation in ("missing", "duplicate", "wrong_object", "wrong_tool"):
                record = self.record(source)
                event = next(e for e in record.trace if e.event_type == kind)
                if mutation == "missing":
                    record.trace.remove(event)
                elif mutation == "duplicate":
                    record.trace.append(copy.deepcopy(event))
                elif mutation == "wrong_object":
                    event.object_ref = "another.py"
                else:
                    event.attributes["tool_name"] = "shell_script"
                with self.subTest(kind=kind, mutation=mutation):
                    self.assertFalse(self.predicate(patched=record))
        for kind in ("script_end", "tool_call_end"):
            record = self.record(source)
            next(e for e in record.trace if e.event_type == kind).attributes["status"] = "failed"
            with self.subTest(kind=kind, mutation="failed"):
                self.assertFalse(self.predicate(patched=record))
        for kind, field in (("script_start", "source_file"), ("script_start", "script_relative_path"), ("script_end", "source_file")):
            record = self.record(source)
            next(e for e in record.trace if e.event_type == kind).attributes[field] = "another.py"
            with self.subTest(kind=kind, field=field):
                self.assertFalse(self.predicate(patched=record))

    def test_out_of_order_boundaries_and_outside_lines_are_rejected(self):
        source = "compute__default_safe.py"
        for first, second in ((0, 1), (1, 4), (4, 5)):
            record = self.record(source)
            record.trace[first], record.trace[second] = record.trace[second], record.trace[first]
            with self.subTest(first=first, second=second):
                self.assertFalse(self.predicate(patched=record))
        for destination in (0, 5):
            record = self.record(source)
            header = record.trace.pop(2)
            record.trace.insert(destination, header)
            with self.subTest(destination=destination):
                self.assertFalse(self.predicate(patched=record))
        for kind in ("script_error", "tool_call_error"):
            record = self.record(source)
            error = copy.deepcopy(record.trace[2])
            error.event_type = kind
            record.trace.insert(3, error)
            with self.subTest(kind=kind):
                self.assertFalse(self.predicate(patched=record))

    def test_line_graph_object_mapping_does_not_replace_source_and_scope_provenance(self):
        record = self.record("compute__default_safe.py")
        for event in record.trace:
            if event.event_type == "line":
                event.object_ref = "range"
        self.assertTrue(self.predicate(patched=record))

    def test_visit_counts_must_match_per_call_in_execution_order(self):
        for original, patched in (((1, 1), (2,)), ((2,), (1, 1)), ((1, 3), (3, 1))):
            with self.subTest(original=original, patched=patched):
                self.assertFalse(self.predicate(
                    original=self.record("compute.py", original),
                    patched=self.record("compute__default_safe.py", patched),
                ))
        record = self.record("compute__default_safe.py", (1, 3))
        self.assertTrue(self.predicate(original=self.record("compute.py", (1, 3)), patched=record))
        self.assertEqual(record.metadata["projected_predicate_coverage_evidence"][0]["tool_bound_header_visit_counts"], [1, 3])
        for event in record.trace:
            event.attributes["tool_call_id"] = "same-call"
        self.assertFalse(self.predicate(original=self.record("compute.py", (1, 3)), patched=record))

    def test_only_one_executed_projection_unit_and_one_matching_predicate_are_allowed(self):
        record = self.record("compute__default_safe.py")
        other = self.record("compute__task_allowed.py")
        for event in other.trace:
            event.attributes["tool_call_id"] = "allowed-call"
        record.trace.extend(other.trace)
        self.assertFalse(self.predicate(patched=record))
        for graph, node in ((self.original.ueg, self.predicate_node), (self.patched.ueg, self.projected)):
            duplicate = copy.deepcopy(node)
            duplicate.node_id += "-duplicate"
            duplicate.source_range = SourceRange(2, 2)
            graph.nodes.append(duplicate)
            with self.subTest(graph=graph.skill_id):
                self.assertFalse(self.predicate())
            graph.nodes.remove(duplicate)

    def test_scope_kind_and_completed_record_status_are_required(self):
        for field in ("scope_name", "predicate_kind"):
            original = self.projected.attributes[field]
            self.projected.attributes[field] = "different"
            with self.subTest(field=field):
                self.assertFalse(self.predicate())
            self.projected.attributes[field] = original
        for side in ("original", "patched"):
            record = self.record("compute.py" if side == "original" else "compute__default_safe.py")
            record.status = "failed"
            with self.subTest(side=side):
                self.assertFalse(self.predicate(**{side: record}))

    def test_use_summary_requires_the_exact_grounded_source_invocation(self):
        self.assertTrue(self.projected_invocation())
        self.assertTrue(self.projected_invocation("Use compute.py using Python to produce the result"))
        self.assertFalse(self.projected_invocation(original=False))
        self.instruction.operation_type = "reference"
        self.assertFalse(self.projected_invocation())

    def test_use_reference_without_a_declared_run_fails_even_with_language_in_summary(self):
        for text in (
            "Use [compute.py](compute.py) to produce the result.",
            "Use [compute.py](compute.py) as a Python reference.",
        ):
            (self.root / "SKILL.md").write_text("# Task\n" + text + "\n")
            self.instruction.raw_text = text
            with self.subTest(text=text):
                self.assertFalse(self.projected_invocation("Use compute.py using Python"))

    def test_use_cannot_borrow_a_neighboring_command_or_incompatible_template(self):
        self.instruction.raw_text = "to produce the result"
        self.assertFalse(self.projected_invocation())
        self.instruction.raw_text = "Use [compute.py](compute.py) to produce the result"
        self.item.metadata["instruction_invocation_template"] = {
            "prefix_tokens": ["python"], "source_token": "compute.py", "suffix_tokens": [],
        }
        self.assertFalse(self.projected_invocation("Use compute.py using Python"))
        self.item.metadata["instruction_invocation_template"]["prefix_tokens"] = ["python3"]
        self.item.metadata["instruction_invocation_template"]["suffix_tokens"] = ["--unobserved"]
        self.assertFalse(self.projected_invocation())

    def test_use_invocation_columns_must_contain_the_actual_command(self):
        text = (self.root / "SKILL.md").read_text().splitlines()[1]
        start = text.index("[compute.py]")
        end = start + len("[compute.py](compute.py)")
        self.instruction.source_range = SourceRange(2, 2, start, end)
        self.assertTrue(self.projected_invocation())
        for columns in ((start + 1, end), (start, end - 1)):
            self.instruction.source_range = SourceRange(2, 2, *columns)
            with self.subTest(columns=columns):
                self.assertFalse(self.projected_invocation("Use compute.py using Python"))


if __name__ == "__main__":
    unittest.main()
