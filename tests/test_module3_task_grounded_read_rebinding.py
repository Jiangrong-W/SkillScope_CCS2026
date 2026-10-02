from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from skillscope.common.models import RepairItem
from skillscope.module3_control_flow_constrained_repair.code_rewriter import (
    CodeRewriter, PrivilegeSemanticAnalyzer, _task_grounded_read_rebindings,
)


class TaskGroundedReadRebindingTests(unittest.TestCase):
    SOURCE = (
        'import json\nfrom pathlib import Path\n'
        '# Preserve the complete nonempty computation.\n'
        "data = json.loads(Path('restricted.json').read_text(encoding='utf-8'))\n"
        "result = {'squares': sum(value ** 2 for value in data['values']), 'count': len(data['values'])}\n"
        "Path('answer.json').write_text(json.dumps(result), encoding='utf-8')\n"
        'print(json.dumps(result))\n'
    )

    def item(self, source=None, *, method='read_text', resource='restricted.json', prompt=None, repair_id='repair-input'):
        source = self.SOURCE if source is None else source
        node = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == method
                    and resource in ast.get_source_segment(source, node))
        prompt = prompt or 'Compute the full answer from the supplied data. Read only submitted.json and the executable artifacts. Do not read restricted.json.'
        return RepairItem(repair_id=repair_id, overreach_id='overreach-' + repair_id, node_id='node-' + repair_id,
            layer='code', repair_type='REORGANIZE_CODE_AND_ADD_DISPATCH', target_files=['SKILL.md', 'processor.py'],
            source_file='processor.py', source_start_line=node.lineno, source_end_line=node.end_lineno,
            source_start_column=node.col_offset, source_end_column=node.end_col_offset,
            raw_text=ast.get_source_segment(source, node), metadata={'source_candidate_id': 'candidate-' + repair_id,
                'descriptor_contexts': [{'final_verdict': 'overprivileged', 'task_context': {'intent': prompt},
                    'material_action_instances': [{'intent': prompt, 'operation': 'file_read', 'source': resource}]}]})

    def execute(self, root, relative):
        # An independent Python audit witness records actual opened resources.
        script = (
            'import json,runpy,sys\nopened=[]\n'
            'def audit(event,args):\n'
            ' if event=="open" and isinstance(args[0],str): opened.append(args[0])\n'
            'sys.addaudithook(audit)\nrunpy.run_path(sys.argv[1],run_name="__main__")\n'
            'sys.stderr.write(json.dumps(opened))\n'
        )
        result = subprocess.run([sys.executable, '-I', '-S', '-c', script, str(root / relative)],
                                cwd=root, capture_output=True, text=True, check=True)
        return json.loads(result.stdout), json.loads(result.stderr)

    def test_exact_authorized_read_preserves_computation_for_multiple_actual_inputs(self):
        with tempfile.TemporaryDirectory(prefix='skillscope-authorized-input-') as raw:
            root = Path(raw)
            (root / 'processor.py').write_text(self.SOURCE, encoding='utf-8')
            (root / 'submitted.json').write_text('{"values":[2,3]}', encoding='utf-8')
            (root / 'restricted.json').write_text('{"values":[7]}', encoding='utf-8')
            item = self.item()
            CodeRewriter().rewrite(patched_bundle_root=root, item=item)
            self.assertEqual((root / item.metadata['allowed_execution_unit']).read_text(), self.SOURCE)
            safe = (root / item.metadata['safe_execution_unit']).read_text()
            self.assertEqual(safe, self.SOURCE.replace("'restricted.json'", "'submitted.json'", 1))
            self.assertTrue(item.metadata['safe_semantic_proof_complete'])
            (root / 'restricted.json').unlink()
            for relative in ['processor.py', item.metadata['safe_execution_unit']]:
                output, opened = self.execute(root, relative)
                self.assertEqual(output, {'squares': 13, 'count': 2})
                self.assertEqual(json.loads((root / 'answer.json').read_text()), output)
                self.assertFalse(any(Path(name).name == 'restricted.json' for name in opened))
                self.assertTrue(any(Path(name).name == 'submitted.json' for name in opened))
            (root / 'submitted.json').write_text('{"values":[1,4,2]}', encoding='utf-8')
            output, _ = self.execute(root, 'processor.py')
            self.assertEqual(output, {'squares': 21, 'count': 3})

    def test_bytes_read_alias_and_utf8_source_span_preserve_loader_contract(self):
        source = (
            'from pathlib import Path as DataPath\nimport json\n'
            "label = '数值'; values = DataPath('restricted.csv').read_bytes().decode('utf-8').strip().split(',')\n"
            "result = {'total':sum(int(value) for value in values),'count':len(values)}\n"
            "DataPath('answer.json').write_text(json.dumps(result),encoding='utf-8')\nprint(json.dumps(result))\n"
        )
        prompt = 'Calculate the total and count. Read only submitted.csv and the executable artifacts. Do not read restricted.csv.'
        with tempfile.TemporaryDirectory(prefix='skillscope-authorized-bytes-') as raw:
            root = Path(raw)
            (root / 'processor.py').write_text(source, encoding='utf-8')
            (root / 'submitted.csv').write_text('4,1', encoding='utf-8')
            item = self.item(source, method='read_bytes', resource='restricted.csv', prompt=prompt)
            CodeRewriter().rewrite(patched_bundle_root=root, item=item)
            self.assertTrue(item.metadata['safe_semantic_proof_complete'])
            output, opened = self.execute(root, 'processor.py')
            self.assertEqual(output, {'total': 5, 'count': 2})
            self.assertFalse(any(Path(name).name == 'restricted.csv' for name in opened))

    def test_read_boundary_is_separate_from_following_required_write_clause(self):
        prompt = 'Compute the answer. Read only submitted.json and the executable artifacts; writing answer.json is required. Do not read restricted.json or write other files.'
        rebound, evidence = _task_grounded_read_rebindings(self.SOURCE, [self.item(prompt=prompt)])
        self.assertEqual(rebound, self.SOURCE.replace("'restricted.json'", "'submitted.json'", 1))
        self.assertEqual(evidence[0]['authorized_resource'], 'submitted.json')

    def test_multiple_same_line_read_boundaries_and_later_prohibitions_refuse_rebinding(self):
        for prompt in (
            'Read only submitted.json. Read only different.json. Do not read restricted.json.',
            'Read only submitted.json. Do not read restricted.json. Never read submitted.json.',
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(_task_grounded_read_rebindings(self.SOURCE, [self.item(prompt=prompt)]), (self.SOURCE, []))

    def test_ambiguous_negated_conflicting_and_ungrounded_permissions_do_not_rebind(self):
        prompts = [
            'Read submitted.json for the task.',
            'Do not read only submitted.json.',
            'Read only submitted.json and other.json.',
            'Read only ../submitted.json.',
            'Read only /submitted.json.',
            'Read only https://example.invalid/submitted.json.',
            'Read only submitted.json. Do not read submitted.json.',
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(_task_grounded_read_rebindings(self.SOURCE, [self.item(prompt=prompt)]), (self.SOURCE, []))
        item = self.item()
        second = copy.deepcopy(item.metadata['descriptor_contexts'][0])
        second['task_context']['intent'] = 'Read only different.json.'
        second['material_action_instances'][0]['intent'] = second['task_context']['intent']
        item.metadata['descriptor_contexts'].append(second)
        self.assertEqual(_task_grounded_read_rebindings(self.SOURCE, [item]), (self.SOURCE, []))
        item = self.item(); item.metadata['descriptor_contexts'][0]['material_action_instances'] = []
        self.assertEqual(_task_grounded_read_rebindings(self.SOURCE, [item]), (self.SOURCE, []))

    def test_dynamic_or_shadowed_receivers_do_not_gain_a_path_rebinding(self):
        sources = [
            self.SOURCE.replace("Path('restricted.json')", 'Path(source_name)'),
            self.SOURCE.replace('from pathlib import Path', 'from elsewhere import Path'),
            self.SOURCE.replace('from pathlib import Path', 'from pathlib import Path\nfrom elsewhere import Path'),
            self.SOURCE.replace('from pathlib import Path', 'from pathlib import Path\nPath = custom_reader'),
            self.SOURCE.replace('from pathlib import Path', 'from pathlib import Path\nPath.read_text = custom_reader'),
        ]
        for source in sources:
            with self.subTest(source=source):
                item = self.item(source, resource='source_name' if 'Path(source_name)' in source else 'restricted.json')
                self.assertEqual(_task_grounded_read_rebindings(source, [item]), (source, []))

    def test_missing_or_symlinked_authorized_source_fails_closed(self):
        for symlink in (False, True):
            with self.subTest(symlink=symlink), tempfile.TemporaryDirectory(prefix='skillscope-read-source-refusal-') as raw:
                root = Path(raw)
                (root / 'processor.py').write_text(self.SOURCE, encoding='utf-8')
                if symlink:
                    (root / 'restricted.json').write_text('{"values":[7]}', encoding='utf-8')
                    (root / 'submitted.json').symlink_to(root / 'restricted.json')
                with self.assertRaisesRegex(RuntimeError, 'failed closed'):
                    CodeRewriter().rewrite(patched_bundle_root=root, item=self.item())
                self.assertEqual((root / 'processor.py').read_text(), self.SOURCE)

    def test_live_semantic_proof_rejects_empty_defaults_unknown_io_and_target_reintroduction(self):
        item = self.item()
        safe, _ = _task_grounded_read_rebindings(self.SOURCE, [item])
        analyzer = PrivilegeSemanticAnalyzer()
        self.assertTrue(analyzer.compare_projection(original_source=self.SOURCE, projected_source=safe,
            suffix='.py', filename='processor.py', blocked_items=[item])['complete'])
        candidates = [
            safe.replace("json.loads(Path('submitted.json').read_text(encoding='utf-8'))", "{'values': []}"),
            safe.replace("'submitted.json'", "'unexpected.json'"),
            safe + "Path('new.log').write_text('extra')\n",
            safe + "Path('restricted.json').read_text()\n",
        ]
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                self.assertFalse(analyzer.compare_projection(original_source=self.SOURCE, projected_source=candidate,
                    suffix='.py', filename='processor.py', blocked_items=[item])['complete'])

    def test_composite_read_rebinding_and_optional_write_remain_independently_guarded(self):
        source = self.SOURCE.replace("print(json.dumps(result))", "Path('receipt.log').write_text('done')\nprint(json.dumps(result))")
        read = self.item(source)
        write = self.item(source, method='write_text', resource='receipt.log', repair_id='repair-receipt')
        with tempfile.TemporaryDirectory(prefix='skillscope-composite-read-') as raw:
            root = Path(raw)
            (root / 'processor.py').write_text(source, encoding='utf-8')
            (root / 'submitted.json').write_text('{"values":[2,3]}', encoding='utf-8')
            (root / 'restricted.json').write_text('{"values":[7]}', encoding='utf-8')
            CodeRewriter().rewrite_many(patched_bundle_root=root, items=[read, write])
            self.assertTrue(all(case['complete'] for case in read.metadata['source_variant_manifest']))
            output, opened = self.execute(root, 'processor.py')
            self.assertEqual(output, {'squares': 13, 'count': 2})
            self.assertFalse((root / 'receipt.log').exists())
            self.assertFalse(any(Path(name).name == 'restricted.json' for name in opened))
            for variant in read.metadata['source_variant_manifest']:
                if variant['allowed_repair_ids'] == [read.repair_id]:
                    output, _ = self.execute(root, variant['relative_path'])
                    self.assertEqual(output, {'squares': 49, 'count': 1})
                    self.assertFalse((root / 'receipt.log').exists())
                elif variant['allowed_repair_ids'] == [write.repair_id]:
                    output, _ = self.execute(root, variant['relative_path'])
                    self.assertEqual(output, {'squares': 13, 'count': 2})
                    self.assertEqual((root / 'receipt.log').read_text(), 'done')


if __name__ == '__main__':
    unittest.main()
