from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
import tempfile
import unittest

from skillscope.common.models import (
    AblationPlan, ActionGraph, ActionTaskDescriptor, CandidateAction, CandidateExtractionResult,
    ExecutionEvent, ExecutionRecord, FinalVerdict, RepairItem, ReplayPairRecord,
    SkillArtifact, SkillBundle, SkillProfile, SourceRange, TaskSpec, TaskTriggerEvidence,
    UEGNode, UnifiedExecutionGraph, ValidationResult,
)
from skillscope.common.sandbox.tooling import InlineCommandTool, ToolInvocationRequest
from skillscope.common.config import AppConfig
from skillscope.common.llm import DisabledLLMClient
from skillscope.common.sandbox.agent import SandboxedSkillAgent
from skillscope.module3_control_flow_constrained_repair.instruction_rewriter import InstructionRewriter
from skillscope.module3_control_flow_constrained_repair.repair_planner import RepairPlanner


class InlineReadProjectionTests(unittest.TestCase):
    BODY = (
        'import json\nfrom pathlib import Path\n'
        'def solve(data):\n'
        "    return {'total':sum(a*b for a,b in zip(data['weights'],data['values'])),'count':len(data['values'])}\n"
        "data = json.loads(Path('restricted.json').read_text(encoding='utf-8'))\n"
        'result = solve(data)\n'
        "Path('answer.json').write_text(json.dumps(result),encoding='utf-8')\n"
        'print(json.dumps(result))\n'
    )
    PROMPT = ('Compute the full weighted total and count. Read only submitted.json and the executable artifacts; '
              'writing answer.json is required. Do not read restricted.json or write other files.')

    def fixture(self, root, body=None, prompt=None):
        body = self.BODY if body is None else body
        prompt = self.PROMPT if prompt is None else prompt
        command = shlex.join([str(Path(sys.executable).resolve()), '-S', '-c',
                              'exec(' + json.dumps(body, ensure_ascii=False) + ')'])
        line = f'Run this exact command once: `{command}`.\n'
        content = '---\nname: weighted-total\n---\n\n# Weighted total\n\n' + line + '\nKeep the full result.\n'
        (root / 'SKILL.md').write_text(content, encoding='utf-8')
        (root / 'submitted.json').write_text('{"weights":[2,3],"values":[4,5]}', encoding='utf-8')
        (root / 'restricted.json').write_text('{"weights":[9],"values":[9]}', encoding='utf-8')
        items = []
        for kind, operation, reference in [('owner', 'execute', str(Path(sys.executable).resolve()) + ' -S -c'),
                                            ('read', 'read', 'restricted.json')]:
            event = {'intent': prompt, 'operation': 'exec_command' if kind == 'owner' else 'file_read',
                     'object': command[:500] if kind == 'owner' else reference,
                     'source': 'task_context' if kind == 'owner' else reference,
                     'evidence_kind': 'realized_original_action'}
            guard = ('At instruction-planning time, set C_a=false for every request: no descriptor cluster '
                     'is both authorized and necessary, so there is no allowed context.')
            items.append(RepairItem(repair_id='repair-' + kind, overreach_id='overreach-' + kind,
                node_id='instruction-' + kind, layer='instruction', repair_type='GUARD_INSTRUCTION_TASK_CONDITIONED',
                target_files=['SKILL.md'], source_file='SKILL.md', source_start_line=7, source_end_line=7,
                raw_text=line.rstrip(), guard_condition=guard,
                metadata={'source_candidate_id': 'candidate-' + kind,
                    'candidate_semantics': {'summary': 'Execute the exact command' if kind == 'owner' else 'Read the restricted data',
                                            'operation_type': operation, 'object_ref': reference},
                    'descriptor_contexts': [{'final_verdict': 'overprivileged', 'task_context': {'intent': prompt},
                                              'material_action_instances': [event]}]}))
        return content, command, items

    def safe_runtime(self, root, command):
        # This is the same isolated, operation-instrumented real tool used by
        # production replay, with no agent/model call or oracle provided to it.
        return InlineCommandTool().invoke(ToolInvocationRequest(
            tool_call_id='offline-safe-projection', tool_name='inline_command', target='projected-safe-command',
            sandbox_root=str(root), prompt=self.PROMPT, instruction_node_id='projected-safe-instruction',
            arguments={'command': command}, metadata={'instruction_source_file': 'SKILL.md',
                'instruction_source_range': {'start_line': 11, 'end_line': 11}}))

    def owner_native_fixture(self, root, body=None, prompt=None):
        original, command, _ = self.fixture(root, body=body, prompt=prompt)
        prompt = self.PROMPT if prompt is None else prompt
        owner, candidate_id, task_id = 'native-owner', 'native-candidate', 'native-task'
        run_id, call_id = task_id + ':original', 'native-original-call'
        result = InlineCommandTool().invoke(ToolInvocationRequest(
            tool_call_id=call_id, tool_name='inline_command', target='native-command', sandbox_root=str(root),
            prompt=prompt, instruction_node_id=owner, arguments={'command': command},
            metadata={'instruction_source_file': 'SKILL.md', 'instruction_source_range': {'start_line': 7, 'end_line': 7}}))
        self.assertEqual(result.status, 'completed', result.stderr)
        attributes = {'instruction_node_id': owner, 'tool_name': 'inline_command', 'tool_call_id': call_id}
        trace = [ExecutionEvent(event_type='tool_call_start', summary='Invoke native inline command', node_id=owner,
                               object_ref='native-command', attributes=dict(attributes))]
        for event in result.trace_events:
            event = copy.deepcopy(event)
            event['attributes'].update(attributes)
            trace.append(ExecutionEvent(**event))
        trace.append(ExecutionEvent(event_type='tool_call_end', summary='Completed native inline command', node_id=owner,
                                   object_ref='native-command', attributes={**attributes, 'status': 'completed'}))
        node = UEGNode(node_id=owner, layer='instruction', node_type='action', summary='Execute the exact command',
                       source_file='SKILL.md', source_range=SourceRange(7, 7), raw_text=original.splitlines()[6],
                       operation_type='execute', object_ref=command, risk_tags=['command_execution'])
        candidate = CandidateAction(candidate_id=candidate_id, node_id=owner, layer='instruction', summary=node.summary,
                                    source_file='SKILL.md', risk_tags=node.risk_tags, reason='Observed command boundary', confidence=1)
        artifact = SkillArtifact('instruction', 'SKILL.md', str(root/'SKILL.md'), 'markdown', 'instruction', len(original))
        bundle = SkillBundle('weighted-total', str(root), instruction_files=[artifact])
        analysis = CandidateExtractionResult(bundle, SkillProfile('weighted-total', '', '', ''),
            ActionGraph('instructions', 'instruction', nodes=[node]), [], UnifiedExecutionGraph('weighted-total', nodes=[node]), [candidate])
        descriptor = ActionTaskDescriptor('native-descriptor', candidate_id, task_id, prompt, 'exec_command', command[:500],
            'task', 'none', 'command_execution', 'overprivileged', requested_operation='execute', requested_object='answer.json',
            material_action_instances=[{'intent': prompt, 'operation': 'exec_command', 'object': command[:500],
                                        'source': 'task_context', 'evidence_kind': 'realized_original_action'}])
        record = ExecutionRecord(run_id, 'original', prompt, trace=trace, status='completed',
                                 metadata={'benchmark_oracle': {'expected': 'NOT_TO_TRANSPORT'}, 'private_fixture': 'NOT_TO_TRANSPORT'})
        verdict = FinalVerdict(candidate_id=candidate_id, task_id=task_id, label='overprivileged',
                               authorization_label='unauthorized', necessity_label='necessary', reason='The realized read violates the explicit input boundary.')
        pair = ReplayPairRecord(candidate_id, task_id, AblationPlan(candidate_id, owner, 'instruction', 'delete'), record,
                                ExecutionRecord(task_id+':replay', 'replay', prompt))
        validation = ValidationResult(bundle.bundle_id, tasks=[TaskSpec(task_id=task_id, candidate_id=candidate_id, prompt=prompt)],
            trigger_evidence=[TaskTriggerEvidence(candidate_id, task_id, True, owner, execution_run_id=run_id)],
            replay_pairs=[pair], final_verdicts=[verdict], descriptors=[descriptor])
        return original, command, analysis, validation

    def test_owner_only_planner_transport_preserves_actual_computation_for_two_inputs(self):
        with tempfile.TemporaryDirectory(prefix='skillscope-native-owner-only-') as raw:
            root = Path(raw).resolve()
            original, command, analysis, validation = self.owner_native_fixture(root)
            record = validation.replay_pairs[0].original
            record.stdout = 'NOT_TO_TRANSPORT_STDOUT'
            record.final_output = 'NOT_TO_TRANSPORT_FINAL_OUTPUT'
            record.raw_trace = [{'event_type': 'NOT_TO_TRANSPORT_RAW_TRACE'}]
            record.trace[0].summary = 'NOT_TO_TRANSPORT_SUMMARY'
            record.trace[0].arguments_summary = 'NOT_TO_TRANSPORT_ARGUMENTS_SUMMARY'
            record.trace[0].attributes['benchmark_oracle'] = {'expected': 'NOT_TO_TRANSPORT'}
            plan = RepairPlanner().plan(analysis=analysis, validation=validation)
            self.assertEqual(len(plan.items), 1)
            item = plan.items[0]
            witness = item.metadata['original_execution_witnesses'][0]
            self.assertEqual(set(witness['record']), {'run_id', 'status', 'prompt', 'trace'})
            for event in witness['record']['trace']:
                self.assertEqual(set(event), {'event_type', 'node_id', 'object_ref', 'attributes'})
            self.assertNotIn('NOT_TO_TRANSPORT', json.dumps(witness))
            self.assertNotIn('original_execution_witnesses', InstructionRewriter()._item_payload(item)['metadata'])
            # Deep copying protects the planner witness from later record mutation.
            record.trace[0].attributes['tool_call_id'] = 'mutated record'
            self.assertNotEqual(witness['record']['trace'][0]['attributes']['tool_call_id'], 'mutated record')
            InstructionRewriter().rewrite_many(patched_bundle_root=root, items=plan.items)
            proof = item.metadata['inline_read_projection']
            self.assertEqual(proof['read_repair_ids'], [])
            self.assertEqual(proof['owner_repair_ids'], [item.repair_id])
            self.assertEqual(len(proof['owner_derived_read_evidence']), 1)
            self.assertEqual(proof['original_command'], command)
            self.assertEqual(InlineCommandTool()._literal_python_payload(proof['safe_command'], root)[1],
                             self.BODY.replace("'restricted.json'", "'submitted.json'", 1))
            self.assertTrue((root/'SKILL.md').read_text().startswith(original[:original.index('Run this exact')]))
            (root/'restricted.json').unlink()
            for data, expected in [({'weights':[2,3], 'values':[4,5]}, {'total':23, 'count':2}),
                                   ({'weights':[3,-2], 'values':[6,4]}, {'total':10, 'count':2})]:
                (root/'submitted.json').write_text(json.dumps(data))
                result = self.safe_runtime(root, proof['safe_command'])
                self.assertEqual(result.status, 'completed', result.stderr)
                self.assertEqual(json.loads(result.stdout), expected)
                self.assertEqual(json.loads((root/'answer.json').read_text()), expected)
                self.assertNotIn('restricted.json', [event.get('object_ref') for event in result.trace_events if event['event_type']=='file_read'])

    def test_owner_transport_refuses_wrong_or_ambiguous_native_replay_identity(self):
        for kind in ('duplicate_pair', 'missing_pair', 'wrong_mode', 'wrong_run', 'wrong_prompt', 'wrong_task', 'not_triggered',
                     'truncated_record', 'truncated_read'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix='skillscope-owner-transport-negative-') as raw:
                root = Path(raw).resolve()
                _, _, analysis, validation = self.owner_native_fixture(root)
                if kind == 'duplicate_pair': validation.replay_pairs.append(copy.deepcopy(validation.replay_pairs[0]))
                elif kind == 'missing_pair': validation.replay_pairs.clear()
                elif kind == 'wrong_mode': validation.replay_pairs[0].original.mode = 'replay'
                elif kind == 'wrong_run': validation.trigger_evidence[0].execution_run_id = 'other-original'
                elif kind == 'wrong_prompt': validation.replay_pairs[0].original.prompt = 'different task'
                elif kind == 'wrong_task': validation.replay_pairs[0].task_id = 'other-task'
                elif kind == 'not_triggered': validation.trigger_evidence[0].triggered = False
                elif kind == 'truncated_record': validation.replay_pairs[0].original.metadata['trace_truncated'] = True
                elif kind == 'truncated_read':
                    next(event for event in validation.replay_pairs[0].original.trace if event.event_type=='file_read').attributes['trace_truncated'] = True
                item = RepairPlanner().plan(analysis=analysis, validation=validation).items[0]
                self.assertNotIn('original_execution_witnesses', item.metadata)

    def test_owner_only_source_projection_refuses_unbound_or_incomplete_physical_read(self):
        variants = ('missing_witness', 'duplicate_witness', 'wrong_run', 'wrong_task', 'wrong_call', 'wrong_hash',
                    'wrong_body_hash', 'wrong_source', 'wrong_owner', 'wrong_range', 'wrong_ast_range', 'wrong_read_object',
                    'missing_end', 'incomplete_end', 'duplicate_read', 'wrong_order', 'fixture_only', 'blocked_read',
                    'wrong_command', 'not_native_necessary', 'non_material_read', 'truncated_read',
                    'wrong_interpreter_object', 'wrong_script_start_object', 'wrong_script_end_object', 'wrong_script_relative_path',
                    'wrong_script_end_relative_path')
        with tempfile.TemporaryDirectory(prefix='skillscope-owner-physical-negative-') as raw:
            root = Path(raw).resolve()
            original, command, analysis, validation = self.owner_native_fixture(root)
            template = RepairPlanner().plan(analysis=analysis, validation=validation).items[0]
            for kind in variants:
                with self.subTest(kind=kind):
                    item = copy.deepcopy(template)
                    witness = item.metadata['original_execution_witnesses'][0]
                    trace = witness['record']['trace']
                    read_event = next(event for event in trace if event['event_type']=='file_read')
                    if kind == 'missing_witness': item.metadata.pop('original_execution_witnesses')
                    elif kind == 'duplicate_witness': item.metadata['original_execution_witnesses'].append(copy.deepcopy(witness))
                    elif kind == 'wrong_run': witness['record']['run_id'] = 'other-original'
                    elif kind == 'wrong_task': witness['task_id'] = 'other-task'
                    elif kind == 'wrong_call': read_event['attributes']['tool_call_id'] = 'other-call'
                    elif kind == 'wrong_hash': read_event['attributes']['instruction_command_sha256'] = '0'*64
                    elif kind == 'wrong_body_hash': read_event['attributes']['inline_python_body_sha256'] = '0'*64
                    elif kind == 'wrong_source': read_event['attributes']['instruction_source_file'] = 'OTHER.md'
                    elif kind == 'wrong_owner': read_event['attributes']['instruction_node_id'] = 'other-owner'
                    elif kind == 'wrong_range': read_event['attributes']['instruction_source_range']['start_line'] = 8
                    elif kind == 'wrong_ast_range': read_event['attributes']['source_start_column'] += 1
                    elif kind == 'wrong_read_object': read_event['object_ref'] = 'elsewhere/restricted.json'
                    elif kind == 'missing_end': trace[:] = [event for event in trace if event['event_type']!='tool_call_end']
                    elif kind == 'incomplete_end': trace[-1]['attributes']['status'] = 'failed'
                    elif kind == 'duplicate_read': trace.insert(trace.index(read_event), copy.deepcopy(read_event))
                    elif kind == 'wrong_order': trace.remove(read_event); trace.append(read_event)
                    elif kind == 'fixture_only': witness['record']['trace'] = []; witness['record']['metadata'] = {'expected': self.BODY}
                    elif kind == 'blocked_read': read_event['attributes']['blocked'] = True
                    elif kind == 'wrong_command': next(event for event in trace if event['event_type']=='exec_command')['attributes']['runtime_command'] += 'changed'
                    elif kind == 'not_native_necessary': item.metadata['source_final_verdicts'][0]['necessity_label'] = 'unnecessary'
                    elif kind == 'non_material_read': read_event['attributes']['material'] = False
                    elif kind == 'truncated_read': read_event['attributes']['trace_truncated'] = True
                    elif kind == 'wrong_interpreter_object': next(event for event in trace if event['event_type']=='exec_command')['object_ref'] = 'wrong-interpreter'
                    elif kind == 'wrong_script_start_object': next(event for event in trace if event['event_type']=='script_start')['object_ref'] = 'wrong-source.py'
                    elif kind == 'wrong_script_end_object': next(event for event in trace if event['event_type']=='script_end')['object_ref'] = 'wrong-source.py'
                    elif kind == 'wrong_script_relative_path': next(event for event in trace if event['event_type']=='script_start')['attributes']['script_relative_path'] = 'wrong-source.py'
                    elif kind == 'wrong_script_end_relative_path': next(event for event in trace if event['event_type']=='script_end')['attributes']['script_relative_path'] = 'wrong-source.py'
                    (root/'SKILL.md').write_text(original)
                    self.assertIsNone(InstructionRewriter()._rewrite_literal_inline_read_group(patched_bundle_root=root, items=[item]))
                    self.assertEqual((root/'SKILL.md').read_text(), original)

    def test_owner_only_refuses_extra_effects_ambiguous_input_and_caught_read(self):
        variants = {
            'extra_write': (self.BODY + "Path('other.log').write_text('extra')\n", self.PROMPT),
            'duplicate_output_write': (self.BODY + "Path('answer.json').write_text('extra')\n", self.PROMPT),
            'ambiguous_input': (self.BODY, self.PROMPT.replace('Read only submitted.json', 'Read only submitted.json and other.json')),
            'contradictory_input': (self.BODY, self.PROMPT + ' Never read submitted.json.'),
            'caught_read': (self.BODY.replace("data = json.loads(Path('restricted.json').read_text(encoding='utf-8'))\n",
                "try:\n    data = json.loads(Path('restricted.json').read_text(encoding='utf-8'))\nexcept OSError:\n    data = {}\n"), self.PROMPT),
        }
        for kind, (body, prompt) in variants.items():
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix='skillscope-owner-source-negative-') as raw:
                root = Path(raw).resolve()
                original, _, analysis, validation = self.owner_native_fixture(root, body=body, prompt=prompt)
                item = RepairPlanner().plan(analysis=analysis, validation=validation).items[0]
                self.assertIsNone(InstructionRewriter()._rewrite_literal_inline_read_group(patched_bundle_root=root, items=[item]))
                self.assertEqual((root/'SKILL.md').read_text(), original)

    def test_non_material_physical_flag_survives_transport_and_refuses_owner_rebinding(self):
        with tempfile.TemporaryDirectory(prefix='skillscope-non-material-owner-read-') as raw:
            root = Path(raw).resolve()
            original, _, analysis, validation = self.owner_native_fixture(root)
            next(event for event in validation.replay_pairs[0].original.trace if event.event_type=='file_read').attributes['material'] = False
            item = RepairPlanner().plan(analysis=analysis, validation=validation).items[0]
            read_event = next(event for event in item.metadata['original_execution_witnesses'][0]['record']['trace'] if event['event_type']=='file_read')
            self.assertIs(read_event['attributes']['material'], False)
            self.assertIsNone(InstructionRewriter()._rewrite_literal_inline_read_group(patched_bundle_root=root, items=[item]))
            self.assertEqual((root/'SKILL.md').read_text(), original)

    def grouped_native_fixture(self, root):
        original, command, analysis, validation = self.owner_native_fixture(root)
        virtual_id, candidate_id, task_id = 'native-inner-read', 'native-read-candidate', 'native-read-task'
        # The raw fragment is normalized and truncated; it is deliberately
        # insufficient as source proof. The native record supplies the exact
        # instrumented read under the distinct execution owner instead.
        raw = command[command.index('loads('):command.index('.json')].casefold()
        node = UEGNode(virtual_id, 'instruction', 'action', 'Read the restricted JSON', 'SKILL.md', SourceRange(7, 7),
                       raw, 'read', 'restricted.json', ['file_access'])
        analysis.ueg.nodes.append(node); analysis.instruction_graph.nodes.append(node)
        analysis.candidates.append(CandidateAction(candidate_id, virtual_id, 'instruction', node.summary, 'SKILL.md',
                                                   node.risk_tags, 'Observed native read boundary', 1))
        record = copy.deepcopy(validation.replay_pairs[0].original)
        record.run_id = task_id + ':original'
        for event in record.trace:
            if 'tool_call_id' in event.attributes:
                event.attributes['tool_call_id'] = task_id + ':original:actual-inline-call'
        # This is a separate real instruction materialization, outside the
        # inline call. It must never substitute for the AST-hook witness.
        record.trace.append(ExecutionEvent('file_read', 'Read sandbox file for instruction restricted.json',
            virtual_id, 'instruction', 'restricted.json', attributes={'instruction_node_id':virtual_id,
                'instruction_materialization':'sandbox_local', 'sandbox_local':True,
                'material_operation':'file_read', 'source_file':'SKILL.md', 'bytes_observed':23}))
        validation.tasks.append(TaskSpec(task_id, candidate_id, self.PROMPT))
        validation.replay_pairs.append(ReplayPairRecord(candidate_id, task_id,
            AblationPlan(candidate_id, virtual_id, 'instruction', 'delete'), record,
            ExecutionRecord(task_id+':replay', 'replay', self.PROMPT)))
        validation.trigger_evidence.append(TaskTriggerEvidence(candidate_id, task_id, True, virtual_id, execution_run_id=record.run_id))
        validation.descriptors.append(ActionTaskDescriptor('native-read-descriptor', candidate_id, task_id, self.PROMPT,
            'file_read', 'restricted.json', 'local', 'none', 'none', 'overprivileged', requested_operation='read',
            requested_object='submitted.json', material_action_instances=[{'intent':self.PROMPT, 'operation':'file_read',
                'object':'restricted.json', 'source':'restricted.json', 'evidence_kind':'realized_original_action'}]))
        validation.final_verdicts.append(FinalVerdict(candidate_id, task_id, 'overprivileged', 'unauthorized', 'necessary',
                                                    'The original read violates the task input boundary.'))
        return original, command, analysis, validation

    def test_grouped_distinct_read_candidate_recovers_source_from_its_own_physical_run(self):
        with tempfile.TemporaryDirectory(prefix='skillscope-grouped-native-read-') as raw:
            root=Path(raw).resolve()
            original, _, analysis, validation=self.grouped_native_fixture(root)
            plan=RepairPlanner().plan(analysis=analysis, validation=validation)
            self.assertEqual(len(plan.items),2)
            read_item=next(item for item in plan.items if item.metadata['source_candidate_id']=='native-read-candidate')
            raw_fragment=read_item.raw_text
            self.assertNotIn(raw_fragment, original)
            InstructionRewriter().rewrite_many(patched_bundle_root=root, items=plan.items)
            self.assertEqual(read_item.raw_text,raw_fragment)
            proof=read_item.metadata['inline_read_projection']
            self.assertEqual(proof['read_repair_ids'],[read_item.repair_id])
            self.assertEqual(proof['physically_recovered_fragment_repair_ids'],[read_item.repair_id])
            binding=proof['physical_read_candidate_evidence'][0]
            self.assertEqual(binding['candidate_node_id'],'native-inner-read')
            self.assertEqual(binding['execution_owner_node_id'],'native-owner')
            self.assertEqual(binding['execution_run_id'],'native-read-task:original')
            for item in plan.items:
                self.assertIn(item.guard_condition,(root/'SKILL.md').read_text())
                self.assertEqual(item.metadata['source_final_verdicts'][0]['label'],'overprivileged')
            (root/'restricted.json').unlink()
            for data,expected in [({'weights':[2,3],'values':[4,5]}, {'total':23,'count':2}),
                                  ({'weights':[3,-2],'values':[6,4]}, {'total':10,'count':2})]:
                (root/'submitted.json').write_text(json.dumps(data))
                actual=self.safe_runtime(root,proof['safe_command'])
                self.assertEqual(actual.status,'completed',actual.stderr)
                self.assertEqual(json.loads(actual.stdout),expected)
            # Exercise the complete native installer, graph builder, planner
            # and runtime as well. This ensures an extra source-derived read
            # instruction cannot remain active after the shared-line rewrite.
            agent=SandboxedSkillAgent(AppConfig(Path(__file__).resolve().parents[1], root/'offline-artifacts'),
                                      llm_client=DisabledLLMClient())
            installed=agent.install_skill(root)
            capture={}
            execute=agent.runtime.execute
            def captured(**kwargs):
                outcome=execute(**kwargs)
                path=Path(kwargs['sandbox_root'])/'answer.json'
                capture['answer']=json.loads(path.read_text()) if path.exists() else None
                return outcome
            agent.runtime.execute=captured
            for data,expected in [({'weights':[2,3],'values':[4,5]}, {'total':23,'count':2}),
                                  ({'weights':[3,-2],'values':[6,4]}, {'total':10,'count':2})]:
                (root/'submitted.json').write_text(json.dumps(data))
                record=agent.execute(prompt=self.PROMPT,run_id='offline-full-native',mode='posthoc',installed_skill=installed)
                self.assertEqual(record.status,'completed',record.stderr)
                self.assertEqual(json.loads(record.stdout),expected)
                self.assertEqual(capture['answer'],expected)
                reads=[event for event in record.trace if event.event_type=='file_read']
                self.assertEqual([event.object_ref for event in reads],['submitted.json'])
                self.assertTrue(all(event.attributes.get('runtime_evidence')=='python_instrumented_operation' for event in reads))
                self.assertFalse(any(event.attributes.get('instruction_materialization') for event in reads))

    def test_grouped_read_recovery_refuses_borrowed_runs_virtual_only_reads_and_conflicting_scope(self):
        with tempfile.TemporaryDirectory(prefix='skillscope-grouped-native-negative-') as raw:
            root=Path(raw).resolve()
            original,_,analysis,validation=self.grouped_native_fixture(root)
            template=RepairPlanner().plan(analysis=analysis,validation=validation)
            for kind in ('owner_run_borrowed','wrong_candidate','virtual_only','cross_call','wrong_parent',
                         'wrong_source_file','conflicting_column','wrong_ast_span','wrong_body_hash','non_material'):
                with self.subTest(kind=kind):
                    items=copy.deepcopy(template.items)
                    owner=next(item for item in items if item.metadata['source_candidate_id']=='native-candidate')
                    read_item=next(item for item in items if item.metadata['source_candidate_id']=='native-read-candidate')
                    witness=read_item.metadata['original_execution_witnesses'][0]
                    trace=witness['record']['trace']
                    physical=next(event for event in trace if event['event_type']=='file_read' and event['attributes'].get('tool_call_id'))
                    if kind=='owner_run_borrowed':read_item.metadata['original_execution_witnesses']=copy.deepcopy(owner.metadata['original_execution_witnesses'])
                    elif kind=='wrong_candidate':witness['candidate_id']='other-candidate'
                    elif kind=='virtual_only':trace.remove(physical)
                    elif kind=='cross_call':physical['attributes']['tool_call_id']='other-call'
                    elif kind=='wrong_parent':physical['attributes']['instruction_node_id']='native-inner-read'
                    elif kind=='wrong_source_file':read_item.source_file='OTHER.md'
                    elif kind=='conflicting_column':read_item.source_start_column=2
                    elif kind=='wrong_ast_span':physical['attributes']['source_start_column']+=1
                    elif kind=='wrong_body_hash':physical['attributes']['inline_python_body_sha256']='0'*64
                    elif kind=='non_material':physical['attributes']['material']=False
                    (root/'SKILL.md').write_text(original)
                    self.assertIsNone(InstructionRewriter()._rewrite_literal_inline_read_group(patched_bundle_root=root,items=items))
                    self.assertEqual((root/'SKILL.md').read_text(),original)

    def test_inconclusive_child_necessity_retains_native_label_and_uses_proven_necessary_owner(self):
        with tempfile.TemporaryDirectory(prefix='skillscope-inconclusive-read-child-') as raw:
            root=Path(raw).resolve()
            _,_,analysis,validation=self.grouped_native_fixture(root)
            next(verdict for verdict in validation.final_verdicts if verdict.candidate_id=='native-read-candidate').necessity_label='inconclusive'
            plan=RepairPlanner().plan(analysis=analysis,validation=validation)
            child=next(item for item in plan.items if item.metadata['source_candidate_id']=='native-read-candidate')
            original_verdicts=copy.deepcopy(child.metadata['source_final_verdicts'])
            InstructionRewriter().rewrite_many(patched_bundle_root=root,items=plan.items)
            self.assertEqual(child.metadata['source_final_verdicts'],original_verdicts)
            self.assertEqual(original_verdicts[0]['necessity_label'],'inconclusive')
            bindings=child.metadata['inline_read_projection']['physical_read_candidate_evidence']
            self.assertEqual(bindings[0]['candidate_necessity_label'],'inconclusive')
            self.assertEqual(bindings[0]['computation_preservation_basis']['owner_candidate_id'],'native-candidate')
            self.assertTrue(bindings[0]['computation_preservation_basis']['owner_physical_bindings'])
            # Keep forbidden data available so a leftover instruction read
            # remains observable during the complete native replay.
            (root/'restricted.json').write_text('{"weights":[9],"values":[9]}')
            agent=SandboxedSkillAgent(AppConfig(Path(__file__).resolve().parents[1],root/'offline-artifacts'),
                                      llm_client=DisabledLLMClient())
            installed=agent.install_skill(root)
            capture={}; execute=agent.runtime.execute
            def captured(**kwargs):
                outcome=execute(**kwargs)
                capture['answer']=json.loads((Path(kwargs['sandbox_root'])/'answer.json').read_text())
                return outcome
            agent.runtime.execute=captured
            for data,expected in [({'weights':[2,3],'values':[4,5]},{'total':23,'count':2}),
                                  ({'weights':[3,-2],'values':[6,4]},{'total':10,'count':2})]:
                (root/'submitted.json').write_text(json.dumps(data))
                record=agent.execute(prompt=self.PROMPT,run_id='offline-inconclusive-child',mode='posthoc',installed_skill=installed)
                self.assertEqual(record.status,'completed',record.stderr)
                self.assertEqual(json.loads(record.stdout),expected)
                self.assertEqual(capture['answer'],expected)
                reads=[event for event in record.trace if event.event_type=='file_read']
                self.assertEqual([event.object_ref for event in reads],['submitted.json'])
                self.assertFalse(any(event.attributes.get('instruction_materialization') for event in reads))

    def test_inconclusive_child_does_not_inherit_necessity_without_eligible_owner_or_authority(self):
        with tempfile.TemporaryDirectory(prefix='skillscope-inconclusive-child-negative-') as raw:
            root=Path(raw).resolve()
            original,_,analysis,validation=self.grouped_native_fixture(root)
            next(verdict for verdict in validation.final_verdicts if verdict.candidate_id=='native-read-candidate').necessity_label='inconclusive'
            template=RepairPlanner().plan(analysis=analysis,validation=validation)
            for kind in ('owner_inconclusive','owner_unnecessary','owner_authority_unknown','owner_missing_trace',
                         'child_authority_unknown','child_not_overprivileged','child_unnecessary','child_invalid_necessity',
                         'child_missing_trace','child_resource_mismatch'):
                with self.subTest(kind=kind):
                    items=copy.deepcopy(template.items)
                    owner=next(item for item in items if item.metadata['source_candidate_id']=='native-candidate')
                    child=next(item for item in items if item.metadata['source_candidate_id']=='native-read-candidate')
                    if kind=='owner_inconclusive':owner.metadata['source_final_verdicts'][0]['necessity_label']='inconclusive'
                    elif kind=='owner_unnecessary':owner.metadata['source_final_verdicts'][0]['necessity_label']='unnecessary'
                    elif kind=='owner_authority_unknown':owner.metadata['source_final_verdicts'][0]['authorization_label']='inconclusive'
                    elif kind=='owner_missing_trace':owner.metadata.pop('original_execution_witnesses')
                    elif kind=='child_authority_unknown':child.metadata['source_final_verdicts'][0]['authorization_label']='inconclusive'
                    elif kind=='child_not_overprivileged':child.metadata['source_final_verdicts'][0]['label']='not_overprivileged'
                    elif kind=='child_unnecessary':child.metadata['source_final_verdicts'][0]['necessity_label']='unnecessary'
                    elif kind=='child_invalid_necessity':child.metadata['source_final_verdicts'][0]['necessity_label']='unknown-future-label'
                    elif kind=='child_missing_trace':child.metadata.pop('original_execution_witnesses')
                    elif kind=='child_resource_mismatch':child.metadata['candidate_semantics']['object_ref']='different.json'
                    (root/'SKILL.md').write_text(original)
                    self.assertIsNone(InstructionRewriter()._rewrite_literal_inline_read_group(patched_bundle_root=root,items=items))
                    self.assertEqual((root/'SKILL.md').read_text(),original)

    def test_overlapping_owner_and_read_keep_computation_and_real_runtime_goal(self):
        with tempfile.TemporaryDirectory(prefix='skillscope-inline-read-projection-') as raw:
            root = Path(raw).resolve()
            original, command, items = self.fixture(root)
            notes = InstructionRewriter().rewrite_many(patched_bundle_root=root, items=items)
            self.assertTrue(notes)
            projected = (root / 'SKILL.md').read_text()
            proof = items[0].metadata['inline_read_projection']
            safe_command = proof['safe_command']
            _, safe_body = InlineCommandTool()._literal_python_payload(safe_command, root)
            self.assertEqual(safe_body, self.BODY.replace("'restricted.json'", "'submitted.json'", 1))
            self.assertEqual(proof['original_command'], command)
            self.assertEqual(proof['original_body_sha256'], hashlib.sha256(self.BODY.encode()).hexdigest())
            self.assertTrue(proof['semantic_projection']['complete'])
            self.assertEqual(proof['read_repair_ids'], ['repair-read'])
            self.assertEqual(proof['owner_repair_ids'], ['repair-owner'])
            self.assertTrue(projected.startswith(original[:original.index('Run this exact')]))
            self.assertTrue(projected.endswith('\nKeep the full result.\n'))
            self.assertEqual(len(re.findall(r'(?<!`)`([^`]+)`(?!`)', projected)), 2)
            for item in items:
                self.assertIn(item.guard_condition, projected)
                self.assertEqual(item.metadata['instruction_projection_strategy'], 'fallback_literal_inline_authorized_read')
            (root / 'restricted.json').unlink()
            result = self.safe_runtime(root, safe_command)
            self.assertEqual(result.status, 'completed', result.stderr)
            self.assertEqual(json.loads(result.stdout), {'total': 23, 'count': 2})
            self.assertEqual(json.loads((root / 'answer.json').read_text()), {'total': 23, 'count': 2})
            reads = [Path(event.get('object_ref', '')).name for event in result.trace_events if event['event_type'] == 'file_read']
            self.assertIn('submitted.json', reads)
            self.assertNotIn('restricted.json', reads)
            (root / 'submitted.json').write_text('{"weights":[3,-2],"values":[6,4]}')
            changed = self.safe_runtime(root, safe_command)
            self.assertEqual(changed.status, 'completed', changed.stderr)
            self.assertEqual(json.loads(changed.stdout), {'total': 10, 'count': 2})

    def test_one_read_candidate_uses_the_same_literal_projection(self):
        with tempfile.TemporaryDirectory(prefix='skillscope-inline-single-read-') as raw:
            root = Path(raw).resolve()
            _, _, items = self.fixture(root)
            InstructionRewriter().rewrite(patched_bundle_root=root, item=items[1])
            self.assertEqual(items[1].metadata['inline_read_projection']['owner_repair_ids'], [])
            result = self.safe_runtime(root, items[1].metadata['inline_read_projection']['safe_command'])
            self.assertEqual(json.loads(result.stdout), {'total': 23, 'count': 2})

    def test_unique_embedded_source_excerpt_can_share_its_command_owner_range(self):
        with tempfile.TemporaryDirectory(prefix='skillscope-inline-read-excerpt-') as raw:
            root = Path(raw).resolve()
            _, command, items = self.fixture(root)
            begin, end = command.index('json.loads'), command.index('.read_text')
            items[1].raw_text = command[begin:end]
            self.assertNotEqual(items[1].raw_text, items[0].raw_text)
            InstructionRewriter().rewrite_many(patched_bundle_root=root, items=items)
            result = self.safe_runtime(root, items[1].metadata['inline_read_projection']['safe_command'])
            self.assertEqual(result.status, 'completed', result.stderr)
            self.assertEqual(json.loads(result.stdout), {'total': 23, 'count': 2})

    def test_overlap_refuses_ambiguous_mismatched_shadowed_or_extra_effects_without_partial_write(self):
        variants = {
            'ambiguous_authorization': (self.BODY, self.PROMPT.replace('Read only submitted.json', 'Read only submitted.json and other.json')),
            'read_source_mismatch': (self.BODY.replace("'restricted.json'", "'different.json'"), self.PROMPT),
            'multiple_reads': (self.BODY + "other = Path('restricted.json').read_text()\n", self.PROMPT),
            'dynamic_receiver': (self.BODY.replace("Path('restricted.json')", "Path(source)"), self.PROMPT),
            'shadowed_path': (self.BODY.replace('def solve(data):', 'Path = custom_path\ndef solve(data):'), self.PROMPT),
            'extra_write': (self.BODY + "Path('extra.log').write_text('extra')\n", self.PROMPT),
            'extra_write_same_output': (self.BODY + "Path('answer.json').write_text('extra')\n", self.PROMPT),
            'extra_delete': (self.BODY + "Path('extra.log').unlink(missing_ok=True)\n", self.PROMPT),
            'extra_command': (self.BODY + 'import subprocess\nsubprocess.run(["echo","extra"])\n', self.PROMPT),
            'contradictory_read': (self.BODY, self.PROMPT + ' Never read submitted.json.'),
            'contradictory_write': (self.BODY, self.PROMPT + ' Never write answer.json.'),
        }
        for name, (body, prompt) in variants.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix='skillscope-inline-read-negative-') as raw:
                root = Path(raw).resolve()
                original, _, items = self.fixture(root, body, prompt)
                metadata = copy.deepcopy([item.metadata for item in items])
                with self.assertRaisesRegex(RuntimeError, 'failed closed'):
                    InstructionRewriter().rewrite_many(patched_bundle_root=root, items=items)
                self.assertEqual((root / 'SKILL.md').read_text(), original)
                self.assertEqual([item.metadata for item in items], metadata)

    def test_native_owner_task_or_executable_mismatch_does_not_gain_read_substitution(self):
        for kind in ('wrong_command', 'wrong_task', 'not_native_verdict', 'missing_material_read', 'unobserved_read', 'empty_guard'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix='skillscope-inline-owner-negative-') as raw:
                root = Path(raw).resolve()
                original, _, items = self.fixture(root)
                if kind == 'wrong_command':
                    items[0].metadata['descriptor_contexts'][0]['material_action_instances'][0]['object'] = 'other-executable -c something'
                elif kind == 'wrong_task':
                    context = items[0].metadata['descriptor_contexts'][0]
                    context['task_context']['intent'] = self.PROMPT + ' Different task.'
                    context['material_action_instances'][0]['intent'] = context['task_context']['intent']
                elif kind == 'not_native_verdict':
                    items[0].metadata['descriptor_contexts'][0]['final_verdict'] = 'inconclusive'
                elif kind == 'missing_material_read':
                    items[1].metadata['descriptor_contexts'][0]['material_action_instances'] = []
                elif kind == 'unobserved_read':
                    items[1].metadata['descriptor_contexts'][0]['material_action_instances'][0]['evidence_kind'] = 'static_intention'
                elif kind == 'empty_guard':
                    items[0].guard_condition = ''
                with self.assertRaisesRegex(RuntimeError, 'failed closed'):
                    InstructionRewriter().rewrite_many(patched_bundle_root=root, items=items)
                self.assertEqual((root / 'SKILL.md').read_text(), original)

    def test_absent_or_symlinked_authorized_input_is_refused(self):
        for kind in ('missing', 'symlink'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix='skillscope-inline-input-negative-') as raw:
                root = Path(raw).resolve()
                original, _, items = self.fixture(root)
                (root / 'submitted.json').unlink()
                if kind == 'symlink':
                    (root / 'submitted.json').symlink_to(root / 'restricted.json')
                with self.assertRaisesRegex(RuntimeError, 'failed closed'):
                    InstructionRewriter().rewrite_many(patched_bundle_root=root, items=items)
                self.assertEqual((root / 'SKILL.md').read_text(), original)


if __name__ == '__main__':
    unittest.main()
