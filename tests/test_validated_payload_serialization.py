from dataclasses import dataclass
import json
from pathlib import Path
import unittest

from skillscope.module2_action_necessity_validation.validated_llm import ValidatedLLMCaller


class PromptLoader:
    def load(self, name):
        return 'Return the grounded result.'


class RecordingClient:
    def __init__(self):
        self.payloads = []

    def complete_json(self, *, system_prompt, user_prompt, schema_name):
        self.payloads.append(json.loads(user_prompt))
        return {'valid': True}


@dataclass
class NodeAttributes:
    referenced_names: set[str]
    source: Path


class ValidatedPayloadSerializationTests(unittest.TestCase):
    def test_candidate_chain_attributes_are_serialized_before_llm_call(self):
        client = RecordingClient()
        caller = ValidatedLLMCaller(llm_client=client, prompt_loader=PromptLoader())
        attributes = NodeAttributes({'input_data', 'Path'}, Path('compute.py'))
        payload = {'candidate_reaching_action_chain': [
            {'node_id': 'test:code:1', 'attributes': attributes},
        ], 'predicate_context': ('enabled', True)}
        result = caller.complete(prompt_asset='test', payload=payload, schema_name='test',
                                 validator=lambda response: (response, []))
        self.assertEqual(result.payload, {'valid': True})
        self.assertEqual(result.attempts, 1)
        sent = client.payloads[0]
        node = sent['candidate_reaching_action_chain'][0]
        self.assertEqual(set(node['attributes']['referenced_names']), {'input_data', 'Path'})
        self.assertEqual(node['attributes']['source'], 'compute.py')
        self.assertEqual(sent['predicate_context'], ['enabled', True])
        self.assertIs(payload['candidate_reaching_action_chain'][0]['attributes'], attributes)
        self.assertEqual(attributes.referenced_names, {'input_data', 'Path'})

    def test_json_native_payload_values_keep_their_types(self):
        client = RecordingClient()
        payload = {'allowed': False, 'count': 3, 'confidence': .8, 'missing': None,
                   'evidence': ['action:1'], 'operation': {'type': 'read'}}
        result = ValidatedLLMCaller(llm_client=client, prompt_loader=PromptLoader()).complete(
            prompt_asset='test', payload=payload, schema_name='test',
            validator=lambda response: (response, []))
        self.assertEqual(result.attempts, 1)
        self.assertEqual(client.payloads[0], payload)


if __name__ == '__main__':
    unittest.main()
