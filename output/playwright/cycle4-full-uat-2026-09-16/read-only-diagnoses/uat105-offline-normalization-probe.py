"""Offline source-function replay; no repository module imports, network, or DB.
Exact AST bodies for extraction, adapter normalization, dispatch, analyze and
plaintext caller run with only environment/converter/adapter seams substituted.
The adapter returns retained response data; it never calls a provider.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import inspect
import json
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/Users/macbook-dev/Documents/GitHub/tldw_server2')
FILES = {
    'extract': 'tldw_Server_API/app/core/Chat/chat_helpers.py',
    'summary': 'tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py',
    'document': 'tldw_Server_API/app/core/Ingestion_Media_Processing/Plaintext/Plaintext_Files.py',
}
class QuietLog:
    def __getattr__(self, name):
        return lambda *a, **k: self
class ProviderError(Exception):
    pass
class Adapter:
    response = None
    calls = 0
    def chat(self, *a, **k):
        self.calls += 1
        return self.response
adapter = Adapter()
ns = dict(
    __name__='offline_frozen_source_probe', Path=Path, time=time, copy=copy,
    inspect=inspect, json=json, logging=QuietLog(),
    _adapter_provider_name=lambda name: name,
    bind_provider_call_credentials=lambda provider, envelope, **kw: (envelope, None),
    ensure_app_config=lambda config: config,
    load_and_log_configs=lambda: {},
    get_registry=lambda: SimpleNamespace(get_adapter=lambda provider: adapter),
    resolve_provider_model=lambda provider, config: 'offline-retained-model',
    resolve_provider_api_key_from_config=lambda provider, config: None,
    _resolve_adapter_timeout=lambda provider, config: None,
    _build_summary_prompt=lambda text, prompt: text,
    _resolve_default_system_prompt=lambda: 'offline-fixture-summary',
    extract_text_from_input=lambda value: value,
    _SUMMARIZATION_NONCRITICAL_EXCEPTIONS=(ValueError, TypeError, AttributeError, LookupError, RuntimeError),
    _SUMMARY_ADAPTER_EXCEPTIONS=(ProviderError, ValueError, TypeError, AttributeError, LookupError, RuntimeError),
    ChatConfigurationError=ProviderError,
    SummaryProviderError=ProviderError,
    _PLAINTEXT_NONCRITICAL_EXCEPTIONS=(ValueError, TypeError, AttributeError, LookupError, RuntimeError),
    PandocMissing=ProviderError,
    log_counter=lambda *a, **k: None,
    log_histogram=lambda *a, **k: None,
)
for key, wanted in [
    ('extract', ['extract_response_content']),
    ('summary', ['_summarize_via_adapter', '_dispatch_to_api', 'analyze']),
    ('document', ['process_document_content']),
]:
    filename = str(ROOT / FILES[key])
    tree = ast.parse(Path(filename).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in nodes} == set(wanted)
    mod = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), filename, 'exec'), ns)

cedar = json.loads((ROOT / 'output/playwright/cycle4-full-uat-2026-09-16/multi/cedar-ingest-media.json').read_text())
cedar = cedar.get('body', cedar)
aster = json.loads(Path('/private/tmp/uat-cycle4-single-aster-media.json').read_text())['body']
raw = cedar['processing']['analysis']
envelope = ast.literal_eval(raw)
source = cedar['content']['text']
ns['convert_document_to_text'] = lambda *a, **k: (source, 'txt', {})
message = envelope['choices'][0]['message']
shape = {
    'stored_analysis_characters': len(raw), 'source_characters': len(source),
    'choice_finish_reason': envelope['choices'][0]['finish_reason'],
    'message_content': message.get('content'),
    'reasoning_characters': len(message.get('reasoning_content') or ''),
    'completion_tokens': envelope.get('usage', {}).get('completion_tokens'),
    'stored_analysis_equals_provider_repr': raw == str(envelope),
    'positive_single_analysis_characters': len(aster['processing']['analysis']),
}
def wrapped(content, finish='stop'):
    return {'choices': [{'message': {'role': 'assistant', 'content': content, 'reasoning_content': 'REASONING_SENTINEL'}, 'finish_reason': finish}], 'model': 'fixture', 'usage': {'completion_tokens': 1}}
cases = [
    ('retained_empty_length', envelope),
    ('empty_stop', wrapped('')),
    ('null_stop', wrapped(None)),
    ('malformed_dictionary', {'unexpected': 'ENVELOPE_SENTINEL'}),
    ('positive_same_model_text', wrapped(aster['processing']['analysis'])),
    ('plain_string', 'Safe user-facing summary'),
    ('partial_answer_length', wrapped('Partial user-facing answer', 'length')),
    ('content_parts', wrapped([{'type': 'text', 'text': 'User-facing content block'}])),
]
records = []
for name, response in cases:
    adapter.response = response
    result = ns['process_document_content'](Path('/private/tmp/offline-cedar.txt'), False, None, True, False, 'llama', None, None, None)
    analysis = result['analysis']
    records.append({
        'case': name, 'status': result['status'], 'warnings': result['warnings'],
        'source_preserved': result['content'] == source,
        'analysis_type': type(analysis).__name__,
        'analysis_characters': len(analysis) if isinstance(analysis, str) else None,
        'analysis_equals_response_repr': analysis == str(response) if not isinstance(response, str) else False,
        'reasoning_leaked': isinstance(analysis, str) and ('reasoning_content' in analysis or 'REASONING_SENTINEL' in analysis),
    })
# Existing caller control: safe explicit failure leaves source and creates Warning.
ns['analyze'] = lambda **kw: 'Error: Provider returned no usable answer.'
control = ns['process_document_content'](Path('/private/tmp/offline-cedar.txt'), False, None, True, False, 'llama', None, None, None)
result = {
    'method': __doc__, 'shape': shape, 'cases': records,
    'existing_warning_control': {'status': control['status'], 'analysis': control['analysis'], 'source_preserved': control['content'] == source, 'warnings': control['warnings']},
    'adapter_calls': adapter.calls, 'provider_network_calls': 0, 'database_writes': 0,
    'source_hashes': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in FILES.values()},
}
Path('/private/tmp/uat105-offline-normalization-result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
assert records[0]['status'] == 'Warning' and records[0]['analysis_type'] == 'NoneType', 'UAT105 reproduced: retained empty completion is accepted as successful raw-envelope analysis'
