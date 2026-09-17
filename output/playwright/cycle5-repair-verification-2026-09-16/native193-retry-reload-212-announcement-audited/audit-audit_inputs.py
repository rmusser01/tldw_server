"""Read only the explicit native evidence allowlist; retain reduced audit facts."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / '.tmp/uat198-181-native-20260917'
OUT = Path(__file__).resolve().parent
NAMES = [
    'alice-controlled-provider-fault-installed.txt',
    'alice-controlled-provider-prompt.txt',
    'alice-controlled-provider-send.txt',
    'alice-controlled-provider-error.txt',
    'alice-controlled-provider-error-events.txt',
    'alice-controlled-provider-retry.txt',
    'alice-controlled-provider-fault-receipt.txt',
    'alice-controlled-provider-retry-settled.txt',
    'alice-controlled-provider-retry-events.txt',
    'alice-controlled-retry-reload.txt',
    'alice-controlled-retry-reload-settled.txt',
    'alice-controlled-retry-reload-events.txt',
    'character212-open.txt', 'character212-settled.txt', 'character212-events.txt',
    'alice-reviewed-characters.txt', 'alice-reviewed-chat-open.txt',
    'alice-reviewed-chat-initial-events.txt',
]


def read_result(name):
    text = (BASE / name).read_text()
    return json.loads(text.split('### Result\n', 1)[1].split('\n### Ran', 1)[0])


def body(event):
    value = event.get('body')
    return json.loads(value) if isinstance(value, str) else value


def digest(path):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


chat_id = '471f52fa-22c5-4eba-a3d6-ee5e6982884f'
ids = ['1e23f18b-4842-4799-931f-df2b2ea75448',
       'cb4a633c-3ef2-48bb-8658-d48702fa1ed6', 'pa_5455-6f36-098-dd2b']
retry = read_result('alice-controlled-provider-retry-events.txt')
reload = read_result('alice-controlled-retry-reload-events.txt')
chars = read_result('character212-events.txt')
fault = read_result('alice-controlled-provider-fault-receipt.txt')
events = retry['events']
responses = [e for e in events if e.get('event') == 'response']
complete = [e for e in responses if '/complete-v2' in e.get('url', '')]
assert [e['status'] for e in complete] == [400, 200]
assert body(complete[0])['detail']['error_code'] == 'model_not_available'
assert body(complete[0])['detail']['provider'] == 'ollama'
assert body(complete[0])['detail']['model'] == 'uat031-unavailable'
assert complete[1].get('body') is None  # No claim to a captured raw SSE body.
created = [e for e in responses if e.get('status') == 201 and e['url'].endswith('/chats/')]
assert len(created) == 1 and body(created[0])['id'] == chat_id
assert str(body(created[0])['character_id']) == '4'
persist = [e for e in responses if '/completions/persist' in e.get('url', '')]
assert len(persist) == 1 and persist[0]['status'] == 200
assert body(persist[0]) == {'chat_id': chat_id, 'assistant_message_id': ids[2], 'saved': True}
history = [e for e in reload['events'] if e.get('event') == 'response'
           and '/messages' in e.get('url', '') and e.get('status') == 200]
assert len(history) == 6
for event in history:
    rows = body(event)['messages']
    assert [r['id'] for r in rows] == ids
    assert all(r['conversation_id'] == chat_id for r in rows)
assert 'ALICE CHARACTER RETRY VERIFIED.' in reload['body']
assert "getByRole('button', { name: 'Retry same model' }).click()" in (BASE / 'alice-controlled-provider-retry.txt').read_text()
assert 'await page.reload();' in (BASE / 'alice-controlled-retry-reload.txt').read_text()
char_query = [e for e in chars['events'] if e.get('event') == 'response'
              and '/characters/query' in e.get('url', '') and e['status'] == 200]
assert len(char_query) == 1
catalogue = body(char_query[0])
assert catalogue['total'] == 1 and [str(x['id']) for x in catalogue['items']] == ['4']
assert '1 character found' in (BASE / 'character212-settled.txt').read_text()
assert '1 characters found' in (BASE / 'alice-reviewed-characters.txt').read_text()
assert any(e.get('event') == 'identity' and e.get('identity', {}).get('id') == 2
           for e in chars['events'])
save_requests = [e for e in events + reload['events']
                 if e.get('event') == 'request' and e.get('method') in {'POST', 'PUT', 'PATCH'}
                 and any(s in e.get('url', '') for s in ['/notes', '/flashcards'])]
assert save_requests == []
rows = body(history[0])['messages']
facts = {
    'chat_id': chat_id,
    'character_id': 4,
    'authenticated_identity': {'id': 2, 'username': 'alice'},
    'created_at_response': created[0]['at'],
    'completion_responses': [{'at': e['at'], 'status': e['status'],
                              'body_read_at': e.get('bodyReadAt')} for e in complete],
    'negative_error': body(complete[0]),
    'fault_receipt': fault,
    'persist_response': {'at': persist[0]['at'], 'body': body(persist[0])},
    'reload_capture_at': reload['at'],
    'reload_history_responses': [{'at': e['at'], 'status': e['status'],
                                  'ordered_ids': [r['id'] for r in body(e)['messages']]}
                                 for e in history],
    'canonical_rows': [{k: r.get(k) for k in ('id', 'sender', 'timestamp', 'conversation_id')}
                       for r in rows],
    'fresh_chat_greeting_save_requests_in_supplied_events': 0,
    'character_query': {'at': char_query[0]['at'], 'total': catalogue['total'], 'ids': [4]},
    'character_capture_at': chars['at'],
    'native_announcement': '1 character found',
    'assertions': 'passed',
}
(OUT / 'verified-facts.json').write_text(json.dumps(facts, indent=2) + '\n')
manifest = {
    'captured_at': datetime.now(timezone.utc).isoformat(),
    'scope': '18 explicit safe native inputs plus two official read-only task snapshots',
    'normalization': 'none; hashes cover original bytes; native inputs were not copied or changed',
    'inputs': [digest(BASE / name) for name in NAMES],
    'task_snapshots': [digest(OUT / name) for name in ('task13260.131.txt', 'task13260.150.txt')],
}
(OUT / 'input-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'assertions': 'passed', 'native_inputs': len(NAMES),
                  'stable_reload_responses': len(history), 'greeting_save_requests': len(save_requests)}))
