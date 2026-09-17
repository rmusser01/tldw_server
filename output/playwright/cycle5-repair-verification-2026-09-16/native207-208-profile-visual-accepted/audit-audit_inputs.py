"""Verify allowlisted native catalogue/fallback receipts without accessing auth data."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / '.tmp/uat198-181-native-20260917'
OUT = Path(__file__).resolve().parent
NAMES = ['bob-account-switch-reviewed-events.txt'] + [
    'bob-reviewed-' + part + '.txt' for part in (
        'characters-open', 'characters-settled', 'character-chat-open',
        'character-chat-settled', 'character-chat-events', 'character-reload',
        'character-reload-settled', 'character-reload-events')
] + [
    'alice-controlled-provider-retry.txt',
    'alice-controlled-provider-retry-settled.txt',
    'alice-controlled-provider-retry-events.txt',
    'alice-controlled-retry-reload.txt',
    'alice-controlled-retry-reload-settled.txt',
    'alice-controlled-retry-reload-events.txt',
]


def result(name):
    s = (BASE / name).read_text()
    return json.loads(s.split('### Result\n', 1)[1].split('\n### Ran', 1)[0])


def body(event):
    v = event.get('body')
    return json.loads(v) if isinstance(v, str) else v


def digest(path):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


event_files = [n for n in NAMES if n.endswith('-events.txt')]
events = {}
captures = []
for name in event_files:
    capture = result(name)
    captures.append({'input': name, 'at': capture['at'], 'url': capture['url']})
    for e in capture['events']:
        # Inputs are cumulative; count each exact event only once.
        events[json.dumps(e, sort_keys=True)] = e
events = sorted(events.values(), key=lambda e: e.get('at', ''))
responses = [e for e in events if e.get('event') == 'response']
profiles = [e for e in responses if '/api/v1/persona/profiles' in e.get('url', '')]
resolvers = [e for e in responses if '/api/v1/visual-identities/bindings/resolve' in e.get('url', '')]
assert len(profiles) == 14 and all(e['status'] == 200 for e in profiles)
assert len(resolvers) == 6 and all(e['status'] == 200 for e in resolvers)
switch = '2026-09-17T07:22:53.195Z'
alice_profiles = [e for e in profiles if e['at'] < switch]
bob_profiles = [e for e in profiles if e['at'] > switch]
assert len(alice_profiles) == len(bob_profiles) == 7
for group, profile_id, created in (
    (alice_profiles, 'research_assistant', '2026-09-16 22:23:13.441000+00:00'),
    (bob_profiles, 'research_assistant:3', '2026-09-17 07:25:12.418000+00:00'),
):
    for e in group:
        rows = body(e)
        assert len(rows) == 1 and rows[0]['id'] == profile_id
        assert rows[0]['created_at'] == rows[0]['last_modified'] == created
        assert rows[0]['version'] == 1
for e in resolvers:
    v = body(e)
    assert v['actor_kind'] == 'character'
    assert v['actor_id'] == (4 if e['at'] < switch else 3)
    assert v['fallback_reason'] == 'metadata_backend_unsupported'
    assert v['resolution_source'] == 'placeholder'
    assert all(v[k] is None for k in ('pack_id', 'pack_version_id', 'asset_id', 'asset_url', 'preview_url'))
assert [body(e)['requested_expression_key'] for e in resolvers] == ['neutral', 'happy', 'neutral', 'happy', 'neutral', 'neutral']
identities = [e for e in events if e.get('event') == 'identity']
assert any(e['identity'] == {'id': 2, 'username': 'alice'} for e in identities)
assert any(e['identity'] == {'id': 3, 'username': 'bob'} for e in identities)
assert 'await page.reload();' in (BASE / 'bob-reviewed-character-reload.txt').read_text()
assert "getByRole('button', { name: 'Chat as Helpful AI Assistant' }).click()" in (BASE / 'bob-reviewed-character-chat-open.txt').read_text()
for name in ('bob-reviewed-character-chat-settled.txt', 'bob-reviewed-character-reload-settled.txt'):
    s = (BASE / name).read_text()
    assert 'Character: Helpful AI Assistant' in s
    assert 'Add expression images for Helpful AI Assistant.' in s
facts = {
    'captures': captures,
    'deduplication': 'exact event JSON across cumulative captures',
    'identity_events': [{'at': e['at'], 'identity': e['identity']} for e in identities],
    'profile_responses': [{'at': e['at'], 'status': e['status'],
                           'rows': [{k: p.get(k) for k in ('id', 'created_at', 'last_modified', 'version')}
                                    for p in body(e)]} for e in profiles],
    'resolver_responses': [{'at': e['at'], 'status': e['status'], 'body': body(e)} for e in resolvers],
    'assertions': 'passed',
}
(OUT / 'verified-facts.json').write_text(json.dumps(facts, indent=2) + '\n')
manifest = {
    'captured_at': datetime.now(timezone.utc).isoformat(),
    'scope': '15 explicit safe native inputs plus two official read-only task snapshots',
    'normalization': 'none; original bytes hashed; native inputs not copied or changed',
    'inputs': [digest(BASE / n) for n in NAMES],
    'task_snapshots': [digest(OUT / n) for n in ('task13260.145.txt', 'task13260.146.txt')],
}
(OUT / 'input-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'assertions': 'passed', 'inputs': len(NAMES),
                  'unique_profile_responses': len(profiles),
                  'unique_resolver_responses': len(resolvers)}))
