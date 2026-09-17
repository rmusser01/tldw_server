"""Audit recorded API/UI ownership evidence without executing native scripts."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / '.tmp/uat198-181-native-20260917'
OUT = Path(__file__).resolve().parent
NAMES = [
    'character194-native-api.mjs', 'character194-native-api-result.json',
    'character194-native-continuation.mjs', 'character194-native-continuation-result.json',
    'alice194-characters-open.txt', 'alice194-characters-settled.txt', 'alice194-characters-events.txt',
    'bob194-characters-open.txt', 'bob194-characters-settled.txt', 'bob194-characters-events.txt',
    'bob194-reload.txt', 'bob194-reload-settled.txt', 'bob194-reload-events.txt',
]


def digest(path):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def result(name):
    s = (BASE / name).read_text()
    return json.loads(s.split('### Result\n', 1)[1].split('\n### Ran', 1)[0])


def select(records, actor, method, path):
    rows = [e for e in records if e.get('actor') == actor and e.get('method') == method and e.get('path') == path]
    assert rows
    return rows


first = json.loads((BASE / 'character194-native-api-result.json').read_text())
second = json.loads((BASE / 'character194-native-continuation-result.json').read_text())
assert first['passed'] is False and len(first['records']) == 23
assert second['passed'] is True and len(second['records']) == 15
assert first['aliceId'] == second['aliceId'] == 5
assert first['bobId'] == second['bobId'] == 6
a, b = first['records'], second['records']
for run in (a, b):
    for actor, owner in (('Alice', 2), ('Bob', 3)):
        identity = select(run, actor, 'GET', '/api/v1/auth/me')[0]
        assert identity['status'] == 200 and identity['body']['id'] == owner
        logout = select(run, actor, 'POST', '/api/v1/auth/logout')
        assert len(logout) == 1 and logout[0]['status'] == 200
    assert not any('cleanupError' in e for e in run)
for method, path in (
    ('GET', '/api/v1/characters/5'),
    ('PUT', '/api/v1/characters/5?expected_version=1'),
    ('DELETE', '/api/v1/characters/5?expected_version=1'),
):
    assert select(a, 'Bob', method, path)[0]['status'] == 404
original = select(a, 'Alice', 'POST', '/api/v1/characters/')[0]
assert original['status'] == 201 and original['body']['id'] == 5 and original['body']['version'] == 1
unchanged = select(a, 'Alice', 'GET', '/api/v1/characters/5')[0]['body']
assert unchanged['version'] == 1 and unchanged['description'] == original['body']['description']
owned_update = select(a, 'Alice', 'PUT', '/api/v1/characters/5?expected_version=1')[0]
assert owned_update['status'] == 200 and owned_update['body']['version'] == 2
bob_created = select(a, 'Bob', 'POST', '/api/v1/characters/')[0]
assert bob_created['status'] == 201 and bob_created['body']['id'] == 6
assert bob_created['body']['name'] == original['body']['name']
assert select(a, 'Alice', 'GET', '/api/v1/characters/6')[0]['status'] == 404
for actor, own, foreign in (('Alice', 5, 6), ('Bob', 6, 5)):
    listed = select(a, actor, 'GET', '/api/v1/characters/')[-1]['body']
    assert own in [r['id'] for r in listed] and foreign not in [r['id'] for r in listed]
foreign_restore = select(b, 'Bob', 'POST', '/api/v1/characters/5/restore?expected_version=3')[0]
missing_restore = select(b, 'Bob', 'POST', '/api/v1/characters/999999999/restore?expected_version=3')[0]
assert select(a, 'Bob', 'POST', '/api/v1/characters/5/restore?expected_version=3')[0]['status'] == 409
assert foreign_restore['status'] == missing_restore['status'] == 409
assert re.sub(r'\d+', '<id>', foreign_restore['body']['detail']) == re.sub(r'\d+', '<id>', missing_restore['body']['detail'])
deleted = [e for e in b if e.get('actor') == 'Alice' and 'deleted_only=true' in e.get('path', '')][0]
assert deleted['status'] == 200 and deleted['body']['total'] == 1
assert deleted['body']['items'][0]['id'] == 5 and deleted['body']['items'][0]['version'] == 3
restored = select(b, 'Alice', 'POST', '/api/v1/characters/5/restore?expected_version=3')[0]
assert restored['status'] == 200 and restored['body']['version'] == 4
assert restored['body']['description'] == owned_update['body']['description']
assert select(b, 'Bob', 'GET', '/api/v1/characters/5')[0]['status'] == 404
assert select(b, 'Bob', 'GET', '/api/v1/characters/6')[0]['body']['version'] == 1
ui = []
assert 'await page.reload();' in (BASE / 'bob194-reload.txt').read_text()
for name, owner, own, foreign, description in (
    ('alice194-characters-events.txt', 2, 5, 6, 'Synthetic Alice-only UAT194 violet compass updated.'),
    ('bob194-characters-events.txt', 3, 6, 5, 'Synthetic Bob-only UAT194 amber map.'),
    ('bob194-reload-events.txt', 3, 6, 5, 'Synthetic Bob-only UAT194 amber map.'),
):
    capture = result(name)
    identities = [e for e in capture['events'] if e.get('event') == 'identity']
    assert identities[-1]['identity']['id'] == owner
    responses = [e for e in capture['events'] if e.get('event') == 'response' and '/characters/query' in e.get('url', '')]
    response = responses[-1]
    payload = json.loads(response['body'])
    assert response['status'] == 200 and payload['total'] == 2
    assert own in [x['id'] for x in payload['items']] and foreign not in [x['id'] for x in payload['items']]
    assert description in capture['body']
    opposite = 'Synthetic Bob-only UAT194 amber map.' if owner == 2 else 'Synthetic Alice-only UAT194 violet compass updated.'
    assert opposite not in capture['body']
    ui.append({'input': name, 'capture_at': capture['at'], 'identity': identities[-1]['identity'],
               'query_at': response['at'], 'status': response['status'],
               'items': [{k: row.get(k) for k in ('id', 'name', 'description', 'version')} for row in payload['items']]})
facts = {
    'original_harness_passed': False, 'original_records': 23,
    'original_stop': 'foreign restore expected404, actual409; original evidence remains unchanged',
    'continuation_passed': True, 'continuation_records': 15,
    'foreign_restore': foreign_restore, 'nonexistent_restore': missing_restore,
    'alice_before_restore': {'id': 5, 'version': 3},
    'alice_restored': {'id': 5, 'version': 4, 'description': restored['body']['description']},
    'bob_preserved': {'id': 6, 'version': 1, 'description': bob_created['body']['description']},
    'logout_responses': 4, 'logout_statuses': [200, 200, 200, 200],
    'ui': ui, 'assertions': 'passed',
}
(OUT / 'verified-facts.json').write_text(json.dumps(facts, indent=2) + '\n')
(OUT / 'input-manifest.json').write_text(json.dumps({
    'captured_at': datetime.now(timezone.utc).isoformat(),
    'scope': '13 explicit safe API/script/UI inputs and one official task snapshot',
    'normalization': 'none; original inputs unchanged',
    'inputs': [digest(BASE / n) for n in NAMES],
    'task_snapshot': digest(OUT / 'task13260.132.txt'),
}, indent=2) + '\n')
print(json.dumps({'assertions': 'passed', 'inputs': len(NAMES), 'original_passed': False,
                  'continuation_passed': True, 'ui_captures': len(ui)}))
