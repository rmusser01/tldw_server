"""Audit retained synthetic receipts only; no HTTP, DB, or browser operations."""

import hashlib
import json
from pathlib import Path

SOURCE = Path('.tmp/uat198-181-native-20260917')
OUT = Path('.tmp/uat209-210-native-independent-20260917')
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()

notes = json.loads((SOURCE / 'notes209-native-api-result.json').read_text())
assert notes['passed'] and len(notes['records']) == 23
records = notes['records']
ids = {'Alice': notes['aliceId'], 'Bob': notes['bobId']}
owners = {'Alice': '2', 'Bob': '3'}
for actor, owner in owners.items():
    mine = [r for r in records if r['actor'] == actor]
    assert next(r for r in mine if r['path'] == '/api/v1/auth/me')['body']['id'] == int(owner)
    catalogs = [r for r in mine if r['method'] == 'GET' and r['path'].startswith('/api/v1/notes/?')]
    assert len(catalogs) == 2
    for catalog in catalogs:
        assert catalog['status'] == 200
        assert all(n['client_id'] == owner for n in catalog['body']['notes'])
        assert all(k['client_id'] == owner for n in catalog['body']['notes'] for k in n.get('keywords') or [])
    foreign = ids['Bob' if actor == 'Alice' else 'Alice']
    foreign_requests = [r for r in mine if r['path'] == '/api/v1/notes/' + foreign]
    assert [(r['method'], r['status']) for r in foreign_requests] == [('GET', 404), ('PATCH', 404)]
    own = [r for r in mine if r['path'] == '/api/v1/notes/' + ids[actor]]
    assert [(r['method'], r['status'], r['body']['version']) for r in own] == [('PATCH', 200, 2), ('GET', 200, 2)]
    assert own[0]['body'] == own[1]['body']
original = '/api/v1/notes/b83dca90-fab0-4c6f-8c0f-6f1e93dfffc8'
original_reads = [r for r in records if r['path'] == original]
assert [(r['actor'], r['status']) for r in original_reads] == [('Alice', 200), ('Bob', 404), ('Alice', 200)]
assert original_reads[0]['body'] == original_reads[2]['body']
assert original_reads[0]['body']['version'] == 1

extracted = {}
for name in ['alice209-ui-save-events.txt', 'alice209-ui-reload-events.txt', 'bob209-notes-events.txt', 'bob209-ui-reload-events.txt']:
    raw = (SOURCE / name).read_text()
    capture = json.JSONDecoder().raw_decode(raw[raw.index('{'):])[0]
    events = [e for e in capture['events'] if e.get('at', '') >= '2026-09-17T08:15:00'
              and (e['event'] == 'identity' or '/api/v1/notes' in e.get('url', ''))]
    result = {'source': str(SOURCE / name), 'source_sha256': sha(SOURCE / name),
              'captured_at': capture['at'], 'extraction': 'Unmodified event objects at/after 08:15Z, identity or /api/v1/notes URL; all other cumulative events omitted.',
              'events': events}
    target = name.replace('.txt', '.scoped.json')
    (OUT / target).write_text(json.dumps(result, indent=2) + '\n')
    extracted[name] = events

def body(event):
    value = event.get('body')
    return json.loads(value) if isinstance(value, str) else value

save = extracted['alice209-ui-save-events.txt']
saved_responses = [e for e in save if e['event'] == 'response' and e.get('status') == 200
                   and e.get('url', '').endswith('/notes/' + ids['Alice']) and e['at'] >= '2026-09-17T08:30:45']
assert {body(e)['version'] for e in saved_responses} == {3, 4}
assert all(body(e)['content'] == 'Saved from Alice browser for UAT209. Teal lantern remains private.' for e in saved_responses)
reloads = {}
for actor, cutoff, expected_count, expected_version in [('Alice', '2026-09-17T08:32:00', 4, 4), ('Bob', '2026-09-17T08:37:30', 3, 2)]:
    events = [e for e in extracted[actor.lower() + '209-ui-reload-events.txt'] if e['at'] >= cutoff]
    identities = [e for e in events if e['event'] == 'identity']
    assert identities and all(e['identity']['id'] == int(owners[actor]) for e in identities)
    catalog = next(e for e in events if e['event'] == 'response' and '/notes/?' in e.get('url', ''))
    assert catalog['status'] == 200
    rows = body(catalog)['notes']
    assert len(rows) == expected_count and all(n['client_id'] == owners[actor] for n in rows)
    own = next(n for n in rows if n['id'] == ids[actor])
    assert own['version'] == expected_version
    expected_content = 'Saved from Alice browser for UAT209. Teal lantern remains private.' if actor == 'Alice' else 'Saved own Bob UAT209 body.'
    assert own['content'] == expected_content
    keywords = next(e for e in events if e['event'] == 'response' and '/notes/keywords/' in e.get('url', ''))
    assert keywords['status'] == 200 and all(k['client_id'] == owners[actor] for k in body(keywords))
    snapshot = (SOURCE / (actor.lower() + '209-ui-reload-settled.txt')).read_text()
    assert f'Showing 1-{expected_count} of {expected_count}' in snapshot and expected_content in snapshot
    assert 'heading "New note"' in snapshot
    assert 'page.reload()' in (SOURCE / (actor.lower() + '209-ui-reload.txt')).read_text()
    reloads[actor] = {'catalog_at': catalog['at'], 'owner': owners[actor], 'count': expected_count,
                      'saved_version': expected_version, 'keyword_count': len(body(keywords)), 'editor_after_reload': 'New note'}

initial = json.loads((SOURCE / 'graph210-admin-native-api-result.json').read_text())
graph = json.loads((SOURCE / 'graph210-admin-native-continuation-result.json').read_text())
assert initial['passed'] is False and graph['passed'] is True
assert initial['aliceId'] == graph['aliceId']
assert next(r for r in graph['records'] if r['path'] == '/api/v1/auth/me')['body']['id'] == 1
graph_reads = [r for r in graph['records'] if r['path'].startswith('/api/v1/notes/graph')]
assert len(graph_reads) == 2
for r in graph_reads:
    assert r['status'] == 200
    data = r['body']; node_ids = {n['id'] for n in data['nodes']}
    assert node_ids == {graph['aliceId'], 'tag:uat210-admin+tag'}
    assert len(data['edges']) == 1
    assert data['edges'][0]['source'] == graph['aliceId'] and data['edges'][0]['target'] == 'tag:uat210-admin+tag'
    assert data['active_note_count'] == 1 and data['all_notes_eligible'] and not data['truncated']
assert next(r for r in initial['records'] if r['path'].startswith('/api/v1/notes/graph'))['status'] == 200
# The script itself is inspected, never run or retained with credential-access helpers.
assert "n.id==='note:'+alice.id" in (SOURCE / 'graph210-admin-native-api.mjs').read_text()
assert "n.id===alice.id" in (SOURCE / 'graph210-admin-native-continuation.mjs').read_text()

before = json.loads((SOURCE / 'reviewed218-before-restart.json').read_text())
after = json.loads((SOURCE / 'reviewed218-after-health.json').read_text())
assert before['files'] == after['files'] and before['revision'] == after['revision']
relevant = [f for f in after['files'] if f['path'].endswith(('ChaChaNotes_DB.py', 'chacha/note_store.py', 'chacha/keyword_store.py', 'endpoints/notes.py'))]
assert len(relevant) == 4
provenance = {'source_manifests': [{'path': str(SOURCE / n), 'sha256': sha(SOURCE / n)} for n in ['reviewed218-before-restart.json', 'reviewed218-after-health.json']],
              'revision': after['revision'], 'before_at': before['at'], 'after_at': after['at'],
              'all_3668_source_hashes_equal': True, 'relevant_files': relevant,
              'health': json.loads((SOURCE / 'reviewed218-replacement-health.json').read_text())}
(OUT / 'runtime-source-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
summary = {'notes_api_requests': 23, 'notes_api_passed': True, 'original_source_unchanged_version': 1,
           'bidirectional_foreign_get_patch_404': True, 'own_patch_read_versions': 2, 'reloads': reloads,
           'admin_graph_focused_all_200': True, 'graph_original_harness_prefix_failure_preserved': True,
           'graph_note_id': graph['aliceId'], 'application_ownership_only': True,
           'restricted_role_RLS_native_acceptance': False, 'no_browser_or_network_actions_by_reviewer': True}
(OUT / 'audit-verification.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
