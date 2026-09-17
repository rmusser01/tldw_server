"""Read retained synthetic UAT evidence only; write narrow, hash-bound metadata."""
from pathlib import Path
from datetime import datetime
from urllib.parse import urlsplit, parse_qsl
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / '.tmp/uat198-181-native-20260917'
META = ROOT / '.tmp/uat181-full-native-20260917'
OUT = Path(__file__).resolve().parent
inputs = {}

def read(path):
    data = path.read_bytes()
    inputs[str(path.relative_to(ROOT))] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    return data.decode()

def obj(path):
    return json.loads(read(path))

def events(name):
    text = read(BASE / name)
    return json.JSONDecoder().raw_decode(text[text.index('{'):])[0]

def body(event):
    value = event.get('body')
    return json.loads(value) if isinstance(value, str) else value

def moment(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))

def rows(value):
    return {row['path']: row['sha256'] for row in value['files']}

def path(event):
    return urlsplit(event.get('url', '')).path

def excerpt(event):
    # Deliberately excludes response/request bodies, headers, title and text.
    value = {'at': event['at'], 'path': path(event), 'status': event['status']}
    allowed = {'due_status', 'deck_id', 'include_workspace_items', 'limit', 'offset', 'order_by', 'status'}
    query = {k: v for k, v in parse_qsl(urlsplit(event.get('url', '')).query) if k in allowed}
    if query:
        value['query'] = query
    return value

task = next((ROOT / 'backlog/tasks').glob('task-13260.118 - *.md'))
assert 'Original native Flashcards/deck/Notes reads followed by owned API replacement succeed' in read(task)
old_before = obj(BASE / 'reviewed220-before-restart.json')
old_after = obj(BASE / 'reviewed220-after-health.json')
new_before = obj(BASE / 'reviewed222-before-restart.json')
new_after = obj(BASE / 'reviewed222-after-health.json')
assert rows(old_before) == rows(old_after) and rows(new_before) == rows(new_after)
assert len(rows(old_after)) == len(rows(new_after)) == 3668
assert old_after['revision'].startswith('7acc8b001a') and new_after['revision'].startswith('598d377df2')
changed = [p for p in sorted(rows(old_after)) if rows(old_after)[p] != rows(new_after)[p]]
assert changed == ['tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py', 'tldw_Server_API/app/core/StudyPacks/provenance.py']
for task_id in ['222', '223']:
    directory = 'uat222-diagnosis-20260917' if task_id == '222' else 'uat223-repair-20260917'
    manifest = obj(ROOT / '.tmp' / directory / 'owned-manifest.json')
    for file in manifest['files']:
        if '/app/' in file['path']:
            assert rows(new_after)[file['path']] == file['sha256']
stable_owner_paths = [
    'tldw_Server_API/app/core/DB_Management/chacha/operation_scope.py',
    'tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py',
    'tldw_Server_API/app/main.py',
    'tldw_Server_API/app/core/Chat/conversation_enrichment.py',
    'tldw_Server_API/app/services/study_pack_jobs_worker.py',
]
for file in stable_owner_paths:
    assert rows(old_after)[file] == rows(new_after)[file]
assert rows(new_after)[stable_owner_paths[-1]] == '8723ee86e578b05c45bb9a57ce2fd04feaebd6744dc5fc006e144ea0c0cdc78c'

stop = obj(BASE / 'overlap82583-to222-stop.json')
health = obj(BASE / 'overlap89545-health.json')
assert stop['oldPid'] == health['oldPid'] == 82583 and health['newPid'] == 89545
assert stop['signal'] == 'SIGTERM' and stop['rows'][-1]['oldAlive'] and not stop['rows'][-1]['occupied']
assert health['rows'][0]['status'] == 200 and health['rows'][0]['oldAlive'] and health['rows'][0]['newAlive']
metas = [obj(META / (name + '.json')) for name in ['warmed82583-before222-restart', 'overlap89545-after-health', 'after89545-owner-study-reads']]
assert len({m['targetContentDatabaseSha256'] for m in metas}) == 1
assert all(m['transactionReadOnlyVerified'] and not m['locks'] and all(s['state'] == 'idle' for s in m['sessions']) for m in metas)
assert moment(metas[0]['observedAt']) < moment(stop['rows'][0]['at']) < moment(health['rows'][0]['at']) <= moment(metas[1]['observedAt'])

warm = events('alice181-final-warmed-events.txt')
study_capture = events('alice181-final-flashcards-events.txt')
notes = events('alice181-reviewed222-notes-events.txt')
study_ui = read(BASE / 'alice181-final-flashcards-ready.txt')
manage_ui = read(BASE / 'alice181-final-manage-ready.txt')
notes_ui = read(BASE / 'alice181-reviewed222-notes-settled.txt')
assert '8 cards remaining' in study_ui and 'tabpanel "Manage"' in manage_ui and ': 8 cards' in manage_ui
assert 'Showing 1-4 of 4' in notes_ui
assert next(e for e in reversed(warm['events']) if e['event'] == 'identity')['identity']['id'] == 2
assert next(e for e in reversed(notes['events']) if e['event'] == 'identity')['identity']['id'] == 2

responses = [e for e in warm['events'] if e['event'] == 'response']
warm_cards = [e for e in responses if '2026-09-17T09:20:52.000Z' <= e['at'] < stop['rows'][0]['at'] and path(e).startswith('/api/v1/flashcards')]
expected = {'/api/v1/flashcards', '/api/v1/flashcards/decks', '/api/v1/flashcards/source-review-plans/due', '/api/v1/flashcards/review-sessions', '/api/v1/flashcards/analytics/summary', '/api/v1/flashcards/review/next'}
assert {path(e) for e in warm_cards} == expected and all(e['status'] == 200 for e in warm_cards)
assert any(parse_qsl(urlsplit(e['url']).query) and 'due_status=due' in e['url'] for e in warm_cards)
assert all(moment(e['at']) < moment(metas[0]['observedAt']) for e in warm_cards)
assert any(path(e) == '/api/v1/flashcards/decks' and e['status'] == 200 and e['at'] > '2026-09-17T09:20:52.000Z' for e in study_capture['events'] if e['event'] == 'response')
note_paths = {'/api/v1/notes/', '/api/v1/notes/keywords/', '/api/v1/notes/collections'}
warm_notes = [e for e in responses if '2026-09-17T09:18:45Z' <= e['at'] < '2026-09-17T09:18:47Z' and path(e) in note_paths]
assert {path(e) for e in warm_notes} == note_paths and all(e['status'] == 200 for e in warm_notes)
warm_note_payload = body(next(e for e in warm_notes if path(e) == '/api/v1/notes/'))
assert len(warm_note_payload['notes']) == 4 and {row['client_id'] for row in warm_note_payload['notes']} == {'2'}
post_health = [e for e in notes['events'] if e['event'] == 'response' and e['at'] > health['rows'][0]['at']]
post_notes = [e for e in post_health if path(e) in note_paths]
assert {path(e) for e in post_notes} == note_paths and all(e['status'] == 200 for e in post_notes)
assert all(e['status'] == 200 for e in post_health)
new_note_payload = body(next(e for e in post_notes if path(e) == '/api/v1/notes/'))
assert new_note_payload['count'] == new_note_payload['total'] == 4
assert {n['id'] for n in new_note_payload['notes']} == {n['id'] for n in warm_note_payload['notes']}
assert {row['client_id'] for row in new_note_payload['notes']} == {'2'}

result = obj(BASE / 'study-pack-job5-reviewed222-readback-result.json')
records = result['records']
drain = obj(BASE / 'known-api-drain-after222.json')
assert drain['rows'] == [{'pid': 82583, 'alive': True}, {'pid': 89545, 'alive': True}]
assert moment(result['at']) < moment(metas[2]['observedAt']) < moment(drain['at'])
assert result['passed'] and result['aliceId'] == 2 and len(records) == 20
def record(actor, wanted):
    return next(r for r in records if r['actor'] == actor and r['path'] == wanted)
for who, identity in [('Alice', 2), ('Bob', 3)]:
    assert record(who, '/api/v1/auth/me')['body']['id'] == identity
    assert record(who, '/api/v1/auth/logout')['status'] == 200
job = record('Alice', '/api/v1/flashcards/study-packs/jobs/5')['body']
assert job['job']['id'] == 5 and job['job']['status'] == 'completed' and job['error'] is None
pack = record('Alice', '/api/v1/flashcards/study-packs/2')['body']
assert pack == job['study_pack'] and pack['id'] == 2 and pack['deck_id'] == 10 and pack['client_id'] == '2'
assert moment(pack['created_at']) < moment(stop['rows'][0]['at'])
cards = record('Alice', '/api/v1/flashcards?deck_id=10')['body']['items']
assert len(cards) == 3 and all(c['client_id'] == '2' and c['deck_id'] == 10 for c in cards)
assistant_rows = [r for r in records if r['actor'] == 'Alice' and r['path'].endswith('/assistant')]
assert len(assistant_rows) == 3
for row in assistant_rows:
    value = row['body']
    assert row['status'] == 200 and value['study_pack']['id'] == 2 and len(value['citations']) == 1
    assert value['primary_citation'] == value['citations'][0] and value['citations'][0]['client_id'] == '2'
assert record('Bob', '/api/v1/flashcards/study-packs/jobs/5')['status'] == 404
assert record('Bob', '/api/v1/flashcards/study-packs/2')['status'] == 404
assert all(r['status'] == 404 for r in records if r['actor'] == 'Bob' and r['path'].endswith('/assistant'))
assert record('Bob', '/api/v1/flashcards?deck_id=10')['body']['items'] == []
assert all(r['method'] == 'GET' or r['path'] in {'/api/v1/auth/login', '/api/v1/auth/logout'} for r in records)
assert len([r for r in records if r['actor'] == 'Alice' and r['path'].endswith('/jobs/5') and r['body'] == job]) == 2

prior = ROOT / 'output/playwright/cycle5-repair-verification-2026-09-16'
for name in [
    prior / 'followup181-enrichment-ownership-reviewed/independent-REVIEW181-enrichment.md',
    prior / 'followup181-study-worker-reviewed/independent-REVIEW181-study-pack.md',
    ROOT / '.tmp/uat204-independent-20260917/REVIEW204.md',
    ROOT / '.tmp/uat181-http-independent-20260917/REVIEW181.md',
    ROOT / '.tmp/uat171-181-213-214-native-audit-20260917/ACCEPTANCE171-181-213-214.md',
]:
    read(name)

unrelated_errors = [e for e in responses if e['at'] > '2026-09-17T09:18:45Z' and e['status'] >= 400]
assert all(path(e) == '/api/v1/notes/graph' and e['status'] == 403 for e in unrelated_errors)
excerpts = {
    'format': 'Metadata-only excerpts; no request/response bodies, credentials, query values outside allowlist or user content.',
    'warmed_notes': [excerpt(e) for e in warm_notes],
    'warmed_flashcards': [excerpt(e) for e in warm_cards],
    'post_health_notes': [excerpt(e) for e in post_notes],
    'separate_graph_permission_responses': [excerpt(e) for e in unrelated_errors],
    'readback_records': [{k: r[k] for k in ['at', 'actor', 'method', 'path', 'status']} for r in records],
}
summary = {
    'task': 'TASK13260.118', 'verdict': 'Literal native AC3 supported; recommend bounded UAT181 closure with prior reviewed AC1/AC2 evidence.',
    'source': {'old_revision': old_after['revision'], 'new_revision': new_after['revision'], 'files_each': 3668, 'startup_pairs_equal': True, 'changed_backend_paths': changed, 'unchanged_owner_paths': {p: rows(new_after)[p] for p in stable_owner_paths}},
    'replacement': {'stop': stop, 'health': health, 'later_process_snapshot': drain},
    'metadata': [{'observedAt': m['observedAt'], 'sessions': len(m['sessions']), 'states': sorted({s['state'] for s in m['sessions']}), 'tableLocks': len(m['locks']), 'relationScope': m['relationScope'], 'transactionReadOnlyVerified': m['transactionReadOnlyVerified']} for m in metas],
    'same_content_database_sha256': metas[0]['targetContentDatabaseSha256'],
    'native': {'actor': 2, 'warmed_notes_responses': len(warm_notes), 'warmed_flashcards_responses': len(warm_cards), 'first_card_at': min(e['at'] for e in warm_cards), 'last_card_at': max(e['at'] for e in warm_cards), 'post_health_notes_responses': len(post_notes), 'same_four_owned_note_ids': True, 'notes_ui_count': 4, 'manage_ui_cards': 8},
    'study_pack_readback': {'request_count': len(records), 'job_id': 5, 'status': 'completed', 'pack_id': 2, 'deck_id': 10, 'cards': 3, 'citation_contexts': 3, 'owner': 2, 'negative_owner': 3, 'unchanged_repeat_job_read': True, 'scope': 'Fresh API-session canonical reads of an already completed job; no model call, regeneration or new worker-execution claim.'},
    'limits': ['Point-in-time all-public-table metadata, excluding indexes/catalogs; no continuous lock monitoring.', 'Old and replacement processes alive at health and at09:25:26 after canonical readbacks; point-in-time process observations.', 'No claim all Buddy branches, all background jobs, generic non-HTTP ownership, raw-SQL RLS or full fresh-install matrix.', 'Native profile uses privileged service role; owner checks are application-level.', 'No new production changes, test execution, runtime/browser/DB/task/git operations in this audit.'],
}
(OUT / 'evidence-window-excerpts.json').write_text(json.dumps(excerpts, indent=2) + '\n')
(OUT / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n')
(OUT / 'input-manifest.json').write_text(json.dumps({'files': [{'path': p, **v} for p, v in inputs.items()]}, indent=2) + '\n')
print(json.dumps({'verdict': summary['verdict'], 'hash_bound_inputs': len(inputs), 'warmed_flashcards_responses': len(warm_cards), 'post_health_notes_responses': len(post_notes), 'metadata_sessions': [len(m['sessions']) for m in metas], 'readback_requests': len(records)}, indent=2))
