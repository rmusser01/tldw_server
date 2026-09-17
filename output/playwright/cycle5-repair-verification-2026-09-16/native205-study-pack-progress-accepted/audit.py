"""Audit retained UAT205 UI/transport artifacts without invoking native services."""
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
from datetime import datetime
from collections import Counter
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / '.tmp/uat198-181-native-20260917'
OUT = Path(__file__).resolve().parent
inputs = {}

def read(path):
    data = path.read_bytes()
    inputs[str(path.relative_to(ROOT))] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    return data.decode()

def obj(path):
    return json.loads(read(path))

def capture(name):
    text = read(BASE / ('alice205-' + name + '.txt'))
    return json.JSONDecoder().raw_decode(text[text.index('{'):])[0]

def body(event):
    value = event.get('body')
    return json.loads(value) if isinstance(value, str) else value

def route(event):
    return urlsplit(event.get('url', '')).path

def query(event):
    return parse_qs(urlsplit(event.get('url', '')).query)

def at(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))

task = next((ROOT / 'backlog/tasks').glob('task-13260.143 - *.md'))
assert 'native queued-to-terminal acceptance' in read(task)
source = obj(BASE / 'native205-source.json')
reviewed = obj(ROOT / '.tmp/uat205-repair-20260917/owned-manifest.json')
known = {r['path']: r for r in reviewed['paths']}
observed = {r['path']: r['sha256'] for r in source['files']}
drawer_path = 'apps/packages/ui/src/components/Flashcards/components/StudyPackCreateDrawer.tsx'
locale_path = 'apps/packages/ui/src/assets/locale/en/option.json'
for path, sha in observed.items():
    assert hashlib.sha256(read(ROOT / path).encode()).hexdigest() == sha
assert observed[drawer_path] == known[drawer_path]['sha256'] == '90b8e5a09e095c5d1c9e0ed4a73dc09362f89b34b66b163502d1877716dd85d4'
old_locale = obj(ROOT / known[locale_path]['snapshot'])
current_locale = obj(ROOT / locale_path)
locale_keys = ['studyPackJobAccepted', 'studyPackJobQueued', 'studyPackJobRunning', 'studyPackJobStatusUnavailable', 'studyPackResultUnavailable']
for key in locale_keys:
    assert current_locale['flashcards'][key] == old_locale['flashcards'][key]
read(ROOT / '.tmp/uat205-independent-20260917/REVIEW205.md')
worker = obj(ROOT / '.tmp/uat206-repair-20260917/owned-manifest.json')['files'][0]
runtime = obj(BASE / 'reviewed222-after-health.json')
assert {r['path']: r['sha256'] for r in runtime['files']}[worker['path']] == worker['sha256']

ui = {name: read(BASE / ('alice205-' + name + '.txt')) for name in ['create-ready', 'title', 'submit', 'pending', 'running', 'progress-later', 'reload-open', 'reload-ready', 'reload-settled']}
assert 'dialog "Create study pack"' in ui['create-ready']
assert "getByRole('textbox', { name: 'Title', exact: true }).fill('Alice UAT205 progress acceptance 20260917 0928')" in ui['title']
for name in ['submit', 'pending']:
    assert 'Study pack queued. Waiting for generation to start.' in ui[name]
    assert 'button "loading Create study pack" [disabled]' in ui[name]
assert 'status ' in ui['running'] and 'Creating your study pack.' in ui['running']
assert 'button "loading Create study pack" [disabled]' in ui['running']
assert 'await page.reload();' in ui['reload-open']
for name in ['progress-later', 'reload-settled']:
    assert '/flashcards?tab=review&deck_id=11' in ui[name]
    assert 'dialog "Create study pack"' not in ui[name]
    assert '2 cards remaining' in ui[name]
    assert 'Alice UAT205 progress acceptance 20260917 0928' in ui[name]

pending = capture('pending-events')
terminal = capture('progress-later-events')
reload = capture('reload-events')
window = [e for e in terminal['events'] if e['at'] >= '2026-09-17T09:26:00.000Z']
jobs_path = '/api/v1/flashcards/study-packs/jobs'
posts = [e for e in window if e['event'] == 'request' and e.get('method') == 'POST' and route(e) == jobs_path]
accepted = [e for e in window if e['event'] == 'response' and route(e) == jobs_path]
assert len(posts) == len(accepted) == 1
assert accepted[0]['status'] == 202 and body(accepted[0])['job']['id'] == 6 and body(accepted[0])['job']['status'] == 'queued'
intent = body(posts[0])
assert intent['title'] == 'Alice UAT205 progress acceptance 20260917 0928'
assert intent['deck_mode'] == 'new' and len(intent['source_items']) == 1 and intent['source_items'][0]['source_type'] == 'note'
identity = next(e for e in reversed(window) if e['event'] == 'identity' and e['at'] < posts[0]['at'])
assert identity['status'] == 200 and identity['identity']['id'] == 2
polls = [e for e in window if e['event'] == 'response' and route(e) == jobs_path + '/6']
assert all(e['status'] == 200 and body(e)['job']['id'] == 6 and body(e)['error'] is None for e in polls)
states = [body(e)['job']['status'] for e in polls]
assert list(dict.fromkeys(states)) == ['queued', 'running', 'completed']
assert Counter(states) == {'queued': 8, 'running': 23, 'completed': 1}
completed = polls[-1]
pack = body(completed)['study_pack']
assert pack['id'] == 3 and pack['deck_id'] == 11 and pack['client_id'] == '2'
assert pack['title'] == intent['title']
assert not [e for e in window if route(e) == jobs_path + '/5']
assert not [e for e in window if e['event'] == 'response' and e['status'] >= 400]

# Captured page body is still queued between the settled initial poll and next GET.
next_poll = next(e for e in window if e['event'] == 'request' and route(e) == jobs_path + '/6' and e['at'] > polls[0]['at'])
assert polls[0]['at'] < pending['at'] < next_poll['at']
assert 'Study pack queued. Waiting for generation to start.' in pending['body']
post_complete = [e for e in window if e['event'] == 'response' and e['at'] > completed['at']]
deck_response = next(e for e in post_complete if route(e) == '/api/v1/flashcards/decks')
deck = next(d for d in body(deck_response) if d['id'] == 11)
assert deck['name'] == intent['title'] and deck['client_id'] == '2'
cards_response = next(e for e in post_complete if route(e) == '/api/v1/flashcards' and query(e).get('deck_id') == ['11'] and query(e).get('due_status') == ['all'])
cards = body(cards_response)['items']
assert cards_response['status'] == 200 and len(cards) == 2 and all(c['deck_id'] == 11 and c['client_id'] == '2' for c in cards)
assert urlsplit(terminal['url']).path == '/flashcards' and parse_qs(urlsplit(terminal['url']).query) == {'tab': ['review'], 'deck_id': ['11']}
assert parse_qs(urlsplit(reload['url']).query) == {'tab': ['review'], 'deck_id': ['11']}
assert next(e for e in reversed(reload['events']) if e['event'] == 'identity')['identity']['id'] == 2
# The cumulative observer is restored after reload; do not invent uncaptured resource GETs.
new_reload_resource_responses = [e for e in reload['events'] if e['event'] == 'response' and e['at'] > terminal['at'] and route(e) in ['/api/v1/flashcards', '/api/v1/flashcards/decks']]
assert not new_reload_resource_responses
assert len([e for e in reload['events'] if e['event'] == 'request' and e.get('method') == 'POST' and route(e) == jobs_path and e['at'] >= posts[0]['at']]) == 1

def event_excerpt(e):
    value = {'at': e['at'], 'event': e['event'], 'path': route(e)}
    for k in ['status', 'method']:
        if k in e:
            value[k] = e[k]
    if e in polls or e in accepted:
        value['job_id'] = body(e)['job']['id']
        value['job_status'] = body(e)['job']['status']
    return value

excerpts = {'creation': [event_excerpt(e) for e in posts + accepted], 'polls': [event_excerpt(e) for e in polls], 'canonical_success_reads': [event_excerpt(deck_response), event_excerpt(cards_response)], 'ui': {'queued': 'Study pack queued. Waiting for generation to start.', 'running': 'Creating your study pack.', 'submit_disabled_in_queued_and_running_snapshots': True, 'queued_gap_capture_at': pending['at'], 'next_poll_request_at': next_poll['at'], 'terminal_capture_at': terminal['at'], 'terminal_route': '/flashcards?tab=review&deck_id=11', 'reload_capture_at': reload['at'], 'reload_settled_route': '/flashcards?tab=review&deck_id=11', 'visible_cards': 2}}
summary = {'task': 'TASK13260.143', 'verdict': 'Native queued-to-terminal AC3 supported; recommend bounded205 closure with previously reviewed automated failure/account controls.', 'source': {'native_observed_at': source['at'], 'native_receipt_revision': source['revision'], 'files': source['files'], 'drawer_matches_reviewed205': True, 'five205_locale_values_unchanged': True, 'backend_runtime_revision': runtime['revision'], 'reviewed206_startup_hash_matches': True}, 'native': {'actor': 2, 'create_post_count_in_window': len(posts), 'job_id': 6, 'accepted_at': accepted[0]['at'], 'status_counts': Counter(states), 'first_running_at': polls[8]['at'], 'completed_at': completed['at'], 'elapsed_acceptance_to_completion_seconds': (at(completed['at'])-at(accepted[0]['at'])).total_seconds(), 'pack_id': 3, 'deck_id': 11, 'card_ids': [c['uuid'] for c in cards], 'same_selected_deck_after_actual_reload': True, 'new_post_reload_resource_get_captured': False, 'original_job5_not_requested_in_window': True}, 'limits': ['Source receipt is after this UI capture; exact drawer bytes match reviewed205 and relevant locale values match. No interpreter/bundle memory inspection.', 'Queued/running snapshots and actual-hook tests support polling-gap protection; no continuous DOM or screen-reader test.', 'No native failed/cancelled/unusable-result/account-switch journey; those remain separately attributed to prior automated controls.', 'Actual reload UI reselects deck11 and shows two cards, but observer has no new direct deck/card GET after reload.', 'No claim of exact requested card count, model quality, inference request count, full matrix or persistent pending-job recovery across reload.', '206 worker startup is a separate repair; this audit changes no flag/runtime/config and proves no general worker availability policy.']}
(OUT / 'evidence-window-excerpts.json').write_text(json.dumps(excerpts, indent=2) + '\n')
(OUT / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n')
(OUT / 'input-manifest.json').write_text(json.dumps({'files': [{'path': p, **v} for p, v in inputs.items()]}, indent=2) + '\n')
print(json.dumps({'verdict': summary['verdict'], 'hash_bound_inputs': len(inputs), 'creation_posts': len(posts), 'poll_states': Counter(states), 'job': 6, 'pack': 3, 'deck': 11, 'visible_cards': 2}, indent=2))
