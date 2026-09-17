"""Bounded verification of retained UAT211 browser receipts; no live access."""
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / '.tmp/uat198-181-native-20260917'
OUT = Path(__file__).resolve().parent
NAMES = [
    'alice211-answer.txt', 'alice211-citrine-cancel.txt', 'alice211-citrine-editor.txt',
    'alice211-cram-filter.txt', 'alice211-cram-open.txt', 'alice211-cram-ready.txt',
    'alice211-cram-schedule.txt', 'alice211-editor-cancel.txt', 'alice211-final-events.txt',
    'alice211-flashcards-open.txt', 'alice211-flashcards-settled.txt', 'alice211-full-reload.txt',
    'alice211-good-rating.txt', 'alice211-greeting-editor.txt', 'alice211-greeting-tag-added.txt',
    'alice211-greeting-tag-input.txt', 'alice211-greeting-tag-saved.txt',
    'alice211-manage-after-rating.txt', 'alice211-manage-open.txt', 'alice211-rating-events.txt',
    'alice211-reload-events.txt', 'alice211-reload-settled.txt', 'alice211-reloaded-editor.txt',
    'alice211-study-open.txt',
]


def result(name):
    text = (BASE / name).read_text()
    return json.loads(text.split('### Result\n', 1)[1].split('\n### Ran', 1)[0])


def body(event):
    v = event.get('body')
    return json.loads(v) if isinstance(v, str) else v


def digest(path):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


original = '37b10bd7-edf4-4f35-83c1-1490115d8c55'
greeting = '082ec63a-124e-4c47-ac23-c11a2deba70b'
events = result('alice211-final-events.txt')['events']
events = [e for e in events if e.get('at', '') >= '2026-09-17T07:42:00']
requests = [e for e in events if e.get('event') == 'request']
responses = [e for e in events if e.get('event') == 'response']
rating_requests = [e for e in requests if e.get('method') == 'POST'
                   and urlparse(e.get('url', '')).path == '/api/v1/flashcards/review']
assert len(rating_requests) == 1
submitted = body(rating_requests[0])
assert submitted['card_uuid'] == greeting and submitted['rating'] == 3
assert submitted['review_context']['tag_filter'] == 'uat211-greeting'
assert submitted['review_context']['review_mode'] == 'cram'
rating_responses = [e for e in responses if urlparse(e.get('url', '')).path == '/api/v1/flashcards/review']
assert len(rating_responses) == 1 and rating_responses[0]['status'] == 200
rated = body(rating_responses[0])
assert rated['uuid'] == greeting and rated['version'] == 3 and rated['repetitions'] == 1
assert rated['interval_days'] == 0 and rated['queue_state'] == 'learning'
gap = (datetime.fromisoformat(rated['due_at']) - datetime.fromisoformat(rated['last_reviewed_at'])).total_seconds()
assert gap == 600
patches = [e for e in requests if e.get('method') == 'PATCH' and '/api/v1/flashcards/' in e.get('url', '')]
assert len(patches) == 1 and urlparse(patches[0]['url']).path.endswith('/' + greeting)
assert body(patches[0])['expected_version'] == 1 and body(patches[0])['tags'] == ['uat211-greeting']
assert not [e for e in requests if e.get('method') in {'POST', 'PUT', 'PATCH', 'DELETE'}
            and original in e.get('url', '')]
lists = [e for e in responses if urlparse(e.get('url', '')).path == '/api/v1/flashcards'
         and e.get('status') == 200]
original_rows = [x for e in lists for x in body(e)['items'] if x['uuid'] == original]
assert original_rows and all(x['version'] == 2 and x['repetitions'] == 1 for x in original_rows)
post_reload = [e for e in lists if e['at'] >= '2026-09-17T07:54:17']
assert len(post_reload) == 2
for e in post_reload:
    cards = {x['uuid']: x for x in body(e)['items']}
    assert len(cards) == 5 and original in cards and greeting in cards
    assert cards[original]['version'] == 2 and cards[original]['repetitions'] == 1
    assert cards[greeting]['version'] == 3 and cards[greeting]['repetitions'] == 1
    assert cards[greeting]['due_at'] == rated['due_at']
    assert cards[greeting]['last_reviewed_at'] == rated['last_reviewed_at']
    assert cards[greeting]['tags'] == ['uat211-greeting']
    new = [x for x in cards.values() if x['queue_state'] == 'new']
    assert len(new) == 3 and all(x['last_reviewed_at'] is None and x['due_at'] is None for x in new)
preview = [e for e in lists if 'uat211-greeting' in e['url'] and e['at'] < rating_requests[0]['at']]
assert preview
assert any(len(body(e)['items']) == 1 and body(e)['items'][0]['uuid'] == greeting
           and body(e)['items'][0]['next_intervals']['good'] == '10 min' for e in preview)
assert "flashcards-review-rate-3').click()" in (BASE / 'alice211-good-rating.txt').read_text()
assert 'Saved. Next review in 10 minutes (next review gap: 10 minutes).' in (BASE / 'alice211-good-rating.txt').read_text()
assert 'await page.reload();' in (BASE / 'alice211-full-reload.txt').read_text()
assert "name: 'Cancel' }).click()" in (BASE / 'alice211-citrine-cancel.txt').read_text()
for name in ('alice211-citrine-editor.txt', 'alice211-reloaded-editor.txt'):
    s = (BASE / name).read_text()
    assert 'Next review gap' in s and '- text: 10 min' in s
settled = (BASE / 'alice211-reload-settled.txt').read_text()
assert settled.count('Next gap 10 min') == 2
assert settled.count('Next gap —') == 3
assert 'switch [checked]' in (BASE / 'alice211-answer.txt').read_text()
assert '10 min' in (BASE / 'alice211-answer.txt').read_text()
facts = {
    'window_start': '2026-09-17T07:42:00Z',
    'final_capture': result('alice211-final-events.txt')['at'],
    'greeting_id': greeting, 'preserved_original_id': original,
    'tag_patch': {'at': patches[0]['at'], 'method': 'PATCH', 'expected_version': 1, 'tags': ['uat211-greeting']},
    'rating_request': {'at': rating_requests[0]['at'], 'body': submitted},
    'rating_response': {'at': rating_responses[0]['at'], 'status': 200, 'body': rated},
    'computed_gap_seconds': gap,
    'post_reload_lists': [{'at': e['at'], 'status': e['status'], 'cards': [
        {k: x.get(k) for k in ('uuid', 'version', 'repetitions', 'queue_state', 'interval_days',
                              'due_at', 'last_reviewed_at', 'tags')} for x in body(e)['items']]} for e in post_reload],
    'ui': {'toast_gap': '10 minutes', 'manage_gap': '10 min', 'editor_gap': '10 min', 'new_card_gap': '—'},
    'assertions': 'passed',
}
(OUT / 'verified-facts.json').write_text(json.dumps(facts, indent=2) + '\n')
(OUT / 'input-manifest.json').write_text(json.dumps({
    'captured_at': datetime.now(timezone.utc).isoformat(),
    'scope': '24 allowlisted alice211-*.txt native receipts and one official CLI task snapshot',
    'normalization': 'none; original bytes hashed; native inputs unchanged',
    'inputs': [digest(BASE / name) for name in NAMES],
    'task_snapshot': digest(OUT / 'task13260.149.txt'),
}, indent=2) + '\n')
print(json.dumps({'assertions': 'passed', 'inputs': len(NAMES), 'gap_seconds': gap,
                  'rating_requests': len(rating_requests), 'post_reload_lists': len(post_reload)}))
