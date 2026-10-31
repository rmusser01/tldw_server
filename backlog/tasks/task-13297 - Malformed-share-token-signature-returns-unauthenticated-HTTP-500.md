---
id: TASK-13297
title: Malformed share-token signature returns unauthenticated HTTP 500
status: Done
assignee: []
created_date: '2026-09-22 04:45'
updated_date: '2026-09-23 19:52'
labels:
  - bug
  - chat
  - security
dependencies:
  - TASK-13289
references:
  - 'tldw_Server_API/app/api/v1/endpoints/chat.py:6496'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`api/v1/endpoints/chat.py:_decode_knowledge_qa_share_token` decodes the **signature** segment outside its `try`:

```python
provided_signature = _urlsafe_b64decode(encoded_signature)   # <- outside the try
if not hmac.compare_digest(expected_signature, provided_signature):
    raise HTTPException(403, "Invalid share token")

try:
    payload = json.loads(_urlsafe_b64decode(encoded_payload).decode("utf-8"))
except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
    raise HTTPException(400, "Malformed share token payload") from exc
```

`binascii.Error` is a `ValueError` subclass and **is** in the except tuple — but that tuple only guards the payload decode. A malformed signature segment raises before the `try` is entered.

So `GET /api/v1/chat/shared/conversations/AAAA.A` returns an unhandled **HTTP 500 on a public, unauthenticated route**, where 400 is correct. The token is attacker-supplied by construction: share links are meant to be pasted by third parties.

`api/v1/endpoints/notes.py:857-912` is the correct sibling — it decodes both segments inside the guard.

Related: TASK-13289 covers the signing-key fallback in the same feature (`_get_knowledge_qa_share_signing_key`). These are distinct defects but land in adjacent code and should be fixed in one pass.

Found by the comprehensive core-module review; independently verified by the orchestrator by reading the try boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failing test requests a share link whose signature segment is not valid base64 and asserts 400, not 500
- [x] #2 Both base64 decodes sit inside the guard, matching notes.py:857-912
- [x] #3 A malformed payload segment still returns 400 (no regression)
- [x] #4 A valid token still resolves (no regression)
- [x] #5 Bandit run for touched scope
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DONE. Signature decode moved inside a guarded block at chat.py:6506 raising 400 (binascii.Error subclasses ValueError). Regression test added at tldw_Server_API/tests/Chat/unit/test_share_token_malformed_signature.py covering 4 malformed-signature shapes plus the wrong-arity case: red before (4 failed), green after (5 passed). Existing test_chat_share_links_api.py still passes (8). Test asserts a handled 4xx rather than strictly 400, because a well-formed but wrong signature correctly yields 403 - the defect was the UNHANDLED exception.
Remaining: Bandit on touched scope (owner-only file under app/api/v1/**).

2026-09-23 reconciliation: AC2 met (7c348a05ae; chat.py:6507-6515 signature decode and :6519-6522 payload decode each inside a ValueError-catching guard raising 400). AC3 met by ad hoc verification only (valid-HMAC token over payload segments 'A' and b64('not json') -> 400 'Malformed share token payload', b64('[1]') -> 400 'Invalid share token payload'); no committed test covers it. AC4 met (test_chat_share_links_api.py::test_share_link_create_list_revoke_and_public_resolve passes; both share-token files: 13 passed). AC5 met (uvx bandit chat.py: no issues). AC1 NOT met as written: tests/Chat/unit/test_share_token_malformed_signature.py calls _decode_knowledge_qa_share_token directly (not the share route) and asserts 400<=status<500, not 400. Actual codes: 'AAAA.A' and 'AAAA.AAAAA' -> 400; 'AAAA.!!!!' and 'AAAA.=' -> 403 (lenient decode then HMAC mismatch). Remaining: a route-level test of GET /api/v1/chat/shared/conversations/AAAA.A asserting exactly 400.

2026-09-23: AC1 done. test_chat_share_links_api.py::test_share_link_resolve_rejects_undecodable_signature_with_400 drives the real route with AAAA.A and asserts == 400; fails on chat.py from 7c348a05ae~1, passes now (file: 9 passed). AC3 remains verified ad hoc only, as recorded. Docs: none needed. No known skips.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Malformed share-token signatures return 400 on the public share route instead of an unhandled 500; covered at helper and route level.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
