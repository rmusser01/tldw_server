---
id: TASK-13297
title: Malformed share-token signature returns unauthenticated HTTP 500
status: Done
assignee: []
created_date: '2026-09-22 04:45'
updated_date: '2026-09-23 00:12'
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

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed test-first and merged to dev in PR #2980 (merge commit 8045fa2956). A failing test reproduced the defect before any code changed, with controls pinning the behaviour that had to stay unchanged. Qodo review then found follow-on defects in three of this batch's fixes; those were corrected in the same PR before merge.
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
