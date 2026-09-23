---
id: TASK-13289
title: Knowledge-QA share links are forgeable when HMAC key derivation fails
status: Done
assignee: []
created_date: '2026-09-22 04:34'
updated_date: '2026-09-23 23:09'
labels:
  - security
  - chat
  - bug
dependencies: []
references:
  - 'tldw_Server_API/app/api/v1/endpoints/chat.py:6463'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`_get_knowledge_qa_share_signing_key()` at `tldw_Server_API/app/api/v1/endpoints/chat.py:6463-6471` resolves the share-link signing key in three steps. The final fallback is a **hardcoded literal published in this open-source repository**:

```python
explicit = (os.getenv("KNOWLEDGE_QA_SHARE_LINK_SECRET") or "").strip()
if explicit: return explicit.encode("utf-8")
try:
    return derive_hmac_key()
except _CHAT_ENDPOINT_NONCRITICAL_EXCEPTIONS:
    fallback = (os.getenv("JWT_SECRET_KEY") or "knowledge_qa_share_link_default")
    return fallback.encode("utf-8")
```

If `KNOWLEDGE_QA_SHARE_LINK_SECRET` is unset, `derive_hmac_key()` raises one of the broad noncritical exception types, and `JWT_SECRET_KEY` is unset, then every share token is signed with `knowledge_qa_share_link_default`. Anyone reading this repo can then mint a token that `_build_knowledge_qa_share_token` (`:6483`) and its verifier accept.

The key is memoized with `@lru_cache(maxsize=1)`, so a single transient `derive_hmac_key()` failure at first use pins the weak key for the remaining process lifetime.

Verified: the literal appears exactly once in the codebase, at `chat.py:6470`, and is the `or` fallback — not a test fixture.

Found by the comprehensive core-module review (AuthNZ reviewer, flagged cross-scope; independently verified by the orchestrator).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failing test proves a token signed with the literal fallback is currently accepted by the verifier
- [x] #2 Key derivation failure is fatal for share-link minting rather than silently downgraded: the endpoint returns an error instead of issuing a weakly-signed token
- [x] #3 No hardcoded signing-key literal remains anywhere in the share-link path
- [x] #4 The lru_cache does not pin a degraded key for the process lifetime after one transient failure
- [x] #5 Existing share tokens signed with a legitimate key still verify, or the rotation/invalidation is documented
- [x] #6 Bandit run for touched scope
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fix commit 598ecd89b8. _get_knowledge_qa_share_signing_key no longer falls back to JWT_SECRET_KEY or the literal 'knowledge_qa_share_link_default'; a derive_hmac_key() failure now raises HTTP 503 for both minting (create/list share links) and resolving. lru_cache does not cache exceptions, so a transient failure is retried on the next call (test_transient_failure_is_not_pinned). verify_signed_token in core/Utils/base64url.py already rejects a missing/empty key (TypeError) and needed no change; the weakness was the key source, not the verifier.
Regression test tldw_Server_API/tests/Chat/unit/test_share_link_signing_key_fail_closed.py: 3 tests, all FAIL on ea1cbc6941 (DID NOT RAISE: forged literal-key token accepted, weak token minted, degraded key pinned) and pass after.
tests/Chat/unit: before 4 failed/1730 passed (3 new red + pre-existing test_chat_helpers::TestLoadConversationHistory::test_load_history); after 1 failed/1733 passed (same pre-existing failure only).
AC5: tokens signed with KNOWLEDGE_QA_SHARE_LINK_SECRET or derive_hmac_key() keep verifying (unchanged key path; existing share-link tests pass). Tokens minted while the degraded fallback was active stop verifying - intended invalidation, documented in the commit message.
Bandit (uvx bandit -q -ll chat.py): no findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Share-link signing now fails closed: no hardcoded or JWT-secret fallback key, 503 on derivation failure, transient failures not pinned by lru_cache. Regression tests red-before/green-after; Chat unit suite shows no new failures.
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
