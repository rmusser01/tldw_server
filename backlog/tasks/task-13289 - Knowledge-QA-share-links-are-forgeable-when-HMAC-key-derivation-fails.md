---
id: TASK-13289
title: Knowledge-QA share links are forgeable when HMAC key derivation fails
status: Done
assignee: []
created_date: '2026-09-22 04:34'
updated_date: '2026-09-22 14:28'
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
- [ ] #1 A failing test proves a token signed with the literal fallback is currently accepted by the verifier
- [ ] #2 Key derivation failure is fatal for share-link minting rather than silently downgraded: the endpoint returns an error instead of issuing a weakly-signed token
- [ ] #3 No hardcoded signing-key literal remains anywhere in the share-link path
- [ ] #4 The lru_cache does not pin a degraded key for the process lifetime after one transient failure
- [ ] #5 Existing share tokens signed with a legitimate key still verify, or the rotation/invalidation is documented
- [ ] #6 Bandit run for touched scope
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed test-first and merged to dev in PR #2980 (merge commit 8045fa2956). A failing test reproduced the defect before any code changed, with controls pinning the behaviour that had to stay unchanged. Qodo review then found follow-on defects in three of this batch's fixes; those were corrected in the same PR before merge.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
