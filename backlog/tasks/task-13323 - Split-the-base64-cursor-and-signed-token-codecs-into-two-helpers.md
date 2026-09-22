---
id: TASK-13323
title: Split the base64 cursor and signed-token codecs into two helpers
status: To Do
assignee: []
created_date: '2026-09-22 04:56'
labels:
  - duplication
  - security
  - migration
dependencies: []
references:
  - 'tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py:128'
  - 'tldw_Server_API/app/core/Sync/v2/service.py:10488'
  - 'tldw_Server_API/app/core/AuthNZ/api_key_crypto.py:118'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
23 production decode sites across 22 files, in TWO notational spellings (-len(x) % 4 and (4 - len(x) % 4) % 4), so a grep-based fix will miss sites.

Strictness has diverged three ways with no owner: 5 strict (validate=True + altchars), 2 canonicalization-checked (notes.py re-encodes and compares), 16 lax (bare decode, which silently DISCARDS out-of-alphabet characters rather than raising). The endpoints layer adds SIX error contracts for a malformed cursor: 400 in three places, 413-or-400 in one, HTTP 500 in one, and silently-ignore-and-restart-from-page-1 in two (workflows.py:2192, :2593).

THE DESTINATION MUST BE TWO HELPERS. The sites split into opaque pagination cursors and HMAC-signed capability tokens. A single flattened codec that grows a verify=False default is worse than the duplication.
Promote mcp_unified_endpoint.py:128-158 for the opaque form. Promote Sync/v2/service.py:10488-10530 for the signed form - it is the reference implementation in the repo (pre-decode size bound, validate=True, explicit altchars, post-decode bound, version check, hmac.compare_digest, TTL/skew bounds). By contrast api_key_crypto.py:_b64decode, same trust class, has neither validate=True nor any length bound.

Source: synthesis F22
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Two separate helpers exist; the signed one cannot be used without signature verification
- [ ] #2 api_key_crypto gains a length bound and alphabet validation
- [ ] #3 Malformed cursors produce one documented status, not six
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
