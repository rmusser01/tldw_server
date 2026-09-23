---
id: TASK-13323
title: Split the base64 cursor and signed-token codecs into two helpers
status: In Progress
assignee: []
created_date: '2026-09-22 04:56'
updated_date: '2026-09-23 19:39'
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
- [x] #2 api_key_crypto gains a length bound and alphabet validation
- [ ] #3 Malformed cursors produce one documented status, not six
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
HELPERS LANDED, three sites migrated, 20 remain.

NEW: core/Utils/base64url.py with TWO decode entry points, deliberately not one:
  decode_opaque_cursor_segment - bounded, alphabet-validated (validate=True + altchars). Non-canonical encodings accepted; nothing keys off the cursor string.
  decode_signed_token_segment - the same, PLUS canonical-form enforcement, so two distinct strings cannot decode to the same signed bytes.
Neither verifies a signature; that stays with the caller holding the key. The naming is the guard: a caller cannot reach for the unsigned path by accident, which a single helper with a verify=False default would have allowed.
Base64SegmentError subclasses ValueError, so existing except ValueError/binascii.Error handlers at the call sites still catch it - verified before migrating.

SECURITY FIX AT THE SAME TIME (finding authnz-11). core/AuthNZ/api_key_crypto.py:
- _b64decode now uses the strict signed decoder. It was a bare urlsafe_b64decode with NO validate and NO length bound, so a corrupted segment decoded to different valid bytes instead of raising.
- verify_kdf_hash called pbkdf2_hmac(..., dklen=len(expected)) OUTSIDE its try. A stored hash of the form "scheme$iters$salt$" splits cleanly into four parts, decodes to b"", and dklen=0 RAISES - so the function escaped as an exception where both callers (key_resolution.py:93, api_key_manager.py:526) expect a bool, giving 500 instead of 401 for every request presenting that key. Now guarded explicitly and the KDF call is inside a try.
- iterations are bounded (1..10M): an unbounded count was a hang rather than a False.
Test: tests/AuthNZ/unit/test_api_key_kdf_hash_robustness.py, 11 cases. Red before (6 failed), green after. 111 passed across the 7 suites touching api_key_crypto.

ALSO MIGRATED:
- MCP prompts_catalog.py - was LAX; gains alphabet validation. 87 passed.
- chacha shared_workspace_chat_store.py - was already strict; migrated for consistency.

Tests: tests/Utils/test_base64url_codecs.py, 17 cases, including one that DEMONSTRATES the defect the 16 lax copies carry by calling the stdlib the way they do and showing it silently accepts a tampered segment.

STILL OPEN: 20 of the 23 sites, 7 of them owner-only under app/api/v1/**. Two notational spellings exist in the wild, so a grep-based sweep will miss sites - enumerate by AST.

2026-09-23 reconciliation:
AC2 met - commit dddcbd3469: api_key_crypto._b64decode now calls decode_signed_token_segment(max_encoded_len=_MAX_HASH_SEGMENT_LEN=512) (validate=True + altchars, pre-decode length bound, canonical form); tests/AuthNZ/unit/test_api_key_kdf_hash_robustness.py covers out-of-alphabet and empty segments. With tests/Utils/test_base64url_codecs.py: 28 passed.
AC1 NOT met - two helpers exist (core/Utils/base64url.py: decode_opaque_cursor_segment, decode_signed_token_segment), but the second clause fails by design: decode_signed_token_segment does not verify any signature (its docstring says so) and can be called with no key. Only naming separates them. Meeting the AC needs a signed-token helper that takes the key and does hmac.compare_digest itself (Sync/v2/service.py:10488-10530 pattern), or an AC amendment if naming is accepted as the guard.
AC3 NOT met - only 3 of 23 sites migrated (api_key_crypto, MCP prompts_catalog, chacha shared_workspace_chat_store). endpoints/workflows.py:2194 and :2597 still use bare urlsafe_b64decode and the six divergent malformed-cursor contracts are untouched; no single documented status exists.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
