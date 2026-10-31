---
id: TASK-13323
title: Split the base64 cursor and signed-token codecs into two helpers
status: Done
assignee: []
created_date: '2026-09-22 04:56'
updated_date: '2026-09-23 20:53'
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
- [x] #1 Two separate helpers exist; the signed one cannot be used without signature verification
- [x] #2 api_key_crypto gains a length bound and alphabet validation
- [x] #3 Malformed cursors produce one documented status, not six
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

2026-09-23 completion (commits 2284010c6c, 85d613ebb5, fe515313ae on top of 0eacb750fd):

AC1 MET - 2284010c6c: core/Utils/base64url.py gains verify_signed_token(token, key, *, max_encoded_len, sign_encoded_payload). It REQUIRES a non-empty bytes key (TypeError otherwise), bounds the whole token pre-decode, decodes both segments strictly (validate=True + altchars, canonical form), and returns the payload only after hmac.compare_digest; mismatch raises SignatureMismatchError (subclass of Base64SegmentError/ValueError). decode_signed_token_segment is gone; the strict single-segment decoder is renamed decode_canonical_segment and documented as NOT for tokens (its only caller, api_key_crypto, decodes stored KDF salt/digest, not a signed token).
Migrated to verify_signed_token: endpoints/chat.py share tokens (400 malformed / 403 mismatch unchanged; sign_encoded_payload=True), core/Chat/chat_loop_approval.py (HEAD accepted "<token>!!!!" as valid - test red on HEAD with (True, None); non-object payload no longer escapes as AttributeError), endpoints/notes.py attachment cursor, core/Notes_Graph/suggestion_api.py cursor codec.

AC3 MET - fe515313ae: one contract, 400 detail "Invalid cursor", documented in Docs/API-related/Pagination_Cursors.md (linked from API_README.md).
- workflows.py runs list + run events: silent restart-from-page-1 -> 400 (decode_opaque_cursor_segment).
- notes.py attachment cursor: 413-or-400 -> 400 "Invalid cursor".
- Notes_Graph suggestion cursor: 422 -> 400 (structured code notes_graph_cursor_invalid kept; frontend keys on code).
- audio_history, audio_jobs: already 400; bare urlsafe_b64decode replaced (it silently repaired "<cursor>!!!!").
- The former HTTP 500 site was the chat share token, already fixed to 400 before this pass; now covered by verify_signed_token plus a roundtrip/tamper test.

Tests (red on HEAD, green after): tests/Utils/test_base64url_codecs.py (wrong key, tampered payload, no key, malformed, non-canonical, oversized, encoded-payload MAC); tests/Workflows/test_malformed_cursor_400.py (5 red -> green); tests/AudioJobs/test_audio_cursor_strictness.py (2 red -> green); tests/Chat_NEW/unit/test_chat_loop_approval.py (2 red -> green); tests/Notes/test_notes_attachment_sync_api.py and tests/Notes_Graph/unit/test_suggestion_api.py updated from 413/422 to 400 (red on HEAD).
Suite runs after: Utils+Chat/unit+Notes+AudioJobs+TTS history+AuthNZ/unit 3979 passed, 15 failed, 7 errors - all 22 also fail with base-commit sources (Postgres/env, pre-existing). Workflows 1593 passed, 4 failed - same 4 fail on HEAD (arxiv lib missing, webhook tests). Notes_Graph 639 passed.
Bandit: uvx bandit -q -ll on all touched sources - rc 0, no findings.

OUT OF SCOPE (not pagination cursors / different token formats), left as is:
- Sync/v2/service.py _decode_pull_token_segment: already strict reference impl; sync protocol token with its own codes (sync_pull_token_invalid / sync_pull_token_too_large=413); not a list-endpoint cursor.
- mcp_unified_endpoint.py _parse_safe_config_query: base64url JSON config param, already strict, 400.
- Prototype_Workspaces/access.py resume cookie: 3-part "ptca.<p>.<s>" format, compares encoded signatures.
- Local_LLM/llamacpp_snapshot_operations.py, services/admin_data_ops_service.py: hexdigest signatures over the encoded body - different token format.
- services/admin_system_service.py: JWT header/payload inspection, not verification.
- AuthNZ/session_manager.py: Fernet ciphertext, not base64 segments.
- MCP filesystem_receipts.py, services/connectors_worker.py (Gmail API bodies), Character_Chat character_io.py (card import), Visual_Identities/source_context.py (heuristic), Slides/standalone_html_registry.py (keyring secret config): not cursors or signed tokens.
- Core cursors already strict (validate=True + altchars): Workspaces membership_models / file_inventory_models, Notes_Graph graph_service x2, media_db email_search_cursor. Their HTTP status mapping was not re-audited in this pass; follow-up if a full-API cursor sweep is wanted.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Two helpers in core/Utils/base64url.py: decode_opaque_cursor_segment for opaque cursors and verify_signed_token, which requires the key and checks hmac.compare_digest before returning a payload (decode_signed_token_segment removed; strict stored-value decoder renamed decode_canonical_segment). Chat share tokens, chat-loop approval tokens (which accepted junk-appended tokens), notes attachment cursors and notes graph suggestion cursors now verify through it. Every malformed pagination cursor on workflows runs/events, audio history, audio jobs, notes attachments and notes graph suggestions returns 400 "Invalid cursor" (was silent restart, 413, 422), documented in Docs/API-related/Pagination_Cursors.md. api_key_crypto hardened earlier (dddcbd3469). Non-cursor, differently formatted token sites are listed in the notes as out of scope.
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
