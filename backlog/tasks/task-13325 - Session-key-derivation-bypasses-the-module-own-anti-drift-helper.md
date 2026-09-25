---
id: TASK-13325
title: Session key derivation bypasses the module own anti-drift helper
status: Done
assignee: []
created_date: '2026-09-22 04:56'
updated_date: '2026-09-23 23:17'
labels:
  - duplication
  - authnz
  - security
dependencies: []
references:
  - 'tldw_Server_API/app/core/AuthNZ/crypto_utils.py:142'
  - 'tldw_Server_API/app/core/AuthNZ/session_manager.py:459'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
crypto_utils.derive_hmac_key_candidates exists and its module docstring explicitly scopes it to "JWTService, APIKeyManager, CSRF, and SessionManager". session_manager.py:_derive_secret_key_candidates (459-505) re-implements it and disagrees on all four sub-decisions:
- salt: per-secret domain-separated plus a legacy fixed-salt candidate, vs one static global b"session_encryption_salt_v1"
- iterations: 100,000 vs 600,000
- source ordering: canonical SHA-256 pre-hashes SINGLE_USER_API_KEY "for parity with legacy logic", the copy feeds raw bytes
- memoization: canonical memoized with a documented rationale, copy not

The same file IMPORTS the canonical at :42-43 and uses it correctly at :884-886 - one file, both the adoption and the bypass. Drift is already visible: crypto_utils grew a production guard against the deterministic test fallback and a JWT_PUBLIC_KEY exclusion; session_manager carries its own separately-written comment about excluding public keys.

Adoption changes the derived Fernet keys, so the current derivation must be appended as a trailing rotation candidate - the mechanism already exists (_fernet_candidates, decrypt_token walks it).

Source: synthesis F24
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 session_manager uses derive_hmac_key_candidates
- [x] #2 Existing encrypted sessions still decrypt via a trailing rotation candidate
- [x] #3 Test asserts both paths derive the same key set
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ae073d8cb8. session_manager._derive_secret_key_candidates now returns urlsafe_b64(derive_hmac_key_candidates(settings)) first, then the old derivation (moved verbatim to _derive_legacy_secret_key_candidates: static salt, 600k rounds, raw secrets incl. JWT secrets in single-user mode) as trailing rotation candidates. Canonical ValueError (no secret configured) is caught -> [] since derived keys are only a fallback behind the persisted/explicit SESSION_ENCRYPTION_KEY. Found while writing AC#2's test: decrypt_token's candidate walk never worked -- Fernet.InvalidToken was not in the caught tuple, so the first non-matching candidate aborted; and both decrypt failure raises called InvalidSessionError(msg), which takes no args (TypeError). Fixed both; otherwise adoption would have logged out every session encrypted under a derived key. Tests: tests/AuthNZ/unit/test_session_manager_key_derivation.py, 4 tests, all 4 fail on ea1cbc6941 session_manager, pass now. AuthNZ/unit + AuthNZ_Unit before: 21 failed/7 errors/2389 passed; after: 21/7/2393, identical failure set. Bandit -ll clean. Docs: none needed (internal). Cost note: init still runs the 600k-round legacy PBKDF2 per secret, as before; drop _derive_legacy_secret_key_candidates once old derived-key sessions have expired.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Session key derivation now goes through crypto_utils.derive_hmac_key_candidates with the old derivation kept as trailing decrypt-only candidates. Also fixed the rotation walk itself (uncaught InvalidToken, bad InvalidSessionError constructor), without which the trailing candidates were unreachable.
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
