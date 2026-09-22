---
id: TASK-13325
title: Session key derivation bypasses the module own anti-drift helper
status: To Do
assignee: []
created_date: '2026-09-22 04:56'
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
- [ ] #1 session_manager uses derive_hmac_key_candidates
- [ ] #2 Existing encrypted sessions still decrypt via a trailing rotation candidate
- [ ] #3 Test asserts both paths derive the same key set
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
