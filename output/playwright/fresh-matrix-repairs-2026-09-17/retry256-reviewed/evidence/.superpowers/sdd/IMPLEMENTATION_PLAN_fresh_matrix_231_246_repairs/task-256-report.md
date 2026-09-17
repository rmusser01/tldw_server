# Task 256: Retry Identity Repair Report

## Change

The ordinary explicit Retry completed-tail guard uses matching valid client correlation IDs to identify an answered replay. A valid, different ID may persist a new identical question after a strict model-selection 400 prevented its first attempt from being saved. Missing, malformed, or legacy saved IDs retain the existing conservative content-based 409 behavior. Unanswered and error-tail logic was not changed.

The endpoint regression is parameterized for real SQLite and the repository's official PostgreSQL fixture. It proves this sequence: successful question A; exact repeated question B rejected with `model_not_available` before persistence; explicit Retry for B succeeds with a second canonical user/assistant pair and canonical acknowledgement; a later Retry for B receives 409. The controlled provider callback checks that both equal user turns reach the model payload while metadata does not.

## Verification

- Causal RED: both SQLite and real PostgreSQL reached the old completed-tail 409. The PostgreSQL RED used a narrow prepared-revision overlay and restored the repaired source unconditionally.
- GREEN: the official dual-backend causal endpoint test passed (`2 passed`); focused retry controls passed 117 tests; image-recovery controls passed 31 tests, and the official PostgreSQL strict-image snapshot passed separately.
- Ruff, compile, production Bandit, and `git diff --check` passed. Full details, command receipts, exit codes, static findings, and hashes are in `.tmp/uat-repairs-231-246/retry256/verification.md`.

## Remaining Concern

The initial PostgreSQL setup errors were caused by sandbox network isolation; elevated official-fixture verification corrected that result. Parent review and native acceptance remain required.

## Review Round 1 Follow-up

Added actual SQLite/PostgreSQL answered-tail controls for matching IDs, absent request ID, malformed request ID, and malformed persisted legacy ID. All four fail closed with no added canonical row and no Retry provider call. The official control run passed 8 cases and the combined causal plus control run passed 10. No production code changed in this follow-up.
