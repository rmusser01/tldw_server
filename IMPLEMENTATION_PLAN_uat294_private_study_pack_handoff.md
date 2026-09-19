# UAT294: keep Study Pack source details out of account-switch history

Tracking: TASK13260.231. The requester authorized repairing the UAT findings before another full matrix. Native PostgreSQL and SQLite multi-user evidence reproduces private Note title/source-ID disclosure through Back and reload. The current URL builder and parser bypass the existing private Flashcards transfer mechanism.

## Stage 1: reproduce the privacy failure
**Goal:** Protect the precise browser-history boundary with failing tests.
**Success Criteria:** Existing route builder exposes private data and legacy parser accepts unowned data in causal regression tests.
**Tests:** Study Pack route privacy and legacy-link rejection.
**Status:** Complete — both original privacy regressions failed before implementation.

## Stage 2: reuse owner-checked transfers
**Goal:** Transfer typed Study Pack intents through the existing temporary, owner-bound Flashcards storage and authority lifecycle.
**Success Criteria:** URLs contain opaque tokens only; destination validates the current owner, consumes the record, clears URL state, and invalidates rendered details on account change. Existing generator handoffs retain their behavior. Note, Media, and Message actions preserve the source on failure.
**Tests:** Owner/server mismatch, expiration, replay, cancellation, cleanup, and both payload kinds; actual source actions and destination behavior.
**Status:** Complete — typed Study Pack records reuse the existing storage, authority, cancellation, TTL, and cleanup paths. Both consumers scrub all private URL fields. Note and Message producers verify captured ownership; Media guards the selected owner and item.

## Stage 3: verify and accept
**Goal:** Verify the repair in focused tests and the original native scenarios.
**Success Criteria:** Relevant tests/lint/review pass; actual reciprocal Back/reload checks hide prior metadata and owner handoffs prefill correctly. Track failures honestly; frozen UAT archives remain unchanged.
**Tests:** Focused Flashcards transfer and source integration suites, native PostgreSQL and SQLite multi-user acceptance. Bandit is applicable only if Python changes; otherwise record JavaScript-only scope.
**Status:** In Progress — 124 focused tests pass. Independent review identified two regressions (single-user Note principal handling and mixed legacy URL cleanup); each was reproduced red, corrected, and verified green. Review now has no actionable findings. ESLint reports the same 90 warnings and zero errors as HEAD. Frontend typecheck reports the same 93 errors as frozen merged3cff, with zero new diagnostics. Bandit is not applicable to this TypeScript-only implementation. Native SQLite verifies owned Note/Media prefill, opaque navigation, and cross-tab logout clearing; reciprocal history checks and PostgreSQL acceptance remain to finish. Frozen matrix archives are unchanged; targeted SQLite uses its retained backend and the changed checkout frontend with an isolated build directory.
