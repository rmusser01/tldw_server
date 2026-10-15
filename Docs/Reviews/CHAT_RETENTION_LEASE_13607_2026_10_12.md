# Original-source retention preparation lease — TASK-13607

The private gateway composes typed original-request reservation and existing attempt policy. It grants bounded preparation eligibility, not quota, verified uploads, claims, ready bundles or public capability.

NativeForkStore.claim_retention_attempt requires the owning transaction and an outer savepoint around reservation and claim. Typed reservation supplies owner/request/digest/projection/source admission under operation→workspace→conversation locks. Saved outcomes return before preparation clock/claim. For preparing state, production UTC time is captured after lock/read waits, then generic claim_attempt grants the lease; deterministic-clock callers remain unchanged.

Shared claim_attempt rejects non-integer/negative generation for preparing rows before parsing/increment. Actual SQLite0.5 previously emitted/persisted1.5 despite downstream candidate refusal. The guard refuses without mutation; terminal handling is unchanged.

Active lease remains pending without renewal. Expired lease with any non-discarded candidate requires reconciliation; only durable cleanup permits generation advancement. Old generations cannot reserve work and candidate/quota IDs remain terminal. Outer savepoint contains post-grant failures even when caught while preserving unrelated caller work. AuthNZ/filesystem work stays outside the transaction.

ADR required: no new; ADR049/050 and proposed071 govern source/key identity. No production retention caller/public route is added; final source/Sync/byte/publication admission remains pending.

## Verification

SupportedPython3.12.13 after environment activation, official PostgreSQL fixture TLDW_TEST_NO_DOCKER=1; no alternative provisioning.

- Causal5failed/5officialPGskip/12warnings3.45s: four missing gateway cases and actual fractional generation accepted at shared seam.
- Initial lease/reservation green21pass/21skip12warnings11.39s.
- Added clock-order/concurrent grants and actual cleanup targeted8pass/8skip/115deselected12warnings4.50s. Real reservation precedes clock; lease has full300seconds. Concurrent callers yield one attempt and one pending result/one generation.
- Final five-suite connected160pass/162skip12warnings73.57s, exit0: lease/reservation, native transactions, complete native I/O and actual fork commit.
- Actual guarded cleanup settles an uncharged/unwritten reserved retention candidate. Renewal remains blocked until discarded; next generation then advances and the old worker refuses. This does not qualify charged reclamation, actual uploaded retention or quota publication.
- Strengthened saved-result cases poison preparation clock, claim and source reader for committed/gone seeded results; changed cases2pass/2officialPGskip/12deselected12warnings2.40s. Exact result/record unchanged. Seeded results do not qualify publication.
- Post-grant fault calls actual lease update then raises; caller catches inside larger transaction. Reservation/lease disappear while unrelated caller conversation persists. Stale source cannot rotate the original lease.
- Three-file matched/direct Ruff0/0; new gateway and only added IO control formatted with AST preservation, new test Black/supported compile all3/diff pass. No broad formatting. Actual one-production-file Bandit baseline/current0/0 findings/0 errors, coverage verified.
- Both final independent source/fixture reviewers bounded CLEAR; reviewers ran no tests.

Physical PostgreSQL/concurrency remains unqualified; official fixture and existing backend-specific controls account for skips. Private evidence /private/tmp/chat-retention-lease-13607-*; raw logs/secrets are not published.

No runtime/profile/auth/key/card/DB changes, new browser/window/tab/send or baseline rewrite. Existing owned services remain untouched. Quota-backed retention, reference mapping/original hash, actual claim/byte publication, active Sync/native bridge, UI/send integration and broader all-ten NN/G/installed-extension/Compact/AT/real-model/study gates remain open.

Initial normal exact-owned eight-file hooks pass (exit0). House Ruff/Black hook filters skip native paths; meaningful manual checks above pass. All three bounded stages complete; only own plan retires to /private/tmp/IMPLEMENTATION_PLAN_chat_retention_lease_13607_completed.md. Final normal exact-owned seven-file hooks pass (exit0), no bypass. Dev advanced to2a8ffa78a839a53171a27bc1c5f3ce811bb892b3 during validation; its notification-schema changes require a separate integration checkpoint after this local lease commit. Master TASK13458/H2 parent13261.4 remain In Progress.
