---
id: TASK-13148
title: Bootstrap Personal Context canonical profile
status: Done
assignee:
  - '@codex'
created_date: '2026-08-30 22:21'
updated_date: '2026-08-31 05:04'
labels:
  - personal-context
  - sync
  - security
dependencies:
  - TASK-13147
references:
  - >-
    backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
documentation:
  - Docs/superpowers/plans/2026-08-28-personal-context-04-sync-multidevice.md
  - IMPLEMENTATION_PLAN_personal_context_bootstrap.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose an authenticated, cursor-consistent Personal Context bootstrap that serializes first-link ownership, returns canonical manifest/scopes/heads with wrapped integrity-key material, and prevents pre-reconciliation uploads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The server serializes first-link profile ownership and returns the canonical manifest, scopes, object heads, purge generation, and one consistent bootstrap cursor for the authenticated user.
- [x] #2 Bootstrap distributes the server-owned integrity key only to an authenticated registered device using the existing wrapped Sync key-record path; plaintext key material never enters logs, diagnostics, or durable bootstrap metadata.
- [x] #3 Pre-reconciliation Personal Context uploads fail closed, retries are idempotent, and mismatched user, device, schema, quota, or purge generation produce stable content-free outcomes.
- [x] #4 The bootstrap contract supports Chatbook reviewed reconciliation and full integrity rebaseline without making Sync history the canonical profile authority.
- [x] #5 Targeted bootstrap, Sync, and Personalization tests plus Ruff, compilation, Bandit, diff hygiene, and independent review pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED tests for authenticated cursor-consistent bootstrap, registered-device wrapping, idempotency, and the pre-reconciliation upload fence.
2. Serialize first-link ownership and return canonical manifest, scopes, object heads, purge generation, quotas, and one bootstrap cursor through the profile service boundary.
3. Deliver the server-owned integrity key through the existing Sync key-record enrollment/rewrap path without plaintext persistence.
4. Enforce stable content-free failures for user/device/schema/quota/generation mismatches and keep Sync history non-authoritative.
5. Run targeted tests, Ruff, compilation, Bandit, diff hygiene, independent review, and commit.

ADR required: no (existing)
ADR path: backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
Reason: ADR-002 already governs canonical server ownership, key custody, whole-object Sync transport, and the service boundary used by bootstrap.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented an authenticated Personal Context first-link boundary over the same
canonical profile records used by the server; Sync remains encrypted transport
and coordination state, never a second profile authority. When no canonical
profile exists, bootstrap reserves only random profile identity and wrapped key
custody, then returns a deterministic transient manifest/global-scope snapshot.
It does not persist canonical manifest, scope, record, or proposal replicas until
the reviewed `/personal-context/complete` request materializes those exact IDs,
versions, and cursor. A cancelled preview therefore leaves canonical content
absent, and an explicit server-side profile creation can safely adopt the same
reservation instead of reporting corrupt/locked state.

The final implementation also fences stale key, purge-generation, profile,
authority, and link transitions; persists receipts with backend-appropriate
locking/CAS semantics; preserves unrelated enrollment metadata; rotates device
wrappers when the registered public key changes; and maps all bootstrap failures
to stable content-free HTTP outcomes. Schema, quota, and purge-generation 409s
now include a typed `attention` object with exact content-free schema bounds,
required/available/insufficient quotas, or expected/current purge generations;
no canonical body or key material is included. ADR-002 remains the governing
decision; no new ADR was required.

Verification after the post-closure contract correction:

- `163 passed` across the Personal Context repository/service, bootstrap, and
  authenticated Sync endpoint modules.
- `174 passed, 13 deselected` for non-PostgreSQL Sync store coverage.
- `4 passed` for executable PostgreSQL Personal Context lock/CAS transaction
  contracts.
- Ruff reported `All checks passed`; Bandit exited 0 for every touched production
  module; Python 3.11 compilation and `git diff --check` exited 0.
- Earlier implementation rounds received independent spec/code-quality approval;
  this correction is ready for the controller-owned follow-up review.

Known verification limits: the environment did not provide a live PostgreSQL
fixture, so backend behavior is covered by executable transaction contracts
rather than a live database integration run. The repository-wide suite was not
run because project guidance requires explicit opt-in; verification stayed on
the affected Personalization and Sync modules. The production-factory timestamp
incident and resulting cross-layer testing rule are recorded in
`backlog/docs/lessons-testing-evidence.md`.

Post-closure structured-attention correction: quota incompatibility now includes every required quota name in available_quotas, using an explicit safe zero for unsupported names. The HTTP boundary strictly validates discriminated attention, bounded content-free values, semantic consistency, and reason-code/kind agreement; malformed, mismatched, or extra-field mappings are omitted while preserving the stable 409 reason/message. TDD evidence: focused RED 3 failed/3 passed; GREEN 6 passed; affected endpoint/bootstrap modules 125 passed. Ruff, Python compilation, Bandit (0 findings/errors), Chatbook strict-parser compatibility, and diff hygiene passed. TASK-13148 status intentionally remains unchanged for controller review.

Final review correction: an unsupported quota now behaves as a zero-capacity
quota, so `future_sync_quota: 0` succeeds while any positive minimum remains a
typed incompatibility with a safe zero in `available_quotas`. The unused
`ensure_sync_profile()` helper and its test fake were removed; the existing
absent-bootstrap regression continues to prove that planning reserves only
content-free identity/key control state and materializes no canonical profile
content. TDD evidence: focused RED `2 failed, 33 deselected`; focused GREEN
`2 passed, 33 deselected`; affected bootstrap/endpoint modules `127 passed`.
TASK-13148 status remains unchanged for controller review.

Final quota-cardinality correction: successful and incompatible quota maps are
now request-focused whenever requirements are supplied. A valid maximum-size
32-name request therefore returns exactly 32 safe availability entries instead
of being expanded with five unrelated server defaults; unknown zero minima
remain successful at zero and positive unknown minima remain typed deficits at
zero. The successful response schema independently enforces the same entry,
name, and strict-integer bounds. Service and authenticated endpoint regressions
cover 32 unknown names, exact map completeness, typed attention survival, and
content-free error output. TASK-13148 status remains unchanged for controller
review.

Transport-watermark correction: successful bootstrap now exposes a separate
`sync_transport_cursor`, signed with the existing private-pull codec and scoped
to the authenticated dataset, registered device, negotiated version set, and
all Personal Context domain/version streams. The original semantic `cursor`
remains the canonical review/receipt identity and is not accepted by private
pull. Bootstrap first enrolls only content-free transport domain control state,
then holds the Sync dataset-row ordering lock while it captures stream
watermarks and reads the canonical snapshot. This is not a cross-database
transaction: the required ordering is canonical commit followed by Sync
envelope append, and relevant envelope appends share the dataset-row lock. The
watermark is retained with the device key record and semantic cursor so retrying
the same reviewed plan cannot advance it; token signing time/expiry may refresh
without skipping a post-review change. The token lifetime is 30 days with five
minutes of bounded clock skew. Successful quota responses now also include each
valid requested unknown zero-minimum key at effective value `0`; positive
unknown requirements remain typed incompatibilities with available value `0`.
Focused tests cover retained history, post-boundary delivery, scope/version
rejection, slow review, retry stability, SQLite interleaving, and an executed
PostgreSQL lock contract. Live PostgreSQL was unavailable. TASK-13148 status is
intentionally unchanged.

Final projection-watermark and request-boundary correction: bootstrap now
refuses to sign a transport watermark while any accepted Personal Context
envelope in a negotiated stream is pending, failed, conflicted, or otherwise
not durably applied/superseded. The check runs under the same dataset-row guard
used by append, materialization, and replay, preventing a later repair from
creating canonical content at or below the reviewed cursor. The stable 409 is
content-free and instructs the caller to repair Sync and retry.

The authenticated request schema now bounds `required_quotas` to 32 safe ASCII
names and strict non-Boolean integers in `[0, 2**63 - 1]`. Successful responses
continue to cover every valid requested name; unknown zero minima are returned
as zero and positive unsupported minima retain typed incompatibility. Meaningful
RED was `4 failed` for projection state and `8 failed` for malformed quota
requests. GREEN was `46 passed` bootstrap, `102 passed` endpoints, `187 passed,
2 skipped` store, and `23 passed` replay/repair. Ruff, compilation, Bandit, and
diff checks are recorded in the Task 3a report. TASK-13148 status remains
unchanged for controller review.
<!-- SECTION:NOTES:END -->

## Progress

- [x] Added the bootstrap contract before production implementation, including
  canonical heads/cursor, registered-device wrapping, compatibility failures,
  pre-link fencing, completion, and plaintext-canary coverage.
- [x] Implemented the server-owned bootstrap and completion boundary with
  canonical Personal Context reads and opaque Sync dataset/key-record state.
- [x] Ran focused SQLite-backed Sync and Personal Context regressions plus
  Python 3.11 compilation and diff hygiene; results are recorded in the slice report.
- [x] Controller verification completed with the authenticated endpoint graph,
  focused SQLite persistence coverage, PostgreSQL transaction-contract coverage,
  Ruff, Bandit, compilation, diff hygiene, and two independent review passes.

## Review round 1 progress

- [x] Hardened factory custody with registered-device RSA-OAEP wrapping and
  server-owned authority selection; added typed API bootstrap/completion routes.
- [x] Reserved Personal Context enrollment metadata/domains and made completion
  receipts device-specific so one device cannot unlock another's pushes.
- [x] Added the dedicated canonical snapshot transaction and atomic persisted
  Sync device receipt path. TASK-13148 remains In Progress for controller closure.

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
