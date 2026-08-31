# TASK-13148 — Bootstrap Personal Context canonical profile

## Stage 1: Contract and RED tests

**Goal:** Pin one cursor-consistent, authenticated bootstrap response and the
pre-reconciliation upload fence.

**Success criteria:** Focused tests fail because bootstrap does not yet expose
canonical profile heads, registered-device key delivery, or link state.

**Tests:** `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py`

**Status:** Complete — contract test was authored before implementation; focused RED
collection/fixture failures and final GREEN evidence are recorded in the task report.

## Stage 2: Canonical bootstrap transaction

**Goal:** Serialize first-link ownership and return the manifest, scopes,
eligible object heads, purge generation, quotas, and one bootstrap cursor
without creating canonical content before reviewed completion.

**Success criteria:** Repeated requests are idempotent; mismatched users,
devices, schema, quotas, and purge generations fail with stable content-free
outcomes; no bootstrap path treats Sync history as canonical storage.

**Tests:** Focused bootstrap plus Personalization repository/service tests.

**Status:** Complete — canonical service remains authoritative. An absent profile
reserves only random identity and wrapped key custody; the deterministic
manifest/global-scope plan remains transient until reviewed completion persists
those exact canonical objects. Compatibility failures expose typed content-free
attention facts while retaining the existing stable 409 reason codes.

## Stage 3: Wrapped integrity-key delivery and link fence

**Goal:** Use the existing Sync key-record enrollment/rewrap path to deliver
the server-owned integrity key only to the authenticated registered device and
block Personal Context uploads until reconciliation completes.

**Success criteria:** No plaintext key or profile body appears in durable
bootstrap metadata, logs, or diagnostics; a completed bootstrap supports the
Chatbook full integrity rebaseline.

**Tests:** Bootstrap security, replay, authorization, and pre-link push tests.

**Status:** Complete — registered-device wrapped key records and the narrow
post-reconciliation upload transition are implemented.

## Stage 4: Verification and review

**Goal:** Complete targeted regressions, static/security gates, task evidence,
and independent review.

**Success criteria:** Scoped Sync and Personalization tests plus Ruff,
compilation, Bandit, diff hygiene, and review pass; TASK-13148 is complete.

**Status:** Complete — the controller reran the affected authenticated endpoint,
bootstrap, service, and store suites; executed the PostgreSQL Personal Context
transaction contracts; and completed Ruff, Bandit, Python 3.11 compilation,
diff hygiene, spec review, and code-quality review.

## Review round 1 progress

- [x] Reserved Personal Context domains and metadata from generic dataset enrollment.
- [x] Added per-device completion receipts and made push require the requesting
  device's matching profile/key/purge receipt.
- [x] Added RSA-OAEP wrapping to the production factory, server-owned authority
  selection, and typed bootstrap/completion API routes.
- [x] Added a bounded Personalization DB snapshot transaction with key identity
  and complete eligible head coverage; Sync completion persists a device-bound
  receipt in its own atomic Sync DB transaction.

## ADR check

ADR required: no (existing)

ADR path: `backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md`

Reason: ADR-002 already governs canonical server ownership, key custody,
whole-object Sync transport, and the service boundary used by bootstrap.

## Final verification evidence

- `163 passed` — Personal Context repository/service, bootstrap, and
  authenticated Sync endpoint modules after the non-mutating review correction.
- `174 passed, 13 deselected` — non-PostgreSQL Sync store tests.
- `4 passed` — executable PostgreSQL Personal Context lock/CAS transaction
  contracts. A live PostgreSQL fixture was unavailable in this environment.
- Ruff, Bandit, Python 3.11 compilation, and `git diff --check` exited cleanly.
- Independent spec and code-quality reviews, including a narrow post-Ruff
  follow-up, reported no actionable findings.

The full repository suite was not run because repository guidance requires
explicit user opt-in; the verification set was limited to affected modules.
