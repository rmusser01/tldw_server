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
eligible object heads, purge generation, quotas, and one bootstrap cursor.

**Success criteria:** Repeated requests are idempotent; mismatched users,
devices, schema, quotas, and purge generations fail with stable content-free
outcomes; no bootstrap path treats Sync history as canonical storage.

**Tests:** Focused bootstrap plus Personalization repository/service tests.

**Status:** Complete — canonical service remains authoritative; first-link creation,
opaque binding state, compatibility checks, and cursor-bounded snapshots are implemented.

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

**Status:** In Progress — focused SQLite regressions, compilation, and diff hygiene
pass; Ruff/Bandit are unavailable in the isolated runner and PostgreSQL-only tests
remain for controller verification.

## ADR check

ADR required: no (existing)

ADR path: `backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md`

Reason: ADR-002 already governs canonical server ownership, key custody,
whole-object Sync transport, and the service boundary used by bootstrap.
