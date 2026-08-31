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

## Structured-attention contract correction

- [x] Unknown required quota names remain content-free and are represented in
  `available_quotas` with an explicit zero, so strict clients can verify the
  complete required/available relationship.
- [x] The HTTP boundary validates bootstrap attention against the strict
  discriminated schema and its matching stable reason code before serialization.
  Malformed, mismatched, or extra-field attention is omitted rather than echoed.
- [x] RED evidence: the focused attention selection produced `3 failed, 3 passed`
  for the missing zero-valued quota and two untrusted-attention leaks.
- [x] GREEN evidence: the same selection produced `6 passed`; the complete
  affected endpoint/bootstrap modules produced `125 passed`.
- [x] Ruff, Python 3.11 compilation, Bandit (`0` findings/errors), and range diff
  hygiene passed. Chatbook's strict typed response parser accepted the corrected
  unknown-quota response shape.

TASK-13148's existing status is intentionally unchanged; controller review owns
any subsequent lifecycle update.

## Final compatibility and cleanup correction

- [x] Treat an unsupported quota as available at zero: a requested minimum of
  zero is satisfied, while a positive minimum remains a typed, content-free
  incompatibility with `available_quotas[name] == 0`.
- [x] Remove the unused `ensure_sync_profile()` service helper and test fake so
  canonical materialization remains reachable only through the reviewed
  plan/completion boundary.
- [x] Retain the absent-profile regression proving bootstrap reserves identity
  and key custody without materializing canonical profile content.
- [x] Focused RED produced `2 failed, 33 deselected`; focused GREEN produced
  `2 passed, 33 deselected`; the complete affected bootstrap and endpoint
  modules produced `127 passed`.

TASK-13148 remains unchanged for controller review.

## Sync transport-watermark correction

- [x] Successful bootstrap now returns `sync_transport_cursor` in addition to
  the unchanged semantic `cursor`. The semantic cursor remains the receipt and
  completion identity; it is never accepted by private Sync pull.
- [x] The transport cursor uses the existing signed pull-token codec and binds
  the authenticated dataset, registered device, complete negotiated adapter
  version set, Personal Context domain/version streams, and one sequence
  watermark per stream. Its 30-day lifetime, plus five minutes of bounded clock
  skew, covers the durable review window.
- [x] Before reading the canonical snapshot, bootstrap enrolls only content-free
  Personal Context transport domains. It then locks the Sync dataset row and
  captures the stream boundary while that lock remains held across the canonical
  snapshot read. This is deliberately an ordering fence, not a cross-database
  transaction: canonical mutations must commit before their Sync envelope is
  appended, and every relevant envelope append takes the same dataset-row lock.
- [x] The boundary is durably associated with the device key record and semantic
  bootstrap cursor. Retrying the same reviewed plan may issue a newly signed
  token but cannot advance the embedded watermarks; a new semantic plan may
  capture a new boundary.
- [x] Unknown valid zero-minimum quota requirements are included in successful
  response `quotas` with effective value `0`; positive unknown requirements
  retain typed incompatibility with available value `0`.

TDD coverage includes the exact HTTP quota field, retained multi-revision
history, post-boundary delivery, signed scope rejection, slow-review expiry,
stable retry watermarks, deterministic SQLite interleaving, and an executed
PostgreSQL `FOR UPDATE` transaction contract. A live PostgreSQL fixture remains
unavailable; the limitation is explicit rather than represented as live-DB
evidence. TASK-13148's status remains unchanged.

## Projection-safe watermark and quota request correction

- [x] Reject bootstrap while any accepted Personal Context envelope at the
  proposed transport boundary is not durably `applied` or `superseded`.
  Snapshot capture, envelope append, materialization, and replay coordinate on
  the existing dataset-row guard; bootstrap never signs a cursor that can hide
  a later replay at or below its watermarks.
- [x] Map incomplete projection to one stable content-free 409 reason with
  explicit repair/retry guidance. Materialization errors, canonical bodies,
  IDs, keys, and ciphertext never enter the response.
- [x] Enforce the public `required_quotas` contract at HTTP parsing: at most 32
  entries; lower-case ASCII identifier names of at most 64 characters; strict
  built-in non-Boolean integers between zero and `2**63 - 1`.
- [x] Preserve the effective quota response contract: every valid requested
  quota is represented on success, including unknown zero-minimum names at
  value zero; positive unsupported minima retain typed incompatibility.
- [x] Cover the durable-append/materialization crash window, failed replay,
  conflict/unknown states, successful guarded repair, post-boundary visibility,
  and the executed PostgreSQL lock/query contract.

Focused meaningful RED: projection `4 failed, 42 deselected`; quota parsing
`8 failed, 93 deselected`. GREEN: bootstrap `46 passed`; endpoints `102 passed`;
store `187 passed, 2 skipped`; replay/repair `23 passed`; schemas/models
`126 passed`. Ruff, Python compilation, Bandit, and diff hygiene passed.
Repository-wide tests remain intentionally unrun under the explicit opt-in
policy. TASK-13148's status is unchanged for controller review.
