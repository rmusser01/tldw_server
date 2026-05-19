# Characters Backend Remediation Design

## Summary

This spec covers remediation for the full set of backend findings from the
Characters review, not just the Stage 5 chat-coupling subset.

The goal is to fix the real persistence and contract boundaries in the
Characters and ChaChaNotes backend so that:

1. character lifecycle semantics are consistent and defensible
2. version history and revert are lossless for avatar changes going forward
3. import and export behavior is explicit and round-trip-safe within declared
   limits
4. chat and message caps are enforced atomically under concurrency
5. request throttling fails closed once character rate limiting is enabled
6. exemplar and world-book API contracts match actual behavior
7. the mixed-suite `503` issue is fixed or narrowed to an evidenced residual
   boundary

## Scope

In scope:

- `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py`
- `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- `tldw_Server_API/app/api/v1/endpoints/character_messages.py`
- `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py`
- `tldw_Server_API/app/core/Character_Chat/character_rate_limiter.py`
- `tldw_Server_API/app/core/Character_Chat/character_limits.py`
- `tldw_Server_API/app/core/Character_Chat/modules/character_db.py`
- `tldw_Server_API/app/core/Character_Chat/modules/character_chat.py`
- `tldw_Server_API/app/core/Character_Chat/modules/character_io.py`
- `tldw_Server_API/app/core/Character_Chat/world_book_manager.py`
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Characters, Character_Chat, ChaChaNotesDB, Streaming, and related integration
  tests that encode the current incorrect behavior
- ChaChaNotes schema migration tests needed to verify SQLite and Postgres-safe
  behavior for any touched persistence path

Out of scope:

- frontend changes
- unrelated cleanup in the dirty main workspace
- broad redesign of the Characters subsystem beyond what is required to fix the
  reviewed findings
- non-Characters modules except where direct test fixtures or DB abstractions
  must change to keep backend parity

## Findings Covered

This remediation is intended to address these reviewed findings:

- restore-on-active-row incorrectly succeeds
- avatar state is missing from character version snapshots and revert behavior
- malformed YAML and card-like text imports silently fall back to synthetic
  characters
- image-file imports bypass the normal avatar normalization contract
- PNG export can generate artifacts that importer limits later reject
- request-level character rate limiting fails open unless RG is fully active
- chat and message caps are non-atomic under concurrency
- hybrid exemplar fallback can under-report `total`
- world-book delete and detach responses populate the wrong identifier field
- import and exemplar fallback paths have reviewed low-severity performance
  issues
- mixed-suite `503` failures point to ChaChaNotes init or shutdown lifecycle
  state that still needs root-cause confirmation

## Locked Decisions

- Restore of an already-active character becomes a conflict, not idempotent
  success.
- Empty `{}` character updates remain explicit no-ops and do not become
  validation errors.
- Character version snapshots become lossless for avatar state going forward.
- Avatar snapshot payloads use a JSON-safe representation, specifically
  `image_base64`, rather than raw binary.
- Legacy snapshots that predate avatar capture remain readable but are treated
  as incomplete historical records. The system will not pretend they are
  lossless.
- Reverting to a legacy snapshot that lacks avatar state preserves the current
  avatar instead of clearing it or guessing at older bytes. Legacy revert is
  therefore explicitly content-only for avatar state.
- Internal version snapshots may be lossless without forcing the public
  `/versions` and `/versions/diff` APIs to return raw avatar payloads by
  default. Default version-history responses should stay compact and expose
  avatar presence or diff semantics without bloating normal API responses.
- The compact public avatar-history shape is locked now rather than left as an
  implementation choice:
  - `/versions` entries expose `payload.image_present: bool`
  - `/versions` entries expose `payload.image_digest: str | null`, where the
    digest is derived from normalized avatar bytes and is safe to compare
  - `/versions/diff` reports avatar changes only through
    `changed_fields[].field == "image_present"` and
    `changed_fields[].field == "image_digest"` and must never return raw
    `image_base64` by default
- Plain-text synthetic character import remains allowed only for explicitly
  plain-text inputs. Structured card inputs are determined by the import
  classifier and parser path, not just filename extension examples. Structured
  card candidates such as JSON, YAML, and PNG-embedded card metadata are
  rejected when parse or validation fails.
- Image-container imports get an explicit split:
  - image files with embedded character metadata are structured-card imports and
    must fail if metadata parsing or validation fails
  - image files without embedded character metadata may only use image-only
    import behavior when the caller explicitly allows that mode
  - image files without embedded metadata must not fall through to synthetic
    plain-text character import
- Image-file imports and base64/image-field imports must converge on one avatar
  normalization contract before persistence.
- PNG export must honor the same effective metadata-size contract as the PNG
  importer. If an export cannot be re-imported under the declared limit, that
  PNG export should fail instead of producing a misleading artifact.
- Character request throttling fails closed once enabled:
  - `429` when RG explicitly denies
  - `503` when throttling is enabled but RG is disabled, not enforcing, or
    unavailable
- Chat and message caps must be enforced through DB-backed atomic operations,
  including multi-message persistence paths.
- Multi-message persistence that spans a slow model call must use a DB-backed
  reservation with explicit acquire, consume, release, and expiry semantics
  rather than holding a DB lock open across external network work.
- Hybrid exemplar fallback must return semantically correct pagination metadata.
- World-book delete and detach responses must match their declared schema.
- Any persistence or migration changes must remain SQLite-safe and Postgres-safe
  or be explicitly documented as deferred with justification. Silent parity
  drift is not acceptable.
- The `503` investigation must end in one of two states:
  - deterministic reproducer, root-cause fix, regression test
  - narrowed failing boundary with instrumentation evidence, a durable repro
    harness, and explicit residual-risk documentation

## Approaches Considered

### Recommended: Persistence-First Remediation

Fix the underlying DB and contract boundaries first, then align endpoint and
test behavior around those corrected invariants.

Pros:

- matches the actual source of the reviewed defects
- avoids patching only API symptoms
- gives the strongest guarantee for direct library callers and API callers
- supports proper concurrency fixes instead of best-effort endpoint checks

Cons:

- touches more files than a shallow API patch
- includes migration and backend-parity work

### Alternative: Endpoint Containment

Keep most persistence behavior as-is and patch the FastAPI entry points.

Pros:

- smaller change set
- faster to land

Cons:

- would leave direct DB or library callers on the old contracts
- would not truly solve the concurrency findings
- does not satisfy the user’s request for full remediation

### Rejected: Broad Subsystem Refactor

Redesign character lifecycle, chat persistence, import/export, and init
lifecycle as one large cleanup.

Reason rejected:

- too large for a focused remediation pass
- too much risk in a dirty repository with unrelated work already present

## Design

### 1. Character Lifecycle Integrity

`ChaChaNotes_DB.restore_character_card(...)` becomes the source of truth for
restore semantics.

New contract:

- missing row: conflict
- deleted row with mismatched `expected_version`: conflict
- deleted row within retention window and matching version: restore succeeds
- active row: conflict, not idempotent success

This remains DB-first so API endpoints and direct library callers share the same
behavior.

`update_character_card(...)` keeps empty payloads as no-ops. The no-op contract
will be documented and regression-tested explicitly so it is no longer an
accidental loophole.

### 2. Lossless Character Version History

Character sync snapshots must carry avatar state in a JSON-safe payload shape.

Going forward, the `sync_log.payload` for character create and update entries
will include:

- the existing character snapshot fields
- `image_base64` when an avatar is present
- `image_base64: null` when the avatar is explicitly absent

This avoids storing raw bytes in JSON while preserving a reversible snapshot.

Revert and diff behavior:

- `_CHARACTER_REVERT_FIELDS` and related version-diff helpers will expand to
  include avatar state
- revert logic will restore avatar state from `image_base64` through the same
  character storage path used by normal writes
- legacy snapshots without `image_base64` remain valid history entries, but
  avatar restoration from those older entries is explicitly incomplete
- when reverting to a legacy snapshot without avatar state, the current avatar
  is preserved unchanged and only the text or metadata fields revert

Public version API shaping:

- internal sync snapshots remain complete
- `/versions` entries expose compact avatar fields in `payload`:
  - `image_present: bool`
  - `image_digest: str | null`
- `/versions/diff` reports avatar changes only through `image_present` and
  `image_digest` field entries and must not include raw `image_base64` by
  default
- revert continues to read the complete internal snapshot rather than the
  compact public payload shape
- if full avatar snapshot inspection is needed later, that should be an
  explicit opt-in behavior rather than the default history payload shape

Migration strategy:

- assuming no intervening schema work lands first, introduce ChaChaNotes schema
  version `45`
- implementation must revalidate the current head schema version when the
  remediation branch is cut and use the next available version if `45` is no
  longer correct
- the selected schema version refreshes the character sync triggers so new
  snapshots contain
  `image_base64`
- no historical backfill of old sync rows is required for correctness; mixed
  old/new history is supported intentionally

### 3. Import And Export Contract Remediation

#### 3.1 Card-like Parse Failures

Character import behavior becomes mode-explicit:

- inputs classified as structured card candidates by the validated import type
  and parser path must parse and validate successfully or the import fails
- inputs classified as explicit plain-text imports remain eligible for
  synthetic plain-text import behavior

This preserves useful plain-text import without silently accepting malformed
card files as something else.

Image-container classification rule:

- PNG, WEBP, and JPEG imports with embedded character metadata are structured
  card mode and must succeed as structured cards or fail clearly
- PNG, WEBP, and JPEG imports without embedded character metadata are not
  structured cards
- those metadata-less image imports may only succeed through explicit
  image-only import mode and otherwise fail with the existing
  `missing_character_data` style contract

#### 3.2 Unified Avatar Normalization

Image-file imports and structured card imports must converge on one avatar
normalization path before persistence.

Implementation target:

- all avatar inputs are normalized into the same persisted representation
- import transport format no longer determines storage bytes
- the normalization path should avoid avoidable duplicate full-buffer work where
  it naturally can, because the current split also contributes to the reviewed
  performance issue

#### 3.3 PNG Export And Import Ceiling

PNG export must not emit an artifact that current import rules later reject.

New contract:

- before returning a PNG export, the server must verify that the embedded
  metadata fits within the same effective ceiling the importer enforces
- if the card is too large for PNG metadata, the PNG export fails with a clear
  client-facing error and callers can use JSON/CCV3 export instead

This is preferable to silently generating a non-round-trippable PNG.

### 4. Strict Quota And Rate-Limit Enforcement

#### 4.1 Request-Level RG Behavior

`CharacterRateLimiter.check_rate_limit(...)` becomes fail-closed once character
request throttling is enabled.

New contract:

- limiter disabled: allow
- limiter enabled and RG denies: `429`
- limiter enabled and RG disabled: `503`
- limiter enabled and RG not enforcing request throttles: `503`
- limiter enabled and RG decision unavailable: `503`

This makes “enabled” operationally meaningful and testable.

#### 4.2 Atomic Chat And Message Caps

Endpoint count-then-insert checks are no longer the source of truth.

Instead, `ChaChaNotes_DB` will expose guarded persistence helpers for:

- atomic chat creation under per-user chat caps
- atomic single-message persistence under per-chat message caps
- atomic multi-message persistence for flows that may commit both a user message
  and an assistant reply

Implementation shape:

- SQLite path uses the existing `BEGIN IMMEDIATE` transaction semantics and
  keeps the count and write in one write-serialized transaction
- Postgres-safe behavior uses the same DB abstraction but must also serialize
  the relevant quota decision, for example by locking a stable guard row or an
  equivalent DB-backed synchronization target rather than relying on
  read-committed count queries alone
- assistant-persisting flows that call an external model before writing must
  not keep a transaction open across that model call
- those flows therefore require a durable DB-backed reservation record with:
  - a stable reservation key or request id
  - reserved slot count
  - conversation scope
  - expiry or cleanup semantics
  - an explicit consume step on successful persistence
  - an explicit release step when the request fails or abandons persistence
- the exact serialization primitive is an implementation detail, but the spec
  requirement is strict: two concurrent requests must not both consume the last
  remaining slot

Multi-message contract:

- if a request needs capacity for both a user message and an assistant reply,
  it must reserve capacity as one unit before the slow model call
- the request either persists the full allowed batch or fails cleanly
- partially consuming the last slot and leaving the request in a half-persisted
  state is not acceptable
- expired or abandoned reservations must stop counting against the cap without
  requiring manual operator cleanup

Existing endpoint pre-checks may remain as fast-fail hints if useful, but the
DB helper is the only authority.

### 5. Exemplar And World-Book Contract Fixes

Hybrid exemplar fallback must stop deriving `total` from a truncated candidate
window.

Target behavior:

- fallback responses expose a real total count for the search space they claim
  to paginate
- deeper pages do not require candidate-pool growth proportional to
  `offset + limit` when embeddings are unavailable

This should address both the correctness issue and the reviewed deeper-page
performance concern in the same design.

World-book delete and detach responses must populate the correct identifier
field so response payloads match their schema and stop misleading clients.

### 6. ChaChaNotes `503` Investigation And Availability Hardening

This workstream must follow root-cause debugging, not speculative fixes.

Investigation flow:

1. reproduce the mixed-suite failure path using the previously failing slices
2. add narrow instrumentation around:
   - shared init-event waiting
   - executor-backed DB creation
   - default-character task scheduling
   - default-character future draining
   - shutdown executor lifecycle
3. identify the failing boundary before changing behavior

Likely failure classes to distinguish:

- stale cached instance or init-event state
- executor shutdown or reuse ordering
- overlap between request warmup and shutdown drain
- default-character background work blocking or poisoning later initialization

Acceptable outcomes:

- deterministic mixed-suite reproducer, narrow fix, regression test
- or a narrowed and evidenced failing boundary plus a durable harness and
  explicit residual-risk note if the failure remains nondeterministic in this
  environment

Unacceptable outcome:

- adding retries, timeouts, or resets without evidence that they fix the root
  cause

### 7. Backend Parity And Migration Strategy

Any touched persistence behavior must keep SQLite and Postgres-safe behavior in
sync.

Requirements:

- the selected next schema version captures any new trigger or auxiliary-table
  changes required by this remediation
- Postgres schema initialization and migration translation must remain valid for
  the same logical changes
- tests must cover migrated SQLite behavior and at least one Postgres-safe
  parity check for touched schema logic where practical

If the final implementation requires an auxiliary guard table for strict quota
serialization, that table is part of the selected schema bump and must exist in
both backend paths.

### 8. Worktree And Repository Discipline

The repository is already dirty with unrelated work. Implementation for this
remediation will happen in an isolated git worktree after the spec and plan are
approved.

This remediation must avoid:

- reverting unrelated user changes
- bundling unrelated cleanup
- broad file churn outside the in-scope Characters and ChaChaNotes backend
  surfaces

## Testing Strategy

This work will be implemented with TDD.

Required failing tests first:

- restore on active character now conflicts
- empty `{}` updates still succeed as no-ops
- avatar-only edits appear in version history and diff output
- avatar-only edits can be reverted from new-format snapshots
- legacy snapshots without avatar state remain readable and behave as explicitly
  incomplete history
- reverting to a legacy snapshot preserves the current avatar unchanged
- structured card parse failures are rejected
- explicit plain-text import mode still works as the intended synthetic mode
- image-file import and structured import produce equivalent normalized avatar
  storage
- oversized PNG metadata export is rejected before returning a non-round-trip
  PNG
- `/versions` and `/versions/diff` stay compact by default even when internal
  snapshots now include avatar state
- RG fail-closed behavior returns the expected `429` and `503` cases
- concurrent chat create cannot exceed configured cap
- concurrent message or multi-message persist cannot exceed configured cap
- abandoned or failed multi-message reservations are released or expire
  correctly
- hybrid exemplar fallback returns correct `total`
- world-book delete and detach responses carry the correct identifier field
- mixed-suite `503` reproducer or narrowed harness verifies the availability
  fix or residual boundary

Test categories:

- unit tests for DB lifecycle and rate-limiter contracts
- migration tests for SQLite schema evolution to v45
- Postgres-safe schema or translation checks for any new DB objects or trigger
  semantics
- integration tests for endpoint behavior where public contract changes
- concurrency tests for strict cap enforcement
- targeted mixed-suite or harness tests for the `503` investigation

## Risks And Mitigations

### Risk: Avatar Snapshots Inflate Sync Payloads

Base64 avatar state increases sync-log payload size.

Mitigation:

- reuse the normalized avatar representation already enforced for persisted
  character images
- treat legacy snapshots as incomplete instead of attempting expensive
  historical backfills
- keep default version-history responses compact even if internal snapshots grow

### Risk: Quota Serialization Becomes Backend-Specific

A SQLite-only transaction fix would leave Postgres vulnerable to the same race.

Mitigation:

- require explicit backend-safe serialization semantics in the DB abstraction
- add parity checks for the chosen serialization approach
- require reservation cleanup semantics so failed model calls do not leak quota

### Risk: Import Contract Change Breaks Implicit Workflows

Some callers may currently rely on malformed YAML silently becoming a synthetic
character.

Mitigation:

- preserve plain-text synthetic import only for explicit plain-text inputs
- make parse-failure rejection explicit for structured card mode
- cover the new contract with endpoint tests

### Risk: `503` Investigation Stays Nondeterministic

The failing path may remain hard to reproduce in a single-file rerun.

Mitigation:

- require a durable harness or narrowed failing boundary, not just a hunch
- keep instrumentation narrow and remove or reduce it once the failing boundary
  is understood

## Verification Contract

Before completion:

- run focused Characters, Character_Chat, ChaChaNotesDB, Streaming, and any
  new migration or concurrency tests covering the changed behavior
- run the mixed-suite or harness test used for the `503` investigation
- run Bandit from the project virtual environment on touched backend paths
- verify only the intended remediation files changed in the isolated worktree

## Success Criteria

This remediation is successful if:

- restore of an active character no longer succeeds as an idempotent restore
- empty `{}` updates remain supported no-ops
- avatar-only edits are represented in version history, diff output, and revert
  for new-format snapshots
- reverting to a legacy snapshot preserves avatar state instead of clearing it
- malformed structured card inputs fail clearly, while explicit plain-text
  imports still work intentionally
- avatar normalization is transport-independent
- PNG export either produces a re-importable artifact or fails clearly when the
  metadata ceiling would be exceeded
- default version-history APIs remain compact even though internal snapshots are
  lossless for avatar state
- chat and message caps hold under concurrent access, including multi-message
  persistence flows
- request throttling behaves fail-closed once enabled
- exemplar fallback pagination metadata is correct and world-book response
  payloads match their schema
- the mixed-suite `503` issue is fixed with regression coverage or narrowed to a
  clearly documented residual boundary with evidence
- SQLite and Postgres-safe behavior remain aligned for touched DB paths
