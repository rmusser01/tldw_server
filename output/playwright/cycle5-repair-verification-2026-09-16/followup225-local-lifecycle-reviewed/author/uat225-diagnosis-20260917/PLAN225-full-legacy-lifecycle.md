# TASK13260.164 / UAT225 — full inactive-Sync suggestion lifecycle

Design: `FULL-LIFECYCLE-DESIGN225.md`. Parent-approved retirement direction; Source013's Stage A remains separate and does not close UAT225. No new persistent namespace/table/schema or Job payload rekey.

## Stage 1: Causal boundaries and local product adapter

**Goal:** Reproduce missing local decisions/lifecycle on real SQLite and official PostgreSQL. Implement only a disjoint, not-yet-wired local adapter after meaningful RED.

**Success criteria:** Actual lower-level link/organization stores preserve deterministic identity, owner validation, guarded product/receipt atomicity, normalized collision behavior and caller rollback. Adapter is not imported by runtime/factory until shared release.

**Tests:** New `test_suggestion_legacy_mutations.py` and `test_suggestion_legacy_lifecycle.py`; reuse existing fixtures and official required-PG runner. No model inference. Capture original RED before implementation, including factory absence. Existing registered canonical fixtures are controls, not proof of fresh local authority.

**Status:** Complete — disjoint causal RED and 18-case GREEN retained. Shared integration is now implemented after Stage A release; the original checkpoint remains historical evidence.

## Stage 2: Shared local lifecycle integration

**Goal:** After Stage A freeze/review, add exact local authority through existing store, factory and Jobs/maintenance paths.

**Success criteria:** Fresh unbound owner can admit, retrieve/generate via bounded fake provider, publish, read, decide, cancel and reconcile; all scopes remain owner/dataset exact. Same-transaction note invalidation includes local data. No public arbitrary legacy scope or relaxed foreign authority.

**Tests:** Actual route/factory/store and real Jobs backend; late enqueue, cancellation, replay, stale source/target, cleanup, unchanged canonical controls, restricted-role PG.

**Status:** Complete — real local factory, Jobs admission/worker publication, decisions, note invalidation and maintenance integrated. Committed168 supplies merge survivor resolution. Review exposed and causal tests proved a SQLite prior-device tag consumer mismatch; the local-only correction preserves canonical Sync and PG ownership. See the final implementation report and retained original candidate.

## Stage 3: Canonical enrollment retirement

**Goal:** Reserve only the actual canonical row with all unrelated flags false; retire old local authority atomically, then drain rows/Jobs through bounded internal maintenance.

**Success criteria:** Both profile default-creation paths and direct bootstrap entries fence before product snapshots; local mutation acquires authority lock before product locks. Existing equal flags preserved; mismatch aborts; later domain binding works. Old payloads/terminal envelopes unchanged; no late publication/product acceptance; unbound late Jobs still discovered and cancelled.

**Tests:** Real PG barriers and SQLite connection controls at admission/enqueue/bind, claim/guard, keyword/membership, stage/publish, profile bootstrap and Personal Context binding. Interrupted/resumed enrollment, caller rollback, all-false row, existing same/different target, original immutable receipt retention and bounded retirement cleanup.

**Status:** Complete — permanent SQLite/official-PG REDs precede reservation and retirement implementation. Both profile entrypoints, direct bootstrap preconditions, concurrent authority locks, late enqueue beyond the admission grace, fair budgeted discovery and immutable retention controls are implemented.

## Stage 4: Review and acceptance handoff

**Goal:** Verify integrated source and disclose remaining boundaries.

**Success criteria:** Required PG + SQLite relevant suites have no skips; Ruff/Bandit/compile/diff checks recorded; exact source/test manifest and independent review clear. Parent-owned native acceptance remains separate. Local keyword-merge redirect gap must be causally classified and resolved/explicitly bounded before claiming complete documented parity.

**Tests:** Existing suggestion API/retrieval/lifecycle/acceptance/Jobs/maintenance suites plus Notes Sync bootstrap/binding. No broad reruns absent a concrete concern.

**Status:** In Progress — final source/tests frozen for independent review; required-PG focused173 and adjacent200 final suites pass with zero skips. Parent-owned native acceptance remains pending. No UAT225 or full-matrix completion claim.

## Leases and prohibitions

- Stage A shared suggestion paths released by parent after commit043b8cd1dd. Keyword/schema/organization store changes remain separately owned under168. Preserve Stage A source/test snapshots and explicit unavailable-decision controls.
- Parent owns Backlog, tracker, browser/native data/runtime/config, staging and commits.
- No new authority infrastructure, broad SQL rewrite, provider calls, token/session access, private native DB inspection or schema mutation outside official disposable fixtures.
- Preserve original Stage A 4 RED / 4 controls and every later failed fixture/candidate receipt. Do not overwrite causal evidence.
