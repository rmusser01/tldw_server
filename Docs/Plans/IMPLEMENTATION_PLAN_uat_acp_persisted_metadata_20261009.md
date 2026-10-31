# C05 persisted session metadata after runtime loss

Associated task: TASK-13260.278.18.83.45 (root-owned; scoped note recorded before edits).

## Contract and ADR assessment

Persisted session ownership authorizes stored detail, usage, events, artifacts, diagnostics, audit and reconciliation reads after a restart. Runtime execution, control, live streams, fork, rollback and checkpoints continue to require the existing current runner ownership check. Missing and foreign records remain `404 session_not_found`. No session restoration, ID reuse or output reconstruction is introduced.

ADR-016 (`Docs/ADR/016-acp-session-and-orchestration-persistence.md`) governs persistent ACP records; ADR-018 (`Docs/ADR/018-resource-governance-endpoint-policy-and-route-map.md`) governs existing endpoint authorization. This repairs their current contract and introduces no new durable architecture rule, so no new ADR is needed.

Named production scope: ACP endpoint, response schema, non-starting runner accessor; shared UI ACP types/store/parent/chat panel. The accessor supports existing direct and sandbox clients without starting either. Explicit `runtime_available=false` prevents connection and execution controls; missing legacy flags retain prior behavior. Stored status and transcript contents remain unchanged.

## Stage 1: Causal ownership and UI controls
**Goal**: Demonstrate stored owner reads fail without runtime and that unavailable stored sessions incorrectly offer runtime controls.
**Success Criteria**: Real temporary ACP DB/store reopened after writing, normal cookie authentication and actual endpoints establish owner/foreign/missing and execution guard behavior. Mounted parent/chat tests exercise actual REST mapping/store and runtime-unavailable controls.
**Tests**: New `test_acp_persisted_session_reads.py`; new `ACPPlayground.persisted-history.test.tsx`.
**Status**: Complete

## Stage 2: Minimal persisted-read/runtime split
**Goal**: Reuse one exact persisted-owner helper for named stored reads and project truthful current runtime availability.
**Success Criteria**: Owner reads succeed, foreign/missing remain404, all existing execution/stream guard calls unchanged; no runner startup for reads; unavailable UI controls disabled without changing legacy behavior.
**Tests**: Causal modules in supported Python3.12 and configured UI/Web clients; genuinely affected store/chat/read contracts.
**Status**: Complete

## Stage 3: Scoped validation and independent review
**Goal**: Freeze source/tests after normal scoped checks and independent review.
**Success Criteria**: AST/compile and baseline/current Ruff/Bandit/ESLint introduce no findings; full new bodies and connected consumers reviewed. Root owns combined types/build and actual live acceptance.
**Tests**: Scoped static/security and exact hash guards; no runtime/provider replay.
**Status**: In Progress

## Attempt ledger

Maximum three distinct causal/repair attempts per issue; fixture/setup failures are reported separately. Exact pre-unit production bytes are retained in functions memory only; no proof snapshots or manifests.

Initial Python red: five failures/natural1; real reopened store/auth/runtime-absence preconditions passed. Owner read/list failures demonstrated the missing contract. Runtime-start-call assertions in foreign/missing cases also failed as expected; the combined output was truncated, so the remaining runtime-negative failure is not claimed as causal before its exact diagnostics are recovered during current qualification.

UI attempt1 causal red: three failures/two legacy-positive passes/natural1; actual parent attempted connection, store retained waiting-permission state, chat offered Stop. Attempt2 after minimal source: four pass/one failure/natural1; every functional runtime fence passed, but the final test GET-only oracle incorrectly treated omitted `RequestInit.method` as non-GET. BEFORE fixture correction: use native fetch's documented omitted-method GET default in that oracle; preserve no-POST/no-socket/control assertions and qualify only that failed ID. Initial runner path was absent before Vitest started and is a setup failure, not a causal attempt.

UI failed-ID fixture correction natural0/one pass/four intentionally deselected,1.41s Vitest. Five distinct new cases are retained, not one final combined green command. Python attempt2: four pass/one failed,831 warnings,18.78s pytest/23.309s outer; every stored read and REST runtime-refusal oracle passed. The remaining WebSocket negative used untrusted `http://testserver` Origin and correctly received4403 before ownership. BEFORE correcting only that fixture: use existing configured trusted `http://localhost:3000`, retaining exact4404 ownership oracle; only failed ID qualifies next. The prior WebSocket result is excluded from causal ownership evidence.

Python failed-ID correction: natural0, one pass/235 warnings,7.65s pytest/10.849s outer. Five distinct initial cases are retained. Genuinely affected current-client/store qualification: natural0,65 passed/535 warnings/no skips,13.54s pytest/17.605s outer; three new accessor cases plus62 existing `test_acp_session_store.py`/`test_acp_sessions_db.py`. Total70 distinct Python cases (eight new/62 existing), not one combined70-green command. The direct positive uses existing controlled stdlib ACP peer/public session creation; sandbox coverage verifies its actual client-selection branch and missing ownership without starting containers. Live sandbox/provider execution is outside acceptance.

Configured UI affected five existing modules: natural0,18 passed/no skips,2.20s Vitest. Retained new five plus existing18 gives23 distinct UI cases across split green commands. Configured Web full six modules: natural0,23 passed/no skips,1.74s Vitest. Exact existing modules: store `acp-sessions.test.ts`, `acp-sessions.persistence.test.ts`, ACPPlayground `ACPChatPanel.test.tsx`, `ACPPlayground.connection.test.tsx`, `ACPSessionPanel.test.tsx`. Installed Node20.19.5/Vitest4.0.18/8192MB; normal defaults. AntD/JSDOM stylesheet parse notices are retained as advisories.

Static source-only primary venv Python3.11: all four Python files AST/compile pass; baseline/current Bandit zero findings/errors (new test initially had three B108 metadata literals, now narrowly explained nosec annotations, unchanged semantics). Ruff: endpoint/new test0, runner retains one identical B009, schema retains I001+UP037, zero new findings. New import-format warning fixed without behavior changes. One Ruff diff command hit sandbox cache permissions after emitting its suggested import-only diff; subsequent stdin static checks succeeded and no environment workaround was added. All five TypeScript files parser diagnostics0. Apps-base configured ESLint baseline/current0 errors, two identical inherited store warnings, no new warnings; inherited missing apps/pages advisory. Early verbose static output truncation was a reporting setup failure and corrected by printing only lint diagnostics.

Self-review: full changed functions and both new modules read. Exact source-segment comparison confirms `_require_session_access`, REST prompt/cancel/close/teardown/reconcile/updates, WS/SSH/SSE stream, fork/rollback/checkpoints unchanged. Named stored-read helpers preserve existing redaction, lineage, audit, pagination and current-config/runtime owner verification. No persisted runtime restoration or transcript reconstruction. Source/tests frozen for root independent review, combined types/build and actual live owner-history acceptance, which remain open.
