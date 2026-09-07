# Writing feedback Service Prompts

Approved scope: expose Writing Playground mood and Echo instructions in the existing shared Service Prompts settings. No new settings or persistence system.

- `writing.feedback.mood`: literal `system_semantics` and `classification_semantics`; the one-word/seven-mood contract and passage carrier remain fixed.
- `writing.feedback.echo`: atomic literal `alex_system`, `sam_system`, `max_system`, `riley_system`, `jordan_system`; persona identities, rotation and passage carrier remain fixed.
- Preserve byte-equivalent default chat payloads, model/temperature/token settings, debounce/threshold behavior, text caps and 20-reaction history.
- Capture a fresh owner-bound snapshot per request. Keep at most one scope lease per feedback channel while state is visible; abort/discard stale work and clear feedback on scope changes. Release leases on replacement or unmount.
- Reuse packaged fallback only for older-server 404/missing definitions. Other lookup errors must not send a default request.

## Stage 1: Definitions and Settings

**Goal:** Register both prompts with shared client types, labels and packaged defaults.
**Success Criteria:** Atomic save/reset and older-server behavior match existing Service Prompts.
**Tests:** Registry/API and shared snapshot/Settings tests.
**Status:** Complete

## Stage 2: Scoped feedback requests

**Goal:** Apply snapshots in the existing feedback hook.
**Success Criteria:** Custom semantics reach generation with locked contracts; stale results never update feedback.
**Tests:** Hook payload, enablement, validation, limits, scope-switch and lifecycle regressions.
**Status:** Complete

## Stage 3: Verification and review

**Goal:** Verify both clients' shared integration and touched backend scope.
**Success Criteria:** Focused tests, lint, Bandit and code review completed; limitations recorded in TASK-13215.
**Tests:** Focused Vitest and pytest suites; touched-scope lint and Bandit.
**Status:** Complete

## Verification

- Backend registry and API: 97 passing tests, including atomic feedback save/reset.
- Shared client: 281 passing focused tests across Settings, feedback hooks, Writing Agent, snapshots and scope transport helpers.
- Direct-browser refresh/scope transport: 47 passing tests.
- English locale mirrors: 12 new entries match across both bundles.
- Bandit: no findings. Ruff: clean. ESLint: no errors; 10 existing explicit-any warnings in untouched portions of the shared server service.
- Global shared-UI typecheck: 158 existing diagnostics, none in changed files (required an 8 GiB Node heap).
- Independent review approved after a test-first fix preventing late scope errors from cancelled requests from clearing newer feedback.
- Full repository tests, production builds and live browser smoke tests were not run.
