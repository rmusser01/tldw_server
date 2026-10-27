# First-run setup before connection — TASK-13272

Implements the approved all-ten chat review: a new extension user must be able to establish a connection before first-run backend state is requested. Full WebUI, extension Options, and the home wizard use the same connection decision. Missing configuration is a normal setup state; supplied but unreachable configuration keeps its real error/retry behavior. No synthetic backend readiness or setup completion is introduced.

ADR check: ADR required: no. Existing canonical browser request transport and connection-account ownership conventions govern; ADR-059 governs task editing. This repairs use of existing transport, scoped ephemeral loading and existing setup components, without a public API, durable schema or dependency change.

## Stage 1: Qualify initial-load and connection boundaries
**Goal**: Reproduce requests/errors before configuration and stale setup results after a connection change.
**Success Criteria**: Actual canonical transport distinguishes absent extension config from valid quickstart, runtime cookie, hosted and advanced transport. Held old-target results cannot certify a new target. Existing same-owner remount deduplication and completed state with metadata failure remain covered.
**Tests**: Actual hook with declared configuration/backend IO; no backend reads without a target, save-config resume, configured failure, held account/server replacement and remount controls. Existing setup transport authentication fallback tests remain intact. Physical direct and worker tests qualify original provider payload routing, replacement/ABA, readiness continuation and in-flight cancellation; the retired direct ambient-config read gate is recorded separately.
**Status**: Complete

## Stage 2: Repair shared loading and setup entry
**Goal**: Resolve configured transport before loading and scope existing ephemeral state to its original connection.
**Success Criteria**: Mounted config/account changes resume loading; stale state/error/catalogs retire. Every initial read and setup mutation carries the original configuration snapshot and owner cancellation signal through the shared domain; worker requests retain the existing authority/epoch and cancellation guards. Unconfigured /setup and home show actionable connection choices, no Retry/operator recovery or backend step writes. Existing configured wizard steps, recovery, and Skip continue to work.
**Tests**: Existing SetupEntryChoice, actual Wizard and route tests plus preconnection click/no-write controls; actual hook refusal/positive controls from Stage 1.
**Status**: In Progress

## Stage 3: Verify, review and commit
**Goal**: Close the bounded root while continuing the complete remediation queue.
**Success Criteria**: Affected tests, both client compilers, authored static checks, normal hooks, original-source review and canonical tracking pass. Current browser evidence is separately identified; no participant or packaged-extension evidence is inferred from doubles.
**Tests**: Hook/entry/wizard/route/first-run API/authority/handoff suites, owned formatting/diff/task gates and a current-bundle first-run walkthrough where available. No authored Python: Bandit is not applicable to TS/Markdown changes.
**Status**: Not Started
