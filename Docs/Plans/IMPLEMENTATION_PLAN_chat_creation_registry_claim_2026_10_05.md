# Existing registry creation reservation prerequisite — TASK-13454.5

Baseline1a479bf485; current dev025627214c unchanged. Add only opt-in fail-closed reads and conditional expected-null claims to the existing bucket. No new storage authority, backend, queue or dependency.

## Stage 1: Reproduce ambiguity and duplicate reservations
**Goal**: strict read and physical-context claim behavior fails against current code.
**Success Criteria**: IO/corrupt/expired data is never absence; two same-key claims cannot both succeed.
**Tests**: actual registry and WebUI storage shim with serialized Web Lock/protocol failure doubles.
**Status**: Complete

## Stage 2: Extend existing operations
**Goal**: opt-in strict unpinned read and exact absence CAS.
**Success Criteria**: fail closed before unavailable/lockless writes; unchanged ordinary draft behavior.
**Tests**: new cases plus existing recovery/tabs/draft tests.
**Status**: Complete

## Stage 3: Verify and locally commit
**Goal**: bounded reviewed prerequisite.
**Success Criteria**: relevant suites/types/extension compile/no added lint/diff/two reviewers.
**Tests**: Bandit N/A to TypeScript/docs-only production. Native creation record, cold/repeat retry blocker, known-ID review UI, greeting opening, Sync, native browser/live evidence remain required integrations; this unit is not their acceptance.
**Status**: Complete

## Verification
Initial prerequisite red: 9 failures/1 pass. Reviewer-discovered stored literal null red: 2 failures. Final: 6 suites/106 passing with the actual WebUI storage shim and a serialized Web Lock double. WebUI typecheck, extension entrypoint compile, no added lint (2 files), and diff check pass. Actual installed Plasmo uses undefined for absence; safeStorageSerde preserves malformed values and stored null. Both independent reviewers found no remaining Critical/Important issue after correction. Logs: /private/tmp/chat-creation-registry-claim-final-{green,types,extension,lint}.log and /private/tmp/chat-creation-registry-null-red.log.

Evidence limits: independent bucket instances and a lock double, not actual physical browser contexts or native create/duplicate POST prevention. Extension compile is not a packaged build. No Python production touched; Bandit N/A. Actual durable intent/caller/UI and zero-provider greeting integration remain required under13454.
