---
id: TASK-12987
title: Add dedicated audio.cpp batch STT provider
status: In Progress
assignee: []
created_date: '2026-07-26 03:27'
updated_date: '2026-07-26 08:26'
labels:
  - stt
  - benchmark
  - audio-cpp
dependencies: []
references:
  - Docs/superpowers/specs/2026-07-25-audio-cpp-batch-stt-provider-design.md
  - >-
    Docs/superpowers/plans/2026-07-25-audio-cpp-batch-stt-provider-implementation-plan.md
  - 'https://github.com/0xShug0/audio.cpp/blob/main/app/server/README.md'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a first-class external-server-only audio.cpp STT adapter and provider registration. The adapter must connect to a user-managed audiocpp_server, validate the pinned HTTP contract, support ordinary batch transcription and the native STT benchmark, and never download, build, launch, restart, terminate, or silently fall back from audio.cpp.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Registers canonical audio-cpp provider with audiocpp and audio_cpp aliases.
- [ ] #2 Uses network-free planning followed by consent-gated health/model discovery and WAV multipart transcription.
- [ ] #3 Records descriptive audio.cpp backend/model metadata while leaving weight identity unresolved and policy gates ineligible.
- [ ] #4 Supports strict and normalized benchmark scoring with separate cold-first and warm timing.
- [ ] #5 Normal CI uses fake transports and upstream-shaped fixtures; live audio.cpp coverage is opt-in.
- [ ] #6 Configuration and user documentation describe setup, limitations, network consent, and true cold-start procedure.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-07-25-audio-cpp-batch-stt-provider-implementation-plan.md task-by-task using TDD, focused commits, independent review, Bandit, and PR gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-07-25: Human-approved external-server-only design is recorded in Docs/superpowers/specs/2026-07-25-audio-cpp-batch-stt-provider-design.md. Scope is WAV batch transcription, network-free planning, consent-gated discovery, unresolved weight identity, and no fallback, retry, download, conversion, or process supervision. Independent design review and re-review approved the hardened spec.

The approved implementation plan is Docs/superpowers/plans/2026-07-25-audio-cpp-batch-stt-provider-implementation-plan.md. Independent plan review led to cross-event-loop single-flight cache semantics, raw-origin path validation, and complete declared PCM-payload validation. Plan re-review approved it. Execution uses subagent-driven TDD with spec and quality reviews per task.

Task 1 complete at 2c7058d348. TDD RED: 8 expected failures and 2 passes from missing metadata opt-in. GREEN: 15 finalizer tests and 107 adapter tests; Ruff, scoped Bandit, compile, and diff checks clean. Spec review approved. Quality review approved after isolating the 9-key allowlist regression from metadata-size validation.

Task 2 complete at d9d4e52d7112a3885ce1258a88e07f7c96668d0a. Initial TDD RED covered the missing module and selector behavior. Review-driven RED then confirmed 6 origin boundary-whitespace failures, 2 host-validation failures (IPvFuture and 254-character DNS), and 1 leaked exploding-__float__ failure. GREEN: 127 dedicated audio.cpp tests and 12 config regressions; Ruff, compileall, diff check, and scoped Bandit with 0 findings. Independent spec re-review returned Spec compliant; independent quality re-review found no issues and approved proceeding. Commit scope is exactly the five planned Task 2 files.

Active stage: Task 3, bounded pinned audio.cpp HTTP response contract and strict PCM RIFF/WAVE validation. No HTTP execution, discovery cache, or adapter registration is in scope for this stage.

Task 3 complete at 6046350092. TDD evidence: initial contract/WAV RED 54 failures plus 1 provenance pass, then 55 GREEN; clarified boundary RED 16 then GREEN 16; JSON exception RED/GREEN 1; wave.open mandate RED/GREEN 1; quality-fix selection RED 3 of 9 with 6 characterization passes, then 9 GREEN. Final full module 195 passed. Ruff format/check, compileall, git diff checks, and scoped Bandit (0 findings/errors) passed. Spec review first required replacing the bespoke RIFF parser with standard wave.open; amended re-review passed. Quality review found relative-base duplication and a potentially hanging FIFO regression plus four test-hardening items; all were fixed, and final spec and quality re-reviews approved with no remaining issues. Scope is exactly the three planned Task 3 files and contains no HTTP/cache/adapter behavior. Active stage: Task 4, secure discovery single-flight cache and no-retry multipart execution over an injected fakeable transport.

Task 4 quality-review fix pass: verified and addressed seven Important findings plus both Minor findings with TDD. RED evidence: 30 selected regressions produced 20 expected failures/10 characterization passes, demonstrating unconsumed cancelled-follower exceptions, single-cancel cleanup, loop-keyed default client leakage risk, same-loop sync deadlock risk, omitted frozen-route invariants, post-buffer response limits, status-after-body classification, timing-based cross-loop tests, and synchronous WAV validation. A separate cancellation-handoff RED showed an off-loop WAV handle could be abandoned after caller cancellation. GREEN implementation adds identity-safe leader-loop tracking and fail-fast, consumes follower wrapper exceptions without cancelling shared work, cancellation-hardened response/client/WAV cleanup, one trust_env=False exact-transport client per real transcription, complete route invariant checks, status-before-content handling, off-loop leak-free WAV validation, deterministic event-based cross-loop tests, and an opt-in decoded-body streaming limit in afetch that skips non-2xx bodies (including the HTTPX error-body hook) and observes at most limit+1 bytes. Verification: 487 audio.cpp + full HTTP-client tests passed; Ruff passed on new/changed scope (existing http_client TRY203 and legacy test import-order baselines excluded); compileall and git diff --check passed; Bandit scanned both touched production files with 0 findings.

Task 4 second quality-review fix pass: strict TDD RED selected 8 bounded-transport regressions with 5 expected failures and 3 characterization passes. Failures proved HTTPX and aiohttp consumed compressed 2xx bodies, bounded requests did not override caller Accept-Encoding, and aiohttp did not disable auto-decompression. GREEN forces Accept-Encoding identity, disables aiohttp auto-decompression only for bounded calls, rejects encoded 2xx responses before body iteration, reconstructs loaded HTTPX responses before restoring copied original headers, skips encoded non-2xx bodies while preserving usable status/headers, retains exact limit+1 streaming bounds, and marks diagnostic cross-loop threads daemon while keeping release/join cleanup. Verification: focused 8 passed; full audio.cpp plus HTTP-client suites 491 passed; Ruff lint clean with documented legacy TRY203/I001 exclusions; compileall and git diff --check clean; Bandit scanned 6506 LOC across both Task 4 production modules with 0 findings/errors. Whole-file Ruff formatting remains a pre-existing baseline and was not mechanically rewritten.

Task 4 final hardening after quality approval: bounded HTTPX identity responses now use raw iteration, reject the first boundary-crossing transport chunk before copying it, and never allocate an extra decoded rechunk buffer. The regression distinguishes raw from decoded iteration, uses an oversized first chunk plus an unconsumed sentinel, and clarifies that the application accumulator is bounded while transport-owned chunk allocation remains the HTTP transport's responsibility. Main-agent verification: 7 focused bounded-response tests passed; full audio.cpp plus HTTP-client suites passed 492/492; focused Ruff passed with only documented legacy TRY203/I001 exclusions; Bandit remained 0 findings/errors over the touched production scope. Task 4 final HEAD will be amended with this hardening. Active stage after final re-review: Task 5 provider registration and immutable planning.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
