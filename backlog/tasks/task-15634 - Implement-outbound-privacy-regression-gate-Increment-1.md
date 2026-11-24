---
id: TASK-15634
title: Implement outbound privacy regression gate Increment 1
status: In Progress
labels:
- security
- privacy
- ci
dependencies:
- TASK-15633
references:
- Docs/Design/2026-11-24-outbound-destination-inventory-design.md
- Docs/superpowers/plans/2026-11-24-outbound-privacy-increment-1.md
- Docs/ADR/097-reviewed-outbound-destinations-and-privacy-modes.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the reviewed product destination and asset inventory plus the required CI and local regression check for covered startup, setup, and application runtime flows. Scope this task to Increment 1.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A versioned inventory validates exact origins and real asset identities, distinguishes declared from transport-verified evidence, and guards new usage in its covered paths.
- [x] #2 The same local and required-CI runner exercises fresh server startup, first-run setup, selected runtime behavior, and opted-in synthetic download and provider paths with no real external traffic.
- [x] #3 Guarded production browser smoke covers the shipped setup wizard, Next.js including its server process, and the packaged Chrome worker and content script. Search results use a bundled icon.
- [x] #4 License-admitted required contexts fail on violations, missing evidence, or failed coverage. Relevant changes trigger the check, diagnostics are redacted, and coverage limits are documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: yes. Proposed ADR-097 covers the durable inventory and required CI rules for this increment. The linked plan documents the five test-first stages and the supported Linux validation profile.
Plan verification on 2026-11-24: existing required-gate classifier and direct HTTP guard baseline passed 99 tests in a plugin-disabled run. Offline planning probes exercised the real download SDK fixture redirect on 0.36.0 and 1.33.0 and the real installer and provider adapter paths with synthetic transports. Linux namespace and production browser execution remain implementation acceptance checks.
Requester-approved plan review amendments: use disposable Linux test containers with disabled external networking and no host service socket mounts, require stop-file ASGI shutdown and final evidence, specify runtime Next auth and full Chromium/cache settings, verify exact SDK repository identity and zero raw attempts in both opt-in children, and trigger required checks on both projected package source trees. ADR-097 remains the governing proposed decision. Verified document links, stage order, example syntax, and review-fix coverage. Container and browser preflights remain implementation acceptance checks. No runtime or workflow code changed.
Implementation started in the isolated outbound-egress-design worktree. Stage commits will record test-first verification and limitations.
Stage 1: observed missing-validator test failure, then 73 passed with the direct HTTP guard. Bandit on the validator and its unit tests and git diff --check passed. Package registry and downstream hosts are unverified where installer argv cannot identify them.
Stage 2: 39 harness and strict-guard tests passed after red runs, and the existing guard suite delivered 27 passes and one expected subprocess-only skip. The real Linux Docker controls and guarded default smoke passed, including startup, local setup state and notes, ASGI shutdown, and zero recorded attempts. Follow-up review regression tests require the preflight at internal entry points and reject local-only config edits before image preparation. The smoke uses the ASGI shutdown-complete message because Uvicorn 0.54 changed a private field name. Bandit and Ruff passed.
Stage 3: 77 tests passed in the isolated Linux backend image (Hugging Face Hub 1.33.0, zero skips). All 11 opt-in subprocess cases also passed with Hub 0.36.0 under macOS OS-level network denial in Python 3.12. The fixtures assert both the built-in and configured OpenAI destinations and leave HF_ENDPOINT unset on the successful Hub cases. Pip/uv and Git coverage is limited to exact argv dispatch. Ruff and Bandit passed.
Stage 4: all four browser scenarios ran together in disconnected Linux Docker and passed with zero violations. Wizard persistence, quickstart WebUI cookie and bundled search icon, the packaged copilot worker POST and rendered reply, and independent content-script activation are proved. Added the app early ONNX telemetry opt-out after a native DNS attempt was reproduced and verified absent. The missing-credentials check now precedes DNS on sample OpenAI endpoints, verified by 63 provider-readiness tests. The WebUI models catalog uses four exact unavailable local provider fixtures. Only Chromium’s verified failed UDP IPv6 route query is classified as platform behavior; actual network traffic remains subject to the exact fixture rules. 18 harness tests, 35 Add Sources tests, both TypeScript checks, Ruff, Bandit, and whitespace checks passed.
Stage 5: ran both image preparations and all seven scenarios from the final checkout in disconnected Linux Docker. Backend and browser independently passed serially in 93.26 and 86.29 seconds. The final harness and workflow suite passed 395 focused tests, and independent review reran 261 CI contract and runner tests. The workflow sentinel propagated a nonzero exit. Missing and stale preparation records failed closed. Report scans found no raw credentials or URLs. actionlint 1.7.12, Ruff on harness code, Bandit with no new findings, and git diff --check passed. GitHub-hosted timing awaits the first branch run.
Pre-push review follow-up authorized. Verify three stages test-first: dedicated license-admitted CI jobs and browser path detection, Chromium worker WebSocket detection, and safe diagnostics plus exception anchor validation. ADR required: no new ADR. The follow-up implements the existing Increment 1 decisions in proposed ADR-097 and the linked implementation plan.
Pre-push stage 1: auth, shared API and smoke inputs now trigger a dedicated browser job without selecting the frontend unit shards. Both dedicated runner jobs reclaim unused SDK space and record cold build time and lowest sampled Docker free disk. Required contexts fail for a selected job that fails, is cancelled or skips. Red run reproduced the missing flag and runner topology; 255 classifier and workflow contract tests passed with the fix, as did actionlint, Ruff and whitespace checks. Bandit found no new findings. Hosted x64 metrics await GitHub execution.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented and verified the versioned outbound inventory, isolated backend and browser fixtures, and matching local and required CI checks. Fixed unsolicited ONNX Runtime startup telemetry, DNS lookups for credential-free sample OpenAI endpoints, and the Research Workspace remote favicon. Exactly seven covered scenarios passed with no violations. Package and Git validation covers installer arguments and the selected vendor downloads use synthetic transports.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
