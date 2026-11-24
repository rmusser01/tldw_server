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
- [ ] #1 A versioned inventory validates exact origins and real asset identities, distinguishes declared from transport-verified evidence, and guards new usage in its covered paths.
- [ ] #2 The same local and required-CI runner exercises fresh server startup, first-run setup, selected runtime behavior, and opted-in synthetic download and provider paths with no real external traffic.
- [ ] #3 Guarded production browser smoke covers the shipped setup wizard, Next.js including its server process, and the packaged Chrome worker and content script. Search results use a bundled icon.
- [ ] #4 License-admitted required contexts fail on violations, missing evidence, or failed coverage. Relevant changes trigger the check, diagnostics are redacted, and coverage limits are documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: yes. Proposed ADR-097 covers the durable inventory and required CI rules for this increment. The linked plan documents the five test-first stages and the supported Linux validation profile.
Plan verification on 2026-11-24: existing required-gate classifier and direct HTTP guard baseline passed 99 tests in a plugin-disabled run. Offline planning probes exercised the real download SDK fixture redirect on 0.36.0 and 1.33.0 and the real installer and provider adapter paths with synthetic transports. Linux namespace and production browser execution remain implementation acceptance checks.
Requester-approved plan review amendments: use disposable Linux test containers with disabled external networking and no host service socket mounts, require stop-file ASGI shutdown and final evidence, specify runtime Next auth and full Chromium/cache settings, verify exact SDK repository identity and zero raw attempts in both opt-in children, and trigger required checks on both projected package source trees. ADR-097 remains the governing proposed decision. Verified document links, stage order, example syntax, and review-fix coverage. Container and browser preflights remain implementation acceptance checks. No runtime or workflow code changed.
Implementation started in the isolated outbound-egress-design worktree. Stage commits will record test-first verification and limitations.
Stage 1: observed missing-validator test failure, then 73 passed with the direct HTTP guard. Bandit on the validator and its unit tests and git diff --check passed. Package registry and downstream hosts are unverified where installer argv cannot identify them.
Stage 2: 39 harness and strict-guard tests passed after red runs, and the existing guard suite delivered 27 passes and one expected subprocess-only skip. The real Linux Docker controls and guarded default smoke passed, including startup, local setup state and notes, ASGI shutdown, and zero recorded attempts. Follow-up review regression tests require the preflight at internal entry points and reject local-only config edits before image preparation. The smoke uses the ASGI shutdown-complete message because Uvicorn 0.54 changed a private field name. Bandit and Ruff passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
