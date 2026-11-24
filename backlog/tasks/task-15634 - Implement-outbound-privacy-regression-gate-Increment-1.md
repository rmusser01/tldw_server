---
id: TASK-15634
title: Implement outbound privacy regression gate Increment 1
status: To Do
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
