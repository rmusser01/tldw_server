---
id: TASK-530.15
title: Add strict Skills live integration certification
status: In Progress
labels:
- skills
- webui
- extension
- e2e
- release-gate
priority: high
parent_task_id: TASK-530
documentation:
- Docs/superpowers/specs/2026-07-15-skills-live-integration-certification-design.md
modified_files:
- Docs/superpowers/specs/2026-07-15-skills-live-integration-certification-design.md
- backlog/tasks/task-530.15 - Add-strict-Skills-live-integration-certification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an explicit release-gate command that certifies the /skills lifecycle against a runner-managed disposable backend in both the WebUI and the packaged browser extension. The extension path must prove that every Skills API request is owned by the exact MV3 background service worker, with no direct-page fallback. Keep existing mocked suites unchanged, keep the gate out of default PR CI, and avoid product-code changes unless the live gate reproduces a real defect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A single explicit command creates an isolated single-user backend profile, starts required processes, runs both surfaces sequentially, and fails rather than skips on missing prerequisites.
- [ ] #2 WebUI completes empty-state, create, exact search, dry render, reload persistence, Trash, restore, second Trash, purge, and direct API postcondition checks.
- [ ] #3 The packaged MV3 extension completes the target-specific lifecycle and every /api/v1/skills request is owned by chrome-extension://<extension-id>/background.js with zero page-owned fallback requests.
- [ ] #4 The extension relay ledger records sanitized request ownership and outcomes, with exactly one successful create, dry execute, restore, and purge plus two successful Trash mutations.
- [ ] #5 The runner attempts both surfaces, records infrastructure failures explicitly, permits at most one same-port backend restart for second-surface evidence, and never converts a crashed run into a pass.
- [ ] #6 Disposable runtime data and browser profiles are removed; retained evidence is sanitized and limited to logs, JSON results, relay ledger, and failure screenshots.
- [ ] #7 Focused unit tests cover environment isolation, commands, aggregation, restart, teardown, artifact redaction, profile ownership, and relay-ledger rules.
- [ ] #8 Existing mocked WebUI and extension Skills suites remain unchanged and pass as regression verification.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
