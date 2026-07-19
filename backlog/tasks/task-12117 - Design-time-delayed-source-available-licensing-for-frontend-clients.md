---
id: TASK-12117
title: Design time-delayed source-available licensing for frontend clients
status: Done
labels:
- licensing
- frontend
- design
priority: high
documentation:
- Docs/superpowers/specs/2026-07-19-frontend-source-available-licensing-design.md
modified_files:
- Docs/superpowers/specs/2026-07-19-frontend-source-available-licensing-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Document the approved licensing design for protecting the WebUI, browser extension, shared UI package, and admin UI under BSL 1.1 with a rolling 24-month transition to AGPL-3.0-only, while keeping the server and remaining repository GPL-3.0-only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design records the protected path boundary and unchanged GPL backend boundary.
- [x] #2 Design defines permitted use, Competing Offering constraints, rolling Change Dates, CLA, trademark policy, and selective commercial licensing.
- [x] #3 Design documents provenance, artifact, CI, legal-review, and release gates.
- [x] #4 Approved design is committed without modifying unrelated user work.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Approved design recorded at `Docs/superpowers/specs/2026-07-19-frontend-source-available-licensing-design.md`.
- Self-review completed: no placeholders; nested documentation and third-party exclusions made explicit; SPDX/package notice method clarified.
- Verification: placeholder scan clean and `git diff --check` will be run on the staged task scope.
- Bandit is not applicable because this task changes Markdown records only.
- Legal review is intentionally a release gate for later implementation, not a blocker to completing this design task.
- Commit `5b10f82218` records the approved design and task without staging unrelated work.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Recorded the user-approved policy for BSL 1.1 protection of the WebUI, extension, shared UI, and admin UI; rolling 24-month conversion to AGPL-3.0-only; GPL backend separation; permitted-use and competing-offering boundaries; CLA, trademark, commercial-license, provenance, artifact, CI, and release controls.
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
