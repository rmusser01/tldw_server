---
id: TASK-516
title: Design chat rails UX rebaseline and remediation from origin/dev
status: Done
labels:
- chat
- ux
- frontend
- spec
priority: high
documentation:
- Docs/superpowers/specs/2026-05-27-chat-rails-ux-rebaseline-design.md
modified_files:
- Docs/superpowers/specs/2026-05-27-chat-rails-ux-rebaseline-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design the clean origin/dev rebaseline workflow for verifying the /chat cockpit rails, re-running the /chat UX evaluation on the correct rail-enabled page, and planning remediation for remaining chat and extension handoff issues.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Draft a design spec that records the branch-provenance cause for missing siderails, defines the clean origin/dev rebaseline workflow, and gates implementation on a refreshed rail-enabled /chat audit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the chat rails UX rebaseline design. The spec records the branch-provenance cause for the missing-siderails observation, defines the clean origin/dev branch/worktree workflow, requires exact git provenance evidence, requires screenshot-backed rail-enabled /chat verification, and gates implementation fixes on a refreshed audit artifact.
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
