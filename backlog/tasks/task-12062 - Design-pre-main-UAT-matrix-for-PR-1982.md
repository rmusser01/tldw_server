---
id: TASK-12062
title: Design pre-main UAT matrix for PR 1982
status: Done
labels:
- uat
- release
- design
modified_files:
- Docs/superpowers/specs/2026-06-29-pre-main-uat-matrix-design.md
- backlog/tasks/task-12062 - Design-pre-main-UAT-matrix-for-PR-1982.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the approved pre-main UAT design for PR #1982 covering Docker and local single-user WebUI, live OpenAI and llama.cpp provider gates, basic document/character-chat journey, and advanced power knowledge journey.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wrote and committed the approved pre-main UAT matrix design at `Docs/superpowers/specs/2026-06-29-pre-main-uat-matrix-design.md`. The design covers Docker and local single-user WebUI environments, live OpenAI and llama.cpp provider gates, basic document ingest plus roleplay character-chat journey, advanced power knowledge workflow, bounded matrix control, isolation, evidence, severity, and fix policy. Spec review subagent approved with no blocking issues. Advisory notes for the implementation plan: define exact provider model/config and make the core answer path explicit. Verification: `git diff --check` passed for the spec/task files. Bandit skipped because this is documentation-only.
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
