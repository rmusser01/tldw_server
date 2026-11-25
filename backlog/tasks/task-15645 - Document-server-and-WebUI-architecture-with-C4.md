---
id: TASK-15645
title: Document server and WebUI architecture with C4
status: Done
labels:
- documentation
- architecture
references:
- Docs/Architecture/c4.md
- Docs/Architecture/server.md
- Docs/Architecture/webui.md
- Docs/Architecture/admin-ui.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give new contributors an architecture walkthrough and experienced contributors a source-linked module reference for the FastAPI server, end-user WebUI, and standalone Admin Panel.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Provide a system context and container view that distinguishes the two WebUI deployments and their API boundaries.
- [x] #2 Provide selective component views and source entry points for FastAPI, the end-user WebUI, and the standalone Admin Panel.
- [x] #3 Link the reference from contributor entry points and verify the Mermaid diagrams and relative links.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. Documents existing boundaries. See ADR-003 for Jobs and Scheduler, ADR-021 for service lifecycle, ADR-025 for provider routing, ADR-049 for chat history ownership, and ADR-052 for admin claims.
Verification on origin/dev worktree base 6d31f3d1a7. All five Mermaid diagrams parsed and rendered in Chromium with Mermaid 11.12.2. Checked 121 changed relative links and anchors, README and docs index path hygiene, public/private documentation boundary, and Backlog task IDs. Fifteen focused CI tests passed. Git whitespace check passed. Bandit covered the touched documentation directory and found zero Python lines. This documentation change required no runtime test run.
Pull request [#3603](https://github.com/rmusser01/tldw_server/pull/3603) targets dev.
PR #3603 follow-up. The onboarding docs gate reproduced six MkDocs warnings for relative links from the published architecture overview to repository-only files. Made those links point to GitHub and regenerated the tracked published overview. The fresh published tree matches the tracked files, a refreshed strict MkDocs build passed, and all 214 docs tests passed. See the [failed check](https://github.com/rmusser01/tldw_server/actions/runs/45443228379).
Renumbered the C4 task after TASK-15633 was assigned to the legacy media source hash parser on dev. The duplicate ID had blocked backend-required on PR #3603.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the C4 context and container views, source-linked component views of FastAPI and the two web clients, and request walkthroughs for grounded chat, research and admin operations. Linked the reference from the README and architecture index.
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
