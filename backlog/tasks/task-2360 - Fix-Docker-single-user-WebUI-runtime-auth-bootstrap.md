---
id: TASK-2360
title: Fix Docker single-user WebUI runtime auth bootstrap
status: In Progress
labels:
- docker
- webui
- auth
- setup
priority: High
documentation:
- Docs/superpowers/specs/2026-06-24-docker-webui-runtime-auth-bootstrap-design.md
modified_files:
- Docs/superpowers/specs/2026-06-24-docker-webui-runtime-auth-bootstrap-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address Docker single-user startup/auth issues by designing and implementing a runtime WebUI auth bootstrap, setup remote-write configuration, and related Docker/docs/test updates. Track stale mcp_unified Docker guidance separately in the design rather than adding a nonexistent root package copy in this branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design first under Docs/superpowers/specs, then wait for explicit user review before implementation planning. Runtime-auth bootstrap should keep the WebUI image generic and avoid baking SINGLE_USER_API_KEY into NEXT_PUBLIC_X_API_KEY for Docker single-user quickstart.
<!-- SECTION:PLAN:END -->

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
