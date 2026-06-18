---
id: TASK-2386
title: Implement Workspace Phase 2 frontend context contracts pilot
status: In Progress
labels:
- workspace
- phase2
- frontend
- acp
priority: High
references:
- https://github.com/rmusser01/tldw_server/issues/1993
- https://github.com/rmusser01/tldw_server/issues/1984
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track #1993 implementation for server-authoritative frontend Workspace context contracts, using Research Workspace and ACP Playground as the pilot surfaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Spec documents the server Workspace model as the authoritative frontend contract source.
- [ ] #2 Shared frontend contract types/helpers normalize server workspace, membership, active context, eligibility, and recovery responses without inventing parallel semantics.
- [ ] #3 Research Workspace pilot consumes the shared contract for active workspace context/recovery copy.
- [ ] #4 ACP Playground pilot consumes the shared contract for session workspace state and mismatch/recovery copy.
- [ ] #5 Tests prove global browse/list rendering is not filtered by active workspace context.
- [ ] #6 Focused frontend tests pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- User clarification: goal is to unify on the server Workspace model. Client-local Research Workspace state can cache/hydrate and decorate, but server workspace identity, memberships, eligibility decisions, and recovery reason codes are authoritative.
- Keep #1993 scoped to frontend/client contracts plus Research Workspace and ACP Playground pilots. Do not build #1994 activity/index UI in this slice.
- Design spec: `Docs/superpowers/specs/2026-06-18-workspace-frontend-server-context-contract-design.md`.
- Implementation plan: `Docs/superpowers/plans/2026-06-18-workspace-frontend-server-context-contract.md`.
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
