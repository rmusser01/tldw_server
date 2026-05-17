---
id: TASK-422
title: Design Next dev server memory leak investigation
status: Done
labels:
- performance
- webui
- extension
- investigation
priority: high
documentation:
- Docs/superpowers/specs/2026-05-17-next-dev-server-memory-leak-investigation-design.md
modified_files:
- Docs/superpowers/specs/2026-05-17-next-dev-server-memory-leak-investigation-design.md
- backlog/tasks/task-422 - Design-Next-dev-server-memory-leak-investigation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a design/spec for investigating the WebUI/extension Next dev server memory growth where the `next-server` process reached roughly 6 GB RSS. Scope is evidence gathering and investigation design only: identify the target process, define measurement strategy, triage branches, guardrails, and expected deliverables before implementation or fixes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Spec documents the confirmed target as the Next dev server rather than the FastAPI backend.
- [ ] #2 Spec defines process telemetry, dev-server log, and browser/request correlation measurements.
- [ ] #3 Spec defines triage branches for idle growth, route-load growth, request storms, and extension/background traffic.
- [ ] #4 Spec records guardrails for no code changes during first-pass evidence gathering and no process restart before evidence capture.
- [ ] #5 Backlog task links the design spec and records verification/skip notes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and critique-hardened the Next dev server memory leak investigation design spec at `Docs/superpowers/specs/2026-05-17-next-dev-server-memory-leak-investigation-design.md`. The spec confirms the target process as the high-RSS Next dev server rather than the FastAPI backend, defines the process/log/browser-request measurement plan, records triage branches for idle growth, route-load growth, request storms, extension/background traffic, and dev-tooling isolation, and sets guardrails against premature fixes or restarting the high-RSS process before evidence capture. The user-requested critique pass patched the spec to require fresh process rediscovery, process-tree sampling, worktree-specific log paths, self-watch loop checks, route cache versus unbounded-growth separation, and a durable evidence artifact path. No runtime code was changed; Bandit was skipped as docs-only.
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
