---
id: TASK-422
title: Design Next dev server memory leak investigation
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-17 21:57'
labels:
  - performance
  - webui
  - extension
  - investigation
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-17-next-dev-server-memory-leak-investigation-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a design/spec for investigating the WebUI/extension Next dev server memory growth where the `next-server` process reached roughly 6 GB RSS. Scope is evidence gathering and investigation design only: identify the target process, define measurement strategy, triage branches, guardrails, and expected deliverables before implementation or fixes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec documents the confirmed target as the Next dev server rather than the FastAPI backend.
- [x] #2 Spec defines process telemetry, dev-server log, and browser/request correlation measurements.
- [x] #3 Spec defines triage branches for idle growth, route-load growth, request storms, and extension/background traffic.
- [x] #4 Spec records guardrails for no code changes during first-pass evidence gathering and no process restart before evidence capture.
- [x] #5 Backlog task links the design spec and records verification/skip notes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and critique-hardened the Next dev server memory leak investigation design spec at `Docs/superpowers/specs/2026-05-17-next-dev-server-memory-leak-investigation-design.md`. The spec confirms the target process as the high-RSS Next dev server rather than the FastAPI backend, defines the process/log/browser-request measurement plan, records triage branches for idle growth, route-load growth, request storms, extension/background traffic, and dev-tooling isolation, and sets guardrails against premature fixes or restarting the high-RSS process before evidence capture. The first critique pass patched the spec to require fresh process rediscovery, process-tree sampling, worktree-specific log paths, self-watch loop checks, route cache versus unbounded-growth separation, and a durable evidence artifact path. The second critique pass patched the spec to distinguish RSS from JavaScript heap, external/native memory, file-descriptor growth, and build-cache growth before assigning blame to application code, and to require user agreement before intrusive diagnostics. No runtime code was changed; Bandit was skipped as docs-only.
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
