---
id: TASK-546
title: Design explicit sidepanel chat WebUI handoff
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-29 05:11
labels:
- chat
- extension
- ux
dependencies: []
references:
- TASK-531
- TASK-534
documentation:
- Docs/superpowers/specs/2026-05-29-sidepanel-chat-webui-handoff-design.md
modified_files:
- Docs/superpowers/specs/2026-05-29-sidepanel-chat-webui-handoff-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Document the approved explicit sidepanel-to-WebUI /chat handoff design: route-only launch remains default, while a separate Continue in WebUI action transfers the current sidepanel draft plus visible page context through a one-time short-lived handoff token.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec documents explicit Continue in WebUI handoff in ControlRow while preserving the route-only open action.
- [x] #2 Spec defines fail-closed handoff storage with read-back verification, unguessable IDs, TTL, one-time consume, and payload bounds.
- [x] #3 Spec defines WebUI import behavior including composer prefill, existing-draft conflict choices, imported-context banner, request inclusion, removal behavior, and hash-router cleanup.
- [x] #4 Spec defines focused regression tests and implementation slices for storage, sidepanel action, WebUI consumption, and packaged extension smoke.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/specs/2026-05-29-sidepanel-chat-webui-handoff-design.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design direction approved in brainstorming: separate Continue in WebUI action, ephemeral token, composer prefill without auto-send, visible already-captured page context only, one-time short-lived consume, sidepanel draft preserved. Local spec review added ControlRow placement, fail-closed storage, request inclusion semantics, safe consume timing, hash-router cleanup, non-overwrite behavior for existing WebUI drafts, role-play query merge rules, unguessable IDs, concrete payload bounds, and malformed package rejection.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reviewed and hardened the sidepanel-to-WebUI /chat handoff design before implementation planning. The spec now preserves the route-only open action, places the explicit Continue in WebUI action in ControlRow, defines a fail-closed local storage service contract, makes imported context part of the next chat request, protects existing WebUI drafts, handles hash-router query cleanup, and defines concrete regression coverage.
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
