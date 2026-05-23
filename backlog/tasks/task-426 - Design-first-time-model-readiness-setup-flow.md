---
id: TASK-426
title: Design first-time model readiness setup flow
status: Done
labels:
- design
- setup
- webui
- embeddings
- audio
documentation:
- Docs/superpowers/specs/2026-05-18-first-time-model-readiness-setup-design.md
modified_files:
- Docs/superpowers/specs/2026-05-18-first-time-model-readiness-setup-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a design spec for a first-time setup/readiness flow that exposes curated choices for chat provider/endpoint/model, embedding model provisioning, transcription readiness, and secondary TTS readiness through backend /setup and native WebUI first-run setup using shared /api/v1/setup/* APIs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design-only task. Spec captures the approved first-time model readiness setup architecture, components, data flow, permissions, error handling, and testing plan. No runtime implementation in this task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created Docs/superpowers/specs/2026-05-18-first-time-model-readiness-setup-design.md as a design-only spec for first-time model readiness setup. The spec records the approved unified readiness wizard direction, native WebUI backed by setup APIs, curated profiles, explicit Provision now gating, lane readiness semantics, permission boundaries, error handling, and test strategy. A follow-up critique pass hardened the design by making restart/admin/remote-blocked states overlays instead of lane statuses, treating TTS as secondary metadata inside the speech lane, requiring WebUI fallback when the backend local setup guard blocks native first-run setup, requiring pollable provisioning instead of long-held HTTP requests, clarifying secret handling, and gating trusted custom HF models behind advanced acknowledgement. Verification: rg found no unfinished-work markers; git diff checks passed for touched docs/task metadata. Bandit skipped because this task changes documentation/task metadata only.
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
