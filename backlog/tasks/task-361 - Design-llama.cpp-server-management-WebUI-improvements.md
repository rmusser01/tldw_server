---
id: TASK-361
title: Design llama.cpp server management WebUI improvements
status: Done
assignee: []
created_date: '2026-05-15 03:20'
updated_date: '2026-05-15 03:24'
labels:
  - design
  - llamacpp
  - webui
  - self-hosted
dependencies: []
references:
  - 'https://github.com/m94301/llama-studio'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a reviewed design spec for improving self-hosted llama.cpp server management in the WebUI. The design should stay conservative for v1: one managed llama.cpp server, guided setup, safe model inventory, warnings-first hardware guidance, explicit provider wiring, and clear active-vs-saved config state. It should be grounded in the existing llama.cpp handler/API/WebUI contracts and note which llama-studio ideas are intentionally reused or deferred.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec documents the approved v1 scope and explicitly defers multi-session llama.cpp management.
- [x] #2 Spec covers backend/API facade, config persistence, model inventory, hardware guidance, WebUI workflow, safety/error handling, testing, and rollout.
- [x] #3 Spec addresses review risks: active vs saved config divergence, restart semantics, path-based model identity, provider wiring boundaries, option safety, optional metadata/hardware probes, and bounded log access.
- [x] #4 Spec is reviewed for internal consistency before requesting user review.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create the approved llama.cpp server management design spec under Docs/superpowers/specs.
2. Include the review corrections as explicit requirements: active vs saved config, restart semantics, model identity, provider wiring boundaries, option safety, optional probes, and bounded log access.
3. Run a lightweight docs verification pass and manual spec consistency review.
4. Commit the spec and Backlog task record, then request user review before implementation planning.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created Docs/superpowers/specs/2026-05-15-llamacpp-server-management-webui-design.md. Ran git diff --check on the spec and reviewed for missing placeholders, contradictions, and V1 scope creep. Tightened launch-profile language so profile persistence is deferred rather than required V1 scope. No Bandit run: documentation-only design task with no Python/code changes.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a reviewed design spec for llama.cpp server management WebUI improvements. The spec keeps V1 aligned to the current single managed llama.cpp server architecture while adding a guided admin flow for config, validation, inventory, warnings-first hardware guidance, explicit provider wiring, bounded log access, and rollout/testing expectations. The review pass tightened risky areas before handoff: saved vs active config must be explicit, restart semantics must be honest, registered model paths require stable model IDs, provider wiring stays opt-in, dynamic option discovery cannot bypass backend safety, metadata/hardware probes are best-effort, and launch-profile storage is deferred from required V1 scope.

Verification: git diff --check passed for the spec. Manual spec review found and resolved one V1 scope ambiguity around per-model launch profiles. Bandit was skipped because this task changed documentation and Backlog records only.
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
