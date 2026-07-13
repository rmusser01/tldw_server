---
id: TASK-12112
title: Design user-customizable service prompt registry and settings
status: In Progress
labels:
- prompts
- design
- webui
- browser-extension
- backend
references:
- TASK-2341
documentation:
- Docs/superpowers/specs/2026-07-12-user-customizable-service-prompts-design.md
modified_files:
- Docs/superpowers/specs/2026-07-12-user-customizable-service-prompts-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design a governed, per-user Service Prompt Registry that exposes a curated allowlist of backend-owned content-generation prompts through a dedicated shared WebUI/browser-extension settings page. The design must preserve deployment defaults and explicit request overrides, provide strict variable validation, preview, revisions, background-job pinning, context-integrity enforcement, and broad content-facing service migration while keeping reusable Prompt Library, Prompt Studio, security, routing, judge, and machine-protocol prompts separate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Document the Service Prompt Registry architecture, ownership boundaries, precedence, persistence, revision, and job-pinning contracts.
- [ ] #2 Define the authenticated API and shared WebUI/browser-extension settings experience, including validation, preview, comparison, reset, history, conflicts, capability states, and responsive behavior.
- [ ] #3 Define the allowlist eligibility policy, context-integrity behavior, privacy/security safeguards, failure semantics, deployment-default compatibility, and explicit exclusions.
- [ ] #4 Define a broad content-facing migration inventory strategy, reviewable rollout slices, verification matrix, and measurable completion gates.
- [ ] #5 Review the written design through the required independent spec-document-reviewer loop and record the approved spec.
<!-- AC:END -->

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
