---
id: TASK-510
title: Create PRD for Mermaid diagram rendering in assistant chat markdown
status: Done
labels:
- docs
- prd
- chat
- frontend
references:
- https://github.com/ggml-org/llama.cpp/pull/24032
modified_files:
- Docs/superpowers/specs/2026-06-04-chat-mermaid-diagrams-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a product requirements/design document for adding Mermaid.js diagram rendering to assistant-facing chat markdown surfaces. Scope is PRD/spec only; implementation planning follows separately after user review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created Docs/superpowers/specs/2026-06-04-chat-mermaid-diagrams-design.md as the PRD for assistant-facing Mermaid diagram rendering in shared chat markdown. Incorporated upstream llama.cpp PR #24032 lessons, local React Markdown/Mermaid anchors, requirements, settings, security, accessibility, performance, acceptance criteria, and test plan. Spec review loop completed: first review found tool-message ambiguity; PRD was revised to scope v1 to assistant-role/greeting assistant markdown only; second review approved. Follow-up local review applied user-approved improvements: clarified generated-SVG handling for the preview dialog, pinned the chat setting to ChatSettingsConfig/DEFAULT_CHAT_SETTINGS, deferred aliases/raw SVG/artifact-viewer unification, and marked the PRD ready for implementation planning. Verification: checked for TODO/TBD/placeholders and contradictions in the touched PRD sections; Bandit not applicable because this task only changes documentation and Backlog metadata.
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
