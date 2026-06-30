---
id: TASK-2264
title: Design chat Mermaid card artifact rail support
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-06 23:51'
labels:
  - chat
  - mermaid
  - webui
  - artifacts
  - spec
dependencies: []
references:
  - ggml-org/llama.cpp#24032 review context
  - 'tldw_server PR #2268 merged Mermaid assistant chat rendering'
  - TASK-495 OpenUI dynamic chat rendering
documentation:
  - Docs/superpowers/specs/2026-06-06-chat-mermaid-card-artifact-rail-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a design spec for opening or pinning assistant Mermaid diagram blocks as chat cards/artifacts using the existing shared renderer and artifact rail boundaries, preserving Mermaid source, assistant-only behavior, and current inline markdown rendering.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design spec covers product goals, non-goals, UX, data shape, security, test plan, risks, and implementation stages for Mermaid chat cards.
- [ ] #2 Spec reuses the existing chat artifact rail and diagram artifact kind instead of creating a Mermaid-specific card system.
- [ ] #3 Spec preserves assistant-only Mermaid rendering and leaves user messages unchanged.
- [ ] #4 Backlog task references the design spec and records markdown-only verification.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/specs/2026-06-06-chat-mermaid-card-artifact-rail-design.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Drafted a PRD/design spec for opening assistant Mermaid diagram blocks into the existing chat artifact rail as diagram artifacts. The design intentionally avoids backend persistence, keeps Mermaid source canonical, preserves assistant-only markdown gating, and documents markdown-only verification with Bandit not applicable for this design-only change.

Verification: git diff --check passed. Bandit skipped because this change only adds Markdown design/task documentation and touches no executable Python code.

Design review update: tightened the PRD to make Mermaid artifact actions explicitly opt-in per markdown surface, keep QuickChat/reasoning/fallback surfaces unchanged by default, and require message/context-aware artifact ids so jump-to-source cannot collide across repeated diagrams.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Draft PRD reviewed and amended. It is ready for implementation approval.
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
