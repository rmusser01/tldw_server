---
id: TASK-546
title: Design Explainer workspace page
status: Done
labels:
- frontend
- design
- explainer
priority: Medium
references:
- https://breakdowner.exe.xyz/
- TASK-546
documentation:
- Docs/superpowers/specs/2026-06-09-explainer-workspace-design.md
modified_files:
- Docs/superpowers/specs/2026-06-09-explainer-workspace-design.md
- backlog/tasks/task-546 - Design-Explainer-workspace-page.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a design spec for a persisted Explainer workspace inspired by Breakdowner, with Goal and Sources tabs, recursive explanation trees, citation-aware grounding modes, and backend persistence from day one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Write a design-only spec for the persisted Explainer workspace, review it for product/architecture risks, update the Backlog task with touched files and verification, then commit the spec and task record.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the Explainer workspace design spec after review. The spec now makes Chatbook export a first-release requirement: exporting creates one Chatbook item containing the complete explainer session tree, clarifying questions, selected answers, citations, grounding metadata, generation metadata, and rendered reading form. It also notes the current Chatbooks implementation constraint: either add a first-class explainer_session ContentType or use generated_document with metadata.subtype = explainer_session as a compatibility bridge. Verification: reviewed the edited spec against the existing Chatbooks schemas/models/service shape; scoped diff review showed only the spec and TASK-546 changes; ASCII scan found no non-ASCII content. Bandit skipped because this revision only changes documentation and Backlog metadata.
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
