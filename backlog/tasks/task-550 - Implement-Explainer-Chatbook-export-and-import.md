---
id: TASK-550
title: Implement Explainer Chatbook export and import
status: To Do
labels:
- backend
- chatbooks
- explainer
- implementation
priority: High
references:
- TASK-546
- TASK-547
- Docs/superpowers/specs/2026-06-09-explainer-workspace-design.md
- Docs/superpowers/plans/2026-06-09-explainer-workspace-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/Explainer/chatbook_adapter.py
- tldw_Server_API/app/core/Chatbooks/chatbook_models.py
- tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py
- tldw_Server_API/app/core/Chatbooks/chatbook_service.py
- tldw_Server_API/app/api/v1/endpoints/explainer.py
- tldw_Server_API/app/api/v1/schemas/explainer.py
- tldw_Server_API/tests/Explainer/test_explainer_chatbook_export.py
- tldw_Server_API/tests/Chatbooks/test_explainer_session_content_type.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement Task 3 from Docs/superpowers/plans/2026-06-09-explainer-workspace-implementation-plan.md: first-class Chatbook explainer_session export/import plus generated_document subtype import fallback. Follow TDD and keep Explainer serialization in core/Explainer/chatbook_adapter.py.
<!-- SECTION:PLAN:END -->

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
