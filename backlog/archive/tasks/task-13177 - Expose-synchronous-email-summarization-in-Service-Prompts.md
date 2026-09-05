---
id: TASK-13177
title: Expose synchronous email summarization in Service Prompts
status: In Progress
assignee: []
created_date: '2026-09-05 16:08'
labels: []
dependencies: []
documentation:
  - Docs/Design/email-summary-service-prompt.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add owner-scoped email system instructions through existing Service Prompts settings. Repair the email form provider/recursive option wiring and let the shared analyzer resolve configured or keyless credentials so the setting reaches actual analysis. User approved the bounded scope on 2026-09-05.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Email summarization is editable through the existing shared Settings and generic service prompt API.
- [ ] #2 One owner-specific prompt snapshot covers each synchronous email request including supported containers; explicit prompts, empty values, defaults and reset retain precedence.
- [ ] #3 Canonical and legacy provider fields and recursive summaries reach analysis; configured credentials and keyless providers are not blocked by the email processor.
- [ ] #4 Disabled analysis and nested attachment behavior remain unchanged; corrupt overrides fail before input processing and temporary lookup connections close on their worker.
- [ ] #5 Focused backend, shared UI, OpenAPI, lint and Bandit checks pass with regression coverage.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the approved design and establish the email/Service Prompts baseline. 2. Add failing end-to-end prompt and analysis wiring tests plus Settings coverage. 3. Implement minimal registry, form, endpoint and processor changes. 4. Verify, independently review, update task and commit.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
