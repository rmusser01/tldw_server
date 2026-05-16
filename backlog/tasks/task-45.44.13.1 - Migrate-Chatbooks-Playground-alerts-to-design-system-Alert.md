---
id: TASK-45.44.13.1
title: Migrate Chatbooks Playground alerts to design-system Alert
status: Done
assignee: []
created_date: '2026-05-16 00:57'
updated_date: '2026-05-16 03:24'
labels:
  - design-system
  - webui
  - product-state
dependencies: []
references:
  - apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx
  - >-
    apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx
  - >-
    apps/packages/ui/src/components/Option/Chatbooks/__tests__/ContentTypePicker.error-state.test.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
documentation:
  - >-
    Docs/superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-bodies/migration-long-tail.md
parent_task_id: TASK-45.44.13
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate ChatbooksPlaygroundPage product-state AntD Alert usage to the shared design-system Alert primitive while preserving visible copy, import/export error handling, OpenWebUI preview/hydration feedback, and job status messaging.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ChatbooksPlaygroundPage no longer imports or renders AntD Alert for product-state messages.
- [x] #2 Focused Chatbooks tests assert representative warning/error/info states render the shared design-system Alert marker.
- [x] #3 The design-system product-state baseline no longer contains ChatbooksPlaygroundPage AntD Alert exceptions.
- [x] #4 Focused tests, design-system verifier, locale/JSON checks if touched, git diff --check, and applicable TypeScript filtering are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Migrated ChatbooksPlaygroundPage and ContentTypePicker product-state alerts from AntD Alert to the design-system Alert primitive. Added DS marker assertions for load-error, hydration-warning, and capability-unavailable states. Verification: Chatbooks focused Vitest 2 files passed; product-state guard test passed; verify:design-system-state passed; baseline JSON parse passed; git diff --check passed. Full UI tsc was run and remains blocked by pre-existing repo-wide unrelated TypeScript errors outside touched files. Bandit skipped because this task only touches frontend TypeScript/JSON/backlog files.

PR #1738 review fixes: made static Chatbooks notices polite status regions, rendered preview/hydration warning arrays as block children instead of newline-joined text, and added focused assertions for include-all status notices and hydration warning block rendering. Verification: ContentTypePicker.error-state.test.tsx passed; targeted OpenWebUI hydration preview test passed; product-state guard test passed; verify:design-system-state passed; baseline JSON parse passed; git diff --check passed. Full OpenWebUI import file still hit existing local 10s timeouts in unrelated tests; full UI tsc remains blocked by pre-existing unrelated errors outside touched paths. Bandit not applicable for frontend TypeScript/backlog-only changes.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Chatbooks Playground product-state alert UI now uses the shared design-system Alert primitive, with stale Chatbooks Alert baseline entries removed and focused tests covering design-system Alert markers for representative states.
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
