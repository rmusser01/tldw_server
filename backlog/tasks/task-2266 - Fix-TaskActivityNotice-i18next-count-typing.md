---
id: TASK-2266
title: Fix TaskActivityNotice i18next count typing
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-07 00:16
labels:
- webui
- notes
- typescript
- verification
dependencies: []
references:
- TASK-2265 verification blocker
modified_files:
- apps/packages/ui/src/components/Notes/TaskActivityNotice.tsx
- apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.task-activity.test.tsx
- apps/packages/ui/src/public/_locales/en/option.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the UI TypeScript verification blocker where TaskActivityNotice passes a string label through the reserved i18next count option, which must remain numeric for typed translation options.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 TaskActivityNotice passes a numeric value through the i18next reserved count option.
- [x] #2 TaskActivityNotice still renders the human-readable task count label in the summary text.
- [x] #3 A regression test covers the translation option shape so future TypeScript checks catch regressions.
- [x] #4 Full UI TypeScript verification passes after the fix.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause: TaskActivityNotice used count for display text, but i18next reserves count for numeric pluralization. Fixed by passing count: events.length and interpolating countLabel separately. Updated the English locale template from {{count}} to {{countLabel}}.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the UI TypeScript blocker by keeping i18next count numeric in TaskActivityNotice and introducing countLabel for the rendered task-count phrase. Added regression coverage for the translation option shape and verified the notes regression, Mermaid chat artifact suites, full UI TypeScript check, diff whitespace, and locale JSON parsing. Bandit is not applicable because only TypeScript, test, JSON locale, and Backlog task files were touched.
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
