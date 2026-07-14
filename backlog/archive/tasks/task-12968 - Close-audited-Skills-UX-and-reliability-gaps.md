---
id: TASK-12968
title: Close audited Skills UX and reliability gaps
status: Done
labels:
- skills
- webui
- ux
- accessibility
- reliability
priority: high
modified_files:
- Docs/Design/2026-07-14-skills-ux-gap-closure-design.md
- apps/packages/ui/src/assets/locale/en/option.json
- apps/packages/ui/src/public/_locales/en/option.json
- apps/packages/ui/src/components/Common/WorkspaceConnectionGate.tsx
- apps/packages/ui/src/components/Option/Skills/Manager.tsx
- apps/packages/ui/src/components/Option/Skills/SkillDetailsDrawer.tsx
- apps/packages/ui/src/components/Option/Skills/SkillDrawer.tsx
- apps/packages/ui/src/components/Option/Skills/SkillPreview.tsx
- apps/packages/ui/src/components/Option/Skills/SkillsWorkspace.tsx
- apps/packages/ui/src/components/Option/Skills/skill-form-utils.ts
- apps/packages/ui/src/components/Option/Skills/skills-query-state.ts
- apps/packages/ui/src/components/Option/Skills/__tests__
- apps/packages/ui/src/i18n/icu-format.ts
- apps/packages/ui/src/i18n/__tests__/icu-format.test.ts
- apps/packages/ui/src/services/tldw/domains/workspace-api.ts
- apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.skills.test.ts
- apps/packages/ui/src/types/skill.ts
- apps/tldw-frontend/e2e/utils/skills-fixtures.ts
- apps/tldw-frontend/e2e/workflows/tier-5-specialized/skills.spec.ts
- tldw_Server_API/app/api/v1/endpoints/skills.py
- tldw_Server_API/app/api/v1/schemas/skills_schemas.py
- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
- tldw_Server_API/app/core/Skills/skills_service.py
- tldw_Server_API/tests/Skills
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the confirmed /skills UX, accessibility, safety, responsive, and power-user gaps found in the 2026-07-14 beginner and expert workflow review. Keep the change isolated from MCP catalog-render work. Prefer existing Skills APIs and frontend patterns; add backend persistence only where the requested history/trash workflow cannot be delivered honestly in the client.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rows provide persistent view, use-in-chat, copy-invocation, duplicate, test, edit, export, and delete workflows with clear feedback.
- [x] #2 Beginner authoring defaults to structured fields with validated generated SKILL.md, while advanced users can edit raw source.
- [x] #3 Dirty editor/import drafts are protected across every close path and recoverable for the browser session.
- [x] #4 Pressing Enter in test arguments performs dry render only; explicit live execution is required, and stale async results never appear for a different skill.
- [x] #5 Search, form fields, upload, dialogs, focus behavior, headings, errors, and touch targets meet the audited keyboard and screen-reader requirements.
- [x] #6 Filters are compact, active constraints are visible/removable, no-results recovery clears filters, and query/filter/sort/pagination state is URL-backed.
- [x] #7 Selection persists across pages and filters; bulk export and existing bulk operations use the full selected set predictably.
- [x] #8 The page is usable at 390x844 without body overflow and retains an efficient desktop table workflow.
- [x] #9 Delete supports immediate undo, conflict recovery preserves user work, and version/history or trash behavior is implemented only with durable truthful persistence.
- [x] #10 Focused unit tests and deterministic browser UAT cover beginner, expert, mobile, accessibility, stale-request, dirty-draft, URL-state, and failure workflows.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design: Docs/Design/2026-07-14-skills-ux-gap-closure-design.md

Baseline: focused Skills Vitest suite passed on origin/dev (6 files, 74 tests).

Scope decision: implement durable Trash and immediate Undo; do not add unbounded active revision snapshots, a second named-view persistence system, or unrelated MCP catalog work.

All five implementation stages completed. Review follow-ups were addressed with focused regressions: lossless raw source, explicit conflict review, 100-item bulk selection enforcement, browser-history restoration, read-only View skill routing, retryable post-commit cleanup, cross-process Trash serialization, interrupted-delete recovery, fail-closed bundle validation, and cancellation-safe lock/transaction completion.

Final independent review reported no actionable findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed the audited /skills beginner, power-user, accessibility, reliability, and responsive gaps. Added guided authoring and session draft recovery; safe dry-run keyboard behavior and stale-request cancellation; complete details/use/copy/duplicate/edit/export/delete actions; compact URL-backed filters and view state; cross-page selection with the backend's 100-item cap; an equivalent 390px mobile workflow; lossless raw-source editing; and durable per-user Trash with versioned restore, permanent purge, immediate Undo, explicit conflict recovery, retryable cleanup, cross-process locking, interrupted-delete reconciliation, and cancellation-safe transactions. Also fixed ICU first-value memoization and replaced brittle modal-animation assertions with user-visible close behavior. Verification passed: 144 focused frontend tests, 253 backend Skills tests, 13 deterministic Playwright scenarios, TypeScript typecheck, focused ESLint, Ruff, Python compilation, Bandit with 0 findings, and git diff --check. Independent final review found no actionable issues. Known skips/blockers: none. The cited MCP catalog service-locator function is absent from current origin/dev and no unrelated MCP change is included.
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
