---
id: TASK-12969
title: Close audited Skills UX and reliability gaps
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-15 04:17'
labels:
  - skills
  - webui
  - ux
  - accessibility
  - reliability
dependencies: []
references:
  - 'PR #2732'
  - Docs/Design/2026-07-14-skills-ux-gap-closure-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement and verify the confirmed /skills UX, accessibility, safety, responsive, power-user, and reliability gaps from the 2026-07-14 workflow review. Keep the change isolated from unrelated MCP and Research Workspace work.
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
- [x] #9 Delete supports immediate undo, conflict recovery preserves user work, and Trash behavior uses durable truthful persistence.
- [x] #10 Focused unit tests and deterministic browser UAT cover beginner, expert, mobile, accessibility, stale-request, dirty-draft, URL-state, and failure workflows.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented beginner and power-user Skills workflows: guided/raw authoring, session draft recovery, persistent row actions, details/test/edit/export flows, compact URL-backed filters, cross-page selection, bounded bulk export, responsive mobile behavior, and accessibility/focus improvements.
Hardened durable Trash/restore/purge behavior with version checks, cross-process locking, update/delete serialization, cancellation-safe mutations, first-use reconciliation, fail-closed archive/symlink handling, and retryable cleanup.
Scoped SkillsService reuse to the owning CharactersRAGDB lifetime, bounded If-Match parsing, ignored stale async file-preview results, and restored focus through details-to-test transitions.
Rebased onto latest dev and renumbered this Skills task from TASK-12968 to TASK-12969 because dev independently assigned TASK-12968 to Research Workspace work.
Reopened after Qodo posted six review threads. Added sanitized contextual Trash logging; missing return annotations; real Skills ingress mapping through the Resource Governor; the missing unit marker; TASK-12969 references on conditional symlink skips; and fail-closed validation for malformed registry rows with missing directory paths. The Resource Governor fix intentionally uses the active centralized enforcement path rather than the documented no-op compatibility dependency.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented and merged the audited Skills UX and reliability gap closure in PR #2732. The work covers beginner and power-user workflows, accessibility, URL-backed filtering, selection and bulk actions, durable Trash, cancellation-safe UI state, bounded imports, and crash-safe atomic bundle publication/recovery. Final recovery review added version-marked backups, reversible rollback, fail-closed ambiguous-state handling, and replacement-marker reconciliation before every lock-held sync debounce.

Recorded verification: 312 backend Skills tests passed; 173 focused frontend Skills/ICU/client tests passed; Playwright passed 13 deterministic scenarios with 3 documented optional live-backend skips; TypeScript, Python compile, locale JSON parsing, resource-governance tests, and git diff checks passed; Bandit reported 0 findings. Ruff was clean on the changed Skills Python scope, and independent re-review confirmed all findings resolved.

PR #2732 merged into dev as bb0c9d6bd565e669db0f00f7b248717c4ae5247f on 2026-07-15 with all inline review threads resolved. Known policy exception: the mandatory requester-authored Change summary placeholder was not replaced before merge.
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
