---
id: TASK-303.2
title: Harden moderation content rules UI
status: Done
assignee: []
created_date: '2026-05-12 18:21'
updated_date: '2026-05-12 19:27'
labels:
  - moderation
  - webui
  - rules
dependencies:
  - TASK-303.1
documentation:
  - >-
    Docs/superpowers/plans/2026-05-12-moderation-review-rules-remediation-implementation-plan.md
  - >-
    Docs/superpowers/specs/2026-05-12-moderation-review-rules-remediation-design.md
parent_task_id: TASK-303
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 2 of the moderation review/rules remediation plan. The Content Rules surface at /moderation/rules should become lint-aware, safer for destructive raw edits/uploads, easier to scan/filter, and clearer about test-result explanations while preserving the existing backend moderation service contract and the Stage 1 route split.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Loaded blocklist rows use backend lint metadata when present and otherwise merge lintBlocklist results by line index.
- [x] #2 Comments and blank rows are represented as non-active rows and excluded from active counts and status totals.
- [x] #3 Content Rules include search, active-only filtering, pattern/action/category filters, and sortable rows without regressing existing rule editing workflows.
- [x] #4 Raw replace and file upload paths show lint/diff preview confirmation before updateBlocklist can be called, and invalid lint rows block confirmation.
- [x] #5 Managed delete and raw replace expose a session undo path until reload or the next conflicting load.
- [x] #6 Test sandbox results include a clear explanation for disabled engine, disabled phase, no match, matched rule, user override, and global fallback outcomes.
- [x] #7 Focused Vitest coverage proves the new row normalization, filters/sorting, preview confirmation, undo path, and result-explanation behavior.
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Stage 2 Content Rules hardening. Managed blocklist rows now merge backend lint metadata or frontend lint fallback, classify comments and blanks as non-active, and support search, filters, sorting, active counts, and session undo for managed deletes. Raw replace and blocklist upload now require lint preview confirmation before updateBlocklist runs, invalid rows block confirmation, and confirmed raw replacements can be undone during the session. Test sandbox results now explain disabled engine, disabled phase, no-match, matched-rule, per-user override, and global fallback outcomes. Policy and user override copy now separates global read-only policy from runtime and per-user scopes. Verification: focused Stage 2 Vitest red then green, full ModerationPlayground Vitest directory plus moderation service contract passed, Playwright/CDP moderation route check passed, git diff --check passed. TypeScript full check still fails only on known unrelated baseline errors in EmbeddingsModelSelectionConfig, persona-visuals, and vnPlay. Bandit skipped because this slice touched frontend TypeScript and docs only.
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
