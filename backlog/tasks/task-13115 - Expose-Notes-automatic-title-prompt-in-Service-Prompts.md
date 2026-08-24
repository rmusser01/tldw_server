---
id: TASK-13115
title: Expose Notes automatic-title prompt in Service Prompts
status: Done
assignee: []
created_date: '2026-08-24 04:47'
updated_date: '2026-08-24 05:45'
labels:
  - service-prompts
  - notes
  - settings
dependencies: []
references:
  - Docs/Design/service-prompt-inventory.md
  - >-
    Docs/superpowers/specs/2026-07-12-user-customizable-service-prompts-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the existing LLM-backed Notes automatic-title prompt as one bounded Service Prompts definition while preserving current title gates, provider configuration, output constraints, normalization, and heuristic fallback behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The registry exposes notes.title.generate with editable literal system and title_instruction parts, and the packaged path preserves the existing provider payload.
- [x] #2 Note create, title suggestion, and bulk create resolve one owner-scoped immutable prompt only when an LLM strategy is active; bulk reuses one revision and heuristic paths perform no prompt read.
- [x] #3 Maximum length, title-only output, content shaping, provider options, normalization, and feature and strategy gates remain code-owned and unchanged.
- [x] #4 Prompt resolution or validation failures fail closed before affected provider dispatch or persistence, while existing provider-unavailable, provider-error, and empty-output cases retain heuristic fallback.
- [x] #5 The catalog-driven WebUI and extension Settings page exposes localized Notes title metadata and uses the existing generic save and reset path without new UI or storage infrastructure.
- [x] #6 Focused backend, frontend Settings, locale, compile, lint, diff, and Bandit verification pass for the touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement test-first in three bounded stages: register the definition and metadata; resolve and consume one immutable prompt in Notes title paths; run focused regression, security, and cross-host Settings verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented test-first. Verification: backend Service Prompt/title matrix 86/86; complete Notes title integration 12/12; legacy Notes title regressions 3/3; shared UI Settings/domain 143/143; extension TypeScript compile; i18n duplicate/coverage/sync dry-run; focused Ruff and ESLint; Bandit zero findings; git diff --check clean. Independent final code review found no remaining actionable issues. Repo-wide tldw-frontend typecheck remains outside this task's gate because origin/dev has unrelated settings navigation and skills-certification baseline errors; the extension compile covers the touched shared UI.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added notes.title.generate with editable literal system and title_instruction parts, wired lazy owner-scoped resolution into create, suggest, and bulk LLM title paths, and exposed localized metadata through the existing Service Prompts Settings UI. The implementation keeps title constraints and heuristic/provider fallback behavior code-owned, avoids Prompts DB access for explicit/heuristic/disabled/replayed work, reuses one successful bulk snapshot, and fails closed before provider dispatch or persistence on prompt failures.
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
