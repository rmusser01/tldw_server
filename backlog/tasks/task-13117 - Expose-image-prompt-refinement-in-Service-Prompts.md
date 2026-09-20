---
id: TASK-13117
title: Expose image prompt refinement in Service Prompts
status: Done
assignee: []
created_date: '2026-08-24 16:51'
updated_date: '2026-08-24 20:01'
labels:
  - service-prompts
  - image-generation
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
Add the existing Playground image-prompt refinement instructions as one bounded Service Prompts definition while keeping prompt carriers, output contract, provider settings, normalization, and user review behavior code-owned.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The registry exposes image.prompt.refinement with editable literal system_semantics and rewrite_semantics parts whose packaged defaults preserve the current provider messages.
- [x] #2 Each Refine action consumes one immutable account/server-scoped snapshot and binds the chat request to its request scope and invalidation signal.
- [x] #3 Prompt mode, backend, original prompt, context cues, output-only contract, truncation, provider/model settings, response normalization, and user review remain locked and unchanged.
- [x] #4 Scope changes and cancellation fail closed without applying a candidate; ordinary provider and empty-result errors preserve current safe notification behavior.
- [x] #5 The catalog-driven WebUI and extension Settings page exposes localized metadata through the existing generic save and reset flow, with older-server packaged fallback.
- [x] #6 Focused registry, rendering, consumer, Settings, locale, compile, lint, diff, and security verification pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement test-first in three bounded stages: register defaults and metadata; consume one immutable snapshot in the existing Refine action; run focused regression, scope, compatibility, security, and extension verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented test-first: registered image.prompt.refinement with two literal semantic parts; kept prompt carriers and output contract code-owned; bound each Refine request to one immutable Service Prompt scope; added current-server, catalog-404, and catalog-200-with-missing-definition compatibility; exposed generic localized Settings metadata. Verification: UI focused matrix 200/200; backend registry/API 79/79; extension compile passed; Ruff passed; ESLint 0 errors with 18 pre-existing warnings in large shared files; Bandit 0 findings and 0 errors; locale/fixture JSON valid; git diff --check clean. Independent review found the rolling-upgrade catalog gap, which was reproduced RED, fixed narrowly, and re-reviewed with no remaining findings. Known unrelated baseline: apps/tldw-frontend bun run typecheck still fails only in unchanged settings-nav-config.ts and skills-certification test files; no changed file appears in the errors. No standalone user documentation change was needed because the existing catalog-driven Settings UI exposes the new definition.

Pull request: https://github.com/rmusser01/tldw_server/pull/2815

Qodo review on PR #2815 identified catalog/detail rolling-upgrade skew: an advertised image.prompt.refinement detail can 404 from an older instance. Reopened to add a narrowly scoped compatibility fallback and regression while preserving abort, scope-change, 500, and unrelated-definition failures.

Qodo rolling-upgrade finding reproduced RED and fixed narrowly: only an advertised image.prompt.refinement detail 404 uses packaged semantics; 412 scope changes, aborts, 500s, and unrelated definition failures still propagate. Verification after the fix: focused service-prompts 80/80; affected shared-UI matrix 203/203; browser extension compile passed; focused ESLint passed; git diff --check clean. Independent re-review found no actionable issues and marked the fix ready.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Exposed Playground image-prompt refinement through the existing Service Prompts registry and Settings UI. Users can customize only refinement and rewrite semantics; scope-bound dispatch, locked request carriers, safe fallback behavior, and older-server compatibility are preserved without adding storage, endpoints, or a new settings system.
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
