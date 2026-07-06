---
id: TASK-12570
title: Implement Skills accessibility copy i18n and state polish
status: Done
labels:
- skills
- webui
- accessibility
- i18n
- ux
priority: high
ordinal: 530.14
parent_task_id: TASK-530
documentation:
- Docs/superpowers/plans/2026-06-30-skills-accessibility-copy-i18n-state-polish.md
modified_files:
- Docs/superpowers/plans/2026-06-30-skills-accessibility-copy-i18n-state-polish.md
- apps/packages/ui/src/assets/locale/en/option.json
- apps/packages/ui/src/components/Option/Skills/Manager.tsx
- apps/packages/ui/src/components/Option/Skills/SkillDrawer.tsx
- apps/packages/ui/src/components/Option/Skills/SkillPreview.tsx
- apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx
- apps/packages/ui/src/components/Option/Skills/__tests__/SkillPreview.test.tsx
- apps/packages/ui/src/components/Option/Skills/__tests__/skills-locale-keys.test.ts
- apps/packages/ui/src/public/_locales/en/option.json
- apps/tldw-frontend/e2e/workflows/tier-5-specialized/skills.spec.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue TASK-530 after TASK-530.13 by hardening the Skills manager and test-run/create flows for accessibility, user-facing state clarity, stable locale copy, responsive behavior, and keyboard workflows. Keep scope frontend, docs, and tests unless investigation proves a backend contract issue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Row, toolbar, and modal actions expose stable accessible names and preserve predictable keyboard focus order.
- [ ] #2 Loading, non-blocking status, blocking errors, confirmations, and success states are announced with appropriate status or alert semantics.
- [ ] #3 Persistent Skills copy uses stable English locale keys for the touched WebUI and extension-visible surfaces.
- [ ] #4 Responsive and extension-width layouts keep toolbar, table actions, and test-run/create flows reachable without overlap.
- [ ] #5 A keyboard-only user can search, open test run, enter arguments, run or close it, open the create drawer, and cancel with predictable focus return.
- [ ] #6 Focused Vitest and Playwright coverage verifies the changed behaviors.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-30-skills-accessibility-copy-i18n-state-polish.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented scoped Skills accessibility/state polish and addressed PR review feedback. Added permanently mounted screen-reader live status regions for Skills list loading and SkillPreview render/test-run states. Added scoped focus return for test-run modal and create/edit drawer flows using Skills-local data attributes, after-close hooks, an idempotent drawer closed-state fallback, and the empty-state create selector fallback, avoiding the previous document-wide label search and fixed retry behavior. Added WebUI English locale keys in assets/locale alongside extension _locales copy, with test coverage for both locale sources. Updated mocked tier-5 Skills journey to current Test run semantics, case-insensitive search matching, and explicit blank-argument preservation, plus keyboard/mobile-width coverage for search, row test run, argument entry, execution result, Escape close focus return, New Skill open, and cancel focus return.

Verification after review fixes and CodeRabbit follow-up:
- bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx src/components/Option/Skills/__tests__/SkillPreview.test.tsx src/components/Option/Skills/__tests__/skills-locale-keys.test.ts --reporter=dot: PASS, 54 tests.
- TLDW_WEB_URL=http://localhost:18087 TLDW_WEB_CMD='bun run dev -- -p 18087' npx playwright test e2e/workflows/tier-5-specialized/skills.spec.ts --project=tier-5 --reporter=line: PASS, 5 tests before the CodeRabbit follow-up; latest rerun attempt was blocked by environment escalation limits before the local server could start.
- node -e JSON parse check for apps/packages/ui/src/public/_locales/en/option.json and apps/packages/ui/src/assets/locale/en/option.json: PASS.
- git diff --check origin/dev...HEAD: PASS.
- Bandit: skipped because no Python files were touched.

Environment note: the worktree's checked-in antd symlink target is absent locally, so focused frontend verification temporarily repointed apps/packages/ui/node_modules/antd to the installed Bun cache target, then restored the tracked symlink target before final status.
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
