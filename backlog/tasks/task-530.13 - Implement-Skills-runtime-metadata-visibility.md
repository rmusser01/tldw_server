---
id: TASK-530.13
title: Implement Skills runtime metadata visibility
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-30 02:13'
labels:
  - skills
  - webui
  - safe-operations
  - backend
  - frontend
dependencies: []
parent_task_id: TASK-530
priority: high
ordinal: 530.13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue TASK-530 Safe Operations after TASK-530.12 by exposing read-only Skills runtime declaration metadata so users can understand whether a skill may use fork execution, model calls, model overrides, or declared tools before running it. Keep scope limited to structured metadata and UI visibility; do not add a policy editor, RBAC enforcement changes, tool permission mutation, or execution behavior changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Skills list and detail responses expose structured runtime declaration metadata derived from existing frontmatter/registry fields without adding persisted columns.
- [x] #2 Runtime metadata names avoid permission guarantees and distinguish declared tools from actually available executable tools.
- [x] #3 The Skills manager can show a compact runtime summary in the table without breaking older responses that lack the new metadata.
- [x] #4 The Skill test-run modal shows runtime impact before Render prompt only or Run test actions.
- [x] #5 Focused backend and frontend tests cover runtime metadata derivation, list response shape, table visibility, and test-run disclosure.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Spec: Docs/superpowers/specs/2026-06-29-skills-runtime-metadata-visibility-design.md
Plan: Docs/superpowers/plans/2026-06-29-skills-runtime-metadata-visibility.md

Touched implementation files:
- tldw_Server_API/app/core/Skills/runtime_metadata.py
- tldw_Server_API/app/api/v1/schemas/skills_schemas.py
- tldw_Server_API/app/api/v1/endpoints/skills.py
- tldw_Server_API/app/core/Skills/skills_service.py
- tldw_Server_API/tests/Skills/integration/test_skills_api.py
- apps/packages/ui/src/types/skill.ts
- apps/packages/ui/src/components/Option/Skills/Manager.tsx
- apps/packages/ui/src/components/Option/Skills/SkillPreview.tsx
- apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx
- apps/packages/ui/src/components/Option/Skills/__tests__/SkillPreview.test.tsx

Verification:
- source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Skills/integration/test_skills_api.py -q: PASS, 74 passed.
- bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx src/components/Option/Skills/__tests__/SkillPreview.test.tsx --reporter=dot: PASS, 46 passed. Local worktree needed the tracked antd symlink temporarily pointed at the installed Bun cache, then restored.
- git diff --check: PASS.
- source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Skills/runtime_metadata.py tldw_Server_API/app/api/v1/schemas/skills_schemas.py tldw_Server_API/app/api/v1/endpoints/skills.py tldw_Server_API/app/core/Skills/skills_service.py -f json -o /tmp/bandit_skills_runtime_metadata_TASK_530_13.json: PASS, zero findings.
- NODE_OPTIONS=--max-old-space-size=8192 bunx tsc -p tsconfig.json --noEmit --pretty false: FAIL on existing baseline errors outside this task and one pre-existing row-selection prop typing issue in Skills Manager; no runtime metadata type errors observed before the baseline failure list.

Scope notes:
- No policy editor, RBAC change, database migration, or skill execution behavior change.
- Runtime metadata is derived from existing frontmatter/registry fields.
- Test-generated untracked watchlist template artifacts were removed from the worktree.

PR: https://github.com/rmusser01/tldw_server/pull/2549

Review follow-up after PR #2549 rebase:
- Rebased codex/skills-runtime-metadata onto latest origin/dev.
- Addressed Qodo wording feedback by aligning allowed_tools schema descriptions around declared tool strings.
- Addressed Gemini inline comments by defaulting explicit null runtime fields to inline/false, normalizing single-string allowed_tools before counting declarations, and defaulting legacy frontend context to inline.
- Added regression coverage for legacy null API fields, single-string tool declarations, context payload null context, and legacy UI list responses opening the test-run preview.
- Addressed CodeRabbit follow-up by preserving explicit runtime.execution_mode values, separating fork/inline disclosure from model-call allowance in SkillPreview, using repo-relative verification commands, and removing a duplicate final-summary marker.
- Verification: pytest tldw_Server_API/tests/Skills/integration/test_skills_api.py -q passed (74 tests); Vitest Manager/SkillPreview passed (46 tests); git diff --check passed; Bandit touched backend scope passed with zero findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented read-only Skills runtime metadata visibility for TASK-530.13. Backend list/detail/context responses now expose structured runtime declarations plus raw declared tools/model values. The Skills manager has an optional Runtime column with legacy-response fallback, and the test-run modal shows selected-skill runtime impact before dry render or test execution. Import review copy now says Declared tools. Focused frontend tests, full Skills API integration tests, diff check, and Bandit pass; full UI typecheck remains blocked by existing baseline errors.
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
