---
id: TASK-530.2
title: Implement Skills guided template authoring
status: Done
labels:
- skills
- webui
- ux
priority: medium
parent_task_id: TASK-530
modified_files:
- apps/packages/ui/src/components/Option/Skills/skill-form-utils.ts
- apps/packages/ui/src/components/Option/Skills/__tests__/skill-form-utils.test.ts
- apps/packages/ui/src/components/Option/Skills/SkillDrawer.tsx
- apps/packages/ui/src/components/Option/Skills/__tests__/SkillDrawer.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next Skills beginner activation slice from TASK-530: add deterministic skill template content utilities and new-skill SkillDrawer template controls so beginners can start from valid SKILL.md drafts without losing manual edits silently.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
['Implemented deterministic Skills starter templates for summarizer, explainer, extractor, and blank drafts. Template generation normalizes user-entered names into valid skill IDs and falls back to template-specific names when empty.', 'SkillDrawer now defaults new skills to the Summarizer template, exposes template choices through an AntD Radio button group, updates generated drafts when the untouched name changes, and prompts before replacing manually edited SKILL.md content.', 'Added SkillDrawer guided-template tests and expanded skill-form-utils tests. Verified RED first: missing buildSkillTemplateContent and missing template selector failed as expected.', 'Verification: focused Skills Vitest suite passed 27 tests; git diff --check passed; Bandit touched-scope report produced 0 findings / 0 LOC because the touched scope is TypeScript.', 'Typecheck: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --project tsconfig.json --noEmit no longer reports Skills errors. It still fails on pre-existing Notes test prop mismatches unrelated to this branch.', 'Browser smoke: local Next dev server served /skills, but the page was blocked by backend readiness because no local API server was available. The route-level recovery state rendered; the drawer path remains covered by component tests.']
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented guided starter templates for new Skills. Added deterministic SKILL.md generation for summarizer, explainer, extractor, and blank templates with normalized skill names. Updated SkillDrawer so new skills start from a beginner-friendly Summarizer draft, users can switch templates, generated drafts follow untouched name changes, and manually edited content is protected by a replace confirmation. Added focused utility and drawer tests. Verification: focused Skills Vitest suite passed 27 tests; git diff --check passed; Bandit touched-scope report had 0 findings / 0 LOC. Package typecheck no longer reports Skills errors but still fails on unrelated existing Notes test prop mismatches. Browser smoke reached /skills but was blocked by backend readiness because no local API server was available.
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
