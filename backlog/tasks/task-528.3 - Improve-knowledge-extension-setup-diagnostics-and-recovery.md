---
id: TASK-528.3
title: Improve /knowledge extension setup diagnostics and recovery
status: Done
labels:
- extension
- knowledge
- ux
- accessibility
priority: high
parent_task_id: TASK-528
documentation:
- Docs/superpowers/plans/2026-06-07-knowledge-extension-setup-diagnostics-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Improve the extension /knowledge setup-required state so users can tell whether server URL, API key, host permission, allowlist, or backend reachability is blocking Knowledge QA. Keep the page scoped to Knowledge QA and do not add flashcard behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Extension /knowledge setup gate shows concrete configuration and reachability checks.
- [ ] #2 The UI explains next actions for missing URL, missing API key, missing host permission, and unreachable backend.
- [ ] #3 The setup/tour interaction does not distract from the blocking recovery path.
- [ ] #4 Automated coverage verifies setup-required, unconfigured, and unreachable-backend states.
- [ ] #5 WebUI and extension differences are documented where setup behavior must differ.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-06-07-knowledge-extension-setup-diagnostics-plan.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TASK-528.3 completed implementation and source coverage. Verification: `bunx vitest run src/components/Option/KnowledgeQA/__tests__/KnowledgeQA.connection.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeQALayout.behavior.test.tsx` passed 20 tests from apps/packages/ui; `bun run compile` passed from apps/extension. Extension runtime Playwright verification was attempted but blocked before browser launch because WXT production build hung twice after duplicated-import warnings; both hung processes were terminated and the blocker is documented in the plan. Bandit not applicable because no Python files were touched.
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
