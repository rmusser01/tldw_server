---
id: TASK-12101
title: Rebase PR 2573 and address current review feedback
status: Done
ordinal: 12101
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase the WebUI llama.cpp/auth/character chat PR onto latest dev, verify active review feedback against current code, implement valid fixes, and record focused verification before pushing the PR branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR branch is rebased onto latest origin/dev without touching unrelated dirty files.
- [x] #2 Current actionable PR review comments are evaluated and valid issues are fixed or documented if not valid.
- [x] #3 Focused tests cover the review fixes and relevant character chat/WebUI paths.
- [x] #4 Verification, typecheck/Bandit status, and any remaining known blockers are recorded before pushing.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Confirmed `origin/dev` is an ancestor of the PR head after fetching/rebasing.
- Active unresolved review threads were avatar-related; tightened whitespace handling before `createImageDataUrl()` and added explicit invalid-embedded-image fallback coverage.
- Verified older review-body issues against current code: runtime placeholder key readiness, shared connection single-user key checks, runtime env-auth opt-out, env-only custom OpenAI catalog config, and draft tracked-character priority were already represented in branch code/tests.
- Added `apiBearer?: string` to `TldwConfig` because runtime bootstrap intentionally strips stale bearer auth from stored config.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Focused frontend tests passed: `bun run vitest run src/types/__tests__/assistant-selection.test.ts src/utils/__tests__/image-utils.test.ts src/services/tldw/__tests__/TldwModels.test.ts src/store/__tests__/connection.test.ts src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx --reporter=dot` (80 tests).
- Runtime bootstrap tests passed: `bun run vitest run __tests__/extension/runtime-bootstrap.test.ts --reporter=dot` (24 tests).
- Backend provider readiness tests passed using the root venv: `python -m pytest tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_readiness.py -q` (6 tests).
- `git diff --check` passed.
- Bandit passed with zero findings for `tldw_Server_API/app/api/v1/endpoints/llm_providers.py`; JSON written to `/tmp/bandit_pr2573_rebase.json`.
- Typecheck status: `apps/packages/ui` still fails on unrelated baseline errors after increasing heap; `apps/tldw-frontend` still fails on unrelated baseline errors. The prior PR-specific `runtime-bootstrap.ts` `apiBearer` type errors are resolved.
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
