---
id: TASK-13111
title: Expose automatic conversation-title prompt in Service Prompts
status: In Progress
created_date: 2026-08-23 21:58
labels:
- service-prompts
- webui
- settings
priority: Medium
references:
- Docs/Design/service-prompt-inventory.md
- Docs/superpowers/specs/2026-07-12-user-customizable-service-prompts-design.md
documentation:
- Docs/superpowers/plans/2026-08-23-conversation-title-service-prompt.md
modified_files:
- tldw_Server_API/app/core/Prompt_Management/service_prompts.py
- tldw_Server_API/tests/Prompt_Management/test_service_prompts.py
- tldw_Server_API/tests/Prompt_Management/test_service_prompts_api.py
- apps/packages/ui/src/utils/__fixtures__/service-prompt-rendering.json
- apps/packages/ui/src/services/title.ts
- apps/packages/ui/src/services/__tests__/title.service-prompt-scope.test.ts
- apps/packages/ui/src/services/service-prompts.ts
- apps/packages/ui/src/services/__tests__/service-prompts.test.ts
- apps/packages/ui/src/services/tldw-server.ts
- apps/packages/ui/src/services/tldw/domains/service-prompts.ts
- apps/packages/ui/src/services/tldw/domains/__tests__/service-prompts.test.ts
- apps/packages/ui/src/components/Option/Settings/ServicePromptsSettings.tsx
- apps/packages/ui/src/components/Option/Settings/__tests__/ServicePromptsSettings.test.tsx
- apps/packages/ui/src/assets/locale/en/settings.json
- apps/packages/ui/src/public/_locales/en/settings.json
- Docs/superpowers/plans/2026-08-23-conversation-title-service-prompt.md
updated_date: 2026-08-23 23:18
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the existing automatic conversation-title generation prompt as a bounded Service Prompts definition so users can customize it without changing the current title feature flag, fallback behavior, model invariants, or request-scope safety.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The service-prompt registry exposes chat.title.generation with one user_template part requiring exactly {query}, and save/reset APIs work through the existing generic storage path.
- [ ] #2 Workflow Prompts always shows a localized Conversation title definition and links users to Chat settings for enabling automatic title generation.
- [ ] #3 Every automatic title-generation path renders one immutable scope-bound snapshot while preserving toolChoice none, saveToDb false, the disabled-by-default flag, and existing fallback cleanup.
- [ ] #4 Ordinary prompt load, render, and model failures return the caller fallback; abort and request-scope changes fail closed without stale title writes.
- [ ] #5 Older servers without the service-prompt catalog use the packaged title default with byte-equivalent provider content.
- [ ] #6 Backend, frontend service, Settings, locale-sync, caller-path, compile, lint, diff, and Bandit checks pass for the touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute the four-stage tracked plan at Docs/superpowers/plans/2026-08-23-conversation-title-service-prompt.md: (1) register the backend definition and golden API contract, (2) bind central title generation to a scope-checked immutable snapshot, (3) expose localized Settings guidance without duplicating enablement, and (4) run verification, security review, code review, and finalization.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Corrected the design reference after verifying the latest-dev path; focused baseline is green (frontend 182/182, backend 76/76).
Created and self-reviewed the four-stage implementation plan. The worktree is based on origin/dev 7a536e5d; focused baseline: frontend 182/182 and backend 76/76.
Task 4 pre-final-review verification (2026-08-23):
- PASS: from apps/packages/ui, `bunx vitest run src/services/__tests__/title.service-prompt-scope.test.ts src/services/__tests__/service-prompts.test.ts src/services/tldw/domains/__tests__/service-prompts.test.ts src/components/Option/Settings/__tests__/ServicePromptsSettings.test.tsx src/hooks/chat-helper/__tests__/saveMessageOnSuccess.scope.test.ts src/hooks/chat-helper/__tests__/saveMessageOnError.test.ts src/hooks/chat/__tests__/useChatActions.service-prompts.test.tsx` => 7 files / 244 tests passed.
- PASS: from repository root, `../../.venv/bin/python -m pytest -q tldw_Server_API/tests/Prompt_Management/test_service_prompts.py tldw_Server_API/tests/Prompt_Management/test_service_prompts_api.py` => 78 passed, 9 warnings in 2.01s.
- PASS: `../../.venv/bin/python -m bandit -r tldw_Server_API/app/core/Prompt_Management/service_prompts.py -f json -o /tmp/bandit_task_13111.json` => 0 findings; totals: 363 LOC, 0 skipped tests, 0 nosec.
- PASS: from apps/extension, `bun run compile` => exit 0; `bun run locales:sync --dry-run settings.json` => no pending write, preserving 173 public-only keys.
- PASS: `git diff --check` => exit 0 (only host Darwin temp-directory warnings).
- BASELINE/ENVIRONMENT BLOCKERS (no branch files implicated): from apps/tldw-frontend, `bun run typecheck` reports TypeScript errors exclusively in untouched scripts/__tests__/skills-certification-{evidence,lifecycle,profile,runner}.test.ts. The branch diff contains no apps/tldw-frontend paths; errors are isolated to that unrelated test suite. From apps/packages/ui, required `bunx eslint ...` resolves ESLint 10.9.0, then fails because there is no eslint.config.* or .eslintrc in that package or any parent directory; package.json has no eslint dependency/config and this branch changes neither.
- Supplemental attempt from apps/tldw-frontend with its config reported all nine UI inputs ignored as outside the config base path, so it is not treated as a lint pass.
- Mutation-focused self-review passed by direct source/test inspection: registry removal is caught by backend/client catalog tests; .replace/reparse is caught by one-pass inserted-brace fixture tests; expected scope mismatch is checked before catalog I/O; 412/AbortError rethrow and no stale writes are covered in title/persistence callers; release is asserted after all exits; generic logging has sentinel coverage; Settings is catalog-driven and always visible; provider invariants toolChoice none/saveToDb false/model are asserted.
- Central rationale: one immutable scope-bound snapshot protects every automatic-title caller from account/server changes while ordinary prompt/model failures retain completed-chat fallback behavior.
- Broad final review and final metadata cleanup remain controller-owned; status stays In Progress and the plan documentation link remains until that review package is generated. See .superpowers/sdd/2026-08-23-conversation-title-service-prompt/task-4-report.md for full command output.
Final-review fix evidence (2026-08-23):
- Newly test-asserted (not merely source-inspected): generateTitle selects the caller model with toolChoice "none", saveToDb false, and the distinct snapshot request scope; invokes exactly one HumanMessage with snapshot.scopeSignal; and returns a real removeReasoning-cleaned title. The packaged-template case now uses fixture.defaults["chat.title.generation"].user_template and hand-derived provider bytes.
- Newly test-asserted scope paths: a mismatched user ID is evaluated after a fully matching multi-user target resolves and rejects canonical 412 before catalog/detail traffic; a fully matching single-user server, null account, and expected API-key fingerprint reaches both catalog and detail reads.
- PASS: from apps/packages/ui, bunx vitest run src/services/__tests__/title.service-prompt-scope.test.ts src/services/__tests__/service-prompts.test.ts => Test Files 2 passed (2), Tests 92 passed (92). Expected stderr: five generic "Error generating title" fallback logs and Node localStorage experimental warnings.
- PASS: from repository root, ./apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs [the nine Task 4 UI paths] => 0 errors, 11 warnings. Warnings are no-explicit-any at title.service-prompt-scope.test.ts:7:57 and tldw-server.ts:112:34, 179:31, 180:33, 190:42, 191:17, 211:19, 228:26, 391:29, 665:29, 666:32; ESLint also printed the existing pages-directory configuration notice. No unrelated warnings changed.
- PASS post-fix commit: git diff --check 7a536e5d7aa0666cdd7af94b68a4256d315296f7..HEAD => exit 0 with no whitespace output.
- Production source remained unchanged in this review-fix wave; these tests lock invariants that the prior Task 4 report had described from source inspection. Status stays In Progress for controller final cleanup.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
