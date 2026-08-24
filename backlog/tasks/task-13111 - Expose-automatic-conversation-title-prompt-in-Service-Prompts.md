---
id: TASK-13111
title: Expose automatic conversation-title prompt in Service Prompts
status: Done
created_date: 2026-08-23 21:58
labels:
- service-prompts
- webui
- settings
priority: Medium
references:
- Docs/Design/service-prompt-inventory.md
- Docs/superpowers/specs/2026-07-12-user-customizable-service-prompts-design.md
documentation: []
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
updated_date: 2026-08-24 04:19
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the existing automatic conversation-title generation prompt as a bounded Service Prompts definition so users can customize it without changing the current title feature flag, fallback behavior, model invariants, or request-scope safety.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The service-prompt registry exposes chat.title.generation with one user_template part requiring exactly {query}, and save/reset APIs work through the existing generic storage path.
- [x] #2 Workflow Prompts always shows a localized Conversation title definition and links users to Chat settings for enabling automatic title generation.
- [x] #3 Every automatic title-generation path renders one immutable scope-bound snapshot while preserving toolChoice none, saveToDb false, the disabled-by-default flag, and existing fallback cleanup.
- [x] #4 Ordinary prompt load, render, and model failures return the caller fallback; abort and request-scope changes fail closed without stale title writes.
- [x] #5 Older servers without the service-prompt catalog use the packaged title default with byte-equivalent provider content.
- [x] #6 Backend, frontend service, Settings, locale-sync, caller-path, compile, lint, diff, and Bandit checks pass for the touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute the four-stage plan: (1) register the backend definition and golden API contract, (2) bind central title generation to a scope-checked immutable snapshot, (3) expose localized Settings guidance without duplicating enablement, and (4) run verification, security review, code review, and finalization.
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
Final cleanup (2026-08-23):
- Broad final review identified three verification gaps; test-only commits 00ef66d49d and 0b23b09b6d addressed them. The scoped fix re-review found all three addressed and no new Critical/Important breakage.
- Focused fix verification: from apps/packages/ui, `bunx vitest run src/services/__tests__/title.service-prompt-scope.test.ts src/services/__tests__/service-prompts.test.ts` => 2 files / 92 tests passed. The locked contract now observes distinct snapshot request scope, exact provider options, one HumanMessage, snapshot signal identity, real removeReasoning cleanup, real packaged fixture bytes, user-mismatch pre-catalog 412, and matching-scope catalog/detail traffic.
- Pinned lint verification: `./apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs [nine Task 4 UI paths]` => 0 errors / 11 pre-existing no-explicit-any warnings plus the existing pages-directory configuration notice.
- Full branch whitespace: `git diff --check 7a536e5d7aa0666cdd7af94b68a4256d315296f7..HEAD` => exit 0.
- Known baseline/tooling caveats retained: `bun run typecheck` in apps/tldw-frontend has unrelated diagnostics only in untouched scripts/__tests__/skills-certification-{evidence,lifecycle,profile,runner}.test.ts; the original packages/ui `bunx eslint` command cannot discover an ESLint config, so the repository-pinned lint command above is the applicable gate. Existing host Darwin confstr() temporary-directory warnings are benign.
- Final rationale: one centrally loaded/rendered immutable request-scope-bound title snapshot protects every automatic-title caller while preserving ordinary prompt/model fallback behavior so completed chats remain usable. Settings shows the prompt without duplicating the Chat enablement control.
PR #2811 follow-up (2026-08-24): reopened for requested rebase onto latest dev and verified review/CI follow-up. Qodo reported two bounded items in generateTitle: add safe failure-stage/type/status context without logging prompt/provider payloads, and honor a pre-aborted caller before the settings read. Changes will follow RED→GREEN and be reverified before the task returns to Done.
PR #2811 review follow-up completed (2026-08-23):
- Rebased cleanly onto the then-current origin/dev 5c268daa7adde606c80908cb48e0dca7bce19553 after a fresh fetch.
- Addressed both Qodo findings with RED→GREEN coverage: pre-aborted callers now fail before reading title settings; ordinary failures log only a fixed coarse stage (settings/snapshot/render/model/invoke/response), never authored prompt, query, provider payload, or raw error text.
- RED evidence: focused title suite failed 7 assertions on the old implementation (settings read occurred for pre-abort; all safe-stage log assertions missing). GREEN evidence: focused title suite 19/19.
- PASS focused UI matrix: 7 files / 247 tests.
- PASS backend prompt-management matrix: 2 files / 78 tests, 9 warnings.
- PASS extension compile; locale sync regenerated the public bundle after the dev rebase and the follow-up dry-run reports no pending write.
- PASS pinned repository ESLint: 0 errors / 11 pre-existing no-explicit-any warnings plus the existing pages-directory notice.
- PASS Bandit on service_prompts.py: 0 findings, 363 LOC, 0 nosec/skips.
- PASS full git diff --check.
- Old-head CI classification: exact origin/dev produces the same OpenAPI hash/count drift as the PR (07868bfa..., 2021 paths / 2958 schemas), so no unrelated fingerprint was imported here. The old mypy NumPy-stub parse failure and Watchlists/Prompt Improvement E2E failures are likewise outside the branch diff.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Exposed chat.title.generation as a Service Prompts definition and Settings entry. A centrally loaded and rendered immutable request-scope-bound title snapshot customizes every automatic-title caller while preserving disabled-by-default behavior, caller model/toolChoice none/saveToDb false, ordinary fallback-completed chats, old-server packaged compatibility, and fail-closed cancellation/account/server/API-key-scope changes. Settings exposes the prompt and directs enablement to Chat settings without duplicating the toggle.
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
