---
id: TASK-12974
title: Implement user-customizable Service Prompts v1 vertical slice
status: In Progress
labels:
- service-prompts
- implementation
priority: high
references:
- TASK-12973
documentation:
- Docs/superpowers/specs/2026-07-12-user-customizable-service-prompts-design.md
- Docs/superpowers/plans/2026-07-15-user-customizable-service-prompts-v1.md
modified_files:
- apps/packages/ui/src/utils/__fixtures__/service-prompt-rendering.json
- tldw_Server_API/app/core/Prompt_Management/service_prompts.py
- tldw_Server_API/tests/Prompt_Management/test_service_prompts.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved four-definition Service Prompts v1 plan across the backend, shared WebUI/browser-extension Settings surface, legacy migration, and named runtime consumers using TDD and staged review gates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A static four-definition registry, strict renderer, one-table owner-scoped override store, and four Service Prompt API operations implement the approved contract.
- [ ] #2 Workflow prompts Settings is shared by WebUI and extension, supports scoped migration/save/reset/conflict/corrupt states, and keeps Prompt Library navigation and backup disclosure accurate.
- [ ] #3 All named RAG, rewrite, web-search, Compare, legacy Sidepanel, and synchronous Translation consumers use immutable server-backed request snapshots with preserved no-override semantics.
- [ ] #4 Focused backend/frontend/E2E verification, OpenAPI checks, lint/type/build checks, Bandit, and git hygiene pass; CI shards remain deliberately unchanged.
- [ ] #5 Each implementation task passes independent specification and code-quality review, followed by a final whole-feature review.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Tasks 1-7 from Docs/superpowers/plans/2026-07-15-user-customizable-service-prompts-v1.md sequentially. Use red-green-refactor for each behavior, commit each dependency stage, and complete spec-compliance then code-quality review before advancing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Execution started in isolated worktree /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/service-prompts-v1-plan on branch codex/service-prompts-v1-plan. .worktrees is gitignored; linked the existing project .venv and installed the existing Bun workspace dependencies without tracked lockfile changes. Focused pre-change baseline passed: 196 backend tests across Prompts DB, Translation error mapping, and router-group contracts; 10 frontend tests across Settings routing/index and chat-pipeline error recovery. CI shard edits remain explicitly out of scope per requester.
Task 1 registry/validator/renderer/resolver TDD evidence (2026-07-16): RED command `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Prompt_Management/test_service_prompts.py -v` failed during collection with the expected missing `Prompt_Management.service_prompts` ModuleNotFoundError. The same command is GREEN after implementation: 47 passed, 2 existing warnings, 0 failures. Independent review identified ignored-fixture staging and two JSON decoder corruption edges; the fixture is force-staged, and regression tests first reproduced plain ValueError (over-limit integer) and RecursionError (excessive nesting) before both were normalized to revision-only ServicePromptCorruptOverride errors. Canonical default byte lengths/SHA-256 values match the five source strings. Ruff check/format checks pass, and Bandit reports zero findings for service_prompts.py. Task 1 deliberately calls the narrow get_service_prompt_override interface that Task 2 will add; no direct SQL or Prompts_DB.py changes were made.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
