---
id: TASK-77
title: Phase 2.2 minimal persona notes router conditional cleanup AI
status: Done
assignee: []
created_date: '2026-05-05 16:35'
updated_date: '2026-05-05 17:31'
labels:
  - phase2.2
  - router-cleanup
  - issue-1116
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1309'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue issue #1116 Phase 2.2 after PR #1309. Convert the next small minimal-test optional persona/notes registrations from eager try/import RouterSpec blocks to ImportedRouterSpec-backed lazy router specs. Scope is limited to persona, archetype_endpoints, and notes in tldw_Server_API/app/api/v1/router_groups/minimal.py. Preserve prefixes, tags, route_key behavior, default_stable behavior, and the existing minimal-test broad skip behavior for import-time failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persona, persona archetype, and notes minimal optional specs defer module import and router attribute lookup until registration
- [x] #2 Existing route metadata is preserved for persona, persona archetype, and notes
- [x] #3 Focused router-group tests cover lazy behavior and broad import failure skipping with red-green verification
- [x] #4 Router-group contract tests and touched-source Bandit pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Keep the persona/notes lazy-spec implementation unchanged unless verification shows the review comments require source changes.
2. Replace the branch-local task record's review-flagged wording with red-green.
3. Harden test_iter_minimal_optional_router_specs_defers_persona_notes_attr_lookup so selected persona/archetype/notes imports through builtins.__import__ are recorded and asserted absent during spec construction.
4. Run the focused router-group tests, Bandit on touched Python scope, git diff --check, update the task record, commit, push, and re-check unresolved PR review threads.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started after PR #1309 merge was verified at merge commit dda18105ac1f739d05c373e68f06b2c269e14dc4. Worktree: /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/phase2-2-minimal-persona-notes-router-conditionals-ai. Branch: codex/phase2-2-minimal-persona-notes-router-conditionals-ai. Baseline full test_router_groups_contract.py passed with 79 tests before edits.

Implemented the persona/notes minimal router tranche. Added red-green focused contract coverage proving persona, archetype_endpoints, and notes defer module import/router attribute lookup until registration and preserve broad skip_exceptions=(Exception,) behavior for RuntimeError import failures. Converted only those three eager try/import RouterSpec blocks to ImportedRouterSpec-backed lazy specs while preserving prefixes, tags, route_key, and default_stable behavior. Verification: focused selection persona_notes_attr_lookup or persona_notes_runtime_import_failures failed red before the source change and passed after implementation; full test_router_groups_contract.py passed with 81 tests; test_main_router_contract.py passed with 6 tests; Bandit on minimal.py reported 0 results and 0 errors; git diff --check was clean.

Review follow-up for PR #1313 started. Verified two unresolved CodeRabbit threads: task wording red-green hyphenation and persona/notes lazy test import tracking through builtins.__import__.

PR #1313 review follow-up implemented. Replaced branch-local task wording with red-green and hardened the persona/notes lazy-spec test to record selected endpoint imports through builtins.__import__ before asserting none occur during spec construction. Verification: focused router contract selection passed with 2 passed and 79 deselected; full test_router_groups_contract.py passed with 81 passed; git diff --check passed; Bandit on test_router_groups_contract.py with B101 skipped reported 0 results; Bandit on router_groups/minimal.py reported 0 results.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted minimal optional persona, persona archetype, and notes router registrations to ImportedRouterSpec-backed lazy specs. The change keeps the prior minimal-test broad import-failure skip behavior by explicitly setting skip_exceptions=(Exception,) while deferring endpoint imports until registration. Added focused regression coverage for lazy router resolution and RuntimeError import-failure skipping. Verification covered red-green focused tests, full router_groups contract tests, main router contract tests, Bandit, and diff check.

PR #1313 review follow-up addressed both unresolved CodeRabbit threads: task wording now uses red-green consistently, and persona/notes lazy-spec coverage now tracks selected builtins.__import__ imports during spec construction.
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
