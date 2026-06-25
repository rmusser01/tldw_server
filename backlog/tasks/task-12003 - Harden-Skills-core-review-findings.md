---
id: TASK-12003
title: Harden Skills core review findings
status: Done
created_date: 2026-06-24 01:46
labels:
- skills
- backend
- security
priority: high
references:
- Review findings from current thread
modified_files:
- IMPLEMENTATION_PLAN_skills_core_review_hardening_12003.md
- backlog/tasks/task-12003 - Harden-Skills-core-review-findings.md
- tldw_Server_API/app/api/v1/endpoints/chat.py
- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
- tldw_Server_API/app/core/Skills/__init__.py
- tldw_Server_API/app/core/Skills/context_integration.py
- tldw_Server_API/app/core/Skills/skill_executor.py
- tldw_Server_API/app/core/Skills/skills_service.py
- tldw_Server_API/tests/Skills/integration/test_skill_mcp_integration.py
- tldw_Server_API/tests/Skills/unit/test_skill_executor.py
- tldw_Server_API/tests/Skills/unit/test_skill_registry_queries.py
- tldw_Server_API/tests/Skills/unit/test_skills_service.py
updated_date: 2026-06-25 03:10
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the current-module Skills core review findings: make skill file/registry mutations safer under conflicts, deny tools by default for fork skills without allowed-tools, bound zip import reads before decompression, version and fail supporting-file updates correctly, and provide async context integration for async chat paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Skills updates and deletes do not leave SKILL.md/supporting files mutated when optimistic registry updates conflict or fail.
- [x] #2 Fork-mode skills with no allowed-tools do not receive advertised tools and cannot execute arbitrary tools by default.
- [x] #3 Zip import rejects oversized SKILL.md/supporting files and suspicious archive payloads before unbounded reads.
- [x] #4 Supporting-file-only updates surface I/O failures and bump skill versions.
- [x] #5 Async chat/context call paths avoid synchronous Skills registry/filesystem scans.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented review hardening for Skills core: file updates now snapshot and restore touched files on registry conflicts/failures; deletes mark the registry before removing files; fork-mode skills deny tools by default and ignore model tool calls when no tools were advertised; zip import validates SKILL.md/supporting-file sizes and aggregate limits from ZipInfo before reads; supporting-file-only updates raise storage errors and bump registry versions; async Skills context helpers are wired into chat request handling. Verification: targeted regressions passed (10 passed); broader Skills suite passed (155 passed); compileall passed for Skills core and chat endpoint; Bandit passed on touched production paths with zero findings, report at /tmp/bandit_skills_core_review_12003.json.
Final verification after adding the zip entry-count guard: targeted zip tests passed (2 passed), broader Skills selection passed (156 passed), compileall passed, and Bandit passed again with zero findings at /tmp/bandit_skills_core_review_12003.json.
PR review follow-up started: rechecked branch against latest origin/dev, collected Gemini/Qodo review comments, and will address rollback robustness, async filesystem offloading, deleted-skill restore behavior, helper docstrings, and tool-denial observability.
PR review follow-up completed: verified branch is based on latest origin/dev; addressed Gemini/Qodo comments by adding restore_skill_registry for deleted row recovery, wiring sync restore/delete rollback through it, offloading new async-path filesystem work through asyncio.to_thread, adding helper docstrings, using missing_ok unlink, catching unexpected registry update failures with rollback, and returning a non-empty denied-tools fork output. Verification: targeted review-fix tests passed (4 passed); broader Skills/registry/API selection passed (162 passed); compileall passed; git diff --check passed; Bandit passed on expanded touched production scope with zero JSON results at /tmp/bandit_skills_core_review_12003_rebased.json.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed all reviewed Skills core findings plus PR review follow-up comments. Verified against latest origin/dev with targeted review regressions, the broader Skills/registry/API suite (162 passed), compileall, diff check, and Bandit with zero findings.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused Skills unit/integration tests cover the fixed behaviors and pass.
- [x] #8 Bandit scans touched backend Python paths with no new findings.
- [x] #9 Implementation plan is created and updated through completion.
<!-- DOD:END -->
