---
id: TASK-12027
title: Implement Web_Scraping refactor Phase 0 import inventory and guardrails
status: Done
created_date: 2026-07-04 00:14
labels:
- web-scraping
- refactor
- phase-0
priority: High
references:
- /Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-phase-0-inventory/Docs/superpowers/plans/2026-07-03-web-scraping-phase-0-import-inventory.md
- /Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-phase-0-inventory/Docs/superpowers/specs/2026-07-03-web-scraping-refactor-design.md
modified_files:
- Helper_Scripts/web_scraping_refactor_inventory.py
- tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py
- Docs/Design/web_scraping_refactor_import_inventory.json
- Docs/Design/WebScraping_Refactor_Import_Inventory.md
- Docs/Design/WebScraping.md
- Docs/superpowers/specs/2026-07-03-web-scraping-refactor-design.md
- backlog/tasks/task-12027 - Implement-Web-Scraping-refactor-Phase-0-import-inventory-and-guardrails.md
updated_date: 2026-07-04 01:11
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved Phase 0 implementation plan for the Web_Scraping modular refactor: add import inventory tooling, generated compatibility artifacts, dependency guardrails, documentation links, and focused verification without moving runtime behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 AST import inventory helper is implemented and tested.
- [x] #2 JSON and Markdown import inventory artifacts are generated and freshness-tested.
- [x] #3 Guardrail tests prevent future internal Web_Scraping packages from importing legacy wrapper files.
- [x] #4 Design docs link to the generated inventory artifacts.
- [x] #5 Focused pytest, helper compile check, Bandit, and scoped whitespace verification are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute the tasks in Docs/superpowers/plans/2026-07-03-web-scraping-phase-0-import-inventory.md using subagent-driven development with review after each task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-03: Created in isolated worktree /Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-phase-0-inventory after baseline focused test passed: preflight policy context suite returned 18 passed.
2026-07-03: Task 1 complete. Implemented `Helper_Scripts/web_scraping_refactor_inventory.py` and initial scanner tests. Initial implementer commit `9b3410587475` passed spec review but code-quality review found mixed import sorting, missed relative imports, output path containment, and lint issues. Fix commit `802aae316a` resolved those findings. Verification after fixes: `python -m pytest tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py -q` returned 7 passed; scoped py_compile passed; ruff check passed; Bandit on helper returned 0 findings; default inventory scan over app/tests completed with 273 records. Spec and code-quality re-reviews approved Task 1.
2026-07-03: Task 2 complete. Added freshness test and generated `Docs/Design/web_scraping_refactor_import_inventory.json` plus `Docs/Design/WebScraping_Refactor_Import_Inventory.md` in commit `a25ed4e6cd`. Inventory contains 273 records across 9 targets and includes WebScraping/WebSearch tests plus app callers in Collections, RAG, MCP, Workflows, Watchlists, Research, and services. Verification: initial freshness test failed before artifacts, generation exited 0, JSON validation passed, full inventory tests returned 8 passed. Spec and code-quality reviews approved with no findings.
2026-07-03: Task 3 complete. Added legacy-wrapper dependency guardrail tests in commit `b67cabb4a7`. Verification: focused guardrail tests returned 2 passed; full inventory test file returned 10 passed; ruff passed. Raw Bandit on the test file reports existing low-severity B101 pytest assert findings, while new asserts use targeted nosec suppressions and `bandit -s B101` is clean. Spec and code-quality reviews approved Task 3 with no blocking findings.
2026-07-03: Task 4 complete. Linked generated import inventory artifacts from `Docs/Design/WebScraping.md` and the refactor design spec in commit `83378d5926`. Verification: documentation grep found both artifact references, scoped whitespace check passed, and both spec and code-quality reviews approved with no findings.
2026-07-03: Task 5 final verification complete. Focused tests: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py -q` returned 10 passed, 32 warnings. Compile check: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m py_compile Helper_Scripts/web_scraping_refactor_inventory.py` exited 0. Bandit: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit Helper_Scripts/web_scraping_refactor_inventory.py -f json -o /tmp/bandit_web_scraping_phase0_inventory.json` exited 0 with 0 errors and 0 results. Scoped whitespace check over touched files exited 0. Worktree had only the Backlog task file untracked after final verification.
2026-07-03: Final whole-branch review found a Phase 0 freshness gap: the guard compared only the inventory JSON `records` field and did not protect JSON metadata or the Markdown artifact. Fixed in commit `ee8e53af83` by regenerating both artifacts in the freshness test and comparing the full JSON payload plus tracked Markdown content. Fresh verification after the fix: focused pytest returned 10 passed, 32 warnings; helper py_compile exited 0; Bandit on `Helper_Scripts/web_scraping_refactor_inventory.py` exited 0 with 0 errors and 0 results; scoped whitespace check over touched files exited 0.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Phase 0 import inventory and guardrails for the Web_Scraping modular refactor. Added an AST inventory helper, generated JSON/Markdown import artifacts, freshness tests, dependency guardrail tests for future internal package dependencies, docs links, and focused verification without moving runtime behavior.
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
