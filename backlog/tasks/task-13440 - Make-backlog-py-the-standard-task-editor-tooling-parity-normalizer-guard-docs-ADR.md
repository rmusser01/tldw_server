---
id: TASK-13440
title: Make backlog-py the standard task editor (tooling parity, normalizer, guard,
  docs, ADR)
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Owner decision (2026-10-03): tools/backlog-py replaces the Node Backlog.md CLI/MCP as the standard task editor. Today the two editors corrupt each other's files: the Node CLI writes SECTION:NOTES and, when it edits a backlog-py task, wraps the existing SECTION:IMPLEMENTATION_NOTES in a second NOTES section and duplicates FINAL_SUMMARY markers (even on a label-only edit). 331 task files on dev already mix the formats. The Node CLI also picks task ids from local branches only, so ids collide across sessions. This task covers the tooling PR: close the agent-used CLI/MCP gaps in backlog-py (labels on create/edit; acceptance criteria on create; --ac/--remove-ac and title edits; notes replace), add a normalize command that rewrites Node-style/nested/duplicated sections into backlog-py's canonical form, add a CI ratchet that every task file parses with backlog-py and the count of Node-style sections cannot grow, update AGENTS.md and the backlog-py README for the cutover, and record the decision in a new ADR (ADR-002's 'require a task' decision stands; this changes the tool).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 backlog-py CLI and MCP support labels on create/edit, acceptance criteria on create, adding/removing criteria and retitling on edit, with tests
- [x] #2 A backlog-py normalize command rewrites SECTION:NOTES, nested NOTES/IMPLEMENTATION_NOTES and duplicated FINAL_SUMMARY markers into one canonical form, idempotently, with tests on real mixed fixtures
- [x] #3 A CI-enforced test checks every backlog task file parses with backlog-py and the count of Node-style sections does not exceed a recorded baseline
- [x] #4 AGENTS.md and tools/backlog-py/README.md direct agents to backlog-py (command name backlog-py; explicit-id creation guidance), and a new ADR records the cutover
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started 2026-10-03 in worktree chore/backlog-py-standard. ADR check: ADR required: yes; ADR path: Docs/ADR/059-backlog-py-task-editor-cutover.md (next free number; no open PR claims 059); reason: changes the repository workflow tool that ADR-002's task requirement runs on. Plan: (1) CLI/MCP parity: -l/--labels, --ac, --remove-ac, -t/--title, --notes, --dep; (2) task normalize [--check] [paths] + normalize-before-edit so backlog-py edits never mix formats; (3) CI ratchet in tests/CI run by backend-required contracts step; (4) AGENTS.md, README, ADR-059, Docs/Published refresh.
Delivered in PR #3142 (https://github.com/rmusser01/tldw_server/pull/3142, base dev, not merged). Touched: tools/backlog-py/src (cli/main.py, core/repository.py, markdown/task_parser.py, mcp/tools.py), tools/backlog-py/tests (test_normalize.py, test_task_mutations.py, test_mcp_resources.py, fixtures/repos/mixed with 7 real broken task files), tools/backlog-py/README.md, tools/backlog-py/docs/agent-critical-parity.md, tldw_Server_API/tests/CI/test_backlog_task_format_ratchet.py, tldw_Server_API/tests/CI/test_path_classifier.py, Helper_Scripts/ci/path_classifier.py, .github/workflows/backend-required.yml, pyproject.toml (pytest pythonpath), AGENTS.md, Docs/ADR/059-backlog-py-task-editor-cutover.md, Docs/ADR/README.md, Docs/Published/ADR (refresh script). Verification: tools/backlog-py/tests 139 passed (101 before); backend-required contracts step run locally as CI runs it 193 passed; tldw_Server_API/tests/CI + tests/Docs/test_docs_published_refresh.py + tests/lint/test_first_party_source_pinning.py pass (two first-run failures were environmental: python not on PATH for a workflow-shell test, and the new ADR untracked; both pass once fixed). Normalizer checked in memory over all 3,879 backlog task files: no text line lost or added, frontmatter bytes kept, idempotent, one BEGIN/END pair per section; on a copy, task normalize took --check from 2,308 to 0 with identical task list output. Ratchet baseline 2,308, enforced in backend-required 'Enforce CI contracts and code ratchets'; verified it fails at baseline-1. Ruff clean on touched files; uvx bandit -ll on touched backlog-py source: no issues. Known gaps (ADR-059): no --plan/priority/assignee/parent options, no MCP server adapter; ratchet does not run on backlog-only PRs (owner decision whether to add backlog/tasks/** to BACKEND_GLOBS).
Review follow-up (orchestrator): backlog-only PRs are now covered by a backlog-task-format pre-commit hook (normalize --check on changed backlog/tasks files, in an isolated env with click/loguru/PyYAML), which runs locally and in the non-required run-pre-commit CI job. Verified: passes on a canonical file and fails on a Node-format file. ADR-059 updated, Docs/Published refreshed (docs refresh test 33 passed), test_run_local_ci 10 passed.
Qodo review of #3142 addressed (ccbc5ec0d1, 1b5e211bd3), each bug test-first. Bugs: removing a criterion now drops its indented continuation lines and multi-line criteria are written as indented continuations; task normalize resolves relative paths against the --cwd project root; MCP notesSet/notesAppend/finalSummary accept text or lists (one per line) and other text fields reject containers instead of writing a repr; --append-notes is repeatable in order; files the normalizer cannot repair are named (ratchet and edit errors). Ratchet redesign: the global count/baseline (2,308) is gone. test_backlog_task_format_ratchet.py now checks only the task files changed since the merge base with BACKLOG_TASK_FORMAT_BASE (backend-required passes the admitted/PR base; local default origin/dev), each of which must parse and be canonical, so a backlog-only PR can no longer make an unrelated backend PR fail; the backlog-task-format pre-commit hook still flags task-only PRs. AC #3's 'every file parses and count <= baseline' wording is superseded by this per-change check (ADR-059 and its Docs/Published mirror updated). License-first workflow contract base-sha counts bumped by one for the new env line. Rules: docstrings on normalizer helpers, module unit markers and annotations on new tests. Verification: tools/backlog-py/tests 145 passed; contracts step as CI runs it 201 passed; tldw_Server_API/tests/CI plus Docs/test_docs_published_refresh.py 485 passed, 4 skipped; ruff clean on touched files; uvx bandit -ll on touched backlog-py source: no issues. origin/dev had no new commits, so there was nothing to merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
backlog-py is now the repository's task editor (ADR-059). It gained labels, criteria add/remove, retitle, notes replace and dependencies on create across CLI and MCP; a lossless, idempotent 'task normalize [--check]' that turns Node SECTION:NOTES, nested NOTES/IMPLEMENTATION_NOTES, repeated sections and duplicated FINAL_SUMMARY markers into one canonical form; and normalize-before-edit so its own edits never mix formats again. A CI ratchet in backend-required asserts every task file parses and caps the files needing normalization at 2,308 (TASK-13441 takes it to zero); backlog-py's own tests now run there too. AGENTS.md, the backlog-py README and ADR-059 direct agents to backlog-py, forbid Node CLI/MCP mutations, and explain choosing a task id above the max across origin/dev and open PRs. PR #3142, unmerged.
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
