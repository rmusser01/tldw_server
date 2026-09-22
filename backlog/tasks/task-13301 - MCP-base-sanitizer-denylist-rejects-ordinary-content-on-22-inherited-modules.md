---
id: TASK-13301
title: MCP base sanitizer denylist rejects ordinary content on 22 inherited modules
status: To Do
assignee: []
created_date: '2026-09-22 04:52'
labels:
  - bug
  - mcp
dependencies: []
references:
  - 'tldw_Server_API/app/core/MCP_unified/modules/base.py:779'
  - >-
    tldw_Server_API/app/core/MCP_unified/modules/implementations/web_tool_base.py:51
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
BaseModule.sanitize_input rejects the substrings ["';", "\";", "--", "/*", "*/", "xp_", "sp_"], applied to every tool call via tool_execution/security.py:harden_and_sanitize_tool_arguments.

Executed: NotesModule.sanitize_input({"content": "Heading\n---\nbody"}) raises ValueError and surfaces as JSON-RPC -32602. Also rejected: "run tests in src/*.py", "SELECT 1 -- note", git pathspec "-- src/app.py", and the filename "exp_data.csv". These are pure data handed to parameterised queries, so the denylist buys no injection protection.

22 modules inherit it unchanged. web_tool_base.py:51-70 already fixed it for web tools only and its docstring diagnoses the exact problem. The base sanitizer test uses os.urandom(4).hex() as its safe fixture, which can never contain a denied substring, so nothing asserts legitimate content survives.

Related: TASK-13294 (same function, tab-stripping defect).

Source: synthesis F5
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Denylist removed from BaseModule.sanitize_input; control-char strip retained
- [ ] #2 Base-level test asserts legitimate content containing -- and /* survives
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
