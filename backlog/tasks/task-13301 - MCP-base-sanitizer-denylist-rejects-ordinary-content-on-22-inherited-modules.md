---
id: TASK-13301
title: MCP base sanitizer denylist rejects ordinary content on 22 inherited modules
status: Done
assignee: []
created_date: '2026-09-22 04:52'
updated_date: '2026-09-22 22:28'
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
- [x] #1 Denylist removed from BaseModule.sanitize_input; control-char strip retained
- [x] #2 Base-level test asserts legitimate content containing -- and /* survives
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in 475bdfb929, alongside TASK-13294 (same function).

AC1: the denylist ["';", '";', "--", "/*", "*/", "xp_", "sp_", "\\x00"] is removed from BaseModule.sanitize_input; the control-character strip is retained as CONTROL_CHARS_RE.sub(""). The docstring records why it went: these values are data handed to parameterised DB_Management queries, so the denylist bought no injection protection while refusing a Markdown "---" rule, "SELECT 1 -- note", the git pathspec "-- src/app.py", the glob "src/*.py" and the filename "exp_data.csv". Its "\\x00" entry was a literal backslash-x-0-0 and never matched anything; real NULs are removed by the control-character strip.

AC2: test_base_sanitizer_preserves_content.py replaces the os.urandom(4).hex() fixture that could never contain a denied substring. 9 parametrised ordinary-content cases plus tab/CR preservation, a Makefile recipe, tab-indented Python, and a real NUL still being stripped.

Beyond the stated scope: web_tool_base's sanitize_input override existed only to escape this denylist, and filesystem, run_command and sandbox each carried their own near-identical copy for the same reason, drifted to three different whitespace classes. With the denylist gone all four are dead weight, so all four are deleted and the base is the one implementation. test_no_module_shadows_the_base_sanitizer keeps it that way. CONTROL_CHARS_RE moved to base.py; web_tool_base re-exports it for web_fetch's URL validation.

Note: removing the denylist changed test_validation_and_sanitization.py::test_deep_argument_sanitization_blocks_nested_patterns, which asserted a nested "/* injected */" raised ValueError. Its real coverage was the recursion, not the denylist, so it now asserts recursion against the surviving behaviour and is renamed ...recurses_into_nested_values, with the reason recorded in the test body rather than changed silently.

Verification: MCP_unified in-app 13 failed / 3301 passed vs baseline 13 / 3281, identical failure set. tests/MCP + MCP_Hub + MCP_unified 4 failed, unchanged. tests/sandbox + Services 24 failed, identical with and without the change (stash-isolated). Bandit clean over the five touched files (run via uvx; bandit is CI-only, not a local dependency).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Denylist removed from BaseModule.sanitize_input and the four modules that had cloned overrides to escape it (filesystem, run_command, sandbox, web_tool_base) collapsed onto the single base implementation. New test file asserts legitimate content survives, replacing a hex fixture that could never have failed.
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
