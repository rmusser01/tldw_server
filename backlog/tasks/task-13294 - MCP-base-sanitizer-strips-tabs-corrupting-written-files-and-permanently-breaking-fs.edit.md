---
id: TASK-13294
title: >-
  MCP base sanitizer strips tabs, corrupting written files and permanently
  breaking fs.edit
status: To Do
assignee: []
created_date: '2026-09-22 04:44'
labels:
  - bug
  - mcp
  - security
dependencies: []
references:
  - 'tldw_Server_API/app/core/MCP_unified/modules/base.py:766'
  - >-
    tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py:1480
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`core/MCP_unified/modules/base.py:sanitize_input` filters control characters with:

```python
return "".join(ch for ch in s if ch >= " " or ch == "\n")
```

It keeps `\n` and **drops `\t` and `\r`**. It is applied to every tool call via `tool_execution/security.py:harden_and_sanitize_tool_arguments`, and 22 modules inherit it unchanged.

Executed proof:
```
input : 'all:\n\tgcc -o x x.c\n'
output: 'all:\ngcc -o x x.c\n'   # the required tab is gone
```

Two consequences:
1. **`fs.write` silently corrupts tab-significant files and reports success.** A Makefile loses its required tabs. The `expected_sha256` receipt hashes the on-disk pre-image, so the integrity guard structurally cannot catch it.
2. **`fs.edit` is permanently unusable on tab-indented files.** It does exact string replacement, so an `old_string` containing a tab can never match content whose tabs were stripped on the way in.

The correct whitespace class already exists in the same package — `filesystem_module.py:_sanitize_patch_diff` preserves `\n`, `\r` **and** `\t`, with a docstring saying so. Its per-tool exemption table is dead code because the blanket sanitizer already ran upstream.

**Second defect in the same function (same fix window).** The denylist at `:779-788` rejects `["';", '";', "--", "/*", "*/", "xp_", "sp_"]` on pure data bound to parameterised queries, so it buys no injection protection while rejecting ordinary content: Markdown `---` rules, `src/*.py` pathspecs, `SELECT 1 -- note`, git `-- path`, and filenames like `exp_data.csv`. `web_tool_base.py:51-70` already fixed this for web tools only; its docstring diagnoses the problem.

The base sanitizer test uses `os.urandom(4).hex()` as its "safe" fixture — a hex string can never contain a denied substring, so nothing asserts that legitimate content survives.

Found by the comprehensive core-module review; the tab-stripping behaviour independently reproduced by the orchestrator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A failing test writes tab-indented content through fs.write and asserts the tab survives round-trip
- [ ] #2 A failing test proves fs.edit can match an old_string containing a tab
- [ ] #3 sanitize_input preserves \\t and \\r, matching filesystem_module._sanitize_patch_diff
- [ ] #4 The denylist no longer rejects ordinary content -- Markdown ---, src/*.py, git -- path, exp_ filenames -- or is removed as ineffective for parameterised queries
- [ ] #5 The base sanitizer test uses a fixture that could actually contain denied substrings and asserts legitimate content survives
- [ ] #6 filesystem_module's now-dead exemption table is removed or made reachable
- [ ] #7 Bandit run for touched scope
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
