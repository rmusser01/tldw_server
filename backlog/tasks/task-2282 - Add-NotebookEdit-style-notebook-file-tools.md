---
id: TASK-2282
title: Add NotebookEdit-style notebook file tools
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-27 20:33
labels:
- mcp
- filesystem
- notebooks
- tools
- agentic-execution
dependencies: []
references:
- https://code.claude.com/docs/en/tools-reference
documentation:
- Docs/Design/2026-06-27-mcp-notebook-edit-tools-design.md
- Docs/superpowers/plans/2026-06-27-mcp-notebook-edit-tools.md
modified_files:
- Docs/Design/2026-06-27-mcp-notebook-edit-tools-design.md
- Docs/superpowers/plans/2026-06-27-mcp-notebook-edit-tools.md
- apps/mcp-unified/src/mcp_unified/USER_GUIDE.md
- apps/mcp-unified/src/mcp_unified/profiles/presets.py
- tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py
- tldw_Server_API/app/core/MCP_unified/modules/implementations/notebook_files.py
- tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_notebook_tools.py
- tldw_Server_API/app/core/MCP_unified/tests/test_notebook_files.py
- tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement notebook-safe MCP tools modeled after Claude Code NotebookEdit. Support reading notebook structure and editing cells by cell id with replace, insert, and delete modes; require notebook path grants; preserve JSON validity; avoid raw whole-notebook overwrites for cell edits; include validation, diff summaries, redacted telemetry, and tests for Jupyter notebooks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design approved and committed. Implementation plan added at Docs/superpowers/plans/2026-06-27-mcp-notebook-edit-tools.md. Baseline before notebook edits: filesystem module tests had 103 passed and 1 pre-existing glob metadata failure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented notebook-safe MCP file tooling for TASK-2282 and addressed PR review feedback after rebasing on latest `origin/dev`. Original implementation added `notebook.read` for structure-first `.ipynb` inspection with bounded optional source previews and read receipts, plus `notebook.edit_cell` for replace, insert, and delete by stable cell id with preimage checks, optional lock leases, dry-run support, write-size limits, and stale code-output clearing on code-cell replacement. Review remediation added module/class/function docstrings in the new notebook test modules, validates existing notebook cell types during parsing, strips code-only `outputs` and `execution_count` when replacing a code cell with markdown/raw, maps oversized notebook reads/edits to `notebook_too_large`, and adds receipt-authorized edit coverage.

Verification:
- Red run before fixes: `pytest test_notebook_files.py test_filesystem_notebook_tools.py -q` failed on invalid existing cell types, code-only field stripping, and oversize reason-code mapping as expected.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_notebook_files.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_notebook_tools.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py -q` -> 80 passed, 4 warnings.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q` -> 103 passed, 1 failed. The failure is the known pre-existing `test_filesystem_glob_marks_file_size_unavailable` glob metadata failure recorded before notebook changes.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/MCP_unified/modules/implementations/notebook_files.py tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py apps/mcp-unified/src/mcp_unified/profiles/presets.py -f json -o /tmp/bandit_mcp_notebook_edit_tools_review.json` -> exit 0, zero findings, output at `/tmp/bandit_mcp_notebook_edit_tools_review.json`.
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
