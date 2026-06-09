---
id: TASK-2332
title: Implement MCP filesystem unified diff parser and governed patch primitive
status: Done
labels:
- mcp
- filesystem
- security
- policy
- implementation
priority: High
documentation:
- Docs/superpowers/specs/2026-06-09-mcp-filesystem-diff-parser-design.md
- Docs/superpowers/plans/2026-06-09-mcp-filesystem-diff-parser-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-06-09-mcp-filesystem-diff-parser-design.md
- Docs/superpowers/plans/2026-06-09-mcp-filesystem-diff-parser-implementation-plan.md
- tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py
- tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_diff.py
- tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_patch_parser.py
- tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next MCP filesystem slice: a robust unified diff parser and governed fs.patch primitive that prefers structured patching over raw edits, while keeping fs.read/fs.write/fs.patch under workspace/path/action grants with hashes, safe failures, and audit-friendly result metadata.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Unified diff parser preserves `\ No newline at end of file` markers for additions, removals, and context lines without corrupting line counts.
- [x] #2 Unified diff parser accepts safe file paths containing spaces in standard `---` / `+++` headers while continuing to reject unsafe absolute, drive-qualified, empty, dot, and traversal paths.
- [x] #3 `fs.patch` end-to-end application preserves missing final-newline content and continues to require hashes or read receipts for existing-file edits.
- [x] #4 Delete and rename diffs remain explicitly unsupported until separate path-policy action verbs exist.
- [x] #5 Focused parser/module tests, Bandit on touched Python scope, and `git diff --check` pass before PR.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design and implementation plan added:
- `Docs/superpowers/specs/2026-06-09-mcp-filesystem-diff-parser-design.md`
- `Docs/superpowers/plans/2026-06-09-mcp-filesystem-diff-parser-implementation-plan.md`

Baseline verification from clean `origin/dev` worktree:
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_patch_parser.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q`
  - Result: 93 passed, 4 warnings.

RED parser tests before implementation:
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_patch_parser.py -q`
  - Result: 3 expected failures for path truncation at spaces, forced final newline on additions, and orphan no-newline marker acceptance.

Implementation:
- Extended parsed hunk lines with `has_trailing_newline`.
- Attached `\ No newline at end of file` markers to the preceding hunk line and rejected orphan/duplicate markers.
- Updated in-memory patch application so added lines do not force a final newline when the diff says none exists.
- Changed header path parsing to preserve unambiguous paths with spaces while retaining normalization checks.
- Added end-to-end `fs.patch` coverage for no-final-newline output.

PR review follow-up:
- Reopened after rebasing onto `origin/dev` on 2026-06-09.
- Addressed still-valid review findings: helper docstrings, stripping space-separated header timestamp metadata without truncating paths containing spaces, and preserving tab-separated diff header metadata through `fs.patch` input sanitization.

Verification:
- Parser focused: 14 passed, 4 warnings.
- Parser + filesystem module focused: 97 passed, 4 warnings.
- PR review focused RED tests before implementation:
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_patch_parser.py::test_parse_unified_diff_preserves_safe_paths_with_spaces tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py::test_filesystem_patch_preserves_tab_header_metadata_during_sanitization -q`
  - Result: 2 expected failures for header metadata parsed as part of the path.
- PR review focused verification:
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_patch_parser.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q`
  - Result: 99 passed, 4 warnings.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m py_compile tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_diff.py tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
  - Result: passed with no output.
- Review Bandit: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_diff.py tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py -f json -o /tmp/bandit_mcp_fs_diff_parser_v2_review.json`
  - Result: 0 findings, 2583 LOC scanned.
- Review `git diff --check`
  - Result: passed with no output.
- Bandit: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_diff.py tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py -f json -o /tmp/bandit_mcp_fs_diff_parser_v2.json`
  - Result: 0 findings, 2554 LOC scanned.
- `git diff --check`
  - Result: passed with no output.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Improved the MCP filesystem unified diff parser used by `fs.patch`. The parser now preserves `\ No newline at end of file` markers, rejects orphan/duplicate no-newline markers, and keeps safe header paths with spaces instead of truncating them. `fs.patch` now preserves missing-final-newline output end to end while continuing to use the existing path scope, hash/read-receipt, atomic write, dry-run, and rollback behavior.

Delete and rename patch application remain intentionally unsupported until separate path-policy action verbs are implemented.
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
