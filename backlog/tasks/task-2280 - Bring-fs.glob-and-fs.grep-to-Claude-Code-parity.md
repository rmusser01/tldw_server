---
id: TASK-2280
title: Bring fs.glob and fs.grep to Claude Code parity
status: Done
labels:
- mcp
- filesystem
- search
- tools
- agentic-execution
references:
- https://code.claude.com/docs/en/tools-reference
modified_files:
- pyproject.toml
- mcp_unified/USER_GUIDE.md
- tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py
- tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py
- Docs/superpowers/plans/2026-06-09-mcp-glob-grep-parity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enhance filesystem search tools to match useful Claude Code Glob/Grep behavior. Cover mtime-sorted capped glob results with truncation, configurable gitignore handling, grep output modes (files_with_matches, content, count), language/type filters, multiline support where safe, direct-file searches for ignored files, path grants, redacted telemetry, and focused tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] `fs.glob` returns capped results sorted by newest modification time by default and supports `sort_by: "path"` for deterministic path order.
- [x] `fs.glob` supports opt-in `.gitignore` filtering with `respect_gitignore: true`.
- [x] `fs.grep` supports `files_with_matches`, `content`, and `count` output modes, defaulting to `files_with_matches`.
- [x] `fs.grep` supports `glob` and `type` narrowing without broadening existing include/exclude filters.
- [x] Directory `fs.grep` respects root `.gitignore` by default, while direct-file `base_path` searches can inspect an ignored file when profile/path policy allows it.
- [x] `fs.grep` supports bounded multiline regex search for `files_with_matches` and `count` modes.
- [x] `fs.glob` and `fs.grep` responses include redacted scalar eval metadata without file contents or absolute paths.
- [x] Focused regression tests cover the new behavior and adjacent command-runtime tests remain green.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-09-mcp-glob-grep-parity-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Claude-style filesystem search parity in the workspace-bounded `FilesystemModule`.

- Added `pathspec>=0.12.1` as an explicit dependency for root `.gitignore` gitwildmatch parsing.
- Added `fs.glob.sort_by`, defaulting to `modified_at`, and `fs.glob.respect_gitignore`, defaulting to `false`.
- Added `fs.grep.output_mode`, `glob`, `type`, `respect_gitignore`, and safe `multiline` options. Directory grep defaults to `.gitignore` filtering; direct-file grep intentionally bypasses gitignore filtering while still using workspace path resolution and profile/path policy.
- Added safe execution eval metadata for `fs.glob` and `fs.grep`; metadata records tool/result/truncation shape only.
- Kept all traversal and reads under existing workspace containment, file size, total byte, file count, and walk-entry limits.
- PR review fixes hardened `.gitignore` loading against symlinks, undecodable bytes, and parse failures; added diagnostics for ignored filesystem errors; and short-circuited file-match grep scans.
- PR review docstring pass added concise docstrings for newly introduced helpers and regression tests.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verification recorded: python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q => 82 passed; python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_run_command_module.py tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_registry.py -q => 21 passed; git diff --check => clean; bandit filesystem_module.py => 0 findings.
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
