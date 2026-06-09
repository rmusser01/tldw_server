---
id: TASK-2279
title: Add Claude-style exact fs.edit primitive
status: Done
labels:
- mcp
- filesystem
- tools
- agentic-execution
references:
- https://code.claude.com/docs/en/tools-reference#edit-tool-behavior
- Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md
- https://github.com/rmusser01/tldw_server/pull/2321
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement a Claude-style fs.edit primitive for exact string replacement as a complement to fs.patch. The tool should enforce read-before-edit via read receipts or hashes, require exact old_string matching, reject fuzzy/regex behavior, require uniqueness unless replace_all is explicit, follow action-aware path grants, and integrate with tool-use redaction and hooks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] `fs.edit` is exposed by the filesystem MCP module with strict schema metadata, `path_scope_action="edit"`, and bounded edit eval metadata.
- [x] `fs.edit` performs exact literal UTF-8 replacement only, rejects missing matches, and rejects non-unique matches unless `replace_all=true`.
- [x] Existing-file edits require either `expected_sha256` or a valid `fs.read` receipt and recheck the preimage before writing.
- [x] Results return structured metadata without raw file content and support `dry_run=true`.
- [x] Path-policy tests cover `fs.edit` as an `edit` action.
- [x] User guide documentation describes when to use `fs.edit` versus `fs.patch` and `fs.write`.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use the existing filesystem module/read-receipt/preimage helpers. Add tests first for tool metadata, exact edit success, duplicate handling, missing preimage, read-receipt success/mismatch, dry-run, and binary rejection. Then add fs.edit schema/validation/dispatch and a small _edit_file helper that reuses workspace resolution, read receipt validation semantics, _assert_preimage_unchanged, and _atomic_write_text_file.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `fs.edit` tool definition, argument validation, async dispatch, exact replacement helper, edit-specific preimage/read-receipt validation, and structured eval metadata.
- Added regression coverage for metadata, exact replacement, required preimage, missing/non-unique old strings, `replace_all`, dry-run, read receipts, receipt context mismatch, binary rejection, and path-grant enforcement.
- Updated `mcp_unified/USER_GUIDE.md` to document `fs.edit` as the small exact-replacement primitive while keeping `fs.patch` preferred for diff-first edits.
- Addressed PR review feedback by preserving raw exact-match strings, enforcing expected-SHA precedence over receipts, rejecting overlapping matches, using no-follow descriptor reads for edit preimages, adding post-read size enforcement, and adding docstrings for the new helpers.
- Skipped the `_atomic_write_text_file` AttributeError finding as invalid: the helper is a `@staticmethod` on `FilesystemModule`, not a module-level function, and the edit path now calls it through the class explicitly.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Implemented the Claude-style exact `fs.edit` primitive as a bounded `edit` action in the existing filesystem module.
- Verification run before commit: focused filesystem/path-enforcement pytest suite, ruff on touched Python, py_compile on the production module, Bandit on the production module, and `git diff --check`.
- Draft PR: https://github.com/rmusser01/tldw_server/pull/2321
- Review follow-up verification: review regression tests, full focused filesystem/path-enforcement suite, ruff, py_compile, Bandit on production filesystem module, and `git diff --check`.
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
