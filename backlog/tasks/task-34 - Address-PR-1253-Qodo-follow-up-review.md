---
id: TASK-34
title: Address PR 1253 Qodo follow-up review
status: Done
assignee: []
created_date: '2026-05-04 04:03'
updated_date: '2026-05-04 04:48'
labels:
  - codegraph
  - mcp
  - review
dependencies:
  - TASK-31
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1253'
  - 'https://github.com/rmusser01/tldw_server/pull/1258'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address post-merge Qodo review findings from PR #1253 on a follow-up branch against dev. Scope: CodeGraph extractor resilience for ast.parse ValueError, codegraph.files limit clamping, extractor package module docstring, missing test helper type hints, and indexer I/O improvements for binary and inventory-only files. Preserve merged CodeGraph behavior and add focused regression coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Python extractor/indexer converts malformed source parse failures into per-file extraction_failed results without aborting the whole run.
- [x] #2 codegraph.files clamps user limits to the configured max_search_results cap and reports truncation consistently.
- [x] #3 Indexer avoids full-file reads for binary skips and inventory-only languages by using a small binary probe plus streaming hashing.
- [x] #4 Qodo maintainability findings for extractor package docstring and test helper type hints are addressed.
- [x] #5 Focused CodeGraph/MCP tests Bandit touched scope and git diff whitespace check pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified PR #1253 is merged. Open follow-up branch codex/codegraph-qodo-followup from origin/dev to address Qodo findings on merged CodeGraph code.

Implemented Qodo follow-up fixes: Python AST parsing catches ValueError; indexer converts extractor ValueError into per-file extraction_failed rows; codegraph.files limit is clamped by max_search_results; inventory-only files use a binary probe and streaming file hash instead of full read_bytes; extractor package now has a module docstring; test policy helpers have type hints.

Verification: focused CodeGraph/MCP suite passed with 51 passed and 5 warnings; Ruff check on touched files passed; Bandit touched scope wrote /tmp/bandit_codegraph_qodo_followup.json with 0 results and 0 errors; git diff --check passed.

Opened draft follow-up PR #1258 against dev for the Qodo review fixes. PR remains draft pending a human-authored Change summary.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed post-merge Qodo review findings from PR #1253 on follow-up branch codex/codegraph-qodo-followup. The CodeGraph extractor/indexer now reports parse/extractor ValueError as per-file extraction failures, codegraph.files clamps large limits, inventory-only indexing avoids full-file reads via binary probing and streaming hashing, and the maintainability comments are fixed with a package docstring and typed test helpers.
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
