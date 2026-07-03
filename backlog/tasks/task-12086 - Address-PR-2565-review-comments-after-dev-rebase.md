---
id: TASK-12086
title: Address PR 2565 review comments after dev rebase
status: Done
labels:
- mcp
- docs
- review
priority: high
ordinal: 12086
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Evaluate and address active review comments and CI/code-scanning issues on PR #2565 after rebasing the standalone MCP docs branch onto latest dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm PR branch is based on latest origin/dev and inspect active review threads/checks.
2. Classify stale/off-diff feedback versus valid current PR issues.
3. Implement valid fixes in the standalone docs package, host DocsModule shim, tests, and documentation task files.
4. Run focused regression tests, formatting, Bandit, and diff checks.
5. Commit, force-push with lease, and update/resolve PR feedback where possible.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Confirmed `codex/mcp-docs-stage1` is already rebased onto latest fetched `origin/dev` (`9445a1731989a2d8b8328f89c0a3fcd4bd3c43c0`); `git rebase origin/dev` reported the branch is up to date.
- Addressed current PR review issues in the standalone docs package:
  - HTTP fetcher: chunked transfer decoding, distinct unsupported content-encoding reason, missing content-type denial under allowlists, outbound header/target CRLF rejection, and populated resolver private-address metadata.
  - Extraction/importers: rich extraction gated to HTML content, charset parsing with spaced parameters, directory imports skipping unsupported files, chunk overlap validation, Markdown fenced-code heading suppression, and improved static HTML text joining/heading flushing.
  - Store/retrieval: total search counts independent of page size, context snippet trimming consistency, empty collection preservation, FTS cleanup trigger, and removal of redundant scoped index definitions.
  - Provider/host shim: clear required-field validation, docs query strings no longer blocked by the generic `--` sanitizer, host execution offloaded through `asyncio.to_thread`, and missing public docstrings added.
  - Config/errors/tests: stricter boolean/integer coercion, `DocsError` initialized as a normal exception, and docs tests marked as unit tests.
- Addressed additional active review items:
  - Context builder pages beyond the first search window when document diversity limits would otherwise under-fill the requested chunk count.
  - Standalone default DB path is absolute under the user data directory instead of CWD-relative.
  - Store status fallbacks log debug context, and document upserts use UPSERT/RETURNING with legacy unique-index migration support.
  - Import-boundary test now clears cached docs modules before checking optional extractor imports.
  - Acquisition test doubles moved into a shared docs test helper module.
  - Backlog/spec review comments fixed: unique Stage 2 docs task ID (`TASK-12076.1`), non-empty Stage 1 plan acceptance criteria, aligned document types in the catalog spec, and plan helper naming aligned with insert-only behavior.
- Addressed Cubic follow-up review items:
  - Optional integer fields now reject malformed string values with a `DocsError` instead of leaking `ValueError`.
  - `default_scope` is parsed from module settings, including mapping values for owner/profile scopes.
  - Static HTML extraction now treats `div` as a block boundary and preserves `pre` text indentation.
  - Markdown parsing records section `end_char` offsets and only closes fenced code blocks with a matching fence marker.
  - Directory imports materialize keyword/collection iterables once so one-shot iterators apply to every imported file.
  - URL fetching denies non-2xx responses before ingestion and stores query-distinct canonical URLs without exposing query strings in public policy details.
  - Context7-compatible `get-library-docs` advertises and applies the optional `tokens` budget.
  - Host docs write tools reject whitespace-only `document_id` values before provider execution.
  - Stage 1 plan/spec snippets and Backlog task metadata now match the implemented schema/status state.
- Stale/off-diff review feedback remains to resolve on GitHub: calendar/frontend CodeQL and Gemini comments refer to files that are not in the current PR diff after retargeting/rebasing to `dev`.
- Verification:
  - `python -m black --check apps/mcp-unified/src/mcp_unified/docs tldw_Server_API/app/core/MCP_unified/adapters/docs tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py tldw_Server_API/tests/MCP_unified/docs` passed.
  - `python -m pytest tldw_Server_API/tests/MCP_unified/docs -q` passed: 187 passed, 4 warnings.
  - `python -m pytest tldw_Server_API/tests/MCP_unified/docs tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py tldw_Server_API/app/core/MCP_unified/tests/test_write_tools_validators.py -q` passed: 244 passed, 4 warnings.
  - Follow-up focused suite passed: 170 passed, 4 warnings.
  - Follow-up full docs plus adjacent MCP suite passed: 252 passed, 4 warnings.
  - `git diff --check` passed.
  - `python -m bandit -r apps/mcp-unified/src/mcp_unified/docs tldw_Server_API/app/core/MCP_unified/adapters/docs tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py -f json -o /tmp/bandit_mcp_docs_pr2565_review.json` passed with `errors: []` and `results: []`.
- Follow-up user requirement handled: standalone MCP docs web extraction remains optional via a new `docs-web` extra that installs `beautifulsoup4>=4.12.0` and `trafilatura>=1.6.0`.
- Added standalone package metadata for the `docs-web` extra and updated the runtime package boundary gate to include it.
- Added regression coverage for the packaging extra and for BeautifulSoup fallback extraction when trafilatura is unavailable.
- Verification follow-up:
  - Red proof before packaging update: focused docs package metadata test failed with `KeyError: 'docs-web'` while existing extractor tests passed.
  - `python -m pytest tldw_Server_API/tests/MCP_unified/docs tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py tldw_Server_API/app/core/MCP_unified/tests/test_write_tools_validators.py -q` passed: 254 passed, 4 warnings.
  - `git diff --check` passed.
  - `python -m bandit apps/mcp-unified/src/mcp_unified/package_metadata.py tldw_Server_API/tests/MCP_unified/docs/test_docs_package_metadata.py tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_extract.py -f json -o /tmp/bandit_mcp_docs_web_extra_clean.json` passed with 0 results. The broader touched runtime boundary test file still has existing test-only Bandit baseline findings (assert/subprocess patterns), with the changed assertion locally marked `# nosec B101`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #2565 was confirmed rebased onto latest `origin/dev`, then current review feedback was addressed across the standalone MCP docs corpus and host docs shim. The changes harden URL fetching, local import parsing, HTML/Markdown parsing, SQLite/FTS behavior, context building, provider validation, async host execution, config coercion, Backlog/spec metadata, and docs test coverage. Follow-up Cubic review items were fixed and verified with Black, focused docs tests, adjacent MCP tests, `git diff --check`, and Bandit with no findings.
Follow-up packaging requirement completed: standalone MCP docs web scraping/extraction support now uses an optional `docs-web` extra for BeautifulSoup4 and trafilatura, preserving locked-down/base installs while advertising and testing the rich HTML extractor dependencies.
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
