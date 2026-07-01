---
id: TASK-12078
title: Implement standalone MCP docs Stage 2 URL acquisition
status: In Progress
priority: high
documentation:
- Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-url-acquisition-design.md
- Docs/superpowers/plans/2026-06-30-standalone-mcp-docs-url-acquisition-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement optional single-page URL acquisition for the standalone MCP docs corpus using the committed implementation plan. Preserve the standalone import boundary, keep web acquisition disabled by default, use TDD for each behavior slice, and complete subagent-driven review checkpoints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 docs.ingest_url is hidden when disabled and stale direct calls return capability_disabled before policy or network work.
- [x] #2 Source policy implements locked_down, local_first, and online_capable decisions with structured domain/wildcard/prefix matching, safe URL normalization, approval_required without fetch, denied-domain precedence, unsupported scheme denial, and credential URL denial.
- [x] #3 URL fetcher uses injectable resolver/transport, validates unsafe IP ranges, prevents DNS rebinding through validated-address transport binding, handles manual redirects with per-hop policy/DNS/IP checks, enforces redirect limits, content-type limits, transferred and decoded body limits, and robots fail-closed behavior.
- [x] #4 Extraction uses lazy optional trafilatura/bs4 imports with stdlib HTML/text fallback and preserves source_url/canonical_uri without pretending URLs are local paths.
- [ ] #5 Approved fake URL ingestion writes to SQLite/FTS5, applies keywords and collections, and is retrievable via docs.search and docs.context.
- [ ] #6 MCP provider and host shim expose docs.ingest_url only when enabled, validate url arguments, categorize it as ingestion, and report disabled/enabled/extractor status in docs.status.
- [ ] #7 Import-boundary tests verify mcp_unified.docs has no top-level tldw_Server_API, requests, httpx, aiohttp, playwright, trafilatura, or bs4 imports; tests do not use live internet.
- [ ] #8 Focused docs MCP tests and Bandit on touched Python paths pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Subagent-driven execution in progress.

Task 1 complete and approved.
- Commits: e65ab32a4c5fc910a7bab97e835a9b728826310d, 526c2de2925a06ded051ac3367e1bc8d67fd2fb3.
- Red evidence: focused settings/provider pytest failed from missing URL settings/status fields; review-fix regressions failed for nan/inf timeout and blank user-agent handling.
- Green evidence: focused settings/provider pytest passed after implementation and hardening (`39 passed` reported by spec/code reviewers); Bandit touched-scope checks reported no findings.
- Review gates: spec compliance approved; code quality approved after non-finite timeout, blank user-agent fallback, and full web_policy status coverage fixes.

Task 2 complete locally after subagent usage limit interrupted final worker.
- Commits: 0679b705c24137d38623fdab3ed5e973e603a980, 088f4b2983eb9e0e0aacdb22b2d5b70e033bd6bb, e761c71356daf12ccc5b3f9630492f5035d57839, b19743788d2616402726dfb1bdc18cceebeb3996, 64e5742e8b67a94f0bbba2e541b7b95c803bcf5f, 8273abbee193e7e9e829fe2326dcbd2024412e9c, 43aed74c93c10359888de7a6f212797f2e6d7a5a.
- Red evidence: policy tests failed before implementation for missing package, missing canonical_url, matched_rule query leakage, unsafe/local host allow, dot-segment prefix bypass, invalid config acceptance, legacy/local aliases, legacy numeric hosts, ambiguous host syntax, raw tab/newline URLs, and malformed numeric-looking hosts.
- Green evidence: focused policy tests passed (`46 passed, 6 warnings`), import-boundary tests passed (`3 passed, 6 warnings`), Bandit acquisition scope reported `results: []`.
- Review gates: spec compliance and code-quality issues were addressed iteratively; final local review verified pure policy/no network I/O, query/fragment redaction, fail-closed host handling, dot-segment rejection, and invalid-config fail-fast behavior.

Task 3 complete locally after subagent usage limit interrupted worker availability.
- Red evidence: fetcher tests initially failed for missing fetcher module; review-driven regressions failed for missing query-preserving request target, uncaught resolver failures, and uncaught transport failures.
- Green evidence: focused fetcher tests passed (`14 passed, 6 warnings`); full current docs MCP tests passed (`126 passed, 6 warnings`); Black check reported 5 touched files unchanged.
- Bandit evidence: `/tmp/bandit_mcp_docs_fetcher.json` reported `errors: []` and `results: []` for `mcp_unified/docs/acquisition` plus the new fetcher test file.
- Review gates: local spec review verified injectable resolver/transport, deny-before-transport private IP handling, validated-address transport gating, no-fetch-before-approval, per-hop redirect policy/DNS/IP checks, redirect limits, content-type limits, transferred/decoded body limits, query redaction in results, and robots fail-closed behavior.

Task 4 complete locally.
- Red evidence: extraction tests initially failed for missing `mcp_unified.docs.acquisition.extract`; review-driven warning regression failed until static fallback reported rich-extractor fallback state.
- Green evidence: extraction/importer tests passed (`10 passed, 6 warnings`); full current docs MCP tests passed (`130 passed, 6 warnings`); Black check reported 5 touched files unchanged.
- Bandit evidence: `/tmp/bandit_mcp_docs_extract.json` reported `errors: []` and `results: []` for `mcp_unified/docs/acquisition`, `mcp_unified/docs/importers`, and the new extraction test file.
- Review gates: local spec review verified lazy `importlib` optional extractor checks, no top-level trafilatura/bs4 imports, stdlib HTML/text fallback, URL `source_url`/`canonical_uri` preservation, optional `source_path`, and local importer compatibility.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
