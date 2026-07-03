---
id: TASK-12078
title: Implement standalone MCP docs Stage 2 URL acquisition
status: Done
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
- [x] #1 docs.ingest_url is hidden when disabled and stale direct calls return capability_disabled before policy or network work.
- [x] #2 Source policy implements locked_down, local_first, and online_capable decisions with structured domain/wildcard/prefix matching, safe URL normalization, approval_required without fetch, denied-domain precedence, unsupported scheme denial, and credential URL denial.
- [x] #3 URL fetcher uses injectable resolver/transport, validates unsafe IP ranges, prevents DNS rebinding through validated-address transport binding, handles manual redirects with per-hop policy/DNS/IP checks, enforces redirect limits, content-type limits, transferred and decoded body limits, and robots fail-closed behavior.
- [x] #4 Extraction uses lazy optional trafilatura/bs4 imports with stdlib HTML/text fallback and preserves source_url/canonical_uri without pretending URLs are local paths.
- [x] #5 Approved fake URL ingestion writes to SQLite/FTS5, applies keywords and collections, and is retrievable via docs.search and docs.context.
- [x] #6 MCP provider and host shim expose docs.ingest_url only when enabled, validate url arguments, categorize it as ingestion, and report disabled/enabled/extractor status in docs.status.
- [x] #7 Import-boundary tests verify mcp_unified.docs has no top-level tldw_Server_API, requests, httpx, aiohttp, playwright, trafilatura, or bs4 imports; tests do not use live internet.
- [x] #8 Focused docs MCP tests and Bandit on touched Python paths pass.
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
Task 5 complete locally.
- Red evidence: service test first failed because disabled web acquisition still fetched and created a document; rich-title regression preserved the URL page heading when trafilatura supplies body text.
- Green evidence: focused service tests passed (`5 passed, 6 warnings`); full docs MCP tests passed (`135 passed, 6 warnings`); Black check reported 5 touched files unchanged; `git diff --check` was clean.
- Bandit evidence: `/tmp/bandit_mcp_docs_service.json` reported `errors: []` and `results: []` for `mcp_unified/docs/acquisition`, `mcp_unified/docs/importers`, and the Task 5 service/extraction tests.
- Review gates: verified disabled calls return `capability_disabled` before policy/resolver/transport work, approval-required and robots fail-closed paths do not fetch, approved fake URL ingestion writes SQLite/FTS5 content with keywords and collections, docs.search/docs.context retrieve it, unchanged detection works, and rich extraction preserves static title/sections metadata.
Task 6 complete locally.
- Red evidence: provider/shim tests first failed because `docs.ingest_url` was unknown, not advertised when enabled, `docs.status` lacked `web_extractors` and enabled availability, and the host shim did not reject blank URL arguments; a review-driven provider validation regression failed until enabled direct calls rejected blank URLs before acquisition.
- Green evidence: provider/shim tests passed (`16 passed, 6 warnings`); full docs MCP tests passed (`141 passed, 6 warnings`); Black check reported 4 touched files unchanged; `git diff --check` was clean.
- Bandit evidence: `/tmp/bandit_mcp_docs_provider_url.json` reported `errors: []` and `results: []` for the provider, host shim, and Task 6 tests.
- Review gates: verified `docs.ingest_url` remains hidden when disabled, stale disabled direct provider calls return `capability_disabled`, enabled provider definitions mark the tool as ingestion/write-capable, enabled direct calls validate non-empty URL and delegate keywords/collections/title to the acquisition service, `docs.status` reports enabled availability plus extractor names, and the host shim rejects blank URL before provider execution.
AC #1 resolved with Task 6 plus the Task 5 service guard: disabled configurations do not advertise `docs.ingest_url`, stale provider calls return `capability_disabled`/`web_acquisition_disabled`, and service-level disabled calls return before policy, resolver, or transport work.
Task 7 complete locally.
- Red evidence: focused boundary/config tests failed because the docs MCP config lacked explicit locked-down URL defaults (`web_source_profile` missing).
- Green evidence: focused import-boundary/config tests passed (`5 passed, 6 warnings`); full docs MCP tests passed (`143 passed, 6 warnings`); Black check reported 2 touched Python test files unchanged; `git diff --check` was clean.
- Bandit evidence: `/tmp/bandit_mcp_docs_boundaries.json` reported `errors: []` and `results: []` for the Task 7 Python test files; config YAML was non-code.
- Review gates: verified the standalone docs package has no AST-level imports of `tldw_Server_API`, `requests`, `httpx`, `aiohttp`, `playwright`, `trafilatura`, or `bs4`; package import does not load `trafilatura` or `bs4`; tests use fake/no-live-internet paths; and repo MCP config keeps web acquisition disabled with explicit locked-down URL defaults.
Task 8 final verification complete.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/docs -q --tb=short` -> `143 passed, 6 warnings`.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified -k "docs or write_tools or validator" -q --tb=short` -> `167 passed, 419 deselected, 52 warnings`.
- Import smoke: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -c "import importlib, sys; optional=['trafilatura','bs4','requests','httpx','aiohttp','playwright']; [sys.modules.pop(name, None) for name in optional]; module=importlib.import_module('mcp_unified.docs'); print(module.DocsSettings.from_mapping({}).enable_web_acquisition); loaded=[name for name in optional if name in sys.modules]; print('loaded_optional=', loaded)"` -> `False` and `loaded_optional= []`.
- Formatting: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m black --check mcp_unified/docs/acquisition mcp_unified/docs/settings.py mcp_unified/docs/importers/base.py mcp_unified/docs/importers/html.py mcp_unified/docs/importers/local.py mcp_unified/docs/mcp_module.py tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py tldw_Server_API/tests/MCP_unified/docs` -> `24 files would be left unchanged` after Black formatted `mcp_unified/docs/acquisition/policy.py`.
- Bandit: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r mcp_unified/docs/acquisition mcp_unified/docs/settings.py mcp_unified/docs/importers/base.py mcp_unified/docs/importers/html.py mcp_unified/docs/importers/local.py mcp_unified/docs/mcp_module.py tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py tldw_Server_API/tests/MCP_unified/docs -f json -o /tmp/bandit_mcp_docs_url_acquisition.json` -> `errors: []`, `results: []`.
- Security invariant review: every item in the Stage 2 checklist is covered by passing docs tests, including hidden disabled tool, stale `capability_disabled`, approval-required no-fetch, locked-down/local-first/online-capable policy behavior, domain/wildcard/prefix handling, credential/scheme/denied-domain precedence, private/reserved IP denial, validated-address transport requirement, redirect revalidation/limits, content-type/body limits, robots fail-closed, lazy optional extractors, URL ingestion into store/search/context with keywords/collections, and disabled repo config.
- Touched files: `mcp_unified/docs/acquisition/*`, `mcp_unified/docs/settings.py`, `mcp_unified/docs/importers/{base,html,local}.py`, `mcp_unified/docs/mcp_module.py`, `tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py`, `tldw_Server_API/Config_Files/mcp_modules.yaml`, docs MCP tests under `tldw_Server_API/tests/MCP_unified/docs`, and this Backlog task.
- Known skips/blockers: none. Web acquisition remains disabled by default; optional rich extractors remain lazy and non-required.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented optional standalone MCP docs URL acquisition with source policy, safe resolver-bound fetching, lazy extraction fallback, SQLite/FTS5 ingestion, MCP provider exposure, host shim validation, disabled-by-default config, and import-boundary safeguards. Final verification passed: docs MCP tests `143 passed`, adjacent MCP selection `167 passed, 419 deselected`, import smoke kept optional web dependencies unloaded, Black passed on the touched Python scope, and Bandit reported no findings.
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
