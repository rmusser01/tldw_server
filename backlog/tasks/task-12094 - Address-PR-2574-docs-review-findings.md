---
id: TASK-12094
title: Address PR 2574 docs review findings
status: Done
labels:
- mcp
- docs
- review-fix
priority: High
references:
- https://github.com/rmusser01/tldw_server/pull/2574
modified_files:
- apps/mcp-unified/src/mcp_unified/docs/acquisition/fetcher.py
- apps/mcp-unified/src/mcp_unified/docs/acquisition/service.py
- apps/mcp-unified/src/mcp_unified/docs/sync.py
- tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_fetcher.py
- tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_service.py
- tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py
- tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix review findings from the MCP docs source sync PR: prevent query-bearing URL secrets from leaking through ingested document metadata/MCP outputs when query persistence is disabled, and make respect_robots fail closed until a robots checker exists.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regressions for query-bearing URL redaction across ingest/search/get/context outputs. 2. Add/fix failing regression for respect_robots fail-closed behavior without resolving or fetching. 3. Patch acquisition/fetcher behavior minimally. 4. Run focused docs tests, Bandit on touched source, and diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR 2574 review findings. URL fetching now fails closed with robots_unavailable before DNS/transport when respect_robots=True and no robots checker is available. URL ingestion now sanitizes query-bearing document canonical_uri/citations and clears document source_url when persist_url_query_strings=False. Source sync now applies the same default sanitization when redirects land on query-bearing URLs, while preserving raw query URLs only for opt-in persist_url_query_strings=True paths. Added regressions for docs.ingest_url, docs.search, docs.get, docs.context, docs.list documents, and URL source sync outputs with ?token=secret. Verification: focused red/green regressions passed after the fix; /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/docs -q -> 254 passed, 4 warnings; /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r apps/mcp-unified/src/mcp_unified/docs/acquisition/fetcher.py apps/mcp-unified/src/mcp_unified/docs/acquisition/service.py apps/mcp-unified/src/mcp_unified/docs/sync.py -f json -o /tmp/bandit_mcp_docs_review_fix.json -> no findings; git diff --check -> clean. Follow-up review found a source-sync redirect leak; fixed and covered in test_docs_source_sync.py.
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
