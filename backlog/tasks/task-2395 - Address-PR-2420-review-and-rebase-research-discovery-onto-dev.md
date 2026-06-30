---
id: TASK-2395
title: Address PR 2420 review and rebase research discovery onto dev
status: Done
labels:
- research
- review
- pr
references:
- https://github.com/rmusser01/tldw_server/pull/2420
modified_files:
- tldw_Server_API/app/api/v1/endpoints/research_discovery.py
- tldw_Server_API/app/api/v1/schemas/research_discovery_schemas.py
- tldw_Server_API/app/core/Research/discovery/catalog.py
- tldw_Server_API/app/core/Research/discovery/identity.py
- tldw_Server_API/app/core/Research/discovery/oa.py
- tldw_Server_API/app/core/Research/discovery/router.py
- tldw_Server_API/app/core/Research/discovery/service.py
- tldw_Server_API/app/core/exceptions.py
- tldw_Server_API/tests/Research/test_research_discovery_catalog.py
- tldw_Server_API/tests/Research/test_research_discovery_endpoint.py
- tldw_Server_API/tests/Research/test_research_discovery_identity.py
- tldw_Server_API/tests/Research/test_research_discovery_router.py
- tldw_Server_API/tests/Research/test_research_discovery_service.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #2420 onto the latest dev branch, evaluate and address concrete review comments and CI issues, retarget the PR to dev, and verify the touched research discovery scope before pushing updates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR branch is rebased onto the latest origin/dev and ready to retarget to dev.
- [x] #2 Concrete PR review findings are evaluated and addressed or documented.
- [x] #3 Focused discovery and adjacent research tests pass on the rebased branch.
- [x] #4 Formatting, compile, diff-check, and touched-scope Bandit verification pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Resolved the dev rebase conflict by preserving the dev lazy router registry and adding research_discovery as an ImportedRouterSpec. Addressed review findings for endpoint rate limiting, typed exception mapping, bounded filters, SQLite snapshot offloading/cleanup, OA resolver limit/concurrency, metadata URL redaction, malformed URL parser safety, source rate-limiter isolation, docstrings, and test markers.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2420's research discovery branch onto latest origin/dev, preserving the dev lazy router registration shape and adding the discovery router as an ImportedRouterSpec. Addressed review findings by adding route rate-limit dependencies, replacing string-parsed endpoint errors with typed research discovery exceptions, bounding filter payload size/depth/key count, offloading snapshot cleanup/write work to a thread, opportunistically deleting expired snapshots, applying total_limit before OA resolver work, capping OA resolver concurrency, redacting unsafe URLs under benign metadata keys, hardening malformed URL property access, isolating rate-limiter failures per source, adding requested test markers/raw regexes/docstrings, and adding focused regression tests. Verification passed: focused review tests 77 passed; full discovery slice 103 passed; adjacent research tests 25 passed; Black check clean; compileall succeeded; git diff --check clean; Bandit touched implementation scope reported 0 findings.
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
