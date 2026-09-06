---
id: TASK-13195
title: Expose synchronous web article summarization in Service Prompts
status: In Progress
assignee: []
created_date: '2026-09-05 22:49'
updated_date: '2026-09-06 00:05'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2907'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Approved bounded slice: add system and summary instructions to existing shared Service Prompts Settings for /media/process-web-scraping. Resolve one authenticated owner configuration per request, preserve explicit prompts and engine-specific defaults, disabled-analysis and persistence behavior, fix individual-URL system-prompt forwarding, and leave scheduled workflows and /ingest-web-content unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared WebUI and extension edit and reset web system and summary instructions using existing storage
- [x] #2 One owner-specific prompt configuration is reused across pages with independent explicit-part precedence
- [x] #3 Engine-specific defaults, inactive analysis and persistence behavior remain compatible; individual URL system prompt reaches the model
- [x] #4 Focused backend and UI regressions, Bandit and independent review pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record scope and baseline. 2. Add failing model-facing tests for request scope and forwarding; implement minimal core integration. 3. Add registry and shared Settings metadata and tests. 4. Verify defaults, scope isolation, fallback paths, security and API contract; review and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented owner-bound web summary overrides, registry and shared Settings metadata. RED tests caught missing prompt lookup, unnecessary default-file loading, and persistence owner mismatch; fixed all three. Independent review found an outdated strict fallback helper contract test; updating signature and forwarding assertions without weakening it. Shared Settings 76 passed; prompt client/domain 124 passed. Bandit on all seven touched runtime files: zero findings. Broader scraping regression and API validation in progress.

Compatibility consumers: 92 passed (usage events, custom headers, permission claims, media patchpoints, workflow adapters). Strict fallback signature/forwarding regression: 4 passed after review fix. Settings: 76 passed in shared config and 76 passed in WebUI config. OpenAPI export and TypeScript generation succeeded; fingerprint unchanged and check passed. Python compileall passed. Ruff has only two verified pre-existing findings (endpoint B004 and enhanced scraper UP015); new helper/tests clean.

Design: Docs/Design/web-summary-service-prompt.md. Final focused backend run: 125 passed (35 owner/model-facing web integration cases plus 90 registry/API cases). Independent review: no production correctness or over-engineering findings; sole strict-signature test finding fixed and verified. Temporary dependency symlinks removed; existing dependency trees untouched. Broad 2,414-case scraping run still in progress.

Broad regression discovered host-level multiprocessing semaphore exhaustion: fixture-generator tests fail creating multiprocessing.Event with OSError(28), before application code. Reproduced with standalone stdlib-only Python probe both inside and outside sandbox. Disk has 293 GiB free. No host limits changed and no unrelated processes killed. Full broad verification cannot be considered green; implementation remains staged/uncommitted pending this verification limitation.

Stopped only this task’s broad pytest run after confirming host resource failure: 1,224 passed, 2 skipped, 11 failed at interruption. One failure was the strict signature test collected before its fix (fresh rerun 4 passed); remaining failures are being classified from the final trace. No feature commit or PR created.

Correction after full failure-log inspection: the broad run’s non-semaphore failure was NOT the already-fixed forwarding test. It was unchanged test_phase3_preflight_characterization.py::test_analyzer_and_public_entry_point_signatures_match_current_inventory, whose expected PEP 604 union spelling differs from Python 3.11 inspect output using typing.Optional. Canonical article source and inventory test are unchanged; enhanced scrape_article argument/return AST is byte-for-byte equivalent structurally to base. The other 10 failures are SemLock OSError(28). Fresh strict-forwarding test remains green. Work is staged, not committed; awaiting direction on proceeding with these unrelated verification limitations.

User explicitly approved committing and opening a PR with the documented broader-test limitations. Proceeding to publish against dev; no merge requested in this step.

Published implementation commit d328765206 on codex/web-summary-service-prompt. PR #2907 opened against dev: https://github.com/rmusser01/tldw_server/pull/2907. Verification limitations are explicit in the PR body. Task remains In Progress pending PR review and integration.

Qodo review posted: zero bugs and three rule findings. Verified two documentation gaps (legacy test helper docstring; resolver module/parameter/return/cleanup contract). Addressing those without behavior changes. Third finding repeats the documented broader-suite limitation explicitly approved by requester; will reply with evidence and preserve the limitation rather than claim a full green suite.

Qodo documentation fixes verified: 35 web-summary integration tests passed; Ruff and format checks passed; touched resolver Bandit reported zero findings. Only docstrings and tracking notes changed. Replied to the broader-suite finding with the existing requester-approved limitation and evidence; required CI remains enforced.

Rebased cleanly onto newly advanced dev 946e591ee9 (pixel-migu addition). git range-diff confirms all three existing PR patches are unchanged by the rebase. Rebased verification: 129 focused integration/registry/API/strict-contract tests passed; Bandit across all seven touched runtime files found zero issues. Qodo had cleared all findings on the pre-rebase head; publishing the rebased head for current-head review and CI.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added user-editable web article system and summary instructions through existing Service Prompts storage and shared Settings, scoped to synchronous web scraping. One immutable owner snapshot preserves explicit overrides and engine defaults across pages; persistence uses the same authenticated owner. Added model-facing, API, Settings and compatibility coverage. Focused tests and Bandit pass; broader suite limitations are documented and requester approved PR publication with them.
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
