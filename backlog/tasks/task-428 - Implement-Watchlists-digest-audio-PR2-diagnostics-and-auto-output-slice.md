---
id: TASK-428
title: Implement Watchlists digest audio PR2 diagnostics and auto-output slice
status: Done
references:
- Docs/superpowers/plans/2026-05-18-watchlists-digest-audio-briefing-implementation-plan.md
- Docs/superpowers/specs/2026-05-18-watchlists-digest-audio-briefing-prd-design.md
- https://github.com/rmusser01/tldw_server/pull/1838
modified_files:
- Docs/superpowers/plans/2026-05-18-watchlists-digest-audio-briefing-implementation-plan.md
- tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py
- tldw_Server_API/app/api/v1/endpoints/watchlists.py
- tldw_Server_API/tests/Watchlists/test_preview_endpoint.py
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobFormModal.tsx
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobFormModal.live-summary.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR2 from the Watchlists digest/audio implementation plan. Scope: source validation diagnostics API for draft/saved source testing plus scheduled output/delivery contract visibility in /watchlists. Preserve existing news/OSINT/CTI workflows and do not start guided pipeline or audio artifact persistence work in this slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Draft and saved source tests return source preview diagnostics for scrape-rule site/forum sources.
- [x] Scheduled watchlist monitor payloads can declare `output_prefs.auto_output.enabled` for recurring output/delivery/audio runs.
- [x] Manual/test-only output creation remains explicit and does not set scheduled `auto_output`.
- [x] Delivery status metadata labels include skipped delivery outcomes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-18-watchlists-digest-audio-briefing-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Reused `validate_selector_rules` for source preview diagnostics rather than adding a second selector validation path.
- Kept diagnostics optional on `PreviewResponse` so existing job preview callers are not forced to handle source-only metadata.
- Preserved existing monitor form behavior by only enabling `auto_output` when a recurring schedule combines with delivery/audio output or an existing enabled auto-output preference.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR2 implementation plus review-fix pass.

Initial implementation:
- Added source preview diagnostics to draft/saved source tests, including fetch mode, selector validation errors/warnings, and dedupe preview key for scrape rules.
- Added scheduled auto-output payload creation for guided pipeline drafts and recurring monitor forms when delivery/audio output is configured.
- Added skipped delivery label normalization for output metadata.

Review-fix pass:
- Addressed Gemini review comments by making selector validation errors/warnings tolerant of missing/None lists.
- Normalized frontend template version once before reuse in pipeline payload construction.

Verification after review fixes:
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Watchlists/test_fetchers_scrape_rules.py tldw_Server_API/tests/Watchlists/test_preview_endpoint.py -q -> 9 passed.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Watchlists/test_job_output_prefs_roundtrip.py tldw_Server_API/tests/Watchlists/test_newsletter_briefing_gaps.py -q -> 47 passed.
- bun run test -- src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts src/components/Option/Watchlists/JobsTab/__tests__/JobFormModal.live-summary.test.tsx src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts --maxWorkers=1 --no-file-parallelism -> 42 passed.
- git diff --check -> passed.
- Bandit on touched backend files -> 0 findings.
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
