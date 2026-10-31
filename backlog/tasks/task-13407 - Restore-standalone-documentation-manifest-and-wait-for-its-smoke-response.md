---
id: TASK-13407
title: Restore standalone documentation manifest and wait for its smoke response
status: To Do
created_date: 2026-10-01 08:10
references:
- https://github.com/rmusser01/tldw_server/pull/3023
- TASK-13377.9
documentation:
- Docs/Operations/Email_Real_PST_Validation_2026-09-26.md
modified_files:
- apps/tldw-frontend/lib/documentation.ts
- apps/tldw-frontend/e2e/smoke/all-pages.spec.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fresh PR3023 browser diagnostics captured a late GET /api/documentation/manifest HTTP500 in the native advanced standalone bundle. apps/tldw-frontend/lib/documentation.ts resolves repository sources only from cwd or ../..; Next standalone changes cwd under .next/standalone/apps/tldw-frontend and the existing CI staging copies public/static without the Docs sources. The all-pages smoke assertion can finish before this request completes. This inherited defect is not introduced or suppressed by TASK13377.9, remains outside its allowlist, and is a separate frontend repair. Independently reviewed source diagnosis is supported; reproduce the complete deployment/source layout before selecting a minimal fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reproduce the manifest HTTP500 from the supported standalone bundle with the intended documentation sources present, then verify a minimal source/runtime repair returns the expected manifest.
- [ ] #2 Add a deterministic smoke check that waits for the documentation manifest response and content, so a late HTTP500 fails instead of arriving after the assertion.
- [ ] #3 Preserve path-containment and document-source validation; do not allowlist HTTP500, broaden suppression, disable guards or count metadata smoke success as documentation acceptance.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Official CLI duplicate search 'documentation manifest' completed: /tmp/email_pr3023_ux_docs_followup_search_20261001.txt; existing returned standalone entries concern MCP documentation and do not cover this WebUI unit. Explicit ID13407 follows verified TASK13406. CLI creation is unavailable for this task due to retained repeated stack-overflow diagnostics, so official MCP is used. Raw late-response evidence /tmp/email_pr3023_ux_raw_diagnostics_20261001.jsonl. Immutable test-maintenance patch d001ed8bde3cc2aff39c1a4abe4faa59f68961c282241cae14f19f998740966f reviewed independently; reviewer says this separate inherited defect does not block that patch, but prohibits any zero-raw-error or functional documentation acceptance claim. No human assignee is invented.
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
