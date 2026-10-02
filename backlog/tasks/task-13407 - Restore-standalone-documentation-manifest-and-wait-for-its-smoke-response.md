---
id: TASK-13407
title: Restore standalone documentation manifest and wait for its smoke response
status: In Progress
assignee: []
created_date: '2026-10-01 08:10'
updated_date: '2026-10-02 01:39'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/3023'
  - TASK-13377.9
documentation:
  - Docs/Operations/Email_Real_PST_Validation_2026-09-26.md
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

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Official CLI duplicate search 'documentation manifest' completed: /tmp/email_pr3023_ux_docs_followup_search_20261001.txt; existing returned standalone entries concern MCP documentation and do not cover this WebUI unit. Explicit ID13407 follows verified TASK13406. CLI creation is unavailable for this task due to retained repeated stack-overflow diagnostics, so official MCP is used. Raw late-response evidence /tmp/email_pr3023_ux_raw_diagnostics_20261001.jsonl. Immutable test-maintenance patch d001ed8bde3cc2aff39c1a4abe4faa59f68961c282241cae14f19f998740966f reviewed independently; reviewer says this separate inherited defect does not block that patch, but prohibits any zero-raw-error or functional documentation acceptance claim. No human assignee is invented.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

2026-10-01: Human authorized completing the documentation runtime and deterministic smoke repair. Isolated follow-up branch starts at dev d81c13fddd1dac1948b30401af0388932a0af8f2. Official MCP read calls did not return; official CLI fallback used. Reproduce the deployed standalone layout, preserve containment and source validation, then verify actual manifest/content. Plan: Docs/Plans/IMPLEMENTATION_PLAN_email_followup_burndown_20261001.md.

TASK13407: original nested standalone with intended Docs/Published present reproduced HTTP500; nested-cwd unit regression RED and actual deployed error retained. Nearest ancestor lookup and Next tracing include real published docs. Browser gate waits for manifest, AuthNZ guide content and rendering; delayed manifest500/content500 regressions and page-close pending-wait regression pass after real REDs. Containment/extensions/API source validation unchanged. Existing extension documentation source layout is unchanged; no extension-docs acceptance claim. README deployment instructions updated. Local verification completed 2026-10-02 UTC (October 1 local): final production build/token sync/bundle budget passed; relocated standalone includes 324 published server documents, actual manifest/content HTTP200 and traversal/unsupported-source HTTP400. Final strict ordinary smoke105 passed, classifier7 passed and development forced-boundary16 passed; XML has zero failures/errors/skips. Focused unit suite16 passed across5files. Touched-scope ESLint and smoke TypeScript pass; diff check passes. Scopes overlap and are not full repository or hosted CI certification. Production build/ordinary browser runner use local Node26; final development and recovery runner use installed Node20.19.5, unchanged 30-second navigation gates, precompiled admin route and task-local16GB heap. Two prior Node26 development runs each had2 navigation failures/14passes: cold compilation and measured Next memory restart; original logs retained, not green credit. No Python source change; Bandit N/A. Shared installations/Postgres unchanged. Independent source review of immutable patch5cfa514a8f4f7f64b71ca767c7a301fc0ed15890f3c524e40ef5a5737eee122f is clear after actual response-wait RED1/GREEN1; required response assertions remain fail-closed. Original failed/setup/rejected diagnostics retained under /tmp/email_followup_*_20261001. Official CLI interactive editor removed orphaned summary markers without changing historical notes. Local source verified; publication/new PR merge not yet claimed.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
