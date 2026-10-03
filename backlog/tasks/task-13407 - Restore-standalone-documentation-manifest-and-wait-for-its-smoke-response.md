---
id: TASK-13407
title: Restore standalone documentation manifest and wait for its smoke response
status: In Progress
assignee: []
created_date: 2026-10-01 08:10
updated_date: 2026-10-03 01:44
labels: []
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/pull/3023
- TASK-13377.9
- https://github.com/rmusser01/tldw_server/pull/3077
documentation:
- Docs/Operations/Email_Real_PST_Validation_2026-09-26.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fresh PR3023 browser diagnostics captured a late GET /api/documentation/manifest HTTP500 in the native advanced standalone bundle. apps/tldw-frontend/lib/documentation.ts resolves repository sources only from cwd or ../..; Next standalone changes cwd under .next/standalone/apps/tldw-frontend and the existing CI staging copies public/static without the Docs sources. The all-pages smoke assertion can finish before this request completes. This inherited defect is not introduced or suppressed by TASK13377.9, remains outside its allowlist, and is a separate frontend repair. Independently reviewed source diagnosis is supported; reproduce the complete deployment/source layout before selecting a minimal fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduce the manifest HTTP500 from the supported standalone bundle with the intended documentation sources present, then verify a minimal source/runtime repair returns the expected manifest.
- [x] #2 Add a deterministic smoke check that waits for the documentation manifest response and content, so a late HTTP500 fails instead of arriving after the assertion.
- [x] #3 Preserve path-containment and document-source validation; do not allowlist HTTP500, broaden suppression, disable guards or count metadata smoke success as documentation acceptance.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Official CLI duplicate search 'documentation manifest' completed: /tmp/email_pr3023_ux_docs_followup_search_20261001.txt; existing returned standalone entries concern MCP documentation and do not cover this WebUI unit. Explicit ID13407 follows verified TASK13406. CLI creation is unavailable for this task due to retained repeated stack-overflow diagnostics, so official MCP is used. Raw late-response evidence /tmp/email_pr3023_ux_raw_diagnostics_20261001.jsonl. Immutable test-maintenance patch d001ed8bde3cc2aff39c1a4abe4faa59f68961c282241cae14f19f998740966f reviewed independently; reviewer says this separate inherited defect does not block that patch, but prohibits any zero-raw-error or functional documentation acceptance claim. No human assignee is invented.

2026-10-01: Human authorized completing the documentation runtime and deterministic smoke repair. Isolated follow-up branch starts at dev d81c13fddd1dac1948b30401af0388932a0af8f2. Official MCP read calls did not return; official CLI fallback used. Reproduce the deployed standalone layout, preserve containment and source validation, then verify actual manifest/content. Plan: Docs/Plans/IMPLEMENTATION_PLAN_email_followup_burndown_20261001.md.

TASK13407: original nested standalone with intended Docs/Published present reproduced HTTP500; nested-cwd unit regression RED and actual deployed error retained. Nearest ancestor lookup and Next tracing include real published docs. Browser gate verifies the manifest contains AuthNZ and waits for the selected document content and rendering; the separate direct HTTP probe verifies real AuthNZ guide content; delayed manifest500/content500 regressions and page-close pending-wait regression pass after real REDs. Containment/extensions/API source validation unchanged. Existing extension documentation source layout is unchanged; no extension-docs acceptance claim. README deployment instructions updated. Local verification completed 2026-10-02 UTC (October 1 local): final production build/token sync/bundle budget passed; relocated standalone includes 324 published server documents, actual manifest/content HTTP200 and traversal/unsupported-source HTTP400. Final strict ordinary smoke105 passed, classifier7 passed and development forced-boundary16 passed; XML has zero failures/errors/skips. Focused unit suite16 passed across5files. Touched-scope ESLint and smoke TypeScript pass; diff check passes. Scopes overlap and are not full repository or hosted CI certification. Production build/ordinary browser runner use local Node26; final development and recovery runner use installed Node20.19.5, unchanged 30-second navigation gates, precompiled admin route and task-local16GB heap. Two prior Node26 development runs each had2 navigation failures/14passes: cold compilation and measured Next memory restart; original logs retained, not green credit. No Python source change; Bandit N/A. Shared installations/Postgres unchanged. Independent source review of immutable patch5cfa514a8f4f7f64b71ca767c7a301fc0ed15890f3c524e40ef5a5737eee122f is clear after actual response-wait RED1/GREEN1; required response assertions remain fail-closed. Original failed/setup/rejected diagnostics retained under /tmp/email_followup_*_20261001. Official CLI interactive editor removed orphaned summary markers without changing historical notes. Local source verified; publication/new PR merge not yet claimed.

Published reviewed fixes as d4693fefe37bc75d9d9b54026c77c12de3200c76 on codex/email-followup-closeout-20261001; normal push verified. Draft PR3077 contains the implementation and this tracking closeout. Repair acceptance is locally complete; this new PR is not merged and its human-written Change summary and hosted CI are pending. Source receipt /tmp/email_followup_closeout_receipt_20261001.json binds ten exact source paths and frozen verification artifacts. Browser coverage checks the AuthNZ manifest entry plus selected-document content/render; the real AuthNZ guide HTTP probe is separate. Task-owned servers stopped after identity verification; shared installations and unrelated resources preserved. Completed owned plan retained at /tmp/email_followup_plan_completed_20261001.md after all stages are complete.

Merge-gate audit correction: implementation and local verification are complete, but the repository-wide Definition of Done requires a human-written Change summary for this new AI-authored PR. PR3077 is draft and awaits that human input plus hosted CI. Restored In Progress with this explicit remaining DoD item; no source changes or repeat tests. The prior completed-plan snapshot is retained as /tmp/email_followup_plan_pre_gate_diagnostic_20261001.md, and the owned plan remains Stage3 In Progress until this requirement is satisfied. TASK13178 remains Done because its implementation PR2887 was already merged.
2026-10-02: Requester supplied PR3077 Change summary directly in this chat; published verbatim. Human gate satisfied. Refreshing reviewed branch onto dev 86e287fee7bfa1a1588639232e35db3666851ded before hosted checks and actual merge. Earlier receipts retain original source/transitive/dependency bindings. Task remains In Progress pending refreshed verification and merge.
Current-dev refresh verified 2026-10-02: fresh Node20.19.5 standalone build moved outside checkout returns manifest324serverdocs/AuthNZcontent200 and traversal/unsupportedsource400. Strict browser106passes include actual manifest AuthNZ entry, selected-document response/content/render, delayed500 negativecases and page-close handling; direct AuthNZ probe is separate. Unit84/8files/build/lint/types/diff pass. Native backend Python3.11.13/FastAPI0.142.1/Pydantic2.11.7 normal TEST_MODE0, isolated task data; startup guard rejection for outside-pytest TEST_MODE1 retained as setup diagnostic. Original certificates keep original inputs and environments. Ten source hashes/29frozen artifact hashes in /tmp/email_followup_devrefresh_receipt_20261002.json; independently clear source patch80ab627d9c43b712adacc18850159417e549ded2cabd8eda9341f65fec645c8b. Task-owned servers verified stopped; shared installations unchanged. Human Change summary supplied verbatim; source publication/hosted CI/actualmerge pending, task In Progress.
PR3077 published ready at31f8e8e8164ca01ffe01f3465507574af9f353d2, requester Change summary verified verbatim, currentdev86e287. Qodo review5964101702 explicitly bound to31f8 reports two rule issues (sync filesystem setup and combined validation test) and two docs packaging/root-discovery findings. Read-only inspection confirms WebUI Docker builder copies no Docs roots; Docs/User_Documentation is absent/optional, actual required source is Docs/Published. Root discovery climbs past the supported standalone bundle. Minimal follow-up under this task: async focused filesystem regressions, fail-closed discovery limited to cwd through its two supported parent levels, required published Docs builder copy, actual container manifest/content verification. Existing source/transitive certificates remain historically bound; current CI still active; no reruns/cancellations/merge.
Qodo four findings repaired with immutable source patch67edd79bcc9dbca33ef99205b8e105a58f0201ce756b65adb727e02280f7ffbb independently clear. Async focused fixture cases retain path/file-type assertions; actual unchanged discovery RED2/7pass served unrelated host docs, repaired complete library GREEN9 rejects that fallback. Actual baseline Dockerfile.webui production image returned manifestHTTP500 with /app/Docs/Published absent. Required Published source copied beforeNextbuild; repaired actual production image (Linux/Node20.20.2, UID10002, cwd/app/apps/tldw-frontend) serves324published-serverdocs and AuthNZ guideHTTP200, traversal/filetype/sourceHTTP400. Frozen logs/metadata/statuses/source/artifact hashes in /tmp/email_followup_qodo_receipt_20261003.json and Docker revalidation receipt; exactownedcontainers stopped, shared installations/unowned resources untouched. ESLint0/diffcheck pass, new test formatting pass; inherited library full-file formatting failure and baseline retained. No Python delta, Bandit N/A. Earlier31f8 84unit/106browser and all original certificates retain their exact earlier source/transitive/dependency bindings and were not rerun for this four-file follow-up. Human summary remains verbatim. Normal publication, new-head Qodo/hosted CI and actualmerge pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Standalone documentation restored in moved native and actual Docker production builds. Required Published docs supplied before tracing; lookup limited to supported cwd/two-parent layout and fails closed without bundle docs. Qodo4source findings addressed with native RED2/GREEN9 and actual DockerRED500/GREEN324docs/AuthNZ200 plus path/filetype/source400; immutable patch independently clear. Earlier certificates retain original bindings. Human Change summary supplied verbatim; refreshed publication/hostedCI/actualmerge pending, task InProgress. No Python delta; BanditN/A.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 New AI-authored PR has a requester-written Change summary explaining what changed and why these implementation choices were made.
<!-- DOD:END -->
