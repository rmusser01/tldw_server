---
id: TASK-13453
title: Implement Knowledge UX audit repairs and enhancements
status: Done
labels:
- knowledge
- ux
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the user-approved K01-K15 repairs and five enhancement themes from Docs/Reviews/KNOWLEDGE_NNG_UX_REVIEW_2026_10_04.md. Defaults approved: ask added items after ingest; continue in Research Workspace with sources attached. Coordinate four reviewable implementation units and browser verification on latest dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All K01-K15 have implemented fixes and regression evidence
- [x] #2 Named source-set handoffs, readiness summaries, outcome recipes, saved-output review loop and extension parity are implemented
- [x] #3 Research continuation preserves source identity, excerpts, trust and inspectable attachments
- [x] #4 Both WebUI and native extension are verified; checks, review, security scope and owned-service cleanup recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
User approved the audited solutions and both default choices on 2026-10-04. Work happens in the attached isolated checkout; original dirty checkout is untouched. ADR check: no new ownership or public-API rule; ADR-007 canonical ResearchWorkspace, ADR-008 split persistence and source lineage, and ADR-053 mixed-source retrieval govern. Reuse existing ingestion for non-media evidence snapshots and existing shared UI on both surfaces. Reassess ADR scope only if those existing contracts cannot express a required repair.
Approved design: Docs/Design/2026-10-04-knowledge-ux-remediation.md. Four-unit execution plan: IMPLEMENTATION_PLAN_knowledge_ux_remediation_20261004.md. All implementation is based on 75ab224081bf140ef52017c1a9b0a04f6878d488 plus the committed audit. Sequential implementation and independent task reviews; no new packages or public API.
Final latest-dev refresh encountered upstream active TASK-13452 for cached chat settings, colliding with this already-completed historical Knowledge audit. Archived only our completed audit through official backlog CLI before refresh; its full record and Docs/Reviews/KNOWLEDGE_NNG_UX_REVIEW_2026_10_04.md remain retained. Implementation tracking stays TASK-13453 and four children; upstream task is untouched.
Final controller acceptance on latest dev025627214c3aeda1b2e9af5c6a9f85c636a2ec02/product03f4e0627b: 123 suites/1741tests, both officialtypes and devbuilds pass; shared352/0added. Four task reviews plus final scoped fixes and native trust scoped review PASS/APPROVED. All actual workflows and native exact captured-note/consistenttrust proof pass; keyboard/narrow geometry and current canonical Research restore verified. All4owned ports closed, browsercontexts closed, only owned generatedoutputs/dependencylinks/plan/scratch removed. Fifteen issues, five enhancement themes, walkthrough, limitations,15screenshots,safe workflow-proof and all17chronological rulings preserved in Docs/Reviews/KNOWLEDGE_UX_REMEDIATION_2026_10_04.md and its assets/reviews/verification.md. Branch/worktree kept local; no push/merge/publish.
Final explicit-file documentation/Backlog hooks PASS exit0; git diff --check PASS; no owned browser-profile processes remain. All final guide local links,15screenshots,17chronological rulings and runtime-credential-free publishable text/manifest verified. Final documentation commit retains reports/proof and removes only completed owned plan.
User requested PR creation against dev after completed verification. Published codex/knowledge-ux-remediation-20261004 and created draft PR https://github.com/rmusser01/tldw_server/pull/3196 targeting dev. Draft reflects repository human-written Change summary merge gate; actual implementation, rationale, test/build/review evidence and limits are in the description. No merge performed. PR attached to current Codex task; worktree retained.
Reopened for PR #3196 delivery: rebased all 21 commits onto dev 5775d3fbbe3ed4b38a57f9a4b72775e434d06685; git range-diff confirms every patch unchanged. User explicitly authorized merge after checks without Qodo (workspace billing block). Merge queue variable is unset, so use manual strict-head required gates and merge commit. Investigating prior E2E Critical failure and refreshing verification before merge.
Follow-up CI root cause: generic combined file-input locator selected the newly added reattach input before qi-file-input, so both real journeys remained at Configure 0 items. Added a real-browser helper regression (same filename assertion RED; GREEN 1 passed) and changed the shared helper to its existing upload test ID. Independent scoped review approved with zero actionable findings. Fresh client type checks passed; canonical clipper covering API44 passed; Bandit production169LOC zero findings/errors; explicit-file hooks passed. Broad127 suites initially1908/1910 with default5-second CurrentChatModelSettings timeouts; isolated default runs reproduced timing/cleanup cascades. Reassessed after three runs: frontend-required explicitly uses maxWorkers1 and testTimeout15000; validating under that existing CI contract, without changing application/tests timeout settings.
CI-compatible cache verification passed all30 existing prompt-settings tests, including the four remount scenarios. Rebase/repair implementation and local required validation complete; all original patches unchanged, domain regressions GREEN, API44/types/Bandit/hooks passed. Full127-suite follow-up continues alongside remote required gates; PR #3196 merge remains pending strict current-head/current-dev CI. No Qodo findings were available; requester explicitly waived that unavailable review.
Final complete affected-suite refresh on rebased product: 127 suites / 1910 tests PASS in532.89s using the existing frontend-required maxWorkers1/testTimeout15000 contract. This resolves the earlier default5-second cache timeouts; no application or committed timeout settings changed. Exact log /private/tmp/knowledge-ux-rebase-tests-ci-settings.log. Fresh dev recheck remains5775d3fb. Required remote license gate passed; remaining checks are queued for hosted runners. Record final local evidence now before any required product gates start.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed K01-K15 and all five approved enhancement themes on latestdev025627, including actual captured-note continuation, canonical evidence/draft/save persistence, eligible retries, accessible compact interaction and honest citation qualification. Final product03f4e0627b passed123 suites/1741 tests,44 canonical API tests, officialtypes/devbuilds, touchedPython Bandit169LOC0findings and independent scoped reviews. Real workflows, scanner/browser limits and all17decisions are retained in the final review guide. Owned services/build artifacts cleaned; local branch/worktree retained for review. Final explicit-file hooks and documentation commit complete the tracking record.
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
