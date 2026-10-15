---
id: TASK-13663
title: Migrate prompt and cited-source journeys to original server-owned completion
  receipts
status: Done
labels:
- chat
- testing
- ux
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two critical ordinary journeys still wait for browser message append POSTs and save_to_db false. Current shared native completion saves on server. Observe the original streamed admission/result, reject duplicate browser writes, and preserve system/source grounding, owner/selection binding, canonical pair and cold-reload checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both journeys consume original native completion admission/result and assert one server-owned completion with no browser message append.
- [x] #2 Exact prompt/source content, owner/digest/revision, canonical parent/IDs, bookmark and cold equality assertions remain.
- [x] #3 Receipt validation tests and actual owned native qualification, static/source review and final critical runner limits are recorded before closure.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR3229 critical job115425762846 cases8/9:90s awaitassistant browserPOST; source tests stillassertsave_to_dbfalse. ProductionTASK13573 server-settlement is qualified and unchanged. Reuseexisting captureStreamedResponseBody and character journey receipt checks; no genericGET inference of acknowledgement, no replay/extra appends. ADR required:no, fixture contract migration underexistingADR049/068 native admission; no new product protocol. Task13662 navigation remainsInProgress/fullcriticalpending.
Plan Docs/Plans/IMPLEMENTATION_PLAN_chat_native_journey_receipts_13663.md; review Docs/Reviews/CHAT_NATIVE_JOURNEY_RECEIPTS_13663_2026_10_13.md. Independent bounded review resolved listener lifetime, repeated terminal metadata, upstream pre-save stop, and raw-versus-rendered answer comparison. Targeted discovery2/2 and lint0. Actual headless native attempt1 reproduced missing final ID in prompt upstreamstop; ingest failed model-menu visibility before completion. Corrected2case runner active; no full qualification claim. Bandit N/A TS-only.
Actual final headless native Playwright2/2 pass0skip51.2s against owned backend19767 via same-origin productionpreview19769 and repositoryci-journeys downstream19768. Earlier legitimateKnowledgeQA writes excluded by attaching observer at ordinaryChatstep start; stays throughcoldreload. Complete-source wait preservesexactgrounding. PromptandIngest receipts/canonicalpairs/bookmarks/coldfullresponseequality executed. Final scopedlint0, boundedindependentreviewclear. No product/type/buildclaim; Bandit N/A TypeScriptfixtures. Fullcritical/cockpit/master and finalwholePRreview/rebase/requiredCI/mergeremainpending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated two ordinary journey fixtures to original server-owned completion receipts while preserving exact prompt/source, ownership, canonical parents/IDs, bookmark and cold reload checks. Successful terminal metadata must agree; one completion and zero ordinaryChat browser appends are verified. Actual2cases pass0skip51.2s; lint and bounded review clear. No wholePR readiness claim.
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
