---
id: TASK-13686
title: Reconcile released admin migration with native chat quota upgrade
status: Done
labels:
- chat
- integration
- migrations
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR3229 must integrate remote dev e3987dda after PR3218. Released SQLite admin step101 conflicts with the unmerged native quota step101. Preserve released version identity and prior preview data through native step102, regenerate the merged API contract, and qualify shared integration before exact-head CI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh, version100, released admin101 and prior preview native101 upgrades preserve account identity and quota charges while ensuring both admin indexes and native quota schema.
- [x] #2 Canonical merged fingerprint and path union pass drift checks; connected layout/chat qualification, Bandit and normal hooks pass.
- [x] #3 Two independent reviewers clear the final integration before required exact-head statuses and merge.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent TASK13458; implementation plan Docs/Design/IMPLEMENTATION_PLAN_chat_final_ci_roots_13685.md. ADR assessment: governing ADR049/050 ownership/settings and ADR059 task editing; existing versioned AuthNZ migrations remain governing, no new durable architecture rule or ADR. Human merge authorization retained. Physical extension/AT/touch/native zoom/participant acceptance remain unrun.
Qualification: SQLite fresh/prior100/admin101/preview101 and repeated charged bootstrap plus PG; preview-only RED established missing indexes; full native/admin/backfill62pass. Shared layout/header/Admin-provider7fullsuites76pass each native/WebUI/extension; connectedtypes2roots0errors. Corrected canonicalAPI2131paths3301schemas driftchecks pass; blank config projection excluded. Buildh9z4J2Xz0llyzyJmvHbIO97sourcehashes969manifests1131staticrefs0missing;23realservercockpit pass113s, original9conversations28rows andcompact4rows unchanged. Bandit67vs59devfiles14identicalfindings0introduced0errors. Independent reviewers clear source/API integration. Upstream Admin cache concern trackedseparatelyTASK13687; no chatcache/ownership impact.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Preserved released admin101 and reconciled native quota through idempotent102, including earlier preview101 ledger/index upgrades. All62backend tests,76shared UI cases ineachthreeprofiles,2type roots,canonicalAPI2131/3301 driftchecks,97sourcehash productionbuild,23livecockpit and unchangeddevStrictMode26requestratchet pass. Original9conversations28rows/compact4rows unchanged. Bandit14baseline0introduced0errors; normal10filehooks pass. Two independent source reviewers clearintegration. All7final-head GitHub statuses/authorizedmerge remain separate landing gates underTASK13458; upstream Admin concern trackedTASK13687. Physical extension/AT/touch/nativezoom/participantacceptance unrun.
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
