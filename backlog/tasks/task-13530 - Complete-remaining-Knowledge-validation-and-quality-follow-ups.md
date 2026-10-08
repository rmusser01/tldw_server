---
id: TASK-13530
title: Complete remaining Knowledge validation and quality follow-ups
status: In Progress
labels:
- knowledge
- ux
- follow-up
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Burn down the six remaining items from the merged Knowledge workflow workstream on current dev, retain actual native/device/participant evidence, and reconcile TASK-13514 with merged PR3205. Reuse existing capture, ingestion, Notes, Research, Sync and test tooling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Final native capture, desktop assistive checks and available actual-device validation are completed or have exact external blockers recorded.
- [x] #2 External-web capture and explicit refresh preserve versions and original evidence with regression coverage.
- [x] #3 The SQLite startup race and scoped frontend test/lint debt are repaired or precisely qualified.
- [x] #4 TASK-13514 and closeout reports accurately distinguish merged work from remaining qualifications.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent closeout task. Existing TASK-13512 owns native/accessibility/participant validation; TASK-13514.1 owns SQLite startup. Baseline dev26ae4fd679. User approved burndown and foreground/VoiceOver checks. ADR assessment and stage plan follow investigation; no task claimed Done without fresh evidence.
Verified local SQLite transaction repair and all32 Notes fixture repairs;866 owning frontend tests pass, touched lint and production Bandit clean, client types pass. TASK-13514 now Done from fresh GitHub PR3205 merge/ancestry proof at e3c345b76f2d93527488b2015b9c3a2649c1f981. Current evidence/qualifications in Docs/Reviews/KNOWLEDGE_BURNDOWN_2026_10_07.md. CDP canonical capture/save/reopen verified, actual Ask exposed extension artifact React dispatcher failure under investigation; native context menu/spoken/device/participants still open. Public capture/refresh proposal and ADR066 await owner design approval. Publication remains open; do not mark the six-item burndown complete.
Final local repair/validation supersedes the React investigation: SQLite caller transactions preserved;32 Notes fixture cases repaired with866 passes; native WXT dedupe/matching18.3.1 pins verified by5 tests and all5 bundle receipts; one-line initial source-selection fix verified with224 QA tests. Actual CDP scoped local-model Ask returns24 November2026 with one mapped citation to canonical Note revision1, web fallback disabled and no page errors; source preview preserves exact UUID, full text, original URL/date. Available browser evidence and exact remaining native/spoken/Safari/iOS/human blockers are recorded in the report, satisfying parentAC1 without falsely closing TASK13512. Direct-panel chat analysis stream error and capture-origin wording stay qualified under13512. TASK13514 accurately Done from merged PR3205. Stage2 Complete;3 requires design approval;4 remains In Progress for external/native observations;5 publication/review remains open. All sanitized receipts now retained with report.
Final owned cleanup verified: Chrome closed through CDP; API/WebUI/mock/article/CDP ports19131/19132/19133/19135/19136 closed and recorded processes gone; existing model9099 remains HTTP200. Four owned dependency links, extension link forest/Jiti cache, generated client outputs and private fixture/browser profiles removed. Primary dependency targets unchanged; managed branch/worktree and sanitized report artifacts retained for review. Cleanup receipt retained with report.
Pre-commit gates and staged diff checks pass for the verified fix/report commit. Unpublished branch rebased cleanly onto current dev5ec8c7939f46d6baecf8767caf070cb4a2eb335c; intervening Admin readiness files have no overlap. Rebased Notes and QA source hashes match verified receipts. Proposal links updated to reachable rebased design commit746598afff328fd0585677003bd916b7471a106f. Prepare draft follow-up PR; new human Change summary and required statuses remain merge gates.
Published and attached draft PR3211 against dev: https://github.com/rmusser01/tldw_server/pull/3211 . Current verified base5ec8c7939f46d6baecf8767caf070cb4a2eb335c; no overlap with intervening Admin readiness. Final strict MkDocs build passes22.81s; normal hooks/diff checks pass and primary branch/head/status exactly preserved. Initial GitHub checks are queued/running, not all green. New human-owned Change summary remains required before ready/merge; ADR066/public-capture design approval and TASK13512 external qualifications remain separate. Do not mark this burndown complete merely because the verified repair is published.
Task6 current reconciliation (2026-10-08): Verified local explicit capture/refresh criterion with retained versions/original evidence and meaningful owner/failure/retry regressions; TASK-13530.1 and TASK-13531 locally Done. TASK-13514 stays Done on dev; SQLite13514.1/frontend13530.2 already Done merged PR3211. All exact external native/assistive/device/participant limits remain TASK-13512 In Progress. Three public acquisition probes all429; mocked extraction-response versus real canonical persistence/FTS/citation clearly qualified. Parent remains In Progress for controller review/publication coordination, not an unresolved capture implementation failure. Historical fullUI2155/1/2156exit1 and isolated same-code CSV passing rerun stay visible for whole-branch triage; no exact active timing owner established. Canonical evidence: Docs/Reviews/KNOWLEDGE_CAPTURE_REFRESH_2026_10_07.md and sanitized artifacts/KNOWLEDGE_CAPTURE_REFRESH_2026_10_07.json. Local product head63a3abd7ff0e3ece0e82bb72f5eb7fdbb6532fec; Task6 base same head. Independent Task6/whole-branch review, publication, new requester-written Change summary, seven updated-head required CI checks and merge remain controller gates. ADR required: yes; accepted Docs/ADR/066-explicit-web-capture-and-refresh-snapshots.md governs explicit capture/fresh refresh/current-head RAG, composing existing007/018/026/031/034/036/042/065 without rationale changes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verified local explicit capture/refresh criterion with retained versions/original evidence and meaningful owner/failure/retry regressions; TASK-13530.1 and TASK-13531 locally Done. TASK-13514 stays Done on dev; SQLite13514.1/frontend13530.2 already Done merged PR3211. All exact external native/assistive/device/participant limits remain TASK-13512 In Progress. Three public acquisition probes all429; mocked extraction-response versus real canonical persistence/FTS/citation clearly qualified. Parent remains In Progress for controller review/publication coordination, not an unresolved capture implementation failure. Historical fullUI2155/1/2156exit1 and isolated same-code CSV passing rerun stay visible for whole-branch triage; no exact active timing owner established. Canonical evidence: Docs/Reviews/KNOWLEDGE_CAPTURE_REFRESH_2026_10_07.md and sanitized artifacts/KNOWLEDGE_CAPTURE_REFRESH_2026_10_07.json. Local product head63a3abd7ff0e3ece0e82bb72f5eb7fdbb6532fec; Task6 base same head. Independent Task6/whole-branch review, publication, new requester-written Change summary, seven updated-head required CI checks and merge remain controller gates. ADR required: yes; accepted Docs/ADR/066-explicit-web-capture-and-refresh-snapshots.md governs explicit capture/fresh refresh/current-head RAG, composing existing007/018/026/031/034/036/042/065 without rationale changes.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
