---
id: TASK-13260
title: Run fresh single-user and multi-user UAT across core workflow loops
status: In Progress
created_date: 2026-09-15 01:27
labels:
- uat
- testing
- documentation
documentation:
- Docs/Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md
updated_date: 2026-09-15 05:07
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Exercise fresh-data single-user and multi-user installations in the current checkout and walk through the user-defined core workflow loops A, B, and C. Keep a running tracker of bugs, failures, UX issues, workarounds, evidence, and coverage gaps. Awaiting the user's A/B/C definitions; clean dependency installation is not certified by reused environments.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Exercise single-user fresh setup and agreed core workflow loops A, B, and C.
- [ ] #2 Exercise multi-user fresh setup, admin and ordinary accounts, and agreed core workflow loops A, B, and C.
- [x] #3 Record all observed product issues with reproducible steps, expected and actual behavior, and evidence.
- [x] #4 Verify account isolation and document environmental limitations and untested scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Prepare isolated configurations, SQLite databases, ports, and browser sessions; exercise fresh setup and independent ingestion/chat/auth paths; obtain A/B/C definitions and walk through each; consolidate evidence and keep the running tracker current.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Checkpoint: 15 product findings recorded in Docs/Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md (five P1, six P2, four P3), with six environment/tooling observations. Fresh isolated single-user and multi-user SQLite state initialized using existing dependencies; no clean dependency/Docker/Postgres certification. Single-user real local chat and reload persistence pass; source ingestion and media search pass. Admin plus Alice/Bob authenticate, and API note/media isolation and ordinary-user admin denial pass. Multi-user onboarding skip fails CSRF; Alice's Notes UI is blocked by /api/v1/health/live requiring system.logs. Knowledge QA fails provider validation after optional RAG setup was deferred. User definitions for A/B/C remain outstanding, so neither loop coverage nor overall UAT is complete. Selected screenshots, snapshots, and API results retained under output/playwright/fresh-install-2026-09-14; private credentials/raw logs excluded. UAT services and synthetic state retained on ports 18000/18080 and 18001/18081 for continuation; original 8000 and model 9099 untouched. Backlog CLI collision replaced original untracked 13259; semantic content restored through MCP from task history, with original timestamp values documented in its recovery note. This UAT uses explicit 13260. No product code modified; Bandit/application tests are not applicable to the documentation/evidence checkpoint.
Repair pass completed on codex/fresh-install-uat-fixes. All19 recorded product findings have verified repairs/dispositions; children13260.1-.4 Done. Final focused verification488frontend tests/248Python tests; touchedBandit0findings; ESLint0errors; same90baselineTypeScript diagnostics. Live ordinary Notes, unverified-admin diagnostics and UI user creation, cited Cedar QA with honest relevance, and all-security-filtered no-generation control verified. Existing Notes fixture and prompt-loader baseline failures documented. Full fresh-install workflow UAT has not been rerun; authoritative named frontend journeys/shared real-server workflows and fixture/provenance/no-skip gaps are in the tracker. Next full pass needs fresh single/multi profiles and explicit acceptance matrices; no invented A/B/C mapping. All runtime data remains isolated on18000/18001 and18080/18081; original8000/model9099 unchanged.
User continued into the full workflow UAT after repair commit68863b90b7. Testing fresh configuration, empty databases and new browser sessions in both modes using current installed dependencies. New isolated API ports18100/18101 and WebUI18180/18181; prior UAT and production retained. Parent runs single-user, independent agent runs multi-user; product revision frozen, all failures recorded before any further fixes. Cover named corejourneys plus shared save/review/restore workflows, strengthen provenance/prompt/no-skip checks, and record fixture limitations honestly.
2026-09-15 full fresh-configuration/data UAT completed on frozen product revision 68863b90b7 for both auth modes. Named frontend journeys and four shared workflows were attempted through normal UI with real llama.cpp. UAT-020 through UAT-045 record 26 new findings, including account metadata leakage, ordinary-user retrieval, tracked Chat, flashcard content/generation, ingestion false success, and UX problems. Blocked downstream steps are not passes. Exact Wikipedia URL attempted in both modes; Wikimedia robot-policy response was incorrectly stored as successful article content. Synthetic public sources provide separate positive controls. Single-user derived note link, card scheduling/reload and media delete/restore pass mechanically, with recorded content/UX failures. Multi-user backend foreign read/write checks pass; recent-note metadata privacy fails. Finishing evidence/secret review and documentation commit. No product edits during this rerun; fresh dependency installation and Docker/Postgres remain untested.
Final UAT evidence review passed: 49 retained evidence files plus hash manifest; all JSON parsed, runtime-secret/token scans passed, finding IDs and severity counts reconciled, no Pending cells in rerun matrix, git diff whitespace check passed. Independent review identified and closed the dedicated multi-user Review gap through visible Cedar selection after its second analysis. Explicitly retained the blocked fresh-run confidential-content RAG policy control and exact Wikipedia denial. Removed only the completed UAT execution plan; tracker preserves stage outcomes. Product acceptance remains blocked by new findings, so task remains In Progress; no additional repair or full rerun was silently claimed.
Final artifact correction: discarded one empty prompt-confirmation snapshot and explicitly documented that live observation; normalized only trailing blank lines in text captures. Final inventory is 48 evidence files plus manifest. Repeated credential/token scan, JSON validation, ID/severity/matrix checks and staged diff whitespace check all pass. No acceptance status was upgraded because of evidence cleanup.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Initial UAT findings001-019 were repaired and committed through 68863b90b7, with targeted verification recorded. The subsequent fresh single/multi workflow run has completed with 26 newly recorded findings020-045 (6P1,14P2,6P3). Every core/shared workflow step has an outcome or dependency block in the tracker. Main open priorities: recent-note account metadata privacy, ordinary-user source retrieval, tracked multi-user Chat, grounded Flashcards, hidden ingest analysis failure, and character/context mismatch. Notes linking, card scheduling, explicit media analyses, single-user restore and backend cross-account denials have passing controls with limitations. Existing dependencies were reused; no clean-machine or Docker/Postgres sign-off. Evidence stored under output/playwright/full-workflow-uat-2026-09-15. New defects remain open; acceptance is not complete and this is not release sign-off.
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
