---
id: TASK-13701
title: Restore settings fixture source and native storage contracts
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current model prompt and conversation numeric fixtures omit the owned fresh/accepted source, actor scene export and real native browser storage required by settings readers. Restore those declared dependencies and retain all save, cancellation, recovery and numeric assertions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both complete settings fixture suites pass with original49prompt and16numeric cases retained
- [x] #2 Actual source/account retirement guards and native unsupported-control refusals remain intact
- [x] #3 Relevant settings regressions, normal client checks, independent review and hooks pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required:no; test-dependency repair only. Representative nativeRED3failed62skipped; modeldialogtargetnull/noConversationtab, numericdurable readunavailable. HighestID13638 acrossremote/open/local; use13639. PlanDocs/Plans/IMPLEMENTATION_PLAN_chat_settings_fixture_contracts_13639.md. PR https://github.com/rmusser01/tldw_server/pull/3229.
Initial native65:61passed4failed. Actual new draft after authretirement; added noSave/noActor write/noClose/nooldprompt before freshsource. Isolated unchanged reset-default5.059s exceeds5s; reset-template4.829s passes. Scoped cached-remount10s matches adjacent two-mount tests, no globaltimeout change. Followup5passed44skipped. Finalnative qualification running; bounded independent review noCritical/Important. Actor durableSave remains mocked limitation. Review Docs/Reviews/CHAT_SETTINGS_FIXTURE_CONTRACTS_13639_2026_10_13.md. Publicf5d nine CI failures, notmerge-ready.
Finalnative202passed5suites; WebUI249passed4suites includingactual Actor read89/ownership95. Both normalclient sourcecompilers exit0; excludesdedicatedtest-file typechecking. Readonly boundedreview CLEAR, staleSave pre-renewal noClose/nooldprompt assertionsadded. Bandit N/A TypeScript tests. Finalnormalhooks pending; wholePR PostgreSQL/CI/frontend/H2 remainopen.
Normal exact4file hooks allpass; no bypass. Completed ownplan retired/private/tmp. Independent boundedreview clear; otherwholePR failures and finalreview/rebase/merge remainpending.
2026-10-15: Renumbered this PR's historical TASK-13639 to TASK-13701 after latest dev introduced an unrelated task with the old ID. The requester explicitly approved the proposed four manual task-ID renames and four report-heading updates. Backlog-py has no existing-task renumber operation; the approved exception changes filename/frontmatter ID, followed by this ordinary backlog-py note. Historical report filenames and original verification content remain unchanged. ADR assessment: no new architecture decision; metadata collision repair under ADR-059.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Restored physical native storage and actual fresh source/controller account authority in both settings suites. Native202passed; WebUI249passed with actual Actor read and ownership services; both normalclient sourcecompilers0; exactfilehooks pass. Retirement proves noSave/noActor write/noClose/nooldprompt before new owned source. Scoped10s real cachedremount aligns adjacent two-mount tests, no globalbudget change. Bandit N/A TS tests; actual Actor Save remains a mocked component boundary, no installedextension/AT or wholePR readinessclaim.
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
