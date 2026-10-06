---
id: TASK-13506
title: Distinguish chat history mode and Character Library read failures
status: Done
labels:
- chat
- ux
- webui
- extension
dependencies:
- TASK-13505
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue approved stages1/2 with the browser-observed false status issues: a failed Character Library read is converted into successful empty results, and the shared chat header calls history eligibility Saved beside pending recovery. Reuse the existing query error/Retry renderer and shared history-mode copy; preserve existing fallback compatibility and notification behavior. No additional persistence store, transport change, backend change or modification of a pending PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The shared header uses the existing history-mode label/help and cannot claim messages were saved from the temporary flag alone; Temporary describes new history eligibility and retained local drafts truthfully.
- [x] #2 Main Character Library read failures reach existing error/Retry UI rather than successful empty onboarding; genuine successful empty data and supported legacy fallback remain distinct; a successful explicit Retry returns ready results.
- [x] #3 Red/green actual-query and header checks, targeted renderer suites, relevant client/lint/diff gates and both independent reviews are recorded with current dev and native browser limits.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Official search Character Library error / Saved header returned no bounded duplicate. Open PR inventory /private/tmp/chat-status-open-pr-paths.json has no main query ownership; PR3147 touches ChatHeader with a separate acknowledged-write tracking proposal. It was read as reference only; no attachment/remote update/integration requested. Its persistence-store approach is not added here. Current dev94854ca3db is integrated in292ea526d1. Browser proof shows 0 characters/No characters yet after failed query reads, and Saved mode beside acknowledged-but-unqualified creation recovery. Roots: useCharacterData final catch returns buildEmptyCharacterQueryResponse instead of rejecting; ChatHeader derives Saved solely from temporaryChat. Existing usePersistenceMode helpers already provide honest labels. No Python implementation; Bandit N/A. Parent program still owns successful native browser generation, physical storage, responsive/assistive/native-extension and unsupported backend enhancement gates.
Verified red4/21 before source fix; green55 in5 actual-query/header/renderer/mode suites plus4 targeted current Manager legacy/error cases. Both authorized independent source reviews clear. Tests use IO doubles, mocked AntD/fallback translation and disabled query retries; production retains its existing one-retry policy. Final shared WebUI types, extension entrypoint/config compile, no-new touched lint and diff checks pass. No locale changes required because existing mode strings are reused. Live IAB header shows History on beside retained acknowledged creation; Library currently displays all4 owned cards. Post-change failed browser reads were not imposed; transport cause/native generation/physical storage/assistive/packaged extension remain parent gates. Dev94854ca3db remains current, integrated292ea526d1; follows Quick Chat checkpointfb0cf6ca79. Unique completed status plan removed per project guidance. No Python implementation; Bandit N/A.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reused shared history-mode copy in the header so eligibility no longer implies saved messages. Main Character Library failures now reject into existing error/Retry feedback while genuine empty results and supported legacy fallback remain distinct. The implementation adds4 lines and removes27 in2 source files, with no new persistence/error framework or transport behavior.59 relevant checks, both client gates, no-new lint/diff and two independent reviews pass against current dev. Browser header wording and current successful4-card loading are verified; exact test/native limits remain documented in the master design. Broader all-ten program stays open.
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
