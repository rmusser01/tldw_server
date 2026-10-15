---
id: TASK-13710
title: Fix post-merge Chat UX reassessment findings across both clients
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement all six findings and remaining improvements from TASK-13709 on latest dev in a new PR. User approved character-first roleplay setup and filtered paginated compact character history, root-cause repairs, independent review and merge after checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Disconnected and unconfigured Chat uses one accurate connection-first recovery path without weakening ownership or draft guards
- [x] #2 Both extension starter prompts update controlled draft state and survive interaction
- [x] #3 Compact tracked sessions use server character filtering and pagination with honest loading stale error and empty states and disambiguating metadata
- [x] #4 Roleplay setup leads with Character and provides optional details and accessible current/proposed preview labels
- [x] #5 Task language customization disclosure and active legacy character-picker error recovery address remaining report improvements
- [x] #6 Regression tests production verification independent review and new PR against latest remote dev qualify the change before merge
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design approved by user in direct reply to TASK-13709 review: recommended character-first and filtered paginated history. ADR required: no; local shared-component state, recovery and presentation repairs reuse current APIs and trust boundaries. Governing ownership/persistence rules are unchanged. Existing isolated managed remediation worktree reused; new branch codex/chat-ux-followup-20261015 begins origin/dev 2e48dc91848887dcb74ebe39d6ff877fed4d64bd. Plan: Docs/Design/IMPLEMENTATION_PLAN_chat_ux_followup_13710.md. Preserve original nine conversations/28 messages and unrelated primary-checkout work.
Stage1 setup/readiness and controlled starter group47 passes after observed regression failures. History21 passes and scoped independent reviewer clears final-page boundary fix. Customization test reassessment after three verification passes: old group-containment assertions were still checking relocated Tools/Compare in primary groups; inspected complete four-layout test, retained all controls/focus/visibility assertions and explicitly opens Advanced. No test timeouts/retries changed. Stage2/3 integration and final qualification remain ongoing.
Stages 1–3 implemented. Shared starter fix proven red on old source (both suggestions lose text after controls) and green through actual production message effect and submit callback: 34 tests pass, exactly one send each. Independent follow-up reviewer cleared duplicate setup guidance and stronger starter evidence at source/unit level; installed-browser confirmation pending. Fresh frozen-lock Bun production WebUI build, WebUI typecheck, strict shared-boundary types, Chrome production extension build and compile all pass. Touched ESLint31 files has0 errors and163 baseline warnings, unchanged. Shell qualification:86/86 adjacent tests plus corrected64/64 shell tests pass; old labels/navigation expectations updated to intended behavior. Original9 chats/28 rows remain deep-equal to immutable baseline. No Python changed: Bandit not applicable to source scope.
Qualification report: Docs/Reviews/CHAT_UX_FOLLOWUP_13710_QUALIFICATION_2026_10_15.md. WebUI touched16 suites362 tests pass. Installed production extension confirms both real starter/control cycles and all6 authoritative older page2 rows (147 filtered character sessions), Previous navigation and responsive roleplay. Settled WebUI drawer390px wide at viewport390. Source provenance4318 production app files matches;969 emitted JSON manifests reference1130 existing JS/CSS assets. Frozen lock unchanged. Independent follow-up source review has no actionable findings. Physical AT, permission-grant UI and Firefox/Edge runtime not checked. New PR/CI/human Change summary remain before merge.
New PR against remote dev: https://github.com/rmusser01/tldw_server/pull/3248. Exact source commit98eda36cfa19705c9ff8308c0c45905e6e3ac5ea independently cleared with no actionable technical findings. Normal pre-commit checks explicitly passed on all41 files (no pre-commit hook installed); configured local fast CI passed under project Python3.12. Frozen source/build proof and after-walkthrough original9/28 preservation pass. Required exact-head GitHub CI and requester-written Change summary are pending before merge. Merge queue variable unset; no auto-merge armed.
Requester supplied the human-written Change summary for PR3248 in this chat on2026-10-15; added verbatim under Change summary in the PR body. Human ownership gate is satisfied. Latest remote dev remains2e48dc91848887dcb74ebe39d6ff877fed4d64bd; branch contains it with no missing dev commits. Four required gates pass on44d4d3581451d8475d076f79e07e3eca4a9b30a2; remaining exact-head CI still running. No app code changed after the independent review.
CI frontend shard4 exposed stale fixtures: WebLayout tour tests mock useConnectionState without useIsConnected; shared Layout tour cases omit a connected setup state; research-status integration globally mocks react-query but omits useQueryClient, also failing on base. CI ratchet rejects the mixed changed-identity replay; do not alter its pins or policy. Plan: reproduce package-owned failures, set connected tour prerequisites, complete query mock using existing adjacent real-QueryClient partial-mock pattern, preserve assertions, and rerun affected package suites plus independent scoped review before repush.
CI fixture follow-up: full WebLayout47 tests pass; native shared Layout/research43 tests pass after reproduced22 failures. Four corrected 320/390px cockpit smoke cases pass against unchanged fresh production app, preserving geometry/overflow/focus assertions. Independent CI diagnosis found shard1 has2 missed Buddy disclosure expectations (338 other cases pass); update that integration contract to verify primary Role-play visibility and hidden-then-visible Buddy through Advanced. CI ratchet/workflow pins and assertions remain intact; no production edits are needed.
Final CI fixture repairs independently reviewed with no actionable findings and no weakened assertions. Fresh frozen-dependency deterministic CI-config run: native UI257 tests plus WebUI47 tests pass,304 total; fixture hashes unchanged during the run. Both Buddy cases reproduced red and pass green with primary Role-play visibility plus hidden-before/visible-after Advanced assertions. All4 corrected responsive cockpit smoke cases pass. Combined final PR lint36 TS/TSX files:0 errors,198 warnings identical to base after line-shift normalization. Production4318-file hash proof and1130 emitted references still match. Human Change summary is verified verbatim in PR3248. Local implementation/qualification is complete; fresh exact-head GitHub checks remain the enforced integration gate before the authorized merge.
Implementation and local qualification are complete, following existing repository tasks that close verified repair units while whole-PR CI/merge remains enforced separately. All acceptance and local definition-of-done items are checked. Independent scoped CI-repair review is clear and all304 deterministic CI-config cases pass. The owned implementation plan is retired with its history preserved; integration continues in this chat on PR3248, which must pass all seven statuses on its final head before merge. No claim that final remote CI or merge has completed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed all six post-merge Chat findings and agreed improvements using current shared connection, controlled-form, owner-scoped history and staged roleplay patterns. Added regression and CI fixture coverage without changing CI pins, disabling tests or weakening assertions. Independent reviews clear the source and fixture repairs. Fresh production builds/typechecks,362 shared WebUI regressions,304 deterministic CI-config fixture/integration cases and4 responsive smoke cases pass; protected9 conversations/28 rows remain unchanged. PR3248 targets latest remote dev and includes the requester-written Change summary. Merge is authorized only after the final exact-head required checks pass.
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
