---
id: TASK-13537
title: Retain owned Temporary chat transcripts across reloads
status: Done
labels:
- chat-ux
- persistence
- webui
- extension
- temporary
dependencies:
- TASK-13457
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A completed native Temporary conversation disappears after reload. Implement the accepted D7 Q3 policy using the existing full-page session and owner-bound sidepanel snapshots: retain messages on-device until closed or cleared, with no server-history creation or automatic resend.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Full-page WebUI and extension persist and cold restore exact Temporary history/messages with the matching verified fresh draft handle; legitimate null/temp sentinel is admitted while saved IDs are refused.
- [x] #2 Sidepanel durable read/write retains only a matching owned fresh Temporary transcript and preserves existing no-queue/no-active-command rules.
- [x] #3 Temporary transcript changes invalidate obsolete saves; wrong-owner, account, New and Clear/Close boundaries cannot restore stale or foreign rows.
- [x] #4 Valid Temporary sessions remain restorable beyond the ordinary bookmark expiration until cleared or closed; ordinary sessions retain their existing expiration.
- [x] #5 Native completed-turn plus next-draft reload, relevant full/compact/store regressions, both client types and scoped static/security assessment are recorded with honest Temporary copy.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Accepted design: Docs/Design/2026-10-04-d7-conversation-model.md §8 Q3 and parent design §Accepted D7 reconciliation explicitly supersede prior transcript exclusion. Plan: Docs/Design/IMPLEMENTATION_PLAN_chat_temporary_reload_13537.md. ADR required: no new decision; adopt the accepted Q3 using existing session/snapshot lifetimes, ownership and cleanup, with no competing transcript registry, history authority or outbox. ADR-049/050 govern ownership/lifetime, ADR-059 task editing. Native causal before/after: /private/tmp/chat-native-temporary-completed-before-reload-20261007.png and /private/tmp/chat-native-temporary-lost-after-reload-20261007.png. Shared root rejects real historyId=temp, omits transcript fields, and clears fresh restoration; sidepanel durable normalization intentionally strips arrays. Existing 24h expiry conflicts with Q3. Duplicate search found only unrelated TASK-13389 Temporary saved-chat read access. Highest local/dev/open-PR task ID13536; fresh origin/dev2c5f19d032 remains included in owned HEAD2bea3bf4b2. Entire NN/G program and stages remain In Progress.
Causal red: six genuine failures /249 passing ownership/storage controls in five suites, /private/tmp/chat-temporary-reload-13537-red.log. Full-page real store/hook cannot retain completed transcript, sidepanel current owned snapshot is stripped in both route configurations, and normal24h expiry rejects Temporary. Production repair uses three existing modules: add qualified transcript to existing session; admit only null/temp sentinel and observe/fence arrays; pagehide flush; restore exact arrays with basic persisted-shape guards; retain owned sidepanel arrays while clearing mutation authority and queued work. Full-page transcript uses existing sessionStorage pin and is omitted from localStorage fallback, so closing the browser tab ends its transcript without suppressing reload. Current owner/scope guards remain.
Current source/labels checkpoint:444passed in12affected suites; both types pass;17authored TS/TSX files have24baseline/24current/zeroadded lint findings and14owned formatting regions pass. Both independent reviewers clear bounded source/privacy/lifetime behavior after compact row validation, held-save/New/account and quota privacy controls. Native current-build copy verified; completed-send proof is pending a separately tracked startup catalogue race TASK-13538. Correct runtime cookie policy/backend models are200; native dropdown remains Loading after startup. Stage2/3complete,Stage4In Progress; no completion or packaged lifecycle claim.
Final closeout supersedes the earlier pending native checkpoint. All four owned plan stages are Complete; only this completed plan is retired. Final formatted combined Temporary/catalogue matrix:543passed in17suites, no skips (/private/tmp/chat-temporary-and-model-13537-13538-final-matrix.log). Both WebUI typecheck and extension compile pass. Effective unchanged-rule ESLint covers all19authored TS/TSX files:34baseline/34current/zeroadded findings;14Temporary owned formatting regions plus the catalogue test region pass, and git diff --check passes. Both authorized independent reviewers clear the final bounded source. Bandit N/A: authored TS/TSX/JSON/docs only, no Python repository changes. Official English locale sync is scoped to seven intentional keys, with combined messages parity and unrelated/non-English bytes retained.
Actual current native WebUI/local deterministic provider completes two Temporary rows and reloads them with exact next draft Next Temporary turn 13537 — UNSENT plus two trailing spaces; provider resolves Healthy without manual cache refresh after TASK-13538. Before/after evidence:/private/tmp/chat-native-temporary-retained-before-reload-13537-20261007.png,/private/tmp/chat-native-temporary-reloaded-13537-20261007.png and matching DOM JSON. New saved chat invokes the existing Clear boundary; subsequent actual reload has zero articles, empty composer and History on (/private/tmp/chat-native-temporary-cleared-after-reload-13537-20261007.png and JSON). Authenticated actual server content search returns zero matches for the unique Temporary question. Original ordinary4/Mira5 rows retain exact pre-check SHA256 values (/private/tmp/chat-native-temporary-server-history-13537-20261007.json). This uses disposable cookie/data and native UI, without interception or browser-storage seeding. Current origin/dev2c5f19d032 is included.
Remaining limits: full/compact ownership, malformed-record, pagehide/held-save, expiry and quota controls declare their lower storage/account/server boundaries. This does not prove packaged extension worker, browser crash/opener/duplicate-tab behavior, physical touch,200% zoom, VoiceOver, offline-owner verification or generated roleplay quality. Tab-local full-page transcript excludes the localStorage fallback; compact logical tabs retain only qualified snapshots and delete them through existing Close behavior. Entire all-ten program and TASK-13457/13458 remain In Progress, including D7 adoption, scoped ACK/provenance/lore, ThemePreview/hover and native extension/accessibility/participant gates.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the accepted on-device Temporary lifetime through existing full-page and sidepanel snapshots. Exact owned messages and the next draft survive reload; New/Clear, owner and malformed-record boundaries remain fenced, without server-history creation or queued replay. Shared copy now explains device retention. Final543checks, both client types, zero added scoped lint and two independent source reviews pass; native reload/New and server absence are recorded. No new ADR, dependency or protocol; parent all-ten review remains In Progress.
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
