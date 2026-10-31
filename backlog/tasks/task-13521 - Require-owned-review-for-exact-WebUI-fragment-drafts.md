---
id: TASK-13521
title: Require owned review for exact WebUI fragment drafts
status: Done
labels:
- chat
- ux
- webui
- extension
dependencies:
- TASK-13457
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Approved Stage4 continuation. WebUI fragment drafts bypass the actual hydrated composer via an unowned global event; the actual Form ignores ifEmptyOnly, and document waits can cancel delivery permanently. Preserve exact raw text, route hash and legacy encoded-query payloads before local storage IDs, and reuse the existing owned Insert/Replace/Cancel review flow. Draft review must be independent of automatic settings/document IO. Broader portable settings/document authority, source opener lifetime, native and D7 gates remain parent13457. ADR assessment: reuse ADR049 history-selection ownership and ADR050 lifecycle, no durable architecture or wire-format change; no new ADR needed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exact fragment draft text, including empty and whitespace-only strings, survives decoding without normalization.
- [x] #2 Actual composer import hook waits for hydration and always offers explicit Insert/Replace/Cancel for hash and encoded-query drafts; source and destination text remain intact until a current choice.
- [x] #3 Fragment review is independent of parent settings/document IO; stale routes, owners, H1 selections, input revisions and expiry cannot apply or erase another route token.
- [x] #4 Affected regressions, both clients, no-added lint, independent reviews, documentation and task gates pass with remaining parent/native limits recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/Design/IMPLEMENTATION_PLAN_chat_webui_fragment_draft_13521.md. Fresh ID census67 open PRs max13518, local13520, dev13510. Production source captures raw textarea text; root receiver and actual composer hook are separate. Bandit N/A TypeScript/TSX/docs-only scope.
Failure-first codec/hook13 cases, actual root review and independent edge findings reproduced: /private/tmp/chat-webui-fragment-draft-red.log, root-red.log, review-edge-red.log, revealed-query-red.log. Exact raw draft now uses actual owned review with stable hash/query transport classification; resolving hash marks hidden query explicit-review-required; matching original parent cleanup preserves replacement packets. Actual useComposerText cold durable restore(hash/query) and real synchronous account retirement pass6 copied-draft checks. Both final independent source reviewers clear bounded draft scope. Both client gates/no-added lint7TSfiles pass. Full matrix first run251pass plus one existing coordinator5000ms timeout under overlapping checks; serial rerun underway without assertion/timeout changes. Automatic settings/document qualification and receipts, source WebUI opener input lifetime, native and D7 remain parent scope.
Final serial7suite252pass at /private/tmp/chat-webui-fragment-final-serial.log; fresh WebUI types/extension entrypoint compile/no-addedlint7TSfiles pass. All three independent route findings reproduced red then green; both final source reviews clear. Actual cold durable composer restoration hash/query and synchronous account watcher pass. Master TASK13521 checkpoint records evidence and declared Form/IO/native limits; own completed plan retired. Latest remote dev7ba48 verified unchanged and included. ADR required:no; existing Docs/ADR/049-chat-history-selection-ownership.md and Docs/ADR/050-native-chat-fork-storage-lifecycle.md apply. Bandit N/A TS/TSX/docs. Automatic settings/document/context authority/receipts, actual source WebUI opener input lifetime, unknown-owner local review, native/packaged/accessibility and D7 remain parent13457.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved exact WebUI fragment and encoded-query drafts into the existing hydrated owned composer Insert/Replace/Cancel review. Removed unowned event delivery and document IO coupling. Expiry preserves transport classification; cleanup matches original packet and hidden query fallback requires review.252 affected tests pass, both clients check, no added lint, two independent reviews clear. Broader Stage4/all-ten/native/D7 work stays In Progress.
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
