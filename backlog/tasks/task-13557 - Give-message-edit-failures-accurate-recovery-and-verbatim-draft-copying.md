---
id: TASK-13557
title: Give message edit failures accurate recovery and verbatim draft copying
status: Done
labels:
- frontend
- chat
- ux
- bug
dependencies:
- TASK-13556
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Plain Save and Save & Send show one generic Try saving again message and may expose raw server codes in notifications. Distinguish conflicts, busy state, changed account/edit, unsupported payload/history/sync and unconfirmed writes; preserve exact drafts and offer explicit verbatim copying before a user reloads. Reuse common editor, existing structured API error helper, translations and platform clipboard; no automatic retries or new mutation authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Failure feedback names the relevant condition and a usable recovery without claiming an unknown write was rejected or showing raw codes, paths or secrets.
- [x] #2 Common full/compact editor retains exact text and offers explicit verbatim copy with honest clipboard success/failure and stale-result guards.
- [x] #3 Shared handler notifications and deployed English copy agree with the editor; causal tests, both compilers, touched lint and two independent reviews qualify the unit while parent all-ten work stays open.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required:no; existing ADR049 owner/selection and ADR059 tracking apply. This is presentation and explicit user clipboard action only, with no public contract, persistence or server authority change. Highest task13556 checked across origin/dev, open PR paths and both local backlogs. Latest dev97ea9cd5fa remains included at HEAD43899c2065. User authorized all identified issues and enhancements; no new design approval required.
ADR required:no; ADR049/059 govern. Final authored code: shared EditMessageForm and createEditMessage, private allowlisted feedback helper, English assets/public mirrors, actual editor/handler regression tests. Initial red22 failures/14 positives, review busy-retirement red6/2, stale-start red2/31 filtered. Final37 suites1788 runner passes =1787 positive plus inherited Regenerate it.fails excluded from acceptance; both compilers exit0, six authored TS files six baseline/current warnings zero errors/introduced; TS structure plus compiled JSX runtime/comment-equivalent formatting, exact locale mirrors and diff check. Two independent source reviews CLEAR. Logs /private/tmp/chat-edit-recovery-13557-{qualified,matrix,web-types-final,extension-types-final,static,format,review-red,entry-red}.log. Bandit N/A: TS/TSX/JSON/docs only. Latest dev97ea9cd5fa included. Three stages complete, own plan retired to /private/tmp/chat-edit-recovery-13557-completed-plan.md. H1 copy cases use a declared controller fence plus actual draft store; no native clipboard/browser/packaged/physical/AT/participant acceptance. Already-issued platform clipboard calls cannot be cancelled; retired receipts cannot certify a newer draft. Parent all-ten gates remain open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Message edit failures now give condition-specific, translated recovery while preserving the exact draft and uncertain write status. Explicit verbatim Copy is fenced at rendered entry and completion, with honest failures and no stale receipt/busy-state transfer. Shared editor and handler roots cover existing callers. Final1787 positive client regressions, both compilers/static/format and two bounded reviews qualify this unit; whole-program/native acceptance remains open.
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
