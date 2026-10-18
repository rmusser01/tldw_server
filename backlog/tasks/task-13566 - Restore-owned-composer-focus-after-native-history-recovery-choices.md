---
id: TASK-13566
title: Restore owned composer focus after native history recovery choices
status: Done
labels:
- chat
- ux
- accessibility
- webui
- extension
dependencies:
- TASK-13458
- TASK-13563
- TASK-13564
- TASK-13565
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current task-owned native WebUI on latest dev verifies explicit Load latest preserves the exact draft and changes only the selected path, but removes its focused recovery control and leaves BODY active. Shared Latest and Branch actions make no focus request. Restore focus through the existing composer event only after the exact owned choice is accepted; preserve view/draft/account/mount/activity and user focus ownership through held and deferred work. No automatic send or hidden reparenting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Successful shared Latest and Branch choices return desktop composer focus once after their exact owned acceptance, retaining draft text and requiring manual Send.
- [x] #2 Failed, refused, held or retired choices and New, same-ID H1, account/source, unmount, busy/editor and later focus changes cannot steal focus or adopt an unrelated replacement view; own accepted Latest epoch remains usable.
- [x] #3 Meaningful controller/component/event controls, both client compilers, static/security scope, two independent reviews and current existing-browser native evidence pass; mobile, packaged, physical, AT and all-ten limits remain explicit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task allocated after remote/open-PR/local max13565 and duplicate recovery-focus search. Fresh dev97ea9cd5fa is included in HEAD7c3fa4d8fd. Original native proof /private/tmp/chat-native-compatibility-13565-native-proof.json and native-latest-focus.log records BODY active after explicit Latest. Source-only branch focus gap is not yet a native observation. Reach: real useHistorySelection accepted choice callbacks, shared HistorySelectionReview actions and original source/focus lease, existing full/compact focus event handlers, actual controller/component and neighboring controls. ADR assessment:no new durable rule/API/persistence/dependency; existing accepted D7 and ADR049/050 source authority, ADR059 tracking. Reuse existing private accepted-open callback pattern and composer focus event, never a global document search or a newly sampled unrelated view.
Qualified four causal controls reproduce missing accepted callbacks and actual composer focus before production. The optional private accepted-choice callback binds original source/owner/capture/activity; Latest callback exceptions stay outside its state-changing catch. Shared Review reuses existing StateAction button refs, greeting focusin/pointerdown ownership and one cancelable RAF plus existing focusComposer event; passive paths and full mobile/noncollapsed-selection policy stay unchanged. Independent review identified Branch same-view rearm ABA. Two actual controller/old-frame controls fail before exact operation epoch/kind pin and pass afterward. Final44-suite connected matrix1976/1976 passes; after repairing one introduced test-fixture render side effect, final H1/Review178/178 and WebUI type gate pass, prior extension type gate passes. Four touched TS/TSX files0 introduced static findings; owned formatting AST-equivalent. Both authorized independent source reviewers CLEAR. No repository Python authored; Bandit N/A for this unit. Build/current native proof pending; all-ten master remains In Progress.
Final actual native evidence: build D7F825-h96xHLo8ujYwqc tokens/799 assets and unchanged600/900KB budgets594.9/791.1 pass. Same existing headless browser verifies actual strict-false409 requests8568/8650, Latest18 selected rows and Branch18/armed, exact multiline/trailing-space draft focused with Send enabled. Canonical entire responses deep-equal21/23 rows; no exact input admitted or generation request. Evidence /private/tmp/chat-native-recovery-focus-13566-native-proof.json plus latest/branch DOM logs/screenshots, readbacks and request/response bodies. Fresh origin/dev97ea9cd5fa remains ancestor. Completed own plan preserved at /private/tmp/chat-native-recovery-focus-13566-completed-plan.md and removed only from repo. Full native proof is collapsed desktop; controlled compact event primitive is distinct from packaged/mobile/range/physical/AT acceptance. Parent/all ten remains In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the shared recovery focus root: only exact accepted Latest/Branch actions can request the existing composer event, with original-source/owner/view/epoch/activity/editor and deliberate focus lifetime through one RAF. Same-view Branch ABA is causally tested and pinned. Callback failure cannot undo accepted history; passive paths/text/transport remain unchanged. Connected1976/44 plus final H1/Review178, both types,0 new static findings, equivalent formatting, two independent source reviews and current actual native proof pass. Bandit N/A, no repository Python. Preserved mobile/noncollapsed-selection policy, packaged/physical AT and broader all-ten gates remain explicitly open.
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
