---
id: TASK-15430
title: Retarget the shared current-account menu to the dev header
status: In Progress
created_date: 2026-11-06 23:46
references:
- https://github.com/rmusser01/tldw_server/pull/3400
updated_date: 2026-11-06 23:53
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update existing draft PR #3400 against dev. Reuse the current composer behavior and preserve both header layouts, notification and account authority, keyboard focus, and existing preferences.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Only the public shared menu changes missing from dev remain in the existing draft.
- [x] #2 Verify account-menu, notification, keyboard, and compact composer behavior on dev.
- [x] #3 Account actions remain visible and receive pointer input above Settings page navigation in both shared header shells.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design and implementation plan: Docs/Design/2026-11-02-header-account-menu.md. ADR required: no. The change keeps the existing authentication, routing, and user-preference contracts.
Dev 2f402bd7e3 already provides persisted Tools disclosure and visible model, attachments and Send on mobile and desktop. Port the original public account component into the shared header control; keep both layouts and test forward/reverse keyboard focus. Reviewed the integration independently. All 184 focused tests in 12 suites passed, as did WebUI typecheck. ESLint: zero errors, one pre-existing share-label warning. Bandit: no Python in the touched directory. No full production build, live sign-in or physical-device verification. The human-authored summary remains a separate merge requirement.
2026-11-06 authorized two-PR landing: preserve current dev port fd63fcf8cdc76fe7769939b7fc51b90a83fcc032 and original70d8 refs. Current3368 a7afb61aa8 is already published and waiting healthy required CI; do not rebase3400 until3368actuallymerges. Read-only newhead review clean. Prospective combined source snapshot passed413tests34suites, normalTS, full6123filelint0failures775 unchangedfrozenerrors, advancedwebpack/token/budget594.6KB pass. Production browser with syntheticJWT/APIfixture exposed real light-theme Account settings/Settings contrast3.13:1 (#5c8dff onwhite), below4.5; globalAntDanchorcolor overrides row text token. Capture /private/tmp/chat-stream-landing-20261106/pr3400-browser-scenes/diagnostic-chat-desktop-light.png. Fix root style interaction with failing-first browser regression, preserve all identity/account/notification/focus contracts. Priorprecommit42314877093/job128226580785 ran currentfd63but compared obsolete mainbase64ec68(9182paths), so its unrelated whitespace fixes are not current7file devdelta defects. Neweventafterproperdevintegrationrequired, no blindretry or unrelated edits. Human summary stillpending; devtargetconfirmed andalreadyappliedexternally. ADRassessment remains existingcontracts/no newADR.
Audit correction2026-11-06: 3.13contrast report was a false positive from my temporary MCP audit forcing html light while AntD retained dark palette. New real browser test using consistent stored theme PASSED on unchanged product; no styling/production repair made. User explicitly corrected. Added8 account popup acceptance cases across chat/settings xdesktop/mobile xlight/dark preservingaxe/geometry/44px/identity/settingsroute/focus/Escape/reverseTab. Initial13case combined browserrun7pass6fail: all6 desktop/settingsaccount cases pass;2chatmobilefixtures entered defaultFocus and hidheader, corrected fixture toseedexistingcockpit and exitFocus throughsameUI as incumbenttest. Four existingcalmheader cases fail secondBrowsepagesmenu click outsideviewport; exactdesktopdarkcase alsofailsidenticallyonpreservedpre-account3368artifact. Root unproven, no production/testworkaround or timeoutchange. One private strongerDOM-retirementprobe pending, no blindretries. Initialfalse-positiveharness screenshots preserved andnotusedasproductred. Product fd63 port remains byte-identical.
Confirmed new integration defect: mobile Settings page navigation (z20) paints later than the shell header wrapper (alsoz20), covering Account settings/Settings popup rows despite valid bounding boxes and zero axe findings. Screenshot inspection caught it; added trial pointer checks for every account action. Failing-first realproduction-browser case settings/mobile/light fails with the page-navigation subtree intercepting pointer events (pr3400-occlusion-red.log). Minimal root correction raises only shell-header wrapper fromz20toz30, above page navigation and below existingmodal/dialoglayers; no dimensions, auth/notification behavior, dependencies or workflows changed. Existingthreecomparablelayers reviewed: Settings asidez20, headerwrappers, shortcut overlayz30; same-level later overlay stillwins. No newADR: preserve existing shell/content/modal separation. Final qualification pending. OldMoremenu retirement-only privateprobe also failed sameoutsideviewport; do not ship that diagnosticwait or blindlyretry. No source/testassertion workaround for inheritedMore case.
Fresh review caught incompletecaller coverage before greenclaim: Next _app renders @web WebLayout, whose separate shellheaderwrapper also remainsz20. Apply identical z30correction in BOTH shared Layout and frontendWebLayout. All header shell callers searched; existingmodal/loadinglayers retained. Stopped only own incomplete native compiler by exact observed commandmatch toavoidwaste; rebuildbothcallers before browsergreen. No source/branchremoteupdateyet.
Final local qualification after both shell fixes: 558 unit cases across38suites pass (413sharedUI/34 plus145Web/4). Normal frontend TypeScript8GB passes. Full shared hook guard6123files0failures775 unchangedfrozenerrors; installed scopedESLint8files0errors1inheritedChatHeaderexplicit-any warning. Exact final advancedwebpack wrapper,token sync,bundlebudget pass594.6KB/604KB; doesnotcertifyCI Turbopack/Ubuntu. Realproduction Chromium eight account cases PASS with allrow pointertrial/axe/44px/geometry/identity/settings-route/focus/Escape/reverseTab assertions. pr3400-occlusion-red.log showsblockedpointer beforefix; pr3400-occlusion-green.log8/8green afterbothcallers. CorrectedSettingsmobilelight/dark screenshots visuallyinspected; no occludedrows. Independentfinalreview clean, missedWebcallerresolved. Impeccabledetector0; scopedreport /private/tmp/chat-stream-landing-20261106/pr3400-account-audit.md. Ownports18318/18320stopped andMCPowncontextclosed. ExistingMoremenudiagnostic remainsunprovenafterbounded3comparisons, notpatched/retried/weakened. NoPythonchanged; Bandit cannotvalidateTS. Humanownwords summaryor explicitwaiver stillpending; do notinferfrom3368. Preservefd63and70d8refs; localfixwillrebaseonlyafter3368actualmerge, thenfreshCI exactheadrequired. No remote3400updateyet.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Retargeted the existing account menu to dev while preserving the current composer and both shared headers. The account popup now stays above Settings navigation in both shell wrappers; failing-first pointer coverage verifies the correction. Local qualification passed558unit tests/38suites,8production browser cases,normalTypeScript,fullguard andbuild/token/budget; fresh independentreview clean. Synthetic accounts were used; live identity providers, packaged extension and devices were not checked. Four inherited More-menu browser failures are separately documented. Requester summary/waiver, serial rebase after3368 and protected exact-head CI/merge remain pending.
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
