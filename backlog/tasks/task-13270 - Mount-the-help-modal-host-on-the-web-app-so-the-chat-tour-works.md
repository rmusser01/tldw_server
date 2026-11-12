---
id: TASK-13270
title: Mount the help modal host on the web app so the chat tour works
status: Done
assignee: []
created_date: 2026-09-19 14:47
updated_date: 2026-09-19 14:47
labels:
- chat
- webui
- onboarding
- ux-audit
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Take a quick tour on the chat empty state is a no-op. Verified live: the DOM is byte-identical before and after the click, and dispatching the open event by hand also does nothing. The host that renders the help modal is mounted only when the layout is running headerless, which is false for a signed-in user on /chat. Three correctly-targeted chat tutorials and a mounted tutorial runner are therefore dead code on the web app. A unit test asserts the button works, against a mock of the store action. The button is also 118x16, below the accessibility target floor.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Clicking Take a quick tour on /chat produces a visible result.
- [x] #2 The help modal host is mounted for signed-in users on /chat.
- [x] #3 The existing chat tutorials are reachable from the chat surface.
- [x] #4 A test exercises the rendered outcome, not only that the store action was called.
- [x] #5 The control meets the minimum target size.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Accepted all-ten original high-priority novice help blocker remains present on current dev97ea9cd5fa/HEADbcec751c5c. WebLayout mounts existing demand-loaded PageHelpModalHost only for hidden header; normal signed-in chat Tour changes the real help store with no host. Shared extension Layout/EventHosts already mount host unconditionally. Reuse that root host once in WebLayout, preserve lazy loading/failure/Escape/focus guards and existing tutorial runner. Shared empty-state Tour control gains44px target. Reach real WebLayout + shared Empty + real help store/host/modal/tutorial catalog, existing layout/empty/help/runner tests, both clients; no tutorial registry redesign or shortcut consolidation under this unit. ADR assessment: presentation host reachability/accessibility only, no durable rule; ADR049/050 ownership unaffected,059 tracking. Own3-stage plan Docs/Plans/IMPLEMENTATION_PLAN_chat_tour_host_13270.md. Parent13458/all ten remain open.
Two actual composed rendered-outcome regressions fail causally before production change: header-visible Tour never renders Page Help, and registered tutorial Start is unreachable. Headerless modal/catalog/Escape/focus/draft positive passes;27/29 existing controls pass. A redundant existing isMac mock export was removed before final causal red. Use real WebLayout/Empty/help store/host/lazy-loaded modal/catalog, with declared unrelated shell/services and mocked TutorialRunner; native actual runner proof remains pending. Minimal production root now mounts existing demand-loaded host once in every WebLayout and adds44px shared Tour target. No new modal, registry, dependency or behavior for settings/transport. Stage1 complete, affected gates/source review/native pending.
2026-10-09: Independent source reviews caught a new two-dialog collision in the initial unconditional host patch. Do not close or ship that patch alone. TASK-13271 is now In Progress for the shared ownership correction; verify real legacy/header events and keyboard outcomes together. Newly reachable tutorial Start foreground uses white against dark primary (3.13:1); use existing primary-foreground token and verify effective contrast with the paired native tour check. Initial six-suite run: 53 pass; four pre-existing p1 source-anchor failures in unmodified PlaygroundForm, Characters Manager, Flashcards ImportExport and WorldBooks Manager require baseline attribution and a real chat-target outcome, not a false all-green claim.
The initial computed 3.13 contrast note was an inference from a raw token, not effective rendered color; native light/dark normal/hover contrast will be measured on the qualified build. Both source reviews caught and cleared page-local keyboard competition, stacked Escape dismissal, retained hidden AntD panel blocking and paired-token transition. Controls failed before repair. Existing ancestor visibility helper exported/reused without behavior change. Connected matrix67/2298 and both final client compilers pass;24 authored TS/TSX scopes add zero static findings. Supplemental EventOnlyHosts fixture now declares its actual Router boundary; final matrix will include it. Real retained-wrapper test is explicitly a DOM fixture; native actual AntD and TutorialRunner proof remains pending.
Native first-time headerless Casual Tour opens and reaches model/input/slash targets, then silently skips missing prompt and New chat targets and ends without completion. Fix the tutorial definition to use permanent Customize/response-settings and composer entry targets, retain truthful saved/temporary guidance and configured-shortcut reference. Native Help also exposed missing Web tutorial namespace (steps rendered as the key); add its existing resource to Web i18n registration and lazy alternate-language mapping, not a replacement formatter. This extends the paired help qualification at its actual source; keep unrelated tutorial failures explicit.
Current causal target controls reproduced raw comma selector DOM-order choice, collapsed Pro missing first model control, and actual Joyride alertdialog keyboard collision. Shared Runner now chooses first visible ordered target, re-resolves the original selector after React removal, and retains bounded retries/SSR guards. Basics model falls back to permanent input; headerless full extension keeps the same lazy Runner. Final72-suite matrix2327 positive checks passes; both types and31 TS scopes add zero static findings. Native namespace absence is repaired in the existing demand-loaded Help module: fresh eager EN registration failed the real600KB budget at600.6KB, so English resources now arrive only with requested Help; real Web init asserts absence at startup then exact ICU5 steps. Alternate-language mapping remains aligned. Shared inherited Characters/Flashcards/WorldBooks selector failures remain outside chat acceptance; chat-only source membership3 positives is not layout geometry. Final build/native qualification underway.
Final2349/75 connected checks and lazy namespace33-scope static gates pass; buildz07nNJMX7YprvcScWBZyI passes798 assets/tokens595.6/791.8KB. Actual headless first-time Basics runs all5 visible steps, literal question-mark positive/outside-editable negative/inside-editable negative-over-tour, actual lazy CmdK stack2->1->0 with close/initiator focus and exact multiline draft. Effective Start contrast dark6.709/4.836 and light4.618/6.500, normal/hover0s and44px; theme restored. Native Close then exposed controlled Runner ignoring ACTIONS.CLOSE: active guide persisted. Two causal callback negatives reproduce ignore and false Complete; shared close now retires retries/ends without completion, real Joyride composition passes. A new build/native Close verification is pending. Actual AntD context and current settings modals block Help while visible/reopen after close, but both unmount their role=dialog after close; retained-hidden native geometry is not claimed, only declared DOM fixture covers retention.
Final bounded qualification passes:75 suites2352 positive, both clients,33 TS scopes0 introduced findings, owned formatting and both source reviewers. BuildJ_E3ie_TkxILUHn1UmyPL actual Close removes guide/exact draft preserved; tokens798assets595.6/791.8KB unchanged budgets. Canonical25-row whole response deep-equal. Existing8GB compiler heap resolves default4GB buildOOM; no product budget/config change. Integrated current devcd5160201c no source interactions. Detailed master supplement/finiteclosureledger and native receipts retain limits: paused offline reload reseeded exact draft, actual AntD wrappers unmount, Tools/Voice full geometry/packaged/physical/AT/participants unproved; tour Close BODY focus is13568, shortcut capability gap remains13271. Own3-stage plan completed/retired to /private/tmp/IMPLEMENTATION_PLAN_chat_tour_host_13270_completed.md. Bandit N/A authored TS; imported dev production Python checked separately.
Integrated-dev verification complete:37 inherited scheduler tests pass, imported3production Python files have zero Bandit findings, Web types pass, both authorized source reviewers CLEAR interaction assessment. Imported Python bytes match dev; native help proof is unchanged-code bounded and retained against its exact build. No authored repository Python.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Normal signed-in Web chat Tour now reaches the existing demand-loaded help host; shared Basics has real visible ordered targets, demand-only count resources and working Close/Finish behavior. Unified keyboard help replaces competing surfaces while navigation uses purpose-specific labels. Current source/tests/types/static/budget and actual headless Basics/Close pass. Independent capability and owned-focus follow-ups remain explicitly tracked; this completes the bounded tour unit, not the all-ten program.
<!-- SECTION:FINAL_SUMMARY:END -->
