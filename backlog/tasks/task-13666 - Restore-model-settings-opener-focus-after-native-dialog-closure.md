---
id: TASK-13666
title: Restore model settings opener focus after native dialog closure
status: Done
labels:
- chat
- accessibility
- ux
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Actual cockpit Save publishes Temperature and Override but leaves document focus on body. Parents conditionally unmount CurrentChatModelSettings immediately, bypassing the native Modal/Drawer close lifecycle and racing focus restoration. Keep each opened overlay mounted through closure, then restore its valid opener without adding arbitrary delays or keeping heavy editor subscriptions mounted while closed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Save, Cancel and dismiss finish native close lifecycle before editor unmount and opener focus restoration
- [x] #2 Full, compact and layout callers preserve lazy mounting and existing scoped save/dirty/pending guards
- [x] #3 Mounted regressions, actual cockpit focus oracle, scoped static gates and independent review pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment:no new durable architecture rule; existing dialog and original-source ownership contracts govern a lifecycle fix. Governing native history settings and focus-return helpers remain intact. Actual RED cockpit spec2183, private /private/tmp/chat-pr3229-cockpit-13656-native-results.json. Search found no matching model-settings-focus task; highest ID across fetched origin/dev, all open PR files and local tasks13665. PR https://github.com/rmusser01/tldw_server/pull/3229. Design/implementation plan must precede source edits.
Design/plan saved under Docs/Design/2026-10-13-chat-model-settings-close-lifecycle.md and Docs/Plans/IMPLEMENTATION_PLAN_chat_model_settings_close_13666.md. Independent design review favors immediate controlled open=false plus close presence through actual Modal/Drawer completion; all3owners and allcompact event/slash setter paths must use same state. Old close callback must not unmount reopened editor or clear/focus newer scope. Current open/history/account/model/lifetime guard must be rechecked inside scheduled focus. Shared focus-return helper semantics stay unchanged. Real actual RED already captured; no arbitrary delays or permanently mounted heavy editor.
One model-settings-specific close state now shared by Full, compact useComposerEvents (all event/slash entry paths), and Layout. Immediate open=false preserved; present persists only through native completion. Revision/lifetime rejects old completion after reopen or unmount; Layout route/header retires outstanding presence. Actual native Modal.afterClose/Drawer.afterOpenChange(false) forwarded. Full clears scope and restores source-guarded focus only on matching completion, with revision and draft/history/account/model/native lease recheck inside scheduled frame; shared focus utility unchanged. Mounted lifecycle4checks pass. Parent7controls first5passed; remaining2initiallyfailed due missing importedstore then modifying separate actualstore while Form usesdeclaredmock boundary—excluded fixture failures, reassessed and nowusesexistingForm publicwriters. Both parent7 and realnativeModal/Drawer2 underverification. No browserfocusfix or completedtask claim yet.
Held-close default composer reopen regression reproduced the explicit settings-scope leak (expected empty default scope, actual openai:previous). Shared setter will initialize default scope/provenance; external explicit-scope event remains separate. Native full six-suite run: 276 passed, six settings tests timed out at unchanged 5-second deadlines. Isolated one-worker settings run remains active; approval review correctly prevented imported-source mutation during that run. No completion or browser qualification claimed.
Central default setter fix applied; external explicit-scope event retains raw opening path. Eight actual-parent focus/reopen cases pass. Independent bounded source re-review clear. Three settings-suite verification attempts failed: broad six-suite run six timeouts/276 passes; isolated full one-worker run 44 timeouts/7 passes; isolated existing explicit-scope case five-second timeout. Stop repeating this issue and reassess comparable real AntD harnesses/configuration. No deadlines/assertions relaxed; no full-suite, browser-focus or completion claim. Native six-root source compiler pending.
Measured unchanged explicit-scope test: 81 original computed-style calls consumed4144ms of4798ms; render237ms/tab-found1129/tab-clicked2106/Save-found4199/close4249. Scope role queries to unique native submit action parent and prompt Reset field; retain visible role lookup, original 5s deadline and all assertions. Diagnostic instrumentation removed. Subsequent native full run281passed/two remaining Reset timeouts; focused scoped Reset/duplicate-close five cases pass. Repeated false RED reproduced stale captured completion; one-line no-op fixes it, without claiming an actual AntD leak. All six native suites running again.
Connected timeout reassessment continued from measured style-query cost: full native277pass/seven Cancel/Close/Apply/discard timeouts. Scope unique actual literal button candidates and retain visible semantic role within parent; seven affected cases pass focused15.62s with original5s deadlines. Latest full native six-suite run pending. Review artifact Docs/Reviews/CHAT_MODEL_SETTINGS_CLOSE_13666_2026_10_13.md records exclusions and pending browser/source/build gates.
Final connected native and WebUI profiles each6suites284pass0skip. Final six production-root dependency compilers native/WebUI exit0. Actual root11file lint0errors158unchangedHEADwarnings0ignored0new, preserving rule identity while normalizing embedded source-line shifts only. Original cockpit case now additionally checks native Close/Escape and 5s opener focus; not run yet. Source-matched isolated production build in progress; current preview unchanged. Bandit N/A TS/docs only.
Final bounded independent review clear; native/WebUI6suites284each0skip, final six-root types0both,11file real lint0errors158unchangedwarnings0new0ignored. Production build0/customdist tokens0/budget590.2<=600/heaviest869<=900;25sourcehashesmatch/969JSON/1130staticrefs/0missing; builduuJqBl4XVXSE2DePcX9jf. Original cockpit prompt/model/MCP case plus Close/Escape1pass0skip0flake5.6s, original5s focus oracles. Original9/28deepEqualtruebaselineRewrittenfalse; synthetic4rowsunchanged. Review Docs/Reviews/CHAT_MODEL_SETTINGS_CLOSE_13666_2026_10_13.md supersedes intermediate pending notes. Own completed plan retired. BanditN/A TS/docs only; installedextension/AT/participant/packageCI/wholeproject not claimed. WholePR review/rebase/merge and remaining13656/13667 stay open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fix the shared model-settings close lifecycle across full, compact and layout surfaces, retaining immediate controlled closure and lazy mounting until native completion. Preserve original source/revision focus, initialize default reopen scope, and make repeated close idempotent. Both connected284test profiles, final production-root types/lint, source-matched build/assets/budgets and actual Save/Close/Escape cockpit focus pass; original data unchanged. No whole-PR completion or merge approval.
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
