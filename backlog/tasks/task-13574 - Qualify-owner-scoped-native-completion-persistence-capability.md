---
id: TASK-13574
title: Qualify owner-scoped native completion persistence capability
status: Done
labels:
- chat
- ux
- root-cause
- d7
dependencies:
- TASK-13573
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bounded prerequisite for accepted D7 ordinary server settlement TASK13573. Opt in to the existing authenticated H1 capture to discover whether this exact owner supports native selected completion. Preserve legacy strict clients, older server omission and enrolled Sync browser-managed history; no dispatch or receipt behavior is changed in this unit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An opted-in captured send returns a strict owner-scoped native completion capability; active scoped Sync reports unsupported and original completion still refuses a later enrollment before writes.
- [x] #2 Legacy unrequested responses omit the new field; omitted older-server capability selects the established client-managed lane, with local/fork and direct SDK capture semantics unchanged.
- [x] #3 Client capture requests preserve original account, CID, workspace and signal; malformed or foreign capability responses are rejected with no writes.
- [x] #4 Focused authenticated contract, actual client transport/parser, connected history, both types, security/static/hooks and independent review evidence are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent plan Docs/Design/IMPLEMENTATION_PLAN_chat_native_server_settlement_13573.md. Actual mounted Full/Compact dispatch RED is /private/tmp/chat-native-server-settlement-13573-dispatch-red.log: both still send save_to_db false. Test bytes restored exactly after RED; no production switch yet. Public opt-in capability rule needs proposed ADR067. Existing snapshot, input and final admission authority do not change. Search found no duplicate capability task; latest origin/dev and open PR ID scan highest TASK13573/ADR066. Two authorized source reviewers assess compatibility; one root runner only.
ADR required yes: proposed Docs/ADR/067-native-completion-capture-capability.md. Opt-in query preserves old strict JSON. Strict proof carries selected_history_v1 policy, supported ownership flag and exact storage_context_digest; only captured send may carry it. Original completion retains all admission/Sync validation. Plan Docs/Design/IMPLEMENTATION_PLAN_chat_native_completion_capability_13574.md. Source tests added before production; no dispatch switch or new registry/helper.
Causal RED qualified before production: authenticated API4 missing-capability failures/3 legacy-fork positives; client2 suites5 missing-parser/opt-in transport failures160 positives. Evidence /private/tmp/chat-native-completion-13574-api-red.log and -client-qualified-red.log. Initial base-client fixture assumption and wrong runner cwd were corrected and excluded from product evidence. Minimal five production files now implement only opt-in capture eligibility; capability context strict matching and unchanged primary Sync gate.
Final client connected7 suites638/638 pass (-client-final-connected.log), including actual composed public method and domain-on-base distinction. Authenticated history/Sync/server-generation run125PASS5skips (-api-connected.log); final actual service-backed scoped Sync35/35 (-sync-final.log, XML). Source factory/auth remain declared fixture boundaries, not physical enrollment qualification. Both actual compilers exit0 empty logs. Five TS scopes baseline249/current249/0 introduced; four Python scopes Ruff0/Bandit0 production/test nonassert findings, required pytest B101391->423 recorded. Two I001 formatting findings fixed with unchanged Python AST; owned TS executable AST equivalent. Both authorized bounded reviewers CLEAR, public method gap closed. Current origin/dev3b68f2dbc2daa1b94099a8a3d56c809fd2c808db is ancestor. Pending exact-file normal hooks and final commit; no native dispatch or package/browser/study proof. Scope-specific property tests not needed for fixed enum/boolean/string contract; strict malformed matrix and authenticated lifecycle cover boundaries.
All17 exact owned files pass normal pre-commit checks, including canonical backlog, raw-patch guards, syntax/private-key/conflict/whitespace gates; no bypass. No imported source or fixture changed during runner/compiler. Parent TASK13573/master13458/all-ten stay In Progress. Completed bounded plan retires after normal commit to /private/tmp/IMPLEMENTATION_PLAN_chat_native_completion_capability_13574_completed.md. Packaging, live native server settlement, physical enrollment/AT, participant measurements and unavailable PostgreSQL are not qualified here.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added strict opt-in original-owner native completion persistence eligibility to existing scoped H1 capture. Legacy/fork/failure JSON and older-server omission remain compatible; global enrolled Sync is unsupported, independent Workspace supported, lookup failure closes eligibility, and primary admission still rechecks enrollment before any write. No generation dispatch or save/recovery behavior changed. Final638 client controls, backend125PASS5skips plus service-backed Sync35PASS, both types,0 introduced lint/security, executable/AST formatting,17-file normal hooks and two independent source reviews qualify the bounded prerequisite. Parent ordinary receipt/partial/recovery and overall UX closure remain open.
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
