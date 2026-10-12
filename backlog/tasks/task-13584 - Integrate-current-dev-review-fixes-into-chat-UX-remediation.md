---
id: TASK-13584
title: Integrate current dev review fixes into chat UX remediation
status: Done
labels:
- chat
- ux
- integration
dependencies:
- TASK-13583
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the explicit latest-dev requirement after origin/dev advanced from 39b53a876f2cfefcede02c92ae7dacf88f7a98ec to 2a935383ac1305fac3af26f2596c203b31a29909. Integrate the seven-file reviewed upstream delta without losing accepted chat owner, recovery, roleplay or storage contracts, then qualify affected shared-client behavior, both types and the current native bundle. Keep all-ten/public H2/physical AT/study gates open.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current dev is an ancestor of the remediation branch and any merge conflicts preserve both reviewed upstream behavior and accepted chat contracts.
- [x] #2 Affected shared-client suites, both client types, introduced static/security checks and normal exact-owned hooks pass with limitations recorded.
- [x] #3 Current built chat still preserves the canonical conversations and existing one-tab workflow; tracking and closure ledger record this bounded integration without overall acceptance claims.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no new decision. Existing ADR049/050 ownership/lifecycle and ADR059 canonical tracking govern. Highest local/dev/open-PR ID13583 verified before explicit13584. Supersedes only the freshness checkpoint of completed13580; that prerequisite remains credited. Source integration waits for the current13583 connected runner and commit. No primary checkout or unowned browser mutation.
TASK13583 committed normally at7f8a472f3fafbaed66ca900f8652a23d3f55d886 (eight exact-owned files). Integrating origin/dev2a935 now, seven upstream files relative to prior39b53a. Upstream fixes research-action closure cache, media keyword debounce, model-index migration failure, ingest fetch strategy typing and popup loader diagnostics. Stage1 integration begins with no active source runner/compiler; primary checkout and unowned browser remain untouched.
Stage2 source integration required three bounded upstream corrections: remove the existing ViewMediaPage timer now that useMediaSearch owns the 300ms debounce; permit the actual worker's optional per-job callback alongside batching while requiring at least one fetch function; scope storage runtime.lastError to failed set callbacks and assert both migration writes were reached. Qualified causal RED: /private/tmp/chat-latest-dev-13584-causal-red-qualified.log (300ms dispatch missing and no failed write reached); corrected GREEN42/42 then final12 affected suites167/167 at /private/tmp/chat-latest-dev-13584-final-client-qualified.log. All eight changed TS files have zero introduced lint findings versus exact HEAD (78 retained baseline findings): /private/tmp/chat-latest-dev-13584-static-final.log. Initial page fixture reached an empty library and one overly broad storage fixture edit was corrected; these diagnostic runs are not acceptance evidence. Both independent source reviewers give bounded CLEAR. Normal WebUI compiler is running; extension types/current build/native preservation still pending. Six untouched server-load-state fixture failures recur at exact HEAD7f8a472f; exact-head-baseline-proof.json confirms all temporary baseline sources restored byte-for-byte. No tests disabled, no public H2 or whole-program acceptance. Greeting picker now belongs to PlaygroundForm, so old PlaygroundChat greeting expectations need a separately tracked migration with real current-host controls. Bandit: no Python change in this seven-file upstream TS/backlog integration; prior owned13583 final Bandit0 findings/errors remains credited for unchanged Python scope. ADR049/050/059 continue governing; no new durable decision.
Final normal WebUI compiler configuration exits0 with node --max-old-space-size=8192; the first default4GB heap run exited134 from V8 allocation exhaustion and is excluded. Normal extension tsconfig.compile.json exits0 before the set-e current-build wrapper begins. Build is in progress. Fresh ID scan for inherited fixture follow-up confirms origin/dev still2a935383ac1305fac3af26f2596c203b31a29909; TASK13585 now tracks all six exact-HEAD fixture failures with current real-host coverage. No tests are disabled. Final12affected suites167/167 and eight-TS baseline78/current78/zero introduced findings remain acceptance evidence; the earlier ten-file command omitted two suffix-qualified suite paths and is excluded from the final count. Both source reviewers confirm final bounded CLEAR including the fixture getter.
Stage2 complete: both client types and current build/tokens/budgets pass (593.3KB shared600/795.6KB chat900). One restored persistent owned headless tab1280x900 verifies finder search/clear/global Server10 total, then original two articles and blank911px input. Canonical GETs before and after preserve9 conversations28 rows deep-equal; baseline remains unchanged. Final supported CLI callback/host-written native-final.json and screenshot qualify the result; earlier capture syntax/require diagnostics are excluded. No new visible Chrome or unowned IAB mutation. Current native visual rail/recovery-label crowding remains in open docking/layout qualification. Source merge hooks/actual retained-parent commit now pending; status/AC stay open until that succeeds. Final source reviews bounded CLEAR; inherited six fixtures tracked13585 remain unqualified here.
Actual normal merge commit064be335948e7bfaca14f44839562de7fdac0b82 exits0 and dev2a935 is a verified ancestor. Normal exact-owned hooks first remove whitespace from two formatter-indented test blocks; all other checks pass and post-hook two suites11/11 pass. Final normal hooks are rerun before recording commit. Both reviewers confirm final callback-scoped runtime getter CLEAR. All three bounded stages,3AC and6DOD now complete; only this task own completed plan retires to /private/tmp/IMPLEMENTATION_PLAN_chat_latest_dev_13584_completed.md. TASK13585 remains In Progress for the six inherited fixtures; all master/public H2/physical qualifications stay open. No push, PR or merge-to-dev performed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Integrated current dev2a935 with preserved research callback/policy freshness, one shared300ms keyword debounce, supported batch callback typing and verified failed migration writes. Final12affected client suites167/167, both client compilers, introduced static0, current build/tokens/budgets and actual native9/28 preservation pass. Source merge064be33594 retains current dev. Six exact-HEAD inherited fixture failures remain explicitly tracked13585; all-ten, public H2, actual quota/adoption, package, physical AT and participant studies are not credited.
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
