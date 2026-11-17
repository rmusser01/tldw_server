---
id: TASK-13576
title: Keep server chat finders current after acknowledged writes
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
Continue accepted D7 mirror/view-only UX root after TASK13573. Actual Compact creates and server-settles a native chat while its overview is mounted; closing exposes the linked mirror as Local until the stale server list refreshes. Canonical reopen remains correct. Trace original scoped mutation/read boundaries, refresh existing shared finders after acknowledged changes without polling or per-token requests, and preserve qualified mirror origin without treating summary/link IDs as turn-save authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Actual Full and Compact server finders refresh after acknowledged native creation or writes without reloading, polling or duplicating sends.
- [x] #2 Owned linked mirrors retain accurate origin and canonical reopen while genuine device-local and failed or foreign ownership remain distinct.
- [x] #3 Failed or unknown outcomes, account/scope changes, concurrent writes and held reads retain original trust boundaries; summaries grant no mutation or turn-save authority.
- [x] #4 Causal tests, connected consumers, both client types, static/security/hooks, current native and latest-dev evidence are recorded with package and participant limitations.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Master13458 remains In Progress. Native13573 Full6/Compact4 exact canonical and cold proofs pass; observed Recent Local badge before overview refresh is a separate finder root. No new receipt/run protocol in this unit. ADR assessment pending source traces; likely existing ADR049/059 for read cache coherence. Two authorized independent reviewers are tracing origin and freshness separately; no edits/runners delegated.
Reviewed roots: existing profile is required in reader lease; apply global/local qualification before bounded DB limit and merge. Native strict resultId also qualifies compatibility/Retry settlement. Actual Compact second turn emitted fourth refresh through legacy SDK linkage callback after H1 saver; suppress H1 linkage there while preserving SDK. Fixture reassessment after three diagnostic attempts: original Retry consumer stubbed invalidation; broader useMessageOption lacked its presentation mount, then added model-sync revisions. Those diagnostics are excluded as product evidence. Preserve original action/presentation fixture and connect actual shared cache helper instead; no assertion relaxation.
Broadened 34-suite run exposed 64 saved-normal failures; exact committed HEAD control reproduces all64 (70pass), caused by actor mock omitting real shouldInjectActorForTemplates export. Repair that connected fixture using actual export. Sixteen unhandled SDK/Roleplay errors come from existing query doubles omitting newly required cancelQueries; update those contract doubles, never make production cancellation optional. Retry callback diagnostic was scoped to wrong helper instance; connect the intended useFullConsumer callback exactly. Preserve all original assertions.
Final connected34 suites1834/1834 pass with native creation/input/result acknowledgements, original account/cancel fences, SDK/Local/Temporary/Roleplay/voice callers, and no unhandled errors. Missing-final admitted-input causal RED2/controls6pass; duplicate/missing-admission/Retry preserve distinct receipts. Both bounded independent reviewers CLEAR including admission leaf. Final20TS static baseline138/current138/0introduced; AST-equivalent changed ranges applied, unsafe formatting candidates discarded. Exact original native Full6/Compact4/finder-baseline2 response equality passes before new build. Sandbox-local HTTP denial and initial test-name/cwd diagnostics excluded. Both types/build/native/hooks still pending. Bandit N/A: only TypeScript/tests/docs/task records authored, no Python source.
Current native finder GREEN: verified webpack/tokens/budgets593.2KB shared/600 and795.4KB chat/900; same isolated API/provider/profile and one headless tab. Full primed Find then creates/two native turns, immediate Server result without reload, search/close/reopen canonical4 and blank input, no reopen POST. Compact390x844 primed Recent, creates/two native turns, closes only view, immediate Recent Server, canonical reopen4 and cold4 blank input/zero POST. Each surface one creation/two completions/no browser message POSTs. Exact original Full6/Compact4/baseline2 and newFull4/newCompact4 responses preserved. Private native-full/compact/cold proofs and PNGs under chat-server-finder-freshness-13576 prefix. Fresh origin/dev3b68f2dbc2daa1b94099a8a3d56c809fd2c808db included. Current35 affected suites1838pass aggregate (34connected1834 plus selected-model4), both compilers0; final19TS baseline137/current137/0introduced. Unrelated render bench1 failure reproduces exact HEAD before transport; removed unnecessary bench edit, no timing qualification. Bandit N/A TS-only. Normal owned hooks and commit pending. Exclude guessed accessible label/count and covered/hidden underlying-header waits, initial canonical role-vs-sender assertion and runner-cwd diagnostic; actual observed controls/receipts qualify. Follow-ups: native Copy notification; Compact New resets Provider to Default while retaining selected model and leaves overlay open; clarify account total vs matching Find count. Package/physical AT/participant and all-ten queue remain open.
Normal exact-owned hooks pass: EOF, whitespace, line endings, secret/merge guards, canonical backlog format and existing test safety guards; Python-only hooks skip authored TS scope. Final 19 TS baseline137/current137/0introduced supersedes earlier checkpoint counts. All four bounded stages complete; own plan archived privately and retired. Incremental commit follows; no PR/push/merge or broad master closure.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed shared server finder freshness at exact native creation, input-admission and result acknowledgements. Cancelled obsolete pending reads, rechecked original account lifetimes, refreshed invalidated remounts, and qualified existing-profile/global mirror origin before paging and merge. Full and Compact canonical reopen remain read-only. Connected 35 affected suites/1838 tests aggregate, both types, zero introduced static findings, current webpack/tokens/budgets and two independent source reviews pass. Current native Full/Compact two-turn finder/reopen/cold proofs preserve all original canonical responses. Normal exact-owned repository hooks pass; no Python source for Bandit. Adjacent Copy, Compact handoff/provider and count clarity plus original broader queue remain open.
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
