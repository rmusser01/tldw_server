---
id: TASK-13593
title: Restore current chat readiness and greeting fixture contracts
status: Done
labels:
- chat
- ux
- tests
dependencies:
- TASK-13584
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The latest-dev integration investigation proved six obsolete PlaygroundChat.server-load-state expectations fail at exact branch HEAD7f8a472f before upstream changes. Refresh the query readiness and responsive spacing doubles to the current authoritative catalog; migrate greeting expectations to the real current composer/picker behavior without removing ownership, mirror-isolation or loaded-conversation coverage. Preserve the existing public first-message-only greeting policy and qualify connected current-host controls. This is a bounded test repair in the already authorized all-ten root-cause queue, not a production rollback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Catalog readiness and responsive empty-state fixture expectations reflect actual production contracts and all original readiness assertions pass.
- [x] #2 Fresh owned greeting, foreign mirror isolation and saved owned conversation controls exercise the current real composer/picker host without appending a new opening to loaded history.
- [x] #3 Affected connected tests and introduced static/format/hooks pass; exact-HEAD diagnostic failures and fixture reach limitations remain recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new durable rule. ADR049/050 and the accepted explicit greeting/owner design govern; ADR059 governs tracking. Search found no existing server-load-state repair task; explicit13585 follows highest13584 across local/origin-dev/open-PR files. Fresh origin/dev remains2a935383ac1305fac3af26f2596c203b31a29909. Original six failure proof: /private/tmp/chat-latest-dev-13584-exact-head-baseline.log and exact-head-baseline-proof.json. Real PlaygroundForm.role-play-starter integration already has UAT355 plain-mirror and owned-context controls; extend/reuse those actual-host controls instead of introducing another Form harness. No source/test edits until current13584 compiler/build qualification finishes.
Stage1 complete: exact HEAD proves six inherited failures; query fixture omits isSuccess/isFetching now required by useChatModelCatalog, spacing asserts pre-responsive classes, and greeting expectations target PlaygroundChat after the real picker moved to PlaygroundForm. Existing actual Form UAT355 controls are present but assert context only. Stage2 will preserve original readiness assertions, verify the Chat host does not own the opening control, and strengthen actual Form/picker/hydration controls for fresh owned, foreign mirror and saved owned cases. No production source rollback; SDK/scope/controller fixtures remain declared. Plan Docs/Design/IMPLEMENTATION_PLAN_chat_current_fixtures_13585.md is staged in three bounded steps. Source edits wait for13584 build to finish.
Stage2 qualified: nine connected frontend suites301/301 actual exit0 with Node26 and house frontend config/maxWorkers2. Original six Chat fixture failures are repaired against isSuccess/isFetching, responsive spacing and real composer greeting ownership. First broad run195PASS/3FAIL was diagnostic: two unchanged queue cases reproduce at exact committed d46e2b99f (baseline exit1; edited fixture restored byte-exact). Actual disabled UI showed invalid non-64-hex account-a draft owner causing creation-recovery refusal; valid declared owner hash and matching default-card storage locator fix the fixture, not production. Queue diagnostics showed saved native Character uploads correctly refused before live preparation. Held-preparation controls now use acknowledged seeded local ownership; an added actual Form native-upload negative preserves this protected contract and exact source draft. Fresh owned real picker/hydration is enabled; loaded owned greeting uses a nonempty transcript/history and remains unchanged; foreign mirror produces no greeting read or append. Real creation recovery/picker/hooks/queue leaves qualify alongside declared SDK/controller/Dexie-engine fixtures. Static two TS27→27 zero introduced; owned-statement Prettier keeps emitted AST equal. Evidence /private/tmp/chat-current-fixtures-13585-{final-connected.log,queue-head-baseline.log,queue-head-baseline-proof.json,disabled-cause.log,causal-green.log,static-final.log,format.log}. Stronger UAT failure and diagnostic cwd error are excluded. No production code/build changed; no broad NN/G/native/AT/package/study closure. Stage3 reviews/types/hooks/commit pending.
Stage3: both approved independent reviewers source-only bounded CLEAR. Normal WebUI and extension compile profiles actual exit0; two TS27→27 zero introduced lint; 9 connected301/301; emitted-AST-equivalent owned formatting and diff check pass. Bandit N/A TS-test/Markdown-only, no authored Python/production behavior. Native layout investigation is separate: current1280×900 runtime tab covers560 square px of Recover drafts and right-end hit reaches runtime. No layout fix or all-ten closure inferred. Fresh origin/dev advanced during final recording to590e7e5794224028c760a42bc5f031562ce7a6c3 (API contract package PR3219); it is not yet in HEAD and is the next bounded integration before layout code. Normal owned hooks and incremental commit are the final recording steps; own plan retires privately only after three stages complete.
Identity repair under TASK-13592: upstream dev independently used TASK-13585 for LLM accounting. This completed chat fixture record now uses collision-safe TASK-13593. The canonical backlog-py parser/frontmatter editor/atomic writer preserved every description, criterion, note, summary and completion marker. Historical evidence paths and commit13585 references remain original evidence locations.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Restored the six current-host fixture contracts without rolling back production: authoritative catalog readiness, responsive spacing, actual Form greeting hydration and mirror/saved-context isolation. Corrected the invalid synthetic owner hash and eligible local held-upload fixtures; retained native Character upload refusal and nonempty saved transcript/history. Nine connected suites301/301, both normal client compilers, zero introduced lint and both independent source reviews pass. No tests disabled. SDK/controller/storage/submission boundaries remain doubles; no actual generation, packaged extension, physical AT, participant or all-ten claim. Latest newly fetched dev590e7e5 integration and confirmed rail obstruction follow next.
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
