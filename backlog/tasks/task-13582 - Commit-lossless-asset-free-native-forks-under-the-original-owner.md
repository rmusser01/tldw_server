---
id: TASK-13582
title: Commit lossless asset-free native forks under the original owner
status: Done
labels:
- chat
- ux
- h2
- persistence
dependencies:
- TASK-13581
- TASK-13261.4
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue accepted chat remediation and H2 design: implement the existing private store atomic commit and original-key receipt recovery for neutral asset-free text and closed inert tool replay. Keep public capabilities/routes disabled until complete H2 qualification; refuse assets and character bindings rather than dropping retained data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Commit and same-key resolve return a strict immutable original child/message map with owner/request/child binding validation and no mutable-source replay.
- [x] #2 An active caller transaction validates original request and live attempt, coherently reprojects retained source content/context, and atomically inserts independent graph/settings/protection/bundle/metadata/receipt.
- [x] #3 Closed inert tool calls and empty results retain truthful house placeholders, remapped IDs/pins and fresh child-owned settled graph provenance; source admission is never copied.
- [x] #4 Unsupported assets, native source claims/references/bundles, character binding and non-discarded candidates fail before child allocation; stale/expired/replaced attempts and workspace closure cannot commit.
- [x] #5 Real SQLite and official PostgreSQL fixtures where available qualify rollback, exact replay, concurrent key, source mutation/deletion and child tombstones with meaningful existing connected controls.
- [x] #6 Bounded independent reviews, formatting/static/security checks and normal owned hooks pass; task/plan/ledger record remaining public, asset, character and usability gates.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: existing accepted ADR049 source-owner boundaries and ADR050 native storage lifecycle govern; ADR059 canonical tracking applies. Reuse approved Docs/Design/2026-09-18-chatbook-h2-native-fork-design.md. Two independent reviewers approve this internal subset with receipt validation, coherent retained comparison, checked same-connection metadata writer, unsupported source inventory and active transaction as required seams. Do not edit or retire historical parent plans; primary checkout and unowned browser untouched.
Own five-stage plan Docs/Design/IMPLEMENTATION_PLAN_chat_native_fork_commit_13582.md. Fresh origin/dev39b53a876f2cfefcede02c92ae7dacf88f7a98ec included; local13581/openPR13534/dev13560 maxima checked before explicit13582 allocation.
Causal8 missing-commit failures and6 corrupt-receipt failures precede production edits. First green13pass/14officialPGskips exposes one fixture getter-shape assumption; corrected settings[settings], no production fix. Subsequent real store lifecycle60pass/63officialPGskips. Source/independent review finds initial legacy PG guard rejection: scoped wrapper state differs from legacy db._local. Guard-only1fail/1pass RED verifies it; bind actual current raw checkout plus same DB and positive tx_depth. This guard check is not physical PG qualification. Unsupported carriers and rolled-back writers preserve original state. More legacy/character/workspace/concurrency/empty-assistant controls follow.
Final supported Python3.12.13 connected seven-file run:304 passed/180 official PostgreSQL fixture skips/7 warnings367.70s, exit0; no physical PostgreSQL claim. Full-green46/44 precedes five additional final boundary tests. Causal legacy PG guard RED and two native creation marker RED checks fixed; both authorized final source reviews bounded CLEAR. Three-owned-file Ruff empty, production Bandit0 findings/0errors, full store/new test Black and owned range existing fixture Black pass; inherited existing whole-file formatting excluded. Public client/routes/capabilities unchanged. Ledger records remaining assets/quota/frozen character/protected routes/client/public/AT/study gates. Own plan stages1–4 complete,5InProgress; normal exact-owned hooks and incremental commit next.
Normal exact-owned hooks pass without bypass. All five stages complete; own plan retired to /private/tmp/IMPLEMENTATION_PLAN_chat_native_fork_commit_13582_completed.md. No unrelated artifacts/primary checkout/runtime/browser changed. Incremental commit follows; physical PostgreSQL, public H2 and all-ten acceptance remain open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Private native store now atomically creates one independent lossless neutral asset-free child and strict original-key receipt, preserving metadata/closed inert tools/pins/fresh graph and refusing unsupported carriers before allocation. Supported connected304pass/180officialPGskips; bounded reviews clear, Ruff/Black/Bandit and normal owned hooks pass. No public route or capability enabled. Remaining physical assets/quota/frozen character/protected protocol/client/usability gates stay open.
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
