# Combined conversation discovery — TASK-13455.5

Baseline `8ea7af620c`, latest `origin/dev 025627214c` integrated without conflicts. Approved design `Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md` governs the one-query and visible-success mirror rule. Post-rebase cached-dialog settings/composer/assistant check: 3 suites / 104 passed. Reuse current sidebar source preference, ServerChatList/LocalChatList, useServerChatHistory, scoped Dexie abstraction and stored-message preview. No new search/index/persistence authority, offline outbox or H1 adoption for cached previews.

All is the fresh default source with separate named Server and On this device groups and independent existing paging; retain explicit saved Server/local/Folders choices. Both use the same query. Scoped cached mirrors join On this device only through an explicit read policy; legacy local reads and H1 guards remain strict. Cached copies are read-only and labelled as stored copies, not live server state. Deduplicate only a proven exact server identity visible in current successful results for the same verified account and query; a stale page, changed filter/query/account, error, pending verification or outage cannot hide the only reachable copy. Preserve source drafts and Temporary view during preview.

## Stage 1: Reproduce query, mirror and scope gaps
**Goal**: red-first tests through current list/DB/query paths.
**Success Criteria**: combined query, stored mirrors, wrong scope, outage/stale visibility, duplicate identities, first verification and storage failures have meaningful reproductions.
**Tests**: mounted source/list tests, scoped Dexie tests and held transport/account callbacks.
**Status**: Complete

## Stage 2: Extend existing source and qualified read paths
**Goal**: complete reachable discovery with honest origin and state.
**Success Criteria**: one query reaches both paged groups; explicit source choices remain; cached mirrors are qualified read-only previews; deduplication depends on currently visible successful server results; existing local selection/H1 boundaries retained.
**Tests**: actual sidebar/list plus DB/search/preview paths; content matches beyond first page, duplicate titles and 2000-row fixture.
**Status**: Complete

## Stage 3: Verify failures, ownership and keyboard behavior
**Goal**: safe recovery and no private stale content/action.
**Success Criteria**: failures are distinct from empty/no matches, retry works without mutation, account/query/page/filter transitions cannot suppress copies or dispatch obsolete selections, nested preview focus returns correctly.
**Tests**: offline/failed refresh/failed next page, held scope/read, profile/account/source ABA, native IndexedDB where available, affected English parity/types/lint.
**Status**: Complete

## Stage 4: Review, record and commit
**Goal**: independent reviewed incremental delivery.
**Success Criteria**: no substantiated Critical/Important findings; task/plan/diff checked; limitations explicit; only previous own completed composer plan removed after archive8ea7af620c.
**Tests**: both authorized independent reviewers, bounded regressions and applicable security validation. Bandit N/A for TS/docs-only production. Native/live/VoiceOver/full-stage gates require later evidence.
**Status**: Complete

Verification checkpoint: qualified red cases9/7 + workspace1 + management4 + feedback1 + final row/busy/policy4 reproduced before repairs. Corrected IO fixtures preserve real stable storage/owner/date contracts; fixture/import failures are excluded from red evidence. Final affected regression14suites403passed. Frontend types and extension entrypoint compile passed (not packaged-build acceptance). Native Chromium production-method IndexedDB harness:2002 qualified histories,67pages,median9ms/p9512ms on this machine,content search1ms, exact successful-visible dedup and retired-copy recovery,zero foreign exposure/preview writes; local H1 policy unchanged. English sidebar parity73keys; zero added lint findings across29touchedTS/TSXfiles. Both independent final reviewers found no remaining substantiated Critical/Important introduced issue; their review did not rerun checks and explicitly retained adjacent and native/live evidence limits. Bandit N/A: production changes are TS/TSX/JSON; temporary verification Python scripts are not repository production edits. TASK-13455.6 tracks the existing nested Folder/Tag operation continuation risk; it remains the next unit and Stage2 stays open. Native UI geometry, default-motion focus, VoiceOver, live-provider and packaged gates are not closed by this harness.
