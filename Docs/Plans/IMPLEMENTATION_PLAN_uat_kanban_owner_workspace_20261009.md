# Kanban owner workspace boundary (2026-10-09)

Associated task: TASK-13260.278.18.83.45. Root recorded the actual multi-user held-editor failure before edits. Alice and Bob have distinct owned board/card UUIDs with numeric IDs 1. Backend list/detail ownership is correct; no stale write was attempted live.

ADR required: no. [ADR-020](../ADR/020-db-management-per-user-paths-and-content-backend.md) governs per-user content ownership. This correction reuses canonical connection authority, existing scoped request/abort support and QueryClient ownership; it changes no API, storage, authentication or durable rule.

## Stage 1: Attribute and demonstrate
**Goal**: Trace every Kanban query, mutation, nested detail, archive Undo, import and export path; retain exact pre-unit source in memory and demonstrate owner retirement through mounted UI and real transport.
**Success Criteria**: A bounded causal red reaches the held editor/cache/request boundary; same-owner and legacy controls remain meaningful.
**Tests**: Real QueryClient/AntD/service/request-core with external fetch fixtures; owner/account/server/unavailable changes, duplicate numeric IDs, pending/late transport.
**Status**: Complete

## Stage 2: Repair the local boundary
**Goal**: Key the complete workspace by canonical authority; scope all query keys and API requests; retire asynchronous UI effects and external Undo callbacks.
**Success Criteria**: Requests capture current same-owner config and a retiring workspace signal. Nested pagination/fanout cannot dispatch after retirement. Old ACK/error/download cannot affect a successor.
**Tests**: Causal green plus nested comments/checklists/labels, archives/import/export, exact payload/refusal, token-refresh and single-user controls.
**Status**: Complete

## Stage 3: Qualify and hand off
**Goal**: Run genuinely affected UI/Web suites, scoped baseline/current lint and syntax; freeze exact source/test hashes for independent review and root combined typing/build/live.
**Success Criteria**: Source review is clear; combined typing/build and held-editor live recovery are complete.
**Tests**: Normal configured Vitest, Node20.19.5, 8192 MB, outer600/case300; scoped ESLint and TypeScript syntax. Bandit N/A (TypeScript only).
**Status**: In Progress

## Scope and design
Nine production paths: services/kanban.ts; new Kanban scope helper; index.tsx, BoardView.tsx, CardDetailPanel.tsx, ChecklistSection.tsx, LabelManager.tsx, ArchivedItemsDrawer.tsx and ImportPanel.tsx. Existing normalization, request bodies and named service exports remain compatible. No generic transport changes.

The outer wrapper uses useCanonicalConnectionConfig and deriveConnectionAuthorityId. The local provider owns an abortable client and owner-prefixed keys; latest same-owner config is captured on each request. Child callbacks and retained Undo/file readers check retirement. QueryClient data from an old owner cannot be used under a successor's numeric IDs.

## Attempt ledger and limitations
Maximum three failed repair attempts per issue, then STOP and source-backed reassessment. Initial write orchestration had a JavaScript template-literal quoting error before tool execution; no files changed and no tests ran (operator failure, not causal red).
No live runtime/browser/provider/profile/Git/proof action. Root owns combined types/build and actual held Alice-to-Bob recovery. Frontend fixtures do not establish backend HTTP authentication or real browser storage-event delivery.

### Initial baseline STOP/reassessment
1. Normal UI command completed, but full JSON capture was tool-truncated; no retained natural exit/count/assertion evidence. No causal claim.
2. Bounded verbose extraction: natural1/8.440s, five failures including same-owner control. Error-type extraction omitted Testing Library message. No causal claim.
3. JSON reporter: natural1/8.374s, 0passed/5failed, all fail before owner assertions at findByRole(dialog,name:/Edit Card/). STOP.

Installed @rc-component/util1.7.0 returns literal test-id in NODE_ENV=test. AntD Drawer uses it for title/aria-labelledby on the full page, where real Select/overlay IDs collide. Repo alternatives: OsceResultsPanel171 and ItemsWorkspace264/309/382 use unique text → closest actual dialog; ACP tablet148–155 identifies unique content. The supported fixture correction locates unique Edit Card title then its closest role=dialog and verifies private description/comment/visibility. Retirement checks the exact original node. No ID/event/mock/timeout change or assertion weakening. Root explicitly released ONE distinct corrected-fixture causal phase under standing human authorization; the original three-run STOP remains recorded and is not a cap reset.

The initial fixture-edit guard expected five textual openCard() calls, but the five leaves are represented by four textual calls (one it.each). It failed before any write/test; exact source inventory corrected the guard to four.

Pre-unit ESLint bounded extraction natural0/1.290s: eight existing production paths, 0errors/14 inherited warnings. Earlier full ESLint source payload capture was truncated and is not qualification.

Corrected-fixture causal command: natural1/7.008s, four source failures at exact original Alice dialog retirement (account/server/unresolved/pending save), one same-owner refreshed-token exact PATCH control passed. Initial visible/private description/comment and saved Alice URL/token reached before source assertions.

Production orchestration initially wrote the scoped service factory then stopped at a guarded import matcher; no test ran against that incomplete state. Corrected matcher completed seven consumers/new provider without rewriting the factory. This is an operator edit failure, not a failed behavioral repair.

Added meaningful connected controls before first green: nested checklist/labels/comments numeric collisions; admitted and unadmitted import; retired export/download; archive drawer/retained Undo; comments-page/checklist-detail continuations; single-user/version headers; current-owner404 refusal. Fixtures map response owner from actual dispatched Authorization, not mutable current-account state.

First repaired-source UI new module: natural1/15.056s,14passed/1failed. The sole new archive Undo fixture fails before retirement at finding the actual Undo button. All four causal owner controls and ten connected/compatibility controls passed. Next bounded command targets only this unqualified case; no14-pass replay. First current lint0errors found six new missing callback dependencies; correct dependencies while retaining original API bodies.

Sole Undo attempt2: natural1/3.944s. Added actual archive dispatch/message.open oracles; actual visible Undo was reached, then same notice remained during JSDom exit animation. Installed static-message GlobalHolder renders outside local App/ConfigProvider, so local motion:false does not apply. Correct only this fixture with canonical ConfigProvider.config holderRender using real motion:false; restore configuration/destroy messages afterward. No DOM/style/event mock or timeout increase. Attempt3 targets sole unqualified Undo,14 passes retained.
Scoped new dependency warnings are corrected by including the local stable client/isActive/key callbacks; API and validation bodies unchanged.

Undo attempt3 sole UI: natural0/2.860s,1passed/14filtered;15 distinct UI owner-module passes are retained by14+1 split, not a combined15 execution.
Five affected original modules UI: natural0/3.399s,25passed/no skips.
First configured Web six-module execution: natural1/14.975s,39passed/1failed. Sole Undo could not open after earlier static messages. STOP/reassessment: three failed executions for this issue (first UI open, second UI exit, first Web open), with one sole UI pass retained.

Installed AntD GlobalHolderWrapper reads global holderRender during sync and changes its React tree; flushMessageQueue captures newMessage.instance only if absent. Switching holderRender after prior notices can remount the holder while preserving the old cached API, explaining why the isolated first-message case passed and mixed lifecycle failed. Alternatives read: existing WritingPlayground/Items/Skills message.destroy cleanup; mounted motion:false ConfigProvider pattern; installed canonical ConfigProvider.config global static-holder pattern. The smallest distinct fixture correction establishes one stable holderRender before any static message (beforeAll), destroys notices afterEach, restores global config only afterAll. No production, assertion, event, geometry or timeout change. Root coordination was sent. Standing all43 fix/validation authorization covers this distinct fixture-only stable-holder phase after documented STOP/reassessment; no cap reset or unchanged run. Any further failure stops this path.
Current production/newtest lint0errors/13 inherited warnings (baseline14, removed unused bgUpload),0new; four existing fixture baseline/current0errors/12samewarnings. AST equivalence confirms all48 original service/helper function signatures/bodies unchanged,42APIreturned exports matched. Fourteen touched TS/TSX syntax checks0errors.


## Frozen qualification handoff
Distinct stable-holder fixture-only Web sole case: natural0/3.199s,1passed/14filtered. Preserve39 first-Web passes;40 distinct Web controls by39+1 split, not a combined40 run. UI40 distinct controls by14+1+25 split, not a combined40 run. The final holder setup moves the same real motion:false wrapper to beforeAll and restores only afterAll; all original assertions/interactions remain. Prior three-failure STOP and source-backed reassessment remain visible.
Normal configured UI/Web Vitest with Node20.19.5/8192 MB/outer600/case300, no worker/config/cache override or dependency changes. Fifteen new owner controls and25 existing affected controls (empty-state2, due-date5, archive1, checklist4, shared service13); no unchanged date helper or route-only suites replayed.
Final new-test scoped ESLint natural0/0errors/0warnings and TypeScript syntax0 after holder correction. Production/helper syntax retained9 zero, existing-fixture syntax4 zero. Scoped lint production13 inherited warnings vs14 baseline; existing fixtures12 same warnings; no new findings. Bandit N/A TypeScript-only. All48 pre-unit service/helper function signatures/bodies compare equal through TypeScript AST printing,42API exports returned exactly. Unchanged BoardGallery byte hash393613b89bbcfe8f5c91644002b12724e506d630aea48678f6aa341887f8dd84 retained.
Production and all test bodies frozen for independent actual-source review. Root's combined configured types/build and real held-editor recovery remain open; Stage3 stays In Progress. No all43/backend auth/real browser event-delivery acceptance claim.
