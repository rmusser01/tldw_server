# Qualified Folders discovery — TASK-13455.7

Baseline866d9176e8; latest origin/dev025627214c freshly verified. The approved Chat UX design, ADR-049/050 and reviewed Stage2 ownership repairs govern this adjacent unit. Reuse the parent ChatSidebar verified owner rather than resolve another identity. Extend the existing folder operation/store, scoped getChat API, React Query and real FolderTree. No new persistent folder authority, service, owner resolver or transfer contract.

Folder names/counts and title lookup IDs require a fresh successful organization snapshot for the captured view. Unqualified cache and different-account data cannot supply current results. Abort retired reads and guard each title batch, selected lookup, create dispatch and feedback. Global scope is explicit; foreign workspace metadata is unavailable rather than selection authority. Retry and partial-title feedback remain reachable; background title loading does not hide the usable folder tree. Temporary uses existing H1 selection while management is disabled imperatively, not only with pointer-events CSS.

A create action holds a separate dialog feedback token within the qualified view: Cancel prevents obsolete feedback or closing a later dialog. A dispatched create may settle on its captured account; no undo or cross-account replay is claimed. Account/New/view retirement also prevents its store ACK publication. Current failures retain the name and explain retry; null ACK cannot claim success.

## Stage 1: Reproduce current gaps
**Goal**: red-first behavior checks with mounted FolderChatList/real tree/store/query and API/getChat IO boundaries.
**Success Criteria**: old cached names, missing account scope, late selected read/title batch, false success on failed create and Temporary keyboard management reproduce their gaps.
**Tests**: meaningful mounted interactions; qualified parent count/prop wiring; store helper checks only if needed.
**Status**: Complete

## Stage 2: Repair existing read/action lifetimes
**Goal**: useful qualified Folders discovery and creation.
**Success Criteria**: fresh names/counts; captured/global title and selection reads; retired batches/selection/feedback stopped; readable loading/partial/retry/offline states; normal creation and H1 selection work.
**Tests**: red cases pass, replacement account and New revision, real Tree stale handler, read retry and held Cancel/create failure; no authority from old cache.
**Status**: Complete

## Stage 3: Verify and independently review
**Goal**: reviewed shared WebUI/full-extension behavior.
**Success Criteria**: relevant regression, WebUI typecheck/extension compile, zero added touched lint, English mirror parity and two independent read-only reviews. Bandit N/A unless Python production changes become necessary.
**Tests**: new mounted tests plus prior sidebar discovery, scoped transport and organization regressions. Test doubles and native/live evidence boundaries recorded.
**Status**: Complete

## Stage 4: Record and commit
**Goal**: reviewed local incremental delivery with accurate remaining work.
**Success Criteria**: normalized task/final summary/plan; only owned changes staged; parent Stage2 stays open for complete native/live acceptance. Previous owned .6 plan may be removed after archive866d9176e8.
**Tests**: diff/task checks and local commit without push/PR/merge.
**Status**: Complete

Final evidence: 10 suites/115tests; WebUI types and extension entrypoint compile; zero added lint11TS/TSX; nine new English source/packaged keys; clean diff; two final independent reviews. Meaningful organization-refusal red2 then green preserves the existing observer. Bandit N/A to TypeScript/JSON/docs-only production scope. Real store/query/tree/AntD with reduced test motion; parent count/props and real getChat transport up to stated IO doubles. Actual H1, packaged/native default motion, VoiceOver, native UI scale and live provider/cookie checks remain later program gates. Previous owned .6 plan archived866d9176e8 and removed.
