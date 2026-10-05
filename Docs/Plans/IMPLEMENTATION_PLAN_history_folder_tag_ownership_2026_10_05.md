# History folder/tag operation ownership — TASK-13455.6

Baseline33151cc597; latest origin/dev025627214c integrated. Approved Chat UX design and the TASK-13455.5 review govern this adjacent follow-up. Reuse ServerChatList's captured action, the two actual bulk picker dialogs, existing folder store, folder API and resource client. No new owner resolver, folder service, persistent cache authority or automatic retry.

Carry the captured request scope and current-view predicate from ServerChatList into both dialogs and every nested operation. Require a fresh successful scoped folder snapshot before selection or Apply. Keep its qualifier transient in the existing store; unqualified persisted/Dexie data cannot authorize an owned mutation. Owned calls skip the existing unqualified Dexie cache. Retired operations cannot dispatch another request, publish ACK data, clear a newer selection or show obsolete feedback. Earlier requests already dispatched may settle on their captured account, but are never replayed under replacement credentials. API single-flight keys isolate captured scopes; the real transport retains expected-user checks and only the exact existing folder/tag paths gain permission.

## Stage 1: Reproduce retired continuations and account joins
**Goal**: meaningful red-first tests through actual dialog, store and API transport paths.
**Success Criteria**: held tag creation, first folder batch, stale store ACK/cache, unqualified cached IDs, captured resource requests and cross-account single-flight each reproduce their gap.
**Tests**: actual mounted modal interactions, store with API/DB IO boundaries, folder-api to real bgRequest with credential/fetch boundaries.
**Status**: Complete

## Stage 2: Extend existing captured operation paths
**Goal**: safe useful folder/tag management on verified history.
**Success Criteria**: guarded reads/writes/chunks/callbacks, fresh account-qualified data, no owned writes into unqualified cache, retryable read errors, no widened unrelated transport policy; fresh apply still works.
**Tests**: red cases become green; normal create/tag/folder Apply and current error feedback; replacement-account and narrow policy rejection tests.
**Status**: Complete

## Stage 3: Verify and review both shared entrypoints
**Goal**: independent reviewed incremental delivery.
**Success Criteria**: bounded regression/types/extension entrypoint compile, zero added touched lint, affected English parity, two independent reviewers and diff/task checks. Bandit is required on the touched notes endpoint after the cookie-session guard repair; retain a baseline comparison for existing findings.
**Tests**: actual dialog/store/transport regressions plus prior sidebar management checks. Full packaged builds, native motion/geometry/VoiceOver/live-provider and D7/program gates remain later evidence.
**Status**: Complete

## Stage 4: Record and commit
**Goal**: keep the parent stage accurate and commit only this unit locally.
**Success Criteria**: task final summary records behavior, exact checks and limitations; plan status complete; own previous completed discovery plan may be removed after its archive33151cc597. Parent stage stays open until its remaining gates are verified.
**Tests**: final diff/task normalization and reviewed local commit without push, PR or merge.
**Status**: Complete

Review correction: scoped cookie-session requests rely on the existing backend expected-user assertion. Extend it on the exact folder/tag reads/writes before DB dependency/body execution. Verify actual API responses for changed cookie principal and current/legacy callers. Existing wire collection_id mapping and API paging also need actual-shape transport regressions before claiming useful fresh organization data.

Final evidence: 8 frontend suites/233passed (218+15 bounded runs), 29focused and133existing backend regressions, WebUI types, extension entrypoint compile, no added lint11TS/TSX, five English mirror keys, Black/diff checks and Bandit0 on notes.py. Two independent final reviews pass after meaningful red-first wire/cookie/cancellation/loading corrections. This unit is locally committed with its task; parent Stage2 remains open. Live cookie-session, packaged parent/New integration, default motion/geometry and VoiceOver are not proved. Legacy Folders follows in TASK-13455.7; compact/Option folder consumers are recorded under Stage4.
