# Native first-create recovery integration — TASK-13454.6

Baseline cdedb99ef6; latest fetched dev025627214c integrated. Reuse existing pending-chat-promotion and local registry, chatDraftOwnerKey, fresh addresses, composer recovery, H1 loading and fences. No replay queue, new store or authority. Unknown ordinary UUID creation has no universal replay guarantee.

## Stage 1: Reproduce actual first-create loss
**Goal**: mounted native first-send demonstrates duplicate creation after a lost acknowledgement and no retained late ID.
**Success Criteria**: red cases show actual dispatch count, private original text, cold/new/account boundaries.
**Tests**: existing mounted H1/draft/model/tools fixture plus actual registry/storage shim with lock/protocol doubles.
**Status**: Complete

## Stage 2: Retain and gate dispatch
**Goal**: durable first-create attempt before POST; late ACK retained without stale binding.
**Success Criteria**: fail-closed availability and exact source key; no unknown replay; no credentials/owner grants in records; existing H1 fencing retained.
**Tests**: concurrent claims, failed/late ACK, repeated/cold attempts and drift.
**Status**: Complete

## Stage 3: Present qualified recovery
**Goal**: existing composer recovery reveals uncertain creation and original input; explicit known-ID review uses H1 qualified loading.
**Success Criteria**: no POST/provider in review; copied text cannot silently bypass uncertainty; empty greeting attempts remain reachable.
**Tests**: UI/copy provenance and existing composer recovery regressions.
**Status**: Complete

## Stage 4: Verify and commit
**Goal**: reviewed actual integration with bounded claims.
**Success Criteria**: relevant tests/types/entrypoint compile/no added lint/diff/two reviews. Ordinary promotion, greeting-only startup, Sync/native/live/full builds remain separately required.
**Tests**: Bandit N/A if production remains TypeScript only.
**Status**: Complete


Verification: mounted first-create regressions initially failed five cases; recovery UI failed two cases. Review found additional receipt lifetime, held-write/GET navigation and displayed-source ABA gaps. Meaningful final race evidence is `/private/tmp/native-creation-review-races-red.log` (four failures), `/private/tmp/native-creation-observer-aba-red.log` (one failure), and `/private/tmp/native-creation-retention-race-red3.log` (two failures). Earlier retention logs contained fixture errors and are not cited as behavioral proof. Storage spies are now restored between tests; this fixes recursive spy reuse without weakening assertions.

Final ten affected suites295passed; WebUI typecheck and extension entrypoint compile exit0; baseline-relative lint on all touched TS files adds no diagnostics; English asset/public recovery copy matches; git diff --check clean. Logs: `/private/tmp/native-creation-recovery-final-{green3,types3,extension3,lint2}.log`. Both authorized independent source reviewers report no remaining Critical/Important finding. Bandit N/A: TypeScript and documentation only.

The creation record is an owner/scope-qualified locator, never an H1 grant. Prepared reservations can be reclaimed only before dispatch via exact conditional writes. Dispatched uncertainty cannot be replayed. Copied draft aliases follow the original late acknowledgement. Original text stays reachable after binding until verified native settlement; private cleanup failure retains it. Copy and Review fence the displayed source revision and current account. Review retains next input under the same preparation/navigation fence, then loads actual owned history with no create/provider call. The accepted owner's lifetime is independent of the preparation composer/settings/Stop guard.

Evidence limits: serialized lock, storage and protocol doubles exercise actual composer/registry/H1 code; no physical multi-browser race, native product UI, VoiceOver, packaged build, Sync v2, backend lost-response-after-commit or universal replay guarantee. Unknown ACK discovery remains manual and cannot identify the server chat. Later manual sending can conservatively leave a retained record. Ordinary creation/promotion, greeting-only startup, full preflight, P9 and all broader program acceptance remain open. Completed registry prerequisite plan is archived in cdedb99ef6 and removed in this unit per repository policy.
