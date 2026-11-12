# Original typed retention reservation — TASK-13606

The generic operation store accepts retention metadata but does not establish a strict frozen request/source binding. This private typed adapter validates the original operation intent and admits initial/resubmitted preparation safely. It creates no claims, lease, quota, bytes, ready bundle or public capability.

## Behavior

NativeForkStore.reserve_retention_operation requires the actual owning transaction, strict model revalidation, recomputed semantic digest and exact authenticated client/owner/scope. The full canonical request and source_conversation_id are frozen at INSERT. Owner/key locks precede workspace/conversation source locks.

Existing keys resolve before mutable source admission. An absent observation is followed by INSERT ON CONFLICT and a real row reread; a concurrent winning committed/terminal result still resolves first. Committed receipts use existing strict original-request/result validation; minimal permanent terminal tombstones remain resolvable without source/canonical body. No new key or current source is substituted.

Preparing admission requires exact original canonical/source binding and supported projection, then compares the original incarnation-bound source revision. Unqualified generic/legacy preparing rows are not silently adopted. Stale/deleted/recreated/protected sources refuse resume without deleting/rebinding a prior key or advancing its generation. Generic resolve remains available to report pending independently of current source. Lookup and resume are distinct; later publication must repeat source/Sync/lease/byte admission.

A local savepoint contains rejected initial reservation and preserves unrelated caller work even when the caller catches the error. Existing preparing state remains frozen. Source deletion has no new pending-operation deletion hook; pending keys stay reconcilable and source change never implies rejection/expiry or permission to release live bytes.

ADR required: no new decision; ADR049/050 and proposed071 govern immutable keys/source identity. No schema, public API or representation admission change. Native-source bridge, active Sync policy, actual preparation/lease/quota/claims/publication and selected-view/client admission remain pending.

## Verification

SupportedPython3.12.13 after project environment activation, official PostgreSQL fixture TLDW_TEST_NO_DOCKER=1; no alternative provisioning.

- Causal10failed/10officialPGskip/12warnings5.41s at missing typed method; green10pass/10skip12warnings5.00s after implementation and lookup/resume distinction.
- Initial five-suite connected151pass/136skip12warnings52.11s preceded a reviewer-found unsupported preparing projection. Actual real-row causal1fail/1PGskip/30deselected12warnings1.43s; preparing binding now checks supported native-fork-v1, terminal lookup unchanged.
- Final five-suite connected152pass/137skip12warnings50.88s, exit0: typed reservation, structural receipts, ordinary source revisions, operation transactions and actual fork commit.
- Actual concurrent same-key callers converge on one row. SQLite exercises real BEGIN IMMEDIATE serialization; PostgreSQL control forces both actual absent-key observations before INSERT, but is officially skipped here. A separate injected stale absence over a real seeded committed row verifies conflict-winner reclassification before a poisoned source reader. This is branch evidence, not fabricated concurrent publication.
- Exact request/source/digest frozen once, changed owner/request/digest/raw input refuse without mutation. Actual source deletion and history mutation refuse resume while generic lookup preserves pending. No generation/candidate/claim/quota work is granted. New stale-source rejection rolls back only its reservation and preserves actual caller conversation work.
- Seeded committed/gone receipts resolve with source reader poisoned; minimal gone tombstone lacks original body/source/target yet remains owner/digest-resolvable. Seeded receipts are structural fixtures and do not prove a retention publisher or bytes.
- Final two-file direct and matched-filename Ruff0/0; new method formatting preserves AST, new test Black check and supported compile bothfiles pass. Actual one-production-file Bandit baseline/current0/0 findings/0 errors; coverage verified. Diff-check passes; no warning/test suppression.
- Both independent final source reviews bounded CLEAR. Reviewers ran no tests.

Physical PostgreSQL execution/concurrency remains unqualified; official unavailable fixture and existing backend-specific controls account for skips. Raw logs/configuration/headers/transcripts remain private under /private/tmp/chat-retention-reserve-13606-*.

No runtime/profile/auth/key/card/DB changes, new browser/window/tab/send or baseline rewrite. Original owned services remain untouched. Public retention recovery/Sync admission, byte/claim publication, source native bridge, original-hash replace/Restore, UI/send integration and all-ten NN/G/installed-extension/Compact/physical AT/real-model/study gates remain open.

Initial normal exact-owned seven-file hooks pass (exit0). House Ruff/Black hook filters skip these paths; meaningful manual checks above pass. All three bounded stages complete; only own plan retires to /private/tmp/IMPLEMENTATION_PLAN_chat_retention_reservation_13606_completed.md. Final normal exact-owned six-file hooks pass (exit0), no bypass. Fresh dev remains a744577356ecd4db2d101689b5564a7a7c3c3f5a with zero missing commits; local commit follows. Master TASK13458/H2 parent13261.4 remain In Progress.
