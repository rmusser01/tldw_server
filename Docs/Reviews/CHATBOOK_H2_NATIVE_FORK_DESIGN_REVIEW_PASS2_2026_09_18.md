# H2 native fork design review — pass 2

Date: 2026-09-18. Independent requesting-code-review seat; follow-up to `/private/tmp/chatbook-h2-design-review-1.md`.

Reviewed worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design`, branch `codex/chatbook-h2-native-fork`. Source baseline remains `ac76c4bc5b035561bf816009c1326a114e87def9`.

## Verdict

**R1, R2 and R3 are closed in the revised design and implementation plan. No remaining concrete P1/P2 finding in this focused re-review.** The design can proceed to implementation. This verdict does not qualify any H2 production behavior, SQLite/PostgreSQL transaction, browser flow, storage race or parity gate.

Read the actual revised `Docs/Design/2026-09-18-chatbook-h2-native-fork-design.md` and the now-present `IMPLEMENTATION_PLAN_chatbook_h2_native_fork.md`, including the applied quota L/N correction and generic native bundle descriptor. Rechecked the relevant original source rather than treating the proposed features as implemented. No applications, tests, database operations, repository edits, Git mutations, main checkout or UAT changes were performed. Only this temporary review report was written by this review seat.

## Finding dispositions

### R1 — recoverable soft-deletion assets: closed

The design at line 85 now separates receipt finality from asset lifetime. Single, bulk and workspace soft deletion burn the creation receipt and deny ordinary access while preserving recoverable claims, bytes and their storage charge. Restore reactivates that retained state without making the old creation key replayable. Hard purge or an explicit irreversible retention action releases claims. Section 8 at line 146 repeats the distinction.

The implementation plan's Task 1.2 at line 170 requires this in the deletion/restore adapters and explicitly tests cleanup between soft deletion and restore. Task 2.1 at line 216 keeps adopted assets on the same reclamation machinery only after hard purge/final claim release. H2-A4 at design line 177 includes cleanup-before-restore and hard-purge qualification.

This matches the existing recoverable conversation lifecycle (`tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py:1248-1255,1318-1325`) without confusing restoration with retrying the creation operation. The previous path from trash to irreversible external-asset cleanup is no longer allowed by the contract.

### R2 — stale physical publisher after cleanup: closed

The design at lines 146-148 now requires a stable native per-candidate filesystem guard outside the directories being deleted. Every chunk and final publisher acquires it, checks durable generation/candidate state, closes the DB transaction and performs I/O under the guard. Cleanup commits non-adoptable `reclaiming` first, closes the transaction, then drains the same guard and deletes both final bytes and upload chunks before finalizing. A writer arriving afterward must observe the durable rejected/reclaiming/discarded state and cannot recreate files.

The DB/filesystem ordering is explicit: no DB lock is held while waiting for the guard, and no byte I/O occurs inside the DB transaction. This avoids exchanging the late-publication bug for a cross-lock deadlock. Existing store adoption still serializes against `reclaiming` under the operation/candidate transaction.

Task 2.1 at lines 182-216 names the actual source helpers, specifies the shared guard and demonstrates the correct reclamation sequence. It also handles the fact that `LocalSyncBlobStore.discard_upload` suppresses removal errors by verifying that the exact upload directory is absent before completion. The plan requires barriers before chunk publication and final rename, process-death/retry cases, and real temporary byte roots with native ownership assertions.

The source mismatch identified in pass 1 is explicitly addressed: `Sync/v2/blob_store.py:122` publishes independently of the GC-only lock at `:275-285`, so the design no longer assumes that existing GC lock alone fences publication. It adds a narrow native wrapper while preserving the byte machinery.

### R3 — existing quota authority and accounting formula: closed

The revised design at lines 150-152 places the aggregate budget in `StorageQuotaService`/AuthNZ and adds an operation/candidate-keyed reservation ledger before native writes. The ChaCha intent and external reserve/release boundaries have same-key reconciliation; adoption retains the existing charge, and release requires guarded, confirmed removal. Missing or uncertain owner state cannot free a charge. This keeps quota work outside the native chat commit.

The final applied accounting formula is sufficiently explicit:

- L is legacy usage excluding the new managed `chat_assets` tree. Persisted legacy counters and their caches contain L only.
- N is fresh active native ledger usage. Admission and public usage summaries compute L+N exactly once.
- Recalculation stores L, returns the public aggregate separately, and never writes L+N back into the L counter.
- Native reserve/adopt/release do not also send native byte deltas through existing `update_usage`.
- Applicable user/team/org pools follow the same split.

This closes the double-counting ambiguity found during the second pass. It matters because the baseline `storage_quota_service.py:131-151` stores its calculated total into `users.storage_used_mb`, while `:201-215` later reads/caches that field. The revised contract deliberately changes the stored component and explicitly adapts public serializers; simply adding N to both existing code paths would not meet it.

Task 2.2 at lines 222-238 identifies the service, AuthNZ migration/row adapters, SQL-owning reservation store and real-fixture tests. Its final test sequence covers cache-prime → reserve → publish → recalculate → reserve/release for user and applicable shared pools, with unchanged aggregate usage across recalculation and no double charge. It also covers response-loss replay, adopted/soft-deleted charge retention, native concurrency and reserve/release crash boundaries.

The guarantee is now calibrated to the actual scope. Native reservations serialize against other native reservations and are visible to existing preflight/accounting. The design explicitly preserves and discloses existing non-native check-then-write races; it does not claim a new universal hard quota or require a broad quota-platform replacement.

## Additional checks

- Source-independent committed resolution, authenticated namespace/digest checks, direct-owner receipt RLS, permanent accepted-key tombstones and no source/child cascading receipt FK remain intact.
- The retention boundary continues to authorize a new native uploaded copy explicitly. No external source ID, URL, job/draft handle or file hash becomes byte authority or a fallback owner.
- The final design at lines 166-168 now gives every H2 child a native bundle/required-projection descriptor, including neutral children without a snapshot or attachments. Cold canonical settings are therefore not contingent on character binding or a local browser receipt.
- Current native bundle/character/asset requirements remain guarded on typed read/write/send paths; legacy pending native operations are not converted into atomic retries. No additional live-card, foreign-owner or legacy mutation fallback is introduced by these fixes.
- Tests listed in the plan are future obligations. The document still distinguishes an ordinary/image milestone from rich asset/character qualification and overall parity.

**Assessment:** No further concrete P1/P2 repair is required by this review seat before implementing the revised contract. The planned real-store, real-byte-root, actual-writer and browser tests remain the evidence needed to claim implementation completion.
