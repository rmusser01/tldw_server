# H2 native workspace lifecycle review

Date: 2026-09-18. Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design`, branch `codex/chatbook-h2-native-fork`.

This is a read-only, source-grounded review of the workspace lifecycle boundary before Tasks 1.2 and 4.1. No repository files were edited, applications or tests run, or additional agents dispatched. H2's storage/coordinator behavior is planned, not implemented or qualified by this review. Line references below describe the files read during this review.

## Verdict

**One concrete P2; no P1.** The proposed operation/conversation locking does not serialize native fork or native retention admission with workspace deletion. A short, durable workspace admission closure before enumeration closes the race without turning the existing cascade into one large transaction. The contract below is a bounded repair requirement, not a claim that production already implements it.

### W1 — [P2] Close native workspace admission before enumerating its conversations

**Design locations:** [scope contract:34](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/Docs/Design/2026-09-18-chatbook-h2-native-fork-design.md:34), [deletion/locking:92](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/Docs/Design/2026-09-18-chatbook-h2-native-fork-design.md:92), [final admission:102](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/Docs/Design/2026-09-18-chatbook-h2-native-fork-design.md:102), [context mutation:171](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/Docs/Design/2026-09-18-chatbook-h2-native-fork-design.md:171). Plan coverage belongs in [Task 1.2:171](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/IMPLEMENTATION_PLAN_chatbook_h2_native_fork.md:171), Task 2.3 and [Task 4.1:352](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/IMPLEMENTATION_PLAN_chatbook_h2_native_fork.md:352).

**Failure schedule:**

1. Workspace deletion enumerates the live conversations in W, including source S, then pauses before deleting S.
2. An already prepared native fork locks its operation and S, verifies the unchanged owner/scope/retained content and commits child C plus its receipt in W. The current proposed final transaction never checks or locks W.
3. Deletion resumes against its earlier list. It deletes S but misses C, then marks W deleted. C remains live, with live assets/charges, and its creation receipt remains committed.
4. For hard deletion, the same enumeration gap allows C to survive until the workspace row is removed. The existing foreign key changes `C.workspace_id` to NULL while `C.scope_type` remains `workspace` and the receipt's immutable scope remains W. The intended child deletion/tombstone adapter never ran.

This breaks native scope integrity and the promised workspace deletion/receipt lifecycle. It is not evidence of cross-owner access or automatic conversion to global scope; those stronger claims are unnecessary to establish the P2.

**Source evidence / existing guards do not close it:**

- [ChaChaNotes_DB.py:27619](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:27619) enumerates before the separate message and conversation mutations at 27638/27649; the workspace update occurs in a later transaction at 27652–27667. The initial version check at 27612 is also outside that final transaction.
- [hard_delete_workspace:27677](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:27677) similarly enumerates, deletes individual conversations, then removes W at 27689. Both PostgreSQL's [scope migration:3526](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:3526) and SQLite's [migration:9550](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:9550) use `ON DELETE SET NULL`.
- [chat.py:6410](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/tldw_Server_API/app/api/v1/endpoints/chat.py:6410) verifies conversation existence, owner and stored scope pair, not the parent workspace's current lifecycle. Scope normalization at 6336 does not supply such a check. [ConversationStore.add_conversation:415](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py:415) normalizes the pair and inserts it; the foreign key verifies existence, not open admission.
- Workspace RLS at [pg_rls_policies.py:639](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py:639) is owner-based. The saved-view owner/active row lock helper at [ChaChaNotes_DB.py:27756](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:27756) is a useful implementation precedent, but is not used to fence this native admission/enumeration boundary.

## Required narrow protocol

### 1. Durable closure with an explicit transaction boundary

Add a protected workspace boolean, for example `native_chat_admission_closed`, default false. Keep it separate from `system_operation_state`: that existing field only admits NULL, `staged` or `publication_pending` for shared workspace cloning ([schema:6651](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:6651)). An ordinary visible workspace here means exactly `system_operation_state IS NULL`, as used by [get_workspace:27345](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:27345); H2 needs no new publication state.

Before either workspace deletion enumerates conversations, perform a short **outermost** DB transaction:

1. Lock the owned workspace row (PostgreSQL `FOR UPDATE`; SQLite's existing immediate write transaction). Bind ownership to the authenticated/database principal and compare the actual `client_id`; do not treat a supplied workspace ID as authority.
2. For soft deletion, require an ordinary non-deleted workspace and recheck the caller's `expected_version` under that lock. Wrong owner, missing/hidden workspace or stale version changes nothing. For hard purge, allow an owned ordinary row whether active or already soft-deleted. Today's hard-delete signature has no expected-version argument: retain that existing entry-point contract unless intentionally extended; it still must lock and validate the current owned row. If an expected version is supplied by a future/internal guarded entry point, check it here. Preserve the existing idempotent absent-row hard-purge behavior.
3. Set the admission-closed flag idempotently, then commit and release W before enumeration or any conversation/message deletion. Keep this internal flag outside the public workspace version counter so the existing final `expected_version + 1` transition remains usable. Public metadata changes can still cause the existing final version conflict; they cannot reopen native admission.

The flag is internal lifecycle authority, not a client-editable workspace option. No expiration, failed cascade, metadata update, import, sync, or normal upsert clears it.

**Outermost-only is an intentional internal API restriction.** A nested `with db.transaction()` does not establish this boundary. PostgreSQL only commits at depth zero ([BackendManagedTransaction:628](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:628)); SQLite only commits an outermost transaction ([TransactionContextManager:44682](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:44682)). Both staged deletion entry points must reject an unsupported enclosing transaction **before mutation**. Do not force-commit caller work, use a second connection to evade its transaction, or hold W through the legacy cascade. Any newly discovered production caller requiring an encompassing transaction is an explicit compatibility blocker to resolve, not permission to weaken this boundary.

### 2. One lock order for native admission and scope-sensitive writes

Native final fork/retention admission uses:

`operation → workspace (if scoped) → source/target conversation → native claim/reference/candidate rows`

Acquire relevant operations/rows in deterministic sorted order where there are multiple. Nonlocking reads may discover identities, but validate their owner, immutable scope and bindings again after the ordered locks. Under the W lock require the authenticated owner, `deleted = false`, `system_operation_state IS NULL`, and admission open; hold W until the child/claim/receipt transaction commits. Do not lock a conversation first and then wait for W.

Apply the same workspace-before-conversation guard to H2 asset context mutations and protected chat restore. Those operations need not invent an operation row if their existing protocol has none. A native retention source becomes protected in the same transaction that commits its first typed claims, so final deletion checks include it. Early capture/preparation checks can improve errors, but cannot replace this final transaction check.

The protected native bundle/asset access guard and receipt child reauthorization must also recognize the closed workspace and return its lifecycle denial. This does not require changing ordinary workspace listing or legacy content access. Read-only checks do not authorize reopening; any read path that takes these row locks follows the same workspace-before-conversation order.

The begin-close transaction acquires W only; it never waits for operation or conversation locks. After it commits, the cascade uses the existing receipt-aware `operation → conversation` deletion adapters without reacquiring W after a conversation lock. Destructive deletion, tombstoning and candidate cleanup must remain permitted for closed workspaces; otherwise closure would prevent its own completion. Preserve the existing message-writer ordering, and do not introduce old-message row locks after the conversation lock.

No filesystem work, guard drain, quota service call, network call or provider work belongs inside these database transactions. Existing candidate cleanup and quota reconciliation run after the relevant DB state commits.

### 3. Failure, retry and final protected-row check

After closure commits, run the existing per-message/per-conversation cascade. Preserve ordinary legacy deletion behavior, quiz/deck handling and the post-success sharing hook. This repair does not promise to serialize unrelated legacy concurrent conversation creation.

If a cascade, protected child tombstone transition, or final workspace CAS fails, leave admission durably closed and return a truthful retryable deletion/lifecycle error (for example `workspace_delete_incomplete` with retry guidance). Do not report successful deletion, reset the flag, or imply that every still-existing child's receipt is gone. Keep the owned workspace discoverable for deletion retry while it is not yet deleted. Retrying with its current public version re-enters the already-closed state and re-enumerates the remaining conversations; it does not resume a stale enumeration. A conflicting metadata update therefore cannot strand the user without a deletion retry path.

Before the final workspace transition, reacquire W in a short transaction, revalidate ownership/closure and the applicable expected version, and query for remaining protected conversations. Soft deletion requires no live protected conversation; hard deletion requires no protected conversation, including soft-deleted ones, to remain. Use the Task 1.2 protected binding/projection columns, covering neutral native children and typed-retention sources, not the optional character binding or a browser receipt. If a protected row remains, abort/release this transaction and return the retryable lifecycle error. **Never acquire/wait for its operation lock while holding W** to repair it inline. Re-enumeration and per-child repair happen after releasing W. Only then may the final workspace deletion proceed, preventing `ON DELETE SET NULL` from detaching an H2 child.

This check is a defensive stop for protected rows, not a new general workspace cascade engine. Existing raw/admin destructive paths touching protected native rows must use the same adapter or preserve their immutable receipt/tombstone and scope invariants.

### 4. Restore, generic writes and receipts

There is no workspace reopen/restore API in the current ChaCha/workspaces code. Do not imply one exists or add one to make this repair work. H2 introduces no automatic reopening. Existing [restore_conversation:1318](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py:1318) can restore an ordinary chat; for a protected H2 chat it must first require its original owned workspace to be ordinary, active and admission-open. It cannot reopen W. A successful individual chat restore in an open workspace retains the already-gone creation key, as the design already requires. A future workspace reopen feature needs an explicit lifecycle contract; it is outside this repair.

Protect the immutable native owner/scope/binding and deleted state from generic bypasses. In particular [upsert_conversation_from_sync:567](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py:567) currently overwrites `client_id`, `deleted`, `scope_type` and `workspace_id` on conflict (581–585). Imports/sync/generic writers must not reparent or revive protected rows, strip the protected descriptor, or clear workspace closure. Reject unsupported protected transitions; no new scope-move feature or global fallback is required. This is specific to H2 protected state, not a redesign of ordinary legacy updates.

Keep authenticated namespace/digest and existing-receipt resolution **before source admission/routing**:

- Actual child soft/hard deletion changes its creation receipt to `gone` in the same native transaction. Soft deletion retains claims/bytes/charges; hard purge releases claims through the existing guarded cleanup protocol.
- A still-existing child in a closed workspace is currently inaccessible under that scope. Resolve/replay returns the lifecycle denial without disclosing its child/map, but **must not burn a committed receipt merely because W is closing**. Only actual missing/deleted child evidence supports the existing conservative `gone` repair. Closure is not child deletion.
- Gone/rejected/expired tombstones remain owner/scope/digest-resolvable without a live source, child or workspace. They never enter creation again.
- A preparing fork/retention that loses to closure cannot publish a child or claims. Fence/reject or reconcile that same key using the existing operation protocol; clean up unadopted candidates/charges outside the DB transaction. Do not release any committed live claim solely because the workspace flag is closed.

## Caller and test compatibility

Production-call search covered `tldw_Server_API/app`, `apps` and `Helper_Scripts`. The only found ChaCha soft-delete caller is [workspaces.py:1267](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/tldw_Server_API/app/api/v1/endpoints/workspaces.py:1267); it calls directly, without an enclosing transaction. Its sharing cleanup hook runs afterward at 1271 and should remain outside the database operation. No production ChaCha hard-delete caller was found in those trees. `agent_orchestration.py:1309` calls the unrelated `Orchestration_DB.delete_workspace`, not this API. No real caller requiring a surrounding transaction was found in this inventory.

Two existing delete-first concurrency tests deliberately enclose the entire method and must be adapted, not removed or disabled:

- SQLite: [test_workspace_soft_delete_then_saved_view_mutation_fails_after_serialization:541](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/tldw_Server_API/tests/ChaChaNotesDB/test_workspace_source_saved_views_db.py:541), enclosing transaction at 555 and deletion at 558.
- PostgreSQL: [test_postgres_saved_view_mutations_serialize_with_workspace_soft_delete:655](/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design/tldw_Server_API/tests/ChaChaNotesDB/test_workspace_source_saved_views_postgres.py:655), `delete_first` enclosing transaction at 724 and deletion at 727.

Move their delete-first holding barrier into a test seam around the **short final workspace transition**, after `deleted = true` is written but before that transaction commits. The saved-view mutation must still wait, then fail with `source_view_not_found` and empty metadata after commit, exactly as asserted today. Do not move this particular assertion to the begin-close seam: the new flag fences H2 native admission, while the existing saved-view helper checks `deleted`; ordinary saved-view behavior during the earlier cascade is not being redefined. Mutation-first test barriers around the saved-view transaction can remain. Add separate H2 tests at begin-close, because those prove the new native invariant.

## Focused acceptance evidence required

Use the existing real SQLite/PostgreSQL repository fixtures and deterministic barriers; this review does not execute them.

1. **Admission wins:** hold native final admission after acquiring W; deletion waits. Let the fork commit, then close/enumerate/delete. The child is included; its receipt becomes gone; soft-delete claims stay recoverable/charged, and hard-purge cleanup follows the existing contract. No protected child ends with a NULL workspace.
2. **Closure wins:** pause after begin-close commits but before enumeration. An already-prepared fork and native retention each fail final admission without a child/new claim. Their same keys cannot create after closure; candidate reconciliation remains safe. Also cover the context mutation/restore guard.
3. **Failure and retry:** inject failure just after closure, during per-child cascade and at final workspace version CAS. Closure persists, surviving receipts are not falsely gone, the response is retryable, and retry with the current workspace version re-enumerates and finishes. Stale-version/wrong-owner begin-close writes nothing.
4. **Receipt separation:** resolve a committed still-existing child during closure: lifecycle denial, stored receipt still committed. Delete that child and resolve the same key again: gone even if source/workspace has since disappeared, with no recreation. Digest mismatch still conflicts before disclosure.
5. **Protected residual and bypasses:** leave a protected row at the final check; abort without attempting operation locks under W. Generic restore/sync/import cannot reopen or reparent it; H2 protection covers plain neutral forks and typed-retention sources. Ordinary unrelated/global deletion behavior remains covered by existing tests.
6. **Transaction boundary and ordering:** both engines reject nested staged deletion before changing any state, without committing unrelated caller work. Adapt the two named saved-view race tests as above and preserve their assertions. Exercise fork/retention/closure/delete interleavings with the defined lock order and bounded joins; no conversation-then-workspace wait is allowed.

The lifecycle field/guard and deletion store protocol belong in Task 1.2; native retention/context use belongs in Task 2.3; capture/commit/resolve routing and HTTP lifecycle behavior belong in Task 4.1. The issue is closed at design level when those contracts and acceptance cases are explicit in the design and plan; production qualification still requires the implementation and tests.
