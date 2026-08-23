# Recipient Shared Research Workspace Data Plane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/research-workspace?shared={share_id}` a fail-closed, recipient-facing view of the owner's canonical shared workspace, with bounded source inspection, recipient-owned history, and grounded chat that can retrieve and cite only currently shared media.

**Architecture:** A route gate keeps local and shared Research Workspace state completely separate. Backend access, chat orchestration, and recipient persistence are split across `SharedWorkspaceAccessService`, `SharedWorkspaceChatService`, and `SharedWorkspaceChatStore`; the read plane resolves owner workspace/media data only after authorization, while completed chat turns and idempotency receipts live in the recipient's ChaChaNotes database. Shared retrieval uses an explicit owner-media allowlist and owner embedding namespace, then generation uses recipient-resolved credentials through the direct chat adapter path with no tools, streaming, cache, or fallback answer.

**Tech Stack:** FastAPI, Pydantic v2, SQLite/PostgreSQL through `CharactersRAGDB`, Jobs status projection, unified RAG retrieval, recipient BYOK and LLM adapter registry, React 18, React Router, TanStack Query-compatible auth fetch, Vitest, Pytest, Bandit, Next.js, and Chrome DevTools Protocol.

**Spec:** `Docs/Design/TASK_12020_40_shared_research_workspace_recipient_data_plane.md`

## Global Constraints

- Preserve `/research-workspace?shared={share_id}` as the only shared Research Workspace URL. Add no redirect, alias, compatibility route, or local-workspace fallback.
- Keep `/research` separate from Research Workspace.
- A present `shared` parameter always selects the shared boundary, even when malformed, unauthorized, revoked, deleted, or operationally unavailable.
- Never mount the local `ResearchWorkspace`, hydrate its workspace Zustand state, or call local workspace, Studio, notes, MCP, ACP, sandbox, artifact, or mutation APIs while shared mode is unresolved or active.
- Shared mode in this task is source read/preview plus grounded chat only. `view_chat_add` and `full_edit` remain policy ceilings; shared writes remain unavailable. Clone remains TASK-12020.41.
- Every recipient endpoint requires `sharing.read`, the existing sharing RBAC rate-limit resource, an authoritative active team/org membership lookup rather than token claims alone, and neutral non-enumerating `404` behavior for missing, revoked, deleted-target, or out-of-scope shares.
- Authorize before opening owner content databases. Reauthorize and revalidate selected source snapshots before generation and immediately before persistence.
- Retrieve only from the owner's Media database and `user_{owner_user_id}_media_embeddings` namespace with a non-empty explicit media allowlist. Do not pass owner ChaChaNotes data into retrieval.
- Disable RAG cache, generation, web fallback, research loops, tools, MCP, ACP, sandbox calls, provider fallback, and streaming on this path.
- Use recipient BYOK or server credentials resolved for the authenticated recipient. Never select credentials merely because a user owns the shared content; when the owner opens their own share URL, their credentials are valid only because that same principal is the recipient.
- Budget the complete model prompt against a server-owned context-window value before generation. Token counting and truncation must be local-only; shared source text must never be sent to a provider tokenizer or token-count endpoint.
- Do not return owner media IDs, database paths, filesystem paths, secrets, raw provider configuration, raw exceptions, queries, prompts, or retrieved excerpts in errors or audit events.
- Keep responses bounded: source pages at 50 by default and 200 maximum; history at 30 by default and 100 maximum; previews at 3,000 characters by default, 12,000 maximum, and 10 chunks maximum; chat scope at 500 sources maximum; citations at 20 maximum, 1,000 quote characters each, and 16,000 aggregate quote characters.
- Persist no user message for failed chat. A successful user message, assistant message, bounded citation metadata, and completed receipt commit atomically.
- Treat the active `CharactersRAGDB.client_id` as the sole recipient tenant key. Normalize it to a non-empty string inside `SharedWorkspaceChatStore`; never accept a recipient identity from an HTTP payload or a Sharing-service call.
- Force PostgreSQL RLS on both recipient chat tables. Policies must bind rows to `app.current_user_id`, the recipient-owned conversation, and the matching thread; application predicates remain defense in depth rather than the tenancy boundary.
- Do not use Jobs or `workspace_operations` for synchronous shared chat. Continue using Jobs only as the owner source ingestion/status authority.
- Keep browser-extension capture and destination contracts unchanged. The extension-hosted Research Workspace route wrapper adopts the same fail-closed gate, but the extension must not receive a writable shared destination from this work.
- Use actual backend and WebUI processes for live acceptance. Browser interaction for live acceptance must use CDP, never native computer control or mocked network responses.
- Do not stage or modify the unrelated untracked watchlist templates in this worktree.

## Delivery Stages

### Stage 1: Fail-Closed Boundary
**Goal:** Prevent local-data leakage and remove unconstrained recipient content paths before replacement APIs exist.
**Success Criteria:** Shared URLs never mount local Research Workspace; old full-media and unsafe chat routes are absent.
**Tests:** Route-gate unit tests and route-absence API tests.
**Status:** Not Started

### Stage 2: Recipient Persistence
**Goal:** Add backend-parity schema and concurrency-safe recipient chat storage.
**Success Criteria:** SQLite and PostgreSQL pass thread, receipt, lease, replay, rollback, cascade, policy-catalog, and cross-recipient isolation tests.
**Tests:** Migration and `SharedWorkspaceChatStore` suites.
**Status:** Not Started

### Stage 3: Authorized Read and Chat Plane
**Goal:** Add access resolution, bounded reads, frozen source scope, verified retrieval, direct generation, typed errors, and audit.
**Success Criteria:** The API returns only authorized shared data and rejects every out-of-scope retrieval or lifecycle change before disclosure/persistence.
**Tests:** Access-service, read-endpoint, retrieval, generation, concurrency, and security suites.
**Status:** Not Started

### Stage 4: Dedicated Recipient UI
**Goal:** Build the compact Sources/Chat shared surface with durable history, evidence inspection, power-user selection, and truthful state handling.
**Success Criteria:** Shared UI exposes only server-allowed actions, preserves draft/selection on failure, and passes desktop/mobile keyboard and responsive tests.
**Tests:** Typed client, controller, route, component, accessibility, and request-ledger suites.
**Status:** Not Started

### Stage 5: Integrated and Live Acceptance
**Goal:** Close cross-layer security gaps and prove the owner/member/nonmember/revocation matrix against real processes.
**Success Criteria:** Focused verification, Bandit, OpenAPI/docs checks, CDP UAT, evidence screenshots, and Backlog closeout are complete.
**Tests:** Full focused matrix plus live CDP walkthrough and API race probe.
**Status:** Not Started

## Dependency and Parallelization Map

```text
Task 1 fail-closed boundary
  |
  +--> Task 2 schema --> Task 3 store --------------------+
  |                                                        |
  +--> Task 4 access/helpers --> Task 5 read API ----------+--> Task 10 integration
  |                         \                              |
  |                          +--> Task 6 retrieval --------+--> Task 7 safe chat API
  |                                                        |
  +--> Task 8 frontend client/controller ------------------+--> Task 9 shared UI
                                                           |
                                                           +--> Task 11 live CDP UAT
```

- After Task 1, Tasks 2, 4, and 8 may run in parallel in separate worktrees or agents.
- Task 3 depends on Task 2. Task 5 depends on Tasks 3 and 4. Task 6 depends on Task 4. Task 7 depends on Tasks 3 and 6.
- Task 9 depends on the Task 5/7 API contracts and Task 8 controller. Task 10 begins after Tasks 5, 7, and 9. Task 11 is last.
- Each task below is one review checkpoint and one focused commit. Keep every intermediate commit fail closed.

## Execution Preflight

- [ ] Fetch `origin/dev` and rebase this feature worktree before Task 1. Do not stage or alter the two unrelated untracked watchlist templates.
- [ ] Re-run the file/signature reconnaissance referenced by each task after the rebase and update only stale line-level assumptions; preserve the approved contracts.
- [ ] Confirm the current ChaChaNotes schema version before writing migration tests. `origin/dev` commit `2e0815c1e4` was inspected on 2026-08-21 and is schema V60, so this plan reserves V61. If a newer migration lands first, change every V61 file/symbol/version reference to the next free version and record that drift in TASK-12020.40 before editing schema code.
- [ ] Run the existing focused Research Workspace, sharing, AuthNZ sharing, and ChaCha migration tests once to establish a post-rebase baseline. A pre-existing failure is recorded; it is not silently reclassified as part of this feature.

---

### Task 1: Install the Fail-Closed Route Gate and Remove Unsafe Recipient Paths

**Files:**
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/shared-workspace-route-state.ts`
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/ResearchWorkspaceRouteGate.tsx`
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/SharedResearchWorkspace/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx`
- Delete: `apps/packages/ui/src/components/Option/ResearchWorkspace/SharedWorkspaceContext.tsx`
- Delete: `apps/packages/ui/src/components/Option/ResearchWorkspace/SharedWorkspaceBanner.tsx`
- Modify: `apps/packages/ui/src/routes/option-research-workspace.tsx`
- Modify: `apps/tldw-frontend/extension/routes/option-research-workspace.tsx`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sharing.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sharing_schemas.py`
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspaceRouteGate.test.tsx`
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.desktop-layout.test.tsx`
- Test: `tldw_Server_API/tests/Sharing/test_shared_workspace_chat_security.py`
- Test: `tldw_Server_API/tests/Sharing/test_sharing_endpoints.py`

**Interfaces:**
- Produces `parseSharedWorkspaceRoute(search)` and `ResearchWorkspaceRouteGate`.
- Temporarily renders a scoped unavailable shared surface for valid share IDs until Task 8 connects bootstrap data.
- Removes `GET /api/v1/sharing/shared-with-me/{share_id}/media/{media_id}` and the existing unconstrained `POST .../{share_id}/chat` with no replacement alias.
- Keeps the named local `ResearchWorkspace` export so its existing focused tests remain stable.

- [ ] **Step 1: Write parser and mounting-boundary tests**

```tsx
it.each(["?shared=", "?shared=0", "?shared=-1", "?shared=1.5", "?shared=01", "?shared=1&shared=2", `?shared=${Number.MAX_SAFE_INTEGER + 1}`])(
  "fails closed for %s",
  (search) => expect(parseSharedWorkspaceRoute(search)).toEqual({ kind: "shared-invalid" })
)

it("does not import or mount the local workspace for a valid shared route", async () => {
  renderGate("?shared=42")
  expect(await screen.findByRole("heading", { name: /shared workspace unavailable/i })).toBeVisible()
  expect(localWorkspaceFactory).not.toHaveBeenCalled()
  expect(localWorkspaceApiRequests()).toEqual([])
})
```

Also cover no parameter -> local, exactly one positive base-10 safe integer -> shared-valid, route parameter replacement, parameter removal, and invalid shared state focus.

- [ ] **Step 2: Run the focused frontend tests and confirm the red state**

Run: `cd apps/packages/ui && bunx vitest run src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspaceRouteGate.test.tsx src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.desktop-layout.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: the route-state module and gate do not exist, and both wrappers still import the local component.

- [ ] **Step 3: Implement strict route parsing and lazy branch loading**

```tsx
export type SharedWorkspaceRouteMode =
  | { kind: "local" }
  | { kind: "shared-invalid" }
  | { kind: "shared-valid"; shareId: number }

export const parseSharedWorkspaceRoute = (search: string): SharedWorkspaceRouteMode => {
  const values = new URLSearchParams(search).getAll("shared")
  if (values.length === 0) return { kind: "local" }
  if (values.length !== 1 || !/^[1-9][0-9]*$/.test(values[0])) {
    return { kind: "shared-invalid" }
  }
  const shareId = Number(values[0])
  return Number.isSafeInteger(shareId)
    ? { kind: "shared-valid", shareId }
    : { kind: "shared-invalid" }
}
```

Use `useLocation().search` as the reactive source. Render separate lazy imports for local and shared branches under `Suspense`; key the shared branch by `shareId` so a parameter change unmounts old state synchronously. The suspense fallback is the unresolved state and uses stable workspace-height geometry.

- [ ] **Step 4: Move both wrappers to the shared gate and strip shared logic from local Research Workspace**

Replace each wrapper's `<ResearchWorkspace />` with `<ResearchWorkspaceRouteGate />` without changing its bounded height classes. Remove `sharedShareId`, metadata fetching, `SharedWorkspaceProvider`, and both `SharedWorkspaceBanner` render sites from the local component. Delete the context and banner after `rg` confirms they have no remaining consumers.

- [ ] **Step 5: Write and run backend route-removal tests**

```python
def test_legacy_shared_full_media_route_is_absent(client, auth_headers):
    response = client.get(
        "/api/v1/sharing/shared-with-me/12/media/99",
        headers=auth_headers,
        follow_redirects=False,
    )
    assert response.status_code == 404
    assert "location" not in response.headers

def test_unsafe_shared_chat_route_is_absent_until_safe_replacement(client, auth_headers):
    response = client.post(
        "/api/v1/sharing/shared-with-me/12/chat",
        headers=auth_headers,
        json={"query": "sentinel"},
        follow_redirects=False,
    )
    assert response.status_code == 404
```

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_chat_security.py tldw_Server_API/tests/Sharing/test_sharing_endpoints.py -q`

Expected after implementation: old route assertions pass; obsolete tests for the removed unsafe contract are deleted or rewritten to assert route absence.

- [ ] **Step 6: Run all route-gate and static layout checks**

Run: `cd apps/packages/ui && bunx vitest run src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspaceRouteGate.test.tsx src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.desktop-layout.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: local mode mounts the local component, every shared state avoids it, and wrapper height parity remains intact.

- [ ] **Step 7: Commit the fail-closed boundary**

```bash
git add apps/packages/ui/src/components/Option/ResearchWorkspace apps/packages/ui/src/routes/option-research-workspace.tsx apps/tldw-frontend/extension/routes/option-research-workspace.tsx tldw_Server_API/app/api/v1/endpoints/sharing.py tldw_Server_API/app/api/v1/schemas/sharing_schemas.py tldw_Server_API/tests/Sharing
git commit -m "fix(workspaces): fail closed for recipient shares"
```

### Task 2: Add ChaChaNotes Schema V61 for Shared Threads and Receipts

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py`
- Create: `tldw_Server_API/tests/DB_Management/test_chacha_migration_v61.py`
- Create: `tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v61.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v52.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v52_integration.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_chacha_migration_v39.py`

**Interfaces:**
- Bumps `CharactersRAGDB._CURRENT_SCHEMA_VERSION` from 60 to 61 after the required rebase preflight.
- Adds `shared_workspace_chat_threads` and `shared_workspace_chat_requests` with equivalent SQLite/PostgreSQL constraints.
- Stores `recipient_user_id` and historical `owner_user_id` as non-empty `TEXT`, matching ChaChaNotes' canonical `client_id` tenant representation; `share_id` remains `INTEGER`/`BIGINT`.
- Adds `build_shared_workspace_chat_rls_sql()` to the canonical ChaCha policy set and force-enables recipient-scoped read/write policies for both new PostgreSQL tables.
- Adds source-version 60 to `_sqlite_linear_migration_steps()` and the PostgreSQL initializer chain.

- [ ] **Step 1: Write failing SQLite and PostgreSQL migration tests**

Assert fresh schema and V60 upgrade behavior, primary/foreign keys, non-empty text tenant keys, positive share IDs, status/source-mode checks, unique conversation mapping, composite receipt-to-thread mapping, cascade on hard conversation deletion, and idempotent initializer reruns. For PostgreSQL, assert both relations have RLS enabled and forced, and that their named policies contain `USING` and `WITH CHECK` ownership predicates. Update earlier migration tests to distinguish their historical step correctness from "initializer reaches current V61" rather than hard-coding an older version as current.

- [ ] **Step 2: Run migration tests and confirm the red state**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_chacha_migration_v61.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v61.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v52.py tldw_Server_API/tests/DB_Management/test_chacha_migration_v39.py -q`

Expected: V61 migration modules/tables are absent and the rebased registry currently stops at source version 59.

- [ ] **Step 3: Add the SQLite V60 -> V61 migration**

Use the following contract, with `DATETIME` timestamps and `INTEGER` identity fields in SQLite:

```sql
CREATE TABLE IF NOT EXISTS shared_workspace_chat_threads (
  recipient_user_id TEXT NOT NULL CHECK(length(trim(recipient_user_id)) > 0),
  share_id INTEGER NOT NULL CHECK(share_id > 0),
  conversation_id TEXT NOT NULL UNIQUE REFERENCES conversations(id) ON DELETE CASCADE,
  owner_user_id TEXT NOT NULL CHECK(length(trim(owner_user_id)) > 0),
  workspace_id TEXT NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (recipient_user_id, share_id),
  UNIQUE (recipient_user_id, share_id, conversation_id)
);

CREATE TABLE IF NOT EXISTS shared_workspace_chat_requests (
  recipient_user_id TEXT NOT NULL CHECK(length(trim(recipient_user_id)) > 0),
  share_id INTEGER NOT NULL CHECK(share_id > 0),
  request_id TEXT NOT NULL,
  request_fingerprint TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('in_progress','retryable','completed','conflicted')),
  lease_epoch INTEGER NOT NULL DEFAULT 1 CHECK(lease_epoch >= 1),
  lease_token TEXT,
  lease_expires_at DATETIME,
  source_mode TEXT CHECK(source_mode IN ('all','include')),
  source_ids_json TEXT,
  source_snapshot_hash TEXT,
  provider TEXT,
  model TEXT,
  user_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
  assistant_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
  error_code TEXT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at DATETIME,
  PRIMARY KEY (recipient_user_id, share_id, request_id),
  FOREIGN KEY (recipient_user_id, share_id, conversation_id)
    REFERENCES shared_workspace_chat_threads(recipient_user_id, share_id, conversation_id)
    ON DELETE CASCADE
);
```

Add indexes for `(conversation_id)`, `(status, lease_expires_at)`, `(status, updated_at)` for bounded conflicted-receipt cleanup, and `(share_id, updated_at)`, then update `db_schema_version` to 61. Keep the migration additive and idempotent.

- [ ] **Step 4: Add the PostgreSQL-equivalent migration and guarded forced-RLS policy builder**

Use `TEXT` for recipient and historical owner IDs, `BIGINT CHECK(share_id > 0)` for share IDs, `TIMESTAMPTZ` for timestamps, existing text conversation/message FKs, and the same check/unique/cascade semantics. Apply it through `_apply_postgres_migration_script(..., expected_version=61)`.

Add `build_shared_workspace_chat_rls_sql()` using guarded `DO` blocks, and extend `build_chacha_rls_sql()` with it. Each policy must have identical `USING` and `WITH CHECK` ownership expressions:

- thread rows require `recipient_user_id = current_setting('app.current_user_id', true)` and a live recipient-owned conversation whose `id`, `client_id`, and `deleted` state match the row;
- request rows require the same recipient key, a matching visible thread for `(recipient_user_id, share_id, conversation_id)`, and a recipient-owned conversation; when message references are non-null, each referenced message must belong to the same conversation and recipient.

The V61 PostgreSQL migration must create both tables, apply only this reviewed policy block, verify `relrowsecurity`, `relforcerowsecurity`, and both policy catalog rows, and then advance the schema version. The normal initializer still applies the complete canonical ChaCha policy set after all migrations.

- [ ] **Step 5: Run migration parity and registry tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_chacha_migration_v61.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v61.py tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v52.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v52_integration.py tldw_Server_API/tests/DB_Management/test_chacha_migration_v39.py -q`

Expected: all selected tests pass; the PostgreSQL integration test may skip only through the repository fixture's standard unavailable signal.

- [ ] **Step 6: Commit schema V61**

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py tldw_Server_API/tests/DB_Management
git commit -m "feat(sharing): add recipient chat receipt schema"
```

### Task 3: Implement the Fenced Recipient Chat Store

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/chacha/shared_workspace_chat_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/__init__.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Create: `tldw_Server_API/tests/DB_Management/test_shared_workspace_chat_store.py`
- Create: `tldw_Server_API/tests/DB_Management/test_shared_workspace_chat_store_postgres.py`

**Interfaces:**

```python
class SharedWorkspaceChatStore:
    def get_or_create_thread(
        self, *, share_id: int, owner_user_id: str,
        workspace_id: str, workspace_name: str,
    ) -> SharedWorkspaceChatThread: ...

    def get_thread(self, *, share_id: int) -> SharedWorkspaceChatThread | None: ...

    def claim_request(
        self, *, share_id: int, request_id: UUID,
        request_fingerprint: str, conversation_id: str,
        lease_seconds: int, now: datetime,
    ) -> SharedWorkspaceChatClaim: ...

    def freeze_sources(
        self, *, claim: SharedWorkspaceChatClaim, source_mode: str,
        source_ids: tuple[str, ...], snapshot_hash: str,
        provider: str, model: str,
    ) -> bool: ...

    def mark_retryable(self, *, claim: SharedWorkspaceChatClaim, error_code: str) -> bool: ...
    def mark_conflicted(self, *, claim: SharedWorkspaceChatClaim, error_code: str) -> bool: ...

    def complete_turn(
        self, *, claim: SharedWorkspaceChatClaim, query: str, answer: str,
        citations: list[dict[str, Any]], provider: str, model: str,
        source_mode: str, effective_source_count: int,
    ) -> StoredSharedWorkspaceTurn: ...

    def load_completed_turn(self, *, share_id: int, request_id: UUID) -> StoredSharedWorkspaceTurn | None: ...
    def list_messages(self, *, share_id: int, before: str | None, limit: int) -> SharedWorkspaceMessagePage: ...
    def purge_expired_conflicts(self, *, now: datetime, limit: int = 100) -> int: ...
```

`SharedWorkspaceChatClaim.disposition` is one of `claimed`, `replay`, `in_progress`, or `request_id_conflict`. A fingerprint conflict never mutates or invalidates the existing claimant. `conflicted` is reserved for a frozen source mismatch.

This store is deliberately located under `core/DB_Management`: it is the persistence abstraction allowed to own SQL. It normalizes its private recipient key from `str(db.client_id)` at construction and rejects a missing/blank client ID. No public method accepts a recipient identity. `core/Sharing` services call this interface and contain no raw SQL.

- [ ] **Step 1: Write failing thread and claim tests**

Cover concurrent first-thread creation, canonical conversation fields and `client_id`, matching claim, fingerprint conflict without receipt mutation, active lease retry timing, expired/retryable reclaim with incremented epoch/token, and completed replay. Assert lease duration clamps to 5-30 minutes and every row key is derived from the bound DB client rather than a caller value.

- [ ] **Step 2: Write failing fenced-write and rollback tests**

Cover stale tokens failing `freeze_sources`, failure transitions, and completion; strict citation metadata failure rolling back both messages and receipt state; completed references loading the exact stored turn; history cursor ordering; conversation hard-delete cascade; and bounded deletion of `conflicted` receipts older than 24 hours without deleting completed or retryable receipts. On PostgreSQL, use two `CharactersRAGDB` instances over the same backend and prove recipient B cannot select, claim, update, or delete recipient A's thread/receipts, even when A's share and conversation IDs are known.

- [ ] **Step 3: Run the SQLite store tests and confirm the red state**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_shared_workspace_chat_store.py -q`

Expected: the store module does not exist.

- [ ] **Step 4: Implement thread and receipt claims with transaction races handled outside failed transactions**

Create the conversation with `source="shared_workspace"`, `external_ref=f"share:{share_id}"`, `scope_type="global"`, `workspace_id=None`, and `client_id=self._recipient_user_id`. Insert conversation and thread mapping in one outer `db.transaction()`. On SQLite `IntegrityError`, PostgreSQL `BackendDatabaseError`, or mapped `ConflictError`, let the transaction roll back the losing conversation and then reload the winning thread in a fresh transaction.

Claims use insert-first semantics. Reclaim uses one compare-and-swap predicate over request key, fingerprint, previous epoch/status/expiry; generate the replacement token with `secrets.token_urlsafe(32)`. Parse timestamps as aware UTC values and cap retry timing returned to clients.

Before a new claim, opportunistically delete at most 100 `conflicted` receipts whose `updated_at` is older than 24 hours. Keep this cleanup in its own short transaction so cleanup failure never weakens claim correctness; backend failures are normalized through the existing `CharactersRAGDBError` family.

- [ ] **Step 5: Implement frozen scope, strict completion, replay, and opaque history cursors**

Use fenced updates containing request key, `lease_epoch`, `lease_token`, and `status='in_progress'`. Validate source JSON before writing: sorted unique strings, 1-500 entries, and bounded serialized length.

For completion, generate both message IDs first and perform this in one outer transaction:

```python
with self._db.transaction() as conn:
    self._assert_current_claim(conn, claim)
    user_id = self._db.add_message({
        "id": user_message_id,
        "conversation_id": claim.conversation_id,
        "sender": "user",
        "content": query,
    })
    assistant_id = self._db.add_message({
        "id": assistant_message_id,
        "conversation_id": claim.conversation_id,
        "sender": "assistant",
        "content": answer,
    })
    conn.execute(STRICT_MESSAGE_METADATA_UPSERT, (..., json.dumps({"rag_context": rag_context})))
    updated = conn.execute(FENCED_COMPLETE_SQL, (..., claim.lease_epoch, claim.lease_token))
    if updated.rowcount != 1:
        raise StaleSharedWorkspaceChatClaim()
```

The stored `rag_context.retrieved_documents` contains only the same bounded citation quotes returned to the recipient, with canonical `source_id` values and no media IDs/full documents. Do not call the best-effort metadata helpers.

Encode history cursors as validated base64url JSON over `(timestamp, last_modified, message_id)` and query with the equivalent stable descending tuple/disjunction, `limit + 1`, then return chronological order. Invalid cursors raise a domain input error.

- [ ] **Step 6: Run SQLite and PostgreSQL store suites**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_shared_workspace_chat_store.py tldw_Server_API/tests/DB_Management/test_shared_workspace_chat_store_postgres.py tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py -q`

Expected: concurrency, fencing, rollback, replay, pagination, cascade, and cross-recipient RLS isolation pass on both backends; PostgreSQL skips only through the standard fixture when unavailable.

- [ ] **Step 7: Commit the store**

```bash
git add tldw_Server_API/app/core/DB_Management/chacha/shared_workspace_chat_store.py tldw_Server_API/app/core/DB_Management/chacha/__init__.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/DB_Management/test_shared_workspace_chat_store.py tldw_Server_API/tests/DB_Management/test_shared_workspace_chat_store_postgres.py
git commit -m "feat(sharing): persist fenced recipient chat turns"
```

### Task 4: Add the Access Service and Reusable Workspace Read Helpers

**Files:**
- Create: `tldw_Server_API/app/core/Sharing/shared_workspace_access_service.py`
- Create: `tldw_Server_API/app/core/Workspaces/job_status.py`
- Create: `tldw_Server_API/app/core/Workspaces/source_preview.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/shared_workspace_repo.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- Create: `tldw_Server_API/tests/Sharing/test_shared_workspace_access_service.py`
- Modify: `tldw_Server_API/tests/Sharing/test_shared_workspace_repo.py`
- Create: `tldw_Server_API/tests/AuthNZ/integration/test_authnz_sharing_postgres.py`
- Create: `tldw_Server_API/tests/Workspaces/test_workspace_job_status.py`
- Create: `tldw_Server_API/tests/Workspaces/test_workspace_source_preview.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class SharedWorkspaceAccessContext:
    share_id: int
    workspace_id: str
    owner_user_id: int
    recipient_user_id: int
    share_scope_type: Literal["team", "org"]
    share_scope_id: int
    access_level: str
    allow_clone: bool
    owner_display_name: str
    shared_at: str | None
    workspace: dict[str, Any]
    policy_actions: dict[str, dict[str, Any]]

class SharedWorkspaceAccessService:
    async def resolve(
        self, *, share_id: int, recipient_user_id: int,
    ) -> SharedWorkspaceAccessContext: ...
```

The constructor receives the share repository, `AuthnzUsersRepo`, and an async owner-ChaCha loader. The context's scope fields are internal inputs for recipient credential resolution and are never serialized. `resolve()` loads the owner user and owner ChaCha database only after the authoritative share/membership query passes.

- [ ] **Step 1: Write failing access-order and disclosure tests**

Assert owner access, active team/org member access, neutral `SharedWorkspaceNotFound` for missing/revoked/out-of-scope/deleted workspace, rejection after membership suspension/removal even when token claims remain stale, authoritative Shared-with-me listing after the same membership changes, operational `SharedWorkspaceUnavailable` after authorization, no owner/user/owner-loader call for denied users, sanitized owner display name, and deny-by-default policy actions.

- [ ] **Step 2: Run access tests and confirm the red state**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_access_service.py -q`

Expected: the access service is absent.

- [ ] **Step 3: Add one authoritative accessible-share repository query**

Add `SharedWorkspaceRepo.get_active_share_for_user(share_id: int, user_id: int) -> dict[str, Any] | None` and `list_active_shares_for_user(user_id: int) -> list[dict[str, Any]]`. Use parameterized AuthNZ queries for both backends that return unrevoked shares only when the caller is the owner or has an active membership in the exact target team/org; also require the target team/organization to remain active. Preserve the existing Shared-with-me rule that omits shares owned by the current user. Return `None`/an empty list for missing, revoked, deleted-scope, suspended, or out-of-scope rows. Do not trust `User.team_ids`, `User.org_ids`, or request-state claims as the authorization authority. Backend failure raises the repository's normal database error and is mapped to a recipient-safe unavailable response without opening owner databases.

- [ ] **Step 4: Implement access resolution and policy projection**

Owners may use their own active share URL, but receive the recipient action projection. Resolve `owner_display_name` from a bounded username only after access succeeds; use `"Workspace owner"` when unavailable, never an owner ID. The access service projects policy ceilings only; source/provider readiness is overlaid by the bootstrap builder in Task 5. Policy results are explicit objects:

```python
{
    "inspect_sources": {"allowed": True, "reason_code": None},
    "ask_grounded_questions": {"allowed": True, "reason_code": None},
    "add_sources": {"allowed": False, "reason_code": "shared_write_not_available"},
    "edit_workspace": {"allowed": False, "reason_code": "shared_write_not_available"},
    "clone_workspace": {"allowed": False, "reason_code": "clone_deferred"},
}
```

- [ ] **Step 5: Extract the Jobs lookup without changing local workspace behavior**

Move `_safe_list_jobs`, `_dedupe_jobs_by_identity`, and `_list_recent_media_ingest_jobs` into `core/Workspaces/job_status.py` as:

```python
def list_recent_workspace_source_ingest_jobs(
    job_manager: JobManager | None, *, owner_user_id: int | str
) -> list[dict[str, Any]]: ...
```

Keep the existing workspace-source and legacy media job queries, 500-item bounds, deduplication, and fail-open optional enrichment. Update local workspace endpoints to call the public helper and prove identical output with focused tests.

- [ ] **Step 6: Extract and extend the bounded preview builder**

Move local preview helpers into `core/Workspaces/source_preview.py` and expose:

```python
def build_workspace_source_preview(
    *, workspace_id: str, source: dict[str, Any], source_status: dict[str, Any],
    media_db: Any | None, max_chars: int, chunk_limit: int,
    focus_chunk_index: int | None = None,
) -> dict[str, Any]: ...
```

When `focus_chunk_index` is present, validate it as non-negative and fetch a centered window of at most `chunk_limit` active chunks. Keep the total preview and chunk bounds unchanged. The local endpoint continues to return its current response shape; recipient schemas in Task 5 omit media IDs and sanitize URLs.

- [ ] **Step 7: Run access and helper regression tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_access_service.py tldw_Server_API/tests/Sharing/test_shared_workspace_repo.py tldw_Server_API/tests/AuthNZ/integration/test_authnz_sharing_postgres.py tldw_Server_API/tests/Workspaces/test_workspace_job_status.py tldw_Server_API/tests/Workspaces/test_workspace_source_preview.py -q`

Expected: authorization order and helper behavior pass, and local preview/status contracts remain unchanged.

- [ ] **Step 8: Commit access and reusable read helpers**

```bash
git add tldw_Server_API/app/core/Sharing/shared_workspace_access_service.py tldw_Server_API/app/core/AuthNZ/repos/shared_workspace_repo.py tldw_Server_API/app/core/Workspaces/job_status.py tldw_Server_API/app/core/Workspaces/source_preview.py tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/tests/Sharing/test_shared_workspace_access_service.py tldw_Server_API/tests/Sharing/test_shared_workspace_repo.py tldw_Server_API/tests/AuthNZ/integration/test_authnz_sharing_postgres.py tldw_Server_API/tests/Workspaces
git commit -m "feat(sharing): authorize canonical shared workspace reads"
```

### Task 5: Replace Recipient Read APIs with Typed Bounded Envelopes

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/shared_workspace_recipient_schemas.py`
- Create: `tldw_Server_API/app/api/v1/utils/shared_workspace_recipient_route.py`
- Modify: `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sharing.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sharing_schemas.py`
- Create: `tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_endpoints.py`
- Modify: `tldw_Server_API/tests/Sharing/test_sharing_endpoints.py`

**Interfaces:**
- Replaces the legacy bootstrap and raw-list source contracts.
- Adds `GET .../{share_id}/chat/messages` and source-ID preview.
- Uses a typed `detail` object for every recipient error.
- Extends `require_permissions()` and `rbac_rate_limit()` only with optional caller-supplied error details; existing callers retain current behavior.
- Mounts replacement routes on a dedicated recipient subrouter whose `APIRoute` maps authentication and request-validation failures without changing global API behavior.

- [ ] **Step 1: Write failing schema and endpoint tests**

Cover request `extra="forbid"`, exact typed 401/403/422/429/503 bodies, bootstrap bounds, allowed-action defaults, source pagination/order/search/state filtering, URL sanitization, no media IDs or file paths, source-ID preview and chunk focus, empty history without thread creation, history pagination, identical neutral 404 bodies, and operational partial errors capped at eight. Assert malformed chat JSON and Pydantic body failures return `invalid_shared_chat_request`, not FastAPI's default validation array.

- [ ] **Step 2: Run recipient endpoint tests and confirm the red state**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_endpoints.py -q`

Expected: typed schemas/endpoints are absent and existing responses violate the new envelope.

- [ ] **Step 3: Define explicit recipient schemas**

Create models for `SharedWorkspaceBootstrapResponse`, `SharedWorkspaceGenerationDefault`, `SharedWorkspaceSourcePage`, `SharedWorkspaceSource`, `SharedWorkspaceSourcePreview`, `SharedWorkspaceMessagePage`, `SharedWorkspaceMessage`, `SharedWorkspaceCitation`, `SharedWorkspaceAllowedAction`, `SharedWorkspacePartialError`, `SharedWorkspaceErrorDetail`, and pagination. Do not subclass owner-facing `ShareResponse`; recipient schemas expose only the approved fields.

Request models use:

```python
model_config = ConfigDict(extra="forbid")
```

Response schemas also forbid accidental additions during construction tests. Use stable reason/code literals or enums where practical.

Centralize recipient projection bounds rather than relying on database values: owner display name 128 characters, workspace/source titles 512, workspace description 2,000, source type 64, origin host 255, partial-error messages 320, and all URLs/identifiers at their schema limits. Reject non-finite scores and invalid timestamps instead of leaking raw values.

- [ ] **Step 4: Add the route-scoped validation/authentication error mapper**

Create `SharedWorkspaceRecipientRoute(APIRoute)` and use it only on a subrouter mounted at `/shared-with-me/{share_id}` under the existing `/sharing` router. Its handler catches `RequestValidationError` and unauthenticated `HTTPException(401)` raised while resolving route dependencies, returning the same `SharedWorkspaceErrorDetail` envelope used by the service. Map POST-chat validation to `invalid_shared_chat_request` and other malformed recipient requests to `invalid_shared_workspace_request`. Keep response schemas and request-body OpenAPI generation intact; do not install a global exception handler or alter owner-sharing, clone, token, or admin routes.

- [ ] **Step 5: Add typed permission and rate-limit detail options**

Extend the dependency factories without changing default callers:

```python
def require_permissions(*permissions: str, detail: Any | None = None): ...
def rbac_rate_limit(resource: str, *, detail: Any | None = None): ...
```

When supplied, only the corresponding 403/429 uses that detail. Preserve `_tldw_rate_limit_resource` metadata. Every recipient route passes `sharing_permission_required`; read routes pass `shared_workspace_rate_limited`, while chat passes `shared_chat_rate_limited`. Add focused dependency and privilege-introspection regression tests. Do not rely on rate limiting as authorization.

- [ ] **Step 6: Implement bootstrap, source page, preview, and history endpoints**

Build one endpoint dependency factory for `SharedWorkspaceAccessService`. Each endpoint resolves access first, then opens owner media or recipient ChaCha resources as needed. Replace `GET /shared-with-me` claim-loop discovery with `list_active_shares_for_user()` before owner-name enrichment, preserving its current response shape. Bootstrap returns first 50 sources and latest 30 messages. Critical share/workspace/source-membership failures reject the envelope; optional Jobs/history/provider-readiness failures become bounded safe `partial_errors` and disable only dependent actions.

Build final response `allowed_actions` from a fresh copy of `context.policy_actions` plus current source/provider readiness. `inspect_sources` remains available for an empty-but-readable source set; `ask_grounded_questions` is disabled with a stable reason when no source is retrieval-capable or the server cannot resolve a default generation target. Include a bounded `generation_default` object with provider, model, readiness, and a stable reason code; provider/model are non-empty only when ready, while an unavailable default has null provider/model and a non-empty reason. Never include credential source, key presence, endpoint, or raw provider diagnostics. Task 5 may initially inject a fail-closed unavailable resolver, and Task 7 wires the canonical target and exact-share-scope credential check before the UI consumes this contract. Never mutate the access context or infer permissions in the frontend.

Sanitize source URLs with `urllib.parse`: permit only HTTP/HTTPS and return only normalized `scheme://host[:port]` origin data, with no credentials, path, query, or fragment; when safe origin reconstruction is not possible, return only a bounded `origin_host`. Never emit `media_id`. Preview passes canonical `source_id` to current membership resolution and maps the internal preview into the recipient schema. Apply `q` before status projection where possible; apply derived `state` filtering before offset/limit, and cap every returned page even though summary computation may inspect the workspace's complete current source set.

- [ ] **Step 7: Run read-plane tests and existing sharing regressions**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_endpoints.py tldw_Server_API/tests/Sharing/test_sharing_endpoints.py tldw_Server_API/tests/Sharing/test_cross_user_access.py -q`

Expected: all selected tests pass; clone behavior remains unchanged and separate.

- [ ] **Step 8: Commit the typed read plane**

```bash
git add tldw_Server_API/app/api/v1/API_Deps/auth_deps.py tldw_Server_API/app/api/v1/endpoints/sharing.py tldw_Server_API/app/api/v1/schemas/shared_workspace_recipient_schemas.py tldw_Server_API/app/api/v1/schemas/sharing_schemas.py tldw_Server_API/app/api/v1/utils/shared_workspace_recipient_route.py tldw_Server_API/tests/Sharing
git commit -m "feat(sharing): add bounded recipient workspace reads"
```

### Task 6: Freeze Shared Source Scope and Fail Closed on Retrieval Provenance

**Files:**
- Create: `tldw_Server_API/app/core/Sharing/shared_workspace_chat_service.py`
- Create: `tldw_Server_API/tests/Sharing/test_shared_workspace_chat_retrieval.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class SharedSourceSnapshotItem:
    source_id: str
    media_id: int
    media_uuid: str
    content_hash: str
    readiness_class: str

@dataclass(frozen=True)
class SharedSourceSnapshot:
    mode: Literal["all", "include"]
    items: tuple[SharedSourceSnapshotItem, ...]
    snapshot_hash: str

@dataclass(frozen=True)
class VerifiedSharedEvidence:
    label: str
    source_id: str
    source_title: str
    content: str
    score: float
    chunk_index: int | None
    start_char: int | None
    end_char: int | None

class SharedWorkspaceChatService:
    def resolve_source_snapshot(...): ...
    def revalidate_source_snapshot(...): ...
    async def retrieve_verified_evidence(...): ...
```

- [ ] **Step 1: Write failing source-scope and snapshot tests**

Cover `all`, `include`, duplicate requested IDs, two canonical sources that reference the same media row, empty/unknown IDs, nonqueryable sources, 500 cap, 501-source subset conflict, source/media remap, UUID/hash change, deletion, trash, readiness loss, unrelated source changes, and frozen `all` retries. Hash canonical JSON over sorted source IDs and only the authorization/content fields in the approved design. When selected source rows share a media ID, retrieve that media once and map citations deterministically to the lexicographically smallest selected canonical source ID.

- [ ] **Step 2: Write failing retrieval isolation tests**

Use two in-workspace media items, one unrelated owner sentinel media item, and owner note/chat sentinels. Assert the pipeline receives only `sources=["media_db"]`, explicit non-empty `include_media_ids`, owner namespace, `media_db`, no `chacha_db`/notes path, and a locked retrieval-only policy with cache, profiles, classification, expansion, LLM reranking, adaptive reruns, generation, and external fallback disabled. Inject an out-of-scope/provenance-less document, non-media source marker, pipeline error, or unexpected generated answer and assert the entire result fails before shared generation.

- [ ] **Step 3: Run retrieval tests and confirm the red state**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_chat_retrieval.py -q`

Expected: `SharedWorkspaceChatService` does not exist.

- [ ] **Step 4: Implement frozen source authorization snapshots**

Resolve current source rows from owner ChaCha, then fetch media with `include_deleted=True, include_trash=True` so deleted/trash transitions are observable. Require a positive media ID, stable media UUID, content hash, live state, and retrieval capability from `build_source_status_projection`. Store no owner media IDs in the receipt; they remain only in the in-memory snapshot object and its hash input.

For a reclaimed receipt with frozen source IDs/provider/model, rebuild and compare exactly those IDs rather than expanding `mode=all` against current membership.

- [ ] **Step 5: Implement retrieval with complete-result provenance validation**

```python
result = await unified_rag_pipeline(
    query=query,
    sources=["media_db"],
    media_db_path=owner_media_db_path,
    notes_db_path=None,
    media_db=owner_media_db,
    chacha_db=None,
    include_media_ids=list(snapshot.media_ids),
    index_namespace=f"user_{owner_user_id}_media_embeddings",
    enable_cache=False,
    adaptive_cache=False,
    search_depth_mode=None,
    rag_profile=None,
    expand_query=False,
    enable_prf=False,
    enable_hyde=False,
    enable_gap_analysis=False,
    enable_reranking=False,
    reranking_strategy="none",
    enable_generation=False,
    enable_post_verification=False,
    adaptive_rerun_on_low_confidence=False,
    fallback_on_error=False,
    enable_web_fallback=False,
    enable_document_grading=False,
    enable_query_rewriting_loop=False,
    enable_query_decomposition=False,
    enable_query_classification=False,
    enable_research_loop=False,
    enable_discussion_search=False,
    search_url_scraping=False,
    enable_query_reformulation=False,
    chat_history=None,
    enable_suggestions=False,
    enable_structured_response=False,
    enable_image_search=False,
    enable_video_search=False,
    enable_streaming=False,
    top_k=20,
)
```

Build these kwargs through one immutable shared-retrieval policy object and pass no caller-owned `resolved_request`, `retrieval_plan`, profile, metadata, or extra kwargs. Explicitly disable every pipeline feature that can invoke generation, provider-backed transformation, query rewriting, adaptive execution, external retrieval, caching, or user-history expansion. Add a signature-sentinel test over all `enable_*`, `fallback_*`, `adaptive_*`, source/database, profile, history, and request/plan parameters; every security-sensitive parameter must be pinned by the policy or listed on a reviewed inert allowlist, so a new parameter forces review.

Treat any pipeline error as `retrieval_unavailable`. Also reject the full result when `generated_answer` is non-empty, metadata reports generation/external sources, or any document lacks `source=media_db`, a parseable `metadata.media_id`, membership in the frozen media set, or a canonical source mapping. Deduplicate repeated retrieved chunks before assigning stable evidence labels `E1` through `E20`, then retain only bounded text needed for generation/citations.

- [ ] **Step 6: Run the retrieval security suite**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_chat_retrieval.py -q`

Expected: source lifecycle conflicts, sentinel isolation, notes exclusion, empty evidence, and provenance rejection all pass.

- [ ] **Step 7: Commit source scoping and retrieval**

```bash
git add tldw_Server_API/app/core/Sharing/shared_workspace_chat_service.py tldw_Server_API/tests/Sharing/test_shared_workspace_chat_retrieval.py
git commit -m "feat(sharing): scope shared retrieval to canonical sources"
```

### Task 7: Add Recipient Provider Resolution, Grounded Generation, and Safe Chat API

**Files:**
- Create: `tldw_Server_API/app/core/Chat/chat_target_resolution.py`
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py`
- Modify: `tldw_Server_API/app/core/Sharing/shared_workspace_chat_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sharing.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/shared_workspace_recipient_schemas.py`
- Create: `tldw_Server_API/tests/Chat/test_chat_target_resolution.py`
- Create: `tldw_Server_API/tests/Sharing/test_shared_workspace_chat_generation.py`
- Create: `tldw_Server_API/tests/Sharing/test_shared_workspace_chat_endpoint.py`

**Interfaces:**
- Produces `resolve_chat_target(requested_provider, requested_model) -> ResolvedChatTarget` from core Chat code and a locally enforced grounded-prompt budget.
- Adds strict `SharedWorkspaceChatRequest` and typed `SharedWorkspaceChatResponse`.
- Restores `POST /api/v1/sharing/shared-with-me/{share_id}/chat` only through the safe service.

- [ ] **Step 1: Write failing provider-target tests**

Cover explicit provider/model, provider-qualified model IDs, server default provider/model, provider override policy, local providers without API keys, key-required providers without recipient/server credentials, no configured provider, unknown adapter, and no silent cross-provider fallback. Assert bootstrap projects the exact same default target, reports null provider/model plus a stable reason when unavailable, and never serializes credential source or endpoint metadata.

- [ ] **Step 2: Extract canonical provider/default-model resolution from `chat.py`**

Move the current `_config_default_llm_provider`, `_get_default_provider`, and nested default-model logic into `core/Chat/chat_target_resolution.py`. Reuse `resolve_provider_and_model`, provider overrides, adapter-registry aliases, environment defaults, and configured models. Keep small compatibility wrappers in `chat.py` only where existing tests patch those names.

```python
@dataclass(frozen=True)
class ResolvedChatTarget:
    provider: str
    model: str

def resolve_chat_target(
    *, requested_provider: str | None, requested_model: str | None
) -> ResolvedChatTarget: ...
```

The helper raises a typed core configuration error when no usable provider/model exists; it does not invoke routing fallback. Wire the same resolver into bootstrap so `generation_default` is the server-resolved provider/model the UI selects first. The ordinary model catalog remains a discovery list, never proof that credentials exist or that a model is authorized; every submitted target is resolved and validated again by the chat endpoint.

- [ ] **Step 3: Write failing generation and citation tests**

Assert recipient `resolve_byok_credentials()` inputs, `request=None`, explicit empty/non-empty team and org lists for exactly the current share scope, and a separately derived trusted-base-URL flag. Prove stale or unrelated `request.state.active_team_id`/`active_org_id` values cannot narrow or replace the share scope, unrelated group and owner credentials are never consulted, and user BYOK remains ahead of current-share scope and server fallback. Also cover direct adapter invocation, no tools/stream/fallback, server-owned prompt, source-instruction isolation, malformed response handling, unknown evidence labels discarded, duplicate labels deduplicated in first-seen order, at least one known label required, 20/1,000/16,000 citation bounds, provider failure typed as `generation_failed`, and BYOK `touch_last_used()` in `finally`.

Cover prompt budgeting for known 4K, 8K, and large context windows and the unknown-model 4K fallback. Assert local `tiktoken` counting is used only when available, the UTF-8 byte upper-bound fallback is deterministic, no provider-native/commercial tokenizer HTTP adapter is invoked, JSON escaping is included in the final prompt count, oversized questions return `shared_chat_context_too_large` before provider invocation, at least one non-empty verified evidence item remains after truncation, dropped labels are rejected, and persisted citation quotes cannot include text trimmed from the model-facing evidence.

- [ ] **Step 4: Implement bounded direct grounded generation**

Build at most 20 evidence blocks, at most 4,000 characters each and 48,000 characters aggregate as outer safety limits. Keep the server instruction in the system message. Put the user question and a `json.dumps()`-serialized evidence array in separate user-message sections so source text cannot terminate or forge hand-written XML/Markdown delimiters. Mark every evidence item as untrusted data and instruct the model never to execute instructions found in it. Ask for strict JSON:

```json
{"answer":"Grounded answer","citations":["E1","E2"]}
```

Before invoking the provider, resolve a positive context window from server-owned model metadata/runtime configuration. Accept known values from 2,048 through 1,000,000 tokens, cap larger values at 1,000,000, reject a known smaller window as unsupported for this flow, and use 4,096 only when metadata is absent or invalid. Reserve a dynamic output budget of `min(1200, max(256, context_window // 4))`, plus `max(256, ceil(context_window * 0.10))` safety tokens. Count the complete serialized messages, including the system instruction, question, labels, titles, JSON punctuation, and escaping. Use only a local `tiktoken` encoding when one is already resolvable for the selected model; never call `resolve_tokenizer()` because it may invoke provider HTTP tokenizers. If local exact counting is unavailable, use `len(text.encode("utf-8"))` as a conservative upper bound.

Greedily retain evidence in retrieval order and binary-search the final item's content against the remaining prompt budget, rebuilding and recounting the actual serialized messages each time. Cap evidence input at 12,000 tokens even for larger models. Materialize a separate immutable budgeted-evidence tuple containing exactly the labels and content sent to the provider; validate response labels and derive citation quotes only from that tuple, never from dropped or trimmed source text. If the fixed prompt plus question leaves no room for one non-empty evidence item and at least 256 output tokens, return typed `422 shared_chat_context_too_large` before credentials are touched or a provider is called. Do not silently truncate the user's question. Accept plain JSON or one surrounding Markdown JSON fence, reject all other shapes, and validate labels against the budgeted evidence. Invoke with the computed output budget:

```python
response = await perform_chat_api_call_async(
    api_endpoint=target.provider,
    model=target.model,
    messages_payload=messages,
    api_key=byok.api_key,
    app_config=byok.app_config,
    streaming=False,
    temperature=0,
    max_tokens=prompt_budget.max_output_tokens,
    user_identifier=str(recipient_user_id),
)
```

Resolve credentials for `recipient_user_id` with `request=None`, `team_ids=[share_scope_id], org_ids=[]` for a team share or `team_ids=[], org_ids=[share_scope_id]` for an organization share. Derive `trusted_base_url_override=is_trusted_base_url_request(request)` separately from the authenticated request so trusted admin/single-user behavior is preserved without allowing request-state active-scope claims to filter the authoritative share scope. User BYOK remains first priority, the current share scope is second, and server configuration is the final allowed fallback; unrelated group and owner credentials are excluded. Use `provider_requires_api_key()` before the call. Do not expose credential source or adapter error text in bootstrap, responses, errors, or audit.

- [ ] **Step 5: Write the orchestration tests before adding the endpoint**

Cover authorize -> claim -> generation-rate reservation -> freeze -> retrieve -> reauthorize/revalidate -> resolve target/context budget/credentials -> generate -> reauthorize/revalidate -> atomic complete; completed replay authorization without rate reservation/retrieval/generation; active/mismatched claim mappings without rate reservation; transient retryable transitions; source conflict transitions; no persistence on any failure; recipient/share chat rate limiting; and bounded audit metadata with no query/answer/excerpts.

- [ ] **Step 6: Implement the strict request/response and endpoint**

```python
class SharedWorkspaceSourceScope(BaseModel):
    mode: Literal["all", "include"]
    source_ids: list[str] = Field(default_factory=list, max_length=500)
    model_config = ConfigDict(extra="forbid")

class SharedWorkspaceChatRequest(BaseModel):
    request_id: UUID
    query: str = Field(min_length=1, max_length=10_000)
    source_scope: SharedWorkspaceSourceScope
    provider: str | None = Field(default=None, max_length=128)
    model: str | None = Field(default=None, max_length=512)
    model_config = ConfigDict(extra="forbid")
```

Fingerprint the trimmed query, normalized mode, sorted unique requested IDs, and normalized requested provider/model. Claim first; completed replay and active/conflicting receipts return without reserving generation capacity. For a newly claimed or reclaimed request, apply the chat `ConversationRateLimiter` using `user_id=str(recipient_id)` and `conversation_id=f"shared:{recipient_id}:{share_id}"` so Resource Governor receives the recipient/share dimension. A rate-limit rejection returns the receipt to `retryable` before responding. Map every domain outcome to the exact typed error table in the spec, including pre-provider `shared_chat_context_too_large`. Emit audit only after a stored completion or mapped failure, as best effort that cannot turn success into failure; include only share ID, actor ID, effective source count, provider/model, outcome, replay flag, and timings.

- [ ] **Step 7: Run provider, generation, endpoint, and concurrency suites**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/test_chat_target_resolution.py tldw_Server_API/tests/Sharing/test_shared_workspace_chat_generation.py tldw_Server_API/tests/Sharing/test_shared_workspace_chat_endpoint.py tldw_Server_API/tests/DB_Management/test_shared_workspace_chat_store.py -q`

Expected: all selected tests pass with no fallback answer and no out-of-scope evidence.

- [ ] **Step 8: Commit safe shared chat**

```bash
git add tldw_Server_API/app/core/Chat/chat_target_resolution.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/app/core/Sharing/shared_workspace_chat_service.py tldw_Server_API/app/api/v1/endpoints/sharing.py tldw_Server_API/app/api/v1/schemas/shared_workspace_recipient_schemas.py tldw_Server_API/tests/Chat/test_chat_target_resolution.py tldw_Server_API/tests/Sharing
git commit -m "feat(sharing): add grounded recipient workspace chat"
```

### Task 8: Add the Typed Frontend Client and Abortable Shared Controller

**Files:**
- Create: `apps/packages/ui/src/types/shared-workspace.ts`
- Create: `apps/packages/ui/src/services/tldw/domains/shared-workspaces.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/index.ts`
- Modify: `apps/packages/ui/src/services/tldw/api-error.ts`
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/SharedResearchWorkspace/useSharedResearchWorkspace.ts`
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/SharedResearchWorkspace/shared-research-workspace-reducer.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/SharedResearchWorkspace/index.tsx`
- Create: `apps/packages/ui/src/services/tldw/domains/__tests__/shared-workspaces.test.ts`
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/shared-research-workspace-reducer.test.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspaceRouteGate.test.tsx`

**Interfaces:**

```ts
export const sharedWorkspacesApi = {
  bootstrap(shareId: number, signal?: AbortSignal): Promise<SharedWorkspaceBootstrap>,
  listSources(shareId: number, query: SharedSourceQuery, signal?: AbortSignal): Promise<SharedSourcePage>,
  previewSource(shareId: number, sourceId: string, chunkIndex?: number, signal?: AbortSignal): Promise<SharedSourcePreview>,
  listMessages(shareId: number, before?: string, signal?: AbortSignal): Promise<SharedMessagePage>,
  ask(shareId: number, request: SharedChatRequest, signal?: AbortSignal): Promise<SharedChatResponse>,
}
```

The controller state has `bootstrap`, `sourceQuery`, `selectedSourceIds`, `messages`, `draft`, `pendingSubmission`, `preview`, and in-pane error slots. Allowed actions initialize to denied.

- [ ] **Step 1: Write failing client contract tests**

Assert auth fetch use, exact canonical paths/query parameters, source-ID URL encoding, `AbortSignal` propagation, no mutation/local paths, strict typed-error normalization, bounded `generation_default` parsing, `shared_chat_context_too_large` handling, and no retry of neutral 404. Extend `StructuredApiErrorDetail` with normalized `code`, `recovery_action`, and non-negative `retry_after_ms`.

- [ ] **Step 2: Write failing reducer/controller lifecycle tests**

Cover bootstrap initialization including fail-closed generation defaults, queryable sources initially selected, route change immediately clearing all previous share data and aborting requests, source refresh reconciling removed/nonqueryable selections, older-history deduplication, draft clearing only after successful stored response, exact-payload retry with the same request UUID after an ambiguous network failure, edit-after-failure invalidating that retry receipt and allocating a new UUID, source conflict forcing refresh/new UUID, rate-limit countdown, context-budget errors preserving the exact draft, and stale responses ignored.

- [ ] **Step 3: Run frontend data tests and confirm the red state**

Run: `cd apps/packages/ui && bunx vitest run src/services/tldw/domains/__tests__/shared-workspaces.test.ts src/components/Option/ResearchWorkspace/__tests__/shared-research-workspace-reducer.test.ts src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspaceRouteGate.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: client/controller modules are absent.

- [ ] **Step 4: Implement the typed API client**

Use `getTldwServerURL`, `fetchWithTldwAuth`, and `buildTldwApiError`. Build query strings with `URLSearchParams`; omit empty values. Keep this recipient client separate from owner mutation hooks in `useSharing.ts`.

- [ ] **Step 5: Implement the reducer and abortable controller**

Use one `AbortController` per bootstrap/page/preview/submission operation and a monotonically increasing share generation. Cleanup aborts every controller. Dispatch `resetForShare` before starting a new share request. UUID creation occurs once in `submitDraft`, and `pendingSubmission` stores the immutable normalized payload that produced it. `retryPending` may reuse the UUID only when resending that exact payload. Editing the draft, source scope, provider, or model after failure clears the retryable pending receipt from client state, so the next submit gets a new UUID; `shared_source_changed` always refreshes and requires a new UUID.

- [ ] **Step 6: Connect the temporary shared shell to bootstrap state**

Replace Task 1's static unavailable surface with stable loading, neutral not-found, unavailable, and loaded placeholders driven only by the shared controller. Do not add local store imports.

- [ ] **Step 7: Run client/controller/route tests**

Run: `cd apps/packages/ui && bunx vitest run src/services/tldw/domains/__tests__/shared-workspaces.test.ts src/components/Option/ResearchWorkspace/__tests__/shared-research-workspace-reducer.test.ts src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspaceRouteGate.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: all selected tests pass, including synchronous stale-share clearing.

- [ ] **Step 8: Commit the frontend data plane**

```bash
git add apps/packages/ui/src/types/shared-workspace.ts apps/packages/ui/src/services/tldw apps/packages/ui/src/components/Option/ResearchWorkspace/SharedResearchWorkspace apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__
git commit -m "feat(workspaces): add recipient shared data client"
```

### Task 9: Build the Dedicated Sources and Chat Recipient Surface

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/SharedResearchWorkspace/index.tsx`
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/SharedResearchWorkspace/SharedWorkspaceHeader.tsx`
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/SharedResearchWorkspace/SharedWorkspaceSourcesPane.tsx`
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/SharedResearchWorkspace/SharedWorkspaceChatPane.tsx`
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/SharedResearchWorkspace/SharedWorkspacePreview.tsx`
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SharedResearchWorkspace.test.tsx`
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SharedResearchWorkspace.accessibility.test.tsx`
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SharedResearchWorkspace.responsive.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/ShareDialog.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/playground.json`
- Modify: `apps/packages/ui/src/public/_locales/en/playground.json`
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/playground-locale-mirror.test.ts`

**Interaction Contract:**
- Desktop: compact header plus two stable panes, Sources and Chat.
- Mobile: compact header plus Sources/Chat tabs; preview is a full-height sheet.
- No trust bar, shared-workspace banner, migration banner, status bar, onboarding banner, Studio, General Chat, notes, MCP, ACP, sandbox, artifacts, or mutation toolbar.

- [ ] **Step 1: Write failing loaded-state and action-authority tests**

Assert workspace identity, owner label, access tooltip, server `allowed_actions` as the only control authority, no hidden mutation controls/requests, no local feature requests, all queryable initially selected, disabled reason labels, select-all/clear, server search/state filters, pagination, 500-source subset state, and no chat submission with zero selected sources.

- [ ] **Step 2: Write failing chat/evidence/error tests**

Assert model list loading through `fetchChatModels`, bootstrap `generation_default` seeding the effective provider/model even when absent from the generic catalog, current scope count, one UUID per submission, persisted response insertion, reload/history/upward pagination, citation buttons, preview focus by chunk, removed-source copy, provider/retrieval/context-budget errors preserving draft and selection, source conflict refresh/new UUID, and rate-limit retry timing.

- [ ] **Step 3: Write failing accessibility and responsive tests**

Cover heading focus after route load, tab semantics, checkbox labels, keyboard source/citation activation, preview focus return, composer label, submission and loading live regions, no horizontal overflow at 390x844 and 1440x900, and stable control dimensions during loading/dynamic labels.

- [ ] **Step 4: Run shared UI tests and confirm the red state**

Run: `cd apps/packages/ui && bunx vitest run src/components/Option/ResearchWorkspace/__tests__/SharedResearchWorkspace.test.tsx src/components/Option/ResearchWorkspace/__tests__/SharedResearchWorkspace.accessibility.test.tsx src/components/Option/ResearchWorkspace/__tests__/SharedResearchWorkspace.responsive.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: dedicated panes and interactions are absent.

- [ ] **Step 5: Implement the compact header and responsive shell**

Use a back link to `/shared-with-me`, literal workspace name, `Shared by {owner}`, and one compact access badge with tooltip. For `view_chat_add`/`full_edit`, explain that the granted tier is the ceiling and shared editing is not yet available. Use fixed/minmax grid tracks and `min-w-0`; avoid decorative cards and nested cards.

- [ ] **Step 6: Implement Sources, preview, and bulk scope controls**

Use checkboxes, search input, state menu, concise readiness badge, pagination controls, and icon buttons with tooltips where symbols are familiar. Source row activation opens `SharedWorkspacePreview`; citation activation passes `chunk_index`. Keep disabled sources visible with a specific reason. Render titles, preview text, citation quotes, and model output as escaped text/Markdown through the existing sanitized renderer; never use `dangerouslySetInnerHTML`. Any displayed origin link uses the server-projected origin only and `rel="noopener noreferrer"`.

- [ ] **Step 7: Implement chat history, composer, model selection, and citations**

Reuse `fetchChatModels`, `useModelSelector`, `ChatModelSelectorDropdown`, and `resolveStartupSelectedModel` with component-local selected-model state, but seed selection from the server `generation_default` ahead of local/global startup preferences. If that exact default is absent from the generic catalog, inject one bounded display option from its provider/model values instead of silently selecting another provider. Treat the generic catalog as discovery only; derive provider from selected model metadata and let the backend re-resolve credentials and policy on every submission. Render recipient messages and bounded citation buttons; load older messages upward without duplicate IDs. Disable send truthfully when the server action, generation default, selected target, or source scope is unavailable.

- [ ] **Step 8: Add owner-facing revocation/provider copy in the existing Share dialog**

Place one compact informational paragraph in `ActiveSharesTab`, near the tables:

> Revoking access prevents future workspace reads and questions. It does not erase content or answers recipients saved while they had access. Recipients may use their own configured model provider, which can receive selected shared passages when they ask a question.

Do not add this copy as a Research Workspace banner.

- [ ] **Step 9: Add localized shared-workspace copy and preserve the English mirror**

Add a `sharedWorkspace` namespace under `playground` for headings, filters, readiness, errors, recovery actions, accessibility names, and owner copy. Use direct state-specific copy in the relevant pane/surface, not banners:

- unavailable/revoked: `This shared workspace isn't available.` and `Return to Shared with me`;
- empty: `This workspace has no shared sources yet.`;
- processing: `Shared sources are still processing. You can inspect available items while you wait.`;
- no provider: `Choose a configured model before asking a question.` with `Open model settings`;
- source conflict: `The shared source set changed. Refresh sources before trying again.`;
- removed citation: `This source is no longer shared.`;
- access tooltip: `This access level is the owner's policy ceiling. Editing shared content is not available here yet.`

Keep both English JSON files byte-equivalent and run the mirror test.

- [ ] **Step 10: Run the complete focused frontend suite**

Run: `cd apps/packages/ui && bunx vitest run src/components/Option/ResearchWorkspace/__tests__ src/services/tldw/domains/__tests__/shared-workspaces.test.ts src/components/Option/Playground/__tests__/playground-locale-mirror.test.ts --maxWorkers=1 --no-file-parallelism`

Expected: all shared, local Research Workspace, locale, accessibility, and responsive tests pass.

- [ ] **Step 11: Commit the recipient UI**

```bash
git add apps/packages/ui/src/components/Option/ResearchWorkspace/SharedResearchWorkspace apps/packages/ui/src/components/Option/ResearchWorkspace/ShareDialog.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__ apps/packages/ui/src/assets/locale/en/playground.json apps/packages/ui/src/public/_locales/en/playground.json apps/packages/ui/src/components/Option/Playground/__tests__/playground-locale-mirror.test.ts
git commit -m "feat(workspaces): build shared recipient research surface"
```

### Task 10: Close Cross-Layer Security, Contract, and Documentation Gaps

**Files:**
- Create: `tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_security_matrix.py`
- Modify: `tldw_Server_API/tests/Sharing/test_cross_user_access.py`
- Modify: `apps/tldw-frontend/e2e/utils/page-objects/ResearchWorkspacePage.ts`
- Create: `apps/tldw-frontend/e2e/workflows/research-workspace.shared-recipient.spec.ts`
- Modify: `Docs/Design/Pagination_Completion_Matrix.md`
- Modify: `Docs/User_Guides/Server/Organizations_and_Sharing.md`
- Modify: `Docs/Published/User_Guides/Server/Organizations_and_Sharing.md`
- Modify: `Docs/Development/Research_Workspace_Final_UAT_Runner.md`
- Modify: `apps/tldw-frontend/lib/api/openapi.fingerprint.json`

- [ ] **Step 1: Add the backend owner/member/nonmember security matrix**

Create deterministic fixtures for owner, authorized team member, nonmember, two workspace sources, unrelated owner sentinel media, owner note/chat sentinels, and recipient local workspace sentinels. Cover neutral 404 equivalence, permission 403, source listing/preview, all/subset chat, completed replay, concurrent matching requests, mismatched fingerprint, revocation/membership/source/media changes at each service boundary, and no saved messages on failure.

- [ ] **Step 2: Add browser-level request-ledger and interaction coverage**

The Playwright CI spec may stub deterministic API responses for component workflow coverage, but it must fail on any request to local workspace, Studio, notes, MCP, ACP, sandbox, artifact, source mutation, or old full-media paths while `shared` is present. Cover desktop/mobile navigation, selection, filters, preview, chat, citations, reload, and revoked state. Live truth remains Task 11.

- [ ] **Step 3: Update current API/docs contracts and remove obsolete active references**

Replace the raw-list pagination matrix row with the new envelope. Document recipient-owned transcript semantics, recipient provider disclosure, revocation limits, read/chat-only scope, and canonical routes in both user-guide copies. Historical plans may retain old route names as history; current docs/tests/UI must not advertise the removed full-media endpoint.

- [ ] **Step 4: Regenerate and verify OpenAPI artifacts**

Run: `cd apps/tldw-frontend && bun run generate:api-types`

Expected: the fingerprint changes for the typed bootstrap/source/preview/history/chat contracts, and generated types contain no old recipient `SharedMediaResponse` operation.

Run: `cd apps/packages/ui && bun run verify:openapi`

Expected: client path verification passes.

- [ ] **Step 5: Run the focused backend matrix**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing tldw_Server_API/tests/DB_Management/test_chacha_migration_v61.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v61.py tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py tldw_Server_API/tests/DB_Management/test_shared_workspace_chat_store.py tldw_Server_API/tests/DB_Management/test_shared_workspace_chat_store_postgres.py tldw_Server_API/tests/AuthNZ/integration/test_authnz_sharing_postgres.py tldw_Server_API/tests/Chat/test_chat_target_resolution.py tldw_Server_API/tests/Workspaces/test_workspace_job_status.py tldw_Server_API/tests/Workspaces/test_workspace_source_preview.py -q`

Expected: all selected tests pass; PostgreSQL may skip only through the repository fixture's unavailable signal.

- [ ] **Step 6: Run frontend, type, lint, design-state, and E2E checks**

Run: `cd apps/packages/ui && bunx vitest run src/components/Option/ResearchWorkspace/__tests__ src/services/tldw/domains/__tests__/shared-workspaces.test.ts src/components/Option/Playground/__tests__/playground-locale-mirror.test.ts --maxWorkers=1 --no-file-parallelism`

Run: `cd apps/tldw-frontend && bun run typecheck`

Run: `cd apps/tldw-frontend && bunx eslint extension/routes/option-research-workspace.tsx e2e/utils/page-objects/ResearchWorkspacePage.ts e2e/workflows/research-workspace.shared-recipient.spec.ts ../packages/ui/src/components/Option/ResearchWorkspace ../packages/ui/src/services/tldw/domains/shared-workspaces.ts ../packages/ui/src/types/shared-workspace.ts`

Run: `cd apps/packages/ui && bun run verify:design-system-state`

Run: `cd apps/tldw-frontend && bunx playwright test e2e/workflows/research-workspace.shared-recipient.spec.ts --project=chromium --reporter=line --workers=1`

Expected: all focused checks pass. Any unrelated baseline failure is recorded with command/output and must not hide a touched-scope failure.

- [ ] **Step 7: Run Bandit on every touched backend path**

Run: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Sharing tldw_Server_API/app/core/DB_Management/chacha/shared_workspace_chat_store.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py tldw_Server_API/app/core/AuthNZ/repos/shared_workspace_repo.py tldw_Server_API/app/core/Workspaces/job_status.py tldw_Server_API/app/core/Workspaces/source_preview.py tldw_Server_API/app/core/Chat/chat_target_resolution.py tldw_Server_API/app/api/v1/API_Deps/auth_deps.py tldw_Server_API/app/api/v1/endpoints/sharing.py tldw_Server_API/app/api/v1/schemas/shared_workspace_recipient_schemas.py tldw_Server_API/app/api/v1/utils/shared_workspace_recipient_route.py -f json -o /tmp/bandit_task_12020_40.json`

Expected: zero new findings in touched code.

- [ ] **Step 8: Commit integrated tests and docs**

```bash
git add tldw_Server_API/tests/Sharing tldw_Server_API/tests/DB_Management tldw_Server_API/tests/Chat tldw_Server_API/tests/Workspaces apps/tldw-frontend/e2e apps/tldw-frontend/lib/api/openapi.fingerprint.json Docs/Design/Pagination_Completion_Matrix.md Docs/User_Guides/Server/Organizations_and_Sharing.md Docs/Published/User_Guides/Server/Organizations_and_Sharing.md Docs/Development/Research_Workspace_Final_UAT_Runner.md
git commit -m "test(workspaces): verify shared recipient isolation"
```

### Task 11: Run Real Backend, WebUI, and CDP Acceptance and Close the Task

**Status:** Fix Round 2/5 implementation and fresh live evidence passed from reviewed head `5a648f8532`; pending controller review. The Backlog task remains In Progress and PR preparation is intentionally omitted by executor direction.

**Files:**
- Create: `apps/tldw-frontend/scripts/shared-research-workspace-cdp-uat.mjs`
- Create: `apps/tldw-frontend/__tests__/shared-research-workspace-cdp-uat.test.ts`
- Create: `apps/tldw-frontend/scripts/local-llm-forwarding-probe.mjs`
- Create: `apps/tldw-frontend/__tests__/local-llm-forwarding-probe.test.ts`
- Modify: `apps/tldw-frontend/scripts/research-workspace-uat-runner.mjs`
- Create: `Docs/Reviews/assets/2026-08-21-shared-research-workspace-recipient-uat/README.md`
- Create: `Docs/Reviews/assets/2026-08-21-shared-research-workspace-recipient-uat/evidence.json`
- Create: `Docs/Reviews/assets/2026-08-21-shared-research-workspace-recipient-uat/desktop-shared-workspace.png`
- Create: `Docs/Reviews/assets/2026-08-21-shared-research-workspace-recipient-uat/desktop-grounded-answer.png`
- Create: `Docs/Reviews/assets/2026-08-21-shared-research-workspace-recipient-uat/mobile-shared-workspace.png`
- Create: `Docs/Reviews/assets/2026-08-21-shared-research-workspace-recipient-uat/mobile-source-preview.png`
- Create: `Docs/Reviews/assets/2026-08-21-shared-research-workspace-recipient-uat/revoked-share.png`
- Update through Backlog CLI: `TASK-12020.40`

**Fix Round 1/5 closeout:** Partition direct/background GET reuse by resolved server and safe auth scope; fence assistant restoration against newer explicit selections; replace the transition observer with separate bounded owner-management and member-Chats operation policies; enforce multiset expected-error correlation and a closed evidence schema; and prove provider-context exclusion through an unchanged-body local forwarding probe. The fresh final31 fixture passed all 15 acceptance checks with closed strict/transition ledgers and five separately inspected screenshots. Focused changed behavior passed `191` tests, the required matched suite passed `111`, and the two correctly resolved package tests passed `13`.

**Fix Round 2/5 scope:** Bind direct GET reuse to the immutable request configuration used by that exact request; require exact transition operation status contracts and persona-isolation identity proofs; and reject non-loopback or malformed provider-probe traffic before any upstream fetch. Each finding proceeds through an isolated RED/GREEN cycle before a fresh live CDP run.

**Fix Round 2/5 closeout:** The four findings completed isolated RED/GREEN cycles. The amended frontend matrix passed `159` runner/probe tests plus `101` correctly resolved shared-package tests. Fresh `final32-fix2-1787446794-16413` passed all canonical 15 checks with closed strict and exact-status transition ledgers, settings `2x200`, race `200/409/200/409` with equal replay hashes, three unchanged clean loopback-probe requests, exact distinct persona identities, and five visually inspected screenshots. No Python differs from the reviewed base; focused lint, script syntax, evidence leak scans, protected-file modes, and diff checks are recorded in the Task 11 report. Status remains pending controller review; no PR/push.

- [x] **Step 1: Write runner contract tests before the live script**

Assert required admin and disposable-account credentials, API/Web/CDP URLs, evidence directory, local-provider preference, redacted output, nonzero exit on any undeclared HTTP failure/console error/page error/request failure, and no computer-control fallback. Secrets may come only from environment variables and must never appear in command logs, screenshots, or `evidence.json`.

- [x] **Step 2: Implement an explicit CDP live runner**

Connect to an already running Chrome instance with:

```js
const browser = await chromium.connectOverCDP(process.env.TLDW_CDP_URL)
```

Use admin-authenticated API calls only for fixture provisioning and browser CDP for all product UI interaction. Derive unique owner/member/nonmember usernames from a run ID, create them with the disposable fixture password, create the team/share, and register cleanup metadata without attempting destructive cleanup on failure. Open three isolated CDP browser contexts, log each persona in through the WebUI, and assert storage/cookies do not cross contexts. Provision the owner workspace, two real ingested files, unrelated owner sentinel media, and recipient local sentinel content. Poll the canonical workspace source/status envelope until both shared sources are `queryable`; time out with the final bounded status payload rather than proceeding early.

- [x] **Step 3: Run the deterministic runner tests**

Run: `cd apps/tldw-frontend && bunx vitest run __tests__/shared-research-workspace-cdp-uat.test.ts __tests__/research-workspace-uat-runner.test.ts --maxWorkers=1 --no-file-parallelism`

Expected: configuration, redaction, failure classification, and evidence checks pass.

- [x] **Step 4: Start actual services and run the CDP matrix**

Use the repository virtual environment for the multi-user backend and the Next.js WebUI. Configure the local OpenAI-compatible `llama.cpp` provider at `http://127.0.0.1:9099/v1/chat/completions` when healthy; otherwise select another configured provider and record its effective provider/model.

Require `TLDW_SHARED_UAT_ADMIN_USERNAME`, `TLDW_SHARED_UAT_ADMIN_PASSWORD`, and `TLDW_SHARED_UAT_FIXTURE_PASSWORD`. The runner exchanges the admin credentials for a short-lived token in memory, creates run-scoped accounts, and never persists credentials. Optional owner/member/nonmember username prefixes may change readable evidence labels but are not identity inputs.

Run the live harness with explicit environment values:

```bash
cd apps/tldw-frontend
TLDW_E2E_SERVER_URL=http://127.0.0.1:18001 \
TLDW_WEB_URL=http://127.0.0.1:18082 \
TLDW_CDP_URL=http://127.0.0.1:9222 \
TLDW_SHARED_UAT_ADMIN_USERNAME="$TLDW_SHARED_UAT_ADMIN_USERNAME" \
TLDW_SHARED_UAT_ADMIN_PASSWORD="$TLDW_SHARED_UAT_ADMIN_PASSWORD" \
TLDW_SHARED_UAT_FIXTURE_PASSWORD="$TLDW_SHARED_UAT_FIXTURE_PASSWORD" \
node scripts/shared-research-workspace-cdp-uat.mjs
```

The run must prove: shared identity/source isolation; all-source and subset answers; verified citation preview; no unrelated owner or recipient sentinel in context/answer/citations; history after reload; owner recipient-style view; malformed/nonmember neutral failures; revocation; saved transcript visibility in Chats; blocked revoked preview; clean desktop/mobile layouts; and no extra banner bars.

- [x] **Step 5: Run the API idempotency race probe against the same backend**

Race two matching request IDs, assert one generated/stored turn and replay equivalence, then reuse the ID with a changed fingerprint and assert typed `409 request_id_conflict`. Record status codes, request IDs, response hashes, and redacted timings in `evidence.json`.

- [x] **Step 6: Inspect all screenshots and the request/error ledger**

Verify visually that controls/text do not overlap, the header is compact, Sources/Chat are the only core panes/tabs, evidence is readable, mobile has no horizontal overflow, and revoked state contains no owner/local data. The ledger must contain no undeclared status >= 400, status 0, failed request, page error, console error, runtime overlay, old full-media call, local workspace call, or mutation/tool request in shared mode.

- [x] **Step 7: Run final repository checks and inspect the diff**

Run: `git diff --check`

Run: `git status --short`

Run: `rg -n "shared-with-me/.+/media/|SharedWorkspaceBanner|SharedWorkspaceProvider" apps/packages/ui/src apps/tldw-frontend tldw_Server_API/app Docs/User_Guides Docs/Published/User_Guides`

Expected: clean whitespace; only intended files plus the two pre-existing untracked watchlist templates; no active old endpoint/banner/provider references.

- [x] **Step 8: Update Backlog and commit live evidence**

Use `backlog task edit TASK-12020.40` to check all acceptance criteria/definition-of-done items, append exact verification commands/results and skips, link the design/plan/PR, and add a final summary covering the security boundary, recipient persistence, UI, and live evidence.

```bash
git add apps/tldw-frontend/scripts/shared-research-workspace-cdp-uat.mjs apps/tldw-frontend/__tests__/shared-research-workspace-cdp-uat.test.ts apps/tldw-frontend/scripts/research-workspace-uat-runner.mjs Docs/Reviews/assets/2026-08-21-shared-research-workspace-recipient-uat backlog/tasks/task-12020.40\ -\ Bind-recipient-shared-workspace-sources-and-chat-to-the-canonical-share.md
git commit -m "test(workspaces): record shared recipient live acceptance"
```

- [ ] **Step 9: Prepare the PR without bypassing the human merge gate**

Push the branch and open a PR against `dev` with the task/spec/plan links and exact verification evidence. The human requester must write the required `Change summary` in their own words explaining what changed and why these implementation choices were made; do not fabricate that human-owned summary.

## Final Review Checklist

- [ ] Shared route parsing accepts exactly one positive base-10 safe integer and otherwise fails closed.
- [ ] Local Research Workspace never mounts or fetches while `shared` is present.
- [ ] Old full-media recipient route is absent with no redirect or alias.
- [ ] Every recipient read/chat route enforces global permission and current share membership.
- [ ] Current membership comes from authoritative AuthNZ queries; stale team/org claims cannot list or open a share.
- [ ] Missing/revoked/deleted/out-of-scope responses are indistinguishable.
- [ ] Recipient 401, permission, validation, rate-limit, conflict, and operational failures use typed bounded details.
- [ ] Source responses and previews expose canonical source IDs only and obey all bounds.
- [ ] SQLite/PostgreSQL schema, races, fencing, rollback, replay, and cascade behavior match.
- [ ] Recipient tenant keys come only from the active ChaChaNotes `client_id`; PostgreSQL RLS is enabled and forced and blocks cross-recipient reads and writes.
- [ ] All recipient-chat SQL remains inside `core/DB_Management`, and conflicted receipts receive bounded cleanup.
- [ ] Retrieval receives a non-empty exact media allowlist and no owner notes/chats.
- [ ] Every returned document has verified in-scope provenance.
- [ ] Completed/active receipt handling does not consume generation-rate capacity.
- [ ] Generation uses recipient/current-share-scoped credentials with request-state scope filtering disabled, JSON-delimited untrusted evidence, no tools/stream/fallback, and at least one verified citation.
- [ ] Complete prompts fit a locally computed model-context budget; no shared content reaches a remote tokenizer or token-count endpoint.
- [ ] Revocation/source changes are checked before generation and persistence.
- [ ] Failed requests save no messages and preserve the frontend draft/selection.
- [ ] Shared UI contains no trust/shared/migration/status banner stack or local feature controls.
- [ ] Desktop/mobile keyboard, focus, live-region, and overflow checks pass.
- [ ] Current API/docs/OpenAPI artifacts describe only the replacement contract.
- [ ] Bandit reports no new touched-scope findings.
- [ ] Real backend/WebUI/CDP evidence proves the owner/member/nonmember/sentinel/reload/revoke matrix.
- [ ] Backlog closeout and the human-authored PR change summary satisfy repository policy.
