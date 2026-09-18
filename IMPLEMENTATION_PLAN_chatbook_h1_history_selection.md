# Chatbook H1 History Selection and Fork Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Use only delegation authorized by the current session or applicable instructions.

**Goal:** Make normal send and Fork use the user's selected history, isolate current local copies from their source, and preserve the owner of uncertain fork operations.

**Architecture:** A shared immutable selection contract connects per-view cursors to existing storage owners and the existing context composer. Native and Dexie owners validate complete snapshots and preserve immutable reviewed legacy interpretations. Current fork adapters consume stable projections and return explicit outcomes; later H2/H3/H4 implementations retain their separate transaction, recovery and sync gates.

**Tech Stack:** Existing TypeScript/React shared UI, Dexie 4, Vitest 4, Playwright, FastAPI/Pydantic, Python `hashlib`, SQLite/PostgreSQL chat backends and `@noble/hashes` 2.0.1. Follow the checked-out lockfiles and project virtual environment; no added runtime dependency.

**Spec:** [H1 selected history and fork ownership](Docs/Design/2026-09-16-chatbook-h1-history-selection-design.md). Read it and the linked [D1–D6 closure](Docs/Design/2026-09-16-chatbook-chat-parity-review-closure.md) before implementation. Tracking: TASK-13261.1; design: TASK-13261.

## Global Constraints

- Full parity targets are the WebUI and extension full-page UI; the compact sidepanel uses the same selection/action services and expands into the extension's own full-page UI.
- `tldw-agent` executes server-directed OS/tool actions in external environments. It has no Chatbook runtime, chat storage, synchronization, browser hosting or model-hosting responsibility.
- Preserve independent local ownership and optional synchronization. A server/account change cannot silently reassign a local conversation or operation.
- Preserve the client-managed context composer from TASK-188; do not introduce a monolithic server context-preview service.
- Reuse the repository's Python, Pydantic, DB backend, Dexie, TypeScript, Vitest and Playwright toolchains; add no runtime dependency for H1.
- Preserve existing request-scope cancellation and persistence rollback from TASK-13023, and integrate without overwriting TASK-13260.15 or the active UAT chat fixes.
- No normal-chat action may use a rendered index, timestamp order or text equality as message ancestry or mutation identity.
- H1 cannot advertise H2/H3 receipt recovery, rich fork fidelity, H4 sync compatibility or F02 independent generation as implemented.

---

## Execution baseline

Execution started from the reviewed design commit on `codex/chatbook-h1-history-selection`; per-stage status and the execution ledger record progress. This plan is not a passing test report. The reviewed server pin is `59049e094e0845a4611ea725ae19b7c1754ea709`; Chatbook is `24094f23d59c7a9d3cfac964c19fd263bc0393b2`. Source paths below were checked against that server object. New paths are explicitly labeled Create.

Fresh remote verification on 2026-09-17 found server dev unchanged and Chatbook advanced first to `1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6`, then to `c97a64eba54d18f88cecc77bf6233e208df8bf24`. The [source delta review](Docs/Design/2026-09-17-chatbook-chat-parity-source-refresh.md) records the C07/C08 settings/provider/context refinements and the later batch-ingestion/setup/clipboard changes. A third read-only refresh during Task5 preparation found Chatbook `d8fb4053f9a27a799d5cdb8ee58f7fd1de91efce`, adding local Library STT failure diagnostics. A fourth read-only refresh found `e89f28d751bc8a5b4f4545b8894b87437252c657` (PR 2704): real Library excerpts and warm handoffs, workspace link Undo, settings/capture lifecycle and local RAG identity refinements. The preserved source audit maps these to broader parity acceptance; no identified H1 ancestry/admission/fork contract changes follow. Historical evidence retains its original pin; broader parity remains open.

Execution uses the root checkout's existing virtual environment with this isolated worktree as cwd/PYTHONPATH. After a frozen Bun install, `pnpm exec` attempted dependency reconciliation; use the installed `apps/packages/ui/node_modules/.bin/vitest` from the UI package cwd for equivalent focused test commands. The shared JSON fixture is explicitly tracked despite the repository's JSON ignore rule.

- [x] Read the current Backlog workflow and TASK-13261.1; set implementation status to In Progress when actual execution starts. Keep design and implementation completion separate.
- [x] Use the worktree workflow to create an isolated `codex/` branch from freshly verified `dev`. The current `codex/fresh-install-uat-fixes` checkout contains unrelated edits. Do not checkout, reset, stash or stage that work. Bring only the approved design/plan/task artifacts into the isolated worktree.
- [x] Compare the new base against the reviewed pin for all touched paths. Read TASK-188, TASK-13023, TASK-13260.15 and active UAT chronology/acknowledgement/provider fixes. Integrate their invariants rather than replacing the files with pinned copies. Record the actual implementation base in this plan and Backlog.
- [x] Read repository and app AGENTS instructions. Use the existing environment/lockfiles. Shared UI tests use Vitest, not Bun's test runner. Commands below are run from the repository root unless a working directory is stated.
- [x] Capture focused baseline results before each affected stage. Reproduce a target behavioral defect with its test before the corresponding fix; do not demand unrelated full-suite success before a bounded change.

Execution worktree: `.worktrees/chatbook-h1-history-design`, branch `codex/chatbook-h1-history-selection`, implementation base `f00e12a5aa` with reviewed dev `59049e09`. The ledger records the completed native and client scope baselines; the same focused-baseline policy continues for remaining stages. Main UAT reconciliation is read-only, most recently at `a684ef4dd2`, with the scoped mirror helper at `c415ac44c3`.

Keep each task's failing/passing command, relevant changed paths and commit in TASK-13261.1. A commit includes only that task's changes and tracking. Run `git diff --check`, relevant lint/type checks and Bandit for touched Python before its commit; never bypass hooks. A failed unrelated baseline is recorded with evidence, not hidden by disabling tests.

## File responsibilities and interfaces

| Create path | Responsibility |
|---|---|
| `apps/packages/ui/src/types/history-selection.ts` | Shared wire/domain types from spec, tagged normal/comparison fork inputs, structured errors. |
| `apps/packages/ui/src/utils/history-selection.ts` | Pure graph/path validation, canonical digests and comparison projection validation. No database or UI imports. |
| `apps/packages/ui/src/db/dexie/history-selection.ts` | Complete local snapshots, immutable legacy projections, owner-bound bookmarks and selected user admission. |
| `apps/packages/ui/src/services/chat-history-selection.ts` | Existing-owner adapters and client composer/selection lease boundary; no runtime host. |
| `apps/packages/ui/src/hooks/chat/useHistorySelection.ts` | Per-view cursor/revision and conditional result-following, separate from workspace and auth scope. |
| `apps/packages/ui/src/components/Common/Playground/HistorySelectionReview.tsx` | Shared accessible legacy review and stale/unsupported states. |
| `apps/packages/ui/src/db/dexie/fork-operations.ts` | Owner-bound pending dispatch records and same-intent claims; no invented server receipt. |
| `tldw_Server_API/app/core/Chat/history_selection.py` | Immutable selection/snapshot values, canonical encoding and pure native validation. |
| `tldw_Server_API/app/api/v1/schemas/history_selection_schemas.py` | Strict endpoint request/response envelopes using the core selection values. |
| `tldw_Server_API/tests/Chat/fixtures/history_selection_v1.json` | Shared TypeScript/Python contract vectors with manually specified expected paths and digest inputs. |

Existing modules retain their current responsibilities. Add owner methods to `MessageStore` and its `CharactersRAGDB` delegation. Extend the existing API client, mounted hooks, stores and controllers listed per task. Do not create a parallel conversation database, a generic accepted-turn platform or a second settings service.

The following interfaces are the stage boundaries, updated as their owning tasks are implemented. Fork interfaces remain proposed until Stage 4. Implementers may refine internal helpers, but must update consumers and this plan together if a public signature changes.

```typescript
// types/history-selection.ts: use readonly members for captured objects.
type HistoryNodeV1 = {
  id: string
  revision: string
  parent_id: string | null
  role: string
  settled: boolean
  legacy_projection_id?: string
}

// Throws HistorySelectionError with a stable code on missing/cyclic ancestry.
function resolveParentPath(
  nodes: readonly HistoryNodeV1[], cursor: HistoryCursorV1
): readonly HistoryNodeV1[]

function resolveLegacyProjection(
  nodes: readonly HistoryNodeV1[], orderedPathIds: readonly string[],
  cursor: HistoryCursorV1, projectionId?: string
): readonly HistoryNodeV1[]

function resolveHistorySelection(
  snapshot: HistorySelectionSnapshotV1,
  view: HistoryViewSelectionV1,
  purpose: "send" | "fork",
  requestContextDigest: string
): HistoryResolutionV1

// HistoryResolutionV1 = ready(selection, selected rows), legacy_review_required,
// stale_selection, invalid_history, or unsupported_history_capability.
// All non-ready variants contain a code and no writable child/admission.

function captureHistorySnapshot(
  owner: HistoryOwnerV1, view: HistoryCaptureRequestV1["view"],
  purpose: "send" | "fork", signal?: AbortSignal
): Promise<HistoryCaptureResultV1>

function prepareHistoryContext(
  payload: unknown, validate_lease: () => boolean
): PreparedHistoryContextV1

function finalizeHistorySelection(
  owner: HistoryOwnerV1, capture: HistorySelectionCaptureV1,
  prepared: PreparedHistoryContextV1,
  currentView: HistoryViewSelectionV1
): HistorySelectionV1

function confirmLegacyHistoryProjection(
  owner: HistoryOwnerV1, scope: HistoryBookmarkScope,
  confirmation: LegacyHistoryProjectionConfirmV1,
  view: HistoryViewSelectionV1, signal?: AbortSignal
): Promise<LegacyHistoryProjectionV1>

function appendSelectedUser(
  owner: HistoryOwnerV1, selection: HistorySelectionV1,
  input: Message, opts?: HistoryOperationOptions
): Promise<HistoryAdmissionV1>

function settleAcceptedAssistant(
  owner: HistoryOwnerV1, admission: HistoryAdmissionReferenceV1,
  input: Message, opts?: HistoryOperationOptions
): Promise<Message | {id: string}>

function prepareLocalFork(request: ForkRequestV1): Promise<LocalForkProjectionV1>
function commitLocalFork(projection: LocalForkProjectionV1): Promise<ForkResultV1>
function createBranchMessage(deps: BranchDependencies):
  (request: ForkRequestV1) => Promise<ForkResultV1>
```

`HistoryCaptureResultV1` is a successful `captured` result or the same structured non-ready errors as `HistoryResolutionV1`. `HistorySelectionCaptureV1` holds snapshot/selected manifest rows, explicit `selected_content` bound by message ID/revision/order, view fence, purpose and storage-context digest, without a final request digest. The shared `nodes` member is the complete lightweight manifest; selected content carries text and ordered images separately. `bindSelectedHistoryContent` and native `bind_selected_history_content` validate this binding; owners still load and fence the content coherently. `PreparedHistoryContextV1` holds the immutable credential-free composer payload or fork policy, its request-context digest and a small adapter over existing request-scope/context/connection lease validation. `finalizeHistorySelection` must receive these inputs explicitly; it cannot consult the current global draft. Admission receives the finalized value after both phases.

Task2.3 refines the client boundary to take an explicit `HistoryOwnerV1`: local profile/owner/conversation; native canonical conversation with verified request scope, separate workspace scope and lease validator; or an unavailable code. Cancellation belongs to each operation. Finalization is synchronous over the detached canonical JSON payload and originating view; source/storage validation belongs to the actual owner append transaction. Dispatch the same frozen `prepared.payload`. `HistoryBookmarkScope` is `{profile_id, client_session_id}`; native bookmarks use the browser profile without adopting local conversation ownership. Native capability is established by a successful capture on the same live adapter, not by constructing an object containing a stored owner key. The report records exact public types and capability limits; mounted consumers remain Stage3 work.

`BranchDependencies` is the typed extraction of existing handler dependencies plus the selection/operation adapters; it is not a service locator. `LocalForkProjectionV1` contains captured owner/source fences, ordered source revisions, preallocated ID map, allowlisted child rows/files and the immutable request. Preparation does not write a child; commit revalidates the source transactionally.

```python
# MessageStore methods; expose through CharactersRAGDB delegation.
def get_conversation_history_snapshot(
    self, conversation_id: str, *, owner_client_id: str,
    owner_key: str | None = None, projection_id: str | None = None,
    conn=None, lock_for_update: bool = False,
) -> HistorySelectionSnapshotV1: ...

def get_conversation_history_selected_content(
    self, conversation_id: str, message_ids: Sequence[str], *,
    snapshot: HistorySelectionSnapshotV1, owner_client_id: str,
    owner_key: str | None = None, conn=None,
) -> tuple[dict[str, Any], ...]: ...

def confirm_legacy_history_projection(
    self, confirmation: Mapping[str, Any], *, owner_client_id: str,
    owner_key: str | None = None, conn=None,
) -> dict[str, Any]: ...

def validate_history_selection(
    self, conversation_id: str, selection: Mapping[str, Any], *,
    owner_client_id: str, owner_key: str, conn: Any,
) -> tuple[HistorySelectionSnapshotV1, tuple[dict[str, Any], ...]]: ...

def append_selected_history_input(
    self, conversation_id: str, selection: Mapping[str, Any],
    message: Mapping[str, Any], *, owner_client_id: str,
    owner_key: str, conn: Any | None = None,
) -> dict[str, Any]: ...  # full owner-issued HistoryAdmissionV1

def append_selected_history_inputs(
    self, conversation_id: str, selection: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]], *, owner_client_id: str,
    owner_key: str, conn: Any | None = None,
) -> dict[str, Any]: ...  # final-input admission; atomic server chain

def settle_history_admission(
    self, conversation_id: str, reference: Mapping[str, Any],
    message: Mapping[str, Any], *, owner_client_id: str,
    owner_key: str, conn: Any | None = None,
) -> str: ...  # persisted assistant/tool result message ID
```

These are signature declarations, not placeholder implementations. Reuse the actual backend connection type when adding annotations. Transaction ownership follows existing caller-connection conventions; a supplied connection is not committed independently.

Task 2.1's native confirmation returns a detached JSON-shaped `LegacyHistoryProjectionV1` mapping that the API validates against its strict envelope. Its selected-content reader retains parsed `tool_calls` and `extra_metadata` alongside ID/revision/text/ordered images; Task 2.2's implementation carries these through both typed binders and the wire capture. The Task2.2 method names above distinguish client-managed stable-ID admission from atomic server-owned chains; their independent review and fix review are complete. The optional DB-internal owner-key fallback is not a cross-server browser namespace; authenticated adapters supply their verified owner key. Protected legacy descendants require the explicitly selected projection ID; Python's matching resolver uses the keyword-only `projection_id` argument.

## Stage 1: selected-path contract and identity

**Goal:** Implement one deterministic contract before changing mounted send/fork behavior.

**Success Criteria:** Graph resolution, tagged empty boundaries, canonical digests and comparison order agree across fixtures; text equality cannot merge distinct variant identities.

**Tests:** H1-A/B/G contract vectors, invalid graph cases and generated acyclic/cyclic graph properties.

**Status:** Complete.

### Task 1.1: types, pure resolver and canonical vectors

**Files:**

- Create the two shared type/utility modules, core Python module, schema envelope module and JSON fixture listed above.
- Create `apps/packages/ui/src/utils/__tests__/history-selection.test.ts`.
- Create `tldw_Server_API/tests/Chat/unit/test_history_selection.py`.
- Modify `apps/packages/ui/src/utils/message-variants.ts` and its existing `__tests__/message-variants.test.ts`.

**Interfaces:** Consumes the spec's cursor/interpretation/digest definitions. Produces `resolveParentPath`, `resolveHistorySelection`, strict `HistorySelectionV1`, `CompareHistorySelectionV1`, snapshot/admission/projection types and shared canonical vectors.

- [x] Add failing path tests. Include two equal-text assistant IDs, descendants on only one branch, `before_message(root)`, explicit empty, missing parent, duplicate ID, cycle and cross-conversation rejection. Start with this self-contained pure case:

```typescript
import { expect, it } from "vitest"
import { resolveParentPath } from "../history-selection"
import type { HistoryNodeV1 } from "@/types/history-selection"

it("retains the chosen stable branch and an empty before-root boundary", () => {
  const rows: HistoryNodeV1[] = [
    { id: "u1", revision: "1", parent_id: null, role: "user", settled: true },
    { id: "a1", revision: "1", parent_id: "u1", role: "assistant", settled: true },
    { id: "a2", revision: "1", parent_id: "u1", role: "assistant", settled: true },
    { id: "u2", revision: "1", parent_id: "a2", role: "user", settled: true }
  ]
  expect([
    resolveParentPath(rows, { kind: "after_message", message_id: "a1" }).map(r => r.id),
    resolveParentPath(rows, { kind: "before_message", message_id: "u1" }).map(r => r.id)
  ]).toEqual([["u1", "a1"], []])
})
```

- [x] Run `pnpm --dir apps/packages/ui exec vitest run src/utils/__tests__/history-selection.test.ts src/utils/__tests__/message-variants.test.ts`. Verify the target missing resolver/identity behavior fails, not environment setup.
- [x] Implement an ID map, duplicate detection, visited-set parent walk and explicit before/after/empty handling. Reverse the collected ancestor chain once. Do not sort or compare content during ancestry resolution. Add the legacy interpretation adapter without mutating original rows.
- [x] Implement the spec's exact nine-member selection tuple and compact literal-Unicode UTF-8 encoding in the fixture. Include Unicode, image/reference identities, request-versus-storage context digests and fork-excluded settings. Use SHA-256 from existing libraries; Python and TypeScript must match fixed tuple inputs/results. Context hashes are validated by their own owner, not by reserializing another owner's settings.
- [x] Remove text-only variant identity dedup when stable distinct IDs exist; preserve same-ID/server-ID updates and existing no-server-ID-inheritance tests.
- [x] Add a comparison vector with two common rounds, A/B responses, B finishing last and an A-specific follow-up. Validate the A projection's semantic IDs/order independently of stored cross-model parent edges.
- [x] Add Hypothesis properties in the Python pure test: valid ancestor paths contain no repeats, before-boundary is a proper prefix, cycles/missing parents reject, and source input is unchanged. Do not add a new JavaScript property library.
- [x] Run the Vitest command and `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_history_selection.py -q`; review/type-check and commit the contract plus fixtures.

Execution evidence: contract commits `28c7d1904e`, `2fed0871c4`, `77dde3d2b4`; 18 focused Vitest tests and 17 Python tests passed. Independent task review and two scoped fix reviews resolved all identified P1/P2 issues. Focused type checking, Ruff, compileall and Bandit passed; full UI type checking exceeded the default 4 GB heap and remains a Stage 5 qualification item. Owner-specific fork-exclusion behavior remains assigned to Task 4.1. Detailed evidence is in the plan-owned SDD ledger and TASK-13261.1.

## Stage 2: complete owners, legacy acceptance and parent binding

**Goal:** Implement storage-backed selection validation and immutable legacy interpretations without changing the context composer's ownership.

**Success Criteria:** Complete source snapshots and CAS are real owner transactions; user append and assistant settlement cannot lose their accepted parent or accept forged metadata.

**Tests:** H1-C/D/H, more than 20,000 rows, dual review/replay, transaction rollback, metadata forgery, stale parent and SQLite/PostgreSQL concurrency.

**Status:** Complete (owner/API/adapter scope; browser durability remains Stage 5).

### Task 2.1: native owner snapshot, migration and legacy CAS

**Files:**

- Modify `tldw_Server_API/app/core/DB_Management/chacha/message_store.py`.
- Modify `tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py` only to factor reusable owner/fence reading without character readiness.
- Modify `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` for migration, table backend policies and store delegation.
- Modify `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py`.
- Create `tldw_Server_API/tests/DB_Management/test_history_selection_migration.py` and `tldw_Server_API/tests/DB_Management/test_history_selection_transactions.py`.
- Extend the shared history-selection node types, schema, pure resolvers and their two-language tests with protected `legacy_projection_id` provenance for explicit descendants.

**Interfaces:** Consumes Stage 1 core values. Produces complete `get_conversation_history_snapshot` and replay-safe `confirm_legacy_history_projection` with caller transaction support.

Implementation refinement: the native migration also adds nullable protected `messages.history_admission_json`, defaulting absent on legacy rows. Ordinary/imported metadata cannot establish owner-issued acceptance. Snapshots may receive an explicit requested projection and adapter-verified owner namespace; `owner_client_id` stays required. The shared legacy resolver follows matching protected descendants to an immutable base prefix or null without selecting a global latest branch. Actual admission writes remain Task 2.2.

- [x] Add migration tests for an unchanged legacy transcript, both schema backends, owner isolation and multiple immutable projections. Use the neighboring behavior-snapshot migration fixture patterns, but do not inherit that file's autouse pin to schema 65.
- [x] Add a complete reader test to the existing `db` fixture in `test_chacha_message_store.py`:

```python
def test_history_snapshot_preserves_equal_text_distinct_ids(db):
    owner_db = db["db"]
    conversation_id = db["conversation_id"]
    first = owner_db.add_message({
        "conversation_id": conversation_id, "sender": "user", "content": "repeat"
    })
    second = owner_db.add_message({
        "conversation_id": conversation_id, "sender": "user", "content": "repeat"
    })
    snapshot = owner_db.get_conversation_history_snapshot(
        conversation_id, owner_client_id="message-store-user"
    )
    assert {row["id"] for row in snapshot.nodes} == {first, second}
```

- [x] Run the three focused files and verify a behavior/missing-table failure. Implement the next available migration on both backends. The reviewed pin is schema 67; choose the next free version at execution, update every SQLite/PostgreSQL migration path and fresh bootstrap, and test upgrade idempotence. Do not renumber a migration already integrated by UAT.
- [x] Implement complete statement/transaction-consistent reads with distinct conversation/history/settings fences and deterministic manifest order. Include metadata/asset revisions; avoid tree limits, display page caps, truncated source projection helpers and binary-heavy review payloads.
- [x] Implement immutable `conversation_history_projections` owner/conversation keys and CAS. Check authorized same-ID replay before fresh-source validation; reject same ID with different content. Persist complete source digest/membership plus selected order. No settings overwrite or parent rewrite occurs.
- [x] Add real competing-connection tests for edit during snapshot/confirmation, confirmation after source change, two views confirming different paths, confirmation response loss followed by append/retry, and deleted conversation access. Verify lock ordering against existing message/metadata edit methods. Admission-specific races are Task 2.2.
- [x] Add a 20,001-row owner manifest fixture and equal-timestamp ties. Check no row disappears even when the display loader would be capped. Measure only enough to detect accidental per-row metadata/asset queries; do not introduce a benchmark framework.
- [x] Run focused SQLite tests and actual PostgreSQL integration using the existing provisioned fixture/`TEST_DATABASE_URL` path. Report unavailable PostgreSQL as unverified, not passing. Run touched-scope Bandit, review and commit.

Execution evidence: `8a293d1ef3` implements schema 68, coherent snapshots and immutable legacy CAS; `c11a8bd9e9` fixes review order and repeated large payload materialization. Initial qualification passed 79 Python and 19 TypeScript tests; the final changed native scope passed 32 real SQLite/PostgreSQL tests, with all 20,001 rows and accepted projection reopen retained. Independent review and scoped re-review resolved all identified native P1/P2 findings. Ruff, compileall, focused TS and touched-scope Bandit passed. The measured same-scope run dropped from 149.80s with observed 26.6 GiB RSS to 43.16s with about 629 MiB peak RSS; these are qualification observations, not a benchmark guarantee. API/admission/settlement and external asset capability checks remain Task 2.2.

### Task 2.2: native selection routes and accepted message persistence

**Files:**

- Modify `tldw_Server_API/app/api/v1/endpoints/chat.py`, `character_messages.py` and `character_chat_sessions.py`.
- Modify `tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py`, `chat_session_schemas.py` and the new selection schema envelopes.
- Modify `tldw_Server_API/app/core/Chat/chat_service.py`, `persistence_service.py` and `DB_Management/chacha/message_store.py`.
- Add a focused `core/Chat/history_context.py` adapter and unit coverage if needed to project already-saved, validated behavior within native selection admission. It must consume captured context without live character/persona/worldbook reloads; precise unsupported state is rejected before writes.
- Extend `DB_Management/ChaChaNotes_DB.py` only for delegation of the new owner methods, and `tests/DB_Management/test_history_selection_transactions.py` for real admission/settlement races.
- Modify `DB_Management/backends/sqlite_backend.py` only to register the fixed snapshot hash function at connection creation. Cover reused A→B→A connections and independent stores with active cursors; remove per-operation registration/cache.
- Extend the shared history-selection core/wire/types/binders and their focused tests as required to transport coherent tool/history fields from the captured native content; preserve the canonical selection tuple.
- Create `tldw_Server_API/tests/Chat_NEW/integration/test_history_selection_api.py`.
- Extend `tldw_Server_API/tests/Sync/test_sync_v2_chat_materializer.py` only to verify imported admission-shaped metadata cannot establish native protected acceptance, using the existing service/store fixtures.
- Extend `tldw_Server_API/tests/Chat_NEW/unit/test_chat_continuation_controls.py`, `tests/Chat_NEW/integration/test_chat_continuation_controls_integration.py`, `tests/Chat/unit/test_chat_service_call_params.py` and `tests/Character_Chat_NEW/integration/test_character_chat_stream_and_persist.py` under `tldw_Server_API/`.
- Extend `tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py` only for the real tool-auto-continuation accepted-parent path.

**Interfaces:** Consumes Task 2.1 owner methods. Produces the two selection/review routes; `MessageCreate.tldw_history_selection_v1` (user) and `MessageCreate.tldw_history_admission_v1` (assistant); accepted input/admission response; character assistant admission reference; and `ChatCompletionRequest.tldw_history_selection_v1` routed through existing `continuation_runtime`.

Implementation binding: derive the API owner key from normalized trusted request base/root_path and authenticated user identity; compare any supplied key rather than treating it as authority. Only the read-only capture request may omit a bootstrap owner key. Its response binds the strict view/snapshot; durable selections and mutations still require the key, stored by the browser under its verified server/account scope. Versioned completion messages are new current inputs, with generated IDs accepted atomically and no replay of a consumed source selection. Native accepted input revision is its row-version string, with a separate protected substantive-state hash detecting metadata/image drift without recursive admission hashing.

Admission refinement: accept a multi-input completion chain in one transaction against the original finalized selection, return the final input admission with the unchanged selection digest/manifest, and bind every accepted input's identity/version/state in protected metadata. Scope consumed-selection checks to the owner/conversation and protected server-completion provenance, including tool-only chains. Client-managed stable-ID matching retries remain distinct. Settlement checks the accepted chain and current authorization, not unrelated later history/settings changes. The complete manifest may include a same-statement, 200-Unicode-character text preview for recognizable legacy review.

Effective behavior refinement: versioned server completion uses the neutral context or a supported, validated saved character behavior projection read in the same owner transaction as selection validation. Support matching self-contained single-character prompt behavior. Do not read/create a live default character before admission, or silently reload live persona/character/exemplar/worldbook sources afterward. Invalid/missing snapshots or effective behavior/assets without a faithful bounded adapter return explicit unsupported capability before append. Record the exact supported and gated states; client-managed/stateless generation stays separate and unsupported states earn no parity credit.

- [x] Add endpoint tests for ownership/workspace scope, before-first/empty, stale manifest, unsupported sync owner and old unversioned request compatibility. Assert mutation counts, not merely response status.
- [x] Add client-managed append tests: parent derived from accepted path, conflicting supplied parent rejected, protected input metadata and user row committed together, rollback on metadata failure. Forge admission via ordinary metadata/settings/sync input and assert it cannot create owner acceptance.
- [x] Add assistant persistence tests for accepted input identity/version, wrong parent, incompatible edit/delete, same stable-ID retry with a different admission, late result after another view chooses a branch, and scope change. Exercise both ordinary generic `addChatMessage` and `CharacterChatStreamPersistRequest`, delegating to the same owner settlement operation. An unrelated branch must not invalidate the original parent.
- [x] Run focused files to red, then add strict request/response envelopes and owner operations. Preserve existing user/access/rate/image validation and use DB abstractions; do not insert direct SQL in endpoints.
- [x] Keep `storage_context_digest` owner-validated. Treat client `request_context_digest` as inert provenance; the client validates its composer lease. The server-owned completion path validates actual supplied request fields plus its storage context. Never treat a client hash as accepted server context.
- [x] Route versioned server-owned completion through selected rows; skip timestamp history reload and signature-overlap dedup only for this path. Append current-turn input rows with explicit parents, then bind assistant/tool/continuation persistence to the returned final input. Test the existing continuation branch that currently passes a null assistant parent.
- [x] Reject versioned sync-routed mutations before any write when no retained-selection/admission adapter exists. Keep default-dataset membership distinct from conversation enrollment. Do not add H4 materialization here.
- [x] Exclude selection/admission fields from provider params, `extra_body` and headers. Verify current normal system/worldbook/character behavior still passes, including multi-image history and existing continuation fixtures.
- [x] Run endpoint/service suites, focused lint/compile and touched-scope Bandit; review and commit the native path.

Execution evidence: implementation `3f8144d505` and fix `db1c6feddf` provide native selection, admission and generic/character settlement. The independent review found six P2 issues; scoped re-review confirms all six addressed and no new P1/P2. Final changed-scope evidence: 67 real SQLite/PostgreSQL tests passed with one intentionally SQLite-only parametrization skipped; 69 endpoint/context tests and one actual-factory compatibility test passed. Compile/diff checks and production Bandit passed with zero findings. Two proven baseline Ruff findings and third-party warnings are disclosed in the report. Client leases, local owners and mounted surfaces remain subsequent tasks; this does not close overall H1.

### Task 2.3: local owner projections, admission and bookmarks

**Files:**

- Create `apps/packages/ui/src/db/dexie/history-selection.ts`.
- Modify `apps/packages/ui/src/db/dexie/schema.ts`, `types.ts`, `chat.ts`, `helpers.ts` and `server-chat-mirror.ts` only at selection/admission interfaces.
- Create `apps/packages/ui/src/db/dexie/__tests__/history-selection.test.ts`.
- Create `apps/packages/ui/src/services/chat-history-selection.ts` and `src/services/__tests__/chat-history-selection.test.ts`.
- Modify `apps/packages/ui/src/services/tldw/TldwApiClient.ts` and `src/services/tldw/domains/chat-rag.ts` for strict selection/admission requests. The domain mixin supplies the live `addChatMessage` method; changing only the base-class duplicate does not update runtime behavior.
- Extend `src/services/tldw/service-prompt-scope-error.ts` and its existing tests with only the exact new versioned selection, legacy-confirmation and character-settlement routes/methods needed by the captured request lease. The transport rejects an unknown route carrying `servicePromptConfig`; do not remove or broadly widen that guard.

**Interfaces:** Consumes Stage 1 types and Task 2.2 routes. Produces `captureHistorySnapshot`, `finalizeHistorySelection`, `confirmLegacyHistoryProjection`, owner-scoped bookmark load/save, and immutable local accepted user metadata for later settlement.

- [x] Add tests for local/profile versus server/account identity, two independent client sessions, explicit empty bookmark, stale deleted target and existing scope invalidation. An old mirror with only `server_chat_id` stays unbound until explicit verified connection binding; migration must not adopt the currently logged-in account or upload content. Use existing scoped mirror test patterns; preserve pending UAT fixes.
- [x] Add the next free Dexie schema migration. The reviewed pin is version 14; do not assume 15 is still free. Add `historySelections` keyed by `[profile_id+client_session_id+owner_key+conversation_id]`, `historyProjections` keyed by `[owner_key+conversation_id+projection_id]`, and narrowly scoped local admission metadata in the existing message records. Migrate structure only; no timestamp ancestry rewrite.
- [x] Implement complete local snapshot reads and synchronous canonical hashes inside the transaction using existing Noble utilities. Do not keep a Dexie transaction alive across HTTP, Plasmo storage or Web Crypto awaits. Required external state that cannot be fenced must be excluded or explicitly unsupported under the spec.
- [x] Confirm local legacy projection and bookmark atomically; for server projections commit the bookmark only after owner acknowledgement. Persist a generated confirmation ID before dispatch and resolve matching replay through the owner. Verify two reviewed interpretations and their explicit descendants survive reopen.
- [x] Implement local selected user append and assistant settlement with protected metadata and explicit parents, not only a read-only selection helper. A local mirror of a server chat must use server admission and existing scoped mirroring, not local ownership.
- [x] Add the server owner adapter and strict version response validation. Return a captured snapshot/resolved rows before client composition; finalize the request-context digest only with the explicit prepared composer payload and a current lease. A missing/older selection endpoint stops dependent mutation; never fall back to display arrays. Keep workspace routing in its existing parameter object, separate from owner namespace.
- [x] Run `pnpm --dir apps/packages/ui exec vitest run src/db/dexie/__tests__/history-selection.test.ts src/services/__tests__/chat-history-selection.test.ts src/services/__tests__/chat-surface-scope.test.ts src/services/tldw/__tests__/TldwApiClient.request-scope.test.ts`. Transaction mocks are orchestration evidence only; real IndexedDB commit/rollback is required in Stage 5.
- [x] Review/type-check and commit the owner adapters.

Execution evidence: `f93b78c5d9` implements local/native client owners and Dexie version15; reviewed fix `0bcdc845ef` resolves all three reported P1/P2 findings, with no new P1/P2 in scoped re-review. Original affected scope passed187 tests; final changed scope passed233 tests across6 files with no skips, including actual domain/proxy error propagation and concurrent pending-confirmation cleanup. Focused owner TypeScript, formatting and diff checks pass. Expanded proxy-test typing reproduces nine exact baseline errors on unchanged lines; expected negative transport logging is disclosed. Native writes require a live capability capture, mirror IDs need authoritative association, and a durable dispatch marker preserves uncertain confirmations. Storage doubles are orchestration evidence only; real IndexedDB, mounted behavior and the stated unsupported-mode limits remain explicit downstream gates.

## Stage 3: mounted UI, hydration and normal send

**Goal:** Use the new contract in the actual WebUI, extension and sidepanel call paths.

**Success Criteria:** Swiping, reopening, legacy review, preparing a request and accepting a late response all use the intended scoped cursor and parent; the existing client composer and native saved-behavior adapter retain their distinct context responsibilities.

**Tests:** H1-A/B/C/D/G, actual mounted submit and restore paths, in-flight admission races and comparison regressions.

**Status:** Complete (mounted/client and native acknowledgement scope; final real-browser qualification remains Stage 5).

### Task 3.1: shared view cursor and history review

**Files:**

- Create `apps/packages/ui/src/hooks/chat/useHistorySelection.ts` and `src/hooks/chat/__tests__/useHistorySelection.test.tsx`.
- Create `apps/packages/ui/src/components/Common/Playground/HistorySelectionReview.tsx` and neighboring `__tests__/HistorySelectionReview.test.tsx`.
- Modify `apps/packages/ui/src/components/Option/Playground/PlaygroundChat.tsx`, `Playground.tsx`, `components/Sidepanel/Chat/body.tsx` and `routes/sidepanel-chat.tsx`.
- Place the single full-page history-selection provider above participating sidebar loaders in both actual shells: shared `components/Layouts/Layout.tsx` and `apps/tldw-frontend/components/layout/WebLayout.tsx`. Gate the provider to the actual H1 chat route and reuse it inside Playground, with a standalone fallback only when absent. Verify route entry/exit and both real layout boundaries; the shared layout intentionally bypasses its root under Next.js and cannot alone establish WebUI coverage. Passive `useMessageOption` consumers must not hydrate; explicitly retain scoped hydration in the existing non-H1 surface owners.
- Extend `components/Sidepanel/Chat/SidepanelHeaderSimple.tsx`, `ControlRow.tsx` and their existing route/handoff tests only where necessary to route H1 expansion to the extension full page with the scoped selection reference. Keep explicit WebUI draft/page-context handoff distinct.
- Modify `apps/packages/ui/src/hooks/usePlaygroundSessionPersistence.tsx`, `hooks/chat/useServerChatLoader.ts`, `store/playground-session.tsx`, `store/sidepanel-chat-tabs.tsx` and `store/option/types.ts`.
- Extend `hooks/useLoadLocalConversation.ts` and `hooks/chat/useServerChatHistoryId.ts` at their actual hydration/mirror ownership boundaries; extend `hooks/chat/__tests__/useServerChatHistoryId.test.tsx` and add focused local-load race/selection tests. The former directly formats and installs raw history after asynchronous reads; the latter currently reuses a bare server ID and can link the current local history. H1 loads must preserve selection fences and verified ownership through these real callbacks.
- Extend `apps/packages/ui/src/db/dexie/helpers.ts` only at the selected-history formatting boundary. The live hydration imports `formatToMessage`/`formatToChatHistory` there; avoid legacy variant collapse and timestamp sorting for an explicit selected path. Keep canonical provider roles/tool fields distinct from presentation role normalization.
- Extend `apps/packages/ui/src/hooks/__tests__/usePlaygroundSessionPersistence.test.tsx`, `useServerChatLoader.scope.test.tsx` and `src/store/__tests__/playground-session-store.test.ts`; run the existing `useServerChatLoader.test.ts` regression suite unchanged. New loader scope/enablement tests belong in the mounted scope suite rather than a redundant change to the older loader suite.
- Modify canonical `apps/packages/ui/src/assets/locale/en/playground.json`; regenerate `apps/packages/ui/src/public/_locales/en/playground.json` through the existing locale script. Use the shared Playground namespace in both shells.

**Interfaces:** Consumes owner adapter snapshots/bookmarks. Produces scoped view selection, explicit stale/review states and selected rendering for every hydration/control path.

- [x] Write mounted tests that open the same conversation in A and B, choose different variants, reopen A's bookmark and deliver a result captured in B. Assert selected IDs/revisions and visible history, not only setter calls.
- [x] Add a legacy review test with an omitted alternative, a before-first cursor and a source mutation during confirmation. Confirming a rendered subset cannot mark the full manifest reviewed. Include keyboard selection, accessible error/focus behavior and virtualized row identity.
- [x] Run failing tests, then implement the hook with one per-view revision and conditional result following. Do not use the single existing `tldw-playground-session` localStorage record as global selection authority.
- [x] Integrate both swipe controls and all listed hydration paths. Make formatters accept a resolved selection; their legacy no-selection signature can remain for read-only callers but must not enter H1 mutation paths. A missing bookmarked ID produces a visible stale choice.
- [x] Integrate the shared review UI in both full-page shells and compact sidepanel. Carry scoped selection/projection references into extension full-page expansion. Do not imply D3 active stream handoff is implemented.
- [x] Add review/pending/stale strings to canonical `apps/packages/ui/src/assets/locale/en/playground.json`, using neighboring Playground namespaces and existing English fallback. Run `node scripts/sync-public-locales.js playground.json` from `apps/extension` and review generated changes; do not hand-edit public locale output or invent translations. Direct Node avoids pnpm11's attempted dependency reconciliation in this frozen Bun environment. Record any additional generated locale files in Backlog.
- [x] Run the mounted/hydration suites, review/type-check and commit.

Execution evidence: `9a2c54ff7f` mounts independent selection, complete legacy review, scoped restore and extension expansion; `5881b02870` fixes destination URL replay, shared-session validity overriding tab ownership, and unreadable/unbindable legacy mirrors. Independent task review and scoped fix review are complete with all three P2 and both minor items addressed and no new P1/P2. The final amended scope passed64/64 tests across5files, including actual mounted Playground/session/controller/Bind compositions; earlier shared246 cases and22 WebUI boundary cases passed across recorded runs. Core TypeScript, formatting and whitespace checks pass. Expanded shared/WebUI typing retains two unchanged prompt-sync diagnostics; ordinary model-count test stdout is disclosed. Real IndexedDB/browser/visual proof and actual send/settlement remain subsequent tasks.

### Task 3.2: freeze normal request preparation and accepted settlement

**Files:**

- Modify `apps/packages/ui/src/hooks/chat/useChatActions.ts`, `hooks/useMessage.tsx`, `hooks/useMessageOption.tsx`, `hooks/chat-modes/normalChatMode.ts` and `hooks/chat-helper/index.ts` (the actual `saveMessageOnSuccess` implementation).
- Extend `hooks/chat-modes/chatModePipeline.ts` and `src/types/chat-modes.ts` only at the prepared-request/admission/settlement boundary. The final provider payload is assembled in the shared pipeline after asynchronous prompt preparation, dynamic UI and steering injection; finalization before those steps would bind the wrong request. Preserve all other modes' existing behavior when H1 is absent.
- Extend `src/models/index.ts`, `src/models/ChatTldw.ts` and `src/services/tldw/TldwChat.ts` only as required to capture the actual resolved model/tool/provider inputs before finalization and use an explicit stateless client-managed inference request. The current model builder rereads ambient settings/tools and defaults server autosave from the active conversation. Reuse existing normalization and preserve legacy behavior outside H1.
- Modify `apps/packages/ui/src/utils/generate-history.ts` only for explicit input normalization/identity-safe versioned assembly.
- Extend `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx`, `hooks/chat-modes/__tests__/normalChatMode.overlay.test.ts` and `hooks/chat-helper/__tests__/saveMessageOnSuccess.scope.test.ts`.
- Add focused shared-pipeline admission coverage alongside the existing `chatModePipeline.abort-lifecycle`, `conversation-id`, `error-recovery.guard` and `provider-recovery` suites; extend the appropriate `saveMessageOnError` regression to ensure a provider error after admission does not append a duplicate user or erase the accepted input.
- Extend relevant model/transport tests beside `src/models/__tests__/pageAssistModel.mcp-tools.test.ts`, `ChatTldw.stream-metadata.test.ts` and `src/services/__tests__/tldw-chat.message-sanitization.test.ts` so the actual outbound request uses the finalized settings/messages and cannot independently autosave an already admitted turn.
- Create `apps/packages/ui/src/hooks/__tests__/useMessage.history-selection.test.tsx` and `src/hooks/chat/__tests__/useChatActions.history-selection.test.tsx`.
- Extend `db/dexie/types.ts`, `db/dexie/history-selection.ts`, the existing selection controller/review component and their tests for narrowly scoped turn recovery stored alongside bookmarks. Keep recovery outside transcript rows and preserve pending confirmations through atomic bookmark updates.

**Interfaces:** Consumes immutable owner selection/admission plus existing composer/connection leases. Produces exact normal provider history and parent-bound settlement, while retaining current comparison `historyForModel` inputs.

- [x] Add a failing full-page and sidepanel fixture: swipe to A1 while raw history's newest variant is A2, send once, and inspect the actual normal-mode/provider input. Assert A1 is present and A2 is absent even when its text matches A1.
- [x] Add two deferred cases: change the origin selection during preparation (zero admission requests); change it after admission dispatch (retain original pending intent and handle returned accepted/unknown result). Do not assert the remote owner observed a local swipe.
- [x] Add composer lease invalidation while admission is in flight. If the user append succeeded, preserve the accepted unsent input and issue no provider call under changed credentials/settings. An unrelated view selection alone must not invalidate the accepted request. Preserve TASK-13023's scope rollback behavior.
- [x] Call `captureHistorySnapshot`, feed its selected rows into current TASK-188 composer, then call `finalizeHistorySelection` with the exact immutable prepared payload and current view/lease. Bind client-only prompts/overlays to `request_context_digest`, owner settings/assets to `storage_context_digest`. Validate the client lease immediately before admission dispatch and before provider dispatch; never ask the server to reconstruct local overlays or read ambient drafts inside capture.
- [x] Replace live `baseMessages`/`baseHistory` normal-path authority in both mounted handlers. Keep current normal provider filtering for system/image-generation rows and singular/plural image normalization explicit in regression expectations. Versioned history must not be deduplicated by content signature.
- [x] Move ordinary server-owned chat user admission ahead of inference using the existing scoped conversation create/reuse preparation; keep locally owned chat admission in Dexie. Pass `tldw_history_admission_v1` through generic assistant `addChatMessage`, and suppress the existing post-generation duplicate user append. Add a character-free mounted fixture asserting exactly one user row before provider dispatch and one assistant row under its accepted ID afterward.
- [x] Pass the returned admission to assistant persistence and retain its parent independent of current navigation. Use identity/version retry guards; stale-parent result remains recoverable text in the original scope, with no automatic resend.
- [x] Retain uncertain admission, accepted-but-unsent input and generated-but-unsaved text under the original profile/owner/conversation and immutable operation identity. Save dispatch intent before the owner call. Recovery is client evidence, never owner authority or a selectable transcript node; reopening under a fresh view can inspect/copy it without retrying. Bookmark/confirmation updates cannot erase recovery, and an account change cannot expose another owner's data.
- [x] Run new mounted tests plus existing overlay, multi-image and scope regression suites. Confirm no production route relies solely on an unused extracted hook. Review/type-check and commit.

Task boundary refinement, 2026-09-17: ordinary and overlay sends use the client-managed composer and separate owner admission/settlement in Task3.2. Existing tracked-character sends are a separate server-managed implementation that does not use that composer. Task3.2 explicitly blocks their legacy timestamp path when H1 selection is active; the required Task3.3 below closes that temporary gate before Stage3 can complete. This is sequencing, not character parity credit or removal from H1.

Execution evidence: `0e71a732d4` implements ordinary/overlay selected-history preparation, owner admission, accepted settlement and separate recovery; `16e911c79a` fixes load receipts, activity ownership and omitted slash injection settings; `9213012401` closes native import-await origin gaps. The independent task review and two scoped fix reviews resolved R1/R2/R3 and the required R4 before-first regression gap, with no open or new fix-diff findings. Original affected coverage passed 216/216 tests in 22 files; fix round 1 passed 87/87 in 7 files; fix round 2 passed 40/40 in 4 files. These suites overlap, so the counts are not cumulative. Focused TypeScript, formatting and whitespace pass. Expanded typing retains two documented unchanged prompt-sync diagnostics. Mounted tests execute actual handlers/controller/navigation/pipeline/model/transport with mocked external boundaries; real browsers/IndexedDB remain Stage5. Tracked-character completion remains required Task3.3 below.

### Task 3.3: integrate tracked-character sends with versioned native completion

**Files:**

- Modify the actual tracked-character branches in `apps/packages/ui/src/hooks/chat/useChatActions.ts` and `hooks/useMessage.tsx`, with their mounted character/history-selection suites.
- Extend existing `services/tldw/TldwApiClient.ts` types and live `services/tldw/domains/chat-rag.ts` completion transport only for the existing versioned native completion body and separately scoped workspace query. Preserve the captured request lease and exact route guard; no broad proxy authorization.
- Reuse Task3.2's narrow recovery and view-fencing interfaces. Add a focused shared helper only if it replaces the same operation in both mounted handlers.
- Correct the verified native persistence-acknowledgement gap through `tldw_Server_API/app/api/v1/endpoints/chat.py`, `core/Chat/chat_service.py`, `streaming_pipeline.py` and `streaming_utils.py` only where needed: versioned admitted completion must return the saved result identity even when `CHAT_STREAM_INCLUDE_METADATA` is disabled. Preserve the legacy flag behavior for unversioned clients. Add focused endpoint/pipeline/stream tests for successful save, failed save and disabled metadata; the reviewed Task2.2 saved-behavior adapter remains the context authority.

**Interfaces:** Native tracked characters use the existing versioned server-owned `POST /api/v1/chat/completions` path: finalized selection plus new current inputs, `save_to_db=true`, verified conversation/workspace and frozen explicit model/provider/request values. The owner validates selected history and saved behavior and atomically admits the new inputs. Client-managed normal/overlay requests retain Task3.2's stateless inference path. Do not send a full client-composed historical transcript to server-owned admission, append the user twice, or invent a saved-prompt preview service.

- [x] Establish the verified originating native owner/capture and finalize the explicit request before dispatch. Use only new system/user/tool input supported by the native contract; do not fetch a mutable live character to substitute for saved behavior. Precisely gate unsupported persona, multi-character, steering or asset requirements before the operation rather than dropping them.
- [x] Integrate both actual mounted tracked-character send branches. Prove selected A1 excludes alternate A2, including equal text, and saved supported single-character behavior survives live source-card modification/deletion. Reuse the actual-factory native evidence and add missing mounted/transport coverage.
- [x] Consume returned admission/result identities from the actual stream and retain accepted/unknown outcomes in the originating scope. Native completion owns its append/settlement; do not add separate generic or legacy persistence. A disconnected or unparseable stream is not proof of non-commit and cannot trigger replay/fallback.
- [x] Make the versioned terminal persistence acknowledgement independent of the optional legacy metadata flag. Require the actual saved ID after successful settlement; failed save or cancellation cannot fabricate it, and provider-supplied admission/result fields cannot substitute for owner acknowledgements. Verify both the native endpoint wiring and streaming handler, including spoofed provider IDs and unchanged unversioned behavior with metadata disabled. Run touched Python checks and Bandit for this narrow correction.
- [x] Validate request/config lease before dispatch and preserve server ownership once dispatched. View changes do not redirect native persistence; conditional stream/display updates cannot overwrite a new view. Account/connection changes stop local consumption without claiming the server rolled back.
- [x] Verify workspace query, explicit provider/model inputs, zero legacy stream calls, exact native persistence ownership, first-send creation fencing and visible unsupported/recovery behavior in both surfaces. Run affected tests/static checks, commit, and pass independent review before Stage3 completion.

Execution evidence: native tracked-character integration `4822de5d71` and recovery-presentation fix `5d62386395` passed independent task review and scoped fix review. The sole P2 (native admission labeled as a response known not to be saved) is resolved with persistence-aware unknown-outcome wording, preserving admission/text and client-managed semantics. Evidence: 222 UI tests, final affected mounted 53 tests, and fix coverage 64 tests; these runs overlap. Native acknowledgement/endpoint tests passed 113 with one pre-existing heartbeat skip and 10 disclosed warnings. Focused types, Ruff, compile and production/test Bandit pass; two known expanded prompt-sync diagnostics remain. The endpoint tests use real SQLite, not new PostgreSQL coverage. Actual browser/IndexedDB and combined first-create/real-loader qualification remain Stage 5.

## Stage 4: isolated local forks and honest pending outcomes

**Goal:** Repair the current copy path and remove automatic owner-changing/duplicate fallback.

**Success Criteria:** Supported copies own their IDs/files, unsupported required state causes zero child writes, and unknown/partial server outcomes survive reload under the original owner.

**Tests:** H1-E/F/G, exact membership/order, cross-model parent normalization, source deletion/edit/file removal, deferred API failures and two-view dispatch claims.

**Status:** Complete (both tasks reviewed, including Task 4.2 fix1; real storage/surface qualification remains Stage 5).

### Task 4.1: one allowlisted local copy projector

Dependency closure required: the unused `types/history-selection.ts` fork result scaffold must be completed to spec section8 before live use (owner_key on all variants, child_id, committed message_map, distinct blocked state). Task4.1 owns this narrow type correction and actual handler assertions; Task4.2 consumes the completed contract. Comparison capture must use a coherent owner read and explicit model projection, retaining Task2.3's normal-send comparison gate.

Action boundary: existing H1 admission does not support same-parent regeneration or a read-only edit-and-resend boundary override. Preserve their explicit capability rejection before any copy/write/display truncation; do not duplicate capability checks or resubmit retained users to simulate support. Plain local scoped edits, leaf-only deletion guarded in the owner transaction, and safe forks remain required. Native selected edit/delete needs an owner-safe mutation adapter; current ambient metadata/version helpers and an unguarded native delete are gated before requests or display changes. Broader regenerate/edit-and-resend parity stays open. Task4.1 may gate the unsafe native index replay temporarily; Task4.2 must implement the supported limited native projection and outcomes before H1 qualification.

**Files:**

- Modify `apps/packages/ui/src/db/dexie/branch.ts`, `types.ts` and `helpers.ts` where exports/call signatures change.
- Modify the existing `db/dexie/history-selection.ts` seam narrowly for coherent owner-authorized fork capture and retained-policy digests. Preserve normal-send and mirror ownership gates; exclude source-only ingestion controls from fork identity while changed retained files/content still reject inside commit. No parallel owner service.
- Fix-review closure includes the existing `services/chat-settings.ts` reader/writer, `hooks/chat/useChatSettingsRecord.ts`, `routes/sidepanel-chat-resume.ts` and an optional HistoryInfo external-write guard. Local conversation keys use browser-local storage with non-destructive one-time legacy migration. Read failures/malformed settings are unavailable, required unsupported context blocks before writes, and overlapping writes cannot evade the owner-transaction fence. Preserve excluded-summary invariance and positive plain forks; do not create another settings authority.
- Modify `apps/packages/ui/src/db/dexie/chat.ts` where existing stable-ID edit/delete operations need conversation ownership enforcement for selected-history controls. Its existing import/history-delete transactions also preserve destination settings guards or reject pending-guard erasure before removing any rows; route actual delete helpers through the existing atomic operation. Reuse these seams instead of adding parallel mutation/import/recovery services.
- Modify `apps/packages/ui/src/hooks/handlers/messageHandlers.ts`, `hooks/chat/useChatActions.ts`, `hooks/chat/chat-action-utils.ts`, `hooks/useMessage.tsx` and `hooks/useMessageOption.tsx`.
- Modify `apps/packages/ui/src/components/Option/Playground/PlaygroundChat.tsx`, `PlaygroundCompareCluster.tsx` and `components/Sidepanel/Chat/body.tsx`.
- Create `apps/packages/ui/src/db/dexie/__tests__/branch-projection.test.ts`.
- Extend `apps/packages/ui/src/hooks/handlers/__tests__/messageHandlers.branch.test.ts` and `src/components/Option/Playground/__tests__/PlaygroundChat.per-model-routing.integration.test.tsx`.
- Extend `src/db/dexie/__tests__/message-target-by-id.test.ts` and the mounted history-selection suites for editing/deleting the selected variant with an unpersisted greeting and hidden alternatives present.

**Interfaces:** Consumes normal `HistorySelectionV1` or explicit `CompareHistorySelectionV1`. Produces `prepareLocalFork`, `commitLocalFork` and the typed branch request/result contract.

- [x] Add a failing copy fixture with source/server IDs, nontrivial parent order, a session file keyed by `sessionId`, source ingestion control fields and an unsupported protected asset. Verify exact ID membership/order rather than timestamp prefix. Missing/duplicate IDs must fail.
- [x] Implement explicit field constructors for conversation, messages, file records and allowed metadata. Preallocate the entire ID map, then remap every retained reference. For normal explicit graphs preserve selected edges; for legacy and comparison materialization build a linear child chain in approved order.
- [x] Add a two-round comparison case where the last rendered response belongs to B and common U2 parents to B1. A's fork contains U1/A1/U2/A2 and its own follow-up in semantic order; U2's new parent is A1's child ID. It neither copies nor follows B1.
- [x] Fix file ownership by writing `sessionId` for the child and fresh local file identities; remove source draft/job/batch/idempotency/control references. Do not invoke cancellation/deletion of source processing while building a copy.
- [x] Encode the spec allowlist and unsupported-required-state gate before transaction write. Character/default snapshots, unfenced external settings, remote protected references and unsupported rich assets must not silently become default/plain children. Source-only provenance remains inert; source sync enrollment is never child upload authority.
- [x] Validate source revisions inside the existing Dexie transaction and commit messages/history/files together. Remove both index-based prefix copying and the second snapshot-copy fallback. A definite abort has no child; uncertainty retains the operation result for Task 4.2.
- [x] Convert all live branch controls to stable boundary requests, including greeting offsets and comparison. Do not retain an index overload in a mutation path for convenience.
- [x] Preserve stable identity in adjacent H1 edit/delete/regenerate controls now that rendered paths differ from persisted row order. A UI position may select the visible target synchronously, but every owner mutation must carry its scoped stable ID; no fallback to index helpers or text equality. Edit-and-send/regenerate must capture an explicit path boundary and must not delete all later timestamp rows or unrelated alternatives. Reuse existing ID-addressed DB seams with real conversation checks and observable failures; temporary/unsaved rows must never be treated as persisted authority. Include source/child and hidden-alternative isolation tests.
- [x] Run projector and mounted branch/compare tests; type-check both consumers, review and commit.

Execution evidence: implementation `eb5e0683e5`, fix1 `1a5492c542` and fix2 `28748d402b`. Independent reviews resolved P1 child-controller adoption, mutation ownership and external-settings availability. Final affected fix2 checks passed 213 tests across 16 files and distinct focused consumer type checks; prior broader fix1 coverage passed 307 tests across 19 files. Counts overlap and are not additive. Actual WebUI/extension storage implementations are covered; real IndexedDB timing, file removal and product builds remain Stage 5. The Minor saved-copy/open-failure toast is assigned to Task 4.2. See the durable verification record for exact evidence and conservative pending-guard limitations.

### Task 4.2: pending operation store and no automatic fallback

Dependency closure: integrate the existing `chat-action-utils.ts`, `useHistorySelection.ts`, both mounted chat hooks and their action/recovery presentation. Preserve the exact child-load receipt from Task 4.1; a displayed child ID is not owner adoption. Include the relevant mounted, owner-adapter and outcome UI tests, and fix the reviewed misleading post-commit failure notification. Known native fork candidates also require a coordinated uncached owner-scoped settings path in the existing `useServerChatLoader`, `Playground` attachment restore/persist and `useChatSettingsRecord` seams. Extend existing getChat/settings transport options with requestScope/signal and the optional expected-user endpoint dependency where required; do not introduce another persistent cache or silently import browser state. Preserve explicit settings edits and reopen after those edits; plain-copy eligibility is not a permanent reading restriction.

Review closure must keep completed owner/candidate/settings qualification independent of ancestry readiness, including ordinary and known-candidate legacy conversations. Invalidated owner leases must clear presented outcomes and fence held reads/manual refresh without erasing durable records. Cover these through the actual controller and its settings/loader consumers; do not substitute mocked readiness for the reviewed failing paths.

Native eligibility requires a narrow addition to the existing same-statement snapshot/capture seam: the opaque storage-context digest alone cannot prove that a source has no required settings or assistant behavior. Task 4.2 may extend `core/DB_Management/chacha/message_store.py`, `core/Chat/history_selection.py`, `api/v1/schemas/history_selection_schemas.py`, the existing `chat.py` capture response and their real owner/API tests to return a minimal purpose-specific eligibility proof tied to that captured context. Update shared strict types/validation coherently. Do not add a parallel endpoint, settings service or raw context payload; an older server without affirmative proof is unsupported for this copy. Preserve existing send behavior and owner/sync gates. Approved proof: optional `snapshot.native_fork_context` with `policy: plain_v1`, the same `storage_context_digest`, and `supported: boolean`. Support only absent settings or an exactly empty JSON object, no behavior row, and all assistant identity fields SQL NULL. Settings/behavior row presence also enters the context digest; malformed/null/nonempty stored settings never certify plain context. Real SQLite/PostgreSQL and API tests cover the narrow policy, drift and older-server compatibility.

**Files:**

- Create `apps/packages/ui/src/db/dexie/fork-operations.ts` and `src/db/dexie/__tests__/fork-operations.test.ts`.
- Modify `apps/packages/ui/src/db/dexie/schema.ts`, `types.ts`, `hooks/handlers/messageHandlers.ts`, `services/chat-history-selection.ts` and `components/Common/Playground/HistorySelectionReview.tsx` for outcome display.
- Extend `apps/packages/ui/src/hooks/handlers/__tests__/messageHandlers.branch.test.ts` and the new operation-store tests.

**Interfaces:** Consumes immutable `ForkRequestV1` and owner namespace. Produces owner-scoped prepared/dispatching/unknown/partial/completed/rejected records, atomic dispatch claim and `ForkResultV1` without automatic replay.

Final consumer checking also found numeric `createChatBranch` callers in Playground's timeline action and ResearchWorkspace/ChatPane. Include their stable-message-ID adaptations and focused routing tests in this task, retaining comparison qualification and unsupported-mode gates. Missing identities must never fall back to rendered positions.

- [x] Add deferred server-create tests: reject before dispatch, lose creation response, acknowledge child then fail the second copied row, and abort while a response is pending. Assert calls to local copy and second server create remain zero after possible side effects.
- [x] Add operation tests for two views dispatching the same intent, reload of `dispatching`, scope/account switch and same ID with different digest. Retain candidate child ID and original request; new account sees none of the old owner's pending data.
- [x] Add `forkOperations` keyed by `[owner_key+operation_id]` with an indexed active intent identity and atomic prepared-to-dispatching claim. Freeze a single operation ID before asynchronous dispatch; attaching to an existing pending intent never dispatches it again. Temporary owner state remains memory-only.
- [x] Map errors by side-effect knowledge: pre-dispatch failure is rejected/blocked, uncertain creation is unknown, known partial child is partial. Mark old multi-request success `legacy_completed`; do not issue an atomic receipt claim.
- [x] Remove the catch fallback to local creation and the retry through a second local snapshot copier. Preserve the source view after failure; never navigate to partial child as successful completion or automatically clean it up.
- [x] Render pending state after reload with a known candidate link for inspection, no automatic retry, and a plain-language explanation of uncertainty and the consequences of another copy. Keep the H2 reconciliation boundary in technical documentation. A deliberate new user operation must be distinct from retrying the same unresolved operation. Do not reset the old namespace.
- [x] Run deferred handler/store tests and focused UI tests. Review/type-check and commit.

Execution evidence: Task 4.2 implementation `ec5f19816c`, locale completion `ac6919ae93` and fix1 `c884e6ba25` passed independent task review and scoped re-review. F1 invalid-owner outcome display and F2 legacy settings gating are addressed. Initial checks passed 628 client tests/35 files and 127 native tests with one intentional SQLite-only skip; the affected final fix passed 289 tests/16 files. Counts overlap. Focused core types pass; wider consumer checks retain the same 11 documented baseline diagnostics. Real IndexedDB migration/claims, successful native fork → send → reopen in each shell, builds and final branch review remain Stage 5. Minor notification failure-step wording remains for final review.

## Stage 5: real owner and surface qualification

**Goal:** Prove the implemented H1 behavior through both product shells and close only the H1 acceptance criteria.

**Success Criteria:** Required owner/API/UI evidence is recorded with no unresolved P1/P2 on H1. No skipped mode is counted as passing parity.

**Tests:** H1-A through H1-H, relevant baseline regressions, live IndexedDB rollback/reopen and real SQLite/PostgreSQL admission.

**Status:** In Progress (Task5.1 implementation and fix1 passed scoped review through `d6129ace01`; final whole-branch review and its deferred findings remain open).

Qualification checkpoint: Task5 fix1 passed independent scoped review for all four original findings and three confirmed evidence gaps. Final affected verification is174unit tests/11files and4/4 repeated full-tip cases in each shell, with no skips/retries/unexpected/flaky outcomes. The29named-case/31actual-result union per shell explicitly reuses earlier unchanged passing cases. Both builds complete; the full frontend typecheck retains90byte-identical initial-base diagnostics, and final lint reports0errors/148warnings versus0/147base. The finalized WebUI artifact still has two traced-copy ENOENT warnings from old test profiles; standalone deployment completeness is unverified despite passing static-token and bundle checks. Whole-branch triage must assess the inner positional message key, saved-copy notification issue and all prior observations. See the [verification record](Docs/Reviews/CHATBOOK_H1_HISTORY_SELECTION_VERIFICATION_2026_09_17.md) and [scoped review](Docs/Reviews/CHATBOOK_H1_TASK5_FIX1_REVIEW_2026_09_17.md).

### Task 5.1: browser flows, regression and release record

**Files:**

- Create `apps/tldw-frontend/e2e/workflows/chat-history-selection.spec.ts`.
- Create `apps/extension/tests/e2e/chat-history-selection.spec.ts`.
- Reconcile the stale header-to-WebUI expectation in `apps/extension/tests/e2e/sidepanel-options-handoff.spec.ts`: selected-history expansion opens the extension full page; retain the explicit composer draft-handoff and settings-sharing coverage. The existing composer action is labeled “Continue in WebUI” but also opens extension `options.html`; qualify that actual destination without claiming external server-WebUI transfer. Use existing isolated HTTP/IndexedDB test helpers or a narrow test-only helper, without a production testing backdoor.
- Update `Docs/Reviews/CHATBOOK_H1_HISTORY_SELECTION_VERIFICATION_2026_09_17.md`, created as an explicitly incomplete checkpoint during execution. Retain precise commands, review dispositions and capability limits; replace pending statuses only with qualifying evidence.
- Regenerate `apps/tldw-frontend/lib/api/openapi.fingerprint.json` using `apps/tldw-frontend/scripts/generate-api-types.mjs`; its `lib/api/generated/openapi.json` and `schema.d.ts` outputs remain ignored build artifacts.
- Update this plan, H1 spec implementation status, TASK-13261.1 and the parity inventory only where evidence supports a changed status.

**Interfaces:** Consumes all prior stage behavior. Produces recorded acceptance results and an independently reviewed H1 change.

- [x] Build browser fixtures with real IndexedDB transactions and the actual mounted UI. Use existing extension persistent-context/build helpers and WebUI config; mock provider transport deterministically, not the local DB commit. Separate real backend admission assertions from mocked transport fault injection.
- [x] Run each full-page shell: select different variants in two views, normal send, before-first/empty, legacy review with alternatives, reopen, fork supported local content, edit/delete child, remove copied file, and reload an unknown server fork. Include the compact sidepanel controls and expansion into the extension full page.
- [x] Exercise real IndexedDB abort between child writes and reopen after successful commit. Assert source rows, files and server/sync control IDs are untouched. Include the comparison child-chain scenario and >20,000-row virtualized legacy review. Verified local comparison handoff/reopen must retain readable rows and model-qualified fork controls even while normal selected-send capture is unsupported; preserve the owner/view fence and never grant authority from display rows. An unloaded feature preference must not disable or persist over the saved comparison state; qualify delayed preference hydration and genuinely disabled preferences through both active comparison hooks. For a verified restored local owner, shared-session comparison fields must not overwrite the per-conversation compareStates hydration; preserve ordinary, unowned and native restore semantics and stale-view fencing.
- [x] Bound rendering of the selected full-page transcript with the installed virtualizer so confirming the >20,000-row path through its last included message remains usable. Preserve the complete selected owner/provider data, stable identities, comparison blocks and existing search/timeline/edit navigation to offscreen rows; prove full-tip reopen/end-row access and independent before-first behavior. This closes the concrete browser OOM discovered during qualification.
- [x] Prevent legacy automatic server save from reuploading/retargeting H1-owned selected conversations during hydration or navigation; preserve genuine unowned draft saves and defer automatic mirror linking until matching native owner verification. Use a create-capable HTTP fixture, assert no unsolicited creates for local/native selection, and cover held old-save responses without retargeting.
- [x] Qualify known native fork child adoption/reopen with poisoned unscoped browser settings, held owner/record/settings responses, scoped explicit settings patches and a later nonplain child. Verify pending resolution cannot import browser state, profile/workspace changes cannot retarget outcomes, and actual extension transport accepts only the authorized scoped metadata/settings paths.
- [x] Reject native metadata scope/missing-record failures coherently across controller and display identity. A null display ID while the H1 owner remains native must not permit local/scratch settings writes. Cover the actual loader/settings failure, the hydration gap, stale responses after navigation and positive new-chat/reset behavior; retain durable known fork outcomes. Present settings rejection without a runtime crash, using the existing string error-label resource rather than its containing translation object.
- [x] Preserve plain native sidepanel sends by omitting synthetic context overrides only when composition is a verified no-op. Explicit or inherited optional assets, including unmatched assets, and material transformations remain explicit and unsupported by this native path. Cover real composition-hook boundaries, first native send/ACK/reopen and same-page held-create navigation.
- [x] Run the focused browser commands:

```bash
# cwd apps/tldw-frontend, using a dedicated H1 server URL/command
node_modules/.bin/playwright test e2e/workflows/chat-history-selection.spec.ts --project=chromium --workers=1
# cwd apps/extension, after building current source
node_modules/.bin/playwright test tests/e2e/chat-history-selection.spec.ts --project=chromium-extension --workers=1
```

- [x] Run the new native selection/API tests plus existing continuation, multi-image, system-message and provider-call-parameter suites. Use the repository virtual environment. Confirm actual PostgreSQL execution separately from SQLite; use the existing fixture's unavailable signal only, not custom silent skips.
- [x] Activate the root checkout's virtual environment, then run `node apps/tldw-frontend/scripts/generate-api-types.mjs` and `node apps/extension/scripts/verify-openapi-client-paths.mjs` from this worktree. Review the fingerprint and preserve ignored generated artifacts as build outputs. Run each app's installed `tsc` from its package cwd: frontend `NODE_OPTIONS=--max-old-space-size=8192 node_modules/.bin/tsc --noEmit`; extension `node_modules/.bin/tsc --noEmit -p tsconfig.compile.json`. Run relevant shared Vitest suites and project lint checks on touched files. These equivalent script commands avoid pnpm11 dependency reconciliation in the frozen Bun environment.
- [x] Run touched Python security validation:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Chat/history_selection.py tldw_Server_API/app/core/Chat/history_context.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/core/Chat/streaming_pipeline.py tldw_Server_API/app/core/Chat/streaming_utils.py tldw_Server_API/app/core/Chat/persistence_service.py tldw_Server_API/app/core/DB_Management/chacha/message_store.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/backends/sqlite_backend.py tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/app/api/v1/endpoints/character_messages.py tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py tldw_Server_API/app/api/v1/schemas/history_selection_schemas.py tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py -f json -o /tmp/bandit_chatbook_h1.json
```

If additional Python is touched, add it to the scope. Compare existing findings with the baseline; fix all new findings introduced by the change. Record code compile/lint checks and `git diff --check`.

- [ ] Perform independent review of the actual diff, especially admission dispatch races, multi-view legacy interpretations, copy allowlist/parent remapping, source isolation and unknown outcomes. Fix each P1/P2 and rereview affected behavior; do not close on a list of proposed fixes.
- [ ] Fill the verification record with commit/base, commands, pass/fail/skip counts, artifacts, supported modes and remaining H2/H3/H4/F02 work. All required H1 modes must pass before TASK-13261.1 is Done. Keep the overall parity inventory honest.
- [ ] Commit only this work and its tracking. A draft PR may summarize the concrete problem and validation when requested/authorized; merging still follows the repository's human-authored Change summary rule. Remove only this implementation plan after all stages are complete if following the repository cleanup convention, retaining its committed history and durable verification links.

## Acceptance-to-task map

| Spec gate | Implementation tasks | Required proof |
|---|---|---|
| H1-A selected identity/two views | 1.1, 2.2–2.3, 3.1–3.3, 5.1 | Actual selected provider input, stable fork boundary and conditional view advance. |
| H1-B boundaries/bookmarks | 1.1, 2.3, 3.1, 5.1 | Empty and before-first survive restore; deleted target stays explicit. |
| H1-C legacy complete/CAS | 2.1–2.3, 3.1, 5.1 | Complete owner manifest, dual immutable projections, replay and later descendants. |
| H1-D admitted parent/composer | 2.2–2.3, 3.2–3.3, 5.1 | Native and client-managed input/assistant/tool parent binding with real owner transactions. |
| H1-E local source isolation | 4.1, 5.1 | Reopened child edits/deletion/file removal leave source unchanged; unsupported state writes nothing. |
| H1-F unknown operation | 4.2, 5.1 | Lost/partial/reloaded outcomes keep owner/op/digest without a second application dispatch. Record browser/intermediary delivery retries separately; native owner deduplication remains H2. |
| H1-G comparison/surfaces/scopes | 1.1, 3.1–3.3, 4.1, 5.1 | Actual full-page/sidepanel paths and A/B comparison child chain. |
| H1-H capability/compatibility | 2.2–2.3, 3.2–3.3, 5.1 | Unsupported mode rejects before mutation; protected metadata/provider fields stay isolated. |

H2 starts after the H1 native contract is stable; H3 starts after the H1 browser contract is stable. They need not wait for each other's implementations. H4, F02, queue/voice and ACP retain their separate specifications and acceptance gates.
