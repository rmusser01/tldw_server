# Persona Workspace Parity Implementation Plan

> **For agentic workers:** Use executing-plans or subagent-driven-development to implement one reviewable stage at a time. Stage 1 implementation and local verification are complete, pending PR merge; later stages remain pending.

**Goal:** Bring Persona/Workspace defaults into reviewed behavioral parity with Chatbook dev and safely adopt them in personal Research Workspace conversations.

**Architecture:** Retain per-user Persona and Workspace ownership and the existing conversation identity model. Resolve startup defaults once, preserve explicit choices, and use MCP Hub for permission-profile authority. Research Workspace adopts the resulting contract after backend gates pass.

**Tech Stack:** Python/FastAPI/Pydantic, ChaChaNotes SQLite/PostgreSQL, MCP Unified/Hub, pytest; existing React/TypeScript chat hooks and Vitest for the later integration stage.

**Spec:** `Docs/Design/2026-09-13-persona-workspace-parity-assessment.md`; existing product contract `Docs/Product/Workspace_Persona_Defaults_PRD.md`.

**Tracking:** #2950; assessment TASK-13243.

| Stage | Backlog task | Dependency |
| --- | --- | --- |
| 1: Resolver hardening | TASK-13244 | None |
| 2: Choices and provenance | TASK-13245 | TASK-13244 |
| 3: Tool-profile parity | TASK-13246 | TASK-13245; focused #1922 contract |
| 4: Provisioning/backfill | TASK-13248 | TASK-13245 and TASK-13246 |
| 5: Research adoption/closeout | TASK-13247 | TASK-13248 and all preceding backend gates |

## Global Constraints

- Pin and recheck both dev SHAs at implementation start and closeout. Assessment baseline: server `c70387f496d82fcee92926bf3715bf5cd240ba88`; Chatbook `4cba44e6124a107cfb3210bf0985b84d514a96da`.
- Backend first. Stage 5 includes a separately reviewed frontend integration; no Buddy/animation or design-system backlog changes.
- No Persona prompt/name/avatar/policy snapshots in Workspace defaults. Voice/style stay null.
- Existing session identity > explicit new-session choice > Workspace default > system fallback. Explicit None is a choice, not missing input.
- Preserve server auth, ownership, scope, optimistic locking, approval, and revocation checks. Local Chatbook ids are not server ids.
- Any new SQL remains within DB_Management. Use existing Jobs for a user-visible bulk backfill; do not add a new scheduler.
- One implementation PR per reviewable unit; split a stage further when its contract and runtime changes need separate review. Stage 3 is gated by a focused #1922 contract, not permission to enable an unvalidated profile string.
- Every stage needs its Backlog task, regression evidence, touched-code Bandit results, and a scoped commit. Keep required human-written Change summary policy for implementation PRs.

## Stage 1: Effective Resolver Privacy And Diagnostics

**Goal:** Make the existing V1 effective-default API a reliable basis for parity.
**Success Criteria:** Hidden references are redacted; malformed defaults remain distinguishable from unset; disabled/inactive states are explicit; transient DB failures retain existing error mapping; the regression baseline passes.
**Tests:** Effective-state API matrix, log redaction, invalid persisted JSON, and historical migration coverage.
**Status:** Complete (implementation and local verification; [PR #2957](https://github.com/rmusser01/tldw_server/pull/2957) pending merge after planning PR #2952).

**Files:**
- Modify `tldw_Server_API/app/api/v1/endpoints/workspaces.py` (existing parse/resolve/projection helpers).
- Modify `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` (row normalization only if needed to preserve invalid-storage state).
- Test `tldw_Server_API/tests/Workspaces/test_workspace_assistant_defaults_api.py`.
- Test `tldw_Server_API/tests/ChaChaNotesDB/test_workspace_assistant_defaults_db.py`.

**Interfaces:** Keep `WorkspaceResponse.assistant_defaults` and `effective_assistant_default` wire shapes. Management-authorized stored defaults may retain their reference; effective `permission_denied` must have null kind/id/label/memory mode. Retain existing deleted-Persona management behavior unless a deliberate API change is tested and documented. Any DB-only corruption flag is computed on load and is neither persisted nor added to the public response.

- [x] Repair the baseline migration test first: replace the version-rewound modern DB setup with an actual v48 fixture or isolated v48-to-v49 migration unit case plus a valid historical full-upgrade fixture. Do not change production migration registry checks. Demonstrate that the test still fails if the assistant-default column migration is removed.
- [x] Add API tests in the existing fixture module. Start with a hidden/missing Persona id inserted via the DB helper; use `_install_workspace_overrides` and GET the Workspace. Assert only the effective view is redacted and that a legitimate settings owner still receives the stored reference.

```python
def test_effective_default_redacts_permission_denied(workspace_app, db):
    row = db.upsert_workspace("ws-private", "Private default")
    db.update_workspace(
        row["id"],
        {"assistant_defaults_json": _assistant_defaults_payload("hidden-persona")},
        row["version"],
    )
    _install_workspace_overrides(workspace_app, db)
    try:
        with TestClient(workspace_app) as client:
            response = client.get("/api/v1/workspaces/ws-private")
        assert response.status_code == 200
        assert response.json()["effective_assistant_default"] == {
            "status": "unavailable", "source": "workspace",
            "assistant_kind": None, "assistant_id": None, "label": None,
            "persona_memory_mode": None, "degraded_reason": "permission_denied",
        }
    finally:
        _clear_workspace_overrides(workspace_app)
```

- [x] Add log redaction coverage for `_parse_workspace_assistant_defaults` with a unique private marker in an invalid assistant_kind and an unknown key. Capture a Loguru warning sink and assert neither marker nor full payload appears. Exercise non-string dict keys as well: logging itself must not raise while sorting heterogeneous keys.
- [x] Add absent/null, invalid JSON string, non-object JSON, invalid object, inactive Persona, feature-disabled Persona, and DB-error cases. Use a dedicated test fixture for malformed storage; keep SQL in the existing DB test layer. Assert no profile lookup occurs when Persona support is disabled, and preserve mapped 5xx on DB failure.
- [x] Run the new regressions red. Apply the smallest changes: log only bounded error category/type and Workspace context; redact permission-denied effective identity; preserve a corruption indicator through DB normalization; consult the same Persona feature policy used by its existing endpoints. Do not import an API module into DB_Management.
- [x] Run both test files, `git diff --check`, and Bandit on the two touched production files. Record failures separately from skips. Commit as `fix: harden workspace persona default resolution` and link TASK-13244/#2950.

**Verification:** 159 tests passed across Workspace defaults DB/API, Workspace CRUD API, and chat conversation unit tests. Isolated v48-to-v49 migration plus the retained historical v4 schema full-upgrade path replace the invalid version rewind. Mutation checks fail when column creation or the corruption flag is removed. Bandit reports zero findings/errors for both production files and test scans (test assertions excluded). API-only and whole-stage independent reviews found no actionable issues. Expanded v59 coverage had 55 passes and three pre-existing source-catalog fixture failures, reproduced with the original DB normalizer restored; no migration guard changed. See the assessment's Stage 1 record for baselines and remaining validation limits.

## Stage 2: Durable Choices And Startup Provenance

**Goal:** Persist explicit None and the actual origin of a conversation's Persona before enabling provisioning or new consumers.
**Success Criteria:** Clear survives restart/backfill; saved chats are independent of default changes; old API callers retain behavior; default resolution and provenance cannot be forged by a caller.
**Tests:** Schema/DB/API round trips, create/retry/resume, stale Workspace version, identity selection precedence, and memory-mode behavior.
**Status:** In Progress. Contract design TASK-13245.1 is reviewed and approved; implementation starts with 2A only (TASK-13245.2), stacked on design PR #2958. Depends on Stage 1. Contract: [choice/provenance design](../../Design/2026-09-13-persona-workspace-choice-provenance-design.md).

### Slice 2A Execution

**Goal:** Persist and expose durable Workspace opt-out without adding startup provenance or provisioning.
**Success Criteria:** Registered v68 migrations preserve legacy storage; defaults and opt-out update atomically; read-only API state survives lifecycle operations; new clone/import destinations conservatively opt out.
**Tests:** SQLite/PostgreSQL fresh and genuine v67 upgrades, clear/set/omit/conflict/restart, clone/import, API read-only projection, cached-writer hazard and offline upgrade checks.
**Status:** In Progress. Baseline: 60 Workspace defaults DB/API tests passed. Fetched server dev still uses schema v67; implementation branch `codex/persona-workspace-explicit-none` starts at reviewed design `0f227ffa1f4e496e6b040425bf82de476c3ef14e`.

- [x] Write and observe failing persistence, migration, lifecycle, and API regressions.
- [x] Implement paired storage writes and current registered migrations; protect clone/import choices without adding Persona-sharing authority.
- [ ] Document offline upgrade/rollback limits, validate both supported backends, run touched-scope Bandit, and review independently.
- [x] Commit and open a draft implementation PR; keep 2B/2C/2D and the parent issue open. [PR #2959](https://github.com/rmusser01/tldw_server/pull/2959), stacked on design PR #2958; implementation commit `a664bfc9e2`.

**2A verification:** Broad Workspace CRUD/defaults/creation/import/clone and migration regression: 246 passed, 16 PostgreSQL skips, zero failures. A further 21 historical migration tests passed. After the final compatible-initializer race guard, the targeted storage/bootstrap/historical rerun passed 37 tests; the independent review rerun passed six failure/rollback/interleaving cases with no remaining findings. Review found and fixed a real SQLite implicit-commit gap: all legacy and shared schema helpers now finish before the final migration transaction. A compatible competing initializer is rechecked under the transaction without rerunning the backfill. This deterministic interleaving test is not a live multiprocess certification. PostgreSQL remains unreachable even outside the sandbox; live fresh/upgrade/rollback tests must pass before a PostgreSQL rollout. Bandit production/test scopes: zero findings/errors (test assertions excluded); compilation succeeds. Ruff is clean on other touched files and retains four unchanged endpoint BLE001 findings, reproduced at HEAD. [Maintenance runbook](../../Code_Documentation/Workspace_Persona_Defaults.md) documents cached-writer hazards and unsupported mixed-version/old-binary rollback.

**2A baseline refresh:** Server dev `beac8e9449b0e2fa90bdab89cf8cbf7905d2b915` and Chatbook dev `fbf374c9d32144d5dd23fd44c9adc0e77f00d58f` verified on 2026-09-13; scoped diffs from the assessment baselines show no changes to the relevant Workspace defaults implementation, tests, or Chatbook migration. This slice adds durable choice only, not full parity.

**Files:**
- Modify `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` and `tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py`.
- Modify `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`, `chat_session_schemas.py`, and `chat_conversation_schemas.py` in the same schemas directory.
- Modify `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py` and the Workspace defaults PATCH path.
- Reuse existing `tldw_Server_API/app/core/Workspaces/assistant_defaults.py`, whose startup resolver is already called by chat creation; share Stage 1 effective resolution without a parallel resolver.
- Extend existing Workspace defaults tests and `tldw_Server_API/tests/Chat/unit/test_chat_conversations_api.py`; add `tldw_Server_API/tests/Workspaces/test_workspace_assistant_startup.py`.

**Proposed wire contract to finalize before implementation:** dedicated `POST /api/v1/chats/workspace-startup`, not additional ignored fields on the legacy create route.

```json
{
  "scope_type": "workspace",
  "workspace_id": "workspace-id",
  "workspace_assistant_selection": "inherit",
  "workspace_assistant_default_version": 7
}
```

`workspace_assistant_selection` is required on the dedicated strict route with an extra-forbid request model. Legacy `/chats/` keeps today's behavior: omitted identity on new Workspace chats already inherits; explicit null in kind/id/character_id suppresses it; explicit identities win and parent/global requests bypass defaults. Old servers silently ignore unknown body fields, so strict clients must use the distinct route and must never downgrade rejected or timed-out requests to legacy creation. Strict requests require an idempotency key and Workspace scope, excluding parent/Character options and all supplied identity fields (including null). `inherit` requires a positive strict-integer Workspace version; `none` rejects a supplied version. Do not remove legacy null semantics. The proposed read-only `assistant_startup` field and strict-protocol retry boundary require approval before implementation.

- [ ] Add `assistant_defaults_explicit_none` to Workspace storage/read contract with current registered migrations for SQLite/Postgres. Clearing sets it, saving a default resets it, and omitted PATCH leaves it unchanged. Legacy null defaults migrate conservatively as explicit opt-out/unknown intent; explicit provisioning can override only after owner confirmation. Do not retrofit schema v49 or skip current registry requirements.
- [ ] Require a drained offline maintenance upgrade for incompatible storage changes: stop old API/background/direct DB writers before migration, then restart only compatible binaries. Cached handles are not fenced by the existing initialization-time version guard. Rehearse the cached-writer hazard and upgrade procedure on SQLite/PostgreSQL; do not claim rolling-upgrade or old-binary rollback support.
- [ ] Persist startup source and originating Workspace version with conversation identity atomically. Use a dedicated bounded provenance field/object, not generic source/external_ref. Legacy chats retain unknown source. Response/list/resume preserve it; forks record validated lineage without claiming today's default. Metadata-only sync updates preserve local provenance and identity changes invalidate it; untrusted imports cannot forge origin. Workspace Sync and cross-server verified provenance transport are not enabled here.
- [ ] Add strict versioned selection through the existing authenticated resolver while retaining legacy inheritance. Validate Workspace version inside the creation transaction; conflict before inserting when stale. Workspace startup currently bypasses Sync v2, so add the proposed owner/scope-bound DB receipt for strict-protocol calls. Exact accepted replay retains original identity while its binding is unchanged; changed requests or subsequent binding/scope edits conflict, never undo legitimate edits. Recheck receipts after blocking PostgreSQL locks before evaluating current defaults. Do not advertise retry safety for unchanged legacy callers or reuse expiring asynchronous Workspace operation receipts.
- [ ] Bound permanent receipt count across each owner's Workspaces, including deleted/invalidated tombstones. The design proposes a configurable 10,000-receipt default; capacity blocks unseen keys, never accepted replay, and deletion cannot recycle keys. Check and insert under owner-scoped transaction serialization; test concurrent admissions in different Workspaces and create/delete loops.
- [ ] Deliver sync/direct mutation and import-authority safeguards with 2B provenance, and receipt invalidation/deletion/capacity plus send-time admission with the 2C route. Keep behavior inactive if its introducing slice is split across PRs. 2D verifies these safeguards; it must not be their first implementation.
- [ ] Characterize prompt precedence using `tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py` and `test_persona_prompt_assembly.py` under `tldw_Server_API/tests/Chat/`. Preserve Persona boundaries, exemplars, memory rules and explicit prompt behavior. Check inactive/deleted profiles before send; a saved unusable Persona must not silently fall back. Compare Chatbook custom-prompt bypass and supported profile fields and record an explicit disposition.
- [ ] Repair the stale Persona integration credential fixture before claiming prompt/memory regression coverage. The design baseline had 13 passes and 9 missing-credential 503 failures; supplying its existing dummy credential through the current snapshot loader in memory yielded 22 passes. See the design validation record; this diagnostic is not an unmodified-suite pass.
- [ ] Regression table: unset/new, explicit Persona, explicit Character, explicit None, inherit/read-only, confirmed read-write, old caller, resumed chat after default edit, revoked Persona, stale Workspace version, concurrent clear, retry after accepted create, fork, and wrong-owner/scope. Inherit read-write only from a currently confirmed saved default; client-provided provenance is not authority.
- [ ] Verify migrations using the repo's database fixtures, run relevant chat/default API suites and Bandit, and commit the storage and API work in separately testable increments. Update the assessment matrix before moving on.

## Stage 3: MCP Hub Permission-Profile Parity

**Goal:** Reproduce Chatbook's Workspace policy binding using server-owned MCP Hub authorization.
**Success Criteria:** A visible profile can be bound only by an authorized user; profile changes/deletion apply to later calls; bindings cannot widen effective permissions or bypass Persona confirmations.
**Tests:** Profile reference validation, narrowing, revocation, policy precedence, stale approval, and owner/scope isolation.
**Status:** Not Started. Focused contract under #1922 is a prerequisite to runtime activation.

**Files and ownership:**
- Contract `Docs/Product/Persona_Tool_Administration_PRD.md` and a focused design under `Docs/Design/` for the #1922 Workspace-binding slice.
- Reuse `tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py` profile CRUD/visibility helpers and its existing service/DB policy evaluator; do not create a second permission store.
- Narrowing remains in `tldw_Server_API/app/core/Persona/policy_evaluator.py` and MCP Unified admission/execute paths.
- Workspace schema/resolver changes belong in the Stage 2 files; add tests under `tldw_Server_API/tests/Workspaces/` and the existing MCP Hub test suite.

- [ ] Resolve profile identity/lifecycle first: Chatbook local ws-* strings versus server integer profile ids, owner scope, assign authority versus use authority, deleted/disabled profile behavior, revision checks, and mapping for a server-connected Chatbook. Define a server-issued reference; never coerce a local ws-* string into a server id.
- [ ] Specify policy order from the current server evaluator and prove it with cases: server deny + profile allow stays denied; Persona deny stays denied; Persona ask survives remembered profile allow; call caps remain enforced; Workspace move re-evaluates Workspace policy without rebinding Persona identity; pending approval is revalidated after policy revision.
- [ ] Add a read-only effective policy preview that uses the same resolver as enforcement, with hidden tools/profile names redacted. Preview is not grant authority.
- [ ] Activate non-null tool_policy_profile_id only once both reference validation and every affected execution path enforce the binding. Until then writes remain rejected. Stage 3 cannot be marked complete for a schema-only unlock or display-only preview.
- [ ] Run focused authorization and runtime tests, Bandit, and profile deletion/revocation regressions. Link the resulting focused PRs to #1922 and #2950. A separate design review is part of this stage's work, not an untracked deferral.

## Stage 4: Idempotent Persona Provisioning And Backfill

**Goal:** Give explicit Workspaces a reusable user-owned default Persona/profile pair, matching Chatbook's convenience behavior without duplicates or resurrected opt-outs.
**Success Criteria:** Concurrent creation/retry produces one pair; clear wins over delayed provisioning; legacy migration respects prior intent; partially created resources are recoverable; existing chats remain unchanged.
**Tests:** Create, repeat, concurrency, crash between resources, retry, backfill preview/commit, clear race, archive/delete/rename, feature-disabled and policy-denied provisioning.
**Status:** Not Started. Depends on Stages 2 and 3.

**Files:**
- Add `tldw_Server_API/app/core/Workspaces/assistant_provisioning.py` for orchestration.
- Reuse `tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py` profile creation and MCP Hub profile service; transaction/claim/receipt storage belongs in DB_Management.
- Integrate Workspace create paths in `tldw_Server_API/app/api/v1/endpoints/workspaces.py` and bulk work through existing Jobs APIs/WorkerSDK.
- Add `tldw_Server_API/tests/Workspaces/test_workspace_assistant_provisioning.py` and DB concurrency cases.

- [ ] Define a per-owner/workspace provisioning receipt keyed by the Workspace creation identity, not its display name. Record each resource id before progressing so retry can reuse it. Use existing transactions/unique constraints/CAS, never an in-memory mutex as the multi-worker guarantee.
- [ ] Create a normal Persona with a bounded Workspace-derived name and seed prompt; reference it from defaults. Do not invent a source Character or rename the Persona when the Workspace is renamed. Create the MCP Hub profile through normal ownership/grant checks and keep its initial effective permissions within the already-authorized baseline.
- [ ] Recheck expected Workspace version/opt-out state before binding. If clear/rebind/archive/delete won the race, do not overwrite it. Record orphan resources for cleanup/reuse without deleting a Persona/profile that now has another reference.
- [ ] Add an owner-reviewed backfill preview listing eligible Workspaces and proposed references, then an idempotent Jobs commit path with status/cancel/retry. Legacy null records are not automatic consent. Already-bound and explicit-None records remain untouched unless the owner explicitly requests provisioning for that record.
- [ ] Enable auto-provision only for new explicit owner-controlled Workspaces after the contract is reviewed. Global/fallback contexts and shared-recipient Workspaces stay excluded. Feature-disabled/unavailable provisioning keeps Workspace creation successful but records a visible retryable status; never falsely report an available default.
- [ ] Run DB/provider-mocked concurrency and failure tests, Jobs ownership/idempotency tests, and Bandit. Cross-client tests compare logical outcomes, not local/server id equality.

## Stage 5: Research Workspace Adoption And Parity Closeout

**Goal:** Apply the completed backend contract to new personal Research Workspace conversations in both normal and RAG modes.
**Success Criteria:** First send persists correct identity/provenance before generation, resume preserves it, explicit None works, source restrictions remain enforced, and all parity rows have tested or agreed dispositions.
**Tests:** New/resumed chat, explicit override/None, stale scope, retry, no-source and selected-source normal/RAG behavior, unavailable Persona, memory mode, and cross-client contract cases.
**Status:** Not Started. Depends on the backend gates above; frontend work is a separate reviewed slice.

**Files:**
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/ChatPane/index.tsx`.
- Reuse `apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx` semantics and `apps/packages/ui/src/hooks/useMessageOption.tsx`.
- Modify shared startup in `apps/packages/ui/src/hooks/chat/useChatActions.ts` and `personaServerChat.ts` in that directory; verify `apps/packages/ui/src/hooks/chat-modes/ragMode.ts` and its shared pipeline pass the bound conversation through generation.
- Extend ResearchWorkspace/ChatWorkspace component tests, `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx`, and backend Persona conversation integration tests.

- [ ] Wire effective defaults only after Workspace hydration is ready. Preserve explicit choice and persisted identity; send the Stage 2 opt-in resolution/version request rather than presenting a stale inherited label as authority.
- [ ] Ensure the new RAG conversation uses the same Persona startup contract before retrieval/generation. Verify every RAG generation route, including any server-generated answer shortcut, uses the bound Persona context or explicitly disables the unsupported combination with an actionable state. Do not silently generate under a plain assistant while displaying a Persona.
- [ ] Preserve strict selected-source behavior: no evidence/retrieval failure must not fall back to unrestricted general chat. Persona scopes may narrow source access but cannot broaden Workspace/user access.
- [ ] Render inherited/explicit/None/unavailable status from persisted provenance, not equality with today's Workspace default. Test leaving and returning to a Workspace while settings or a send request is in flight.
- [ ] Confirm ordinary global chat, Chat Workspace, temporary chats, and explicit Character selection retain behavior. Shared-recipient Research chat stays governed by #2737 and is not enabled by this stage.
- [ ] Run focused Vitest suites and backend tests, then browser UAT for the affected surface. Record fresh Chatbook/server SHAs and compare the same cases on both. Update the PRD and #2950 with actual evidence; keep any unresolved parity row open, including Stage 3/4 dependencies.

## Verification And Review Checklist

- [x] Before Stage 1, reproduce the recorded 32-pass/1-fail baseline and repair the historical migration fixture without suppressing its assertion.
- [ ] Unit/API tests for a stage go red for the intended behavior before production changes and green afterward. Mock LLM/network work; use existing SQLite/PostgreSQL fixture infrastructure.
- [ ] Preserve production-path test coverage: ordinary chat memory tests are not evidence that Research Workspace RAG memory already works.
- [ ] Each runtime PR runs touched-scope Bandit and focused regressions, with exact results attached to its Backlog task. No production-code scan is needed for this documentation-only planning PR.
- [ ] Verify tests for malformed payloads assert private values are absent from logs and effective responses.
- [ ] Review idempotency across the entire multi-resource operation; a stable job id or a single once-per-database flag alone does not prove provisioning is retry safe.
- [ ] Refresh the parity matrix and link PR evidence. Stage 5 cannot close gaps merely because they have future trackers.

## Assessment Self-Review

Reviewed against #2950 and the existing PRD: reference ownership, explicit None, precedence, read-write confirmation, prompt/RAG semantics, provisioning, profile ids/authority, sharing boundaries, and latest-dev refresh are assigned above. Review corrections incorporated: explicit None must precede provisioning; legacy null cannot prove opt-in; the RAG branch must preserve identity; corruption must survive DB normalization as a diagnostic; retry must preserve accepted resolution; permission binding must not be confused with permission grants. Detailed policy and multi-resource provisioning contracts are stage deliverables, not pre-approved runtime changes.
