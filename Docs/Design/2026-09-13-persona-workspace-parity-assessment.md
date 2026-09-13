# Persona Workspace Defaults Parity Assessment

Status: Assessment complete; Stage 1 implemented and locally verified, pending merge. Stages 2-5 remain pending. Proposed later-stage decisions below require review as part of the implementation plan.

Tracking: [#2950](https://github.com/rmusser01/tldw_server/issues/2950), TASK-13243. Predecessor [#1911](https://github.com/rmusser01/tldw_server/issues/1911) remains closed for completed V1 scope.

Contract: [Workspace Assistant Defaults PRD](../Product/Workspace_Persona_Defaults_PRD.md).
Execution: [staged implementation plan](../superpowers/plans/2026-09-13-persona-workspace-parity-implementation-plan.md).

## Baselines And Method

| Repository | Branch | Inspected commit |
| --- | --- | --- |
| rmusser01/tldw_server | dev | `c70387f496d82fcee92926bf3715bf5cd240ba88` |
| rmusser01/tldw_chatbook | dev | `4cba44e6124a107cfb3210bf0985b84d514a96da` |

Both remote baselines were checked on 2026-09-13. Server code was read from an isolated worktree; Chatbook code was read with `git show origin/dev:<path>` against the verified commit, not its potentially modified checkout. Refresh both baselines at each implementation PR and at parity closeout. A changed SHA requires a scoped diff review, not silently reusing this matrix as current evidence.

This assessment covers Persona identity, Workspace defaults, startup/persistence, provisioning, and their permission dependencies. It is not an audit of every Persona, Buddy, avatar, scheduling, or multi-agent feature. Parity means equivalent agreed behavior with documented authority differences; it does not mean copying a local implementation into a multi-user server.

## Outcome

The shared V1 substrate is present: reference-backed defaults, read-only/read-write metadata, explicit save confirmation, effective-default responses, and creation-time application in Chat Workspace/Console. The server also already applies defaults on new Workspace chat creation when identity is omitted and honors explicit-null opt-out (see the Stage 2 correction below). Full parity is not established. Chatbook adds automatic provisioning, a durable Workspace explicit-None flag, and named Workspace permission profiles. Research Workspace needs end-to-end adoption verification on the server; Chatbook's implemented design explicitly excludes that surface too.

The first recommended backend work is to harden effective-default resolution and its regression baseline. Opt-out/provenance storage follows. Permission-profile binding must be designed against MCP Hub before provisioning can reproduce Chatbook's Persona-plus-profile behavior. Research Workspace integration follows those backend contracts.

## Evidence Index

Paths below are relative to the indicated repository and refer to the pinned commits above.

| ID | Repository | Path / symbols |
| --- | --- | --- |
| S1 | server | `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`: `WorkspaceAssistantDefaults`, `WorkspaceEffectiveAssistantDefault`, `WorkspacePatchRequest` |
| S2 | server | `tldw_Server_API/app/api/v1/endpoints/workspaces.py`: `_parse_workspace_assistant_defaults`, `_effective_workspace_assistant_default`, `_get_workspace_persona_profile`, `_ws_to_response` |
| S3 | server | `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`: `_load_workspace_assistant_defaults_json`, `_workspace_row_to_dict`, `update_workspace` |
| S4 | server | `tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py`: `_normalize_conversation_assistant_identity`, conversation create/update methods |
| S5 | server | `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`: session creation, ownership/scope validation, conversation metadata; `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py` |
| S6 | server | `tldw_Server_API/app/core/Chat/chat_service.py`: `_build_persona_chat_projection`, `_resolve_assistant_context_for_chat`; `tldw_Server_API/app/core/Persona/memory_integration.py` |
| S7 | server | `apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx`; `apps/packages/ui/src/hooks/useMessageOption.tsx`; `apps/packages/ui/src/hooks/chat/personaServerChat.ts` |
| S8 | server | `apps/packages/ui/src/components/Option/ResearchWorkspace/ChatPane/index.tsx`; `apps/packages/ui/src/hooks/chat/useChatActions.ts`: `ensureWorkspaceServerChatForTurn`, RAG/normal send branches; `apps/packages/ui/src/hooks/chat-modes/ragMode.ts` |
| S9 | server | `tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py`: visible permission-profile lookup, profile CRUD and policy assignment; `tldw_Server_API/app/core/Persona/policy_evaluator.py` |
| S10 | server | `tldw_Server_API/app/core/Workspaces/assistant_defaults.py`: `resolve_new_conversation_assistant`; `tldw_Server_API/tests/Workspaces/test_workspace_assistant_creation.py`; S5 `_active_chat_sync_service` |
| C1 | Chatbook | `tldw_chatbook/Workspaces/models.py`; `tldw_chatbook/Workspaces/assistant_defaults.py` |
| C2 | Chatbook | `tldw_chatbook/Workspaces/registry_service.py`: defaults setters/clearers; `tldw_chatbook/DB/migrations/workspaces_v7_to_v8_explicit_none.sql` |
| C3 | Chatbook | `tldw_chatbook/Workspaces/agent_provisioning.py`: `WorkspaceAgentProvisioner`, `run_workspace_agent_backfill` |
| C4 | Chatbook | `tldw_chatbook/Chat/console_assistant_defaults.py`: `resolve_new_console_assistant`, `build_persona_agent_system_prompt`; `tldw_chatbook/Chat/console_chat_store.py` creation boundary |
| C5 | Chatbook | `tldw_chatbook/Chat/console_persona_assignment.py`: target/revision/idle guards and memory confirmation. Domain behavior only; its Buddy UI is excluded. |
| C6 | Chatbook | `Docs/superpowers/specs/2026-08-29-workspace-assistant-defaults-design.md`; `Tests/Workspaces/test_workspace_assistant_defaults.py` |

## Parity Matrix

"Aligned" below means matching inspected contracts, not newly certified runtime parity. These rows describe the original pinned baseline; the Stage 1 implementation record below supersedes its resolver findings. Stage numbers refer to the linked plan.

| Area | Server evidence | Chatbook evidence | Assessment and disposition |
| --- | --- | --- | --- |
| Reference-backed Persona defaults | S1/S3 store kind/id/memory mode; voice/style/tool profile null-only | C1/C2 store references; voice/style null-only; tool profile may be a string | Core identity aligned. Tool-profile difference belongs to Stage 3. Do not introduce profile snapshots. |
| Save confirmation | S1 requires literal true for read-write and consumes confirmation before DB update | C2 and C6 reject unconfirmed read-write saves | Broadly aligned. Preserve server-side enforcement; test changing Persona while memory writes are enabled (Stages 2/4). |
| Effective response and privacy | S2 can return unavailable + permission_denied with assistant id/kind/memory mode still populated | C1 returns null identity fields for unavailable lookup results | Confirmed effective-projection gap, Stage 1. Stored settings are a separate authorized management view. No cross-user exploit is claimed from static inspection of per-user DBs. |
| Corrupt storage and logging | S3 converts invalid JSON to None; S2 cannot then distinguish corruption from absence. S2 logs `str(exc)` and arbitrary input keys for invalid objects | C6 tests that malformed values are absent from warning logs; loader similarly clears bad data | Server diagnostics/privacy gap, Stage 1. Do not copy the loss of corruption state. Both sides need contract tests for absence versus invalid data. |
| Feature-disabled and inactive behavior | S2 rejects inactive saves but does not consult the Persona enabled switch during effective resolution; S6 only rejects missing profiles in the inspected resolver | C4 checks missing/archived Workspace; C5 rejects inactive assignment, whereas C1 does not check is_active | Validation is not uniform in either project. Stage 1 effective-state checks; Stage 2 ordinary send checks. Use server capability policy, not Chatbook's weaker branch as the desired behavior. |
| New-chat default selection | S7 supplies hook defaults; S5 already invokes S10 for omitted Workspace identity, with explicit-null and parent/global bypass | C4 resolves at Console creation when no explicit identity/custom settings apply | Creation-time inheritance already exists on both. Stage 2 preserves legacy behavior and adds a strict versioned/retry-safe protocol; do not create a second resolver. |
| Explicit None | S1/S3 persist null but have no distinct assistant-default opt-out bit | C2 persists assistant_defaults_explicit_none; C3 excludes it from backfill | Confirmed storage gap. Stage 2 must land before auto-provision/backfill. A chat-level None is distinct from clearing a Workspace setting. |
| Conversation persistence/resume | S4/S5 retain kind/id/mode; S7 derives inherited provenance partly by comparing the current default | C4 persists chosen identity; C5 guards exact assignment targets | Identity substrate exists; durable inherited-versus-explicit provenance needs a server contract. Matching ids is not proof of origin. Stage 2. |
| Prompt composition | S6 starts from Persona name/system_prompt and has separate exemplar/memory integration | C4 uses name/system_prompt/personality/description through Character-card text composition | Non-identical semantics. Stage 2 must test shared supported fields and document canonical precedence. Preserve richer server behavior; do not copy a rendered prompt into Workspace defaults. |
| Normal/RAG execution | S8 takes RAG branch before tracked-Persona normal send; its create payload omits identity, but S5/S10 can resolve it server-side | C6 explicitly defers Research Workspace adoption | A missing request identity is not proof of missing stored identity. End-to-end generation/memory/provenance display remain unverified; Stage 5 gates both modes and source restrictions. |
| Automatic provisioning/backfill | No equivalent in inspected Workspace create/default paths | C3 creates a Persona and ws-* profile; backfill retries and skips explicit None | Confirmed feature gap, Stage 4 after Stages 2/3. Server must be safe under multiple workers; local best-effort creation alone is insufficient. |
| Workspace tool policy | S1 rejects non-null profile; S9 already provides MCP Hub profile control plane and Persona narrowing | C1/C3/C6 use named local permission profiles with Persona narrowing | Stage 3 with #1922. Local ws-* strings are not server MCP Hub integer profile ids. Binding a profile must not confer grant authority. |
| Target/concurrency safety | S3 uses optimistic versions for Workspace mutations; chat adoption depends on hook state and scope | C5 rechecks exact target and active work before publication | Preserve server optimistic locking. Add stale-scope/default race tests; do not add active-chat reassignment as an incidental startup feature. |
| Authority and sharing | Server owner/auth checks, per-user data, Workspace scope and sharing | C6 local-only V1, explicit server-authority/sync deferral | Intentional architecture difference, not grounds to omit authorization tests. Server references never resolve against a client's local Persona namespace. Shared-recipient chat remains #2737. |

## Proposed Contract Decisions

1. Keep Workspace defaults optional. Represent unset versus explicitly cleared separately before provisioning. Legacy null rows have unknown intent: conservatively preserve them; use a previewed owner-authorized backfill to opt eligible legacy Workspaces in. Do not infer consent from null.
2. Keep existing conversations independent of default edits. Add creation provenance as bounded metadata (selection source, originating Workspace id/version); persist it with assistant identity in the same transaction. Do not reuse generic `source`/`external_ref`, which already describe unrelated conversation provenance. Existing rows get unknown origin, not guessed inheritance.
3. Preserve existing Workspace-only implicit inheritance and explicit-null opt-out for old `/chats` callers. New adoption clients can explicitly request a versioned, retry-safe Workspace resolution or explicit None. Resolution uses the authenticated owner and expected Workspace version. A stale default conflicts before creation; accepted retries reuse the original identity. Workspace chat currently bypasses Sync v2, so its replay guarantee requires a DB-owned transaction, not reliance on global Sync receipts.
4. Preserve custom system-prompt intent without stripping required Persona safety/policy/exemplar sections. Chatbook currently bypasses default inheritance when a custom prompt is supplied; server canonical precedence needs explicit characterization before extending adoption. Preserve existing explicit Persona behavior.
5. Persona id/memory mode are conversation-scoped; Workspace tool policy is evaluated for the current Workspace at tool-call admission. A profile reference never grants more than AuthNZ, deployment policy, MCP Hub, Persona rules, and approvals permit. Existing conversations can encounter new/revoked policy on a later call without changing their Persona identity.
6. Create user-owned default Personas only through the normal Persona service/DB ownership boundary, without requiring a fabricated Character card. No account-wide background rollout until opt-out, idempotency, and profile authority are implemented. Workspace deletion/rename does not delete/rename a reusable Persona without a separate explicit user action.
7. Keep unavailable new defaults nonblocking for opening a Workspace and make the fallback visible. An existing saved Persona that becomes unusable must not silently change identity. A temporary DB failure stays a retryable service failure, not a successful `none` result that loses user intent.

These proposals preserve the accepted V1 no-snapshot contract. They do not establish a cross-app Persona sync format or claim that local profile ids can be used on the server.

## Verification Evidence

Fresh baseline command from the isolated server worktree, using the shared project virtualenv:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_workspace_assistant_defaults_db.py tldw_Server_API/tests/Workspaces/test_workspace_assistant_defaults_api.py tldw_Server_API/tests/Chat/unit/test_chat_conversations_api.py -q --tb=short --disable-warnings
```

Result: **32 passed, 1 failed, 6 warnings**. The failure is `test_v48_sqlite_migration_adds_workspace_assistant_defaults_column`: it constructs a current DB, rewinds the schema version to 48, then hits `Notes attachment v59 registry collision requires explicit repair`. This is a pre-existing test-fixture failure on the unmodified pinned baseline, not proof of a broken production v48 migration. Stage 1 repairs the test using a historical fixture or an isolated migration test plus a genuine old-schema upgrade case; never bypass the registry repair guard.

Existing server tests also cover Persona prompt assembly and read-only/read-write memory behavior in `tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py`; those tests were inspected but not executed in this pass. Chatbook tests were inspected, not run. No browser UAT or fresh cross-client certification was performed. Documentation-only changes require no production Bandit scan; code-stage scans remain mandatory.

## Review And Exit Criteria

- Missing functionality is assigned to Stages 1-5; no stage treats a PRD marked implemented as runtime proof.
- V1 stays completed; expanded parity remains open in #2950, with permission-profile work explicitly dependent on #1922.
- First implementation PR: Stage 1 only. It can be reviewed independently and does not need provisioning or UI work.
- Later policy/provisioning stages must produce their focused design decisions before enabling behavior. A tracker or plan alone never closes their parity rows.
- Close #2950 only with fresh baseline SHAs, passing behavioral evidence, and user-agreed disposition of any remaining cross-project difference.

## Stage 1 Implementation Record

TASK-13244 implements resolver hardening in [PR #2957](https://github.com/rmusser01/tldw_server/pull/2957) on `codex/persona-workspace-resolver-hardening`, stacked on planning PR #2952. Server dev was refreshed to `e157b6d1306a133e93595a8d457ecac76ac770fa`; Chatbook dev to `392ce191fd28953550f85154ea1f8e4eda4ab7f3`. Both were rechecked at local closeout on 2026-09-13. Scoped diffs since the assessment changed neither the server Stage 1 files nor the inspected Chatbook Workspace/default contracts.

- Effective `permission_denied` and `persona_feature_disabled` responses redact identity; the settings-owner view retains references. Disabled reads skip profile lookup, non-null saves return 503, and clearing still works. Deleted/inactive distinctions and mapped lookup failures remain intact.
- Raw storage that cannot decode as an object produces a computed private corruption flag, consumed by the API as `invalid_default`. SQL NULL remains unset; no new persistent/public field or Persona snapshot was introduced. Validation logs contain only fixed categories, known types, and Workspace-id presence.
- The obsolete migration fixture was replaced by isolated v48-to-v49 coverage plus a real upgrade from the retained historical v4 schema. Disabling column creation fails the migration test; removing the corruption flag fails malformed-storage projection cases. Production migration guards were not changed.

Final focused command, using the shared activated virtualenv:

```bash
python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_assistant_defaults_api.py tldw_Server_API/tests/ChaChaNotesDB/test_workspace_assistant_defaults_db.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py tldw_Server_API/tests/Chat/unit/test_chat_conversations_api.py -q --tb=short --disable-warnings
```

Result: **159 passed, 6 warnings**, no skips. Bandit on both production files returned zero findings/errors; the test scan also returned zero with expected test assertions excluded. Ruff was clean for the DB module and tests; the endpoint retains the same four pre-existing BLE001 warnings outside changed code. Both test files and changed production sections satisfy Black; full-file DB formatting still reports unrelated existing drift. API-only and whole-stage independent reviews found no actionable issues.

An additional v59 migration suite run had **55 passes and three failures**: `test_sqlite_v59_initializer_serializes_on_one_schema_authority`, `test_sqlite_v58_to_v59_creates_empty_canonical_registry`, and `test_sqlite_fresh_and_v58_upgrade_registry_schema_are_identical`. All three reproduced with the original HEAD DB normalizer restored in memory, failing with `Notes task v59 SQLite source catalog drifted`; their version-rewound fixtures are outside this Persona change. They were not suppressed or silently counted as passing. Live PostgreSQL, Chatbook runtime tests, browser UAT, and the repository-wide suite were not run. This stage establishes the tested resolver contract, not full cross-client parity or Research Workspace adoption.

## Stage 2 Baseline Correction

TASK-13245.1 found that `resolve_new_conversation_assistant` and its endpoint integration/tests were present since commit `77e2f3765b22ce3166187b564a7bbcb88ba2880b`, before the original assessment. They were missed in that pass, not newly introduced by Stage 1. Fresh `test_workspace_assistant_creation.py` execution passed **9 tests** including HTTP persistence of inherited and explicit-null choices. The matrix above corrects the startup and RAG claims accordingly. Stage 1's effective-state hardening and its recorded tests remain valid; startup reuse of that resolver still belongs to Stage 2.

Server dev rechecked at `beac8e9449b0e2fa90bdab89cf8cbf7905d2b915`, Chatbook dev at `392ce191fd28953550f85154ea1f8e4eda4ab7f3`; scoped startup/schema/DB/Sync files are unchanged. The [Stage 2 contract proposal](2026-09-13-persona-workspace-choice-provenance-design.md) keeps legacy behavior, introduces durable Workspace opt-out and honest local provenance, and scopes new transactional replay guarantees to strict-protocol callers. Workspace Sync activation and verified cross-server provenance transport remain outside this slice.
