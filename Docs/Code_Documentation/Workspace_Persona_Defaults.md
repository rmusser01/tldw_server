# Workspace Persona Choices And Local Startup History

Stages 2A and 2B of [#2950](https://github.com/rmusser01/tldw_server/issues/2950) add durable opt-out and local startup provenance to the existing Workspace Persona defaults. The [approved design](../Design/2026-09-13-persona-workspace-choice-provenance-design.md) and [implementation plan](../superpowers/plans/2026-09-13-persona-startup-provenance-implementation-plan.md) define the boundaries. The implementation is one activation unit: do not deploy its intermediate storage, creation or projection commits separately.

This slice does not add auto-provisioning, a strict startup route, receipts, replay guarantees, send-time admission changes, Workspace Sync, Research RAG adoption, MCP profile activation, Buddy behavior or UI changes. Stages 2C/2D and later work remain open.

## Read And Write Contract

Workspace management responses expose `assistant_defaults_explicit_none` as a read-only boolean. Continue using the existing version-checked PATCH command to change the choice:

| Operation | Stored defaults | Opt-out |
| --- | --- | --- |
| Normal Workspace creation | SQL NULL | false |
| PATCH omits `assistant_defaults` | unchanged | unchanged |
| PATCH `assistant_defaults: null` | SQL NULL | true |
| PATCH valid Persona defaults | reference object | false |
| Existing legacy SQL NULL at upgrade | unchanged | true |
| Existing legacy non-null at upgrade | unchanged, including malformed values | false |

Clearing an already-empty default still records opt-out and advances the Workspace version. Defaults and opt-out change in one optimistic-locking update; conflicts leave both untouched. Renames, upserts, archive/unarchive, and soft deletion retain the choice. There is no public reset-to-unset command. Sending the derived field directly in PUT/PATCH is rejected with 422, including null.

The migration's conservative opt-out does not prove historical user intent. Corrupt non-null defaults, including a default paired with true, remain unavailable in the effective management projection (`invalid_default`); explicit owner clear/save repairs the pair. Persona references remain user-owned and reference-backed, with no snapshots.

Clone snapshots and the current Research Workspace import format do not represent a trustworthy portable Persona choice. New clone and import targets therefore opt out. Importing into an existing Workspace preserves its choice; clone publication preserves the target's opt-out. No cross-owner Persona reference is copied.

## Startup Selection

The existing `POST /api/v1/chats/` route applies defaults only to new personal Workspace chats without an explicit assistant or validated parent lineage. Original request omission/null intent, not equality with today's default, determines the source:

| Validated creation request | Public source | Origin references |
| --- | --- | --- |
| Workspace, assistant omitted, available configured default | `workspace_default` | Selected Workspace ID/version |
| Workspace, assistant omitted, unset or cleared default | `system_fallback` | Examined Workspace ID/version |
| Workspace, explicit normalized Persona or Character | `explicit` | null |
| Workspace, untracked with a supplied null identity field | `explicit_none` | null |
| Workspace, validated parent conversation lineage | `fork` | null |
| Global API/Sync, legacy/direct creation or imports | `unknown` | null |

Supplying null for `assistant_kind`, `assistant_id` or `character_id` suppresses inheritance when the normalized choice is untracked. A non-null tracked choice wins over a simultaneous nullable compatibility field. Explicit choices and forks do not consult the Workspace default's availability, and forks do not gain new parent-assistant inheritance behavior.

A configured unavailable default fails closed: 503 when Persona support is disabled, otherwise 409. Missing, deleted or staged Workspaces return 404. Existing archived-Workspace behavior is retained. An unset or cleared default produces an untracked chat, not a provisioned Persona.

Scope, quota and parent/message checks happen before creation. For non-Character Workspace creation, the final Workspace and selected Persona are locked in that order, then identity, default title and provenance are inserted in the same transaction. A default changed during preflight cannot produce a stale identity paired with a new version. Nonempty caller-supplied titles are preserved. Character creation forwards a separate trusted value inside the existing factory transaction; its preflight and existing greetings, participants, presets and behavior snapshots are unchanged. This is atomic local selection, not a retry/idempotency contract.

## Read-Only Provenance

Existing chat and conversation metadata responses expose the same `assistant_startup` object, including create/detail/list/update/restore and conversation tree metadata. Public Character library list/metadata/search results also use checked projection; the internal storage column is never a public field.

```json
{
  "schema_version": 1,
  "source": "workspace_default",
  "workspace_id": "workspace-id",
  "workspace_version": 2
}
```

The object has exactly these four keys and a 1024-byte encoded UTF-8 JSON limit. Schema version is the strict integer 1. `workspace_default` requires a nonempty ID and positive strict-integer version; `system_fallback` has both references or neither. All other sources have null references. It contains no Persona name, prompt, assistant ID, memory mode or policy snapshot. A legacy Workspace ID too large for a complete object produces a bounded input error before new-chat insertion; existing access is unaffected and IDs are never truncated.

NULL, corrupt or oversized storage projects to `unknown` with null references. A reference is disclosed only while its originating Workspace remains visible through the authenticated user's DB. Reading the conversation does not itself authorize its old origin. Deleted, missing or staged origins redact the whole value to `unknown`; archived but visible origins remain valid historical references. Redaction never changes stored history. Repeated origins may share a request-local visibility lookup, never a cross-request/user cache. Storage failures are errors, not successful `unknown` responses.

Startup creation and reference projection check the DB handle's construction-time `owner_user_id`, supplied by the trusted per-user dependency. Cached handles retain their first caller's `client_id` for existing writer attribution, even when voice or a background worker initializes them before REST. That alias is not owner authority. Public Character library projection uses the same owner identity. Standalone `CharactersRAGDB` callers default ownership to their initial `client_id`; callers with separate attribution must supply the trusted owner explicitly, never from request metadata or inferred paths.

Both `assistant_startup` and `assistant_startup_json` are rejected in create/update request bodies, including null. Ordinary DB insertion/update dictionaries reject the same keys. Only the internal typed insertion argument can create trusted provenance. The value describes local startup history, not current Persona eligibility, resume eligibility or Character behavior-snapshot authority.

## Lifecycle And Transport

Actual normalized changes to `(assistant_kind, assistant_id, character_id, persona_memory_mode)` clear origin atomically with the identity update, including Sync replacements. Changing the identity back does not restore the old value. Metadata, empty updates, normalized no-ops, settings/history changes and scope-only moves preserve it. A title-only Sync envelope is not necessarily metadata-only: the existing replacement semantics determine the resulting binding. Soft deletion, tombstones and restore preserve storage; hard deletion and account erasure remove it with the row. The initializer's historical missing-Character-field repair preserves provenance for equivalent normalized bindings and clears it for invalid or changed bindings; this is not a general identity repair service. The origin has no Workspace foreign key, so deletion cannot rewrite surviving history elsewhere.

Current Chatbook, OpenWebUI, legacy history and Sync formats do not transport verified provenance. Existing exports and outgoing Sync field lists omit it; imports/new remote rows start unknown even if the source supplies either key. Nested OpenWebUI source metadata may retain those strings as uninterpreted data, never as assistant selection or authoritative provenance. This does not introduce shared-recipient adoption or cross-server reference copying.

Workspace clone snapshots contain Workspace metadata, memberships, sources, notes and artifacts, not conversations. They do not carry conversation startup history; the snapshot model and clone service required no changes for 2B.

## Required Maintenance Upgrade

Schema v68 adds a non-null choice bit with a false default (SQLite integer with a 0/1 constraint; PostgreSQL boolean), then backfills SQL NULL defaults to true in the registered migration transaction. Fresh databases use the same registered migration chain. SQLite completes legacy initialization and source-catalog checks through v67 before starting the final migration transaction: legacy script helpers can implicitly commit, so none may run after the new writes. Historical schemas and validation guards are unchanged.

Schema v69 adds nullable `conversations.assistant_startup_json TEXT` through registered v68-to-v69 migrations. SQLite bounds the byte length of a BLOB cast; PostgreSQL uses `octet_length`. Existing rows remain NULL: there is no inference, backfill, snapshot table, origin index or Workspace foreign key. The SQLite post-v67 tail applies the pending v68/v69 steps within its final transaction; PostgreSQL uses its existing initializer transaction.

1. Schedule an offline maintenance window. Block incoming writes and drain active requests/jobs. Stop every old API process, worker, and direct database writer, including idle processes holding cached handles. Inventory every affected per-user ChaChaNotes database and PostgreSQL deployment before proceeding.
2. Take a consistent, restorable pre-upgrade backup while writes are stopped. Use the deployment's existing database backup procedure, including SQLite WAL handling or PostgreSQL backup tooling. Do not copy only an active SQLite main file.
3. Deploy the compatible binary and initialize each affected database through its normal `CharactersRAGDB` configuration/ownership path, with user traffic and workers still stopped. Use one migration initializer per database during maintenance. Initialization applies the registered steps through v69; do not patch schema markers or add columns manually. Inspect failures before proceeding to another database.
4. Verify each database reports schema v69. Check representative legacy-null defaults are opted out, configured defaults retain their references, malformed defaults remain unchanged, and pre-upgrade conversations have unknown origin. In a disposable Workspace, test clear/save/read across a restart, stale-version rejection, inherited and explicit-null creation, metadata preservation and identity-change invalidation. Check safe origin projection after origin access is lost while the conversation remains readable.
5. Restart only compatible API and worker binaries and reopen traffic after all checks pass. Keep the maintenance window closed if writer quiescence or database coverage cannot be guaranteed.

Initialization-time version checks reject an older binary opening migrated data. They do **not** fence an older process's already-open handle: former clear/save statements can corrupt the choice pair, and old identity writers do not invalidate new provenance. This release does not support mixed-version rolling upgrades. The backend regression includes the cached-writer hazard as an explicit deployment limitation, not a claim that runtime fencing exists.

## Recovery And Verification

A failed v68-to-v69 migration transaction retains v68 without the new column; the failed SQLite post-v67 tail rolls back its pending migration writes. Failures in preceding legacy validation/initialization must not apply later columns or backfills. This is not a new atomicity guarantee for upgrades between older historical versions. After successful migration, prefer a compatible fix-forward binary. Never downgrade schema markers, drop choice/provenance columns or run old writers against v69. Restoring the full pre-upgrade backup in an offline recovery loses subsequent writes, so reconcile those separately with explicit operator approval; an old-binary rollback retaining post-upgrade data is not supported.

Backend regressions are in `tldw_Server_API/tests/DB_Management/test_workspace_persona_optout_v68.py`; they use the repository's PostgreSQL fixture and SQLite database initialization. They cover fresh/legacy storage, transitions, restart, lifecycle, rollback, and old cached writers. API regressions remain in `tests/Workspaces/test_workspace_assistant_defaults_api.py`. A skipped PostgreSQL fixture is not PostgreSQL rollout evidence: require a successful live run before deploying that backend.

Local provenance coverage additionally lives in `test_conversation_assistant_startup.py`, `test_workspace_assistant_creation_atomic.py`, the Workspace provenance/projection suites, and existing chat/Character/transport suites. Required live PostgreSQL tests cover both blocking orders for default changes and creation, and local/Sync mutation races. See the implementation plan and TASK-13245.4 for final gate evidence; intermediate-stage passes are not the complete activation gate.

### Local 2B Execution Evidence (2026-09-14)

The parent ran the final gates on implementation commit `6a8da29037107da8f901c7095841b9a70712a8ec`; the documentation stage does not rerun those suites or scans.

| Gate | Recorded result |
| --- | --- |
| Core, API, SQLite and required live PostgreSQL matrix (17 files) | 678 passed, 26 warnings, zero failures/skips, 237.84s; `/tmp/persona-stage5-core-final.log`. PostgreSQL 18.6, official fixture with `TLDW_TEST_POSTGRES_REQUIRED=1`. |
| Full six-file Character/library/import/export regression | 394 passed, three pre-existing skips, 23 warnings, zero failures, 403.29s, exit 0; `/tmp/persona-stage5-character-transport-final.log`. Skips: Resource Governor limits, V3 format fixture, removed streaming route. |
| Exact 29 changed Python files | Compilation and whitespace checks passed. Ruff: 42 findings versus 48 at execution base `8876d2d187`, none new by file/code/message comparison. |
| Bandit | Production: zero findings/errors. Tests, excluding only B101 assertions: eight findings, all reproduced at base (nine there); no new suppressions. |
| Independent review and publication | Stages 1-4 individually approved; final whole-branch review and one draft implementation PR remain parent-owned pending gates. The human-written Change summary remains required. |

The two disjoint final test gates total 1,072 passed, three pre-existing skips and zero failures. These results do not certify repository-wide tests, live providers, Chatbook runtime, browser UAT, or PostgreSQL Character startup. Warnings and base-matched static findings are retained, not described as a clean lint run. Commands, artifacts and remaining gates are recorded in the focused plan and TASK-13245.4. Both parent test processes have exited; the parent retains the task-created PostgreSQL container for potential independent-review probes and owns its later cleanup.

### Historical 2A Evidence

Stage 2A was verified on PostgreSQL 18.6 on 2026-09-13: all six PostgreSQL opt-out cases passed. The combined SQLite/PostgreSQL opt-out, defaults DB/API, PostgreSQL clone lifecycle and v67 migration regression passed 114 tests with no failures or skips. Tests used the existing `pg_database_config` fixture with `TLDW_TEST_POSTGRES_REQUIRED=1`, which fails rather than skips if the server is unavailable. This validates the tested backend paths; deployments still require the per-database offline checks above.

## Compatibility Limits

Stage 2B verification reproduced a pre-existing PostgreSQL Character factory failure in `WorldBookService._initialize_tables`: the backend connection wrapper lacks the context-manager protocol used by that preflight. It fails before the new transaction/INSERT. SQLite Character regressions and PostgreSQL non-Character Persona creation are covered; they do not certify PostgreSQL Character startup. This slice does not change WorldBook initialization.

Chatbook source comparison informs the design, not full runtime parity. The approved server behavior intentionally fails closed for unavailable configured defaults; Chatbook's console resolver can degrade to an untracked assistant with a notice and treats supplied custom prompts differently. Local IDs and prompt snapshots are not copied between systems. Cross-client strict startup and other parity work remain separate future gates.

Parent-fetched dev refs checked on 2026-09-14: server `1e0bb6feddab5a9e4be794ea44ea21b9ea29bf30`, Chatbook `2a10cc3a307c368d14bc73be9a8b1cd9109dfeb4`. Since the planning refs (`ebdeeac384c58559fa90fd3a5f79f5262ae190d5` / `4631b60f8dd9623fc55bf16f4a37e29fcb1240c7`), server changes are VZ drill tooling/docs and Chatbook changes are Notes/import UI/release files; neither changes the compared Persona/Workspace contract paths. Server dev remains schema v67; this stack advances 2A v68 to 2B v69. Source comparison does not replace runtime validation or authorize a rebase/deployment. TASK-13245, #2950, 2C/2D and later parity work remain open.
