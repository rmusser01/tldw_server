# UAT193 / UAT194 / UAT195 approved bounded repair

Tasks: TASK-13260.131 (UAT193 default bootstrap/name uniqueness) and TASK-13260.132 (UAT194 foreign character visibility/mutation under the configured bypass role). Parent approved the bounded store repair and PG67→68 migration after causal controls; shared schema source released at07e0abf1c4. Historical hold notes below describe the earlier decision sequence.

Native evidence: Alice user 2 can GET character 2 (Helpful AI Assistant), whose stored client_id is 1. POST /chats/ reaches the immutable behavior factory and rejects that foreign character. The factory owner predicate must remain unchanged.

Source evidence: the shared PostgreSQL content backend ignores the SQLite per-user path. CharacterStore ID/name/list/query reads have no PostgreSQL owner predicate. Default dependency bootstrap reuses the first name match. The base schema has globally unique character names; scoping the name lookup alone would turn reuse into a name-conflict failure. Character update SQL also rewrites client_id to the current instance, so mutation authorization needs a bounded control before choosing a repair.

Causal tests use two actual CharactersRAGDB instances with client IDs 1 and 2 on the same official PostgreSQL fixture database. SQLite controls use separate user files, matching the existing physical isolation contract. Tests exercise real dependency default-seeding, real character and chat HTTP routes, real quota counts and behavior factory. Request authentication is supplied as a fixed owner principal and the DB dependency returns the corresponding real instance; no claim of a full AuthNZ login/cache test. Frequency throttling and sync transport are disabled locally. No model calls.

Expected boundaries: default owned by each user, visible default can create a chat, user-created owned character succeeds, foreign detail/list/query/name reads are isolated, equal display names work for different users, missing/deleted/foreign create produces no conversation, and foreign update cannot transfer ownership.

## Causal result

Official required-PG run: 8 failed, 16 passed, zero skipped (24 collected), 37.61 seconds. Eight PostgreSQL failures establish foreign default reuse, visible-default POST 400, foreign detail/list/query/by-name exposure, globally colliding display names and foreign PUT changing stored owner 1 to 2. All 12 SQLite controls and four PostgreSQL controls (owned chat creation; missing, deleted and foreign rejection without a partial chat) pass. No production or test bytes changed during the run.

## Proposed minimum coherent repair — requires parent approval

**Role-contract qualification added after parent review:** the following store proposal is held pending the deployment-role decision. Existing forced character RLS rejects foreign reads/updates under a verified non-bypass role. The 194 exposure/mutation is proven under fixture/native RLS-bypass roles, not under ordinary enforced RLS. The PG name migration is independently necessary. Do not expand the entire store just to accommodate an unsupported test configuration without a product-contract decision.

1. Preserve the factory's PostgreSQL owner check, world-book ownership checks and immutable behavior materialization. Preserve SQLite's physical per-user isolation and existing sync-client semantics.
2. Make CharacterStore's PostgreSQL character-card reads consistently owner-scoped: ID/batch/name, list/query/count, setup selectors, full-text/tag search and raw tag reads. Apply the owner predicate even when deleted rows are requested. Use parameterized predicates local to character-card SQL; do not change the generic execute helper. Existing system_init rows are not a blanket exception to ownership: retain their data, but seed each user's normal Helpful AI Assistant through the existing owner-aware path.
3. Guard character mutations at the SQL boundary with the same current owner, including update/delete/restore and their preflight/fallback reads. A previously read ID or matching version is not authorization. Keep versions, tombstones, creation IDs, sync behavior and transaction decisions intact. Exemplar/tag child operations and route preflights must be checked for the same parent-ownership requirement before claiming the character surface is isolated.
4. Replace PostgreSQL's global character-name uniqueness with uniqueness on `(client_id, name)` through the normal versioned migration path. Existing globally unique rows already satisfy this narrower key; preserve all rows, IDs, ownership and content. Keep same-owner duplicate/tombstone reservation behavior and SQLite name uniqueness. Add a catalog/data-preservation migration test, including repeat initialization and existing default rows. Do not merely rename Alice's default, transfer the admin row or drop uniqueness.
5. Once these contracts are in place, the current default bootstrap can find or create the user's own default without accepting a foreign by-name match. Add defense-in-depth validation only if needed by the reviewed store API. Exercise the actual initializer/dependency two-user path after the 181 plumbing freeze, plus explicit/nested caller transaction rollback, snapshot immutability, same-owner conflict, deleted/default behavior and foreign mutation/child controls.

Expected production paths: `chacha/character_store.py` for the table-local ownership contract; `ChaChaNotes_DB.py` for versioned PostgreSQL name-constraint migration only (must coordinate with the 181 owner); optionally `ChaCha_Notes_DB_Deps.py` only for a demonstrated bootstrap invariant not provided by the corrected store. Endpoint/factory changes are not currently required by the reproduced failure. This is larger than a one-line factory change and must be reviewed as such before release.

No production edits yet. Do not relax factory ownership, alias a foreign row, rewrite native data, or introduce a global read-helper policy. This diagnosis does not certify all other shared-PG resources, world-book CRUD or sync entry points; any further concrete findings need explicit scope and tracking.

## Exact proposed PostgreSQL schema entry

- Change only `_POSTGRES_SCHEMA_VERSION` from 67 to 68. Leave `_CURRENT_SCHEMA_VERSION` (SQLite) at 67; the existing backend-specific constants and initializer already distinguish these heads.
- Add `_migrate_from_v67_to_v68_postgres(conn)` adjacent to the existing 66→67 PostgreSQL migration and invoke it under `target_version >= 68 and current_version < 68`, directly after the existing PG67 branch. Update the local version to 68 before `_runtime_schema_version` is set. Do not change earlier migrations or the SQLite linear registry.
- In the caller's existing schema transaction, identify the character table's exact single-column `name` unique constraint from `pg_constraint`/`pg_attribute`, using its table OID and column identities. Create fixed `character_cards_client_id_name_key UNIQUE (client_id, name)`, then drop only the validated former name constraint using the backend's identifier quoting. Use no CASCADE. Preserve any unrelated constraints/indexes and all row data; unexpected catalog shape must fail with a schema error rather than silently dropping unknown constraints.
- Confirm the resulting owner/name key and absence of a remaining global name-only key, then update `db_schema_version` from 67 to 68 and verify it. Reopening version68 must leave rows and constraints unchanged. A controlled failure during migration must roll back both catalog changes and version advancement.
- The new permanent migration tests build **real v67** through the actual initializer with a subclass capped at `_POSTGRES_SCHEMA_VERSION=67`, assert the historical version/global-name constraint, seed live/default/deleted rows, then reopen through the normal constructor. They do not relabel a current database as old or bypass migration methods.

Current schema/control RED: four expected failures, two passes, zero skips. The same-owner bootstrap concurrency barrier passes; different-owner simultaneous bootstrap returns the same ID and fails. The non-bypass role policy control passes. The first schema test attempt prematurely closed the shared factory backend in fixture cleanup; it is retained separately and is not counted as causal RED. Corrected tests retain the original database lease while reopening. Its one orphan NOLOGIN test role was removed using an exact-prefix/flags/single-role official-fixture cleanup control, which passed.

Before production release: choose the supported role handling for194, then add the applicable foreign delete/restore/batch/setup/child and owned-operation controls, migration rollback fault, and explicit caller transaction controls. Existing factory/SQLite behavior remains a required gate.

## Bounded predicate coverage proposed for an accepted privileged service role

The inspected runtime accepts its configured content role and has no privilege rejection. If that accepted service configuration remains supported, limit defense-in-depth to **CharacterStore's existing character-card and exemplar surface**, following the local owner-predicate convention already used by the factory and the Notes ADR. This is not a proposal to change every shared-content store or global RLS implementation.

| Boundary | Exact methods / treatment |
| --- | --- |
| Character fetches | `get_character_card_by_id`, `get_character_cards_by_ids`, `get_character_card_by_name`; PG `client_id = db.client_id` even with deleted rows |
| Browse and setup | `list_character_cards`, `query_character_cards`, `query_character_setup_options`, `get_character_setup_option_by_id`; predicate applies before count, pagination, sort and projection; correlated conversation counts/last-used filters must exclude foreign conversations |
| Search and tags | `search_character_cards`, `_search_cards_by_tags_json` and any reachable PG fallback, `_get_raw_character_tags`; `manage_character_tags` obtains only owned candidates and relies on protected updates |
| Card writes | `update_character_card`, `soft_delete_character_card`, `restore_character_card`; owner predicate in SQL UPDATE and every version/tombstone/fallback read, not only endpoint preflight; no-op/idempotent paths must not report foreign ownership success |
| Exemplar children | Existing add/get/list/update/delete/search methods derive owner from the linked character, without a redundant owner column. Filter reads and writes by an owned parent using a local EXISTS/join predicate; preserve their current child deletion semantics |

One small private helper may provide the static-alias PG-only predicate and bound owner argument, if that avoids mismatched SQL/parameter branches. Do not parse/arbitrarily rewrite SQL, change backend execute semantics, alter the factory, add RLS exceptions, change scope-context ownership, or touch another store.

Required controls before GREEN: two real owners under the fixture bypass role; unchanged restricted-role policy; per-file SQLite with different sync-client IDs; foreign detail/list/query/setup/batch/name/search/tag and card/exemplar mutation rejection; owned operations; duplicate names within owner and across owners; deleted/tombstone behavior; same-owner and two-owner bootstrap concurrency; actual default/helper → GET → chat factory; immutable snapshots; explicit/nested caller rollback and no partial conversation after rejected creation. Further services outside this exact surface require a separate reproduced boundary before any change.

## Approved implementation and UAT195 attribution

The parent accepted privileged configured service roles for this bounded character contract because startup accepts the role and the Notes precedent requires local predicates. Ordinary restricted-role RLS controls pass; no ordinary-role leakage is claimed. Production edits are limited to CharacterStore and the PG schema constant/migration/registration in ChaChaNotes_DB. Deps and the factory remain unchanged.

UAT195 is separately associated with TASK13260.133. The driver replay proves IndeterminateDatatype42P18 for the isolated `? IS NULL` parameter with both omitted and supplied optional filters. Eleven permanent PostgreSQL regressions fail while eleven SQLite controls pass; adding `CAST(? AS TEXT)` at the existing optional predicates supplies the type without changing filtering. The prior suggestion that supplied filters were an owned positive was rejected by the probe; owned listing is the passing positive.

Three extra migration RED controls establish missing upgrade behavior for atomic rollback, an unrelated unique constraint, and a conflicting destination constraint. All earlier failed harness attempts remain retained and are identified separately.
