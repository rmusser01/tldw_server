# Stage 1 — Architecture survey, inventory, and the service/store pair

## Scope

Module shape and size/churn ranking for `tldw_Server_API/app/core/Sync/`; the two hot files
(`v2/service.py`, `v2/store.py`); the composition root (`v2/factory.py`); layering against
`Docs/Architecture.md`; and the test surface reachable by import-grep. Correctness of the
push/pull/conflict core is stage 2; the domain-adapter/materializer/bootstrap family is stage 3;
blob and retention boundaries are stage 4.

## Code Paths Reviewed

- `v2/service.py:SyncV2Service (1303-11042)` — 165 methods, 9,740 class lines
- `v2/service.py` module level — 35 dataclasses (`SyncV2Settings` 832, `SyncPushResult` 1128,
  `SyncRestorePreview` 1282, `SyncDiagnosticsReport` 1079, …) and 38 free functions
- `v2/store.py:SyncV2Store (110-2253)` — 138 DB-touching methods, 7 guard context managers
- `v2/store.py:materialization_guard (120-158)`, `conflict_resolution_guard (161-167)`,
  `conflict_resolution_savepoint (170-177)`, `personal_context_authority_guard (179-191)`,
  `personal_context_bootstrap_guard (194-208)`, `retention_guard (220-228)`,
  `retention_domain_guard (231-243)`, `blob_write_guard (246-259)`
- `v2/store.py:transition_notes_task_readiness (618-647)` vs
  `v2/store.py:transition_notes_task_activity_readiness (650-679)`
- `v2/factory.py:sync_v2_service_for_user (126-246)`, `_sync_v2_settings_from_env (460-469)`,
  `_sync_v2_bool_env (472-476)`
- `v2/security.py:_TRUE_ENV_VALUES (20)`, `_parse_bool (184-189)`
- `v2/personal_context_ongoing_contract.py:158` — the module's only `core/ -> app/api/` import
- `Sync_Client.py:ClientSyncEngine (156-…)`, `Sync_Client.py:main (1043-1057)`
- `sync_contract.py (1-41)`

### Size × churn (top of `2026-09-21-stage1-churn-baseline.txt`)

| File | LOC | commits/12mo |
| --- | ---: | ---: |
| `v2/service.py` | 11,361 | 127 |
| `v2/store.py` | 2,253 | 76 |
| `v2/models.py` | 2,238 | 45 |
| `v2/profile.py` | 1,645 | 38 |
| `v2/notes_moodboard_studio_contract.py` | 1,592 | 7 |
| `v2/notes_organization_coordinator.py` | 1,334 | 12 |
| `v2/server_origin_batch.py` | 1,329 | 21 |
| `v2/notes_task_contract.py` | 1,202 | 4 |
| `Sync_Client.py` | 1,112 | 16 |

### Longest methods on `SyncV2Service`

`restore_preview (5315-5960)` 646 · `_attachment_lifecycle_diagnostics (7859-8407)` 549 ·
`push (4491-5023)` 533 · `_expand_task_client_push (3874-4181)` 308 ·
`stage_personal_context_authority (1868-2154)` 287 · `pull (5025-5276)` 252 ·
`resolve_conflict (6890-7103)` 214 · `retention_dry_run (3338-3497)` 160 ·
`_push_task_client_group (4266-4422)` 157 · `retention_compact (3188-3336)` 149 ·
`capabilities (2551-2697)` 147 · `_pull_versioned (10159-10304)` 146

## Tests Reviewed

100 test files reach `core.Sync` by import-grep (`2026-09-21-stage1-test-inventory.txt`);
72 live in `tldw_Server_API/tests/Sync/` (80,854 LOC). **This module is not untested.**

- `tests/Sync/test_sync_v2_service.py` (9,370 LOC, 165 tests) — the primary behavioural suite for
  `SyncV2Service`. Covers capabilities, device lifecycle, push/pull, conflict paths, restore.
  Downgrades risk on the orchestration surface, but is SQLite-only.
- `tests/Sync/test_sync_v2_store.py` (7,109 LOC) — store/`SyncDatabase` contract, including
  fake-backend PostgreSQL lock-ordering assertions. Contains one long-standing red test (below).
- `tests/Sync/test_sync_v2_endpoints.py` (4,271 LOC) — the API boundary.
- Six `*_postgres_contract.py` files (notes, notes_link, notes_organization, notes_task,
  notes_task_activity, notes_task_readiness) plus five personal-context files that parametrise
  `["sqlite", "postgres"]` (`test_sync_v2_personal_context_activation.py:33`). 16 of 72 files
  mention Postgres at all.
- `tests/MediaDB2/test_sync_client.py` — the only substantive consumer of `Sync_Client.py`.
- There is **no `tests/Sync/conftest.py`**; `sync_store`, `_clock`, `_envelope`,
  `_ready_encryption` are redefined per file (e.g. `test_sync_v2_service.py:92,106,108,218`;
  `test_sync_v2_chat_materializer.py:22,39`; `test_sync_v2_endpoints.py:114,126`).

## Validation Commands

```
$ find tldw_Server_API/app/core/Sync -name '*.py' | xargs wc -l | sort -rn | head -1
   43905 total          # 65 .py files

$ grep -rl "core\.Sync" tldw_Server_API/tests | wc -l
     100

$ grep -c "return self\.db\." tldw_Server_API/app/core/Sync/v2/store.py
     125

# AST classification of SyncV2Store methods that call self.db.*
WITH connection: 76
WITHOUT connection: 62

$ grep -rn "utcnow()" tldw_Server_API/app/core/Sync/
(no output)                     # zero tz-naive utcnow sites in this module

$ grep -rn "from tldw_Server_API.app.api" tldw_Server_API/app/core/Sync/
tldw_Server_API/app/core/Sync/v2/personal_context_ongoing_contract.py:158   # 1 site, schema-only, deferred

$ grep -rn "ClientSyncEngine" tldw_Server_API/app/ | grep -v Sync_Client.py
(no output)                     # zero production importers

$ python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_service.py -q -p no:randomly
165 passed, 11 warnings in 17.76s

$ python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_store.py \
      tldw_Server_API/tests/Sync/test_sync_v2_retention.py \
      tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py -q -p no:randomly
FAILED tldw_Server_API/tests/Sync/test_sync_v2_store.py::test_postgres_personal_context_receipt_locks_binding_before_upsert
1 failed, 266 passed, 2 skipped, 13 warnings in 12.24s

$ python -m pytest tldw_Server_API/tests/Sync -q -p no:randomly
E   ModuleNotFoundError: No module named 'psycopg'
ERROR tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_postgres_contract.py
ERROR tldw_Server_API/tests/Sync/test_sync_v2_notes_task_postgres_contract.py
!!! Interrupted: 2 errors during collection !!!
7 warnings, 2 errors in 3.17s

# Full directory, with the two uncollectable files excluded (background run, exit 0):
$ python -m pytest tldw_Server_API/tests/Sync -q -p no:randomly \
      --ignore=.../test_sync_v2_notes_task_postgres_contract.py \
      --ignore=.../test_sync_v2_notes_task_activity_postgres_contract.py
12 failed, 2908 passed, 38 skipped, 62 warnings in 10895.35s (3:01:35)

# Five of the twelve failure names survived the run's output buffer:
FAILED .../test_sync_v2_personal_context_exchange_gate.py::test_mixed_selected_conflicts_with_exact_proof_resolve_in_request_order
FAILED .../test_sync_v2_personal_context_exchange_gate.py::test_mixed_exact_proof_preserves_native_notes_resolution_actions[overwrite]
FAILED .../test_sync_v2_personal_context_exchange_gate.py::test_mixed_exact_proof_preserves_native_notes_resolution_actions[duplicate_rename]
FAILED .../test_sync_v2_server_origin_capture.py::test_workspace_chat_api_write_stays_direct_when_sync_active
FAILED .../test_sync_v2_store.py::test_postgres_personal_context_receipt_locks_binding_before_upsert

# All of them reproduce in isolation — deterministic, not a test-ordering artifact:
$ python -m pytest .../test_sync_v2_personal_context_exchange_gate.py \
      .../test_sync_v2_server_origin_capture.py -q -p no:randomly
4 failed, 127 passed, 13 warnings in 32.57s

# test_sync_v2_personal_context_exchange_gate.py:772
E   AssertionError: assert ['mixed-exact-note'] == ['mixed-exact-note', 'mixed-exact-personal']
E     Right contains one more item: 'mixed-exact-personal'

# test_sync_v2_server_origin_capture.py:1484
E   assert 404 == 201
```

## Findings

### FINDING sync-4

```
axis:        encapsulation
class:       n/a
severity:    Medium
sites:       v2/service.py:SyncV2Service (1303-11042) — 165 methods / 9,740 class lines;
             v2/service.py module level — 35 dataclasses + 38 free functions;
             deferred sibling imports at v2/service.py:1881, 3772, 3805, 3945, 3987, 3999,
             4006, 4239, 4387, 4763, 6841, 9950 (28 function-local imports total);
             the cycle's other half: v2/notes_task_coordinator.py:37 (TYPE_CHECKING import of
             SyncV2Service), v2/personal_context_activation.py, v2/personal_context_conflicts.py,
             v2/server_origin.py, v2/server_origin_batch.py, v2/replay.py — each takes
             `service: SyncV2Service` as a parameter
canonical:   NONE
destination: Follow the shipped in-repo precedent: core/DB_Management/Media_DB_v2.py (121
             commits of churn) was split into the core/DB_Management/media_db/ package
             (api.py, constants.py, errors.py, legacy_content_queries.py, runtime/). The
             equivalent for Sync is a core/Sync/v2/service/ package with one responsibility per
             module — `transport.py` (push/pull/cursors), `conflicts.py`, `retention.py`,
             `blobs.py`, `diagnostics.py`, `results.py` (the 35 DTOs), `settings.py` — and the
             coordinators depending on those narrow modules instead of on the whole service.
knowledge:   "What one Sync operation is allowed to do" is spread over one 9,740-line class whose
             collaborators reach back into it. The deferred imports are not a style choice: they
             exist because service.py and six sibling modules form an import cycle that a
             top-level import would break.
scenario:    n/a (encapsulation)
impact:      Medium, not High, because the cycle is currently stable. The cost is change
             amplification and review blindness: 127 commits/12mo land in one file, every new
             collaborator must either be injected through the 22-parameter constructor or added
             as another function-local import, and there is no module boundary that a reviewer or
             a lint rule can hold. Two of the findings in this ledger (sync-1, sync-10) are
             defects that survive precisely because the code that should agree with itself is
             400 lines apart in the same file.
cost-driver: n/a
tests:       import-grep reachability, not measured coverage — tests/Sync/test_sync_v2_service.py
             (165 tests, all passing), tests/Sync/test_sync_v2_endpoints.py,
             tests/Sync/test_sync_v2_factory.py
effort:      expensive — needs Docs/Design/, an ADR, a Backlog task, and a staged
             IMPLEMENTATION_PLAN. The DTO extraction (35 dataclasses -> results.py) is the cheap
             first stage and breaks no cycle.
owner-only:  no
confidence:  confirmed (the size, the 28 deferred imports, the cycle); assumption (that the
             media_db/ package shape is the right target boundary — that needs the design doc)
```

### FINDING sync-5

```
axis:        duplication
class:       true-duplication
severity:    Medium
sites:       v2/store.py:SyncV2Store (110-2253) — 125 one-line `return self.db.<same-name>(...)`
             forwarders over ~138 DB-touching methods;
             the five structurally identical non-re-entrant guards:
               v2/store.py:conflict_resolution_guard (161-167)
               v2/store.py:personal_context_authority_guard (179-191)
               v2/store.py:retention_guard (220-228)
               v2/store.py:retention_domain_guard (231-243)
               v2/store.py:blob_write_guard (246-259)
             each of which is `with self.db.materialization_transaction(keys) as connection:
             guarded = copy(self); guarded._connection = connection; yield guarded`, versus
               v2/store.py:materialization_guard (120-158) — the only one with the
               `if self._connection is not None: ... yield self; return` re-entrancy short-circuit;
             the consequence at the call site: v2/service.py:_ensure_personal_context_conflict_candidate
             (9940-9971), which hand-rolls `if store._connection is None:` at :9955 — the only
             place in the whole repo that reads `SyncV2Store._connection` from outside store.py;
             the split the signature does not show: 76 forwarders pass `connection=self._connection`
             and 62 do not (full list produced by AST in the validation run)
canonical:   NONE — `SyncV2Store` is itself the intended seam
destination: Keep SyncV2Store, but shrink it to what it actually owns: one
             `v2/store_guards.py` (or a `_guard()` helper inside store.py) owning
             "open-or-join one Sync transaction and hand back a bound store", parameterised by
             the lock keys, and used by all seven guards so re-entrancy is decided in one place.
             The 125 forwarders should be generated or removed, not hand-written.
knowledge:   Two pieces of knowledge are duplicated. (a) "How a Sync transaction is opened and
             bound to a store copy" — written six times, with the re-entrancy rule in only one.
             (b) "Whether this read participates in the caller's open transaction" — encoded 138
             times as the presence or absence of one keyword argument, invisible at the call site.
scenario:    Non-transactional read inside a held guard, one confirmed instance:
             v2/personal_context_activation.py:295 calls `guarded.get_dataset_envelope_range(...)`
             while holding `personal_context_authority_guard` (opened at :272), and
             v2/store.py:get_dataset_envelope_range (1843-1845) forwards WITHOUT `connection=`,
             so SyncDatabase.get_dataset_envelope_range (Sync_DB.py:10701-10704) opens
             `self.backend.transaction()` of its own. On SQLite this is safe today — verified
             directly: the outer connection stays `in_transaction=True` across the nested call and
             a subsequent rollback still discards the outer writes. Whether it is safe on
             PostgreSQL depends on the backend's connection handling, which nothing in the Sync
             layer asserts and no Sync test covers. Labelled probable-risk, not a defect.
impact:      Medium. The immediate cost is that a reviewer cannot tell a transactional store call
             from a non-transactional one, and adding a sixth guard means re-deciding the
             re-entrancy question by copy-paste. The latent cost is the SQLite/PostgreSQL split
             the repo already has precedent for.
cost-driver: n/a
tests:       import-grep reachability, not measured coverage — tests/Sync/test_sync_v2_store.py
             (7,109 LOC), tests/Sync/test_sync_v2_personal_context_activation.py (parametrised
             sqlite/postgres). No test asserts that a store read inside a guard joins that guard's
             transaction.
effort:      moderate — the guard consolidation is mechanical and well covered; the
             connection-threading audit needs a per-method decision.
owner-only:  no
confidence:  confirmed (the six duplicated guard bodies, the 125 forwarders, the 76/62 split, the
             single external `_connection` read); probable-risk (the PostgreSQL behaviour of the
             one untransacted read inside a guard)
```

### FINDING sync-11

```
axis:        duplication
class:       true-duplication
severity:    Medium
sites:       NOTE (2026-09-22): between this audit's runs and its write-up, a concurrent session
             in this working tree applied `pytest.importorskip("psycopg")` to both files below.
             The collection-abort half of this finding is therefore already fixed and uncommitted
             locally; the conftest, the 12 red tests, and the CI gate remain.
             Unguarded module-scope psycopg imports that abort collection of the whole directory
             (as observed during this audit, before that change):
               tests/Sync/test_sync_v2_notes_task_postgres_contract.py:10
               tests/Sync/test_sync_v2_notes_task_activity_postgres_contract.py:10
             (the four sibling *_postgres_contract.py files do not import psycopg at module scope);
             12 failing tests in the directory, of which these five were recovered from the run
             output and all five reproduce deterministically in isolation:
               tests/Sync/test_sync_v2_store.py:test_postgres_personal_context_receipt_locks_binding_before_upsert
                 (3036-3066) — red since 2026-09-03: its fake dataset row (`metadata_json` at
                 :275) omits `link_state`, which
                 core/DB_Management/Sync_DB.py:complete_personal_context_link_receipt (3968-3974)
                 began requiring in commit 8c97f181e5;
               tests/Sync/test_sync_v2_personal_context_exchange_gate.py:772
                 (`test_mixed_selected_conflicts_with_exact_proof_resolve_in_request_order`) plus
                 the two `test_mixed_exact_proof_preserves_native_notes_resolution_actions`
                 parametrisations — a mixed notes/personal-context resolution batch returns only
                 the notes item in `resolved`; the personal-context item is silently dropped.
                 That symptom is exactly what core/Sync/v2/service.py:resolve_conflicts_batch
                 (6871-6872) produces: `except Exception: rejected.append(index); continue` turns
                 ANY failure inside `PersonalContextConflictService.resolve_batch_item` into an
                 indistinguishable per-item rejection, so the suite cannot say whether this is a
                 real regression or a stale fixture. Last touched 2026-09-04 (42bb7cf0f1);
               tests/Sync/test_sync_v2_server_origin_capture.py:1484
                 (`test_workspace_chat_api_write_stays_direct_when_sync_active`) — asserts 201,
                 gets 404. Last touched 2026-08-10 (1058c09dd4);
             the remaining seven failure names scrolled out of the run's output buffer and were
             not recovered;
             no shared fixture module — `sync_store`/`_clock`/`_envelope`/`_ready_encryption`
             redefined per file, e.g. tests/Sync/test_sync_v2_service.py:92,106,108,218;
             tests/Sync/test_sync_v2_chat_materializer.py:22,39;
             tests/Sync/test_sync_v2_endpoints.py:114,126;
             tests/Sync/test_sync_v2_diagnostics.py:30,46;
             tests/Sync/test_sync_v2_attachment_refs.py:31;
             tests/Sync/test_sync_v2_domain_adapters.py:71;
             tests/Sync/test_sync_v2_media_compat.py:44;
             tests/Sync/test_sync_v2_attachment_materializer.py:72
canonical:   a `tests/Sync/conftest.py` — which does not exist
destination: `tests/Sync/conftest.py` owning exactly the shared Sync v2 test fixtures
             (`sync_store`, `_clock`, `_ready_encryption`, `_envelope`) and nothing else; plus
             `pytest.importorskip("psycopg")` in the two unguarded contract files.
knowledge:   "How you stand up a Sync v2 store/service for a test" is written out ~10 times.
             When the `SyncV2Settings` constructor or the encryption-readiness helper changes,
             every copy is a separate edit.
scenario:    n/a (test health)
impact:      Medium. Three compounding facts: (1) `pytest tldw_Server_API/tests/Sync` does not run
             at all in an environment without psycopg — it aborts at collection, so all 70 other
             files are skipped silently rather than the two Postgres files being skipped;
             (2) neither blocking gate executes this directory — `backend-required.yml:193-195`
             runs only `tldw_Server_API/tests/unit` with `-m "unit and not e2e and not jobs"`,
             and `coverage-required.yml:154-157` runs only `tests/unit` + `tests/sanity_tests`
             (plus the AuthNZ floor at :167-169); (3) as a direct consequence **12 tests are
             red**, at least one since 2026-09-03 (~18 days), and the three exchange-gate
             failures sit on the personal-context conflict path where `resolve_conflicts_batch`'s
             blind `except Exception` (core/Sync/v2/service.py:6871-6872) makes a genuine
             regression and a stale fixture look identical from the outside.
             A contributing factor worth stating plainly: the directory takes **3h 01m** to run,
             which is why nobody runs it — the CI step proposed below has to be a scoped subset
             or a nightly, not this command.
             This is NOT the repo-wide `--cov-fail-under=12` complaint: it is that the largest
             behavioural suite protecting a 43,905-LOC module is outside every contractual gate.
cost-driver: n/a
tests:       n/a — this finding is about the tests
effort:      cheap for the two `importorskip` lines and the conftest. Triaging 12 red tests and
             picking a gate-able subset out of a 3-hour suite is moderate, and the three
             exchange-gate failures need the blind `except` at service.py:6871-6872 to surface
             the real error before anyone can tell what they mean. The CI step has a precedent to
             copy (`coverage-required.yml:160-169`, "AuthNZ coverage floor").
owner-only:  no
confidence:  confirmed (the collection abort, the 12 failures, and the five named failures
             reproducing in isolation were all observed in the validation runs above);
             assumption (that all 12 are pre-existing rather than environment-specific — the
             working tree carries unrelated uncommitted edits from a concurrent session, though
             none under core/Sync/)
```

### FINDING sync-14

```
axis:        duplication
class:       true-duplication
severity:    Low
sites:       Dead module: Sync_Client.py (1,112 LOC, 16 commits/12mo) —
               Sync_Client.py:ClientSyncEngine (156-…), Sync_Client.py:main (1043-1057).
               Zero importers under app/; the only consumers are
               tests/MediaDB2/test_sync_client.py and tests/DB_Management/test_media_db_api_imports.py:91.
             Coercion duplication (this module's landing of repo cluster C2, two copies, both
             with the same truthy set and both missing the canonical spelling `y`):
               v2/security.py:_TRUE_ENV_VALUES (20) + v2/security.py:_parse_bool (184-189)
               v2/factory.py:_sync_v2_bool_env (472-476)
             versus the shared-but-bypassed core/MCP_unified/environment.py:is_truthy (17), whose
             `_TRUTHY` set is `{"1","true","yes","y","on"}`.
canonical:   core/MCP_unified/environment.py:is_truthy (17) — exists, is bypassed, but is
             MCP-package-scoped and should not be imported across that boundary as-is
destination: `core/Utils/coercion.py` owning scalar and environment coercion only
             (`is_truthy`, `env_bool`, `env_int`, `env_optional_positive_int`). Explicitly NOT
             `Utils/Utils.py` and NOT `http_client.py`.
knowledge:   "Which strings mean true" — one operator-facing contract, currently three answers in
             the repo and two inside this module.
scenario:    n/a
impact:      Low on both halves. `Sync_Client.py` is a pure maintenance tax — 16 commits in 12
             months (ruff, bandit, linting, a db-ownership refactor) on code nothing in the server
             calls. The truthy divergence is fail-closed: `SYNC_V2_SERVER_TRUSTED_ENABLED=y`
             silently leaves server-trusted encryption disabled and surfaces as a readiness
             blocker rather than as an insecure default, so it is an operator footgun, not a hole.
cost-driver: n/a
tests:       import-grep reachability, not measured coverage — tests/MediaDB2/test_sync_client.py
             covers Sync_Client.py; tests/Sync/test_sync_v2_security.py and
             tests/Sync/test_sync_v2_factory.py cover the coercion sites.
effort:      cheap for the coercion consolidation. For Sync_Client.py, confirm intent first — it
             is still named as a target in Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md:143
             and Docs/Plans/2026-03-16-media-db-v2-remaining-gaps-plan.md:89, so deleting it is a
             product decision, not a cleanup.
owner-only:  no
confidence:  confirmed (zero production importers; the two truthy sets); assumption (that
             Sync_Client.py is genuinely retired rather than staged for a desktop client)
```

## Suggested Refactor/Actions

1. `sync-11` first — it is the cheapest and it unblocks measuring everything else. Add
   `pytest.importorskip("psycopg")` to the two contract files, add `tests/Sync/conftest.py`, fix
   the `link_state` fixture, then add a "Sync contract floor" step to
   `coverage-required.yml` modelled line-for-line on the existing "AuthNZ coverage floor"
   (`:160-169`). Small enough to ship without a design doc; needs a Backlog task.
2. `sync-5` next — consolidate the seven guards onto one parameterised `_guard()` and delete the
   `store._connection` read at `v2/service.py:9955`. Mechanical, well covered by
   `tests/Sync/test_sync_v2_store.py`. Separately, sweep the 62 non-threading forwarders and
   decide each one explicitly; `get_dataset_envelope_range` is the one with a live caller inside
   a guard.
3. `sync-14` coercion half — create `core/Utils/coercion.py` and route `v2/security.py` and
   `v2/factory.py` through it. Cross-module cluster; coordinate with the C2 owner rather than
   fixing Sync in isolation.
4. `sync-4` last and only behind a design doc — `Docs/Design/2026-MM-DD-sync-v2-service-decomposition-design.md`,
   an ADR (it changes a module boundary that four existing ADRs reference), a Backlog task linking
   both, and `IMPLEMENTATION_PLAN_sync-service-decomposition.md` with the DTO extraction as
   stage 1. Point at the `media_db/` migration as the template; do not invent a new shape.
5. `sync-14` Sync_Client half — raise as a product question, not a PR.
