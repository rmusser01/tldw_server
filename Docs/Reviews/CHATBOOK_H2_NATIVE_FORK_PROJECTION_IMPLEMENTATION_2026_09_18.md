# Task 1.1 implementation report

Status: DONE. Backlog: TASK-13261.3. Date: 2026-09-18.
Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chatbook-h1-history-design`.
Branch: `codex/chatbook-h2-native-fork`.
Base: `100dc8f677b54afb4311c2374cf5f3e0cd095dc0`.
Commit: `addee6cb1af31e15eaa841a53ef186b75b28bd3f` — `feat(chat): define native fork retained projection`.

## Implemented

- Closed/frozen Pydantic native scope, asset manifest, capture/request/resolve, binding template/descriptor, fork and retention operation results, retention request/context mutation and disabled-by-default capability contracts. H1 `state`, `child_id` and JSON-object `message_map` are preserved; message/asset maps are detached immutable mappings with JSON serializers. Terminal variants cannot carry a child/map.
- The exact fourteen-member §3 semantic tuple, compact literal-Unicode UTF-8 SHA-256, and six fixed independently constructed fixture vectors (Unicode, empty cursor, reviewed legacy projection, changed title, text representation, explicit missing marker). Request title/owner/scope/selected membership/assets/fidelity are semantic; H1 fences, selection revision and raw context hints are not included in this digest.
- Immutable `AuthorizedNativeOwner` and `ProjectedNativeForkContext`; a binding template has no allocated child identity.
- `MessageStore.read_native_fork_source`: caller-transaction-capable owner lock, complete H1 manifest/graph resolution, coherent resume read, selected content read and full H1 source/fence/context comparison. The adapter distinguishes an absent settings row from unreadable required JSON via non-null settings revision. It uses existing stores/SQL; no byte transfer/live behavior/provider call is introduced. H1 send functions and digest rules are unchanged.
- `capture_native_fork` returns the canonical request skeleton, exact typed embedded-image references/revisions and ordered hashes, required-effect inventory, and an explicit omitted-history-pins review reason. Legacy primary-image reference IDs are retained exactly rather than replaced with invented positional IDs.
- `project_native_fork_context` preserves canonical accepted snapshot/envelope values and independent canonical settings, with no live card/preset/worldbook lookup. Unknown settings/metadata/prompt extension carriers, unsupported personas/routing, malformed required envelope/state, unclassified derived memory, and old generated-image text mirrors reject explicitly.
- `project_native_fork_messages(rows, context=projected_context)` exports recursively immutable sanitized retained rows for the later commit remapper. It preserves exact selected text/order/images, recognizes accepted legacy/renamed character senders, rejects active rows, and validates closed function-call/result groups. Replay arguments must be finite JSON without structured source credentials; unsupported function-call/placeholder carriers reject. Historical citations retain bounded display excerpts/labels, not external URLs, source dereference IDs or saved query credentials.
- Retained row revisions hash sanitized semantic content, not H1 raw revisions, metadata timestamps or source admission JSON. H1 raw revisions remain inside the coherent adapter's validation only. A later commit must use this adapter plus these projectors, compare semantic retained revisions/digest, then allocate fresh message/replay/reference IDs and fresh child settlement/admission authority.

## Exact exclusions and accepted policies

- Public `summary`: removes `content`, `sourceRange`, `updatedAt`, `compressedCount`; preserves `enabled`, `thresholdMessages`/`messageThreshold`, `windowMessages`/`recentWindowMessages`.
- Materialized `values.behavior_controls.auto_summary.summary` is removed; normalized enabled/threshold/window controls remain.
- Materialized `values.behavior_controls.applied_overrides.summary` receives the same public summary projection.
- Public `roleplayPendingGreetingV1` and materialized `values.greeting` are removed. Declarative `behavior_controls.greeting`, public greeting selection policy and the accepted base snapshot greeting definition remain. This grants no seeding/live-lookup capability.
- Public source settings bookkeeping `updatedAt` and extraction side-effect setting `characterMemoryExtraction` are removed.
- History pins are filtered to selected rows in public settings and materialized controls/applied overrides; omitted pins are surfaced in capture review. Their IDs are still source references to be remapped at the later child commit.
- Accepted `default_memory` in both base and overridden participants drops explicit `summary`/`compaction` entries. Explicit creation-request content and manual entries remain. Known non-manual category entries remain only without a `source_conversation_id`; source-derived non-manual entries reject with `unsupported_native_fork_unclassified_derived_memory`. Unknown kinds/source provenance reject. Snapshot and materialized base/envelope digests are rebuilt through existing canonical builders after exclusions.

## Root-confirmed protocol clarifications used

These match the task's `contracts-and-constraints.md` clarifications and the root's subsequent memory/greeting rulings:

- Scope: `{kind:"global"}` or `{kind:"workspace", workspace_id}`. Global rejects even a null supplied workspace field; workspace requires a nonempty ID. Tuple form is `["global",null]` / `["workspace",id]`.
- Fidelity: `strict` / `allow_unavailable`. Disposition: `retained` / `unavailable`; `unavailable_v1` requires null hash and explicit degraded review.
- Finite roles: `message_image`, `document_context`, `generated_image`, `reference_image`, with role/representation coherence.
- Operation tags: `native_fork_v1` / `native_asset_retention_v1`.
- Asset hash: lowercase 64-hex SHA-256. Accepted behavior digests retain `sha256:`.
- Limits: identifiers 256, owner key 1024, title 200, with exact validated text retained.
- Internal projected snapshot schema is string `"1"`; canonical snapshot payload retains integer `schema_version: 1`.
- Canonical accepted rich records/inventory do not enable composition capabilities. Unknown required provenance/effects fail closed.
- Drop optional materialized `values.greeting` source pending/selected authority; keep declarative controls and accepted base definition.
- For memory, known non-manual categories with a source conversation are unclassified derived memory, not silently authored data.

## TDD and verification evidence

All Python commands used the project environment first:

```sh
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
```

All test commands ran from the assigned worktree with `PYTHONPATH="$PWD"`. After the first red run, `LOGURU_AUTOINIT=false` avoided inherited import/closed-capture logging noise.

Initial RED:

```sh
PYTHONPATH="$PWD" python -m pytest -o addopts='' tldw_Server_API/tests/Chat/unit/test_native_fork_projection.py -q
```

`/tmp/native_fork_task_1_1_red.log`: expected collection failure, `ModuleNotFoundError: ...native_fork_schemas`; 1 error, 4 warnings. Implementation did not exist when the tests were written/run.

Incremental behavior RED → GREEN, all using the same focused module and `-q`:

| Log | Observed result / break caught |
|---|---|
| `/tmp/native_fork_task_1_1_green_attempt1.log` | 25 passed, 4 warnings, 11.47s |
| `/tmp/native_fork_task_1_1_red2.log` | 7 failed, 33 passed: exact request serialization, image manifest, inventory, unknown prompt carrier, replay credentials/finite arguments |
| `/tmp/native_fork_task_1_1_green2.log` | 40 passed, 4 warnings, 12.17s |
| `/tmp/native_fork_task_1_1_red3.log` | 6 failed, 43 passed: citation projector missing, malformed policies accepted, malformed required controls leaked AttributeError |
| `/tmp/native_fork_task_1_1_red_memory.log` | `-k category_memory`: 1 failed, 1 passed, 49 deselected; source-derived category incorrectly retained |
| `/tmp/native_fork_task_1_1_green3.log` | 51 passed, 4 warnings, 13.97s |
| `/tmp/native_fork_task_1_1_red4.log` | `-k 'pending_greeting or malformed_metadata or accepted_character_sender'`: 6 failed, 51 deselected |
| `/tmp/native_fork_task_1_1_green4.log` | 57 passed, 4 warnings, 14.85s |
| `/tmp/native_fork_task_1_1_regressions.log` | Initial combined final matrix: 119 passed, 4 warnings, 15.29s |
| `/tmp/native_fork_task_1_1_red5.log` | `-k 'primary_image or participant_name'`: 2 failed, 75 deselected; legacy `:primary` reference rewritten and accepted renamed sender rejected |
| `/tmp/native_fork_task_1_1_red6.log` | `-k placeholder_metadata`: 2 failed, 77 deselected; unsupported placeholder could hide retained text |
| `/tmp/native_fork_task_1_1_final_tests.log` | Final combined matrix: **123 passed, 4 warnings in 16.10s** (79 new tests plus 44 existing H1 tests) |

Final focused/H1 command:

```sh
PYTHONPATH="$PWD" LOGURU_AUTOINIT=false python -m pytest -o addopts='' \
  tldw_Server_API/tests/Chat/unit/test_native_fork_projection.py \
  tldw_Server_API/tests/Chat/unit/test_history_selection.py \
  tldw_Server_API/tests/Chat/unit/test_history_context.py -q
```

Warnings were the existing Starlette/httpx deprecation, unknown pytest `plugins` config, Pydantic `schema` shadow warning, and Python `crypt` deprecation. No tests skipped. No broad unrelated suite was run.

Syntax/lint/format:

```sh
python -m compileall -q tldw_Server_API/app/api/v1/schemas/native_fork_schemas.py tldw_Server_API/app/core/Chat/native_fork_projection.py tldw_Server_API/app/core/DB_Management/chacha/message_store.py tldw_Server_API/tests/Chat/unit/test_native_fork_projection.py
python -m ruff check tldw_Server_API/app/api/v1/schemas/native_fork_schemas.py tldw_Server_API/app/core/Chat/native_fork_projection.py tldw_Server_API/app/core/DB_Management/chacha/message_store.py tldw_Server_API/tests/Chat/unit/test_native_fork_projection.py
python -m black --check tldw_Server_API/app/api/v1/schemas/native_fork_schemas.py tldw_Server_API/app/core/Chat/native_fork_projection.py tldw_Server_API/tests/Chat/unit/test_native_fork_projection.py
git diff --cached --check
```

Compile/ruff/diff checks passed. New files were Black-formatted; only the added adapter lines of the existing message store were formatted (`python -m black --line-ranges 371-432 .../message_store.py`) to avoid unrelated churn. The last placeholder change was Black-formatted and rechecked with Ruff/compile before the final tests.

Security:

```sh
python -m bandit -r tldw_Server_API/app/api/v1/schemas/native_fork_schemas.py tldw_Server_API/app/core/Chat/native_fork_projection.py tldw_Server_API/app/core/DB_Management/chacha/message_store.py -f json -o /tmp/bandit_native_fork_task_1_1.json
python -m bandit tldw_Server_API/tests/Chat/unit/test_native_fork_projection.py -s B101 -f json -o /tmp/bandit_native_fork_task_1_1_tests.json
```

Both JSON reports contain **0 findings, 0 scan errors**. Test-only B101 was excluded because these are pytest assertions, not production authorization checks. The production scan includes the entire touched existing message store; its old `nosec` explanatory comments produce Bandit parser warnings but no findings. No new suppression was added to production.

## Self-review and commit scope

Reviewed new schemas/projector/test source, canonical fixture assertions, staged file list, complete adapter diff, and final incremental diffs. Self-review fixes received their own failing tests before implementation: legacy primary image identity, accepted renamed participant sender, malformed false/empty metadata, materialized greeting exclusion, and replay placeholder guards. Checked retained source references remain inert and no child IDs/receipt/route/effect activation are created here.

The commit contains exactly:

1. `tldw_Server_API/app/api/v1/schemas/native_fork_schemas.py`
2. `tldw_Server_API/app/core/Chat/native_fork_projection.py`
3. `tldw_Server_API/app/core/DB_Management/chacha/message_store.py`
4. `tldw_Server_API/tests/Chat/unit/test_native_fork_projection.py`
5. `tldw_Server_API/tests/fixtures/native_fork_v1.json`

The fixture needed exact-path `git add -f` because `.gitignore:281` ignores `*.json`. No broad force-add was used. Git index writes required sandbox escalation; approved staging/commit operations targeted only these five files. Commit used normal `git commit` with no hook bypass or hook/config change. Git printed an existing unreachable-loose-object/gc warning; no repository maintenance was attempted.

Root-owned design/plan/Backlog edits and `.next-live-tier-h1-production/` remain unstaged and untouched by this unit. This report is the requested local task artifact, not part of the implementation commit.

## Limits / next-unit integration requirements

- This unit does not register public H2 routes, create receipts/children, retain external bytes, implement the frozen composer, or advertise enabled H2 effects. Overall H2 is not qualified by these tests.
- Capture currently creates typed manifests for existing owned embedded DB images. Typed external document/generated-image claims come from the later asset-store/service unit. Legacy generated text mirrors without native revision authority explicitly reject even with `allow_unavailable`; an unknown carrier is not converted to a missing marker.
- The accepted rich nested values remain canonical storage with required-effect inventory. Actual send/composer capability qualification must still reject every unknown or incompletely frozen effect before writes; storage preservation alone is not send fidelity.
- Later commit uses `read_native_fork_source(..., conn=transaction)` and the same projectors. It must not compare request semantic retained revisions to H1 raw revisions or reuse H1 source admission authority. It allocates/remaps child/replay/reference IDs and filters/remaps retained source pin references after projection.
- New real-database capture tests used SQLite. PostgreSQL transaction/concurrency qualification belongs to the later native operation/commit stage; no new PostgreSQL runtime claim is made here.
- No unresolved P1/P2 implementation concern is known after this self-review. Independent review has not yet run; root owns it.
