# Move API schemas consumed by core into core-owned models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut every `core.* -> api.v1.schemas` edge from the layering baseline by moving the models core reads into core packages, re-exported unchanged from `api/v1/schemas`, with the OpenAPI document byte-identical.

**Architecture:** For each bounded context, move exactly the definitions core imports, plus the definitions those reference (their closure), verbatim into a core module. The api module imports them back, so endpoints, services and tests keep their import paths. Core importers are retargeted, and each PR deletes or lowers its `layering_baseline.txt` lines.

Two edges are cut by inversion instead of a move. The Sync contract exporter moves up to the API layer (S9). `ChaChaNotes_DB` stops re-validating an API payload that its only caller already validated (S6).

Ten statements belong to sibling tasks:
- TASK-13950.13 PR C1 deletes `chat_service`'s two key-resolution imports. It also moves `get_api_keys` and the provider-name lists into `core/config_llm_providers.py`.
- TASK-13950.14 retargets the eight `DEFAULT_LLM_PROVIDER` importers.

S10 deletes the last lines after `.14` slice 2 merges.

**Tech Stack:** Python 3.12/3.13, Pydantic v2, FastAPI, pytest, the TASK-13950.9 layering contract (`tldw_Server_API/tests/lint/test_layering_contract.py`), `Helper_Scripts/export_openapi_schema.py`.

**Spec:** `backlog/tasks/task-13950.15 - Move-API-schemas-consumed-by-core-into-core-owned-models.md` (acceptance criteria #1-#3; AC #1 as amended on 2026-10-26: "Schemas imported by core move to core/<context>/models (or *_schemas in core) with re-exports from api/v1/schemas; services importers of moved names stay on the api re-export and move in TASK-13950.37"); roadmap `Docs/superpowers/plans/2026-10-25-architecture-remediation-roadmap.md`, `## Phase 1`, row `.15` (effort M; 1 PR per context, Evaluations first; add an OpenAPI snapshot diff test before the first move).

## Measured starting point

Measured on `origin/dev` `35e711080a` merged with `origin/chore/13950-9-layering-contract` `9bff409144` (PR #3337, stacked on #3334). On that merge the layering ratchet passes. `python -m tldw_Server_API.tests.lint.test_layering_contract` reports 262 pairs, 683 statements, and a largest package SCC of `{'module_level': 102, 'with_function_level': 130}`. Neither #3334 nor #3337 is on dev yet.

There are 34 `core.* -> api.v1.schemas` pairs with 92 statements. Line numbers are on the measured merge. `fn` means a function-level import, and `try` means one under `try/except ImportError`. The owner is the slice below, `.13` for TASK-13950.13 PR C1, or `.14` for TASK-13950.14.

| Baseline line | Importing module:line, schema module -> owner |
|---|---|
| `core.Audio -> api.v1.schemas: 1` | Realtime.default_pipeline:14 audio_schemas -> S3 |
| `core.Audio_Studio -> api.v1.schemas: 1` | providers.speech:8 audio_schemas -> S3 |
| `core.Audiobooks -> api.v1.schemas: 3` | alignment_utils:10, subtitle_generator:14, tag_parser:9 audiobook_schemas -> S7 |
| `core.Buddy -> api.v1.schemas: 2` | service:13 buddies, turns:16 buddy_turns -> S8 |
| `core.Chat -> api.v1.schemas: 6` | chat_helpers:14 chat_request_schemas -> S10; chat_loop_store:10 chat_loop_schemas -> S8; chat_service:2156(fn), chat_service:2251(fn) `chat_request_schemas as _schemas_mod` (API keys) -> .13; chat_service:2821(fn) `_is_server_managed_chat_request_field` -> S10; chat_target_resolution:11 `DEFAULT_LLM_PROVIDER` -> .14 |
| `core.Chunking -> api.v1.schemas: 1` | auto_boundary_assistant:630(fn) `DEFAULT_LLM_PROVIDER` -> .14 |
| `core.Claims_Extraction -> api.v1.schemas: 1` | `__init__`:1 claims_schemas -> S7 |
| `core.Collections -> api.v1.schemas: 1` | reading_service:470(fn) media_request_models -> S7 |
| `core.DB_Management -> api.v1.schemas: 3` | ChaChaNotes_DB:43677(fn) quizzes -> S6; Evaluations_DB:23 evaluation_recipe_schemas, Evaluations_DB:29 evaluation_schemas_unified -> S1 |
| `core.Data_Tables -> api.v1.schemas: 1` | jobs_worker:40 `DEFAULT_LLM_PROVIDER` -> .14 |
| `core.Evaluations -> api.v1.schemas: 22` | embeddings_abtest_jobs_worker:33, embeddings_abtest_runner:8, embeddings_abtest_service:13, recipe_runs_jobs:8, recipe_runs_jobs:9, recipe_runs_jobs_worker:14, recipe_runs_jobs_worker:15, recipe_runs_service:14, recipe_runs_service:20, recipes.base:8, recipes.embeddings_recipe_hints:13, recipes.embeddings_retrieval:10, recipes.persona_dialogue_tree_robustness:8, recipes.rag_answer_quality:11, recipes.rag_retrieval_tuning:10, recipes.registry:7, recipes.reporting:7, recipes.summarization_quality:9, synthetic_eval_generation:9, synthetic_eval_repository:10, synthetic_eval_service:10 -> S1; recipes.rag_retrieval_tuning_execution:11 rag_schemas_unified -> S2 |
| `core.File_Artifacts -> api.v1.schemas: 2` | file_artifacts_service:16, jobs_worker:30 -> S8 |
| `core.Flashcards -> api.v1.schemas: 2` | generation:5 flashcards, source_review:12 study_packs -> S6 |
| `core.Ingestion_Media_Processing -> api.v1.schemas: 1` | document_upload_preflight:8 document_upload_processing -> S7 |
| `core.Local_LLM -> api.v1.schemas: 4` | llamacpp_acquisition_jobs:7, llamacpp_acquisition_service:20, llamacpp_inventory_service:10, llamacpp_profile_capabilities:10 -> S8 |
| `core.MCP_unified -> api.v1.schemas: 1` | modules.implementations.rag_module:12 rag_schemas_unified -> S2 |
| `core.Monitoring -> api.v1.schemas: 1` | topic_monitoring_service:32 -> S8 |
| `core.Persona -> api.v1.schemas: 2` | archetype_loader:15 archetype_schemas -> S8; live_tts:10 audio_schemas -> S3 |
| `core.Prompt_Management -> api.v1.schemas: 1` | prompt_improvement_dispatch:12 `ChatCompletionRequest` -> S10 |
| `core.RAG -> api.v1.schemas: 5` | rag_service.response_mapping:7, rag_service.source_health:8, rag_service.transport:13, rag_service.unified_pipeline:2673(fn,try), rag_service.unified_pipeline:9151(fn,try) -> S2 |
| `core.Research -> api.v1.schemas: 2` | service:13, streaming:8 research_runs_schemas -> S8 |
| `core.Research_Workspace -> api.v1.schemas: 5` | capabilities:16 -> S4; output_jobs:18 audio_schemas -> S3; output_jobs:19 `DEFAULT_LLM_PROVIDER` -> .14; output_jobs:20 research_workspace_outputs, output_jobs:28 workspace_schemas -> S4 |
| `core.Scheduled_Tasks -> api.v1.schemas: 1` | recurring_question_rag_adapter:9 -> S2 |
| `core.Setup -> api.v1.schemas: 2` | provider_catalog:5, provider_validation:10 -> S8 |
| `core.Sharing -> api.v1.schemas: 2` | shared_workspace_clone_jobs_worker:23, shared_workspace_clone_operations:16 -> S4 |
| `core.Skills -> api.v1.schemas: 1` | skill_executor:385(fn) `DEFAULT_LLM_PROVIDER` -> .14 |
| `core.Streaming -> api.v1.schemas: 2` | speech_chat_service:34 audio_schemas -> S3; speech_chat_service:41 `DEFAULT_LLM_PROVIDER` -> .14 |
| `core.StudyPacks -> api.v1.schemas: 3` | generation_service:14 `DEFAULT_LLM_PROVIDER` -> .14; generation_service:15, jobs:11 study_packs -> S6 |
| `core.Sync -> api.v1.schemas: 1` | v2.personal_context_ongoing_contract:158(fn) sync_v2_models -> S9 |
| `core.TTS -> api.v1.schemas: 3` | realtime_session:14, tts_jobs_worker:20, tts_service_v2:25 -> S3 |
| `core.VN_Assets -> api.v1.schemas: 2` | preflight:5, service:14 -> S5 |
| `core.VN_Play -> api.v1.schemas: 2` | setup_options:11 vn_asset_schemas, setup_options:15 vn_play_schemas -> S5 |
| `core.Workflows -> api.v1.schemas: 3` | adapters.audio.multi_voice_tts:82(fn), adapters.audio.tts:13(try) -> S3; adapters.llm.llm:52(fn) `DEFAULT_LLM_PROVIDER` -> .14 |
| `core.Workspaces -> api.v1.schemas: 2` | assistant_defaults:13 chat_session_schemas, assistant_defaults:14 workspace_schemas -> S4 |

Per owner: S1 23, S2 8, S3 10, S4 7, S5 4, S6 5, S7 6, S8 15, S9 1, S10 3, `.13` C1 2, `.14` 8. The total is 92. This task cuts 82.

Drift from the backlog figures (dev `dca11342de`):
- Evaluations is still 22 and Research_Workspace is still 5.
- `services -> api.v1.schemas` is now 36, not 35.
- TASK-13950.14 says "six core modules read `DEFAULT_LLM_PROVIDER`". Eight do: `Chat.chat_target_resolution:11` and `Streaming.speech_chat_service:41` are missing from its list. `.14`'s own plan covers all eight.
- The 11 core files importing `chat_request_schemas` hold 13 statements, and 11 after `.13` C1.

Every test patch target into a module this task touches names `chat_request_schemas.API_KEYS`, `.get_api_keys` or `.load_and_log_configs`. That surface belongs to `.13` C1, which rewrites it (`.14` plan, R2 table). An AST count of string targets plus module-object `setattr`/`patch.object` calls finds 35 targets in 9 files. `test_config_providers_endpoints.py` holds 4 more in a `_target` variable. V5b's `grep` control finds the 16 string forms. None of them names a definition this task moves.

Per amended AC #1, services importers of moved names keep using the api re-export; TASK-13950.37 moves them. They are `services/audiobook_jobs_worker.py`, `services/outputs_service.py`, `services/study_pack_jobs_worker.py`, `services/media_ingest_jobs_worker.py`, `services/osce_generator.py` and `services/quiz_generator.py`.

## Global Constraints

Copied from the roadmap (binding):
- **Branching and PR size.** Every PR targets `dev`. One backlog task per PR. A task with more than about 400 changed lines is split into ordered PRs, each passing CI on its own.
- **Raising a baseline.** It is allowed only through a documented bump-with-reason entry, reviewed in the PR, so that security fixes inside frozen functions are not blocked.
- **Line-keyed legacy baseline.** `Helper_Scripts/ci/scope_predicate_baseline.txt` is still keyed by line number until `.23` re-keys it. Any PR that shifts lines in a file it lists regenerates it with `--write-baseline`.
- **Shims.** A move keeps a re-export shim at the old import path until `git grep` shows no importer and no string monkeypatch target still uses it.
- **Postgres proof.** Postgres-dependent acceptance criteria are proven in the Postgres shard with `TLDW_TEST_POSTGRES_REQUIRED=1`. A local skip is not evidence.
- **Backlog edits** use `tools/backlog-py` only; stage with `git add -A backlog/tasks`. Cite duplicate ids (TASK-12108, TASK-12013) by title.
- **Review gates.** Each AI-authored PR carries a `## Change summary` (ADR-004) and gets a clean-subagent `/superpowers:requesting-code-review` before merge. Each task's own plan gets an independent review plus a dry run before execution.
- **Owner decisions (2026-10-25)** (moderation, keywords, scheduler, admin): none bears on this task.

Task-specific:
- **Move the closure, nothing more.** Move what a core module imports, plus every definition those read: class bases, field annotations, defaults, validators, module constants, and the module-level `try`/`if` blocks that bind those names. Models only endpoints use stay in `api/v1/schemas`. Each task lists the exact names.
- **Verbatim.** Moved code is cut and pasted unchanged, in source order. The only allowed changes are a `# noqa` comment, the new module's import lines, and imports retargeted per R2. Moved code keeps any mypy errors it already has; V6 compares per group. Reviewers read the move with `git diff --color-moved=dimmed-zebra`.
- **The api re-exports are permanent.** AC #1 makes `core-owned model + api re-export` the end state, as in `api/v1/schemas/history_selection_schemas.py`. Endpoints, services and tests keep importing from `api/v1/schemas`, so they need no retargeting. The Shims rule's removal condition does not apply to these re-exports.
- **Consumers keep `from <module> import <Name>`.** A retarget changes only the module path. Patch targets such as `core.TTS.tts_service_v2.OpenAISpeechRequest` keep resolving to the same object.
- **No laundering.** A cut removes the import. Moving it under `TYPE_CHECKING`, into `importlib`, into `services`, or into another api module is not a cut. The one existing `TYPE_CHECKING` import, `ChaChaNotes_DB.py:63`, is type-only; the contract skips it by design, and Task 10 records it as AC #2's one exception.
- **The OpenAPI artifacts never change.** `apps/packages/api-types/openapi.fingerprint.json` and `apps/packages/api-types/paths.d.ts` stay byte-identical in every PR.
- **Sibling ownership.**
  - `.13` C1 deletes `chat_service.py:2156` and `:2251`. It also creates `core/config_llm_providers.py` and moves `get_api_keys`, `_BASE_SUPPORTED_PROVIDER_NAMES_LIST` and `ALL_SUPPORTED_PROVIDER_NAMES_LIST` there.
  - `.14` retargets the eight `DEFAULT_LLM_PROVIDER` importers, appends the default-provider resolver to that module, and deletes the schema's import-time config reads.
  - This task touches none of that before S10.
- **`LAYERS` in `test_layering_contract.py` is not edited.**
- **Sibling plans touch the same lines.** Each owning task applies its step only if the sibling has merged first; V1 reports any stale entry either way. The order-dependent gaps on each sibling's side are listed so that sibling can amend its plan.
  - **`.13`** (`2026-10-26-task-13950-13-llm-gateway.md`):
    - PR B replaces `core/Chat/__init__.py` with a docstring. That makes Task 8's `__init__` note moot.
    - PR C1 lowers `core.Chat -> api.v1.schemas` from 6 to 4.
    - Sibling side: `.13` PR B's red test `test_importing_does_not_load[...core.Chat...]` expects `core/Chat/__init__.py` to pull in `api.v1.schemas.chat_loop_schemas` through `chat_loop_store`. If S8 lands first, that test no longer fails before B.
  - **`.14`** (`2026-10-26-task-13950-14-schema-defaults.md`):
    - It adds `tldw_Server_API/tests/lint/test_schemas_config_boundary.py` with two allowlists. In `IMPORT_TIME_CONFIG_ALLOWLIST`, the `api/v1/schemas/rag_schemas_unified.py` entry goes stale in Task 2. In `CHAT_REQUEST_SCHEMAS_CORE_IMPORTERS`, the `core.Chat.chat_helpers`, `core.Chat.chat_service` and `core.Prompt_Management.prompt_improvement_dispatch` entries go stale in Task 10.
    - Sibling side: if S2 lands before `.14` slice 1, `.14` must not seed the RAG entry. Its R4 already notes this.
  - **`.16`** (`2026-10-26-task-13950-16-db-management-logic.md`):
    - PR C moves `core/Sync/v2/personal_context_ongoing_contract.py` into `core/Sync_Contracts/`, with a module alias at the old path, and adds `core.Sync_Contracts -> api.v1.schemas: 1  # bump:`. That affects Task 9.
    - Its `test_db_management_logic_imports.py` counts any module `DB_Management` imports that reaches DB code. Task 1 pins this independently.
    - Sibling side: if S9 lands before PR C, `.16` Task 4's `--write-baseline` never emits `core.Sync_Contracts -> api.v1.schemas: 1`.
- **ADR-004 waiver.** The owner waived ADR-004 Change summaries for TASK-13950 PRs. Each PR's `## Change summary` section records the waiver.

## Review Focus

1. **OpenAPI drift from a move.** A component can be renamed or reshaped two ways. First, Pydantic module-qualifies colliding names: 50 components today, for example `tldw_Server_API__app__api__v1__schemas__evaluation_schemas_unified__EvaluationResponse`. Of the 59 component keys containing `__`, the other 9 are FastAPI `Body_*` names. Second, a moved field default can lose its module-level source, such as RAG's `_DEF_*` settings defaults or `_compat.Field(example=...)`. No moved name is module-qualified today. Every task pins this with V2 (fingerprint) and V3 (raw-document `cmp`). A negative control in the dry run confirmed that V2 fails on a renamed component.
2. **Models outside OpenAPI change shape silently.** Seventeen moved models are not components, so the fingerprint cannot see them:
   - S1: `SyntheticEvalDraftSampleCreate`, `SyntheticEvalPromotionCreate`, `SyntheticEvalReviewActionCreate`;
   - S4: `WorkspaceOperationBase`, `_RecipientModel`;
   - S7: `AddMediaForm`, `AudioVideoOptions`, `ChunkingOptions`, `PdfOptions`, `AlignmentPayload`, `AlignmentWord`, `ChapterPreview`;
   - S8: `ResearchArtifactManifestEntry`, `ResearchCheckpointSummary`, `ResearchRunSnapshotResponse`, `BuddyModel`;
   - S10: `BaseMessage`.

   V3 pins them. It diffs the validation and serialization JSON schema of every name in `CORE_OWNED` against the merge base, and each PR body records the per-model digests for the slice's members of this set (R6).
3. **Import cycles through a heavy package `__init__`.** A re-export now imports a core package first. Heavy ones are `core.RAG` (4.6 s cold; imports `unified_pipeline`), `core.Claims_Extraction`, `core.VN_Assets`, `core.VN_Play`, `core.Sharing`, `core.Local_LLM`, `core.StudyPacks` and `core.Persona`. Also, `api/v1/schemas/__init__.py` imports `chat_loop_schemas`. V4 pins this with a fresh-interpreter import of every api and core module in `CORE_OWNED`. `tests/Claims/test_claims_analytics_exports_api.py::test_claims_export_filter_schema_imports_in_fresh_interpreter` already does this for `claims_schemas`.
4. **A cut that hides the import instead of removing it.** The layering contract counts function-level and `try`-guarded imports. The identity test asserts each moved class's `__module__` is the core module. Task 10 Step 5 greps core import statements for `api.v1.schemas` and expects exactly the `ChaChaNotes_DB.py:63` `TYPE_CHECKING` line.
5. **Patch or reload drift.** A test that patches or `importlib.reload`s an api module stops affecting moved code. `tests/Config/test_env_file_selection.py:85-87` reloads `chat_request_schemas`. Today every patch target names an attribute that `.13` C1 rewrites and that does not move. Each task's V5b greps the slice's modules for patch targets naming a moved definition and expects none. Task 10 re-runs the scan after `.13` and `.14` land.

---

## PR slices

Merge order is S1 to S10. Each slice is based on dev after the previous slice merges. Slices may be prepared in parallel: on a `layering_baseline.txt` conflict, take dev's side, re-apply the slice's deltas (R1), and run `--tighten`. Append `CORE_OWNED` rows in slice order. "Hand" counts hand-written lines; the S1 dry run measured about 169 before this round's two added tests. "Moved" counts verbatim cut-and-paste lines (`git diff --color-moved` shows them).

| Slice | Context | Base | Statements cut | Hand | Moved |
|---|---|---|---|---|---|
| S1 (Task 1) | Evaluations + Evaluations_DB | dev after #3334 and #3337 | 23 | ~195 (incl. 3 tests) | ~180 |
| S2 (Task 2) | RAG + `_compat.Field` | dev after S1 | 8 | ~100 (incl. docs, backlog notes) | ~2,170 |
| S3 (Task 3) | Speech: TTS request + speech chat | dev after S2 | 10 | ~85 | ~310 |
| S4 (Task 4) | Workspaces, Research_Workspace, Sharing | dev after S3 | 7 | ~155 | ~450 |
| S5 (Task 5) | VN_Assets, VN_Play | dev after S4 | 4 | ~100 | ~420 |
| S6 (Task 6) | Flashcards, StudyPacks, ChaCha OSCE import | dev after S5 | 5 | ~90 (incl. the fail-closed test) | ~105 |
| S7 (Task 7) | Ingestion, Collections, Claims, Audiobooks | dev after S6 | 6 | ~120 (incl. one doc) | ~465 |
| S8 (Task 8) | Small contexts: Local_LLM, Buddy, Persona, Research, Setup, Monitoring, File_Artifacts, Chat loop | dev after S7 | 15 | ~210 | ~390 |
| S9 (Task 9) | Sync contract exporter (inversion) | dev after S8 | 1 | ~35 | ~45 |
| S10 (Task 10) | Chat request model; AC close-out | dev after S9 **and** `.14` slice 2 | 3 | ~80 (incl. docs, ratchet edits) | ~975 (re-measured) |

The roadmap asks for one PR per context, which would be 25 or more PRs.

- **Bundles accepted (controller ruling 2026-10-26).** S7 and S8 bundle small contexts: each has 1-4 statements and follows the same recipe, and each PR stays at about 120-210 hand-written lines, which keeps it reviewable. That reviewability is what the roadmap's one-context-per-PR rule exists for.
- S6 and S9 stay separate because they are inversions, not moves.
- The work is closer to L than the roadmap's M.

Conflict hazards with open PRs (#3321-#3349) on 2026-10-26:

| PR | File | Slice | Handling |
|---|---|---|---|
| #3321 | `api/v1/schemas/chat_request_schemas.py` | S10 | S10 re-measures the closure on its base, which is after `.13` C1, which is after #3321 |
| #3331 | `core/RAG/rag_service/unified_pipeline.py` (listed in `scope_predicate_baseline.txt`) | S2 | Locate the two imports by their quoted text, not by line number |
| #3328, #3339, #3342 | `core/DB_Management/ChaChaNotes_DB.py` | S6 | Small hunk. Locate `import_osce_quiz_entry_atomic` by name |
| #3325, #3326, #3327, #3330 | `pyproject.toml` | S2 | One added per-file-ignore line. Rebase on conflict |

---

## The move recipe (every slice)

The rows of `CORE_OWNED` are the single list of what moved. The identity test, V3, V4 and V6 all read them.

- **R1 (red).** Append the slice's rows to `CORE_OWNED` in `tldw_Server_API/tests/unit/test_core_owned_schema_reexports.py` (Task 1 creates the file).

  Baseline edits are deltas. Each task gives the delta and the line's value on the measured merge. Apply the delta to the value on the slice base, because `.13` C1, `.14` and earlier slices lower some of the same lines. A line that reaches 0 is deleted.

  Run V1a. Expected:
  - Each new row fails. A new core module fails with `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.<new module>'`; an existing one fails with `AttributeError: module '<existing core module>' has no attribute '<name>'`.
  - `test_layering_contract.py::test_imports_follow_the_layering_contract` and `test_core_to_api_import_boundary.py::test_core_to_api_imports_only_fall` both fail. They report `core.<X> -> api.v1.schemas: <n> statements > baseline <m>`, or `..., not in the baseline` for a deleted line. A raised bump line shows `stale, baseline <n> but <m> measured`.
- **R2 (move).** For each row, cut the listed definitions from `api/v1/schemas/<api module>.py` and paste them verbatim, in source order, into the core module. The core module starts with:
  - a one-line docstring: `"""Core-owned <topic> models, re-exported by the API schemas package (TASK-13950.15)."""`;
  - the source's `from __future__ import annotations` line, if it has one;
  - the source's import lines the moved code uses, so the same objects resolve (for example the same `Field`).

  Some of those source imports read an api module that re-exports a core owner. Retarget each such import to the owner. There are five today: `evaluation_recipe_schemas` (`RunStatus`), `research_workspace_outputs` and `shared_workspace_recipient_schemas` (`workspace_schemas` names), `buddy_turns` (`buddies` names), `archetype_schemas` (`persona.PersonaConfirmationMode`). S10's `history_selection_schemas` import is a sixth.

  The core module then imports nothing under `tldw_Server_API.app.api` or `tldw_Server_API.app.services`.
- **R3 (re-export).** In the api module, import every name in the row back with one block:

  ```python
  # Core-owned since TASK-13950.15; re-exported for the API layer.
  from tldw_Server_API.app.core.<module> import (  # noqa: F401 - re-exported (TASK-13950.15)
      ...
  )
  ```

  `ruff --fix` would otherwise delete the block. Ruff honours a `noqa` on the first line of a multi-line import (checked with ruff 0.16.5).
  - If the api module already imports from that core module, add the names to the existing import and put the `noqa` comment there. In S1, `evaluation_schemas_unified.py` already imports `normalize_run_status` from `run_state`.
  - An api module left with no definitions becomes the `history_selection_schemas.py` shape: a docstring `"""Compatible API exports of core-owned <topic> schemas."""`, the import block, and `__all__` listing the names. In that case the `noqa` is not needed.
  - An existing `__all__` is left as it is.
  - Then remove imports that the move leaves unused in the api module (ruff F401 on that file lists them). Keep one only if `git grep -n "from tldw_Server_API.app.api.v1.schemas.<m> import"` shows an importer reading that name from the module. In S1 these are `datetime`, `Enum` and the `RunStatus` import in `evaluation_recipe_schemas.py`, and `Enum` in `synthetic_eval_schemas.py`.
- **R4 (retarget).** In each listed core statement, replace `tldw_Server_API.app.api.v1.schemas.<m>` with the core module path. Names, aliases, `try` blocks, indentation and line count stay the same.

  Leave import order alone, even where ruff then reports I001. Re-sorting would shift lines, and `unified_pipeline.py` is line-keyed in `scope_predicate_baseline.txt`; ruff is non-blocking in CI.

  A statement whose names now live in two core modules splits into two statements.
- **R5 (green).** Run V1a; every test passes. Then run `$PY -m tldw_Server_API.tests.lint.test_layering_contract --tighten`. `git diff -- tldw_Server_API/tests/lint/layering_baseline.txt` must equal the R1 edit exactly (`--tighten` changed nothing). Record the report's `largest package SCC` line in the PR body.
- **R6 (verify).** Create the base worktree once, run V1-V6, and remove the worktree:

  ```bash
  git worktree add --detach "$TMPDIR/t1315-base" "$(git merge-base origin/dev HEAD)"
  # V1 V2 V3 V4 V5 V5b V6
  git worktree remove --force "$TMPDIR/t1315-base"
  ```

  Each V must give its stated pass. The PR body records three things:
  - V3's two sha256 values and `IDENTICAL`;
  - the per-model digest lines that V3 prints for the slice's members of Review Focus 2's set (ruling 1);
  - V5's two summary lines.
- **R7 (commit).** One commit per slice: `refactor(<context>): core owns the <context> models api/v1/schemas re-exports (TASK-13950.15 S<n>)`.

Verification commands. Run them from the slice worktree root. `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`; `B="$TMPDIR/t1315-base"`.

- **V1a.** Fast loop:

  ```bash
  $PY -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1315-v1a" \
    tldw_Server_API/tests/unit/test_core_owned_schema_reexports.py \
    tldw_Server_API/tests/lint/test_layering_contract.py tldw_Server_API/tests/lint/test_core_to_api_import_boundary.py
  ```

  Pass: `N passed`, with no failures.
- **V1.** Both backend-required ratchet steps, run exactly as the workflow on the branch defines them. This picks up ratchets appended later, such as #3331's and `.14`'s:

  ```bash
  PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" $PY - <<'PY'
  import os, subprocess, yaml
  steps = yaml.safe_load(open(".github/workflows/backend-required.yml"))["jobs"]["backend-required"]["steps"]
  for name in ("Enforce tenant isolation ratchets", "Enforce CI contracts and code ratchets"):
      step = next(s for s in steps if s.get("name") == name)
      env = {**os.environ, **{k: str(v) for k, v in step.get("env", {}).items()}}
      env.update(BACKLOG_TASK_FORMAT_BASE="origin/dev", PYTEST_ADDOPTS=f"-p no:cacheprovider --basetemp={os.environ['TMPDIR']}/t1315-v1")
      print(f"== {name}", flush=True)
      subprocess.run(["bash", "-c", step["run"]], env=env, check=True)
  PY
  ```

  Pass: both `==` headers print, each pytest run ends in `passed`, and the script exits 0. The runner takes about 3 minutes.

  On the measured merge, 3 tests fail for reasons unrelated to this task, and they fail the same way on the base:
  - `function_size_baseline.json`: `api.v1.endpoints.chat:create_chat_completion` is 2870 lines against a ceiling of 2865 (two tests);
  - `charactersragdb_surface_baseline.json`: `delegated_methods` is 306 against 302.

  Both are #3334 seeds that dev has outgrown, and #3334 re-measures them before it merges. A failure that is also present on the slice's merge base is recorded in the PR, not fixed here.
- **V2.** OpenAPI contract gate (the AC #3 snapshot):

  ```bash
  make openapi-drift-check VENV_PYTHON=$PY && git diff --exit-code origin/dev -- apps/packages/api-types/
  ```

  Pass: `[openapi-export] OK — OpenAPI fingerprint matches the checked-in snapshot.` and `[openapi-export] OK — committed path union matches the canonical contract.` on stderr, and no diff.
- **V3.** Raw OpenAPI document and moved-definition shapes, against the merge base. Write `$TMPDIR/t1315_shapes.py` once:

  ```python
  """Dump the raw OpenAPI document and the shape of every CORE_OWNED name (TASK-13950.15)."""
  import hashlib, importlib, importlib.util, inspect, json, sys
  from pathlib import Path
  sys.path[:0] = [".", "Helper_Scripts"]
  from pydantic import TypeAdapter
  import export_openapi_schema as exp
  NOT_IN_OPENAPI = {  # Review Focus 2: shapes the fingerprint cannot see
      "SyntheticEvalDraftSampleCreate", "SyntheticEvalPromotionCreate", "SyntheticEvalReviewActionCreate",
      "WorkspaceOperationBase", "_RecipientModel", "AddMediaForm", "AudioVideoOptions", "ChunkingOptions",
      "PdfOptions", "AlignmentPayload", "AlignmentWord", "ChapterPreview", "ResearchArtifactManifestEntry",
      "ResearchCheckpointSummary", "ResearchRunSnapshotResponse", "BuddyModel", "BaseMessage",
  }
  spec = importlib.util.spec_from_file_location("core_owned", sys.argv[1])
  table = importlib.util.module_from_spec(spec); spec.loader.exec_module(table)
  doc = {"openapi": exp._load_schema(), "shapes": {}}
  for api_module, _core, names in table.CORE_OWNED:
      module = importlib.import_module(f"{table.API}.{api_module}")
      for name in names:
          obj = getattr(module, name)
          if inspect.isfunction(obj):
              shape = inspect.getsource(obj)
          else:
              try:
                  shape = {m: TypeAdapter(obj).json_schema(mode=m) for m in ("validation", "serialization")}
              except Exception:  # constants and types pydantic cannot schema
                  shape = obj.__qualname__ if isinstance(obj, type) else repr(obj)
          doc["shapes"][f"{api_module}.{name}"] = shape
  Path(sys.argv[2]).write_text(json.dumps(doc, default=str) + "\n", encoding="utf-8")  # unsorted: raw order counts
  Path(sys.argv[2] + ".digests").write_text("".join(
      f"{key} {hashlib.sha256(json.dumps(shape, sort_keys=True, default=str).encode()).hexdigest()}\n"
      for key, shape in doc["shapes"].items() if key.rsplit(".", 1)[1] in NOT_IN_OPENAPI), encoding="utf-8")
  ```

  Then run it on the base and on the branch:

  ```bash
  T="$PWD/tldw_Server_API/tests/unit/test_core_owned_schema_reexports.py"
  (cd "$B" && PYTHONHASHSEED=0 $PY "$TMPDIR/t1315_shapes.py" "$T" "$TMPDIR/t1315-base.json")
  PYTHONHASHSEED=0 $PY "$TMPDIR/t1315_shapes.py" "$T" "$TMPDIR/t1315-head.json"
  cmp "$TMPDIR/t1315-base.json" "$TMPDIR/t1315-head.json" && echo IDENTICAL
  shasum -a 256 "$TMPDIR/t1315-base.json" "$TMPDIR/t1315-head.json"
  diff "$TMPDIR/t1315-base.json.digests" "$TMPDIR/t1315-head.json.digests" && cat "$TMPDIR/t1315-head.json.digests"
  ```

  Pass: `IDENTICAL`, two equal sha256 values, and an empty `diff`. Paste the printed digest lines into the PR body; there are none for S2, S3, S5, S6 and S9. Two runs on one tree were measured byte-identical at 6.1 MB. The base resolves every name through the old api module, so the comparison is old definition against moved definition.
- **V4.** Fresh-interpreter import of every api and core module in `CORE_OWNED`:

  ```bash
  $PY - <<'PY'
  import importlib.util, os, subprocess, sys
  spec = importlib.util.spec_from_file_location("t", "tldw_Server_API/tests/unit/test_core_owned_schema_reexports.py")
  t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
  mods = sorted({f"{t.API}.{a}" for a, _, _ in t.CORE_OWNED} | {f"{t.CORE}.{c}" for _, c, _ in t.CORE_OWNED})
  bad = [m for m in mods if subprocess.run([sys.executable, "-c", f"import {m}"], env={**os.environ, "PYTHONPATH": "."}, capture_output=True).returncode]
  print("FAILED:", bad) if bad else print(f"OK {len(mods)} modules")
  PY
  ```

  Pass: `OK <n> modules`.
- **V5.** Every test file that imports a module the slice touched, on the branch and on the base, compared:

  ```bash
  bash <<'SH'
  set -u; PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python; B="$TMPDIR/t1315-base"
  REGEX='<slice regex>'
  run() { (cd "$1" && git grep -l -P "$REGEX" -- 'tldw_Server_API/tests/*.py' 'tldw_Server_API/app/core/*/tests/*.py' \
    | xargs "$PY" -m pytest -q -rfE -p no:cacheprovider --basetemp "$TMPDIR/t1315-v5-$2" > "$TMPDIR/t1315-v5-$2.txt" 2>&1); }
  run "$PWD" head; run "$B" base
  for s in head base; do
    grep -E '^(FAILED|ERROR) ' "$TMPDIR/t1315-v5-$s.txt" | sed 's/ - .*//' | sort -u > "$TMPDIR/t1315-v5-$s.fails"
    tail -1 "$TMPDIR/t1315-v5-$s.txt"
  done
  comm -23 "$TMPDIR/t1315-v5-head.fails" "$TMPDIR/t1315-v5-base.fails"
  SH
  ```

  Pass: `comm` prints nothing, which means no failure or error on the branch that the base lacks. The two summary lines also show the same `skipped` count.

  S1's selection is 140 files and 1,870 items, about 36 minutes per tree, so run the two trees concurrently if time matters. On the measured merge, the same 8 order-dependent failures occur on both trees: 7 in `tests/Evaluations/test_embeddings_abtest_*` and `Usage/test_quota_evals_chatbooks.py::test_resolve_job_limits_reads_the_resolver`.
- **V5b.** Patch targets that name a moved definition through the old path.

  First, string targets. This form catches `patch`, `patch.dict`, multi-line `monkeypatch.setattr` and `import_module` strings:

  ```bash
  git grep -n -P '["'"'"']tldw_Server_API\.app\.api\.v1\.schemas\.(<slice api modules>)\.(<slice names>)\b' -- '*.py'
  ```

  Pass: no output. As a positive control, the same command with `chat_request_schemas` and `API_KEYS|get_api_keys` finds 16 lines on the measured merge.

  Second, object-form targets:

  ```bash
  git grep -n -P 'v1\.schemas import (<slice api modules>)\b|import tldw_Server_API\.app\.api\.v1\.schemas\.(<slice api modules>) as' -- 'tldw_Server_API/tests/*.py'
  ```

  Pass: every hit is one of the known module-object imports, and none patches a moved name. The known imports are:
  - `tests/TTS/test_tts_module.py:245` (`audio_schemas`, an import check only);
  - `tests/Workspaces/test_workspace_source_saved_views_api.py:25` (`workspace_schemas`; it reads the `WorkspaceSourceSavedView*` models, which stay in api);
  - `tests/Sync/test_sync_v2_models.py:5` (`sync_v2_models`);
  - 11 statements in 9 files that import `chat_request_schemas` as a module to patch `.13` C1's `API_KEYS`, `get_api_keys` and `load_and_log_configs`.
- **V6.** Type check and Bandit on the slice's changed and new app files. The list comes from the working tree plus untracked files, so it works before R7 commits. Write `$TMPDIR/t1315_mypy.py` once:

  ```python
  """V6: mypy errors per moved group must not grow (TASK-13950.15). Args: table, head output, base output."""
  import collections, importlib.util, re, sys
  spec = importlib.util.spec_from_file_location("t", sys.argv[1])
  t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
  def counts(path):
      found = (re.match(r"(\S+\.py):\d+: error:", line) for line in open(path, encoding="utf-8"))
      return collections.Counter(m[1] for m in found if m)
  head, base = counts(sys.argv[2]), counts(sys.argv[3])
  parent = {}
  def find(f):
      while parent.setdefault(f, f) != f:
          f = parent[f]
      return f
  for api_module, core_module, _ in t.CORE_OWNED:  # a moved definition's errors travel within its group
      parent[find(f"tldw_Server_API/app/api/v1/schemas/{api_module}.py")] = find(f"tldw_Server_API/app/core/{core_module.replace('.', '/')}.py")
  groups = collections.defaultdict(set)
  for f in set(head) | set(base) | set(parent):
      groups[find(f)].add(f)
  grew = [(sorted(g), sum(head[f] for f in g), sum(base[f] for f in g)) for g in groups.values()]
  grew = [x for x in grew if x[1] > x[2]]
  print("\n".join(f"GREW {h} > {b}: {g}" for g, h, b in grew) or f"OK {sum(head.values())} head / {sum(base.values())} base errors")
  ```

  Then run it:

  ```bash
  bash <<'SH'
  set -u; PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python; B="$TMPDIR/t1315-base"
  { git diff --name-only --diff-filter=d "$(git merge-base origin/dev HEAD)" -- 'tldw_Server_API/*.py'
    git ls-files --others --exclude-standard -- 'tldw_Server_API/*.py'; } | sort -u > "$TMPDIR/t1315-changed.txt"
  uvx mypy --python-executable "$PY" --follow-imports=silent --ignore-missing-imports $(cat "$TMPDIR/t1315-changed.txt") > "$TMPDIR/t1315-mypy-head.txt"
  (cd "$B" && uvx mypy --python-executable "$PY" --follow-imports=silent --ignore-missing-imports \
    $(for f in $(cat "$TMPDIR/t1315-changed.txt"); do [ -f "$f" ] && echo "$f"; done) > "$TMPDIR/t1315-mypy-base.txt")
  "$PY" "$TMPDIR/t1315_mypy.py" tldw_Server_API/tests/unit/test_core_owned_schema_reexports.py "$TMPDIR/t1315-mypy-head.txt" "$TMPDIR/t1315-mypy-base.txt"
  uvx bandit -q -ll $(grep '/app/' "$TMPDIR/t1315-changed.txt")
  SH
  ```

  Pass: `OK <n> head / <m> base errors`, and Bandit reports no medium or high issue (`-ll`; retargeted files already carry LOW findings such as B404 and B311, so a low-severity gate could never pass from S3 on).

  Groups join each api module with the core module or modules its names moved to. Moved code keeps errors it already had, so a group's head total must not exceed its base total. Every other changed file is compared to itself. Known moved errors, all `unused-ignore` except where noted:
  - S2: 7 in `rag_schemas_unified`, one of them the `no-redef` duplicate that Task 2 files a follow-up for; and 2 in `_compat` into `core/Utils/pydantic_compat.py`;
  - S7: 1 `prop-decorator` in `AddMediaForm`.

  The S1 dry run measured 206 errors on both sides. CI runs mypy with `continue-on-error`, so this step is the gate.

No moved code holds a function in `tldw_Server_API/tests/lint/function_size_baseline.json` (`grep -c "api.v1.schemas" ...` is 0), so no "moved from" size reasons are needed. Retargets keep line counts, so the frozen `unified_rag_pipeline` (7,063 lines), `unified_batch_pipeline` and `chat_service` functions do not grow.

---

### Task 1: S1, Evaluations (identity test, recipe/abtest/synthetic models, RunStatus)

**Files:**
- Create: `tldw_Server_API/tests/unit/test_core_owned_schema_reexports.py`
- Create: `tldw_Server_API/app/core/Evaluations/evaluation_recipe_schemas.py`, `tldw_Server_API/app/core/Evaluations/embeddings_abtest_schemas.py`, `tldw_Server_API/app/core/Evaluations/synthetic_eval_schemas.py`
- Modify: `tldw_Server_API/app/core/Evaluations/run_state.py` (append `RunStatus`)
- Modify (R3): `tldw_Server_API/app/api/v1/schemas/evaluation_schemas_unified.py`, `evaluation_recipe_schemas.py`, `embeddings_abtest_schemas.py`, `synthetic_eval_schemas.py`
- Modify (R4): the 21 `core.Evaluations` statements marked S1 in the starting-point table, plus `tldw_Server_API/app/core/DB_Management/Evaluations_DB.py:23` and `:29`
- Modify: `tldw_Server_API/tests/lint/layering_baseline.txt`
- Modify: `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py` (one assertion)

**Interfaces:**
- Produces: `CORE_OWNED: tuple[tuple[str, str, tuple[str, ...]], ...]`, `API = "tldw_Server_API.app.api.v1.schemas"` and `CORE = "tldw_Server_API.app.core"` in the test module; later slices append rows. It also produces these definitions:
  - `core.Evaluations.run_state.RunStatus(str, Enum)`;
  - `core.Evaluations.evaluation_recipe_schemas`: `ConfidenceSummary`, `RecipeManifest`, `RecipeRunRecord`, `RecommendationSlot`, `ReviewState`;
  - `core.Evaluations.embeddings_abtest_schemas`: `ABTestArm`, `ABTestChunking`, `ABTestCleanupPolicy`, `ABTestLimits`, `ABTestQuery`, `ABTestReRanker`, `ABTestRetrieval`, `EmbeddingsABTestConfig`;
  - `core.Evaluations.synthetic_eval_schemas`: `SyntheticEvalDraftSampleCreate`, `SyntheticEvalPromotionCreate`, `SyntheticEvalProvenance`, `SyntheticEvalReviewActionCreate`, `SyntheticEvalReviewActionType`, `SyntheticEvalReviewState`.

`RunStatus` goes into the existing `run_state.py`, because that module already holds the run-status vocabulary (`TERMINAL_RUN_STATUSES`, `normalize_run_status`). Add `from enum import Enum`. `core.Evaluations.evaluation_recipe_schemas` imports `RunStatus` from there.

The new modules import only stdlib (`datetime`, `enum`, `typing`), `pydantic` and `run_state`, and `core/Evaluations/__init__.py` is empty. So the S1 bump adds no DB-reaching import. Step 1's second test pins that directly, without depending on `.16`'s ratchet landing first.

- [ ] **Step 1: Write the tests (red).**

  The identity test file. Its docstring is the module's documentation of the pattern.

```python
"""api/v1/schemas re-exports the models core owns (TASK-13950.15).

Core must not import api/v1/schemas (tests/lint/test_layering_contract.py), so the models core
reads are defined in core and re-exported here, as api/v1/schemas/history_selection_schemas.py
does. A row names an api module, the core module that defines the names, and every name that
moved. The re-export must be the same object, so FastAPI, isinstance checks and the OpenAPI
components see one definition.
"""

from __future__ import annotations

import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

API = "tldw_Server_API.app.api.v1.schemas"
CORE = "tldw_Server_API.app.core"
REPO_ROOT = Path(__file__).resolve().parents[3]

# (module under API, module under CORE, the names that moved). Slices append rows in order.
CORE_OWNED: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("evaluation_schemas_unified", "Evaluations.run_state", ("RunStatus",)),
    ("evaluation_recipe_schemas", "Evaluations.evaluation_recipe_schemas", ("ConfidenceSummary", "RecipeManifest", "RecipeRunRecord", "RecommendationSlot", "ReviewState")),
    ("embeddings_abtest_schemas", "Evaluations.embeddings_abtest_schemas", ("ABTestArm", "ABTestChunking", "ABTestCleanupPolicy", "ABTestLimits", "ABTestQuery", "ABTestReRanker", "ABTestRetrieval", "EmbeddingsABTestConfig")),
    ("synthetic_eval_schemas", "Evaluations.synthetic_eval_schemas", ("SyntheticEvalDraftSampleCreate", "SyntheticEvalPromotionCreate", "SyntheticEvalProvenance", "SyntheticEvalReviewActionCreate", "SyntheticEvalReviewActionType", "SyntheticEvalReviewState")),
)


@pytest.mark.parametrize(("api_module", "core_module", "names"), CORE_OWNED, ids=[f"{a}->{c}" for a, c, _ in CORE_OWNED])
def test_api_schema_module_re_exports_the_core_owned_definition(
    api_module: str, core_module: str, names: tuple[str, ...]
) -> None:
    api = importlib.import_module(f"{API}.{api_module}")
    core = importlib.import_module(f"{CORE}.{core_module}")
    for name in names:
        assert getattr(api, name) is getattr(core, name), f"{API}.{api_module}.{name} is not {core.__name__}.{name}"
        obj = getattr(core, name)
        if inspect.isclass(obj) or inspect.isfunction(obj):
            assert obj.__module__ == core.__name__, f"{name} is defined in {obj.__module__}"


def test_the_evaluations_models_load_no_db_management_code() -> None:
    """S1 raises core.DB_Management -> core.Evaluations to 3; the raise must stay a model-only import."""
    mods = ("run_state", "evaluation_recipe_schemas", "embeddings_abtest_schemas", "synthetic_eval_schemas")
    code = "import importlib, sys\n" + "".join(f"importlib.import_module('{CORE}.Evaluations.{m}')\n" for m in mods)
    code += "print('\\n'.join(sorted(sys.modules)))"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    run = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert [m for m in run.stdout.split() if ".core.DB_Management" in m] == []
```

  A prototype on the measured merge confirmed both directions:
  - The DB test fails red with `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Evaluations.evaluation_recipe_schemas'` in its assertion message.
  - Pointed at `core.DB_Management.Evaluations_DB`, the same probe does list `DB_Management` modules, so the detector works.

  In `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`, add this test next to `test_backend_required_enforces_layering_ratchets`. It uses that file's existing `REPO_ROOT` and `yaml`.

```python
@pytest.mark.unit
def test_backend_required_unit_smoke_runs_the_unit_tests() -> None:
    """TASK-13950.15: the core-owned schema re-export test lives in tests/unit; this step is its required gate."""
    import shlex

    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/backend-required.yml").read_text())
    steps = workflow["jobs"]["backend-required"]["steps"]
    gate = next(step for step in steps if step.get("name") == "Backend unit smoke")
    assert not gate.get("continue-on-error", False)  # nosec B101 - regression assertion
    assert "tldw_Server_API/tests/unit" in shlex.split(gate["run"].replace("\\\n", " "), comments=True)  # nosec B101
```

  It passes on arrival, because it pins an existing gate. The dry run confirmed the identity test runs in that step under its environment (4 passed).

  Then edit `layering_baseline.txt` (R1). Values are on the measured merge:
  - `core.DB_Management -> api.v1.schemas: 3` becomes `core.DB_Management -> api.v1.schemas: 1` (−2);
  - `core.DB_Management -> core.Evaluations: 1` becomes `core.DB_Management -> core.Evaluations: 3  # bump: Evaluations_DB reads RunStatus and the recipe-run models that moved from api.v1.schemas into core.Evaluations; core.DB_Management -> api.v1.schemas falls by the same 2 (TASK-13950.15)` (+2);
  - `core.Evaluations -> api.v1.schemas: 22` becomes `core.Evaluations -> api.v1.schemas: 1` (−21).

  The bump is the only raise in this task. The phase exit allows `DB_Management -> domain` entries that target model modules (TASK-13950.16 AC #5). This slice also completes the `Evaluations_DB` half of `.16` AC #3.

- [ ] **Step 2: Run V1a.** Expected: 7 failed and 17 passed.
  - Three identity cases fail with `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Evaluations.evaluation_recipe_schemas'` (and the abtest and synthetic modules).
  - The fourth identity case fails with `AttributeError: module 'tldw_Server_API.app.core.Evaluations.run_state' has no attribute 'RunStatus'`.
  - `test_the_evaluations_models_load_no_db_management_code` fails with the `ModuleNotFoundError` above in its message.
  - `test_imports_follow_the_layering_contract` and `test_core_to_api_imports_only_fall` fail, with these lines between them:
    - `core.Evaluations -> api.v1.schemas: 22 statements > baseline 1`;
    - `core.DB_Management -> api.v1.schemas: 3 statements > baseline 1`;
    - `core.DB_Management -> core.Evaluations: stale, baseline 3 but 1 measured`.

  The dry run matched every identity and contract line.
- [ ] **Step 3: Apply R2-R4.**
  - Recipe models move from `evaluation_recipe_schemas.py` (5 names, about 62 of 230 lines).
  - `RunStatus` moves from `evaluation_schemas_unified.py` (7 lines). Merge its R3 import into that file's existing `from ...run_state import normalize_run_status`. That module's colliding `EvaluationResponse`, `WebhookTestRequest` and `WebhookTestResponse` stay put; moving them would rename their components.
  - Abtest moves 8 names (49 of 153 lines), and synthetic moves 6 names (59 of 216 lines).
  - Remove the now-unused api imports named in R3, and retarget the 23 statements.
- [ ] **Step 4: Run R5.** Expected: V1a passes with 24 passed. `--tighten` leaves the baseline diff equal to Step 1's three lines and reports 662 statements (683 − 23 + 2).
- [ ] **Step 5: Run R6.**
  - V5 regex: `schemas\.(evaluation_recipe_schemas|evaluation_schemas_unified|embeddings_abtest_schemas|synthetic_eval_schemas)|core\.Evaluations|Evaluations_DB` (140 test files on the measured merge).
  - V5b: use the api modules `evaluation_recipe_schemas|evaluation_schemas_unified|embeddings_abtest_schemas|synthetic_eval_schemas` with the 20 names above.
  - V3: prints digests for the three `SyntheticEval*Create` models.
  - If `.16` PR A is on the base, `tldw_Server_API/tests/lint/test_db_management_logic_imports.py` passes with `db_management_logic_imports.txt` unchanged.
  - Each V gives its stated pass.
- [ ] **Step 6: Commit (R7).**

  ```bash
  git add tldw_Server_API/tests/unit/test_core_owned_schema_reexports.py tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py \
    tldw_Server_API/app/core/Evaluations tldw_Server_API/app/core/DB_Management/Evaluations_DB.py tldw_Server_API/app/api/v1/schemas \
    tldw_Server_API/tests/lint/layering_baseline.txt
  ```

  In the PR body, say AC #3 is enforced by backend-required's existing "OpenAPI contract drift gate" (V2), so no new snapshot test is added.

### Task 2: S2, RAG and the compat `Field`

**Files:**
- Create: `tldw_Server_API/app/core/RAG/rag_schemas_unified.py`
- Modify: `tldw_Server_API/app/core/Utils/pydantic_compat.py`. Append `Field` and its `_IS_PYDANTIC_V2` probe verbatim. The file already has a per-file `BLE001` ignore (`pyproject.toml:1311`), so no `noqa` is needed.
- Modify (R3):
  - `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`;
  - `tldw_Server_API/app/api/v1/schemas/_compat.py`, which becomes a shim of `Field` with `__all__ = ["Field"]`. Its importers are unchanged: `feedback_schemas` and `rag_schemas_unified` import it relatively; Collections feeds, Collections WebSub and reading import it absolutely.
- Modify (R4). Locate each import by its quoted text, because #3331 may shift `unified_pipeline.py`:
  - `core/RAG/rag_service/response_mapping.py:7`, `source_health.py:8`, `transport.py:13`;
  - `unified_pipeline.py:2673` and `:9151`, each `from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGResponse`, edited in place with the same line count. The file is in `Helper_Scripts/ci/scope_predicate_baseline.txt` (entry `:3889`); if a line shifts, run `$PY Helper_Scripts/ci/scope_predicate_ratchet.py --write-baseline`;
  - `core/Evaluations/recipes/rag_retrieval_tuning_execution.py:11`;
  - `core/MCP_unified/modules/implementations/rag_module.py:12`;
  - `core/Scheduled_Tasks/recurring_question_rag_adapter.py:9`.
- Modify: `pyproject.toml`. Next to `"tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py" = ["BLE001"]`, add `"tldw_Server_API/app/core/RAG/rag_schemas_unified.py" = ["BLE001"]`.
- Modify (docs, DoD #3). Then run `bash Helper_Scripts/refresh_docs_published.sh` and stage its `Docs/Published` diff.
  - `Docs/Architecture/rag.md:12`: the Schemas cell becomes `` `app/core/RAG/rag_schemas_unified.py`, re-exported by `app/api/v1/schemas/rag_schemas_unified.py` ``.
  - `Docs/Code_Documentation/RAG-Developer-Guide.md:12` (mirrored): it becomes `` - Public schemas: defined in `tldw_Server_API/app/core/RAG/rag_schemas_unified.py`, re-exported from `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`. ``
- Modify: `tldw_Server_API/tests/lint/layering_baseline.txt`, `tldw_Server_API/tests/unit/test_core_owned_schema_reexports.py`.

**Interfaces:**
- Consumes: Task 1's `CORE_OWNED`.
- Produces:
  - `core.RAG.rag_schemas_unified`: `KnowledgeSourceEmbeddingStatus`, `KnowledgeSourceHealthEntry`, `KnowledgeSourceHealthResponse`, `KnowledgeSourceIndexStatus`, `UnifiedBatchRequest`, `UnifiedRAGRequest`, `UnifiedRAGResponse`;
  - `core.Utils.pydantic_compat.Field(*args: Any, **kwargs: Any)`.

Two module-level blocks move with the classes:
- the Pydantic v1-fallback `try` (`field_validator`, `model_validator`);
- the `RAG_CONTEXTUAL_DEFAULTS` settings `try` (`_settings`, `_ctx_defaults`, `_DEF_SIB_WIN`, `_DEF_INC_PARENT`, `_DEF_PARENT_MAX_TOK`, `_DEF_INC_SIB`, `_DEF_ENH_CHUNK`, `_DEF_FTS_LVL`).

About 2,145 of the module's 2,264 lines move. `UnifiedBatchResponse` and `ImplicitFeedbackEvent` stay in api. If they read a moved private, import it back as well. If they still use `field_validator`, `model_validator` or `Field`, the api module keeps its own copy of the Pydantic-fallback `try` block and its `from ._compat import Field`. Those are imports, not definitions, so keeping a copy duplicates no definition.

`UnifiedBatchRequest` declares `adaptive_advanced_rewrites` twice (`:1946` and `:2063`; mypy `no-redef`). It moves verbatim, and a follow-up task removes the duplicate.

- [ ] **Step 1: Red (R1).** Append these rows:
  - `("rag_schemas_unified", "RAG.rag_schemas_unified", ("KnowledgeSourceEmbeddingStatus", "KnowledgeSourceHealthEntry", "KnowledgeSourceHealthResponse", "KnowledgeSourceIndexStatus", "UnifiedBatchRequest", "UnifiedRAGRequest", "UnifiedRAGResponse"))`;
  - `("_compat", "Utils.pydantic_compat", ("Field",))`.

  Baseline deltas (measured values): delete `core.Evaluations -> api.v1.schemas` (1 after S1), `core.MCP_unified -> api.v1.schemas: 1`, `core.RAG -> api.v1.schemas: 5` and `core.Scheduled_Tasks -> api.v1.schemas: 1`.

  If `.14` slice 1 is on the base, also delete the `"api/v1/schemas/rag_schemas_unified.py"` entry from `IMPORT_TIME_CONFIG_ALLOWLIST` in `tldw_Server_API/tests/lint/test_schemas_config_boundary.py`. The import-time `settings` read moves with `UnifiedRAGRequest` into `core/RAG/rag_schemas_unified.py`. `.14`'s guard does not look there, and config reads are allowed there, but the read is still made at import. Step 6 records that in TASK-13950.21, which that entry's reason names.
- [ ] **Step 2: Run V1a.** Expected: the `rag_schemas_unified` row fails with `ModuleNotFoundError`, the `_compat` row with `AttributeError`, and the two contract tests report the 4 pairs as `not in the baseline`.
- [ ] **Step 3: Apply R2-R4 and the doc edits.**
- [ ] **Step 4: Run R5.** Expected: V1a passes and `--tighten` changes nothing.

  Then run this. Expected: `False`.

  ```bash
  $PY -c "import sys, tldw_Server_API.app.core.RAG; print('tldw_Server_API.app.api.v1.schemas.rag_schemas_unified' in sys.modules)"
  ```

  Importing `core.RAG` must not load the api module, so the re-export can never re-enter a half-initialized `core.RAG`. On the measured merge, `core/RAG/__init__.py`'s import closure (540 app modules) references the schema module only through `unified_pipeline`'s function-level imports.
- [ ] **Step 5: Run R6.**
  - V5 regex: `schemas\.(rag_schemas_unified|_compat|collections_feeds_schemas|collections_websub_schemas|feedback_schemas|reading_schemas)|core\.RAG|core\.Utils\.pydantic_compat|rag_retrieval_tuning_execution|implementations\.rag_module|recurring_question_rag_adapter`. The in-app pathspec picks up the `core/MCP_unified/tests` files over `rag_module`.
  - V5b uses the 8 names.
  - V1 includes `tests/lint/test_scope_predicate_ratchet.py`, which must pass without a baseline rewrite.
  - Also run `$PY -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1315-docs" tldw_Server_API/tests/Docs`. Expected: no failures.
  - Cold-importing the api module now costs about 4 s, because it runs `core/RAG/__init__.py`. The app imports `core.RAG` anyway; note it in the PR.
- [ ] **Step 6: Backlog notes.** Pick the follow-up's id with the AGENTS.md Backlog.md id rule.

  ```bash
  PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13950.21 --append-notes "TASK-13950.15 S2 moved the import-time RAG_CONTEXTUAL_DEFAULTS / RAG_DEFAULT_FTS_LEVEL settings read and the _DEF_* Field defaults from api/v1/schemas/rag_schemas_unified.py to core/RAG/rag_schemas_unified.py, verbatim; they are still read at import."
  PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Remove the duplicate UnifiedBatchRequest.adaptive_advanced_rewrites field" --id TASK-13950.<next free n> -d "core/RAG/rag_schemas_unified.py declares adaptive_advanced_rewrites twice in UnifiedBatchRequest (mypy no-redef); TASK-13950.15 S2 moved it verbatim." --ac "UnifiedBatchRequest declares adaptive_advanced_rewrites once, and make openapi-drift-check passes without regenerating the fingerprint"
  git add -A backlog/tasks
  ```

- [ ] **Step 7: Commit (R7).**

### Task 3: S3, speech request and speech-chat models

**Files:**
- Create: `tldw_Server_API/app/core/TTS/speech_schemas.py`, `tldw_Server_API/app/core/Streaming/speech_chat_schemas.py`
- Modify (R3): `tldw_Server_API/app/api/v1/schemas/audio_schemas.py`
- Modify (R4):
  - `core/Audio/Realtime/default_pipeline.py:14`, `core/Audio_Studio/providers/speech.py:8`, `core/Persona/live_tts.py:10`, `core/Research_Workspace/output_jobs.py:18`;
  - `core/Streaming/speech_chat_service.py:34`, split into a `core.TTS.speech_schemas` import of `OpenAISpeechRequest` and a `core.Streaming.speech_chat_schemas` import of the `SpeechChat*` names. The import sits outside the size-frozen `run_speech_chat_turn`, and the file is not in `scope_predicate_baseline.txt`;
  - `core/TTS/realtime_session.py:14`, `core/TTS/tts_jobs_worker.py:20`, `core/TTS/tts_service_v2.py:25`;
  - `core/Workflows/adapters/audio/multi_voice_tts.py:82`;
  - `core/Workflows/adapters/audio/tts.py:13`, keeping its `try/except ImportError`.
- Modify: `layering_baseline.txt`, `test_core_owned_schema_reexports.py`

**Interfaces:**
- Produces:
  - `core.TTS.speech_schemas`: `NormalizationOptions`, `OpenAISpeechRequest` (138 lines);
  - `core.Streaming.speech_chat_schemas`: `SpeechChatLLMConfig`, `SpeechChatRequest`, `SpeechChatResponse`, `SpeechChatSTTConfig`, `SpeechChatTTSConfig`, `SpeechChatTiming`, `SpeechChatTokenUsage` (172 lines).

The two closures are disjoint. Every consumer -> `core.TTS` edge already exists (Audio, Audio_Studio, Persona, Research_Workspace, Streaming, Workflows), so no new domain edge appears.

- [ ] **Step 1: Red (R1).** Append these rows:
  - `("audio_schemas", "TTS.speech_schemas", ("NormalizationOptions", "OpenAISpeechRequest"))`;
  - `("audio_schemas", "Streaming.speech_chat_schemas", ("SpeechChatLLMConfig", "SpeechChatRequest", "SpeechChatResponse", "SpeechChatSTTConfig", "SpeechChatTTSConfig", "SpeechChatTiming", "SpeechChatTokenUsage"))`.

  Baseline deltas (measured values):
  - delete `core.Audio -> api.v1.schemas: 1`, `core.Audio_Studio -> api.v1.schemas: 1`, `core.TTS -> api.v1.schemas: 3`;
  - `core.Persona` −1 (2 to 1), `core.Research_Workspace` −1 (5 to 4), `core.Streaming` −1 (2 to 1), `core.Workflows` −2 (3 to 1).
- [ ] **Step 2: Run V1a.** Expected: 2 rows fail with `ModuleNotFoundError`, and the contract tests fail on the 7 pairs.
- [ ] **Step 3: Apply R2-R4.**
- [ ] **Step 4: Run R5.**
- [ ] **Step 5: Run R6.** V5 regex: `schemas\.audio_schemas|core\.(TTS|Streaming|Audio|Audio_Studio)\b|Persona\.live_tts|Research_Workspace\.output_jobs|Workflows\.adapters\.audio`. `services/audiobook_jobs_worker.py` and `services/outputs_service.py` keep importing `OpenAISpeechRequest` from the api module (`.37`).
- [ ] **Step 6: Commit (R7).**

### Task 4: S4, Workspaces, Research_Workspace and Sharing

**Files:**
- Create the following, all under `tldw_Server_API/app/`: `core/Research_Workspace/research_workspace_capabilities.py`, `core/Research_Workspace/research_workspace_outputs.py`, `core/Workspaces/workspace_schemas.py`, `core/Workspaces/chat_session_schemas.py`, `core/Sharing/shared_workspace_recipient_schemas.py`.
- Modify (R3):
  - `api/v1/schemas/research_workspace_capabilities.py` and `research_workspace_outputs.py` become whole-module shims with `__all__`, because every definition moves;
  - `api/v1/schemas/workspace_schemas.py`;
  - `shared_workspace_recipient_schemas.py`, which keeps its `__all__`;
  - `chat_session_schemas.py`, whose colliding `MessageResponse` stays put.
- Modify (R4): `core/Research_Workspace/capabilities.py:16`, `core/Research_Workspace/output_jobs.py:20` and `:28`, `core/Workspaces/assistant_defaults.py:13` and `:14`, `core/Sharing/shared_workspace_clone_jobs_worker.py:23`, `core/Sharing/shared_workspace_clone_operations.py:16`
- Modify: `layering_baseline.txt`, `test_core_owned_schema_reexports.py`

**Interfaces:**
- Produces:
  - `core.Workspaces.workspace_schemas`: the 12 names in Step 1's third row (133 of 1,692 lines);
  - `core.Workspaces.chat_session_schemas`: `ALLOWED_ASSISTANT_KINDS`, `ALLOWED_PERSONA_MEMORY_MODES`, `ChatSessionCreate` (131 lines);
  - `core.Research_Workspace.research_workspace_capabilities` (6 names) and `core.Research_Workspace.research_workspace_outputs` (6 names), as listed in Step 1;
  - `core.Sharing.shared_workspace_recipient_schemas` (10 names).

  The outputs and sharing modules import the artifact and operation models from `core.Workspaces.workspace_schemas` (R2's retarget rule).

`ChatSessionCreate` reads `core.Chat.assistant_startup`, `core.Chat.conversation_create_idempotency` and `core.Workspaces.chat_startup_schemas`. That is why Workspaces owns it: `core.Workspaces -> core.Chat` already exists, and a `core.Chat` home would add a new `core.Chat -> core.Workspaces` edge.

One new direct domain edge appears, `core.Research_Workspace -> core.Workspaces`. It is unratcheted, and both packages are already mutually reachable, so the SCC does not grow.

- [ ] **Step 1: Red (R1).** Append these rows:
  - `("research_workspace_capabilities", "Research_Workspace.research_workspace_capabilities", ("ResearchWorkspaceCapabilitiesResponse", "ResearchWorkspaceCapability", "ResearchWorkspaceCapabilityId", "ResearchWorkspaceCapabilityMode", "ResearchWorkspaceCapabilityStatus", "ResearchWorkspaceOverallStatus"))`;
  - `("research_workspace_outputs", "Research_Workspace.research_workspace_outputs", ("ResearchWorkspaceOutputArtifactType", "ResearchWorkspaceOutputSettings", "ResearchWorkspaceOutputStatus", "ResearchWorkspaceOutputStatusResponse", "ResearchWorkspaceOutputSubmitRequest", "ResearchWorkspaceOutputSubmitResponse"))`;
  - `("workspace_schemas", "Workspaces.workspace_schemas", ("WorkspaceArtifactRedaction", "WorkspaceArtifactResponse", "WorkspaceArtifactReviewState", "WorkspaceAssistantDefaultDegradedReason", "WorkspaceAssistantDefaults", "WorkspaceAssistantKind", "WorkspaceEffectiveAssistantDefault", "WorkspaceEffectiveAssistantDefaultSource", "WorkspaceEffectiveAssistantDefaultStatus", "WorkspaceOperationBase", "WorkspaceOperationStatus", "WorkspacePersonaMemoryMode"))`;
  - `("chat_session_schemas", "Workspaces.chat_session_schemas", ("ALLOWED_ASSISTANT_KINDS", "ALLOWED_PERSONA_MEMORY_MODES", "ChatSessionCreate"))`;
  - `("shared_workspace_recipient_schemas", "Sharing.shared_workspace_recipient_schemas", ("CloneWarningCode", "SharedWorkspaceCloneCounts", "SharedWorkspaceCloneError", "SharedWorkspaceCloneOperationResponse", "SharedWorkspaceCloneProgress", "SharedWorkspaceCloneReadiness", "SharedWorkspaceCloneResult", "SharedWorkspaceCloneWarning", "ShortCode", "_RecipientModel"))`.

  Baseline deltas (measured values):
  - `core.Research_Workspace` −3 (4 after S3 to 1; the remaining statement is `.14`'s);
  - delete `core.Sharing -> api.v1.schemas: 2` and `core.Workspaces -> api.v1.schemas: 2`.
- [ ] **Step 2: Run V1a.** Expected: 5 rows fail with `ModuleNotFoundError`, and the contract tests fail on 3 pairs.
- [ ] **Step 3: Apply R2-R4.**
- [ ] **Step 4: Run R5.**
- [ ] **Step 5: Run R6.** V5 regex: `schemas\.(research_workspace_capabilities|research_workspace_outputs|workspace_schemas|shared_workspace_recipient_schemas|chat_session_schemas)|core\.(Research_Workspace|Workspaces|Sharing)\b`. V3 prints digests for `WorkspaceOperationBase` and `_RecipientModel`.
- [ ] **Step 6: Commit (R7).**

### Task 5: S5, visual-novel assets and play setup

**Files:**
- Create: `tldw_Server_API/app/core/VN_Assets/vn_asset_schemas.py`, `tldw_Server_API/app/core/VN_Play/vn_play_schemas.py`
- Modify (R3): `api/v1/schemas/vn_asset_schemas.py`, `api/v1/schemas/vn_play_schemas.py` (keeps its `__all__`)
- Modify (R4): `core/VN_Assets/preflight.py:5`, `core/VN_Assets/service.py:14`, `core/VN_Play/setup_options.py:11` and `:15`
- Modify: `layering_baseline.txt`, `test_core_owned_schema_reexports.py`

**Interfaces:**
- Produces:
  - `core.VN_Assets.vn_asset_schemas`: `ReviewStatus`, `SlotStatus`, `VALID_REVIEW_STATUSES`, `VNAssetBulkReviewRequest`, `VNAssetCleanupRequest`, `VNAssetCleanupResponse`, `VNAssetGenerationPreflightResponse`, `VNAssetGenerationRequest`, `VNAssetGenerationStatusResponse`, `VNAssetItemResponse`, `VNAssetManifestResponse`, `VNAssetPackCreate`, `VNAssetPackResponse`, `VNAssetPackUpdate`, `VNAssetPromptPreviewRequest`, `VNAssetPromptPreviewResponse`, `VNAssetReadinessResponse`, `VNAssetReviewRequest`, `VNAssetSlotCreate`, `VNAssetSlotPreflight`, `VNAssetSlotResponse`, `VNAssetSlotUpdate` (301 of 510 lines; reads `core.VN_Assets.constants`);
  - `core.VN_Play.vn_play_schemas`: `VNPlayMode`, `VNPlaySetupAssetPackOption`, `VNPlaySetupCharacterOption`, `VNPlaySetupCompatibility`, `VNPlaySetupCompatibilityStatus`, `VNPlaySetupDefaults`, `VNPlaySetupEmptyState`, `VNPlaySetupEmptyStateScope`, `VNPlaySetupOptionsResponse`, `VNPlaySetupPagination`, `VNPlaySetupPaginationSet`, `VNPlaySetupScriptVersionOption`, `VNPlaySetupTrustLevel`, `VNPlaySetupTrustSource`, `VNPlaySetupWarning`, `VNPlaySetupWarningSeverity`, `VNPlaySetupWarningSummary` (119 of 782 lines).

- [ ] **Step 1: Red (R1).** Append two rows: `("vn_asset_schemas", "VN_Assets.vn_asset_schemas", (<the 22 names above, in that order>))` and `("vn_play_schemas", "VN_Play.vn_play_schemas", (<the 17 names above, in that order>))`. Delete `core.VN_Assets -> api.v1.schemas: 2` and `core.VN_Play -> api.v1.schemas: 2`.
- [ ] **Step 2: Run V1a.** Expected: 2 rows fail with `ModuleNotFoundError`, and the contract tests fail on 2 pairs.
- [ ] **Step 3: Apply R2-R4.**
- [ ] **Step 4: Run R5.**
- [ ] **Step 5: Run R6.** V5 regex: `schemas\.(vn_asset_schemas|vn_play_schemas)|core\.VN_(Assets|Play)\b`. V3 compares `VALID_REVIEW_STATUSES` (a set) by `repr`; `PYTHONHASHSEED=0` keeps that stable.
- [ ] **Step 6: Commit (R7).**

### Task 6: S6, flashcards, study packs, and the ChaCha OSCE import

**Files:**
- Create: `tldw_Server_API/app/core/Flashcards/flashcard_schemas.py`, `tldw_Server_API/app/core/StudyPacks/study_pack_schemas.py`
- Modify (R3): `api/v1/schemas/flashcards.py`, `api/v1/schemas/study_packs.py`
- Modify (R4): `core/Flashcards/generation.py:5`, `core/Flashcards/source_review.py:12`, `core/StudyPacks/generation_service.py:15`, `core/StudyPacks/jobs.py:11`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`, method `import_osce_quiz_entry_atomic` (`:43672-43680`; locate by name, because #3328, #3339 and #3342 touch the file)
- Test: `tldw_Server_API/tests/Quizzes/test_osce_import_v2.py` (one test)
- Modify: `layering_baseline.txt`, `test_core_owned_schema_reexports.py`

**Interfaces:**
- Produces:
  - `core.Flashcards.flashcard_schemas`: `FlashcardCardType`, `FlashcardGenerateRequest`, `FlashcardGenerationType`, `FlashcardPlanItem`;
  - `core.StudyPacks.study_pack_schemas`: `StudyPackCreateJobRequest`, `StudyPackSourceSelection`, `StudyPackSourceType`;
  - `CharactersRAGDB.import_osce_quiz_entry_atomic(self, entry: OsceQuizExportV2) -> dict[str, Any]`.

The ChaCha change is an inversion, not a move.
- The only caller, `api/v1/endpoints/quizzes.py:349`, passes an `OsceQuizExportV2` that `_QUIZ_EXPORT_V2_ENTRY_ADAPTER` (a strict `TypeAdapter`) already validated. `git grep` finds no other caller, tests included.
- `OsceQuizExportV2.model_validate(instance)` returns the instance unchanged, because the model does not set `revalidate_instances`.

The edit makes three changes:
- the annotation narrows to `entry: OsceQuizExportV2`, keeping the existing `TYPE_CHECKING` import at `:63`;
- the line `from tldw_Server_API.app.api.v1.schemas.quizzes import OsceQuizExportV2` is deleted;
- `validated = OsceQuizExportV2.model_validate(entry)` becomes `validated = entry`.

The docstring becomes `"""Rekey, downgrade, and atomically persist one portable OSCE entry the caller validated."""`.

An unvalidated `Mapping` now fails closed: `validated.stations` raises `AttributeError` before anything is written. The new test pins that.

Moving the roughly 200-line OSCE closure (`quizzes.py` plus `osce.py`) would need a new `core.Quizzes` package and a new ratcheted `DB_Management` pair. TASK-13950.37 can retarget the `TYPE_CHECKING` line if it moves the OSCE schemas.

- [ ] **Step 1: Red (R1).** Append these rows:
  - `("flashcards", "Flashcards.flashcard_schemas", ("FlashcardCardType", "FlashcardGenerateRequest", "FlashcardGenerationType", "FlashcardPlanItem"))`;
  - `("study_packs", "StudyPacks.study_pack_schemas", ("StudyPackCreateJobRequest", "StudyPackSourceSelection", "StudyPackSourceType"))`.

  Baseline deltas (measured values):
  - delete `core.DB_Management -> api.v1.schemas` (1 after S1) and `core.Flashcards -> api.v1.schemas: 2`;
  - `core.StudyPacks` −2 (3 to 1; the remaining statement is `.14`'s).

  Append to `test_osce_import_v2.py`, using its existing `quizzes_db`, `_osce_entry` and `_station_export` fixtures:

```python
def test_db_import_fails_closed_on_an_unvalidated_mapping(quizzes_db: CharactersRAGDB) -> None:
    with pytest.raises(AttributeError):
        quizzes_db.import_osce_quiz_entry_atomic(_osce_entry(_station_export()))
    for table in ("quizzes", "osce_stations"):
        assert quizzes_db.execute_query(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"] == 0  # nosec B608
```

- [ ] **Step 2: Run V1a and the new test.** Expected from V1a: 2 rows fail with `ModuleNotFoundError`, and the contract tests fail on 3 pairs. Then run:

  ```bash
  $PY -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1315-osce" "tldw_Server_API/tests/Quizzes/test_osce_import_v2.py::test_db_import_fails_closed_on_an_unvalidated_mapping"
  ```

  Expected: `Failed: DID NOT RAISE AttributeError`. Today the method validates the dict and persists it; this was checked on the measured merge.
- [ ] **Step 3: Apply R2-R4 and the ChaCha edit.**
- [ ] **Step 4: Run R5 and the new test.** Expected: V1a passes, and the new test passes.
- [ ] **Step 5: Run R6.** V5 runs `tldw_Server_API/tests/Quizzes` plus the regex `schemas\.(flashcards|study_packs)|core\.(Flashcards|StudyPacks)\b`.

  `test_osce_import_v2.py` covers the ChaCha path: the mixed rekey and downgrade, rollback of the second station, invalid-sibling isolation, and now fail-closed. Its Postgres case must also pass in the Postgres shard:

  ```bash
  TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1315-pg" \
    "tldw_Server_API/tests/Quizzes/test_osce_import_v2.py::test_postgres_v2_import_ignores_owner_and_provenance_claims"
  ```

  Pass: `1 passed`, not skipped.
- [ ] **Step 6: Commit (R7).**

### Task 7: S7, ingestion, collections, claims and audiobooks

**Files:**
- Create the following, all under `tldw_Server_API/app/`: `core/Ingestion_Media_Processing/document_upload_processing.py`, `core/Ingestion_Media_Processing/media_request_models.py`, `core/Claims_Extraction/claims_schemas.py`, `core/Audiobooks/audiobook_schemas.py`.
- Modify (R3):
  - `api/v1/schemas/document_upload_processing.py`;
  - `api/v1/schemas/media_request_models.py`, where the colliding `SearchRequest` stays put;
  - `api/v1/schemas/claims_schemas.py`;
  - `api/v1/schemas/audiobook_schemas.py`, which keeps its `__all__`.
- Modify (R4):
  - `core/Ingestion_Media_Processing/document_upload_preflight.py:8`;
  - `core/Collections/reading_service.py:470`, function level;
  - `core/Claims_Extraction/__init__.py:1`, which imports the package's own submodule;
  - `core/Audiobooks/alignment_utils.py:10`, `subtitle_generator.py:14`, `tag_parser.py:9`.
- Modify (docs, DoD #3): `Docs/API-related/OCR_API_Documentation.md:42` (mirrored) becomes `` Reference (code): `tldw_Server_API/app/core/Ingestion_Media_Processing/media_request_models.py`, re-exported from `tldw_Server_API/app/api/v1/schemas/media_request_models.py`. `` Then run `bash Helper_Scripts/refresh_docs_published.sh` and stage its `Docs/Published` diff.
- Modify: `layering_baseline.txt`, `test_core_owned_schema_reexports.py`

**Interfaces:**
- Produces:
  - `core.Ingestion_Media_Processing.document_upload_processing`: `DocumentModeCapability`, `DocumentProcessingMode`, `DocumentProcessingStatus`, `DocumentUploadPreflightFile`, `DocumentUploadPreflightItem`;
  - `core.Ingestion_Media_Processing.media_request_models`: `AddMediaForm`, `AudioVideoOptions`, `AutoChunkingGoal`, `ChunkMethod`, `ChunkingMode`, `ChunkingOptions`, `MediaType`, `OcrMode`, `PdfEngine`, `PdfOptions` (364 of 1,069 lines; reads `core.Ingestion_Media_Processing.Email.attachment_policy`);
  - `core.Claims_Extraction.claims_schemas`: `FVAAdjudicationResult`, `FVAClaimResult`, `FVAEvidenceItem`;
  - `core.Audiobooks.audiobook_schemas`: `AlignmentEngine`, `AlignmentPayload`, `AlignmentWord`, `ChapterPreview`, `SubtitleFormat`, `SubtitleMode`, `SubtitleVariant`.

- [ ] **Step 1: Red (R1).** Append four rows, one per module above, each with its names in the listed order. Delete `core.Audiobooks -> api.v1.schemas: 3`, `core.Claims_Extraction -> api.v1.schemas: 1`, `core.Collections -> api.v1.schemas: 1` and `core.Ingestion_Media_Processing -> api.v1.schemas: 1`.
- [ ] **Step 2: Run V1a.** Expected: 4 rows fail with `ModuleNotFoundError`, and the contract tests fail on 4 pairs.
- [ ] **Step 3: Apply R2-R4 and the doc edit.**
- [ ] **Step 4: Run R5.**
- [ ] **Step 5: Run R6.**
  - V5 regex: `schemas\.(document_upload_processing|media_request_models|claims_schemas|audiobook_schemas)|core\.(Claims_Extraction|Audiobooks)\b|document_upload_preflight|Collections\.reading_service`. It includes `tests/Claims/test_claims_analytics_exports_api.py::test_claims_export_filter_schema_imports_in_fresh_interpreter`, the existing cycle pin for `claims_schemas`.
  - Also run `tldw_Server_API/tests/Docs`.
  - V3 prints digests for the 7 S7 members of Review Focus 2's set.
  - The frozen `api.v1.API_Deps.media_add_deps:get_add_media_form` (435 lines) keeps importing `AddMediaForm` from the api module, unchanged.
- [ ] **Step 6: Commit (R7).**

### Task 8: S8, small contexts and the chat-loop events

**Files:**
- Create the following, all under `tldw_Server_API/app/`:
  - `core/Local_LLM/llamacpp_admin_schemas.py`;
  - `core/Buddy/buddy_schemas.py`, one module for both `buddies.py` and `buddy_turns.py`, because `BuddyTurnCreate` reads `BuddyModel` and `ResourceId`;
  - `core/Persona/archetype_schemas.py`;
  - `core/Research/research_runs_schemas.py`;
  - `core/Setup/setup_schemas.py`;
  - `core/Monitoring/monitoring_schemas.py`, while the colliding `WatchlistDeleteResponse` stays in api;
  - `core/File_Artifacts/file_artifacts_schemas.py`;
  - `core/Chat/chat_loop_schemas.py`.
- Modify (R3): `api/v1/schemas/llamacpp_admin_schemas.py`, `buddies.py`, `buddy_turns.py`, `archetype_schemas.py` (keeps `__all__`), `persona.py`, `research_runs_schemas.py`, `setup_schemas.py`, `monitoring_schemas.py`, `file_artifacts_schemas.py`, `chat_loop_schemas.py`
- Modify (R4):
  - `core/Local_LLM/llamacpp_acquisition_jobs.py:7`, `llamacpp_acquisition_service.py:20`, `llamacpp_inventory_service.py:10`, `llamacpp_profile_capabilities.py:10`;
  - `core/Buddy/service.py:13`, `core/Buddy/turns.py:16`;
  - `core/Persona/archetype_loader.py:15`;
  - `core/Research/service.py:13`, `core/Research/streaming.py:8`;
  - `core/Setup/provider_catalog.py:5`, `core/Setup/provider_validation.py:10`;
  - `core/Monitoring/topic_monitoring_service.py:32`;
  - `core/File_Artifacts/file_artifacts_service.py:16`, `core/File_Artifacts/jobs_worker.py:30`;
  - `core/Chat/chat_loop_store.py:10`.
- Modify: `layering_baseline.txt`, `test_core_owned_schema_reexports.py`

**Interfaces:**
- Produces:
  - `core.Local_LLM.llamacpp_admin_schemas`: `LlamaCppAcquisitionJobListResponse`, `LlamaCppAcquisitionJobResponse`, `LlamaCppAsset`, `LlamaCppAssetDownloadRequest`, `LlamaCppAssetImportPreviewResponse`, `LlamaCppAssetMetadata`, `LlamaCppAssetsResponse`, `LlamaCppInventoryItem`, `LlamaCppInventoryResponse`, `LlamaCppModelMetadata`;
  - `core.Buddy.buddy_schemas`: `BuddyCreate`, `BuddyModel`, `BuddyName`, `PersonaPackSource`, `ResourceId`, `StarterSource` (from `buddies`), `BuddyTurnCreate` (from `buddy_turns`);
  - `core.Persona.archetype_schemas`: `ArchetypeBuddyDefaults`, `ArchetypeMCPConfig`, `ArchetypePersonaDefaults`, `ArchetypePolicyDefaults`, `ArchetypeStarterCommand`, `ArchetypeSummary`, `ArchetypeTemplate`, `ArchetypeToolOverride` (from `archetype_schemas`), plus `PersonaConfirmationMode`, a one-line `Literal` alias from `persona`;
  - `core.Research.research_runs_schemas`: `ResearchArtifactManifestEntry`, `ResearchCheckpointSummary`, `ResearchRunResponse`, `ResearchRunSnapshotResponse`;
  - `core.Setup.setup_schemas`: `SetupProviderCatalogEntry`, `SetupProviderCatalogResponse`, `SetupProviderType`, `SetupProviderValidationResponse`;
  - `core.Monitoring.monitoring_schemas`: `Watchlist`, `WatchlistRule`;
  - `core.File_Artifacts.file_artifacts_schemas`: `AsyncMode`, `ExportFormat`, `ExportMode`, `FileArtifact`, `FileCreateOptions`, `FileCreateRequest`, `FileExportInfo`, `FileExportRequest`, `FileType`, `FileValidationIssue`, `FileValidationResult`;
  - `core.Chat.chat_loop_schemas`: `ChatLoopEvent`, `ChatLoopEventType`.

- [ ] **Step 1: Red (R1).** Append one row per (api module, core module) pair with the names above, in the listed order. That is 10 rows: `buddies` and `buddy_turns` both point to `Buddy.buddy_schemas`, and `archetype_schemas` and `persona` both point to `Persona.archetype_schemas`.

  Baseline deltas (measured values):
  - delete `core.Buddy: 2`, `core.File_Artifacts: 2`, `core.Local_LLM: 4`, `core.Monitoring: 1`, `core.Persona` (1 after S3), `core.Research: 2` and `core.Setup: 2`, each `-> api.v1.schemas`;
  - `core.Chat` −1 (6 on the measured merge; 4 once `.13` C1 has merged, 3 once `.14` has).
- [ ] **Step 2: Run V1a.** Expected: 10 rows fail with `ModuleNotFoundError`, and the contract tests fail on 8 pairs.
- [ ] **Step 3: Apply R2-R4.** `api/v1/schemas/__init__.py` imports `chat_loop_schemas`, so every schema import runs `core/Chat/__init__.py`. Before `.13` PR B, that init imports only `chat_loop_store`, which will import only `core.Chat.chat_loop_schemas`. After PR B the init is a docstring.
- [ ] **Step 4: Run R5.**
- [ ] **Step 5: Run R6.**
  - V5 regex: `schemas\.(llamacpp_admin_schemas|buddies|buddy_turns|archetype_schemas|persona|research_runs_schemas|setup_schemas|monitoring_schemas|file_artifacts_schemas|data_tables_schemas|chat_loop_schemas)|core\.(Local_LLM|Buddy|Persona|Research|Setup|Monitoring|File_Artifacts)\b|chat_loop_store`.
  - `data_tables_schemas` is included because that api module imports moved `file_artifacts_schemas` names (`FileCreateOptions` and others), so its tests exercise the re-export.
  - The in-app pathspec includes `core/MCP_unified/tests/test_extraction_contracts.py::test_catalog_loader_uses_standalone_catalog_schema`.
  - V4 must also pass `$PY -c "import tldw_Server_API.app.api.v1.schemas"` in a fresh interpreter.
  - V3 prints digests for the 3 `Research*` models and `BuddyModel`.
- [ ] **Step 6: Commit (R7).**

### Task 9: S9, Sync contract exporter moves to the API layer (inversion)

**Files:**
- Modify: `tldw_Server_API/app/core/Sync/v2/personal_context_ongoing_contract.py`. Delete `export_personal_context_ongoing_contract` (`:153` to the end of the function) and its `__all__` entry.
- Modify: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`:
  - append the function verbatim, dropping its function-level import of this module's own 13 models;
  - add `PERSONAL_CONTEXT_ONGOING_ENDPOINTS` and `PersonalContextCleanupAck` to the existing import from `core.Sync.v2.personal_context_ongoing_contract` (`:32`);
  - add `"export_personal_context_ongoing_contract"` to `__all__`;
  - change the docstring to `"""Export the version-one API contract from the API models."""`.
- Modify: `Helper_Scripts/generate_personal_context_ongoing_contract.py:15-17`, importing from `tldw_Server_API.app.api.v1.schemas.sync_v2_models`.
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_ongoing_contract.py:15-28`. Move `export_personal_context_ongoing_contract` from the core import block (`:18-28`) into the `sync_v2_models` import block (`:15-17`).
- Modify: `layering_baseline.txt`

**Interfaces:**
- Produces: `tldw_Server_API.app.api.v1.schemas.sync_v2_models.export_personal_context_ongoing_contract() -> dict[str, object]`, with the same output as before.

If `.16` PR C is on the base, adjust two things:
- the function lives in `tldw_Server_API/app/core/Sync_Contracts/personal_context_ongoing_contract.py` (the old path is a module alias); edit that file;
- the baseline line to delete is `core.Sync_Contracts -> api.v1.schemas: 1  # bump: ...`, not `core.Sync -> api.v1.schemas: 1`.

`sync_v2_models.py` keeps whichever import path the base uses.

This edge needs no model move. The only core reader is the function that exports the API models' JSON schema. Exporting the API contract is an API-layer job, so the function moves up. Its two importers are retargeted in the same PR, so no shim is left; a shim in core would recreate the edge. AC #1 covers schemas that core imports, and here none is imported once the exporter leaves core.

- [ ] **Step 1: Record the contract digest on the base.**

  ```bash
  $PY -c "import hashlib,json;from tldw_Server_API.app.core.Sync.v2.personal_context_ongoing_contract import export_personal_context_ongoing_contract as e;print(hashlib.sha256((json.dumps(e(),sort_keys=True,separators=(',',':'),ensure_ascii=True)+'\n').encode()).hexdigest())"
  ```

  Expected: one sha256 (`8292b30c8557f2d2423015aec1ba2b32cee0eea42fa0b39abfd66636fcc932ea` on the measured merge; the dry run reproduced it).
- [ ] **Step 2: Red.** Delete the baseline line. Run V1a. Expected: `core.Sync -> api.v1.schemas: 1 statement, not in the baseline` (or `core.Sync_Contracts -> ...`).
- [ ] **Step 3: Move the function and retarget the two importers.**
- [ ] **Step 4: Green.**
  - Run V1a. Expected: it passes.
  - Run Step 1's command with the import path changed to `tldw_Server_API.app.api.v1.schemas.sync_v2_models`. Expected: the digest equals Step 1's.
  - Run `$PY -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1315-sync" tldw_Server_API/tests/Sync`. Expected: no failures, including `test_contract_generator_writes_reproducible_schema_and_manifest` (the subprocess run of the helper script) and `test_sync_v2_store.py::test_core_models_import_without_api_schema_module`.
- [ ] **Step 5: Run V1, V2, V3 and V6.** V3 has no new rows, but its raw OpenAPI comparison still runs. Create and remove the base worktree as in R6.
- [ ] **Step 6: Commit** `refactor(sync): the API layer exports the personal-context contract from its own models (TASK-13950.15 S9)`.

### Task 10: S10, chat request model, and close-out

**Base:** dev after S9 and after `.14` **slice 2** merges, which itself follows every `.13` PR. Slice 2 removes the config reads that `ChatCompletionRequest`'s closure includes today (`_config`, `_env_default_provider`, `_test_mode_enabled`, `DEFAULT_LLM_PROVIDER`). By then, `.13` C1 has moved `ALL_SUPPORTED_PROVIDER_NAMES_LIST` into `core/config_llm_providers.py`.

**Files:**
- Create: `tldw_Server_API/app/core/Chat/chat_request_schemas.py`
- Modify (R3): `api/v1/schemas/chat_request_schemas.py`
- Modify (R4): `core/Chat/chat_helpers.py:14`, `core/Chat/chat_service.py:2821` (same line count, because `chat_service` functions are size-frozen), `core/Prompt_Management/prompt_improvement_dispatch.py:12`.

  `.13` C1 deletes `chat_service.py:2156` and `:2251`, and `.14` retargets all eight `DEFAULT_LLM_PROVIDER` importers to `core/config_llm_providers.py`. Expected leftovers: none. A leftover found on the base is retargeted to `core/config_llm_providers.py` here.
- Modify: `tldw_Server_API/tests/lint/test_schemas_config_boundary.py` (from `.14`). Delete the three entries this slice makes stale from `CHAT_REQUEST_SCHEMAS_CORE_IMPORTERS`: `core.Chat.chat_helpers`, `core.Chat.chat_service`, `core.Prompt_Management.prompt_improvement_dispatch`.
- Modify (docs, DoD #3): `Docs/Code_Documentation/Chat_Developer_Guide.md:73` (mirrored) becomes `` - API schemas: `tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py` (re-exports `ChatCompletionRequest`, defined in `tldw_Server_API/app/core/Chat/chat_request_schemas.py`) ``. Then run `bash Helper_Scripts/refresh_docs_published.sh`.
- Modify: `layering_baseline.txt`, `test_core_owned_schema_reexports.py`, and the paragraph of the `tldw_Server_API/tests/lint/test_core_to_api_import_boundary.py` docstring that begins "Schema-only imports from `api/v1/schemas/*`". That paragraph now says core reads no api schema since TASK-13950.15, and the remaining core -> api pairs are true inversions.

**Interfaces:**
- Consumes: `core/config_llm_providers.py` from `.13` C1 (`get_api_keys`, `ALL_SUPPORTED_PROVIDER_NAMES_LIST`) and `.14` (the default-provider resolver). After `.14`, `ChatCompletionRequest`'s closure reads them from there instead of from module-level config reads.
- Produces: `core.Chat.chat_request_schemas` with `ChatCompletionRequest`, `_is_server_managed_chat_request_field` and their closure. Re-measure the closure on the base. Today it is about 48 top-level names and about 973 lines, including `BaseMessage`, the `ChatCompletion*MessageParam` family, `ToolDefinition`, `ResponseFormat`, `ResearchChatContext*`, the `MAX_*` limits and the provider-name lists. It reads `core.Chat.history_selection_schemas` (R2 retarget), `core.LLM_Calls.*` and `core.custom_openai_providers`.

- [ ] **Step 1: Re-scan patch targets on the base.**

  ```bash
  git grep -n -P "chat_request_schemas" -- 'tldw_Server_API/tests/*.py'
  ```

  Expected: every patch, `setattr` and `reload` hit names only:
  - `load_and_log_configs`, `get_api_keys` or `DEFAULT_LLM_PROVIDER`, if `.13` or `.14` left any;
  - or the module itself, the `test_env_file_selection.py:85-87` import-and-reload.

  None names a definition in the closure. A hit that does name one is retargeted to `tldw_Server_API.app.core.Chat.chat_request_schemas.<name>` in this PR.
- [ ] **Step 2: Red (R1).** Append the row `("chat_request_schemas", "Chat.chat_request_schemas", (<the re-measured closure names>))`. Delete every remaining `core.* -> api.v1.schemas` baseline line. Run V1a. Expected: the row fails with `ModuleNotFoundError`, and the contract tests list every remaining importing module.
- [ ] **Step 3: Apply R2-R4 and the doc edit.**
- [ ] **Step 4: Run R5.**
- [ ] **Step 5: AC #2 check.** The second command matches import statements only. The R2 docstrings name the API schemas package in prose, so a plain text grep would also match them.

  ```bash
  grep -E '^core\.[^ ]+ -> api\.v1\.schemas:' tldw_Server_API/tests/lint/layering_baseline.txt
  git grep -nP '^\s*(from|import)\s+(tldw_Server_API\.app\.api\.v1(\.schemas\b|\s+import\s+schemas\b)|\.+api\.v1\b)' -- 'tldw_Server_API/app/core/*.py' ':!tldw_Server_API/app/core/*/tests/*'
  ```

  Expected: the first command prints nothing. The second prints exactly one line, the type-only `TYPE_CHECKING` import (see Task 6):

  ```text
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:63:    from tldw_Server_API.app.api.v1.schemas.quizzes import OsceQuizExportV2
  ```

  On the measured merge the same command prints 93 lines: the 92 statements plus that one.
- [ ] **Step 6: Run R6.**
  - V5 regex: `schemas\.chat_request_schemas|core\.Chat\.(chat_helpers|chat_service|chat_target_resolution)|prompt_improvement_dispatch|speech_chat_service`.
  - Also run `tldw_Server_API/tests/Config/test_env_file_selection.py` and `tldw_Server_API/tests/Docs`.
  - V3 prints the `BaseMessage` digest.
  - In the PR body, record the contract report: pairs, statements, and both SCC sizes.
- [ ] **Step 7: Backlog close-out** with tools/backlog-py.

  First check that AC #1 carries the amended wording:

  ```bash
  grep -c "services importers of moved names stay on the api re-export and move in TASK-13950.37" "backlog/tasks/task-13950.15 - Move-API-schemas-consumed-by-core-into-core-owned-models.md"
  ```

  Expected: `1`. If it prints `0`, stop and ask the controller, because AC #1 changed again. Otherwise run:

  ```bash
  PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13950.15 --check-ac 1 --check-ac 2 --check-ac 3 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 --append-notes "<S1-S10 PR numbers; final contract report line; DB_Management -> core.Evaluations bump (S1) with its no-DB-code test; inversions S6/S9; AC #2 exception: the type-only TYPE_CHECKING import at ChaChaNotes_DB.py:63, which is syntactically module level and creates no runtime edge; services importers left on the re-exports for TASK-13950.37>" --final-summary "<what moved, where, how OpenAPI identity was proven (V2 + V3 per slice, digests in each PR), and the four developer docs updated>"
  PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13950.37 --append-notes "TASK-13950.15 left these services importers on the api re-exports of core-owned models: services/audiobook_jobs_worker.py, outputs_service.py, study_pack_jobs_worker.py, media_ingest_jobs_worker.py, osce_generator.py, quiz_generator.py. Retarget them to the core modules listed in tldw_Server_API/tests/unit/test_core_owned_schema_reexports.py."
  git add -A backlog/tasks
  ```

  Fill the two `<...>` texts in at execution. This one call writes the notes and the summary, so DoD #5 holds.
- [ ] **Step 8: Commit (R7)** with message `refactor(chat): core owns ChatCompletionRequest; core imports no api schema (TASK-13950.15 S10)`.

---

## Self-review

- **Spec coverage.**
  - AC #1, as amended (core-owned models re-exported from api; services importers stay on the re-export for `.37`): Tasks 1-8 and 10. Task 10 Step 7 checks the amended wording before checking the AC.
  - AC #2 (no core -> api.v1.schemas imports; entries removed): every task's R1 and R5; Task 10 Steps 2 and 5, with the `TYPE_CHECKING` exception recorded.
  - AC #3 (OpenAPI unchanged): V2 in every task. The existing required gate is the snapshot test the roadmap asks for, so it is not duplicated (ruling accepted). V3 adds raw-byte identity and per-model digests for the 17 models outside OpenAPI, recorded in each PR.
  - Ruling 1 (S1 bump with proof): `test_the_evaluations_models_load_no_db_management_code`.
  - Roadmap "Evaluations first": S1.
  - DoD #3 (docs): four developer-doc lines in S2, S7 and S10. DoD #4 (Bandit): V6.
- **Placeholders.** Three remain, and each depends on execution-time facts: Task 10's re-measured names (after `.14` slice 2), the backlog note texts (PR numbers), and the follow-up task id (the next free id). Task 5's rows refer to the listed name order instead of repeating 39 names.
- **Type consistency.** `CORE_OWNED`, `API` and `CORE` have one shape, used by the test, V3, V4 and V6.
- **Proportion.** Moved code is described by name lists, not reproduced. Two helper scripts (V3, V6) and three tests carry code, because their bodies are the verification.
- **Review and dry-run findings folded in.**
  - Review: I1-I4 and M1-M13.
  - Dry run: I1, I2 and m1-m9.
  - M14: the controller accepted the S7 and S8 bundles (ruling 2026-10-26).
  - Declined: dry-run m2's alternative of re-sorting imports, because line-keyed baselines forbid shifting lines; R4 says to leave the order alone.
