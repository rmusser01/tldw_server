# Decompose unified_rag_pipeline (TASK-13950.24) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `unified_rag_pipeline` becomes a ~338-line kwargs shim over `run_rag_pipeline(rcfg: RagRunConfig)`, a ~175-line orchestrator with no nested defs. The 6,740-line body moves into named stage modules (retrieval, rerank, generation, post-processing and ten more), and RAG output stays identical throughout.

**Architecture:** Twelve tasks and seventeen PRs (the last is backlog-only), merged strictly in order.
1. A golden-output test and a path-gated required job land first (Task 1). A stdlib move tool and AST pins follow (Task 2).
2. Three in-place rewrites turn the body's names into `rcfg.<group>.<name>` (read-only parameters, Task 3) and `state.<name>` (mutated parameters, shared locals and hoisted helpers, Task 4). After them, every remaining move is verbatim.
3. Stages move out tail-first, one stage group per PR (Tasks 5-9), so no move shifts a statement that a later PR moves.
4. The entry point splits, then the 14 direct callers pass a config (Tasks 10-11). Task 12 closes out.

`Helper_Scripts/rag_split/rag_split.py` produces every rewrite and move; its `verify` proves each one against the PR's base (Appendix A).

**Tech Stack:** Python 3.12 (stdlib `ast`, `symtable`, `dataclasses`), pytest, GitHub Actions (`backend-required.yml`), the AST lint ratchets under `tldw_Server_API/tests/lint/`.

**Spec:**
- `backlog/tasks/task-13950.24 - Decompose-unified-rag-pipeline-into-a-typed-config-and-stages.md`, AC #1-#5.
- The roadmap's Global Constraints and Phase 3 `.24` row ("golden-output test; config object (no behaviour change); one stage group per PR"): `Docs/superpowers/plans/2026-10-25-architecture-remediation-roadmap.md`.
- Research at dev 5033a7689b, git-ignored, so the executor's ledger workspace holds it: `.superpowers/sdd/2026-11-24-task-13950-24-rag-pipeline-decompose/research-A.md` (anatomy, regions R00-R30, Appendix A's parameter-to-group column) and `research-B.md` (callers, tests, ratchets). The planner reran every number below.

## Measured (dev 5033a7689b)

- **File and function.** `core/RAG/rag_service/unified_pipeline.py` is 9,582 lines. `unified_rag_pipeline` spans `:2016-9067`, 7,052 lines. Its signature takes 312 of them (`:2016-2327`) and its docstring 25 more (`:2328-2352`, `body[0]`). Its `function_size_baseline.json` ceiling is 7,063 lines / 234 params.
- **Parameters.** 233 named plus `**kwargs`. The body reads 13 hidden kwargs keys (15 reads; `metadata` three times), with `.get` defaults `True` for `include_sources`/`include_metadata`, `False` for `enable_expansion`, and `None` for the other ten.
  - 35 parameters are reassigned in the body (AST Store or `nonlocal`): `adaptive_time_budget_sec, date_range, enable_cache, enable_date_filter, enable_discussion_search, enable_hyde, enable_multi_vector_passages, enable_numeric_fidelity, enable_parent_expansion, enable_reranking, enable_web_fallback, expand_query, fts_level, generation_prompt, hybrid_alpha, hyde_model, hyde_provider, include_parent_document, low_confidence_behavior, media_db_path, mv_max_spans, mv_span_chars, mv_stride, nli_model, numeric_fidelity_behavior, parent_context_size, parent_max_tokens, query, require_hard_citations, reranking_strategy, resolved_request, retrieval_plan, sql_retriever, strict_extractive, top_k`.
  - Body lines that read read-only parameters: 565. Lines that touch mutated parameters: 354. Lines that touch `result`: 583.
- **Structure.** 57 `enable_*` flags; 5 are never tested by an `if` (`enable_graph_retrieval, enable_streaming, enable_batch, enable_enhanced_chunking, enable_parent_expansion`).
  - One `try` spans `:3000-9020`, with 89 body statements, three handlers and a 203-line `finally`. `return` occurs only at `:2566`, `:2585` (empty query), `:8806` (fallback dict), `:9034` and `:9053` (tail).
  - There are 42 depth-1 nested defs and classes. 21 are used outside the top-level statement that defines them:
    - 18 prologue closures, 442 lines with `_EarlyReturn` (R14 lists them);
    - `_EarlyReturn`;
    - `_prefetch_routed_external_docs`, used only by the next statement;
    - `_execute_retrieval_variant`, defined in R14 and used in R15.
  - `:8451-8454` test `"old_nf_missing" in locals()` and three siblings; their bindings are at `:8391-8413`.
- **Names.** In `unified_pipeline.py`, `rcfg` and `state` occur 0 times. `cfg` occurs 13 times and `s` 36 times, so neither can be the new names.
- **Patching.** Tests patch 57 distinct module names of `unified_pipeline`, in string and attribute forms. The top ones are `MultiDatabaseRetriever` (93 string + 42 attribute refs), `AnswerGenerator` (32 + 14) and `get_shared_cache` (18 + 6). `test_hyde.py:887` patches `unified_pipeline.logger`. No test patches a stdlib name on the module. Rerun: `B_targets.py`, and `git grep -E 'unified_pipeline\.(time|os|asyncio|datetime|json|re)\b'`, which is empty.
  - `rag_unified.py:696-713` assigns `up.MultiDatabaseRetriever` and `up.RetrievalConfig` at runtime.
  - About 60 test lines in 25 files patch `unified_rag_pipeline` itself, 15 of them in `MCP_unified/tests/test_rag_module.py`.
- **Error paths.** Only five calls in the `try` body sit outside an inner handler that catches `RuntimeError` or `Exception`:
  - `_normalize_pipeline_sources` (`:3002`; its inner `try` catches `ValueError` only);
  - `assess_query_for_clarification` (`:3191`) and `spell_check_query` (`:3229`);
  - `TableProcessor` (`:5508`);
  - `coordinate_standard_result_evidence` (`:6941`).

  A failing retriever is caught at `:4818` (`document_retrieval_failed`), and a failing `AnswerGenerator` at its own handler (`answer_generation_failed`). An invalid `sensitivity_level` raises at `:5378`, but `:5491` catches it as `"Security filter failed: …"`, so research A's finding 6 is wrong.

  `build_retrieval_only_result` (`:6933`, in `try[69]` with no inner handler) runs when `enable_generation=False`. Making it raise `RuntimeError` reaches the 7-type handler (`:8794`). The dry run measured both outcomes:
  - `fallback_on_error=True` returns the `:8806` dict, whose `timings` holds `total`, written by `finally` after the return;
  - `fallback_on_error=False` returns `errors == ["pipeline_failed"]`.
- **Function-local imports.** 69 import statements sit inside the function. They resolve at call time from their source module, so a patch on `up` cannot reach them. Appendix C lists the ones the golden cases reach.
- **Source-reading tests.** `Sharing/test_shared_workspace_chat_retrieval.py:909` pins `set(signature.parameters)`. Its `test_hidden_pipeline_kwargs_reads_are_explicitly_reviewed_absent` parses `inspect.getsource(unified_rag_pipeline)` and expects exactly the 13 literal kwargs reads. `transport.py:99` returns the whole payload only while the function has `**kwargs`.
- **Request schema.** `UnifiedRAGRequest` (`core/RAG/rag_schemas_unified.py:78`) has 230 fields. 193 are shared with the parameters, and 191 of those have equal defaults; the two that differ are `sources` (`None` vs `['media_db']`) and `enable_generation` (`True` vs `False`). 37 fields are request-only and 40 parameters are pipeline-only (`B_schema.py`, `default_mismatch.py`).
- **Callers.** There are 14 direct calls in 12 app files (research B §1a). Three more use the function by reference: `streaming_executor.py:741`, `shared_workspace_chat_service.py:491` and `rag_unified.py:2386`. Inside the module, the adaptive rerun at `:8275` passes 79 keywords (73 `x=x`, 6 computed).
- **Ratchet hits inside the function.**
  - `scope_predicate`: one, `:3769` `if RewriteCache and user_id` (R12).
  - `keyword_identity`: one, `:5225` (R15).
  - BLE001: six, at `:3431`, `:3465`, `:3689`, `:7373`, `:8858` and `:8999`; there are none elsewhere in the file (`ruff check --isolated --select BLE001`).
  - Env reads: `SEARCH_MAX_ITERATIONS_*` (R10), `RAG_GUARDRAILS_STRICT` and `RAG_NUMERIC_FIDELITY_BEHAVIOR` (R11), three `RAG_RERANK_CALIB_*`/`RAG_MIN_RELEVANCE_PROB` (R19), `RAG_NLI_MODEL(_PATH)` (R24) and `RAG_ADAPTIVE_TIME_BUDGET_SEC` (R26).
  - The file's `global` (`:1291`) and its test-mode branch (`:9133`) are outside the function.
- **Untested flags.** No test sets `enable_pre_retrieval_clarification`, `enable_observability`, `enable_content_policy_filter`, `enable_html_sanitizer` or `enable_discussion_search` to True. Measured with `git grep -h -E '<flag>.{0,4}(=|:) *True' -- tldw_Server_API/tests tldw_Server_API/app/core/MCP_unified/tests`. Nine more appear one to three times.
- **No RAG test runs on a dev PR.** `ci.yml`'s RAG shards run only at `nightly`/`release` tier.
- **Gate cost, local** (serial, `TEST_MODE=true DISABLE_HEAVY_STARTUP=1`, Docker `postgres:18` up; the three `tests/RAG` PostgreSQL files also with `TLDW_TEST_POSTGRES_REQUIRED=1`):
  - Appendix D's 55 files ran 1,051 tests in 166 s (963 in 125.6 s, plus 88 in 40.0 s), with 0 skips.
  - One local failure: `test_rag_profiles.py::TestSlidesSourceLocalRerankers::test_flashrank_accepts_complete_non_listwise_local_bundle`, because flashrank is not installed in the venv.
  - With `-n 2`, all 51 RAG_NEW files that mention the pipeline (the excluded three included) took 123 s, so xdist buys nothing.
  - Excluded, and failing or skipping on dev: `RAG_NEW/test_unified_pipeline.py` (4 teardown errors from live DNS to `api.openai.com`, plus 3 `live_network_baseline` entries), `integration/test_rag_agentic_api.py` (skips on HTTP 500) and `integration/test_rag_benchmarks_ablation.py` (skips without lifespan).
- **Trigger rate.** Of 196 merges to dev in the 30 days to 2026-11-24, 19 touched the RAG globs (R18). 26 did with `pyproject.toml`/`uv.lock`, and 47 (24%) with `backend-required.yml`/`path_classifier.py` too.
- **Import.** `python -X importtime` puts `unified_pipeline` at 1.47 s cumulative.
- **Open PR #3480** (draft, CONFLICTING, 419 files) changes `unified_pipeline.py` +68/−53 in 14 hunks: module imports (2), a decorator on the function, the signature (`max_generation_tokens` 500→4096), R05, R06 (`_with_timeout`), R08, R14, R15 (a new media-context block), R16 (2, security admin role), R20 (sibling inclusion removed), R22 and R29 (faithfulness). It also touches 27 other RAG files.
- **Python and disk.** Python is 3.12.13 in the venv and in CI, and `.gitignore:281` ignores `*.json`. Disk was 235 GiB free.

## Rulings

R1-R9 are the controller's and binding; R10-R22 are the planner's. Each gives its why and, where it matters, its cost if wrong.

1. **R1, no behaviour change.** Every PR keeps RAG output identical; AC4 is the contract.
   - Defects found on the way are filed at close-out, never fixed here: the `enable_generation` default mismatch, the 9 never-read parameters, the rerun's 154 dropped knobs and the others in Task 12.
   - Inert parameters stay in the signature, and in `RagRunConfig.inert`.
2. **R2, the golden test lands first** (Task 1), before any code moves.
   - Research B's G1-G12 become G01-G19 (Appendix C). The extra cases cover gaps: a cache that really hits (G07), the outer ladder with `fallback_on_error` True and False (G12a/G12b, through `build_retrieval_only_result`; see Measured), a security-filter failure (G13) and explicit source selection (G19).
   - A stage-order spy records every fake collaborator call.
   - The golden JSON is committed (`git add -f`, the `.21` precedent).
   - Seeded-change tests prove the comparison is sensitive.
   - The golden file does not change after PR A.
3. **R3, the gate.** A new path-gated required job, `rag-pipeline`, follows the `chacha-owner-isolation` precedent (R18). It runs the golden test and Appendix D's list, and it lands in PR A.
4. **R4, patch compatibility.** Every name moved code reads from `unified_pipeline`'s namespace is read late, through the module object `up` (R11). Existing patches stay live with no test churn.
   - A pin fails if a pipeline module binds a name that tests patch on `unified_pipeline` (Task 2).
5. **R5, the config.** `RagRunConfig` is a frozen dataclass of 17 frozen sub-config dataclasses (R12).
   - `unified_rag_pipeline` keeps its exact signature and `**kwargs`.
   - The new entry point is `run_rag_pipeline(rcfg: RagRunConfig)`.
   - The request schema needs no new builder: `RagRunConfig.from_kwargs(**<the dict the caller builds today>)` is it (R12).
6. **R6, the adaptive rerun.** It keeps its literal 79-keyword call to the kwargs shim `up.unified_rag_pipeline`; the explicit keyword list is the allow-list.
   - `test_rerun_passes_the_frozen_argument_set` pins the names, their order and the six computed values.
   - Golden G10 pins the values the rerun receives.
   - Never `dataclasses.replace(rcfg, …)`: that would carry the 154 knobs the rerun resets to defaults today.
7. **R7, AC2 and AC3.**
   - The orchestrator ends at about 175 lines with no nested defs, and the shim at about 338 (signature 312 + docstring 25 + one `return`).
   - Every new function is at most 280 lines, which leaves margin under the 300-line THRESHOLD.
   - Every PR `--tighten`s `unified_rag_pipeline`'s ceiling. Other baselines are re-keyed, never widened (R20).
   - The `_EarlyReturn`/`finally` ordering, the spans, the closures and `locals()` are treated in R14 and R15.
8. **R8, slicing.** One stage group per PR, tail-first, each under about 2,500 changed lines (see the slices table).
   - Each PR passes the golden test unchanged, Appendix D's gate list, its own C4 files and every ratchet.
   - Each PR proves its move with `rag_split verify` (R16).
9. **R9, #3480.** Its hunks span almost every region, so its first collision is Task 3, whose rewrite touches every read-only parameter read. Don't wait for it. The owner rebases it onto the stage modules, and Appendix B maps each hunk's region to its module.
10. **R10, rewrite in place, then move verbatim.**
    - A function local is semantically one slot of one frame, and an attribute of a per-call object is the same.
    - So Tasks 3-4 rewrite every name that crosses a planned function boundary: read-only parameters to `rcfg.<group>.<name>`, everything else to `state.<name>`.
    - From then on a move only dedents, prefixes module globals and fixes relative imports.
    - `rcfg` and `state` are the new names because the file uses neither (Measured).
    - **Cost if wrong:** none in behaviour, because verify proves each rewrite.
11. **R11, the late-resolution seam.** Every pipeline module whose code reads a `unified_pipeline` global (`run_state.py` from D2, and every stage module) imports `from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up`.
    - Moved code reads every `unified_pipeline` module global as `up.<name>`, annotations included. Pipeline modules carry no `from __future__ import annotations`, because `unified_pipeline` has none and nested-def annotations evaluate at def time today.
    - Two kinds of name are imported directly instead:
      - names bound by a stdlib import in `unified_pipeline` (module in `sys.stdlib_module_names`), so the env-read ratchet still sees `os.getenv`;
      - pipeline-package names, such as `_run_state` or `RagRunConfig`.
    - **Cycle rule:**
      - Runtime imports among `unified_pipeline`, `run_state` and the stage modules are module imports only: `from .pipeline import rerank as _rerank` and `from .pipeline import run_state as _run_state` in `unified_pipeline`, `from . import run_state as _run_state` in a stage module, and `up`.
      - Names are read as attributes at call time (`_rerank.run_rerank(...)`, `_run_state.PipelineState(rcfg)`, `except _run_state.EarlyReturn:`).
      - `run_config` imports neither, so its names may be imported by name anywhere.
      - Stage signatures annotate `state: "PipelineState"` from a `TYPE_CHECKING` import.
    - Any import order is then cycle-safe, because `from pkg import mod` falls back to `sys.modules` for a partially initialised module. Subprocess tests pin it (Task 5).
    - **Why not a `deps` object:** it would cost about 300 test edits in 35 files.
12. **R12, `RagRunConfig`.**
    - **Frozen stdlib dataclasses with `slots=True`, not pydantic.** Validators would coerce or reject values the function accepts today (R1). Runtime handles (DB objects, callbacks) need no `arbitrary_types_allowed`. Core owns the type. `dataclasses.fields` lets pins enumerate every field.
    - **17 groups, with research A Appendix A's membership:** `request` (3), `scope` (10), `runtime` (11), `retrieval` (24), `processing` (20), `query_prep` (20), `cache` (4), `safety` (14), `rerank` (8), `generation` (18), `citations` (6), `verification` (27), `self_correct` (28), `research` (12), `observability` (14), `resilience` (5), `inert` (9).
    - **The 13 hidden keys are typed fields:**
      - `request`: `metadata`, `include_sources`, `include_metadata`;
      - `query_prep`: `enable_expansion`;
      - `scope`: `workspace_id`;
      - `runtime`: the three db paths, `prompts_db` and `faithfulness_llm`;
      - `verification`: the three `claims_budget_*` keys.
    - Unknown keys are ignored, exactly as the function ignores them; no `extras` bag, because nothing reads one.
    - `from_kwargs` copies values without copying objects, and field defaults equal the signature's.
    - Every caller hands `from_kwargs` the dict it hands the shim today, so the two request/function default differences resolve exactly as they do now, and need no special case.
13. **R13, `PipelineState`.** It is a plain class with `__slots__`, one slot per shared name, and no defaults.
    - Reading an unset slot raises `UnboundLocalError` from `__getattr__`. Today that read raises `UnboundLocalError`, a `NameError` the 7-type handler does not catch. A plain `AttributeError` would be caught and turn into `pipeline_failed`.
    - Its `__init__` stores `rcfg` and copies the 35 mutated parameters from it (`MUTATED_PARAMS`).
14. **R14, closures.**
    - **Hoisted (Task 4, PR D2).** The 18 prologue closures become `PipelineState` methods, with the leading underscore dropped (`state.sync_effective_query`). A bound method substitutes for the closure name exactly, and hoisting removes 442 lines from the function, which pays for D3's publish lines. In their bodies, `state` becomes `self` and `rcfg` becomes `self.rcfg`. The 18:
      - `_resolve_effective_enable_generation`, `_planned_index_namespace`, `_refresh_retrieval_snapshot_from_current_knobs`, `_retrieval_plan_for`, `_sync_effective_query`, `_latch_optional_provider_failure`;
      - `_latched_optional_provider_failure`, `_ensure_profile_resolution_metadata`, `_record_profile_degradation`, `_build_cache_namespace`, `_get_cache_instance`, `_with_timeout`;
      - `_resilient_call`, `_apply_generation_gate`, `_build_multi_retriever`, `_build_pipeline_db_paths`, `_apply_workspace_filtering_to_result`, `_decorate_calibration_metadata`.
    - **`_EarlyReturn` becomes the module class `EarlyReturn`** in `run_state.py`, read as `_run_state.EarlyReturn` (R11). This is equivalent: both raise sites (`:3049`, `:3223`) sit inside the same call's `try`, so no instance escapes to another call.
    - **Published.** A closure defined in one planned function and used in another (for example `_execute_retrieval_variant`, or the claims closures) stays where it is and gets `state.<name> = <name>` right after its def. Its free locals become state.
    - Nested defs stay nested inside stage functions; the orchestrator has none.
15. **R15, the partition (Appendix B).**
    - **Spans and wrappers.** A function is a run of whole statements of one block. A hand-entered span (`_otel_cm.__enter__()` … `finally: __exit__`) stays in one wrapper function with its `try`, and the wrapper calls the leaves.
    - **`locals()`.** `adaptive_rerun` holds both the bindings and the `locals()` checks.
    - **Kept inline in `run_rag_pipeline`, so no moved code contains `return`:**
      - the empty-query early return (R02);
      - the except ladder, in today's order (`_run_state.EarlyReturn` pass → provider failures re-raise → the 7 types append `pipeline_failed` and return the fallback dict whose `metadata`/`timings` alias `state.result`'s → `finally: await _post_processing.finalize_run(rcfg, state)`);
      - the tail (R30).
    - **Tail-first order.** Each move PR replaces only statements after every not-yet-moved one, so Appendix B's anchors stay valid.
16. **R16, the verify check is worth building** (R8 left it optional).
    - The golden covers 20 configurations, not the roughly 400 branches.
    - Until Task 10, a missed rename would read the shim's untouched parameter, a silent stale value that no `NameError` reveals.
    - Moves change namespaces (`up.`/`state.`/`rcfg.`), which is where silent drift hides.
    - `verify` compares ASTs, independent of `apply`'s text edits. Its output goes in each PR body, and `test_new_manifests_verify_against_the_pr_base` reruns it in the Enforce step on the PR's new manifests.
17. **R17, AC1's "all 15 external callers"** means the 14 direct call sites in 12 files. Each calls `run_rag_pipeline(RagRunConfig.from_kwargs(**<today's kwargs>))`.
    - These keep the kwargs shim:
      - `streaming_executor`'s injected `standard_pipeline` (15 kwargs fakes);
      - `SharedWorkspaceChatService.rag_pipeline` (3 fakes, plus the Sharing sentinel that pins the kwargs call);
      - `transport.py:99`'s signature filter.
    - Task 12 amends AC1 to say so.
    - **Cost if wrong:** one follow-up migrates the two seams, about 18 fakes.
18. **R18, the job.**
    - Job id `rag-pipeline`, named "RAG pipeline (golden + pipeline tests)". Flag `rag_pipeline_changed`. Its globs are R18's list in Task 1.
    - The test list `RAG_PIPELINE_GATE_TESTS` and `RAG_PIPELINE_GATE_EXCLUDED` (name → reason) live in `Helper_Scripts/ci/path_classifier.py`, and the workflow lists the same files literally.
    - It runs serially, with a `postgres:18` service and `TLDW_TEST_POSTGRES_REQUIRED=1` for its PostgreSQL cases (in `test_slides_source_retrieval_hardening`, `test_restricted_postgres_media_retrieval` and `test_sql_retriever_integration`), plus chacha's junit check (each listed file has ≥1 pass; 0 skips).
    - **Cost:** 166 s of tests locally, plus job setup. It runs on about 24% of dev merges with the chacha-style globs, or 13% without the workflow and classifier globs. PR A records the CI wall time. The job runs in parallel with `non-utc-tests`, and `backend-required` waits for both.
    - **Why not the Enforce step:** that runs on every backend PR. **Why not nightly:** that fails AC5.
19. **R19, the golden.** Appendix C has the cases. Two pins make it hard to fool:
    - `test_golden_cases_cover_every_live_flag_both_ways`: each of the 52 live flags is True in ≥1 case and False in ≥1 case, using effective values (case kwarg, else signature default; `None` counts as False).
    - The normalized record keeps `metadata`'s insertion order and `timings`' keys, so stage order and the post-return `finally` stay visible.
    - No case patches a stdlib name on `up`: those are imported directly by pipeline modules (R11), so such a patch would go inert after a move and fail spuriously.
20. **R20, ratchets.**
    - **`function_size`:** `--tighten` in every PR, with no new entries; every new function is ≤280 lines.
    - **`env_read`:** a moved read keeps its code. The new module gets an entry `{"reads": n, "import_time": 0, "reason": "moved from core/RAG/rag_service/unified_pipeline.py (TASK-13950.24 PR <x>)"}`, and `unified_pipeline.py`'s entry is `--tighten`ed. Reading env through the config would change read timing (R1).
    - **`scope_predicate`:** `--write-baseline` re-keys `:3769` in PR K.
    - **`keyword_identity`:** `:5225` moves to `pipeline/augment.py: 1  # bump: moved from unified_pipeline.py (TASK-13950.24 PR I2)`, and the old entry is deleted.
    - **BLE001:** no new per-file ignore. Each moved blind except gets `# noqa: BLE001 - moved from unified_pipeline.py (TASK-13950.24)`, and K deletes `pyproject.toml:1220` once `ruff check --select BLE001 unified_pipeline.py` is clean.
    - **Untouched:** `global_statements`, `test_mode_branch`, `test_quality` (no test is renamed) and `layering` (moves stay inside `core.RAG`).
21. **R21, the Sharing sentinel moves with the hidden reads (Task 3).**
    - `_literal_outer_pipeline_kwargs_reads` analyzes `hidden_kwarg_values`, which holds the 13 literal reads, and accepts a `FunctionDef` root.
    - A new assertion says `unified_rag_pipeline`'s own source reads no `kwargs` key.
    - The signature sentinel is untouched. The PR body calls out the security-test edit.
22. **R22, caller imports.** A caller imports `RagRunConfig, run_rag_pipeline` from `unified_pipeline` in the statement that imported `unified_rag_pipeline`, keeping the `_unified_pipeline.<name>` attribute form where it has one.
    - Test patches move to `run_rag_pipeline` on the same module object.
    - Import-statement counts don't change.

## Global Constraints

Copied into every implementer brief.
- **PRs:** target `dev`, branch from the latest `origin/dev`, one PR per slice in the table below, merged strictly in order (every PR from C on edits `unified_pipeline.py`).
  - Carry a `## Change summary`; the owner waived ADR-004 sign-off for TASK-13950.
  - Get a clean-subagent code review before merge.
  - Never arm auto-merge.
  - Compare non-required E2E results with dev before merging.
  - Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Git:** never `git stash`, `--no-verify`, force-push or `git clean -fdx`.
- **Python:** `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. There is no `python` on PATH. The shell is zsh, so `$VAR` lists don't word-split; use `${=VAR}`, arrays or `$(…)`.
- **Disk:**
  - Refuse to start a pytest run under 100 GiB free (`df -g /System/Volumes/Data`).
  - Each run uses `-p no:cacheprovider --basetemp "$TMPDIR/t24-<task>-<n>"`, deleted right after.
  - At most `-n 2`, one run at a time. The gate list runs serially.
  - No repo exports or extra worktrees.
- **Tests and network:**
  - `unset OPENAI_API_KEY ANTHROPIC_API_KEY`.
  - `HTTP_PROXY`/`HTTPS_PROXY=http://127.0.0.1:9`, `NO_PROXY=localhost,127.0.0.1`.
  - Never pass `-p pytest_asyncio.plugin` or `-p xdist.plugin`. Never delenv `PYTEST_CURRENT_TEST`.
  - A chat/LLM path with a commercial provider calls the real API unless the adapter is stubbed. Every golden case and every new test stubs it.
  - PostgreSQL runs set `TLDW_TEST_POSTGRES_REQUIRED=1`. The plugin auto-starts Docker `postgres:18`; don't set `TLDW_TEST_NO_DOCKER`.
- **Known local failures (not regressions):**
  - `test_runtime_package_discovery` (setuptools);
  - 3 in `test_merge_queue_gate_workflows` and 3 in `test_required_workflow_contracts`;
  - `test_rg_route_map_lint` and `test_route_auth_ratchet`, which flake under `-n 2` load;
  - `test_rag_profiles.py::…::test_flashrank_accepts_complete_non_listwise_local_bundle` (flashrank not installed).

  Confirm each against a dev baseline before dismissing it.
- **Bandit:** `uvx --offline bandit -q` on changed app and `Helper_Scripts` files. No new findings against the base. Moved code keeps its `# nosec` comments.
- **Backlog:** `PYTHONPATH=tools/backlog-py/src $PY -m backlog_py task …` only, with explicit ids. Stage with `git add -A backlog/tasks`.
- **Subagents:** implementers never dispatch subagents.
- **Baselines:** never widen any.
  - The only allowed new entries are R20's re-keys, each with the reason shape given there.
  - `function_size` is only ever `--tighten`ed; it gets no new entries.
- **Moves:**
  - Every rewrite and move is `rag_split apply` output, committed with its `.py` manifest.
  - A hand edit after `apply` must be a declared manifest item (`noqa`, `doc`), or `verify` fails.
  - A rebase re-runs `apply` on the new dev, never a hand merge.
  - Before opening a PR, run `gh pr list --base dev --search unified_pipeline` and check the open PRs' files. If one edits a span you move, merge or rebase that one first, #3480 excepted (R9).
- **Golden:** the golden JSON never changes after PR A, and no test is edited to pass. A golden diff is a behaviour change: find it, never regenerate.

## Review Focus

1. **An inert patch after a move.** A test patches a `unified_pipeline` name that moved code no longer reads there, and it passes vacuously.
   - Pinned by `test_pipeline_modules_read_patched_names_through_the_pipeline_module` and its seeded self-test (Task 2), and by `verify`'s `up.` check (V2).
   - For the entry point, pinned by `test_entry_point_patches_target_a_module_that_calls_it` (Task 11).
2. **Output drift on a branch the golden can't see.** The 5 flags listed under Measured "Untested flags" are turned on by no test but the golden.
   - Pinned by `test_golden_cases_cover_every_live_flag_both_ways` (Task 1). `verify`'s AST identity (V1-V7) covers every branch of every move, and its output is in every move PR.
3. **Owner and workspace scoping, or the sharing boundary, drifts.**
   - Golden G07 runs a cache miss and then a hit with a `workspace_id` kwarg and a scoped runtime user inside `scoped_context`, and pins the cache namespace. G19 pins explicit source selection and workspace filtering.
   - `Sharing/test_shared_workspace_chat_retrieval.py` is in the gate list (Task 1), and its sentinel is re-pointed, not weakened (R21).
   - Task 9 re-keys the `if user_id` predicate.
4. **Exceptions and `finally` order change.**
   - Golden G08 (`EarlyReturn`); G12a (the `:8806` dict is returned before `finally` writes `timings["total"]` into it); G12b (`pipeline_failed` on the response path); G13 (the security filter's own handler records its failure).
   - `test_unset_state_slot_reads_like_an_unbound_local` (Task 4).
   - V1 forbids moving `return`, `break` or a handler boundary.
5. **The rerun or a streaming caller changes arguments.**
   - `test_rerun_passes_the_frozen_argument_set` (Task 2) and the rerun record in golden G10.
   - `test_streaming_executor.py` and `test_rag_stream_parity.py` are in the gate. The streaming and sharing seams stay on the shim (R17).

## CI-equivalent checks

- **C1:** `tldw_Server_API/tests/lint`, `tldw_Server_API/tests/Helper_Scripts/test_rag_split.py` and `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`. Add the whole `tldw_Server_API/tests/CI` when a workflow, the classifier or a contract file changes.
- **C2 (move and rewrite PRs):** paste both outputs into the PR body.
  - `$PY -m Helper_Scripts.rag_split.rag_split verify <manifest> --base origin/dev` reports 0 differences;
  - `ruff check --select F821,F841,F401,F811,BLE001 <new and changed pipeline modules>` is clean.
- **C3:** Appendix D's gate list (`RAG_PIPELINE_GATE_TESTS`), serially, with `TLDW_TEST_POSTGRES_REQUIRED=1`: 0 failed apart from the known local failure, 0 skipped. Record the wall time.
- **C4:** the task's extra files.
- **C5:** bandit.

A run:
```bash
df -g /System/Volumes/Data   # refuse under 100 GiB free
unset OPENAI_API_KEY ANTHROPIC_API_KEY
export HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 NO_PROXY=localhost,127.0.0.1
GATE=$(PYTHONPATH=. $PY -c 'from Helper_Scripts.ci.path_classifier import RAG_PIPELINE_GATE_TESTS as t; print(" ".join(t))')
BT="$TMPDIR/t24-<task>-<n>"
PYTHONPATH=. TEST_MODE=true DISABLE_HEAVY_STARTUP=1 TLDW_TEST_POSTGRES_REQUIRED=1 \
  $PY -m pytest -q -p no:cacheprovider --basetemp "$BT" -rs ${=GATE}
rm -rf "$BT"
```

## PR slices and projections

The `unified_rag_pipeline` params stay 234 throughout (the shim keeps the signature). Lines are measured at each PR and recorded in its body; "~" values are estimates from dev spans.

| PR | Task | content | `unified_rag_pipeline` lines after | changed lines (est.) |
|---|---|---|---|---|
| A | 1 | golden + `rag-pipeline` job | 7,052 | ~1,300 + golden data |
| B | 2 | `rag_split` tool + structure pins | 7,052 | ~1,100 |
| C | 3 | `RagRunConfig`; read-only params → `rcfg.*` | ~7,053 | ~1,950 |
| D1 | 4 | `PipelineState`; `result` + 35 mutated params → `state.*` | ~7,054 | ~1,950 |
| D2 | 4 | hoist 18 closures + `EarlyReturn`; their free locals → `state.*` | ~6,595 | ~1,300 |
| D3 | 4 | remaining crossing locals → `state.*`; publish stage closures | ~6,600 | ~1,500 |
| E | 5 | post-processing (R27-R29) | ~6,135 | ~1,000 |
| F | 5 | verification + claims (R23-R26) | ~5,100 | ~2,150 |
| G | 6 | generation (R22) | ~4,520 | ~1,250 |
| H | 6 | rerank + post-retrieval (R19-R21) | ~3,710 | ~1,700 |
| I1 | 7 | processing + self-correct (R17-R18) | ~3,145 | ~1,200 |
| I2 | 7 | augment + filters (R15-R16) | ~2,560 | ~1,250 |
| J | 8 | retrieval + cache lookup (R13-R14) | ~1,775 | ~1,650 |
| K | 9 | scope, query prep, research (R07-R12) | ~785-820 (AC3 may be met here) | ~2,150 |
| L | 10 | prepare (R00-R06) + `run_rag_pipeline` split + docs | ~338, AC3 met (`run_rag_pipeline` ~175) | ~900 |
| M | 11 | 14 callers + test retargets | ~338 | ~900 |
| N | 12 | close-out | ~338 | backlog |

`function_size_baseline.json` ends at `unified_rag_pipeline` about 338/234; it stays listed because 338 > 300.

---

### Task 1: Golden-output test and the `rag-pipeline` required job (R2, R3, R18, R19, AC4, AC5)

**Files:**
- Create:
  - `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_golden.py` (shard `rag-new-unit-unified-pipeline`);
  - `tldw_Server_API/tests/RAG_NEW/unit/unified_pipeline_golden.json` (`git add -f`).
- Modify:
  - `Helper_Scripts/ci/path_classifier.py`, `.github/actions/detect-required-gate-changes/action.yml` and `.github/workflows/backend-required.yml`;
  - `tldw_Server_API/tests/CI/test_path_classifier.py` and `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`.

**Interfaces:**
- Produces, in `path_classifier.py`:
  - `RAG_PIPELINE_GLOBS: list[str]`;
  - `RAG_PIPELINE_GATE_TESTS: tuple[str, ...]` (Appendix D);
  - `RAG_PIPELINE_GATE_EXCLUDED: dict[str, str]` (path → reason, Appendix D);
  - `classify_paths(...)["rag_pipeline_changed"]`.
- Produces, in the golden module:
  - `GOLDEN_CASES: dict[str, GoldenCase]`, where `GoldenCase(kwargs: dict[str, Any], entry: Literal["pipeline", "batch"] = "pipeline", runs: int = 1, patches: tuple[str, ...] = ())`. `patches` names Appendix C's per-case patch sets.
  - `run_case(name: str, monkeypatch) -> dict` returns `{"result": [normalized per run], "trace": [spy calls]}`;
  - `normalize(result: Any) -> dict`;
  - `INERT_FLAGS = frozenset({"enable_graph_retrieval", "enable_streaming", "enable_batch", "enable_enhanced_chunking", "enable_parent_expansion"})`;
  - the write mode: with `TLDW_WRITE_RAG_GOLDEN=1`, `test_golden_case_matches` writes its case into the JSON instead of asserting. It runs under pytest, so RAG_NEW's autouse fixtures and `TEST_MODE` apply; there is no standalone CLI.

**The globs (R18), each with its reason:**
- `tldw_Server_API/app/core/RAG/**`, `tldw_Server_API/app/api/v1/endpoints/rag_*.py` and `tldw_Server_API/app/api/v1/schemas/rag_schemas*.py`: the pipeline, and the code that turns a request into its kwargs. `rag_*.py` matches `rag_unified.py` and `rag_health.py`, not `storage*.py`.
- `tldw_Server_API/tests/RAG_NEW/**`, `tldw_Server_API/tests/RAG/**`, `tldw_Server_API/tests/Sharing/test_shared_workspace_chat_retrieval.py` and `tldw_Server_API/tests/_plugins/postgres.py`: every gate file, their conftests, and the PostgreSQL fixture they use. Editing a gate test runs the gate.
- `.github/workflows/backend-required.yml` and `Helper_Scripts/ci/path_classifier.py`: editing the job runs the job (the chacha precedent).
- `pyproject.toml` and `uv.lock`: a dependency bump can change RAG output.

The job (R18) copies `chacha-owner-isolation` step for step: `if`, needs, the postgres service, checkout, setup-ffmpeg (portaudio), setup-python-deps and the env block. It does not set `MINIMAL_TEST_APP`, because the RAG integration tests use the full app. Its run is `pytest -q --disable-warnings -p pytest_asyncio.plugin -p pytest_timeout --timeout=180 --durations=15 --junitxml="$report" <Appendix D files>`, followed by the junit check with the same file list, and it has `timeout-minutes: 20`.

`backend-required` gets `rag-pipeline` in `needs`, plus the step "Require the RAG pipeline gate to have passed". That step accepts exactly `true:true:success|true:false:skipped|false:*:skipped) ;;` over `backend_changed:rag_pipeline_changed:result`.

- [ ] **Step 1: Write the failing CI tests.**
  - In `test_path_classifier.py`:
    - `test_rag_pipeline_paths_enable_the_rag_job` is parametrized over `core/RAG/rag_service/unified_pipeline.py`, `core/RAG/rag_schemas_unified.py`, `api/v1/endpoints/rag_unified.py`, `api/v1/schemas/rag_schemas_simple.py`, `tests/RAG_NEW/unit/test_hyde.py`, `tests/RAG/test_unified_schema_and_pipeline.py`, the Sharing test, `pyproject.toml` and `.github/workflows/backend-required.yml`, and asserts the flag is True.
    - `test_unrelated_backend_change_leaves_the_rag_job_off` uses `core/Chat/chat_service.py` and asserts `backend_changed` is True and `rag_pipeline_changed` is False.
  - In the contracts file:
    - add `rag-pipeline` to `ORIGINAL_JOB_NAMES` (after `chacha-owner-isolation`, before `backend-required`), `ORIGINAL_DEPENDENCIES` (`("changes",)`, and `backend-required` now needs it), `DIRECT_ADMISSION_JOBS` and `FETCH_DEPTH_CHECKOUTS`;
    - add `rag_pipeline_changed` to `CHANGE_CLASSIFIER_OUTPUTS`;
    - add the `extra_condition` branch, as for chacha;
    - add `_rag_pipeline_job_problems(workflow) -> list[str]`, modeled on `_owner_isolation_job_problems`. Its list is `path_classifier.RAG_PIPELINE_GATE_TESTS`, not a marker scan.
  - Then add these tests:
    - `test_backend_required_runs_the_rag_pipeline_gate`, which asserts `== []`;
    - mutations, each returning exactly one problem: every check and mutation test of `_owner_isolation_job_problems` (`:2181-2365`) under a `rag_gate` name. That covers a file dropped or commented out, `continue-on-error`, the gate not waiting, PostgreSQL not required, subset selection (`-k`/`-m`/`--deselect`), a missing junit check, a file the check does not list and a check that does not exit, plus `…_when_the_accepted_outcomes_change`;
    - `test_rag_pipeline_globs_cover_every_gate_test`: every `RAG_PIPELINE_GATE_TESTS` path matches `RAG_PIPELINE_GLOBS`;
    - `test_rag_pipeline_gate_lists_every_pipeline_test_file`: every `test_*.py` under `tests/RAG_NEW` and `tests/RAG` whose text contains `unified_pipeline` is in `RAG_PIPELINE_GATE_TESTS ∪ RAG_PIPELINE_GATE_EXCLUDED`, the excluded reasons are non-empty, and the two sets are disjoint.
- [ ] **Step 2: Run them, and expect FAIL** (no flag, no job): `PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t24-1-1" tldw_Server_API/tests/CI/test_path_classifier.py tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`
- [ ] **Step 3: Implement the classifier, the action output, the `changes` output, the job and the check step.**
  - Comment above the job: `# RAG pipeline (TASK-13950.24): the golden-output test and the pipeline tests, only when RAG paths change.`
  - Rerun Step 2; expect PASS, apart from the known local failures.
- [ ] **Step 4: Build the harness (Appendix C).**
  - **Patch points (B3).** A fake goes on `up` where `up` binds the name. Otherwise it goes on the source module of a function-local import (`monkeypatch.setattr(<source module>, "<name>", fake)`), exactly as Appendix C's patch sets list.
    - A function-local `from X import y` reads `X.y` each time it runs, and every move keeps the import function-local with the same absolute module (V2), so these patches stay live through every PR.
    - The live-network teardown guard in `conftest.py` fails any case that reaches an unpatched network collaborator.
  - **The spy.** Each fake records `(collaborator, method, normalized args)` into one `Recorder`.
    - Dataclass args are recorded by value with `dataclasses.asdict`, and pydantic args with `.model_dump(mode="json")`, recursively, through the same rules: documents become ids and floats are rounded to 6 places.
    - So `RetrievalConfig`, `RetrievalPlan`, `RerankingConfig` and `GenerationConfig` show every knob.
    - Only opaque handles (DB objects, callables, `credential_runtime`) become their type name.
  - Every case:
    - sets `enable_cache=False`, except G07 and G19;
    - deletes `RAG_GUARDRAILS_STRICT`, `RAG_NUMERIC_FIDELITY_BEHAVIOR`, `RAG_NLI_MODEL`, `RAG_NLI_MODEL_PATH`, `RAG_ADAPTIVE_TIME_BUDGET_SEC`, `RAG_RERANK_CALIB_BIAS`, `RAG_RERANK_CALIB_W_CE`, `RAG_MIN_RELEVANCE_PROB` and `SEARCH_MAX_ITERATIONS_*` from the env;
    - passes an explicit `date_range` wherever dates matter;
    - stubs every LLM, reranker, embedding and web collaborator.
  - Start each case's fakes from the RAG_NEW test named for it in Appendix C.
  - **`normalize`:**
    - a pydantic result uses `.model_dump(mode="json")` and a dict is kept;
    - add `metadata_keys: list(metadata)` and `timings_keys: sorted(timings)`, then drop the `timings` values and `total_time`;
    - map `feedback_id`, `trace_id` and `query_id` to `"<id>"`;
    - map values under an explicit `VOLATILE_KEYS` (start: `{"at", "timestamp"}`) to `"<t>"`. There is no suffix rule, because `time_budget_sec` and `adaptive_time_budget_sec` are echoed config and must stay visible;
    - round floats to 6 places.
- [ ] **Step 5: Probe determinism.** Run every case 3 times in one process and once in a second process; all normalized outputs must be equal. Fix a mismatch by adding the varying key to `VOLATILE_KEYS`, with a comment naming the stage that writes it, never by sorting documents or a blanket rule. Once Step 6's `test_golden_case_matches` exists, write the golden with `TLDW_WRITE_RAG_GOLDEN=1` on unmodified dev code.
- [ ] **Step 6: Write the remaining golden tests.**
  - `test_golden_case_matches[<case>]`, parametrized, asserts `run_case(case) == golden[case]` and shows a unified diff of the JSON on failure.
  - `test_golden_cases_cover_every_live_flag_both_ways` (R19).
  - `test_golden_detects_a_seeded_change`, parametrized. Each mutation must make one case differ (assert `!=`):
    - `up.coordinate_standard_result_evidence` wrapped to add a metadata key, against G02;
    - `up.CitationGenerator = None`, against G03;
    - `up._SEARCH_DEPTH_PRESETS["quality"]["top_k"] = 24`, against G14;
    - a fake reranker that reverses its order, against G03;
    - G04 run with `fts_level="chunk"`: the recorded `RetrievalConfig` must differ, which proves the spy sees knobs (I2).
- [ ] **Step 7: Mutate by hand** (PR evidence, not committed).
  - Change `:5225`'s `any(` to `all(`, run G04 and paste the failing diff into the PR body, then revert.
  - G04 reaches that line, because it sets `keyword_filter=["observatory", "telescope"]`. Its fixture doc `d2` contains only "observatory", so `all` drops it and the documents and trace change.
  - (`:6087` runs only for `two_tier` with the reranker missing, which no case uses.)
- [ ] **Step 8: Verify.**
  - Run C1 plus the whole `tldw_Server_API/tests/CI`, and C3 (Appendix D plus the golden).
  - Run `$PY Helper_Scripts/ci/test_quality_triage.py --enforce`; it must report no new offense.
  - Run `$PY Helper_Scripts/ci/check_shard_coverage.py` and confirm it covers the golden file.
- [ ] **Step 9: Push, then read the `rag-pipeline` job's wall time and `--durations` from PR A's own run** (it triggers, because the PR changes the classifier).
  - Record the time in the PR body and the ledger.
  - This is the first run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and explicit `-p`; local runs cannot rehearse it (Global Constraints).
  - If a listed file fails or skips on dev code in CI, move it to `RAG_PIPELINE_GATE_EXCLUDED` with the failure as its reason, and list it for Task 12's follow-ups. Never edit the test.
  - This never applies to the golden file. A CI divergence there means a collaborator is not faked, so fake it and regenerate the golden from dev in PR A.
- [ ] **Step 10: Commit** `test(rag): golden-output test and the path-gated rag-pipeline required job (TASK-13950.24 PR A)`.

### Task 2: The `rag_split` tool and the structure pins (R4, R6, R16, AC5)

**Files:**
- Create:
  - `Helper_Scripts/rag_split/__init__.py` and `Helper_Scripts/rag_split/rag_split.py` (Appendix A);
  - `tldw_Server_API/tests/Helper_Scripts/test_rag_split.py`;
  - `tldw_Server_API/tests/lint/test_rag_pipeline_structure_pins.py` (AST-only apart from Task 5's subprocess import tests; `_ast_cache.parse_tree` for app files).
- Modify:
  - `.github/workflows/backend-required.yml`: add both new files mid-list in "Enforce CI contracts and code ratchets", with the comment line `# The RAG split tool and structure pins (TASK-13950.24) guard each move PR.` in the comment block above the step, never inside the continued command;
  - `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`: add both files to `TEST_SEAM_GATE_SUITES`.

**Interfaces:**
- Produces Appendix A's CLI and manifest format.
- Produces, in the pins file:
  - `patched_pipeline_names(test_roots: Iterable[Path]) -> set[str]`, which collects patch targets on `unified_pipeline` in every form:
    - string targets in `patch`/`mock.patch`/`mocker.patch`/`monkeypatch.setattr`;
    - `setattr`/`monkeypatch.setattr`/`patch.object(<alias>, "<Name>", …)`, where `<alias>` is bound in the file by any import of `…rag_service.unified_pipeline`;
    - `<alias>.<Name> = …` assignments.

    It scans `tldw_Server_API/tests`, `tldw_Server_API/app/core/MCP_unified/tests` and `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`, prefiltered by the text `unified_pipeline`.
  - `pipeline_module_violations(pkg_dir: Path, patched: set[str]) -> list[str]`.
  - `RERUN_KEYWORDS: tuple[str, ...]`, the 79 keyword names of the `:8275` call in order, copied with `[k.arg for k in call.keywords]` at dev 5033a7689b.
  - `RERUN_COMPUTED: dict[str, str]`:

    ```python
    {"expand_query": "rerun_expand",
     "enable_cache": "False if adaptive_rerun_bypass_cache else enable_cache",
     "enable_generation": "bool(adaptive_rerun_include_generation)",
     "enable_post_verification": "False", "_adaptive_rerun": "True", "enable_claims": "False"}
    ```

- [ ] **Step 1: Write the failing tool self-tests** (each builds a mini pipeline module in `tmp_path` and uses `--base-dir`, so no git is needed):
  - `test_apply_then_verify_round_trips` (a rewrite, a hoist, a publish and two moves);
  - `test_rewrite_skips_shadowed_names` (a nested-def parameter, a comprehension variable, a lambda default and a class-body name, each spelled like a mapped name);
  - `test_rewrite_drops_mapped_names_from_nonlocal`;
  - `test_hidden_kwarg_reads_are_rewritten` (`kwargs.get("k", d)` becomes `rcfg.<g>.k` only when `d` equals the manifest's default);
  - `test_move_prefixes_module_globals_with_up`, `test_move_keeps_stdlib_names_bare` and `test_relative_import_level_is_bumped`;
  - `test_verify_rejects_a_changed_statement`, `…_a_missing_up_prefix`, `…_a_crossing_local`, `…_a_return_in_a_span`, `…_a_break_outside_its_loop` and `…_a_name_collision_with_rcfg_or_state`;
  - `test_shared_reports_a_local_read_before_assignment_in_another_cell`, `test_shared_ignores_a_temp_rebound_in_every_cell` and `test_shared_keeps_a_closure_local_rebound_only_before_its_def` (the `sgl` shape, A.3);
  - `test_verify_rejects_an_unbumped_relative_import_in_a_hoist` and `test_verify_rejects_a_stdlib_name_bound_differently` (V6, V9);
  - `test_new_manifests_verify_against_the_pr_base`:
    - it finds every manifest added since `git merge-base $BACKLOG_TASK_FORMAT_BASE HEAD` (falling back to `origin/dev` when the variable is unset) and runs `verify` against that merge base, expecting 0 differences;
    - with no new manifest it passes trivially;
    - it runs in the Enforce step, so R16's proof is gated, not just pasted;
  - `test_refuses_python_older_than_3_12`.
- [ ] **Step 2: Run them, and expect FAIL.**
- [ ] **Step 3: Implement `rag_split.py`** (Appendix A). Scope resolution uses `symtable`; definite assignment follows A.3.
- [ ] **Step 4: Write the pins.**
  - `test_patch_scan_finds_the_known_targets`: the scan contains `{"MultiDatabaseRetriever", "AnswerGenerator", "get_shared_cache", "create_reranker", "logger", "unified_rag_pipeline", "RetrievalConfig"}`.
  - `test_patch_scan_sees_each_form`: a seeded tmp test file with one of each form yields one name each.
  - `test_pipeline_modules_read_patched_names_through_the_pipeline_module`: `pipeline_module_violations(rag_service/pipeline, patched) == []`. A violation is any module-level or function-level binding (import, def, class, assignment) of a patched name, or any bare `Name` load of one, in a pipeline module. So is `<alias>.<patched name>`, where `<alias>` is bound by an import of a module other than `unified_pipeline`. It is vacuous until Task 3 creates the package.
  - `test_patched_name_import_in_a_pipeline_module_is_reported`: a seeded tmp module with `from ..database_retrievers import MultiDatabaseRetriever` gives one violation, and so does `from .. import database_retrievers as dr` followed by `dr.MultiDatabaseRetriever(...)`.
  - `test_rerun_passes_the_frozen_argument_set`: across `unified_pipeline.py` and `pipeline/**/*.py`, there is exactly one call whose func is `unified_rag_pipeline` or `up.unified_rag_pipeline`. Calls whose enclosing top-level function is `unified_batch_pipeline`, `simple_search` or `advanced_search` are excluded, nested defs included. The call has no positional or `**` arguments, and `[k.arg …] == RERUN_KEYWORDS`.

    A `NodeTransformer` then rewrites every `rcfg.<group>.<x>` and `state.<x>` chain inside each value expression to `Name(x)`. After it, each identity keyword's value is its own name, and each `RERUN_COMPUTED` value's `ast.unparse` equals the frozen text.
- [ ] **Step 5: Verify.** Run C1, `tests/CI`, and the two new files under the step's env (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1`; neither is async). Run bandit on `Helper_Scripts/rag_split/rag_split.py`.
- [ ] **Step 6: Commit** `test(rag): rag_split move tool and pipeline structure pins (TASK-13950.24 PR B)`.

### Task 3: `RagRunConfig`; read-only parameters read through `rcfg` (R5, R12, R21, AC1 part 1)

Projection: ~7,053 lines / 234 params.

**Files:**
- Create:
  - `tldw_Server_API/app/core/RAG/rag_service/pipeline/__init__.py` (a docstring only, no imports);
  - `…/pipeline/run_config.py`, which imports nothing from `unified_pipeline`;
  - `tldw_Server_API/tests/RAG_NEW/unit/test_rag_pipeline_types.py`, added to `RAG_PIPELINE_GATE_TESTS` and the job list;
  - `Helper_Scripts/rag_split/manifests/t03_config.py`.
- Modify:
  - `unified_pipeline.py`: its first statement and the rewrite;
  - `tldw_Server_API/tests/Sharing/test_shared_workspace_chat_retrieval.py` (R21);
  - the pins file;
  - `function_size_baseline.json` (`--tighten`);
  - `path_classifier.py` and `backend-required.yml` (the new gate file).

**Interfaces:**
- Produces:
  - 17 `@dataclass(frozen=True, slots=True)` classes: `RequestOptions`, `ScopeOptions`, `RuntimeHandles`, `RetrievalOptions`, `ProcessingOptions`, `QueryPrepOptions`, `CacheOptions`, `SafetyOptions`, `RerankOptions`, `GenerationOptions`, `CitationOptions`, `VerificationOptions`, `SelfCorrectOptions`, `ResearchOptions`, `ObservabilityOptions`, `ResilienceOptions` and `InertOptions`. Every field has the signature's annotation and default, and `query` has no default.
  - `@dataclass(frozen=True, slots=True) class RagRunConfig`, with fields `request, scope, runtime, retrieval, processing, query_prep, cache, safety, rerank, generation, citations, verification, self_correct, research, observability, resilience, inert`.
  - `RagRunConfig.from_kwargs(**kwargs: Any) -> RagRunConfig` and `RagRunConfig.from_call_locals(values: Mapping[str, Any]) -> RagRunConfig`. The second calls `from_kwargs(**{k: v for k, v in values.items() if k != "kwargs"}, **values["kwargs"])`.
  - `RagRunConfig.as_kwargs() -> dict[str, Any]`: 246 keys, shallow.
  - `hidden_kwarg_values(**kwargs: Any) -> dict[str, Any]`: exactly 13 literal `kwargs.get("<k>"[, d])` reads, with the defaults in Measured.
  - `FIELD_GROUP: Mapping[str, str]`: field → group, used by the tool and the pins.
- The body's first statement is `rcfg = RagRunConfig.from_call_locals(locals())`.
- Rewrite map: each of the 198 non-mutated parameters becomes `rcfg.<group>.<name>`, and each `kwargs.get("<k>"[, d])` becomes `rcfg.<group>.<k>`. The 35 mutated parameters stay bare until Task 4.

- [ ] **Step 1: Write the failing tests in `test_rag_pipeline_types.py`:**
  - `test_every_parameter_is_exactly_one_config_field`: the 233 named parameters plus the 13 hidden keys equal `FIELD_GROUP`'s keys. The 233 parameters fall into groups of R12's sizes, and the 13 hidden keys into R12's hidden-key groups.
  - `test_config_defaults_equal_the_signature_defaults`: compare each field default with the signature default, using `is` for `None`/bool and `==` otherwise.
  - `test_hidden_kwarg_values_reads_the_thirteen_keys`: `hidden_kwarg_values()` equals the 13 defaults.
  - `test_from_call_locals_keeps_every_value_by_identity`: give each of the 246 keys a distinct `object()` sentinel; `as_kwargs()[k] is v` for all of them.
  - `test_unknown_keys_are_ignored`: `from_kwargs(query="q", corpus="x")` has no trace of `corpus`.
  - `test_from_kwargs_honours_exactly_the_config_fields`: for every hidden key `k`, the value in `from_kwargs(query="q", **{k: sentinel})` equals `hidden_kwarg_values(**{k: sentinel})[k]`. Every `FIELD_GROUP` key is honoured, and no other key is.
  - `test_config_is_frozen`: assignment raises `FrozenInstanceError`.
  - `test_config_tracks_the_request_schema`:
    - shared names with `UnifiedRAGRequest.model_fields` = 194 (193 + `workspace_id`);
    - `REQUEST_ONLY` = the frozen 36 names;
    - `CONFIG_ONLY` = the frozen 52 names (40 pipeline-only + 12 hidden);
    - the shared defaults differ exactly on `{"sources", "enable_generation"}`.
  - In the pins file:
    - `test_config_is_built_first_from_the_call_locals`: the first statement after the docstring (`body[1]`; `body[0]` is the docstring at `:2328`) is exactly that assignment;
    - `test_body_reads_read_only_parameters_through_rcfg`: after that statement, no `Name` resolves to a non-mutated parameter (`symtable`);
    - `test_every_rcfg_read_names_a_config_field`: every `rcfg.<g>.<f>` chain in `unified_pipeline.py` and `pipeline/**` is a real group and field.
- [ ] **Step 2: Run them, and expect FAIL.**
- [ ] **Step 3: Write `run_config.py`.** Membership follows research A Appendix A's sub-config column. If the ledger copy is gone, regenerate it with `subconfigs.py` from the research directory. Group names map as `sources`→`scope`, `query`→`query_prep`, `selfcorrect`→`self_correct` and `agentic`→`research`.
- [ ] **Step 4: Write `t03_config.py`, then run `rag_split apply`, then `verify`; expect 0 differences.**
- [ ] **Step 5: Re-point the Sharing sentinel (R21).** `test_hidden_pipeline_kwargs_reads_are_explicitly_reviewed_absent` now asserts both:
  - `_literal_outer_pipeline_kwargs_reads() == expected_absent` (the source is now `hidden_kwarg_values`);
  - `_literal_outer_pipeline_kwargs_reads(inspect.getsource(unified_rag_pipeline)) == set()`.

  A new Sharing-owned assertion pins what the pipeline honours: `set(FIELD_GROUP) - set(inspect.signature(unified_rag_pipeline).parameters) == SHARED_RETRIEVAL_POLICY.reviewed_absent_kwarg_names`. So a new config field fails the Sharing test, not only the RAG one. The analyzer's parametrized self-tests keep passing.
- [ ] **Step 6: Verify.** `function_size --tighten`, then C1, C2, C3 (the golden unchanged) and C5.
- [ ] **Step 7: Commit** `refactor(rag): RagRunConfig; the pipeline reads read-only parameters through rcfg (TASK-13950.24 PR C)`.

### Task 4: `PipelineState` (R10, R13, R14; PRs D1, D2, D3)

Three PRs, each green alone. Projections: D1 ~7,054, D2 ~6,595, D3 ~6,600 lines / 234 params.

**Files:**
- Create: `…/pipeline/run_state.py`, plus manifests `t04a_state.py`, `t04b_hoist.py` and `t04c_shared.py`.
- Modify: `unified_pipeline.py`, `test_rag_pipeline_types.py`, the pins file and `function_size_baseline.json` (`--tighten`).

**Interfaces:**
- Produces:
  - `MUTATED_PARAMS: tuple[tuple[str, str], ...]`, the 35 `(group, name)` pairs of Measured, frozen.
  - `class PipelineState` with `__slots__: tuple[str, ...]`, `__init__(self, rcfg: RagRunConfig) -> None` and `__getattr__(self, name: str) -> Any`.
    - `__getattr__` raises `UnboundLocalError(f"cannot access local variable '{name}' where it is not associated with a value")` for a slot name, and `AttributeError` otherwise.
  - `class EarlyReturn(Exception)`, with the same docstring (D2).
  - The 18 methods of R14 (D2), with the same parameters as their closures.
- The statement after it (`body[2]`) is `state = _run_state.PipelineState(rcfg)`, and `unified_pipeline` imports `from .pipeline import run_state as _run_state` (R11).

- [ ] **Step 1 (D1): Write the failing tests.**
  - `test_mutated_parameters_are_frozen`: the names in `MUTATED_PARAMS` equal the test's frozen 35-name literal (Measured), and each pair's group is `FIELD_GROUP[name]`.
  - `test_unset_state_slot_reads_like_an_unbound_local`: `PipelineState(cfg).result` raises `UnboundLocalError`, `.nope` raises `AttributeError`, and `state.top_k == cfg.retrieval.top_k`.
  - In the pins file:
    - `test_state_slots_cover_every_state_attribute`: every `state.<a>` (Load/Store/Del) in `unified_pipeline.py` and `pipeline/**` is a slot or a method, and every `self.<a>` in `PipelineState` likewise;
    - `test_body_reads_no_parameter_after_the_prologue`: no `Name` resolves to any parameter after `body[2]`;
    - `test_mutated_parameters_are_never_read_from_rcfg`: no `rcfg.<g>.<x>` with `x` in `MUTATED_PARAMS` outside `PipelineState.__init__`.
- [ ] **Step 2 (D1):** create `run_state.py` with slots `("rcfg", *mutated, "result")`. Then `apply` and `verify` `t04a_state.py`: `result` and the 35 become `state.<name>`, and mapped names are dropped from `nonlocal`. Run C1-C3, `--tighten`, C5. Commit `refactor(rag): PipelineState holds the result and the mutated parameters (TASK-13950.24 PR D1)`.
- [ ] **Step 3 (D2):**
  - Run `rag_split shared --hoist <the 18 names>`. It lists the hoisted bodies' free function locals, such as the seven `retrieval_*` names, `optional_provider_failure_code`, `cache_instance` and `cache_setup_attempted`.
  - `t04b_hoist.py`: those locals become `state.<name>`, the 18 closures become methods (R14), and `_EarlyReturn` becomes `_run_state.EarlyReturn`. `run_state.py` gains the `up` import at this point.
  - Pin `test_hoisted_helpers_are_methods_not_closures`: none of the 18 names, nor `_EarlyReturn`, is defined inside `unified_rag_pipeline`, and each method exists.
  - Run `apply`, `verify`, C1-C3, `--tighten`, C5.
  - Commit `refactor(rag): hoist the pipeline's request helpers onto PipelineState (TASK-13950.24 PR D2)`.
- [ ] **Step 4 (D3):**
  - `t04c_shared.py` writes Appendix B's partition as `cells`, with spans at D3's base. `rag_split shared --manifest t04c_shared.py` lists every local that crosses a cell (R15), plus the closures used outside their cell. All of them become `state.<name>` (and `__slots__` entries), each such closure is published, and R26's four `locals()` names must not be among them.
  - The five published closures are `_execute_retrieval_variant`, `_analyze`, `_mark_claims_used`, `_run_claims_engine` and `_retrieve_for_claim`. A.3's def-order rule keeps `_analyze`'s import-bound `sgl` local to `run_claims` (bound at `:7574` before the def at `:7590`; its other bindings at `:6129` and `:7098` run in earlier cells), so `apply` never meets a mapped import binding. If `shared` ever reports an import-, `except`- or walrus-bound name, stop and report it; never hand-edit.
  - Run `apply` and `verify`; the PR body carries `shared`'s final run showing 0 crossings. Then C1-C3, `--tighten`, C5.
  - Commit `refactor(rag): shared pipeline locals live on PipelineState (TASK-13950.24 PR D3)`.

### Move recipe (Tasks 5-10)

1. Branch from the latest `origin/dev`, and check open PRs per Global Constraints.
2. Write `Helper_Scripts/rag_split/manifests/t<NN>_<pr>.py`. Its moves come from Appendix B, with spans at your base; the `# ========== X ==========` headers are the anchors. Leaves come before their wrapper. Each entry has a one-line `doc` and the `noqa` lines for its blind excepts.
3. `rag_split shared --manifest <m>` must report 0 crossing names. If it reports any, add them to the manifest's `rewrite` and as `PipelineState.__slots__` entries, and list them in the PR body.
4. `rag_split apply <m>`, then `rag_split verify <m> --base origin/dev` (0 differences), then C2's ruff.
5. Ratchets: `function_size --tighten`, plus the task's re-keys (R20).
6. Run C1, C3 (the golden unchanged), C4 and C5.
7. Commit `refactor(rag): move <group> into pipeline/<module>.py (TASK-13950.24 PR <x>)`. The PR body carries the verify and ruff output, the gate wall time, the measured `unified_rag_pipeline` lines/params and the re-keys.

### Task 5: Post-generation stages (PRs E, F; AC2 post-processing)

- [ ] **PR E:**
  - `pipeline/post_processing.py`: `finish_answer`, `store_and_observe` and `finalize_run` (the `finally` body). The `finally:` clause keeps one line, `await _post_processing.finalize_run(rcfg, state)`.
  - BLE001 `noqa` at `:8858` and `:8999`. Golden: G07, G12a, G12b, G14, G15 and G16 cover these stages.
  - Import-order pins, the first stage module's PR, added to the structure pins file:
    - `test_pipeline_modules_import_up_only_as_a_module`: no pipeline module imports a name from `unified_pipeline`;
    - `test_unified_pipeline_imports_pipeline_modules_as_modules`: every `ImportFrom` of `.pipeline` in `unified_pipeline.py` binds module names only, apart from `RagRunConfig` from `run_config`;
    - `test_pipeline_modules_import_in_either_order`: three subprocesses import first `…pipeline.run_state`, then `…pipeline.post_processing`, then `…unified_pipeline`; each then reads `unified_pipeline.unified_rag_pipeline` and exits 0.
  - C4: `test_unified_pipeline_focused.py` with `-k "cache or feedback"`, and `test_pipeline_post_verification_metadata.py`.
  - Projection ~6,135.
- [ ] **PR F:**
  - `pipeline/verification.py`: `check_citations_and_groundedness`, `run_numeric_fidelity`, and `run_post_verification` with the leaves `verify_answer` and `adaptive_rerun`.
  - `pipeline/claims.py`: `run_claims` with the leaves `prepare_claims_job` and `execute_claims`. `_analyze`, `_mark_claims_used`, `_run_claims_engine` and `_retrieve_for_claim` are published.
  - `env_read` re-keys: `pipeline/verification.py` (`RAG_ADAPTIVE_TIME_BUDGET_SEC`) and `pipeline/claims.py` (`RAG_NLI_MODEL(_PATH)`), with the measured reads.
  - `test_rerun_passes_the_frozen_argument_set` now finds the call in `verification.py`.
  - C4: `test_post_verifier.py`, `test_rag_provider_credentials.py` (BYOK `mark_used` ordering), `test_generation_executor.py` and `test_strict_extractive_and_citations.py`.
  - Projection ~5,100.

### Task 6: Generation; rerank and post-retrieval (PRs G, H; AC2 generation, rerank)

- [ ] **PR G:**
  - `pipeline/generation.py`: `prepare_generation`, and `run_generation` with the leaves `generate_standard_answer`, `synthesize_multi_turn`, `finish_generation` and `answer_when_gated`. The `rag.generation` span stays in `run_generation`.
  - BLE001 `noqa` at `:7373`. Golden: G02, G12a, G12b (`build_retrieval_only_result` is in `prepare_generation`) and G15.
  - C4: `test_pipeline_generation_controls.py`, `test_pipeline_two_tier_gate.py`, `test_unified_pipeline_structured_writer.py` and `test_batch_round2_flags.py`.
  - Projection ~4,520.
- [ ] **PR H:**
  - `pipeline/rerank.py`: `run_rerank` with the leaves `select_reranker`, `apply_rerank` and `calibrate_rerank`. The `rag.rerank` span stays in `run_rerank`.
  - `pipeline/post_retrieval.py`: `run_web_fallback`, `explain_and_expand_sources` and `build_citations_and_chains`.
  - `env_read` re-key: `pipeline/rerank.py` (3). Golden: G03, G09, G11, G14 and G15.
  - C4: `test_learned_fusion_simple.py`, `test_evidence_chain_provider_isolation.py` and the reranker files `tests/RAG_NEW/unit/test_reranker*.py`.
  - Projection ~3,710.

### Task 7: Document stages (PRs I1, I2)

- [ ] **PR I1:**
  - `pipeline/processing.py`: `process_tables_and_vlm` and `run_evidence_accumulation`.
  - `pipeline/self_correct.py`: `grade_documents` and `run_rewrite_loop`.
  - Golden: G11, G14. C4: `tests/RAG/test_doc_researcher_features.py` (already in the gate) and `tests/RAG_NEW/unit/test_agentic_chunker_vlm.py`.
  - Projection ~3,145.
- [ ] **PR I2:**
  - `pipeline/augment.py`: `augment_passages` and `run_gap_analysis` (which includes the keyword filter). `_execute_retrieval_variant` is read as `state._execute_retrieval_variant`.
  - `pipeline/filters.py`: `apply_content_filters` and `apply_security_filter`.
  - `keyword_identity` re-key (R20). Golden: G04, G06, G10, G11, G13, G19.
  - C4: `test_prf_and_spans.py` and `test_unified_pipeline_security_cache_helpers.py`.
  - Projection ~2,560.

### Task 8: Retrieval (PR J; AC2 retrieval)

- [ ] **PR J:**
  - `pipeline/retrieval.py`: `lookup_retrieval_cache`, and `run_retrieval` with the leaves `prepare_retriever`, `merge_hyde_and_variants`, `run_prf`, `run_decomposition` and `record_retrieval_diagnostics`.
  - The `rag.retrieval` span and its 79-line handler stay in `run_retrieval`. The decomposition's `Semaphore`/`create_task`/`wait` block and its `finally` cancellation move whole in `run_decomposition`.
  - `up.MultiDatabaseRetriever` keeps `rag_unified.py:696-713`'s runtime override live.
  - Golden: G01, G04, G05, G07, G19.
  - C4:
    - `test_unified_pipeline_decomposition.py`, `test_hyde.py`, `test_knowledge_source_retrieval_coverage.py` and `test_retrieval_plan_usage.py`;
    - `tests/DB_Management/test_media_db_api_imports.py`, which reloads `unified_pipeline`; run it with `TLDW_TEST_POSTGRES_REQUIRED=1`.
  - Projection ~1,775.

### Task 9: Scope, query preparation and research (PR K)

- [ ] **PR K:**
  - `pipeline/scope.py`: `resolve_sources`, which may `raise EarlyReturn`.
  - `pipeline/query_prep.py`: `clarify_and_spellcheck`, `run_query_classification` (which keeps `_prefetch_routed_external_docs` nested), `apply_guardrail_defaults`, `run_query_expansion`, `route_and_weight` and `prepare_hyde_and_temporal_filters`.
  - `pipeline/research.py`: `run_research_loop`.
  - Re-keys:
    - `scope_predicate`: `--write-baseline` turns `…unified_pipeline.py::unified_rag_pipeline if user_id` into `…pipeline/query_prep.py::run_query_expansion if user_id`. The baseline line count is unchanged; hand-check the diff.
    - `env_read`: `pipeline/research.py` (1) and `pipeline/query_prep.py` (2).
    - BLE001 `noqa` at `:3431`, `:3465` and `:3689`, then delete `pyproject.toml`'s `unified_pipeline.py` BLE001 line once `ruff check --select BLE001` on that file is clean.
  - Golden: G04, G08, G09, G17.
  - C4: `test_pre_retrieval_clarification_*.py`, `test_quick_wins_spell_check_compat.py`, `test_temporal_filters.py`, `test_unified_pipeline_query_expansion.py`, `test_search_depth_explicit_overrides.py` and `test_research_agent.py`.
  - Measure and record `unified_rag_pipeline`'s lines (projected ~785-820, at the 800 edge). AC3 is met at Task 10 either way.

### Task 10: The entry point (PR L; AC1 part 2, AC2, AC3)

Projection: `unified_rag_pipeline` ~338 lines / 234 params, and `run_rag_pipeline` ~175 lines / 1 param.

**Files:**
- Create `pipeline/prepare.py`: `prepare_request` (R00 and R01's remainder), `apply_search_depth_presets` (R03) and `init_run` (R04-R06's remainder).
- Modify `unified_pipeline.py`, the pins file, `function_size_baseline.json`, `Docs/Code_Documentation/RAG-Developer-Guide.md` (a "Pipeline layout" section: the modules, `RagRunConfig`, `run_rag_pipeline`, and the `up.`/`state`/`rcfg` rules) and its `Docs/Published` mirror.

**Interfaces:**
- Produces `async def run_rag_pipeline(rcfg: RagRunConfig) -> UnifiedPipelineResult` in `unified_pipeline.py`, defined right after the shim.
- The shim's body is its docstring plus `return await run_rag_pipeline(RagRunConfig.from_call_locals(locals()))`, and the docstring names `run_rag_pipeline`.
- `run_rag_pipeline`'s body is today's body minus its docstring (`body[0]`) and the `rcfg = …` statement (`body[1]`), in place:
  - `state = _run_state.PipelineState(rcfg)`;
  - three `prepare` calls, with R02 inline between them;
  - the `try` of stage calls and the inline ladder (R15);
  - the `finally` call;
  - the inline tail.

- [ ] **Step 1: Write the pins.**
  - `test_orchestrator_has_no_nested_defs`: no `FunctionDef`, `AsyncFunctionDef`, `Lambda` or `ClassDef` inside `run_rag_pipeline`.
  - `test_shim_only_builds_the_config`: `unified_rag_pipeline`'s body is the docstring plus that one `return`.
  - Re-point Task 3's and Task 4's body pins at `run_rag_pipeline`, whose `body[0]` is `state = …` (on the shim, `body[1]` is now the `return`). The "no parameter read after the prologue" pin then checks that `run_rag_pipeline` reads no signature parameter name, only `rcfg` and `state`.
  - Extend Task 5's `test_pipeline_modules_import_in_either_order` with a `…pipeline.prepare`-first subprocess.
- [ ] **Step 2: Move the prologue.** Use `t10_entry.py` with the recipe's Steps 2-4.
- [ ] **Step 3: Split the entry point.**
  - The manifest's `entry_split` item makes `run_rag_pipeline` and the shim. `verify` V8 checks that the moved body is unchanged apart from the removed first statement.
  - `function_size --tighten`: the entry becomes ~338/234, and `run_rag_pipeline` needs no entry (under 300).
- [ ] **Step 4: Update the guide.** Run `Helper_Scripts/refresh_docs_published.sh` and commit the `Docs/Published` diff in this PR.
- [ ] **Step 5: Verify.**
  - Run C1-C5.
  - Run `test_rag_integration.py:59` and `test_rag_provider_credentials.py:1173` (`inspect.signature` checks); both are in the gate.
  - Measure AC3 with the function-size ratchet's counter.
- [ ] **Step 6: Commit** `refactor(rag): run_rag_pipeline(rcfg) orchestrator; unified_rag_pipeline is a kwargs shim (TASK-13950.24 PR L)`.

### Task 11: Callers pass the config (PR M; AC1, R17, R22)

Projection: unchanged, ~338 / 234.

**Files:**
- Modify the 14 sites:
  - `api/v1/endpoints/rag_unified.py` `:922`, `:928` and `:1614`;
  - `api/v1/endpoints/slides.py:2750`;
  - `core/Data_Tables/jobs_worker.py:856`;
  - `core/Evaluations/embeddings_abtest_service.py:683`, `eval_runner.py:814`, `recipe_runs_jobs_worker.py:400` and `recipes/rag_answer_quality_execution.py:426`;
  - `core/MCP_unified/modules/implementations/rag_module.py:1002` and `slides_module.py:1649`;
  - `core/Scheduled_Tasks/recurring_question_rag_adapter.py:215`;
  - `core/Workflows/adapters/rag/search.py:131`;
  - `services/quality_eval_scheduler.py:81`.
- Modify every test that patches `unified_rag_pipeline` to intercept one of these callers. That is about 45 of the roughly 60 patch lines; patches that intercept batch, streaming, sharing or the rerun stay.

**Interfaces:**
- Each site becomes `await run_rag_pipeline(RagRunConfig.from_kwargs(<exactly today's arguments>))`, through the caller's existing import statement or `_unified_pipeline.` attribute (R22).
  - The endpoint and MCP sites pass their `pipeline_kwargs` (transport's payload) unchanged.
  - `recurring_question_rag_adapter.py:215` becomes `RagRunConfig.from_kwargs(**request.model_dump())`. Its 36 request-only keys are ignored, exactly as the shim ignores them, and `workspace_id` lands in `rcfg.scope.workspace_id`.
- Produces, in the pins file, `PIPELINE_ENTRY_PATCH_ALLOWLIST: dict[tuple[str, str], str]`, mapping (test file, function) → reason.
- A retargeted fake is `async def fake(rcfg): kwargs = rcfg.as_kwargs(); …`. An assertion that a key is absent becomes an assertion on that field's default value.

- [ ] **Step 1: Write the entry-point pins.**
  - `test_entry_point_patches_target_a_module_that_calls_it`: for every test patch of `unified_rag_pipeline` or `run_rag_pipeline` on a caller module X (any form the Task 2 scanner knows), X's source must load that name. Prove it with `test_entry_point_patch_on_a_module_that_dropped_the_call_is_reported`, a seeded tmp module.
  - That rule is blind to the roughly 40 patches on `unified_pipeline` itself, which still loads the name (batch, `simple_search`, `advanced_search`, the rerun). So `test_pipeline_entry_patches_are_allow_listed` collects every `(test file, enclosing test or fixture function)` that patches `unified_pipeline.unified_rag_pipeline`, by string target or alias. That set must equal `PIPELINE_ENTRY_PATCH_ALLOWLIST`'s keys exactly, so both new and stale entries fail.
    - Each value is a frozen reason from {`batch`, `streaming`, `sharing`, `rerun`, `simple_search`}.
    - Step 2 fills the list from its classification table, after retargeting the patches that meant a migrated caller (for example `rag_unified`'s non-streaming path).
- [ ] **Step 2: Migrate the 14 sites and the tests.** The PR body carries a table: every patch line → retargeted, or kept with its reason (batch / streaming / sharing / rerun).
- [ ] **Step 3: Verify.**
  - Run C1-C3 and C5.
  - C4, the callers' tests, with `TLDW_TEST_POSTGRES_REQUIRED=1`:
    - `app/core/MCP_unified/tests/test_rag_module.py` and `test_slides_module.py`;
    - `tests/Evaluations/{test_embeddings_abtest_retrieval,test_rag_pipeline_runner,test_recipe_runs_jobs_worker,test_evaluations_stage2_user_isolation_and_usage_accounting}.py` and `tests/Evaluations/unit/test_eval_runner.py`;
    - `tests/Workflows/adapters/test_rag_adapters.py`, `tests/Slides/test_slides_api.py` and `tests/DataTables/test_data_tables_worker.py`;
    - `tests/AuthNZ/integration/test_rag_media_permissions_claims.py`, `tests/AuthNZ/integration/test_auth_principal_media_rag_invariants.py` and `tests/AuthNZ_SQLite/test_rag_media_permissions_sqlite.py`;
    - `tests/Sharing/test_cross_user_access.py`.
  - Run `test_endpoint_monkeypatch_ratchet` (its count must not rise; `--tighten` if it falls).
- [ ] **Step 4: Commit** `refactor(rag): callers pass a RagRunConfig to run_rag_pipeline (TASK-13950.24 PR M)`.

### Task 12: Close-out (AC1-AC5, DoD)

**Files:** backlog-py only: `backlog/tasks/task-13950.24 - …md` and the follow-up tasks.

- [ ] **Step 1: Check each AC on dev,** recording the command and the numbers.
  - **AC1:** the 14 sites. An AST scan of `tldw_Server_API/app/**/*.py` for `Call` nodes whose func is named `unified_rag_pipeline` (Name or Attribute) finds only `unified_pipeline.py`'s batch, `simple_search` and `advanced_search` callers and `pipeline/verification.py`'s rerun. A text grep would also match docstrings and READMEs. R17's seams call the function by reference.
  - **AC2:** the module list and `test_orchestrator_has_no_nested_defs`.
  - **AC3:** the ratchet's measurement (~338) and the baseline diff from 7,063.
  - **AC4:** the golden is unchanged since PR A (`git log` on the JSON shows one commit), and the gate is green on dev. Also run the whole `tests/RAG_NEW` and `tests/RAG` once on dev, serially with `TLDW_TEST_POSTGRES_REQUIRED=1`, and record the counts against a run at 5033a7689b's results. That covers AC4's "existing RAG_NEW suite" beyond the gate's 48 files.
  - **AC5:** the contract tests of Tasks 1 and 2.
- [ ] **Step 2: Amend the ACs** with `task edit`.
  - AC1: "`run_rag_pipeline(rcfg)` accepts one typed `RagRunConfig` (17 grouped sub-configs); `unified_rag_pipeline` keeps its kwargs signature as a shim that builds the config; the 14 direct external call sites in 12 files pass the config; the streaming and shared-workspace injection seams and transport's signature filter keep the shim (R17)."
  - AC2: "…no nested defs in the orchestrator (`run_rag_pipeline`)."
  - AC5: add "…or backend-required's path-gated `rag-pipeline` job, whose result backend-required requires, with workflow-contract assertions".
- [ ] **Step 3: File follow-ups,** each depending on TASK-13950.24:
  - the `enable_generation` default mismatch (`True` vs request `False`);
  - remove or wire the 9 never-read parameters, the 7 forward-only ones and the 4 dead writes at `:3928-3934`;
  - reject unknown keyword arguments in `RagRunConfig.from_kwargs`;
  - the adaptive rerun resets 154 knobs to defaults (owner decision);
  - an honest return type for `run_rag_pipeline`, which returns 3 shapes today;
  - an invalid `sensitivity_level` skips the security filter (it fails open), recording only `Security filter failed: …`; G13 pins today's behaviour;
  - one BYOK adapter for the five near-identical ones;
  - migrate the streaming and shared-workspace seams to `RagRunConfig` (R17);
  - read pipeline env through `config_sections` into `RagRunConfig`;
  - replace `locals()` in `adaptive_rerun` with explicit initialisers;
  - the 3 gate exclusions (Appendix D) and anything Task 1 Step 9 added;
  - update TASK-13519's notes: its reranker and expansion changes now target `pipeline/rerank.py` and `pipeline/retrieval.py`.
- [ ] **Step 4: Record** in the notes and the final summary: the rulings, the PR numbers, the measured lines per PR, the gate wall times (local and CI), the bandit results and the known skips. Set the task to Done, and stage with `git add -A backlog/tasks`.

## Self-review

- **Spec coverage:**

  | AC | Tasks |
  |---|---|
  | AC1 | 3 (config), 10 (entry and shim), 11 (callers), 12 (amendment, R17) |
  | AC2 | 5-10 (modules), 10 (`test_orchestrator_has_no_nested_defs`) |
  | AC3 | 10 (~338; K may already be ≤800); `--tighten` in every PR |
  | AC4 | 1 (golden + gate), every PR's C3, 12 (one full RAG_NEW + RAG run) |
  | AC5 | 1 (job + contracts), 2 (pins, tool self-tests and `test_new_manifests_verify_against_the_pr_base` in the Enforce step + `TEST_SEAM_GATE_SUITES`), 12 (amendment) |
  | DoD | 12; docs in 10; bandit in every PR |

- **Names across tasks:**
  - `rcfg`, `state`, `up`, `_run_state`, `RagRunConfig`, `PipelineState`, `EarlyReturn`, `MUTATED_PARAMS`, `FIELD_GROUP`, `hidden_kwarg_values`, `from_kwargs`, `from_call_locals`, `as_kwargs`, `run_rag_pipeline`, `RAG_PIPELINE_GATE_TESTS`, `TLDW_WRITE_RAG_GOLDEN` and `PIPELINE_ENTRY_PATCH_ALLOWLIST` are each defined once (Tasks 1-4, 10, 11) and used with the same spelling after.
  - The golden ids are G01-G11, G12a, G12b and G13-G19 everywhere (R2, R16, Review Focus, Tasks 5-9, Appendix C).
  - "First statement" always means the first statement after the docstring (`body[1]`).
  - The stage function names are Appendix B's. None collides with a `unified_pipeline` module name (checked by script; `accumulate_evidence` was renamed `run_evidence_accumulation` for that reason).
- **Fails-before:**
  - Task 1's classifier and contract tests;
  - Task 2's tool self-tests (V6, V9 and A.3's def-order rule included) and `test_patched_name_import_in_a_pipeline_module_is_reported`;
  - Task 3's types tests;
  - Task 4's `test_unset_state_slot_reads_like_an_unbound_local`, `test_mutated_parameters_are_frozen` and `test_hoisted_helpers_are_methods_not_closures`;
  - Task 10's shim and orchestrator pins;
  - Task 11's seeded entry-point test, and `test_pipeline_entry_patches_are_allow_listed` before Step 2 fills the list;
  - Task 1's `fts_level` seeded check, against a spy that records type names.
- **Passes before and after:** the golden (every PR from B on), `test_rerun_passes_the_frozen_argument_set`, the Sharing signature sentinel and, from Task 3, its `FIELD_GROUP` assertion.
- **Proportion:** the plan holds signatures, test names and tables; the only code is the rerun's frozen values.

---

## Appendix A: `Helper_Scripts/rag_split/rag_split.py`

**A.1 CLI** (`$PY -m Helper_Scripts.rag_split.rag_split <cmd>`; stdlib only; exits 2 on Python < 3.12):
- `shared [--manifest M] [--hoist N1,N2,…] (--base REV | --base-dir DIR)`: prints the crossing names for the manifest's cells (A.3), or the free function locals of the named closures. It writes nothing.
- `apply M`: edits the working tree from manifest items, as text edits at AST positions, applied bottom-up per file:
  - `rewrite`, `hidden` and `publish` in place;
  - `hoist` into `run_state.py`;
  - `moves` and `entry_split`.

  It creates a pipeline module with its header if absent, and appends functions in manifest order. It fails closed on a mapped name bound by `except … as`, a walrus, `import`, `global` or a class body, on a default or decorator of a hoisted def that reads a function local, and on any V8 collision.
- `verify M (--base REV | --base-dir DIR)`: A.2's checks against the head working tree. It prints one line per difference and exits 1 if there are any.

**A.2 verify checks:**
- **V1:** each move span is whole statements of one block. It contains no `return`, no `yield`, no `break`/`continue` outside a loop it contains, and no `nonlocal`/`global`.
- **V2:** each moved function body equals the base span's AST (`ast.dump(..., include_attributes=False)`) after exactly these transforms:
  - every `Name` resolving to a `unified_pipeline` module global becomes `up.<name>`, except stdlib-import names and `pipeline`-package names (R11);
  - each relative `ImportFrom` level gains 1;
  - inner leaves become their calls;
  - the function may start with a docstring.
- **V3:** in head, the base span is exactly one statement, `[await] _<module>.<func>(rcfg, state)`. The function is `async def` iff the span awaits at its own scope, and its signature is `(rcfg: RagRunConfig, state: PipelineState) -> None`.
- **V4:** no crossing. A span's live-in names are within `{rcfg, state}`, builtins and module globals, and no name bound in the span is live-in at another cell (A.3).
- **V5 (`rewrite`/`hidden`/`publish`):** head equals the base with only these changes:
  - each mapped `Name` (Load/Store/Del) resolving to the function scope becomes its target;
  - mapped names are dropped from `nonlocal` (an empty statement is removed);
  - `kwargs.get("<k>"[, d])` becomes `rcfg.<g>.<k>` where `d` equals `FIELD_GROUP`'s field default;
  - `state.<n> = <n>` follows each published def.
- **V6 (`hoist`):** each method body equals the closure body after the V5 map, then `state`→`self` and `rcfg`→`self.rcfg`, then every V2 transform: globals to `up.` and the relative-import level. That matters because `_get_cache_instance`'s `from .advanced_cache import register_semantic_cache` (`:2906`) sits under a swallowing `except ImportError`. The closure is gone from the base function.
- **V7:** every other line of every touched file equals the base, apart from the import lines the manifest declares.
- **V8:** `rcfg`, `state`, `up` and `_run_state` are unused as names in the base function. After the manifest's moves, the `entry_split` body equals the base body minus its first statement after the docstring.
- **V9:** every bare stdlib-import name loaded in a pipeline module is bound there by an import equal to `unified_pipeline`'s binding of that name: the same module, attribute and alias. So `from datetime import datetime` cannot become `import datetime`, a mismatch ruff cannot see.

**A.3 cells and liveness:**
- **Cells.** The cells are each move's span, plus the residual function (everything not in a span).
- **Live-in.** A name is live-in at a cell if the cell may read it before a definite assignment, with the standard rules:
  - an `if` merges the intersection of its branches;
  - a loop body may run zero times;
  - `try` handlers start from the state before the `try`;
  - `raise`, `return`, `break` and `continue` end a path;
  - a nested def, lambda or comprehension reads its free names at its definition point.
- **Crossing.** A crossing name is one live-in at one cell and bound in another.
  - So is a free local `n` of a closure referenced outside its defining cell C, unless both hold:
    - C definitely assigns `n` before the def;
    - no other cell binds `n` at a line after the def.
  - Cells run in source order, so a binding in an earlier cell cannot reach a closure that captures C's later binding. This keeps `_analyze`'s `sgl` local.
- **Scope.** Resolution uses `symtable`, and the self-tests cover comprehensions, lambdas, class bodies and `nonlocal`.

**A.4 manifests:** `Helper_Scripts/rag_split/manifests/t<NN>_<pr>.py` holds one `MANIFEST = {...}` literal, read with `ast.literal_eval` and never imported (`.gitignore` ignores `*.json`). Its keys:
- `base`: the dev sha;
- `rewrite`: `{name: target}`;
- `hidden`: `{key: [group, default]}`;
- `publish`: `[name]`;
- `hoist`: `{closure: method}`;
- `moves`: `[{"name", "module", "span": [a, b], "doc", "noqa": [line]}]`;
- `cells`: `[[a, b]]`;
- `entry_split`: `true`.

## Appendix B: the partition (dev 5033a7689b; `try[i]` = the i-th statement of the `:3000` try body)

| module (PR) | function: span | lines |
|---|---|---|
| `post_processing` (E) | `finish_answer`: try[80..85] 8521-8705 · `store_and_observe`: try[86..88] 8708-8788 · `finalize_run`: finally 8818-9020 | 185 · 81 · 203 |
| `verification` (F) | `check_citations_and_groundedness`: try[71..76] 7480-7570 · `run_numeric_fidelity`: try[78] 8002-8151 · `run_post_verification`: try[79] 8155-8517, wrapper ~39, with leaves `verify_answer` 8164-8266 and `adaptive_rerun` 8270-8492 | 91 · 150 · 39 + 103 + 223 |
| `claims` (F) | `run_claims`: try[77] 7571-7999, wrapper ~113, with leaves `prepare_claims_job` 7654-7792 and `execute_claims` 7793-7971 | 113 + 139 + 179 |
| `generation` (G) | `prepare_generation`: try[66..69] 6895-6948 · `run_generation`: try[70] 6950-7476, wrapper ~95, with leaves `generate_standard_answer` 7007-7065, `synthesize_multi_turn` 7067-7312, `finish_generation` 7313-7374 and `answer_when_gated` 7408-7476 | 54 · 95 + 59 + 246 + 62 + 69 |
| `rerank` (H) | `run_rerank`: try[57] 6078-6452, wrapper ~70, with leaves `select_reranker` 6110-6248, `apply_rerank` 6249-6330 and `calibrate_rerank` 6333-6419 | 70 + 139 + 82 + 87 |
| `post_retrieval` (H) | `run_web_fallback`: try[58] 6456-6535 · `explain_and_expand_sources`: try[59..61] 6538-6686 · `build_citations_and_chains`: try[62..65] 6688-6891 | 80 · 149 · 204 |
| `processing` (I1) | `process_tables_and_vlm`: try[51..52] 5504-5684 · `run_evidence_accumulation`: try[53..54] 5687-5803 | 181 · 117 |
| `self_correct` (I1) | `grade_documents`: try[55] 5807-5894 · `run_rewrite_loop`: try[56] 5898-6075 | 88 · 178 |
| `augment` (I2) | `augment_passages`: try[42..44] 4915-5050 · `run_gap_analysis`: try[45..46] 5053-5231 | 136 · 179 |
| `filters` (I2) | `apply_content_filters`: try[47..49] 5234-5360 · `apply_security_filter`: try[50] 5363-5501 | 127 · 139 |
| `retrieval` (J) | `lookup_retrieval_cache`: try[40] 4123-4183 · `run_retrieval`: try[41] 4186-4913, wrapper ~128, with leaves `prepare_retriever` 4212-4351, `merge_hyde_and_variants` 4357-4504, `run_prf` 4506-4561, `run_decomposition` 4564-4764 and `record_retrieval_diagnostics` 4766-4816 | 61 · 128 + 140 + 148 + 56 + 201 + 51 |
| `scope` (K) | `resolve_sources`: try[0..9] 3001-3181 minus D2's four hoisted defs (try[2], [3], [8], [9]), so try[0..1] and try[4..7] | ~77 |
| `query_prep` (K) | `clarify_and_spellcheck`: try[10..12] 3184-3237 · `run_query_classification`: try[13..18] 3239-3466 · `apply_guardrail_defaults`: try[20..29] 3696-3759 · `run_query_expansion`: try[30..31] 3762-3871 · `route_and_weight`: try[32..36] 3874-3991 · `prepare_hyde_and_temporal_filters`: try[37..39] 3994-4119 | 54 · 228 · 64 · 110 · 118 · 126 |
| `research` (K) | `run_research_loop`: try[19] 3471-3691 | 221 |
| `prepare` (L) | `prepare_request`: top statements 2353-2549 minus hoisted · `apply_search_depth_presets`: 2601-2652 · `init_run`: 2655-2998 minus hoisted | ~107 · 52 · ~83 |
| inline in `run_rag_pipeline` | R02 2552-2598, the ladder 8790-8814, the `finally` call, the tail 9022-9067 | 47 · 25 · 1 · 46 |

Every function is ≤ 246 lines, and the orchestrator makes 34 stage calls.

## Appendix C: golden cases

Every case sets `enable_cache=False`, except G07 and G19. The fakes start from the RAG_NEW file in brackets.

**Base patch set (B3).** An autouse fixture in the golden module installs these for every case:
- **On `up`:**
  - `MultiDatabaseRetriever`, returning a fixed 6-document fixture. Doc `d2` contains only "observatory" (Step 7); the fixture also has a PII doc, an injection doc and parent/chunk-index metadata.
  - `AnswerGenerator`.
  - `create_reranker`, with a deterministic score map. Reranking is on by default, the default strategy is flashrank, and flashrank is not installed locally.
- **On source modules (function-local imports):**
  - `LLM_Calls.Summarization_General_Lib.analyze` (scripted; `:5059`, `:6129`, `:7098`, `:7574`, `:8877`);
  - `core.config.load_and_log_configs` (a fixed dict; `:4000`, `:5063`, `:6130`, `:6213`, `:6226`, `:8876`);
  - `core.config.rag_strict_extractive`, which returns `False` (`:3728`);
  - `RAG.rag_service.generation_defaults.resolve_generation_defaults` (fixed; `:5066`, `:8882`);
  - `RAG.rag_service.payload_exemplars.maybe_record_exemplar` (a recorder; 5 sites);
  - `RAG.rag_service.advanced_cache.register_semantic_cache` (a recorder; `:2906`).
- **Never reached:** no case passes `media_db`/`media_db_path`, so the media-DB FTS fallbacks (`:4364`, `:4859`) stay unreachable.

| id | kwargs | extra patches (`up.` or source module) | covers |
|---|---|---|---|
| G01 `minimal` | `top_k=5`; every default-True flag False (`enable_reranking, enable_generation, enable_injection_filter, enable_chunk_citations, enable_query_reformulation, enable_research_action_dedup`) | none | base retrieval, result shape [`test_unified_pipeline_focused`] |
| G02 `answer` | `enable_generation=True, search_depth_mode="balanced"` | none | R03 presets, standard generation |
| G03 `rerank_citations` | `enable_reranking, reranking_strategy="hybrid", enable_citations, require_hard_citations, enable_learned_fusion` | none | R19, R21, R23 [`test_learned_fusion_simple`] |
| G04 `query_prep` | `expand_query, expansion_strategies=["synonym","acronym"], keyword_filter=["observatory","telescope"], enable_intent_routing, enable_dynamic_granularity, adaptive_hybrid_weights, auto_temporal_filters` (no temporal cue), `spell_check=True` | `up.`: `multi_strategy_expansion`, `QueryRouter`, `GranularityRouter`, `spell_check_query` | R08, R11, R12, keyword filter [`test_unified_pipeline_query_expansion`] |
| G05 `variants` | `enable_hyde, enable_prf, enable_query_decomposition, max_subqueries=2, enable_text_late_chunking, enable_date_filter, date_range={…}` | `up.`: `generate_hypothetical_answer`, `hyde_embed_text`, `apply_prf`; `RAG.rag_service.agentic_chunker._decompose_query` (`:4583`) | R14 leaves [`test_hyde`, `test_unified_pipeline_decomposition`] |
| G06 `safety` | `enable_security_filter, enable_injection_filter, enable_content_policy_filter, enable_html_sanitizer, chunk_type_filter` | none (real filters) | R16 |
| G07 `scoped_cache` | `enable_cache=True, workspace_id="ws-1"` (kwarg), `enable_generation=False, enable_reranking=False, adaptive_cache=False, credential_runtime=SimpleNamespace(_user_id=7)`; `runs=2` inside `scoped_context(user_id=7)` | `up.get_shared_cache`: an in-memory factory copied from `test_unified_pipeline_focused.py:51` (that helper is private) | R05, R13, R28. The dry run measured `cache_hit` False then True, with one namespace `owner:…|authorization:…|workspace:…|retrieval:…` |
| G08 `clarify` | `enable_pre_retrieval_clarification=True` | `up.assess_query_for_clarification` returns clarify | `EarlyReturn`, `finally`, tail [`test_pre_retrieval_clarification_pipeline`] |
| G09 `classify` | `enable_query_classification, enable_query_reformulation, enable_discussion_search, enable_web_fallback` | `up.`: `classify_and_reformulate`, `fallback_to_web_search`; `RAG.rag_service.query_classifier.reformulate_query` (`:3437`); `Web_Scraping.WebSearch_APIs`: `search_discussions`, `perform_websearch`, `process_web_search_results` (`:3265`, `:3294`) | R09, R20 web fallback |
| G10 `rerun` | `enable_post_verification, adaptive_rerun_on_low_confidence=True, enable_gap_analysis` | `up.PostGenerationVerifier` (low confidence); a recording wrapper on `up.unified_rag_pipeline` | R15 gap analysis, R26 rerun kwargs (R6) [`test_post_verifier`] |
| G11 `processing` | `enable_multi_vector_passages, enable_precomputed_spans, enable_numeric_table_boost, enable_table_processing, enable_vlm_late_chunking, enable_knowledge_strips, include_sibling_chunks, sibling_window=1, enable_evidence_accumulation` | `up.`: `apply_multi_vector_passages`, `apply_precomputed_spans`, `TableProcessor`, `process_knowledge_strips`, `EvidenceAccumulator`; `Ingestion_Media_Processing.VLM.registry.get_backend` returns `None` (`:5545`) | R15, R17, R20, R21 [`test_prf_and_spans`] |
| G12a `fallback_dict` | `enable_generation=False, fallback_on_error=True, enable_resilience=True, retry_attempts=2, circuit_breaker=False` | `up.build_retrieval_only_result` raises `RuntimeError` | the 7-type handler. The `:8806` dict is returned with `timings_keys` holding `total`, written by `finally` after the return |
| G12b `pipeline_failed` | `enable_generation=False, fallback_on_error=False` | as G12a | the response path with `errors == ["pipeline_failed"]`; the tail still coordinates evidence |
| G13 `security_filter_error` | `enable_security_filter=True, sensitivity_level="bogus"` | none | the `:5491` handler records `Security filter failed: …` and the filter is skipped (fail-open, today's behaviour) |
| G14 `self_correct` | `enable_document_grading, enable_query_rewriting_loop, enable_web_fallback, enable_utility_grading, enable_fast_hallucination_check, search_depth_mode="quality"` | the `up.` grader, rewriter and `fallback_to_web_search` bindings these stages call; the base `analyze` backstop | R18, R20, R23, R27 |
| G15 `generation_extras` | `enable_abstention, enable_multi_turn_synthesis, enable_suggestions, enable_structured_response, enable_evidence_chains` | `up.`: `generate_suggestions`, `EvidenceChainBuilder` and the writer functions; the base `analyze` covers the critique (`:7098`) | R21, R22, R27 [`test_pipeline_generation_controls`, `test_unified_pipeline_structured_writer`] |
| G16 `verification` | `enable_claims, enable_numeric_fidelity, enable_faithfulness_eval, faithfulness_llm=<fake>` (kwarg), `enable_monitoring, enable_observability, enable_performance_analysis, collect_feedback, highlight_results, track_cost` | `up.`: `ClaimsEngine`, `check_numeric_fidelity`; `Claims_Extraction.claims_engine._resolve_claims_llm_config` (`:7580`); `core.config.settings` (`:7698`, `:7796`); `RAG.rag_service.faithfulness.FaithfulnessEvaluator` (`:8865`) | R24, R25, R27-R29 [`test_generation_executor`] |
| G17 `research` | `enable_research_loop, enable_research_progress` (recording callback), `enable_image_search, enable_video_search, enable_research_action_dedup` | `up.`: `research_loop`, `create_default_registry` | R10 [`test_research_agent`] |
| G18 `batch` | `unified_batch_pipeline(queries=["q1","q2","q1"], max_concurrent=2, enable_generation=False)` | none | batch wrapper, duplicate reuse |
| G19 `explicit_sources` | `include_media_ids=<two fixture media ids>, workspace_id="ws-1", enable_cache=True` | none | R07 explicit source selection (`cache_bypassed.reason == "explicit_source_selection"`), workspace filtering |

Research B's G1-G12 map to G01, G02, G03, G04, G05, G06, G07, G08/G09, G10, G11, G12a and G18. If a case meets an LLM, config or network path not listed here, the live-network guard or the determinism probe shows it; add the patch to this table in PR A.

## Appendix D: the gate list (`RAG_PIPELINE_GATE_TESTS`, under `tldw_Server_API/tests/`)

Measured locally: 1,051 tests in 166 s, 0 skips.
- **Added files:** Task 1 adds the golden file and Task 3 adds `RAG_NEW/unit/test_rag_pipeline_types.py`.
- **`RAG_NEW/integration/`:** `test_rag_ablate_api`, `test_rag_batch_checkpoint_api`, `test_rag_batch_resume_api`, `test_rag_convenience_endpoint_credentials`, `test_rag_doc_researcher_api`, `test_rag_integration`, `test_rag_request_contract_parity`, `test_rag_stream_parity`, `test_rag_strict_extractive_nli_api`.
- **`RAG_NEW/`:** `property/test_rag_properties`, `test_prf_and_spans`.
- **`RAG_NEW/unit/`:**
  - `test_batch_round2_flags`, `test_evidence_chain_provider_isolation`, `test_generation_executor`, `test_hyde`, `test_knowledge_source_retrieval_coverage`, `test_learned_fusion_simple`, `test_pipeline_generation_controls`, `test_pipeline_post_verification_metadata`, `test_pipeline_two_tier_gate`, `test_post_retrieval_coordinator`, `test_post_verifier`;
  - `test_pre_retrieval_clarification_metrics`, `test_pre_retrieval_clarification_pipeline`, `test_quick_wins_spell_check_compat`, `test_rag_endpoint_contract_cleanup`, `test_rag_profiles`, `test_rag_provider_credentials`, `test_rag_result_mapping_core`, `test_rag_transport_helpers`, `test_rag_unified_response_mapping`, `test_rag_unified_search_agent_defaults`, `test_retrieval_plan_usage`, `test_retrieval`;
  - `test_search_depth_explicit_overrides`, `test_slides_source_retrieval_hardening`, `test_source_contract`, `test_standard_core_contract_threading`, `test_streaming_executor`, `test_strict_extractive_and_citations`, `test_temporal_filters`, `test_unified_pipeline_decomposition`, `test_unified_pipeline_focused`, `test_unified_pipeline_profile_metadata`, `test_unified_pipeline_query_expansion`, `test_unified_pipeline_security_cache_helpers`, `test_unified_pipeline_structured_writer`, `test_unified_pipeline`.
- **`RAG/`:** `test_unified_pipeline_document_serialization`, `test_unified_schema_and_pipeline`, `test_doc_researcher_features`, `test_knowledge_qa_live_regressions`, `test_restricted_postgres_media_retrieval`, `test_sql_retriever_integration`.
- **`Sharing/`:** `test_shared_workspace_chat_retrieval`.

`RAG_PIPELINE_GATE_EXCLUDED`:
- `RAG_NEW/test_unified_pipeline.py`: 4 tests attempt live DNS to `api.openai.com` (teardown errors), and 3 are `live_network_baseline` entries.
- `RAG_NEW/integration/test_rag_agentic_api.py`: it skips on HTTP 500.
- `RAG_NEW/integration/test_rag_benchmarks_ablation.py`: it skips without TestClient lifespan.

## Review dispositions (fix round 1)

The sources are `plan-review.md` (I, M) and `plan-dryrun.md` (B, N).

**Fixed:**
- **B1/I1:**
  - G12 split into G12a/G12b, through `up.build_retrieval_only_result` raising `RuntimeError` with `enable_generation=False`. Measured gives the reachability proof (`:6933` is in `try[69]` with no inner handler).
  - G13 relabelled to the security filter's own handler.
  - Review Focus 4 corrected. The false Task 12 follow-up became the real fail-open one.
- **B2:** G07 uses the dry run's config; `include_media_ids` moved to the new G19.
- **B3:** Appendix C's base and per-case patch sets name every source module, and Task 1 Step 4 explains why they stay live.
- **I2:** the spy records dataclass and pydantic args by value, with an `fts_level` seeded check.
- **I3:** A.3's def-order rule; D3 names the five published closures.
- **I4:** the Sharing-owned `FIELD_GROUP` assertion, plus `test_from_kwargs_honours_exactly_the_config_fields`.
- **I5:** `PIPELINE_ENTRY_PATCH_ALLOWLIST` with exact-set equality.
- **I6:** V6 applies every V2 transform, plus new V9 and two self-tests.
- **M1/N1:** the hand mutation is now `:5225` against G04, with the reason it shows.
- **M2/N5:** an explicit `VOLATILE_KEYS`, with no suffix rule.
- **M3a/N2:** `TLDW_WRITE_RAG_GOLDEN` runs under pytest. **M3b:** the base set fakes `create_reranker`. **M3c:** Step 9 never excludes the golden.
- **M4:** every chacha contract check and mutation is copied, plus the globs-cover-gate test and the `ORIGINAL_JOB_NAMES` position.
- **M5:** the `<alias>.<patched>` bypass is flagged.
- **M6:** `test_new_manifests_verify_against_the_pr_base` runs in the Enforce step.
- **M7:** a full RAG_NEW + RAG run in Task 12.
- **M8:** the recipe adds slots.
- **M9/N9:** the docstring is `body[0]`, and "first statement" means after it. **M9:** seventeen PRs; the pins file wording; the AC1 AST scan.
- **N3:** Step 9 says it is the first autoload-off run.
- **N4:** the cache factory is copied.
- **N7/N8:** the rerun pin uses a `NodeTransformer` and excludes by enclosing top-level function.
- **N10:** the shim is ~338.
- **N11:** "233 named parameters", with 15 hidden reads in Measured.
- **N14:** the scope row is fixed, K is ~785-820, and AC3 is met at L.
- **N15:** the AST scan replaces the grep.

**Accepted as is:**
- **M10:** the exclusions, PR sizes and tool value. The reviewer agreed with them, so nothing changes.
- **N6:** the "5 never tested by an `if`" stays. It is the inert set `INERT_FLAGS` uses. The dry run's 8 count call-argument uses, which the coverage pin already handles by effective value.
