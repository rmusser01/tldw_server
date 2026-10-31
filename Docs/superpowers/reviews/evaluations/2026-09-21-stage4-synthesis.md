# Stage 4 — Prior-findings reconciliation, priority order, and coverage boundaries

## Scope

Three jobs, no new findings:

1. Reconcile the 16 findings in [`../evals-module/README.md`](../evals-module/README.md) against the
   current working tree — still-live or already-addressed, with the evidence for each verdict.
2. Put this audit's 17 findings into one priority order, with the design-first gate marked.
3. State plainly what this audit did not cover, so the gap is visible rather than implied.

Per the README's rules, this file **points back at** the stage files; it does not restate them.
Findings live in
[stage 2](2026-09-21-stage2-scoring-and-aggregation.md) and
[stage 3](2026-09-21-stage3-shared-helpers-and-efficiency.md).

## Code Paths Reviewed

Only the sites named by the prior ledger, re-opened to check whether the described behaviour is
still present. Each is cited inline in the reconciliation table below.

## Tests Reviewed

No new test reading in this stage. The per-finding test mapping is in stages 1-3.

## Validation Commands

Each verdict below was reached by opening the cited lines in the current tree. The two verdicts
that rest on executed output rather than reading are:

```
$ python3 -c "
from tldw_Server_API.app.core.DB_Management.backends.base import QueryResult
print('first is a property :', isinstance(QueryResult.__dict__['first'], property))
print('scalar is a property:', isinstance(QueryResult.__dict__['scalar'], property))"
first is a property : True
scalar is a property: True
```
(prior finding 7 — `fetch_one`/`fetch_value` returning bound methods — is resolved by `first` and
`scalar` now being properties at `DB_Management/backends/base.py:259` and `:271`)

```
$ grep -n "run_status\|alias=\"status\"" tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_crud.py | head -3
372:    run_status: Optional[str] = Query(None, alias="status"),
379:            status=run_status,
```
(prior finding 8 — the `status` module shadowed by a query parameter — is resolved by the rename)

## Findings

None. This stage records verdicts on someone else's findings and orders this audit's own.

### Prior-findings reconciliation vs `../evals-module/`

| # | Prior finding (abbreviated) | Verdict | Evidence in the current tree |
| ---: | --- | --- | --- |
| 1 | Rate-limit subject is the raw single-user API key, not the stable user id | **STILL LIVE** | `evaluations_auth.py:146-168` — in `single_user` mode `verify_api_key` still `return token`, and the test-mode branch at `:143-144` returns the token too. |
| 2 | `verify_api_key` TEST_MODE shortcut inconsistent with `get_eval_request_user` | **PARTIALLY ADDRESSED** | Both branches now share one gate, `_evals_test_mode_bypass_enabled()` = `is_explicit_pytest_runtime() and TEST_MODE` (`evaluations_auth.py:52-53`, used at `:143` and `:166`). The residual asymmetry stands: single_user returns the token, multi_user returns the literal `"test_user"` (`:169`). |
| 3 | `_apply_rate_limit_headers` defaults `PerMinute-Remaining` to 0 on allowed paths | **NOT RE-VERIFIED** | Deliberately out of this audit's scope (auth/rate-limit surface). No claim either way. |
| 4 | `cancel_run` rewrites completed/failed runs to `cancelled` | **ADDRESSED** | `unified_evaluation_service.py:664-678` now normalizes the current status and gates the direct DB write on `can_transition_run_status(current_status, "cancelled")`, returning False when the transition is illegal. |
| 5 | Explicit `db_path` bypasses the evaluations path-containment rule | **ADDRESSED** | `evaluation_manager.py:_get_db_path (103-133)` now routes the explicit path through `resolve_trusted_database_path(..., label="evaluations_db_path", extra_roots=[base_resolved])` and falls back to the default on failure. |
| 6 | `resolve_trusted_database_path` compares a resolved root to a merely normalized candidate (symlink alias) | **NOT RE-VERIFIED** | Lives in `DB_Management/db_path_utils.py`, outside this module's boundary. Flagged for whoever audits DB_Management. |
| 7 | `BackendAdapter.fetch_one`/`fetch_value` return bound methods | **ADDRESSED** | `QueryResult.first` and `.scalar` are `@property` (`DB_Management/backends/base.py:259, :271`, executed above), so the attribute access at `db_adapter.py:271, :279` is correct. |
| 8 | `list_runs` shadows the FastAPI `status` module with a query parameter | **ADDRESSED** | Renamed to `run_status` with `alias="status"` (`evaluations_crud.py:372, :379`, executed above). |
| 9 | `pipeline_presets` keyed by `name` alone, so presets collide across users on a shared backend | **STILL LIVE — and the read-side fix masks it** | Reads are now user-filtered (`Evaluations_DB.get_pipeline_preset:2864-2870` applies `_append_user_filter`), but the DDL is still `name TEXT PRIMARY KEY` (`:513-519`) and the write is still `ON CONFLICT(name) DO UPDATE SET config = excluded.config, user_id = COALESCE(excluded.user_id, ...)` (`:2851-2857`). On a shared Postgres backend, user B saving a preset named `default` **overwrites user A's config and reassigns the row's `user_id` to B**; A's now-filtered read then returns nothing, so A's preset silently disappears rather than being visibly shared. The filter hides the collision instead of preventing it. |
| 10 | Ephemeral RAG indexes named `{namespace}_{cfg_id}` collide across concurrent runs | **NOT RE-VERIFIED for the collision** | This audit worked the adjacent lifecycle defect instead — see [evaluations-017](2026-09-21-stage3-shared-helpers-and-efficiency.md), the create-before-register ordering that orphans collections past the TTL reaper. The naming question is unchanged as far as this audit observed. |
| 11 | `RecipeRegistry` silently overwrites on duplicate `recipe_id` | **STILL LIVE** | `recipes/registry.py:43-45` — `for recipe in recipe_iterable: self._recipes[recipe.recipe_id] = recipe`, no duplicate check, no warning. |
| 12 | The benchmark registry advertises benchmarks the loader map cannot resolve | **STILL LIVE** | `benchmark_loaders.load_benchmark_dataset:420-430` knows 9 names; `benchmark_registry.py:120` registers `simpleqa_verified`, which is not among them. Unknown names fall through to source-format sniffing and return `[]` when no source is given (`:436-438`). |
| 13 | Benchmark routes coerce `User.id` to `int`, breaking tenant-style string ids | **ADDRESSED** | `evaluations_benchmarks.py:32-33` — `_get_evaluation_manager_for_user(identity)` now passes `identity.user_scope` straight through; no `int()` coercion remains. |
| 14 | The dataset read route bypasses the service and calls `svc.db.get_dataset(...)` | **STILL LIVE** | `evaluations_datasets.py:180-186` still reaches through the service to its `.db` attribute with `include_samples`/`sample_limit`/`sample_offset`. (Separately improved: the route now uses the canonical `build_offset_pagination_meta`.) |
| 15 | Webhook ownership normalization disagrees with the service-binding coercion | **NOT RE-VERIFIED** | Endpoint-identity surface, out of this audit's scope. No claim. |
| 16 | A/B test idempotent replay returns a hard-coded `status='running'` | **ADDRESSED** | `evaluations_embeddings_abtest.py:196-205` — the replay branch now sets the `X-Idempotent-Replay` header and returns `_abtest_status_response(test_id, row)`, i.e. the real persisted status. |

Tally: **6 addressed** (4, 5, 7, 8, 13, 16), **1 partially addressed** (2), **5 still live**
(1, 9, 11, 12, 14), **4 not re-verified** (3, 6, 10, 15) because they sit outside this audit's
stated boundary. Finding 9's verdict is sharper than the prior ledger's: the read-side filter that
was added since makes the cross-user write **silently destructive** rather than merely leaky.

### This audit's findings in priority order

| Rank | Finding | Axis | Sev | Design-first? | Owner-only? |
| ---: | --- | --- | --- | --- | --- |
| 1 | [evaluations-008](2026-09-21-stage3-shared-helpers-and-efficiency.md) BYOK guard lost when a recipe was extracted | duplication | High | no | no |
| 2 | [evaluations-009](2026-09-21-stage3-shared-helpers-and-efficiency.md) `_call_adapter_text` x4; the runner's copy drops credential validation | duplication | High | ADR + design note | no |
| 3 | [evaluations-001](2026-09-21-stage2-scoring-and-aggregation.md) alias keys double-counted in the overall RAG score | correctness | High | no | no |
| 4 | [evaluations-007](2026-09-21-stage3-shared-helpers-and-efficiency.md) `created` computed in host-local time from naive UTC, six copies (ADR-014 drift) | correctness | High | **yes** | **yes** (4 of 6 sites) |
| 5 | [evaluations-003](2026-09-21-stage2-scoring-and-aggregation.md) one sample's chunk stats published as a config-level mean and used to pick the winner | correctness | High | no | no |
| 6 | [evaluations-002](2026-09-21-stage2-scoring-and-aggregation.md) correct 1-5 normalizer is dead; 11 inline copies use a different formula | duplication | High | **yes** | no |
| 7 | [evaluations-012](2026-09-21-stage3-shared-helpers-and-efficiency.md) weak test-mode gate; last webhook fallback drops the user filter | encapsulation | Medium | no | no |
| 8 | [evaluations-011](2026-09-21-stage3-shared-helpers-and-efficiency.md) webhook schema owned twice with a drifted FK | duplication | Medium | no | no |
| 9 | [evaluations-010](2026-09-21-stage3-shared-helpers-and-efficiency.md) audit bridge clone drops requester attribution | duplication | Medium | ADR (shared with #2) | no |
| 10 | [evaluations-016](2026-09-21-stage3-shared-helpers-and-efficiency.md) unbounded accumulation, post-load pagination, silent 1000-row truncation | efficiency | Medium | partly (schema) | no |
| 11 | [evaluations-015](2026-09-21-stage3-shared-helpers-and-efficiency.md) serialized LLM fan-out; `asyncio.run` per sample | efficiency | Medium | no | no |
| 12 | [evaluations-013](2026-09-21-stage3-shared-helpers-and-efficiency.md) five recipes re-derive six decision helpers; four confidence formulas | duplication | Medium | **yes** | no |
| 13 | [evaluations-014](2026-09-21-stage3-shared-helpers-and-efficiency.md) 705-line connection pool warmed and unused | efficiency | Medium | no | one line is |
| 14 | [evaluations-005](2026-09-21-stage2-scoring-and-aggregation.md) unmeasurable samples enter the mean as 0.0 | correctness | Medium | no | no |
| 15 | [evaluations-006](2026-09-21-stage2-scoring-and-aggregation.md) same judged metric, opposite failure semantics in two evaluators | duplication | Medium | no | no |
| 16 | [evaluations-017](2026-09-21-stage3-shared-helpers-and-efficiency.md) ephemeral collection created before it is registered | sequential-coupling | Medium | no | no |
| 17 | [evaluations-004](2026-09-21-stage2-scoring-and-aggregation.md) `aggregation_weights` cannot exclude `rag_overall` | correctness | Medium | no | no |

### Proposed Backlog tasks

Proposals only. The Backlog is the ledger of record and is managed through its MCP/CLI; **this audit
is read-only and created no task files.**

1. `evals: restore BYOK decryption guard on the rag_answer_quality execution path` — evaluations-008.
2. `evals: single owner for adapter-text calls with credential context` — evaluations-009 + -010, ADR + design note.
3. `evals: fix RAG overall-score alias double-count` — evaluations-001.
4. `evals: normalize persisted timestamps to UTC before Unix conversion` — evaluations-007, design-first, owner-only paths.
5. `evals: chunk stats must be per-config means, not one sample's` — evaluations-003.
6. `evals: one owner for judge-score normalization and judge-failure semantics` — evaluations-002 + -006, design-first.
7. `evals: tighten webhook test-mode gate and remove the user-filterless fallback` — evaluations-012.
8. `evals: one owner for the webhook schema` — evaluations-011.
9. `evals: batch-path cost reductions` — evaluations-015 + the cheap half of -016 + -017.
10. `evals: shared recipe decision helpers` — evaluations-013, design-first.
11. `evals: adopt or delete the Evaluations connection pool` — evaluations-014.
12. `evals: aggregate over measured samples only` — evaluations-005 + -004.
13. `evals: pipeline_presets primary key must include user_id` — prior finding 9, whose still-live write path is now silently destructive.

## Suggested Refactor/Actions

- **Sequence.** Ranks 1-3 and 5 are small, independent, and each needs only a failing test first.
  Rank 4 and ranks 6, 12 gate on a written decision because they change values already returned to
  clients. Ranks 8, 11, 13 are deletions or ordering changes and should not be bundled with the
  design-first work.
- **Base branch.** `CONTRIBUTING.md:86,121` says PRs target `dev`; `origin/HEAD` resolves to `main`.
  This audit assumed **`dev`** for any resulting PR. Confirm before opening one.
- **Gates.** Bandit runs on touched scope (ADR-005); the credential- and tenant-boundary changes
  (ranks 1, 7) must clear HIGH/CRITICAL. Nothing proposed here adds a CI gate, so the six
  contractual gates in `Docs/Development/CI_REQUIRED_GATES.md` are unaffected.
- **ADRs.** Nothing proposed contradicts ADR-012, -013, -014 or -015. Rank 4 restores ADR-014
  conformance. Rank 2 and rank 6 each warrant a new ADR because neither the placement of a shared
  adapter-text/audit bridge nor the canonical judge-score scale is settled by any existing record —
  ADR-015 assigns scoring to "dedicated evaluator modules" without saying they must agree.

## Not Covered

Stated plainly rather than implied.

- **The auth, rate-limit and endpoint-identity surface.** Prior findings 1, 2, 3, 15 and the whole
  of `../evals-module/` slice 1 and slice 4. Re-walking them would have duplicated an existing
  ledger; where a verdict was cheap to reach I recorded it above, and where it was not I wrote
  "not re-verified" instead of guessing.
- **`user_rate_limiter.py` beyond its persistence layer.** 1,353 LOC and the module's second-highest
  churn (32 commits). I read `_connect`/`_run_db`/`_init_database`/`_write_request_usage` for the
  connection and retention findings and stopped there. The tier logic, quota arithmetic, ledger
  backfill (`_backfill_legacy_daily_usage_to_ledger`) and the Resource-Governance integration
  (`_rg_evaluations_enabled` and friends, `:1152-1240`) are unreviewed. That is the largest single
  gap in this audit.
- **`metrics.py` and `metrics_advanced.py` beyond their collector declarations.** The colliding
  metric names are recorded in stage 3's "Noted, not filed"; I did not verify the `/metrics` export
  surface, so I make no claim about whether the advanced collectors are actually scraped.
- **`webhook_security.py` (830 LOC).** Only `_close_response` and the SSRF-adjacent call into
  `webhook_validator.resolve_safe_delivery_target_async` were touched. HMAC signing, replay
  protection and the allowlist itself are unreviewed.
- **`circuit_breaker.py`, `config_manager.py`, `config_validator.py`, `db_adapter.py`,
  `benchmark_registry.py`, `simpleqa_eval.py`, `web_retrieval_quality.py`,
  `article_extraction_benchmark.py`, `wordbench_runner.py`, `qa_benchmark_helper.py`,
  `persona_chat_judge*.py`, `synthetic_eval_*.py`, `ocr_evaluator.py`.** Read for the specific
  patterns each stage was hunting (aggregation math, duplicated helpers, cost drivers) and cited
  where they matched, but not audited end to end.
- **Postgres behaviour was reasoned about, never executed.** Every Postgres claim in this ledger —
  the `_row_to_dataset_dict` fall-through in evaluations-007, the `pipeline_presets` upsert
  collision in prior finding 9, the `Evaluations_DB.backend` re-resolution in evaluations-015 — was
  derived by reading the DDL and the branch, not by running against a cluster. The repo's own
  precedent (the SQLite/Postgres split being tested on SQLite only) says this is exactly where to
  be sceptical, and the honest label is probable-risk.
- **Nothing was measured.** No profiler, no query counter, no load. Every efficiency finding names
  a cost driver and what it scales with, which is what the rubric asks for, but the wall-clock
  magnitudes are unmeasured and labelled as assumptions.
- **The local venv is missing two declared core dependencies** (`sklearn`, `hypothesis`), so 8 of
  the module's test modules could not be collected and the property-based invariants could not be
  run. See the README's environment note. Only `tests/Evaluations/unit/test_rag_evaluator.py`
  (34 passed / 6 env-failed) was actually executed.
