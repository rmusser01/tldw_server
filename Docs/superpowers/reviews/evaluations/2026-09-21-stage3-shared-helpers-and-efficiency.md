# Stage 3 — Shared helpers, persistence ownership, and batch-workload cost

## Scope

Three questions, all about things that are written more than once or cost more than once:

1. **Duplication / adoption gaps.** Which helpers in this module are re-derived, and which of the
   copies is correct? Which existing canonical helpers are bypassed?
2. **Persistence and layering ownership.** Who owns the evaluations schema, the connections, and
   the audit bridge — and where does a second owner exist?
3. **Efficiency.** Evaluation runs are batch workloads. Where does cost scale with sample count,
   dataset size, context count or config-grid size, and what is the driver?

Scoring math is stage 2. Auth, run lifecycle and endpoint CRUD were covered by
[`../evals-module/`](../evals-module/README.md) and are not re-walked here.

## Code Paths Reviewed

**Timestamp conversion**
- `DB_Management/Evaluations_DB.py:_ensure_unix_timestamp (2465-2506)`, and its callers
  `_row_to_evaluation_dict (2592)`, `_row_to_dataset_dict (2641)`, `_row_to_run_dict (2608+)`.
- `Evaluations/unified_evaluation_service.py:_extract_created_ts (1496-1508)`.
- `api/v1/endpoints/evaluations/evaluations_datasets.py:_normalize_dataset_timestamps (~40-64)`.
- `api/v1/endpoints/evaluations/evaluations_rag_pipeline.py:to_ts (72-80, 116-124, 174-182)`.
- `Evaluations/eval_runner.py:_send_webhook (2350-2360)`.
- Schema origin: `Evaluations_DB.py:379-552` (SQLite `TEXT DEFAULT CURRENT_TIMESTAMP`) vs
  `:706-944` (Postgres `TIMESTAMPTZ DEFAULT NOW()`), plus `:251`
  (`sqlite3.register_adapter(datetime, lambda d: d.isoformat(sep=" "))`).

**Provider/credential plumbing**
- `Evaluations/ms_g_eval.py:_validate_provider_credentials (66-83)`, `_call_adapter_text (85-134)`.
- `Evaluations/eval_runner.py:_call_adapter_text (99-138)` and its call sites
  `_eval_label_choice (1844)`, `_eval_nli_factcheck (2038)`.
- `Evaluations/cli/evals_cli.py:_call_adapter_text (58-89)`,
  `Evaluations/cli/evals_cli_enhanced.py:_call_adapter_text (122-153)`.
- `LLM_Calls/adapter_utils.py` — `normalize_provider (53)`, `ensure_app_config (143)`,
  `resolve_provider_model (160)`, `resolve_provider_api_key_from_config (183)`,
  `get_adapter_or_raise (202)`, `split_system_message (209)`.
- `Evaluations/recipe_runs_jobs_worker.py:_run_config_for_execution (115-127)`,
  `_contains_redacted_value (130+)`, dispatch at `:1091-1104`.
- `Evaluations/recipes/rag_answer_quality_execution.py:_run_config_for_execution (82-84)` and its
  three call sites `:95, :122, :144`.

**Persistence and audit ownership**
- `Evaluations/evaluation_manager.py:__init__ (46-62)`, `_connect (64-75)`, `_run_db (77-101)`,
  `_get_db_path (103-…)`, `_init_database (194-313)` incl. the webhook DDL at `:256, :278`.
- `Evaluations/webhook_manager.py:_webhook_schema_sql (170-247)`, `_get_webhooks (540-618)`,
  `_deliver_webhook (620-731)`.
- `Evaluations/user_rate_limiter.py:_connect (228-248)`, `_run_db (250-278)`,
  `_init_database (280-336)`, `_write_request_usage (833-857)`.
- `Evaluations/connection_pool.py` (all 705 lines), `EvaluationsConnectionManager (587-645)`,
  `get_connection_manager (651-657)`, `get_connection (664-667)`.
- `services/startup_evaluations_warmup.py:_warm_evaluations_connection_manager (29-34)`,
  `services/shutdown_evaluations_resources.py:46`.
- `Evaluations/audit_adapter.py (1-295)` vs `Embeddings/audit_adapter.py (1-248)`.
- `core/testing.py:is_test_mode (42-48)` vs
  `api/v1/endpoints/evaluations/evaluations_auth.py:_evals_test_mode_bypass_enabled (52-53)`.

**Recipe helper family**
- `recipes/{summarization_quality,rag_answer_quality,rag_retrieval_tuning,embeddings_retrieval,persona_dialogue_tree_robustness}.py`
  and `recipe_runs_service.py`, `recipe_runs_jobs_worker.py`,
  `recipes/rag_answer_quality_execution.py`. Specific symbols cited per finding.

**Efficiency**
- `eval_runner.py:_execute_rag_pipeline_run (613-891)`, `_build_ephemeral_index (1031-1185)`,
  `_process_batch (1294-1360)` (the correct shape), `_execute_evaluation (373-422)`.
- `rag_evaluator.py:_evaluate_context_precision (832-884)`, `_evaluate_context_relevance (886-945)`.
- `recipe_runs_jobs_worker.py:_execute_rag_retrieval_tuning_recipe_run (873-983)` (`asyncio.run` at
  `:916`), `_execute_summarization_recipe_run (789-870)`,
  `_collect_embeddings_candidate_results (645-721)`.
- `embeddings_abtest_service.py:run_vector_search_and_score (605-863)` (reranker at `:779`),
  `compute_significance (984-1027)`, `build_collections_vector_only (300-602)`.
- `Evaluations_DB.py:backend (279-302)`, `_resolve_backend (308-321)`, `get_connection (246-259)`,
  `_row_to_dataset_dict (2633-2665)`, `get_abtest_arms (1407-1424)`, `get_abtest_queries (1463-1479)`.
- `recipes/rag_answer_quality_execution.py:_execute_candidate (235-343)`,
  `_resolve_live_contexts (397-418)` (`asyncio.run` at `:408`).
- `ms_g_eval.py:run_geval (172-400)`, metric loop `:358-380`.
- `benchmark_loaders.py:load_benchmark_dataset (404-457)`.
- `unified_evaluation_service.py:evaluate_qa3 (1119-1280)`, loop `:1203-1213`.
- `webhook_manager.py:send_webhook (499-538)` (correct gather) vs `_deliver_webhook (620-731)`.

## Tests Reviewed

| Test file | What it protects | Downgrades risk? |
| --- | --- | --- |
| `tests/DB_Management` (2 files reaching `core.Evaluations`) | Evaluations DB row mapping. | **No** for timestamps — CI hosts run UTC, where the defect in evaluations-007 is a zero offset. |
| `tests/Evaluations/unit/test_webhook_manager_backend_schema.py` | `WebhookManager`'s SQLite and Postgres DDL. | Partly. It pins *one* owner's DDL, which is why the second owner (evaluations-011) is invisible: nothing asserts the two agree. |
| `tests/Evaluations/integration/test_webhook_multi_user_api.py` | Per-user webhook isolation over the API. | **No** for evaluations-012 — the leak is behind `is_test_mode()`, which is true while the suite runs, so the fallback is the path under test rather than the path being excluded. |
| `tests/Evaluations/unit/test_specialized_provider_auth_failures.py:56` | Monkeypatches `ms_g_eval._call_adapter_text` to reject unresolved credentials. | Yes for G-Eval, and it is the direct evidence for evaluations-009: the *guarded* copy is the one under test. |
| `tests/LLM_Calls/test_provider_adapter_runtime_boundary.py:654, :665` | Exercises `ms_g_eval._call_adapter_text` and `prompt_studio ...EvaluationManager._call_adapter_text`. | Partly — it knows about two of the seven copies and not about `eval_runner`'s. |
| `tests/Evaluations/test_evaluations_audit_adapter.py` | The Evaluations audit bridge. | Partly. It pins the bridge's own behaviour; nothing compares it against the Embeddings twin (evaluations-010). |
| `tests/Services/test_startup_evaluations_warmup.py`, `tests/Services/test_shutdown_evaluations_resources.py` | That the pool starts and stops cleanly. | **No** for evaluations-014 — neither asserts anything is served from the pool, which is the finding. |
| `tests/AuthNZ/unit/test_test_mode_runtime_guard.py:122` | Monkeypatches `connection_pool.get_connection_manager` to raise, checking the guard. | Confirms the pool is only ever *constructed*, never drawn from. |
| `tests/Evaluations/test_rag_pipeline_runner.py`, `tests/Evaluations/test_evaluations_unified.py` | The rag_pipeline sweep. | Partly — functional coverage only; no assertion on call counts, so evaluations-015 is not detectable by them. |

Reachability by import-grep. Not measured coverage.

## Validation Commands

```
$ TZ=America/Los_Angeles python3 -c "
from datetime import datetime, timezone
s='2026-09-21 21:06:55'   # exactly what SQLite CURRENT_TIMESTAMP writes (UTC, naive)
print('naive  ->', int(datetime.fromisoformat(s).timestamp()))
print('true utc ->', int(datetime(2026,9,21,21,6,55,tzinfo=timezone.utc).timestamp()))
print('delta seconds =', int(datetime.fromisoformat(s).timestamp()) - int(datetime(2026,9,21,21,6,55,tzinfo=timezone.utc).timestamp()))"

naive  -> 1790050015
true utc -> 1790024815
delta seconds = 25200
```

```
$ diff <(sed -n '58,89p' .../Evaluations/cli/evals_cli.py) <(sed -n '122,153p' .../Evaluations/cli/evals_cli_enhanced.py) && echo BYTE-IDENTICAL
BYTE-IDENTICAL

$ grep -rn "_call_adapter_text" --include='*.py' tldw_Server_API/app | grep "def " | wc -l
       7

$ grep -n "_run_config_for_execution(" \
    .../Evaluations/recipe_runs_jobs_worker.py .../Evaluations/recipes/rag_answer_quality_execution.py
recipe_runs_jobs_worker.py:115:def _run_config_for_execution(...)       <- decrypts + guards
recipe_runs_jobs_worker.py:593, :682, :798, :881, :995                  <- 5 guarded call sites
rag_answer_quality_execution.py:82:def _run_config_for_execution(...)   <- 3 lines, no decrypt
rag_answer_quality_execution.py:95, :122, :144                          <- 3 UNguarded call sites

$ grep -rn "rate_limit_tracking" --include='*.py' tldw_Server_API/app | grep -v user_rate_limiter.py
(no output)

$ for s in get_connection_manager get_connection_health get_connection_stats; do \
    grep -rn "\b$s\b" --include='*.py' tldw_Server_API/app | grep -v core/Evaluations/connection_pool.py; done
tldw_Server_API/app/services/startup_evaluations_warmup.py:31
tldw_Server_API/app/services/startup_evaluations_warmup.py:34
tldw_Server_API/app/services/shutdown_evaluations_resources.py:46
   (no call site anywhere checks out a connection)

$ for f in webhook_manager user_rate_limiter evaluation_manager synthetic_eval_repository webhook_security; do \
    printf "%s " $f; grep -cE "CREATE TABLE|INSERT INTO|UPDATE .+ SET|DELETE FROM|SELECT " $f.py; done
webhook_manager 15
user_rate_limiter 14
evaluation_manager 14
synthetic_eval_repository 10
webhook_security 4

$ grep -rn "from tldw_Server_API.app.api" --include='*.py' tldw_Server_API/app/core/Evaluations | grep -v __pycache__ | wc -l
      25
$ ... of which importing api.v1.endpoints or api.v1.API_Deps (true inversion):
embeddings_abtest_service.py -> api.v1.endpoints.embeddings_v5_production_enhanced   (x2)
embeddings_abtest_runner.py  -> api.v1.endpoints.embeddings_v5_production_enhanced
audit_adapter.py:20          -> api.v1.API_Deps.Audit_DB_Deps
   (the remaining 21 are api.v1.schemas.* imports — the mild, schemas-in-the-wrong-package shape)
```

## Findings

### FINDING evaluations-007 — the OpenAI-compatible `created` timestamp is computed in host-local time from a UTC-naive SQLite string, in six places

```
axis:        correctness
class:       divergent-copies
severity:    High
sites:       CANONICAL, AND ALSO WRONG:
               DB_Management/Evaluations_DB.py:_ensure_unix_timestamp (2489-2504)
             BYPASSING COPIES, same defect:
               Evaluations/unified_evaluation_service.py:_extract_created_ts (1503-1507)
               api/v1/endpoints/evaluations/evaluations_datasets.py:_normalize_dataset_timestamps (51-58)
               api/v1/endpoints/evaluations/evaluations_rag_pipeline.py:to_ts (72-80)
               api/v1/endpoints/evaluations/evaluations_rag_pipeline.py:to_ts (116-124)
               api/v1/endpoints/evaluations/evaluations_rag_pipeline.py:to_ts (174-182)
             SAME BUG, DIFFERENT SHAPE:
               Evaluations/eval_runner.py:_send_webhook (2357) int(datetime.utcnow().timestamp())
               Evaluations/metrics.py:477 datetime.utcnow().isoformat() into the health payload
             SCHEMA ORIGIN:
               Evaluations_DB.py:379-552 (SQLite TEXT DEFAULT CURRENT_TIMESTAMP -> UTC, naive)
               Evaluations_DB.py:706-944 (Postgres TIMESTAMPTZ DEFAULT NOW() -> aware datetime)
canonical:   Evaluations_DB.py:_ensure_unix_timestamp (2465-2506) — it IS the designated converter
             (6 call sites inside the DB class) but it carries the identical defect, so promoting it
             is not enough; it has to be fixed first.
destination: tldw_Server_API/app/core/Evaluations/timestamps.py — one responsibility: converting a
             persisted evaluations timestamp (SQLite naive-UTC text, Python-adapted ISO text, or a
             Postgres aware datetime) into the Unix `created` integer ADR-014 requires. Not Utils.py.
knowledge:   "SQLite writes CURRENT_TIMESTAMP as UTC with no offset marker, and Python's
             `datetime.fromisoformat` on such a string yields a NAIVE datetime whose `.timestamp()`
             is interpreted in the host's local zone." Re-derived six times, wrong six times.
scenario:    On the default SQLite backend a row written by `DEFAULT CURRENT_TIMESTAMP` reads back
             as `"2026-09-21 21:06:55"`. `.replace("Z","+00:00")` is a no-op on it, `fromisoformat`
             returns a naive datetime, and `.timestamp()` treats it as local. Executed above on a
             `TZ=America/Los_Angeles` host: **the API returns a `created` 25,200 seconds off**. Every
             CI runner and most containers are UTC, where the offset is exactly zero — which is why
             no test catches it and why it will surface only on a self-hosted box with a real TZ,
             the deployment model this project targets. On the Postgres backend the same converters
             receive a `datetime` object: `_ensure_unix_timestamp:2483` handles it correctly, but
             `evaluations_datasets.py:47-58` matches none of its isinstance branches and falls
             through to `int(datetime.now(timezone.utc).timestamp())` at :60 — so every dataset's
             `created` equals the time of the request that read it. Same field, two different wrong
             answers, split by backend — the dual-backend divergence class with precedent.
             `Evaluations_DB.py:251` compounds it: rows written from Python go through
             `isoformat(sep=" ")`, which for an aware datetime DOES carry `+00:00`, so one column
             holds two formats and only one of them round-trips correctly.
impact:      ADR-014 makes Unix `created` part of the public API contract. This violates it silently,
             by an amount that varies with the host, on the default backend.
tests:       tests/DB_Management (2 files by import-grep); every evaluations endpoint test asserting
             `created`. None can detect it: CI is UTC.
effort:      moderate — six edits, three of them under `api/v1/endpoints/`. Cheap per site; the cost
             is that it changes returned values on non-UTC hosts and needs a migration note for
             already-persisted rows.
owner-only:  YES for the four sites under tldw_Server_API/app/api/v1/** (CONTRIBUTING.md owner-only
             path). The Evaluations_DB and core sites are not.
confidence:  confirmed (the naive-parse defect, executed with observed output); confirmed (the
             SQLite DDL writes naive UTC); confirmed (the Postgres fall-through to now() in the
             datasets converter)
```

### FINDING evaluations-008 — extracting one recipe into its own module dropped the BYOK decryption and redaction guard

```
axis:        duplication
class:       divergent-copies
severity:    High
sites:       GUARDED (correct):
               Evaluations/recipe_runs_jobs_worker.py:_run_config_for_execution (115-127)
               used by :593 (embeddings), :682 (embeddings candidates), :798 (summarization),
                       :881 (rag_retrieval_tuning), :995 (persona_dialogue_tree_robustness)
             UNGUARDED:
               Evaluations/recipes/rag_answer_quality_execution.py:_run_config_for_execution (82-84)
               used by :95 (execute_rag_answer_quality_recipe_run),
                       :122 (execute_fixed_context_...), :144 (_execute_live_end_to_end_...)
             DISPATCH THAT ROUTES PAST THE GUARD:
               Evaluations/recipe_runs_jobs_worker.py:1098-1104
canonical:   recipe_runs_jobs_worker.py:_run_config_for_execution (115-127) — this is the correct
             copy: it calls decrypt_recipe_run_config_from_metadata() and raises when the stored
             config still contains redacted values and no BYOK key is configured.
destination: Evaluations/recipe_runs_config.py — one responsibility: materialising a recipe run's
             effective run_config from its persisted metadata, including decryption and the
             redaction guard. Both modules import it; neither keeps a private copy.
knowledge:   "a persisted recipe run_config may hold encrypted or redacted secrets, and executing
             one without decrypting it is not allowed"
scenario:    `recipe_runs_jobs_worker.py:1098` dispatches `recipe_id == "rag_answer_quality"` to
             `execute_rag_answer_quality_recipe_run`, which immediately calls its OWN three-line
             `_run_config_for_execution` (`rag_answer_quality_execution.py:82-84`):
             `dict(metadata.get("run_config_internal") or metadata.get("run_config") or {})`.
             No `decrypt_recipe_run_config_from_metadata`, no `_contains_redacted_value` check. A
             rag_answer_quality run created with a BYOK provider key, persisted with that key
             encrypted, then executed on a worker where `BYOK_ENCRYPTION_KEY` is unset: the five
             sibling recipes raise "Recipe run secret-bearing config is encrypted but cannot be
             decrypted. Configure BYOK_ENCRYPTION_KEY for recipe worker execution."
             rag_answer_quality instead proceeds with the raw stored value — the ciphertext
             envelope, or the redaction placeholder — which then flows into `_resolve_provider_model`
             (:867-882) and the candidate provider config as if it were a real credential.
impact:      A security control that exists, is correct, and is applied to five of six recipes is
             absent from the sixth — and the sixth is the one that was refactored out into its own
             module. This is the exact failure mode a shared helper prevents.
tests:       tests/Evaluations (recipe-run worker tests reachable by import-grep). Nothing asserts
             the two copies agree; the guard's own test, if any, exercises the worker copy.
effort:      cheap — the correct implementation already exists eleven lines long; the fix is to
             import it rather than re-declare it. Behaviour-preserving for the five guarded paths.
owner-only:  no
confidence:  confirmed (both implementations, the dispatch, and the three unguarded call sites);
             probable-risk (the downstream consequence of the placeholder reaching the provider —
             traced through _resolve_provider_model but not executed)
```

### FINDING evaluations-009 — four copies of `_call_adapter_text` inside the module; the one the runner uses drops BYOK credential validation

```
axis:        duplication
class:       divergent-copies
severity:    High
sites:       Evaluations/ms_g_eval.py:_call_adapter_text (85-134)          <- correct
             Evaluations/eval_runner.py:_call_adapter_text (99-138)        <- no credential plumbing
             Evaluations/cli/evals_cli.py:_call_adapter_text (58-89)       }
             Evaluations/cli/evals_cli_enhanced.py:_call_adapter_text (122-153) } byte-identical pair
             LIVE CALL SITES OF THE UNGUARDED COPY:
               Evaluations/eval_runner.py:_eval_label_choice (1843-1853)
               Evaluations/eval_runner.py:_eval_nli_factcheck (2037-2047)
             THREE FURTHER COPIES OUTSIDE THE MODULE (context, not scope):
               Web_Scraping/WebSearch_APIs.py:437, WebSearch/Web_Search.py:188,
               Prompt_Management/prompt_studio/evaluation_manager.py:45
             FIVE MORE BYTE-IDENTICAL FUNCTIONS ACROSS THE SAME CLI PAIR:
               list_benchmarks  evals_cli.py (164-183) == evals_cli_enhanced.py (440-459)
               register         evals_cli.py (391-409) == evals_cli_enhanced.py (774-792)
               validate         evals_cli.py (415-462) == evals_cli_enhanced.py (798-845)
               health           evals_cli.py (529-564) == evals_cli_enhanced.py (849-886)
               validate_api_config  cli/api_utils.py (111-132) == evals_cli_enhanced.py (215-236)
canonical:   ms_g_eval.py:_call_adapter_text (85-134) is the correct copy. It calls
             _validate_provider_credentials (66-83) -> is_runtime_issued_provider_call_credentials,
             deep-copies provider_credentials.app_config, sets credentials_resolved, and passes
             PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY through to the adapter. The primitives all six
             copies stitch together already live in LLM_Calls/adapter_utils.py (normalize_provider
             :53, ensure_app_config :143, resolve_provider_model :160,
             resolve_provider_api_key_from_config :183, get_adapter_or_raise :202,
             split_system_message :209) — what is missing there is the ~30-line wrapper.
destination: LLM_Calls/adapter_text.py — one responsibility: "given a provider, messages and an
             optional runtime credential context, return the adapter's text reply." It sits beside
             adapter_utils.py, which already owns every primitive it needs. Explicitly NOT
             http_client.py and NOT Utils.py.
knowledge:   "how a caller-supplied BYOK credential context reaches the provider adapter, and what
             happens when it is absent or forged"
scenario:    `eval_runner._call_adapter_text` has no `provider_credentials` or `credentials_resolved`
             parameter at all. Its api_key line (:129) is an unconditional
             `api_key or resolve_provider_api_key_from_config(provider, cfg)`. So an evaluation of
             type `label_choice` or `nli_factcheck` run by a user in a BYOK deployment, where the
             caller's credentials were resolved into a ProviderCallCredentials context, calls the
             provider with the SERVER's configured key instead — silently, once per sample. The
             G-Eval path through `ms_g_eval` raises `ChatConfigurationError("Provider credential
             context is invalid.")` under the same conditions. Two evaluation types therefore bypass
             a gate the third enforces, and the bypass shows up as a billing and attribution
             discrepancy rather than an error.
impact:      High: the divergence is on a credential boundary, it is live on two evaluation types,
             and the test that would catch it (`test_specialized_provider_auth_failures.py:56`)
             patches the guarded copy only.
tests:       tests/Evaluations/unit/test_specialized_provider_auth_failures.py:56 (ms_g_eval copy);
             tests/LLM_Calls/test_provider_adapter_runtime_boundary.py:654,665 (ms_g_eval and
             prompt_studio copies). The eval_runner copy is reached by
             tests/Evaluations/unit/test_eval_runner.py by import-grep, with no credential assertion.
effort:      moderate — promoting one wrapper is mechanical, but the eval_runner copy's extra
             `system_message`/`response_format`/`max_tokens` parameters must be preserved, and the
             CLI pair should collapse into cli/api_utils.py (which already hosts the sixth
             byte-identical function) rather than into the new module.
owner-only:  no
confidence:  confirmed (four in-module copies, the byte-identical CLI pair verified by diff, and the
             two live call sites of the unguarded copy); confirmed (the credential parameter is
             absent, not merely defaulted)
```

### FINDING evaluations-010 — the Evaluations audit bridge is a drifted clone of the Embeddings one and silently drops requester attribution

```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       Evaluations/audit_adapter.py:_emit (147-179)          <- no ip_address/endpoint/method
             Embeddings/audit_adapter.py:_emit (118-150)           <- accepts and forwards all three
             shared-but-separately-maintained machinery, near-identical in both files:
               _int_env  (Evaluations :36-45 catches Exception; Embeddings :35-44 catches (TypeError, ValueError))
               _ensure_sync_loop / _SYNC_LOOP_THREAD (Evaluations :82-88; Embeddings :83-89)
               _shutdown (Evaluations :99-113 suppresses Exception; Embeddings :100-114 suppresses (RuntimeError, TypeError, ValueError))
               _schedule (Evaluations :118-145; Embeddings :117-144)
             callers that can no longer attribute:
               Evaluations/audit_adapter.py:log_webhook_registration_async (216-220)
               Evaluations/audit_adapter.py:log_webhook_unregistration_async (222-226)
             both files also carry the SAME core -> api layering inversion:
               Evaluations/audit_adapter.py:20 and Embeddings/audit_adapter.py:23 import
               api/v1/API_Deps/Audit_DB_Deps
canonical:   NONE — two siblings, no designated owner. Embeddings/audit_adapter.py:_emit is the more
             correct of the two: it builds a full AuditContext(user_id, ip_address, endpoint, method).
destination: core/Audit/sync_bridge.py — one responsibility: running mandatory audit writes from
             synchronous call sites on a dedicated loop and blocking until flushed. It owns the
             thread, the timeout env var and `_emit`; the two modules keep only their event
             vocabularies. This also removes one of the two `api/v1/API_Deps` imports from core/.
knowledge:   "how a synchronous caller emits a mandatory audit event, and what context that event
             must carry"
scenario:    `Evaluations/audit_adapter._emit (147-179)` constructs `AuditContext(user_id=user_id)`
             and nothing else. `log_webhook_registration_async (216-220)` emits
             `UEvent.SECURITY_VIOLATION` on a failed registration. That audit row therefore has a
             null `ip_address`, `endpoint` and `method`. An operator investigating repeated failed
             webhook registrations — the signature of someone probing the SSRF allowlist — cannot
             correlate those rows to a source address, while the byte-adjacent Embeddings bridge
             records exactly that for its own SECURITY_VIOLATION events. Separately, each module
             starts its own daemon thread and event loop (`evaluations-audit-sync-loop` and
             `embeddings-audit-sync-loop`) for the same job.
impact:      Medium: the audit record exists, so nothing is lost outright — but the field that makes
             a security event actionable is absent on one of two identical bridges, and the
             divergence is invisible because each file has its own tests.
tests:       tests/Evaluations/test_evaluations_audit_adapter.py (import-grep). Pins this bridge's
             behaviour in isolation; nothing compares the two.
effort:      moderate — mechanical to unify, but the two `_int_env`/`_shutdown` exception-tuple
             differences must be reconciled deliberately (the Embeddings narrow tuples are the
             better choice; the Evaluations bare `except Exception` is the drift).
owner-only:  no
confidence:  confirmed (the clone relationship and the missing parameters, verified by diff);
             confirmed (SECURITY_VIOLATION is among the events that lose attribution)
```

### FINDING evaluations-011 — two modules own the same webhook tables, with a drifted foreign key, so the effective schema depends on which one initialises first

```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       Evaluations/evaluation_manager.py:_init_database (254-297)
               webhook_registrations DDL at :255-274, webhook_deliveries DDL at :277-297
             Evaluations/webhook_manager.py:_webhook_schema_sql (170-247)
               Postgres branch :171-208, SQLite branch :210-246
             the drift: evaluation_manager.py:295 adds
               FOREIGN KEY (evaluation_id) REFERENCES internal_evaluations(evaluation_id)
               which webhook_manager.py's SQLite branch (:231-246) does NOT have
             FK enforcement is ON for the evaluation_manager path:
               evaluation_manager.py:71-72 -> DB_Management/sqlite_policy.py:configure_sqlite_connection
               with the module default foreign_keys=True (sqlite_policy.py:38, :52)
canonical:   NONE. `webhook_manager.py:_webhook_schema_sql` is the better owner — it is the module
             that actually reads and writes these tables, it handles both backends, and it has a
             dedicated test (`tests/Evaluations/unit/test_webhook_manager_backend_schema.py`).
             `evaluation_manager.py`'s copy is annotated in-code as
             "# Create webhook registrations table (needed for webhook tests)" (:254) — a test
             convenience that became a second production schema owner.
destination: n/a — the fix is deletion, not a new module: evaluation_manager stops declaring tables
             it does not use, and the fixture that needed them calls WebhookManager's initialiser.
knowledge:   "the shape of webhook_registrations and webhook_deliveries, and whether a delivery row
             is foreign-keyed to internal_evaluations"
scenario:    `CREATE TABLE IF NOT EXISTS` is a no-op when the table exists, so whichever module
             touches a given evaluations DB file first wins. If `EvaluationManager._init_database`
             runs first — it does on the `benchmark_api.py:23` import path and on any CLI entry
             (`cli/evals_cli.py:114`, `cli/evals_cli_enhanced.py:384`, `cli/benchmark_cli.py:333`) —
             `webhook_deliveries` is created WITH the FK to `internal_evaluations(evaluation_id)`.
             `WebhookManager._create_delivery_record` (:649) then inserts a row whose
             `evaluation_id` is a run or evaluation id from the `runs`/`evaluations` tables, which is
             not present in `internal_evaluations`. With `PRAGMA foreign_keys=ON` — the
             sqlite_policy default that `evaluation_manager._connect` applies — that insert fails.
             If `WebhookManager` initialises first, the same insert succeeds, because its DDL has no
             such FK. Webhook delivery persistence therefore succeeds or fails according to the
             import order of two modules that do not reference each other.
impact:      Medium: it needs a specific ordering to bite, and the affected write is a delivery
             audit row rather than the delivery itself. But it is a schema owned twice, which means
             the next column added to one copy silently does not exist in databases initialised by
             the other.
tests:       tests/Evaluations/unit/test_webhook_manager_backend_schema.py (pins one owner);
             tests/Evaluations/integration/test_api_endpoints.py, test_production_features_integration.py
             (import-grep). None asserts the two DDLs agree.
effort:      cheap — delete the duplicate DDL from `evaluation_manager._init_database` and have the
             test fixture that needed it construct a WebhookManager. Gated on finding that fixture.
owner-only:  no
confidence:  confirmed (both DDLs, the FK drift, and that foreign_keys defaults ON via
             sqlite_policy); probable-risk (that the ordering actually occurs in a deployed
             configuration — reasoned from the CLI and benchmark_api import paths, not executed)
```

### FINDING evaluations-012 — webhook lookup uses a weaker definition of "test mode" than the module's auth layer, and its last fallback drops the user filter

```
axis:        encapsulation
class:       divergent-copies
severity:    Medium
sites:       Evaluations/webhook_manager.py:_get_webhooks (585-593)  <- fallback 1, keeps user_id
             Evaluations/webhook_manager.py:_get_webhooks (601-613)  <- fallback 2, NO user_id filter
             gate used:      core/testing.py:is_test_mode (42-48) — env var only
             gate available: api/v1/endpoints/evaluations/evaluations_auth.py:_evals_test_mode_bypass_enabled (52-53)
                             — is_explicit_pytest_runtime() AND TEST_MODE
canonical:   evaluations_auth.py:_evals_test_mode_bypass_enabled (52-53) is the correct predicate for
             a test-only relaxation inside this module: it requires PYTEST_CURRENT_TEST to be set,
             so it cannot be turned on by an environment variable alone.
destination: n/a — the fix is to use the stricter predicate that already exists, or to remove the
             second fallback entirely.
knowledge:   "what counts as test mode, and what a test-mode relaxation is permitted to relax"
scenario:    `core/testing.is_test_mode()` returns True whenever `TEST_MODE` or `TLDW_TEST_MODE` is
             truthy in the environment — no pytest check, no production-environment check (the file
             has an `is_production_like_env()` helper at :69 that this path does not consult).
             `webhook_manager._get_webhooks` line 604 runs
             `SELECT id, url, secret, retry_count, timeout_seconds FROM webhook_registrations
              WHERE active = ?` with **no `user_id` predicate**. So on any deployment where
             `TEST_MODE=1` is set — a staging box, a container that inherited the flag, a developer
             running against shared data — a user whose own event matches no webhook receives every
             other user's active webhook rows including their `secret`, and the evaluation payload
             is then delivered to those users' URLs by `send_webhook (499-538)`. The same module's
             auth layer twelve files away refuses to relax anything unless pytest is actually
             running. Two answers to the same question, and the weaker one guards the cross-tenant
             path.
impact:      Medium rather than High because it requires TEST_MODE to be set, which is not the
             default. It is High if TEST_MODE ever reaches a shared environment, and the repo's own
             cross-user isolation audit records that isolation mechanisms in this codebase tend to
             default off.
tests:       tests/Evaluations/integration/test_webhook_multi_user_api.py;
             tests/Evaluations/test_evaluations_webhooks_endpoint_sanitization.py (import-grep).
             The integration test runs WITH test mode on, so this fallback is inside the path under
             test rather than excluded from it.
effort:      cheap — swap the predicate, or delete the second fallback. The risk is that some test
             depends on the relaxation, which is exactly what should be found out.
owner-only:  no
confidence:  confirmed (the query without user_id, the gate, and the gate's env-only definition);
             probable-risk (that TEST_MODE reaches a multi-tenant environment — a deployment
             property this audit cannot observe)
```

### FINDING evaluations-013 — the five recipe modules each re-derive the same six decision helpers, with four different confidence formulas and two different single-candidate contracts

```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       _detect_dataset_mode (6):
               recipes/summarization_quality.py (181-188)  } byte-identical
               recipes/rag_answer_quality.py (666-673)     }
               recipes/rag_retrieval_tuning.py (890-897)   }
               recipes/embeddings_retrieval.py (316-323)   }
               recipes/persona_dialogue_tree_robustness.py (492-499) }
               recipe_runs_service.py (499-516)  <- DRIFTED: treats "mixed" as a validation error
             _confidence_score (4), three formulas:
               recipes/rag_answer_quality.py (929-934)   } byte-identical, /10.0 denominator
               recipes/embeddings_retrieval.py (493-498) }
               recipes/summarization_quality.py (336-346)  <- unscaled margin term
               recipes/rag_retrieval_tuning.py (443-447)   <- /25.0 denominator, rounds to 3dp
             _winner_margin (4), two contracts:
               recipes/summarization_quality.py (327-334)  -> 0.0 for a single candidate
               recipes/rag_answer_quality.py (911-927)     -> 0.0
               recipes/embeddings_retrieval.py (476-491)   -> 0.0
               recipes/rag_retrieval_tuning.py (434-441)   -> 1.0 for a single candidate
             _normalize_weights (3), three validation strictnesses:
               recipes/summarization_quality.py (166-179)          silently defaults on bad input
               recipes/rag_answer_quality.py (426-444)             raises on unknown keys
               recipes/rag_answer_quality_execution.py (1071-1083) raises on non-Mapping, IGNORES unknown keys
             _reserve_review_sample (5):
               recipes/rag_answer_quality.py (675-683) } byte-identical, ceil(n*0.2) floored at 3
               recipes/summarization_quality.py (190-198) }
               recipes/embeddings_retrieval.py (325-333)      same math, key renamed to sample_query_ids
               recipes/persona_dialogue_tree_robustness.py (502-510)  n//5 floored at 1
               recipes/rag_retrieval_tuning.py (899-920)      parameterized from run_config
             _extract_sample_id (6), two fallback index bases:
               recipes/rag_answer_quality.py (650-664) == recipes/rag_answer_quality_execution.py (975-989)
               recipes/summarization_quality.py (399-408) ~ recipe_runs_jobs_worker.py:_extract_summarization_sample_id (259-268)
               recipes/rag_retrieval_tuning.py (922-927) == recipe_runs_jobs_worker.py:_extract_rag_sample_id (334-339)
               first group returns f"sample-{index}", second returns f"sample-{index + 1}"
canonical:   NONE. `recipes/base.py` exists and is the natural home but currently holds only a thin
             manifest base class.
destination: tldw_Server_API/app/core/Evaluations/recipes/decisions.py — one responsibility: the
             shared decision vocabulary every recipe needs (dataset mode, review sampling, winner
             margin, confidence, best-slot picking, sample identity). Each recipe keeps only its own
             metric vocabulary and weights.
knowledge:   "how a recipe decides which candidate won, how confident it is, and how it identifies a
             sample"
scenario:    Two concrete divergences, both live. (1) `rag_retrieval_tuning._winner_margin (434-441)`
             returns `1.0` when there is exactly one candidate; the other three return `0.0`. That
             value feeds `_confidence_score(winner_margin=...)` eleven lines below, where
             `margin_factor = min(max(1.0,0)/0.25, 1.0) = 1.0`, so a single-candidate
             rag_retrieval_tuning run with 25+ samples and zero spread reports
             `0.5*1 + 0.5*1 - 0 = confidence 1.0` — maximum confidence in a comparison with nothing
             to compare against. The same single-candidate run under `summarization_quality` reports
             `0.45 + 0.35 + 0 - 0 = 0.80`. (2) `rag_answer_quality._normalize_weights (426-444)`
             REJECTS an unknown weight key with a ValueError listing the allowed keys, while
             `rag_answer_quality_execution._normalize_weights (1071-1083)` — the executor for the
             same recipe — silently ignores it. A run_config with a typo'd weight therefore passes
             validation only if it reaches the executor first, and the weight the user thought they
             set is dropped without a word.
impact:      Medium: these are not crashes, they are the recipe system reporting different
             confidence for the same evidence depending on which recipe you ran, and accepting a
             config its own validator would reject. Every new recipe added copies one of the four
             formulas at random.
tests:       tests/Evaluations (recipe tests by import-grep, incl. test_rag_pipeline_runner.py and
             the recipe-run worker tests). Each recipe's helper is exercised through its own recipe;
             nothing cross-checks them.
effort:      moderate — mechanical extraction, but picking ONE confidence formula changes reported
             confidence for at least three of the four recipes, so it needs a short design note and
             a decision about which formula is intended.
owner-only:  no
confidence:  confirmed (every site and every divergence listed, verified by reading the bodies);
             confirmed (the single-candidate confidence arithmetic, computed by hand from the two
             cited functions)
```

### FINDING evaluations-014 — a 705-line connection pool is warmed at startup, holds ten idle SQLite connections per worker, and serves nothing

```
axis:        efficiency
class:       adoption-gap
severity:    Medium
sites:       Evaluations/connection_pool.py (entire file, 705 lines; 14 commits in 12 months)
               ConnectionPool (124-586), _initialize_pool (188-198), _create_connection (200-238)
               PooledConnection._configure_connection (67-80) incl. PRAGMA mmap_size=268435456 (:75)
               EvaluationsConnectionManager (587-645), pool_size default 10 (:619)
               get_connection (664-667), get_connection_async (669-672),
               get_connection_health (674-679), get_connection_stats (681-686)
             ONLY EXTERNAL CONSUMERS:
               services/startup_evaluations_warmup.py:31,34  (constructs it)
               services/shutdown_evaluations_resources.py:46 (tears it down)
             THE THREE PATHS THAT ACTUALLY OPEN CONNECTIONS INSTEAD:
               Evaluations/evaluation_manager.py:_connect (64-75)    sqlite3.connect per call
               Evaluations/user_rate_limiter.py:_connect (228-248)   sqlite3.connect per call
               DB_Management/Evaluations_DB.py:get_connection (246-259) sqlite3.connect per call
             AND THE ABANDONED WIRING:
               Evaluations/user_rate_limiter.py:87
               "# from ...Evaluations.connection_pool import get_connection  # unused"
canonical:   Evaluations/connection_pool.py IS the designated pooling mechanism. It is bypassed by
             every consumer, which is the adoption gap.
destination: n/a — either wire the three `_connect` paths through the existing manager, or delete
             the file. Both are defensible; keeping it warmed-but-unused is not.
knowledge:   "how the Evaluations module obtains a configured SQLite connection" — currently
             answered four times, and the answer nobody uses is the one that was built for it.
cost-driver: At startup, `_warm_evaluations_connection_manager()` constructs
             `EvaluationsConnectionManager`, whose `ConnectionPool.__init__` calls
             `_initialize_pool` and opens `pool_size` connections (default **10** outside test mode,
             `connection_pool.py:619`), each running `configure_sqlite_connection` plus
             `PRAGMA mmap_size=268435456` (:75). Those file descriptors, WAL reader slots and 256MB
             mmap windows are held for the process lifetime and never read from.
             **Scales with `pool_size` x uvicorn worker count.** Meanwhile every real query pays a
             fresh `sqlite3.connect` + pragma configure, so the cost the pool exists to remove is
             still paid **per DB call**, scaling with request rate and with sample count on the
             batch paths.
             Secondary: the manager defaults its db_path to
             `DatabasePaths.get_evaluations_db_path(DatabasePaths.get_single_user_id())` (:601-604),
             so in multi-user mode the ten connections point at the single-user database.
             Related: `api/v1/endpoints/benchmark_api.py:23` constructs `EvaluationManager()` at
             MODULE IMPORT, which runs `_init_database` and creates that same single-user DB file as
             a side effect of importing a router.
tests:       tests/Services/test_startup_evaluations_warmup.py,
             tests/Services/test_shutdown_evaluations_resources.py,
             tests/AuthNZ/unit/test_test_mode_runtime_guard.py:122 (import-grep). All three assert
             construction and teardown; none asserts a connection is ever served.
effort:      cheap if the decision is "delete" (nothing imports the checkout API). Moderate if the
             decision is "adopt", because the three `_connect` paths have divergent kwargs
             (`row_factory` vs `detect_types`/`timeout`) and two different busy timeouts
             (10_000 in evaluation_manager.py:40 and user_rate_limiter.py:204; 30_000 in
             connection_pool.py:73 and db_adapter.py:133).
owner-only:  no (benchmark_api.py:23 is under api/v1/, so touching that one line IS owner-only)
confidence:  confirmed (no consumer, verified by symbol grep across app/ and tests/); confirmed
             (pool_size 10 and the eager `_initialize_pool`); assumption (the operational cost of
             ten idle connections is a nuisance rather than a limit — not measured)
```

### FINDING evaluations-015 — the batch paths serialize independent LLM calls, and one creates a fresh event loop per sample

```
axis:        efficiency
class:       n/a
severity:    Medium
sites:       recipe_runs_jobs_worker.py:_execute_rag_retrieval_tuning_recipe_run (907-947)
               asyncio.run(...) INSIDE the per-sample loop at :916-921
             recipes/rag_answer_quality_execution.py:_resolve_live_contexts (397-418)
               asyncio.run(...) per sample at :408
             eval_runner.py:_execute_rag_pipeline_run (716-891)
               per-sample awaits at :725 (pipeline), :779 (rag_evaluator), :819 and :827 (custom
               metrics), plus a blocking self.db.update_run_progress at :884 and :734
             rag_evaluator.py:_evaluate_context_precision (846-872)  per-context sequential await
             rag_evaluator.py:_evaluate_context_relevance (898-934)  per-context sequential await
             embeddings_abtest_service.py:run_vector_search_and_score (673-847)
               per-query create_reranker(strat, rconf) at :779, blocking collection.query at :711,
               manager.get_or_create_collection re-looked-up per query at :706
             ms_g_eval.py:run_geval (358-380)  four metric calls in sequence
             recipe_runs_jobs_worker.py:_execute_summarization_recipe_run (809-841)
             unified_evaluation_service.py:evaluate_qa3 (1203-1213)
             wordbench_runner.py:run_benchmark (138-154)
             eval_runner.py:_build_ephemeral_index (1076-1133)
               get_embedding_config() per document at :1094; one create_embeddings_batch per
               document at :1095-1100; one sklearn cosine_similarity call per adjacent chunk pair
               at :1114 and :1130
canonical:   eval_runner.py:_process_batch (1294-1360) — the module's own correct shape:
             `asyncio.Semaphore(max_workers)` at :1320 plus
             `asyncio.gather(*tasks, return_exceptions=True)` at :1358, with per-task timeout and
             exception capture. `response_quality_evaluator.evaluate (143-148)` and
             `webhook_manager.send_webhook (499-538)` also already do it correctly.
destination: n/a — the pattern to copy is in the same file.
knowledge:   n/a (efficiency finding)
cost-driver: **`asyncio.run` per sample** (`recipe_runs_jobs_worker.py:916`): one event-loop
             create + tear-down per sample, `C x N` times, where C = number of candidates and
             N = dataset size. One `asyncio.run` wrapping a `gather` over the dataset would make it
             `C`.
             **Per-context judge calls** (`rag_evaluator.py:846, :898`): **K sequential LLM
             round-trips per sample per metric, K = len(contexts)**; with both context metrics
             enabled that is `2K`. Called from inside the rag_pipeline per-sample loop, so the
             serialized total is `G x N x 2K` where G = len(config_grid).
             **Reranker re-instantiation** (`embeddings_abtest_service.py:779`): for the
             `CROSS_ENCODER` and `FLASHRANK` strategies `create_reranker` constructs a ranking model
             — `A x Q` constructions, A = len(arms), Q = len(queries).
             **G-Eval metric serialization** (`ms_g_eval.py:358-380`): 4 sequential LLM calls per
             invocation, and `run_geval` is called once per sample by both the summarization worker
             and the answer-quality executor, making it the 5x multiplier in `C x N x 5`.
             **Per-document embedding config reload** (`eval_runner.py:1094`): `get_embedding_config()`
             once per corpus document per config, `G x D`; and one embedding round-trip per document
             rather than per corpus, `G x D` instead of `G`.
             **Blocking DB inside async** (`eval_runner.py:884`, `webhook_manager.py:649, :695, :705`):
             a synchronous connect/UPDATE/commit/close on the event loop, `G x N` times for progress
             and `2 + retry_count` times per webhook per event.
tests:       tests/Evaluations/test_rag_pipeline_runner.py, tests/Evaluations/unit/test_eval_runner.py,
             tests/Evaluations (recipe-run worker tests), tests/Evaluations/integration (import-grep).
             All functional; none asserts call counts or concurrency, so none would notice a
             regression here either way.
effort:      moderate per site and independently shippable. The two `asyncio.run`-in-a-loop sites
             are the cheapest and largest wins. `create_reranker` hoisting out of the query loop is
             a two-line change. The rag_evaluator context loops are a direct `gather` with the
             existing `_run_bounded_rag_analyze` already bounded by a daemon boundary.
owner-only:  no
confidence:  confirmed (every site read; the `asyncio.run` placement, the `create_reranker`
             placement and the per-context loops verified directly); assumption (the actual wall-clock
             magnitude — no profiling was run)
```

### FINDING evaluations-016 — run results accumulate whole in memory, dataset "pagination" happens after the full blob is parsed, and one report silently truncates at 1000 rows

```
axis:        efficiency
class:       n/a
severity:    Medium
sites:       eval_runner.py:_execute_evaluation (373, 396, 415-422)
               sample_results accumulates every sample dict; results embeds the full list plus a
               second failed-only list; store_run_results json.dumps the whole thing
             DB_Management/Evaluations_DB.py:store_run_results (2019-2040)  one JSON column
             DB_Management/Evaluations_DB.py:_row_to_dataset_dict (2655-2662)
               sample_limit/sample_offset applied by Python slicing AFTER _json_maybe parses the
               entire samples blob
             DB_Management/Evaluations_DB.py:create_dataset (2042-2070)  samples as one json.dumps
             consumers that then take the whole list:
               eval_runner.py:_get_samples (1242), recipe_runs_service.py:_resolve_dataset (446),
               recipe_runs_jobs_worker.py (148, 167)
             recipe_runs_jobs_worker.py:_collect_embeddings_candidate_results (654)
               db.list_abtest_results(test_id, limit=1000, offset=0) — hard cap, no loop, no total check
             embeddings_abtest_service.py:compute_significance (995)
               db.list_abtest_results(..., limit=100000, offset=0), then an A x A x Q Python loop (1008-1026)
             DB_Management/Evaluations_DB.py:get_abtest_arms (1407-1424), get_abtest_queries (1463-1479)
               no LIMIT, no OFFSET; get_abtest_queries is loaded three times per run
               (embeddings_abtest_service.py:635, :990 and recipe_runs_jobs_worker.py:653)
             benchmark_loaders.py:load_benchmark_dataset (432-455)
               the full dataset is loaded and then sliced to `limit` at :453-455
             user_rate_limiter.py:_write_request_usage (845)
               INSERT into rate_limit_tracking on every gated request; no DELETE anywhere in app/
canonical:   api/v1/utils/pagination.py — build_offset_pagination_meta (12),
             build_cursor_pagination_meta (40), build_page_pagination_meta (57). Every paginated
             read in this module hand-rolls limit/offset instead.
destination: n/a for the memory findings. For pagination, adopt the existing
             api/v1/utils/pagination.py rather than adding anything.
knowledge:   n/a (efficiency finding)
cost-driver: **Unbounded result accumulation**: peak memory and the single-blob write are both
             `O(N x per-sample payload)`, N = sample count, with the failed subset serialized twice
             into the same JSON document. The same shape repeats in
             `recipe_runs_jobs_worker.py (843-870, 949-983)` and
             `embeddings_abtest_service.run_abtest_full (884)`.
             **Post-load dataset pagination**: every "paginated" dataset read costs `O(full dataset)`
             in I/O, JSON parse time and memory regardless of `sample_limit`. Scales with dataset
             size, not page size, and it is on the request path via `recipe_runs_service:446`.
             **Post-load benchmark limit**: `limit=10` against a 4,326-sample SimpleQA split still
             fetches and parses all 4,326 (`benchmark_registry.py:109` records that size).
             **Silent truncation at 1000**: when `arms x queries > 1000`,
             `_collect_embeddings_candidate_results` reports on a truncated subset with no error and
             no indication in the output. This is a correctness cliff reached by a plausible config
             (8 arms x 150 queries), not just a cost.
             **`limit=100000`** in `compute_significance` materialises every result row including
             `ranked_documents`/`ranked_metadatas` JSON it never reads, then runs `O(A^2 x Q)`
             Python.
             **`rate_limit_tracking` grows forever**: one INSERT per gated evaluations request,
             never pruned anywhere in `app/`, degrading the three indexes created at
             `user_rate_limiter.py:320-322`. Scales with total lifetime request count.
tests:       tests/Evaluations (worker and A/B tests by import-grep);
             tests/Evaluations/test_db_adapter.py; tests/DB_Management. None exercises a dataset
             large enough for the 1000-row cap or the post-load slicing to be observable.
effort:      mixed. The 1000-row cap is cheap and should be fixed first — it is a wrong answer, not
             a slow one. Dataset-sample pagination is expensive: it means moving samples out of a
             JSON column into a child table, which is a schema migration and needs the full
             design-first treatment. `rate_limit_tracking` retention is cheap (a periodic delete in
             the existing startup cleanup worker, which already reaps ephemeral collections).
owner-only:  no
confidence:  confirmed (every site read; the post-load slicing and the 1000-row cap verified
             directly); confirmed (no DELETE on rate_limit_tracking anywhere in app/, verified by grep)
```

### FINDING evaluations-017 — ephemeral vector collections are created before they are registered, so a mid-build failure orphans them past the TTL reaper

```
axis:        sequential-coupling
class:       n/a
severity:    Medium
sites:       eval_runner.py:_build_ephemeral_index (1063-1065)
               await adapter.create_collection(collection_name)
             eval_runner.py:_execute_rag_pipeline_run (698-711)
               the build is awaited at :698, appended to built_collections at :705, and only then
               registered at :709 via self.db.register_ephemeral_collection(...)
             eval_runner.py:_execute_rag_pipeline_run (986-998)
               cleanup is a plain trailing block, not a finally, and is gated on
               rp.get("cleanup_collections")
             the reaper that can only see registered collections:
               DB_Management/Evaluations_DB.py:list_expired_ephemeral_collections (2937-2956)
               services/startup_cleanup_workers.py:227-236
canonical:   NONE
destination: n/a — the fix is an ordering change plus a try/finally.
knowledge:   "a vector collection that exists must be discoverable by the reaper that deletes it"
scenario:    `_build_ephemeral_index` creates the collection at :1065 and then spends the rest of
             the function chunking, embedding and upserting (:1076-1133) — every one of those steps
             can raise (embedding provider failure, adapter error, `sklearn` import). The caller
             registers the collection in `ephemeral_collections` only at :709, AFTER
             `_build_ephemeral_index` has returned successfully. So a failure anywhere between
             `create_collection` and the function's return leaves a real collection in the user's
             vector store that was never written to `ephemeral_collections`, and
             `list_expired_ephemeral_collections` (which selects from that table) can never find it.
             The startup cleanup worker will therefore never delete it — the orphan is permanent.
             The trailing cleanup block at :986 does not help either: it is gated on an opt-in
             `cleanup_collections` flag and, because it sits at the end of the function body rather
             than in a `finally`, it is skipped entirely when the config-grid loop raises.
impact:      Medium: the TTL reaper covers the ordinary case, so this only bites on failure — but
             failure during a long embedding sweep is exactly when a config grid has created several
             collections, and the leak is unbounded in time rather than TTL-bounded.
tests:       tests/Evaluations/test_rag_pipeline_runner.py; tests/Evaluations/test_evaluations_unified.py
             (import-grep). Neither exercises a failure between create and register.
effort:      cheap — register before building (the row is cheap and idempotent via
             `INSERT OR IGNORE`, Evaluations_DB.py:2928-2933), and move the cleanup block into a
             `finally`.
owner-only:  no
confidence:  confirmed (the ordering, the reaper's dependence on the table, and that cleanup is not
             in a finally)
```

### Noted, not filed

Real observations that did not clear a drop rule, recorded so the next reviewer does not re-derive them.

- **Bool coercion, five implementations, three truthy sets.** `recipes/rag_retrieval_tuning.py:_parse_bool_value (832-843)` and `recipes/rag_retrieval_tuning_candidates.py:_normalize_bool_value (317-328)` are AST-identical under two names with their own `_TRUE_VALUES` constants; `recipes/persona_dialogue_tree_robustness.py:_coerce_run_config_bool (283-292)` accepts `"y"`/`"n"` that the first two reject and rejects `int` 1 that they accept; `synthetic_eval_generation.py:_coerce_optional_bool (457-469)` returns `None` where the others raise; `config_validator.py:_is_truthy (180-183)` accepts five spellings while `config_validator.py:212` and `config_manager.py:521` — in the same and adjacent files — accept only the literal `"true"`. A canonical exists and is bypassed by all seven: `core/Utils/common.py:6 parse_boolean`, backed by `core/testing.py:30 is_truthy`, with `core/testing.py:35 env_flag_enabled` covering the two `os.getenv` cases exactly. Not filed because the sites are all config-parsing at module boundaries and I could not name a specific future edit that breaks — but it is the cheapest adoption gap in the module.
- **Two Prometheus collector classes define colliding metric names.** `metrics.py:155` and
  `metrics_advanced.py:155` both declare `webhook_deliveries_total`, with label sets
  `['event_type','outcome']` and `['event_type','status']`; `webhook_response_time_seconds` vs
  `webhook_delivery_latency_seconds` differ only in a `2.5` vs `2.0` bucket boundary.
  The collision is defused, not resolved: `get_advanced_metrics (515-530)` forces
  `use_separate_registry=True` and says so in a comment at :520-521, and
  `metrics_advanced.py:538` binds the module-level instance that way. Consequence: the 15 advanced
  collectors live on a registry that is not the default one. I did **not** verify the `/metrics`
  export surface, so I will not assert they are unexported. Also note `metrics.py` repeats a
  `try: Counter(...) except ValueError: REGISTRY._names_to_collectors[name]` idiom 21 times, poking
  a private prometheus_client attribute.
- **`_close_response`** is byte-identical at `webhook_manager.py:75-84` and
  `webhook_security.py:69-78`, with six more identical copies outside the module
  (`Web_Scraping/enhanced_web_scraping.py:1253`, `Web_Scraping/preflight/adapters/http.py:175`,
  `Watchlists/fetchers.py:82`, `RAG/rag_service/quick_wins.py:602`,
  `Ingestion_Media_Processing/Audio/Audio_Transcription_External_Provider.py:81`,
  `Chunking/async_chunker.py:28`, `VoiceAssistant/router.py:84`). Repo-wide, not Evaluations-owned;
  belongs to whoever audits `core/http_client.py`.
- **`eval_runner.py:_extract_content` is byte-identical at :1859-1878 and :2052-2071**, and the
  surrounding blocks `:1855-1932` (label_choice) and `:2048-2128` (nli_factcheck) differ only in log
  strings and comments — roughly 78 duplicated lines. Folded into evaluations-009's site list rather
  than filed separately, since both blocks exist to call the same duplicated adapter wrapper.
- **`persona_chat_judge_execution.py:_prediction_from_response (299)`** does a bare
  `json.loads(response_text)` and returns the `"malformed_json"` error key for any reply wrapped in
  a ``` fence, while `benchmark_utils.py:166-199` has a four-pattern fence-stripping ladder for the
  same job. A real behavioural gap, but the judge prompt demands strict JSON and the strictness may
  be deliberate; without knowing that intent I cannot assert a defect.
- **Layering.** 25 `core/Evaluations -> app/api` imports. 21 are the mild schemas-only shape
  (`api/v1/schemas/evaluation_recipe_schemas`, `evaluation_schemas_unified`,
  `synthetic_eval_schemas`, `embeddings_abtest_schemas`, `rag_schemas_unified`) — the honest reading
  is that those schemas are in the wrong package, not that core is wrong to need them. Four are true
  inversions: `embeddings_abtest_service.py` (x2) and `embeddings_abtest_runner.py` importing
  `api/v1/endpoints/embeddings_v5_production_enhanced`, and `audit_adapter.py:20` importing
  `api/v1/API_Deps/Audit_DB_Deps` (carried inside evaluations-010, whose destination removes it).
  Not filed as its own finding: the proportionate repo-wide response is the AST import-ratchet
  precedent at `tests/lint/test_endpoint_auth_deps_import_boundary.py`, seeded at the current count
  so it can only go down — which is a repo-level recommendation, not an Evaluations one.
- **`rag_evaluator.py:363-369`** — two expression statements with discarded values, left from a
  refactor. Dead and confusing; no behavioural consequence. Already noted in stage 2.

## Suggested Refactor/Actions

1. **Ship the two security-shaped fixes first, separately, each with its own regression test.**
   `evaluations-008` (import the guarded `_run_config_for_execution` instead of re-declaring it) and
   `evaluations-012` (use `_evals_test_mode_bypass_enabled`, or delete the user-filterless fallback).
   Both are small diffs on a credential/tenant boundary. Neither needs a design document; both need a
   test that fails before the change. Bandit runs on touched scope per ADR-005, so confirm the
   touched files clear HIGH/CRITICAL.
2. **`evaluations-007` needs a design note before any edit**, because it changes values already
   persisted and returned. Proposed shape:
   `Docs/Design/2026-MM-DD-evaluations-timestamp-normalization-design.md` covering: fix
   `_ensure_unix_timestamp` to treat an offset-less string as UTC; extract it to
   `core/Evaluations/timestamps.py`; repoint the five bypassing copies; decide whether already-stored
   rows need a backfill. Four of the six sites are under `api/v1/**` and are therefore
   **owner-only** per CONTRIBUTING.md — flag that in the Backlog task.
3. **`evaluations-009` and `evaluations-010` are one question: where does this repo put a
   sync-to-provider or sync-to-audit bridge?** Answer it once, in an ADR, then place
   `LLM_Calls/adapter_text.py` and `core/Audit/sync_bridge.py`. Explicitly rule out `Utils.py` and
   `http_client.py` as destinations in the ADR text. Stage as: (1) new module with the correct
   (`ms_g_eval` / Embeddings) behaviour; (2) migrate the divergent copy and add the credential test
   that would have caught it; (3) migrate the CLI pair into the existing `cli/api_utils.py`;
   (4) delete the copies.
4. **`evaluations-011` and `evaluations-014` are deletions, not refactors.** Remove the duplicate
   webhook DDL from `evaluation_manager._init_database`; decide whether `connection_pool.py` is
   adopted or deleted. Deleting 705 lines that nothing calls is the cheapest maintainability win in
   this module. Do not leave it warmed-but-unused.
5. **`evaluations-013` needs a design note** because unifying `_confidence_score` changes reported
   confidence for three of four recipes. Propose `recipes/decisions.py` with one stated
   responsibility, and stage the migration recipe by recipe so each change is separately reviewable.
   The `_winner_margin` single-candidate contract should be settled first and independently — that
   one is a defect, not a preference.
6. **Efficiency work is independently shippable and should not wait on any of the above.** Order by
   ratio: the 1000-row silent truncation in `evaluations-016` (wrong answer, one line), the two
   `asyncio.run`-in-a-loop sites and the per-query `create_reranker` hoist in `evaluations-015`
   (large win, small diff), the `rag_evaluator` context-loop `gather`s, then
   `rate_limit_tracking` retention in the existing startup cleanup worker. Dataset-sample pagination
   is the only item here that needs a schema migration and the full design-first treatment.
7. **`evaluations-017` is a two-line ordering fix** (register before create, cleanup in a `finally`).
   Ship it with the efficiency batch.
