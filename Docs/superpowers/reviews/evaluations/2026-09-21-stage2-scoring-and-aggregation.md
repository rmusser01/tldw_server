# Stage 2 — Scoring normalization and aggregation math

## Scope

The one thing an evaluations module must get right is the number it returns. This stage audits
every path from "LLM judge emitted a token" to "aggregate score stored on the run": score parsing,
1-5 -> 0-1 normalization, per-metric weighting, mean/pass-rate computation, and the treatment of
samples that failed to measure. Threshold resolution and pass/fail are in scope; persistence,
transport and efficiency are not (stage 3).

Anchored on the behaviour matrix in stage 1.

## Code Paths Reviewed

- `rag_evaluator.py:RAGEvaluator.evaluate (249-428)` — metric dispatch, alias injection at
  `:381-385`, dead expression statements at `:363-369`, overall score at `:415`.
- `rag_evaluator.py:RAGEvaluator._calculate_overall_score (1042-1086)` — equal-weight default at
  `:1057`, weighted sum at `:1063-1068`, observed-range clamp at `:1073-1082`.
- `rag_evaluator.py:RAGEvaluator._normalize_score (1027-1039)` — the correct min-max.
- `rag_evaluator.py:_evaluate_relevance (511-585)`, `_evaluate_faithfulness (587-656)`,
  `_evaluate_answer_similarity (658-830)`, `_evaluate_context_precision (832-884)`,
  `_evaluate_context_relevance (886-945)`, `_evaluate_context_recall (947-999)`.
- `response_quality_evaluator.py:_evaluate_relevance (180-243)`, `_evaluate_completeness (245-308)`,
  `_evaluate_clarity (310-370)`, `_evaluate_accuracy (372-435)`,
  `_evaluate_custom_criterion (505-567)`.
- `ms_g_eval.py:parse_output (425-446)`, `aggregate (137-170)`, `aggregate_llm_scores (572-596)`.
- `eval_runner.py:_eval_summarization (1380-1450)`, `_eval_rag (1453-1515)`,
  `_eval_response_quality (1517-1573)`, `_eval_exact_match (1577-1614)`, `_eval_includes (1616-1658)`,
  `_eval_fuzzy_match (1660-1697)`, `_eval_propositions (1699-1741)`,
  `_eval_label_choice (1743-1938)`, `_eval_nli_factcheck (1939-2133)`.
- `eval_runner.py:_coerce_threshold_value (2132-2141)`, `_extract_threshold_config (2143-2169)`,
  `_resolve_metric_threshold (2171-2184)`, `_evaluate_passed (2186-2214)`,
  `_calculate_aggregate_results (2217-2276)`, `_calculate_metric_stats (2278-2322)`,
  `_calculate_usage (2324-2340)`.
- `eval_runner.py:_execute_evaluation (274-445)` — batching, progress, result assembly.
- `eval_runner.py:_execute_rag_pipeline_run (541-1000)` — config grid, per-config aggregation
  (`:893-928`), leaderboard and best-config selection (`:945-965`).
- `eval_runner.py:_compute_mrr_ndcg (1002-1029)` — **read and found correct**; standard binary-gain
  DCG with an ideal-DCG denominator over `min(|rel|, |retrieved|)`. Not a finding.
- `recipes/rag_answer_quality_execution.py:_coerce_unit_score (1102-1112)`.
- `recipes/summarization_quality.py:_normalize_score (289)` and its call sites `:268-279`.
- `evaluation_manager.py:613, :632` — the 1-10 scoring variants.

## Tests Reviewed

| Test file | What it protects | Downgrades risk? |
| --- | --- | --- |
| `tests/Evaluations/unit/test_rag_evaluator.py:363-375` | Asserts `_normalize_score(1)==0`, `(3)==0.5`, `(5)==1.0`, and clamping at 0 and 6. | **No** — it pins a function with no production callers. It is the evidence *for* EVAL-002, not against it. |
| `tests/Evaluations/property/test_evaluation_invariants.py:102-136` | Hypothesis properties: range, monotonicity, idempotence of `_normalize_score`. | **No** — same misdirection, and it could not be executed locally (`hypothesis` missing from the venv). |
| `tests/Evaluations/unit/test_eval_runner.py` | Runner batching and eval-function dispatch. | Partly — could not be collected locally (`sklearn`). Import-grep reachability only. |
| `tests/Evaluations/test_evaluations_stage3_batch_failfast_and_metrics_none.py` | Batch fail-fast and `metrics=None` handling. | Yes for `_normalize_metrics`; **no** for the mean-with-zeros behaviour in EVAL-004, which is the *intended-looking* path. |
| `tests/Evaluations/unit/test_response_quality_provider_boundary.py` | Provider boundary for `ResponseQualityEvaluator`. | Partly — covers the provider call, not the divergent failure semantics in EVAL-005. |
| `tests/Evaluations/test_rag_pipeline_runner.py` | The rag_pipeline sweep end to end. | **No** — no assertion ties `aggregate.chunk_cohesion` to a per-sample mean (EVAL-003). |
| `tests/Evaluations/unit/test_specialized_provider_auth_failures.py:56` | Monkeypatches `ms_g_eval._call_adapter_text` to reject unresolved credentials. | Yes for the G-Eval path — and it is precisely why the *other* copy going unguarded (stage 3, EVAL-007) is invisible. |

Reachability by import-grep. Not measured coverage.

## Validation Commands

Both numeric claims below were executed against the working tree.

```
$ python3 -c "
from tldw_Server_API.app.core.Evaluations.rag_evaluator import RAGEvaluator
e = RAGEvaluator.__new__(RAGEvaluator)
m  = {'relevance':{'score':1.0},'answer_relevance':{'score':1.0},'context_relevance':{'score':0.0}}
m2 = {'relevance':{'score':1.0},'context_relevance':{'score':0.0}}
print('overall WITH alias dupes :', e._calculate_overall_score(m))
print('overall WITHOUT dupes    :', e._calculate_overall_score(m2))
print('normalize_score(1)       :', e._normalize_score(1))
print('inline raw/5.0 for raw=1 :', 1/5.0)
print('normalize_score(3)       :', e._normalize_score(3), ' vs inline', 3/5.0)"

overall WITH alias dupes : 0.6666666666666666
overall WITHOUT dupes    : 0.5
normalize_score(1)       : 0.0
inline raw/5.0 for raw=1 : 0.2
normalize_score(3)       : 0.5  vs inline 0.6
```

```
$ grep -rn '/ 5\.0\|/5\.0\|/ 3\.0\|/ 10\.0' --include='*.py' tldw_Server_API/app/core/Evaluations | grep -v __pycache__ | wc -l
      17

$ grep -rn "_normalize_score" --include='*.py' tldw_Server_API/app/core/Evaluations | grep -v __pycache__
tldw_Server_API/app/core/Evaluations/rag_evaluator.py:1027:    def _normalize_score(self, score: float) -> float:
tldw_Server_API/app/core/Evaluations/recipes/summarization_quality.py:268:  ... self._normalize_score(metrics.get("grounding"), 1.0),
tldw_Server_API/app/core/Evaluations/recipes/summarization_quality.py:289:    def _normalize_score(self, value: Any, max_score: float) -> float:
  (7 call sites, all of the *summarization_quality* two-argument function — none of RAGEvaluator's)

$ grep -rn "_normalize_score" tldw_Server_API/tests | grep -v __pycache__ | wc -l
       8   (all in tests/Evaluations/unit/test_rag_evaluator.py and tests/Evaluations/property/test_evaluation_invariants.py)
```

## Findings

### FINDING evaluations-001 — alias metric keys are counted twice in the overall RAG score

```
axis:        correctness
class:       n/a
severity:    High
sites:       tldw_Server_API/app/core/Evaluations/rag_evaluator.py:RAGEvaluator.evaluate (381-385)
             tldw_Server_API/app/core/Evaluations/rag_evaluator.py:RAGEvaluator.evaluate (401-409)
             tldw_Server_API/app/core/Evaluations/rag_evaluator.py:RAGEvaluator._calculate_overall_score (1042-1086)
             tldw_Server_API/app/core/Evaluations/eval_runner.py:EvaluationRunner._eval_rag (1490-1507)
canonical:   NONE
destination: n/a — this is a defect, not a consolidation
knowledge:   "which keys in results['metrics'] are distinct metrics" vs "which are back-compat aliases"
scenario:    `evaluate()` records the canonical key and then `setdefault`s an alias pointing at the
             SAME metric dict — `answer_relevance` for `relevance` (:382) and `answer_faithfulness`
             for `faithfulness` (:385). The de-duplication that removes the canonical key
             (:401-409) runs only `if not explicit_metrics`. `eval_runner._eval_rag:1478` always
             passes a non-None `metrics` list (`_normalize_metrics(..., default=[...])` returns a
             list unconditionally, eval_runner.py:217-232), so `caller_provided_metrics` is always
             True on the API path and the de-duplication NEVER runs there. With
             metrics=["relevance","context_relevance"], judge scores relevance=1.0 and
             context_relevance=0.0, `results['metrics']` holds three entries and the equal-weight
             mean is **0.667 instead of 0.5** (executed above). `_eval_rag:1490-1507` then iterates
             the same dict into `scores` and takes `statistics.mean(scores.values())`, so the
             inflated value propagates into per-sample `avg_score`, into `_evaluate_passed`'s
             global-threshold comparison (:1506-1507), and into `mean_score` in
             `_calculate_aggregate_results`. Any metric set containing exactly one of
             relevance/faithfulness plus at least one other metric is scored wrong.
impact:      Silently inflates the headline RAG score whenever relevance or faithfulness is
             requested alongside a non-aliased metric. It moves pass/fail across the default 0.7
             threshold. The wrong number is the product.
tests:       tests/Evaluations/unit/test_rag_evaluator.py; tests/Evaluations/test_evaluation_integration.py;
             tests/Evaluations/test_rag_pipeline_runner.py; tests/Evaluations/unit/test_unified_evaluation_service_mapping.py
             (import-grep reachability; none asserts the arithmetic)
effort:      cheap — exclude alias keys from `_calculate_overall_score`, or emit aliases only in
             the response projection rather than in the scored dict. Well-covered file.
owner-only:  no
confidence:  confirmed (the double-count, executed); confirmed (the always-True explicit_metrics
             path on the runner)
```

### FINDING evaluations-002 — the correct 1-5 normalizer is dead; eleven inline copies use a different formula

```
axis:        duplication
class:       divergent-copies
severity:    High
sites:       CORRECT, UNUSED:
               rag_evaluator.py:RAGEvaluator._normalize_score (1027-1039)   (score-1)/4
             INLINE raw/5.0 — same class, different formula:
               rag_evaluator.py:_evaluate_relevance (556)
               rag_evaluator.py:_evaluate_faithfulness (635)
               rag_evaluator.py:_evaluate_answer_similarity (815)
               rag_evaluator.py:_evaluate_context_precision (862)
               rag_evaluator.py:_evaluate_context_relevance (920)
               rag_evaluator.py:_evaluate_context_recall (985)
               response_quality_evaluator.py:_evaluate_relevance (226)
               response_quality_evaluator.py:_evaluate_completeness (291)
               response_quality_evaluator.py:_evaluate_clarity (353)
               response_quality_evaluator.py:_evaluate_accuracy (418)
               response_quality_evaluator.py:_evaluate_custom_criterion (550)
             THREE FURTHER INCOMPATIBLE SCHEMES:
               eval_runner.py:_eval_summarization (1424) raw/max if raw>=1.0 else raw, max=3 for fluency
               eval_runner.py:_eval_summarization (1435-1439) legacy-string branch divides unconditionally
               recipes/rag_answer_quality_execution.py:_coerce_unit_score (1102-1112) <=1.0 passthrough else clamp(raw/5)
               recipes/summarization_quality.py:_normalize_score (289) divide by a caller-supplied max
               evaluation_manager.py:613 raw/10.0 ; evaluation_manager.py:632 raw/10.0 if raw>1 else raw
canonical:   rag_evaluator.py:RAGEvaluator._normalize_score (1027-1039) — exists, is tested, has
             ZERO production callers
destination: tldw_Server_API/app/core/Evaluations/scoring.py — single responsibility: converting a
             judge's raw score on a declared scale into a 0-1 unit score, and parsing that raw score
             out of a provider reply. It owns exactly two functions (`parse_judge_score` wrapping
             the existing ms_g_eval.parse_output logic, and `to_unit_score(raw, scale)`). Not Utils.py.
knowledge:   "what a 1-5 Likert judge score means as a 0-1 number" — re-derived 16 times with five
             different answers.
scenario:    `_normalize_score(1)` returns 0.0; the eleven inline copies return 0.2 for the same
             input (executed above). A retrieval system whose judge rates every context "1 =
             completely irrelevant" reports `context_relevance = 0.2`, not 0.0 — a 20% floor under
             every metric in the family, which also shifts `_calculate_overall_score`'s
             observed-range clamp (:1073-1082) and every threshold comparison. At the other end the
             two formulas agree (5 -> 1.0), so the divergence is invisible on happy-path fixtures
             and only distorts the bottom of the range, where evaluations matter most.
             `_eval_summarization`'s two branches disagree with each other: a dict reply of
             `{"fluency": 0.8}` scores 0.8, the identical information as the string `"fluency: 0.8"`
             scores 0.8/3 = 0.267.
impact:      Five answers to one question, spread across the files a new evaluator author will copy
             from. Adding a metric means picking a formula at random. The correct one is the one
             nobody calls.
tests:       tests/Evaluations/unit/test_rag_evaluator.py:363-375 and
             tests/Evaluations/property/test_evaluation_invariants.py:102-136 both pin the DEAD
             function's contract, which is why the divergence survived
effort:      moderate — mechanical per site, but it CHANGES REPORTED SCORES, so it needs a design
             note and a decision on whether to keep /5.0 (raise `_normalize_score` to match the
             copies) or adopt (s-1)/4 (change published numbers). That decision is the expensive
             part, not the edit.
owner-only:  no (all sites are under core/)
confidence:  confirmed (the divergence and the dead canonical, executed); assumption (which formula
             the project intends — the ADRs do not say)
```

### FINDING evaluations-003 — rag_pipeline reports one arbitrary sample's chunk stats as a config-level mean, and feeds it into best-config selection

```
axis:        correctness
class:       n/a
severity:    High
sites:       tldw_Server_API/app/core/Evaluations/eval_runner.py:_execute_rag_pipeline_run (909-917)
             tldw_Server_API/app/core/Evaluations/eval_runner.py:_execute_rag_pipeline_run (919-928)
             tldw_Server_API/app/core/Evaluations/eval_runner.py:_execute_rag_pipeline_run (931-943)
             tldw_Server_API/app/core/Evaluations/eval_runner.py:_execute_rag_pipeline_run (956-965)
canonical:   the sibling helper `_mean_score` (897-903) in the same function, used for every other field
destination: n/a
knowledge:   "a config-level aggregate is a mean over that config's samples"
scenario:    Lines 909-914 pick the FIRST per-sample record that carries `chunk_index_stats` and
             break. `cohesion_mean` and `separation_mean` (:916-917) are read straight off that one
             sample and then published in `config_summary["aggregate"]` (:941-942) alongside
             `retrieval_cov_mean`, `mrr_mean` and `ndcg_mean`, which ARE means computed by
             `_mean_score`. The field names say `_mean`; the values are single observations. Those
             same two values are then multiplied into `config_score` (:925-926), which selects
             `best_config` (:956-961) and sorts the leaderboard (:965). Concretely: a sweep with
             `aggregation_weights={"rag_overall":0.5,"chunk_cohesion":0.5}` over two configs picks
             whichever config's FIRST sample happened to chunk coherently, not the config that
             chunks better on average — and rerunning the same sweep with the dataset in a
             different order can pick the other config.
impact:      The rag_pipeline sweep exists to choose a configuration. This makes that choice depend
             on dataset ordering. Reported as a mean, so no operator would question it.
tests:       tests/Evaluations/test_rag_pipeline_runner.py; tests/Evaluations/test_evaluations_unified.py
             (import-grep reachability; neither asserts chunk-stat provenance)
effort:      cheap — `chunk_cohesion`/`chunk_separation` go through the existing `_mean_score`
             helper eleven lines above, or are renamed to drop the `_mean` claim.
owner-only:  no
confidence:  confirmed
```

### FINDING evaluations-004 — `aggregation_weights` cannot exclude `rag_overall`, and `config_score` is an unnormalized sum

```
axis:        correctness
class:       n/a
severity:    Medium
sites:       tldw_Server_API/app/core/Evaluations/eval_runner.py:_execute_rag_pipeline_run (919-928)
             tldw_Server_API/app/core/Evaluations/eval_runner.py:_execute_rag_pipeline_run (963-965)
canonical:   NONE
destination: n/a
knowledge:   "how a caller expresses which signals rank a config sweep"
scenario:    Line 920 reads `weights.get("rag_overall", 1.0)` while every sibling term defaults to
             0.0. A caller who supplies `aggregation_weights={"mrr": 1.0}` — the natural way to say
             "rank by MRR" — gets `config_score = mean_overall + mrr_mean`, because the absent
             `rag_overall` key silently re-enters at weight 1.0. There is no value the caller can
             pass to remove it except an explicit `{"rag_overall": 0.0, "mrr": 1.0}`, which is not
             discoverable from the field name. Separately the expression is a weighted SUM with no
             division by the weight total, so with `{"rag_overall":1,"mrr":1,"ndcg":1}` the reported
             `config_score` reaches 3.0 and sits in the leaderboard response next to `overall`,
             which is 0-1. Ranking stays monotone within one run (same weights everywhere), so this
             is a usability/contract defect rather than a mis-ranking.
impact:      Medium: the user-facing knob for "rank my sweep by X" does not do what it says, and the
             number it produces is not on the scale the surrounding fields use.
tests:       tests/Evaluations/test_rag_pipeline_runner.py (import-grep reachability; no weights assertion)
effort:      cheap — default `rag_overall` to 0.0 when any weight key is supplied, and divide by
             `sum(weights.values())`. Changing the divisor changes published numbers, so pair it
             with a short design note.
owner-only:  no
confidence:  confirmed (the default re-entry); confirmed (the unnormalized sum); probable-risk (that
             callers actually pass partial weight maps — no fixture in-repo does)
```

### FINDING evaluations-005 — samples that failed to measure enter the mean as 0.0

```
axis:        correctness
class:       n/a
severity:    Medium
sites:       tldw_Server_API/app/core/Evaluations/eval_runner.py:_process_batch (1332-1339)
             tldw_Server_API/app/core/Evaluations/eval_runner.py:_calculate_aggregate_results (2240-2254)
             tldw_Server_API/app/core/Evaluations/eval_runner.py:_calculate_aggregate_results (2262-2270)
canonical:   NONE
destination: n/a
knowledge:   "a sample we could not measure is not a sample that scored zero"
scenario:    A timed-out or errored sample returns `{"sample_id": ..., "error": ...}` with no
             `avg_score` (`_process_batch:1333, :1337`). `_calculate_aggregate_results` reads
             `result.get("avg_score")`, finds None, increments `failed_samples`, and then **appends
             0.0 to `all_scores`** (:2244-2248). `mean_score`, `std_dev` and `min_score` are
             computed over that list (:2263-2266), and `pass_rate = passed_count / len(results)`
             (:2268) divides by the full sample count including unmeasurable ones. Concretely: a
             120-sample run where the provider rate-limits half the batch and the measured half
             averages 0.84 reports `mean_score = 0.42` and `min_score = 0.0` — indistinguishable
             from a model that genuinely scored 0.42. `failed_samples: 60` is present in the same
             dict, so the information to correct it exists and is simply not applied.
impact:      Medium rather than High because `failed_samples` is reported alongside, so a careful
             operator can detect it — but every downstream comparison, leaderboard and regression
             check reads `mean_score`, not the pair.
tests:       tests/Evaluations/test_evaluations_stage3_batch_failfast_and_metrics_none.py;
             tests/Evaluations/unit/test_eval_runner.py; tests/Evaluations/test_error_scenarios.py
             (import-grep reachability)
effort:      cheap — compute the aggregate over measured samples only and report
             `measured_samples` next to `total_samples`. Behaviour change, so it needs a line in
             the response docs.
owner-only:  no
confidence:  confirmed
```

### FINDING evaluations-006 — two evaluators implement the same judged metric with opposite failure semantics

```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       tldw_Server_API/app/core/Evaluations/rag_evaluator.py:_evaluate_relevance (511-585)
               — failure path :575-584 -> raise_detached_error(ValueError(...))
             tldw_Server_API/app/core/Evaluations/response_quality_evaluator.py:_evaluate_relevance (180-243)
               — failure path :236-242 -> return ("relevance", {"score": 0.0, ...})
             sibling fail-soft copies in the same file:
               response_quality_evaluator.py:_evaluate_completeness (301-307)
               response_quality_evaluator.py:_evaluate_clarity (363-369)
               response_quality_evaluator.py:_evaluate_accuracy (428-434)
               response_quality_evaluator.py:_evaluate_custom_criterion (560-566)
             sibling fail-loud copies in the same file:
               rag_evaluator.py:_evaluate_faithfulness (648-655)
               rag_evaluator.py:_evaluate_answer_similarity (823-829)
               rag_evaluator.py:_evaluate_context_recall (992-998)
canonical:   NONE — and the handling deliberately diverged once already: rag_evaluator.py:827 still
             carries the comment "Raise exception instead of returning 0.0 (fixing error handling
             issue)", i.e. exactly this fix was applied to one file and not the other
destination: the same tldw_Server_API/app/core/Evaluations/scoring.py proposed in evaluations-002,
             which should own the judge-call-and-parse wrapper including its failure contract
knowledge:   "what a judged metric reports when the judge could not be reached"
scenario:    The prompts, the 1-5 rubric, the `"You are an evaluation expert. Provide only numeric
             scores."` system message and the `float(score_str.strip())/5.0` parse are the same in
             both. Only the except block differs. When the judge provider is down,
             `RAGEvaluator.evaluate` records the metric in `failed_metrics` and sets
             `partial_results: True` (:395-397), so the caller knows the number is missing;
             `ResponseQualityEvaluator.evaluate` records `relevance = 0.0` as an ordinary measured
             value, which flows into `overall_quality`, into `eval_runner._eval_response_quality`'s
             `avg_score` (:1562) and into the stored run aggregate with no marker at all. A single
             provider outage therefore produces a partial-results run on one code path and a
             confidently-terrible score on the other.
impact:      Two evaluators the same API surface presents as peers disagree on whether "the judge
             failed" is a measurement. This is the same class of bug as evaluations-005, but here
             it is not even detectable downstream.
tests:       tests/Evaluations/unit/test_response_quality_provider_boundary.py;
             tests/Evaluations/unit/test_specialized_provider_auth_failures.py;
             tests/Evaluations/test_error_scenarios.py; tests/Evaluations/unit/test_rag_evaluator.py
             (its TestErrorHandling class pins the fail-loud contract for rag_evaluator only)
effort:      cheap to align the five `response_quality_evaluator` handlers with the already-chosen
             fail-loud policy; the decision is already documented in rag_evaluator.py:827's comment.
owner-only:  no
confidence:  confirmed (the divergence); confirmed (that the project already decided fail-loud is
             correct, per the in-code comment)
```

### Not raised, and why

- `_compute_mrr_ndcg (1002-1029)` — read line by line; the binary-gain DCG and the ideal-DCG
  denominator over `min(|rel|, |retrieved|)` are both standard. **Correct. Dropped.**
- `_calculate_metric_stats (2278-2322)` — handles the empty and non-numeric cases explicitly and
  falls back to inferring metric names from the observed score blobs. **Correct. Dropped.**
- `_extract_threshold_config` / `_evaluate_passed (2143-2214)` — the missing-per-metric-threshold
  case logs and fails closed (:2199-2206), which is the safe direction. **Correct. Dropped.**
- `rag_evaluator.py:363-369` — two expression statements whose values are discarded, left over from
  a refactor that removed the assignments. Genuinely dead, genuinely confusing to read, but with no
  behavioural consequence and no drop-rule scenario. Noted here, **not** filed as a finding.
- `_evaluate_context_precision` vs `_evaluate_context_relevance` failure asymmetry — `precision`
  relies on the outer handler while `relevance` guards `ValueError` inline, but `ValueError` IS in
  `_RAG_EVAL_NONCRITICAL_EXCEPTIONS` (rag_evaluator.py:35-52), so both append 0.0. Looks like a
  divergence, isn't. **justified-divergence, dropped** (their duplication is carried in stage 3).

## Suggested Refactor/Actions

Ordered by ratio of consequence to effort.

1. **evaluations-001 and evaluations-003 are one-line-class fixes with wrong-number consequences.**
   Do these first, independently, each with a regression test that asserts the arithmetic:
   `_calculate_overall_score` must ignore alias keys; `chunk_cohesion`/`chunk_separation` must go
   through `_mean_score`. Small enough not to need a design document.
2. **evaluations-005 and the measured/total split** — one change to `_calculate_aggregate_results`
   plus a `measured_samples` field. Small, but it changes a published response, so note it in
   `Docs/Evals/` rather than shipping silently.
3. **evaluations-002 and evaluations-006 belong to one design decision, not two patches.** Both
   resolve by giving the module a single owner for "judge score in, unit score out, and what
   happens when the judge fails". That owner should be a new
   `tldw_Server_API/app/core/Evaluations/scoring.py` with exactly that one responsibility — it must
   NOT be added to `core/Utils/Utils.py`. Because adopting `(s-1)/4` would change every published
   RAG and response-quality score, this needs the full treatment:
   `Docs/Design/2026-MM-DD-evaluations-scoring-normalization-design.md`, an ADR (it supersedes
   nothing but it IS a decision that ADR-015's "dedicated evaluator modules own scoring behavior"
   does not settle), a Backlog task linking both, and an `IMPLEMENTATION_PLAN_evaluations-scoring.md`
   staged as: (1) introduce `scoring.py` with both formulas and a feature-flagged selector;
   (2) migrate `rag_evaluator`; (3) migrate `response_quality_evaluator`; (4) migrate the runner and
   recipe variants; (5) delete the dead `_normalize_score` and repoint its tests.
4. **evaluations-004** — fold into the same design note as (3) if the team wants to renormalize
   `config_score`; otherwise ship the `rag_overall` default fix alone, which is independent.
