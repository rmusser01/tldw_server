# Stage 3 (2026-09-21) Duplication, Utility Consolidation, and Test Topology

## Scope

Axis 1 (duplication / utility consolidation), plus the module's test topology as a review target in its own right. The 2026-04-07 ledger produced ownership and boundary findings; it did not classify anything as `adoption-gap`, `divergent-copies`, or `true-duplication`, and it noted the `tests/RAG` + `tests/RAG_NEW` split without assessing it. Both are covered here.

This stage also answers two questions the repo-wide briefing put to this module directly:
- **C4** — is `rag_service/resilience.py` the best of the eight module-level exponential-backoff implementations, and should it be promoted as canonical? (Answer: no, and the reason matters — see rag-13.)
- **Secondary** — does the bounded-gather duplicated twice in `batch_utils.py` clear the drop rule? (Answer: yes, and for a better reason than the one seeded — see rag-11.)

Read-only. No source file was modified.

## Code Paths Reviewed

- `rag_service/web_fallback.py:_truncate_content_by_tokens (50-91)` and its caller at `(240-241)`
- `core/Workflows/adapters/rag/search.py:_truncate_content_by_tokens (162-206)` and `_RAG_ADAPTER_NONCRITICAL_EXCEPTIONS (32-45)` — out of module, cited as the duplicate
- `rag_service/utils.py:TokenCounter (16-48)`, `normalize_scores (162-195)`
- `rag_service/quick_wins.py:CostTracker.__init__ (323-334)`
- `rag_service/advanced_reranking.py:BaseReranker._normalize_scores (416-427)`
- `rag_service/batch_utils.py:run_batch (84-188)`, `run_batch_indexed (190-277)`
- `rag_service/resilience.py:RetryConfig (87-96)`, `RetryPolicy (230-290)`, `get_coordinator (543-549)`
- `core/http_client.py:_decorrelated_jitter_sleep (2311-2315)`, `_parse_retry_after_delay_seconds (2318-2337)`, `_should_retry (2340-2357)` — out of module, cited as the better implementation
- `core/DB_Management/transaction_utils.py` retry loop (`:54-88`) — out of module, cited as the worst
- `core/LLM_Calls/tokenizer_resolver.py:_resolve_tiktoken_encoding_cached (926-936)`, `resolve_tiktoken_encoding (939-943)` — the canonical tokenizer resolver
- `tests/RAG/conftest.py (39-225)`, `tests/RAG_NEW/conftest.py (28-310)`

## Tests Reviewed

- `tests/RAG/test_batch_utils.py` — exercises both `run_batch` and `run_batch_indexed`. It is the **only** place in the repo that calls `run_batch_indexed`; see rag-11.
- `tests/RAG/test_web_fallback.py` — covers `web_fallback`; `tests/RAG/test_knowledge_trust_contracts.py` and `tests/RAG_NEW/unit/test_rag_profiles.py` reach it indirectly. `tests/Workflows/adapters/test_content_adapters.py` covers the Workflows twin. Neither side tests the other's copy, which is exactly how the two drifted.
- `tests/RAG_NEW/unit/test_advanced_reranking_sanitizers.py`, `test_two_tier_reranker.py` — reranker behavior; nothing pins the score-normalization contract.
- `tests/RAG/test_resilience_sanitizers.py:70-71` — the only direct `resilience.RetryPolicy` exercise, and it passes an explicit `retry_on`, so it never runs the defaults the pipeline uses.
- `tests/RAG/conftest.py:dual_backend_env (141-199)` — the only sqlite/postgres-parametrized RAG fixture. Consumers: `tests/RAG/test_dual_backend_end_to_end.py`, `tests/RAG/test_dual_backend_characters_retriever.py`. Two files.
- `tests/RAG_NEW/conftest.py:_isolate_semantic_cache (42-61)`, `_disable_limit_enforcement (64-69)`, `_isolate_provider_override_cache (71-83)`, `_reset_main_app_lifecycle_between_rag_tests (85-108)` — four autouse fixtures whose scope, by pytest's conftest rules, stops at the `RAG_NEW` directory.
- `grep -rln "normalize_scores\|TokenCounter" tldw_Server_API/tests` returns only `tests/Slides/test_standalone_html_sources.py` — neither shared helper in `rag_service/utils.py` has a direct test.

Import-grep reachability, not measured coverage. The suite was not executed.

## Validation Commands

```bash
diff <(sed -n '50,91p' tldw_Server_API/app/core/RAG/rag_service/web_fallback.py) \
     <(sed -n '162,206p' tldw_Server_API/app/core/Workflows/adapters/rag/search.py)
```
Observed: three hunks only — the docstring, and the two `except` clauses (`except Exception` in RAG vs `except _RAG_ADAPTER_NONCRITICAL_EXCEPTIONS` in Workflows). The 30-line algorithm body is byte-identical.

```bash
diff <(sed -n '116,185p' .../batch_utils.py) <(sed -n '208,277p' .../batch_utils.py)
```
Observed: the only functional differences are `await func(item)` vs `await func(index, item)` and **the absence of the fail_fast warning log in the second copy**.

```bash
grep -n "logger.warning" tldw_Server_API/app/core/RAG/rag_service/batch_utils.py
```
Observed: `159`, `164`, `256`. `run_batch` logs in both the fail_fast and the non-fail_fast branch; `run_batch_indexed` logs only in the non-fail_fast branch.

```bash
grep -rn "run_batch_indexed" tldw_Server_API/app tldw_Server_API/tests
```
Observed: three hits, all in `batch_utils.py` itself (`:9`, `:190`, `:197`) plus `tests/RAG/test_batch_utils.py:7,22,...`. Zero production callers.

```bash
grep -rn "encoding_for_model\|get_encoding(\"cl100k_base\")" tldw_Server_API/app/core/RAG/
```
Observed: 7 lines across 3 files — `quick_wins.py:331,334`; `utils.py:27,30`; `web_fallback.py:67,69,71`.

```bash
grep -rn "RetryPolicy(" tldw_Server_API/app/core/RAG/
```
Observed: `quick_wins.py:639` (`http_client.RetryPolicy(attempts=1, retry_on_unsafe=False)`),
`unified_pipeline.py:2996` (`resilience.RetryPolicy(RetryConfig(...))`),
`resilience.py:489`. Two different classes named `RetryPolicy`, with incompatible constructors, both live inside `core/RAG/`.

```bash
find tldw_Server_API/tests/RAG -name 'test_*.py' | wc -l ; find tldw_Server_API/tests/RAG_NEW -name 'test_*.py' | wc -l
git log --diff-filter=A --format='%ad' --date=short -- tldw_Server_API/tests/RAG_NEW | tail -1
git log --diff-filter=A --format='%ad' --date=short -- tldw_Server_API/tests/RAG | tail -1
git log -1 --format='%ad' --date=short -- tldw_Server_API/tests/RAG
git log -1 --format='%ad' --date=short -- tldw_Server_API/tests/RAG_NEW
```
Observed: `47` and `143`; created `2025-09-04` and `2025-04-19`; last touched `2026-09-18` and `2026-09-19`.

```bash
grep -rl "dual_backend_env" tldw_Server_API/tests
```
Observed: `tests/RAG/test_dual_backend_end_to_end.py`, `tests/RAG/conftest.py`, `tests/RAG/test_dual_backend_characters_retriever.py`.

## Findings

---

### FINDING rag-9 — a token-budget truncator is duplicated verbatim across the module boundary, and both copies re-tokenize O(log n) times where the module's own helper does it in one pass

```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       rag_service/web_fallback.py:_truncate_content_by_tokens (50-91), called at (240)
             core/Workflows/adapters/rag/search.py:_truncate_content_by_tokens (162-206) [out of module]
canonical:   rag_service/utils.py:TokenCounter.truncate (41-48) — already in this package, already
             correct, and already one-pass: `tokens = encoding.encode(text)`, slice, `encoding.decode(...)`.
destination: n/a — `TokenCounter` is the cohesive home and it already exists. The Workflows adapter
             should import it rather than either copy being promoted.
knowledge:   "how many tokens fit, and what a truncated string looks like." The two copies already
             disagree on failure handling, and the third implementation in the same package disagrees on
             algorithm. A future change to the truncation marker, the 4-chars-per-token fallback ratio,
             or the tokenizer choice has to land in three places, and nothing links them.
impact:      Which copy is correct: `TokenCounter.truncate (41-48)` is the least-wrong on algorithm —
             it encodes once and decodes a token slice, O(n). Both `_truncate_content_by_tokens` copies
             binary-search over *character* prefixes, calling `encoding.encode(text[:mid])` inside the
             loop, so they tokenize from the start of the string ~log2(len(text)) times: O(n log n)
             tokenization for a result the canonical helper gets in O(n). On a 100 KB scraped web page
             with a 500-token budget that is ~17 full tokenizations instead of one, per fallback
             document, and `web_search_fallback` runs this per result.
             But `TokenCounter.truncate` is not a drop-in: it has no `"...[truncated]"` marker and no
             tiktoken-unavailable fallback. So the consolidation is "give TokenCounter the marker and the
             fallback, then delete both copies" — not "swap the call."
             The two `_truncate_content_by_tokens` copies also diverge on error handling, which is the
             live-drift evidence: RAG catches bare `Exception` (:65, :88), Workflows catches the explicit
             `_RAG_ADAPTER_NONCRITICAL_EXCEPTIONS` tuple (search.py:32-45, used at :183, :204). That tuple
             does include `ImportError`, so the two behave the same for a missing tiktoken today — but
             the RAG copy will swallow anything at all, including a `MemoryError` on a large encode, and
             silently return a 4-chars-per-token approximation instead.
tests:       tests/RAG/test_web_fallback.py, tests/RAG/test_knowledge_trust_contracts.py,
             tests/RAG_NEW/unit/test_rag_profiles.py (the RAG copy);
             tests/Workflows/adapters/test_content_adapters.py (the Workflows copy).
             `TokenCounter` has no direct test — `grep -rln "TokenCounter" tldw_Server_API/tests`
             returns only tests/Slides/test_standalone_html_sources.py.
effort:      moderate. The consolidation target needs two features added before either caller can move,
             and it is currently untested, so a test for `TokenCounter.truncate` comes first.
             `core/Workflows/adapters/` is not owner-only, so the cross-module edit is allowed.
owner-only:  no
confidence:  confirmed
```

---

### FINDING rag-10 — the reranker base class re-implements the package's own `normalize_scores`, including its degenerate-case policy

```
axis:        duplication
class:       adoption-gap
severity:    Medium
sites:       rag_service/advanced_reranking.py:BaseReranker._normalize_scores (416-427)
             consumers: DiversityReranker.rerank (1204), and every subclass that inherits it
canonical:   rag_service/utils.py:normalize_scores (162-195), `method="minmax"` branch (177-183).
             `database_retrievers.py:59` already imports it as `_normalize_scores` and uses it at
             `:1074`, `:1367`, `:1653`, `:1655`.
destination: n/a — the canonical helper is one directory level away in the same package and is already
             adopted by the sibling module.
knowledge:   "what a normalized relevance score is, and what happens when all inputs are equal." Both
             implementations currently answer the degenerate case with `[0.5] * n` — the same arbitrary
             constant, written twice. Change that policy (to 1.0, or to preserving the input, or to
             raising) on one side and retrieval and reranking start disagreeing about the score of a
             tied result set, silently, in the middle of a pipeline that sorts on exactly that number.
             This is the same knowledge that rag-2 shows is already fragmented eleven ways in
             `database_retrievers.py`; a second uncoordinated copy in the reranker is how it gets to
             twelve.
impact:      Medium. Today the two are behaviorally identical, so nothing is broken — the cost is
             entirely change-amplification, and it is realized the moment anyone revisits scoring, which
             rag-2 says they must. Fixing it now makes the rag-2 work cheaper; leaving it means the
             rag-2 fix has to find this copy.
tests:       tests/RAG_NEW/unit/test_advanced_reranking_sanitizers.py,
             tests/RAG_NEW/unit/test_two_tier_reranker.py, tests/RAG_NEW/unit/test_reranker_metrics.py
             (reranker behavior, which transitively depends on the normalization but does not pin it).
             `utils.normalize_scores` has no direct test.
effort:      cheap mechanically — delete the method, import the canonical one. Gate it on adding a test
             for `normalize_scores` first, since the shared helper is currently unprotected.
owner-only:  no
confidence:  confirmed
```

---

### FINDING rag-11 — `run_batch_indexed` is an 87-line verbatim copy of `run_batch` with zero production callers, and it has already lost a log line

```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       rag_service/batch_utils.py:run_batch (84-188)
             rag_service/batch_utils.py:run_batch_indexed (190-277)
             only caller of the second: tests/RAG/test_batch_utils.py (`:7`, `:22`, and its
             `run_batch_indexed` cases)
canonical:   `run_batch (84-188)` is the correct copy — it is the one the production code actually
             calls (unified_pipeline.py:9346-9360) and the one that logs a fail_fast abort.
destination: n/a. Two options, both cheaper than keeping the copy: delete `run_batch_indexed`, or give
             `run_batch` an `pass_index: bool = False` parameter and make the indexed form two lines.
             The briefing's framing — "differing only `func(item)` vs `func(index, item)`" — understates
             it: `process_item` already receives `index` in both copies, so the difference is literally
             whether one local name is forwarded into the callback.
knowledge:   bounded-concurrency batch semantics: semaphore acquisition, the pre- and post-acquire
             cancel checks, the lock-guarded progress counter, the coroutine-vs-value progress callback,
             ordered-result reconstruction, and the fail_fast abort protocol. Every one of those is
             written twice. The `BatchResult` contract (`:28`) is shared, so a change to it has to be
             threaded through both builders.
impact:      The copy has already drifted, which is the whole argument. `run_batch` logs
             `"Batch item {index} failed (fail_fast=True), cancelling remaining"` at `:159-163` when a
             fail_fast abort fires; `run_batch_indexed` sets `cancel_event` at `:254` and logs **nothing**
             (`:253-256`). A fail_fast batch in the indexed variant therefore stops early with no record
             of why. Nobody noticed because nobody calls it. That is the honest severity story: it is a
             Medium not because 87 duplicated lines are expensive today, but because this is a
             demonstrated instance of a copy silently diverging from its original inside a single file,
             in a repo where the same pattern at DB-backend scale is a known source of live defects.
tests:       tests/RAG/test_batch_utils.py (both functions; the fail_fast log divergence is not
             asserted, which is why it drifted).
effort:      cheap. Deleting the function is a one-line change to `__all__`-equivalent surface plus the
             test cases; parameterizing is ~5 lines. The test file is the only consumer either way.
owner-only:  no
confidence:  confirmed
```

---

### FINDING rag-12 — the tiktoken encoder fallback is written three times inside `core/RAG/` with three different failure policies, bypassing the canonical resolver

```
axis:        duplication
class:       adoption-gap
severity:    Medium
sites:       rag_service/utils.py:TokenCounter.__init__ (26-31)     — catches `KeyError` only
             rag_service/quick_wins.py:CostTracker.__init__ (330-334) — catches bare `Exception`
             rag_service/web_fallback.py:_truncate_content_by_tokens (65-71) — catches bare `Exception`,
               and re-derives the `model or default` branch a third time
canonical:   core/LLM_Calls/tokenizer_resolver.py:_resolve_tiktoken_encoding_cached (926-936) and
             resolve_tiktoken_encoding (939-943) — `@lru_cache(maxsize=128)`, raising a typed
             `TokenizerUnavailable` instead of leaking whichever exception tiktoken happened to throw.
destination: n/a — adopt the resolver. Note the briefing's do-not-seed list correctly separates this from
             the `count_tokens` tokenizer-protocol implementations in `core/Chunking/strategies/tokens.py`,
             which are legitimate and not a target.
knowledge:   "how to get an encoder for a model name, and what to do when you cannot." Three answers in
             one package. Also note what the canonical resolver has that none of the three copies do:
             memoization. `tiktoken.encoding_for_model` and `get_encoding` are not free — the copies
             construct an encoder per `TokenCounter`, per `CostTracker`, and per
             `_truncate_content_by_tokens` *call*.
impact:      The divergence is already observable. `TokenCounter.__init__` catches only `KeyError` (:28).
             `encoding_for_model` raises `KeyError` for an unrecognized model name, which is the case the
             author had in mind — but it also performs a BPE-file fetch on first use, and a network or
             filesystem failure there raises something else. Those escape `TokenCounter.__init__`
             entirely, so constructing a `TokenCounter` can raise where the two sibling copies would have
             silently fallen back to `cl100k_base`. `utils.py:12` imports tiktoken unguarded at module
             scope as well, so the fallback story for "tiktoken not installed at all" differs between
             `utils.py` (import error at module load) and the other two (caught, degraded).
             Severity is Medium rather than High because `tiktoken>=0.5.0` is a hard dependency
             (`pyproject.toml:125`), so the not-installed branch is unreachable in a correct install —
             the live divergence is the fetch-failure one.
tests:       tests/RAG/test_web_fallback.py (web_fallback copy). `TokenCounter` and `CostTracker` have no
             direct tests — `grep -rln "TokenCounter" tldw_Server_API/tests` returns only
             tests/Slides/test_standalone_html_sources.py.
effort:      moderate. Two of the three sites are untested, so adoption needs tests written first. The
             resolver's typed `TokenizerUnavailable` also changes the failure signature at each call
             site, so each needs its own catch decision rather than a mechanical swap.
owner-only:  no
confidence:  confirmed
```

---

### FINDING rag-13 — C4 verdict: `resilience.RetryPolicy` should NOT be promoted as canonical; it is a weaker algorithm sharing a name with the already-adopted `http_client.RetryPolicy`, and both are live inside `core/RAG/`

```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       rag_service/resilience.py:RetryConfig (87-96), RetryPolicy._calculate_delay (281-289)
             rag_service/unified_pipeline.py (2996) — constructs `resilience.RetryPolicy(RetryConfig(...))`
             rag_service/quick_wins.py (639) — constructs `http_client.RetryPolicy(attempts=1, retry_on_unsafe=False)`
             compared against, out of module:
               core/http_client.py:_decorrelated_jitter_sleep (2311-2315)
               core/http_client.py:_parse_retry_after_delay_seconds (2318-2337)
               core/http_client.py:_should_retry (2340-2357)
               core/DB_Management/transaction_utils.py retry loop (54-88)
canonical:   for the *delay schedule*: core/http_client.py:_decorrelated_jitter_sleep (2311-2315),
             together with `_parse_retry_after_delay_seconds` and `_should_retry`.
destination: a new `core/Utils/backoff.py` owning exactly one responsibility — computing the next retry
             delay and classifying whether an attempt is retriable. Seeded by *moving* the three
             http_client functions above into it, with http_client importing them back.
             Explicitly NOT `core/Utils/Utils.py`, and explicitly not by growing `core/http_client.py`
             (6,600 LOC, already a cohesion problem per the briefing). The point of the move is that the
             best implementation currently lives inside the repo's largest junk drawer, which is why
             seven other modules wrote their own instead of importing it.
knowledge:   retry scheduling: base delay, growth, cap, jitter shape, `Retry-After` honouring, and which
             exceptions are terminal.
impact:      **Direct answer to the C4 seed question: no.** Ranking the three implementations the
             briefing named:
             - `http_client` is the best. `_decorrelated_jitter_sleep (2311-2315)` implements
               decorrelated jitter — `min(cap, random.uniform(base, prev * 3))` — which is the algorithm
               that actually de-synchronizes a retrying fleet. It is paired with `Retry-After` parsing
               that handles both delta-seconds and HTTP-date (`:2318-2337`) and with a real retriability
               classifier that treats DNS failures as permanent (`:2340-2357`). It also has ~15
               importers, so it is already the repo's de facto standard.
             - `resilience.RetryPolicy._calculate_delay (281-289)` is middling. It caps correctly but its
               jitter is symmetric +/-25% around the nominal delay
               (`delay * 0.25 * (2 * random() - 1)`), which keeps every client clustered in a narrow band
               around the same instant — it damps a thundering herd far less than decorrelated jitter.
               It has no `Retry-After` awareness and no retriability classifier (`retry_on=[Exception]`,
               `dont_retry_on=[]` by default, which is the rag-7 defect).
             - `transaction_utils.py:54-88` is the worst: `0.1 * (2 ** retry_count)` with no jitter at
               all, which is the textbook synchronized-retry failure mode.
             So promoting `resilience.RetryPolicy` would standardize the repo on the weaker of the two
             algorithms that are actually adopted. It would also entrench a name collision that already
             exists **inside `core/RAG/` itself**: `quick_wins.py:639` constructs
             `http_client.RetryPolicy(attempts=1, retry_on_unsafe=False)` while `unified_pipeline.py:2996`
             constructs `resilience.RetryPolicy(RetryConfig(max_attempts=...))`. Two classes, one name,
             incompatible constructors, one package. A reader who greps `RetryPolicy(` in this module
             gets two unrelated things.
             What `resilience.py` *should* keep is its circuit-breaker and coordinator surface — those
             have real consumers (`api/v1/endpoints/rag_health.py:24`,
             `core/Research_Workspace/capabilities.py:468`) and no competitor.
tests:       tests/RAG/test_resilience_sanitizers.py:70-71 (RetryPolicy with an explicit `retry_on`);
             tests/Infrastructure/test_cb_parity_batch_c.py:310-384 (circuit-breaker parity against the
             RAG coordinator — protects the part of resilience.py that should stay).
effort:      expensive, and it is a repo-wide decision, not a RAG one. This needs the design-first
             treatment: `Docs/Design/YYYY-MM-DD-retry-backoff-consolidation-design.md`, an ADR (the
             choice of jitter algorithm is a decision with operational consequences), a Backlog task
             linking both, and a staged `IMPLEMENTATION_PLAN_retry_backoff.md`. The RAG-local slice —
             stop shadowing the `RetryPolicy` name, and populate `dont_retry_on` per rag-7 — is cheap and
             can go first.
owner-only:  no
confidence:  confirmed (the three implementations and their properties);
             probable-risk (that the weaker jitter has caused an observed incident — no evidence either way)
```

---

### FINDING rag-14 — two permanently-maintained test trees for one module, with disjoint and non-overlapping conftest guarantees

```
axis:        duplication
class:       true-duplication
severity:    Medium
sites:       tldw_Server_API/tests/RAG/ — 47 test files, flat, created 2025-04-19, 98 commits in 12
               months, last touched 2026-09-18
             tldw_Server_API/tests/RAG_NEW/ — 143 test files under unit/ integration/ property/,
               created 2025-09-04, 255 commits in 12 months, last touched 2026-09-19
             tests/RAG/conftest.py:dual_backend_env (141-199) — sqlite+postgres parametrized fixture,
               consumed only by tests/RAG/test_dual_backend_end_to_end.py and
               tests/RAG/test_dual_backend_characters_retriever.py
             tests/RAG_NEW/conftest.py:_isolate_semantic_cache (42-61), _disable_limit_enforcement (64-69),
               _isolate_provider_override_cache (71-83), _reset_main_app_lifecycle_between_rag_tests (85-108)
               — four autouse fixtures
             tests/RAG_NEW/TEST_STRATEGY.md — describes a three-tier pyramid for "the RAG module" with no
               mention of tests/RAG
canonical:   NONE. Neither tree is designated; `_NEW` is not a migration marker here because both trees
             are under active development.
destination: one `tests/RAG/` with the `unit/ integration/ property/` layout `RAG_NEW` already uses,
             one conftest that is the union of the two guarantee sets.
knowledge:   what a RAG test is allowed to assume about its environment. Right now that answer depends
             on which directory the file happens to sit in, and nothing states the rule.
impact:      This is not a tidiness complaint — the two conftests protect *different* things, and pytest
             scopes each one to its own directory:
             - The only sqlite/postgres-parametrized RAG fixture lives in `tests/RAG/conftest.py:141`.
               The 143-file `RAG_NEW` tree cannot use it. `RAG_NEW` has some pgvector-specific files
               (`unit/test_vector_store_parity.py`, `unit/test_pgvector_adapter_sanitizers.py`,
               `integration/test_retriever_pgvector_multi_search.py`), but no dual-backend
               parametrization — so the larger and faster-growing tree is effectively SQLite-only.
               The briefing records this exact shape as a known-real risk with precedent: the
               2026-09-21 cross-user isolation audit found leaks fail to converge partly because the
               SQLite/PostgreSQL split is only tested on SQLite.
             - Conversely, the four autouse isolation fixtures — semantic-cache directory and
               `_SHARED_CACHES` reset, limit-enforcement disable, provider-override cache reset,
               app-lifecycle reset — apply only to `RAG_NEW`. The 47 files in `tests/RAG` run without
               them. `semantic_cache._SHARED_CACHES` is a module-level global and
               `_isolate_semantic_cache` clears it on setup but not on teardown (:58-60), so cross-tree
               state bleed in a single pytest process is a live ordering hazard rather than a
               theoretical one.
             - `TEST_STRATEGY.md` documents a strategy for "the RAG module" that silently means only
               one of its two test trees.
             Change-amplification: a new retriever needs coverage decisions made twice, and a reviewer
             reading either tree gets a false picture of what is protected.
tests:       n/a — the tests are the subject.
effort:      moderate. Mechanically it is a directory move plus a conftest union, and both trees are
             green today. The real cost is that merging the conftests means the 47 `tests/RAG` files
             suddenly acquire four autouse fixtures they have never run under, and the dual-backend
             parametrization becomes visible to files that have never seen Postgres — so it wants a
             staged plan (merge conftests first under the current layout, fix fallout, then move files).
             Per the briefing, the AuthNZ tree is split five ways and Chat/Character_Chat/TTS/
             MediaIngestion all have `_NEW` twins, so this is worth solving as a repo pattern with RAG
             as the pilot rather than as a RAG-only cleanup.
owner-only:  no
confidence:  confirmed (the split, the disjoint fixture scopes, the active maintenance of both);
             probable-risk (that cross-tree `_SHARED_CACHES` bleed has actually caused a flake)
```

## Suggested Refactor/Actions

1. **rag-11 now, as a standalone.** Delete `run_batch_indexed` or fold it into `run_batch` with a flag. Zero production callers, one test file, and it is the cleanest possible demonstration of the pattern for anyone who needs convincing.
2. **rag-10 after a test for `normalize_scores`.** Cheap, and it removes one of the twelve scoring answers rag-2 has to reconcile. Sequence it *before* the rag-2 design work, not after.
3. **rag-9 and rag-12 together.** Both are about `rag_service/utils.py` being the right home that nobody uses. One task: add tests for `TokenCounter`, give `TokenCounter.truncate` the marker and the unavailable-tokenizer fallback, route it through `tokenizer_resolver.resolve_tiktoken_encoding`, then delete the two `_truncate_content_by_tokens` copies and the `quick_wins` encoder block.
4. **rag-13 splits in two.** The RAG-local half (stop shadowing the `RetryPolicy` name; populate `dont_retry_on` per rag-7) is cheap and can ship immediately. The repo-wide half — extracting `core/Utils/backoff.py` from `http_client` and retiring the other seven implementations — needs a design doc and an ADR, and belongs to whoever owns the C4 cluster repo-wide, not to this module.
5. **rag-14 as a pilot.** Propose a Backlog task to merge the two RAG conftests first, under the existing layout, and only then move files. If it works, it is the template for the Chat, Character_Chat, TTS, and MediaIngestion `_NEW` twins. Do not hand-edit Backlog task files; propose through the Backlog MCP/CLI.
6. Nothing in this stage touches `tldw_Server_API/app/api/v1/**`, so none of it is owner-only. The `core/Workflows/adapters/rag/search.py` edit in rag-9 is outside RAG but inside `core/`, which is unrestricted.
