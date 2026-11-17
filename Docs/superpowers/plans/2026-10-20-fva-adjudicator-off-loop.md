# FVA Adjudicator: the LLM Call Runs Off the Event Loop, and a Quota Refusal Is Let Through (TASK-13862)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Review folded in (2026-10-20):** independent review and dry run, `review_plan_fu_fva_adjudicator.md`. Verdict: ready with fixes, with 0 Blocker, 2 Important and 5 Minor, all folded in.
- **I1:** the off-loop test records the user each quota decision is made for, and asserts it is the request's user. Measured: the mutant that moves the call off the loop without the LLM call state now fails (`assert [] == [7]`).
- **I2:** the endpoint has three broken constructor calls, not two. "What was measured", Decision 1 and the follow-up's description are corrected, and the follow-up now says what to do with the `user_id` parameter.
- **M1:** Review Focus 1, the deadline does not bound latency.
- **M2:** Review Focus 4, no test pins the fallback call's move off the loop.
- **M3:** Task 1 Step 4 names the Hypothesis test that also fails on dev.
- **M4:** `bandit -q` prints nothing when clean.
- **M5:** the conflicts table lists the sibling plans.

The changed steps were re-run in a scratch worktree on `origin/dev` `df7c290fbf`, which changes none of this plan's files.

**Goal:** `ClaimAdjudicator` makes its LLM call on the bounded adapter pool, where the LLM quota check at the call runs, and a refusal there fails the claim instead of becoming a neutral stance.

**Architecture:**
- `_invoke_llm_assess_call` calls the sync analyze function straight from a coroutine, on the server loop's thread. There the check at the call cannot wait, so it is skipped (design §5), and the provider call blocks the loop.
- The claims engine already runs its own analyze calls through `_run_bounded_claims_analyze` (`claims_engine.py:108-119`: `SYNC_ADAPTER_CALL_POOL`, a 60 s deadline, drained on timeout). The adjudicator reuses that helper for both of its call shapes. The pool carries the LLM call state into the thread (design §1), so the check runs there.
- Three broad `except Exception` clauses would swallow the refusal once it can be raised: `_llm_assess` (it becomes NEUTRAL 0.5), `_assess_documents` (the document is dropped), and the falsification `try` in `FVAPipeline.process_claim` (the claim keeps its unchallenged verdict). Each gets `except LLMQuotaExceededError: raise` before it, the TASK-13638 pattern. `process_batch` already re-raises a refusal it collects (TASK-13638), so the batch fails with it.

**Tech Stack:** Python 3.12, asyncio, pytest (`asyncio_mode = "auto"`), loguru.

**Spec:**
- `backlog/tasks/task-13862 - FVA-adjudicator-calls-the-sync-analyze-on-the-event-loop.md`
  - AC1: the adjudicator's LLM call runs off the event loop (`asyncio.to_thread` or the bounded adapter pool), with a test.
  - AC2: an LLM quota refusal there is let through, not counted as a failed assessment.
- Governing design: `Docs/Design/2026-10-09-llm-usage-accounting-design.md` §1 (the bounded pools carry the LLM call state), §4b (what the caller sees) and §5 (a sync call on the loop's thread is not checked). Operator page: `Docs/Operations/Usage_Quotas.md`. Parent plan: `Docs/superpowers/plans/2026-10-18-llm-quota-refusal-callers-rag-metric.md` (Review Focus 2 filed this task).

## What was measured

Line numbers and behaviour were measured on `origin/dev` `a6719bf24f`, and the review re-checked them on `2b925562e5`. The review fixes were re-run on `df7c290fbf`. None of the commits between them touches this plan's files, so every number below holds on all three. The probes and their output are in the scratchpad: `fu_probe_fva_endpoint_plain.py`, `fu_probe_fva_endpoint_repaired.py`, `fu_p1_*` and `fu_p1r_*`, and the reviewer's `rv_fva_*`.

- **Callers of `ClaimAdjudicator`** (`git grep` over `tldw_Server_API` and `apps`): one in the app, `FVAPipeline.__init__` (`fva_pipeline.py:168-172`). `create_adjudicator` has no caller outside its tests. The only app caller of `FVAPipeline` is `claims_service.verify_claims_with_fva` (`claims_service.py:4684`), behind `POST /claims/verify/fva` (`endpoints/claims.py:996-1040`). Tests: `tests/Claims_Extraction/test_adjudicator.py`, `test_fva_pipeline.py`, `tests/Claims/test_claims_compat_types.py`.
- **The adjudicator always takes the LLM path in the app.** `FVAPipeline` reads the verifier's `_nli` when it is built (`fva_pipeline.py:164-166`), but `HybridClaimVerifier` loads it lazily on first use (`claims_engine.py:970-1008`), so the adjudicator gets `nli_pipeline=None` and `llm_analyze_fn=_fva_claims_analyze_call`.
- **How other async callers move a sync analyze call off the loop:**
  - the claims engine's extract, verify and APS calls: `await _run_bounded_claims_analyze(self._analyze, ...)` (`claims_engine.py:841`, `:1207`, `:1491`);
  - the PR 2b paths and the HyDE legacy branch (TASK-13638): `await asyncio.to_thread(...)` (`unified_pipeline.py:4133`).
  - This plan uses the first: it is the same module family, it bounds concurrency, and it gives the call the deadline the verifier's call already has.
- **What `POST /claims/verify/fva` returns today, refusal or not: HTTP 500.** `verify_claims_with_fva` has three constructor calls that cannot succeed, all before any LLM call:
  1. `ClaimsJobContext(job_id=str(uuid4()), user_id=...)` (`claims_service.py:4759-4762`), made only when the request sets `fva_config.max_budget_usd`. `ClaimsJobContext` (`budget_guard.py:45-50`) has no `job_id` field.
  2. `MultiDatabaseRetriever(media_db=db)` (`claims_service.py:4770`). The class requires the positional `db_paths` (`database_retrievers.py:4744-4746`). Every request without a budget fails here: `TypeError: MultiDatabaseRetriever.__init__() missing 1 required positional argument: 'db_paths'` (probe `fu_probe_fva_endpoint_plain.py`: `status=500`; the reviewer's `rv_fva_probe_500.py`, through the full `main.app`, with and without `?user_id=2`: `500`).
  3. `RetrievalConfig(top_k=..., search_mode="hybrid")` (`claims_service.py:4782-4785`), which names two fields the dataclass does not have.

  All three have been there since the endpoint landed (`05245a3d59`, 2026-02-05). No test calls it: `tests/Claims_Extraction/test_fva_endpoint.py` checks schemas and request validation only. The operator page's claim that `/claims/verify/fva` answers 402 (`Usage_Quotas.md:95`, `:103`) therefore does not hold. See Decision 1.
- **What the endpoint would do once it runs.**
  - The author's probe `fu_probe_fva_endpoint_repaired.py`:
    - calls 2 and 3 are fixed in a scratch worktree only;
    - the server loop is the accounting loop;
    - `perform_chat_api_call` is replaced by a fake that first runs `check_llm_call_sync()`, as the real one does at `chat_service.py:2636`.

    That fake set the user itself (`llm_call_context(user_id=7)`). The reviewer's `rv_fva_probe_scope_user.py` does not, and leaves the user to the content scope, as the real dependency does. It gives the same answers, with every decision made for user 7, so the request's user reaches the worker thread.

  | Case | dev + repair | dev + repair + this plan |
  |---|---|---|
  | A: every call refused | 402 (the verify call runs on the pool and is checked) | 402 |
  | B: verify admitted, later calls refused | **200**, `final_status: refuted`. Both adjudication calls ran on the loop's thread and were not checked: 1 quota decision for 3 LLM calls. | **402**: 2 decisions |
  | C: nothing refused | 200, 1 decision for 3 calls | 200, 3 decisions; every call off the loop's thread |

  - With `max_budget_usd` set, the repaired endpoint still answered `500` on call 1. With call 1 also fixed, keeping `user_id=user_id` (reviewer's probe), the `claims_verify` usage row was written:
    - for the username (`'alice'`) when the request has no `user_id` parameter;
    - for `'2'` when it passes `?user_id=2`.

    Quota decisions stayed with user 7 in both cases.

- **The RED run** (Task 1 Step 2) on unchanged source: `3 failed`.
  - The off-loop test: `assert [(False, False)] == [(True, True)]`, meaning on the loop's thread and not checked.
  - The two refusal tests: `Failed: DID NOT RAISE LLMQuotaExceededError`, with `LLM assessment failed: 402: ...` logged by `adjudicator.py:389`.
- **Prototype** (this plan applied in a scratch worktree, first on `a6719bf24f`, then with the review's I1 fix on `df7c290fbf`):
  - the 3 new tests: `3 passed`;
  - **mutants** (the reviewer's, against the 3 new tests):
    - dropping any one of the three re-raises is caught (2, 2 and 1 tests fail);
    - putting the first call back on the loop is caught (1 test fails);
    - moving it off the loop through a plain executor that carries no LLM call state is caught by I1's assertion (`assert [] == [7]`);
    - putting the prompt-only fallback back on the loop is not caught (Review Focus 4);
  - `tests/Claims` and `tests/Claims_Extraction`: `1223 passed, 1 skipped` (103 s, on `df7c290fbf`). Both of the reviewer's runs gave `1222 passed, 1 failed`: the Hypothesis test `test_claims_jobs_contracts.py::test_review_metrics_payload_inclusive_window_property`, which fails on unchanged dev in the same run (`1219 passed, 1 failed`) and passes alone (Task 1 Step 4);
  - `Helper_Scripts/ci/test_quality_triage.py --enforce`: exit 0, "OK — no offenses beyond the baseline", and neither test file is in its report;
  - ruff on the four touched files: the same 4 findings as on dev (`adjudicator.py` I001; `test_adjudicator.py` I001 twice and F401 `typing.Optional`), none new;
  - `uvx bandit -q -r` on `adjudicator.py` and `fva_pipeline.py`: no output, exit 0 (without `-q`: "No issues identified").

## Global Constraints

- **Python:** `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. **Tests:** `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q <paths>`.
- **Branch** `fix/fva-adjudicator-off-loop` from `origin/dev`; the PR targets `dev`. Commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `--no-verify`, never `git stash`.
- **Test first:** Task 1's RED run happens on unchanged source and fails for the reasons above.
- **The pattern, verbatim:** `except LLMQuotaExceededError:` then `raise`, directly before the broad clause. No other error handling changes: a provider outage, a parse error, a timeout or a full pool still gives NEUTRAL 0.5, as `test_llm_handles_failure` pins.
- **Imports:** `from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError`. `fva_pipeline.py` already has it (line 21).
- **No new setting, exception type or helper.** `_run_bounded_claims_analyze` is reused as it is.
- **Test-quality ratchet (required gate):** `$PY Helper_Scripts/ci/test_quality_triage.py --enforce` exits 0, and `tldw_Server_API/tests/CI/test_test_quality_ratchet.py` gives `1 passed`. Each new test asserts on a result, not only on a collector or `.status_code`.
- **Lint comparison:** run `$PY -m ruff check --output-format concise <touched .py files>` here and on the same files from `origin/dev` (export them with `git show origin/dev:<path>` into a temporary tree holding the repo's `pyproject.toml`), and diff the two outputs with line and column numbers stripped. Expected: the same 4 findings. Do not run `ruff --fix`: the import blocks of `adjudicator.py` and `test_adjudicator.py` are already unsorted on dev.
- **Line-number baselines:** none of the touched files is in `Helper_Scripts/ci/scope_predicate_baseline.txt`.
- **Docs:** `Docs/Operations/Usage_Quotas.md` and `Docs/Design/2026-10-09-llm-usage-accounting-design.md` have no `Docs/Published` mirror (`Helper_Scripts/refresh_docs_published.sh` copies only `Operations/Env_Vars.md` from Operations, and nothing from `Docs/Design`). Edits are anchored on quoted text.
- **Backlog:** `tools/backlog-py` only (`PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task ...`), `--append-notes`, explicit ids for new tasks, `git add -A backlog/tasks`.
- **Task ids:** compute the next id right before `task create` (Task 2 Step 1), and push soon after; the companion plan `plan_fu_workflow_cancel_siblings.md` also files a task.

## Decisions for the owner (recommendation first)

1. **`POST /claims/verify/fva`, the adjudicator's only caller, answers 500 to every request (measured above).**
   - **Recommended:** land this fix now, and file a follow-up to repair the endpoint (Task 2 Step 1). Until the endpoint runs, the operator page does not list it among the endpoints that answer 402 (Task 2 Step 2).
     - The fix is complete without the endpoint: the pipeline tests show the refusal reaching `process_batch`, which raises it, and the endpoint has no `try` around that call.
     - Reviving the endpoint means three constructor calls, an endpoint test harness, and a decision on the `user_id` query parameter. That last one is an API and quota-accounting choice that deserves its own review:
       - **Where it reaches.** The parameter (`endpoints/claims.py:999`) reaches the usage row only through `ClaimsJobContext` (call 1 above). The charge to another user therefore goes live only when call 1 is fixed with `user_id=user_id` kept as it is.
       - **Who can set it.** Any authenticated caller in multi-user mode. The router has no permission dependency, and `get_auth_principal` only authenticates.
       - **What it moves.** Quota decisions stay with the authenticated user (measured). Only the `claims_verify` usage row moves to the parameter's value, as a string. On SQLite the INTEGER column stores `'2'` as 2, so the row counts toward user 2's monthly token and cost limits.
       - **Without the parameter,** the row carries the caller's username. So the caller's own verify usage never counts toward its limits.
       - Overall: latent today, and once live a quota-accounting integrity bug, not an access-control breach.
     - **The follow-up should admin-gate the parameter,** as the sibling claims endpoints do. `rebuild_claims` (`claims_service.py:4581-4585`) uses `user_id` only when `_legacy_user_has_platform_admin_claims(current_user)`, and otherwise the caller's own `current_user.id`. The follow-up records two alternatives:
       - drop the parameter, which changes the OpenAPI schema and `apps/packages/api-types`;
       - accept it and ignore it.
     - **The one extra cost:** the operator page is edited twice, now to say the endpoint answers 500, and again when the follow-up makes it answer 402.
   - **Alternative:** hold this PR and fold it into the endpoint repair, so that the 402 is tested end to end and the operator page is edited once. The prototype above shows what that PR would measure (cases A-C).
   - Either way the adjudicator change is the same.

## Review Focus

1. **The 60 s deadline does not bound latency; it discards a late answer.**
   - The pool call uses `drain_after_timeout=True`. Past `CLAIMS_PROVIDER_CALL_TIMEOUT_SECONDS` (60 s), `await_bounded_daemon_with_timeout` waits for the worker to finish, then raises `TimeoutError("Claims provider call timed out")` and drops the result. The reviewer measured a 0.2 s deadline on a 1 s call: it raised after 1.01 s.
   - So a call slower than 60 s takes as long as before, but its answer becomes NEUTRAL 0.5 instead of being used, as for any other failure (pinned by `test_llm_handles_failure`).
   - The loop is no longer blocked while it runs. The verifier's calls already behave this way.
2. **A full pool.** `SYNC_ADAPTER_CALL_POOL` holds 32 workers by default (`CHAT_SYNC_ADAPTER_MAX_WORKERS`) and is shared with every sync adapter call. When it is full, `DaemonCapacityError` (a `RuntimeError`) gives NEUTRAL 0.5 and a warning. Before, there was no bound.
3. **An async `llm_analyze_fn`** (the tests' `AsyncMock`) is now called on the worker thread. That only creates the coroutine, which the existing `inspect.isawaitable` branch awaits on the loop. Pinned by the existing `test_adjudicate_with_llm_fallback` and `test_llm_parses_json`.
4. **The prompt-only fallback** (`fn(prompt)` after a `TypeError`) also runs on the pool.
   - `test_llm_assess_falls_back_to_prompt_only_signature` pins that the fallback still works, but no test pins that it runs off the loop: the reviewer's mutant that puts it back on the loop passes both whole test files.
   - The risk is low. In the app the fallback never reaches a provider: `_fva_claims_analyze_call(prompt)` raises `TypeError` for its 4 missing positional arguments.
5. **A refusal stops the claim at its first refused call.** Supporting documents are assessed first, and the claim's other documents are not assessed. In a batch, the other claims run to completion and are then discarded by `process_batch`'s re-raise; their LLM calls are refused before reaching a provider. This plan does not change that.

## ADR assessment

Not needed. ADR-064 (per-user quotas) and ADR-067 (accounting in the provider base class) govern and do not change. Moving a sync LLM call off the loop applies design §4b and §5, which are already approved. "A refusal is let through" is the TASK-13638 rule. No new durable rule.

## Conflicts with other plans

| Place | This plan | Other | How to merge |
|---|---|---|---|
| `Usage_Quotas.md:95` and `:103` | Task 2 Step 2 edits both | `plan_fu_workflow_cancel_siblings.md` edits line 96, the workflow bullet | Adjacent lines: git conflicts for whichever lands second. Take the landed text and re-apply your own sentence edit by its quoted text. The two edits share no sentence. |
| `Usage_Quotas.md:95` | Task 2 Step 2 deletes "claims verification (`POST /claims/verify/fva`), " | `plan_fu_keep_artifact_on_refusal.md` also edits line 95 (it removes three endpoints and the "discard that request's results" sentence), and lines 93 and 98 | Same line: git conflicts for whichever lands second. Re-apply your own deletion by its quoted text. The two edits remove different phrases. |
| `Usage_Quotas.md` | as above | `plan_fu_anthropic_cache.md` inserts a bullet after line 107 (context 105-107); `plan_fu_rg_redis_error.md` edits line 124 | Neither is next to lines 95 and 103, and neither diff's context is touched here, so both apply cleanly in either order. |
| `Usage_Quotas.md` | as above | Open PR #3084 (2300 files) also touches the file | Same rule if it lands first. |
| Design doc, end of file | Task 2 Step 3 appends a revision paragraph | No open PR or plan touches the design doc (checked 2026-10-20, including the 11 open PRs with more than 100 files, paged) | None. |

---

### Task 1: the adjudicator's LLM call runs on the bounded pool, and a refusal fails the claim (AC1, AC2)

**Files:**
- Modify: `tldw_Server_API/app/core/Claims_Extraction/adjudicator.py`
  - imports after line 22 (the `analyze_types` import);
  - before the `except Exception as e:` at line 231 (`_assess_documents`);
  - the `fn(` call at line 302 and the `fn(prompt)` call at line 327 (`_invoke_llm_assess_call`);
  - before the `except Exception as e:` at line 388 (`_llm_assess`).
- Modify: `tldw_Server_API/app/core/Claims_Extraction/fva_pipeline.py`, before the `except Exception as e:` at line 329 (`process_claim`'s falsification `try`).
- Test: `tldw_Server_API/tests/Claims_Extraction/test_adjudicator.py` (import block at lines 8-20; append a class at the end, after line 685).
- Test: `tldw_Server_API/tests/Claims_Extraction/test_fva_pipeline.py` (append after line 746).

**Interfaces:**
- Consumes:
  - `claims_engine._run_bounded_claims_analyze(analyze_fn, *args, **kwargs) -> Any`, which runs `partial(analyze_fn, *args, **kwargs)` on `SYNC_ADAPTER_CALL_POOL`, with a 60 s deadline;
  - `exceptions.LLMQuotaExceededError` (an `HTTPException`, 402; `detail["category"]`).
- Produces:
  - `ClaimAdjudicator.adjudicate(...)`, `ClaimAdjudicator._llm_assess(...)` and `FVAPipeline.process_claim(...)` raise `LLMQuotaExceededError` instead of returning a degraded result;
  - signatures unchanged.

- [ ] **Step 0: start the task and the branch**

```bash
git fetch origin dev --quiet && git switch -c fix/fva-adjudicator-off-loop origin/dev
mkdir -p Docs/superpowers/plans
cp /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/scratchpad/plan_fu_fva_adjudicator.md Docs/superpowers/plans/2026-10-20-fva-adjudicator-off-loop.md
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13862 -s "In Progress" \
  --append-notes "Plan: Docs/superpowers/plans/2026-10-20-fva-adjudicator-off-loop.md. The adjudicator's two analyze calls go through claims_engine._run_bounded_claims_analyze (the pool the verifier uses); a refusal is re-raised before the broad clauses in _llm_assess, _assess_documents and FVAPipeline.process_claim. Measured: POST /claims/verify/fva, the only app caller, answers 500 to every request (three constructor calls that cannot succeed: ClaimsJobContext(job_id=...), MultiDatabaseRetriever(media_db=db) without db_paths, RetrievalConfig(top_k=...)); owner Decision 1."
git add -A backlog/tasks Docs/superpowers/plans/2026-10-20-fva-adjudicator-off-loop.md
```

- [ ] **Step 1: write the failing tests**

In `tests/Claims_Extraction/test_adjudicator.py`, replace the block from `import pytest` through the closing `)` of the `adjudicator import (` statement (dev lines 8-20) with:

```python
import asyncio
import threading

import pytest
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import tldw_Server_API.app.core.Claims_Extraction.adjudicator as adjudicator_module
from tldw_Server_API.app.core.Claims_Extraction.adjudicator import (
    ClaimAdjudicator,
    EvidenceStance,
    EvidenceAssessment,
    AdjudicationResult,
    create_adjudicator,
)
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Usage import llm_accounting, llm_quota_gate, quota_resolver
from tldw_Server_API.app.core.Usage.llm_call_context import llm_call_context
```

Append to the same file:

```python


def _refusal() -> LLMQuotaExceededError:
    return LLMQuotaExceededError(
        category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
    )


class TestClaimAdjudicatorLLMQuota:
    """The adjudicator's LLM call is quota-checked, and a refusal is not an assessment (TASK-13862)."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_llm_call_runs_off_the_loop_where_the_quota_check_runs(self, monkeypatch):
        """On the loop's own thread the check at the call is skipped (LLM usage accounting design section 5)."""
        decided_for: list[int] = []

        async def _admit(user_id, requested_tokens=1):
            decided_for.append(user_id)
            return None

        monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
        quota_resolver.invalidate_all()
        monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _admit)
        monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
        loop_thread = threading.get_ident()
        seen: list[tuple[bool, bool]] = []

        def _analyze(*_args, **_kwargs):
            # (off the loop thread, checked) for the call the provider layer would make here
            seen.append((threading.get_ident() != loop_thread, llm_quota_gate.check_llm_call_sync()))
            return '{"stance": "CONTRADICTS", "confidence": 0.8}'

        adjudicator = ClaimAdjudicator(llm_analyze_fn=_analyze)
        with llm_call_context(user_id=7):
            stance, confidence = await adjudicator._llm_assess("claim", "evidence")

        assert (stance, confidence) == (EvidenceStance.CONTRADICTS, 0.8)
        assert seen == [(True, True)]
        assert decided_for == [7]  # the request's user crossed into the worker thread

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_a_quota_refusal_is_not_a_neutral_assessment(self):
        """On dev the refusal became NEUTRAL 0.5, and the claim kept its verdict."""
        adjudicator = ClaimAdjudicator(llm_analyze_fn=AsyncMock(side_effect=_refusal()))
        claim = MockClaim(id="1", text="The sky is green.")

        with pytest.raises(LLMQuotaExceededError) as refused:
            await adjudicator.adjudicate(
                claim=claim,
                supporting_docs=[Document(id="1", content="The sky is blue.", metadata={}, score=0.8)],
                contradicting_docs=[],
                original_verification=MockClaimVerification(claim=claim),
            )
        assert refused.value.detail["category"] == "llm_tokens_month"
```

`check_llm_call_sync()` is the check `perform_chat_api_call` runs first (`chat_service.py:2636`). It returns `False` when it could not run, on the accounting loop's own thread. It returns `True` when it ran and admitted the call, but also when there was no user to check (`llm_quota_gate.py:163-166`). So `seen == [(True, True)]` alone would pass a move off the loop that loses the request's user. `decided_for == [7]` shows that the decision was made, and made for the request's user. The reviewer's mutant runs the call through a plain `ThreadPoolExecutor`, which carries no LLM call state, and fails it with `assert [] == [7]` (measured, `fu_p1r_mutant.py`).

Append to `tests/Claims_Extraction/test_fva_pipeline.py`:

```python


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_claim_lets_an_adjudication_refusal_through():
    """A refusal during adjudication fails the claim; it does not keep the unchallenged verdict (TASK-13862)."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    low_confidence = MockClaimVerification(
        claim=MockClaim(id="1", text="Test"),
        confidence=0.5,
        evidence=[MockEvidence(doc_id="1", snippet="Evidence")],
    )
    claims_engine = MockClaimsEngine(MockVerifier(verification=low_confidence))
    claims_engine._analyze = AsyncMock(
        side_effect=LLMQuotaExceededError(
            category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
        )
    )
    anti_docs = [Document(id="anti_1", content="Contradicting evidence", metadata={}, score=0.7)]
    pipeline = FVAPipeline(claims_engine, MockRetriever(documents=anti_docs))

    with pytest.raises(LLMQuotaExceededError) as refused:
        await pipeline.process_claim(
            claim=MockClaim(id="1", text="Test claim"),
            query="test",
            documents=[Document(id="1", content="Support", metadata={}, score=0.8)],
        )
    assert refused.value.detail["category"] == "llm_tokens_month"
```

The verification is passed to `MockVerifier`'s constructor, not assigned to an attribute: the test-quality ratchet counts an attribute assignment of a module-defined fake as `stub_injection`, and this file's baseline is already 2.

- [ ] **Step 2: run them on unchanged source**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q \
  "tldw_Server_API/tests/Claims_Extraction/test_adjudicator.py::TestClaimAdjudicatorLLMQuota" \
  "tldw_Server_API/tests/Claims_Extraction/test_fva_pipeline.py::test_process_claim_lets_an_adjudication_refusal_through"
```

Expected: `3 failed` (measured on `a6719bf24f`, and again with I1's assertion on `df7c290fbf`).
- The off-loop test: `assert [(False, False)] == [(True, True)]`.
- The two refusal tests: `Failed: DID NOT RAISE LLMQuotaExceededError`, each with the warning `LLM assessment failed: 402: {'error': 'limit_exceeded', ...}`.

- [ ] **Step 3: implement**

`adjudicator.py`, the imports. After

```python
from tldw_Server_API.app.core.Claims_Extraction.analyze_types import (
    AdjudicatorAnalyzeCallable,
)
```

add:

```python
from tldw_Server_API.app.core.Claims_Extraction.claims_engine import _run_bounded_claims_analyze
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
```

`claims_engine` imports neither `adjudicator` nor `fva_pipeline`, so there is no import cycle; `fva_pipeline` already imports `claims_engine` at runtime (line 40).

`_assess_documents`, before its `except Exception as e:`:

```python
            except LLMQuotaExceededError:
                raise  # the user's limit: not a document that could not be assessed
            except Exception as e:
                logger.warning(f"Failed to assess document {doc.id}: {e}")
                continue
```

`_invoke_llm_assess_call`, the first call. Replace

```python
            response = fn(
                provider,
```

with

```python
            # Off the loop: a sync LLM call on the loop's thread blocks it and cannot be quota-checked.
            response = await _run_bounded_claims_analyze(
                fn,
                provider,
```

The remaining arguments (`prompt, prompt, None, system, temperature, streaming=False, ..., max_tokens=300`) and the `inspect.isawaitable` branch after the call stay as they are. Then replace

```python
            fallback = fn(prompt)
```

with

```python
            fallback = await _run_bounded_claims_analyze(fn, prompt)
```

`_llm_assess`, before its `except Exception as e:`:

```python
        except LLMQuotaExceededError:
            raise  # the user's limit: not a neutral stance
        except Exception as e:
            logger.warning(f"LLM assessment failed: {e}")
            return EvidenceStance.NEUTRAL, 0.5
```

`fva_pipeline.py`, `process_claim`, between `except asyncio.TimeoutError:` and `except Exception as e:`:

```python
                    except LLMQuotaExceededError:
                        raise  # the user's limit fails the claim; it does not keep the unchallenged verdict
                    except Exception as e:
                        logger.error(f"Falsification error for claim {claim.id}: {e}")
```

- [ ] **Step 4: run the new tests, then the neighbourhood**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q \
  "tldw_Server_API/tests/Claims_Extraction/test_adjudicator.py::TestClaimAdjudicatorLLMQuota" \
  "tldw_Server_API/tests/Claims_Extraction/test_fva_pipeline.py::test_process_claim_lets_an_adjudication_refusal_through"
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Claims tldw_Server_API/tests/Claims_Extraction
```

Expected:
- `3 passed`;
- then `1223 passed, 1 skipped` (measured on the prototype, 103 s).

The Hypothesis test `test_claims_jobs_contracts.py::test_review_metrics_payload_inclusive_window_property` can fail in the full run, giving `1222 passed, 1 failed`. Both of the reviewer's runs did. It fails the same way on unchanged dev in the full run (`1219 passed, 1 failed`), and passes alone, for example with `--hypothesis-seed=93458965372782150207670795045311259442`. Rerun it alone and record both results.

Before attributing any other failure to this change, run it on unchanged `origin/dev`. Then the lint comparison (Global Constraints): the same 4 findings.

- [ ] **Step 5: commit**

```bash
git add -A backlog/tasks tldw_Server_API/app/core/Claims_Extraction/adjudicator.py tldw_Server_API/app/core/Claims_Extraction/fva_pipeline.py \
  tldw_Server_API/tests/Claims_Extraction/test_adjudicator.py tldw_Server_API/tests/Claims_Extraction/test_fva_pipeline.py
git commit -m "fix(claims): the FVA adjudicator calls the LLM on the bounded pool, where the quota check runs, and lets a refusal through (TASK-13862)" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 2: the operator page, the design note, the endpoint follow-up, and close-out

**Files:** Modify `Docs/Operations/Usage_Quotas.md` and `Docs/Design/2026-10-09-llm-usage-accounting-design.md`. Backlog: TASK-13862 and one new task.

**Interfaces:** Consumes Task 1's behaviour. Produces the follow-up task id `$FVA_EP` used in the docs below.

If the owner takes Decision 1's alternative, stop after Task 1: the endpoint repair absorbs this task.

- [ ] **Step 1: file the endpoint follow-up first, so the docs can name it.** Compute the highest id with the AGENTS.md recipe, extended to page through open PRs with more than 100 files (`gh pr list --json files` returns at most 100 per PR):

```bash
git fetch origin dev --quiet
HIGHEST=$( {
  git ls-tree -r --name-only origin/dev backlog/
  gh pr list --state open --limit 500 --json files --jq '.[].files[].path'
  for n in $(gh pr list --state open --limit 500 --json number,changedFiles --jq '.[] | select(.changedFiles > 100) | .number'); do
    gh api --paginate "repos/rmusser01/tldw_server/pulls/$n/files" --jq '.[].filename'
  done
} | sed -nE 's|^backlog/.*task-([0-9]+)[ .].*|\1|p' | sort -n | tail -1 )
echo "$HIGHEST"
FVA_EP="TASK-$((HIGHEST + 1))"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "POST /claims/verify/fva answers 500 to every request" --id "$FVA_EP" -l claims,quotas,backend \
  -d "verify_claims_with_fva (Claims_Extraction/claims_service.py) has three constructor calls that cannot succeed, so the endpoint answers 500 to every request before any LLM call: (1) ClaimsJobContext(job_id=..., user_id=...) near line 4759, made when fva_config.max_budget_usd is set; ClaimsJobContext has no job_id field. (2) MultiDatabaseRetriever(media_db=db) near line 4770, without the required positional db_paths. (3) RetrievalConfig(top_k=..., search_mode='hybrid') near line 4782; RetrievalConfig has max_results, use_fts and use_vector. All date from the endpoint's first commit (05245a3d59, 2026-02-05); no test calls it. The user_id query parameter needs a decision before call (1) is fixed. Today it reaches the usage row only through ClaimsJobContext; with call (1) fixed and user_id=user_id kept, the claims recorder (claims_engine.py near line 677) writes each claims_verify usage row for whatever id any authenticated caller passes, and for the caller's username when none is passed, so the caller's own verify usage never counts toward its limits. Quota decisions stay with the authenticated user. Recommended: admin-gate the parameter as rebuild_claims does (claims_service.py near line 4581: the parameter only for _legacy_user_has_platform_admin_claims, else current_user.id), and record the authenticated user's id, not the username. Alternatives: drop the parameter (an OpenAPI and apps/packages/api-types change), or accept and ignore it. Also: retrieval should be scoped to the authenticated user; FVAPipeline reads the verifier's lazily loaded _nli before it is loaded, so the adjudicator never uses NLI. Since the endpoint has never worked, removing it is the other option. Found while planning TASK-13862: its prototype with calls (2) and (3) fixed answered 402 for a refusal at the verify or adjudication call, with every decision made for the request's user." \
  --ac "POST /claims/verify/fva runs the FVA pipeline on the caller's media DB, with an endpoint test that checks a result" \
  --ac "A quota refusal at its verify or adjudication call answers 402, with a test" \
  --ac "Claims usage rows are written for the authenticated user's id unless an admin passes user_id, with a test" \
  --ac "Usage_Quotas.md lists POST /claims/verify/fva among the endpoints that answer 402"
git add -A backlog/tasks
```

Measured on 2026-10-20: the highest id on `origin/dev` was `13866`. Run the command again at execution time: the companion plan files a task too.

- [ ] **Step 2: `Usage_Quotas.md` (Decision 1 recommended)**
  - Line 95, the HTTP endpoint bullet: delete "claims verification (`POST /claims/verify/fva`), " so that the sentence reads "... prompt improvement (`POST /prompts/improve`), and manual character memory extraction (`POST /characters/{id}/memories/extract`)."
  - Line 103: replace "`/claims/verify/fva` answers 402, and a generated artifact whose claims cannot be checked reports the refusal instead of `claim_verification_failed`." with:

    > a generated artifact whose claims cannot be checked reports the refusal instead of `claim_verification_failed`. `POST /claims/verify/fva` answers 500 to every request until $FVA_EP is fixed.

    Write the computed id in place of `$FVA_EP`.

- [ ] **Step 3: the design doc.** At the end of the file, after the "Revision of 2026-10-18, TASK-13638:" paragraph and its bullets, append (use the commit date if it is not 2026-10-20; write the computed id in place of `$FVA_EP`):

```markdown

Revision of 2026-10-20, TASK-13862:

- **The FVA adjudicator's LLM call** runs on the bounded adapter pool, as the claims engine's own calls do, so it is checked at the call (§5). A refusal there fails the FVA batch instead of becoming a neutral stance. Its only caller, `POST /claims/verify/fva`, answers 500 to every request ($FVA_EP).
```

- [ ] **Step 4: the checks**

```bash
$PY Helper_Scripts/ci/test_quality_triage.py --enforce
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/CI/test_test_quality_ratchet.py tldw_Server_API/tests/Docs
uvx bandit -q -r tldw_Server_API/app/core/Claims_Extraction/adjudicator.py tldw_Server_API/app/core/Claims_Extraction/fva_pipeline.py
```

Expected:
- the triage exits 0 with "OK — no offenses beyond the baseline" (measured on the prototype);
- `213 passed` (measured with the Step 2 and 3 edits applied);
- bandit prints nothing and exits 0: with `-q`, a clean run is silent (measured; without `-q` it says "No issues identified"). Bandit is not installed in `.venv`; if `uvx` is unavailable, record the skip.

Then the lint comparison: the same 4 findings.

- [ ] **Step 5: close TASK-13862**

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13862 \
  --append-notes "RED on origin/dev: 3 failed (the off-loop test assert [(False, False)] == [(True, True)]: the call ran on the loop's thread and was not checked; the adjudicate and process_claim refusal tests DID NOT RAISE, the refusal became NEUTRAL 0.5). GREEN: 3 passed. The off-loop test also asserts that the quota decision is made for the request's user (decided_for == [7]); a move off the loop without the LLM call state fails it. tests/Claims + tests/Claims_Extraction: 1223 passed, 1 skipped (the Hypothesis test test_review_metrics_payload_inclusive_window_property can fail in the full run, on dev too, and passes alone). Test-quality ratchet exit 0; ruff: the same 4 findings as dev; bandit: clean." \
  --append-notes "Decision 1, recommended option: fix the adjudicator now; the endpoint repair is $FVA_EP. POST /claims/verify/fva answers 500 to every request (three constructor calls that cannot succeed), so the change is latent until then; a prototype with the endpoint repaired answered 402 for a refusal at the adjudication call (200 with an unchecked call on dev)." \
  --check-ac 1 --check-ac 2 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "ClaimAdjudicator calls its analyze function through claims_engine._run_bounded_claims_analyze, so the call runs on the bounded adapter pool, is checked at the call and no longer blocks the loop. A quota refusal is re-raised before the broad clauses in _llm_assess, _assess_documents and FVAPipeline.process_claim, so it fails the batch instead of becoming a neutral stance. The only caller, POST /claims/verify/fva, answers 500 to every request; its repair is $FVA_EP, and the operator page says so." \
  -s Done
git add -A backlog/tasks
```

The tallies in the first note are the prototype's. If a run gives different numbers, write the numbers it gave, with the dev baseline for any failure.

- [ ] **Step 6: commit and PR**

```bash
git add Docs/Operations/Usage_Quotas.md Docs/Design/2026-10-09-llm-usage-accounting-design.md
git commit -m "docs(quotas): the FVA adjudicator is checked at the call; /claims/verify/fva answers 500 until its repair; close TASK-13862" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git push -u origin fix/fva-adjudicator-off-loop
gh pr create --base dev --title "FVA adjudicator: LLM call off the event loop, quota refusal let through (TASK-13862)" --body-file /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/scratchpad/pr_body_13862.md
```

The PR body needs:
- what changed and why;
- the measured endpoint 500 and Decision 1 as taken;
- the follow-up id;
- a "## Change summary" section. ADR-004: AI-authored PRs need the owner's own Change summary, and no recorded waiver names TASK-13862, so ask the owner for it;
- the attribution line `🤖 Generated with [Claude Code](https://claude.com/claude-code)` at the end.

Then:
- request a clean review (`/superpowers:requesting-code-review` in a fresh subagent) and address it before merging;
- check `gh variable get MERGE_QUEUE` and merge as `CLAUDE.md` says for that mode.

## Out of scope

- **Repairing `POST /claims/verify/fva`**: the follow-up filed in Task 2 Step 1, unless the owner takes Decision 1's alternative.
- **The adjudicator never using NLI in the app**, because `FVAPipeline` reads `_nli` before it is loaded. This is in the follow-up's description.
- **A boundary check (§4a) for FVA**: its LLM use depends on the claim and the evidence, so it is not "always" (TASK-13638, Out of scope).
