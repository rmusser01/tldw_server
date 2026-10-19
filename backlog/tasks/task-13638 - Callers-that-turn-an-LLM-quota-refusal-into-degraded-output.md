---
id: TASK-13638
title: Callers that turn an LLM quota refusal into degraded output
status: Done
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After LLM usage accounting PR 2b (TASK-13437) an LLM call for an over-quota user raises LLMQuotaExceededError at the call. These callers catch it broadly and go on with degraded output instead of reporting it: workflow adapters that return an error output (Workflows/adapters/content/generation.py, text/nlp.py, research/bibliography.py, content/image.py, content/summarize.py, llm/translate.py, which returns the input as a fallback translation; control/flow.py parallel drops it; control/state.py retry retries it); claims verification (Claims_Extraction/claims_service.py, whose _CLAIMS_NONCRITICAL_EXCEPTIONS holds HTTPException, falls back to heuristic verdicts); Prompt Studio test runs and evaluation (test_runner.py, evaluation_manager.py save error outputs and 0.0 scores) and benchmarks (evaluation_manager.py returns score 0.0); character memory extraction (character_memory_extraction.py returns an empty result). Decide for each whether to let the refusal through (except LLMQuotaExceededError: raise) or keep the degraded output and say so.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each listed caller either lets the refusal through or records that it was refused
- [x] #2 The operator page lists what each one does
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From the PR 2b final review: a media item that reuses a transcript calls analyze per chunk inside _PERSISTENCE_NONCRITICAL_EXCEPTIONS (Ingestion_Media_Processing/persistence.py near lines 1701-1710), so every chunk after the first refusal is attempted and refused (one hand-off and counter read each); stop at the first refusal. The Explainer node stores generation_failed for a refusal (Explainer/jobs.py near line 451) while the job's error code says llm_quota_exceeded; carry the exception's failure_code into the node's error metadata.
From the PR #3234 review: HyDE's legacy branch (RAG/rag_service/unified_pipeline.py near line 4063, taken with no credential runtime, so internal callers only) calls the sync sgl.analyze on the server loop, where the check at the call cannot run; move it to asyncio.to_thread. Changing unified_pipeline.py moves line numbers in two baselines: regenerate them (Helper_Scripts/ci/scope_predicate_ratchet.py --write-baseline; Helper_Scripts/web_scraping_refactor_inventory.py --json ... --markdown ...).
Plan: plan_quota_callers_rag_metric.md (PR A, tasks 1-9). Each caller lets LLMQuotaExceededError through, or keeps its artifact and records the refusal; per-caller table in the plan.
RED on origin/dev 4d35534699 (the plan's source files are unchanged since 7c94a2efb1): 23 workflow tests (22 adapter cases DID NOT RAISE, including the evaluations step; the engine test assert 4 == 1); claims 3 failed (DID NOT RAISE); Prompt Studio 7 failed (including _call_llm RuntimeError and the execute endpoint's 502); benchmarks 3 failed (no 402: dataset load and benchmark lookup ran); memory 1; transcript reuse 1 (assert 20 == 1); Explainer 1 failed, 1 passed (control); HyDE 1 (same thread). GREEN: 23, 6, 7, 3, 1, 1, 2 and 1 passed (41 test ids).
Neighbourhood suites: tests/Workflows 1682 passed, 6 skipped; tests/prompt_studio 1180 passed, 17 skipped; tests/Evaluations (without property/ and integration/) with tests/Usage 1151 passed, 7 skipped; tests/MediaIngestion_NEW/unit with the video dedupe integration file and Media/test_media_llm_quota.py 610 passed, 8 skipped, 1 xfailed; both character memory files 51 passed; tests/Explainer 49 passed; the three HyDE files 58 passed; tests/Docs 212 passed. tests/Claims, Claims_Extraction and Flashcards under load average 80-97: 2 failed, 1704 passed, 1 skipped; the failures were the known Hypothesis FailedHealthCheck (too_slow) in test_claims_jobs_contracts.py and test_claims_rebuild_health_reads_persisted (assert 0 == 7), which passed alone; a rerun of tests/Claims gave 1075 passed, 1 skipped, and tests/Claims on an origin/dev export passed it too (1 failed, the same Hypothesis check; 1 error, a Postgres restricted-role test). Test-quality ratchet: triage --enforce exit 0, test_test_quality_ratchet.py 1 passed. scope_predicate_ratchet: baseline exact; the web scraping inventory regenerates unchanged. OpenAPI drift gate exit 0. Ruff on the 38 touched Python files: 35 findings on origin/dev, the same 35 after.
Decisions taken (owner, 2026-10-18): 1, a refused workflow step fails, with its run, and is not retried ('402' in _NON_RETRIABLE_REASONS); 2, benchmark runs get the boundary check at the top of each handler; 3, keep the user's artifact, record the refusal with its code, and file the follow-up. Follow-ups: TASK-13862 (FVA adjudicator sync call on the loop), TASK-13863 (cancel sibling sub-steps in map and parallel), TASK-13864 (add-on paths that still discard the artifact).
Bandit -ll on the 23 touched app files: one Medium/Low B608 in Evaluations/evaluation_manager.py (the internal_evaluations IN-clause query), present on origin/dev at line 855; nothing new.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Every caller listed in the task, plus Prompt Studio's executor and the workflow evaluations step, lets LLMQuotaExceededError through, or keeps its artifact and records the refusal. Sequential loops stop at the first refusal; fan-outs fail with it, and their siblings' calls are refused before reaching a provider (cancelling the siblings is a follow-up). The workflow engine does not retry a 402. Benchmark runs are checked at the request boundary. Usage_Quotas.md lists what each caller does.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
