---
id: TASK-15426
title: Use the user's stored key for media analysis, the study assistants, watchlist
  summaries, evaluation runs and benchmarks
status: In Progress
labels:
- quotas
- backend
- byok
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up of TASK-13956. These features call their provider with the server's key and never with a stored user key: media analysis (the /media endpoints and media ingest jobs, every call through Summarization_General_Lib._summarize_via_adapter with no provider_credentials), the flashcards and quizzes study assistants (study_assistant.generate_study_assistant_reply, perform_chat_api_call_async with no key), watchlist output summaries (outputs_service._llm_reply), evaluation runs (eval_runner, resolve_provider_api_key_from_config) and benchmarks (a raw api_key from the request or the server's). So their calls are the operator's spend and count against the per-user LLM limits, and their boundaries refuse an over-limit user even when the user stored a key for the provider. Product question for the owner: should these features use the user's stored key (moving their cost to that key)? If yes, resolve it through ProviderCredentialRuntime and pass the handle; then each boundary names its provider to enforce_llm_quota in one line, and the ingest job's start check (_admit_or_refuse_analysis) admits an over-limit user whose analysis provider runs on their own key without marking the job admitted. Cost differs by family (TASK-13956 plan review): the study assistants are the cheap first slice (build a runtime in the route, resolve, pass the handle; perform_chat_api_call_async already binds it); benchmarks and watchlist summaries are small to medium (analyze takes provider_credentials); evaluation runs are medium (a runtime that outlives the request); media analysis and the ingest job are about a day with tests (a handle through about ten ingestion libraries, or a context-scoped handle read in _summarize_via_adapter, plus a runtime per job).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner decides whether each listed feature uses the user's stored key
- [ ] #2 Each feature the owner approves resolves the user's key through ProviderCredentialRuntime and passes the handle on every LLM call, with a test
- [ ] #3 Its boundary names its provider to enforce_llm_quota, and the media ingest job's start check admits an over-limit user whose analysis provider runs on their own key without marking the job admitted, with a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
AC #1 answered (owner, 2026-11-07): all four families use the user's stored key, built cheapest first: (a) the flashcards and quizzes study assistants; (b) benchmark runs and watchlist output summaries; (c) evaluation runs; (d) media analysis (the /media endpoints) and media ingest jobs. ADR-004 Change summary waived for every TASK-15426 PR (owner, 2026-11-07: "Yes, waive for all").
Plan, PR A of 4 (study assistants): Docs/superpowers/plans/2026-11-07-own-key-study-assistants.md.
PR A (https://github.com/rmusser01/tldw_server/pull/3444): the flashcards and quizzes study assistants run on the user's own key for the reply's provider (users_own_key_credentials + enforce_llm_quota_on_own_key in usage_quota_deps.py; study_assistant_own_key replaces require_llm_quota on both routes). Over a limit: own key admitted unmarked, server key and no usable provider refused with the same 402. Tests: focused 17 passed; wider set 1041 passed, 1 skipped, 0 failed (34 files; base 1026 passed, 1 skipped). Bandit: no issues. Docs: Usage_Quotas.md, BYOK guide (including last_used_at), ADR-083 amendment (mirrored). The prompt-improvement live-DNS test is left to the live-network-tests task (filed by the benchmark-judge PR).
Plan, PR B of 4 (benchmark runs, watchlist output summaries): Docs/superpowers/plans/2026-11-07-own-key-benchmarks-watchlist-summaries.md.
PR B (https://github.com/rmusser01/tldw_server/pull/3452): benchmark runs (both routes; a raw api_key names no provider) and watchlist output summaries run on the user's own key (enforce_llm_quota_on_own_key; handle through evaluate_custom_metric, both the model response and the judge call, and summarize_items_for_output). Enrichment unchanged. The standalone /benchmarks/{name}/run no longer returns 500 after its LLM calls with the default save_results: it stored its rate limiter as user_id; it now stores the run in the user's own evaluations database under their id. Built on TASK-15485 (#3445, the benchmark judge fix, merged first). Tests: focused 11 passed, 0 network attempts; wider set 1 failed (known probe artifact test_ssrf_protection_private_ip), 534 passed, 4 skipped, 1 xpassed (base 1 failed, 526 passed); Docs/lint/CI/backlog-py 2 failed, 1768 passed, 6 skipped, 1 error (the same environment-only ids as the base). Bandit: one pre-existing finding (B608), none new. Docs updated (mirrored).
Plan, PR C of 4 (evaluation runs): Docs/superpowers/plans/2026-11-07-own-key-evaluation-runs.md. Filed TASK-15491: label-choice and NLI evaluation runs score a failed judge call 0.0 (pre-existing; the owner's 2026-11-10 benchmark ruling applied to runs).
PR C (https://github.com/rmusser01/tldw_server/pull/3460): evaluation runs name their judge provider (run_llm_provider, the runner's own resolution) and run every judge call on the user's own key (handle in eval_config, never stored; summarization, RAG, response quality, label choice, NLI; a direct judge call checks the handle's provider). rag_pipeline, raw api_key and no-LLM runs still refused over a limit. A run already started keeps its key (owner, 2026-11-07); the BYOK guide says to cancel it. Built on PR B (#3452). Follow-up: TASK-15491 (label choice and NLI score a failed judge call 0.0). Tests: focused set 31 passed (0 network attempts); wider set 60 files 825 passed, 42 skipped, plus the 3 PostgreSQL files 90 passed, 0 netprobe attempts; CI suites (Docs, lint, CI, backlog-py) 1773 passed, 6 skipped, 2 environment-only failures and 1 environment-only error (no coverage, no setuptools in the shared venv). Bandit: no issues. Docs updated (mirrored).
PR C final review (fable, clean): Ready to merge Yes, 0 Critical, 0 Important, 4 Minor (deferred). Filed TASK-15492: an evaluation's evaluator_model set to a model name is used as the provider name (pre-existing).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
