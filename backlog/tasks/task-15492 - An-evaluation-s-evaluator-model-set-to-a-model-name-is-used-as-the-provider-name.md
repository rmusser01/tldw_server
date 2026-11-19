---
id: TASK-15492
title: An evaluation's evaluator_model set to a model name is used as the provider
  name
status: Done
labels:
- evaluations
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`EvaluationRunner._resolve_eval_provider_model_api_key` (`tldw_Server_API/app/core/Evaluations/eval_runner.py:228-235`) reads the spec's `evaluator_model`, which the code comment says may name a provider OR a model. It calls `normalize_provider(str(evaluator_model))` and takes the result as the provider whenever it is truthy, falling back to treating the value as a model only when it is empty. `normalize_provider` (`core/LLM_Calls/adapter_utils.py:53`) only strips and lower-cases, so any non-empty value becomes the provider and the model branch never runs. `Evaluations_API_Reference.md` documents `"evaluator_model": "gpt-4"` as the model to use (`:55`, `:553`, `:582`, `:617`); such an evaluation's judge calls go to a provider named `gpt-4` and fail at the adapter. A `provider:model` or `provider/model` value is also taken whole as the provider, because the provider-prefix split below it only looks at the model override.

Pre-existing, the same on the server's key. Since TASK-15426 PR C (#3460) the run route names this provider to the LLM quota, so an over-limit user with a stored key for the real provider is refused, as every over-limit run was before PR C. Found by PR C's final review (2026-11-12).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An `evaluator_model` that is a known provider name (for example `openai`, `anthropic`) still selects that provider, with a test.
- [x] #2 An `evaluator_model` that is a model name (for example `gpt-4`) is used as the model, and the provider comes from `api_name`/`provider` or the default, with a test.
- [x] #3 A `provider:model` or `provider/model` `evaluator_model` selects both, with a test; the run route's `run_llm_provider` names the same provider the runner uses.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Owner decisions 2026-11-16 (plan Docs/superpowers/plans/2026-11-17-evals-runner-cli.md): (1) a label-choice or NLI call that succeeds but returns an unparseable reply keeps accuracy 0.0, since that call is the system under test, not a judge; (2) the known-provider check is the built-in adapter table (ChatProviderRegistry.DEFAULT_ADAPTERS plus canonical_provider_name aliases); (3) --save/--no-save removed from eval custom; (4) an unscored eval custom exits 1; (5) an explicitly named provider stops the prefix:/prefix/ split; (6) eval rag and eval quality runners are filed as TASK-15561, not fixed here; (7) ADR-004 waiver covers this PR. PR #3499.
Evidence: AC1 the two provider-name rows (regression guard, pass on base); AC2 the model rows incl. the meta-llama/llama-3 split fix; AC3 provider:model, provider/model and the run_llm_provider equality assertion on every row.
Final review fix (PR #3499, I1): evaluator_model is split once. openrouter/openai/gpt-4o, ollama:mistral:7b and together/Qwen/Qwen2-72B-Instruct keep their provider (3 rows added to the parametrized test: 3 failed before, 19 passed after).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
evaluator_model is a provider only when it names a known one; provider:model and provider/model give both; an explicitly named provider stops the split; run_llm_provider names the same pair as the runner. test_evaluator_model_is_a_provider_only_when_it_names_one: 14 of 16 rows failed on base, 16 passed GREEN; wider set 316 passed. PR #3499.
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
