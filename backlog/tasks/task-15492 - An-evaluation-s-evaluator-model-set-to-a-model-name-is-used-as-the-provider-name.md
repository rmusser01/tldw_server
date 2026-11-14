---
id: TASK-15492
title: An evaluation's evaluator_model set to a model name is used as the provider
  name
status: To Do
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
- [ ] #1 An `evaluator_model` that is a known provider name (for example `openai`, `anthropic`) still selects that provider, with a test.
- [ ] #2 An `evaluator_model` that is a model name (for example `gpt-4`) is used as the model, and the provider comes from `api_name`/`provider` or the default, with a test.
- [ ] #3 A `provider:model` or `provider/model` `evaluator_model` selects both, with a test; the run route's `run_llm_provider` names the same provider the runner uses.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
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
