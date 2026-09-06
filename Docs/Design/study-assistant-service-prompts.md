# Study Assistant Service Prompts

Approved bounded slice, tracked by TASK-13208 (formerly TASK-13199).

Expose four literal `guidance` parts through existing Service Prompts:
`study.assistant.explain`, `study.assistant.mnemonic`,
`study.assistant.followup`, and `study.assistant.freeform`.
Both flashcard and quiz-question response endpoints consume the selected
action's authenticated-owner configuration once per request.

Keep the shared grounding prefix, bounded context/history, learner message,
provider/model, temperature, token limit, response processing, and persistence
unchanged. Defaults reproduce the existing model messages. Fact-checking keeps
its original instructions and structured-output path without prompt lookup.

Use a shared HTTP dependency for authenticated storage acquisition and safe
error mapping, and the existing Study Assistant core module for action
selection and prompt assembly. Resolve and close prompt storage on the same
worker. Pass only immutable guidance into generation; no global prompt cache,
new storage, jobs integration, or client API fields.

Tests exercise real HTTP routes and owner databases with the model transport
replaced. Cover all four actions on both endpoints, default compatibility,
save/reset, cross-owner isolation, edits during generation, fact-check bypass,
and invalid configuration before model calls or response-message writes.
Verify shared Settings editing, lint, Bandit, API compatibility and code review.
