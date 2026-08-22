---
id: TASK-13110
title: 'Wire automation executors to the server LLM plumbing'
status: Done
assignee:
  - '@robert'
created_date: '2026-08-22 22:30'
updated_date: '2026-08-22 22:30'
labels:
  - scheduled-tasks
  - automation
  - llm
dependencies:
  - task-13021
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The deferred slice of TASK-13021: the agent-task consumer ships with an injected per-family executor seam (`register_executor`), but no production registrant exists — runs currently fail honestly with `no_executor_configured`. This task wires the executors to the server's canonical LLM entrypoint (`perform_chat_api_call_async` in `core/Chat/chat_service.py`, the same surface Flashcards/Research/MCP modules use) so phase-1 scheduled automations actually generate:

- **`recurring_question`**: one completion call — the definition's `input.question` as the user message, a fixed generation-only system prompt, bounded `max_tokens`.
- **`agent_task`** (generation-only): the definition's `input.message` (or `prompt`) as the user message with the same guardrails — tools are already refused upstream by the consumer's phase-1 boundary, so the executor never sees a tool request it must honor.

**Model selection precedence** (per definition, degrading to server defaults): the definition's `input`/config `model` (+ optional `provider`) when present, else the automation executor defaults from config (`[Scheduled_Tasks_Automation]` executor_provider / executor_model), else the server's own default provider/model resolution (omit both and let `perform_chat_api_call_async` resolve). Credentials resolve through the existing provider-config layer — the executor adds no new secret handling.

Registered at worker startup (`run_agent_task_jobs_worker`) so a registered executor always accompanies a consuming worker; registration is idempotent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A production executor for `recurring_question` builds one completion call from `input.question` (system prompt fixed, generation-only) and returns the assistant text
- [x] #2 A production executor for `agent_task` builds the call from `input.message`/`input.prompt` with the same guardrails; both executors raise nothing on missing input except an honest error string recorded as a failed run
- [x] #3 Model/provider precedence: definition-level `model`(+`provider`) → automation config defaults (`executor_provider`/`executor_model`) → server default resolution; pinned by unit test over the resolution function
- [x] #4 Executors are registered at worker startup, idempotently, before the first job is acquired
- [x] #5 The call goes through `perform_chat_api_call_async` with bounded `max_tokens` (default 1000, definition-configurable, capped); no tools/tool_choice are ever passed
- [x] #6 Tests cover: question/message prompt construction, precedence matrix, registration idempotency, and the no-executor→wired transition through `handle_agent_task_job` with the real registry (LLM call mocked at the entrypoint boundary)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: consumes the existing TASK-13021 executor seam and the server's canonical chat entrypoint; no new boundary, storage, or contract — the phase-1 scope itself is ADR-077 decision 4 (owner-accepted).

1. `core/Scheduled_Tasks/automation_executors.py`: resolution helper + the two executors calling `perform_chat_api_call_async`
2. Worker startup registration (idempotent)
3. Tests per the AC matrix with the entrypoint mocked
4. Close-out + PR
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented 2026-08-22 on `feat/executor-wiring` (branched from dev after PR #2803 merged).

**Approach.** `core/Scheduled_Tasks/automation_executors.py` provides one shared generation-only executor registered for both phase-1 families, calling the server's canonical `perform_chat_api_call_async` (the same surface Flashcards/Research/MCP use — `messages` + optional `api_provider`/`model` + `system_message` + bounded `max_tokens`; never `tools`/`tool_choice`). Content extraction reuses `extract_openai_content` from the Workflows adapters. Credentials stay inside the chat entrypoint's provider-config layer — no new secret handling.

**Target resolution** (`resolve_execution_target`): definition `input` overrides (`provider`, `model`, `max_tokens`) → `[Scheduled_Tasks_Automation]` config defaults (`executor_provider`/`executor_model`/`executor_max_tokens`) → both omitted so the entrypoint applies the server's own default resolution. max_tokens defaults 1000, capped 4000, junk-tolerant.

**Prompt construction:** `recurring_question` → `input.question`; `agent_task` → `input.message` falling back to `input.prompt`; missing input raises LookupError (recorded by the consumer as an honest failed run); a definition-level `input.system_prompt` overrides the fixed generation-only system prompt, which itself forbids questions/tools/side effects.

**Registration** at `run_agent_task_jobs_worker` startup, idempotent (same callables re-registered, not re-created) — a consuming worker always has executors. Empty completions raise (the consumer records `failed`, never a blank success).

**Verification.** 10 tests in `test_automation_executors.py`: the four-case precedence matrix (definition wins / config defaults / both silent / max_tokens cap-floor-junk), prompt construction for both families with message→prompt fallback, missing-prompt LookupError, empty-completion RuntimeError, idempotent registration, and the seam transition through the REAL consumer (`no_executor_configured` failure → registered executor success with the run row carrying the generated summary; LLM mocked at the entrypoint boundary; separate run slots so the terminal-row dedupe doesn't replay the failed outcome). Full Notifications suite **234 passed**. The no-executor guard in the consumer remains the fallback if registration ever fails.

**Env gates unchanged**: enable `SCHEDULED_TASKS_AUTOMATION_SCHEDULER_ENABLED` + `AGENT_TASK_JOBS_WORKER_ENABLED` together to run the wired chain end-to-end.

**Files:** `core/Scheduled_Tasks/automation_executors.py` (new), `services/agent_task_jobs_worker.py` (startup registration), `tests/Notifications/test_automation_executors.py` (new), this task file.
