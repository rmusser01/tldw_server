# Chat Macros Design

Date: 2026-07-03
Status: Approved
Backlog: TASK-12124

## Summary

Build a dedicated `Chat_Macros` module for user-defined slash macros in chat and workspace surfaces. The first bundled macro is `/wrapup`, which fans out focused branch prompts, merges their outputs, and posts a structured final result back to the original thread when a durable target exists.

This is not a one-off `/wrapup` feature. `/wrapup` is the reference macro that proves a reusable macro engine for future custom commands.

## Goals

- Let users define custom chat macros with conservative slash commands.
- Support declarative multi-step macro workflows with prompt, branch, merge, and result posting steps.
- Store user macros as file-backed definitions with registry metadata, following the Skills storage pattern.
- Execute user-visible macro runs through Jobs by default, with small durable job payloads and inspectable run records.
- Support configurable output profiles so a macro can produce one concise response, structured sections, or multiple specified pieces.
- Support chat, chat-workspace, research-workspace, and ACP-aware contexts through bounded context snapshots.
- Ship `/wrapup` as the first built-in macro with general closeout defaults and preset overrides.

## Non-Goals

- Do not add general tool or skill execution in v1.
- Do not replace Chat Workflows.
- Do not let user macros shadow core slash commands.
- Do not require all macro runs to create visible forked conversations.
- Do not store full chat history or raw branch transcripts inside Jobs payloads.
- Do not implement workspace-local macro files in v1, though the schema should leave room for them.

## Current Context

The repository already has several relevant systems:

- `tldw_Server_API/app/core/Chat/command_router.py` provides built-in slash commands with command parsing, rate limits, RBAC hooks, and bounded injection.
- `tldw_Server_API/app/core/Skills/` provides file-backed per-user definitions with registry synchronization, import/export, validation, and path-safety patterns.
- `tldw_Server_API/app/core/Jobs/` provides durable background work, leasing, retries, cancellation, events, and admin visibility.
- `tldw_Server_API/app/core/Chat_Workflows/` provides template and dialogue workflows, but it is step/answer/dialogue-oriented rather than command-triggered macro orchestration.
- ACP sessions already support fork lineage and server-side fork creation through `/api/v1/acp/sessions/{session_id}/fork`.

## Recommended Approach

Create a dedicated `Chat_Macros` module. It owns command macro definitions, macro execution, run records, branch records, context snapshots, output profiles, and `/wrapup` built-in behavior.

The module reuses existing systems rather than embedding itself inside them:

- Use Skills-style file storage for macro definitions.
- Use Jobs for background macro execution.
- Use the Chat orchestrator for LLM calls.
- Use workspace/RAG context helpers for selected sources and retrieval snapshots.
- Use ACP fork APIs only when an ACP session is available and the macro requests fork semantics.
- Use the existing command router as the slash-command entry point, but avoid treating macro output as ordinary command injection.

## Alternatives Considered

### Extend Chat Workflows

Pros:
- Reuses template/run concepts and LLM dialogue orchestration.
- Already has a per-user workflow database and API.

Cons:
- Chat Workflows is optimized for step questions, answer recording, and moderated dialogue.
- Command-triggered macros need fan-out, fold-back, command aliases, output profiles, and posting to original threads.
- This would likely overload the Chat Workflows domain boundary.

### Represent Macros As Skills

Pros:
- Reuses file-backed storage, import/export, and permission vocabulary.

Cons:
- Skills are prompt/tool bundles, not chat-command workflow definitions.
- Mixing macro orchestration into Skills would blur two user concepts and make future permissions harder to reason about.

### Dedicated Chat Macros Module

Pros:
- Clear ownership and security model.
- Can reuse Skills, Jobs, Chat, RAG, and ACP without forcing any one of them to absorb macro-specific behavior.
- Easier to keep `/wrapup` as a built-in macro while supporting future custom commands.

Cons:
- Requires new schemas, services, endpoints, worker code, and UI surfaces.

## Architecture

### Backend Modules

New backend package:

```text
tldw_Server_API/app/core/Chat_Macros/
├── README.md
├── models.py
├── parser.py
├── service.py
├── settings.py
├── context_snapshot.py
├── executor.py
├── branch_runner.py
├── output_profiles.py
├── acp_adapter.py
├── jobs.py
└── builtin/
    └── wrapup/MACRO.yaml
```

New API surface:

```text
tldw_Server_API/app/api/v1/endpoints/chat_macros.py
tldw_Server_API/app/api/v1/schemas/chat_macros.py
```

Expected routes:

- `GET /api/v1/chat/macros`
- `GET /api/v1/chat/macros/{name}`
- `POST /api/v1/chat/macros`
- `PUT /api/v1/chat/macros/{name}`
- `DELETE /api/v1/chat/macros/{name}`
- `POST /api/v1/chat/macros/{name}/clone`
- `POST /api/v1/chat/macros/validate`
- `GET /api/v1/chat/macros/settings`
- `PUT /api/v1/chat/macros/settings`
- `POST /api/v1/chat/macros/run`
- `GET /api/v1/chat/macros/runs/{run_id}`
- `POST /api/v1/chat/macros/runs/{run_id}/cancel`

### Ownership Boundaries

`Chat_Macros` owns:

- Macro definitions and registry metadata.
- Macro settings and output profiles.
- Macro invocation parsing after the command router identifies a macro command.
- Context snapshot assembly.
- Macro run and branch lifecycle.
- Background job enqueueing and worker execution.
- Final output assembly and metadata.

`Chat.command_router` owns:

- Slash command detection.
- Core command precedence.
- Routing macro commands to `Chat_Macros`.

`Jobs` owns:

- Durable background execution, leasing, retries, cancellation, events, and queue controls.

`Skills` remains separate:

- Macro schema can reserve future `permissions.skills`, but v1 rejects non-empty skill permissions.

`Chat_Workflows` remains separate:

- No v1 dependency is required, though concepts such as template/run lifecycle can inform implementation.

## Storage Model

### File-Backed Definitions

Per-user macro definitions live under backend-managed user storage:

```text
Databases/user_databases/<user_id>/macros/<macro_name>/MACRO.yaml
Databases/user_databases/<user_id>/macros/<macro_name>/templates/*
```

`MACRO.yaml` is the canonical editable source for user-owned macros. Supporting files are optional and must be path-safe, UTF-8 where text is expected, bounded in count and size, and kept inside the macro directory.

Builtin macros live in package assets and are immutable. Users can disable a built-in macro or clone it into a user-owned macro, but cannot silently edit the bundled source.

### Registry Metadata

The registry stores derived metadata for listing and drift detection:

- `id`
- `user_id`
- `name`
- `command`
- `description`
- `enabled`
- `source`: `builtin` or `user`
- `builtin_version`
- `schema_version`
- `digest`
- `validation_status`
- `validation_error`
- `updated_at`
- `deleted_at`

The registry is not the canonical macro source. It is a sync/cache layer, following the Skills pattern.

### Macro Settings

Macro settings are separate from each macro definition. They own reusable defaults:

- Default execution mode.
- Sync eligibility thresholds.
- Output profiles.
- Retention defaults.
- Max branches.
- Max concurrency.
- Timeout caps.
- Retry caps.
- Token and estimated-cost caps.
- Default model/provider behavior.
- Default partial-failure policy.

Macros reference output profiles by name and may define local overrides within allowed caps.

### Run Records

`Chat_Macros` needs durable run storage outside Jobs payloads:

- `macro_runs`
  - `run_id`
  - `user_id`
  - `macro_name`
  - `macro_command`
  - `macro_source`
  - `macro_version`
  - `macro_digest`
  - `status`
  - `surface`
  - `conversation_id`
  - `workspace_id`
  - `acp_session_id`
  - `normalized_args`
  - `output_profile`
  - `context_snapshot`
  - `model_selection`
  - `status_message_id`
  - `final_message_id`
  - `error_code`
  - `error_message`
  - `created_at`
  - `started_at`
  - `completed_at`

- `macro_run_branches`
  - `branch_id`
  - `run_id`
  - `step_id`
  - `label`
  - `status`
  - `attempt_count`
  - `prompt_digest`
  - `output_text`
  - `citations`
  - `usage`
  - `acp_child_session_id`
  - `retained`
  - `error_code`
  - `error_message`
  - `created_at`
  - `completed_at`

Jobs payloads should carry only:

- `macro_run_id`
- `user_id`
- `macro_digest`
- `normalized_args`

## Macro Definition Format

Example `MACRO.yaml`:

```yaml
schema_version: 1
name: wrapup
command: wrapup
description: Close out the current conversation or workspace.
enabled: true

args:
  preset:
    type: string
    default: general
  keep_forks:
    type: boolean
    default: false
    aliases: ["keep-forks"]
  mode:
    type: string
    default: background
  output_profile:
    type: string
    default: default
    aliases: ["output-profile"]

context:
  surfaces: [chat, chat-workspace, research-workspace, acp]
  include_chat_history: true
  include_workspace_context: auto
  retrieval: auto
  snapshot_at_dispatch: true

execution:
  mode_default: background
  branch_strategy: auto
  max_branches: 6
  max_concurrency: 3
  timeout_seconds: 180
  retries_per_branch: 1
  merge_retries: 1
  partial_failure: best_effort
  retain_scratch_branches: false

steps:
  - id: summary
    type: branch_prompt
    label: Summary
    output: summary
    prompt: "Summarize the current conversation or workspace context."
  - id: decisions
    type: branch_prompt
    label: Decisions
    output: decisions
    prompt: "Extract decisions and rationale from the current context."
  - id: action_items
    type: branch_prompt
    label: Action items
    output: action_items
    prompt: "Extract action items, owners, due dates, and blockers if present."
  - id: open_questions
    type: branch_prompt
    label: Open questions
    output: open_questions
    prompt: "Identify unresolved questions and suggested follow-up prompts."
  - id: merge
    type: merge
    consumes: [summary, decisions, action_items, open_questions]
    output: final
    prompt: "Combine branch outputs using the selected output profile."
  - id: post
    type: post_result
    consumes: [final]

output_profile: default

permissions:
  tool_calls: []
  skills: []
```

V1 step types:

- `prompt`: one LLM prompt producing one named output.
- `branch_prompt`: fan-out branch prompt producing one named output.
- `merge`: consume declared outputs and produce one merged output.
- `post_result`: post final output to the original target if available.

V1 schema reserves `permissions.tool_calls` and `permissions.skills`, but validation rejects non-empty values unless a future guarded capability enables them.

Branch strategy is explicit at the macro or step level:

- `auto`: use ACP forks only when the surface has a resumable ACP session and the step requests fork semantics; otherwise use chat-native branch calls.
- `chat_native`: always run isolated prompt calls against the context snapshot.
- `acp_fork`: require ACP fork support and fail or follow the macro's configured fallback policy if unavailable.

## Command Semantics

Macro command names use conservative identifiers compatible with the current slash parser. V1 supports word-style commands such as:

- `/wrapup`
- `/dev_handoff`
- `/research_digest`

Hyphenated command names should wait unless the command parser is intentionally widened and covered by regression tests.

Core commands always win. User macros cannot shadow:

- `/time`
- `/weather`
- `/skills`
- `/skill`
- Any future built-in command registered as core.

Bundled macros can be disabled. User clones must use a non-conflicting command.

Command args support normalized aliases:

- YAML field: `keep_forks`
- CLI flag: `--keep-forks`
- Normalized arg key: `keep_forks`

Invalid args fail before a run or job is created.

## Execution Flow

1. The user sends a macro command such as `/wrapup --preset dev_handoff --keep-forks`.
2. The command router parses the slash command and checks core command precedence.
3. If the command resolves to an enabled macro, `Chat_Macros` normalizes args and validates the macro definition, digest, settings, and limits.
4. The service resolves model/provider selection. Branch and merge calls default to current chat model settings unless macro settings or definition provide allowed overrides. If no usable model is available, invocation fails early.
5. The service creates a durable `macro_run` with normalized args, selected output profile, target references, and a bounded context snapshot.
6. Background mode enqueues a Jobs record with only run ID, user ID, macro digest, and normalized args.
7. The original chat receives a status message when a durable target exists. If no durable target exists, the API returns a run ID and run detail URL/history reference without promising automatic append.
8. The worker loads the run, rechecks macro digest, marks the run running, and starts branch steps under `max_concurrency`.
9. Each branch output is normalized into `{step_id, label, status, text, citations, usage}`.
10. Failed branches retry once by default. After retries, remaining failures become failed branch records.
11. Merge consumes declared outputs and applies the selected output profile. Merge retries once by default.
12. Final output is posted to the original target when writable. Otherwise it remains inspectable in run history.
13. Scratch ACP forks are closed or hidden unless `--keep-forks` or settings retain them.
14. Run status, usage, timings, retained fork links, and failure metadata remain available through run detail APIs.

## Context Snapshotting

Context settings such as `retrieval: auto` and `include_workspace_context: auto` are resolved at dispatch time into a bounded snapshot. The snapshot should include:

- Chat/conversation message IDs and a bounded text excerpt or digest.
- Workspace ID and selected source IDs.
- Retrieval result IDs, snippets, scores, and source fingerprints when retrieval is used.
- ACP session ID and forkability metadata when available.
- Model settings and output profile name.
- Token estimates.

Background workers use this snapshot instead of re-resolving mutable workspace state. A future macro option can request fresh context explicitly, but v1 should default to snapshot stability.

Snapshots must not include secrets. Large content should be represented by source IDs, fingerprints, snippets, and bounded excerpts.

## `/wrapup` Built-In Macro

Default `/wrapup` preset:

- Summary.
- Decisions.
- Action items.
- Open questions.

Optional preset examples:

- `general`
- `dev_handoff`
- `research`
- `pr_review`

Default behavior:

- Background execution.
- Temporary scratch branches.
- Retry each failed branch once.
- Best-effort partial failure.
- If all branches fail, post a failure report rather than a synthetic wrapup.
- Structured output using the user’s default output profile.

Supported options:

- `--preset <name>`
- `--question <text>` repeated. Each occurrence creates one custom branch prompt with generated IDs such as `custom_1`, `custom_2`, and display labels such as `Custom 1`. Named/custom-labeled branches are supported through the structured API or macro YAML rather than the v1 slash flag.
- `--output-profile <name>`
- `--keep-forks`
- `--sync` only when below configured sync thresholds.
- `--include-branches` if the selected profile supports branch appendices.

## Output Profiles

Output profiles live in Macro Settings and can be referenced by macros.

Examples:

```yaml
output_profiles:
  default:
    format: structured_sections
    sections:
      - summary
      - decisions
      - action_items
      - open_questions
      - failed_branches
  compact:
    format: single_response
  verbose:
    format: structured_sections
    include_branch_outputs: true
```

Profiles control:

- Single message vs structured sections vs multiple pieces.
- Section order.
- Whether branch outputs are included.
- How failed branches are summarized.
- Whether citations/source references are required where available.

Profiles do not grant extra permissions or bypass execution limits.

## ACP And Chat-Native Branches

If an ACP session is available and resumable, branch steps that request fork semantics may use ACP fork/resume. These branches preserve lineage and can be retained with `--keep-forks`.

If ACP is unavailable, not resumable, or not selected for the surface, branches use chat-native prompt calls against the same context snapshot. This fallback should be explicit in run metadata.

Scratch ACP branches should be closed or hidden after completion unless retained. Even when scratch branches are removed, the macro run keeps minimal branch metadata, output, status, and usage.

## Error Handling

- Unknown macro command: no macro run is created; existing command behavior remains intact.
- Disabled macro: return a short chat-visible error.
- Invalid args: fail immediately, no job.
- Model unavailable: fail immediately, no branches.
- Background enqueue failure: return/post failed status when possible.
- Branch failure: retry per branch up to policy; then mark branch failed.
- All branches failed: final output is a failure report.
- Merge failure: retry merge up to policy; then post or retain a structured failure report with successful branch outputs attached according to retention settings.
- Original thread not writable: keep run inspectable and mark final-post status as failed.
- Job cancelled: stop launching new branches; mark queued branches cancelled; cancel in-flight calls where supported; retain run detail.

Provider errors shown to users should be redacted and safe. Raw provider exception details should stay in logs or admin-only diagnostics according to existing logging policy.

## Security And Guardrails

- Path safety follows Skills: no traversal, no symlink escape, bounded files, UTF-8 text validation, digest checks, and atomic writes.
- Macro command names are conservative and cannot shadow core commands.
- V1 rejects tool and skill permissions by default.
- Context snapshots are bounded and secret-safe.
- Jobs enforce owner/user ID, macro digest, branch count, concurrency, timeout, retry count, token budget, and cancellation.
- Branch and merge prompts run through existing chat moderation and provider readiness checks where applicable.
- ACP branches inherit ACP access control and fork-resumability checks.
- Sync mode is allowed only below strict configured limits to avoid HTTP timeout and streaming conflicts.
- Run detail redacts raw branch transcripts according to retention settings.

## Retention

Default retention:

- Keep final output.
- Keep branch metadata.
- Keep normalized successful branch outputs.
- Drop or redact raw branch transcripts after a short configured period.
- Do not retain scratch ACP branches unless `--keep-forks` or settings request retention.

Retention can be configured in Macro Settings but cannot bypass global privacy/security limits.

## Cost And Model Controls

Macro settings should include:

- Maximum branches.
- Maximum estimated input tokens.
- Maximum estimated output tokens.
- Maximum total estimated tokens per run.
- Optional estimated cost cap when provider pricing is available.
- Default branch model/provider.
- Default merge model/provider.
- Fallback behavior when current chat model is unavailable.

The status message should make branch count and selected profile visible so users understand the scope before or as the background run starts.

## Frontend Design

Add a macro manager/settings surface:

- List macros.
- Enable/disable macros.
- Clone built-ins.
- Create/edit user macros.
- Simple form for common macros.
- Advanced YAML/JSON editor for power users.
- Validation preview.
- Output profile editor.
- Run history and run detail drawer.

Chat messages:

- Invocation creates a status message with metadata when a durable target exists.
- UI renders status metadata as a card.
- Final output appears as normal assistant content with macro metadata.
- Run detail can open from either status or final message.
- Cancellation is available from running status cards and run detail.

The backend stores metadata and Markdown/structured output; the card rendering is a frontend concern.

## Testing Strategy

Backend tests:

- Parser/schema tests for valid and invalid `MACRO.yaml`.
- Arg alias tests for `--keep-forks`, `--output-profile`, and normalized snake_case fields.
- Command collision tests for core command precedence.
- Storage tests for filesystem sync, digest drift, import/export, builtin clone, disable behavior, symlink, traversal, and oversized files.
- Command router tests for macro dispatch and disabled macro behavior.
- Service tests for context snapshotting, output profile resolution, model availability checks, branch fan-out, branch output normalization, retries, partial failure, all-failed behavior, and merge failure.
- Jobs worker tests for minimal payloads, idempotency, cancellation, lease-safe completion/failure, and small payload contracts.
- ACP adapter tests for fork path, scratch cleanup, keep-forks, fork-not-resumable fallback, and retained fork links.
- Security tests for forbidden tool/skill permissions, secret-safe context snapshots, and redacted provider errors.

Frontend tests:

- Macro manager list/editor behavior.
- Builtin clone/disable flows.
- Advanced editor validation feedback.
- Output profile editor.
- `/wrapup` status message rendering.
- Final macro output rendering.
- Run detail drawer.
- Cancellation action.
- Durable-target-missing fallback state.

## Acceptance Criteria

- `Chat_Macros` has a clear design boundary separate from Chat Workflows and Skills.
- Macro definitions are file-backed and registry-synced.
- `/wrapup` is specified as a versioned built-in macro.
- Background execution uses Jobs with minimal payloads and durable run records.
- Context is snapshot at dispatch and bounded.
- Output profiles are user-configurable settings.
- Partial failure, cancellation, retention, ACP fallback, and cost controls are explicit.
- V1 rejects tools/skills while preserving schema room for future guarded permissions.

## Open Questions For Implementation Planning

- Which existing per-user database should own `macro_runs`, or should `Chat_Macros` introduce a dedicated per-user database?
- Should macro final output be persisted through the same chat message persistence path as normal assistant responses or through a macro artifact table plus a chat message reference?
- How much of the Skills service can be reused mechanically for path-safe file storage without coupling the domains?
- Which frontend settings route should host Macro Manager, and should it also be reachable from the command palette?
