# MCP Skills Model-Only Runner Design

## Status

- Task: `TASK-2294.3`
- Parent: `TASK-2294`
- Status: Approved design direction; implementation plan pending requester review
- Date: 2026-07-23

## Context

MCP Unified already exposes `skills.list`, `skills.get`, and `skills.render`
through the read-only `skills` module. Those tools provide user-scoped
discovery and bounded dry rendering without model calls, tool execution,
workflow execution, or supporting-file disclosure.

The existing core `SkillExecutor` is not an acceptable MCP execution boundary.
Its fork path can construct and call the separate core `ToolExecutor`, lists
tools outside the complete MCP Unified policy path, and converts some raw tool
errors into model-visible text. Exposing that path through MCP would create a
second security stack beside the extracted MCP tool-execution pipeline.

The full execution problem also contains several independent risks:

- a synchronous run can perform partial writes before a later approval request;
- retrying after partial execution can duplicate effects;
- Skill command declarations and profile permission rules do not currently
  share one canonical parser and matcher;
- MCP has nested-call reporting but no general nested orchestration depth
  contract;
- current write classification is not a sufficient standalone proof that an
  arbitrary nested call is safe to execute;
- durable approval continuation requires persisted run state and authorization
  revalidation.

This design therefore establishes the smallest executable slice: one bounded
model completion with no tools.

## Decision

Add a separate, disabled-by-default `skills_runner` MCP module that exposes one
tool, `skills.run`.

The existing `skills` module remains enabled by default, read-only, and
behaviorally unchanged. The new module has its own health, concurrency,
timeout, circuit-breaker, configuration, risk-tier, and operator opt-in
boundary.

`skills.run` accepts only a verified fork-mode Skill that:

- is visible to the authenticated user;
- permits model invocation;
- declares no tools;
- has no supporting files.

It renders the Skill from one immutable snapshot, closes the Skills database,
and performs exactly one tool-disabled model completion through a narrow
host-provided model invocation port.

## Goals

- Establish an auditable MCP execution boundary without nested tool execution.
- Keep model failures and concurrency isolated from read-only Skill discovery.
- Preserve user isolation, context-integrity checks, and Skill visibility.
- Reuse normal MCP tool and `Skill(skill_name)` authorization.
- Use server-managed provider credentials and existing outbound-egress policy.
- Bound input, prompt, provider output, elapsed time, and cancellation.
- Preserve normal MCP hooks, rate limits, idempotency, audit, metrics, and
  tool-use reporting.
- Return stable, bounded, sanitized results and failures.
- Create a foundation that later nested-execution tasks can extend without
  changing this public contract.

## Non-Goals

This task does not:

- execute any MCP, external MCP, core ToolExecutor, shell, file, browser,
  workflow, Job, Scheduler, or subagent tool;
- support inline-mode Skills;
- read or transmit Skill supporting files;
- accept client-selected providers, endpoints, API keys, credentials, or
  arbitrary model overrides;
- perform a multi-turn model loop;
- support streaming responses;
- persist run transcripts or rendered prompts;
- provide approval continuation or resumable run tokens;
- promise exactly-once execution without a caller-supplied MCP idempotency key;
- change the REST Skills execution endpoint or core `SkillExecutor`;
- change `skills.list`, `skills.get`, `skills.render`, or their schemas;
- define canonical allowed-tool declaration semantics.

Read-only nested MCP execution is deferred to `TASK-2294.4`. Durable effectful
nested execution is deferred to `TASK-2294.5`.

## Why A Separate Module

MCP module risk tiers are registered by module ID. Adding an optional effectful
tool to the existing `skills` module would leave `/api/v1/mcp/status`
describing that module as read-only even when execution was enabled.

A separate module provides the following properties:

- `skills` remains accurately classified as read-only;
- `skills_runner` can be classified as model execution;
- disabled capability appears as an explicit operator opt-in;
- model provider failures do not open the catalog/render circuit breaker;
- model concurrency cannot consume all read-only Skills capacity;
- enabling execution is a visible configuration change;
- tests can assert an exact one-tool execution surface.

## Module And Risk Surface

Add `skills_runner` to the module configuration with:

- `enabled: false`;
- a conservative concurrency limit, initially `2`;
- a module timeout no greater than the hard run timeout;
- settings for bounded prompt and output limits;
- a description that states Skill content and arguments may be sent to the
  configured model provider.

Add a `model_execution` module risk tier:

- label: `Configured model execution`;
- meaning: invokes a configured model and may incur provider cost or outbound
  data transfer, but does not by itself authorize tools;
- explicit opt-in: true;
- common module: `skills_runner`.

When disabled, `skills_runner` appears in `surface.disabled_available` with
operator guidance. It does not contribute `skills.run` to `tools/list`.

No feature flag is added inside the existing `skills` module. Module enablement
is the single capability switch.

## Public Tool Contract

### Tool

Name: `skills.run`

Description requirements:

- state that the tool executes one fork-mode Skill through one configured model
  completion;
- state that Skill content and supplied arguments may be sent to the configured
  provider;
- state that no tools or supporting files are available;
- avoid implying durable, resumable, or exactly-once execution.

Metadata:

- category: `model_execution`;
- `readOnlyHint: false`;
- explicit write/effect classification through
  `SkillsRunnerModule.is_write_tool_call()`.

### Input

```json
{
  "skill_name": "review-paper",
  "arguments": "focus on methodology"
}
```

Schema:

- `skill_name`: required string, canonical Skill-name pattern, maximum 64
  characters;
- `arguments`: optional string, default empty, maximum 10,000 characters and
  40,000 UTF-8 bytes;
- `additionalProperties: false`.

The request cannot contain provider, endpoint, API key, credential, headers,
model, tools, timeout, system prompt, messages, or supporting-file fields.

### Success

```json
{
  "status": "completed",
  "skill_name": "review-paper",
  "skill_version": 4,
  "execution_mode": "fork",
  "output": "Bounded model response",
  "model_source": "skill",
  "tools_used": 0,
  "supporting_files_used": false,
  "dry_run": false
}
```

Contract rules:

- `output` is normalized text from the host model port;
- `skill_version` identifies the immutable snapshot used for the run;
- `model_source` is `skill` when the authored Skill selected a model and
  `server_default` otherwise;
- no provider credential, endpoint, raw provider envelope, rendered prompt,
  Skill content, arguments, supporting-file metadata, absolute path, or raw
  exception text is returned;
- oversized output is rejected atomically rather than truncated;
- the response does not claim that the call is durable or resumable.

Provider and exact model identifiers are not returned in the first slice. They
remain available to existing protected provider observability where already
recorded.

## Eligibility And Snapshot Contract

The runner obtains trusted `user_id` and the per-user ChaCha database path from
the MCP `RequestContext`. It must not derive a path from request arguments.

The loader performs the existing model-visible and context-integrity checks.
Unknown, deleted, cross-user, hidden, integrity-blocked, or
model-invocation-disabled Skills all map to `skill_not_found` so the runner does
not create a visibility oracle.

For a visible Skill, construct one immutable `SkillRunSnapshot` containing only:

- canonical name;
- version;
- instruction content;
- normalized execution mode;
- normalized declared-tool list;
- authored model selector, if present;
- a boolean indicating supporting-file presence.

Before any model invocation:

1. Require execution mode `fork`.
2. Require the parsed declaration field to be a valid empty list after the
   parser's documented default handling. Any malformed, unrecognized, or
   non-empty declaration state is ineligible; it must not normalize into an
   executable no-tools Skill.
3. Require the supporting-file field to be absent or a valid empty list. Any
   malformed, unrecognized, or non-empty state is ineligible.
4. Render arguments with the existing substitution semantics.
5. Require instruction content to be a string and the authored model selector
   to be absent or a bounded string accepted by the model port. Reject control
   characters and URL-shaped model selectors.
6. Enforce the rendered-prompt ceiling.
7. Close all request-scoped database connections.

The model port cannot be called until successful database closure. A database
close failure after snapshot loading fails the request as `skills_unavailable`;
it does not continue with the snapshot.

The snapshot is not reloaded during the run. A concurrent Skill update affects
later calls, not the in-flight call.

## Authorization Contract

Normal `tools/call` processing remains authoritative.

The authorization composition is:

1. coarse MCP method authorization;
2. `skills.run` tool authorization and API-key scope checks;
3. context allow-list and effective profile policy;
4. schema and module argument validation;
5. effect/write checks;
6. existing permission-subject extraction for `skill_name`;
7. `Skill(skill_name)` permission decision and approval lease evaluation;
8. governance preflight and pre-tool hooks;
9. module execution;
10. audit, metrics, post-tool hooks, and tool-use reporting.

`skills.run` is always classified as effectful even though this slice performs
no data mutation. A model call can incur cost and external data transfer, so it
must not inherit read-only API-key or write-disable permissions.

The two authorization dimensions have distinct purposes:

- tool authorization controls whether this caller may perform model execution
  through `skills.run`;
- `Skill(skill_name)` controls access to the selected Skill identity.

An existing Skill approval lease cannot authorize execution by itself because
`skills.run` must also be independently allowed. This task does not introduce a
new permission subject family.

The standard MCP idempotency key is honored. With a supplied key, a successful
effectful call may be replayed from the existing owner-scoped cache. Without a
key, duplicate model invocation remains possible and is not represented as
exactly once.

## Model Invocation Port

The runner depends on a narrow host-neutral port, not on `MCPProtocol`,
`ToolExecutionCoordinator`, the REST endpoint, or core `ToolExecutor`.

Define normalized request and result types in a small MCP module-runtime port
module:

```python
@dataclass(frozen=True, slots=True)
class ModelCompletionRequest:
    system_prompt: str
    user_prompt: str
    authored_model: str | None
    max_output_chars: int
    max_output_bytes: int
    max_provider_response_bytes: int


@dataclass(frozen=True, slots=True)
class ModelCompletionResult:
    content: str
    model_source: Literal["skill", "server_default"]
```

The callable receives the normalized request and trusted MCP
`RequestContext`. It returns only normalized output, not a raw provider
response.

The server composition root supplies the production adapter. The adapter:

- resolves the authenticated user's effective provider credential context
  through the shared provider runtime;
- selects the configured provider;
- treats the Skill-authored model as a model selector only, never as an
  endpoint or credential selector;
- validates the model selector as an opaque identifier of at most 256
  characters, rejects control characters and URL-shaped values, and verifies
  provider/model readiness;
- derives credential scope only from trusted authenticated-principal fields in
  the request context and ignores metadata-supplied keys, endpoints, or
  credential material;
- uses existing provider outbound-egress enforcement;
- uses existing provider usage accounting, token/cost quotas, and concurrency
  controls;
- performs one non-streaming completion with `tools=None`;
- enforces the provider response-body limit before retaining or normalizing an
  oversized envelope;
- rejects provider responses containing tool calls;
- normalizes provider-specific output into `ModelCompletionResult`;
- maps provider errors into internal typed failures without raw error text.

The module must fail closed during initialization or health checking when the
port is absent. No fallback may instantiate `SkillExecutor`, `ToolExecutor`, or
call a provider directly.

## Prompt And Completion Flow

The runner performs exactly one logical completion and one model turn. It adds
no module-level retry. Existing provider transport retries may still occur
inside the production adapter under the shared provider retry policy; those
retries must remain within the same timeout, credential, quota, and
observability context.

System prompt:

```text
You are executing the Skill "<canonical skill name>".
Follow the Skill instructions and return the requested result.
No tools or supporting files are available.

<rendered Skill instructions>
```

User prompt:

```text
Execute the Skill instructions.
```

The authored Skill content remains authoritative within this isolated
completion. The runner does not add follow-up messages, parse model-generated
tool arguments, or execute tool calls.

If the provider returns a tool call despite tools being disabled, fail with
`skill_run_unexpected_tool_call`. Do not expose the attempted tool name or
arguments and do not retry automatically.

## Resource Limits

All settings reject booleans and non-integer coercion. Values are clamped to
hard ceilings at module initialization.

Initial limits:

- arguments: 10,000 characters and 40,000 UTF-8 bytes;
- rendered prompt: default 32,000 characters and 128,000 UTF-8 bytes, with
  hard maximums of 100,000 characters and 400,000 bytes;
- normalized output: default 20,000 characters and 80,000 UTF-8 bytes, with
  hard maximums of 100,000 characters and 400,000 bytes;
- provider response envelope: default 1,000,000 bytes and hard maximum
  2,000,000 bytes;
- effective wall timeout: the minimum of module timeout, configured run
  timeout, and a 120-second hard maximum;
- completions per call: exactly 1;
- tool calls per call: 0;
- supporting files per call: 0;
- module concurrency: default 2.

Argument, prompt, provider-envelope, and normalized-output overflows fail
atomically. The public result never contains a partial or truncated completion.

The provider adapter requests a provider-native token/output cap when
supported. Character and UTF-8 byte limits remain authoritative after
normalization.

## Cancellation And Timeout

Model invocation runs in one retained child task.

On caller cancellation:

1. cancel the child task;
2. drain it so provider cleanup and connection release can complete;
3. re-raise `asyncio.CancelledError`;
4. emit no success result or success event.

On timeout:

1. cancel and drain the child task;
2. raise the stable `skill_run_timeout` failure;
3. do not return provider content received after the deadline.

The production model port must propagate cancellation and cancel its underlying
HTTP operation. Broad exception handlers must not convert cancellation into a
generic failure.

Database cleanup uses the existing retained-lifecycle pattern and completes
before the model child task is created.

## Error Contract

Public failures use stable reason codes and sanitized messages:

| Condition | Public reason |
|---|---|
| Unknown, hidden, blocked, deleted, cross-user, or model-disabled Skill | `skill_not_found` |
| Inline-mode Skill | `skill_run_requires_fork` |
| Skill declares any tools | `skill_run_tools_unsupported` |
| Skill has supporting files | `skill_run_supporting_files_unsupported` |
| Rendered prompt exceeds limit | `skill_run_prompt_too_large` |
| Model port or provider is unavailable | `skill_run_provider_unavailable` |
| Provider egress is denied | `skill_run_egress_denied` |
| Provider usage quota is denied | `skill_run_quota_exceeded` |
| Provider response envelope exceeds limit | `skill_run_provider_response_too_large` |
| Provider returned a tool call | `skill_run_unexpected_tool_call` |
| Normalized output exceeds limit | `skill_run_output_too_large` |
| Wall timeout | `skill_run_timeout` |
| Database cleanup or unexpected internal failure | `skills_unavailable` |

Existing protocol errors remain authoritative for invalid parameters,
permission denial, approval required, governance denial, rate limit, and
idempotency conflicts.

Warnings and errors may log:

- operation;
- component;
- safe reason code;
- exception class;
- bounded filename/function/line traceback metadata;
- canonical Skill name only after visibility authorization;
- Skill version;
- request ID.

Logs must not include Skill content, rendered prompt, arguments, model output,
raw provider error text, credentials, headers, endpoints, database paths, or
source lines.

## Observability

The normal MCP tool-use event is the authoritative parent execution record.
`skills.run` must participate in:

- module and per-tool metrics;
- effectful rate-limit selection;
- owner-scoped idempotency;
- pre-tool and post-tool hooks;
- audit success/failure classification;
- tool-use success, denial, failure, timeout, cancellation, and replay
  reporting where the existing protocol records those outcomes.

The model adapter continues existing provider usage and credential-resolution
observability. The Skills runner does not create a second provider telemetry
format.

Internal safe event metadata may include Skill version, model source, elapsed
duration, and zero tool/supporting-file counts. Public result fields remain the
exact success contract above. Neither event metadata nor MCP audit metadata
records prompt or output content.

## Component Boundaries

`SkillsRunnerModule`

- owns the public tool schema, exact argument validation, write
  classification, request dispatch, public result shape, and stable error
  mapping;
- depends on a snapshot loader and model completion port;
- does not import `MCPProtocol`, core `SkillExecutor`, core `ToolExecutor`, or a
  provider adapter.

`SkillRunSnapshotLoader`

- owns trusted user/path extraction, request-scoped database construction,
  model-visible verified Skill loading, immutable snapshot creation, and
  database cleanup;
- performs no model or tool work;
- exposes no database object after return.

`ModelCompletionPort`

- owns the host-neutral normalized request/result contract;
- contains no credential fields.

Production model adapter

- lives at the server composition boundary;
- owns effective provider credential resolution, configured-provider
  selection, egress enforcement, one completion, response normalization, and
  provider error classification.

`module_surface.py`

- owns the new model-execution risk tier and operator-facing disabled-module
  guidance.

The existing `skills_module.py` remains behaviorally unchanged.

## Configuration

Default configuration:

```yaml
- id: skills_runner
  class: tldw_Server_API.app.core.MCP_unified.modules.implementations.skills_runner_module:SkillsRunnerModule
  enabled: false
  name: Skills Runner
  version: "0.1.0"
  department: agentic_execution
  timeout_seconds: 60
  max_concurrent: 2
  settings:
    max_rendered_skill_chars: 32000
    max_rendered_skill_bytes: 128000
    max_output_chars: 20000
    max_output_bytes: 80000
    max_provider_response_bytes: 1000000
    run_timeout_seconds: 60
```

No credential or endpoint is accepted in this module entry. Provider
configuration remains owned by the existing provider and credential systems.

## Testing Strategy

Implementation follows TDD with fakes at the new ports.

### Surface And Configuration

- default config keeps `skills_runner` disabled;
- disabled status reports the `model_execution` tier and explicit opt-in;
- enabling registers exactly `skills.run`;
- existing `skills` remains in `read_only`;
- existing Skills tool catalog remains exactly list/get/render;
- runner initialization/health fails closed without a model port;
- schema rejects unknown keys, wrong types, booleans as integers, oversized
  names, and oversized arguments.

### Eligibility And Isolation

- visible fork Skill with no tools/files produces one snapshot;
- inline, declared-tool, and supporting-file Skills fail before model call;
- hidden/integrity-blocked/deleted/cross-user/model-disabled cases all return
  `skill_not_found`;
- user database path comes only from trusted context;
- database closes before model invocation;
- close failure prevents model invocation;
- concurrent Skill mutation does not alter the captured version/content.

### Authorization And Effect Classification

- coarse method denial occurs before module execution;
- tool denial and API-key read-only scope reject `skills.run`;
- `Skill(name)` deny rejects;
- `Skill(name)` ask requires the existing approval lease;
- tool allow without Skill allow rejects;
- Skill allow without `skills.run` tool allow rejects;
- `is_write_tool_call()` returns true for `skills.run`;
- write-disable policy rejects before model invocation;
- governance and pre-hook denial prevent model invocation;
- successful post-hooks and tool-use reporting preserve existing behavior.

### Model Port

- receives rendered prompt, fixed user prompt, authored model selector, trusted
  context, and no tools or credentials;
- is called exactly once;
- Skill-authored model is tagged `skill`; missing model is tagged
  `server_default`;
- client cannot override provider, model, endpoint, or key;
- tool-call response fails closed;
- malformed/non-string response fails closed;
- provider unavailable and egress-denied failures map to stable reasons;
- provider usage accounting and quota denial remain authoritative;
- oversized provider envelopes fail before normalization and retention;
- raw provider errors are absent from responses and captured logs.

### Bounds, Cancellation, And Idempotency

- prompt character/byte boundaries succeed at their limits and fail above
  either limit before provider call;
- provider envelope and normalized output character/byte boundaries fail
  atomically above their limits;
- timeout cancels and drains the child task;
- caller cancellation propagates and emits no success;
- standard idempotency replay does not invoke the model a second time;
- a different owner cannot replay another owner's key;
- no-key repeated calls are not claimed as exactly once.

### Regression And Boundaries

- existing Skills module and service suites pass;
- existing protocol authorization, hooks, idempotency, reporting, and module
  surface suites pass;
- import-boundary tests reject imports of `MCPProtocol`, `SkillExecutor`,
  `ToolExecutor`, and provider adapters from the runner module;
- focused integration invokes `skills.run` through public `tools/call` with a
  fake model port;
- Ruff, compile checks, Bandit, `git diff --check`, and package-boundary checks
  pass for touched scope.

## Expected Implementation Scope

The implementation plan may touch:

- `tldw_Server_API/app/core/MCP_unified/modules/base.py` for one explicit,
  typed `model_completion_handler` field on `ModuleConfig`, parallel to the
  existing tool catalog handler;
- a small module-runtime port definitions file;
- a new Skill snapshot loader;
- a new `skills_runner_module.py`;
- `tldw_Server_API/app/core/MCP_unified/server.py` composition wiring;
- `tldw_Server_API/app/core/MCP_unified/module_surface.py`;
- `tldw_Server_API/Config_Files/mcp_modules.yaml`;
- focused MCP Skills runner, server registration, authorization, surface,
  lifecycle, and boundary tests;
- `Docs/MCP/Unified/Modules.md`;
- the task record and implementation plan.

The callable is injected by the server composition root. It must not be placed
in YAML settings or resolved through a service locator.

## Follow-Up Tasks

`TASK-2294.4` will design canonical allowed-tool declaration compilation and
read-only nested MCP execution. It must preserve the caller's authorization
ceiling, add recursion limits and parent-child reporting, and reject calls not
proven read-only.

`TASK-2294.5` will design durable effectful nested execution. It must address
approval continuation, authorization revalidation, retry-safe idempotency,
partial progress, cancellation, quotas, and operator-visible run state. It
uses Jobs by default. Any alternative durable runtime requires separate design
approval with evidence that it preserves equivalent authorization, operations,
and recovery properties. It must never serialize provider credentials.

## Acceptance Criteria

1. A separate `skills_runner` module exposes only `skills.run`, is disabled by
   default, and has an explicit model-execution risk tier; existing read-only
   Skills tools remain unchanged.
2. Only user-scoped, model-visible, integrity-verified fork-mode Skills with no
   declared tools or supporting files are eligible, and all unsupported cases
   fail before model invocation.
3. One immutable Skill snapshot is loaded and the request-scoped database is
   closed before model/provider work.
4. Normal tool and `Skill(skill_name)` authorization applies, and `skills.run`
   is effectful for scope, policy, rate limit, idempotency, hook, audit,
   metrics, and reporting behavior.
5. Model execution uses one injected normalized completion port with
   server-managed credentials and existing egress enforcement; client provider,
   endpoint, credential, and model overrides are impossible.
6. Exactly one logical tool-disabled completion and one model turn occur under
   explicit input, prompt, provider-envelope, output, timeout, and cancellation
   bounds; the module performs no retry.
7. Results and errors are stable and sanitized, with no prompt, raw provider
   payload, credential, endpoint, path, or raw exception leakage.
8. Focused tests and documentation cover the execution contract, and nested
   read-only and effectful execution remain isolated in `TASK-2294.4` and
   `TASK-2294.5`.
