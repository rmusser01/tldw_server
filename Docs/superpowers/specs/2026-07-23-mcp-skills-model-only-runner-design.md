# MCP Skills Model-Only Runner Design

## Status

- Task: `TASK-2294.3`
- Parent: `TASK-2294`
- Status: Security-review revisions applied; requester approval pending
- Date: 2026-07-25

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
- has no supporting files or extra directory entries;
- does not declare an authored model selector.

It renders the Skill from one immutable snapshot, closes the Skills database,
and, on a successful dispatch path, performs exactly one tool-disabled model
completion through a narrow host-provided model invocation port.

## Goals

- Establish an auditable MCP execution boundary without nested tool execution.
- Keep model failures and concurrency isolated from read-only Skill discovery.
- Preserve user isolation, context-integrity checks, and Skill visibility.
- Reuse normal MCP tool and `Skill(skill_name)` authorization.
- Use one operator-configured provider/model pair, server-managed credentials,
  and existing outbound-egress policy.
- Bound input, prompt, provider output, elapsed time, and cancellation.
- Preserve normal MCP hooks, effect classification, rate limits, idempotency,
  audit, metrics, and tool-use reporting.
- Return stable, bounded, sanitized results and failures.
- Ensure expected domain failures do not count as circuit-breaker failures or
  appear as successful tool-use events.
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
- support streaming responses or automatic provider fallback;
- create a transcript or rendered-prompt store; the existing owner-scoped MCP
  idempotency cache remains an explicitly documented bounded exception for a
  successful normalized response when the caller supplies an idempotency key;
- provide approval continuation or resumable run tokens;
- promise exactly-once execution without a caller-supplied MCP idempotency key;
- change the REST Skills execution endpoint or core `SkillExecutor`;
- change `skills.list`, `skills.get`, `skills.render`, or their schemas;
- define canonical allowed-tool declaration semantics;
- support Skill-authored model selection; this is deferred until an explicit
  operator allow-list contract is designed.

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
- a module timeout strictly greater than the run timeout plus its bounded
  cancellation-cleanup allowance;
- `max_retries: 0` so later generic retry behavior cannot silently change this
  module's execution contract;
- one non-secret operator-configured provider/model pair;
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
- `write_capable: true` so `tools/list` authorization agrees with call-time
  effect classification;
- `uses_network: true` so the existing conservative network rate-limit bucket
  applies;
- `rate_limit_fail_closed: true` so an unavailable rate-limit admission path
  cannot silently permit a billable model call;
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
  "tools_used": 0,
  "supporting_files_used": false,
  "dry_run": false
}
```

Contract rules:

- `output` is normalized text from the host model port;
- `skill_version` identifies the immutable snapshot used for the run;
- no provider credential, endpoint, raw provider envelope, rendered prompt,
  Skill content, arguments, supporting-file metadata, absolute path, or raw
  exception text is returned;
- oversized output is rejected atomically rather than truncated;
- the response does not claim that the call is durable or resumable.

Provider and model identifiers are not returned. They remain available only to
existing protected provider observability where already recorded.

## Eligibility And Snapshot Contract

The runner obtains trusted `user_id` and the per-user ChaCha database path from
the explicit fields populated on MCP `RequestContext` by the server. It must not
derive either value from request arguments or `RequestContext.metadata`. The
user ID must normalize to a positive integer before any database access.

Add one Skills-service operation dedicated to execution snapshots. Under the
existing per-user Skills mutation lock, it performs a targeted bounded registry
reconciliation for only the requested canonical name and one
model-visible/context-integrity decision. It must not call the catalog-wide
registry synchronization path, scan or mutate unrelated Skill rows, or compose
separate metadata and content reads.

The targeted reconciliation uses the same captured `SKILL.md` bytes and strict
parse result that form the snapshot. It may insert, restore, update, or tombstone
only the selected registry row with optimistic version checks, and it never
trusts a stored directory path over the expected owner-scoped path. An
integrity-blocked or structurally invalid file is not newly indexed. After any
row mutation, filesystem identity/inventory and the selected registry row are
revalidated before the immutable snapshot can escape the lock.

The execution snapshot path requires a non-null Context Integrity resolver and
a positive digest-allowed decision for the exact purpose `skill_execution`.
The read-only Skills module may retain its existing behavior when the global
resolver is absent, but the effectful runner fails closed as
`skills_unavailable` and does not dispatch a model call.

Unknown, deleted, cross-user, hidden, integrity-blocked, or
model-invocation-disabled Skills all map to `skill_not_found` so the runner does
not create a visibility oracle.

Public failure precedence is fixed. A missing directory/main file, deleted or
cross-user row, integrity denial, malformed/oversize/control-bearing main file,
or any state in which exact current visibility cannot be established returns
`skill_not_found`. A structurally valid current file with
`user-invocable: false` or model invocation disabled also returns
`skill_not_found`. Only after current visibility is proven may eligibility
failures be exposed, in this order: non-fork context, declared tools, extra
directory entries, authored model selector, then rendered-prompt overflow.
Shared dependency unavailability and a concurrent snapshot mutation retain
`skills_unavailable` as specified below.

When the initial inventory contains extra entries, the loader may still read the
bounded regular `SKILL.md` to establish current visibility and deterministic
failure precedence, but it never opens an extra entry. It performs no model call
and does not compute an incomplete canonical digest from a subset of the bundle.

For a visible Skill, construct one immutable `SkillRunSnapshot` containing only:

- canonical name;
- version;
- instruction content;
- normalized execution mode;
- normalized declared-tool list;
- a boolean indicating that the directory inventory contained only `SKILL.md`.

The snapshot loader uses a strict execution parse mode. Existing permissive
frontmatter coercions are not an execution authorization boundary. In strict
mode:

- frontmatter must be a YAML mapping whose complete key allow-list is `name`,
  `description`, `argument-hint`/`argument_hint`,
  `disable-model-invocation`/`disable_model_invocation`,
  `user-invocable`/`user_invocable`, `allowed-tools`/`allowed_tools`, `model`,
  and `context`;
- `name` must be absent, null, or a canonical Skill-name string equal to the
  captured registry and directory name;
- `description` and `argument-hint`/`argument_hint` must be absent, null, or
  strings;
- `context` is required and must be the exact string `fork`;
- `disable-model-invocation`/`disable_model_invocation` must be absent or the
  exact boolean `false`;
- `user-invocable`/`user_invocable` must be absent or the exact boolean `true`;
- `allowed-tools`/`allowed_tools` must be absent, null, or an exact list of
  strings; strings, mappings, scalars, and non-string members are structurally
  invalid, while a non-empty well-typed list is ineligible as
  `skill_run_tools_unsupported`;
- `model` must be absent or null; every non-null value is rejected;
- duplicate YAML keys and simultaneous use of both aliases for any recognized
  field are rejected before normalization;
- unknown frontmatter keys and every type coercion are rejected;
- YAML anchors, aliases, merge keys, explicit tags, and non-core node types are
  rejected; parsing is bounded to 16,384 frontmatter UTF-8 bytes, 64 nodes, and
  depth 4 before object construction;
- every accepted frontmatter string must be strictly UTF-8 encodable and contain
  no prohibited C0/C1 control other than tab, carriage return, or line feed;
- instruction content must decode as valid UTF-8, be a string, and contain no
  prohibited C0/C1 control character other than tab, carriage return, or line
  feed.

Before reading prompt-bearing bytes, the loader performs a no-follow directory
inventory. The root must be a real directory and its exact entry set must be one
regular file named `SKILL.md`. Symlinks, subdirectories, special files, hidden
files, and any other entry fail as supporting files without reading their
contents.

The loader holds an identity-checked directory handle and performs inventory,
open, and stat operations relative to that handle where the platform supports
it. If the platform cannot provide equivalent no-follow, stable-identity
guarantees, the runner is unavailable; there is no permissive path-based
execution fallback.

The loader then opens `SKILL.md` without following links and reads at most the
existing 500,000-byte Skill limit plus one byte before rejecting oversize input.
It does not allocate or decode the unbounded file. It verifies all of the
following before returning:

- file identity and metadata are stable across open/read;
- the post-read directory inventory matches the pre-read inventory;
- the registry row identity and version still match the captured row;
- the SHA-256 hash of the raw `SKILL.md` text matches the captured registry
  `file_hash`;
- the canonical filesystem digest is recomputed from exactly the captured
  `SKILL.md` bytes, canonical Skill asset ID, and `skill_name` metadata using
  the same helper as startup inventory, then accepted by
  `ContextIntegrityResolver.require_digest_allowed()` with
  `purpose="skill_execution"`;
- the strict parsed name and captured registry name agree.

A concurrent mutation or identity mismatch fails closed before model invocation.
It is not retried inside the runner.

Before any model invocation:

1. Require execution mode `fork`.
2. Require the strict declared-tool list to be empty.
3. Require the single-file inventory assertion.
4. Require the authored model selector to be absent.
5. Render arguments with the existing substitution semantics.
6. Build the fixed system/user messages and enforce final combined prompt
   character and UTF-8 byte ceilings, including boilerplate.
7. Close all request-scoped database connections.

The model port cannot be called until successful database closure. A database
close failure after snapshot loading fails the request as `skills_unavailable`;
it does not continue with the snapshot.

The snapshot is not reloaded during the run. A mutation completed after snapshot
verification affects later calls, not the in-flight call.

The module overrides the inherited SQL-oriented input sanitizer. It validates
without mutating text: tabs and line endings are preserved, SQL-like substrings
are allowed, strict UTF-8 encodability is required, and C0/C1 control characters
other than tab, carriage return, and line feed are rejected rather than silently
stripped.

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
effectful call may be replayed from the existing owner-scoped cache. This cache
contains the complete normalized success payload, including model output, for
the operator-configured MCP idempotency TTL and may use Redis. It does not store
separate request, rendered-prompt, Skill, argument, provider-envelope, or
credential fields. The model output itself may quote or reproduce sensitive
input, so operators must treat the entire cached response as sensitive. Expected
failures are not cached. Without a key, duplicate model invocation remains
possible and is not represented as exactly once.

The existing idempotency argument binding separately retains an owner-scoped
argument fingerprint for the same TTL, including after a failed attempt. It does
not retain plaintext arguments. Reusing the same key and arguments may execute
again after a failure; reusing the key with different arguments remains a
conflict.

The owner key is also bound to a canonical authorization/billing-scope
fingerprint derived only from explicit server-authenticated identity fields.
The default personal/no-active-scope case preserves the legacy owner-key shape;
when an active team or organization is present, the key adds a fixed-format
SHA-256 scope digest, never raw scope IDs. The scope fingerprint is included in
the prepared-call HMAC and cache-key relationship check. The same user, public
arguments, and idempotency key under a different active team or organization is
a different replay domain and cannot receive the first scope's output. Generic
request metadata, headers, session/client IDs, and Skill/provider content cannot
influence this dimension.

Idempotency binds the public request arguments, not mutable Skill or provider
state. A replay after the Skill, provider, or model configuration changes returns
the cached original result and its original `skill_version` until TTL expiry. A
new idempotency key is required to execute the newer state. Documentation and
tests must make this stale-by-design replay behavior explicit.

Idempotency lock contention occurs outside the current module timeout. The
runner therefore supplies a bounded contention-wait policy to the generic
idempotency manager: default 5 seconds, hard maximum 30 seconds. The bound
applies only while waiting to become the execution owner in local and Redis
paths; it does not cancel an execution after ownership is acquired. If the bound
expires before a cached result appears, return a stable protocol-level
`idempotency_in_progress` conflict, perform no model dispatch, and leave module
breaker state unchanged.

The idempotency manager invokes an owner execution callback at most once per
request. Backend fallback is allowed only before ownership and only when the
manager can prove that remote lock acquisition did not succeed. An ambiguous
Redis lock-acquisition failure returns breaker-neutral
`idempotency_unavailable` without dispatch; it must not fall through to local
execution.

Once the owner callback starts, no Redis/cache/serialization/release failure may
invoke it again or transition to a second backend execution path. If the callback
returns a valid success but Redis result persistence fails, retain that payload
in the bounded local cache when possible, emit degraded-persistence metrics and
safe logs, release the remote lock best-effort, and return the original success.
The public result is unchanged. This remains best-effort process-local replay,
not a durable exactly-once claim. Cancellation or failure before a valid result
exists remains uncached under the existing ambiguous-outcome rules.

Successful cache payloads are serialized once to strict canonical JSON with
UTF-8, `ensure_ascii=False`, `allow_nan=False`, stable key ordering, compact
separators, and no permissive `default=str` conversion. The serialized bytes are
bounded by the call policy before either backend stores them. Redis stores those
exact bytes; the local cache stores an equivalent decoded copy and returns a
fresh copy on replay so caller mutation cannot alter cached state. Serialization
or size failure after a valid module result is degraded persistence, not a tool
failure or reason to redispatch.

For an idempotent call, response construction and success-cache persistence
occur immediately after the module returns and before cancellation-sensitive
post-execution observers. Post-hook, audit, metrics, reporting, or degraded-cache
observability must preserve the committed module outcome and must never cause
same-request redispatch. A cancellation after module success may still propagate,
but the successfully persisted result remains replayable.

The transition from a valid callback result to a committed idempotency outcome
is cancellation-shielded and bounded by `idempotency_finalize_seconds` (default
5, hard maximum 15). Strict serialization and the fresh bounded local replay
copy are completed synchronously before any remote await. Redis result storage
and lock release then run under the bounded shield. Cancellation is re-raised
only after that finalization finishes or reaches its bound. A remote timeout or
failure retains the process-local committed copy, records degraded persistence,
and never invokes the callback again; cross-process durability is not claimed.
Remote finalization runs in an idempotency-manager-owned task. At the bound it
receives cancellation and a bounded drain attempt; a backend operation that does
not terminate remains owned until completion, marks remote idempotency health
degraded, and is drained on manager shutdown. Any late write may publish only
the already committed exact bytes, and lock release remains ownership-token
checked. No finalization task is left unreferenced and no late outcome can
replace the caller result.

This bounded replay cache is the only runner-owned output retention in this
slice. Documentation must disclose it and must not describe the runner as
strictly non-persistent when idempotency is enabled.

## Generic Expected Tool Failure Contract

The runner must not add Skills-specific exception branches to `protocol.py` and
must not return a normal success dictionary for a failed execution. Before the
runner is added, introduce one host-neutral expected tool-failure outcome in the
extracted MCP execution layer.

As part of that host-neutral prerequisite, move `IdempotencyManager` and its
Redis/local state machine out of `protocol.py` into the extracted tool-execution
package before changing its behavior. `protocol.py` may retain a compatibility
re-export for existing imports, but no idempotency implementation or
Skills/model-specific branch. This keeps the new ownership, contention, and
post-success persistence rules independently reviewable.

The outcome contains only:

- a stable reason code matching a conservative ASCII pattern and bounded length;
- a bounded public message selected from server-defined constants;
- a breaker action fixed by server code: `ignore` for expected domain/policy
  outcomes or `record_failure` for sanitized dependency failures.

`ignore` is neutral: it neither increments nor resets breaker state. This
prevents invalid/ineligible requests from opening the circuit and also prevents
them from masking prior provider failures. `record_failure` increments the
breaker normally. Unexpected exceptions also remain breaker-counted.

Neutrality includes half-open state. An ignored outcome releases any acquired
half-open probe lease but neither closes nor reopens the circuit and does not
advance a success threshold. The next eligible call may perform the real health
probe. A counted failure in half-open reopens the circuit under the existing
backoff policy.

The module breaker wrapper must adapt these actions without treating an ignored
failure as a successful return. One implementation is to let ignored failures
pass through as an exception type excluded from breaker accounting, while
wrapping dependency and unexpected exceptions in one internal counted type and
unwrapping them after breaker processing. Both the injected and fallback module
breaker paths must implement the same semantics. The execution runtime then
handles known variants uniformly:

- classifies metrics, audit, post-hooks, and tool-use reporting as failure;
- emits a standard MCP tool result with `isError: true` and bounded structured
  JSON content containing only `status`, `reason_code`, and `message`;
- does not place the failure in the idempotency result cache;
- never exposes the internal exception type, traceback, or raw message.

The touched module wrapper and breaker integration log only the server-defined
reason code, sanitized exception family, module ID, and bounded structural
metadata. They must not interpolate `str(exception)` or exception repr. Raw
dependency text is retained only where an existing protected sink explicitly
allows it; the Skills runner does not create such a sink.

Unexpected exceptions continue through the existing generic failure path and
continue to count toward the module circuit breaker. Invalid parameters,
permission denial, approval required, governance denial, rate limiting, and
existing idempotency conflicts retain their protocol-level mappings. The new
`idempotency_in_progress` and `idempotency_unavailable` outcomes are also
protocol-level, breaker-neutral failures and never enter the result cache.

The expected-failure type and mapping are generic execution primitives. Neither
their names nor their implementation may mention Skills, models, or providers.
The exact structured error content is:

```json
{
  "status": "failed",
  "reason_code": "stable_server_defined_code",
  "message": "Bounded server-defined message"
}
```

This JSON object is the sole error content item in the repository's established
structured JSON content representation, and the outer `tools/call` result sets
`isError: true`. The outer result may retain the existing safe `module`, `tool`,
and `eval` bookkeeping fields; it contains no other module payload. This is a
tool execution error result, not a JSON-RPC request error.

The extracted execution runtime also honors a host-neutral, server-authored
`rate_limit_fail_closed` tool metadata flag. For such a tool, a rate-limit
dependency error returns sanitized protocol-level `rate_limit_unavailable`
before module execution. It is breaker-neutral and uncached. An actual limit
denial retains the existing rate-limit response. Tools without the flag retain
their current compatibility behavior; no tool-name or Skills-specific branch is
introduced.

Prepared execution must not reread security-relevant behavior from a shared
mutable tool-definition dictionary. Preparation derives a versioned immutable
`PreparedExecutionPolicy` containing only normalized execution controls,
including effect classification, the final rate-limit category,
`rate_limit_fail_closed`, and a bounded idempotency policy containing argument
injection, TTL, contention wait, success finalization, lock, entry-count, and
serialized-result limits.
The prepared-call HMAC covers every policy field; the normalized idempotency-key
digest; the strict canonical argument digest; and digests of strict canonical
JSON tool-definition and scope-reporting snapshots. The argument snapshot is
the immutable dispatch authority; the compatibility `tool_args` view must match
its signed digest but is never passed to a module. All snapshots are bounded and
detached from their mutable sources.
The shared encoder accepts only JSON values with string object keys, uses UTF-8
with `ensure_ascii=False`, `allow_nan=False`, sorted keys, and compact
separators, and has no coercing fallback such as `default=str`. Canonical
tool-definition bytes are limited to 1,000,000 and canonical scope-reporting
bytes to 256,000; exceeding either limit fails preparation.

Before rate admission or idempotency ownership, the runtime verifies the HMAC,
argument, context, and idempotency-scope fingerprints; normalized-key/cache-key
relationship; and current registry binding. The tool must still resolve to the
same operational module ID and instance, and its current canonical definition
digest must match the prepared digest. Missing, replaced, disabled, or
definition-changed tools fail closed as a stale prepared call without dispatch.
Observer-facing
tool-definition and scope dictionaries are decoded as private copies from the
verified canonical snapshots. The runtime consumes only
`PreparedExecutionPolicy` for execution decisions. Registry mutation therefore
either leaves the prepared snapshot unchanged or invalidates the call; it can
never silently rewrite its behavior.

Because rate admission and idempotency contention may await, the complete
prepared integrity check and live module/tool/definition binding are checked
again immediately before module invocation. That second check is inside the
owner callback and occurs after all pre-dispatch waits. The context fingerprint
includes every explicit server-authenticated identity field, including numeric
active team and organization scope when added by the model adapter work. A
stale prepared call is a sanitized, breaker-neutral, uncached protocol failure
and cannot fall back to a previous module object or mutated request context.

Authorization, approval, governance, and effective-policy evaluation use a
bounded request snapshot established during preparation. Changes in external
RBAC or policy stores after preparation apply to newly prepared calls; this
design does not claim continuous mid-request revocation while a bounded rate,
idempotency, or module wait is in progress. The second check does reject any
mutation of the signed request identity/context or prepared policy itself. A
future continuous-revocation feature must add and bind an explicit policy epoch
or lease instead of silently rerunning a second policy transaction with
different side effects.

The policy type and integrity-payload version are explicit compatibility
boundaries. Any future execution control derived from tool metadata, schema, or
configuration must become a typed policy field and be included in the HMAC
payload before the runtime may consume it. The observer snapshot may be used by
reporting, evaluation metadata, and post-hooks, but it is not an authority for
rate limiting, effect classification, idempotency behavior, breaker action, or
dispatch. This is a generic prepared-call invariant and adds no Skills-specific
logic to `protocol.py`.

## Model Invocation Port

The runner depends on a narrow host-neutral port, not on `MCPProtocol`,
`ToolExecutionCoordinator`, the REST endpoint, or core `ToolExecutor`.

Define normalized identity, capability, request, and result types in a small MCP
module-runtime port module:

```python
@dataclass(frozen=True, slots=True)
class ModelInvocationIdentity:
    user_id: int
    active_team_id: int | None
    active_organization_id: int | None
    execution_id: str


@dataclass(frozen=True, slots=True)
class ModelCompletionCapabilities:
    native_async_cancellation: bool
    response_limit_before_decode: bool
    native_max_output_tokens: bool
    tool_suppression: bool
    automatic_retries_disabled: bool


@dataclass(frozen=True, slots=True)
class ModelCompletionRequest:
    system_prompt: str
    user_prompt: str
    max_output_tokens: int
    max_output_chars: int
    max_output_bytes: int
    max_provider_response_bytes: int


@dataclass(frozen=True, slots=True)
class ModelCompletionResult:
    content: str


class ModelCompletionPort(Protocol):
    @property
    def capabilities(self) -> ModelCompletionCapabilities: ...

    async def complete(
        self,
        request: ModelCompletionRequest,
        identity: ModelInvocationIdentity,
    ) -> ModelCompletionResult: ...
```

The injected port object receives the normalized request and immutable
`ModelInvocationIdentity`. `user_id` and each present active-scope ID are
positive integers captured from explicit server-authenticated context fields,
not `RequestContext.metadata`, request arguments, headers, or a client-selected
membership. The MCP authentication adapter must populate those explicit fields;
missing active scope means that precedence level is skipped, never guessed from
the user's other memberships. `execution_id` is a canonical lowercase UUIDv4
generated by the server for this run; it is not derived from JSON-RPC request
IDs, client IDs, metadata, arguments, or Skill content.

The port does not receive mutable `RequestContext`, client/request identifiers,
request metadata, database paths, arbitrary headers, or credential fields. It
returns only normalized output, not a raw provider response. Before credential
resolution or quota reservation, the adapter revalidates that the user is
active, the active team belongs to the active organization when both are
present, and current membership authorizes each supplied active scope. A missing
or revoked supplied scope fails closed; it does not fall through to another
membership or the server credential.

The server composition root captures one non-empty operator-configured provider
and model at initialization and supplies the production adapter. The request
cannot change either value. Auto-routing, provider fallback, authored model
selection, and model aliases that can resolve to a different provider are
disabled.

The adapter:

- resolves user, team, organization, billing, and credential scope from the
  immutable numeric scope identity and authoritative AuthNZ stores;
- preserves the shared provider runtime's explicit user, active-team,
  active-organization, then server credential precedence; only an authoritative
  `absent` result advances to the next level, while invalid, unauthorized,
  revoked, or unavailable scope state fails closed;
- never derives membership, credentials, endpoints, provider, model, or billing
  scope from client/request identifiers or request metadata;
- constructs the shared provider credential runtime from a frozen server config
  snapshot and authoritative membership result;
- fixes the endpoint to the selected provider's frozen server configuration and
  sets credential-runtime base-URL override trust to false; scoped credential
  records may supply secrets but cannot change the runner endpoint;
- uses existing provider outbound-egress enforcement;
- performs the same quota preflight, worst-case token/cost reservation,
  provider concurrency admission, actual-usage reconciliation, and credential
  usage marking required by the normal authenticated provider path;
- fails closed before dispatch when identity, membership, credential, quota,
  accounting, egress, or concurrency services are unavailable;
- retains the conservative reservation under the shared ambiguous-outcome
  policy when final provider usage cannot be determined;
- performs at most one physical provider dispatch for a non-streaming completion
  (`stream=false`) with one choice, `tools=None`, no tool choice, and a required
  provider-native output-token cap;
- disables all server-controlled adapter, SDK, HTTP-client, fallback, and module
  retries and makes one outbound request attempt;
- rejects an advertised response length above the cap when trustworthy, and
  independently enforces the cap with a limit-plus-one incremental read over
  decompressed response bytes before JSON decoding or envelope retention;
- rejects multiple choices, malformed content, and every tool-call response;
- normalizes provider-specific output into `ModelCompletionResult`;
- maps provider errors into internal typed failures without raw error text.

Incremental HTTP response-body consumption exists only to enforce the byte cap;
it is not token streaming, does not expose partial content, and does not change
the one-turn non-streaming completion contract. Wire-byte accounting alone is
insufficient; compressed-response expansion is covered by the decompressed cap.

Breaker action for provider failures derives from trusted credential/runtime
provenance, never request metadata. Missing, invalid, exhausted, or unauthorized
user/team/organization/server credentials are credential-scope-local and must
not mutate the global module breaker. Shared adapter defects, shared
identity/accounting-store outages, and failures of the fixed provider transport
path may record a global breaker failure. Tests must prove one credential
scope's failures cannot open or reset another scope's module circuit.

The breaker is module-wide, so any failure that can be influenced by a selected
Skill, prompt, user-scoped filesystem/database state, credential scope, or one
request's provider content is breaker-neutral. Only failures whose trusted
provenance identifies shared runner infrastructure may use `record_failure`.
HTTP status, exception class, or a stable public reason alone is not sufficient
to classify a failure as shared.

The worst-case reservation must be durably accepted before dispatch. After a
valid billable completion is received, failure to reconcile actual usage or mark
credential usage does not convert that completion into a retriable public
failure: the adapter leaves the conservative reservation intact, emits bounded
degraded-accounting observability, completes credential cleanup, and returns the
validated result. This prevents a post-call accounting outage from inducing a
duplicate paid request. An ambiguous transport outcome has no success result and
also retains the reservation under shared policy.

The adapter records a monotonic dispatch-started transition. Cancellation or
failure before any outbound request bytes are attempted releases the unused
reservation. At or after dispatch starts, cancellation or an unknown transport
outcome retains the reservation until normal reconciliation or ambiguous-outcome
policy resolves it. Tests must cover both sides of this boundary.

The capability descriptor is an enforced contract, not informational metadata.
The module is unavailable during initialization and health checking unless the
port is present, all five capabilities are true, and the configured provider
path is certified to implement them. The first implementation may support only
one provider path; unsupported configured providers fail closed instead of
falling back to a sync adapter.

Capability/readiness checks are configuration- and implementation-based. They
must not issue a model completion, probe with Skill content, or make a
per-request discovery call before the accounted dispatch.

No fallback may instantiate `SkillExecutor`, `ToolExecutor`, call the generic
low-level provider helper directly, or use `await_bounded_sync_call`.

Provider output normalization is deterministic. After bounded envelope parsing,
the adapter requires exactly one textual choice with no tool-call fields,
requires strict UTF-8 encodability with no unpaired surrogate, converts CRLF and
bare CR line endings to LF, rejects empty/whitespace-only output and prohibited
C0/C1 controls other than tab or line feed, then applies the configured
character and UTF-8 byte ceilings. It performs no trimming or Unicode
normalization beyond that line-ending conversion.

## Prompt And Completion Flow

The runner performs at most one physical provider dispatch, one logical
completion, and one model turn. No server-controlled layer retries
automatically, including after connect, timeout, cancellation, rate limit,
malformed response, or an ambiguous transport failure. A caller may deliberately
submit another request, subject to normal idempotency and quota behavior.

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

All settings reject booleans and non-integer coercion. Missing required values
or values outside documented bounds fail module initialization; limits are not
silently clamped.

Initial limits:

- `SKILL.md`: existing hard limit of 500,000 bytes, enforced with a bounded
  limit-plus-one read before UTF-8 decoding;
- YAML frontmatter: hard limits of 16,384 UTF-8 bytes, 64 nodes, and depth 4;
- arguments: 10,000 characters and 40,000 UTF-8 bytes;
- final combined prompt: default 32,000 characters and 128,000 UTF-8 bytes, with
  hard maximums of 100,000 characters and 400,000 bytes;
- normalized output: default 20,000 characters and 80,000 UTF-8 bytes, with
  hard maximums of 100,000 characters and 400,000 bytes;
- requested provider output: default 4,096 tokens and hard maximum 8,192
  tokens, enforced natively before dispatch;
- provider response envelope: default 1,000,000 bytes and hard maximum
  2,000,000 bytes;
- run timeout: default 60 seconds and hard maximum 120 seconds;
- cancellation cleanup allowance: default 5 seconds and hard maximum 15
  seconds;
- idempotency contention wait: default 5 seconds and hard maximum 30 seconds,
  outside but immediately preceding owned module execution;
- idempotency success finalization: default 5 seconds and hard maximum 15
  seconds, cancellation-shielded only after a valid owner result exists;
- serialized idempotency success result: default 256,000 bytes and hard maximum
  1,000,000 bytes, enforced on strict canonical JSON before local or Redis
  retention;
- module timeout: at least one second greater than run timeout plus cleanup
  allowance; configuration violating this relationship fails initialization;
- completions per call: exactly 1;
- tool calls per call: 0;
- supporting files per call: 0;
- module concurrency: default 2.

Argument, prompt, provider-envelope, and normalized-output overflows fail
atomically. The public result never contains a partial or truncated completion.

The provider adapter requires a provider-native token/output cap. Character and
UTF-8 byte limits remain authoritative after normalization. Prompt bytes plus
the configured maximum output tokens also feed the shared quota/cost preflight;
a model dispatch cannot occur on an unbounded or unknown worst-case reservation.

## Cancellation And Timeout

Model invocation runs in one retained child task.

On caller cancellation:

1. cancel the child task;
2. drain it within the configured cleanup allowance so provider cleanup,
   accounting reconciliation, and connection release complete;
3. re-raise `asyncio.CancelledError`;
4. emit no success result or success event.

On timeout:

1. cancel and drain the child task;
2. raise the stable `skill_run_timeout` failure;
3. do not return provider content received after the deadline.

If the child does not terminate inside the cleanup allowance, it remains in a
module-owned lifecycle registry with its result permanently discarded. The
reservation stays in the ambiguous-outcome state, adapter health becomes false,
and the module refuses every new dispatch until the task terminates and health
is explicitly re-established. The condition records the shared dependency
failure `skill_run_provider_unavailable`; caller cancellation still propagates
after this containment state is installed. Module shutdown cancels and drains
all retained children under the existing bounded shutdown policy. No task is
left unreferenced.

The production model port must use a native async transport, propagate
cancellation, and cancel its underlying HTTP operation. A provider path that
uses a sync worker, cannot prove transport cancellation, or cannot reconcile an
ambiguous billable outcome is not eligible. Broad exception handlers must not
convert cancellation into a generic failure.

The outer module timeout is a last-resort containment boundary, not the normal
runner deadline. Tests must prove the inner runner emits `skill_run_timeout`
and completes cleanup before the outer timeout can replace it with the base
module's generic timeout.

Database cleanup uses the existing retained-lifecycle pattern and completes
before the model child task is created.

## Error Contract

Known public failures use stable reason codes and sanitized messages. Breaker
action is explicit so caller-caused ineligibility cannot disable the module
while real dependency failures still provide isolation:

| Condition | Public reason | Breaker action |
|---|---|---|
| Unknown, hidden, blocked, deleted, cross-user, or model-disabled Skill | `skill_not_found` | ignore |
| Inline-mode Skill | `skill_run_requires_fork` | ignore |
| Skill declares any tools | `skill_run_tools_unsupported` | ignore |
| Skill has extra directory entries | `skill_run_supporting_files_unsupported` | ignore |
| Skill declares a model selector | `skill_run_model_selector_unsupported` | ignore |
| Missing, oversize, invalid UTF-8/control-bearing, structurally invalid, or parser-budget-exceeding `SKILL.md` | `skill_not_found` | ignore |
| Rendered prompt exceeds limit | `skill_run_prompt_too_large` | ignore |
| User/team/organization/server credential is missing, invalid, exhausted, or unauthorized | `skill_run_provider_unavailable` | ignore |
| Model port, required capability, shared identity resolver, or fixed provider transport path is unavailable | `skill_run_provider_unavailable` | record failure |
| Provider egress is denied | `skill_run_egress_denied` | ignore |
| Provider usage quota is denied | `skill_run_quota_exceeded` | ignore |
| Provider response envelope exceeds limit | `skill_run_provider_response_too_large` | ignore |
| Provider returned a tool call | `skill_run_unexpected_tool_call` | ignore |
| Provider returned multiple choices or malformed content | `skill_run_invalid_provider_response` | ignore |
| Normalized output exceeds limit | `skill_run_output_too_large` | ignore |
| Wall timeout | `skill_run_timeout` | ignore |
| Provider child failed to terminate inside cleanup allowance | `skill_run_provider_unavailable` | record failure |
| Snapshot changed during capture | `skills_unavailable` | ignore |
| User-scoped Skill filesystem/database or request cleanup failed | `skills_unavailable` | ignore |
| Shared Context Integrity resolver or shared runner service failed | `skills_unavailable` | record failure |

Known runner failures use the generic MCP expected-failure result with
`isError: true`. Existing protocol errors remain authoritative for invalid
parameters, permission denial, approval required, governance denial, rate limit,
and idempotency conflicts. Unclassified internal failures retain the existing
generic error response, count toward the breaker, and never expose a raw message.

Warnings and errors may log:

- operation;
- component;
- safe reason code;
- exception class;
- bounded filename/function/line traceback metadata;
- canonical Skill name only after visibility authorization;
- Skill version;
- the fixed-format server-generated execution ID.

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

Internal safe event metadata may include the server-generated execution ID,
Skill version, elapsed duration, and zero tool/supporting-file counts. Public
result fields remain the exact success contract above. Neither event metadata
nor MCP audit metadata records prompt or output content. Protected provider
observability may retain provider/model and usage metadata under its existing
policy, but not runner prompt or output text.

## Component Boundaries

`SkillsRunnerModule`

- owns the public tool schema, exact argument validation, write
  classification, request dispatch, public result shape, and stable error
  mapping;
- depends on a snapshot loader and model completion port;
- does not import `MCPProtocol`, core `SkillExecutor`, core `ToolExecutor`, or a
  provider adapter.

Generic expected tool-failure outcome

- lives in the host-neutral MCP execution package;
- owns bounded reason/message validation and the `isError: true` mapping;
- preserves existing protocol-level authorization errors;
- applies the server-defined breaker action, prevents known failures from
  entering the idempotency result cache, and records them as failures;
- adds no Skills-specific logic to `protocol.py`.

`tool_execution/idempotency.py`

- owns local/Redis argument binding, replay caching, lock ownership, bounded
  contention, sticky-backend behavior, bounded cancellation-shielded success
  finalization, and degraded-persistence results;
- invokes an owner callback at most once per request;
- exposes a narrow typed interface to the execution runtime;
- leaves only a compatibility re-export in `protocol.py`.

`SkillRunSnapshotLoader`

- owns trusted user/path extraction, request-scoped database construction,
  one strict model-visible verified Skill read, stable no-follow inventory,
  immutable snapshot creation, and database cleanup;
- performs no model or tool work;
- never reads bytes from a supporting file;
- exposes no database object after return.

`ModelCompletionPort`

- owns host-neutral identity, capability, request, and result contracts;
- contains no credential fields.

Production model adapter

- lives at the server composition boundary;
- owns authoritative membership and credential resolution, fixed
  provider/model selection, quota reservation and reconciliation, egress
  enforcement, one native-async bounded transport dispatch, response
  normalization, and provider error classification.

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
  timeout_seconds: 70
  max_concurrent: 2
  max_retries: 0
  settings:
    provider: "${MCP_SKILLS_RUNNER_PROVIDER:-}"
    model: "${MCP_SKILLS_RUNNER_MODEL:-}"
    max_prompt_chars: 32000
    max_prompt_bytes: 128000
    max_output_tokens: 4096
    max_output_chars: 20000
    max_output_bytes: 80000
    max_provider_response_bytes: 1000000
    run_timeout_seconds: 60
    cancellation_cleanup_seconds: 5
    idempotency_wait_seconds: 5
    idempotency_finalize_seconds: 5
    idempotency_result_max_bytes: 256000
```

Provider and model are non-secret fixed selectors. Both must be non-empty when
the module is enabled and must match a certified capability-gated adapter path.
No credential, endpoint, header, fallback list, or provider-retry setting is
accepted in `settings`. Credential and endpoint configuration remains owned by
the existing provider and credential systems; the top-level module retry value
is fixed at zero.

## Testing Strategy

Implementation follows TDD with fakes at the new ports.

### Surface And Configuration

- default config keeps `skills_runner` disabled;
- disabled status reports the `model_execution` tier and explicit opt-in;
- enabling registers exactly `skills.run`;
- existing `skills` remains in `read_only`;
- existing Skills tool catalog remains exactly list/get/render;
- tool metadata includes `write_capable` and `uses_network`, read-only catalog
  callers see it as non-executable, rate limiting selects `network`, and
  `rate_limit_fail_closed` prevents dispatch when admission is unavailable;
- runner initialization/health fails closed without a model port, fixed
  provider/model, or any required capability;
- `max_retries` is zero and invalid/out-of-range settings or timeout/cleanup
  relationships fail init rather than clamp;
- schema rejects unknown keys, wrong types, booleans as integers, oversized
  names, and oversized arguments.

### Generic Expected Failures

- expected outcomes produce `isError: true` with only bounded
  `status`/`reason_code`/`message` error content plus the runtime's existing safe
  outer bookkeeping fields, and do not become JSON-RPC request errors;
- domain/policy outcomes record failure metrics, audit, post-hooks, and tool-use
  events without incrementing or resetting the circuit breaker;
- a domain/policy outcome after a prior dependency failure leaves the existing
  failure count unchanged;
- an ignored half-open outcome releases the probe slot while preserving
  half-open state and success/failure thresholds; the next eligible call can
  probe, while a counted half-open failure reopens with normal backoff;
- known dependency outcomes produce the same sanitized public shape and do
  increment/open the breaker at its configured threshold;
- expected outcomes are not cached by local or Redis idempotency paths;
- the owner callback is invoked at most once even when Redis result persistence,
  serialization, or release fails after a successful execution;
- strict canonical JSON serialization rejects non-JSON/NaN values, enforces the
  result-byte ceiling, has equivalent local/Redis values, and prevents caller
  mutation from changing a later local replay;
- ambiguous Redis lock acquisition fails closed before dispatch, while a
  post-success Redis persistence failure returns the original result, records
  degraded persistence, and at most stores the result in the bounded local
  cache;
- idempotent success persistence precedes cancellation-sensitive post-execution
  observers, whose failure cannot redispatch or replace the committed outcome;
- cancellation at every boundary after a valid callback result waits for the
  bounded shielded commit; a local replay copy exists before Redis awaits, and
  cancellation or remote timeout cannot re-enter the callback;
- a remote finalizer that misses its deadline remains manager-owned, degrades
  remote idempotency health, can publish only the exact committed bytes, and is
  drained at shutdown without orphan work;
- unexpected exceptions remain generic, count toward the breaker, and expose no
  raw message;
- reason/message validation rejects oversized, malformed, or control-bearing
  values and never derives public content from a raw exception message;
- module/breaker logs in the touched path never interpolate exception message or
  repr and contain only tested safe fields;
- no Skills-specific handling is added to `protocol.py`;
- `IdempotencyManager` implementation lives outside `protocol.py`, whose legacy
  import remains a compatibility re-export only;
- generic fail-closed rate-limit metadata returns sanitized
  `rate_limit_unavailable` before module dispatch and leaves breaker and
  idempotency state unchanged;
- prepared execution derives a detached canonical tool-definition snapshot and
  a versioned immutable execution policy, binds both into the prepared-call
  HMAC, and never rereads execution controls from observer-facing metadata;
- the HMAC also binds normalized idempotency-key and bounded scope-reporting
  digests plus the authenticated idempotency-scope fingerprint; cache-key
  relationships are recomputed, and observer dictionaries are private decodes
  rather than shared nested mutable objects;
- strict canonical snapshot serialization rejects non-JSON values, non-string
  keys, NaN, cycles, and exact-byte-limit overflow without `default=str`;
- mutating or replacing the registry/module/definition after preparation makes
  the prepared call stale, while tampering with any snapshot, policy, key, or
  scope field fails before rate admission, idempotency ownership, module
  execution, or model dispatch;
- registry/definition, arguments, identity context, policy, key, or snapshot
  drift during rate or idempotency waits is caught by the second complete check
  inside the owner callback immediately before module invocation;

### Eligibility And Isolation

- visible fork Skill with no tools/files produces one snapshot;
- snapshot loading performs only a targeted reconciliation for the requested
  Skill; it neither invokes catalog-wide synchronization nor reads/mutates an
  unrelated Skill;
- inline, declared-tool, authored-model, and extra-entry Skills fail before the
  model call;
- strict execution parsing rejects coercible strings/scalars, mixed aliases,
  duplicate/unknown keys, non-boolean model-disable/user-invocable values, and
  `user-invocable: false`;
- strict parsing rejects over-budget frontmatter, aliases, anchors, merge keys,
  explicit tags, excessive node depth/count, invalid UTF-8, unpaired
  surrogates, and prohibited prompt controls before model dispatch;
- hidden/integrity-blocked/deleted/cross-user/model-disabled cases all return
  `skill_not_found`;
- malformed/oversize/control-bearing files that prevent exact visibility also
  return `skill_not_found`, and overlapping visible eligibility failures follow
  the documented deterministic precedence;
- an unavailable Context Integrity resolver returns `skills_unavailable` and
  prevents model dispatch;
- the loader checks the canonical `skill:user:<user_id>/<name>` asset digest
  with exact purpose `skill_execution`, independently of the registry raw-text
  hash;
- user identity/database path comes only from explicit trusted `RequestContext`
  fields, and spoofed metadata cannot affect either;
- supporting-file bytes are never opened or read;
- symlinks, directories, hidden files, special files, and concurrent
  inventory/registry/file mutation fail before model invocation;
- selected-row insert/update/restore/tombstone behavior uses optimistic version
  checks and post-mutation revalidation, while invalid/integrity-blocked files
  are not newly indexed;
- database closes before model invocation;
- close failure prevents model invocation;
- mutation completed after snapshot verification does not alter captured
  version/content;
- argument text containing SQL-like substrings is preserved byte-for-byte while
  prohibited control characters are rejected without mutation.

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
- rate-limit denial or admission-service failure prevents model invocation;
- successful post-hooks and tool-use reporting preserve existing behavior.

### Model Port

- receives rendered prompt, fixed user prompt, immutable positive numeric user
  and optional active team/organization identity, server-generated execution
  ID, output-token/character/byte bounds, and no mutable context,
  client/request identifiers, tools, or credentials;
- follows user, active-team, active-organization, then server credential
  precedence; absent advances, while revoked/invalid/unavailable state fails
  closed, and it never guesses among multiple memberships;
- malformed, cross-organization, missing, or revoked active scope fails before
  credential resolution, quota reservation, or provider dispatch;
- dispatches at most once and disables every server-controlled retry/fallback
  layer;
- client and Skill cannot override provider, model, endpoint, or key;
- spoofed client metadata cannot affect membership, billing, credentials,
  provider, or model;
- credential-scope-local failures leave the global module breaker unchanged,
  as do Skill/prompt/content-dependent failures, while failures with trusted
  shared adapter/store/transport provenance use record-failure action;
- unsupported sync/cancellation/response-limit/token-limit adapter paths fail
  readiness without dispatch;
- readiness checks make no model/probe/discovery dispatch;
- tool-call response fails closed;
- multiple-choice and malformed/non-string responses fail closed;
- output normalization has exact tests for line endings, empty/whitespace-only
  content, unpaired surrogates, prohibited controls, and absence of trimming or
  Unicode normalization;
- provider unavailable and egress-denied failures map to stable reasons;
- quota reservation happens before dispatch, actual usage reconciles afterward,
  and ambiguous outcomes retain conservative accounting;
- cancellation before dispatch releases the reservation, while cancellation at
  or after dispatch start retains it under ambiguous-outcome policy;
- post-completion reconciliation/credential-marking failure retains the
  worst-case reservation and valid result rather than creating a retriable
  duplicate-call failure;
- oversized provider envelopes fail before normalization and retention;
- compressed responses are capped on decompressed bytes and cannot bypass the
  envelope limit with a small wire representation;
- raw provider errors are absent from responses and captured logs.

### Bounds, Cancellation, And Idempotency

- prompt character/byte boundaries succeed at their limits and fail above
  either limit before provider call;
- output-token configuration is enforced natively and rejects values above the
  hard maximum;
- provider envelope and normalized output character/byte boundaries fail
  atomically above their limits;
- timeout cancels and drains the child task inside the cleanup allowance and
  returns `skill_run_timeout` before the outer module timeout;
- a child that violates cancellation cleanup remains lifecycle-owned, retains
  ambiguous accounting, makes adapter/module health fail closed, blocks new
  dispatches, and can never publish a late result;
- caller cancellation propagates and emits no success;
- standard idempotency replay does not invoke the model a second time;
- a Skill/config change followed by the same key replays the original output and
  `skill_version`; a new key executes the new state;
- successful idempotency payload retention is documented, contains no separate
  request/prompt/argument fields, and is treated as sensitive because model
  output may reproduce input;
- serialized success cache limits pass at the exact byte boundary and degrade
  without redispatch above it;
- failed execution is never replayed from idempotency cache;
- failed execution may be retried with the same key/arguments while the
  plaintext-free argument fingerprint continues to reject different arguments;
- local and Redis contention waits fail as `idempotency_in_progress` at the
  configured bound without dispatch or breaker mutation;
- Redis lock-acquisition ambiguity fails as `idempotency_unavailable` without
  dispatch, and Redis result-write failure after dispatch produces exactly one
  model call and returns the original success without claiming durable replay;
- a different owner cannot replay another owner's key;
- the same user cannot replay one active team/organization scope's result under
  another active scope with the same key and arguments, while personal/no-scope
  keys preserve compatibility;
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

The security review exposed three ownership boundaries with different blast
radii. Implement them as ordered child tasks with separate plans, commits, test
gates, and review checkpoints.

### `TASK-2294.3.1`: Harden Generic MCP Execution Foundations

May touch the host-neutral tool-execution models, `modules/base.py`, execution
runtime/coordinator, reporting, fail-closed rate-limit admission, idempotency
ownership/finalization, and focused tests. It must remove the post-owner Redis
fallback redispatch path, extract the idempotency implementation from
`protocol.py`, preserve its compatibility import, bind a versioned immutable
execution-policy, canonical tool-definition/scope snapshots, normalized
idempotency-key relationship, and live registry binding into prepared-call
integrity, and must not add a Skills-specific branch to `protocol.py`.

### `TASK-2294.3.2`: Bounded Model Completion Adapter

May add the small model-runtime port definitions, authoritative identity adapter,
explicit optional active-scope fields populated by MCP authentication, the
corresponding prepared-context fingerprint extension, one certified native-async
provider transport path, quota/accounting composition, server composition
wiring, and focused provider/security tests. Active scope must not use generic
request metadata. It depends on `TASK-2294.3.1` and the shared provider-runtime
consolidation in `TASK-12963`; its plan must revalidate the landed provider
interfaces instead of duplicating or racing that work. It must not expose a
public MCP tool.

### `TASK-2294.3.3`: Strict Snapshot Loader And Runner

May add the strict Skills execution-snapshot service operation, a new
`skills_runner_module.py`, one explicit typed `model_completion_port` field on
`ModuleConfig`, server registration, `module_surface.py`, `mcp_modules.yaml`,
focused targeted-reconciliation/runner/authorization/lifecycle tests, and MCP
module documentation. The snapshot operation must not reuse catalog-wide
synchronization. It depends on both prerequisite tasks.

The port is injected by the server composition root. It must not be placed
in YAML settings or resolved through a service locator. YAML contains only the
fixed non-secret provider/model selectors and bounded module settings.

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
2. A generic host-neutral expected tool-failure outcome produces MCP
   `isError: true` results, accurate failure reporting, safe logs, and no
   failure-result caching; closed and half-open domain outcomes are breaker-neutral
   while trusted shared dependency failures count, without Skills logic in
   `protocol.py`. Prepared calls use an HMAC-bound immutable execution policy,
   normalized key relationship, detached canonical tool-definition/scope
   snapshots, and live registry revalidation, so mutable or stale state cannot
   alter admission, idempotency, breaker, reporting, or dispatch behavior after
   preparation.
3. `IdempotencyManager` is extracted from `protocol.py`; bounded sticky
   ownership invokes a callback at most once per request, uses one strict
   byte-bounded local/Redis replay representation, commits valid success under
   bounded cancellation shielding, partitions replay by authenticated active
   scope without exposing raw scope IDs, and cannot redispatch after successful
   owner execution.
4. Only user-scoped, user-invocable, model-visible, integrity-verified fork-mode
   Skills with no declared tools, authored model, or directory entry beyond one
   regular `SKILL.md` are eligible under the explicit resource-bounded strict
   frontmatter contract.
5. One targeted mutation-checked Skill snapshot reconciles no unrelated row,
   validates both registry raw-text hash and canonical Context Integrity digest
   for `skill_execution`, and closes request-scoped database resources before
   model/provider work.
6. Normal tool and `Skill(skill_name)` authorization applies; tool-definition
   and call classification agree that `skills.run` is effectful, network-using,
   and fail-closed when rate-limit admission is unavailable.
7. Model execution uses one capability-gated injected completion port with only
   immutable positive numeric user and explicit active team/organization scope
   identity plus a server-generated execution ID, authoritative membership and
   credential-precedence resolution, fixed provider/model, quota
   reservation/reconciliation, egress enforcement, and shared-provenance breaker
   classification; client and Skill overrides are impossible.
8. At most one physical tool-disabled provider dispatch and one model turn occur
   under native output-token, pre-decode response-byte, prompt, output, timeout,
   cancellation, and retained-child containment bounds; no server-controlled
   layer retries or falls back.
9. Results, errors, logs, and telemetry are stable and sanitized, with no prompt,
   raw provider payload, credential, endpoint, path, client/request identifier,
   or raw exception leakage.
10. Documentation discloses successful normalized-output retention in the
   owner-scoped idempotency cache, including sensitive-output and degraded
   process-local persistence cases; failures and plaintext request/prompt/
   argument fields are excluded while argument fingerprint binding remains for
   its TTL.
11. Focused tests and documentation cover the execution contract, and nested
   read-only and effectful execution remain isolated in `TASK-2294.4` and
   `TASK-2294.5`.
