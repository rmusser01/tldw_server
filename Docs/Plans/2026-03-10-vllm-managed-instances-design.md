# Managed vLLM Instances Design

Date: 2026-03-10
Status: Approved
Scope: managed `vllm` lifecycle, multi-instance routing, local and remote orchestration, chat/embeddings/multimodal inference via existing OpenAI-compatible provider path

## 1. Summary

Add first-class managed `vLLM` instances to `tldw_server` so operators can:

- define multiple structured `vLLM` instance specs
- launch and stop `vLLM` servers locally or on remote hosts
- monitor readiness and health over time
- route chat, embeddings, and multimodal requests to a selected managed instance

The design keeps inference on the existing OpenAI-compatible `vllm` provider adapter, but moves lifecycle management into a dedicated control plane with durable instance storage, request-scoped routing, pluggable executors, and job-backed lifecycle operations.

Implementation scope for v1 is a single `tldw_server` control-plane deployment with SQLite-backed durable instance storage behind a repository abstraction. Postgres-backed shared-state deployments remain a follow-on unless a matching Postgres repository is added during implementation.

## 2. User-Approved Decisions

1. `tldw_server` should own `vLLM` process lifecycle.
2. `vLLM` should run as an external process, not as an in-process runtime inside `tldw_server`.
3. Launch configuration should use structured settings, not a raw operator-provided command string.
4. The design must account for embeddings and multimodal models from day one.
5. Remote orchestration is in scope.
6. Multiple managed `vLLM` instances are required.
7. Remote control should support SSH first and leave room for a future agent-based path.

## 3. Review-Driven Revisions

This design was revised after review to address the highest-risk gaps:

1. Instance specs and runtime metadata must be durable across `tldw_server` restarts.
2. Request routing must be request-scoped and must not mutate global `vllm_api_IP` or related shared config.
3. The executor contract must define direct reachability requirements for inference, especially for future agent mode.
4. Capability handling for embeddings and multimodal models must distinguish operator declarations from active probes.
5. SSH lifecycle management must rely on a stable remote launcher contract, not ad hoc shell backgrounding.
6. Long-running lifecycle operations should be Jobs-backed rather than purely synchronous admin requests.
7. `extra_args` must remain constrained and argv-based, not raw shell text.

## 4. Current State

### 4.1 Existing Capabilities

- `vLLM` already exists as a chat provider in `tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py`.
- The current `vllm` adapter is OpenAI-compatible HTTP and already supports the core request path used by chat.
- Configuration already includes legacy single-endpoint keys such as `vllm_api_IP`, `vllm_model`, sampling defaults, retries, and timeouts.
- The provider registry already exposes `vllm` as a first-class provider name.
- The codebase already has two relevant lifecycle patterns:
  - `llama.cpp`: older singleton process management in `app/core/Local_LLM`
  - `mlx`: newer provider-specific lifecycle endpoints with a dedicated runtime registry

### 4.2 Gaps

- There is no multi-instance `vLLM` registry.
- There is no durable managed runtime model for `vLLM`.
- The existing `vllm` integration assumes a single configured endpoint rather than per-request instance selection.
- There is no local or SSH-backed `vLLM` command builder/supervisor.
- There is no admin API for creating, starting, stopping, probing, or selecting managed `vLLM` instances.
- There is no capability contract for embeddings or multimodal `vLLM` instances.

## 5. Goals And Non-Goals

### 5.1 Goals

- Add durable multi-instance `vLLM` management to `tldw_server`.
- Support both local-host and SSH-managed `vLLM` execution in v1.
- Reuse the existing OpenAI-compatible `vllm` inference adapter rather than inventing a separate inference stack.
- Support routing for chat, embeddings, and multimodal requests from day one.
- Provide explicit admin APIs and UI-ready status models for lifecycle operations.
- Make lifecycle operations operationally safe through structured command building, health checks, and auditable status.

### 5.2 Non-Goals

- No generic managed-runtime framework for every local backend in this phase.
- No promise of inference proxying through a future agent in v1.
- No arbitrary shell command execution for `vLLM` management.
- No automatic inference that every `vLLM` model supports embeddings or every multimodal mode from `/v1/models` alone.
- No hidden mutation of process/global config to swap active `vLLM` endpoints mid-request.

## 6. Proposed Architecture

### 6.1 Durable Instance Registry

Introduce a dedicated `VLLMInstanceRegistry` backed by durable storage.

Each instance record stores:

- identity: `instance_id`, display name, description, tags
- execution mode: `local`, `ssh`, `agent`
- transport configuration:
  - `local`: optional bind-address, working directory, environment/profile references
  - `ssh`: remote host, port, user, launcher path, optional remote workdir, and secret or connection-profile references
  - `agent`: reserved agent host/registration metadata and secret references
- launch spec: structured `vLLM` settings and constrained `extra_args`
- routing policy: enabled, default-route eligibility, optional future tenant/org policy
- capability metadata:
  - `declared_capabilities`
  - `probed_capabilities`
  - `effective_capabilities`
- observed runtime state:
  - `desired_state`
  - `observed_state`
  - last known `base_url`
  - pid/remote pid or executor handle metadata
  - last health check
  - last probe summary
  - last error

The registry is the source of truth for instance configuration. In-memory executor/session state is only a cache of current observation.

The repository contract behind the registry must support more than create/get/list:

- update instance specs for admin PATCH flows
- delete stopped instances safely
- persist lifecycle metadata changes such as desired state, observed state, last known base URL, last error, capability probes, and executor handles
- support concurrency-safe state transitions so Jobs workers and admin reads do not race on partially updated state

Transport/auth material must be stored by reference where possible. Secret-bearing fields should resolve through existing secret/config mechanisms and must not be echoed back verbatim in operator APIs or logs.

### 6.2 Request-Scoped Routing

Managed `vLLM` must not depend on mutating global settings such as `vllm_api_IP`.

Instead, add a request-scoped routing layer:

- request may specify `provider="vllm"` and `provider_instance_id`
- if omitted, routing may fall back to one configured managed default instance
- routing resolves a specific instance, endpoint, model, and capability envelope for the current request only

This resolution must feed the existing `vllm` HTTP adapter as per-request override data.

Result:

- multiple concurrent requests can safely target different `vLLM` instances
- no cross-request leakage occurs
- backward compatibility remains possible through a managed default route

### 6.3 Split Lifecycle From Inference

Keep the existing `vllm` adapter for inference transport and request formatting.

Add a new control-plane package for lifecycle management, for example:

`tldw_Server_API/app/core/VLLM_Management/`

Core responsibilities:

1. instance CRUD and state management
2. structured command building
3. launch/stop/restart/probe orchestration
4. health reconciliation
5. request-time instance resolution

This avoids mixing external process supervision logic into the OpenAI-compatible chat adapter.

### 6.4 Executor Abstraction

Define a common executor interface implemented by:

- `LocalExecutor`
- `SSHExecutor`
- `AgentExecutor` placeholder

Each executor must provide:

- `start(instance_spec) -> lifecycle_result`
- `stop(instance_spec, handle) -> stop_result`
- `restart(instance_spec, handle) -> lifecycle_result`
- `probe(instance_spec) -> probe_result`
- `resolve_logs(instance_spec, handle) -> log_metadata`

Every lifecycle result must include:

- resolved `base_url`
- executor lifecycle handle
- log metadata
- pid/remote pid metadata when known
- a normalized readiness/health contract

This keeps `local`, `ssh`, and future `agent` execution on the same state machine.

### 6.5 SSH Remote Launcher Contract

SSH mode should not use a fragile pattern like:

- `ssh host "vllm serve ... &"`

Instead, define a stable remote launcher contract:

- install or provide a small remote launcher script
- pass structured argv and metadata to that launcher
- launcher starts `vllm`, writes pid/log/status metadata, and returns a normalized handle
- stop/status operations use the same launcher contract

Benefits:

- stable pid tracking
- safer quoting and argument handling
- better orphan cleanup
- predictable log collection and restart behavior

### 6.6 Reachability Rule

Phase 1 inference requires direct reachability from `tldw_server` to the managed `vLLM` HTTP endpoint.

That means:

- local instances must expose a reachable local URL
- SSH-managed instances must expose a reachable remote URL to `tldw_server`
- future agent-managed instances are only inference-capable in v1 if they also expose a directly reachable URL

If future agent-managed instances do not expose direct reachability, the architecture must add inference proxying instead of pretending the current adapter path still works.

### 6.7 Reconciler

Add a background `VLLMReconciler` service that:

- loads persisted instance specs on startup
- probes existing known endpoints
- updates observed state to `healthy`, `unhealthy`, `stopped`, `failed`, or `unknown`
- reapplies monitoring and optional restart policy

The reconciler should not blindly relaunch every instance on startup. Persisted `desired_state` and observed results must drive behavior.

## 7. Launch Specification

### 7.1 Structured Settings

The launch spec should support structured `vLLM` settings such as:

- `model`
- `served_model_name`
- `host`
- `port`
- `dtype`
- `quantization`
- `tensor_parallel_size`
- `pipeline_parallel_size`
- `gpu_memory_utilization`
- `max_model_len`
- `max_num_seqs`
- `trust_remote_code`
- `chat_template`
- `limit_mm_per_prompt`
- `api_key`
- `download_dir`
- `revision`
- `tokenizer`
- `tokenizer_mode`

The exact list should track the supported `vllm serve` surface selected for the release, but the public API should remain structured and typed.

### 7.2 Constrained Extra Args

Allow `extra_args` only as a constrained list of validated pass-through flags.

Rules:

1. `extra_args` must be stored and processed as argv tokens, not raw shell text.
2. shell concatenation is forbidden.
3. dangerous/conflicting flags should be rejected.
4. first-class structured fields win over conflicting pass-through args.

This provides an escape hatch without turning the feature into arbitrary command execution.

## 8. Capability Model

### 8.1 Capability Sources

Track three capability layers:

- `declared_capabilities`
  - operator says this instance is intended for chat, embeddings, vision, audio, or tool use
- `probed_capabilities`
  - health and lightweight runtime probes can confirm support or availability
- `effective_capabilities`
  - routing is allowed to use only what survives policy and probe validation

### 8.2 Why This Split Is Required

`/v1/models` alone is not enough to prove embeddings or multimodal support.

Therefore:

- some capabilities can be probed
- some capabilities must be operator-declared
- some capabilities should remain “unknown” until explicitly validated

Routing should reject requests that require unsupported or unverified capabilities rather than silently failing deeper in the stack.

## 9. Lifecycle Operations And Jobs

Lifecycle operations can take long enough to exceed normal request budgets.

Use Jobs for:

- start
- restart
- likely probe
- possibly stop when remote cleanup is slow

Reasoning:

- model startup can be slow
- SSH reachability can hang or degrade
- readiness and capability checks can take time
- job-backed operations provide better operator visibility and retry behavior

The instance API should return job metadata and current state summaries instead of forcing long synchronous waits.

This follows the repository guidance that new user-visible/admin-visible operational work should default to Jobs.

### 9.1 Worker Contract

Jobs-backed lifecycle support also requires a dedicated worker entrypoint for the `vllm_management` domain.

That worker should:

- use `WorkerSDK` with a `vllm_management` domain/queue contract
- dispatch `start`, `stop`, `restart`, and `probe` handlers
- update repository state transitions and persisted errors
- share the same repository, command-builder, executor, and reconciler abstractions as the API layer

Enqueuing lifecycle jobs without a corresponding worker is explicitly out of scope for an acceptable implementation.

## 10. API Surface

### 10.1 Instance CRUD

- `POST /api/v1/llm/providers/vllm/instances`
- `GET /api/v1/llm/providers/vllm/instances`
- `GET /api/v1/llm/providers/vllm/instances/{instance_id}`
- `PATCH /api/v1/llm/providers/vllm/instances/{instance_id}`
- `DELETE /api/v1/llm/providers/vllm/instances/{instance_id}`

### 10.2 Lifecycle

- `POST /api/v1/llm/providers/vllm/instances/{instance_id}/start`
- `POST /api/v1/llm/providers/vllm/instances/{instance_id}/stop`
- `POST /api/v1/llm/providers/vllm/instances/{instance_id}/restart`
- `POST /api/v1/llm/providers/vllm/instances/{instance_id}/probe`

These endpoints should enqueue Jobs and return:

- instance identifier
- requested action
- job identifier
- current observed state snapshot

### 10.3 Default Routing

- `POST /api/v1/llm/providers/vllm/default`

This endpoint sets or clears the managed default `vLLM` instance used when a request targets `provider="vllm"` without `provider_instance_id`.

### 10.4 Request Contract Extension

Prefer a generic request field:

- `provider_instance_id`

instead of a `vllm`-specific selector.

Initial behavior:

- meaningful for `provider="vllm"`
- ignored or rejected for providers that do not support managed instances

This avoids schema churn if other providers gain managed-instance routing later.

For request types that already distinguish provider and model, the managed-instance extension must preserve that explicit provider selection. In practice, embeddings request schemas should add a `provider` field alongside `provider_instance_id` rather than relying on model-only heuristics.

## 11. Data Model

### 11.1 Persistent Instance Record

Suggested durable fields:

- `instance_id`
- `name`
- `description`
- `tags`
- `execution_mode`
- `transport_config_json`
- `launch_spec_json`
- `routing_policy_json`
- `declared_capabilities_json`
- `desired_state`
- `observed_state`
- `last_known_base_url`
- `last_probe_at`
- `last_probe_summary_json`
- `last_error`
- `executor_metadata_json`
- `created_at`
- `updated_at`

### 11.2 Runtime Observation Fields

Observation-only metadata may include:

- local pid
- remote pid
- launcher handle id
- log path or log locator
- last health latency
- recent readiness failure reason

These should be persisted where useful for operator debugging, but must be treated as observed state rather than configuration truth.

## 12. Error Handling

The control plane should produce explicit, instance-scoped failures for:

- invalid launch configuration
- unknown instance id
- conflicting route/default selection
- unreachable SSH host
- failed remote launcher handshake
- port already in use
- readiness timeout
- direct endpoint not reachable from `tldw_server`
- capability mismatch for embeddings or multimodal requests
- unhealthy or failed target instance

Inference-side routing errors should clearly distinguish:

- instance not found
- instance not healthy
- instance not reachable
- instance lacks required capability

## 13. Security

- Management endpoints remain admin-only.
- Secrets must be stored or referenced securely and redacted from logs and status payloads.
- Command building must remain argv-based and never fall back to raw shell command strings.
- SSH mode should prefer stored connection profiles or secret references rather than inline credentials.
- `extra_args` must be constrained and validated.
- Future agent mode must use explicit host registration and authenticated control-plane requests.

## 14. Validation And Testing

### 14.1 Unit Tests

- instance registry CRUD and state transitions
- command builder flag mapping and conflict resolution
- constrained `extra_args` validation
- request-scoped instance resolution
- effective capability computation
- executor contract normalization

### 14.2 Integration Tests

- create/list/update/delete instance APIs
- start/restart/stop/probe endpoints returning Jobs metadata
- default-instance routing
- explicit `provider_instance_id` routing
- unhealthy-instance rejection paths
- SSH executor mocking with remote launcher contract validation

### 14.3 Contract Tests

- chat requests routed through a managed `vLLM` instance
- embeddings requests routed only to effective embeddings-capable instances
- multimodal requests routed only to effective multimodal-capable instances
- concurrent requests targeting different managed `vLLM` instances without global-config leakage

## 15. Rollout Plan

### Phase 1

- durable instance registry
- local and SSH executors
- structured command builder
- Jobs-backed lifecycle endpoints
- request-scoped routing into the existing `vllm` adapter
- direct-reachability enforcement

### Phase 2

- agent executor implementation
- richer capability probing
- improved logs/metrics surface
- optional restart policies and more advanced routing controls

## 16. Recommended Next Step

Write an implementation plan that stages the work in this order:

1. persistent instance model and registry
2. request-scoped routing and adapter integration
3. local executor and local lifecycle APIs
4. SSH executor with remote launcher contract
5. Jobs-backed orchestration and reconciler
6. capability enforcement, tests, and docs
