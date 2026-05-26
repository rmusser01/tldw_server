# llama.cpp Managed Runtime Roadmap Design

Status: Draft
Date: 2026-05-16
Owner: Core/WebUI maintainers
Scope: Multi-instance llama.cpp management, local model import/register workflows, durable instance profiles, supervisor behavior, and model-family/mmproj support
Tracking: TASK-397
Reference: https://github.com/m94301/llama-studio

## Summary

Expand the merged llama.cpp WebUI management work from a single managed
`llama-server` launcher into a backend-owned managed runtime for self-hosted
users. The runtime should support multiple named llama.cpp instance profiles,
local model asset discovery/import, durable supervised services, and
multimodal/model-family metadata without creating a parallel client-owned
runtime system.

The design intentionally borrows the useful product ideas from llama-studio:
stable model-on-port services, visible GPU/runtime status, saved launch
configuration, per-model logs, and searchable llama-server options. It adapts
those ideas to tldw_server's architecture by making the backend the source of
truth for paths, launch policy, process state, provider wiring, and future API
clients.

The first delivery should not attempt remote model downloads or a full model
marketplace. Local import/register/discovery comes first. Remote download,
curated catalogs, and Hugging Face workflows can be added after the runtime has
a stable asset and profile model.

## Goals

- Manage multiple llama.cpp services from tldw's backend and WebUI.
- Store durable instance profiles that represent desired services, not just UI
  presets.
- Support supervised lifecycle behavior: autostart, health checks,
  restart-on-crash, retry limits, and visible failure state.
- Preserve the current V1 single-server endpoints as compatibility wrappers
  around a default instance profile.
- Expand local model inventory into asset discovery for GGUF files, imported
  folders, and multimodal projectors.
- Support model-family modes across chat, vision, embeddings, rerank, and
  generic llama-server use cases.
- Keep warnings-first hardware guidance while hard-blocking unsafe or invalid
  requests.
- Reuse existing tldw provider/model metadata surfaces so Chat, Knowledge, and
  future clients can reason about local capabilities.

## Non-Goals

- No remote model download workflow in the first runtime roadmap slice.
- No curated model marketplace or recommendation catalog in the first slice.
- No promise of perfect GPU scheduling across NVIDIA, Metal, CPU-only, and
  mixed-device hosts.
- No frontend-owned process manager.
- No silent provider rewiring during profile start.
- No hard failure solely because an advisory VRAM estimate predicts a poor fit.
- No support for arbitrary shell commands; all runtime execution remains scoped
  to configured llama.cpp binaries and validated launch arguments.
- No non-admin user-facing launch controls in the first roadmap stages.

## Existing Context

The merged V1 design introduced a safer single-server admin surface:

- `GET /api/v1/llamacpp/config`
- `PUT /api/v1/llamacpp/config`
- `POST /api/v1/llamacpp/validate`
- `GET /api/v1/llamacpp/inventory`
- `POST /api/v1/llamacpp/models/register-path`
- `POST /api/v1/llamacpp/start-by-model`
- `POST /api/v1/llamacpp/use-in-chat`
- `GET /api/v1/llamacpp/logs/tail`
- `GET /api/v1/llamacpp/hardware`

That work deliberately kept one managed server process and deferred
multi-session management, model downloads/imports, profile storage, and mmproj
pairing. This roadmap is the follow-on design for those deferred capabilities.

The compatibility requirement is important: existing clients that call
`/api/v1/llamacpp/status`, `/start_server`, `/stop_server`, `/start-by-model`,
`/logs/tail`, or `/use-in-chat` should continue to work. Internally, those
endpoints can target a generated or migrated default profile.

## Architecture

The new system has three backend-owned layers.

### Model Asset Inventory

The inventory layer discovers local assets and describes what they appear to
support. It should handle:

- GGUF model files from configured model roots.
- Explicitly registered local GGUF paths.
- Imported local folders under allowlisted roots.
- Companion files such as `mmproj` projectors.
- Unknown or partially parsed assets with warnings instead of whole-scan
  failure.

Inventory metadata is best effort. It can use filename parsing, file stats,
safe GGUF header parsing when available, and known naming conventions. It
should not require a heavy parser before the admin UI can show local files.

### Instance Profile Registry

The registry stores durable desired services. A profile is not merely a saved
launch form; it is the service definition the supervisor reconciles.

Each profile has a stable ID, user-facing name, selected model assets, mode,
host/port, launch arguments, resource policy, health policy, restart policy,
autostart setting, provider alias, and tags.

### Supervisor Runtime

The supervisor owns observed process state. It starts, stops, probes, restarts,
and reports `llama-server` processes according to the registry. It exposes
runtime state separately from profile configuration so the UI can distinguish
"defined", "starting", "healthy", "failed", "stopped", and "paused".

The supervisor should be conservative: it records failures, exposes logs, and
respects retry limits instead of looping indefinitely.

### Persistence Recommendation

Use a dedicated backend-owned profile store rather than expanding `config.txt`
into a multi-profile database. A small SQLite table or JSON document under the
tldw config/data directory is acceptable for the first implementation, but the
contract should be repository/service-shaped so it can move later without
changing API payloads.

`config.txt` should remain the bootstrap source for global llama.cpp settings:
enabled state, executable path, default model roots, safe path allowlist, raw
argument policy, secret policy, default host/port, and log location. Profiles
then reference those global settings and add per-instance desired state.

This avoids turning a comment-preserving config file into a concurrent runtime
state store while preserving the existing setup path.

### Access Control And Scope

Managed llama.cpp profiles are host-level admin resources. In the first stages,
all asset, profile, lifecycle, log, and supervisor endpoints should require the
same admin permission posture as the existing lifecycle endpoints.

Profiles should be deployment-global rather than per-user. Provider wiring is
the boundary where single-user and multi-user behavior matters:

- single-user mode may update the configured llama.cpp provider endpoint as V1
  does today;
- multi-user mode should avoid silently changing every user's provider
  settings and should either require an admin-level global provider action or
  expose the selected profile as an explicit local provider alias.

Logs and runtime state may expose local paths, process IDs, ports, and model
names, so they should stay admin-only unless a later design introduces a
redacted read-only user view.

## Core Data Model

### `LlamaCppAsset`

Represents a local file or group discovered by inventory.

Required fields:

- `asset_id`
- `kind`: `gguf`, `mmproj`, `folder`, `unknown`
- `identity_basis`: `resolved_path`, `content_hash`, or `manual`
- `path`
- `resolved_path`
- `display_name`
- `source`: `models_dir`, `registered_path`, `imported_folder`
- `size_bytes`
- `modified_at`
- `metadata`
- `capabilities`
- `warnings`

For multimodal families, a base GGUF can include candidate
`mmproj_asset_ids`. A projector can also point back to candidate base models.
Pairing is explicit when a profile selects both assets.

Asset IDs should be stable enough for saved profiles but honest about local
files. A path-derived ID is acceptable initially if it uses canonical resolved
paths after allowlist checks. If content hashing is later added, it should
augment identity rather than breaking existing profile references. Symlinks
must be resolved before allowlist checks so an apparent allowed path cannot
escape its root.

### `LlamaCppInstanceProfile`

Represents desired service state.

Required fields:

- `profile_id`
- `name`
- `enabled`
- `mode`: `chat`, `vision`, `embedding`, `rerank`, `server_generic`
- `model_asset_id`
- `mmproj_asset_id`
- `host`
- `port`
- `port_policy`
- `server_args`
- `device_policy`
- `resource_policy`
- `health_policy`
- `restart_policy`
- `autostart`
- `provider_alias`
- `tags`

`mmproj_asset_id` is nullable and is required only for profiles whose selected
mode or asset metadata requires a projector.

`port_policy` should distinguish stable explicit ports from autoselected ports.
Stable explicit ports are the default because self-hosted users often want
fixed endpoints for other tools. Autoselection can be supported, but the
runtime should report the assigned port clearly and must not let two enabled
profiles silently fight over the same explicit port.

### `LlamaCppInstanceRuntime`

Represents observed state.

Required fields:

- `profile_id`
- `state`
- `pid`
- `endpoint`
- `resolved_args`
- `started_at`
- `last_health_at`
- `restart_count`
- `next_restart_at`
- `exit_code`
- `last_error`
- `log_tail_available`
- `warnings`
- `health`

Runtime records may be in memory initially, but the last observed failure state
should be durable enough to survive a tldw restart when supervision is enabled.

## API Shape

The new API should expose profile and runtime concepts directly:

```text
GET    /api/v1/llamacpp/assets
POST   /api/v1/llamacpp/assets/register-path
POST   /api/v1/llamacpp/assets/import-folder

GET    /api/v1/llamacpp/profiles
POST   /api/v1/llamacpp/profiles
GET    /api/v1/llamacpp/profiles/{profile_id}
PUT    /api/v1/llamacpp/profiles/{profile_id}
DELETE /api/v1/llamacpp/profiles/{profile_id}

POST   /api/v1/llamacpp/profiles/{profile_id}/start
POST   /api/v1/llamacpp/profiles/{profile_id}/stop
POST   /api/v1/llamacpp/profiles/{profile_id}/pause
POST   /api/v1/llamacpp/profiles/{profile_id}/resume
POST   /api/v1/llamacpp/profiles/{profile_id}/use-in-chat

GET    /api/v1/llamacpp/instances
GET    /api/v1/llamacpp/instances/{profile_id}
GET    /api/v1/llamacpp/instances/{profile_id}/logs/tail
```

Compatibility wrappers should remain:

- V1 `status` returns the default profile runtime.
- V1 `start-by-model` creates or updates the default profile and starts it.
- V1 `stop_server` stops the default profile.
- V1 `logs/tail` tails the default profile logs.
- V1 `use-in-chat` wires the default running profile.

This lets the WebUI migrate incrementally and avoids breaking existing API
clients while the runtime architecture changes underneath.

## Migration And Compatibility

The first implementation should synthesize or create a reserved default profile
from the existing `[LlamaCpp]` config when no profile store exists. This profile
represents the current single managed server.

Compatibility rules:

- existing V1 endpoints target the reserved default profile;
- existing config endpoints continue to edit global bootstrap/default settings;
- starting by model through V1 updates the default profile's model and launch
  args for that run;
- stopping through V1 stops only the default profile;
- V1 status should not hide other managed profiles, but it should preserve the
  current response shape for clients that expect one active llama.cpp server;
- the new multi-profile endpoints become the authoritative way to inspect all
  managed profiles and runtimes.

Implementation should keep the default profile visibly named in the Admin UI so
users understand why legacy controls and new profile controls affect the same
service.

## Lifecycle Semantics

Profiles declare desired behavior. Runtime reports observed behavior.

- `enabled=true` means the profile may run.
- `autostart=true` means the supervisor should start it when tldw starts.
- Restart policy controls crash recovery.
- Stop actions should distinguish "stop now" from "disable and stop".
- Pause should stop reconciliation without deleting the profile.
- Resume should re-enable reconciliation according to the profile policy.

Health checks should layer from cheap to expensive:

1. Process alive check.
2. Port listener check.
3. OpenAI-compatible `/v1/models` or llama-server health probe when available.
4. Optional smoke request only when the profile opts in.

Failed starts and failed health checks must remain visible with bounded logs,
exit codes, retry counts, and the resolved command after secret redaction.

Lifecycle operations need per-profile locking. Concurrent start, stop, pause,
resume, and profile update requests for the same profile should serialize so
the supervisor cannot race itself into duplicate processes or stale runtime
state. Cross-profile operations may proceed concurrently only when they do not
share explicit ports or mutable global resources.

On tldw shutdown, the supervisor should gracefully stop processes it owns. On
startup, it should not blindly assume old PIDs still belong to it. If later
orphan adoption is added, it must verify command line, endpoint, and profile
identity before taking ownership.

## Safety Policy

The runtime continues V1's safety posture.

Hard failures:

- invalid or non-allowlisted paths
- path traversal attempts
- missing executable or missing model file
- duplicate profile IDs
- invalid host or port
- duplicate enabled explicit host/port combinations
- unsafe raw arguments when unvalidated args are disabled
- CLI secrets when CLI secrets are disabled
- mmproj path outside allowed roots
- profile mode and structured launch arguments that conflict with each other

Warnings:

- likely VRAM/RAM pressure
- unknown asset capability
- inferred rather than proven mmproj pairing
- recoverable port conflicts when `port_policy` allows autoselection
- unsupported or unknown llama-server arguments
- stale imported path
- provider alias conflict
- health probe unavailable

Hardware and GPU scheduling should start advisory. Device policies may express
intent, but the first runtime stages should not claim perfect packing across
platforms. Users should be able to launch experimental configurations after
seeing warnings.

Mode-specific structured settings should own the flags that make a server
behave like chat, vision, embedding, or rerank. Raw custom args must not be able
to override reserved structured flags unless the backend explicitly allows it
and surfaces the override in warnings.

## Model Import And Asset Workflow

The first acquisition workflow is local only:

- rescan configured model roots
- register an allowlisted file path
- import/register an allowlisted folder
- show stale path warnings
- show path and capability warnings
- pair/unpair projector candidates

In this first workflow, "import folder" means register and scan an existing
local folder under an allowlisted root. It does not mean upload, copy, move, or
download model files.

Remote downloads are deferred. This avoids mixing long-running network jobs,
credential policy, partial-file cleanup, disk quota handling, and model trust
metadata into the same slice as process supervision.

The eventual download system should feed the same `LlamaCppAsset` inventory
contract rather than creating a separate model catalog.

## Future Download Workflow

Remote downloads are intentionally later-stage work, but the roadmap should
reserve the shape now. Downloads should be implemented as managed acquisition
jobs that produce local assets only after the file is complete and validated.

Future download requirements:

- support direct URLs and selected model-hosting sources through explicit user
  action;
- show size, destination, license/trust metadata when available, and disk-space
  warnings before starting;
- write partial files to a temporary location and atomically register completed
  assets;
- support cancellation, retry, checksum validation when available, and cleanup
  of incomplete files;
- respect the same allowlisted destination policy as local imports;
- never make downloaded assets executable policy by themselves; users still
  create or update profiles explicitly.

This keeps acquisition, inventory, and runtime supervision connected without
making the first multi-instance milestone depend on network jobs.

## Multimodal And Model-Family Support

The roadmap should model more than text chat from the beginning, but stage the
implementation.

Profile `mode` provides the first explicit model-family contract:

- `chat`
- `vision`
- `embedding`
- `rerank`
- `server_generic`

Vision profiles can select both a base GGUF and an mmproj asset. The initial
behavior should prove launch compatibility and provider metadata first. Richer
Chat image workflows can then route through profiles that expose vision input.

Embedding and rerank modes should not be wedged into chat-only metadata. They
should surface through `/api/v1/llm/models/metadata` with modalities such as
`input=["text"]`, `output=["embedding"]`, and capability flags that existing
Knowledge and provider-selection flows can consume.

## WebUI Workflow

The `/admin/llamacpp` page should become a compact operations console with five
areas.

### Readiness

Binary path, model roots, allowlisted import paths, detected llama-server
version, parsed server options, and config/runtime warnings.

### Assets

GGUF, mmproj, and imported folders. Users can rescan, register a file, import a
folder, inspect inferred capabilities, and pair/unpair projectors.

### Profiles

Durable instance definitions. Users can create a profile from an asset,
duplicate a profile, assign mode, choose port, choose device/resource policy,
choose autostart/restart behavior, and preview resolved launch arguments.

### Runtime

Running, stopped, starting, failed, and paused profiles with endpoint, PID,
health, restart count, log tail, and actions.

### Advanced Args Browser

Use the llama-studio idea of parsing `llama-server --help` into a searchable
option browser. The browser helps users add supported flags, but backend
validation remains authoritative.

## Provider And Routing Behavior

The first profile wiring should stay explicit. Starting a profile does not
silently change Chat settings.

`use-in-chat` should wire a selected running profile into the current llama.cpp
provider settings or a named provider alias. Later routing can expose multiple
managed profiles as separate local provider entries.

Provider/model metadata should use existing `/api/v1/llm/models/metadata`
patterns. Managed profiles can appear as configured local models with
capability and modality metadata derived from profile mode and asset metadata.

## Staged Delivery Plan

### Stage 1: Profiles And Supervisor Contract

Add the profile registry, runtime state model, default-profile migration, V1
compatibility wrappers, profile CRUD, manual start/stop, log tailing, health
state, and backend tests.

Success means the backend can manage more than one durable profile without
requiring the WebUI to own process rules.

### Stage 2: Autostart And Health Recovery

Add startup reconciliation, restart policies, crash tracking, retry limits,
pause/resume, and visible failed-supervision state.

Success means a self-hosted user can configure stable local services that come
back after tldw restarts or llama-server crashes.

### Stage 3: Asset Inventory V2

Expand inventory into assets, imported folders, mmproj discovery, candidate
pairing, capability inference, and stale-path handling.

Success means profiles can be created from local assets and local multimodal
pairs without remote downloads.

### Stage 4: Model-Family Modes

Formalize chat, vision, embedding, rerank, and generic server modes. Surface
managed profile metadata through `/api/v1/llm/models/metadata`.

Success means Chat and Knowledge can distinguish local profile capabilities
without hardcoded frontend assumptions.

### Stage 5: Admin Console UX

Migrate the WebUI from a single-server page to readiness/assets/profiles/runtime
panels. Keep the old single-server flow working through the default profile.

Success means self-hosted users can discover files, define services, supervise
them, and connect selected profiles to tldw Chat from one page.

### Stage 6: Routing And Advanced Workflows

Add multiple provider aliases, stable external endpoint presentation, optional
route selection, and remote download/catalog workflows built on the asset
contract.

Success means tldw can act as the control plane for local llama.cpp services
used by both tldw and external tools.

## Testing Strategy

Backend unit tests:

- profile validation
- path allowlisting
- asset ID stability
- launch argument resolution and redaction
- V1 default-profile migration
- supervisor state transitions
- autostart/restart retry limits
- mmproj candidate pairing
- provider alias conflict handling

API tests:

- asset register/import/list
- profile CRUD
- start/stop/pause/resume
- runtime status
- log tailing
- `use-in-chat`
- V1 wrappers against the default profile

Frontend tests:

- profile creation and duplication
- runtime state rendering
- warning display
- args preview
- mmproj pairing controls
- `use-in-chat` action behavior

E2E tests:

- start two profiles on distinct ports
- verify both appear in runtime state
- stop one profile without affecting the other
- verify warnings are shown rather than hard-blocking advisory resource risks
- verify a duplicate explicit port is rejected or forced through a documented
  autoselect policy
- verify legacy V1 status/start/stop still targets only the default profile

Verification should also include `git diff --check` and Bandit for touched
backend code when implementation begins. This design-only task does not require
Bandit beyond documenting that no Python code was changed.

## Open Questions For Implementation Planning

- Should supervision run in the existing API process only, or should it become a
  service abstraction that can later move to a worker?
- How much of the previous single-server handler can be reused as an
  instance-level process adapter before it becomes clearer to replace it?
- What provider alias shape best fits existing multi-user and single-user
  config paths?
- Which llama-server flags should become first-class structured fields versus
  remaining validated custom args?
