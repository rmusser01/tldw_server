# llama.cpp Server Management WebUI Design

Status: Draft
Date: 2026-05-15
Owner: Core/WebUI maintainers
Scope: Self-hosted llama.cpp setup and single-server management from the WebUI
Tracking: TASK-361
Reference: https://github.com/m94301/llama-studio

## Summary

Improve the existing `/admin/llamacpp` experience so a self-hosted user can
configure, validate, start, stop, inspect, and wire a single managed
`llama-server` process without editing config files by hand.

The first version stays aligned with the current tldw_server architecture: one
managed llama.cpp server process, explicit admin controls, backend-owned safety
checks, and no automatic provider rewiring. Ideas from llama-studio should be
used selectively: guided setup, model inventory, option discoverability, process
logs, and hardware-fit warnings. Multi-session management, GPU scheduling, and
download/import workflows are out of scope for this design.

The most important UX rule is honesty about state. The WebUI must distinguish
saved configuration from active runtime configuration because updating
`config.txt` does not necessarily mutate an already-created
`LlamaCppHandler`. When a change requires an API server restart or a fresh
llama.cpp launch, the page must say so directly.

## Goals

- Let self-hosted admins configure the llama.cpp binary path, model locations,
  launch defaults, and safe path allowlist from the WebUI.
- Keep the current single-server lifecycle model: start, stop, status, metrics,
  and health checks for one managed `llama-server` process.
- Show a model inventory for GGUF files found in configured directories and
  explicitly registered local file paths.
- Warn about likely hardware-fit problems without hard-blocking experimental
  launches.
- Make provider wiring explicit: after a successful managed launch, offer a
  "Use this in Chat" action instead of silently changing chat provider config.
- Preserve the existing backend safety boundary for paths, command-line
  arguments, redaction, and admin-only controls.
- Keep existing lifecycle endpoints compatible where possible.

## Non-Goals

- No multi-session llama.cpp management in V1.
- No GPU scheduler, per-GPU allocation planner, or concurrent process manager.
- No model downloads, uploads, remote imports, or registry browsing.
- No generic config editor for arbitrary sections.
- No automatic chat provider rewiring during server start.
- No hard block based only on VRAM/RAM estimates.
- No new requirement that `llama-server --help` output becomes executable
  policy.

## Existing Project Context

The backend already has a managed llama.cpp integration:

- `tldw_Server_API/app/core/Local_LLM/LlamaCpp_Handler.py` owns one active
  `llama-server` process, validates paths and safe launch arguments, redacts
  command output, probes readiness, and supports rollback on failed swaps.
- `tldw_Server_API/app/core/Local_LLM/LLM_Inference_Schemas.py` defines
  `LlamaCppConfig`, including executable path, models directory, host, port,
  default threads, GPU layers, context size, allowlist paths, port autoselect,
  and log file settings.
- `tldw_Server_API/app/api/v1/endpoints/llamacpp.py` exposes lifecycle,
  status, metrics, model listing, inference, and reranking endpoints. Lifecycle
  endpoints already require admin permissions.
- `tldw_Server_API/app/core/config.py` reads `[LlamaCpp]` config and supports
  cache refresh helpers.
- `tldw_Server_API/app/core/Setup/setup_manager.py` contains the safer
  comment-preserving config update path used by setup flows.

The WebUI already has an admin llama.cpp page:

- `apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx`
  includes status, model selection, start/stop controls, structured settings,
  preset import/export, and raw extra flags.
- `apps/packages/ui/src/utils/build-llamacpp-server-args.ts` converts
  structured UI settings into backend server arguments.
- `apps/packages/ui/src/utils/gguf-model-metadata.ts` extracts basic metadata
  from filenames only.

This design should reshape and extend those surfaces instead of creating a
parallel llama.cpp management system.

## Design Choices

### Single Managed Server

The managed llama.cpp feature remains a single-server console. The WebUI should
show one active process, one active model, one host/port, and one lifecycle
state. If users need multiple simultaneous llama.cpp servers, they can still run
external processes and point tldw_server at them through provider config.

This keeps the V1 design compatible with the current handler and avoids
duplicating llama-studio's broader session manager before the backend has a
process model for it.

### Backend-Owned Admin Facade

Add a narrow admin facade around the existing llama.cpp lifecycle API. The
facade centralizes setup, inventory, validation, and provider wiring so the
frontend does not need to infer server rules from scattered config and status
endpoints.

Candidate endpoints:

```text
GET  /api/v1/llamacpp/config
PUT  /api/v1/llamacpp/config
POST /api/v1/llamacpp/validate
GET  /api/v1/llamacpp/inventory
POST /api/v1/llamacpp/models/register-path
POST /api/v1/llamacpp/start-by-model
POST /api/v1/llamacpp/use-in-chat
GET  /api/v1/llamacpp/logs/tail
```

Existing endpoints such as `start_server`, `stop_server`, `status`, `metrics`,
and `models` should remain compatible. Newer WebUI flows can prefer
`start-by-model`, which accepts a stable inventory `model_id` rather than a raw
filename.

All new endpoints are admin-only and should use the same permission posture as
existing lifecycle controls.

### Saved Config vs Active Runtime

`GET /api/v1/llamacpp/config` must expose both saved and active state:

```json
{
  "saved_config": {
    "enabled": true,
    "executable_path": "vendor/llama.cpp/server",
    "models_dir": "models/gguf_models",
    "host": "127.0.0.1",
    "port": 8080,
    "default_ctx_size": 4096,
    "default_threads": 8,
    "default_n_gpu_layers": 0,
    "allowed_paths": ["models/gguf_models"]
  },
  "active_config": {
    "handler_configured": true,
    "enabled": true,
    "executable_path": "vendor/llama.cpp/server",
    "models_dir": "models/gguf_models",
    "host": "127.0.0.1",
    "port": 8080
  },
  "restart_required": false,
  "restart_reasons": [],
  "env_overrides": {
    "executable_path": false,
    "models_dir": false,
    "port": false
  }
}
```

Config writes must use the existing setup/config update path so comments and
known key validation are preserved. After a successful write, config caches
should be refreshed. The response must still report whether the active handler
will observe the new values immediately.

Conservative V1 restart semantics:

- changing `enabled`, `executable_path`, `models_dir`, `allowed_paths`, or
  managed log path requires an API server restart if the handler has already
  been constructed or omitted during startup;
- changing saved launch defaults affects future launches only after the backend
  observes the refreshed config;
- launch arguments passed directly to a start request apply to that launch
  immediately;
- environment-variable-controlled fields should be shown as locked or
  overridden, with the winning source visible.

### Model Inventory and Identity

Add a backend inventory service for GGUF models. It should scan the configured
models directory and explicit registered local paths, then return stable model
identities. The inventory service becomes the resolver between user-facing model
rows and safe absolute paths.

Inventory rules:

- recursively scan configured model directories for `*.gguf` files, with bounded
  traversal and result limits;
- skip likely multimodal projector files such as `mmproj*.gguf` unless a future
  UI explicitly supports pairing them;
- include top-level file stats always: size, modified time, basename, directory,
  and path status;
- include filename-derived metadata always when possible: quantization, rough
  parameter family, and display name;
- include GGUF header metadata only when a lightweight parser is available and
  safe to run;
- never require a new heavy GGUF parser dependency just to render the basic V1
  inventory;
- return warnings instead of failing the whole inventory when individual paths
  are unreadable or outside the allowlist.

Registered external paths must not overload the existing `model_filename`
contract. The new flow should start by `model_id`, and the backend resolves that
ID to a validated path under `models_dir` or `allowed_paths`.

Example inventory item:

```json
{
  "model_id": "sha256-path-prefix-or-db-id",
  "display_name": "Qwen3 8B Q4_K_M",
  "basename": "qwen3-8b-q4_k_m.gguf",
  "source": "models_dir",
  "path": "/redacted/or/admin-visible/path/qwen3-8b-q4_k_m.gguf",
  "size_bytes": 4920000000,
  "modified_at": "2026-05-15T10:00:00Z",
  "metadata": {
    "quantization": "Q4_K_M",
    "parameter_hint": "8B",
    "context_hint": null
  },
  "warnings": []
}
```

### Launch Defaults and Per-Model Profiles

Global `[LlamaCpp]` config remains the source for handler setup and default
launch values. Per-model usability should not be forced into `config.txt`.

Required V1 should not add new server-side launch profile storage. It should
preserve the current import/export preset workflow and use the stable
`model_id` inventory contract so a later slice can add last-used settings or
named profiles without changing how models are addressed.

If a later launch-profile slice is approved, per-model launch state stores UI
launch preferences only: context size, GPU layers, threads, cache choices,
batching, flash attention, and advanced flags. It must not store secrets or
bypass backend argument validation.

### Hardware Snapshot and Fit Warnings

Add a best-effort hardware snapshot endpoint or include hardware data in the
inventory response. It should collect only local capacity signals useful for
launch guidance:

- system RAM total and available;
- CPU core/thread count;
- GPU name, total VRAM, free VRAM, and driver/runtime hints when available;
- probe availability and errors as structured warnings.

The probe must be optional and cross-platform friendly. Use Python/library paths
already available where possible. GPU details can use optional NVML when present
or a sanitized `nvidia-smi` fallback if the project already accepts that pattern.
macOS, Windows, and non-NVIDIA Linux hosts must degrade to "unknown" instead of
failing setup.

VRAM/RAM estimates are advisory. The UI should warn when a selected model and
launch settings look risky, but it must allow the user to start anyway.

### Option Discovery

Borrow llama-studio's useful idea of parsing `llama-server --help`, but keep the
backend allowlist as the execution authority.

The WebUI may show an option browser with:

- supported structured options already known to tldw_server;
- backend-allowed extra arguments;
- parsed `llama-server --help` options marked as informational or unsupported
  until the backend allowlist accepts them.

Dynamic help parsing must not let new flags bypass the existing denylist,
allowlist, path validation, or CLI secret checks.

### Provider Wiring

After a managed server reaches healthy state, the WebUI should show a clear
"Use this in Chat" action.

`POST /api/v1/llamacpp/use-in-chat` should:

- confirm the managed process is running and ready;
- normalize the provider endpoint to the form expected by the existing local
  provider adapter, such as `http://127.0.0.1:8080` or another agreed base URL;
- update only the llama.cpp provider endpoint field, currently the local API
  `llama_api_IP` setting, unless a future provider schema explicitly adds a
  model field;
- refuse or warn when environment overrides mean the saved config will not win;
- refresh relevant config/provider caches after saving;
- optionally probe `/v1/models` and return the discovered served model names.

Starting a server must not silently change chat behavior. Provider wiring is a
separate explicit action.

### WebUI Workflow

The WebUI should become a guided admin console with three primary areas.

1. Readiness
   - Shows enabled/disabled state, binary path validity, active vs saved config,
     restart requirement, provider wiring status, and recent process health.
   - Offers direct fixes where safe: save config, validate binary, rescan
     models, restart guidance.

2. Inventory
   - Lists discovered GGUF models and registered local model paths.
   - Shows size, metadata hints, source, warnings, and current active model.
   - Provides actions: rescan, register local path, select for launch, show
     launch history/defaults.

3. Launch
   - Keeps common settings visible: context size, GPU layers, threads, host,
     port, and cache/batching basics.
   - Keeps advanced flags collapsed/searchable.
   - Shows hardware-fit warnings and path/port warnings before start.
   - Starts the selected model, displays readiness progress, and then offers
     "Use this in Chat".

The existing advanced options should not disappear; they should be reorganized
so first-time local users can complete the normal path without understanding
every llama.cpp flag.

### Logs

The UI may expose a bounded log tail for the managed process. The endpoint must
only read the configured managed llama.cpp log file or a backend-owned per-launch
log file. It must not accept arbitrary file paths.

Log responses should be bounded by line count or byte count, redact known
secrets, and be admin-only.

## Error States

The WebUI should render first-class states for:

- llama.cpp support disabled in config;
- saved config differs from active runtime config;
- API server restart required;
- invalid or missing binary;
- binary exists but validation fails;
- no GGUF models found;
- registered path rejected by allowlist;
- model file removed since inventory scan;
- port unavailable and autoselect disabled;
- autoselect changed the requested port;
- startup timeout or health check failure;
- process exited after launch;
- hardware probe unavailable;
- provider endpoint currently points somewhere else;
- provider config is locked by environment override.

Failures should point to the next useful action. For example, "No models found"
should offer to rescan, update the models directory, or register a local GGUF
path. "Restart required" should explain that the saved config changed but the
active API process has not reloaded the handler.

## Rollout Plan

Implement this as small reviewable slices:

1. Backend config facade
   - saved vs active config;
   - typed update path through setup manager;
   - validation endpoint;
   - restart-required semantics.

2. Inventory resolver
   - recursive bounded GGUF scan;
   - registered local path support;
   - stable model IDs;
   - start-by-model endpoint.

3. Provider wiring and logs
   - explicit use-in-chat endpoint;
   - cache refresh/probe behavior;
   - bounded managed log tail.

4. WebUI console reshape
   - readiness panel;
   - inventory table;
   - launch panel;
   - warnings-first hardware guidance;
   - explicit provider wiring prompt.

5. Deferred launch profiles
   - use the stable `model_id` contract to support a later last-used settings or
     named-profile slice without making profile storage part of required V1.

## Testing

Backend tests should cover:

- config facade returns saved and active state;
- config writes preserve comments and report restart-required cases;
- environment overrides are reported and not silently overwritten;
- validation handles missing binary, invalid binary, and valid binary help/version
  checks;
- inventory scans nested GGUF files with bounds and skips projector files;
- registered paths require allowlist approval and stable model IDs;
- start-by-model resolves IDs safely and preserves existing start endpoint
  compatibility;
- provider wiring updates only the intended llama.cpp endpoint and refreshes
  caches;
- log tail cannot read arbitrary files and enforces response bounds.

Frontend tests should cover:

- disabled/unconfigured readiness state;
- saved vs active config mismatch;
- restart-required messaging;
- empty inventory and path registration flows;
- warning-but-allow hardware guidance;
- start success followed by explicit "Use this in Chat";
- provider mismatch and environment-locked provider states;
- existing advanced launch settings still build expected server args.

End-to-end smoke coverage should validate the normal self-hosted path using
mocked backend responses: configure, validate, scan inventory, select model,
start, and confirm provider wiring.

## Acceptance Criteria

- `/admin/llamacpp` can guide an admin from unconfigured state to a healthy
  managed llama.cpp process without hand-editing config files.
- The page clearly distinguishes saved config, active runtime config, and
  restart-required changes.
- GGUF inventory supports the configured models directory and explicit local
  path registration without model downloads or uploads.
- Hardware guidance warns but does not block launch.
- Starting a server does not silently change chat provider behavior.
- "Use this in Chat" explicitly wires the running managed server to the
  llama.cpp provider endpoint and reports override/probe results.
- Backend path, argument, secret, and log safety checks remain authoritative.
- Existing llama.cpp lifecycle endpoints remain compatible.
