# llama.cpp Integration Modes

This project supports two distinct `llama.cpp` integration planes. They are both valid, and they are intentionally separate.

## Plane Overview

| Plane | Purpose | Primary Surface | State Owner | Typical Operator |
|---|---|---|---|---|
| managed plane | Manage a local `llama.cpp/server` process | `/api/v1/llamacpp/*` lifecycle endpoints | `LLMInferenceManager` + `LlamaCppHandler` managed process state | Admin/operator |
| provider plane | Route chat requests to an OpenAI-compatible `llama.cpp` endpoint | `provider=llama.cpp` chat adapter path | `llama_api` provider configuration + remote server state | App/client caller |

## Endpoint-to-Plane Mapping

| Endpoint/Path | Plane | Contract |
|---|---|---|
| `POST /api/v1/llamacpp/start_server` | managed plane | Starts or swaps managed local server model |
| `POST /api/v1/llamacpp/start-by-model` | managed plane | Starts managed local server from a stable inventory `model_id` |
| `POST /api/v1/llamacpp/stop_server` | managed plane | Stops managed local server process |
| `GET /api/v1/llamacpp/status` | managed plane | Returns managed process status |
| `GET /api/v1/llamacpp/config` and `PUT /api/v1/llamacpp/config` | managed plane | Reads and updates saved llama.cpp admin configuration |
| `POST /api/v1/llamacpp/validate` | managed plane | Validates a local `llama-server` binary path |
| `GET /api/v1/llamacpp/inventory` | managed plane | Lists bounded GGUF inventory with stable model IDs and metadata hints |
| `POST /api/v1/llamacpp/models/register-path` | managed plane | Legacy inventory-only path registration for local GGUF models |
| `GET /api/v1/llamacpp/assets` | managed plane | Lists GGUF, mmproj, and imported-folder assets for the Assets panel |
| `POST /api/v1/llamacpp/assets/register-path` | managed plane | Registers an allowed local GGUF or mmproj path for the Assets panel |
| `POST /api/v1/llamacpp/assets/import-folder/preview` and `POST /api/v1/llamacpp/assets/import-folder` | managed plane | Previews, then explicitly persists, an allowed local asset folder import |
| `GET/POST/DELETE /api/v1/llamacpp/assets/downloads*` | managed plane | Lists, queues, reads, or cancels Jobs-backed remote GGUF acquisition jobs |
| `GET /api/v1/llamacpp/models` | managed plane | Legacy model-file listing from managed-plane runtime |
| `GET /api/v1/llamacpp/hardware` | managed plane | Returns best-effort local RAM/GPU snapshot and warnings |
| `GET /api/v1/llamacpp/logs/tail` | managed plane | Tails the configured managed llama.cpp log with bounded line count |
| `POST /api/v1/llamacpp/use-in-chat` | bridge action | Explicitly points provider-plane `llama_api` chat endpoint at the running managed server |
| `POST /api/v1/llamacpp/inference` | managed plane | Runs inference against managed server context |
| `POST /api/v1/chat/completions` + `provider=llama.cpp` | provider plane | Sends request through provider adapter to configured `llama_api` endpoint |
| `GET/POST/PATCH/DELETE /api/v1/chat/grammars*` | provider plane support surface | Manages user-scoped saved GBNF grammars used by the provider-plane chat workflow |

## Critical Rule

No shared state is implied between the managed plane and provider plane.

Starting/stopping a managed server does not automatically rewrite provider-plane endpoint configuration, and provider-plane availability does not guarantee managed-plane process readiness.

## Managed Admin WebUI Flow

The `/admin/llamacpp` WebUI is an admin/operator surface for the managed plane. It guides the user through three panels:

1. **Readiness**: Shows saved config, active runtime config, environment overrides, and restart-required reasons. Saved config changes are persisted through `PUT /api/v1/llamacpp/config`, but the active handler only picks up some changes after the API server restarts.
2. **Inventory**: Shows GGUF models discovered from the configured models directory and registered local paths. The UI starts models by stable `model_id` through `POST /api/v1/llamacpp/start-by-model` instead of sending arbitrary absolute paths from the browser.
3. **Launch**: Preserves structured llama.cpp server options, preset import/export, and advanced flags. Hardware data from `GET /api/v1/llamacpp/hardware` is advisory; missing GPU/RAM probes produce warnings and should not disable launch by themselves.

After a managed server is running, the page shows **Use this in Chat**. This is intentionally explicit: it calls `POST /api/v1/llamacpp/use-in-chat` and updates the provider-plane `llama_api` endpoint only after the admin chooses it. Starting a managed server alone does not mutate chat provider settings.

## Managed Runtime Profiles

The managed plane now stores durable llama.cpp runtime profiles. A profile is a backend-owned service definition, not just a saved browser form. It records the profile ID, name, enabled state, mode, selected GGUF asset or model path, optional mmproj projector, host/port policy, structured server arguments, autostart setting, bounded restart policy, provider alias, and tags.

Profile state is persisted in `llamacpp_profiles.json` next to the active tldw config file resolved by setup configuration. Treat this as a server-owned file: use the Admin UI or `/api/v1/llamacpp/profiles*` endpoints instead of editing it while the API server is running.

The legacy one-server API remains a compatibility wrapper around the reserved `default` profile:

- `POST /api/v1/llamacpp/start_server`
- `POST /api/v1/llamacpp/start-by-model`
- `GET /api/v1/llamacpp/status`
- `GET /api/v1/llamacpp/logs/tail`
- `POST /api/v1/llamacpp/use-in-chat`

Those endpoints target only the default profile. Multi-profile operators should use `/api/v1/llamacpp/profiles` for desired service definitions and `/api/v1/llamacpp/instances` for observed runtime state across all profiles.

Profiles can reference assets discovered by the asset inventory or direct model paths that pass the configured allowlist. Local registration and folder import are the preferred ways to make files visible before profile creation. Remote download and future catalog workflows belong to the acquisition path; they do not automatically create profiles, start runtimes, or rewire Chat.

Vision profiles require an explicit mmproj pairing. Use `mmproj_model_id` for an inventory projector asset, or a backend-validated `server_args.mmproj` path. If both are supplied, they must resolve to the same projector file. The backend rejects missing or conflicting projector definitions while keeping resource-fit warnings advisory.

Autostart and restart policy are intentionally bounded. On startup, the reconciler starts only enabled profiles with `autostart=true`. Paused profiles are not restarted until resumed. Restart attempts honor the stored limit, and the profile records bounded last-failure metadata such as state, exit code, model path, and error summary instead of raw logs or environment.

## Model Acquisition And Import

The `/admin/llamacpp` **Assets** panel helps operators make local GGUF and mmproj files visible to the managed plane without changing runtime or provider-plane state by surprise.

Local acquisition has two paths:

1. **Register local asset path**: `POST /api/v1/llamacpp/assets/register-path` records one allowed local GGUF or mmproj file path. Use this when the asset already exists and only that file should be added to inventory.
2. **Import local asset folder**: `POST /api/v1/llamacpp/assets/import-folder/preview` scans a folder first and returns counts, warnings, candidate assets, and `will_persist=false`. The preview does not mutate config. `POST /api/v1/llamacpp/assets/import-folder` is the explicit confirmation step that persists the folder in the managed asset inventory.

Remote acquisition is Jobs-backed. `POST /api/v1/llamacpp/assets/downloads` validates a remote GGUF download request and creates an acquisition job. The worker downloads to a partial file, enforces size/checksum validation when requested, promotes the file atomically, then registers the completed asset. A queued or running download does not appear in normal asset inventory until the worker finishes validation and registration; the WebUI refreshes assets after it observes a newly completed download job.

Acquisition is intentionally not a launch workflow. Registering, importing, or downloading an asset never creates a profile, starts a runtime, or wires the asset into Chat. Operators must still create or update a profile, start the desired runtime, and explicitly use **Use this in Chat** when they want provider-plane routing to change.

Remote download policy is conservative by default:

- download URLs must use `http` or `https`, must not embed credentials, and must not include known secret query keys;
- local/private-network hosts are rejected unless the operator explicitly enables private downloads for the deployment;
- destination paths must resolve under configured llama.cpp roots such as `models_dir` or `allowed_paths`;
- destination filenames must be simple `.gguf` filenames.

## Common Misconfigurations

| Symptom | Plane | Likely Cause | Correct Fix |
|---|---|---|---|
| `503` on `/api/v1/llamacpp/status` saying backend not configured | managed plane | Managed handler disabled or unavailable | Enable `[LlamaCpp] enabled=true` and restart server |
| `/api/v1/llamacpp/inventory` or `/api/v1/llamacpp/models` returns unavailable while `provider=llama.cpp` chat works | managed plane | Provider plane configured, but no managed handler/runtime | Configure managed handler/model directory; do not assume provider mode enables lifecycle API |
| Readiness shows `restart_required=true` after saving config | managed plane | Saved config differs from active handler state | Restart the API server when changing handler startup config such as enabled state, executable path, or models directory |
| Hardware panel shows warnings but launch remains enabled | managed plane | GPU/RAM probing is unavailable or incomplete | Treat hardware warnings as sizing guidance; validate launch settings manually |
| **Use this in Chat** reports an override warning | bridge action | Environment/config override prevents the saved provider endpoint from becoming effective | Remove or update the override, then call **Use this in Chat** again |
| `provider=llama.cpp` chat fails while managed status is running | provider plane | `llama_api` endpoint/auth/config mismatch | Fix provider configuration (`llama_api` host/path/key), validate OpenAI-compatible endpoint |
| Tools payloads rejected for `provider=llama.cpp` | provider plane | Current contract blocks tools for this adapter path | Remove tools/tool_choice or switch provider that advertises tool support |
| llama.cpp grammar/thinking fields are rejected with `400` | provider plane | Resolved provider is not llama.cpp, or `strict_openai_compat` disabled non-standard fields | Route through `POST /api/v1/chat/completions` with llama.cpp selected and check local-provider strict compatibility |
| `thinking_budget_tokens` is ignored by UI or rejected by API | provider plane | No operator-configured upstream request-key mapping | Set `LLAMA_CPP_THINKING_BUDGET_PARAM` or `Local-API.llama_cpp_thinking_budget_param` for the deployment |

## Advanced Controls On The Provider Plane

The llama.cpp provider plane now exposes first-class advanced controls for `POST /api/v1/chat/completions`:

- `grammar_mode`, `grammar_id`, `grammar_inline`, `grammar_override`
- `thinking_budget_tokens` when the deployment advertises support

Important boundaries:

1. These first-class fields are only supported on `POST /api/v1/chat/completions` in v1.
2. `/api/v1/messages` does not yet accept the first-class llama.cpp grammar/thinking fields.
3. `strict_openai_compat` disables these advanced fields because they rely on non-standard request keys.
4. The grammar library endpoints only store user-scoped reusable GBNF text; the actual grammar is resolved into provider-plane `extra_body` at send time.

Provider metadata for this surface is exposed under `GET /api/v1/llm/providers` as `llama_cpp_controls`, including:

- whether grammar is currently supported
- whether thinking-budget control is currently supported
- the reserved raw `extra_body` keys that first-class UI controls can overwrite

## Verification Checklist

1. Confirm which plane your workflow targets first.
2. For lifecycle operations, use only `/api/v1/llamacpp/*` managed plane endpoints.
3. On `/admin/llamacpp`, confirm readiness, inventory, and status before launching.
4. Treat hardware warnings as advisory unless the actual launch fails.
5. Use **Use this in Chat** only after the managed server is running, then validate `provider=llama.cpp` and `llama_api` config independently.
6. If you need per-request thinking budget, verify `LLAMA_CPP_THINKING_BUDGET_PARAM` or `Local-API.llama_cpp_thinking_budget_param` is configured before testing.
7. Treat managed and provider diagnostics separately during incident triage.
