# UAT118 image Retry: vision capability diagnosis

2026-09-16. HEAD at evidence capture: 1b7e7734c9ee6a8afe9f7dcc5de641c20998f9d6.

Scope: read-only source inspection and authenticated GET /v1/models and GET /props on root-owned localhost19099. No inference, browser actions, application launch, configuration write, profile creation, source/test/task/root-document edit, staging or commit performed. Existing9099 was not contacted or changed.

## Outcome

19099 truthfully advertises vision. Ordinary external Llama.cpp/Custom OpenAI discovery loses that capability metadata, so changing only provider URL/key/model is insufficient to make Chat accept images.

An existing configuration-only candidate is available via saved managed-profile metadata: record the real GGUF and matching projector as an enabled vision profile; use the provider-qualified actual ID `llama.cpp/gemma-4-26B-A4B-it`; independently configure the normal Llama.cpp provider to the actual external19099 endpoint. This is a source-supported candidate, not tested native application behavior. Root will validate normal visible configuration after restoring the app. UAT118 remains open. No UAT157 or source repair was created here.

## Direct non-inference evidence

- GET http://127.0.0.1:19099/v1/models: HTTP200; ID `gemma-4-26B-A4B-it`; capabilities `completion,multimodal`; meta.n_ctx=16384.
- GET http://127.0.0.1:19099/props: HTTP200; modalities `vision:true,video:true,audio:false`; build `b10430-4c1a0af40`; matching model alias.
- Props reports loaded weights `/Users/macbook-dev/Working/Language_Models/gemma-4-26B-A4B/gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf`.
- Matching projector file is present: `/Users/macbook-dev/Working/Language_Models/gemma-4-26B-A4B/gemma-4-26B-A4B-it-mmproj-BF16.gguf`. Root reports its owned launch loaded the matching projector; props does not independently expose its path. Match it to root's launch before saving a profile.
- Root separately reports direct public-icon image control HTTP200 in5807ms: “A white speech bubble containing three black dots.” This is root-owned provider evidence, not application UAT or a reviewer inference request.
- Sandbox first returned EPERM. Authorized read-only GETs succeeded after escalation. The API key was read privately from its file, never printed or placed in visible command arguments. Saved responses were checked for the exact key string and contain none.

Key-free evidence:

- `/private/tmp/uat118-vision-models.json`
- `/private/tmp/uat118-vision-props.json`
- `/private/tmp/uat118-vision-capability-response.json`
- `/private/tmp/uat118-vision-diagnosis-manifest.json` — source and response SHA-256 values.

## Exact capability propagation gap

1. `llm_providers.py:1105` `_extract_models_from_response` retains only IDs/names from data/models. Actual `capabilities:[completion,multimodal]` is discarded. `discover_models_from_endpoint` returns flat ModelDiscoveryResult IDs; it does not query props.
2. Catalog assembly uses `get_model_metadata(provider, model)` (line931), backed by static MODEL_METADATA and conservative defaults (lines895-922). This actual model lacks a llama/custom entry, so vision=false and input=[text]. Routing metadata derives flags from existing capabilities; it does not expose a configurable capability override.
3. `get_models_metadata` flattens provider model metadata and then appends managed-profile entries. Provider-level adapter capability envelopes do not replace individual model capabilities.
4. Frontend `domains/models-audio.ts:454` requests `/api/v1/llm/models/metadata`; `model-normalization.ts` and `TldwModels.transformModel` preserve supplied capabilities. They do not independently query19099, or derive vision from generic multimodal strings.
5. `models/index.ts:87-96` requires selected model capabilities to include `vision`. `ChatTldw.ts:467-475` blocks image_url HumanMessage parts before transport without that capability. The ordinary external raw-ID listing therefore remains unsupported despite the actual server's ability.

## Existing controls checked

### Settings → Models

Default provider/model, catalog display, nicknames and generation settings are available. `AddUpdateModelSettings.tsx` edits generation/sampling/context parameters; no consumed vision override was found. Legacy `createModel` stores model ID/name/provider/type. Current pageAssistModel uses backend-derived tldwModels metadata, not those legacy custom records.

### Admin provider override API

`PUT /api/v1/admin/llm/providers/{provider}` accepts enabled/allowed_models/config/credentials. Generic config JSON is accepted, but `apply_llm_provider_overrides_to_listing` only filters/orders models and existing models_info, and sets enabled/default_model. It does not merge arbitrary per-model capabilities or modalities. Setting config.vision or config.models_info would not establish a consumed capability setting.

### Admin → Llama.cpp → Profiles

The existing editor exposes Name, Mode, model asset/direct Model path, mmproj asset or Server args mmproj path, Host, Port, Enabled, Autostart and Provider alias. POST `/api/v1/llamacpp/profiles` creates the profile; PUT `/api/v1/llamacpp/profiles/{profile_id}` updates it.

`llamacpp_profile_capabilities.py` resolves real file paths, validates inventory allowlists and requires a projector for vision mode. Valid vision profiles emit capabilities.chat=true, capabilities.vision=true, input=[text,image], output=[text]; invalid paths produce false capabilities and warnings. Saved profile metadata is available independently of runtime RUNNING state.

## Exact configuration candidate for root

These are proposed normal configuration steps; none were applied here.

1. Configure normal Llama.cpp transport in the isolated fresh API profile:

   - `[Local-API] llama_api_IP = http://127.0.0.1:19099/v1`
   - `[Local-API] llama_model = gemma-4-26B-A4B-it`
   - `[Local-API] llama_api_key` receives the existing private19099 credential through the ordinary private configuration mechanism. Do not copy its value into evidence.

   Catalog provider name is `llama`; Chat provider is `llama.cpp`. The Llama adapter uses the configured endpoint/key and normalized requested model. Custom OpenAI can use the same transport but alone does not fix metadata loss.

2. Ensure the isolated `[LlamaCpp]` model directory/allowed paths include `/Users/macbook-dev/Working/Language_Models/gemma-4-26B-A4B`, through existing settings/assets controls. This lets the profile validate the actual files.

3. Admin → Llama.cpp → Profiles → New profile; save the following ordinary profile settings:

```json
{
  "name": "Gemma 4 26B vision on 19099",
  "enabled": true,
  "mode": "vision",
  "model_id": null,
  "model_path": "/Users/macbook-dev/Working/Language_Models/gemma-4-26B-A4B/gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf",
  "mmproj_model_id": null,
  "host": "127.0.0.1",
  "port": 19099,
  "port_policy": "explicit",
  "server_args": {
    "mmproj": "/Users/macbook-dev/Working/Language_Models/gemma-4-26B-A4B/gemma-4-26B-A4B-it-mmproj-BF16.gguf"
  },
  "autostart": false,
  "provider_alias": "llama.cpp/gemma-4-26B-A4B-it"
}
```

Use the returned profile ID. Real discovered asset IDs may replace direct paths once confirmed to resolve to the same files. This is Save only: root's already-running19099 process is not owned by the application's supervisor.

4. Refresh the catalog via existing UI. Verify the entry has model=`llama.cpp/gemma-4-26B-A4B-it`, provider=`llama.cpp`, source=`managed_llamacpp_profile`, capabilities.vision=true, and no capability warnings. Verify it is selectable and visibly advertises Vision.

5. Select that exact qualified entry and Llama.cpp provider in Chat. Root must capture the application request and actual outbound provider request: expected provider model is precisely `gemma-4-26B-A4B-it`. Only after confirming the mapping should root perform the actual image send/Retry acceptance under its inference lease.

### Why the qualified actual ID matters

Managed entries are appended after ordinary model entries. Reusing the raw model ID risks the first ordinary text-only row winning: `TldwModels.getModel` uses first matching ID (line471). The qualified ID avoids that collision without pretending to be a known vision model.

Backend `_split_inline_provider_model` (`chat_service.py:1644`) recognizes registered provider prefixes, including its own explicit local provider. `normalize_request_provider_and_model` (lines1917-1941) strips `llama.cpp/`, preserving the exact real model suffix. This is existing provider-routing syntax. Native outbound verification is still required.

### Runtime ownership limits

The saved profile supplies validated metadata; it does not adopt root's external process. No external attach/adopt endpoint was found. Use in Chat requires a supervisor-owned RUNNING runtime and only rewrites the global llama endpoint; it does not configure API key/model/capabilities. Start validates port availability and launches a new process. Do not Start on occupied19099. This candidate uses saved-profile metadata plus independently configured global provider transport, and must not be reported as a managed-runtime acceptance test.

## Remaining uncertainty

- Fresh application API/catalog was not running or queried during this diagnosis. Supervisor availability, file allowlist validation, catalog cache refresh, provider readiness and visible selection still require live validation.
- Profile metadata is not bound to the external process. Truthfulness depends on retaining evidence that19099 uses the same actual GGUF+projector and the configured global Llama.cpp endpoint points there.
- No native success is claimed. If normal configuration fails, retain the exact failure without injecting cached capabilities or inventing another model identity.
- UAT118 image send/Retry remains pending. Provider control success alone does not close it.
