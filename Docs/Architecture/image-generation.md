# Image generation

This document describes the provider-neutral image core: six backends behind one synchronous adapter protocol, shared bounds and prompt refinement, reference-image (img2img) resolution from owned media, and the consumers that own persistence and jobs (File Artifacts, Persona Visuals, VN Assets). The core itself has no HTTP or DB layer.

## Authoritative files

All paths under `tldw_Server_API/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| Config | `app/core/Image_Generation/config.py` | `ImageGenerationConfig` (globally cached — reset with the cache-clear helper), `resolve_image_generation_model`; code caps 1024²/1 M pixels/steps 50 vs the shipped config's raised values |
| Registry | `adapter_registry.py` | `DEFAULT_ADAPTERS` (stable_diffusion_cpp, swarmui, openrouter, novita, together, modelstudio), lazy dotted-path adapter resolution |
| Contracts | `adapters/base.py` | `ImageGenRequest` / `ImageGenResult` / the `ImageGenerationAdapter` Protocol — `generate()` is **synchronous** |
| Adapters | `adapters/*_adapter.py` | sd_cpp runs the configured binary as a local subprocess; novita submits + polls; modelstudio has sync/async/auto modes (reference images force sync) |
| Format helpers | `adapters/image_format_utils.py` | streamed fetch with declared-size caps, base64 decode, output validation/conversion |
| Capabilities | `capabilities.py` | reference-image support is **builtin modelstudio-only** (`qwen-image-2.0` families); config can only filter the builtin set, never grant it |
| Reference images | `reference_images.py` | `ManagedReferenceImageRepository` — owned MediaFiles joins only; ownership + storage-root prefix + size + PIL dimension checks |
| Prompt refinement | `prompt_refinement.py` | off/basic/auto (auto enriches only short prompts lacking quality cues) |
| Validation | `request_validation.py` | shared bounds + per-backend `extra_params` allowlists (modelstudio `mode` exempt) |
| Catalog | `listing.py` | `image/<backend>` entries with `is_configured` heuristics, merged into `/api/v1/llm/providers` |
| Consumers | `app/core/File_Artifacts/*`, `app/core/Persona/visual_jobs*.py`, `app/core/VN_Assets/*`, `app/core/Visual_Identities/` | persistence + Jobs ownership (see dataflows) |

## Dataflows

1. **File artifact image (sync default)**: `POST /api/v1/files/create` → the File Artifacts image adapter normalizes (backend resolution + prompt refinement), validates (bounds + allowlist), and generates; bytes persist via the artifact storage path; delivery by export URL or inline base64 under the cap.
2. **Async export**: explicit async (or auto over caps) → 202 + job id → the files-domain Jobs worker re-runs the export durably; failures reset export status with retryable/non-retryable codes (all `reference_image_*` codes are non-retryable).
3. **Reference image (img2img)**: the picker lists owned media (`GET /files/reference-images`); the create payload carries a `reference_file_id`; resolution checks backend capability → model capability → ownership/storage/size/dimension probes → `ResolvedReferenceImage` (data-URL conversion in the modelstudio adapter).
4. **VN asset generation**: pack/slot requests fan out to `vn_assets`-domain Jobs; the worker pins an execution recipe at first claim (settings frozen; the local sd-cpp model is sha256-verified — a changed model file fails `vn_asset_local_model_changed` rather than silently generating differently), acquires the backend concurrency gate, revalidates the claim token and lease, and generates on a worker thread.
5. **Persona visual candidates**: persona-pack jobs generate fixed 1024² PNGs; candidate assets are review-gated with activation as a separate explicit step.

## Boundaries

- The core is provider normalization only — no HTTP, no DB writes, no jobs; consumers own those.
- Adapters are synchronous and blocking (subprocess, sleep-polling); every async caller wraps with `to_thread`.
- Reference images come only from the principal's Media DB via the storage backend — never client paths or raw bytes.
- **No billing/quota category exists for images** — limits are config bounds, inline caps, and file-artifact storage quotas.

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| Backend not enabled / adapter init failure | `image_backend_unavailable` |
| Reference missing/wrong owner/bad path/unsupported | Non-retryable `reference_image_*` job codes |
| Storage backend failure | 503 `reference_image_storage_unavailable` |
| Bounds or allowlist violations | Create rejected (`image_params_invalid`) |
| VN backend busy / claim lost / batch terminal | Retryable `vn_asset_backend_busy` / `vn_asset_variant_claim_lost`; terminal `vn_asset_batch_terminal` |

## Config keys (`[Image-Generation]`)

`default_backend`, `enabled_backends` (JSON array), `max_width/max_height/max_pixels/max_steps/max_prompt_length/inline_max_bytes`, per-backend families (`sd_cpp_*`, `swarmui_*`, `openrouter_image_*`, `novita_image_*`, `together_image_*`, `modelstudio_image_*` incl. timeouts/poll intervals/allowlists), `reference_image_supported_models` (filter-only). Env fallbacks: `OPENROUTER_API_KEY`, `NOVITA_API_KEY`, `TOGETHER_API_KEY`, `DASHSCOPE_API_KEY`/`QWEN_API_KEY`, etc.

## Verified gotchas

1. **Opt-in enablement trap**: the code default for `enabled_backends` is `[]`, and an empty list disables *everything* silently even with `default_backend` set — the shipped config seeds sd_cpp.
2. Config is globally cached and adapters snapshot config at init — tests (and config changes) must reset both the config cache and the registry.
3. Catalog `is_configured` for hosted providers is key-presence only; sd_cpp requires the binary and model files to exist at listing time.
4. The File-Artifacts image adapter uses `asyncio.run` internally for reference resolution — it must only run on the sync export thread, never an event loop.
5. Model-family matching treats `qwen-image-2.0-flash` as family `qwen-image-2.0` (separator- and wildcard-aware).

## Governing decisions and docs

No image-gen ADR; adjacent: ADR-023 (File Artifacts export delegation), ADR-003 (Jobs for the VN/persona/files workers). Current: the module READMEs (`Image_Generation/`, `VN_Assets/`); `Persona_Visual_Packs.md` is accurate on its declared scope (pack review/activation) but predates the VN preflight rework. No dedicated `Code_Documentation` page exists — this doc is the reference.

## Related docs

- [jobs-background.md](./jobs-background.md) — the workers
- [rag.md](./rag.md) — where image models surface in the provider catalog
