# Image Generation

Image_Generation normalizes provider-backed image creation for file artifacts and model catalog listing. It loads backend configuration, registers image adapters, validates provider capabilities such as reference-image support, resolves Media DB reference images through storage backends, and applies deterministic prompt refinement before dispatching to a provider adapter.

## Start Here

- `config.py` loads image generation backends, enabled providers, API-key sources, and request limits.
- `adapter_registry.py` resolves and registers image generation adapters.
- `adapters/base.py` defines `ImageGenRequest`, `ImageGenResult`, and the adapter protocol.
- `capabilities.py` defines reference-image capability contracts.
- `reference_images.py` resolves owned Media DB images for reference-image workflows.
- `prompt_refinement.py` applies deterministic prompt cleanup and quality suffixes.
- Related API surface: `tldw_Server_API/app/api/v1/endpoints/files.py` and `tldw_Server_API/app/api/v1/endpoints/llm_providers.py`.
- Related tests: `tldw_Server_API/tests/Image_Generation/`, `tldw_Server_API/tests/Files/`, and `tldw_Server_API/tests/FileArtifacts/`.

## Responsibilities

- Load image-generation defaults and provider-specific settings from configuration and environment variables.
- Register adapters for Stable Diffusion CPP, SwarmUI, OpenRouter, Novita, Together, and ModelStudio.
- Resolve the effective backend from request hints or configured defaults.
- Validate backend and model reference-image capabilities.
- Load reference image bytes and dimensions from user-owned media storage.
- Refine prompts within configured length limits.
- Expose configured image models to the LLM provider catalog.

## Module Map

- `config.py`: backend configuration, limits, and environment fallback handling.
- `adapter_registry.py`: adapter registration and backend resolution.
- `listing.py`: image-model catalog entries for provider listing.
- `capabilities.py`: reference-image support matrix and capability resolution.
- `reference_images.py`: Media DB and storage-backed reference image lookup.
- `prompt_refinement.py`: deterministic prompt refinement modes.
- `exceptions.py`: image-generation exception types.
- `adapters/`: provider-specific adapter implementations and shared image format helpers.

## How It Connects

- `File_Artifacts/adapters/image_adapter.py` calls this module to validate image artifact requests, resolve reference images, refine prompts, and invoke adapters.
- `files.py` exposes image artifact creation, export, and reference-image picker endpoints.
- `llm_providers.py` includes image models from `listing.py` in `/llm/providers`.
- Persona visual jobs, VN asset generation jobs, and workflow content adapters reuse the same image generation adapter path.
- Media DB and storage abstractions are used for reference images so callers do not pass arbitrary filesystem paths.

## Extension Points

- Add a provider by implementing an adapter under `adapters/`, registering it in `adapter_registry.py`, and adding config parsing in `config.py`.
- Add a provider's model catalog entries in `listing.py`.
- Extend reference-image support by updating `capabilities.py` and adding adapter-level request handling.
- Change prompt refinement in `prompt_refinement.py` and verify the File Artifacts image adapter tests.
- Add accepted output formats in adapter code and the files endpoint export path together.

## Testing

- Provider adapters, config defaults, model listing, reference-image capabilities, prompt refinement, and reference-image lookup are covered in `tldw_Server_API/tests/Image_Generation/`.
- API behavior for file image generation and reference image picker flows is covered in `tldw_Server_API/tests/Files/`.
- File artifact image adapter allowlist behavior is covered in `tldw_Server_API/tests/FileArtifacts/test_image_adapter_allowlist.py`.

## Gotchas

- Reference-image resolution intentionally checks ownership and storage-root safety before loading bytes.
- `ResolvedReferenceImage` accepts either inline bytes or a temp path, not both.
- API keys can come from config or environment; tests commonly patch these paths, so keep provider initialization lazy.
