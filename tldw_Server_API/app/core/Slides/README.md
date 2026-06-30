# Slides

Slides provides the core persistence, generation, rendering, export, asset, and visual-style logic for user presentations. The package backs the `/slides` API with a per-user `Slides.db`, LLM-assisted slide JSON generation, Reveal.js rendering/export helpers, and configurable visual style profiles.

## Start Here

- `slides_db.py` owns the per-user presentations database, version records, FTS support, and visual style persistence.
- `slides_generator.py` turns source text into structured slide JSON through an LLM provider or deterministic test mode.
- `presentation_rendering.py`, `slides_export.py`, and `slides_templates.py` handle render/export/template behavior.
- Related API surface: `app/api/v1/endpoints/slides.py`.
- Related schemas: `app/api/v1/schemas/slides_schemas.py`.
- Related tests: `tests/Slides/`.

## Responsibilities

- Store presentations, versions, metadata, soft-delete state, and searchable presentation text.
- Generate slide decks from source text with optional chunking, summarization, visual-style prompts, and JSON repair/parsing.
- Render presentations with Reveal.js assets and export them to supported formats.
- Manage slide assets, image references, template catalogs, and style-pack CSS.
- Persist and resolve visual styles, style profiles, generated styles, and style packs.
- Enforce optimistic versioning and conflict checks through API/database flows.

## Module Map

- `slides_db.py` - SQLite persistence, versioning, FTS, visual style tables, and soft delete.
- `slides_generator.py` - LLM-backed and test-mode slide generation.
- `presentation_rendering.py` - presentation HTML/rendering helpers.
- `slides_export.py` - export helpers for rendered presentations.
- `slides_assets.py` and `slides_images.py` - asset and image management helpers.
- `slides_templates.py` - template catalog handling.
- `visual_styles.py`, `visual_style_catalog.py`, `visual_style_generation.py`, `visual_style_packs.py`, `visual_style_profiles.py`, and `visual_style_resolver.py` - visual style creation and resolution.
- `revealjs/` and `style_packs/` - vendored/runtime presentation assets used by rendering.

## How It Connects

- `app/api/v1/endpoints/slides.py` wires the module to AuthNZ dependencies, database dependencies, RAG source helpers, Jobs, metrics, render/export, and visual-style endpoints.
- `app/api/v1/API_Deps/Slides_DB_Deps.py` creates the per-user `Slides.db` dependency.
- `app/core/MCP_unified/modules/implementations/slides_module.py` exposes slide operations to MCP tools.
- Chat, RAG, Collections, Media DB, and ChaChaNotes are adjacent sources for presentation generation and asset references.

## Extension Points

- For new presentation fields or version behavior, start in `slides_db.py`, `slides_schemas.py`, and the endpoint ETag/version tests.
- For a new export format, update `slides_export.py` and the `/slides` export endpoint tests.
- For style-pack or template behavior, inspect `slides_templates.py`, `visual_style_*`, and `style_packs/`.
- For generation changes, update `slides_generator.py` with tests covering JSON parsing, token limits, and deterministic test mode.

## Testing

- `tests/Slides/`
- `app/core/MCP_unified/tests/test_slides_module.py`

## Gotchas

- Presentation updates use version/ETag-style conflict behavior; bypassing `slides_db.py` can skip conflict detection.
- The module contains vendored Reveal.js assets. Treat those as runtime assets, not primary application logic.
- Generation paths enforce source size and structured JSON constraints; prompt changes should preserve parseable output.
