# Character_Chat Core Module

Last updated: 2026-06-01

`tldw_Server_API/app/core/Character_Chat/` contains the core helpers used by Character Cards, character-backed chat sessions, world books, chat dictionaries, prompt assembly, and character-chat rate/size guardrails.

The module is backend/runtime code. User-facing API behavior is exposed through the FastAPI routers under `tldw_Server_API/app/api/v1/endpoints/`.

## Current Responsibilities

- Import, validate, normalize, store, load, and export Character Cards.
- Manage character-backed conversations and messages through `CharactersRAGDB`.
- Build character-aware prompt context from card fields, chat history, greetings, author notes, memory, presets, and world books.
- Process world-book/lorebook keyword and regex matches with priority, token budget, diagnostics, and optional recursive scanning.
- Process chat dictionaries with literal/regex replacements, grouping, probability, timed effects, ordering, token budgets, activity records, and version history.
- Enforce character/chat guardrails through `character_limits.py` and `character_rate_limiter.py`.
- Preserve compatibility for older call sites through `Character_Chat_Lib_facade.py`.

This module does not own provider integrations, AuthNZ, BYOK, OpenAI-compatible chat schemas, or generated documentation.

## Module Layout

```text
Character_Chat/
├── Character_Chat_Lib_facade.py
├── __init__.py
├── ccv3_parser.py
├── character_limits.py
├── character_rate_limiter.py
├── chat_dictionary.py
├── chat_grammar.py
├── constants.py
├── regex_safety.py
├── world_book_manager.py
├── world_book_prompt_context.py
└── modules/
    ├── __init__.py
    ├── character_chat.py
    ├── character_db.py
    ├── character_generation_presets.py
    ├── character_io.py
    ├── character_memory_extraction.py
    ├── character_prompt_presets.py
    ├── character_templates.py
    ├── character_utils.py
    ├── character_validation.py
    ├── persona_exemplar_embeddings.py
    ├── persona_exemplar_selector.py
    └── persona_exemplar_telemetry.py
```

Key files:

- `Character_Chat_Lib_facade.py`: stable import surface that re-exports modular helpers for legacy callers.
- `modules/character_io.py`: card import from image/text/JSON/YAML, metadata extraction, sanitization, image validation, and chat-history import helpers.
- `modules/character_validation.py`: Character Card v1/v2 and common external card-shape parsers.
- `ccv3_parser.py`: minimal Character Card v3 validation and mapping.
- `modules/character_db.py`: character CRUD helpers around `CharactersRAGDB`.
- `modules/character_chat.py`: conversation/message helper functions and placeholder-aware history formatting.
- `modules/character_prompt_presets.py`: default and custom character prompt section assembly.
- `modules/character_generation_presets.py`: generation-setting extraction from card extension metadata.
- `world_book_manager.py`: world-book CRUD, entries, import/export, statistics, matching, and context processing.
- `world_book_prompt_context.py`: adapter that builds and injects world-book context into prompt messages.
- `chat_dictionary.py`: chat dictionary CRUD, entry processing, import/export, activity, stats, and version history.
- `character_limits.py`: non-rate character/chat limits from settings or environment.
- `character_rate_limiter.py`: Resource Governor-backed enforcement wrapper for character operations and chat flows.
- `constants.py`: shared bounds for regex, streaming, tool-call metadata, imports, and message content.

## Data Flow

1. API dependencies resolve the authenticated user and per-user `CharactersRAGDB` with `get_chacha_db_for_user`.
2. Character/card endpoints call import, validation, DB, exemplar, world-book, or export helpers.
3. Chat-session endpoints load conversation state, card data, chat settings, messages, and optional participant cards.
4. Prompt assembly applies card context, summaries, greetings, author notes, character memory, message steering, world-book context, and provider/model options.
5. Provider dispatch happens outside this module through shared chat provider helpers.
6. Non-streaming completion responses may persist user and assistant turns through the character chat helpers. Streaming responses are returned as SSE and require a follow-up persist call if the assistant text should be saved.

Persistence flows through `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`, primarily via `CharactersRAGDB`. Important data families include `character_cards`, `character_exemplars`, `conversations`, `messages`, message metadata, conversation settings, world books, world-book entries, character/world-book attachments, chat dictionaries, dictionary entries, dictionary activity, and dictionary revisions.

## Character Card Import And Export

The import path accepts file content from `POST /api/v1/characters/import` after endpoint-level type and size validation. Supported upload extensions are `.png`, `.webp`, `.jpeg`, `.jpg`, `.json`, `.yaml`, `.yml`, `.txt`, and `.md`.

Important behavior:

- Image imports detect PNG, WebP, and JPEG magic bytes. Metadata extraction is most useful for PNG/WEBP card files that carry `chara` metadata.
- Image-only import is explicit. If no card metadata is detected, the endpoint returns `missing_character_data` unless `allow_image_only=true` is provided.
- Text imports support JSON/YAML and text/Markdown containing card JSON.
- Parsed card text is sanitized for null bytes and obvious script-like patterns before persistence.
- Card formats include Character Card v1/v2/v3 and common Tavern/SillyTavern, Pygmalion, Text Generation WebUI, and Alpaca-style shapes.
- Export supports `v3`, `v2`, `json`, and `png`. PNG export embeds v2 card JSON into a PNG `chara` metadata chunk.

## Prompt Assembly

Character prompt assembly is spread across the API endpoint and focused helpers because it combines request flags, chat settings, current DB state, and provider options.

Primary components:

- System prompt and card sections from `modules/character_prompt_presets.py`.
- Character generation defaults from `modules/character_generation_presets.py`.
- Greeting selection and staleness checks in `character_chat_sessions.py`.
- Author note resolution from chat settings and card extension fields.
- Character memory injection through the character-memory endpoint/database path.
- Auto-summary settings stored in conversation settings.
- World-book context from `world_book_prompt_context.py`.
- Message steering flags for continue-as-user, impersonate-user, and forced narration.
- Multi-character participant resolution from `participantCharacterIds` or `participant_character_ids` settings.

Prompt preview is exposed by `POST /api/v1/chats/{chat_id}/prompt-preview` and should remain aligned with `/complete-v2` whenever prompt assembly changes.

## FastAPI Touch Points

Route mounting:

- `characters_endpoint.py` is mounted under `/api/v1/characters`.
- `character_chat_sessions.py` is mounted under `/api/v1/chats`.
- `character_messages.py` is mounted under `/api/v1`.
- `chat_dictionaries.py` is included by the main chat router, mounted under `/api/v1/chat`.
- `character_memory.py` is mounted under `/api/v1/characters` and is used by character memory injection paths.

Major route families:

- Character cards: `/api/v1/characters`, `/api/v1/characters/import`, `/api/v1/characters/{character_id}`, `/api/v1/characters/{character_id}/export`.
- Character versions and restore: `/api/v1/characters/{character_id}/versions`, `/api/v1/characters/{character_id}/versions/diff`, `/api/v1/characters/{character_id}/revert`, `/api/v1/characters/{character_id}/restore`.
- Exemplars: `/api/v1/characters/{character_id}/exemplars...`.
- World books: `/api/v1/characters/world-books...` and `/api/v1/characters/{character_id}/world-books...`.
- Chat sessions and completions: `/api/v1/chats`, `/api/v1/chats/{chat_id}`, `/api/v1/chats/{chat_id}/completions`, `/api/v1/chats/{chat_id}/complete-v2`, `/api/v1/chats/{chat_id}/completions/persist`.
- Prompt helpers: `/api/v1/chats/{chat_id}/prompt-preview`, `/api/v1/chats/{chat_id}/greetings`, `/api/v1/chats/{chat_id}/author-note/info`, `/api/v1/chats/{chat_id}/diagnostics/lorebook`.
- Messages: `/api/v1/chats/{chat_id}/messages`, `/api/v1/messages/{message_id}`.
- Chat dictionaries: `/api/v1/chat/dictionaries...`.

## Settings And Environment

Character/chat limits are read from settings with environment overrides:

- `MAX_CHARACTERS_PER_USER`
- `MAX_CHARACTER_IMPORT_SIZE_MB`
- `MAX_CHATS_PER_USER`
- `MAX_MESSAGES_PER_CHAT`
- `MAX_MESSAGES_PER_CHAT_SOFT`
- `MAX_CHARACTER_IMPORT_AVATAR_SIZE_BYTES`
- `MAX_CHARACTER_IMPORT_AVATAR_SIZE_MB`
- `MAX_MESSAGE_IMAGE_BYTES`

Provider/model defaults for character completions can come from chat settings, configured provider defaults, and character-chat environment overrides such as `CHAR_CHAT_PROVIDER` and `CHAR_CHAT_MODEL`. Strict model selection is controlled outside this module in the chat/provider routing path.

Rate limiting goes through `CharacterRateLimiter`. Current enforcement is Resource Governor-backed when character RG enforcement is enabled; otherwise the legacy shim logs deprecation and allows the request.

## Safety And Boundaries

- Do not bypass `CharactersRAGDB`; this module should use database abstractions rather than ad hoc SQL.
- Keep import sanitization and size checks in place for untrusted cards and images.
- Keep regex length and safety checks aligned between world books, dictionaries, and `regex_safety.py`.
- Treat card fields, world books, dictionaries, author notes, memory, and message history as prompt material that may leave the host when an external provider is used.
- Preserve streaming persistence semantics: streaming responses do not automatically save assistant content.
- Avoid adding global mutable runtime state. Per-request state should flow through dependencies, DB rows, settings payloads, or bounded helpers.

## Extension Guidance

When adding a card field:

1. Update `character_schemas.py`.
2. Update parser/export mappings in `character_validation.py`, `ccv3_parser.py`, and `characters_endpoint.py` as needed.
3. Update DB storage through `ChaChaNotes_DB` migrations/abstractions.
4. Update prompt preset assembly only if the field should affect model context.
5. Add focused tests for parsing, API create/update, export, and prompt assembly where relevant.

When changing prompt assembly:

1. Keep `/completions`, `/complete-v2`, and `/prompt-preview` behavior aligned.
2. Include token/budget diagnostics where context can grow.
3. Verify streaming and non-streaming persistence paths separately.
4. Add regression tests for participant routing if multi-character behavior changes.

When changing dictionaries or world books:

1. Keep regex safety and token budgets intact.
2. Preserve import/export fidelity for JSON paths.
3. Update statistics, versioning, and activity behavior when entry lifecycle changes.
4. Add tests for literal, regex, ordering, budget, and disabled-entry behavior.

## Relevant Schemas

- `tldw_Server_API/app/api/v1/schemas/character_schemas.py`
- `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py`
- `tldw_Server_API/app/api/v1/schemas/world_book_schemas.py`
- `tldw_Server_API/app/api/v1/schemas/chat_dictionary_schemas.py`
- `tldw_Server_API/app/api/v1/schemas/character_memory_schemas.py`

## Relevant Tests

Current targeted suites include:

- `tldw_Server_API/tests/Character_Chat/`
- `tldw_Server_API/tests/Character_Chat_NEW/`
- `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_character_store.py`
- `tldw_Server_API/tests/ChaChaNotesDB/test_character_card_tag_search.py`
- `tldw_Server_API/tests/ChaChaNotesDB/test_character_exemplars_db.py`
- `tldw_Server_API/tests/Character_Chat/unit/test_chat_session_character_scope_api.py`
- `tldw_Server_API/tests/Chat/unit/test_chat_dictionary_endpoints.py`
- `tldw_Server_API/tests/Streaming/test_character_chat_sse_unified_flag.py`
- `tldw_Server_API/tests/RAG/test_dual_backend_characters_retriever.py`

Useful targeted commands:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Character_Chat -q
python -m pytest tldw_Server_API/tests/Character_Chat_NEW -q
python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_dictionary_endpoints.py -q
python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_character_exemplars_db.py -q
```

For docs-only changes to this README, targeted source checks and `git diff --check` are usually sufficient. Run the tests above when behavior or schemas change.
