# Config Files Overview

This document explains the purpose and usage of settings in `Config_Files/config.txt`.
It is the single source of truth for server/runtime configuration. Values are read
at startup by the backend and, for some modules, re-read on demand.

Conventions
- All keys belong to an INI section like `[Chunking]`.
- Booleans: `true|false` (case-insensitive). Integers/floats are unquoted.
- Paths are relative to the repo root unless noted.
- Secrets should be supplied via `.env` where indicated; do not commit secrets.

How to apply changes
- Edit `tldw_Server_API/Config_Files/config.txt` and restart the server.
- Some modules cache settings at import; prefer restart after edits.

---

## Diagnostics
- Admin network diagnostics endpoint: `GET /api/v1/admin/network-info`
  - Returns resolved client IP, proxy headers used, and access decisions for Setup based on toggles and IP lists.
  - Requires admin role. Useful when configuring `trusted_proxies`, allow/deny lists, or remote toggles.

## [Setup]
- `enable_first_time_setup` (bool): Allow interactive/setup workflows on first run.
- `setup_completed` (bool): Marks setup completed to suppress guided flows.
- `allow_remote_setup_access` (bool): If true, allows setup API from non-localhost.
 - `setup_ip_allowlist` (csv): Optional IPs/CIDRs allowed to access `/setup` remotely.
   - Env override: `TLDW_SETUP_ALLOWLIST="203.0.113.0/24"`
 - `setup_ip_denylist` (csv): Optional IPs/CIDRs to explicitly block from `/setup`.
   - Env override: `TLDW_SETUP_DENYLIST="0.0.0.0/0"`

## [UI]
- `quickstart_url` (str): Optional quickstart target for `/api/v1/config/quickstart`.
  - Accepts absolute URLs (`https://...`) or same-origin paths (`/docs`, `/login`, etc.).

## [Server]
- `disable_cors` (bool): Disable browser-origin checks entirely. Leave this `false` for normal self-hosting; localhost and loopback browser access already work by default.
- `cors_allow_credentials` (bool): Include `Access-Control-Allow-Credentials` on CORS responses.
  - Default: `false` (safer; compatible with wildcard origins in local/test).
  - Env override: `CORS_ALLOW_CREDENTIALS=true|false`
- `ALLOWED_ORIGINS` (env): Optional browser-origin allowlist for non-default setups.
  - Local first-run self-hosting already allows common `localhost` and `127.0.0.1` origins.
  - Set this only when your browser UI runs from a different origin, such as another host, LAN IP, or custom port.
- `trusted_proxies` (csv): Proxy IPs/CIDRs trusted for X-Forwarded-For/X-Real-IP processing.
  - If the socket peer is in this list, the server uses the leftmost X-Forwarded-For IP as the client.
  - Env override: `TLDW_TRUSTED_PROXIES="10.0.0.0/8,192.168.0.0/16,127.0.0.1"`

## [Audit]
- `export_stream_auto_max_rows` (int): Auto-switch audit exports to streaming when `max_rows` is above this value (env: `AUDIT_EXPORT_STREAM_AUTO_MAX_ROWS`). Keeps large CSV/JSON/JSONL exports from buffering entirely in memory.

## Workspace Project Roots
Project Workspace host-local roots are gated by `[WORKSPACES].project_root_allowed_base_paths`,
`WORKSPACE_PROJECT_ROOT_ALLOWED_BASE_PATHS`, or `TLDW_WORKSPACE_PROJECT_ROOT_ALLOWED_BASE_PATHS`.
If unset, the Workspace API falls back to ACP workspace roots for compatibility. Ingestion source
roots are not used for project-root writes.

## [Processing]
- `processing_choice` (str): Hardware backend, e.g., `cuda|cpu` for ML pipelines.

## [Media-Processing]
- `max_audio_file_size_mb` (int): Max upload size for audio files.
- `max_pdf_file_size_mb` (int): Max upload size for PDFs.
- `max_video_file_size_mb` (int): Max upload size for videos.
- `max_epub_file_size_mb` (int): Max upload size for EPUB.
- `max_document_file_size_mb` (int): Max upload size for generic docs.
- `pdf_conversion_timeout_seconds` (int): Timeout for PDF conversion.
- `audio_processing_timeout_seconds` (int): Timeout for audio processing.
- `video_processing_timeout_seconds` (int): Timeout for video processing.
- `max_archive_internal_files` (int): Max files extracted from archives.
- `max_archive_uncompressed_size_mb` (int): Max total uncompressed archive size.
- `audio_transcription_buffer_size_mb` (int): Buffer budget for streaming STT.
- `uuid_generation_length` (int): Default short UUID length used in DBs/exports.
- `kept_video_max_files` (int): Retention: count of video temp files.
- `kept_video_max_storage_mb` (int): Retention: MB cap for video temp files.
- `kept_video_retention_hours` (int): Retention: age cap for video temp files.

## [Chat-Dictionaries]
- `enable_chat_dictionaries` (bool): Enable dictionary-based post-gen editing.
- `post_gen_replacement` (bool): Apply post generation replacement list.
- `post_gen_replacement_dict` (path): Markdown file with replacements.
- `chat_dictionary_chat_prompts` (path): Prompt dictionary for chat.
- `chat_dictionary_RAG_prompts` (path): Prompt dictionary for RAG.
- `strategy` (str): Application strategy, e.g., `character_lore_first`.
- `max_tokens` (int): Size cap for dictionary-injected content.
- `default_rag_prompt` (str): Name of default system prompt for RAG.

## [Chat-Module]
- `enable_provider_fallback` (bool): Allow fallback to alt provider on failure.
- `max_base64_image_size_mb` (int): Per-image Base64 size limit.
- `max_text_length_per_message` (int): Characters per user message.
- `max_messages_per_request` (int): Limit messages per API call.
- `max_images_per_request` (int): Max images per API call.
- `max_request_size_bytes` (int): Hard payload cap for requests.
- `streaming_idle_timeout_seconds` (int): Idle WS timeout for streaming.
- `streaming_heartbeat_interval_seconds` (int): Ping interval for streaming.
- `streaming_max_response_size_mb` (int): Max streamed response size.
- `chat_save_default` (bool): Default persistence for conversations.
- `conversation_creation_max_retries` (int)
- `db_transaction_max_retries` (int)
- `rate_limit_per_minute` (int): Global rate per client.
- `rate_limit_per_conversation_per_minute` (int): Per-conversation rate.
- `prompt_guardrails_enabled` (bool): Enable pre-dispatch prompt/cost guardrails for chat and character chat. Default: `false`.
- `prompt_guardrails_default_action` (str): `warn|block`; warning thresholds are non-blocking by default.
- `prompt_guardrails_warn_total_estimated_tokens` / `prompt_guardrails_block_total_estimated_tokens` (int, optional): Total prompt estimate warning and hard-cap thresholds.
- `prompt_guardrails_warn_static_segment_tokens` / `prompt_guardrails_warn_world_book_tokens` (int, optional): Segment-specific warning thresholds for cache-sensitive prompt sections.
- `prompt_guardrails_warn_max_output_tokens` / `prompt_guardrails_warn_choice_count` (int, optional): Warn when output caps or `n` can multiply cost.
- `prompt_guardrails_warn_reasoning_efforts` (csv): Reasoning-effort values that should warn about hidden/reasoning-token risk. Default: `high,xhigh`.
- `prompt_guardrails_warn_on_fingerprint_churn` (bool): Warn when static prompt fingerprints churn between requests, indicating lower cache reuse potential. Default: `true`.
- `history_messages_limit` (int): History window size for context (1-500).
- `history_messages_order` (str): `asc|desc` ordering of loaded history.

## [Character-Chat]
- `CHARACTER_RATE_LIMIT_OPS` (int): Ops allowed per window.
- `CHARACTER_RATE_LIMIT_WINDOW` (int sec): Window size in seconds.
- `MAX_CHARACTERS_PER_USER` (int)
- `MAX_CHARACTER_IMPORT_SIZE_MB` (int)

## [Settings]
- `chunk_duration` (int sec): Media chunk duration for STT.
- `words_per_second` (int): Heuristic for text/audio timing.
- `save_character_chats` (bool)
- `save_rag_chats` (bool)
- `save_video_transcripts` (bool)

## [Auto-Save]
- `save_character_chats` (bool)
- `save_rag_chats` (bool)

## [Prompts]
- `prompt_sample` (str): Example prompt used in demos/tests.
- `video_summarize_prompt` (str): Default summarization prompt.

## [Database]
- `type` (str): `sqlite|postgres|...` DB backend selector.
- `sqlite_path` (path): Main media summary DB when `sqlite`.
- `sqlite_wal_mode` (bool)
- `sqlite_foreign_keys` (bool)
- `backup_path` (path): Backup/export directory.
- `pg_connection_string` (dsn): Single string alternative to fields below.
- `pg_host|pg_port|pg_database|pg_user|pg_password|pg_sslmode` (str): Postgres.
- `pg_pool_size|pg_max_overflow|pg_pool_timeout` (ints/floats): Pool tuning.
- `elasticsearch_host|elasticsearch_port` (str/int): If ES is used.
- `chroma_db_path` (path): ChromaDB path.
- `prompts_db_path|rag_qa_db_path|character_db_path` (path): Other DBs.

## [Chunking]
Global defaults:
- `chunking_method` (str): `words|sentences|paragraphs|tokens|...`.
- `chunk_max_size` (int): Default chunk size (units vary by method).
- `chunk_overlap` (int): Default overlap (units vary by method).
- `adaptive_chunking` (bool): Adaptive window scaling heuristics.
- `chunking_multi_level` (bool): Paragraph-aware multi-level chunking.
- `language` (str): Default language hint.

Operational controls:
- `max_streaming_flush_threshold_chars` (int): Upper bound on stream flush size; 0 disables.
- `json_single_metadata_reference` (bool): Emit one JSON metadata chunk and reference from others.
- `json_metadata_reference_key` (str): Reference field name in JSON chunks.
- `cache_copy_on_access` (bool): Deep-copy cached results (safety) vs direct store (performance).
- `verbose_logging` (bool): Raise chunk creation logs to INFO (otherwise DEBUG).
- `regex_timeout_seconds` (float): Cap regex execution time in `ebook_chapters`.
- `regex_disable_multiprocessing` (bool): Use thread-guarded regex only.
- `regex_simple_only` (bool): Restrict custom chapter regex to safe subset.
- `enable_contextual_retrieval` (bool): Include contextual elements in retrieval.
- `context_window_size` (int): Contextual window for retrieval.
- `include_parent_context` (bool)

Per-media defaults (override global):
For each type in `chunking_types` (`article|audio|book|document|mediawiki_article|mediawiki_dump|obsidian_note|podcast|text|video`) the following are available:
- `<type>_chunking_method` (str)
- `<type>_chunk_max_size` (int)
- `<type>_chunk_overlap` (int)
- `<type>_adaptive_chunking` (bool)
- `<type>_chunking_multi_level` (bool)
- `<type>_language` (str)

## [AuthNZ]
- `auth_mode` (str): `single_user|multi_user`.
- `database_url` (dsn): Auth DB URL; default uses SQLite file.
- `enable_registration` (bool)
- `require_registration_code` (bool)
- `enable_org_scoped_registration_codes` (bool)
- `org_invite_allow_missing_email` (bool): Allow org invite redemption when user email is missing.
- `rate_limit_enabled` (bool)
- `rate_limit_per_minute` (int)
- `rate_limit_burst` (int)
- `access_token_expire_minutes` (int)
- `refresh_token_expire_days` (int)
- `byok_enabled` (bool): Enable per-user BYOK keys (ignored in single_user mode).
- `byok_allowed_providers` (csv): Optional allowlist of providers eligible for BYOK.
- `byok_allowed_base_url_providers` (csv): Optional allowlist of providers that may set BYOK `base_url`.
- `byok_encryption_key` (str): Base64-encoded 32-byte key for BYOK secret encryption (AES-GCM).
- `byok_secondary_encryption_key` (str): Secondary BYOK encryption key for dual-read during rotations.

## [Embeddings]
- `embedding_provider` (str): `openai|huggingface|llama|...`.
- `embedding_model` (str): Model id.
- `onnx_model_path|model_dir` (path): Local model locations.
- `embedding_api_url` (url): Custom API endpoint.
- `chunk_size|overlap` (int): Chunking defaults for embeddings.
- `enable_contextual_chunking` (bool)
- `contextual_llm_model` (str)
- `context_window_size` (int|None)
- `context_strategy` (str)
- `context_token_budget` (int)

## [Claims]
- `ENABLE_INGESTION_CLAIMS` (bool): Extract claims during ingestion.
- `CLAIM_EXTRACTOR_MODE` (str): `heuristic|llm|...`.
- `CLAIMS_MAX_PER_CHUNK` (int)
- `CLAIMS_EMBED` (bool)
- `CLAIMS_EMBED_MODEL_ID` (str)
- `CLAIMS_LLM_PROVIDER|CLAIMS_LLM_MODEL` (str)
- `CLAIMS_LLM_TEMPERATURE` (float)
- `CLAIMS_JSON_PARSE_MODE` (str): `lenient|strict`
- `CLAIMS_ALIGNMENT_MODE` (str): `off|exact|fuzzy`
- `CLAIMS_ALIGNMENT_THRESHOLD` (float)
- `CLAIMS_MONITORING_ENABLED` (bool)
- `CLAIMS_ADAPTIVE_THROTTLE_ENABLED` (bool)
- `CLAIMS_ADAPTIVE_THROTTLE_LATENCY_MS` (float)
- `CLAIMS_ADAPTIVE_THROTTLE_ERROR_RATE` (float)
- `CLAIMS_ADAPTIVE_THROTTLE_BUDGET_RATIO` (float)
- `CLAIMS_REBUILD_ENABLED` (bool)
- `CLAIMS_REBUILD_INTERVAL_SEC` (int)
- `CLAIMS_REBUILD_POLICY` (str)
- `CLAIMS_STALE_DAYS` (int)
- `contextual_llm_model` (str): Claims pipeline LLM.
- `contextual_chunk_method` (str)
- `trusted_hf_remote_code_models` (csv)
- `max_models_in_memory` (int)
- `max_model_memory_gb` (int)
- `model_lru_ttl_seconds` (int)

## [RAG]
- `vector_store_type` (str): `chromadb|pgvector|...`.
- `pgvector_*` (dsn parts): Host/port/db/user/password/sslmode when used.
- `enable_parent_expansion` (bool) & `parent_expansion_size` (int)
- `include_sibling_chunks` (bool) & `sibling_window` (int)
- `include_parent_document` (bool) & `parent_max_tokens` (int)
- `default_llm_provider|default_llm_model` (str)
- `hyde_provider|hyde_model` (str): HyDE settings.
- `semantic_cache_enabled` (bool) & `cache_similarity_threshold` (float)
- `enable_reranking` (bool) & `rerank_top_k` (int)
- Rerankers: `llm_reranker_*`, `llama_reranker_*`, `transformers_reranker_model` (str)
  - Transformers reranker examples: `BAAI/bge-reranker-v2-m3`,
    `mixedbread-ai/mxbai-rerank-large-v2` (auto-enables `trust_remote_code`)
- Feature flags (phased rollout):
  - `enable_structure_index` (bool)
  - `strict_extractive` (bool)
  - `require_hard_citations` (bool)
  - `low_confidence_behavior` (str): `continue|ask|decline`
  - `agentic_cache_backend` (str): `memory|sqlite`
  - `agentic_cache_ttl_sec` (int)
  - `implicit_feedback_enabled` (bool): Enable implicit feedback capture (click/copy/expand/dwell/citation-used).

## [API] (Hosted providers)
For each provider, settings follow the pattern:
- `<provider>_model` (str), `<provider>_streaming` (bool), `<provider>_temperature` (float), `<provider>_top_p|min_p` (float), `<provider>_max_tokens` (int), `<provider>_api_timeout` (int sec), `<provider>_api_retry` (int), `<provider>_api_retry_delay` (int sec)
Providers present: `anthropic, cohere, deepseek, qwen, google, groq, huggingface, mistral, openai, openrouter` and generic OpenAI-compatible custom endpoint blocks.
- `custom-openai-api` maps to `custom_openai_api`; `custom-openai-api-2` maps to `custom_openai2_api` / `custom_openai_api_2`.
- Additional named OpenAI-compatible endpoints are supported as `custom-openai-api-3` through `custom-openai-api-99`. Define them with `custom_openaiN_api_*` or `custom_openai_api_N_*` fields, for example `custom_openai37_api_ip`, `custom_openai37_api_key`, and `custom_openai37_api_model`.
- Custom OpenAI-compatible endpoint URLs can be overridden with environment variables: `CUSTOM_OPENAI_API_IP` (aliases `CUSTOM_OPENAI_API_BASE` and `CUSTOM_OPENAI_API_URL`) for `custom_openai_api`, and `CUSTOM_OPENAI2_API_IP` for `custom_openai2_api`. Backward-compatible aliases `CUSTOM_OPENAI_API_IP_1` and `CUSTOM_OPENAI_API_IP_2` are also accepted. Provider-1 env vars do not override provider-2 config values. For endpoint `N` in `3..99`, use `CUSTOM_OPENAI_API_IP_N`, `CUSTOM_OPENAI_API_BASE_N`, `CUSTOM_OPENAI_API_URL_N`, `CUSTOM_OPENAI_API_KEY_N`, and `CUSTOM_OPENAI_API_MODEL_N` or the equivalent `CUSTOM_OPENAI{N}_API_*` / `CUSTOM_OPENAI_API_N_*` aliases.
- `model_for_summarization` (str): Preferred model for summarize endpoints.
- `default_api` (str): Default provider for chat.
- `default_api_for_tasks` (str): Default provider for batch/tools.

## [Local-API]
Local service backends (Kobold, LLaMA, Oobabooga, Tabby, vLLM, Ollama, Aphrodite). Each has similar knobs:
- `<engine>_api_IP` (url)
- `<engine>_model` (str, when applicable)
- `<engine>_streaming` (bool)
- `<engine>_temperature|top_p|min_p|top_k` (floats/ints)
- `<engine>_max_tokens` (int), `<engine>_api_timeout|_api_retry|_api_retry_delay`
Common: `max_tokens`, `local_api_timeout`, `local_api_retries`, `local_api_retry_delay`, `streaming`, `temperature`, `top_p`, `min_p`.

## [STT-Settings]
- `default_batch_transcription_model` (str): explicit batch/offline default model id when requests omit `model` (default: `parakeet-tdt-0.6b-v3-onnx`; legacy alias `parakeet-onnx` is still accepted).
- `default_streaming_transcription_model` (str): explicit WebSocket streaming default model id when clients omit `model` (default: `parakeet-tdt-0.6b-v3-onnx`; legacy alias `parakeet-onnx` is still accepted).
- `default_transcriber` (str): legacy provider default (`faster-whisper|parakeet|canary|...`) used for compatibility when explicit model defaults are absent.
- `nemo_model_variant` (str), `nemo_device` (str), `nemo_cache_dir` (path)
- `nemo_chunk_duration|nemo_overlap_duration` (sec)
- `parakeet_onnx_model_id` (str): Hugging Face repository id used by Parakeet ONNX loader (default: `istupakov/parakeet-tdt-0.6b-v3-onnx`).
- `parakeet_onnx_revision` (str|empty): optional pinned revision/commit for deterministic ONNX downloads.
- `streaming_fallback_to_whisper` (bool): opt-in fallback when streaming model init fails (default: `false`, fail-fast).
- `mlx_model_id` (str): default `mlx-community/parakeet-tdt-0.6b-v3`
- `mlx_cache_dir` (path): optional local cache override for model artifacts
- `mlx_chunk_duration|mlx_overlap_duration|buffered_chunk_duration|buffered_total_buffer` (sec)
- `buffered_merge_algo` (str): e.g., `lcs`.
- `mlx_decoding_mode` (str): `greedy|beam`
- `mlx_beam_size|mlx_length_penalty|mlx_patience|mlx_duration_reward` (beam controls)
- `mlx_sentence_max_words|mlx_sentence_silence_gap|mlx_sentence_max_duration` (sentence splitting controls)
- `mlx_stream_context_left|mlx_stream_context_right|mlx_stream_depth` (native MLX streaming context/cache controls)
- `mlx_stream_keep_original_attention` (bool): keep upstream full attention model during MLX streaming
- `whisper_compute_type` (str): Optional faster-whisper compute type override; when unset or `"auto"` the server uses `float16` on CUDA and `int8` on CPU; examples: `float16`, `int8`, `int8_float16`.
- Transcript cache toggles:
  - `disable_transcript_cache` (bool): When true, do not write `.segments.json` transcript cache files at all.
  - `disable_transcript_cache_pruning` (bool): When true, keep writing cache files but skip age/size-based pruning (cache may grow without bound).
  - `transcript_cache_max_files_per_source` (int): Max cached transcript files per base source; newest files are kept first. Leave empty for defaults or set 0/negative to disable this limit.
  - `transcript_cache_max_age_days` (int): Delete cached transcripts older than this many days. Leave empty for defaults or set 0/negative to disable.
- `transcript_cache_max_total_mb` (float): Cap total transcript cache size (MB) per directory; oldest files are evicted when exceeded. Leave empty for defaults or set 0/negative to disable.
- `skip_audio_prevalidation` (bool): When true, skip `ffprobe`-based audio validation and rely on `ffmpeg` + STT to surface bad files (useful for high-throughput deployments).
- Runtime transcription requests do not auto-install `parakeet-mlx`; install dependencies during setup/deployment.
- Design reference: `Docs/Design/STT_Parakeet_MLX_Parity.md`

## [external_providers]
- Reserved for plugging in external providers (YAML/INI sub-configs).

## [TTS-Settings]
General and provider-specific:
- Canonical keys:
  - `local_device` (str): `cpu|cuda|auto`.
  - `default_provider` (str), `default_voice` (str), `default_speed` (float)
  - `TTS_VOICE_REGISTRY_ENABLED` (bool-like via env/settings): when `false`, disables persistent `voice_registry.db` and uses runtime/filesystem-only compatibility mode for custom voices (deprecated; planned removal after 2026-12-31).
- Compatibility aliases (deprecated; scheduled for removal after 2026-06-30):
  - `local_tts_device`/`tts_device` -> `local_device`
  - `default_tts_provider` -> `default_provider`
  - `default_tts_voice` -> `default_voice`
  - `default_tts_speed` -> `default_speed`
- Precedence for effective values: environment variables > `config.txt` `[TTS-Settings]` > `tts_providers_config.yaml` > defaults.
OpenAI TTS:
- `default_openai_tts_voice|_speed|_model|_output_format|_streaming`
ElevenLabs TTS (placeholders if not used):
- `default_eleven_tts_*` keys (voice/model/language/tunables)
Google/Edge (placeholders): `default_google_tts_*`, `edge_tts_voice`
AllTalk:
- `default_alltalk_tts_*`, `alltalk_api_ip`
Kokoro (local):
- `kokoro_model_path`, `default_kokoro_tts_*`
KittenTTS (local):
- Configure in `tts_providers_config.yaml` under `providers.kitten_tts`.
- Key fields: `model`, `model_revision`, `device`, `auto_download`, and `extra_params.cache_dir`.
- `model_revision` should remain an immutable Hugging Face commit hash. The shipped default pins the bundled nano model to a vetted commit.
- Voices are static English presets: `Bella`, `Jasper`, `Luna`, `Bruno`, `Rosie`, `Hugo`, `Kiki`, `Leo`.
- First-use downloads can be disabled globally with `auto_download_local_models = false` or per-provider with `providers.kitten_tts.auto_download: false`.
Custom OpenAI-compatible TTS:
- `default_custom_openai_*`
VibeVoice:
- `vibevoice_*` keys: variant/model/device/quantization/streaming params/paths

## [Search-Engines]
- `search_provider_default` (str)
- `search_language_query|results|analysis` (str)
- `search_default_max_queries` (int)
- `search_enable_subquery` (bool) & `search_enable_subquery_count_max` (int)
- `search_result_rerank` (bool)
- `search_result_max|search_result_max_per_query` (int)
- `search_result_blacklist` (json/array)
- `search_result_display_type` (str) & `search_result_display_metadata` (bool)
- `search_result_save_to_db` (bool)
- `search_result_analysis_tone` (str)
- Per-engine keys: API keys/IDs/region/language filters and router URLs.
  - Examples:
    - Google: `search_engine_api_key_google`, `search_engine_id_google`, `search_engine_api_url_google`
    - Brave: `search_engine_api_key_brave_regular`, `search_engine_api_key_brave_ai`
    - Searx: `search_engine_searx_api`
    - Tavily: `search_engine_api_key_tavily`
    - Serper: `search_engine_api_key_serper`, `search_engine_api_url_serper`
    - Exa: `search_engine_api_key_exa`, `search_engine_api_url_exa`
    - Firecrawl: `search_engine_api_key_firecrawl`, `search_engine_api_url_firecrawl`

## [Web-Scraper]
- `web_scraper_api_key|web_scraper_api_url` (str)
- `web_scraper_api_timeout|web_scraper_api_retry|web_scraper_api_retry_delay` (int)
- `web_scraper_retry_count` (int)
- `web_scraper_stealth_playwright` (bool)
- `custom_scrapers_yaml_path` (path, optional): override custom scraper router rules file
- `web_scraper_default_backend` (str, default `auto`): `auto|curl|httpx|playwright`
- `web_scraper_ua_mode` (str, default `fixed`): `fixed|rotate`
  - Crawl flags (env overrides file):
    - `web_crawl_strategy` (str) → `WEB_CRAWL_STRATEGY` (default `default`)
    - `web_crawl_include_external` (bool) → `WEB_CRAWL_INCLUDE_EXTERNAL` (default `false`)
    - `web_crawl_score_threshold` (float) → `WEB_CRAWL_SCORE_THRESHOLD` (default `0.0`)
    - `web_crawl_max_pages` (int) → `WEB_CRAWL_MAX_PAGES` (default `100`)
    - `web_crawl_allowed_domains` (csv) → restrict crawl to specific domains (always includes base host when `include_external` is false)
    - `web_crawl_blocked_domains` (csv) → domains to exclude (subdomains included)
  - Scoring (env overrides file):
    - `web_crawl_enable_keyword_scorer` (bool) → `WEB_CRAWL_ENABLE_KEYWORD_SCORER` (default `false`)
    - `web_crawl_keywords` (csv) → `WEB_CRAWL_KEYWORDS` (e.g., `ai,ml,python`)
    - `web_crawl_enable_domain_map` (bool) → `WEB_CRAWL_ENABLE_DOMAIN_MAP` (default `false`)
    - `web_crawl_domain_map` (json or `domain:score` csv) → `WEB_CRAWL_DOMAIN_MAP`
      - Examples: `{"docs.python.org": 1.0, "github.com": 0.8}` or `docs.python.org:1.0,github.com:0.8`
  - Robots policy:
    - `web_scraper_respect_robots` (bool, default `true`): enforce robots.txt for egress-allowed hosts
      - Resolved per-domain with caching; missing robots.txt fails open.
  - Patterns:
    - Default excludes include common non-content paths (e.g., `/tag/`, `/category/`, `wp-content`, images/docs). These are applied internally.
    - Advanced pattern controls can be added via code (see `URLPatternFilter` in `app/core/Web_Scraping/filters.py`).

## [Logging]
- `log_level` (str): `DEBUG|INFO|WARN|ERROR`.
- `log_file` (path), `log_metrics_file` (path)
- `max_bytes` (int|null): log rotation size
- `backup_count` (int): rotated files kept

## [HTTP-Client]
- Centralized outbound HTTP client configuration (applies to helpers in `tldw_Server_API.app.core.http_client`).
- Defaults are secure-by-default and can be overridden via environment variables or the `[HTTP]` section in `Config_Files/config.txt` (which is mapped into the corresponding `HTTP_*` env vars at startup when unset).

- Timeouts
  - `HTTP_CONNECT_TIMEOUT` (float, default `5.0` seconds)
  - `HTTP_READ_TIMEOUT` (float, default `30.0` seconds)
  - `HTTP_WRITE_TIMEOUT` (float, default `30.0` seconds)
  - `HTTP_POOL_TIMEOUT` (float, default `30.0` seconds)

- Connection limits
  - `HTTP_MAX_CONNECTIONS` (int, default `100`)
  - `HTTP_MAX_KEEPALIVE_CONNECTIONS` (int, default `20`)

- Retries & backoff
  - `HTTP_RETRY_ATTEMPTS` (int, default `3`)
  - `HTTP_BACKOFF_BASE_MS` (int, default `250`)
  - `HTTP_BACKOFF_CAP_S` (int, default `30`)
  - Retries on: 408, 429, 500, 502, 503, 504, and connect/read timeouts. Honors `Retry-After`.

- Redirects & proxies
  - `HTTP_MAX_REDIRECTS` (int, default `5`)
  - `HTTP_TRUST_ENV` (bool, default `false`) — when false, system proxies are ignored
  - `PROXY_ALLOWLIST` (csv of hosts or URLs; deny-by-default)

- JSON & headers
  - `HTTP_JSON_MAX_BYTES` (int, optional) — maximum allowed JSON response size for helpers that enable this guard
  - `HTTP_DEFAULT_USER_AGENT` (string, overrides default `tldw_server/<version> (component)`)

- Transport & TLS
  - `HTTP3_ENABLED` (bool, default `false`) — HTTP/3 (QUIC) behind a flag. Note: currently a no‑op; reserved for future QUIC support.
  - `TLS_ENFORCE_MIN_VERSION` (bool, default `false`) — optional TLS min version enforcement
  - `TLS_MIN_VERSION` (str, default `1.2`)
  - `TLS_CERT_PINS_SPKI_SHA256` (csv of SPKI SHA-256 pins; optional certificate pinning)

- Proxies & Egress
  - `PROXY_ALLOWLIST` (csv of proxy hostnames or URLs; deny-by-default when empty)

`config.txt` mapping for `[HTTP]`
- Keys in `[HTTP]` are convenience aliases that populate the process environment if the corresponding env var is not already set:
  - `connect_timeout` → `HTTP_CONNECT_TIMEOUT`
  - `read_timeout` → `HTTP_READ_TIMEOUT`
  - `write_timeout` → `HTTP_WRITE_TIMEOUT`
  - `pool_timeout` → `HTTP_POOL_TIMEOUT`
  - `retry_attempts` → `HTTP_RETRY_ATTEMPTS`
  - `backoff_base_ms` → `HTTP_BACKOFF_BASE_MS`
  - `backoff_cap_s` → `HTTP_BACKOFF_CAP_S`
  - `max_connections` → `HTTP_MAX_CONNECTIONS`
  - `max_keepalive_connections` → `HTTP_MAX_KEEPALIVE_CONNECTIONS`
  - `trust_env` → `HTTP_TRUST_ENV`
  - `default_user_agent` → `HTTP_DEFAULT_USER_AGENT`
  - `json_max_bytes` → `HTTP_JSON_MAX_BYTES`
  - `http3_enabled` → `HTTP3_ENABLED`
  - `proxy_allowlist` → `PROXY_ALLOWLIST`
  - `enforce_tls_min_version` → `HTTP_ENFORCE_TLS_MIN`
  - `tls_min_version` → `HTTP_TLS_MIN_VERSION`
  - `cert_pins` → `HTTP_CERT_PINS`
  - `allow_redirects` → `HTTP_ALLOW_REDIRECTS`
  - `max_redirects` → `HTTP_MAX_REDIRECTS`
  - `allow_cross_host_redirects` → `HTTP_ALLOW_CROSS_HOST_REDIRECTS`
  - `allow_scheme_downgrade` → `HTTP_ALLOW_SCHEME_DOWNGRADE`

TLS and certificate pinning

By default the HTTP client follows system trust stores. You can optionally enforce a minimum TLS version and use certificate pinning on a per-host basis.

- Env toggles for TLS minimum version:
  - `HTTP_ENFORCE_TLS_MIN` or `TLS_ENFORCE_MIN_VERSION`: set to `1`/`true` to enable
  - `HTTP_TLS_MIN_VERSION` or `TLS_MIN_VERSION`: `1.2` (default) or `1.3`

- Programmatic per-host certificate pinning (leaf certificate SHA-256):

```python
from tldw_Server_API.app.core.http_client import create_async_client, afetch, RetryPolicy

# Map of host -> set of allowed certificate fingerprints (hex sha256 of DER)
pins = {
    "api.openai.com": {"b1e5...deadbeef"},
    "api.groq.com": {"a2c4...c0ffee"},
}

async with create_async_client(enforce_tls_min_version=True, tls_min_version="1.2", cert_pinning=pins) as client:
    resp = await afetch(method="GET", url="https://api.openai.com/v1/models", client=client, retry=RetryPolicy())
    print(resp.status_code)
```

Notes
- Pinning checks the leaf certificate fingerprint (sha256 of the DER cert) before the request proceeds. A mismatch raises an egress/pinning error.
- Env-driven pinning (built-in parser): set `HTTP_CERT_PINS` to a CSV-style mapping of host to pins
  - Example: `HTTP_CERT_PINS="api.openai.com=ab12..|cd34..,api.groq.com=ef56.."`
  - Format: `host=pin1|pin2[,host2=pin3]` where pins are lowercase sha256 hex of the leaf certificate DER.
  - These pins are attached to clients created by `create_client`/`create_async_client` when `cert_pinning` is not provided.

- Egress & SSRF policy
  - All helpers evaluate the central egress policy (`app/core/Security/egress.py`) before any network I/O and on each redirect hop, and validate proxies.
  - Denies unsupported schemes, disallowed ports, denylisted hosts, and private/reserved IPs by default. See `WORKFLOWS_EGRESS_*` env keys in that module for allow/deny behavior.

- Observability
  - Structured logs redact sensitive headers and may include `request_id`, `method`, `host`, `status`, `duration_ms`.
  - Metrics (if telemetry enabled): `http_client_requests_total`, `http_client_request_duration_seconds`, `http_client_retries_total`, `http_client_egress_denials_total`.
  - When tracing is active, `traceparent` header is injected automatically where supported.

X-Request-Id propagation

Outbound helpers auto-inject `X-Request-Id` when present in trace baggage (set via RequestID middleware or `TracingManager.set_baggage('request_id', ...)`). Example:

```
from tldw_Server_API.app.core.Metrics.traces import get_tracing_manager
from tldw_Server_API.app.core.http_client import create_client, fetch

tm = get_tracing_manager()
tm.set_baggage('request_id', 'abc123')

with create_client() as client:
    r = fetch(method='GET', url='http://example.com', client=client)
    assert r.status_code == 200
```

SSE streaming example

```
from tldw_Server_API.app.core.http_client import create_async_client, astream_sse, RetryPolicy

async def consume():
    async with create_async_client() as client:
        policy = RetryPolicy(attempts=3)
        async for ev in astream_sse(url='http://example.com/stream', client=client, retry=policy):
            print(ev.event, ev.data)
```

Downloads with checksum and resume

```
from pathlib import Path
from tldw_Server_API.app.core.http_client import download, adownload, RetryPolicy

dest = Path('/tmp/file.bin')
policy = RetryPolicy(attempts=3)

# Sync
download(
    url='http://example.com/file.bin',
    dest=dest,
    checksum='deadbeef...',  # optional sha256
    resume=True,
    retry=policy,
    require_content_type='application/pdf',  # optional strict content-type
    max_bytes_total=50_000_000,              # optional disk quota guard (bytes)
)

# Async
# await adownload(
#     url='http://example.com/file.bin',
#     dest=dest,
#     resume=True,
#     retry=policy,
#     require_content_type='application/pdf',
#     max_bytes_total=50_000_000,
# )
```

Example (Python)
```
from tldw_Server_API.app.core.http_client import create_async_client, afetch_json

async with create_async_client() as client:
    data = await afetch_json(method="GET", url="https://api.example.com/items", client=client)
```

Downloads and streaming
```
from tldw_Server_API.app.core.http_client import adownload, astream_sse, RetryPolicy

# Reliable file downloads with optional checksum/length validation
await adownload(url="https://host/file.bin", dest="/tmp/file.bin", retry=RetryPolicy(attempts=3))

# Stream SSE events with backpressure-friendly async iteration
async for evt in astream_sse(method="GET", url="https://host/stream"):
    print(evt.event, evt.data)
```

## [Egress]
- Centralized outbound egress policy configuration for HTTP clients and workflows (backed by `app/core/Security/egress.py`).
- Values here are mapped into `EGRESS_*` / `WORKFLOWS_EGRESS_*` env vars when unset.

- Global allow/deny lists
  - `egress_allowlist` → `EGRESS_ALLOWLIST` (csv of hostnames/domains)
  - `egress_denylist` → `EGRESS_DENYLIST`

- Workflows-specific allow/deny overrides
  - `workflows_allowlist` → `WORKFLOWS_EGRESS_ALLOWLIST`
  - `workflows_denylist` → `WORKFLOWS_EGRESS_DENYLIST`

- Ports, profile, private IPs
  - `allowed_ports` → `WORKFLOWS_EGRESS_ALLOWED_PORTS` (csv of ints; default `80,443`)
  - `block_private` → `WORKFLOWS_EGRESS_BLOCK_PRIVATE` (bool; default `true`)
  - `profile` → `WORKFLOWS_EGRESS_PROFILE` (`strict|permissive|custom`)

Notes
- The egress policy denies unsupported schemes, disallowed ports, denylisted hosts, and private/reserved IP ranges by default (when `block_private=true`).
- HTTP helpers in `http_client.py` and workflows/webhook components consult this policy before network I/O and on each redirect hop.
- Local LLM model discovery (e.g., Aphrodite/Oobabooga/VLLM/Ollama endpoints) also uses the same egress policy. If a local endpoint runs on non-default ports like `8080` or on `127.0.0.1`, discovery calls can be blocked unless the port is listed in `WORKFLOWS_EGRESS_ALLOWED_PORTS` and private IP blocking is disabled via `WORKFLOWS_EGRESS_BLOCK_PRIVATE=false`. To avoid discovery entirely, set the provider’s model explicitly in config (e.g., `aphrodite_model=...`).

## [Moderation]
- `enabled` (bool)
- `input_enabled|output_enabled` (bool)
- `input_action|output_action` (str): `block|redact|warn`
- `redact_replacement` (str)
- `blocklist_file|user_overrides_file` (path)
- `per_user_overrides` (bool)
- `pii_enabled` (bool), `categories_enabled` (csv)
- `runtime_overrides_file` (path)

## [Redis]
- `redis_enabled` (bool)
- `redis_host|redis_port|redis_db` (str|int)
- `cache_ttl` (int): default TTL seconds for cached items

## [Web-Scraping]
- `stealth_wait_ms` (int): Delay to allow stealth page prep.

## [personalization]
- `enabled` (bool)
- `alpha|beta|gamma` (floats): Weighting for recommender signals.
- `recency_half_life_days` (int)

## [persona]
- `enabled` (bool)
- `default_persona` (str)
- `voice` (str), `stt` (str): Preferred voice/STT for persona.
- `max_tool_steps` (int)

## [persona.rbac]
- `allow_export` (bool)
- `allow_delete` (bool)

---

Notes
- Some keys are placeholders (`FIXME`); leave empty or set via `.env` when a provider requires credentials.
- When both provider-specific and global keys exist, provider-specific keys take precedence for that provider’s operations.
- For advanced tuning of chunking behavior, see `tldw_Server_API/app/core/Chunking/README.md`.

---

## API Route Toggles

Purpose: allow end users to run only stable API modules and let developers/operators selectively enable in-development endpoints. Route inclusion is evaluated at startup before routers are mounted.

Default behavior
- Stable-only mode is ON by default. A curated set of experimental routes is disabled unless explicitly enabled.
- You can override behavior per route or globally via `config.txt` or environment variables.

Configure in `config.txt`
- Section: `[API-Routes]`

## Streaming

Purpose: control unified streaming behavior and chat stream channel size from `config.txt`.

Section: `[Streaming]`
- `streams_unified` (bool): Enable the centralized SSE/WS streaming helpers for chat and selected SSE endpoints. Standardizes heartbeats, idle timeout handling, and metrics. Accepts `true|false`.
- `chat_stream_channel_maxsize` (int, optional): Bounded size of the per-request in-memory channel used when chat requests are executed via the internal queue. Defaults to `100`. You may also set this in `[Chat-Module]` as `chat_stream_channel_maxsize`; the `[Chat-Module]` value takes precedence.

Section: `[Chat-Module]`
- `chat_stream_channel_maxsize` (int, optional): Same as above; co-located with other chat streaming knobs like `streaming_idle_timeout_seconds` and `streaming_heartbeat_interval_seconds`.

Notes
- Environment variables still override file settings (`STREAMS_UNIFIED`, `CHAT_STREAM_CHANNEL_MAXSIZE`).
- When `streams_unified=true`, the chat `/api/v1/chat/completions` streaming path uses `SSEStream` with bounded queues and heartbeat/idle enforcement.
- Keys:
  - `stable_only` (bool, default: true): When true, disables the experimental set unless explicitly enabled.
  - `disable` (csv): Route keys to always disable.
  - `enable` (csv): Route keys to force-enable (useful when `stable_only=true`).
  - `experimental_routes` (csv): Extend the curated experimental set with more route keys.

Environment overrides
- `ROUTES_STABLE_ONLY=true|false`
- `ROUTES_DISABLE="sandbox,connectors"`
- `ROUTES_ENABLE="workflows"`
- `ROUTES_EXPERIMENTAL="workflows,scheduler"`

Known route keys (not exhaustive)
`health, metrics, monitoring, moderation, audit, auth, auth-enhanced, users, privileges, admin, mcp-catalogs, media, audio, audio-jobs, audio-websocket, chat, characters, character-chat-sessions, character-messages, chunking, chunking-templates, outputs-templates, outputs, embeddings, vector-stores, connectors, claims, media-embeddings, items, reading, watchlists, subscriptions-deprecated, notes, prompts, reading-highlights, prompt-studio, rag-health, rag-unified, workflows, scheduler, research, paper-search, evaluations, ocr, vlm, benchmarks, setup, config, jobs, sync, tools, sandbox, flashcards, personalization, persona, mcp-unified, chatbooks, llm, llamacpp, web-scraping`

Curated experimental set (gated by `stable_only=true` unless explicitly enabled)
`sandbox, connectors, workflows, scheduler, flashcards, personalization, persona, jobs, benchmarks`

Examples
- Run stable-only with workflows enabled:
  - `stable_only = true`
  - `enable = workflows`
- Disable web-scraping and connectors regardless of `stable_only`:
  - `disable = web-scraping, connectors`

Notes
- Control-plane endpoints are also gated:
  - `metrics` key: gates both `/metrics` (Prometheus text) and `/api/v1/metrics` (JSON).
  - `health` key: gates `/health`, `/ready`, and `/health/ready`, as well as health routers (`/healthz`, `/readyz`).
  - `setup` key: gates the Setup UI (`/setup`) and the setup API routes; root (`/`) only redirects to `/setup` when this key is enabled and setup is required.
