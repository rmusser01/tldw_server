# Chunking and embeddings

This document describes text segmentation and vectorization: the strategy-based `Chunker` with templates and the auto planner, the metadata contract, the embeddings provider engine and its async service, per-user ChromaDB collections, and the Jobs/Redis staged pipeline (ADR-022).

## Authoritative files

All paths under `tldw_Server_API/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| Chunker | `app/core/Chunking/chunker.py` | `Chunker`, `chunk_text_hierarchical_tree/flat`, `chunk_text_with_metadata`, `process_text`, `create_chunker` |
| Contracts | `Chunking/base.py` | `ChunkingMethod`, `ChunkResult`, `BaseChunkingStrategy` (grapheme-boundary expansion), `ChunkerConfig` (defaults: words / 400 / 200, 100 MB ceiling) |
| Strategies | `Chunking/strategies/` (14) | words, sentences, paragraphs, tokens, fixed_size, semantic, json_xml, ebook_chapters, propositions, rolling_summarize, structure_aware, code, code_ast |
| Templates | `Chunking/templates.py`, `template_initialization.py` | `TemplateProcessor`/`TemplateManager`/`TemplateClassifier`/`TemplateLearner`; builtin seeding |
| Facade | `Chunking/__init__.py` | `DEFAULT_CHUNK_OPTIONS`, the `improved_chunking_process` shim, `chunk_for_embedding` (adaptive default) |
| Embeddings engine | `Chunking/../Embeddings/Embeddings_Server/Embeddings_Create.py` | provider engine (OpenAI / HF Transformers / ONNX / local API), effective-provider resolution, model storage, token-bucket limiter |
| Async service | `Embeddings/async_embeddings.py` | `AsyncEmbeddingService` + four async providers, batch creation |
| Chroma manager | `Embeddings/ChromaDB_Library.py` | `ChromaDBManager` — per-user paths, `process_and_store_content`, `situate_context` (contextual retrieval), dimension-mismatch collection recreation, simhash near-dup dedupe |
| Jobs adapter | `Embeddings/jobs_adapter.py` | `EmbeddingsJobsAdapter` — root `embeddings_pipeline` job with stages `{chunking, embedding, storage}`, per-stage idempotency `{media_id}:{stage}:{version}` |
| Pipeline services | `app/services/redis_worker.py`, `vector_compactor.py`, `multi_tier_cache.py`, `rate_limiter.py`, `dlq_crypto.py` | Redis Streams stage workers (Jobs owns root status/billing per ADR-022), compaction, caches, backpressure, encrypted DLQ |
| Vector stores (RAG side) | `app/core/RAG/rag_service/vector_stores/` | adapter factory — ChromaDB always registered; PGVector import-conditional |
| Endpoints | `…/endpoints/embeddings_v5_production_enhanced.py`, `media_embeddings.py` | create/models/providers/collections/health/DLQ; media generate/batch/status/search/delete/jobs |

## The three-passes trap (read this first)

One ingestion can be chunked **three independent times**: processor-time (ephemeral result chunks), persist-time (`chunk_text_hierarchical_flat` with its own defaults when options are present), and embeddings-time (`chunk_media_content`, words/1000/200). Counts can disagree across stages — a post-persist consistency check exists precisely for this. Never assume the chunks the API returned are the chunks stored.

## Embeddings dataflow (media)

1. A successful `/media/add` item schedules embeddings (jobs preferred, background fallback per dispatch mode).
2. Jobs path: a root `embeddings_pipeline` job plus stage jobs feed Redis Streams queues (`embeddings:chunking|embedding|storage`) whose workers process stages and upsert to Chroma.
3. Background path: media content → `chunk_media_content` → async batch embeddings → upsert into the per-user `user_{id}_media_embeddings` collection with chunk metadata (media id, offsets, chunk type, title/author, model/provider) → mark processed or embeddings-error.
4. The Chroma manager path (`process_and_store_content`) chunks via `chunk_for_embedding`, optionally contextualizes with `situate_context` (LLM, config-gated), embeds, upserts, and writes vector references back to SQL.

## Boundaries

- ChromaDB is per-user, per-collection: `user_{id}_media_embeddings` / `notes_embeddings` / `character_embeddings` / `kanban_embeddings`. pgvector exists **only** behind the RAG adapter factory (`rag.vector_store_type`, `rag.pgvector.*`) and is import-conditional.
- The embeddings API is OpenAI-compatible; `dimensions` is honored only for OpenAI text-embedding-3 models — HF/ONNX dims are fixed.
- Server chunks carry **no engine version stamp** (chatbook's `parity-1@…` stamp is chatbook-only); identity is method/max/overlap metadata + uuid.

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| Provider failure | Circuit breaker trip → 5xx/503; 429 rate limits; 4xx validation/policy |
| Dimension mismatch on store | Target collection recreated |
| Transient Chroma segment errors | Retried; fresh clients can momentarily miss just-written collections (retrieval retries `get_or_create_collection`) |
| Backpressure | Rejects when queues deep (`EMB_BACKPRESSURE_*`); DLQ with optional encryption and requeue controls |

## Config keys (selection)

Env: `EMBEDDINGS_MAX_BATCH_SIZE`, `_CONNECTION_POOL_SIZE`, `_REQUEST_TIMEOUT`, `_MAX_RETRIES`, cache TTL/size keys, `EMBEDDINGS_ENFORCE_POLICY[_STRICT]`, `_ALLOW_FALLBACK_WITH_HEADER`, `EMBEDDINGS_DIMENSION_POLICY` (reduce|pad|ignore), `EMB_BACKPRESSURE_*`, `REDIS_URL`, `PRELOAD_EMBEDDING_MODELS`, `AUTO_DOWNLOAD_MODELS`, `TRUSTED_HF_REMOTE_CODE_MODELS`, `EMBEDDINGS_REDIS_STREAM_*`, `CHROMADB_FORCE_STUB`. config.txt `[Embeddings]` (capital E): `embedding_provider`, `embedding_model` (provider-resolution default all-MiniLM-L6-v2; core config fallback `Qwen/Qwen3-Embedding-4B-GGUF`; the shipped config sets `Qwen/Qwen3-Embedding-0.6B`), fallback model, `auto_generate_on_upload`, `embedding_dimension`, `enable_contextual_chunking` (false) + contextual LLM model, chroma backend/stub keys. `[Chunking]`: per-type method/size/overlap families + globals, propositions engine keys, streaming flush threshold, JSON metadata-reference keys, regex safety keys.

## Verified gotchas

1. **No `template_library/` directory ships** — builtin templates are hardcoded seeds, despite older docs describing a JSON library on disk.
2. `overlap` is re-clamped to `< max_size` inside strategies; `words`-strategy text is reconstructed by joining tokens (use `align_text_to_source=True` for source slices).
3. `ChunkerConfig.strategy_cache_mode` defaults disagree between dataclass ("shared") and the `Chunker` fallback string ("thread").
4. Several `[Chunking]` booleans parse as raw strings (`'False'`) in the config dict — treat as strings, not bools.
5. `Docs/Code_Documentation/Chunking-Module.md` is stale-ish: it predates timecode maps, frontmatter, the propositions engine, and claims the nonexistent template library.

## Governing decisions

ADR-022 (`Docs/ADR/022-embeddings-api-and-media-pipeline.md` — Jobs root + Redis stages), ADR-020 (per-user paths). Current references: `Embeddings-Developer-Guide.md`, `Embeddings-Documentation.md`, module READMEs.

## Related docs

- [ingestion.md](./ingestion.md) — where chunking is invoked
- [rag.md](./rag.md) — what the vectors serve
