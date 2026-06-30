# Auto Chunk Boundary Assistant Design

## Goal

Add the real LLM-backed boundary assistant deferred by Auto Chunking V1. The assistant refines deterministic Auto Chunking decisions only when a request explicitly sets `auto_chunking_use_llm=true`, and every unavailable, timeout, provider, or invalid-response path must keep the deterministic Auto plan.

## Scope

This is a backend-only slice. It does not change Quick Ingest UI controls or add new public chunking fields. Existing Manual and legacy requests continue to use `prepare_chunking_options_dict()` and template application. Auto requests continue to start with the deterministic planner so the LLM assistant can only refine bounded planner outputs, not replace source text or invent arbitrary chunk payloads.

## Assistant Contract

Introduce a narrow `AutoChunkBoundaryAssistant` interface under `tldw_Server_API/app/core/Chunking/`. The interface accepts a deterministic plan, deterministic chunk options, request profile, media hints, and a bounded text excerpt. It returns a result with either validated refined options plus metadata or a typed fallback reason. The result type is defined before provider calls so tests can exercise the resolver without invoking an LLM.

The concrete assistant wraps the existing async chat service path, `perform_chat_api_call_async`, because this is a legacy async ingestion call site and the LLM adapter guide points legacy async callers there. The prompt requests strict JSON only. Accepted refinements are intentionally small: method, max size, overlap, derived views, and rationale. Raw chunk text, rewritten content, direct chunk bodies, file paths, URLs to fetch, and code-like instructions are ignored.

## Availability

Availability is explicit and separate from provider key presence. The assistant is available only when all of these are true:

- The request is Auto mode and `auto_chunking_use_llm=true`.
- A provider can be resolved from request fields or server defaults.
- A model can be resolved from request fields or provider config defaults.
- The provider has a registered chat adapter.
- Required API-key providers have a resolved API key.

If any requirement fails, the resolver preserves deterministic options and records an `ai_assist_unavailable` fallback reason.

## Data Flow

The existing sync resolver remains the deterministic/manual compatibility path. A new async resolver first calls the sync resolver. If no Auto plan is produced, it returns unchanged results. If Auto is active and LLM assist is not requested, it returns unchanged deterministic results. If opt-in is present, it checks assistant availability and then calls the assistant with bounded input.

On a valid assistant result, the async resolver updates `chunk_options` and `chunking_plan` consistently and sets `used_llm=true`. On timeout, provider error, invalid JSON, invalid values, or unavailable config, it returns the original deterministic `chunk_options` and `chunking_plan`, sets `used_llm=false`, and appends fallback metadata.

## Wiring

Async ingestion paths should call the async resolver:

- `/api/v1/media/add` orchestration in persistence
- media ingest jobs worker
- direct process endpoints that already run in async request handlers
- web/article ingestion services

Template application remains manual/legacy-only. Call sites should only run `apply_chunking_template_if_any()` when the returned `chunking_plan is None`.

## Testing

Tests cover the interface/result shape before provider calls, default no-call behavior, explicit opt-in success, unavailable provider fallback, timeout/provider error fallback, invalid-response fallback, and `used_llm` semantics. Focused integration tests cover at least one async resolver call and one ingestion worker/media-add path so the resolved options used for chunk generation/FTS match the returned metadata.
