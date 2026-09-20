# Secondary LLM Surfaces Atomic Credential Snapshot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every assigned non-Chat/RAG LLM surface dispatch one authoritative server credential/config snapshot and prove configured-key rotations cannot splice generations at the adapter boundary.

**Architecture:** Reuse `resolve_static_server_fallback` as the structured server fallback passed to `resolve_byok_credentials`. Carry the resulting `app_config` together with `credentials_resolved=True` through existing service methods to the final adapter or `Summarization_General_Lib.analyze` call; preserve old callers with optional keyword defaults and capture Prompt Studio HTTP background work once before scheduling.

**Tech Stack:** Python 3.11, FastAPI, asyncio, pytest, existing BYOK runtime and LLM adapter abstractions.

## Global Constraints

- Work is tracked by existing Backlog task `TASK-12963`.
- Server-side credentials only; request-body API keys remain ignored.
- Missing static credentials fail closed and cannot be recovered by a later environment/config read.
- Do not edit Messages, Embeddings, admin provider tests, `user_keys.py`, `byok_runtime.py`, credential database/settings code, or TTS.
- Add no dependencies or new credential abstractions.
- Use RED → GREEN for each stage and do not commit this delegated slice.

---

## Stage 1: Character chat and unified evaluations

**Status:** Complete

**Goal:** Make character router/completion and all three unified evaluation routes preserve the captured key/config generation through their real LLM boundary.

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py`
- Modify: `tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py`
- Modify: `tldw_Server_API/app/core/Evaluations/ms_g_eval.py`
- Modify: `tldw_Server_API/app/core/Evaluations/rag_evaluator.py`
- Modify: `tldw_Server_API/app/core/Evaluations/response_quality_evaluator.py`
- Test: `tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py`
- Test: `tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_session_error_mapping.py`
- Test: `tldw_Server_API/tests/Evaluations/test_evaluations_unified.py`
- Test: `tldw_Server_API/tests/Evaluations/unit/test_unified_evaluation_service_mapping.py`

**Interfaces:**
- Endpoint fallback callable: `resolve_static_server_fallback(provider) -> ServerFallbackCredentials`.
- Evaluation methods gain optional `app_config: dict[str, Any] | None = None` and `credentials_resolved: bool = False` keywords.
- Final adapter/analyzer requests include the captured `api_key`, `app_config`, and `credentials_resolved=True` for resolved endpoint calls.

- [ ] Add parameterized A→B and absent→B character-router and completion regressions. Their fake resolver calls the endpoint fallback, rotates the legacy source, and asserts the provider call still receives the captured snapshot plus the authoritative marker.
- [ ] Run the two character node groups and verify they fail because current fallbacks are strings and dispatch omits `credentials_resolved`.
- [ ] Replace both character fallback closures with the existing structured helper and add the marker at both dispatches.
- [ ] Add G-Eval, RAG, and response-quality endpoint/service regressions that capture analyzer/adapter kwargs for both rotations.
- [ ] Run the evaluation node groups and verify RED on missing `app_config`/`credentials_resolved` propagation.
- [ ] Thread optional snapshot keywords through the evaluation service and evaluators, then pass them to `_call_adapter_text` or `analyze` without changing unrelated evaluation APIs.
- [ ] Re-run the focused character/evaluation groups GREEN.

## Stage 2: Prompt Studio synchronous and HTTP-background evaluation

**Status:** Complete

**Goal:** Resolve one Prompt Studio credential snapshot and prevent scheduled work from recapturing a later key generation.

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_evaluations.py`
- Modify: `tldw_Server_API/app/core/Prompt_Management/prompt_studio/evaluation_manager.py`
- Test: `tldw_Server_API/tests/prompt_studio/unit/test_evaluation_bg_propagation_unit.py`
- Test: `tldw_Server_API/tests/prompt_studio/unit/test_evaluation_endpoint_error_mapping.py`

**Interfaces:**
- `run_evaluation_async(..., byok_resolution: ResolvedByokCredentials | None = None)` accepts the request-captured resolution for FastAPI background execution and resolves only when called without one.
- Evaluation-manager entry points gain optional `credentials_resolved: bool = False`; their adapter bridge places the flag in the actual adapter request.

- [ ] Add sync A→B/absent→B adapter-boundary tests and background tests that schedule with generation A/absent, rotate to B, execute the captured task, and assert no recapture.
- [ ] Run the new Prompt Studio tests and verify RED on string fallback, background re-resolution, and missing adapter marker.
- [ ] Use the structured fallback in sync/background resolution, pass the request-captured resolution into FastAPI background tasks, and propagate the marker through `EvaluationManager`/`TestRunner`.
- [ ] Re-run the focused Prompt Studio tests GREEN.

## Stage 3: Speech chat, audio-streaming LLM, and chat documents

**Status:** Complete

**Goal:** Preserve one snapshot across direct-adapter and compatibility-shim dispatches for the assigned audio and document surfaces.

**Files:**
- Modify: `tldw_Server_API/app/core/Streaming/speech_chat_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat_documents.py`
- Modify: `tldw_Server_API/app/core/Chat/document_generator.py`
- Test: `tldw_Server_API/tests/Audio/test_speech_chat_service.py`
- Test: `tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_document_generator.py`
- Test: `tldw_Server_API/tests/Chat/integration/test_document_generation_endpoints.py`

**Interfaces:**
- Direct adapter request dictionaries and chat compatibility calls receive `credentials_resolved=True`.
- `DocumentGeneratorService.generate_document` and `_call_llm` gain optional `credentials_resolved: bool = False`; the endpoint supplies `True` with its captured snapshot.

- [x] Add parameterized A→B/absent→B speech-chat tests for direct adapter and legacy chat-call paths.
- [x] Add equivalent WebSocket audio tests for adapter, no-adapter shim, and `NotImplementedError` shim paths.
- [x] Add chat-document endpoint-to-service and service-to-chat-boundary rotation tests.
- [x] Run these nodes and verify RED on late key/config recovery or missing authoritative markers.
- [x] Replace string fallbacks, stop downstream key recovery from a captured authoritative snapshot, and add the marker at every existing dispatch branch.
- [x] Re-run the focused audio/document tests GREEN (15 passed).

## Stage 4: Rolling-summarize chunking and high-risk verification

**Status:** In Progress

**Goal:** Ensure template JSON, ordinary JSON, and file-upload rolling summarization all carry the resolved snapshot into `analyze`, then validate the complete delegated slice.

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/chunking.py`
- Modify: `tldw_Server_API/app/core/Chunking/strategies/rolling_summarize.py`
- Test: `tldw_Server_API/tests/Chunking/test_chunking_endpoint.py`

**Interfaces:**
- `_resolve_chunking_byok` accepts a structured fallback resolver instead of a bare key.
- Every rolling-summarize `llm_config` contains `app_config=byok_resolution.app_config` and `credentials_resolved=True`.

- [ ] Add A→B/absent→B assertions for template JSON, ordinary JSON, and file-upload configurations at the `Chunker`/`improved_chunking_process` analyzer boundary.
- [ ] Run the chunking nodes and verify RED because current `llm_config` drops the snapshot and marker.
- [ ] Use `resolve_static_server_fallback` for the chosen provider and add the captured config/marker to all three existing `llm_config` dictionaries.
- [ ] Run focused suites for all touched subsystems, `compileall`, `git diff --check`, and Bandit over the touched production files.
- [ ] Review the final diff for accidental scope overlap, secret-bearing logs/reprs, positional-call compatibility, and background object lifetime; record exact test counts and residual compatibility risks in the handoff.

## Stage 5: Provider policy runtime and lifecycle enforcement

**Status:** In Progress

**Goal:** Route every owned LLM credential lookup through one execution-scoped `ProviderCredentialRuntime`, enforce the eventual model when known, and close/mark runtime ownership at synchronous, streaming, WebSocket, executor, and background boundaries.

**Files:** Existing Stage 1–4 production and regression files only.

- [ ] Add RED assertions that known models reach `ProviderCredentialRuntime.resolve(..., model=...)`.
- [ ] Add lifecycle assertions for `mark_used()` after successful dispatch and `close()` on success/failure/cancellation.
- [ ] Keep Prompt Studio runtime ownership alive through deferred background execution.
- [ ] Verify whether Unified Evaluation selects an eventual model; use `model=None` and document the limitation if it does not.
- [ ] Re-run the complete owned high-risk suite, compile checks, Bandit, and diff review.
