# Auto Chunk Boundary Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicitly opt-in LLM boundary assistant for Auto Chunking that refines deterministic plans and falls back deterministically on every unavailable/error path.

**Architecture:** Keep the current sync resolver as the deterministic/manual compatibility layer. Add a small assistant module and an async resolver that calls the assistant only for Auto requests with `auto_chunking_use_llm=true`. Wire async ingestion paths to the async resolver without changing frontend contracts.

**Tech Stack:** Python, FastAPI async request handlers, existing `perform_chat_api_call_async`, existing LLM provider registry/config helpers, pytest.

---

### Task 1: Boundary Assistant Contract

**Files:**
- Create: `tldw_Server_API/app/core/Chunking/auto_boundary_assistant.py`
- Test: `tldw_Server_API/tests/Chunking/test_auto_boundary_assistant.py`

- [ ] **Step 1: Write failing tests for interface/result and validation**

Tests should assert that `AutoChunkBoundaryAssistant` can be implemented by a fake async assistant, `AutoChunkBoundaryAssistantResult` can represent success and fallback, valid JSON suggestions refine only allowed fields, and invalid method/size/overlap/view values return fallback metadata.

- [ ] **Step 2: Run test to verify it fails**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_auto_boundary_assistant.py -v`
Expected: FAIL because the module and types do not exist.

- [ ] **Step 3: Implement the minimal contract and validators**

Create dataclasses/protocol for request/result/availability. Add bounded helpers for excerpt creation, fallback reason appending, response text extraction, JSON parsing, and suggestion validation.

- [ ] **Step 4: Run test to verify it passes**

Run the same focused test file and confirm PASS.

### Task 2: LLM Adapter

**Files:**
- Modify: `tldw_Server_API/app/core/Chunking/auto_boundary_assistant.py`
- Test: `tldw_Server_API/tests/Chunking/test_auto_boundary_assistant.py`

- [ ] **Step 1: Write failing tests for availability and provider call behavior**

Tests should mock provider/model/config resolution and chat calls. Cover missing opt-in no-call, missing provider/model/adapter/key fallback, explicit success, timeout, provider exception, empty response, and invalid JSON.

- [ ] **Step 2: Run tests to verify failure**

Run the focused assistant test file. Expected: FAIL because concrete adapter behavior is absent.

- [ ] **Step 3: Implement concrete assistant**

Resolve provider from request fields or defaults, resolve model from request fields or config, verify adapter registration and required key availability, call `perform_chat_api_call_async` through `asyncio.wait_for`, and parse strict JSON. Do not send full source text; cap excerpts.

- [ ] **Step 4: Run tests to verify pass**

Run the focused assistant test file and confirm PASS.

### Task 3: Async Resolver

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/chunking_options.py`
- Modify: `tldw_Server_API/app/core/Chunking/auto_planner.py` if plan metadata helper support is needed
- Test: `tldw_Server_API/tests/Chunking/test_auto_chunking_resolver.py`

- [ ] **Step 1: Write failing resolver tests**

Tests should cover default no-call behavior, explicit opt-in success updating both `chunk_options` and `chunking_plan`, fallback preserving deterministic options on timeout/error/invalid response, and `used_llm` semantics.

- [ ] **Step 2: Run resolver tests to verify failure**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_auto_chunking_resolver.py -v`
Expected: FAIL because no async resolver exists.

- [ ] **Step 3: Implement async resolver**

Add `async_resolve_chunking_options_and_plan()` and `async_resolve_chunking_for_result()` wrappers. They call the existing deterministic resolver first, then invoke the assistant only when Auto plan exists and `auto_chunking_use_llm=true`. Preserve sync resolver behavior.

- [ ] **Step 4: Run resolver tests to verify pass**

Run the resolver test file and confirm PASS.

### Task 4: Async Ingestion Wiring

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
- Modify: `tldw_Server_API/app/services/media_ingest_jobs_worker.py`
- Modify: direct process endpoints under `tldw_Server_API/app/api/v1/endpoints/media/process_*.py`
- Modify: `tldw_Server_API/app/services/web_scraping_service.py`
- Modify: `tldw_Server_API/app/services/enhanced_web_scraping_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/ingest_web_content.py`
- Test: existing Auto Chunking media, worker, and web ingestion tests

- [ ] **Step 1: Write or extend failing wiring tests**

At minimum, prove one media-add/persistence path and the jobs worker call the async resolver and pass refined `chunk_options` downstream while returned/stored metadata contains the same plan.

- [ ] **Step 2: Run focused wiring tests to verify failure**

Run the targeted media add, jobs worker, and web Auto Chunking tests. Expected: FAIL while call sites still use the sync resolver.

- [ ] **Step 3: Wire async resolver**

Replace async call sites with awaits to the async resolver. Keep `apply_chunking_template_if_any()` gated by `chunking_plan is None`.

- [ ] **Step 4: Run focused wiring tests to verify pass**

Run the same focused tests and confirm PASS.

### Task 5: Verification and Closeout

**Files:**
- Modify: `backlog/tasks/task-96.8 - Implement-real-Auto-Chunking-boundary-assistant-adapter.md`

- [ ] **Step 1: Run focused backend test suite**

Run the new assistant/resolver tests plus existing Auto Chunking planner, process endpoint, jobs worker, persistence metadata, and web ingest tests.

- [ ] **Step 2: Run Bandit on touched production files**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Chunking/auto_boundary_assistant.py tldw_Server_API/app/core/Ingestion_Media_Processing/chunking_options.py tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py tldw_Server_API/app/services/media_ingest_jobs_worker.py tldw_Server_API/app/services/web_scraping_service.py tldw_Server_API/app/services/enhanced_web_scraping_service.py tldw_Server_API/app/api/v1/endpoints/media -f json -o /tmp/bandit_auto_chunk_boundary_assistant.json`

- [ ] **Step 3: Run whitespace diff check**

Run: `git diff --check`

- [ ] **Step 4: Update Backlog**

Check acceptance criteria and Definition of Done items with verification evidence and add final summary.
