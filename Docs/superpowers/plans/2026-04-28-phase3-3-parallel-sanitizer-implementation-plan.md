# Phase 3.3 Parallel Sanitizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the remaining Phase 3.3 conservative-plus sanitizer work by dispatching independent, covered raw-error cleanup shards in parallel, then integrating only red/green verified patches.

**Architecture:** The parent coordinator owns the shared checkout, candidate matrix, plan updates, commits, and push decisions. Workers operate on disjoint source/test slices in isolated shard worktrees or return patch bundles; they add tests first, patch only the covered fallback branch, and report verification evidence. The parent applies worker output serially and commits small batches after combined verification.

**Tech Stack:** Python, FastAPI, pytest, Loguru, Bandit, git worktrees.

---

## File Structure

**Read-only inputs**
- `Docs/superpowers/specs/2026-04-28-phase3-3-parallel-sanitizer-design.md`
- `Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md`

**Coordinator-owned files**
- Create: `Docs/superpowers/reviews/2026-04-28-phase3-3-remaining-candidate-matrix.md`
- Modify: `Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md`

**Shard source pools**
- RAG tail: `tldw_Server_API/app/core/RAG/rag_service/research_agent.py`, `tldw_Server_API/app/core/RAG/rag_service/document_grader.py`, `tldw_Server_API/app/core/RAG/rag_service/guardrails.py`, `tldw_Server_API/app/core/RAG/rag_service/table_serialization.py`
- API deps and small endpoints: `tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py`, `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py`, `tldw_Server_API/app/api/v1/API_Deps/kanban_deps.py`, `tldw_Server_API/app/api/v1/endpoints/skills.py`, `tldw_Server_API/app/api/v1/endpoints/llamacpp.py`, `tldw_Server_API/app/api/v1/endpoints/chunking.py`, `tldw_Server_API/app/api/v1/endpoints/vector_stores_openai.py`
- Web/search/ingestion helpers: `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py`, `tldw_Server_API/app/core/WebSearch/Web_Search.py`, `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`, `tldw_Server_API/app/core/Ingestion_Media_Processing/PDF/PDF_Processing_Lib.py`, `tldw_Server_API/app/core/Ingestion_Media_Processing/Books/Book_Processing_Lib.py`, `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Lib.py`, `tldw_Server_API/app/core/Ingestion_Media_Processing/Plaintext/Plaintext_Files.py`
- Core services/jobs: `tldw_Server_API/app/core/Chat/chat_orchestrator.py`, `tldw_Server_API/app/core/TTS/tts_service_v2.py`, `tldw_Server_API/app/core/Audio/tts_service.py`, `tldw_Server_API/app/services/document_processing_service.py`, `tldw_Server_API/app/services/ingestion_sources_worker.py`
- MCP small-core: `tldw_Server_API/app/core/MCP_unified/protocol.py`, `tldw_Server_API/app/core/MCP_unified/server.py`, `tldw_Server_API/app/core/MCP_unified/external_servers/manager.py`, `tldw_Server_API/app/core/MCP_unified/modules/implementations/slides_module.py`
- Inventory-only giants: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`, `tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py`, `tldw_Server_API/app/api/v1/endpoints/paper_search.py`, `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py`, `tldw_Server_API/app/core/DB_Management/Prompts_DB.py`, `tldw_Server_API/app/core/DB_Management/PromptStudioDatabase.py`, `tldw_Server_API/app/api/v1/endpoints/sync.py`, `tldw_Server_API/app/api/v1/endpoints/chat.py`, `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`

**Likely test pools**
- `tldw_Server_API/tests/RAG/test_*sanitiz*.py`
- `tldw_Server_API/tests/Chat/test_chacha_db_deps_error_mapping.py`
- `tldw_Server_API/tests/Chat/unit/test_chat_orchestrator_contract.py`
- `tldw_Server_API/tests/Skills/integration/test_skills_api.py`
- `tldw_Server_API/tests/LLM_Local/test_llamacpp_management_api.py`
- `tldw_Server_API/tests/LLamaCpp/test_llamacpp_reranking_endpoints.py`
- `tldw_Server_API/tests/Web_Scraping/test_*.py`
- `tldw_Server_API/tests/WebScraping/test_*.py`
- `tldw_Server_API/tests/Media_Ingestion_Modification/test_*.py`
- `tldw_Server_API/tests/MCP_unified/test_*sanitization.py`

## Global Rules

- All coordinator commands must run from `/Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption`.
- Before any coordinator or worker source edit, verify the checkout with:

```bash
pwd
git branch --show-current
git status --short --branch
```

Expected: `pwd` is `/Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption` for parent work, or an explicitly assigned shard worktree for worker work; branch is `worktree-phase3.3-error-handler-adoption` or the worker's shard branch created from it.
- Activate the project venv before Python commands: `source .venv/bin/activate`.
- Do not push unless the user explicitly asks.
- Do not edit validation-facing `400/422`, not-found `404`, conflict `409`, or public raw-diagnostic contracts already asserted by tests.
- Do not patch giant files unless the candidate has a direct focused red/green test and parent approval.
- Do not add generic sanitizer frameworks. Prefer local fixed labels or existing adjacent helper patterns.
- Do not add `exc_info=True`, `logger.exception(...)`, `logger.opt(exception=True)`, `traceback.format_exc()`, or traceback-bearing metadata.
- Every source edit needs red/green proof on an unpatched baseline for the same branch.
- Every worker report must include skipped candidates and reasons.
- Commands below use concrete default file lists for buildability. If a worker creates a shard-specific sanitizer file named in this plan, include it in the command. If a shard makes no source edits and therefore creates no new test file, omit only that non-existent file and record the omission in the worker report.

Use the full scan regex for every shard refresh unless the task explicitly says otherwise:

```bash
exc_info=|str\(e\)|str\(exc\)|str\(error\)|error=str|detail=str|detail=f\".*\{e\}|detail=f\".*\{exc\}\"
```

Every implementation worker must return this exact evidence template:

```markdown
Status: DONE | DONE_WITH_CONCERNS | BLOCKED
Shard:
Worktree and branch verified:
Files changed:
Candidates patched:
Candidates skipped and reasons:
Red test command and failure summary:
Green focused test command and result:
Full touched test-file command and result:
Bandit command and output path:
Bandit results/errors/skipped reviewed:
New Bandit findings introduced:
Pre-existing Bandit findings retained:
git diff --check result:
Notes for parent integration:
```

## Sanitizer Test Templates

Use these patterns as starting points; adapt to existing fixtures and module style.

```python
def assert_safe_text(value: object, leaked: str = "backend exploded") -> None:
    text = str(value)
    assert leaked not in text
    assert "/tmp/" not in text
    assert "secret" not in text.lower()
```

```python
async def test_endpoint_backend_failure_is_sanitized(monkeypatch):
    async def fail(*args, **kwargs):
        raise RuntimeError("backend exploded /tmp/secret-token")

    monkeypatch.setattr(target_module, "target_dependency", fail)

    with pytest.raises(HTTPException) as exc_info:
        await target_module.endpoint_under_test(...)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to perform operation"
    assert_safe_text(exc_info.value.detail)
```

```python
def test_fail_open_log_is_sanitized(monkeypatch):
    records = []
    sink_id = logger.add(lambda message: records.append(message), format="{message} {extra}")
    try:
        monkeypatch.setattr(target, "dependency", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("backend exploded /tmp/secret-token")))
        assert target.function_under_test(...) == expected_fallback
    finally:
        logger.remove(sink_id)

    rendered = "\n".join(str(record) for record in records)
    assert "backend exploded" not in rendered
    assert "/tmp/secret-token" not in rendered
```

## Task 1: Build Remaining Candidate Matrix

**Files:**
- Create: `Docs/superpowers/reviews/2026-04-28-phase3-3-remaining-candidate-matrix.md`
- Read: `Docs/superpowers/specs/2026-04-28-phase3-3-parallel-sanitizer-design.md`
- Read: `Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md`

- [ ] **Step 1: Record the coordinator baseline**

Run:

```bash
git rev-parse HEAD
git status --short --branch
```

Expected: record the current commit and confirm no source/test edits are present before workers start.

- [ ] **Step 2: Run the raw-error scan**

Run:

```bash
rg -n "exc_info=|str\(e\)|str\(exc\)|str\(error\)|error=str|detail=str|detail=f\".*\{e\}|detail=f\".*\{exc\}\"" tldw_Server_API/app -g '*.py'
```

Expected: output includes both patch candidates and known skip-default files.

- [ ] **Step 3: Create the matrix document**

Write this table header to `Docs/superpowers/reviews/2026-04-28-phase3-3-remaining-candidate-matrix.md`:

```markdown
# Phase 3.3 Remaining Candidate Matrix

Baseline: `<git rev-parse HEAD>`

| Source file | Line | Function/branch | Pattern | Surface | Existing tests reviewed | Proposed safe label/helper | Owned test file | Red-test strategy | Shard | Decision |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
```

- [ ] **Step 4: Fill initial patch/skip decisions**

For each shard source pool, add at least one row per raw-error hit that looks patchable and one summary row for each skip-default giant. Use `Decision` values `patch`, `skip-public-contract`, `skip-validation`, `skip-giant`, `skip-no-focused-test`, or `defer`.

- [ ] **Step 5: Search tests before marking public payload changes patchable**

Run targeted searches like:

```bash
rg -n "last_error|detail|backend exploded|exc_info|sanitiz|HTTPException|500" tldw_Server_API/tests/RAG tldw_Server_API/tests/Web_Scraping tldw_Server_API/tests/WebScraping tldw_Server_API/tests/MCP_unified tldw_Server_API/tests/Media_Ingestion_Modification
```

Expected: matrix rows for public response fields cite exact test files or explicitly state no existing public-contract assertion was found.

- [ ] **Step 6: Commit only if the matrix is stable**

Run:

```bash
git diff --check
git status --short --branch
```

Expected: either leave the matrix unstaged for the first implementation commit, or commit it alone if the parent wants a review checkpoint.

## Task 2A: Parallel Worker - RAG Tail

**Files:**
- Modify only if matrix says `patch`: `tldw_Server_API/app/core/RAG/rag_service/research_agent.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/core/RAG/rag_service/document_grader.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/core/RAG/rag_service/guardrails.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/core/RAG/rag_service/table_serialization.py`
- Test: `tldw_Server_API/tests/RAG/test_document_grader.py`
- Test: `tldw_Server_API/tests/RAG/test_table_serialization_sanitizers.py`
- Test: create focused sanitizer file only if needed, `tldw_Server_API/tests/RAG/test_phase3_3_remaining_sanitizers.py`

- [ ] **Step 1: Refresh shard scan**

Run:

```bash
rg -n "exc_info=|str\(e\)|str\(exc\)|str\(error\)|error=str|detail=str|detail=f\".*\{e\}|detail=f\".*\{exc\}\"" tldw_Server_API/app/core/RAG/rag_service/research_agent.py tldw_Server_API/app/core/RAG/rag_service/document_grader.py tldw_Server_API/app/core/RAG/rag_service/guardrails.py tldw_Server_API/app/core/RAG/rag_service/table_serialization.py
```

Expected: identify only remaining raw-error sites not already intentionally preserved by earlier Phase 3.3 commits.

- [ ] **Step 2: Write failing tests for one to three isolated candidates**

Prefer log-only or metadata-return branches that can be triggered by monkeypatching a dependency. Do not modify `unified_pipeline.py`.

- [ ] **Step 3: Run red tests**

Run the exact focused test selections, for example:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/RAG/test_phase3_3_remaining_sanitizers.py -q
```

Expected: fail because raw exception text, raw path, or traceback metadata is still exposed.

- [ ] **Step 4: Patch minimal sanitizer behavior**

Replace raw exception detail with a fixed label such as `research_agent_error`, `grading_error`, or `table_processing_error`, or with `error_type=type(exc).__name__` in logs. Preserve return shape and status semantics.

- [ ] **Step 5: Run green focused and full touched tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/RAG/test_phase3_3_remaining_sanitizers.py -q
python -m pytest tldw_Server_API/tests/RAG/test_phase3_3_remaining_sanitizers.py tldw_Server_API/tests/RAG/test_document_grader.py tldw_Server_API/tests/RAG/test_table_serialization_sanitizers.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Run Bandit and diff checks**

Run:

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/core/RAG/rag_service/research_agent.py tldw_Server_API/app/core/RAG/rag_service/document_grader.py tldw_Server_API/app/core/RAG/rag_service/guardrails.py tldw_Server_API/app/core/RAG/rag_service/table_serialization.py -f json -o /tmp/bandit_phase3_3_rag_tail.json
git diff --check
git status --short --branch
```

Expected: no new Bandit findings; report any pre-existing findings by file and test impact.

## Task 2B: Parallel Worker - API Deps and Small Endpoints

**Files:**
- Modify only if matrix says `patch`: `tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/api/v1/API_Deps/kanban_deps.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/api/v1/endpoints/skills.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/api/v1/endpoints/llamacpp.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/api/v1/endpoints/chunking.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/api/v1/endpoints/vector_stores_openai.py`
- Test: `tldw_Server_API/tests/Chat/test_chacha_db_deps_error_mapping.py`
- Test: create focused deps sanitizer file only if needed, `tldw_Server_API/tests/API_Deps/test_phase3_3_sanitizers.py`
- Test: `tldw_Server_API/tests/Skills/integration/test_skills_api.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_management_api.py`
- Test: `tldw_Server_API/tests/LLamaCpp/test_llamacpp_reranking_endpoints.py`

- [ ] **Step 1: Refresh shard scan**

Run:

```bash
rg -n "exc_info=|str\(e\)|str\(exc\)|str\(error\)|error=str|detail=str|detail=f\".*\{e\}|detail=f\".*\{exc\}\"" tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py tldw_Server_API/app/api/v1/API_Deps/kanban_deps.py tldw_Server_API/app/api/v1/endpoints/skills.py tldw_Server_API/app/api/v1/endpoints/llamacpp.py tldw_Server_API/app/api/v1/endpoints/chunking.py tldw_Server_API/app/api/v1/endpoints/vector_stores_openai.py
```

Expected: identify candidate `500` fallbacks or fail-open logs; classify `400/404/409/422` as skip.

- [ ] **Step 2: Write red tests only for generic fallback branches**

Use direct async endpoint calls where possible. Preserve any `UserNotFoundError`, `SkillNotFoundError`, `SkillValidationError`, conflict, and validation behavior.

- [ ] **Step 3: Run red focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/API_Deps/test_phase3_3_sanitizers.py tldw_Server_API/tests/Skills/integration/test_skills_api.py tldw_Server_API/tests/LLM_Local/test_llamacpp_management_api.py tldw_Server_API/tests/LLamaCpp/test_llamacpp_reranking_endpoints.py -k "sanitiz or backend or fallback or error" -q
```

Expected: fail on raw detail or raw log content.

- [ ] **Step 4: Patch minimal sanitizer behavior**

For HTTP `500` fallbacks, use existing route-specific strings such as `Failed to list skills`, `Failed to initialize server`, or the established string already used by adjacent tests. For logs, use safe exception-type labels.

- [ ] **Step 5: Run green tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/API_Deps/test_phase3_3_sanitizers.py tldw_Server_API/tests/Skills/integration/test_skills_api.py tldw_Server_API/tests/LLM_Local/test_llamacpp_management_api.py tldw_Server_API/tests/LLamaCpp/test_llamacpp_reranking_endpoints.py -k "sanitiz or backend or fallback or error" -q
python -m pytest tldw_Server_API/tests/API_Deps/test_phase3_3_sanitizers.py tldw_Server_API/tests/Chat/test_chacha_db_deps_error_mapping.py tldw_Server_API/tests/Skills/integration/test_skills_api.py tldw_Server_API/tests/LLM_Local/test_llamacpp_management_api.py tldw_Server_API/tests/LLamaCpp/test_llamacpp_reranking_endpoints.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Run Bandit and diff checks**

Run:

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py tldw_Server_API/app/api/v1/API_Deps/kanban_deps.py tldw_Server_API/app/api/v1/endpoints/skills.py tldw_Server_API/app/api/v1/endpoints/llamacpp.py tldw_Server_API/app/api/v1/endpoints/chunking.py tldw_Server_API/app/api/v1/endpoints/vector_stores_openai.py -f json -o /tmp/bandit_phase3_3_api_deps_small_endpoints.json
git diff --check
git status --short --branch
```

Expected: no new Bandit findings.

## Task 2C: Parallel Worker - Web/Search/Ingestion Helpers

**Files:**
- Modify only if matrix says `patch`: `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/core/WebSearch/Web_Search.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/core/Ingestion_Media_Processing/PDF/PDF_Processing_Lib.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/core/Ingestion_Media_Processing/Books/Book_Processing_Lib.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Lib.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/core/Ingestion_Media_Processing/Plaintext/Plaintext_Files.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_websearch_searx_json.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py`
- Test: `tldw_Server_API/tests/WebScraping/test_scraper_analyzer_sanitizers.py`
- Test: create focused web sanitizer file only if needed, `tldw_Server_API/tests/Web_Scraping/test_phase3_3_sanitizers.py`
- Test: `tldw_Server_API/tests/Media_Ingestion_Modification/test_media_processing.py`
- Test: `tldw_Server_API/tests/Media_Ingestion_Modification/test_ingestion_helpers_stage3.py`
- Test: create focused ingestion sanitizer file only if needed, `tldw_Server_API/tests/Media_Ingestion_Modification/test_phase3_3_sanitizers.py`

- [ ] **Step 1: Refresh shard scan**

Run:

```bash
rg -n "exc_info=|str\(e\)|str\(exc\)|str\(error\)|error=str|detail=str|detail=f\".*\{e\}|detail=f\".*\{exc\}\"" tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py tldw_Server_API/app/core/WebSearch/Web_Search.py tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py tldw_Server_API/app/core/Ingestion_Media_Processing/PDF/PDF_Processing_Lib.py tldw_Server_API/app/core/Ingestion_Media_Processing/Books/Book_Processing_Lib.py tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Lib.py tldw_Server_API/app/core/Ingestion_Media_Processing/Plaintext/Plaintext_Files.py
```

Expected: classify network/backend fail-open logs separately from user-visible parsing errors.

- [ ] **Step 2: Write red tests for helper-level fail-open branches**

Prefer direct function tests with monkeypatched network/parser dependencies. Skip broad ingestion pipeline paths that require real files, external services, or end-to-end setup.

- [ ] **Step 3: Run red focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Web_Scraping/test_phase3_3_sanitizers.py tldw_Server_API/tests/Media_Ingestion_Modification/test_phase3_3_sanitizers.py -q
```

Expected: fail on raw backend exception text, path, URL credential, or traceback metadata.

- [ ] **Step 4: Patch minimal sanitizer behavior**

Keep existing fallback return values. Replace raw log/message fragments with fixed labels such as `web_search_error`, `article_extraction_error`, `pdf_processing_error`, `book_processing_error`, `audio_processing_error`, or safe `error_type`.

- [ ] **Step 5: Run green and Bandit**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Web_Scraping/test_phase3_3_sanitizers.py tldw_Server_API/tests/Media_Ingestion_Modification/test_phase3_3_sanitizers.py -q
python -m pytest tldw_Server_API/tests/Web_Scraping/test_phase3_3_sanitizers.py tldw_Server_API/tests/Web_Scraping/test_websearch_searx_json.py tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py tldw_Server_API/tests/WebScraping/test_scraper_analyzer_sanitizers.py tldw_Server_API/tests/Media_Ingestion_Modification/test_phase3_3_sanitizers.py tldw_Server_API/tests/Media_Ingestion_Modification/test_media_processing.py tldw_Server_API/tests/Media_Ingestion_Modification/test_ingestion_helpers_stage3.py -q
python -m bandit -r tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py tldw_Server_API/app/core/WebSearch/Web_Search.py tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py tldw_Server_API/app/core/Ingestion_Media_Processing/PDF/PDF_Processing_Lib.py tldw_Server_API/app/core/Ingestion_Media_Processing/Books/Book_Processing_Lib.py tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Lib.py tldw_Server_API/app/core/Ingestion_Media_Processing/Plaintext/Plaintext_Files.py -f json -o /tmp/bandit_phase3_3_web_ingestion.json
git diff --check
git status --short --branch
```

Expected: all selected tests pass and no new Bandit findings.

## Task 2D: Parallel Worker - Core Services and Jobs

**Files:**
- Modify only if matrix says `patch`: `tldw_Server_API/app/core/Chat/chat_orchestrator.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/core/TTS/tts_service_v2.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/core/Audio/tts_service.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/services/document_processing_service.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/services/ingestion_sources_worker.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_orchestrator_contract.py`
- Test: `tldw_Server_API/tests/Ingestion_Sources/test_ingestion_sources_worker.py`
- Test: create focused Chat sanitizer file only if needed, `tldw_Server_API/tests/Chat/unit/test_phase3_3_sanitizers.py`
- Test: create focused Audio/TTS sanitizer file only if needed, `tldw_Server_API/tests/Audio/test_phase3_3_tts_sanitizers.py`
- Test: create focused ingestion worker sanitizer file only if needed, `tldw_Server_API/tests/Ingestion_Sources/test_phase3_3_worker_sanitizers.py`

- [ ] **Step 1: Refresh shard scan**

Run:

```bash
rg -n "exc_info=|str\(e\)|str\(exc\)|str\(error\)|error=str|detail=str|detail=f\".*\{e\}|detail=f\".*\{exc\}\"" tldw_Server_API/app/core/Chat/chat_orchestrator.py tldw_Server_API/app/core/TTS/tts_service_v2.py tldw_Server_API/app/core/Audio/tts_service.py tldw_Server_API/app/services/document_processing_service.py tldw_Server_API/app/services/ingestion_sources_worker.py
```

Expected: identify fail-open logging or status/error fields; skip user-facing job error messages unless tests can prove a safe contract change.

- [ ] **Step 2: Write red tests for non-public fail-open logs first**

Prefer log sanitizer tests over public job status changes. If a job status payload currently stores raw user-visible errors and no existing test covers a safe replacement, mark `skip-public-contract`.

- [ ] **Step 3: Run red focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chat/unit/test_phase3_3_sanitizers.py tldw_Server_API/tests/Audio/test_phase3_3_tts_sanitizers.py tldw_Server_API/tests/Ingestion_Sources/test_phase3_3_worker_sanitizers.py -q
```

Expected: fail on raw exception/path/token exposure.

- [ ] **Step 4: Patch minimal sanitizer behavior**

Use fixed log messages or `error_type=type(exc).__name__`. Do not change retryability, job lifecycle state, progress callbacks, or raised exception classes.

- [ ] **Step 5: Verify**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chat/unit/test_phase3_3_sanitizers.py tldw_Server_API/tests/Audio/test_phase3_3_tts_sanitizers.py tldw_Server_API/tests/Ingestion_Sources/test_phase3_3_worker_sanitizers.py -q
python -m pytest tldw_Server_API/tests/Chat/unit/test_phase3_3_sanitizers.py tldw_Server_API/tests/Chat/unit/test_chat_orchestrator_contract.py tldw_Server_API/tests/Audio/test_phase3_3_tts_sanitizers.py tldw_Server_API/tests/Ingestion_Sources/test_phase3_3_worker_sanitizers.py tldw_Server_API/tests/Ingestion_Sources/test_ingestion_sources_worker.py -q
python -m bandit -r tldw_Server_API/app/core/Chat/chat_orchestrator.py tldw_Server_API/app/core/TTS/tts_service_v2.py tldw_Server_API/app/core/Audio/tts_service.py tldw_Server_API/app/services/document_processing_service.py tldw_Server_API/app/services/ingestion_sources_worker.py -f json -o /tmp/bandit_phase3_3_core_services.json
git diff --check
git status --short --branch
```

Expected: all selected tests pass and no new Bandit findings.

## Task 2E: Parallel Worker - MCP Small-Core

**Files:**
- Modify only if matrix says `patch`: `tldw_Server_API/app/core/MCP_unified/protocol.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/core/MCP_unified/server.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/core/MCP_unified/external_servers/manager.py`
- Modify only if matrix says `patch`: `tldw_Server_API/app/core/MCP_unified/modules/implementations/slides_module.py`
- Test: `tldw_Server_API/tests/unit/test_mcp_unified_error_mapping.py`
- Test: `tldw_Server_API/tests/MCP_unified/test_mcp_config_sanitization.py`
- Test: `tldw_Server_API/tests/MCP_unified/test_external_server_manager_sanitization.py`
- Test: `tldw_Server_API/tests/MCP_unified/test_slides_module_exports.py`
- Test: create focused MCP sanitizer file only if needed, `tldw_Server_API/tests/MCP_unified/test_phase3_3_small_core_sanitizers.py`

- [ ] **Step 1: Refresh shard scan**

Run:

```bash
rg -n "exc_info=|str\(e\)|str\(exc\)|str\(error\)|error=str|detail=str|detail=f\".*\{e\}|detail=f\".*\{exc\}\"" tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/server.py tldw_Server_API/app/core/MCP_unified/external_servers/manager.py tldw_Server_API/app/core/MCP_unified/modules/implementations/slides_module.py
```

Expected: classify only direct unit-testable raw-error returns/logs as patchable.

- [ ] **Step 2: Write red tests**

Use existing MCP sanitizer tests where possible. Skip `mcp_hub_management.py` and broad server lifecycle paths unless a direct unit harness already exists.

- [ ] **Step 3: Run red focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/test_phase3_3_small_core_sanitizers.py -q
```

Expected: fail on raw exception/path/token exposure or traceback-bearing metadata.

- [ ] **Step 4: Patch minimal sanitizer behavior**

Use fixed labels or safe exception-type labels. Preserve JSON-RPC error codes, MCP response shape, server lifecycle behavior, and module export semantics.

- [ ] **Step 5: Verify green focused, full touched, Bandit, and diff checks**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/test_phase3_3_small_core_sanitizers.py -q
python -m pytest tldw_Server_API/tests/MCP_unified/test_phase3_3_small_core_sanitizers.py tldw_Server_API/tests/unit/test_mcp_unified_error_mapping.py tldw_Server_API/tests/MCP_unified/test_mcp_config_sanitization.py tldw_Server_API/tests/MCP_unified/test_external_server_manager_sanitization.py tldw_Server_API/tests/MCP_unified/test_slides_module_exports.py -q
python -m bandit -r tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/server.py tldw_Server_API/app/core/MCP_unified/external_servers/manager.py tldw_Server_API/app/core/MCP_unified/modules/implementations/slides_module.py -f json -o /tmp/bandit_phase3_3_mcp_small_core.json
git diff --check
git status --short --branch
```

Expected: red/green evidence exists for every source edit and no new Bandit findings exist.

## Task 2F: Parallel Worker - Skip Inventory for Giants

**Files:**
- Read only: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Read only: `tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py`
- Read only: `tldw_Server_API/app/api/v1/endpoints/paper_search.py`
- Read only: `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py`
- Read only: `tldw_Server_API/app/core/DB_Management/Prompts_DB.py`
- Read only: `tldw_Server_API/app/core/DB_Management/PromptStudioDatabase.py`
- Read only: `tldw_Server_API/app/api/v1/endpoints/sync.py`
- Read only: `tldw_Server_API/app/api/v1/endpoints/chat.py`
- Read only: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- Do not modify source, tests, plan files, or `Docs/superpowers/reviews/2026-04-28-phase3-3-remaining-candidate-matrix.md`; return proposed matrix rows in the worker report for the parent to apply.

- [ ] **Step 1: Count remaining raw-error hits**

Run:

```bash
rg -n "exc_info=|str\(e\)|str\(exc\)|str\(error\)|error=str|detail=str|detail=f\".*\{e\}|detail=f\".*\{exc\}\"" tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py tldw_Server_API/app/api/v1/endpoints/paper_search.py tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py tldw_Server_API/app/core/DB_Management/Prompts_DB.py tldw_Server_API/app/core/DB_Management/PromptStudioDatabase.py tldw_Server_API/app/api/v1/endpoints/sync.py tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py
```

Expected: each giant source file has a count and representative examples.

- [ ] **Step 2: Search for direct coverage**

Run targeted searches for each giant, for example:

```bash
rg -n "ChaChaNotes|mcp_hub_management|paper_search|characters_endpoint|PromptStudioDatabase|sync|chat" tldw_Server_API/tests
```

Expected: record whether a direct focused sanitizer test exists.

- [ ] **Step 3: Prepare matrix rows with skip reasons**

For each giant candidate, prepare rows or grouped summary rows with `skip-giant`, `skip-public-contract`, or `defer`. Do not patch source and do not edit the parent-owned matrix file.

- [ ] **Step 4: Return inventory summary**

Report which giant file would be the best next separate tranche and why. Include exact candidate functions if they appear direct-testable.

## Task 3: Parent Integration Batch 1

**Files:**
- Modify: `Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md`
- Modify: source/test files returned by the first completed safe shard batch

- [ ] **Step 1: Review worker diffs manually**

Check every changed source branch against the sanitizer contract. Reject any patch that changes `400/404/409/422`, broad control flow, retry semantics, or public raw diagnostics without explicit matrix justification.

- [ ] **Step 2: Apply one logical batch**

Apply only compatible shard outputs. Keep conflicting or ambiguous shard output aside for later.

- [ ] **Step 3: Update the Phase 3.3 plan**

Append `**Recent Update**` lines to `Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md` describing what changed, why it is covered, and which verification commands passed.

- [ ] **Step 4: Run combined verification**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/RAG/test_phase3_3_remaining_sanitizers.py tldw_Server_API/tests/API_Deps/test_phase3_3_sanitizers.py tldw_Server_API/tests/Chat/test_chacha_db_deps_error_mapping.py tldw_Server_API/tests/Skills/integration/test_skills_api.py tldw_Server_API/tests/LLM_Local/test_llamacpp_management_api.py tldw_Server_API/tests/LLamaCpp/test_llamacpp_reranking_endpoints.py -q
python -m bandit -r tldw_Server_API/app/core/RAG/rag_service/research_agent.py tldw_Server_API/app/core/RAG/rag_service/document_grader.py tldw_Server_API/app/core/RAG/rag_service/guardrails.py tldw_Server_API/app/core/RAG/rag_service/table_serialization.py tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py tldw_Server_API/app/api/v1/API_Deps/kanban_deps.py tldw_Server_API/app/api/v1/endpoints/skills.py tldw_Server_API/app/api/v1/endpoints/llamacpp.py tldw_Server_API/app/api/v1/endpoints/chunking.py tldw_Server_API/app/api/v1/endpoints/vector_stores_openai.py -f json -o /tmp/bandit_phase3_3_batch1.json
git diff --check
git status --short --branch
```

Expected: tests pass, Bandit introduces no new findings, diff check passes.

- [ ] **Step 5: Commit batch 1**

Run:

```bash
git add Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md Docs/superpowers/reviews/2026-04-28-phase3-3-remaining-candidate-matrix.md tldw_Server_API/app/core/RAG/rag_service/research_agent.py tldw_Server_API/app/core/RAG/rag_service/document_grader.py tldw_Server_API/app/core/RAG/rag_service/guardrails.py tldw_Server_API/app/core/RAG/rag_service/table_serialization.py tldw_Server_API/tests/RAG/test_phase3_3_remaining_sanitizers.py tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py tldw_Server_API/app/api/v1/API_Deps/kanban_deps.py tldw_Server_API/app/api/v1/endpoints/skills.py tldw_Server_API/app/api/v1/endpoints/llamacpp.py tldw_Server_API/app/api/v1/endpoints/chunking.py tldw_Server_API/app/api/v1/endpoints/vector_stores_openai.py tldw_Server_API/tests/API_Deps/test_phase3_3_sanitizers.py tldw_Server_API/tests/Skills/integration/test_skills_api.py tldw_Server_API/tests/LLM_Local/test_llamacpp_management_api.py tldw_Server_API/tests/LLamaCpp/test_llamacpp_reranking_endpoints.py
git diff --cached --check
git commit -m "Phase 3.3: sanitize remaining covered fallbacks batch 1"
```

Expected: branch is clean after commit. Do not push.

## Task 4: Parent Integration Batch 2

**Files:**
- Modify: `Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md`
- Modify: remaining source/test files returned by safe shard workers

- [ ] **Step 1: Review remaining worker outputs**

Reject any shard that lacks red proof, green proof, Bandit review, or clear skip accounting.

- [ ] **Step 2: Apply second logical batch**

Group by compatible source/test ownership. Do not mix unrelated broad files if review becomes hard.

- [ ] **Step 3: Update the Phase 3.3 plan**

Append `**Recent Update**` lines to `Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md` describing the second batch, coverage, and verification evidence before staging.

- [ ] **Step 4: Run combined verification**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Web_Scraping/test_phase3_3_sanitizers.py tldw_Server_API/tests/Media_Ingestion_Modification/test_phase3_3_sanitizers.py tldw_Server_API/tests/Chat/unit/test_phase3_3_sanitizers.py tldw_Server_API/tests/Audio/test_phase3_3_tts_sanitizers.py tldw_Server_API/tests/Ingestion_Sources/test_phase3_3_worker_sanitizers.py tldw_Server_API/tests/MCP_unified/test_phase3_3_small_core_sanitizers.py -q
python -m bandit -r tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py tldw_Server_API/app/core/WebSearch/Web_Search.py tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py tldw_Server_API/app/core/Ingestion_Media_Processing/PDF/PDF_Processing_Lib.py tldw_Server_API/app/core/Ingestion_Media_Processing/Books/Book_Processing_Lib.py tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Lib.py tldw_Server_API/app/core/Ingestion_Media_Processing/Plaintext/Plaintext_Files.py tldw_Server_API/app/core/Chat/chat_orchestrator.py tldw_Server_API/app/core/TTS/tts_service_v2.py tldw_Server_API/app/core/Audio/tts_service.py tldw_Server_API/app/services/document_processing_service.py tldw_Server_API/app/services/ingestion_sources_worker.py tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/server.py tldw_Server_API/app/core/MCP_unified/external_servers/manager.py tldw_Server_API/app/core/MCP_unified/modules/implementations/slides_module.py -f json -o /tmp/bandit_phase3_3_batch2.json
git diff --check
git status --short --branch
```

Expected: tests pass, Bandit introduces no new findings, diff check passes.

- [ ] **Step 5: Commit batch 2**

Run:

```bash
git add Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md Docs/superpowers/reviews/2026-04-28-phase3-3-remaining-candidate-matrix.md tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py tldw_Server_API/app/core/WebSearch/Web_Search.py tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py tldw_Server_API/app/core/Ingestion_Media_Processing/PDF/PDF_Processing_Lib.py tldw_Server_API/app/core/Ingestion_Media_Processing/Books/Book_Processing_Lib.py tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Lib.py tldw_Server_API/app/core/Ingestion_Media_Processing/Plaintext/Plaintext_Files.py tldw_Server_API/tests/Web_Scraping/test_phase3_3_sanitizers.py tldw_Server_API/tests/Media_Ingestion_Modification/test_phase3_3_sanitizers.py tldw_Server_API/app/core/Chat/chat_orchestrator.py tldw_Server_API/app/core/TTS/tts_service_v2.py tldw_Server_API/app/core/Audio/tts_service.py tldw_Server_API/app/services/document_processing_service.py tldw_Server_API/app/services/ingestion_sources_worker.py tldw_Server_API/tests/Chat/unit/test_phase3_3_sanitizers.py tldw_Server_API/tests/Audio/test_phase3_3_tts_sanitizers.py tldw_Server_API/tests/Ingestion_Sources/test_phase3_3_worker_sanitizers.py tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/server.py tldw_Server_API/app/core/MCP_unified/external_servers/manager.py tldw_Server_API/app/core/MCP_unified/modules/implementations/slides_module.py tldw_Server_API/tests/MCP_unified/test_phase3_3_small_core_sanitizers.py
git diff --cached --check
git commit -m "Phase 3.3: sanitize remaining covered fallbacks batch 2"
```

Expected: branch is clean after commit. Do not push.

## Task 5: Final Sweep and Remaining-Items Report

**Files:**
- Modify: `Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md`
- Modify: `Docs/superpowers/reviews/2026-04-28-phase3-3-remaining-candidate-matrix.md`

- [ ] **Step 1: Run final raw-error scan**

Run:

```bash
rg -n "exc_info=|str\(e\)|str\(exc\)|str\(error\)|error=str|detail=str|detail=f\".*\{e\}|detail=f\".*\{exc\}\"" tldw_Server_API/app -g '*.py'
```

Expected: remaining hits are either intentionally preserved or recorded as skipped/deferred.

- [ ] **Step 2: Update matrix decisions**

Mark every reviewed candidate as `patched`, `skip-public-contract`, `skip-validation`, `skip-giant`, `skip-no-focused-test`, or `defer-next-phase`.

- [ ] **Step 3: Run final hygiene**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/RAG/test_phase3_3_remaining_sanitizers.py tldw_Server_API/tests/API_Deps/test_phase3_3_sanitizers.py tldw_Server_API/tests/Web_Scraping/test_phase3_3_sanitizers.py tldw_Server_API/tests/Media_Ingestion_Modification/test_phase3_3_sanitizers.py tldw_Server_API/tests/Chat/unit/test_phase3_3_sanitizers.py tldw_Server_API/tests/Audio/test_phase3_3_tts_sanitizers.py tldw_Server_API/tests/Ingestion_Sources/test_phase3_3_worker_sanitizers.py tldw_Server_API/tests/MCP_unified/test_phase3_3_small_core_sanitizers.py -q
python -m bandit -r tldw_Server_API/app/core/RAG/rag_service/research_agent.py tldw_Server_API/app/core/RAG/rag_service/document_grader.py tldw_Server_API/app/core/RAG/rag_service/guardrails.py tldw_Server_API/app/core/RAG/rag_service/table_serialization.py tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py tldw_Server_API/app/api/v1/API_Deps/kanban_deps.py tldw_Server_API/app/api/v1/endpoints/skills.py tldw_Server_API/app/api/v1/endpoints/llamacpp.py tldw_Server_API/app/api/v1/endpoints/chunking.py tldw_Server_API/app/api/v1/endpoints/vector_stores_openai.py tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py tldw_Server_API/app/core/WebSearch/Web_Search.py tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py tldw_Server_API/app/core/Ingestion_Media_Processing/PDF/PDF_Processing_Lib.py tldw_Server_API/app/core/Ingestion_Media_Processing/Books/Book_Processing_Lib.py tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Lib.py tldw_Server_API/app/core/Ingestion_Media_Processing/Plaintext/Plaintext_Files.py tldw_Server_API/app/core/Chat/chat_orchestrator.py tldw_Server_API/app/core/TTS/tts_service_v2.py tldw_Server_API/app/core/Audio/tts_service.py tldw_Server_API/app/services/document_processing_service.py tldw_Server_API/app/services/ingestion_sources_worker.py tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/server.py tldw_Server_API/app/core/MCP_unified/external_servers/manager.py tldw_Server_API/app/core/MCP_unified/modules/implementations/slides_module.py -f json -o /tmp/bandit_phase3_3_final_parallel_sanitizers.json
git diff --check
git status --short --branch
```

Expected: all touched tests pass, no new Bandit findings, branch status is understood.

- [ ] **Step 4: Commit final report if changed**

Run:

```bash
git add Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md Docs/superpowers/reviews/2026-04-28-phase3-3-remaining-candidate-matrix.md
git diff --cached --check
git commit -m "Phase 3.3: record remaining sanitizer inventory"
```

Expected: commit only if these docs changed after the implementation batches.

## Success Criteria

- Candidate matrix exists and records patch/skip decisions for reviewed remaining candidates.
- At least two independent worker shards can run in parallel without overlapping writes.
- Every source patch has a red/green test.
- Parent combined verification passes before every implementation commit.
- No new Bandit findings are introduced in touched source files.
- Giant or public-contract-sensitive files are skipped with explicit reasons rather than opportunistically edited.
- No push occurs unless the user explicitly requests it.
