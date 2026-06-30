# Research Workspace Vector Indexing Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate TASK-478.30 with a live backend/WebUI run that shows a bounded Research Workspace source progressing through real embeddings/vector indexing to `queryable`, or records a bounded diagnosable failure state.

**Architecture:** Keep source readiness owned by the existing Workspaces status projection: source membership in ChaChaNotes, media readiness in Media DB, in-flight lifecycle in Jobs, and vector storage in ChromaDB. Use live API calls to seed a small media document and start the existing media embeddings Jobs pipeline, then use WebUI/CDP to confirm Research Workspace displays the resulting source status and RAG behavior.

**Tech Stack:** FastAPI Workspaces API, Media embeddings API, Jobs `embeddings_*` worker, ChromaDB, Next.js Research Workspace, Playwright real-backend E2E/CDP.

---

### Task 1: Confirm Live Embeddings Runtime Contract

**Files:**
- Read: `tldw_Server_API/Config_Files/config.txt`
- Read: `tldw_Server_API/app/api/v1/endpoints/media_embeddings.py`
- Read: `tldw_Server_API/app/core/Embeddings/services/jobs_worker.py`
- Update: `backlog/tasks/task-478.30 - Validate-long-running-Research-Workspace-vector-indexing-completion-with-real-embeddings.md`

- [x] **Step 1: Start a live backend outside the sandbox**

Run a single-user backend on an unused local port, with NLTK downloads disabled and normal production/test guardrails intact.

Expected: health endpoint returns `status=ok`.

- [x] **Step 2: Probe embeddings provider availability**

Call the OpenAI-compatible embeddings endpoint and/or media embeddings endpoint with a bounded input and a real provider/model (`huggingface:sentence-transformers/all-MiniLM-L6-v2` preferred over the default Qwen model for local bounded validation).

Expected: either a real embedding vector is returned/queued, or the exact provider/model/dependency failure is captured.

- [x] **Step 3: Decide validation path**

If the provider can produce embeddings, continue to full vector completion. If not, stop short of `Pass`, record the diagnosable blocker in TASK-478.30 and the UAT matrix, and avoid claiming vector-ready behavior.

### Task 2: Drive Source to Vector Completion

**Files:**
- Read: `apps/tldw-frontend/e2e/workflows/research-workspace.real-backend.spec.ts`
- Update if needed: `apps/tldw-frontend/e2e/workflows/research-workspace.real-backend.spec.ts`
- Update: `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`
- Update: `backlog/tasks/task-478.30 - Validate-long-running-Research-Workspace-vector-indexing-completion-with-real-embeddings.md`

- [x] **Step 1: Seed a bounded media document**

Use `POST /api/v1/media/add` with a small text document containing a unique evidence token and `perform_analysis=false`.

Expected: media row exists and exposes text content through `/api/v1/media/{id}`.

- [x] **Step 2: Attach it to a Research Workspace**

Create or load a Research Workspace, add the media item as a workspace source, and read `/api/v1/workspaces/{workspace_id}/sources/status`.

Expected: status reports either `partially_queryable`/`vector_index_pending` before embedding completion or an active job state if processing is still running.

- [x] **Step 3: Start media embeddings and run the existing embeddings worker**

Call `POST /api/v1/media/{media_id}/embeddings` with provider `huggingface`, model `sentence-transformers/all-MiniLM-L6-v2`, small chunk size, and `force_regenerate=true`. Run `python -m tldw_Server_API.app.core.Embeddings.services.jobs_worker` long enough for the root job and stage jobs to complete.

Expected: embeddings job progresses through chunking, embedding, and storage; media `vector_processing` becomes ready; ChromaDB has stored vectors.

- [x] **Step 4: Poll source status until terminal**

Poll `/api/v1/workspaces/{workspace_id}/sources/status` and `/api/v1/media/{media_id}/embeddings/status`.

Expected: source reaches `state=queryable`, `readiness.vector_ready=true`, and embeddings status reports vectors. If it fails, the failure is bounded and diagnostic.

### Task 3: Prove WebUI/RAG Agreement

**Files:**
- Update if needed: `apps/tldw-frontend/e2e/workflows/research-workspace.real-backend.spec.ts`
- Update: `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`
- Update: `backlog/tasks/task-478.30 - Validate-long-running-Research-Workspace-vector-indexing-completion-with-real-embeddings.md`

- [x] **Step 1: Open Research Workspace in Playwright/CDP**

Run the WebUI against the same backend and active workspace.

Expected: the source card agrees with backend status and does not show `Processing` after backend status is vector-ready.

- [x] **Step 2: Run grounded RAG from the selected source**

Ask a question containing the seeded evidence token or call the UI path that triggers `/api/v1/rag/search`.

Expected: RAG request uses the selected media ID and returns grounded evidence from the vector-ready source. If provider chat generation is unavailable, record RAG retrieval evidence and model/provider blocker separately.

### Task 4: Finalize Evidence and Verification

**Files:**
- Update: `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`
- Update: `backlog/tasks/task-478.30 - Validate-long-running-Research-Workspace-vector-indexing-completion-with-real-embeddings.md`

- [x] **Step 1: Update UAT matrix conservatively**

Update RW-UAT-006 and the high-risk remainder only according to the live evidence: pass if vector completion is proven, partial/blocked if provider/runtime prevents completion.

- [x] **Step 2: Run focused verification**

Run focused backend/frontend tests only for files changed. Always run `git diff --check`. Run Bandit if production Python changed; otherwise document skip.

- [x] **Step 3: Close Backlog and commit**

Set TASK-478.30 status and checkboxes according to evidence, record exact commands/outcomes, and commit the plan/docs/tests/backlog changes.
