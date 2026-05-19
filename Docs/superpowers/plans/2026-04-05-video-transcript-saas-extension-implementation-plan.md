# Video Transcript SaaS Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `executing-plans` or `subagent-driven-development` if the user explicitly requests delegation. Do not reintroduce new public `video-lite` backend endpoints while executing this plan.

**Goal:** Deliver the private extension plus hosted transcript workspace by reusing existing backend ingest jobs, media detail, auth, billing, and chat surfaces. Any dedupe behavior must be implemented behind the existing ingest-job worker path rather than through a new OSS product-specific API contract.

**Architecture:** Keep the store-facing extension and hosted app in the private overlay repo. Keep the OSS backend public surface unchanged for this product. Use existing `POST /api/v1/media/ingest/jobs`, existing ingest-job polling, and existing `GET /api/v1/media/{media_id}` detail retrieval. Implement backend dedupe as an internal worker optimization that still creates a normal per-user media record, transcript, and first-ingest summary.

**Tech Stack:** existing media ingest jobs, existing media detail endpoints, existing auth/billing/chat surfaces, internal worker dedupe shim, private Next.js app, private browser extension package, React, Bun/Vitest, pytest

---

## Execution Status

- This plan supersedes any earlier `video-lite` plan steps that required new public backend routes such as `/api/v1/media/video-lite/...`.
- Do not execute any historical task that creates new `video-lite` backend schemas, services, or endpoints.
- The active implementation direction is:
  - private frontend orchestration only
  - existing ingest-job endpoints
  - existing media detail endpoints
  - internal backend worker dedupe only

## File Structure

### Backend In `tldw_server`

- Modify: `tldw_Server_API/app/services/media_ingest_jobs_worker.py`
  - Add internal source normalization and dedupe checks behind the existing ingest-job worker flow.
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
  - Materialize a per-user transcript record from a deduped canonical transcript artifact.
  - Persist the dedicated per-user summary slot in existing media detail metadata.
  - Reuse an existing same-user media row for the same normalized source instead of creating duplicates.
- Modify: any lower-level DB helpers that `persistence.py` already relies on, if required, to:
  - materialize a per-user transcript record from a deduped canonical transcript artifact
  - keep summaries generated per-user rather than copied cross-user
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py`
  - Add coverage for dedupe-hit behavior and per-user persistence semantics.
- Add: `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingest_jobs_video_dedupe.py`
  - Prove the existing public ingest-job and media-detail endpoints work end to end for this flow.

### Private Overlay Repo

- Modify: `/Users/macbook-dev/Documents/GitHub/tldw-video-lite-private/web/lib/video-lite-client.ts`
  - Replace any custom `video-lite` backend-client assumptions with existing ingest-job and media-detail calls.
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw-video-lite-private/web/pages/lite/new.tsx`
  - Submit existing ingest jobs and preserve job tracking state.
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw-video-lite-private/web/pages/lite/video/[sourceKey].tsx`
  - Route around job polling and existing media detail loading rather than custom workspace endpoints.
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw-video-lite-private/web/components/VideoWorkspacePage.tsx`
  - Render transcript from existing media detail `content.text` and summary from existing `processing.safe_metadata.video_lite.summary`.
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw-video-lite-private/extension/src/background/video-lite.ts`
  - Keep the extension launcher as a hosted-app handoff only.
- Modify: private repo tests and verification docs accordingly.

## Task 1: Remove Stale Product-Specific Backend Assumptions From The Plan Surface

**Files:**
- Modify: `Docs/superpowers/specs/2026-04-05-video-transcript-saas-extension-design.md`
- Modify: `Docs/superpowers/plans/2026-04-05-video-transcript-saas-extension-implementation-plan.md`

- [x] **Step 1: Rewrite the design and plan**
  - remove any instruction that adds new public `/api/v1/media/video-lite/*` endpoints
  - lock the approved reuse-only ingest-job design

- [x] **Step 2: Commit the docs update**

## Task 2: Add Internal Dedupe To Existing Ingest Jobs

**Files:**
- Modify: `tldw_Server_API/app/services/media_ingest_jobs_worker.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py`
- Add: `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingest_jobs_video_dedupe.py`

- [ ] **Step 1: Write failing worker tests**

Cover:

- a normal ingest hit still behaves as today
- a dedupe hit skips download/transcription work
- a dedupe hit still creates a normal completed job
- the completed job still returns `result.media_id`
- same-user repeated ingest resolves to the existing `media_id` instead of creating a duplicate media row
- the transcript is still materialized into the requesting user’s own media record
- the summary is generated for that user rather than copied cross-user
- the dedicated summary payload is stored in `processing.safe_metadata.video_lite.summary`
- a completed job means transcript persistence and the initial summary attempt have both finished
- weak or mutable sources fall back to normal ingest when freshness cannot be established cheaply

- [ ] **Step 2: Run the targeted backend tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingest_jobs_video_dedupe.py -v
```

- [ ] **Step 3: Implement the internal dedupe shim**

Requirements:

- normalize the incoming source behind the worker
- check internal dedupe artifacts without exposing a new public API
- treat same-user repeat ingest as idempotent by reusing the existing media row for the same normalized source
- keep the worker orchestration thin and put transcript materialization and summary persistence in `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
- on dedupe hit:
  - skip repeated download/transcription work
  - create or reuse the user-local media row according to same-user idempotency rules
  - persist the transcript into the user’s DB
  - generate that user’s first summary as part of the same ingest lifecycle
- write the summary into the reserved `processing.safe_metadata.video_lite.summary` slot rather than generic `processing.analysis`
- mark the job `completed` only after transcript persistence and the initial summary attempt have both finished
- fall back to the normal ingest path when source freshness cannot be established cheaply
- return the normal completed job result shape with `media_id`

- [ ] **Step 4: Run backend verification**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingest_jobs_video_dedupe.py -v
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/services/media_ingest_jobs_worker.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py \
  -f json -o /tmp/bandit_video_lite_dedupe.json
```

- [ ] **Step 5: Commit**

Suggested message:

```bash
git commit -m "feat: dedupe video ingest behind existing jobs"
```

## Task 3: Rewire The Private Hosted App To Existing Ingest Jobs

**Files:**
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw-video-lite-private/web/lib/video-lite-client.ts`
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw-video-lite-private/web/pages/lite/new.tsx`
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw-video-lite-private/web/pages/lite/video/[sourceKey].tsx`
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw-video-lite-private/web/components/VideoWorkspacePage.tsx`
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw-video-lite-private/tests/web/VideoWorkspacePage.test.tsx`
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw-video-lite-private/tests/web/video-lite-runtime.test.tsx`

- [ ] **Step 1: Write failing private-frontend tests**

Cover:

- hosted intake submits `POST /api/v1/media/ingest/jobs`
- job polling reads existing job status endpoints
- on completion, `result.media_id` drives the workspace fetch
- workspace loads `GET /api/v1/media/{media_id}`
- transcript renders from `content.text`
- summary renders from `processing.safe_metadata.video_lite.summary`
- summary unavailable or failed states render from the reserved per-user summary metadata
- completed jobs without `media_id` are treated as failures

- [ ] **Step 2: Run the targeted frontend tests to verify they fail**

Run:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw-video-lite-private && bun test \
  tests/web/VideoWorkspacePage.test.tsx \
  tests/web/video-lite-runtime.test.tsx -v
```

- [ ] **Step 3: Replace custom backend-client assumptions**

Requirements:

- no calls to custom `/api/v1/media/video-lite/*` endpoints
- use existing ingest-job submission and polling
- use existing media detail fetches
- keep login and upgrade intent preservation in the private app

- [ ] **Step 4: Run private frontend verification**

Run:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw-video-lite-private && bun test \
  tests/shared/video-lite-intent.test.ts \
  tests/web/VideoWorkspacePage.test.tsx \
  tests/web/video-lite-routing.test.ts \
  tests/extension/sidepanel-video.test.tsx
bunx vitest run --environment jsdom tests/web/video-lite-runtime.test.tsx
cd /Users/macbook-dev/Documents/GitHub/tldw-video-lite-private/web && bun run build
```

- [ ] **Step 5: Commit**

Suggested message:

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw-video-lite-private commit -m "refactor: reuse ingest jobs and media detail for video lite"
```

## Task 4: Keep The Extension Launcher Thin

**Files:**
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw-video-lite-private/extension/src/background/video-lite.ts`
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw-video-lite-private/extension/src/routes/sidepanel-video.tsx`
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw-video-lite-private/tests/extension/sidepanel-video.test.tsx`

- [ ] **Step 1: Write failing launcher tests**

Cover:

- launcher deep-links into hosted intake/workspace routes only
- extension does not depend on a custom backend launcher-state endpoint
- transcript/chat intent is preserved in hosted handoff
- login/upgrade routing stays in the hosted app

- [ ] **Step 2: Run the extension tests to verify they fail**

Run:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw-video-lite-private && bun test \
  tests/extension/sidepanel-video.test.tsx -v
```

- [ ] **Step 3: Implement the thinner launcher contract**

Requirements:

- extension remains a launcher only
- hosted app owns ingest submission, status polling, and workspace load
- no custom backend `launcher_access` surface

- [ ] **Step 4: Re-run extension verification**

Run:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw-video-lite-private && bun test \
  tests/extension/sidepanel-video.test.tsx -v
cd extension && bun run build
```

- [ ] **Step 5: Commit**

Suggested message:

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw-video-lite-private commit -m "refactor: keep video-lite extension launcher-only"
```

## Task 5: Final Verification

**Files:**
- Modify: `/Users/macbook-dev/Documents/GitHub/tldw-video-lite-private/docs/verification.md`

- [ ] **Step 1: Run backend verification**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingest_jobs_video_dedupe.py -v
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/services/media_ingest_jobs_worker.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py \
  -f json -o /tmp/bandit_video_lite_dedupe.json
```

- [ ] **Step 2: Run private repo verification**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw-video-lite-private && bun test \
  tests/shared/video-lite-intent.test.ts \
  tests/web/VideoWorkspacePage.test.tsx \
  tests/web/video-lite-routing.test.ts \
  tests/extension/sidepanel-video.test.tsx
bunx vitest run --environment jsdom tests/web/video-lite-runtime.test.tsx
cd /Users/macbook-dev/Documents/GitHub/tldw-video-lite-private/web && bun run build
cd /Users/macbook-dev/Documents/GitHub/tldw-video-lite-private/extension && bun run build
```

- [ ] **Step 3: Update verification docs**

Record:

- no custom backend `video-lite` endpoints
- existing ingest-job flow reused
- existing media detail flow reused
- dedupe lives behind worker internals only
- same-user repeat ingest reuses the existing `media_id` instead of creating duplicate rows
- lightweight Summary tab reads the reserved `processing.safe_metadata.video_lite.summary` field
- completed ingest jobs include the initial summary attempt, with failure recorded as terminal per-user metadata

## Notes For The Implementer

- Do not add new public backend routes for this product.
- Do not add a new OSS `video-lite` workspace contract.
- Treat `POST /api/v1/media/ingest/jobs` as the canonical ingest entry point.
- Treat ingest job `result.media_id` as the canonical handoff into hosted workspace loading.
- Keep transcript and summary storage per-user.
- Keep same-user repeat ingest idempotent at the media-row level.
- Keep the stable lightweight summary in `processing.safe_metadata.video_lite.summary`, not generic `processing.analysis`.
- Keep dedupe internal to worker orchestration plus persistence-layer materialization logic.
- Keep the extension launcher-only in V1.
