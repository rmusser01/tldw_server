## Stage 1: Llama.cpp Runtime Mapping
**Goal**: Use the backend-advertised chat provider for llama.cpp models and send raw model ids in Studio and RAG requests.
**Success Criteria**: Focused unit tests fail before the change and pass after; llama.cpp metadata maps to `llama.cpp` while request bodies use the GGUF filename without a provider prefix.
**Tests**: `e2e-fixture-models.test.ts`, `StudioPane.stage1.test.tsx`, `ragMode.sanitization.test.ts`
**Status**: Complete

## Stage 2: Queryable Live UAT Sources
**Goal**: Seed real-backend Research Workspace UAT documents with chunking enabled so selected sources become queryable before grounded chat assertions.
**Success Criteria**: Failed live RAG waits observe `/api/v1/rag/search` instead of stalling on non-queryable `Processing` sources.
**Tests**: Targeted real-backend Playwright rerun for the previously failing grounded chat/search/Studio scenarios.
**Status**: Complete

## Stage 3: Full Llama.cpp UAT Retry
**Goal**: Re-run the standalone Research Workspace UAT matrix against the local llama.cpp provider.
**Success Criteria**: UAT no longer fails due provider reachability, non-queryable seeded sources, or llama.cpp request shaping; any residual failures are classified with fresh evidence.
**Tests**: `bun run e2e:research-workspace:uat` with `/tmp/pr2535-research-workspace-uat-report-llamacpp*.json` evidence.
**Status**: Complete
