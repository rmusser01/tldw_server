# Summarization, Media, and Audio Service Prompts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate exactly the sixteen approved summarization, media, transcript, Research Studio audio, and slide-summary prompt definitions to the shared service-prompt system while preserving no-override provider-message bytes, explicit override semantics, size limits, provenance safety, and authenticated Media-ingest Jobs pinning.

**Architecture:** Register the sixteen inventory-approved multipart contracts in the shared registry, then resolve each definition once at its server-side inventory boundary. This first rollout domain establishes an authenticated `/api/v1/chat/completions` execution bridge: browser workflows send only a typed `service_prompt` object, and the chat endpoint resolves and assembles provider messages from `AuthPrincipal`, so TypeScript never receives effective/hidden parts or recreates locked assembly. Synchronous backend workflows receive one immutable resolved bundle, while Media audio-analysis Jobs use the protected held-bind-release pin path and verified WorkerSDK context. Existing deterministic source truncation/chunking remains upstream of rendering, and no domain-specific resolver, persistence layer, or template syntax is introduced.

**Tech Stack:** FastAPI/Pydantic, shared `Service_Prompts` registry/resolver/templates/service, Jobs `ServicePromptJobPinner` and `WorkerSDK`, React/TypeScript, existing `tldwRequest`, Vitest/Testing Library, pytest/Hypothesis where useful, Playwright, Bandit.

---

## Work record, prerequisites, and stop conditions

- Backlog task: `TASK-12957` (`Migrate summarization, media, and audio service prompts`). Use the official Backlog MCP/CLI workflow; do not edit task files manually.
- Required foundations: the registry/resolver, persistence/API, protected Jobs pinning, and shared settings UI/client from plans 03–06 must be implemented and green before this domain plan begins.
- Task 2 of this plan establishes the shared browser execution bridge before any browser migration in Tasks 3–5. TASK-12959 and later domain plans depend on and must reuse this bridge; they must not add parallel execution endpoints or browser resolvers.
- Reuse `ServicePromptRegistry`, `PromptExecutionContext`, the shared constrained renderer and budgets, the service/API lifecycle, `ServicePromptJobPinner.enqueue(...)`, and the WorkerSDK verified immutable prompt context. Do not introduce a second resolver, prompt store, browser cache, or pin verifier.
- Before Task 3, verify the Task-2 bridge in `tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py`, `tldw_Server_API/app/api/v1/endpoints/chat.py`, `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`, `apps/packages/ui/src/services/service-prompts.ts`, and `apps/packages/ui/src/services/__tests__/service-prompts.test.ts`. It must be optional, `extra="forbid"`, authenticated, allow only declared named variables/explicit parts/finite selectors, return no prompt body, map shared size failures to 413, and preserve complete provider-message arrays. If any condition is absent, stop and finish Task 2; do not add a domain endpoint, fetch effective bodies, or resolve in TypeScript.
- If an authenticated owner cannot be threaded from the slides API or MCP execution context, stop that caller migration instead of using a userless/global fallback.
- `slides.source.summary` remains a server-only substage selected inside the Slides endpoint when source text exceeds its direct budget. Browser callers send source/options to the Slides API and never receive summary prompt parts; if an active browser path bypasses Slides and invokes chat for this definition, it must use Task 2's typed bridge or remain unmigrated.
- If a legacy provider-message golden cannot be preserved inside the approved contract and budgets, stop and document the exact byte diff in `TASK-12957`; do not silently normalize whitespace or delimiters.
- The current planning-only skip does not waive implementation verification. Focused tests, mandatory full CI shards, Bandit, and `git diff --check` remain required.

## Stage map

| Stage | Tasks | Goal | Success criteria | Status |
| --- | --- | --- | --- | --- |
| 1. Contracts and bridge | Tasks 1–2 | Register exactly sixteen definitions and establish authenticated browser execution | exact-set contracts, complete-message Goldens, typed bridge, and size-boundary tests pass | Not Started |
| 2. Browser runtimes | Tasks 3–5 | Migrate Analysis, Review, transcript, and Research Studio audio | browsers send request-only execution data; server provider-message Goldens pass | Not Started |
| 3. Synchronous server runtimes | Tasks 6–7 | Migrate Slides source summarization and every direct audio-analysis caller | authenticated boundaries resolve once and pass one immutable bundle; no ownerless lookup remains | Not Started |
| 4. Protected Jobs and canary | Tasks 8–9 | Pin Media-ingest analysis and prove cross-surface behavior | held-bind-release, WorkerSDK verification, exact limits, and real-server canary pass | Not Started |
| 5. Release gate | Task 10 | Reconcile, document, secure, and verify the domain | mandatory backend/frontend gates, Bandit, inventory audit, and Backlog finalization pass | Not Started |

## Exact scope lock

Register and migrate these definitions—no more and no fewer:

| Service prompt ID | Boundary | Editable/constrained parts and locked carrier |
|---|---|---|
| `media.analysis.critical` | Analysis modal | `system`, optional `user_prefix`; locked conditional `\n\n` and `media_content` |
| `media.analysis.executive` | Analysis modal | `system`, optional `user_prefix`; locked conditional `\n\n` and `media_content` |
| `media.analysis.qa` | Analysis modal | `system`, optional `user_prefix`; locked conditional `\n\n` and `media_content` |
| `media.audio.analysis` | Media audio analysis | `system`, `user_instruction`; locked transcript/chunk carrier and exact `\n\n\n\n` separator |
| `media.review.bullets` | Review page | `system`, optional `user_prefix`; locked conditional `\n\n` and `selected_content` |
| `media.review.critical` | Review page | `system`, optional `user_prefix`; locked conditional `\n\n` and `selected_content` |
| `media.review.qa` | Review page | `system`, optional `user_prefix`; locked conditional `\n\n` and `selected_content` |
| `media.review.summary.bullets` | Review page | `system`, optional `user_prefix`; locked conditional `\n\n` and `selected_content` |
| `media.review.summary.detailed` | Review page | `system`, optional `user_prefix`; locked conditional `\n\n` and `selected_content` |
| `media.review.summary.executive` | Review page | `system`, optional `user_prefix`; locked conditional `\n\n` and `selected_content` |
| `media.transcript.clean` | Content Review | editable `system_semantics`, locked fidelity constraints, editable `user_instruction`, locked content delimiters |
| `media.transcript.correction` | Content Review | editable `system_semantics`, locked immutable constraints, editable `user_instruction`, locked content delimiters |
| `media.transcript.headings` | Content Review | editable `system_semantics`, locked additive constraints, editable `user_instruction`, locked content delimiters |
| `media.transcript.speakerturns` | Content Review | editable `system_semantics`, locked turn constraints, editable `user_instruction`, locked content delimiters |
| `research.studio.audio` | Research Studio audio overview | `system`, `audio_script_semantics`; locked source carrier |
| `slides.source.summary` | Slides source summarization | `system`, `summary_semantics`; locked source-chunk carrier |

Copy the inventory topology literally (`E-L` = editable literal, `LV-L` = locked visible literal, `LV-T` = locked visible constrained template, `LH-T` = locked hidden constrained template, `G` = the shared general byte cap):

- Analysis: `system (E-L) ⇒ user_prefix (E-L, optional) → conditional two-newline separator (LV-L) → media_content (LH-T)`; render `user_prefix` and `media_content` once each.
- Review: `system (E-L) ⇒ user_prefix (E-L, optional) → conditional two-newline separator (LV-L) → selected_content (LH-T)`; render `user_prefix` and `selected_content` once each.
- Audio analysis: `system (E-L) ⇒ transcript_or_chunk (LH-T) → four-newline separator (LV-L) → user_instruction (E-L)`; render `transcript_or_chunk` once.
- Transcript actions: `system_semantics (E-L) → action-specific constraint (LV-L) ⇒ user_instruction (E-L) → content_delimiters (LV-T)`; render `content` once.
- Research Studio audio: `system (E-L) ⇒ audio_script_semantics (E-L) → source_carrier (LV-T)`; render `source_text` once after the existing 18,000-character shaping.
- Slides source summary: `system (E-L) ⇒ summary_semantics (E-L) → source_chunk (LV-T)`; render each already-split chunk once from the single resolved definition.

All other inventory rows remain unchanged. Do not register opportunistic media defaults or migrate adjacent recursive-document, streaming-insight, Scheduler, or general summarization boundaries in this task.

## Non-negotiable runtime contract

1. Resolve once at the server-side inventory boundary and pass a frozen/immutable bundle through loops. Browser actions cross the existing chat bridge with an ID, runtime variables, allowed explicit parts, and finite selectors only; `chat.py` resolves once. No TypeScript or lower backend layer performs another user/deployment/package lookup.
2. Preserve precedence exactly: explicit request override → authenticated job pin → approved user revision → deployment provider → packaged default.
3. Preserve each inventory row's exact literal-vs-constrained-template declaration. Literal fields—including braces—are never parsed as templates.
4. Preserve explicitness exactly:
   - Browser saved/manual fields are presence-based; a present empty string remains an explicit empty literal part in the typed execution request.
   - `media.audio.analysis` request fields retain their current truthy semantics; empty strings mean omitted/default.
   - Preset selection is a definition selector, while a manual edit, Prompt Library value, or locally saved value is an explicit part override.
5. No-override calls must be byte-equivalent to current provider messages, including whitespace, conditional separators, delimiters, and message order.
6. Enforce UTF-8 budgets at both authoring and execution boundaries:
   - Authored part and expanded variable/rendered part: 65,536 bytes allowed; 65,537 rejected.
   - Authored definition and final rendered bundle: 262,144 bytes allowed; 262,145 rejected.
   - HTTP execution rejection is 413. Non-HTTP execution uses the foundation's canonical stable typed code `service_prompt_size_limit_exceeded`.
7. Do not silently truncate to satisfy service-prompt budgets. Preserve only pre-existing deterministic input shaping: Research Studio source excerpts are capped at 18,000 characters, and Slides uses at most `SLIDES_MAX_SOURCE_CHUNKS` (default 20). Media/review/transcript/audio variables reject rather than truncate.
8. Logs, statuses, events, and API-safe provenance contain definition IDs, source kinds, contract versions, revision/snapshot identifiers, and digests only—never raw prompt bodies, rendered source content, MACs, or keys.
9. Ordinary Jobs payloads never contain raw prompt bodies. For prompt-related data, a prompt-bearing audio-analysis job contains only the protected pin-set references/digests defined by the foundation; its existing non-prompt Media-ingest fields remain unchanged.
10. `media.audio.analysis` declares the finite requirement set exactly `{media.audio.analysis}` only when the input is audio and an analysis provider is active. Enqueue creates/attaches the complete protected pin set before queue release; the worker must consume the verified WorkerSDK context before provider dispatch.

## Authoritative implementation file map

### Shared backend definitions and tests

- Create: `tldw_Server_API/Config_Files/Prompts/service_prompts_summarization_media_audio.prompts.yaml`
- Modify: `tldw_Server_API/app/core/Service_Prompts/registry.py`
- Modify: `tldw_Server_API/app/core/Service_Prompts/templates.py`
- Create: `tldw_Server_API/tests/Service_Prompts/test_registry_summarization_media_audio.py`
- Create: `tldw_Server_API/tests/Service_Prompts/test_runtime_summarization_media_audio.py`
- Modify: `tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py`

The packaged `media.audio.analysis` definition must use compatibility mappings for the existing `audio/Transcription Analysis Summary`, `audio/System Prompt`, and their environment/file override behavior in `tldw_Server_API/Config_Files/Prompts/audio.prompts.yaml`; do not create a competing duplicate default source.

### Authenticated chat execution bridge established by this plan

- Modify: `tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py`
- Create: `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`
- Modify: `apps/packages/ui/src/services/service-prompts.ts`
- Modify: `apps/packages/ui/src/services/tldw-server.ts`
- Modify: `apps/packages/ui/src/services/__tests__/service-prompts.test.ts`
- Modify with transport-redaction cases: `apps/packages/ui/src/services/__tests__/tldw-chat.message-sanitization.test.ts`

### Analysis and Review

- Modify: `apps/packages/ui/src/components/Media/analysisPresets.ts`
- Modify: `apps/packages/ui/src/components/Media/AnalysisModal.tsx`
- Modify: `apps/packages/ui/src/components/Media/__tests__/AnalysisModal.stage3.regression.test.tsx`
- Modify: `apps/packages/ui/src/components/Review/PromptDropdown.tsx`
- Modify: `apps/packages/ui/src/components/Review/ReviewPage.tsx`
- Create: `apps/packages/ui/src/components/Review/__tests__/ReviewPage.service-prompts.test.tsx`

### Transcript Content Review

- Modify: `apps/packages/ui/src/utils/content-review-ai.ts`
- Modify: `apps/packages/ui/src/components/ContentReview/ContentReviewPage.tsx`
- Create: `apps/packages/ui/src/utils/__tests__/content-review-ai.service-prompts.test.ts`
- Create: `apps/packages/ui/src/components/ContentReview/__tests__/ContentReviewPage.service-prompts.test.tsx`

### Research Studio audio

- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/audio-overview-service-prompt.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx`
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/__tests__/audio-overview-service-prompt.test.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx`

### Media audio and Jobs

- Modify: `tldw_Server_API/app/api/v1/schemas/media_request_models.py`
- Modify: `tldw_Server_API/app/api/v1/API_Deps/media_add_deps.py`
- Modify: `tldw_Server_API/app/api/v1/API_Deps/media_processing_deps.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_audios.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/audio_batch.py`
- Modify: `tldw_Server_API/app/services/media_ingest_jobs_worker.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Files.py`
- Create: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_service_prompts.py`
- Create: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_process_audios_service_prompts.py`
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_batch_counts.py`
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py`
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py`
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_process_batch_media_precheck_regressions.py`
- Create: `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_audio_service_prompt_jobs.py`

### Slides

- Modify: `tldw_Server_API/app/core/Slides/slides_generator.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/slides.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/slides_module.py`
- Modify: `tldw_Server_API/tests/Slides/test_slides_generator.py`
- Modify: `tldw_Server_API/tests/Slides/test_slides_api.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_slides_module.py`

### Canary, documentation, and tracking

- Modify: `apps/tldw-frontend/e2e/workflows/service-prompts-settings.spec.ts`
- Modify: `Docs/Design/service-prompt-inventory.md`
- Modify: `Docs/API/service-prompts.md`
- Modify: `tldw_Server_API/Config_Files/Prompts/README.md`
- Update through Backlog MCP/CLI: `TASK-12957`

## Mandatory commit and final-domain gates

Focused red/green commands below supplement this gate; they never replace it. The requester's planning-time shard skip does not carry into implementation. Do not commit if any required shard fails, even for an apparently unrelated or environmental reason. Diagnose under the repository's three-attempt rule, record the evidence in `TASK-12957`, and stop if the gate cannot be made green.

- [ ] From the repository root, always run the complete backend suite:

```bash
source .venv/bin/activate
python -m pytest -v
git diff --check
```

- [ ] For Tasks 1–9, run formatting, lint, and type checks with the exact task-local source/test arrays printed in that task. Those arrays contain only files that exist by that task's commit boundary. Do not substitute the complete-domain arrays below for an early commit.

- [ ] Only at Task 10, after every planned source and test exists, run the complete-domain Python arrays:

```bash
source .venv/bin/activate
FINAL_PYTHON_SCOPE=(
  tldw_Server_API/app/core/Service_Prompts/models.py
  tldw_Server_API/app/core/Service_Prompts/registry.py
  tldw_Server_API/app/core/Service_Prompts/templates.py
  tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py
  tldw_Server_API/app/api/v1/endpoints/chat.py
  tldw_Server_API/app/api/v1/schemas/media_request_models.py
  tldw_Server_API/app/api/v1/API_Deps/media_processing_deps.py
  tldw_Server_API/app/api/v1/API_Deps/media_add_deps.py
  tldw_Server_API/app/api/v1/endpoints/media/process_audios.py
  tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py
  tldw_Server_API/app/core/Ingestion_Media_Processing/audio_batch.py
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Files.py
  tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py
  tldw_Server_API/app/services/media_ingest_jobs_worker.py
  tldw_Server_API/app/core/Slides/slides_generator.py
  tldw_Server_API/app/api/v1/endpoints/slides.py
  tldw_Server_API/app/core/MCP_unified/modules/implementations/slides_module.py
)
FINAL_PYTHON_TEST_SCOPE=(
  tldw_Server_API/app/core/MCP_unified/tests/test_slides_module.py
  tldw_Server_API/tests/Chat/test_service_prompt_execution.py
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_audio_service_prompt_jobs.py
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_batch_counts.py
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_service_prompts.py
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_process_audios_service_prompts.py
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_process_batch_media_precheck_regressions.py
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py
  tldw_Server_API/tests/Service_Prompts/test_registry_summarization_media_audio.py
  tldw_Server_API/tests/Service_Prompts/test_runtime_summarization_media_audio.py
  tldw_Server_API/tests/Slides/test_slides_api.py
  tldw_Server_API/tests/Slides/test_slides_generator.py
)
python -m black --check "${FINAL_PYTHON_SCOPE[@]}" "${FINAL_PYTHON_TEST_SCOPE[@]}"
python -m ruff check "${FINAL_PYTHON_SCOPE[@]}" "${FINAL_PYTHON_TEST_SCOPE[@]}"
python -m mypy "${FINAL_PYTHON_SCOPE[@]}" "${FINAL_PYTHON_TEST_SCOPE[@]}"
```

- [ ] For a commit touching `apps/`, run every frontend shard and the touched-scope format/lint/type/build checks:

```bash
cd apps/tldw-frontend
bun run test:run
bunx vitest run -c vitest.extension.config.ts
cd ../packages/ui
bun run test
cd ../../tldw-frontend
bun run format:check
bun run lint
bunx tsc --noEmit -p ../packages/ui/tsconfig.json
bun run build
```

- [ ] Before each commit, update `TASK-12957` through the official Backlog MCP/CLI workflow with the current stage, touched files, red/green commands, full-gate results, blockers, and summary. Never hand-edit its task Markdown.

## Task 1: Register the sixteen contracts and packaged defaults

**Files:**

- Create: `tldw_Server_API/Config_Files/Prompts/service_prompts_summarization_media_audio.prompts.yaml`
- Modify: `tldw_Server_API/app/core/Service_Prompts/registry.py`
- Create: `tldw_Server_API/tests/Service_Prompts/test_registry_summarization_media_audio.py`
- Modify: `tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py`

- [ ] Move `TASK-12957` to In Progress through the official Backlog workflow and link this plan.
- [ ] Add a failing registry test that asserts the catalog contribution is the exact sixteen-ID set above, each part's literal/constrained mode, editable/locked visibility, assembly order, variable contract, compatibility mapping, safe samples, and rollout availability. Assert representative adjacent inventory rows are absent from this contribution.
- [ ] Add byte-golden fixtures from the current constants/presets for every no-override system and user message. For dynamic content use fixed sentinel values containing braces, Unicode, and delimiter-like text so accidental parsing or normalization is visible. Assert each constrained carrier declares exactly its inventory variable once and startup rejects missing/extra placeholders.
- [ ] Run `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Service_Prompts/test_registry_summarization_media_audio.py`; expect collection/assertion failure because the sixteen definitions and packaged asset do not exist.
- [ ] Add the packaged YAML and registry entries. Keep `media.audio.analysis` wired to the existing compatibility module/key/environment sources rather than copying its legacy defaults into a second provider.
- [ ] Make English labels/descriptions and category/tags/workflow IDs sufficient for Settings UI discovery without exposing implementation module names.
- [ ] Extend API integration assertions so catalog/detail expose only safe metadata and visible editable/locked parts; hidden content remains digest/presence-only.
- [ ] Rerun the focused registry/API tests; expect all sixteen definitions and golden defaults to pass.
- [ ] Run the mandatory commit gate plus this exact Task 1 Python scope; update Backlog:

```bash
source .venv/bin/activate
TASK_PYTHON_SOURCE_SCOPE=(
  tldw_Server_API/app/core/Service_Prompts/registry.py
)
TASK_PYTHON_TEST_SCOPE=(
  tldw_Server_API/tests/Service_Prompts/test_registry_summarization_media_audio.py
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py
)
python -m black --check "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m ruff check "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m mypy "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
```
- [ ] Commit: `feat: register summarization media service prompts (TASK-12957)`.

## Task 2: Establish the authenticated chat execution bridge and exact UTF-8 failures

**Files:**

- Modify: `apps/packages/ui/src/services/service-prompts.ts`
- Modify: `apps/packages/ui/src/services/tldw-server.ts`
- Modify: `apps/packages/ui/src/services/__tests__/service-prompts.test.ts`
- Modify: `apps/packages/ui/src/services/__tests__/tldw-chat.message-sanitization.test.ts`
- Modify: `tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py`
- Modify: `tldw_Server_API/app/core/Service_Prompts/models.py`
- Modify: `tldw_Server_API/app/core/Service_Prompts/registry.py`
- Create: `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`
- Modify: `tldw_Server_API/app/core/Service_Prompts/templates.py`
- Modify: `tldw_Server_API/tests/Service_Prompts/test_registry_summarization_media_audio.py`
- Create: `tldw_Server_API/tests/Service_Prompts/test_runtime_summarization_media_audio.py`
- Modify: `tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py`

### Registry-owned provider-message merge policies

Task 2 establishes one closed, code-defined merge-policy field on the registry definition. The `service_prompt` request object and every TypeScript client type contain no policy, role, insertion-index, or message-order field; a caller cannot select or override how messages are merged. The bridge reads the policy only from the resolved registry definition and supports exactly:

1. `replace_generated_messages`: replace the complete generated provider-message array for standalone workflows; an empty incoming `messages` array is valid only when `service_prompt` is present and its registry definition declares this policy.
2. `replace_copilot_user_text_preserve_non_text`: replace only the copilot current-user text content with the one rendered user text while preserving every authenticated non-text/image content block byte-for-byte and in its existing relative order.
3. `insert_before_current_user`: preserve permitted history and insert the complete resolved message bundle immediately before the final current-user message.
4. `prepend_system_preserve_history`: prepend the resolved system message before permitted agent/continuation history without rewriting, reordering, or dropping any preserved history/current-user message.

Definitions that execute through the chat bridge must declare one of these policies in the registry. The fourteen browser definitions in this task declare `replace_generated_messages`; `media.audio.analysis` and `slides.source.summary` remain non-chat boundaries and do not become browser-executable merely because the dispatcher exists. Tests for the other three policies use synthetic test-only definitions and must not register a seventeenth product ID.

- [ ] Confirm the foundation's single size error code is `service_prompt_size_limit_exceeded`. If the landed foundation uses a different canonical code, stop and normalize the foundation plus every assertion to one code before proceeding; never retain aliases.
- [ ] Write failing schema/endpoint tests for an optional `service_prompt` object with `extra="forbid"`, definition ID, declared runtime variables, allowed explicit parts carrying `literal|template` kind, and allowlisted finite selectors. Explicitly reject client-supplied `merge_policy`, `policy`, `roles`, `order`, `insertion_index`, or equivalent fields. Cover absent-object backward compatibility, malformed/unknown fields, `AuthPrincipal` ownership, cross-user isolation, one resolver call immediately before provider construction, complete provider-message arrays, no prompt body in responses, and fail-closed store/quarantine/configuration errors. Preserve the existing ordinary-chat empty-`messages` rejection and its current sanitized error when `service_prompt` is absent.
- [ ] Add failing backend dispatcher tests for all four registry-owned policies. For `replace_generated_messages`, prove `messages=[]` is accepted only with that declared policy and the rendered bundle becomes the complete provider array. For `replace_copilot_user_text_preserve_non_text`, use a current-user content array containing text plus authenticated image/non-text blocks and prove only the text changes. For `insert_before_current_user`, prove every history byte is preserved and the resolved bundle appears immediately before the final current-user message. For `prepend_system_preserve_history`, prove the resolved system message is first and every incoming history/current-user message remains byte-identical and ordered. Assert incompatible message shapes and any attempt to influence the policy fail before provider dispatch.
- [ ] Add failing backend bridge cases for representative Analysis, transcript, and Research Studio audio shapes: authenticated precedence, exact complete arrays, literal braces, present-empty literal parts, missing/extra variables, illegal part/kind/selector, safe provenance, and zero provider calls on resolution failure.
- [ ] Write failing TypeScript request-builder and transport tests over all four incoming-message shapes: standalone `messages: []`; copilot current-user multimodal content containing text plus image/non-text blocks; web-search history ending in the current-user message that the server will insert before; and agent/continuation history that the server will prepend to. Snapshot each serialized request and prove its ordinary `messages` value, roles, content blocks, and order are transported byte-for-byte unchanged for the server-owned policy. Separately assert `service_prompt` serializes only ID, variables, allowed explicit parts, and finite selectors and cannot emit a role, message order, insertion index, merge-policy name, or policy-like field. Preserve a present empty literal; never call catalog/detail/effective-body APIs; carry no locked strings or client-rendered service-prompt messages; and redact variables/explicit content from logs/errors.
- [ ] Write failing Python boundary tests proving 65,536/65,537 UTF-8 bytes for authored parts and expanded variables/rendered parts, plus 262,144/262,145 for authored definitions and final rendered bundles. Use multibyte cases so character counts cannot pass as byte counts, and cover missing/extra variables plus repetition-count rejection for every constrained carrier shape in this domain. Exercise the aggregate boundary with a synthetic in-test five-part definition whose individual parts remain at or below 65,536 bytes and whose exact assembly/separator bytes are counted; do not register a seventeenth product definition.
- [ ] Assert the settings/preview API and chat execution bridge return 413 and a non-HTTP renderer throws the exact typed code `service_prompt_size_limit_exceeded`; assert no partial provider call and no raw part/variable appears in error text, logs, or provenance.
- [ ] Run `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/__tests__/service-prompts.test.ts ../packages/ui/src/services/__tests__/tldw-chat.message-sanitization.test.ts`; expect failure because no request-only execution builder/transport field exists.
- [ ] Run `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Chat/test_service_prompt_execution.py tldw_Server_API/tests/Service_Prompts/test_runtime_summarization_media_audio.py tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py`; expect failure because the chat request schema and endpoint do not yet execute service prompts.
- [ ] Add the closed `replace_generated_messages | replace_copilot_user_text_preserve_non_text | insert_before_current_user | prepend_system_preserve_history` policy type to the registry-owned definition model and implement one server dispatcher in `chat.py`. The dispatcher obtains the policy from the resolved definition only, validates its required incoming/resolved message shape, and applies the exact transformations above. It must never read a policy, role, or ordering instruction from request data. Assign `replace_generated_messages` to the fourteen scoped browser definitions without adding an ID; keep `media.audio.analysis` and `slides.source.summary` on their existing non-chat boundaries.
- [ ] Add the optional typed object to `ChatCompletionRequest`. In `chat.py`, validate its definition-specific variables/parts/selectors against the registry, build `PromptExecutionContext` from `AuthPrincipal`, resolve exactly once immediately before provider-message construction, then apply the registry-owned dispatcher and pass the immutable merged array to the existing provider path. `messages=[]` is accepted only for a present Service Prompt whose definition declares `replace_generated_messages`; when `service_prompt` is absent, retain the byte-identical ordinary chat fast path and ordinary empty-chat rejection.
- [ ] Add a narrow `buildServicePromptExecution(...)` request builder in `service-prompts.ts` and thread its result unchanged through `tldw-server.ts` as `ChatCompletionRequest.service_prompt`. This is transport only: no policy/role/order field, settings/detail fetch, precedence, rendering, locked assembly, or content persistence in TypeScript.
- [ ] Reuse the foundation's canonical `service_prompt_size_limit_exceeded` exception and map it to 413 in `chat.py`; do not fork budget logic or add a second endpoint. Document the four registry-owned merge policies and the non-selectable client contract so TASK-12959 and later plans reuse them without modifying the dispatcher.
- [ ] Rerun both focused commands; expect all four merge-policy tests, ordinary empty-chat rejection, policy-free typed transport, exact-limit, precedence, no-body, and safe-error tests to pass.
- [ ] Run the mandatory commit/frontend gates plus this exact Task 2 Python scope; update Backlog:

```bash
source .venv/bin/activate
TASK_PYTHON_SOURCE_SCOPE=(
  tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py
  tldw_Server_API/app/api/v1/endpoints/chat.py
  tldw_Server_API/app/core/Service_Prompts/models.py
  tldw_Server_API/app/core/Service_Prompts/registry.py
  tldw_Server_API/app/core/Service_Prompts/templates.py
)
TASK_PYTHON_TEST_SCOPE=(
  tldw_Server_API/tests/Chat/test_service_prompt_execution.py
  tldw_Server_API/tests/Service_Prompts/test_registry_summarization_media_audio.py
  tldw_Server_API/tests/Service_Prompts/test_runtime_summarization_media_audio.py
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py
)
python -m black --check "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m ruff check "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m mypy "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
```
- [ ] Commit: `feat: add authenticated service prompt chat bridge (TASK-12957)`.

## Task 3: Migrate Analysis and Review preset selectors

**Files:**

- Modify: `apps/packages/ui/src/components/Media/analysisPresets.ts`
- Modify: `apps/packages/ui/src/components/Media/AnalysisModal.tsx`
- Modify: `apps/packages/ui/src/components/Media/__tests__/AnalysisModal.stage3.regression.test.tsx`
- Modify: `apps/packages/ui/src/components/Review/PromptDropdown.tsx`
- Modify: `apps/packages/ui/src/components/Review/ReviewPage.tsx`
- Create: `apps/packages/ui/src/components/Review/__tests__/ReviewPage.service-prompts.test.tsx`
- Modify: `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`

- [ ] Add failing Analysis tests for `critical`, `executive`, and `qa` selector-to-ID mapping; one typed `service_prompt` request per Generate action; variables exactly `{media_content}`; custom/saved/Prompt Library values only as explicit literal parts; present empty strings preserved; literal braces transported unchanged; and no client-rendered system/user prompt.
- [ ] Add failing Review tests for the six mappings and the same typed transport/explicitness rules with variables exactly `{selected_content}`. Assert neither component calls catalog/detail or contains the locked conditional separator/carrier assembly.
- [ ] Extend the backend bridge golden table for all nine Analysis/Review IDs, proving the server emits the exact current system/user arrays and conditional `user_prefix + "\n\n" + content` bytes for absent, nonempty, and present-empty explicit prefixes.
- [ ] Run `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Media/__tests__/AnalysisModal.stage3.regression.test.tsx ../packages/ui/src/components/Review/__tests__/ReviewPage.service-prompts.test.tsx && cd ../.. && source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Chat/test_service_prompt_execution.py -k 'media_analysis or media_review'`; expect missing typed execution requests and bridge golden cases.
- [ ] Add `servicePromptId` to only the three eligible Analysis preset records. Track preset selection separately from text editing: choosing a preset selects a definition; any subsequent manual, saved, or Prompt Library value is an explicit literal part.
- [ ] Extend Review `PromptDropdown` with an ID-bearing preset selection callback without changing its existing manual-value contract, then map only the six eligible presets.
- [ ] Preserve existing localStorage keys and presence semantics; do not treat a saved `""` as absent. Build the typed bridge request at the action boundary and pass it unchanged through `createChatCompletion`.
- [ ] Remove client provider-message assembly for the migrated presets. The browser sends content only as a declared runtime variable and sends no locked separator/carrier text; `chat.py` resolves and assembles once from the authenticated principal.
- [ ] Keep existing non-eligible/unselected defaults on their legacy path so this task does not expand the registry set.
- [ ] Rerun the focused Vitest/pytest command; expect typed transport and server-side byte goldens to pass, with no prompt body returned to TypeScript.
- [ ] Run the mandatory commit/frontend gates plus this exact Task 3 Python test scope; update Backlog:

```bash
source .venv/bin/activate
TASK_PYTHON_TEST_SCOPE=(tldw_Server_API/tests/Chat/test_service_prompt_execution.py)
python -m black --check "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m ruff check "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m mypy "${TASK_PYTHON_TEST_SCOPE[@]}"
```
- [ ] Commit: `feat: route browser media prompts through chat bridge (TASK-12957)`.

## Task 4: Migrate transcript correction and rewrite actions

**Files:**

- Modify: `apps/packages/ui/src/utils/content-review-ai.ts`
- Modify: `apps/packages/ui/src/components/ContentReview/ContentReviewPage.tsx`
- Create: `apps/packages/ui/src/utils/__tests__/content-review-ai.service-prompts.test.ts`
- Create: `apps/packages/ui/src/components/ContentReview/__tests__/ContentReviewPage.service-prompts.test.tsx`
- Modify: `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`

- [ ] Add failing utility/component tests that correction, clean, headings, and speaker-turn actions build exactly one typed request with the matching definition ID and variables exactly `{content}`; optional manual values become only declared explicit literal parts, including present empty strings and braces. Assert no constraint/delimiter text, locally rendered prompt message, catalog/detail fetch, or raw body log exists in TypeScript.
- [ ] Extend the backend bridge table for the four definitions, including exact legacy provider bytes `instruction + "\n\n<<<CONTENT>>>\n" + content + "\n<<<END>>>"`, content once, editable/locked order, precedence, literal braces, 413 before dispatch, and sanitized errors.
- [ ] Run `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/utils/__tests__/content-review-ai.service-prompts.test.ts ../packages/ui/src/components/ContentReview/__tests__/ContentReviewPage.service-prompts.test.tsx && cd ../.. && source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Chat/test_service_prompt_execution.py -k media_transcript`; expect hardcoded browser prompt assembly and missing bridge goldens to fail.
- [ ] Refactor `content-review-ai.ts` into narrow typed execution-request builders. Preserve unrelated exported APIs, but remove the four migrated prompt constants/assembly paths once the server packaged goldens cover them.
- [ ] In `ContentReviewPage`, build the request at the action boundary and pass it unchanged through `runChatRewrite`/the chat transport. Do not resolve, render, or append delimiters in the browser.
- [ ] Keep fidelity/immutable/additive/turn constraints and delimiters exclusively in the registered server bundle; `chat.py` renders `content` once and assembles the provider array.
- [ ] Rerun the focused Vitest/pytest command; expect typed requests, four server byte goldens, and UI error/no-dispatch assertions to pass.
- [ ] Run the mandatory commit/frontend gates plus this exact Task 4 Python test scope; update Backlog:

```bash
source .venv/bin/activate
TASK_PYTHON_TEST_SCOPE=(tldw_Server_API/tests/Chat/test_service_prompt_execution.py)
python -m black --check "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m ruff check "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m mypy "${TASK_PYTHON_TEST_SCOPE[@]}"
```
- [ ] Commit: `feat: migrate transcript service prompts (TASK-12957)`.

## Task 5: Migrate Research Studio audio overview

**Files:**

- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/audio-overview-service-prompt.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx`
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/__tests__/audio-overview-service-prompt.test.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx`
- Modify: `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`

- [ ] Add failing TypeScript tests for a request-only helper that emits `definition_id="research.studio.audio"` and variables exactly `{source_text}` plus only declared explicit parts; assert no system/instruction/carrier bytes, settings/detail fetch, local rendering, or raw body logging.
- [ ] Add the backend bridge golden for the current system string and exact user bytes, including the numbered 2–3 minute instructions, blank lines, `Selected sources:\n`, braces, Unicode, authenticated precedence, and safe 413 failure before provider dispatch.
- [ ] Add a boundary assertion that existing source selection still truncates the source excerpt to 18,000 characters before service-prompt byte validation; inputs still exceeding the UTF-8 budget reject and are never truncated again.
- [ ] Run `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/StudioPane/__tests__/audio-overview-service-prompt.test.ts ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx && cd ../.. && source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Chat/test_service_prompt_execution.py -k research_studio_audio`; expect the hardcoded in-hook messages and absent server golden to fail.
- [ ] Extract only a typed audio-overview execution-request builder from the large hook; do not extract or preserve prompt assembly and do not generalize unrelated artifact prompts.
- [ ] In `generateAudioOverview`, perform existing source selection/truncation, pass `source_text` through the typed request, and let authenticated `chat.py` resolve/render once. Retain non-prompt provider/model/TTS options unchanged.
- [ ] Rerun the focused Vitest/pytest command; expect request-only transport, exact server provider bytes, and size handling to pass.
- [ ] Run the mandatory commit/frontend gates plus this exact Task 5 Python test scope; update Backlog:

```bash
source .venv/bin/activate
TASK_PYTHON_TEST_SCOPE=(tldw_Server_API/tests/Chat/test_service_prompt_execution.py)
python -m black --check "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m ruff check "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m mypy "${TASK_PYTHON_TEST_SCOPE[@]}"
```
- [ ] Commit: `feat: migrate research audio service prompt (TASK-12957)`.

## Task 6: Resolve slide source summarization once per presentation

**Files:**

- Modify: `tldw_Server_API/app/core/Slides/slides_generator.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/slides.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/slides_module.py`
- Modify: `tldw_Server_API/tests/Slides/test_slides_generator.py`
- Modify: `tldw_Server_API/tests/Slides/test_slides_api.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_slides_module.py`

- [ ] Add failing generator tests that `slides.source.summary` resolves once before `_chunk_and_summarize`, even for multiple chunks; each chunk renders from that same frozen bundle; and current system/user bytes remain exact.
- [ ] Cover braces/Unicode, 65,536/65,537 rendered source-chunk bytes, no provider call on rejection, and unchanged deterministic chunk splitting capped by `SLIDES_MAX_SOURCE_CHUNKS` (default 20). The cap is input shaping, not post-render truncation.
- [ ] Add failing API and MCP tests that authenticated owner/execution context reaches the generator, approved-user precedence works, safe provenance contains no body, and missing owner context fails closed rather than falling back globally.
- [ ] Run `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Slides/test_slides_generator.py tldw_Server_API/tests/Slides/test_slides_api.py tldw_Server_API/app/core/MCP_unified/tests/test_slides_module.py`; expect resolution-count/context tests to fail.
- [ ] Add a `PromptExecutionContext`/resolved-bundle input at the SlidesGenerator orchestration boundary. Resolve before `_chunk_and_summarize`; render only the constrained `source_chunk` per iteration with the shared renderer.
- [ ] Thread the authenticated principal from the FastAPI endpoint and the authenticated MCP tool execution context. Do not let the generator query an ambient/global user or resolve again per chunk.
- [ ] Preserve all non-summary slide prompts and generation behavior unchanged.
- [ ] Rerun the focused pytest command; expect all byte, count, owner, limit, and chunk-cap tests to pass.
- [ ] Run the mandatory commit gate plus this exact Task 6 Python scope; update Backlog:

```bash
source .venv/bin/activate
TASK_PYTHON_SOURCE_SCOPE=(
  tldw_Server_API/app/core/Slides/slides_generator.py
  tldw_Server_API/app/api/v1/endpoints/slides.py
  tldw_Server_API/app/core/MCP_unified/modules/implementations/slides_module.py
)
TASK_PYTHON_TEST_SCOPE=(
  tldw_Server_API/tests/Slides/test_slides_generator.py
  tldw_Server_API/tests/Slides/test_slides_api.py
  tldw_Server_API/app/core/MCP_unified/tests/test_slides_module.py
)
python -m black --check "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m ruff check "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m mypy "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
```
- [ ] Commit: `feat: migrate slide summary service prompt (TASK-12957)`.

## Task 7: Migrate every synchronous audio-analysis boundary without ownerless fallback

**Files:**

- Modify: `tldw_Server_API/app/api/v1/schemas/media_request_models.py`
- Modify: `tldw_Server_API/app/api/v1/API_Deps/media_processing_deps.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_audios.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/audio_batch.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Files.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
- Create: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_service_prompts.py`
- Create: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_process_audios_service_prompts.py`
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_batch_counts.py`
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_process_batch_media_precheck_regressions.py`

- [ ] Add failing `/api/v1/media/process-audios` tests proving the endpoint obtains `current_user` through `get_request_user`, derives the canonical owner from `current_user.id` rather than the Media DB handle, and resolves `media.audio.analysis` exactly once only when `perform_analysis` is true and `api_name` is nonblank/non-`none`. Missing authentication, unavailable owner state, quarantined/store failure, or resolution failure must return the shared sanitized error and must not call `run_audio_batch`; no ownerless/global/default fallback is allowed.
- [ ] Add two-user endpoint cases with distinct approved revisions plus deployment/package controls. Capture the bundle passed to `run_audio_batch` and prove explicit truthy `custom_prompt`/`system_prompt` parts are highest, empty strings are omitted/default, literal braces remain literal, safe provenance contains only IDs/source kinds/digests, and one user's revision can never reach the other request.
- [ ] Extend `test_audio_batch_counts.py` with failing caller-contract tests: `run_audio_batch` accepts the one immutable resolved bundle, forwards the same object identity to `process_audio_files`, removes raw `custom_prompt_input`/`system_prompt_input` from the lower call, performs no resolver/store access, requires a bundle for active analysis, and permits `None` only when analysis/provider selection means no model dispatch.
- [ ] Add complete provider-message Goldens in `test_audio_service_prompts.py` for the endpoint/caller path and the `process_batch_media` path: packaged, deployment, approved-user, and explicit truthy selections must all preserve the exact system role plus transcript/chunk → four-newline separator → user instruction bytes. Use braces, Unicode, multi-item/chunk input, and the model-download placeholder branch; assert one resolved object feeds every provider call.
- [ ] Add 65,536/65,537 UTF-8 endpoint cases for each reachable explicit/runtime part. Override only `ProcessAudiosForm`/`get_process_audios_form` limits needed to express the approved byte boundary without weakening unrelated `AddMediaForm` fields; 65,537 returns 413 before `run_audio_batch` or `analyze`. Keep the exact 262,144/262,145 aggregate fixture in Tasks 2/9, and add an endpoint adapter test proving the same aggregate exception maps to 413 without pretending this three-text-part definition can naturally reach 256 KiB while every part remains within 64 KiB.
- [ ] Run `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/MediaIngestion_NEW/unit/test_process_audios_service_prompts.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_batch_counts.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_service_prompts.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_process_batch_media_precheck_regressions.py`; expect missing authenticated endpoint resolution and one-bundle caller plumbing failures.
- [ ] Add `current_user: User = Depends(get_request_user)` to `process_audios_endpoint`. After validated inputs exist and before the normal `run_audio_batch` call, build one `PromptExecutionContext` with canonical owner/workflow/request metadata and truthy explicit parts, resolve once, and pass the frozen bundle to `run_audio_batch`. File-rejection/no-input branches never resolve because they cannot dispatch analysis.
- [ ] Add a typed `resolved_service_prompt` parameter to `run_audio_batch`, `process_batch_media` audio dispatch, and `process_audio_files`. `run_audio_batch` and `process_batch_media` only forward a caller-resolved or WorkerSDK-verified bundle; `process_audio_files` renders runtime transcript/chunk variables from that bundle and never queries current user, user store, deployment source, prompt loader, global defaults, or Job pin state.
- [ ] Require the bundle whenever analysis can dispatch. A trusted userless maintenance caller must deliberately create a server-default `PromptExecutionContext` at its own public boundary; absence of owner/bundle is never interpreted as permission to select a global fallback.
- [ ] Replace both model-download placeholder and final analysis prompt loading with the same bundle. Preserve analyzer provider/model arguments, recursive/chunked behavior, locked carrier, warnings, and result shapes; remove the migrated `load_prompt` fallback from these execution branches only after Goldens pass.
- [ ] Rerun the focused pytest command; expect authenticated ownership, precedence, exact provider bytes, single-resolution/object-identity, fail-closed, and size tests to pass.
- [ ] Run the mandatory commit gate plus this exact Task 7 Python scope; update Backlog:

```bash
source .venv/bin/activate
TASK_PYTHON_SOURCE_SCOPE=(
  tldw_Server_API/app/api/v1/schemas/media_request_models.py
  tldw_Server_API/app/api/v1/API_Deps/media_processing_deps.py
  tldw_Server_API/app/api/v1/endpoints/media/process_audios.py
  tldw_Server_API/app/core/Ingestion_Media_Processing/audio_batch.py
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Files.py
  tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py
)
TASK_PYTHON_TEST_SCOPE=(
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_service_prompts.py
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_process_audios_service_prompts.py
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_batch_counts.py
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_process_batch_media_precheck_regressions.py
)
python -m black --check "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m ruff check "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m mypy "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
```
- [ ] Commit: `feat: migrate media audio analysis prompt (TASK-12957)`.

## Task 8: Pin audio-analysis prompts at Media-ingest enqueue and verify before dispatch

**Files:**

- Modify: `tldw_Server_API/app/api/v1/schemas/media_request_models.py`
- Modify: `tldw_Server_API/app/api/v1/API_Deps/media_add_deps.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py`
- Modify: `tldw_Server_API/app/services/media_ingest_jobs_worker.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py`
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py`
- Create: `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_audio_service_prompt_jobs.py`

- [ ] Add failing endpoint tests for the finite requirement declaration: exactly `{media.audio.analysis}` only for audio input with an active analysis provider; all other Media-ingest combinations use the unchanged ordinary Jobs path.
- [ ] Add failing enqueue tests for protected full-bundle pin-set commit → held-job creation → authenticated bind → queued release, all-or-nothing failure, idempotent retry, owner matching, and prompt-related payload data containing only the foundation's pin-set UUID/submission ID/set digest references. Assert submitted literal system/user bodies and unmistakable body sentinels are absent from ordinary payload JSON, events, status, and logs.
- [ ] Add failing API-preflight tests for truthy explicit fields at 65,536/65,537 UTF-8 bytes; return 413 before creating a held job. Keep unrelated legacy schema character limits unchanged until their owning domain migrates.
- [ ] Add failing worker/integration tests for valid verified context, missing/tampered/wrong-owner/wrong-contract/wrong-selection pin, transient protected-store unavailability, expired authentication retention, and operator `bypass_stored_overrides`. The handler/provider must not run on any invalid pin; transient failures retry; integrity failures quarantine; bypass holds without substitution.
- [ ] Add a time-of-use golden: enqueue with an approved revision, then edit/reset it; the worker still emits the originally authenticated provider bytes. Cover request-bound literal overrides and literal braces in the same way.
- [ ] Run `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_audio_service_prompt_jobs.py`; expect pinning, payload-redaction, and verification tests to fail.
- [ ] Route the prompt-bearing branch through `ServicePromptJobPinner.enqueue(...)`; pass the finite declaration and explicit truthy parts to the pinner and let it resolve once. Leave `_create_media_ingest_job` unchanged for non-prompt-bearing jobs.
- [ ] Remove raw `custom_prompt`/`system_prompt` values from the normal queued `options` payload for this branch. Snapshot templates/assembly metadata and request-bound literal parts only in the owner-scoped protected store as defined by plan 05.
- [ ] Adapt the Media worker's WorkerSDK handler closure to accept the verified immutable prompt context, require `media.audio.analysis`, and pass that bundle through `process_batch_media` to audio dispatch. Never reconstruct or resolve from mutable payload fields.
- [ ] Keep all pin verification in WorkerSDK. The Media worker adds only the exact requirement assertion and domain plumbing; it does not verify MACs/digests itself.
- [ ] Rerun the focused pytest command; expect held-bind-release, immutability, error classification, and provider-byte tests to pass.
- [ ] Run the mandatory commit gate plus this exact Task 8 Python scope; update Backlog:

```bash
source .venv/bin/activate
TASK_PYTHON_SOURCE_SCOPE=(
  tldw_Server_API/app/api/v1/schemas/media_request_models.py
  tldw_Server_API/app/api/v1/API_Deps/media_add_deps.py
  tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py
  tldw_Server_API/app/services/media_ingest_jobs_worker.py
  tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py
)
TASK_PYTHON_TEST_SCOPE=(
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_audio_service_prompt_jobs.py
)
python -m black --check "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m ruff check "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m mypy "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
```
- [ ] Commit: `feat: pin audio analysis prompts for ingest jobs (TASK-12957)`.

## Task 9: Add the real domain canary and cross-surface regression matrix

**Files:**

- Modify: `apps/tldw-frontend/e2e/workflows/service-prompts-settings.spec.ts`
- Modify: `tldw_Server_API/tests/Service_Prompts/test_runtime_summarization_media_audio.py`
- Modify: `tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py`
- Modify: `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_audio_service_prompt_jobs.py`

- [ ] Add a failing real-server Playwright canary using the existing mock LLM/provider harness: edit and approve `media.analysis.critical` in WebUI, run the Analysis workflow, and assert exact provider messages; edit the same account from the extension-options hash route and confirm WebUI sees the new pending revision.
- [ ] Keep raw prompt bodies out of screenshots, traces, console output, and fixture names. Use digests/sentinels only where provenance is asserted.
- [ ] Add a backend parameterized matrix over all sixteen IDs proving precedence order, reset/edit behavior, explicit semantics by boundary, literal braces, exact source kind/digests, and no raw body in safe provenance.
- [ ] Add a full size matrix covering exactly-allowed/one-byte-over authored part, authored definition, expanded variable/rendered part, and final bundle across both HTTP and non-HTTP entry points. Assert failures precede all provider/job side effects.
- [ ] Run `cd apps/tldw-frontend && bunx playwright test e2e/workflows/service-prompts-settings.spec.ts --reporter=line`; expect the new domain canary to fail before fixtures/workflow wiring are complete.
- [ ] Run `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Service_Prompts/test_runtime_summarization_media_audio.py tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py tldw_Server_API/tests/Chat/test_service_prompt_execution.py tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_audio_service_prompt_jobs.py`; expect the new cross-domain matrix failures, then make only fixture/wiring corrections needed for them.
- [ ] Rerun both focused commands; expect exact provider bytes and all boundary cases to pass.
- [ ] Run the mandatory commit/frontend gates, Playwright trace/screenshot review, and this exact Task 9 Python test scope; update Backlog:

```bash
source .venv/bin/activate
TASK_PYTHON_TEST_SCOPE=(
  tldw_Server_API/tests/Service_Prompts/test_runtime_summarization_media_audio.py
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py
  tldw_Server_API/tests/Chat/test_service_prompt_execution.py
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_audio_service_prompt_jobs.py
)
python -m black --check "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m ruff check "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m mypy "${TASK_PYTHON_TEST_SCOPE[@]}"
```
- [ ] Commit: `test: verify summarization media prompt rollout (TASK-12957)`.

## Task 10: Document, secure, and complete the rollout

**Files:**

- Modify: `Docs/Design/service-prompt-inventory.md`
- Modify: `Docs/API/service-prompts.md`
- Modify: `tldw_Server_API/Config_Files/Prompts/README.md`
- Update through Backlog MCP/CLI: `TASK-12957`

- [ ] Update the inventory only with implemented migration evidence, source/test call paths, and completed status for the exact sixteen IDs. Do not change approved IDs, contracts, precedence, or eligibility decisions.
- [ ] Immediately after the inventory edit, run `node Helper_Scripts/validate_service_prompt_inventory.mjs .` from the repository root. A nonzero exit blocks documentation closeout and commit; record the JSON counts/reference results in TASK-12957.
- [ ] Document multipart browser execution through the authenticated chat bridge, authenticated `/process-audios` → `run_audio_batch` → `process_audio_files` one-bundle execution, audio Jobs pinning, exact explicit-empty/truthy behavior, stable 413/non-HTTP size errors, safe provenance, existing deterministic truncation/chunking, and fail-closed unsupported-server/owner behavior.
- [ ] Document the packaged YAML plus `media.audio.analysis` compatibility mapping so operators know which deployment/env sources remain authoritative.
- [ ] Run the complete focused backend suite:

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Service_Prompts/test_registry_summarization_media_audio.py \
  tldw_Server_API/tests/Service_Prompts/test_runtime_summarization_media_audio.py \
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py \
  tldw_Server_API/tests/Chat/test_service_prompt_execution.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_process_audios_service_prompts.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_batch_counts.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_service_prompts.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_process_batch_media_precheck_regressions.py \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_audio_service_prompt_jobs.py \
  tldw_Server_API/tests/Slides/test_slides_generator.py \
  tldw_Server_API/tests/Slides/test_slides_api.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_slides_module.py
```

- [ ] Run the complete focused frontend suite from `apps/tldw-frontend`:

```bash
bunx vitest run \
  ../packages/ui/src/services/__tests__/service-prompts.test.ts \
  ../packages/ui/src/services/__tests__/tldw-chat.message-sanitization.test.ts \
  ../packages/ui/src/components/Media/__tests__/AnalysisModal.stage3.regression.test.tsx \
  ../packages/ui/src/components/Review/__tests__/ReviewPage.service-prompts.test.tsx \
  ../packages/ui/src/utils/__tests__/content-review-ai.service-prompts.test.ts \
  ../packages/ui/src/components/ContentReview/__tests__/ContentReviewPage.service-prompts.test.tsx \
  ../packages/ui/src/components/Option/ResearchWorkspace/StudioPane/__tests__/audio-overview-service-prompt.test.ts \
  ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx
bunx playwright test e2e/workflows/service-prompts-settings.spec.ts --reporter=line
```

- [ ] Run Bandit from the project virtual environment and review every new finding:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Service_Prompts \
  tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/chat.py \
  tldw_Server_API/app/api/v1/API_Deps/media_processing_deps.py \
  tldw_Server_API/app/api/v1/endpoints/media/process_audios.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/audio_batch.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Files.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py \
  tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py \
  tldw_Server_API/app/api/v1/schemas/media_request_models.py \
  tldw_Server_API/app/api/v1/API_Deps/media_add_deps.py \
  tldw_Server_API/app/services/media_ingest_jobs_worker.py \
  tldw_Server_API/app/core/Slides/slides_generator.py \
  tldw_Server_API/app/api/v1/endpoints/slides.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/slides_module.py \
  -f json -o /tmp/bandit_task_12113_1.json
```

- [ ] From the repository root, run the exact no-waiver backend and diff gate:

```bash
source .venv/bin/activate
python -m pytest -v
git diff --check
```

- [ ] Run the exact Python source format/lint/type gate:

```bash
source .venv/bin/activate
PYTHON_SCOPE=(
  tldw_Server_API/app/core/Service_Prompts/models.py
  tldw_Server_API/app/core/Service_Prompts/registry.py
  tldw_Server_API/app/core/Service_Prompts/templates.py
  tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py
  tldw_Server_API/app/api/v1/endpoints/chat.py
  tldw_Server_API/app/api/v1/schemas/media_request_models.py
  tldw_Server_API/app/api/v1/API_Deps/media_processing_deps.py
  tldw_Server_API/app/api/v1/API_Deps/media_add_deps.py
  tldw_Server_API/app/api/v1/endpoints/media/process_audios.py
  tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py
  tldw_Server_API/app/core/Ingestion_Media_Processing/audio_batch.py
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Files.py
  tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py
  tldw_Server_API/app/services/media_ingest_jobs_worker.py
  tldw_Server_API/app/core/Slides/slides_generator.py
  tldw_Server_API/app/api/v1/endpoints/slides.py
  tldw_Server_API/app/core/MCP_unified/modules/implementations/slides_module.py
)
PYTHON_TEST_SCOPE=(
  tldw_Server_API/app/core/MCP_unified/tests/test_slides_module.py
  tldw_Server_API/tests/Chat/test_service_prompt_execution.py
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_audio_service_prompt_jobs.py
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_batch_counts.py
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_service_prompts.py
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_process_audios_service_prompts.py
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_process_batch_media_precheck_regressions.py
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py
  tldw_Server_API/tests/Service_Prompts/test_registry_summarization_media_audio.py
  tldw_Server_API/tests/Service_Prompts/test_runtime_summarization_media_audio.py
  tldw_Server_API/tests/Slides/test_slides_api.py
  tldw_Server_API/tests/Slides/test_slides_generator.py
)
python -m black --check "${PYTHON_SCOPE[@]}" "${PYTHON_TEST_SCOPE[@]}"
python -m ruff check "${PYTHON_SCOPE[@]}" "${PYTHON_TEST_SCOPE[@]}"
python -m mypy "${PYTHON_SCOPE[@]}" "${PYTHON_TEST_SCOPE[@]}"
```

- [ ] Run the exact no-waiver frontend/extension/format/lint/type/build gate, then return to the repository root:

```bash
cd apps/tldw-frontend
bun run test:run
bunx vitest run -c vitest.extension.config.ts
cd ../packages/ui
bun run test
cd ../../tldw-frontend
bun run format:check
bun run lint
bunx tsc --noEmit -p ../packages/ui/tsconfig.json
bun run build
cd ../..
```

Full CI shards and any command above may not be waived by focused success. Diagnose failures under the three-attempt rule, record them in TASK-12957, and stop if the gate cannot be made green.
- [ ] Self-review the final diff for exact sixteen-ID scope, once-only server resolution, request-only browser transport, no effective/locked prompt bodies in TypeScript or ordinary Jobs payloads/telemetry, no new TODO without an issue, and unchanged no-override provider bytes.
- [ ] Update `TASK-12957` with plan link, final touched-file list, focused and full verification output, Bandit JSON path/findings, blockers (if any), commit hashes, and final summary.
- [ ] Commit: `docs: document summarization media prompt rollout (TASK-12957)`.

## Definition of done

- All and only the sixteen scoped IDs are available through the shared catalog and execute through their declared boundaries.
- Every boundary resolves once, honors the exact precedence and explicitness contract, and passes an immutable bundle downward.
- `/process-audios` derives ownership from authenticated `current_user`, resolves once before `run_audio_batch`, and passes the same immutable bundle through `process_audio_files`; missing ownership or bundle can never select a global fallback.
- Provider-message goldens are byte-equivalent without overrides and intentional edits change only the declared editable parts.
- All exact byte limits pass/fail at the required edge, with 413 or the one canonical `service_prompt_size_limit_exceeded` code and no downstream side effect.
- Prompt-bearing Media audio Jobs are fully pinned before queue release, contain no raw prompt body in ordinary payloads, and are verified by WorkerSDK before domain dispatch.
- Existing 18,000-character Research Studio shaping and Slides maximum-20 chunk shaping remain deterministic; no new silent truncation exists.
- Safe provenance, logs, statuses, events, E2E artifacts, and Backlog evidence expose no raw prompt or source bodies.
- Focused tests, mandatory full pytest/frontend/extension shards, Playwright canary, formatter/linter/type/build checks, Bandit, and `git diff --check` are green.
