# Documents and Web Service Prompts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate exactly the 32 inventory-approved document, web, note-title, image-refinement, browser-workflow, and writing prompt definitions to the shared Service Prompts system without changing no-override provider-message bytes, legacy explicit-override semantics, output validation, or Jobs integrity.

**Architecture:** Register the 32 atomic contracts in the existing `ServicePromptRegistry`. Synchronous Python consumers resolve one immutable bundle from the authenticated owner at the inventory boundary and render only runtime variables below it. Browser consumers reuse TASK-12957's optional authenticated `service_prompt` extension on `POST /api/v1/chat/completions`; TypeScript sends only definition IDs, runtime variables, finite selectors, permitted history/carriers, and legacy explicit named parts. The scene-annotation producer reuses protected held-bind-release pinning and the worker consumes only WorkerSDK-verified context. No domain-specific resolver, persistence layer, template language, execution endpoint, or client-side prompt resolver is introduced.

**Tech Stack:** FastAPI/Pydantic, shared `Service_Prompts` registry/resolver/templates/service, existing Chat completion bridge, Jobs `ServicePromptJobPinner` and `WorkerSDK`, React/TypeScript shared UI package, pytest/Hypothesis where useful, Vitest/Testing Library, Playwright, Bandit.

---

## Work record, prerequisites, and stop conditions

- Backlog task: `TASK-12959` (`Migrate document and web service prompts`). Use the official Backlog MCP/CLI workflow; do not edit the task file manually.
- Complete TASK-12957 and foundation plans 02–06 first. In particular, TASK-12957 must have landed the optional authenticated chat-execution bridge in:
  - `tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py`
  - `tldw_Server_API/app/api/v1/endpoints/chat.py`
  - `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`
  - `apps/packages/ui/src/services/service-prompts.ts`
  - `apps/packages/ui/src/services/__tests__/service-prompts.test.ts`
- Before browser work, verify that TASK-12957's landed bridge owns and tests all four registry-selected message-merge policies. The request may not select a policy; the registry definition does:
  1. `replace_generated_messages`: replace the complete generated provider-message array for standalone workflows; an empty incoming `messages` array is valid only with a Service Prompt definition declaring this policy.
  2. `replace_copilot_user_text_preserve_non_text`: replace only the copilot user's text parts while preserving authenticated non-text/image parts in their existing order.
  3. `insert_before_current_user`: preserve permitted history and insert the resolved client-web-search context immediately before the current user message.
  4. `prepend_system_preserve_history`: prepend the resolved system message before permitted agent/continuation history without rewriting the preserved history/current user messages.
- Also verify that `buildServicePromptExecution(...)` accepts only a definition ID, declared variables, named explicit parts with `literal|template` kind, and an allowlisted finite selector; that `chat.py` derives ownership from `AuthPrincipal`, resolves exactly once, applies the registry-owned policy, returns no prompt body, and maps shared size failures to 413. If any model, policy, validation, ownership, or error invariant is missing, stop TASK-12959 and amend TASK-12957's plan, Backlog record, implementation, and bridge tests until it is complete and green. TASK-12959 only assigns its definitions to the landed policies and transports their typed requests; it must not add or modify bridge merge behavior, schema semantics, or a parallel endpoint.
- Reuse `ServicePromptRegistry`, `PromptExecutionContext`, `ServicePromptService`, the shared constrained renderer and budgets, `ServicePromptJobPinner.enqueue(...)`, and WorkerSDK's verified immutable prompt context. Do not import settings/detail prompt bodies into the browser or implement precedence, rendering, locked assembly, or pin verification in this domain.
- Stop if any Python boundary cannot receive the authenticated owner, approved-revision source, or verified job context. Never use an ambient/global user or silently fall back across users.
- Stop if a no-override provider-message Golden differs in role, order, whitespace, delimiter, truncation, or byte content. Record the exact diff in TASK-12959 instead of normalizing it.
- Stop if `writing.annotation.scene` cannot declare the finite set exactly `{writing.annotation.scene}` at enqueue or cannot receive the verified pin through WorkerSDK before provider dispatch.
- Core Scheduler definitions remain deferred. Do not register, resolve, or add compatibility aliases for any Scheduler-reachable definition in this task.
- The requester-authorized planning-only CI skip applies only while authoring this document. It does not waive any implementation test, full CI shard, Bandit scan, or commit gate below.

## Exact scope lock: 32 definitions

Register and migrate these IDs—no more and no fewer:

| # | Service prompt ID | Inventory boundary and atomic contract |
|---:|---|---|
| 1 | `chat.document.briefing` | `system` (editable literal) ⇒ `user_instruction` (editable literal) → optional locked focus block → locked conversation context |
| 2 | `chat.document.meetingnotes` | same chat-document assembly; document type `meeting_notes` |
| 3 | `chat.document.qa` | same chat-document assembly; document type `q_and_a` |
| 4 | `chat.document.studyguide` | same chat-document assembly; document type `study_guide` |
| 5 | `chat.document.summary` | same chat-document assembly; document type `summary` |
| 6 | `chat.document.timeline` | same chat-document assembly; document type `timeline` |
| 7 | `documents.copilot.explain` | one editable constrained `user_template`; locked single-user-message/multimodal carrier |
| 8 | `documents.copilot.rephrase` | one editable constrained `user_template`; same carrier |
| 9 | `documents.copilot.summary` | one editable constrained `user_template`; same carrier |
| 10 | `documents.copilot.translate` | one editable constrained `user_template`; same carrier |
| 11 | `image.prompt.refinement` | editable literal `system_semantics` ⇒ locked prompt-mode/backend/original/context carriers → editable literal `rewrite_semantics` |
| 12 | `media.document.insights` | editable literal system semantics → locked JSON contract ⇒ editable literal user semantics → locked document/category carriers |
| 13 | `media.document.summary` | editable literal `system` ⇒ locked document/chunk carrier → locked four-newline separator → editable literal user instruction |
| 14 | `media.text.translation` | editable literal `system` ⇒ editable constrained `user_template` |
| 15 | `notes.title.generate` | editable literal system/title semantics plus locked length-only-title/content carriers |
| 16 | `web.search.client.answer` | one editable constrained `system_template`; registry policy `insert_before_current_user` places the rendered provider `system` message after permitted history/actor messages and immediately before the current provider `user` message |
| 17 | `web.search.snippet.digest` | hidden locked analyzer system ⇒ locked extracted snippets → locked four-newline separator → editable constrained digest template |
| 18 | `workflow.book.analysis.chapter` | locked base system → editable literal style fragment ⇒ locked content carrier |
| 19 | `workflow.book.analysis.characters` | same book assembly; character-analysis preset |
| 20 | `workflow.book.analysis.comprehensive` | same book assembly; comprehensive preset |
| 21 | `workflow.book.analysis.concepts` | same book assembly; key-concepts preset |
| 22 | `workflow.web.summary.brief` | locked base system → editable literal style fragment ⇒ locked title/URL/content carrier |
| 23 | `workflow.web.summary.bullets` | same page assembly; bullets style |
| 24 | `workflow.web.summary.detailed` | same page assembly; detailed style |
| 25 | `writing.agent.brainstorm` | editable literal system → optional locked manuscript context ⇒ permitted existing history ⇒ current user message |
| 26 | `writing.agent.planning` | same agent assembly; planning mode |
| 27 | `writing.agent.quick` | same agent assembly; quick mode |
| 28 | `writing.annotation.scene` | editable system semantics → locked JSON schema ⇒ editable review semantics → locked scene-anchor carrier; protected Jobs path |
| 29 | `writing.annotation.selection` | editable system semantics → locked JSON schema ⇒ editable review semantics → locked selected-text anchor carrier; direct request |
| 30 | `writing.continuation.fill` | editable literal system ⇒ permitted authenticated prefix/suffix/context message carrier |
| 31 | `writing.continuation.predict` | editable literal system ⇒ permitted authenticated prefix/context message carrier |
| 32 | `writing.feedback.echo` | finite selector chooses one of five atomically versioned editable persona systems ⇒ locked passage carrier |

Do not register adjacent/deferred IDs such as `writing.feedback.mood`, client RAG, generic web reports, recursive document workflows, title-generation/report definitions owned by later domains, or any Scheduler definition.

## Non-negotiable compatibility contract

1. Resolution precedence is exactly: explicit request override → authenticated job pin → approved user revision → deployment provider → packaged default. Active stored revisions remain atomic; only explicit request data may replace a declared subset.
2. Resolve once at the matrix boundary. Render runtime variables from that frozen bundle in lower loops; never re-read the prompt store, deployment assets, or pin store per chunk, chapter, scene, or provider retry.
3. Preserve literal/template semantics exactly. Explicit literal braces remain bytes; explicit constrained templates render only declared placeholders with their declared repetition counts. Unknown variables, parts, kinds, selectors, or locked-part overrides fail before provider dispatch.
4. Preserve explicitness exactly:
   - Chat-document saved `system_prompt` and `user_prompt` rows are presence-based literals, including `""`; request `custom_prompt` is truthy-only and replaces only `user_instruction`.
   - Copilot local storage is presence-based: any stored string, including `""`, is an explicit constrained `user_template`; a missing/non-string key means no explicit part.
   - Document-summary `custom_prompt_input` and `system_prompt_input` remain truthy-only literal overrides.
   - Client web-search storage remains truthy-only; missing or empty means no explicit template.
   - Book `custom` selection supplies the exact custom string, including `""`, as a literal `style_fragment`; named presets supply no explicit part.
   - Writing continuation chat mode still bypasses these definitions and keeps its existing explicit chat context.
5. Preserve selectors exactly. Echo accepts only `alex|sam|max|riley|jordan`; all five systems version together. Mode/style/preset-to-definition mappings remain finite and are not user-provided definition IDs.
6. Browser requests use the four TASK-12957 registry-declared policies above only. The client cannot choose roles or insertion positions. TASK-12959 maps each of its definitions to one already-landed policy but does not implement, extend, or branch bridge merge behavior.
7. With no override, compare complete provider-message arrays byte-for-byte. This includes system-message placement, blank lines, four-newline analyzer separators, indentation in legacy web digest templates, focus/context framing, multimodal non-text parts, and all current provider options.
8. Safe provenance contains only definition ID, source kind, contract version, revision/snapshot identifiers, and digests. Responses, logs, errors, audits, metrics, Jobs events/status, Playwright traces, and screenshots contain no prompt body, rendered document/search/scene content, MAC, or key.
9. Output enforcement remains locked: document insights retains JSON response mode and `_normalize_insights`; manuscript annotation parsing/range/schema/dedupe remains unchanged; note-title normalization remains unchanged; image refinement cleanup remains unchanged. Note-title heuristic fallback remains only for the pre-existing provider failure, provider-configuration failure, or empty normalized provider output cases. Service Prompt unavailable, quarantined, validation, or size failures fail closed through the shared typed mapper; they never invoke the heuristic, dispatch a provider after failure, or persist a note/title.
10. Ordinary Jobs payloads never contain raw prompt bodies or rendered scene text. `writing.annotation.scene` may retain its existing non-prompt IDs, provider/model, bounded filters, and focus metadata, plus only the foundation pin-set UUID/submission ID/set digest references.

## Exact UTF-8 budget and error contract

- An authored text part of exactly 65,536 UTF-8 bytes succeeds; 65,537 fails.
- An expanded variable/rendered text part of exactly 65,536 UTF-8 bytes succeeds; 65,537 fails.
- An authored definition of exactly 262,144 UTF-8 bytes succeeds; 262,145 fails.
- A final rendered bundle of exactly 262,144 UTF-8 bytes succeeds; 262,145 fails.
- Settings/preview and chat execution return HTTP 413 before provider dispatch. Direct non-HTTP execution raises the foundation's one exact typed code: `service_prompt_size_limit_exceeded`.
- Use multibyte fixtures so tests count UTF-8 bytes, not Python/JavaScript characters. Final-bundle tests must subtract the exact locked bytes before padding the editable part.
- Exercise the aggregate limit with a synthetic in-test five-part definition, never a product registry entry: four 65,000-byte parts, one 2,132-byte part, and four explicit three-byte `"\n--"` assembly separators total `4 * 65_000 + 2_132 + 4 * 3 = 262_144` bytes. Increasing only the fifth part to 2,133 bytes yields 262,145. Every authored/rendered part remains at or below 65,536; assert the renderer counts the 12 assembly bytes and rejects only the one-byte-over aggregate.
- Never truncate to satisfy these budgets. Preserve only matrix-documented shaping already performed before `G` validation: chat context's existing cap/framing; image original prompt `C1200` and at most four preassembled context cues; insights document `C50000`; translation text `C10000`; note snippet `C2000`; legacy web-search chunk construction/truncation; book content `C30000`; webpage content `C10000`; writing-agent scene `C2000` and ten character/world rows each; echo passage `C1000`. All other overflow rejects.

## Authoritative implementation file map

### Definitions, lifecycle limits, and shared bridge references

- Create: `tldw_Server_API/Config_Files/Prompts/service_prompts_documents_web.prompts.yaml`
- Modify: `tldw_Server_API/app/core/Service_Prompts/registry.py`
- Create: `tldw_Server_API/tests/Service_Prompts/test_registry_documents_web.py`
- Create: `tldw_Server_API/tests/Service_Prompts/test_runtime_documents_web.py`
- Modify: `tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py`
- Reference prerequisite only: `tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py`
- Reference prerequisite only: `tldw_Server_API/app/api/v1/endpoints/chat.py`
- Modify with domain Goldens only: `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`
- Reference prerequisite only: `apps/packages/ui/src/services/service-prompts.ts`
- Modify with domain request fixtures only: `apps/packages/ui/src/services/__tests__/service-prompts.test.ts`

Keep compatibility providers for the existing `chat.prompts.yaml` chat-document keys, `document.prompts.yaml` summary keys, and `summarization.prompts.yaml` analyzer system. Do not copy those defaults into the new asset. The new asset owns only currently hard-coded defaults after their consumers migrate.

### Shared TypeScript chat transport used by this domain

- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwChat.ts`
- Modify: `apps/packages/ui/src/models/ChatTldw.ts`
- Modify: `apps/packages/ui/src/models/index.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts`
- Create: `apps/packages/ui/src/services/__tests__/tldw-chat.service-prompts.test.ts`

These files transport TASK-12957's existing typed object unchanged. They do not resolve, render, fetch prompt detail, or implement assembly.

### Direct Python consumers

- Modify: `tldw_Server_API/app/core/Chat/document_generator.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat_documents.py`
- Modify: `tldw_Server_API/tests/Chat/unit/test_document_generator.py`
- Create: `tldw_Server_API/tests/Chat/unit/test_document_generator_service_prompts.py`
- Modify: `tldw_Server_API/tests/Chat/integration/test_document_generation_endpoints.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/document_insights.py`
- Modify: `tldw_Server_API/tests/Media/test_document_insights.py`
- Create: `tldw_Server_API/tests/Media/test_document_insights_service_prompts_api.py`
- Modify: `tldw_Server_API/app/services/document_processing_service.py`
- Modify: `tldw_Server_API/tests/Services/test_document_processing_service.py`
- Create: `tldw_Server_API/tests/Services/test_document_processing_service_prompts.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/translate.py`
- Create: `tldw_Server_API/tests/Translation/test_translate_service_prompts.py`
- Modify: `tldw_Server_API/tests/Translation/test_translate_endpoint_error_mapping.py`
- Modify: `tldw_Server_API/app/core/Writing/note_title.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/notes.py`
- Modify: `tldw_Server_API/tests/unit/test_note_title_lib.py`
- Create: `tldw_Server_API/tests/unit/test_note_title_service_prompts.py`
- Modify: `tldw_Server_API/tests/integration/test_notes_auto_title_api.py`
- Modify: `tldw_Server_API/app/core/WebSearch/Web_Search.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/research.py`
- Create: `tldw_Server_API/tests/WebSearch/test_websearch_service_prompts.py`
- Modify: `tldw_Server_API/tests/WebSearch/test_websearch_core.py`
- Modify: `tldw_Server_API/tests/WebSearch/unit/test_aggregate_results_schema.py`
- Modify: `tldw_Server_API/tests/WebSearch/integration/test_websearch_endpoint.py`

### Browser consumers

- Modify: `apps/packages/ui/src/services/application.ts`
- Modify: `apps/packages/ui/src/hooks/useMessage.tsx`
- Modify: `apps/packages/ui/src/services/__tests__/application.copilot-prompts.test.ts`
- Create: `apps/packages/ui/src/hooks/__tests__/useMessage.service-prompts.test.tsx`
- Modify: `apps/packages/ui/src/utils/image-prompt-refinement.ts`
- Modify: `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundImageGen.ts`
- Modify: `apps/packages/ui/src/utils/__tests__/image-prompt-refinement.test.ts`
- Modify: `apps/packages/ui/src/services/tldw-server.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/__tests__/normalChatMode.overlay.test.ts`
- Modify: `apps/packages/ui/src/components/Common/Workflow/steps/AnalyzeBookWorkflow.tsx`
- Create: `apps/packages/ui/src/components/Common/Workflow/__tests__/AnalyzeBookWorkflow.service-prompts.test.tsx`
- Modify: `apps/packages/ui/src/components/Common/Workflow/steps/SummarizePageWorkflow.tsx`
- Create: `apps/packages/ui/src/components/Common/Workflow/__tests__/SummarizePageWorkflow.service-prompts.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/AIAgentTab.tsx`
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/AIAgentTab.service-prompts.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/hooks/utils.ts`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx`
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.service-prompts.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingFeedback.ts`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/hooks/__tests__/useWritingFeedback.test.tsx`

### Writing annotations and protected Jobs

- Modify: `tldw_Server_API/app/api/v1/API_Deps/service_prompt_deps.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
- Modify: `tldw_Server_API/app/core/Writing/manuscript_annotations.py`
- Modify: `tldw_Server_API/app/core/Writing/manuscript_annotation_jobs.py`
- Modify: `tldw_Server_API/app/services/writing_annotation_review_jobs_worker.py`
- Modify: `tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py`
- Modify: `tldw_Server_API/tests/Writing/test_manuscript_annotation_review_jobs.py`
- Modify: `tldw_Server_API/tests/Services/test_writing_annotation_review_jobs_worker.py`

### Canary, documentation, and tracking

- Modify: `apps/tldw-frontend/e2e/workflows/service-prompts-settings.spec.ts`
- Modify: `Docs/Design/service-prompt-inventory.md`
- Modify: `Docs/API/service-prompts.md`
- Modify: `tldw_Server_API/Config_Files/Prompts/README.md`
- Update through Backlog MCP/CLI: `TASK-12959`

## Mandatory commit and final-domain gates

Before **every** commit below, run the complete backend and frontend commands in this section from the repository root. Focused tests never replace these gates, and a docs-only or backend-only commit does not waive the frontend shards/checks. For Python formatting, lint, and type checks, Stages 1–5B must use the exact task-local arrays printed in their work unit; those arrays contain only files that exist by that commit boundary. The complete-domain arrays in this section are reserved for Stage 5C, after every planned source and test exists.

```bash
source .venv/bin/activate
python -m pytest -v

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
git diff --check
```

At Stage 5C only, run the exact complete-domain Python format/lint/type scope after all paths exist:

```bash
source .venv/bin/activate

FINAL_PYTHON_CHECK_PATHS=(
  tldw_Server_API/app/core/Service_Prompts/registry.py
  tldw_Server_API/app/api/v1/API_Deps/service_prompt_deps.py
  tldw_Server_API/app/core/Chat/document_generator.py
  tldw_Server_API/app/api/v1/endpoints/chat_documents.py
  tldw_Server_API/app/api/v1/endpoints/media/document_insights.py
  tldw_Server_API/app/services/document_processing_service.py
  tldw_Server_API/app/api/v1/endpoints/translate.py
  tldw_Server_API/app/core/Writing/note_title.py
  tldw_Server_API/app/api/v1/endpoints/notes.py
  tldw_Server_API/app/core/WebSearch/Web_Search.py
  tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py
  tldw_Server_API/app/api/v1/endpoints/research.py
  tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py
  tldw_Server_API/app/core/Writing/manuscript_annotations.py
  tldw_Server_API/app/core/Writing/manuscript_annotation_jobs.py
  tldw_Server_API/app/services/writing_annotation_review_jobs_worker.py
)
FINAL_PYTHON_TEST_PATHS=(
  tldw_Server_API/tests/Chat/integration/test_document_generation_endpoints.py
  tldw_Server_API/tests/Chat/test_service_prompt_execution.py
  tldw_Server_API/tests/Chat/unit/test_document_generator.py
  tldw_Server_API/tests/Chat/unit/test_document_generator_service_prompts.py
  tldw_Server_API/tests/Media/test_document_insights.py
  tldw_Server_API/tests/Media/test_document_insights_service_prompts_api.py
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py
  tldw_Server_API/tests/Service_Prompts/test_registry_documents_web.py
  tldw_Server_API/tests/Service_Prompts/test_runtime_documents_web.py
  tldw_Server_API/tests/Services/test_document_processing_service.py
  tldw_Server_API/tests/Services/test_document_processing_service_prompts.py
  tldw_Server_API/tests/Services/test_writing_annotation_review_jobs_worker.py
  tldw_Server_API/tests/Translation/test_translate_endpoint_error_mapping.py
  tldw_Server_API/tests/Translation/test_translate_service_prompts.py
  tldw_Server_API/tests/WebSearch/integration/test_websearch_endpoint.py
  tldw_Server_API/tests/WebSearch/test_websearch_core.py
  tldw_Server_API/tests/WebSearch/test_websearch_service_prompts.py
  tldw_Server_API/tests/WebSearch/unit/test_aggregate_results_schema.py
  tldw_Server_API/tests/Writing/test_manuscript_annotation_review_jobs.py
  tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py
  tldw_Server_API/tests/integration/test_notes_auto_title_api.py
  tldw_Server_API/tests/unit/test_note_title_lib.py
  tldw_Server_API/tests/unit/test_note_title_service_prompts.py
)
python -m black --check "${FINAL_PYTHON_CHECK_PATHS[@]}" "${FINAL_PYTHON_TEST_PATHS[@]}"
python -m ruff check "${FINAL_PYTHON_CHECK_PATHS[@]}" "${FINAL_PYTHON_TEST_PATHS[@]}"
python -m mypy "${FINAL_PYTHON_CHECK_PATHS[@]}" "${FINAL_PYTHON_TEST_PATHS[@]}"
```

The final Python arrays are the exact complete backend source/test scope for this plan and run only in Stage 5C. Earlier commits use their exact task-local arrays while still running the complete backend and frontend shards above. The frontend sequence is the exact complete WebUI, extension, shared-package test, format, lint, package-UI TypeScript, and build gate. Update TASK-12959 through Backlog MCP/CLI with stage, touched files, red/green evidence, every command/result, and blockers. Do not commit around an unrelated or environmental failure: diagnose it under the repository's three-attempt rule, record the attempts, and stop if any command cannot be green.

## Stage 1: Register the exact contracts and packaged defaults

**Goal:** Add the 32 definitions and their trusted default/deployment mappings without changing a consumer.

**Success Criteria:** The domain contribution is exactly the scope-lock set; every part, role, order, visibility, template mode, selector, variable count, explicit rule, message policy, and safe sample matches the inventory; no adjacent ID appears.

**Tests:** Registry/catalog contracts, strict compatibility-provider behavior, default asset byte Goldens, hidden-part redaction, authored-limit edges.

**Status:** Not Started

**Files:**

- Create: `tldw_Server_API/Config_Files/Prompts/service_prompts_documents_web.prompts.yaml`
- Modify: `tldw_Server_API/app/core/Service_Prompts/registry.py`
- Create: `tldw_Server_API/tests/Service_Prompts/test_registry_documents_web.py`
- Modify: `tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py`

- [ ] Move TASK-12959 to In Progress through the official Backlog workflow and confirm TASK-12957 plus foundation plans 02–06 are complete and green.
- [ ] Write a failing registry test with a literal expected set of all 32 IDs. Assert count `32`, equality in both directions, and representative absence of `writing.feedback.mood`, Scheduler definitions, and later-domain IDs.
- [ ] Add a parameterized contract table for every row above: editable/locked and visible/hidden status, literal/constrained-template kind, provider role/order, declared variables and repetition, optional parts, finite selectors, chat message policy, existing source shaping, and settings metadata.
- [ ] Add no-override packaged-default fixtures captured from the current YAML/constants before deleting any consumer constant. Use braces, Unicode, delimiter-like text, empty optionals, and all selectors so accidental interpolation/normalization is visible.
- [ ] Add API cases proving visible editable/locked parts are present, hidden parts expose digest/presence only, and catalog/provenance never includes source paths or bodies.
- [ ] Add authoring tests using multibyte padding: 65,536-byte part and 262,144-byte definition pass; 65,537 and 262,145 return 413 with `service_prompt_size_limit_exceeded` and no stored pending revision.
- [ ] Run the red command:

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Service_Prompts/test_registry_documents_web.py \
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py -k 'documents_web or size_limit'
```

  Expected failure: all 32 IDs/default assets are unknown and catalog/limit fixtures cannot resolve them.
- [ ] Add the new packaged asset and registry entries. Map chat-document parts to existing `chat/<key>`, document-summary parts to `document/<key>`, and snippet `analyzer_system` to the existing `summarization/Summarization System Prompt` provider with its packaged fallback. Use `service_prompts_documents_web` only for formerly hard-coded defaults; preserve the existing `TLDW_PROMPT_FILE_*` convention and strict configured-source failures.
- [ ] Assign each browser definition to exactly one of TASK-12957's four landed registry-owned message policies. This task adds definition metadata only; it does not add a fifth policy or implement/modify policy dispatch. The client request may select only a definition/finite selector and may not supply role/order/policy.
- [ ] Rerun the same command. Expected pass: exact 32-set, all contracts/default digests, redaction, and authored limits are green.
- [ ] Run the mandatory commit/frontend gates plus this exact Stage 1 Python scope; update Backlog:

```bash
source .venv/bin/activate
TASK_PYTHON_SOURCE_SCOPE=(
  tldw_Server_API/app/core/Service_Prompts/registry.py
)
TASK_PYTHON_TEST_SCOPE=(
  tldw_Server_API/tests/Service_Prompts/test_registry_documents_web.py
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py
)
python -m black --check "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m ruff check "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m mypy "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
```
- [ ] Commit: `feat: register documents web service prompts (TASK-12959)`.

## Stage 2: Verify and extend transport through TASK-12957's chat bridge

**Goal:** Carry the existing typed execution object through every shared chat client used by this domain, without altering the backend bridge or creating client-side resolution.

**Success Criteria:** Both sync and streaming requests transport `service_prompt` unchanged; absent metadata preserves old requests; TASK-12957's four registry-owned policies produce the required standalone, copilot, web-search, and history behavior; bodies/hidden parts never round-trip to TypeScript; TASK-12959 makes no bridge behavior change.

**Tests:** Generic builder fixtures, ChatCompletion request typing, sync/stream propagation, domain bridge error mapping, owner isolation, resolver count, role/order Goldens.

**Status:** Not Started

**Files:**

- Reference prerequisite: `tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py`
- Reference prerequisite: `tldw_Server_API/app/api/v1/endpoints/chat.py`
- Modify: `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`
- Reference prerequisite: `apps/packages/ui/src/services/service-prompts.ts`
- Modify: `apps/packages/ui/src/services/__tests__/service-prompts.test.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwChat.ts`
- Modify: `apps/packages/ui/src/models/ChatTldw.ts`
- Modify: `apps/packages/ui/src/models/index.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts`
- Create: `apps/packages/ui/src/services/__tests__/tldw-chat.service-prompts.test.ts`

- [ ] First inspect the landed TASK-12957 bridge. Assert its model is optional and `extra="forbid"`, supports `definition_id`, `variables`, `explicit_parts` with `literal|template`, and finite `selector`, derives owner from `AuthPrincipal`, resolves once, strips `service_prompt` before provider adapters, and maps invalid/unknown/unavailable/oversized cases to sanitized 422/503/413 responses. Also prove its registry-owned dispatcher already implements all four policies exactly: `replace_generated_messages`, `replace_copilot_user_text_preserve_non_text`, `insert_before_current_user`, and `prepend_system_preserve_history`; the request cannot override the policy; ordinary chat with empty `messages` still rejects. If any assertion fails, stop TASK-12959 and amend/finish TASK-12957's plan, Backlog task, implementation, and tests before returning. Do not patch bridge behavior under TASK-12959.
- [ ] Write failing TypeScript tests for `buildServicePromptExecution(...)` with representative literal/template overrides (including `""` and braces), selector, unknown client fields, no local persistence, and no settings/detail fetch.
- [ ] Write failing transport tests showing `TldwApiClient.ChatCompletionRequest`, `TldwChatOptions`, `ChatTldwOptions`, `PageAssistModelOptions`, and `ChatModePrompt` carry one typed object unchanged through non-streaming and streaming calls. When the object is absent, request snapshots must remain byte-equivalent.
- [ ] Add TASK-12959 regression rows that consume, without redefining, each landed policy: standalone definitions with `messages: []`; copilot replacement of only text while byte-preserving non-text/image parts and order; web-search insertion immediately before the final current-user message; and resolved-system prepend before permitted agent/continuation history while preserving all history/current-user bytes. Empty ordinary chat requests without `service_prompt` must still fail as before.
- [ ] Extend the backend bridge table for the 19 browser IDs in this task. Assert one resolver call from the authenticated owner, exact complete provider messages, source/provenance IDs/digests only, explicit precedence, finite selector rejection, unknown/locked part rejection, 65,536-byte rendered part success, 65,537-byte 413, 262,144-byte bundle success, 262,145-byte 413, and zero provider calls on failure.
- [ ] Run the red commands:

```bash
cd apps/tldw-frontend && bunx vitest run \
  ../packages/ui/src/services/__tests__/service-prompts.test.ts \
  ../packages/ui/src/services/__tests__/tldw-chat.service-prompts.test.ts
```

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Chat/test_service_prompt_execution.py -k documents_web
```

  Expected failure: shared model clients drop the typed object and the bridge has no registered documents/web Golden cases.
- [ ] Import the TASK-12957 execution type/builder; add only optional transport properties and direct field forwarding:

```ts
type DomainChatOptions = {
  servicePrompt?: ServicePromptExecutionRequest
}

const request: ChatCompletionRequest = {
  ...existingFields,
  service_prompt: options.servicePrompt
}
```

- [ ] Pass `ChatModePrompt.servicePrompt` into `pageAssistModel(...)`, then through `ChatTldw`/`TldwChat` for both stream and sync paths. Do not inspect the definition, variables, selector, or explicit parts in these transport classes.
- [ ] Do not modify `chat_request_schemas.py` or `chat.py` under TASK-12959. Registry definition metadata from Stage 1 and domain regression rows are the only backend bridge-facing additions. Any discovered schema, merge-policy, validation, ownership, or error-mapping defect belongs to an amended TASK-12957 and blocks this stage until that dependency is green.
- [ ] Rerun both focused commands. Expected pass: typed transport and all browser-bound bridge contract/limit cases are green.
- [ ] Run the mandatory commit/frontend gates plus this exact Stage 2 Python test scope; update Backlog:

```bash
source .venv/bin/activate
TASK_PYTHON_TEST_SCOPE=(tldw_Server_API/tests/Chat/test_service_prompt_execution.py)
python -m black --check "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m ruff check "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m mypy "${TASK_PYTHON_TEST_SCOPE[@]}"
```
- [ ] Commit: `feat: transport service prompt executions in chat clients (TASK-12959)`.

## Stage 3: Migrate direct Python document, translation, title, and web consumers

**Goal:** Replace direct prompt loading/hard-coding in the 11 non-annotation server-side definitions with one authenticated immutable bundle per public boundary.

**Success Criteria:** Six chat documents, insights, document summary, translation, note title, and legacy web digest emit byte-equivalent provider messages; lower loops do not resolve; legacy explicit and fallback behavior is unchanged. The two annotation definitions remain in Stage 5.

**Tests:** Resolver-count assertions, complete provider-message Goldens, explicit/deployment/approved/default precedence, literal braces/empty behavior, output normalization, direct size errors, endpoint owner/error mapping.

**Status:** Not Started

### Work unit 3A: Chat documents

**Files:**

- Modify: `tldw_Server_API/app/core/Chat/document_generator.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat_documents.py`
- Modify: `tldw_Server_API/tests/Chat/unit/test_document_generator.py`
- Create: `tldw_Server_API/tests/Chat/unit/test_document_generator_service_prompts.py`
- Modify: `tldw_Server_API/tests/Chat/integration/test_document_generation_endpoints.py`

- [ ] Add failing parameterized tests over all six document types. Capture complete `system_message` and user message bytes for default, focus present/absent, context truncation, saved row, request custom prompt, saved-empty row, and literal braces.
- [ ] Assert compatibility precedence: truthy request `custom_prompt` overlays only `user_instruction`; present saved system/user rows (including empty) become explicit literal parts; temperature/max-token settings remain from the saved/default generation settings; the remaining parts use approved/deployment/packaged precedence.
- [ ] Assert `generate_document` calls the resolver once, then `_call_llm` and persistence receive only the frozen rendered messages. Constructor, context formatting, `_call_llm`, save, stream iteration, and prompt-cache reads must not resolve.
- [ ] Add FastAPI integration tests with two authenticated principals that have distinct approved revisions for the same chat-document definition. Capture provider messages for each principal and prove neither revision crosses accounts; missing/mismatched owner context fails before resolution/provider/persistence. For each create/generate document route, submit a one-byte-over runtime/aggregate fixture and assert HTTP 413 with exact code `service_prompt_size_limit_exceeded`, zero provider calls, and no document persistence.
- [ ] Run `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Chat/unit/test_document_generator_service_prompts.py tldw_Server_API/tests/Chat/integration/test_document_generation_endpoints.py -k 'service_prompt or document'`; expect failures because `DocumentGeneratorService` mutates `DEFAULT_PROMPTS` and assembles prompt text directly.
- [ ] Remove prompt-text mutation from `DocumentGeneratorService.__init__`. Keep only immutable non-prompt generation settings. Split saved-row lookup from effective prompt resolution so a row's presence is not confused with a server default.
- [ ] Inject `ServicePromptService` and owner into the request-scoped service from `chat_documents.py`; resolve the selected definition once in `generate_document`, render locked focus/context once, and pass exact strings into `_call_llm`. Existing non-generating constructors used by Chatbooks/Notifications must remain valid and must not resolve.
- [ ] Rerun the focused command; expect six exact message arrays and all endpoint compatibility cases to pass.
- [ ] Run the mandatory commit/frontend gates plus this exact Work unit 3A Python scope; update Backlog:

```bash
source .venv/bin/activate
TASK_PYTHON_SOURCE_SCOPE=(
  tldw_Server_API/app/core/Chat/document_generator.py
  tldw_Server_API/app/api/v1/endpoints/chat_documents.py
)
TASK_PYTHON_TEST_SCOPE=(
  tldw_Server_API/tests/Chat/unit/test_document_generator.py
  tldw_Server_API/tests/Chat/unit/test_document_generator_service_prompts.py
  tldw_Server_API/tests/Chat/integration/test_document_generation_endpoints.py
)
python -m black --check "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m ruff check "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m mypy "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
```
- [ ] Commit: `feat: migrate chat document service prompts (TASK-12959)`.

### Work unit 3B: Insights, document summary, translation, and note title

**Files:**

- Modify: `tldw_Server_API/app/api/v1/endpoints/media/document_insights.py`
- Modify: `tldw_Server_API/tests/Media/test_document_insights.py`
- Create: `tldw_Server_API/tests/Media/test_document_insights_service_prompts_api.py`
- Modify: `tldw_Server_API/app/services/document_processing_service.py`
- Modify: `tldw_Server_API/tests/Services/test_document_processing_service.py`
- Create: `tldw_Server_API/tests/Services/test_document_processing_service_prompts.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/translate.py`
- Create: `tldw_Server_API/tests/Translation/test_translate_service_prompts.py`
- Modify: `tldw_Server_API/tests/Translation/test_translate_endpoint_error_mapping.py`
- Modify: `tldw_Server_API/app/core/Writing/note_title.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/notes.py`
- Modify: `tldw_Server_API/tests/unit/test_note_title_lib.py`
- Create: `tldw_Server_API/tests/unit/test_note_title_service_prompts.py`
- Modify: `tldw_Server_API/tests/integration/test_notes_auto_title_api.py`

- [ ] Add failing insights unit and FastAPI tests for one endpoint-bound resolution, exact system/user bytes, category-present/absent framing, the existing 50,000-character shaping, JSON response mode, approved/deployment/default precedence, sanitized unavailability, and unchanged `_normalize_insights` behavior. Use two authenticated principals with distinct approved `media.document.insights` revisions to prove cross-user isolation and missing-owner failure. Assert a one-byte-over request returns HTTP 413 with exact code `service_prompt_size_limit_exceeded` before provider/cache mutation.
- [ ] Add failing document-summary tests proving truthy explicit system/user literals beat stored/deployment/default parts, empty strings do not override, braces remain literal, the definition resolves once before document/chunk loops, and every `analyze` call uses the same bundle with exact document/chunk + `\n\n\n\n` + instruction bytes. Because this placeholder service has no public production endpoint, exercise two explicit authenticated `PromptExecutionContext` owners with distinct approved revisions and assert missing/mismatched owner input fails closed; never infer owner `1` or use a global fallback.
- [ ] Add failing translation FastAPI tests for exact `{target_language}`/`{text}` rendering once, 10,000-character shaping before `G`, literal explicit system braces, one endpoint resolution, analyzer options, cleanup/error sentinel, and safe 413/503 mapping before `analyze`. Use two authenticated principals with distinct approved `media.text.translation` revisions, prove cross-user isolation and missing-owner failure, and assert one-byte-over returns exact 413 `service_prompt_size_limit_exceeded` before `analyze`.
- [ ] Add failing note-title unit and FastAPI integration tests for the feature/strategy gate, one resolution only when the LLM path is active, exact OpenAI adapter system/user/options, `max_len` clamp 10–255, 2,000-character snippet shaping, and normalization. Exercise all three LLM-capable boundaries—note create with `auto_title=true`, `/api/v1/notes/title/suggest`, and `/api/v1/notes/bulk` items with `auto_title=true`—using two authenticated principals with distinct approved `notes.title.generate` revisions; prove cross-user isolation, missing-owner failure, and one resolver call per generated title.
- [ ] Split note-title failures explicitly. Preserve heuristic fallback only for pre-existing provider failure, provider-configuration failure, or empty normalized provider output. Foundation Service Prompt unavailable/quarantined/validation errors must fail closed through the shared sanitized mapper, and size errors must return HTTP 413 with exact code `service_prompt_size_limit_exceeded`. Assert create/suggest/bulk do not invoke the heuristic, dispatch a provider, or persist any affected note; bulk rejects atomically rather than partially writing earlier items.
- [ ] Run the red command:

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Media/test_document_insights.py \
  tldw_Server_API/tests/Media/test_document_insights_service_prompts_api.py \
  tldw_Server_API/tests/Services/test_document_processing_service_prompts.py \
  tldw_Server_API/tests/Translation/test_translate_service_prompts.py \
  tldw_Server_API/tests/Translation/test_translate_endpoint_error_mapping.py \
  tldw_Server_API/tests/unit/test_note_title_service_prompts.py \
  tldw_Server_API/tests/integration/test_notes_auto_title_api.py
```

  Expected failure: each consumer still loads/constants/assembles directly and records zero resolver calls.
- [ ] Inject the existing service dependency and canonical authenticated owner at every endpoint above. For the placeholder document service, require an explicit `PromptExecutionContext`/resolver input whenever summarization is enabled; do not invent a userless fallback. Catch foundation Service Prompt errors before legacy broad exception/fallback handlers so typed 422/503/413 responses survive unchanged.
- [ ] Resolve once immediately before each inventory boundary (`messages_payload`, pre-chunk document analysis, translation `analyze`, and `_try_generate_title_llm`). Pass the immutable bundle into lower functions and preserve all provider/model/temperature/token options.
- [ ] Remove only migrated semantic constants/imports after byte Goldens pass; keep output schemas, normalizers, gates, and error sentinels locked. Keep the note-title heuristic only for the three pre-existing fallback cases above; never convert a Service Prompt control-plane/integrity/validation/size error into a title.
- [ ] Rerun the focused command. Expected pass: exact bytes, count, precedence, shaping, failure, and output-enforcement tests are green.
- [ ] Run the mandatory commit/frontend gates plus this exact Work unit 3B Python scope; update Backlog:

```bash
source .venv/bin/activate
TASK_PYTHON_SOURCE_SCOPE=(
  tldw_Server_API/app/api/v1/endpoints/media/document_insights.py
  tldw_Server_API/app/services/document_processing_service.py
  tldw_Server_API/app/api/v1/endpoints/translate.py
  tldw_Server_API/app/core/Writing/note_title.py
  tldw_Server_API/app/api/v1/endpoints/notes.py
)
TASK_PYTHON_TEST_SCOPE=(
  tldw_Server_API/tests/Media/test_document_insights.py
  tldw_Server_API/tests/Media/test_document_insights_service_prompts_api.py
  tldw_Server_API/tests/Services/test_document_processing_service.py
  tldw_Server_API/tests/Services/test_document_processing_service_prompts.py
  tldw_Server_API/tests/Translation/test_translate_service_prompts.py
  tldw_Server_API/tests/Translation/test_translate_endpoint_error_mapping.py
  tldw_Server_API/tests/unit/test_note_title_lib.py
  tldw_Server_API/tests/unit/test_note_title_service_prompts.py
  tldw_Server_API/tests/integration/test_notes_auto_title_api.py
)
python -m black --check "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m ruff check "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m mypy "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
```
- [ ] Commit: `feat: migrate document translation title prompts (TASK-12959)`.

### Work unit 3C: Legacy web-search snippet digest

**Files:**

- Modify: `tldw_Server_API/app/core/WebSearch/Web_Search.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/research.py`
- Create: `tldw_Server_API/tests/WebSearch/test_websearch_service_prompts.py`
- Modify: `tldw_Server_API/tests/WebSearch/test_websearch_core.py`
- Modify: `tldw_Server_API/tests/WebSearch/unit/test_aggregate_results_schema.py`
- Modify: `tldw_Server_API/tests/WebSearch/integration/test_websearch_endpoint.py`

- [ ] Add one parameterized failing suite over both byte-identical implementations. Assert the same definition resolves once before each aggregation chunk loop, not per chunk; every chunk renders from the frozen bundle with `question`, source-built finite `chunk_index`, and the same `result_snippets` exactly twice.
- [ ] Capture complete provider requests: snapshotted analyzer system; locked source-built/bounded snippet input; exact `\n\n\n\n`; digest template with current indentation/tags; unchanged temperature/provider/streaming values. Assert failure fallback/failed-chunk accounting and final report behavior remain unchanged.
- [ ] Cover the existing 6,000-character source chunk construction/truncation as the only shaping, literal braces/Unicode in question/snippets, approved/deployment/default precedence, strict configured analyzer default failure, safe provenance, and direct `service_prompt_size_limit_exceeded` before `summarize`.
- [ ] Add FastAPI integration cases for `research.websearch_endpoint` with two authenticated principals and distinct approved `web.search.snippet.digest` revisions. Prove owner forwarding, cross-user isolation, and missing-owner failure; assert an oversize rendered part/bundle returns HTTP 413 with exact code `service_prompt_size_limit_exceeded` before aggregation/provider/cache side effects.
- [ ] Run `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/WebSearch/test_websearch_service_prompts.py tldw_Server_API/tests/WebSearch/integration/test_websearch_endpoint.py`; expect failures because both loops construct `chunk_prompt` directly and the endpoint does not thread owner context.
- [ ] Thread the authenticated execution context from `research.websearch_endpoint` into `analyze_and_aggregate`/`aggregate_results`. Resolve `web.search.snippet.digest` once before the loop and render only per-chunk variables below it.
- [ ] Preserve the existing `summarize(input_data=..., custom_prompt_arg=..., system_message=...)` adapter boundary: supply rendered locked snippets, rendered digest suffix, and the resolved analyzer system so `Summarization_General_Lib` retains its exact four-newline assembly without a second definition lookup.
- [ ] Rerun the focused command; expect both implementations' resolver counts, provider-message Goldens, endpoint owner mapping, and failure behavior to pass.
- [ ] Run the mandatory commit/frontend gates plus this exact Work unit 3C Python scope; update Backlog:

```bash
source .venv/bin/activate
TASK_PYTHON_SOURCE_SCOPE=(
  tldw_Server_API/app/core/WebSearch/Web_Search.py
  tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py
  tldw_Server_API/app/api/v1/endpoints/research.py
)
TASK_PYTHON_TEST_SCOPE=(
  tldw_Server_API/tests/WebSearch/test_websearch_service_prompts.py
  tldw_Server_API/tests/WebSearch/test_websearch_core.py
  tldw_Server_API/tests/WebSearch/unit/test_aggregate_results_schema.py
  tldw_Server_API/tests/WebSearch/integration/test_websearch_endpoint.py
)
python -m black --check "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m ruff check "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m mypy "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
```
- [ ] Commit: `feat: migrate web snippet digest prompt (TASK-12959)`.

## Stage 4: Route every browser workflow through the shared server bridge

**Goal:** Remove provider-facing prompt assembly for the 19 browser definitions while preserving local explicit settings, selectors, runtime shaping, history, multimodal content, and model options.

**Success Criteria:** Each action sends one typed execution object and only allowed runtime/history data; no client performs resolution or locked assembly; server bridge Goldens equal current provider messages.

**Tests:** Request-capture Vitest tests per consumer cluster, 19-ID backend Golden matrix, sync/stream transport, real-server settings canary.

**Status:** Not Started

### Work unit 4A: Copilot, image refinement, and client web search

**Files:**

- Modify: `apps/packages/ui/src/services/application.ts`
- Modify: `apps/packages/ui/src/hooks/useMessage.tsx`
- Modify: `apps/packages/ui/src/services/__tests__/application.copilot-prompts.test.ts`
- Create: `apps/packages/ui/src/hooks/__tests__/useMessage.service-prompts.test.tsx`
- Modify: `apps/packages/ui/src/utils/image-prompt-refinement.ts`
- Modify: `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundImageGen.ts`
- Modify: `apps/packages/ui/src/utils/__tests__/image-prompt-refinement.test.ts`
- Modify: `apps/packages/ui/src/services/tldw-server.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/__tests__/normalChatMode.overlay.test.ts`
- Modify: `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`

- [ ] Add failing copilot tests for four message-type mappings, raw stored-value presence (including empty), missing-key no-override behavior, `{text}` exactly once, braces/Unicode, no raw default prompt in the request, and image calls preserving the current non-text image part while server-rendering the user text.
- [ ] Add failing image-refinement tests for normalized/truncated original prompt, finite strategy/backend, one preassembled optional context-cues variable capped at four entries, no per-entry template iteration, exact model/options, and no system/rewrite defaults in TypeScript requests.
- [ ] Add failing normal-chat tests for local nonempty template as an explicit constrained part, empty local setting as absent, ISO timestamp and normalized result text as variables, current result snippet shaping, actor/base history unchanged, and registry-owned `insert_before_current_user` placement. Compare the complete provider array and prove the rendered web-search message retains role `system`, follows every permitted history/actor message, and immediately precedes the unchanged current `user` message; the client cannot submit role, policy, or insertion fields.
- [ ] Run the red command from `apps/tldw-frontend`:

```bash
bunx vitest run \
  ../packages/ui/src/services/__tests__/application.copilot-prompts.test.ts \
  ../packages/ui/src/hooks/__tests__/useMessage.service-prompts.test.tsx \
  ../packages/ui/src/utils/__tests__/image-prompt-refinement.test.ts \
  ../packages/ui/src/hooks/chat-modes/__tests__/normalChatMode.overlay.test.ts
```

  Expected failure: requests still contain fully rendered prompt bodies and lack `service_prompt` metadata.
- [ ] Add raw local-override accessors without changing the existing settings getters. Copilot treats any stored string as explicit; web search treats only a nonempty string as explicit. Keep `custom` copilot outside this migration.
- [ ] Change image helper output from provider messages to bounded runtime variables/context cues. Use `buildServicePromptExecution(...)` at the action boundary; do not fetch catalog/detail or copy locked/default strings.
- [ ] Put web-search execution metadata on `ChatModePrompt` so the pipeline transports it unchanged while retaining ordinary history/human-message assembly.
- [ ] Extend the server bridge Golden table for these six IDs and rerun Vitest plus `python -m pytest -q tldw_Server_API/tests/Chat/test_service_prompt_execution.py -k 'documents_copilot or image_prompt_refinement or web_search_client'`.
- [ ] Expected pass: browser requests are body-free except explicit local content/runtime data and provider messages remain byte-equivalent.
- [ ] Run the mandatory commit/frontend gates plus this exact Work unit 4A Python test scope; update Backlog:

```bash
source .venv/bin/activate
TASK_PYTHON_TEST_SCOPE=(tldw_Server_API/tests/Chat/test_service_prompt_execution.py)
python -m black --check "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m ruff check "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m mypy "${TASK_PYTHON_TEST_SCOPE[@]}"
```
- [ ] Commit: `feat: route document web browser prompts server side (TASK-12959)`.

### Work unit 4B: Book and webpage workflows

**Files:**

- Modify: `apps/packages/ui/src/components/Common/Workflow/steps/AnalyzeBookWorkflow.tsx`
- Create: `apps/packages/ui/src/components/Common/Workflow/__tests__/AnalyzeBookWorkflow.service-prompts.test.tsx`
- Modify: `apps/packages/ui/src/components/Common/Workflow/steps/SummarizePageWorkflow.tsx`
- Create: `apps/packages/ui/src/components/Common/Workflow/__tests__/SummarizePageWorkflow.service-prompts.test.tsx`
- Modify: `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`

- [ ] Add failing request-capture tests for four preset-to-ID mappings, exact 30,000-character shaping, whole-book/per-chapter loops, and the custom preset as an explicit literal `style_fragment` including empty/braces.
- [ ] Add failing request-capture tests for three page-style-to-ID mappings, exact title/URL variables, 10,000-character content shaping, and unchanged progress/error behavior.
- [ ] Assert both workflows send no base system, style default, wrapper, or content carrier text—only the ID, variables, permitted explicit part, model/options, and `messages: []` allowed by registry policy.
- [ ] Run `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Common/Workflow/__tests__/AnalyzeBookWorkflow.service-prompts.test.tsx ../packages/ui/src/components/Common/Workflow/__tests__/SummarizePageWorkflow.service-prompts.test.tsx`; expect raw `messages` assertions to fail.
- [ ] Replace only provider-message construction with `buildServicePromptExecution(...)`. Resolve once per workflow call at `analyzeContent`/summary creation; reuse the same definition selection for every existing chapter iteration without client-side prompt assembly.
- [ ] Extend the server bridge Golden table for all seven IDs and rerun Vitest plus `python -m pytest -q tldw_Server_API/tests/Chat/test_service_prompt_execution.py -k 'workflow_book or workflow_web_summary'`.
- [ ] Expected pass: exact server provider arrays, input shaping, custom-empty semantics, and workflow behavior are green.
- [ ] Run the mandatory commit/frontend gates plus this exact Work unit 4B Python test scope; update Backlog:

```bash
source .venv/bin/activate
TASK_PYTHON_TEST_SCOPE=(tldw_Server_API/tests/Chat/test_service_prompt_execution.py)
python -m black --check "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m ruff check "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m mypy "${TASK_PYTHON_TEST_SCOPE[@]}"
```
- [ ] Commit: `feat: migrate book and page workflow prompts (TASK-12959)`.

### Work unit 4C: Writing agent, continuation, and echo

**Files:**

- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/AIAgentTab.tsx`
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/AIAgentTab.service-prompts.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/hooks/utils.ts`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx`
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.service-prompts.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingFeedback.ts`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/hooks/__tests__/useWritingFeedback.test.tsx`
- Modify: `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`

- [ ] Add failing agent tests for quick/planning/brainstorm ID mapping, exact temperatures/token limits, current history/current user message, optional context order, scene `C2000` with `...`, ten character/world entries, Untitled/unknown-role/info defaults, and no system/context labels assembled in TypeScript.
- [ ] Add failing continuation tests that only non-chat `predict`/`fill` attach service metadata, preserve the existing plan/context messages/prefix/suffix/stops and sync/stream choice, and leave chat mode's explicit system/context path unchanged.
- [ ] Add failing echo tests for deterministic round-robin selectors, `passage=editorText.slice(-1000)`, debounce/history behavior, no persona prompt bytes in the request, and exact model/temperature/token settings. Assert the adjacent mood branch remains on its current literal request and never receives `writing.feedback.echo` metadata.
- [ ] Run the red command:

```bash
cd apps/tldw-frontend && bunx vitest run \
  ../packages/ui/src/components/Option/WritingPlayground/__tests__/AIAgentTab.service-prompts.test.tsx \
  ../packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.service-prompts.test.tsx \
  ../packages/ui/src/components/Option/WritingPlayground/hooks/__tests__/useWritingFeedback.test.tsx
```

  Expected failure: current clients still send hard-coded systems/personas and have no service-prompt transport option.
- [ ] Replace `SYSTEM_PROMPTS`, `PREDICT_SYSTEM_PROMPT`, `FILL_SYSTEM_PROMPT`, and echo persona prompt bodies only after Stage 1 Goldens exist. Keep non-prompt persona name/emoji/role metadata client-side.
- [ ] Send shaped context/history as the registry-permitted locked carrier and let the server prepend/assemble the resolved system. Do not let TypeScript choose provider roles/order.
- [ ] Refactor `callChat` to accept optional typed execution metadata; use it only for echo. Preserve mood's current literal system/user path for TASK-12962.
- [ ] Extend the server bridge Golden table for all six writing IDs and rerun Vitest plus `python -m pytest -q tldw_Server_API/tests/Chat/test_service_prompt_execution.py -k 'writing_agent or writing_continuation or writing_feedback_echo'`.
- [ ] Expected pass: exact messages, selector rotation, shaping, chat bypass, mood non-regression, and sync/stream behavior are green.
- [ ] Run the mandatory commit/frontend gates plus this exact Work unit 4C Python test scope; update Backlog:

```bash
source .venv/bin/activate
TASK_PYTHON_TEST_SCOPE=(tldw_Server_API/tests/Chat/test_service_prompt_execution.py)
python -m black --check "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m ruff check "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m mypy "${TASK_PYTHON_TEST_SCOPE[@]}"
```
- [ ] Commit: `feat: migrate writing browser service prompts (TASK-12959)`.

## Stage 5: Migrate annotations, prove limits/completeness, and close the domain

**Goal:** Migrate direct selected-text review and protected scene-review Jobs, then prove the complete 32-ID domain through integration, security, documentation, and mandatory full shards.

**Success Criteria:** Selected review resolves once; scene review pins one finite bundle before queue release and consumes verified WorkerSDK context; exact limits/failures precede side effects; all eligible call sites are migrated and all remaining prompt sources are explicitly deferred/excluded.

**Tests:** Selected annotation provider/normalizer Goldens, held-bind-release and WorkerSDK tamper matrix, time-of-use pin immutability, full 32-ID precedence/provenance/limits matrix, real-server UI canary, Bandit/full CI.

**Status:** Not Started

### Work unit 5A: Direct selected-text annotation review

**Files:**

- Modify: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
- Modify: `tldw_Server_API/app/core/Writing/manuscript_annotations.py`
- Modify: `tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py`

- [ ] Add failing unit and FastAPI tests for one authenticated resolution in `create_selected_text_review_annotation`, exact system/user bytes, selected text/focus/category/document metadata once each, approved/deployment/default precedence, braces/Unicode, and safe provenance. Use two authenticated principals with distinct approved `writing.annotation.selection` revisions; prove owner forwarding, cross-user isolation, and missing/mismatched-owner failure.
- [ ] At the direct selected-text review route, submit an oversize rendered part and aggregate bundle and assert HTTP 413 with exact code `service_prompt_size_limit_exceeded`, zero provider calls, and no annotation/anchor persistence.
- [ ] Assert strict parser, exactly-one-annotation contract, category/range validation, anchor persistence, and no-persist-on-invalid-output remain unchanged.
- [ ] Run `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py -k 'selected_text and service_prompt'`; expect direct `build_selected_text_review_prompt` provider assembly and zero resolver calls.
- [ ] Inject the authenticated service/context from `writing_manuscripts.py`, resolve once at selected-review construction, render locked anchor/schema carriers, and pass complete messages to the existing provider/parser path.
- [ ] Rerun the focused command; expect exact bytes, one resolution, size/no-dispatch, and parser/persistence assertions to pass.

### Work unit 5B: Protected scene annotation Jobs

**Files:**

- Modify: `tldw_Server_API/app/api/v1/API_Deps/service_prompt_deps.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
- Modify: `tldw_Server_API/app/core/Writing/manuscript_annotation_jobs.py`
- Modify: `tldw_Server_API/app/services/writing_annotation_review_jobs_worker.py`
- Modify: `tldw_Server_API/tests/Writing/test_manuscript_annotation_review_jobs.py`
- Modify: `tldw_Server_API/tests/Services/test_writing_annotation_review_jobs_worker.py`

- [ ] Add failing enqueue tests declaring the finite requirement exactly `{writing.annotation.scene}` and proving protected pin-set commit → held job → authenticated bind → queued release. Cover all-or-nothing failure, idempotent retry, owner/submission match, request-time 65,536/65,537 explicit part handling, and no prompt body or scene text in ordinary payload/events/status/logs.
- [ ] Add failing WorkerSDK tests for valid verified context and missing/tampered/wrong-owner/wrong-definition/wrong-contract/digest-mismatch/expired-key/operator-bypass cases. Handler/provider must not run on invalid pins; transient store errors retry; integrity errors quarantine; bypass holds without substitution.
- [ ] Add a time-of-use Golden: enqueue an approved scene-review revision, edit/reset it, then execute. The provider receives the originally authenticated template bytes. Runtime scene text/version/focus/anchor metadata is loaded authoritatively after verification and rendered once.
- [ ] Run the red command:

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Writing/test_manuscript_annotation_review_jobs.py \
  tldw_Server_API/tests/Services/test_writing_annotation_review_jobs_worker.py
```

  Expected failure: enqueue still calls `JobManager.create_job` directly and the handler rebuilds raw prompt messages without verified WorkerSDK context.
- [ ] Expose the existing `ServicePromptJobPinner` through `service_prompt_deps.py`; do not add another store/verifier. In `enqueue_scene_annotation_review_job`, call `ServicePromptJobPinner.enqueue(...)` with the authenticated owner, one-time submission, finite definition set, and current explicit parts (none today).
- [ ] Keep normal payload fields bounded and non-prompt. The protected store snapshots complete render-ready templates/assembly metadata; it does not snapshot rendered scene/document variables. Ordinary job payload contains only safe pin references/digests plus current non-prompt IDs/provider/model/filter metadata.
- [ ] Adapt the writing worker's WorkerSDK handler signature to receive verified context. Before scene lookup/provider dispatch, require the pinned bundle:

```python
bundle = worker_context.service_prompts.require("writing.annotation.scene")
```

- [ ] After verification, load the authoritative scene/version, render runtime variables through the frozen bundle, and call the provider. Never call the registry/resolver/pin store from the handler and never accept raw prompt bodies from `job_payload`.
- [ ] Rerun the focused command; expect held-bind-release, pin immutability, integrity classification, payload redaction, and provider-message Goldens to pass.
- [ ] Run the mandatory commit/frontend gates plus this exact combined Work units 5A–5B Python scope; update Backlog:

```bash
source .venv/bin/activate
TASK_PYTHON_SOURCE_SCOPE=(
  tldw_Server_API/app/api/v1/API_Deps/service_prompt_deps.py
  tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py
  tldw_Server_API/app/core/Writing/manuscript_annotations.py
  tldw_Server_API/app/core/Writing/manuscript_annotation_jobs.py
  tldw_Server_API/app/services/writing_annotation_review_jobs_worker.py
)
TASK_PYTHON_TEST_SCOPE=(
  tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py
  tldw_Server_API/tests/Writing/test_manuscript_annotation_review_jobs.py
  tldw_Server_API/tests/Services/test_writing_annotation_review_jobs_worker.py
)
python -m black --check "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m ruff check "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
python -m mypy "${TASK_PYTHON_SOURCE_SCOPE[@]}" "${TASK_PYTHON_TEST_SCOPE[@]}"
```
- [ ] Commit: `feat: pin writing annotation service prompts (TASK-12959)`.

### Work unit 5C: Cross-domain limits, canary, documentation, and final verification

**Files:**

- Create: `tldw_Server_API/tests/Service_Prompts/test_runtime_documents_web.py`
- Modify: `tldw_Server_API/tests/Service_Prompts/test_registry_documents_web.py`
- Modify: `tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py`
- Modify: `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`
- Modify: `apps/tldw-frontend/e2e/workflows/service-prompts-settings.spec.ts`
- Modify: `Docs/Design/service-prompt-inventory.md`
- Modify: `Docs/API/service-prompts.md`
- Modify: `tldw_Server_API/Config_Files/Prompts/README.md`
- Update through Backlog MCP/CLI: `TASK-12959`

- [ ] Add a parameterized matrix over all 32 IDs proving packaged/deployment/approved/explicit/job-pin precedence as applicable, exact literal/template/empty/truthy/selector behavior, one resolution at each matrix boundary, complete no-override provider-message bytes, and safe provenance with no body/path.
- [ ] Add exact multibyte boundary cases across settings API, chat bridge, direct renderer, and Jobs preflight: authored/expanded/rendered part 65,536 pass and 65,537 fail; authored/final bundle 262,144 pass and 262,145 fail. Assert HTTP 413 or exact non-HTTP `service_prompt_size_limit_exceeded`, no partial resolution, no provider/job/persistence side effect, and no silent truncation beyond the listed existing shaping.
- [ ] Add the synthetic in-test five-part aggregate fixture exactly as specified in the budget contract: four 65,000-byte parts + one 2,132-byte part + four `"\n--"` separators (`12` assembly bytes) = 262,144; increasing only the fifth part to 2,133 = 262,145. Assert every part is `<= 65_536`, the fixture is not registered as a 33rd product ID, and both authored/final accounting include the explicit separator bytes.
- [ ] Add/retain direct FastAPI integration assertions for all public Python clusters: chat-document generation, document insights, translation, note create/title-suggest/bulk, research web-search aggregation, and selected-text review. For each, verify authenticated-owner forwarding and two-user isolation. For chat documents, insights, all three note-title boundaries, research, and selected review, assert HTTP 413 with exact response code `service_prompt_size_limit_exceeded` and no provider/job/cache/persistence side effect.
- [ ] Add a real-server Playwright canary using the existing mock provider: edit and explicitly approve `documents.copilot.summary` in WebUI, invoke the Copilot summary action, and assert the provider receives the approved template plus runtime text. Then save a pending edit through the extension-options hash route and verify WebUI observes the same account's pending revision. Keep raw prompt text out of screenshots/traces/console fixture names.
- [ ] Run the new matrix and canary tests before documentation/fixture corrections:

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Service_Prompts/test_runtime_documents_web.py \
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py \
  tldw_Server_API/tests/Chat/test_service_prompt_execution.py -k documents_web
cd apps/tldw-frontend && bunx playwright test e2e/workflows/service-prompts-settings.spec.ts --reporter=line
```

  Expected red result: the all-32 precedence/limit table and real approved-override Copilot canary expose any missing domain wiring or evidence before the closeout edits.
- [ ] Search every legacy source string/key and all provider calls in the touched consumers. Confirm each eligible call now uses an immutable resolved/pinned bundle and every remaining hit is a compatibility asset, locked test fixture, or explicitly deferred/excluded inventory row. Do not delete unrelated constants.
- [ ] Update only these 32 inventory rows with migrated call sites, contract version, Golden test path, and availability. Document chat-execution request usage, error/limit behavior, new packaged asset, strict compatibility providers, and scene pinning in `Docs/API/service-prompts.md` and the Prompts README.
- [ ] Immediately after the inventory edit, run `node Helper_Scripts/validate_service_prompt_inventory.mjs .` from the repository root. A nonzero exit blocks closeout and commit; record the JSON counts/reference results in TASK-12959.
- [ ] Run all focused backend suites together:

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Service_Prompts/test_registry_documents_web.py \
  tldw_Server_API/tests/Service_Prompts/test_runtime_documents_web.py \
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py \
  tldw_Server_API/tests/Chat/test_service_prompt_execution.py \
  tldw_Server_API/tests/Chat/unit/test_document_generator.py \
  tldw_Server_API/tests/Chat/unit/test_document_generator_service_prompts.py \
  tldw_Server_API/tests/Chat/integration/test_document_generation_endpoints.py \
  tldw_Server_API/tests/Media/test_document_insights.py \
  tldw_Server_API/tests/Media/test_document_insights_service_prompts_api.py \
  tldw_Server_API/tests/Services/test_document_processing_service.py \
  tldw_Server_API/tests/Services/test_document_processing_service_prompts.py \
  tldw_Server_API/tests/Translation/test_translate_service_prompts.py \
  tldw_Server_API/tests/Translation/test_translate_endpoint_error_mapping.py \
  tldw_Server_API/tests/unit/test_note_title_lib.py \
  tldw_Server_API/tests/unit/test_note_title_service_prompts.py \
  tldw_Server_API/tests/integration/test_notes_auto_title_api.py \
  tldw_Server_API/tests/WebSearch/test_websearch_service_prompts.py \
  tldw_Server_API/tests/WebSearch/test_websearch_core.py \
  tldw_Server_API/tests/WebSearch/unit/test_aggregate_results_schema.py \
  tldw_Server_API/tests/WebSearch/integration/test_websearch_endpoint.py \
  tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py \
  tldw_Server_API/tests/Writing/test_manuscript_annotation_review_jobs.py \
  tldw_Server_API/tests/Services/test_writing_annotation_review_jobs_worker.py
```

- [ ] Run all focused frontend suites from `apps/tldw-frontend`:

```bash
bunx vitest run \
  ../packages/ui/src/services/__tests__/service-prompts.test.ts \
  ../packages/ui/src/services/__tests__/tldw-chat.service-prompts.test.ts \
  ../packages/ui/src/services/__tests__/application.copilot-prompts.test.ts \
  ../packages/ui/src/hooks/__tests__/useMessage.service-prompts.test.tsx \
  ../packages/ui/src/utils/__tests__/image-prompt-refinement.test.ts \
  ../packages/ui/src/hooks/chat-modes/__tests__/normalChatMode.overlay.test.ts \
  ../packages/ui/src/components/Common/Workflow/__tests__/AnalyzeBookWorkflow.service-prompts.test.tsx \
  ../packages/ui/src/components/Common/Workflow/__tests__/SummarizePageWorkflow.service-prompts.test.tsx \
  ../packages/ui/src/components/Option/WritingPlayground/__tests__/AIAgentTab.service-prompts.test.tsx \
  ../packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.service-prompts.test.tsx \
  ../packages/ui/src/components/Option/WritingPlayground/hooks/__tests__/useWritingFeedback.test.tsx
```

- [ ] Run the real-server canary: `cd apps/tldw-frontend && bunx playwright test e2e/workflows/service-prompts-settings.spec.ts --reporter=line`.
- [ ] Run Bandit on every touched Python source and review/fix every new finding:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Service_Prompts/registry.py \
  tldw_Server_API/app/api/v1/API_Deps/service_prompt_deps.py \
  tldw_Server_API/app/core/Chat/document_generator.py \
  tldw_Server_API/app/api/v1/endpoints/chat_documents.py \
  tldw_Server_API/app/api/v1/endpoints/media/document_insights.py \
  tldw_Server_API/app/services/document_processing_service.py \
  tldw_Server_API/app/api/v1/endpoints/translate.py \
  tldw_Server_API/app/core/Writing/note_title.py \
  tldw_Server_API/app/api/v1/endpoints/notes.py \
  tldw_Server_API/app/core/WebSearch/Web_Search.py \
  tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py \
  tldw_Server_API/app/api/v1/endpoints/research.py \
  tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py \
  tldw_Server_API/app/core/Writing/manuscript_annotations.py \
  tldw_Server_API/app/core/Writing/manuscript_annotation_jobs.py \
  tldw_Server_API/app/services/writing_annotation_review_jobs_worker.py \
  -f json -o /tmp/bandit_task_12113_2.json
```

- [ ] Run the final no-waiver quality/full-shard gate exactly; do not substitute focused success or smaller path sets:

```bash
source .venv/bin/activate
python -m pytest -v

PYTHON_CHECK_PATHS=(
  tldw_Server_API/app/core/Service_Prompts/registry.py
  tldw_Server_API/app/api/v1/API_Deps/service_prompt_deps.py
  tldw_Server_API/app/core/Chat/document_generator.py
  tldw_Server_API/app/api/v1/endpoints/chat_documents.py
  tldw_Server_API/app/api/v1/endpoints/media/document_insights.py
  tldw_Server_API/app/services/document_processing_service.py
  tldw_Server_API/app/api/v1/endpoints/translate.py
  tldw_Server_API/app/core/Writing/note_title.py
  tldw_Server_API/app/api/v1/endpoints/notes.py
  tldw_Server_API/app/core/WebSearch/Web_Search.py
  tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py
  tldw_Server_API/app/api/v1/endpoints/research.py
  tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py
  tldw_Server_API/app/core/Writing/manuscript_annotations.py
  tldw_Server_API/app/core/Writing/manuscript_annotation_jobs.py
  tldw_Server_API/app/services/writing_annotation_review_jobs_worker.py
)
PYTHON_TEST_PATHS=(
  tldw_Server_API/tests/Chat/integration/test_document_generation_endpoints.py
  tldw_Server_API/tests/Chat/test_service_prompt_execution.py
  tldw_Server_API/tests/Chat/unit/test_document_generator.py
  tldw_Server_API/tests/Chat/unit/test_document_generator_service_prompts.py
  tldw_Server_API/tests/Media/test_document_insights.py
  tldw_Server_API/tests/Media/test_document_insights_service_prompts_api.py
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py
  tldw_Server_API/tests/Service_Prompts/test_registry_documents_web.py
  tldw_Server_API/tests/Service_Prompts/test_runtime_documents_web.py
  tldw_Server_API/tests/Services/test_document_processing_service.py
  tldw_Server_API/tests/Services/test_document_processing_service_prompts.py
  tldw_Server_API/tests/Services/test_writing_annotation_review_jobs_worker.py
  tldw_Server_API/tests/Translation/test_translate_endpoint_error_mapping.py
  tldw_Server_API/tests/Translation/test_translate_service_prompts.py
  tldw_Server_API/tests/WebSearch/integration/test_websearch_endpoint.py
  tldw_Server_API/tests/WebSearch/test_websearch_core.py
  tldw_Server_API/tests/WebSearch/test_websearch_service_prompts.py
  tldw_Server_API/tests/WebSearch/unit/test_aggregate_results_schema.py
  tldw_Server_API/tests/Writing/test_manuscript_annotation_review_jobs.py
  tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py
  tldw_Server_API/tests/integration/test_notes_auto_title_api.py
  tldw_Server_API/tests/unit/test_note_title_lib.py
  tldw_Server_API/tests/unit/test_note_title_service_prompts.py
)
python -m black --check "${PYTHON_CHECK_PATHS[@]}" "${PYTHON_TEST_PATHS[@]}"
python -m ruff check "${PYTHON_CHECK_PATHS[@]}" "${PYTHON_TEST_PATHS[@]}"
python -m mypy "${PYTHON_CHECK_PATHS[@]}" "${PYTHON_TEST_PATHS[@]}"

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
git diff --check
```

  No backend, WebUI, extension, shared-UI, Black, Ruff, mypy, format, lint, package-UI TypeScript, build, or diff gate may be waived.
- [ ] Update TASK-12959 with exact test/Bandit/build results, file list, approved inventory row changes, canary evidence, skips (none expected), and final summary. Mark complete only after every acceptance criterion and Definition of Done item is evidenced.
- [ ] Run the mandatory commit and final-domain gates once more and commit: `docs: complete documents web prompt migration (TASK-12959)`.

## Final implementation checklist

- [ ] Exactly 32 IDs are registered; no deferred/excluded/Scheduler ID was added.
- [ ] Every named direct/browser/Jobs consumer resolves or consumes one immutable atomic bundle at the inventory boundary.
- [ ] TASK-12957's shared chat bridge is reused unchanged for domain behavior; all four registry-owned merge policies were verified before Stage 2; no second endpoint or TypeScript resolver exists.
- [ ] Precedence, literal/template, empty/truthy, selector, locked assembly, roles/order, and no-override provider bytes are proven.
- [ ] Exact UTF-8 limits pass/fail at 65,536/65,537 and 262,144/262,145 with HTTP 413 or `service_prompt_size_limit_exceeded`, before side effects.
- [ ] Every direct FastAPI cluster proves authenticated-owner forwarding and two-user isolation; chat documents, insights, note create/suggest/bulk, research, and selected review prove exact 413 `service_prompt_size_limit_exceeded` before side effects.
- [ ] Note-title heuristic fallback is limited to provider failure, provider-configuration failure, and empty normalized output; every Service Prompt unavailable/quarantined/validation/size failure fails closed.
- [ ] `writing.annotation.scene` pins `{writing.annotation.scene}` at enqueue and consumes only verified WorkerSDK context before dispatch; ordinary payloads have no raw prompt bodies.
- [ ] Output parsers/normalizers and existing matrix-documented shaping are unchanged; no service-prompt budget truncation was added.
- [ ] Inventory/API/Prompts docs and TASK-12959 contain final evidence.
- [ ] Focused tests, real-server canary, Bandit, formatting/lint/build, `git diff --check`, and every mandatory full CI shard are green.
