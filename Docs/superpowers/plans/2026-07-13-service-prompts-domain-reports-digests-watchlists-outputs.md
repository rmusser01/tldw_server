# Reports, Digests, Watchlists, and Outputs Service Prompts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate all 21 approved report/output definitions through the shared resolver and protected Jobs path while leaving every broken or schema-dependent watchlist prompt deferred.

**Architecture:** Add the approved code-defined contracts to the existing registry, then migrate consumers in four coherent groups: synchronous backend outputs, three finite Jobs workflows, small browser outputs, and Research Studio work products. Backend boundaries resolve once and pass immutable bundles downward. The twelve browser definitions use TASK-12957's authenticated `service_prompt` extension on `/api/v1/chat/completions` and declare its registry-owned `replace_generated_messages` policy; clients send `messages: []` plus only IDs, finite selectors, explicit named parts, and runtime variables, and cannot select policy, roles, insertion, or ordering. Jobs pin complete render-ready bundles at enqueue and obtain only verified bundles from WorkerSDK context.

**Tech Stack:** Python Service Prompts registry/resolver, FastAPI, existing Jobs/WorkerSDK, React/TypeScript shared UI package, pytest/Hypothesis, Vitest.

---

**Backlog task:** `TASK-12961`

**Prerequisites:** Complete TASK-12960 and foundation plans 02–06. TASK-12957's browser execution bridge must exist in `chat_request_schemas.py`, `chat.py`, and `apps/packages/ui/src/services/service-prompts.ts`; plan 5's held-bind-release pinning and WorkerSDK guard must be available. Stop if either shared path is absent rather than implementing a second resolver, returning hidden parts to the browser, or placing prompt text in job payloads.

Before every commit below, satisfy the umbrella plan's mandatory per-commit gate. The requester's planning-time CI-shard waiver does not waive implementation gates.

## Stage map

| Stage | Tasks | Goal | Success criteria | Status |
| --- | --- | --- | --- | --- |
| 1. Contracts | Task 1 | Register exactly 21 approved definitions | exact-set contracts and complete-message Goldens pass | Not Started |
| 2. Server runtimes | Tasks 2–3 | Migrate synchronous consumers and three protected Jobs | one boundary resolution or verified job pin per execution; no lower-level lookup | Not Started |
| 3. Browser runtimes | Tasks 4–5 | Migrate title, Disco, and ten Research Studio outputs | TypeScript sends typed execution data only; all provider-message Goldens pass | Not Started |
| 4. Limits | Task 6 | Enforce approved UTF-8 budgets | all four boundaries and no-dispatch behavior pass | Not Started |
| 5. Release gate | Task 7 | Reconcile, document, secure, and verify the domain | full mandatory gates, Bandit, inventory validator, and Backlog finalization pass | Not Started |

## Mandatory gate before every commit

Focused red/green commands below supplement this gate; they never replace it. The requester's planning-time shard skip does not carry into implementation. Do not commit if a required command fails, including for an unrelated or environmental reason; diagnose under the repository's three-attempt rule, record the evidence in TASK-12961, and stop if the gate cannot be made green.

```bash
source .venv/bin/activate
python -m pytest -v
git diff --check
```

The complete-domain Python array below is the **final Task 7 gate only**, after every listed file has been created. Tasks 1–6 must instead run their task-local exact arrays below; never invoke a future task's not-yet-created path at an earlier commit.

```bash
source .venv/bin/activate
PYTHON_CHECK_PATHS=(
  tldw_Server_API/app/api/v1/endpoints/data_tables.py
  tldw_Server_API/app/api/v1/endpoints/flashcards.py
  tldw_Server_API/app/api/v1/endpoints/quizzes.py
  tldw_Server_API/app/api/v1/endpoints/research.py
  tldw_Server_API/app/api/v1/endpoints/slides.py
  tldw_Server_API/app/core/Data_Tables/jobs_worker.py
  tldw_Server_API/app/core/Flashcards/study_assistant.py
  tldw_Server_API/app/core/MCP_unified/modules/implementations/slides_module.py
  tldw_Server_API/app/core/Research/jobs.py
  tldw_Server_API/app/core/Research/jobs_worker.py
  tldw_Server_API/app/core/Research/providers/synthesis.py
  tldw_Server_API/app/core/Research/service.py
  tldw_Server_API/app/core/Service_Prompts/registry.py
  tldw_Server_API/app/core/Slides/slides_generator.py
  tldw_Server_API/app/core/StudyPacks/generation_service.py
  tldw_Server_API/app/core/WebSearch/Web_Search.py
  tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py
  tldw_Server_API/app/services/study_pack_jobs_worker.py
  tldw_Server_API/tests/Chat/test_service_prompt_execution.py
  tldw_Server_API/tests/DataTables/test_data_tables_jobs_integration.py
  tldw_Server_API/tests/DataTables/test_data_tables_worker.py
  tldw_Server_API/tests/Flashcards/test_study_assistant_service.py
  tldw_Server_API/tests/Jobs/test_reports_outputs_service_prompt_pins.py
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py
  tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py
  tldw_Server_API/tests/Research/test_research_jobs_service.py
  tldw_Server_API/tests/Research/test_research_jobs_worker.py
  tldw_Server_API/tests/Research/test_research_synthesizer.py
  tldw_Server_API/tests/Service_Prompts/test_registry.py
  tldw_Server_API/tests/Service_Prompts/test_reports_outputs_contracts.py
  tldw_Server_API/tests/Service_Prompts/test_reports_outputs_goldens.py
  tldw_Server_API/tests/Service_Prompts/test_reports_outputs_sync_integration.py
  tldw_Server_API/tests/Slides/test_slides_api.py
  tldw_Server_API/tests/Slides/test_slides_generator.py
  tldw_Server_API/tests/StudyPacks/test_generation_service.py
  tldw_Server_API/tests/StudyPacks/test_study_pack_jobs.py
  tldw_Server_API/tests/StudyPacks/test_study_pack_jobs_worker.py
  tldw_Server_API/tests/WebSearch/integration/test_websearch_endpoint.py
  tldw_Server_API/tests/WebSearch/unit/test_legacy_websearch_sanitizers.py
)
python -m black --check "${PYTHON_CHECK_PATHS[@]}"
python -m ruff check "${PYTHON_CHECK_PATHS[@]}"
python -m mypy "${PYTHON_CHECK_PATHS[@]}"
```

For a commit touching `apps/`, run every frontend shard and the shared-package type/build checks:

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

## Exact approved ID set

```text
chat.title.generation
data.table.generation
playground.disco.skill.comment
research.studio.compare
research.studio.corpus.gaps
research.studio.executive
research.studio.hypotheses
research.studio.literature.matrix
research.studio.proposal
research.studio.report
research.studio.slides
research.studio.summary
research.studio.timeline
research.synthesis.report
slides.deck.generation
study.assistant.explain
study.assistant.followup
study.assistant.freeform
study.assistant.mnemonic
study.pack.generation
web.search.report
```

The watchlist item/group/briefing/topic prompts remain deferred. Do not repair or register them here. Do not register any evaluation, routing, citation-verification, quiz, or core-Scheduler prompt.

## Task 1: Register all 21 contracts and exact packaged defaults

**Files:**

- Modify: `tldw_Server_API/app/core/Service_Prompts/registry.py`
- Modify: `tldw_Server_API/tests/Service_Prompts/test_registry.py`
- Create: `tldw_Server_API/tests/Service_Prompts/test_reports_outputs_contracts.py`
- Create: `tldw_Server_API/tests/Service_Prompts/test_reports_outputs_goldens.py`
- Reference: `Docs/Design/service-prompt-inventory.md`

- [ ] Add a failing exact-set test for the 21 IDs above and a negative test for every deferred watchlist ID/name. Confirm it fails with missing definitions, not by weakening catalog assertions.
- [ ] Add failing table-driven contract assertions for every ID's selector, provider-role order, part IDs, editability/visibility/mode, declared variables, deterministic truncations, finite selectors, output-contract evidence, and direct-versus-Jobs topology exactly as indexed in the inventory. Assert the twelve bridge definitions—`chat.title.generation`, `playground.disco.skill.comment`, and the ten `research.studio.*` IDs—declare exactly TASK-12957's registry-owned `replace_generated_messages` policy; request schemas cannot supply policy, roles, insertion, or ordering. The other nine definitions do not become chat-bridge definitions.
- [ ] Cover the notable multipart contracts explicitly: four independently selected study-assistant systems with one locked context carrier; Slides semantic prefix/schema/guidelines/style/title/source order; Data Tables guidance/carrier/JSON contract; Study Pack generation/JSON/title/request/hidden-source/card order; Research synthesis system/JSON/report/research order; Disco finite skill/outcome variants; and Research Studio JSON schemas/headings as locked parts rather than editable prose.
- [ ] Add failing no-override Goldens that compare the complete ordered provider-message arrays for all 21 definitions. For legacy `chat.title.generation`, normalize the source `{{query}}` marker to the constrained `{query}` contract while asserting byte-equivalent output. For every Studio definition, use the source's current 18,000-character source budget and 12,000-character compatible-artifact budget before the `G` check.
- [ ] Add table-driven precedence/provenance cases spanning file-backed and code-backed definitions: permitted explicit request part → approved user revision → configured deployment provider → packaged default. Assert deployment-provider failure is strict, literal explicit parts preserve braces byte-for-byte (including `research.studio.summary`'s `summaryInstruction`), and safe provenance exposes only definition/source/contract/revision-or-pin/digest metadata—never prompt text, runtime values, file paths, MACs, or keys.
- [ ] Run `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Service_Prompts/test_registry.py tldw_Server_API/tests/Service_Prompts/test_reports_outputs_contracts.py tldw_Server_API/tests/Service_Prompts/test_reports_outputs_goldens.py`; expected red result: all 21 IDs are unknown.
- [ ] Add only these approved definitions to the code-native registry using the packaged-default representation established by plan 3. Remove editable default ownership from consumers as each later task migrates it; locked schemas, evidence carriers, source labels, and repair prompts stay server-managed.
- [ ] Rerun the focused command and confirm exact ID set, contract metadata, and all complete-message Goldens pass.
- [ ] Run the exact Task 1 Python quality gate after its two new test files exist:

```bash
source .venv/bin/activate
TASK_PYTHON_PATHS=(
  tldw_Server_API/app/core/Service_Prompts/registry.py
  tldw_Server_API/tests/Service_Prompts/test_registry.py
  tldw_Server_API/tests/Service_Prompts/test_reports_outputs_contracts.py
  tldw_Server_API/tests/Service_Prompts/test_reports_outputs_goldens.py
)
python -m black --check "${TASK_PYTHON_PATHS[@]}"
python -m ruff check "${TASK_PYTHON_PATHS[@]}"
python -m mypy "${TASK_PYTHON_PATHS[@]}"
```

- [ ] Commit: `feat: register reports output service prompts (TASK-12961)`.

## Task 2: Migrate synchronous backend output boundaries

**Files:**

- Modify: `tldw_Server_API/app/core/Flashcards/study_assistant.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/flashcards.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/quizzes.py`
- Modify: `tldw_Server_API/app/core/Slides/slides_generator.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/slides.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/slides_module.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/research.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py`
- Modify: `tldw_Server_API/app/core/WebSearch/Web_Search.py`
- Modify: `tldw_Server_API/tests/Flashcards/test_study_assistant_service.py`
- Modify: `tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py`
- Modify: `tldw_Server_API/tests/Slides/test_slides_generator.py`
- Modify: `tldw_Server_API/tests/Slides/test_slides_api.py`
- Modify: `tldw_Server_API/tests/WebSearch/integration/test_websearch_endpoint.py`
- Modify: `tldw_Server_API/tests/WebSearch/unit/test_legacy_websearch_sanitizers.py`
- Create: `tldw_Server_API/tests/Service_Prompts/test_reports_outputs_sync_integration.py`

- [ ] Write failing study-assistant tests for each action (`explain`, `mnemonic`, `follow_up`, `freeform`) proving exactly one selected definition resolves for the authenticated owner, the common locked carrier is unchanged, `learner_message` remains literal runtime data, thread history is included only for follow-up, and provider temperature/max tokens stay `0.3`/`1000`.
- [ ] Write failing Slides tests proving endpoint and every MCP generation site pass one resolved `slides.deck.generation` bundle to `_call_llm`; optional title/style parts preserve current omission rules; system bytes remain semantic prefix → locked JSON/layout schema → guidelines → optional style carrier, followed by the user title/source carriers.
- [ ] Write failing web-report tests proving `/research/websearch` resolves `web.search.report` from its authenticated principal before executor dispatch, threads the immutable bundle through `analyze_and_aggregate` to `aggregate_results`, and both byte-identical legacy implementations send the exact system template plus `Follow the above instructions.` user message. Direct legacy calls with no owner deliberately resolve trusted server defaults.
- [ ] In the synchronous integration cases, prove approved-user/deployment-provider/packaged-default precedence and safe provenance at the authenticated boundary; explicit literal braces in study-assistant and Slides parts must reach provider messages unchanged.
- [ ] Run `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Service_Prompts/test_reports_outputs_sync_integration.py tldw_Server_API/tests/Flashcards/test_study_assistant_service.py tldw_Server_API/tests/Slides/test_slides_generator.py tldw_Server_API/tests/WebSearch/integration/test_websearch_endpoint.py`; expected red result: consumers still construct local strings and never call the resolver.
- [ ] At each public endpoint/MCP boundary create `PromptExecutionContext` with canonical owner, workflow, request/trace ID, and only declared explicit parts. Resolve once, then pass the immutable render-ready bundle through existing service calls. Do not put provider/model settings into prompt resolution.
- [ ] Replace local semantic constants with rendering of the resolved bundle. Preserve action/preset selection, source budgets, optional-carrier conditions, JSON response modes, output normalizers, fallback returns, and both web-search adapters. Do not resolve inside Slides loops or web-search chunk-summary loops.
- [ ] Rerun the focused sync suites and all Task 1 Goldens; confirm provider-message equality and one-resolution assertions pass.
- [ ] Run the exact Task 2 Python quality gate after `test_reports_outputs_sync_integration.py` exists:

```bash
source .venv/bin/activate
TASK_PYTHON_PATHS=(
  tldw_Server_API/app/core/Flashcards/study_assistant.py
  tldw_Server_API/app/api/v1/endpoints/flashcards.py
  tldw_Server_API/app/api/v1/endpoints/quizzes.py
  tldw_Server_API/app/core/Slides/slides_generator.py
  tldw_Server_API/app/api/v1/endpoints/slides.py
  tldw_Server_API/app/core/MCP_unified/modules/implementations/slides_module.py
  tldw_Server_API/app/api/v1/endpoints/research.py
  tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py
  tldw_Server_API/app/core/WebSearch/Web_Search.py
  tldw_Server_API/tests/Flashcards/test_study_assistant_service.py
  tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py
  tldw_Server_API/tests/Slides/test_slides_generator.py
  tldw_Server_API/tests/Slides/test_slides_api.py
  tldw_Server_API/tests/WebSearch/integration/test_websearch_endpoint.py
  tldw_Server_API/tests/WebSearch/unit/test_legacy_websearch_sanitizers.py
  tldw_Server_API/tests/Service_Prompts/test_reports_outputs_sync_integration.py
)
python -m black --check "${TASK_PYTHON_PATHS[@]}"
python -m ruff check "${TASK_PYTHON_PATHS[@]}"
python -m mypy "${TASK_PYTHON_PATHS[@]}"
```

- [ ] Commit: `feat: resolve synchronous report prompts (TASK-12961)`.

## Task 3: Pin the three finite Jobs definitions

**Files:**

- Modify: `tldw_Server_API/app/api/v1/endpoints/data_tables.py`
- Modify: `tldw_Server_API/app/core/Data_Tables/jobs_worker.py`
- Modify: `tldw_Server_API/app/core/Research/service.py`
- Modify: `tldw_Server_API/app/core/Research/jobs.py`
- Modify: `tldw_Server_API/app/core/Research/jobs_worker.py`
- Modify: `tldw_Server_API/app/core/Research/providers/synthesis.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/flashcards.py`
- Modify: `tldw_Server_API/app/services/study_pack_jobs_worker.py`
- Modify: `tldw_Server_API/app/core/StudyPacks/generation_service.py`
- Modify: `tldw_Server_API/tests/DataTables/test_data_tables_jobs_integration.py`
- Modify: `tldw_Server_API/tests/DataTables/test_data_tables_worker.py`
- Modify: `tldw_Server_API/tests/Research/test_research_jobs_service.py`
- Modify: `tldw_Server_API/tests/Research/test_research_jobs_worker.py`
- Modify: `tldw_Server_API/tests/Research/test_research_synthesizer.py`
- Modify: `tldw_Server_API/tests/StudyPacks/test_study_pack_jobs.py`
- Modify: `tldw_Server_API/tests/StudyPacks/test_study_pack_jobs_worker.py`
- Modify: `tldw_Server_API/tests/StudyPacks/test_generation_service.py`
- Create: `tldw_Server_API/tests/Jobs/test_reports_outputs_service_prompt_pins.py`

- [ ] Add failing producer tests for the exact finite requirement sets: Data Tables pins only `data.table.generation`; deep research synthesis pins only `research.synthesis.report`; Study Pack creation pins only `study.pack.generation`. Assert the owner/submission/set digest are bound before `queued` and ordinary payloads contain no prompt text, rendered values, MACs, or hidden source bundles.
- [ ] Add failing worker tests proving handlers are not entered for an absent/tampered/wrong-owner pin and that valid WorkerSDK context contains the verified immutable render-ready bundle selected by definition ID. Editing/resetting the user override after enqueue must not change provider messages.
- [ ] Cover approved-user/deployment-provider/packaged-default selection at enqueue and assert the verified WorkerSDK provenance is safe metadata only. The worker must never expose or log pinned template bodies, rendered runtime values, MACs, keys, or protected-store paths.
- [ ] Add complete worker Goldens: Data Tables keeps existing 24,000-character source truncation before `G`, JSON-object response mode, row/column normalizers, and max-row bound; Research synthesis retains 12-source/20-note bounds and independent validation; Study Pack keeps its source evidence bound, JSON/citation schema, requested title, first-attempt bundle, and separately locked repair path.
- [ ] Run `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Jobs/test_reports_outputs_service_prompt_pins.py`; expected red result: producers create ordinary queued jobs without pin requirements.
- [ ] Change the three producers to `ServicePromptJobPinner.enqueue(...)` with their exact finite declaration. Snapshot templates/assembly only; keep runtime documents, tables, sources, evidence, and request JSON in their existing protected/ordinary data boundaries rather than prompt components.
- [ ] In each handler, retrieve the definition from verified WorkerSDK context, render with job runtime values under the same strict renderer, and pass the complete messages to the existing provider call. Remove lower-level prompt reconstruction and never fall back to registry/user state from a worker.
- [ ] Rerun the new pin suite plus all listed domain worker suites and foundation pinning suites for SQLite/PostgreSQL.
- [ ] Run the exact Task 3 Python quality gate after the pin-suite file exists:

```bash
source .venv/bin/activate
TASK_PYTHON_PATHS=(
  tldw_Server_API/app/api/v1/endpoints/data_tables.py
  tldw_Server_API/app/core/Data_Tables/jobs_worker.py
  tldw_Server_API/app/core/Research/service.py
  tldw_Server_API/app/core/Research/jobs.py
  tldw_Server_API/app/core/Research/jobs_worker.py
  tldw_Server_API/app/core/Research/providers/synthesis.py
  tldw_Server_API/app/api/v1/endpoints/flashcards.py
  tldw_Server_API/app/services/study_pack_jobs_worker.py
  tldw_Server_API/app/core/StudyPacks/generation_service.py
  tldw_Server_API/tests/DataTables/test_data_tables_jobs_integration.py
  tldw_Server_API/tests/DataTables/test_data_tables_worker.py
  tldw_Server_API/tests/Research/test_research_jobs_service.py
  tldw_Server_API/tests/Research/test_research_jobs_worker.py
  tldw_Server_API/tests/Research/test_research_synthesizer.py
  tldw_Server_API/tests/StudyPacks/test_study_pack_jobs.py
  tldw_Server_API/tests/StudyPacks/test_study_pack_jobs_worker.py
  tldw_Server_API/tests/StudyPacks/test_generation_service.py
  tldw_Server_API/tests/Jobs/test_reports_outputs_service_prompt_pins.py
)
python -m black --check "${TASK_PYTHON_PATHS[@]}"
python -m ruff check "${TASK_PYTHON_PATHS[@]}"
python -m mypy "${TASK_PYTHON_PATHS[@]}"
```

- [ ] Commit: `feat: pin report output job prompts (TASK-12961)`.

## Task 4: Migrate title and Disco browser outputs through the shared bridge

**Files:**

- Modify: `apps/packages/ui/src/services/title.ts`
- Create: `apps/packages/ui/src/services/__tests__/title.service-prompts.test.ts`
- Modify: `apps/packages/ui/src/utils/disco-skill-check.ts`
- Modify: `apps/packages/ui/src/utils/__tests__/disco-skill-check.test.ts`
- Modify: `apps/packages/ui/src/components/Common/Playground/Message.tsx`
- Create: `apps/packages/ui/src/components/Common/Playground/__tests__/Message.disco-service-prompt.test.tsx`
- Modify: `apps/packages/ui/src/services/service-prompts.ts`
- Modify: `apps/packages/ui/src/services/__tests__/service-prompts.test.ts`
- Modify: `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`

- [ ] Add failing browser tests asserting title generation submits `chat.title.generation` with `{query}` and no raw default; Disco submits `playground.disco.skill.comment` with the finite skill ID/outcome and bounded assistant message, no raw persona/system/guidance text, and unchanged model/tool/temperature/token settings. Both calls send `messages: []`; their request objects contain no merge-policy, role, insertion-index, or ordering field.
- [ ] Add server-bridge Goldens for title's legacy `{{query}}` byte equivalence and Disco's pass/fail variants, 1,500-character truncation plus current ellipsis behavior, finite skill-catalog validation, system/user roles, and locked display contract. Assert the registry applies `replace_generated_messages` to the empty incoming array and that attempts to send policy/roles/order fail before resolution/provider dispatch. Unknown skill/outcome/variables must also fail before provider dispatch.
- [ ] Add browser-bridge precedence/provenance cases proving approved-user, deployment-provider, and packaged-default selection is performed from `AuthPrincipal`; responses and sanitized errors contain only safe provenance and never effective/hidden prompt bodies. Put literal braces in stored approved-user/deployment content for the declared `chat.title.generation` E-T template and Disco E-L guidance parts, then assert constrained-template braces are validated/rendered while literal guidance braces remain bytes. Do not add browser-facing explicit prompt overrides to either definition.
- [ ] From `apps/tldw-frontend`, run `bunx vitest run ../packages/ui/src/services/__tests__/title.service-prompts.test.ts ../packages/ui/src/utils/__tests__/disco-skill-check.test.ts ../packages/ui/src/components/Common/Playground/__tests__/Message.disco-service-prompt.test.tsx`; expected red result: raw prompt strings are still assembled client-side.
- [ ] Reuse the TASK-12957 request builder. Keep feature flags, fallback title, reasoning removal, trigger probability, skill selection, persistence, and response parsing unchanged. `disco-skill-check.ts` may return bounded runtime variables, but it must no longer own the editable prompt body.
- [ ] Rerun focused Vitest and `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Chat/test_service_prompt_execution.py -k 'title_generation or disco_skill'`.
- [ ] Run `source .venv/bin/activate && python -m black --check tldw_Server_API/tests/Chat/test_service_prompt_execution.py && python -m ruff check tldw_Server_API/tests/Chat/test_service_prompt_execution.py && python -m mypy tldw_Server_API/tests/Chat/test_service_prompt_execution.py` as the exact Task 4 Python quality gate, plus the mandatory frontend gate above.
- [ ] Commit: `feat: resolve browser output prompts server side (TASK-12961)`.

## Task 5: Migrate all ten Research Studio work products

**Files:**

- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/literature-workproducts.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx`
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.service-prompts.test.tsx`
- Modify: `apps/packages/ui/src/services/service-prompts.ts`
- Modify: `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`

- [ ] Add a failing exact selector table for summary/report/executive/timeline/compare/slides and literature matrix/corpus gaps/hypotheses/proposal. Assert each action sends its one exact definition ID; no aggregate Studio ID or dynamic user-derived ID is accepted.
- [ ] Add failing request-shape tests proving source excerpts are bounded to 18,000 characters and compatible artifacts to 12,000 before `G`; `research.studio.summary` sends nonblank `summaryInstruction` as the highest-precedence named **literal** explicit part; Slides fallback selects `research.studio.slides` only after the Slides API failure. Every Studio call sends `messages: []`, and no client request can contain merge policy, provider roles, insertion index, or message order.
- [ ] Add server Goldens for role/order, locked headings/schemas/source carriers, JSON response mode for matrix/gaps/hypotheses, optional compatible-artifact carriers, and no raw editable/default/hidden prompt bytes in browser requests or responses. Prove all ten registry definitions apply `replace_generated_messages` to the empty incoming array and reject caller-selected policy/roles/order before provider dispatch.
- [ ] Add explicit `research.studio.summary` cases where nonblank `summaryInstruction` contains `{braces}` and Unicode; assert it remains a literal part. Cover approved-user/deployment-provider/packaged-default precedence and safe provenance without returning bodies or runtime source excerpts.
- [ ] Run `bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.service-prompts.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx` from `apps/tldw-frontend`; expected red result: current builders return complete messages.
- [ ] Refactor the existing Studio builders to return typed finite selector/runtime-variable data only. The request builder supplies the ID/variables/explicit summary part to the existing chat completion call; the server resolver owns semantic and locked message assembly. Preserve source coverage, artifact persistence, traceability, JSON parsing/normalization, error states, and fallback sequencing.
- [ ] Run the focused Vitest files and `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Chat/test_service_prompt_execution.py -k research_studio`; confirm all ten provider-message Goldens pass.
- [ ] Run `source .venv/bin/activate && python -m black --check tldw_Server_API/tests/Chat/test_service_prompt_execution.py && python -m ruff check tldw_Server_API/tests/Chat/test_service_prompt_execution.py && python -m mypy tldw_Server_API/tests/Chat/test_service_prompt_execution.py` as the exact Task 5 Python quality gate, plus the mandatory frontend gate above.
- [ ] Commit: `feat: resolve research studio output prompts (TASK-12961)`.

## Task 6: Enforce every approved byte boundary and failure surface

**Files:**

- Modify: `tldw_Server_API/tests/Service_Prompts/test_reports_outputs_contracts.py`
- Modify: `tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py`
- Modify: `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`
- Modify: `tldw_Server_API/tests/Jobs/test_reports_outputs_service_prompt_pins.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/flashcards.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/quizzes.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/slides.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/research.py`
- Modify: `tldw_Server_API/tests/Flashcards/test_study_assistant_service.py`
- Modify: `tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py`
- Modify: `tldw_Server_API/tests/Slides/test_slides_api.py`
- Modify: `tldw_Server_API/tests/WebSearch/integration/test_websearch_endpoint.py`

- [ ] Parameterize the foundation byte-budget fixture over representative literal, constrained-template, direct-browser, synchronous-backend, and Jobs definitions from this domain. Use multibyte UTF-8 text. Assert authored parts and expanded variables/rendered text parts accept exactly 65,536 bytes and reject 65,537; authored definitions and final rendered bundles accept exactly 262,144 and reject 262,145. Exercise the aggregate boundary with a synthetic in-test five-part contract, counting deterministic separators/assembly bytes, so every individual part stays at or below 65,536 bytes; do not register another product definition.
- [ ] Assert settings API and browser chat execution overflow return `413` with `service_prompt_size_limit_exceeded` and zero provider calls; direct non-HTTP rendering exposes the exact same stable code; Jobs workers classify that deterministic size overflow as a permanent execution error without handler/provider dispatch.
- [ ] Add failing public FastAPI cases for both study-assistant response routes (`/flashcards/{card_uuid}/assistant/respond` and `/quizzes/attempts/{attempt_id}/questions/{question_id}/assistant/respond`), all five Slides generation routes (`/slides/generate` and its `from-chat`, `from-media`, `from-notes`, and `from-rag` variants), and `/research/websearch`. For rendered-part and aggregate overflow, require exact HTTP `413` with response code `service_prompt_size_limit_exceeded`, zero provider calls, and no assistant-message, presentation, cache, or other persistence. Add a narrow shared Service Prompt exception mapper before each endpoint's broad `500` handler; do not remap unrelated failures.
- [ ] Prove no new silent truncation. Preserve only matrix-declared pre-render source operations: Data Tables 24,000-character source cap; Disco 1,500-character message cap/ellipsis; Studio 18,000/12,000 source caps; Slides 200,000-character preflight and optional summarization; Study Pack per-source evidence cap; `study.assistant.explain`, `study.assistant.freeform`, and `study.assistant.mnemonic` `context_json` capped at 6,000 characters; and `study.assistant.followup`'s existing 6,000-character context/thread bound.
- [ ] Run `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Service_Prompts/test_reports_outputs_contracts.py tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py tldw_Server_API/tests/Chat/test_service_prompt_execution.py tldw_Server_API/tests/Jobs/test_reports_outputs_service_prompt_pins.py tldw_Server_API/tests/Flashcards/test_study_assistant_service.py tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py tldw_Server_API/tests/Slides/test_slides_api.py tldw_Server_API/tests/WebSearch/integration/test_websearch_endpoint.py`; confirm all limit, direct-endpoint `413`, no-side-effect, precedence, and provenance assertions pass.
- [ ] Before that commit, run the exact Task 6 Python quality gate over the files in this task:

```bash
source .venv/bin/activate
TASK_PYTHON_PATHS=(
  tldw_Server_API/tests/Service_Prompts/test_reports_outputs_contracts.py
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py
  tldw_Server_API/tests/Chat/test_service_prompt_execution.py
  tldw_Server_API/tests/Jobs/test_reports_outputs_service_prompt_pins.py
  tldw_Server_API/app/api/v1/endpoints/flashcards.py
  tldw_Server_API/app/api/v1/endpoints/quizzes.py
  tldw_Server_API/app/api/v1/endpoints/slides.py
  tldw_Server_API/app/api/v1/endpoints/research.py
  tldw_Server_API/tests/Flashcards/test_study_assistant_service.py
  tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py
  tldw_Server_API/tests/Slides/test_slides_api.py
  tldw_Server_API/tests/WebSearch/integration/test_websearch_endpoint.py
)
python -m black --check "${TASK_PYTHON_PATHS[@]}"
python -m ruff check "${TASK_PYTHON_PATHS[@]}"
python -m mypy "${TASK_PYTHON_PATHS[@]}"
```

- [ ] Commit: `test: enforce report output prompt byte budgets (TASK-12961)`.

## Task 7: Verify completeness, security, and documentation

**Files:**

- Modify: `Docs/Design/service-prompt-inventory.md`
- Modify: `Docs/API/service-prompts.md`
- Modify: `tldw_Server_API/Config_Files/Prompts/README.md`
- Modify: `tldw_Server_API/Config_Files/config.txt`
- Update through official Backlog MCP/CLI: `TASK-12961`

- [ ] Search every source string/constant and all 21 IDs. Confirm all approved consumers resolve through the shared boundary, no frontend file owns an editable default, workers use verified context only, and every remaining watchlist/output prompt is still explicitly deferred/excluded.
- [ ] Update only these 21 matrix rows with migrated call sites, registry contract versions, Golden test paths, and availability. Keep all watchlist rows deferred; this task does not repair their unsupported keyword dispatch.
- [ ] Document the 21 definitions, their synchronous/browser/Jobs execution topology, finite selectors, deployment-provider mappings, and three protected pin sets in `Docs/API/service-prompts.md`, `Config_Files/Prompts/README.md`, and the relevant commented prompt/config examples in `Config_Files/config.txt`. Do not document deferred watchlist definitions as available.
- [ ] Run the affected backend suites together with this copy-pasteable command: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Service_Prompts tldw_Server_API/tests/Chat/test_service_prompt_execution.py tldw_Server_API/tests/DataTables/test_data_tables_jobs_integration.py tldw_Server_API/tests/DataTables/test_data_tables_worker.py tldw_Server_API/tests/Research/test_research_jobs_service.py tldw_Server_API/tests/Research/test_research_jobs_worker.py tldw_Server_API/tests/Research/test_research_synthesizer.py tldw_Server_API/tests/StudyPacks/test_study_pack_jobs.py tldw_Server_API/tests/StudyPacks/test_study_pack_jobs_worker.py tldw_Server_API/tests/StudyPacks/test_generation_service.py tldw_Server_API/tests/Flashcards/test_study_assistant_service.py tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py tldw_Server_API/tests/Slides/test_slides_generator.py tldw_Server_API/tests/Slides/test_slides_api.py tldw_Server_API/tests/WebSearch/integration/test_websearch_endpoint.py tldw_Server_API/tests/WebSearch/unit/test_legacy_websearch_sanitizers.py tldw_Server_API/tests/Jobs/test_reports_outputs_service_prompt_pins.py tldw_Server_API/tests/Jobs/test_service_prompt_pinning_sqlite.py tldw_Server_API/tests/Jobs/test_service_prompt_pinning_postgres.py`.
- [ ] Run the affected browser tests with `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/__tests__/title.service-prompts.test.ts ../packages/ui/src/utils/__tests__/disco-skill-check.test.ts ../packages/ui/src/components/Common/Playground/__tests__/Message.disco-service-prompt.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.service-prompts.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx ../packages/ui/src/services/__tests__/service-prompts.test.ts`, then run the complete frontend gate above.
- [ ] Run `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Service_Prompts tldw_Server_API/app/core/Data_Tables tldw_Server_API/app/core/Research tldw_Server_API/app/core/StudyPacks tldw_Server_API/app/core/Flashcards/study_assistant.py tldw_Server_API/app/core/Slides tldw_Server_API/app/core/MCP_unified/modules/implementations/slides_module.py tldw_Server_API/app/core/WebSearch tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py tldw_Server_API/app/api/v1/endpoints/data_tables.py tldw_Server_API/app/api/v1/endpoints/flashcards.py tldw_Server_API/app/api/v1/endpoints/quizzes.py tldw_Server_API/app/api/v1/endpoints/research.py tldw_Server_API/app/api/v1/endpoints/slides.py tldw_Server_API/app/services/study_pack_jobs_worker.py -f json -o /tmp/bandit_task_12113_4.json`; review and fix every new finding.
- [ ] Run the mandatory gate above one final time, then run `node Helper_Scripts/validate_service_prompt_inventory.mjs .` from the repository root. Record its JSON counts/reference results, security review, touched files, and final summary in TASK-12961. If any mandatory command is unavailable or fails, stop under the three-attempt rule; do not record or use a waiver.
- [ ] Commit: `docs: complete report output prompt migration (TASK-12961)`.

## Stop conditions

- Stop if a browser path would require fetching hidden prompt bodies or reimplementing precedence/rendering in TypeScript.
- Stop if any of the three Jobs producers cannot declare its finite definition set before enqueue or any worker would need a live lower-level lookup.
- Stop if a structured output's independent response-format/normalization contract no longer holds after exact default-message substitution.
- Stop if a watchlist prompt appears necessary for this domain; repair and reclassify it in a separate human-reviewed task.
