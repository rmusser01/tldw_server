# User-Visible RAG Generation Service Prompts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate exactly `rag.client.answer` and `research.explainer.expansion` to the shared Service Prompt runtime while preserving no-override provider messages, client RAG retrieval behavior, and reproducible Explainer Jobs execution.

**Architecture:** Register only the two approved inventory definitions. Browser RAG callers send a typed execution request and a finite formatting-guide boolean through the authenticated `/api/v1/chat/completions` bridge established by TASK-12957; the server resolves once from the authenticated principal at provider dispatch, and TypeScript neither resolves prompts nor receives hidden parts. Explainer enqueue declares one finite requirement, pins the complete bundle before queue release, and its WorkerSDK handler consumes only the verified immutable worker context before provider dispatch. Both paths retain the shared precedence `explicit request -> verified job pin -> approved user revision -> deployment provider -> packaged default` and the shared renderer, budgets, and safe provenance.

**Tech Stack:** Existing Python Service Prompts registry/resolver and protected Jobs pinning, FastAPI/Pydantic chat execution bridge, Explainer Jobs/WorkerSDK, React/TypeScript shared UI package, pytest/Hypothesis, Vitest.

---

**Backlog task:** `TASK-12960`

**Governing artifacts:**

- `Docs/superpowers/specs/2026-07-12-user-customizable-service-prompts-design.md`
- `Docs/superpowers/plans/2026-07-12-user-customizable-service-prompts.md`
- `Docs/superpowers/plans/2026-07-12-service-prompts-03-registry-resolver.md`
- `Docs/superpowers/plans/2026-07-12-service-prompts-04-persistence-api-backup.md`
- `Docs/superpowers/plans/2026-07-12-service-prompts-05-protected-job-pinning.md`
- `Docs/Design/service-prompt-inventory.md`

## Prerequisites and hard boundaries

- Complete foundation plans 02-06 before this domain migration. Reuse their registry, resolver, strict source loader, persistence, FastAPI dependencies, error types, job pinner, protected store, and WorkerSDK context. Do not fork any of those abstractions.
- TASK-12957 must first land one optional typed `service_prompt` execution object on the existing `POST /api/v1/chat/completions` request in `tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py`, resolve it from `AuthPrincipal` in `tldw_Server_API/app/api/v1/endpoints/chat.py`, test it in `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`, and expose a request builder—not a resolver—in `apps/packages/ui/src/services/service-prompts.ts`.
- Before Task 2, verify the landed TASK-12957 bridge accepts the approved finite boolean selector for the locked-visible RAG formatting-guide suffix and preserves the current `appendSystemPromptSuffix(template, systemPromptAppendix)` result in the same final user message. The browser must never send caller-owned suffix text. If the boolean selector is absent, stop and amend TASK-12957's typed selector contract. Do not add a RAG-only endpoint, send hidden parts to the browser, split the suffix into another provider message, or install a TypeScript resolver.
- Resolve exactly once at the matrix boundary. The chat endpoint resolves once per chat request immediately before provider-message construction. Explainer enqueue resolves and pins the complete bundle once; its worker verifies and renders the pin without re-resolving current state.
- No-override provider message arrays must be byte-equivalent to today's arrays. Preserve each part's `literal`/`template` mode, placeholder count, role, order, and separators. Safe provenance may contain IDs, schema/contract versions, source kinds, revision/pin/default/content digests, and locked markers; it must not contain prompt text, variables, file paths, MACs, keys, or hidden content.
- Do not register or migrate retrieval/query rewrite, reranking, judge/evaluator, RAG routing, core Scheduler RAG templates, or any deferred/excluded candidate. In particular, leave `ragQuestionPrompt` and every retrieval/reranking key already in `rag.prompts.yaml` unchanged.

## Approved scope and exact contracts

| Definition | Exact bundle | Runtime variables and topology |
| --- | --- | --- |
| `rag.client.answer` | `user_template` (visible/editable template) then optional `formatting_guide_suffix` (locked/visible literal), selected only by a finite boolean; exactly one provider `user` message and no Service Prompt `system` message | `{context}` and `{question}`, each required and appearing exactly once; overflow rejects; direct authenticated chat request using TASK-12957's registry-owned `replace_copilot_user_text_preserve_non_text` policy |
| `research.explainer.expansion` | provider `system`: `system_semantics` (visible/editable literal) then `grounding_contract` (locked/visible literal); provider `user`: optional `expansion_semantics` (visible/editable literal) then `node_source_carrier` (locked/visible template) | `{session}`, `{node}`, `{source_excerpts}`, `{intent}`, and `{grounding}`, each required and appearing exactly once; finite Explainer Jobs requirement pinned at enqueue |

For Explainer, keep the packaged `expansion_semantics` empty and optional so the default user bytes do not change. The locked assembler adds `expansion_semantics + "\n"` only when that part is non-empty, followed by this carrier:

```text
{session}
Requested intent: {intent}
Grounding mode: {grounding}
{node}
Source context:
{source_excerpts}
```

`session` is the existing two-line `Session title`/`Depth preset` text, `node` is the existing three-line `Node title`/`Node body`/`Selected answer` text, and `source_excerpts` is the exact output of the current `_format_source_block`. This uses each approved variable once while reproducing `ExplainerPrompt.as_messages()` exactly.

For `rag.client.answer`, the locked suffix bytes are the finite `OUTPUT_FORMATTING_GUIDE_SYSTEM_PROMPT_SUFFIX` constant. Server assembly must reproduce today's operation order: select the effective raw `user_template`, trim it, append `"\n\n" + formatting_guide_suffix` only when the boolean is true and the exact suffix is not already contained, then render `{context}` and `{question}`. This preserves deduplication before runtime substitution. The legacy `useMessage.tsx` sidepanel path always selects `false`.

`rag.client.answer` uses the compatible registry policy `replace_copilot_user_text_preserve_non_text` established by TASK-12957. Each browser caller retains one final current-user message as the policy target. Its text is only the same current/effective question already supplied as the `question` variable—not a locally rendered Service Prompt—and the server replaces that text with the one rendered RAG answer message while preserving any authenticated non-text/image parts in order. History stays before that current-user message byte-for-byte. The request object never carries a policy, provider role, insertion position, or assembly order; those remain code-defined registry metadata.

## Mandatory gate before every commit

Focused red/green commands below supplement this gate; they never replace it. The requester's planning-time shard skip does not carry into implementation. Do not commit if any required shard fails, even for an apparently unrelated or environmental reason. Diagnose under the repository's three-attempt rule, record the evidence in TASK-12960, and stop if the gate cannot be made green.

- [ ] From the repository root, always run the complete backend suite:

```bash
source .venv/bin/activate
python -m pytest -v
git diff --check
```

- [ ] For a commit touching Python, run Black, Ruff, and mypy against both exact source and test arrays declared in that task's gate step. Do not shorten an array, omit tests, or substitute a directory-wide approximation.

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

- [ ] Before each commit, update TASK-12960 through the official Backlog MCP/CLI workflow with the current stage, touched files, red/green commands, full-gate results, blockers, and summary. Never hand-edit its task Markdown.

## Task 1: Register only the two RAG-generation contracts and lock default bytes

**Goal:** Make the shared registry and strict source loader aware of the two approved definitions without changing any provider message.

**Success criteria:** The registry exposes exactly these two IDs for this slice; contracts, compatibility mappings, roles, and Goldens pass; excluded/deferred IDs remain absent.

**Status:** Not Started

**Files:**

- Modify: `tldw_Server_API/app/core/Service_Prompts/registry.py`
- Modify: `tldw_Server_API/Config_Files/Prompts/rag.prompts.yaml`
- Create: `tldw_Server_API/Config_Files/Prompts/explainer.prompts.yaml`
- Modify: `tldw_Server_API/tests/Service_Prompts/test_registry.py`
- Modify: `tldw_Server_API/tests/Utils/test_prompt_loader_env_overrides.py`
- Create: `tldw_Server_API/tests/Service_Prompts/test_rag_generation_contracts.py`
- Reference: `Docs/Design/service-prompt-inventory.md`

- [ ] **Step 1 — Red: write exact registry, source, and provider-message tests.** Assert the slice ID set is exactly `{"rag.client.answer", "research.explainer.expansion"}`. Assert part IDs, visibility/editability, literal/template modes, role boundaries, assembly order, finite selectors, required variables, one occurrence per variable, `reject` overflow policy, environment mappings, and strict configured-source failure. Assert `formatting_guide_suffix` is locked-visible, optional, and selected only by a boolean. Assert `rag.client.answer` declares exactly TASK-12957's `replace_copilot_user_text_preserve_non_text` registry policy and cannot accept a policy/role/order override from request data. Assert no retrieval/reranking/judge/evaluator/Scheduler candidate is registered.

```python
assert render("rag.client.answer", context="CTX", question="Q") == [
    {
        "role": "user",
        "content": (
            "You are a helpful AI assistant. Use the following pieces of context to answer the "
            "question at the end. If you don't know the answer, just say you don't know. DO NOT "
            "try to make up an answer. If the question is not related to the context, politely "
            "respond that you are tuned to only answer questions that are related to the context.  "
            "CTX  Question: Q Helpful answer:"
        ),
    }
]
assert render("rag.client.answer", context="CTX", question="Q", formatting_guide=True) == [
    {"role": "user", "content": expected_rag_answer + "\n\n" + EXACT_FORMATTING_GUIDE}
]
assert render(
    "rag.client.answer",
    explicit_user_template=DEFAULT_RAG_TEMPLATE + "\n\n" + EXACT_FORMATTING_GUIDE,
    context="CTX",
    question="Q",
    formatting_guide=True,
) == [{"role": "user", "content": expected_rag_answer + "\n\n" + EXACT_FORMATTING_GUIDE}]
assert render_explainer(fixture) == build_node_expansion_prompt(**fixture).as_messages()
```

- [ ] **Step 2 — Confirm the red state.** Run:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Service_Prompts/test_registry.py \
  tldw_Server_API/tests/Service_Prompts/test_rag_generation_contracts.py \
  tldw_Server_API/tests/Utils/test_prompt_loader_env_overrides.py -k 'rag_client or explainer_expansion or rag_generation'
```

Expected failure: both definition lookups report unknown IDs and both new compatibility keys are unavailable; no pre-existing test should fail.

- [ ] **Step 3 — Green: add the minimum definitions and assets.** Append only a quoted `client_answer` scalar to `rag.prompts.yaml`; do not reformat or modify existing retrieval/reranking keys. Create `explainer.prompts.yaml` with only the editable packaged parts. Use strict compatibility mappings `TLDW_PROMPT_FILE_RAG__CLIENT_ANSWER` and `TLDW_PROMPT_FILE_EXPLAINER__SYSTEM_SEMANTICS` / `TLDW_PROMPT_FILE_EXPLAINER__EXPANSION_SEMANTICS`. Keep the locked grounding contract/carrier code-defined in the registry.

```yaml
# rag.prompts.yaml
client_answer: "You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say you don't know. DO NOT try to make up an answer. If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.  {context}  Question: {question} Helpful answer:"
```

```yaml
# explainer.prompts.yaml
system_semantics: "You expand a persisted research explainer tree. "
expansion_semantics: ""
```

The registry must use the shared model and renderer, for example:

```python
ServicePromptDefinition(
    id="rag.client.answer",
    parts=(
        template_part("user_template", variables=("context", "question")),
        locked_literal_part("formatting_guide_suffix", optional=True, visible=True),
    ),
    selectors=(boolean_selector("formatting_guide_suffix"),),
    provider_assembly=(user_message("user_template", optional_suffix="formatting_guide_suffix"),),
    message_policy="replace_copilot_user_text_preserve_non_text",
    contract_version=1,
)
```

- [ ] **Step 4 — Verify green.** Rerun the Step 2 command. Expected pass: exact IDs/contracts and both complete provider-message Goldens pass; configured nonblank missing/integrity-blocked files fail closed with sanitized configuration errors, while unset/blank mappings select packaged defaults.

- [ ] **Step 5 — Run the mandatory per-commit gate.** Use these exact Python arrays in addition to the full backend/frontend commands above, then record every result in TASK-12960:

```bash
source .venv/bin/activate
PYTHON_SOURCE_CHECK_PATHS=(
  tldw_Server_API/app/core/Service_Prompts/registry.py
)
PYTHON_TEST_CHECK_PATHS=(
  tldw_Server_API/tests/Service_Prompts/test_registry.py
  tldw_Server_API/tests/Utils/test_prompt_loader_env_overrides.py
  tldw_Server_API/tests/Service_Prompts/test_rag_generation_contracts.py
)
python -m black --check "${PYTHON_SOURCE_CHECK_PATHS[@]}" "${PYTHON_TEST_CHECK_PATHS[@]}"
python -m ruff check "${PYTHON_SOURCE_CHECK_PATHS[@]}" "${PYTHON_TEST_CHECK_PATHS[@]}"
python -m mypy "${PYTHON_SOURCE_CHECK_PATHS[@]}" "${PYTHON_TEST_CHECK_PATHS[@]}"
```

- [ ] **Step 6 — Commit only after every gate is green:** `feat: register RAG generation service prompts (TASK-12960)`.

## Task 2: Route every client RAG answer through the authenticated shared bridge

**Goal:** Stop formatting the answer service prompt in the browser while preserving retrieval, query rewrite, chat history, multimodal branches, local-setting precedence, and exact final provider messages.

**Success criteria:** All four active consumers send a typed execution request plus one current-user carrier; the authenticated server resolves once and replaces only that carrier's text under the landed registry policy; history/non-text parts remain byte-equivalent; the browser contains no resolver/hidden parts; explicit local templates remain highest; no-override provider arrays are byte-equivalent.

**Status:** Not Started

**Files:**

- Modify: `apps/packages/ui/src/services/tldw-server.ts`
- Modify: `apps/packages/ui/src/services/service-prompts.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwChat.ts`
- Modify: `apps/packages/ui/src/models/ChatTldw.ts`
- Modify: `apps/packages/ui/src/models/index.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/ragMode.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/documentChatMode.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/tabChatMode.ts`
- Modify: `apps/packages/ui/src/hooks/useMessage.tsx`
- Modify: `apps/packages/ui/src/services/__tests__/service-prompts.test.ts`
- Modify: `apps/packages/ui/src/services/__tests__/tldw-chat.message-sanitization.test.ts`
- Modify: `apps/packages/ui/src/models/__tests__/ChatTldw.stream-metadata.test.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/__tests__/ragMode.sanitization.test.ts`
- Create: `apps/packages/ui/src/hooks/chat-modes/__tests__/rag-client-answer.service-prompts.test.ts`
- Create: `apps/packages/ui/src/hooks/__tests__/useMessage.rag-service-prompt.guard.test.ts`
- Modify: `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`
- Reference prerequisite: `tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py`
- Reference prerequisite: `tldw_Server_API/app/api/v1/endpoints/chat.py`

- [ ] **Step 1 — Verify the prerequisite instead of coding around it.** Read TASK-12957 and its landed bridge tests. Confirm its request object is optional, `extra="forbid"`, uses the authenticated `AuthPrincipal`, accepts named variables, allowed explicit parts, and allowlisted finite selectors, resolves once immediately before provider dispatch, returns no prompt body, and maps shared size errors to `413`. Confirm its registry-owned `replace_copilot_user_text_preserve_non_text` dispatcher replaces current-user text while preserving history and authenticated non-text/image parts in order, and that selector values cannot carry arbitrary text. If any condition is absent, stop; update the prerequisite task/plan rather than creating a second execution path here.

- [ ] **Step 2 — Red: write transport and four-consumer tests.** For RAG mode, document mode, text tab mode, and legacy sidepanel website mode, assert:

  - `service_prompt.definition_id === "rag.client.answer"`;
  - variables are exactly `{context, question}` and raw context/question do not appear in a client-rendered service-prompt message;
  - a nonblank `systemPromptForRag` becomes only the explicit `user_template` with `kind: "template"`; blank/missing local storage sends no explicit part, allowing approved-user/deployment/packaged precedence;
  - an explicit constrained `user_template` containing `{{literal brace}}` plus `{context}` and `{question}` exactly once renders the escaped braces as literal `{literal brace}` bytes, while `{`/`}` inside the runtime context and question values remain data and are never reparsed as template syntax;
  - the client request contains no policy, provider-role, insertion-position, or assembly-order field; `replace_copilot_user_text_preserve_non_text`, the single final `user` role, and its position after history come only from registry metadata;
  - `humanMessage` remains the final current-user carrier: its text is the same current/effective question supplied in `variables.question`, never a rendered/default/local RAG template, and any authenticated non-text/image parts retain their exact bytes and order;
  - `ragQuestionPrompt` query rewrite, retrieval calls, source shaping, actor/history messages, provider/model/tool options, and save behavior remain unchanged;
  - the three pipeline modes send only the formatting-guide boolean; the current locked suffix lands in the same rendered final user message, after `Helpful answer:`, with the same two-newline separator;
  - enabled, disabled, and an explicit/stored template already containing the exact suffix cover current trim/dedup behavior; duplicate suffix bytes are never emitted;
  - legacy `useMessage.tsx` sends no suffix selector (equivalent to `false`), because that path never appended the guide;
  - the tab image branch remains its existing question-plus-image request and does not also apply `rag.client.answer`;
  - the existing document `.substring(0, maxContextSize)`, tab `getTabContents` cap, and legacy HTML/PDF `.slice(0, maxWebsiteContext)` occur before the shared UTF-8 check and no new truncation is added.

Add backend bridge cases for explicit → approved user → deployment → packaged precedence, cross-user isolation, exact complete provider arrays, escaped literal braces in the explicit constrained template, braces in runtime context/question values as data, suffix on/off/dedup, legacy-no-suffix, unknown variable/part/selector and non-boolean selector `422`, and no provider call on resolution failure. For every one of the four consumers, cover empty and multi-turn histories and compare the complete provider array: every prior system/user/assistant/tool or actor-injected message is byte-identical and in the same order, followed by exactly one current `user` message whose text is the server-rendered RAG answer; no raw carrier text survives replacement. Add a compatible non-text carrier fixture and prove only its text is replaced while image/non-text parts and order remain exact. Assert the bridge's safe-provenance keys are exactly the applicable subset of definition ID, schema/contract version, per-part source kinds, explicit/user-revision identifiers or digests, trusted server-default bundle digest, canonical content digest, assembly order, and locked-section markers. Assert provenance, response, errors, and logs contain no prompt/rendered text, raw context/question values, deployment or asset/file paths, owner credentials, MAC/authentication tags, keys, or hidden content.

- [ ] **Step 3 — Confirm the red state.** Run:

```bash
cd apps/tldw-frontend
bunx vitest run \
  ../packages/ui/src/services/__tests__/service-prompts.test.ts \
  ../packages/ui/src/services/__tests__/tldw-chat.message-sanitization.test.ts \
  ../packages/ui/src/models/__tests__/ChatTldw.stream-metadata.test.ts \
  ../packages/ui/src/hooks/chat-modes/__tests__/ragMode.sanitization.test.ts \
  ../packages/ui/src/hooks/chat-modes/__tests__/rag-client-answer.service-prompts.test.ts \
  ../packages/ui/src/hooks/__tests__/useMessage.rag-service-prompt.guard.test.ts
cd ../..
source .venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Chat/test_service_prompt_execution.py -k rag_client_answer
```

Expected failure: the UI still calls `.replace("{context}", ...)`/`.replace("{question}", ...)`, `ChatCompletionRequest` has no propagated domain request, and the backend bridge test has no `rag.client.answer` provider-message case.

- [ ] **Step 4 — Green: thread the existing typed request, not prompt content.** Keep `promptForRag()`'s legacy query-rewrite result, but expose the nonblank local answer value separately from its UI-display fallback so a packaged fallback is never mislabeled as an explicit override. Add a narrow builder around TASK-12957's type:

```ts
const servicePrompt = buildServicePromptExecution({
  definitionId: "rag.client.answer",
  variables: { context, question },
  explicitParts: localAnswerTemplate
    ? { user_template: { kind: "template", content: localAnswerTemplate } }
    : undefined,
  selectors: { formatting_guide_suffix: formattingGuideEnabled }
})
```

Use the exact allowlisted boolean-selector field name landed by TASK-12957; it must carry only `true|false`, never `OUTPUT_FORMATTING_GUIDE_SYSTEM_PROMPT_SUFFIX` text. Do not add an untyped `extra_body` escape hatch. Add `servicePrompt?: ServicePromptExecutionRequest` to `ChatModePrompt` and `PageAssistModelOptions`, then pass it unchanged through `ChatTldwOptions`, `TldwChatOptions`, and `ChatCompletionRequest.service_prompt`. Do not inspect or render it in those layers.

In the three pipeline modes, retain context construction and sources, but replace local RAG-template formatting with a minimal `humanMessage` current-user carrier. Its text is `ctx.message`, matching `variables.question`; if a compatible branch already has authenticated non-text/image parts, retain them unchanged. `chatModePipeline` sends unchanged history, that final carrier, and the execution object; the server's registry policy replaces only the carrier text with the rendered RAG answer. In `useMessage.tsx`, build the same request after retrieval, retain its final current-user carrier with the effective rewritten `query` matching `variables.question`, and pass the request when constructing the final `pageAssistModel`; leave its separate question-rewrite model call untouched. The existing tab image branch still bypasses `rag.client.answer` entirely. Fail closed with the bridge's sanitized error if prompt execution fails—never silently fall back to the browser default.

- [ ] **Step 5 — Verify green.** Rerun the Step 3 commands. Expected pass: all four consumers propagate one typed request without policy/role/order fields plus one current-user carrier, the server resolves once from the principal, local explicit/no-explicit cases select the right source, exact provider arrays preserve every history/non-text byte and replace only the final carrier text (including suffix behavior), and no prompt content is returned by the bridge.

- [ ] **Step 6 — Run the mandatory per-commit gate.** Run the full backend gate even though this stage is frontend-heavy, then every `apps/` shard plus format/lint/type/build. Run Black, Ruff, and mypy on the exact Python test changed by this task:

```bash
source .venv/bin/activate
PYTHON_SOURCE_CHECK_PATHS=()
PYTHON_TEST_CHECK_PATHS=(
  tldw_Server_API/tests/Chat/test_service_prompt_execution.py
)
python -m black --check "${PYTHON_TEST_CHECK_PATHS[@]}"
python -m ruff check "${PYTHON_TEST_CHECK_PATHS[@]}"
python -m mypy "${PYTHON_TEST_CHECK_PATHS[@]}"
```

Record all results in TASK-12960; no shard may be waived. `chat_request_schemas.py` and `chat.py` remain TASK-12957-owned prerequisites, so this task's source array is intentionally empty; any bridge correction is completed and gated under TASK-12957 before this task resumes.

- [ ] **Step 7 — Commit only after every gate is green:** `feat: resolve client RAG answers through service prompts (TASK-12960)`.

## Task 3: Pin the complete Explainer bundle and consume verified WorkerSDK context

**Goal:** Make each node-expansion job reproducible and fail closed before generation when its authenticated prompt pin is missing or invalid.

**Success criteria:** Enqueue pins the one finite complete bundle before queue release; ordinary payloads contain no prompt; the worker uses only verified context; default messages and Explainer normalization/citation behavior stay unchanged.

**Status:** Not Started

**Files:**

- Modify: `tldw_Server_API/app/api/v1/API_Deps/service_prompt_deps.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/explainer.py`
- Modify: `tldw_Server_API/app/core/Explainer/service.py`
- Modify: `tldw_Server_API/app/core/Explainer/jobs.py`
- Modify: `tldw_Server_API/app/core/Explainer/jobs_worker.py`
- Modify: `tldw_Server_API/app/core/Explainer/prompting.py`
- Modify: `tldw_Server_API/tests/Explainer/test_explainer_endpoints.py`
- Modify: `tldw_Server_API/tests/Explainer/test_explainer_jobs.py`
- Create: `tldw_Server_API/tests/Explainer/test_explainer_service_prompts.py`
- Modify: `tldw_Server_API/tests/Jobs/test_service_prompt_worker_guard.py`

- [ ] **Step 1 — Red: test held-bind-release enqueue.** Inject the foundation `ServicePromptJobPinner` through `service_prompt_deps.py` and the endpoint. Assert the declaration is the immutable exact tuple `("research.explainer.expansion",)`, a complete atomic bundle is pinned before `held -> queued`, ownership comes from `CurrentPrincipal` rather than request/job payload fields, and idempotent reuse requires matching owner/submission/set digest/binding. Assert queue/pin failure does not mark the Explainer node queued.

Parameterize enqueue over approved-user revision → deployment provider → packaged default selection with no explicit override. For each source, assert the complete selected editable bundle is pinned once with the expected per-part source kinds and digests before release. Put literal `{stored brace}` and `{deployment brace}` text in the approved and deployment `system_semantics`/`expansion_semantics` E-L parts and prove those braces remain byte-for-byte literal through pinning and worker rendering rather than becoming template syntax.

```python
assert call.prompt_requirements == ("research.explainer.expansion",)
assert call.execution_context.owner_user_id == authenticated_principal.user_id
assert not ({"prompt", "system", "user", "parts", "mac", "auth_tag"} & call.job_payload.keys())
```

- [ ] **Step 2 — Red: test verified worker-only execution.** Cover valid pins from each approved-user/deployment/packaged enqueue case, missing worker context, missing definition, wrong owner/submission/job binding, tampered component/set/binding MAC, digest/contract mismatch, unavailable retained key, transient protected-store failure, and `bypass_stored_overrides`. In every invalid/permanent case assert the handler/generator is never called and the exact foundation failure classification is used; transient store failure is retryable, and a blocked stored-override pin is held without substitution. Prove editing/resetting/acknowledging the active revision after enqueue does not change the provider messages.

Assert WorkerSDK context and persisted generation metadata expose exactly the applicable safe provenance: definition ID, schema/contract version, per-part source kinds, approved-revision or pin/snapshot identifiers and digests, trusted server-default bundle digest, canonical content digest, assembly order, and locked-section markers, plus the existing `promptTemplateVersion`. Assert worker context serialization, generation metadata, job status/events, errors, and logs contain no prompt/rendered text, runtime session/node/source/intent/grounding values, deployment or asset/file paths, owner credentials, MAC/authentication tags, keys, or hidden content.

Also retain Goldens for source-only/no-evidence total fallback, malformed/empty child normalization to `Insufficient source evidence`, citation filtering against authoritative selected context, and atomic persistence. These are locked domain behavior and must not become editable prompt contracts.

- [ ] **Step 3 — Confirm the red state.** Run:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Explainer/test_explainer_service_prompts.py \
  tldw_Server_API/tests/Explainer/test_explainer_jobs.py \
  tldw_Server_API/tests/Explainer/test_explainer_endpoints.py \
  tldw_Server_API/tests/Jobs/test_service_prompt_worker_guard.py -k 'explainer or research_explainer_expansion'
```

Expected failure: enqueue still calls `JobManager.create_job` directly, `build_node_expansion_prompt` still owns literal prompt text, and the handler can dispatch generation without a verified Service Prompt worker context.

- [ ] **Step 4 — Green: replace direct enqueue with the shared pinner.** Keep the ordinary domain payload limited to `session_id`, `node_id`, `intent`, and `answer_revision`; let the shared pinner add only its safe pin-set UUID/submission ID/set digest references. The minimal domain declaration is:

```python
EXPLAINER_SERVICE_PROMPT_REQUIREMENTS = ("research.explainer.expansion",)

return prompt_job_pinner.enqueue(
    requirements=EXPLAINER_SERVICE_PROMPT_REQUIREMENTS,
    execution_context=prompt_execution_context,
    domain=EXPLAINER_DOMAIN,
    queue=EXPLAINER_QUEUE,
    job_type=EXPLAINER_JOB_TYPE,
    payload=payload,
    owner_user_id=owner_user_id,
    idempotency_key=idempotency_key,
)
```

Use the exact Plan 5 method/type names that landed; do not wrap them in a second Explainer pin store. The route must use `CurrentPrincipal` and the shared pinner dependency. Preserve existing priority, retries, idempotency key, terminal-row handling, and node-state transitions.

- [ ] **Step 5 — Green: render only the verified worker bundle.** Update the concrete WorkerSDK handler to accept the immutable verified context supplied by Plan 5. Require `research.explainer.expansion` from that context before loading a generator. After loading and ownership-validating session/node/source data, build only runtime variables, render with the shared renderer, and adapt the two rendered messages to `ExplainerPrompt` for the unchanged generator.

```python
verified = worker_context.service_prompts.require("research.explainer.expansion")
prompt = explainer_prompt_from_verified_bundle(
    verified,
    session=format_session(session),
    node=format_node(node),
    source_excerpts=_format_source_block(source_context),
    intent=effective_intent,
    grounding=session.grounding,
)
generation = await _call_generator(generator, prompt)
```

`build_node_expansion_prompt` may remain only as a deterministic compatibility/Golden helper over the packaged bundle; the production job path must never resolve current state or reconstruct editable/locked prompt bytes itself. Preserve `promptTemplateVersion` and add only safe Service Prompt provenance (`definitionId`, contract/schema version, source kinds, revision/pin/default/content digests) to generation metadata. Never persist prompt bodies, rendered variables, asset paths, MACs, keys, or hidden content.

- [ ] **Step 6 — Verify green.** Rerun the Step 3 command, then run both protected-store backends:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Jobs/test_service_prompt_pinning_sqlite.py \
  tldw_Server_API/tests/Jobs/test_service_prompt_pinning_postgres.py \
  tldw_Server_API/tests/Jobs/test_service_prompt_worker_guard.py
```

Expected pass: a complete verified pin reaches the generator byte-for-byte, invalid pins never dispatch, PostgreSQL/SQLite behavior matches, and ordinary jobs retain the unchanged fast path.

- [ ] **Step 7 — Run the mandatory per-commit gate.** Use these exact source/test arrays for Black, Ruff, and mypy, then record focused, PostgreSQL, and full-backend results in TASK-12960:

```bash
source .venv/bin/activate
PYTHON_SOURCE_CHECK_PATHS=(
  tldw_Server_API/app/api/v1/API_Deps/service_prompt_deps.py
  tldw_Server_API/app/api/v1/endpoints/explainer.py
  tldw_Server_API/app/core/Explainer/service.py
  tldw_Server_API/app/core/Explainer/jobs.py
  tldw_Server_API/app/core/Explainer/jobs_worker.py
  tldw_Server_API/app/core/Explainer/prompting.py
)
PYTHON_TEST_CHECK_PATHS=(
  tldw_Server_API/tests/Explainer/test_explainer_endpoints.py
  tldw_Server_API/tests/Explainer/test_explainer_jobs.py
  tldw_Server_API/tests/Explainer/test_explainer_service_prompts.py
  tldw_Server_API/tests/Jobs/test_service_prompt_worker_guard.py
)
python -m black --check "${PYTHON_SOURCE_CHECK_PATHS[@]}" "${PYTHON_TEST_CHECK_PATHS[@]}"
python -m ruff check "${PYTHON_SOURCE_CHECK_PATHS[@]}" "${PYTHON_TEST_CHECK_PATHS[@]}"
python -m mypy "${PYTHON_SOURCE_CHECK_PATHS[@]}" "${PYTHON_TEST_CHECK_PATHS[@]}"
```

- [ ] **Step 8 — Commit only after every gate is green:** `feat: pin Explainer expansion prompts (TASK-12960)`.

## Task 4: Enforce exact UTF-8 budgets and no-dispatch failures

**Goal:** Prove authored, variable, rendered-part, definition, and final-bundle limits at both direct chat and protected-job execution surfaces.

**Success criteria:** Every exact boundary is tested in UTF-8 bytes; overflow never truncates or reaches a provider; APIs return `413`; non-HTTP execution uses one exact stable code.

**Status:** Not Started

**Files:**

- Modify only if the red tests expose a shared foundation defect: `tldw_Server_API/app/core/Service_Prompts/models.py`
- Modify only if the red tests expose a shared foundation defect: `tldw_Server_API/app/core/Service_Prompts/templates.py`
- Modify only if the red tests expose a shared foundation defect: `tldw_Server_API/app/api/v1/endpoints/service_prompts.py`
- Modify only if the red tests expose a shared foundation defect: `tldw_Server_API/app/api/v1/endpoints/chat.py`
- Modify only if the red tests expose a shared foundation defect: `tldw_Server_API/app/core/Jobs/worker_sdk.py`
- Modify: `tldw_Server_API/tests/Service_Prompts/test_templates.py`
- Modify: `tldw_Server_API/tests/Service_Prompts/test_rag_generation_contracts.py`
- Modify: `tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py`
- Modify: `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`
- Modify: `tldw_Server_API/tests/Explainer/test_explainer_service_prompts.py`
- Modify: `tldw_Server_API/tests/Jobs/test_service_prompt_worker_guard.py`
- Modify: `apps/packages/ui/src/services/__tests__/service-prompts.test.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/__tests__/rag-client-answer.service-prompts.test.ts`

- [ ] **Step 1 — Red: add byte-exact authored and runtime cases.** Use multibyte input so the tests cannot accidentally count code points. A useful exact value is `"é" * 32_768`, which is 65,536 UTF-8 bytes; append `"x"` for 65,537. Assert:

  - authored part: 65,536 succeeds, 65,537 rejects;
  - each expanded/runtime variable: 65,536 succeeds, 65,537 rejects;
  - each expanded/rendered part: 65,536 succeeds, 65,537 rejects;
  - authored definition: 262,144 succeeds, 262,145 rejects;
  - final rendered bundle: 262,144 succeeds, 262,145 rejects.

Use a synthetic in-test five-part definition/bundle fixture to isolate the 262,144/262,145 aggregate boundary while every constituent remains below 65,536. Do not register a third product definition or add it to the catalog. For `rag.client.answer`, construct valid templates that still contain `{context}` and `{question}` exactly once when testing authored sizes.

```python
EXACT_PART = "é" * 32_768
OVER_PART = EXACT_PART + "x"
assert len(EXACT_PART.encode("utf-8")) == 65_536
assert len(OVER_PART.encode("utf-8")) == 65_537
```

- [ ] **Step 2 — Red: assert surface behavior and no dispatch.** The settings/preview API and chat execution bridge must return HTTP `413` with machine code `service_prompt_size_limit_exceeded` and sanitized part/limit metadata. Direct renderer/worker failures must expose the exact same non-HTTP code `service_prompt_size_limit_exceeded`. Assert provider/generator mocks have zero calls, logs/status contain no content, and retry classification follows the foundation (deterministic size overflow is permanent).

Assert no new code uses `slice`, `substring`, byte clipping, or ellipsis to make an oversized prompt pass. The only accepted truncations are pre-existing inventory-documented source caps in `documentChatMode.ts`, `get-tab-contents.ts`, and legacy `useMessage.tsx`, performed before the renderer's `G` check. If the foundation landed any different code or retained an alias, stop Task 4 and normalize the foundation plus every foundation/domain assertion to the exact canonical code `service_prompt_size_limit_exceeded` before continuing. Do not adapt this domain to another code or introduce an alias.

- [ ] **Step 3 — Confirm the red state.** Run:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Service_Prompts/test_templates.py \
  tldw_Server_API/tests/Service_Prompts/test_rag_generation_contracts.py \
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py \
  tldw_Server_API/tests/Chat/test_service_prompt_execution.py \
  tldw_Server_API/tests/Explainer/test_explainer_service_prompts.py \
  tldw_Server_API/tests/Jobs/test_service_prompt_worker_guard.py -k '65536 or 65537 or 262144 or 262145 or size_limit'
cd apps/tldw-frontend
bunx vitest run \
  ../packages/ui/src/services/__tests__/service-prompts.test.ts \
  ../packages/ui/src/hooks/chat-modes/__tests__/rag-client-answer.service-prompts.test.ts
```

Expected failure: at least one surface lacks exact boundary/error propagation assertions; no test may pass because a string was silently truncated.

- [ ] **Step 4 — Green: reuse shared validators and error mapping only.** If an uncovered boundary is real, change only the five explicitly listed shared source files whose layer owns that boundary; otherwise change tests only. Domain consumers must not implement their own counters or error names. Count `len(value.encode("utf-8"))` in the shared validator before rendering/assembly, and recheck rendered parts/final bundle after substitution/assembly.

- [ ] **Step 5 — Verify green.** Rerun Step 3 without `-k` for every listed Python test file. Expected pass: all ten exact allow/reject cases, `413` mappings, stable non-HTTP code, sanitized errors, and zero provider/generator dispatch assertions pass.

- [ ] **Step 6 — Run the mandatory per-commit gate.** Run the full backend suite and every `apps/` shard plus format/lint/type/build. Run Black, Ruff, and mypy against these exact potential shared-foundation source paths and exact changed tests; checking an unchanged potential source is intentional and avoids a conditional placeholder:

```bash
source .venv/bin/activate
PYTHON_SOURCE_CHECK_PATHS=(
  tldw_Server_API/app/core/Service_Prompts/models.py
  tldw_Server_API/app/core/Service_Prompts/templates.py
  tldw_Server_API/app/api/v1/endpoints/service_prompts.py
  tldw_Server_API/app/api/v1/endpoints/chat.py
  tldw_Server_API/app/core/Jobs/worker_sdk.py
)
PYTHON_TEST_CHECK_PATHS=(
  tldw_Server_API/tests/Service_Prompts/test_templates.py
  tldw_Server_API/tests/Service_Prompts/test_rag_generation_contracts.py
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py
  tldw_Server_API/tests/Chat/test_service_prompt_execution.py
  tldw_Server_API/tests/Explainer/test_explainer_service_prompts.py
  tldw_Server_API/tests/Jobs/test_service_prompt_worker_guard.py
)
python -m black --check "${PYTHON_SOURCE_CHECK_PATHS[@]}" "${PYTHON_TEST_CHECK_PATHS[@]}"
python -m ruff check "${PYTHON_SOURCE_CHECK_PATHS[@]}" "${PYTHON_TEST_CHECK_PATHS[@]}"
python -m mypy "${PYTHON_SOURCE_CHECK_PATHS[@]}" "${PYTHON_TEST_CHECK_PATHS[@]}"
```

Record results in TASK-12960; full shards cannot be waived.

- [ ] **Step 7 — Commit only after every gate is green:** `test: verify RAG generation prompt limits (TASK-12960)`.

## Task 5: Prove domain completeness, secure the touched scope, and document rollout

**Goal:** Close the migration with auditable inventory evidence, security checks, complete test shards, and current Backlog records.

**Success criteria:** Every eligible call site uses the shared path; every remaining RAG/research prompt stays explicitly excluded/deferred/locked; Bandit and all gates are green; docs and TASK-12960 record the result.

**Status:** Not Started

**Files:**

- Modify: `Docs/Design/service-prompt-inventory.md`
- Modify: `Docs/API/service-prompts.md`
- Modify: `tldw_Server_API/Config_Files/Prompts/README.md`
- Modify: `tldw_Server_API/Config_Files/config.txt`
- Update through Backlog MCP/CLI: `backlog/tasks/task-12113.3 - Migrate-user-visible-RAG-generation-service-prompts.md`

- [ ] **Step 1 — Audit exact migration coverage.** Search the two legacy defaults, `promptForRag`, local `{context}`/`{question}` replacement, `build_node_expansion_prompt`, direct Explainer `create_job`, and Explainer WorkerSDK registration. Confirm the only product registry IDs added by this task are `rag.client.answer` and `research.explainer.expansion`, every active consumer named by the matrix uses the shared resolver/bridge/pin, and no retrieval/reranking/judge/evaluator/core Scheduler prompt moved.

```bash
rg -n 'DEFAULT_RAG_SYSTEM_PROMPT|promptForRag|replace\("\{context\}"|build_node_expansion_prompt|enqueue_explainer_node_expansion_job|create_job' \
  apps/packages/ui/src \
  tldw_Server_API/app/core/Explainer \
  tldw_Server_API/app/api/v1/endpoints/explainer.py
rg -n 'rag\.client\.answer|research\.explainer\.expansion' \
  tldw_Server_API/app/core/Service_Prompts \
  apps/packages/ui/src \
  tldw_Server_API/app/core/Explainer
```

- [ ] **Step 2 — Update documentation and inventory evidence.** For the two rows only, record migrated call sites, contract/schema versions, compatibility keys, Golden tests, availability, and safe provenance. Explicitly document `rag.client.answer`'s code-defined `replace_copilot_user_text_preserve_non_text` policy, the retained current-user carrier, the active legacy `useMessage.tsx` consumer, and the three pre-existing upstream context truncations; state that clients cannot select policy/role/order and renderer overflow is never silently truncated. Document deployment mappings, strict configured-file failure, precedence, chat bridge behavior, Jobs pinning/retention, `413`/stable non-HTTP size errors, and troubleshooting without publishing prompt bodies or file paths. Do not alter other inventory decisions or counts except through the inventory's validator-driven correction process.

- [ ] **Step 3 — Run focused domain integration suites.** Expected result: all pass with no skip introduced by this task.

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Service_Prompts \
  tldw_Server_API/tests/Chat/test_service_prompt_execution.py \
  tldw_Server_API/tests/Explainer/test_explainer_service_prompts.py \
  tldw_Server_API/tests/Explainer/test_explainer_jobs.py \
  tldw_Server_API/tests/Explainer/test_explainer_endpoints.py \
  tldw_Server_API/tests/Jobs/test_service_prompt_worker_guard.py \
  tldw_Server_API/tests/Jobs/test_service_prompt_pinning_sqlite.py \
  tldw_Server_API/tests/Jobs/test_service_prompt_pinning_postgres.py \
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py
cd apps/tldw-frontend
bunx vitest run \
  ../packages/ui/src/services/__tests__/service-prompts.test.ts \
  ../packages/ui/src/services/__tests__/tldw-chat.message-sanitization.test.ts \
  ../packages/ui/src/models/__tests__/ChatTldw.stream-metadata.test.ts \
  ../packages/ui/src/hooks/chat-modes/__tests__/ragMode.sanitization.test.ts \
  ../packages/ui/src/hooks/chat-modes/__tests__/rag-client-answer.service-prompts.test.ts \
  ../packages/ui/src/hooks/__tests__/useMessage.rag-service-prompt.guard.test.ts
```

- [ ] **Step 4 — Run Bandit on the complete touched Python scope and review the JSON.** Fix every new finding in changed code; an empty command exit is insufficient without reviewing the report.

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Service_Prompts/registry.py \
  tldw_Server_API/app/core/Service_Prompts/models.py \
  tldw_Server_API/app/core/Service_Prompts/templates.py \
  tldw_Server_API/app/api/v1/endpoints/service_prompts.py \
  tldw_Server_API/app/api/v1/endpoints/chat.py \
  tldw_Server_API/app/core/Jobs/worker_sdk.py \
  tldw_Server_API/app/core/Explainer/prompting.py \
  tldw_Server_API/app/core/Explainer/jobs.py \
  tldw_Server_API/app/core/Explainer/jobs_worker.py \
  tldw_Server_API/app/core/Explainer/service.py \
  tldw_Server_API/app/api/v1/API_Deps/service_prompt_deps.py \
  tldw_Server_API/app/api/v1/endpoints/explainer.py \
  -f json -o /tmp/bandit_task_12113_3.json
```

The unchanged potential Task 4 foundation sources are included intentionally so this gate has one fixed scope. A missing TASK-12957 bridge capability remains a prerequisite stop and is corrected and secured under that task, not silently added to this domain's touched scope.

- [ ] **Step 5 — Run the mandatory per-commit gate one final time.** Run these commands exactly from the repository root. The fixed arrays include every Python source that this plan may change and every Python test that it names; unchanged potential foundation sources remain in scope intentionally. No command or path may be waived or shortened.

```bash
source .venv/bin/activate
python -m pytest -v

PYTHON_SOURCE_CHECK_PATHS=(
  tldw_Server_API/app/core/Service_Prompts/registry.py
  tldw_Server_API/app/core/Service_Prompts/models.py
  tldw_Server_API/app/core/Service_Prompts/templates.py
  tldw_Server_API/app/api/v1/endpoints/service_prompts.py
  tldw_Server_API/app/api/v1/endpoints/chat.py
  tldw_Server_API/app/core/Jobs/worker_sdk.py
  tldw_Server_API/app/api/v1/API_Deps/service_prompt_deps.py
  tldw_Server_API/app/api/v1/endpoints/explainer.py
  tldw_Server_API/app/core/Explainer/service.py
  tldw_Server_API/app/core/Explainer/jobs.py
  tldw_Server_API/app/core/Explainer/jobs_worker.py
  tldw_Server_API/app/core/Explainer/prompting.py
)
PYTHON_TEST_CHECK_PATHS=(
  tldw_Server_API/tests/Service_Prompts/test_registry.py
  tldw_Server_API/tests/Utils/test_prompt_loader_env_overrides.py
  tldw_Server_API/tests/Service_Prompts/test_rag_generation_contracts.py
  tldw_Server_API/tests/Chat/test_service_prompt_execution.py
  tldw_Server_API/tests/Explainer/test_explainer_endpoints.py
  tldw_Server_API/tests/Explainer/test_explainer_jobs.py
  tldw_Server_API/tests/Explainer/test_explainer_service_prompts.py
  tldw_Server_API/tests/Jobs/test_service_prompt_worker_guard.py
  tldw_Server_API/tests/Service_Prompts/test_templates.py
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py
)
python -m black --check "${PYTHON_SOURCE_CHECK_PATHS[@]}" "${PYTHON_TEST_CHECK_PATHS[@]}"
python -m ruff check "${PYTHON_SOURCE_CHECK_PATHS[@]}" "${PYTHON_TEST_CHECK_PATHS[@]}"
python -m mypy "${PYTHON_SOURCE_CHECK_PATHS[@]}" "${PYTHON_TEST_CHECK_PATHS[@]}"

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

Then run `node Helper_Scripts/validate_service_prompt_inventory.mjs .` from the repository root and record its JSON output in TASK-12960. Any nonzero exit stops the task.

- [ ] **Step 6 — Finalize TASK-12960 through Backlog MCP/CLI.** Record the human-owned Change summary requirement, what changed and why, exact test/Bandit/inventory outputs, touched files, commits, blockers, and PR link when available. Mark complete only after every definition and call site above is verified. Do not manually edit the task file.

- [ ] **Step 7 — Commit only after every gate is green:** `docs: record RAG generation prompt migration (TASK-12960)`.

## Final stop conditions

- Stop if TASK-12957's chat bridge is absent or cannot apply `replace_copilot_user_text_preserve_non_text` to one retained current-user carrier while preserving all history/non-text bytes, one authenticated server-resolved final user message, the finite boolean-selected locked suffix and its exact trim/dedup placement, principal ownership, and `413` behavior. Do not ship a browser resolver, caller-owned policy/roles/order/suffix text, raw hidden part fetch, `extra_body` workaround, or RAG-only endpoint.
- Stop if Explainer cannot declare the exact finite requirement and bind a complete authenticated bundle before queue release, or if its handler can run without verified WorkerSDK context. Never fall back to current defaults after a pin failure.
- Stop if no-override complete provider arrays differ by even one byte, role, message, separator, placeholder occurrence, or ordering from the Goldens.
- Stop if an implementation needs to expose or persist prompt bodies, variables, deployment paths, hidden content, MACs, or keys in catalog/response/log/status/job payload/provenance.
- Stop if an excluded/deferred retrieval, reranking, judge/evaluator, routing, or core Scheduler prompt appears necessary; return it to the inventory review instead of expanding this task.
- Stop after three failed approaches to the same issue, document the attempts in TASK-12960, and reassess before continuing.
