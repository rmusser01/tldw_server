# Eligible Extraction and Chunking Service Prompts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the two approved extraction/chunking definitions to the shared Service Prompt resolver without changing rolling-summary assembly, mood classification, or any deferred structured-extraction flow.

**Architecture:** Register only `chunking.rolling.summary` and `writing.feedback.mood`. The chunking endpoint resolves one render-ready atomic bundle from the authenticated request and passes it immutably through `Chunker` into every rolling-summary iteration. The browser mood definition declares TASK-12957's registry-owned `replace_generated_messages` policy, so TypeScript sends `messages: []`, the definition ID, and bounded runtime passage; it cannot select policy, roles, insertion, or ordering and never implements resolution or receives hidden prompt parts.

**Tech Stack:** Existing Python Service Prompts registry/resolver, FastAPI chunking and chat endpoints, React/TypeScript shared UI package, pytest/Hypothesis, Vitest.

---

**Backlog task:** `TASK-12962`

**Prerequisites:** Complete TASK-12961 and the dependency-ordered foundation plans 02–06. In particular, TASK-12957 must have shipped the typed `service_prompt` extension on `/api/v1/chat/completions`, implemented in `chat_request_schemas.py` and `chat.py`, plus the request-only helper in `apps/packages/ui/src/services/service-prompts.ts`. Stop rather than creating a TypeScript resolver or a second runtime endpoint if that bridge is absent.

Before every commit below, satisfy the umbrella plan's mandatory per-commit gate. The requester's planning-time CI-shard waiver does not waive implementation gates.

## Stage map

| Stage | Tasks | Goal | Success criteria | Status |
| --- | --- | --- | --- | --- |
| 1. Contracts | Task 1 | Register exactly two approved contracts | exact ID set and default-message Goldens pass | Not Started |
| 2. Rolling runtime | Task 2 | Resolve once across every chunking entry path | template JSON, ordinary JSON, and multipart file requests preserve exact provider bytes and one resolution | Not Started |
| 3. Browser runtime | Task 3 | Move mood classification behind authenticated server execution | no prompt body reaches TypeScript; hook and bridge tests pass | Not Started |
| 4. Limits | Task 4 | Enforce approved UTF-8 budgets | all four boundaries and no-dispatch behavior pass | Not Started |
| 5. Release gate | Task 5 | Reconcile, document, secure, and verify the domain | full mandatory gates, Bandit, inventory validator, and Backlog finalization pass | Not Started |

## Approved scope and contracts

| Definition | Atomic contract | Runtime boundary |
| --- | --- | --- |
| `chunking.rolling.summary` | editable literal `system` ⇒ editable literal `base_instruction` → locked length/continuity branch → optional locked prior context → optional locked preserve-structure sentence → locked segment carrier | `tldw_Server_API/app/api/v1/endpoints/chunking.py` resolves once; `Chunker` and `RollingSummarizeStrategy` receive the immutable render-ready bundle. A finite key-presence selector preserves the current missing-key helpful-assistant packaged source versus present-`None` analyzer deployment/file/default source. |
| `writing.feedback.mood` | editable literal system semantics → locked one-word enum contract ⇒ editable literal classification semantics → locked passage carrier | the existing authenticated chat-completion execution extension resolves server-side with registry-owned `replace_generated_messages`; `useWritingFeedback` sends `messages: []` and `passage=editorText.slice(-500)` only |

OCR, proposition extraction, structured JSON extraction, recursive document workflows, and core-Scheduler-reachable definitions remain deferred or excluded. Do not register them in this task.

## Task 1: Register the two exact contracts and default-message Goldens

**Files:**

- Modify: `tldw_Server_API/app/core/Service_Prompts/registry.py`
- Modify: `tldw_Server_API/tests/Service_Prompts/test_registry.py`
- Create: `tldw_Server_API/tests/Service_Prompts/test_extraction_chunking_contracts.py`
- Reference: `Docs/Design/service-prompt-inventory.md`

- [ ] Add failing registry tests asserting that the available extraction/chunking ID set is exactly `{"chunking.rolling.summary", "writing.feedback.mood"}` and that no deferred/excluded extraction ID appears.
- [ ] Add failing contract tests for the precise selector, role/order, visibility, literal/template mode, optional parts, declared variables, `C153→G` per-summary rolling-context compaction, `C500→G` passage truncation, and safe provenance metadata recorded in the approved matrix. The system selector is a finite enum derived only from `"system_message" in llm_config`: `strategy_missing_key` selects the packaged helpful-assistant bytes without consulting the analyzer deployment mapping, while `analyzer_default` (present `None`) selects `summarization/Summarization System Prompt` through its strict deployment/file/default provider; a present non-`None` value, including `""`, remains the highest literal explicit part. Assert `writing.feedback.mood` declares exactly `replace_generated_messages`, and that request data cannot provide a policy, role, insertion index, or order.
- [ ] Add failing Golden tests that render the packaged defaults and compare complete provider messages byte-for-byte:

```python
assert rolling_messages == [
    {"role": "system", "content": expected_system},
    {"role": "user", "content": expected_base_and_locked_carriers},
]
assert mood_messages == [
    {"role": "system", "content": "You are a mood classifier. Respond with exactly one word."},
    {
        "role": "user",
        "content": (
            "Classify the emotional mood of this text. Respond with ONLY one word from: "
            "tense, romantic, melancholic, action, calm, mysterious, humorous\n\nText: "
            + passage
        ),
    },
]
```

- [ ] Add a two-ID precedence/provenance matrix. For both definitions, prove approved-user revision → deployment provider → packaged default selection; for rolling only, declared explicit literal parts remain highest. Put braces and Unicode in approved/deployment E-L parts and assert byte preservation; configured-source failures are strict and never fall back. Safe provenance must contain exactly definition ID, source kind, contract version, revision-or-pin identifier, and digest metadata, and must exclude prompt text, variables, file paths, MACs, keys, and owner data.

- [ ] Run `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Service_Prompts/test_registry.py tldw_Server_API/tests/Service_Prompts/test_extraction_chunking_contracts.py`; confirm failure reports both unknown IDs before implementation.
- [ ] Add the two code-defined registry entries. Map rolling `base_instruction` to the existing `chunking/rolling_summarization` YAML → Markdown → packaged fallback compatibility source and its existing `TLDW_PROMPT_FILE_*` convention. Encode the finite system-source selector above so stored approved revisions still outrank either server-default branch while direct missing-key and present-`None` no-override bytes remain different exactly as today. Keep mood's current bytes as the immutable packaged default. Encode provider roles and locked assembly in registry metadata; do not leave a second editable default in either consumer.
- [ ] Rerun the focused tests and confirm both IDs, contract assertions, and provider-message Goldens pass.
- [ ] Commit: `feat: register extraction chunking service prompts (TASK-12962)`.

## Task 2: Resolve rolling summarization once and preserve every branch

**Files:**

- Modify: `tldw_Server_API/app/api/v1/endpoints/chunking.py`
- Modify: `tldw_Server_API/app/core/Chunking/__init__.py`
- Modify: `tldw_Server_API/app/core/Chunking/chunker.py`
- Modify: `tldw_Server_API/app/core/Chunking/strategies/rolling_summarize.py`
- Modify: `tldw_Server_API/tests/Chunking/test_chunking_endpoint.py`
- Modify: `tldw_Server_API/tests/Chunking/test_chunker_v2.py`
- Create: `tldw_Server_API/tests/Chunking/test_rolling_summarize_service_prompt.py`

- [ ] Write failing endpoint tests that inject a counting resolver and independently exercise all three active request shapes: template JSON through `TemplateProcessor`, ordinary JSON through `improved_chunking_process`, and multipart file through `improved_chunking_process`. For each, submit `method=rolling_summarize` and prove one resolution for the authenticated owner even when several segments are summarized. Assert owner/request/trace metadata enter `PromptExecutionContext` and provider/model selection does not.
- [ ] Write failing strategy Goldens for first and continuation calls, prior context present/absent, preserve-structure on/off, and the exact order and separators from the matrix. Compare complete system/user provider messages, not isolated prompt substrings.
- [ ] Write failing precedence cases for the existing three-way strategy contract: `llm_config.system_message` present and non-`None` (including `""`) is the literal system part; a missing key uses the helpful-assistant default; a present `None` selects the analyzer deployment/file-backed or packaged default. Separately preserve the endpoint's existing `client_suggested_system_prompt or method_default_system_prompt` behavior instead of silently changing empty request values.
- [ ] Write failing compatibility cases proving explicit literal system/base text containing `{segment}` remains literal, deployment-file failures use the registry's strict managed-source error rather than legacy fallback, and the un-migrated `load_prompt()` API remains unchanged.
- [ ] Run `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Chunking/test_rolling_summarize_service_prompt.py tldw_Server_API/tests/Chunking/test_chunking_endpoint.py -k 'rolling_summarize or service_prompt'`; confirm failures show the endpoint still loads/assembles prompt text inside the lower loop.
- [ ] At each authenticated public chunking boundary, build one `PromptExecutionContext`, translate only a present non-`None` system value into a named literal override, set the finite system-source selector from exact key presence/`None` state, resolve `chunking.rolling.summary`, and pass the immutable render-ready bundle into the configured `Chunker` (template JSON) or `improved_chunking_process` (ordinary JSON and multipart file). Preserve each branch's current `client_suggested_system_prompt or method_default_system_prompt` calculation before deriving that state.
- [ ] Add a narrow optional `service_prompt_bundle` parameter to `improved_chunking_process` in `tldw_Server_API/app/core/Chunking/__init__.py`, then pass it into the `Chunker` constructed there. Add the corresponding narrow constructor argument to `Chunker` and `RollingSummarizeStrategy`. Require the bundle for every migrated provider path; do not add a registry singleton or perform ownerless lookups in `_create_summarization_prompt` or its loop.
- [ ] Replace lower-loop file/default loading with rendering of the already-resolved bundle using the current finite branch values, segment, and preserve flag. Preserve `_create_context_summary` exactly: each completed summary is reduced to at most 150 characters or its first sentence, with the existing three-character ellipsis yielding a maximum of 153 characters, before the last `context_window` items are joined under the existing `Previous context:` wrapper. Do not truncate the assembled prior-context message. Keep provider, model, temperature, max-token, and error handling unchanged.
- [ ] Rerun the focused tests and existing rolling cases in `test_chunker_v2.py`; confirm one resolution, exact bytes, and unchanged no-LLM behavior.
- [ ] Commit: `feat: resolve rolling summary prompt once (TASK-12962)`.

## Task 3: Move mood classification to server-side prompt execution

**Files:**

- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingFeedback.ts`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/hooks/__tests__/useWritingFeedback.test.tsx`
- Modify: `apps/packages/ui/src/services/service-prompts.ts`
- Modify: `apps/packages/ui/src/services/__tests__/service-prompts.test.ts`
- Modify: `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`

- [ ] Add a failing hook test that enables mood analysis and asserts the request uses `/api/v1/chat/completions` with `messages: []`, `service_prompt.definition_id === "writing.feedback.mood"`, `variables === {passage: editorText.slice(-500)}`, no raw mood system/user instruction in the browser request, no policy/role/insertion/order field, and unchanged model/temperature/max-token values.
- [ ] Add failing server-bridge coverage for this allowlisted ID: authenticated owner context is used, the registry applies `replace_generated_messages` to the empty incoming array, server-rendered messages equal the Task 1 Golden, caller-selected policy/roles/order and unknown parts/variables fail with the bridge's stable `422` code, and hidden/locked text is absent from the response and logs.
- [ ] From `apps/tldw-frontend`, run `bunx vitest run ../packages/ui/src/components/Option/WritingPlayground/hooks/__tests__/useWritingFeedback.test.tsx ../packages/ui/src/services/__tests__/service-prompts.test.ts`; confirm the request-body assertion fails because the hook still sends raw prompt messages.
- [ ] Reuse the typed request builder created by TASK-12957. Change only the mood branch to submit the ID and passage variable. Preserve debounce, cancellation, last-500-character slicing, enum normalization, stale-response suppression, and feature toggles. Do not change the separately migrated echo branch.
- [ ] Run the focused Vitest command and `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Chat/test_service_prompt_execution.py -k writing_feedback_mood`; confirm both client request shape and server provider-message bytes pass.
- [ ] Commit: `feat: resolve writing mood prompt server side (TASK-12962)`.

## Task 4: Prove approved byte limits at both execution surfaces

**Files:**

- Modify: `tldw_Server_API/tests/Service_Prompts/test_extraction_chunking_contracts.py`
- Modify: `tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py`
- Modify: `tldw_Server_API/tests/Chat/test_service_prompt_execution.py`
- Modify: `tldw_Server_API/tests/Chunking/test_rolling_summarize_service_prompt.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chunking.py`
- Modify: `tldw_Server_API/tests/Chunking/test_chunking_endpoint.py`

- [ ] Add parameterized UTF-8 byte tests using multibyte content where character length differs from byte length. For authored parts and expanded variables/rendered text parts, assert exactly 65,536 bytes succeeds and 65,537 fails. For authored definitions and final rendered bundles, assert exactly 262,144 bytes succeeds and 262,145 fails. Use a synthetic in-test five-part definition/bundle so the aggregate boundary can be crossed while every constituent remains at or below 65,536 bytes; count all deterministic separators/assembly bytes explicitly. Do not register a third product definition.
- [ ] Assert the settings API maps authored part/definition overflow to HTTP `413` with `service_prompt_size_limit_exceeded`. Assert the chat execution bridge maps variable/rendered-part/bundle overflow to HTTP `413` with the same code and without provider dispatch. Assert direct rolling-strategy overflow exposes the exact stable non-HTTP code `service_prompt_size_limit_exceeded` before analyzer dispatch.
- [ ] Exercise template JSON, ordinary JSON, and multipart file requests at the authenticated public chunking boundary. For a one-byte-over rendered part and aggregate bundle, assert exact HTTP `413` with response code `service_prompt_size_limit_exceeded`, zero analyzer calls, and no chunk/file persistence. Add a narrow Service Prompt size-exception branch before the endpoint's broad `500` mapping; unrelated failures retain existing status/details.
- [ ] Prove no overflow is silently truncated. The only allowed truncations are the matrix-recorded source operations: each rolling summary is compacted by `_create_context_summary` before the selected context-window entries and wrapper are assembled, and mood uses the last 500 passage characters before the UTF-8 `G` check. Assert the assembled rolling prior-context message itself is never silently truncated.
- [ ] Run `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Service_Prompts/test_extraction_chunking_contracts.py tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py tldw_Server_API/tests/Chat/test_service_prompt_execution.py tldw_Server_API/tests/Chunking/test_rolling_summarize_service_prompt.py tldw_Server_API/tests/Chunking/test_chunking_endpoint.py`; confirm all settings, bridge, direct-strategy, public-endpoint boundary, and no-dispatch assertions pass.
- [ ] Commit: `test: enforce extraction prompt byte budgets (TASK-12962)`.

## Task 5: Verify domain completeness and document the rollout

**Files:**

- Modify: `Docs/Design/service-prompt-inventory.md`
- Modify: `Docs/API/service-prompts.md`
- Modify: `tldw_Server_API/Config_Files/Prompts/README.md`
- Modify: `tldw_Server_API/Config_Files/config.txt`
- Update through official Backlog MCP/CLI: `TASK-12962`

- [ ] Search `rolling_summarization`, `MOOD_PROMPT`, the two exact legacy default strings, and all `load_prompt(`/chat dispatches in the touched consumers. Confirm every eligible call uses the resolver/bridge and every remaining extraction/chunking prompt is still explicitly locked, deferred, or excluded in the matrix.
- [ ] Update the two matrix rows with migrated call sites, registry contract version, Golden test paths, and availability. Do not change the other 230 decisions as part of this task.
- [ ] Document the two definition IDs, rolling system-source selector, `chunking/rolling_summarization` compatibility mapping and `TLDW_PROMPT_FILE_*` deployment override in `Docs/API/service-prompts.md`, `Config_Files/Prompts/README.md`, and the relevant commented prompt/config example in `Config_Files/config.txt`. Do not document mood prompt bodies as a browser-side configuration surface.
- [ ] Run the domain suites together:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Service_Prompts \
  tldw_Server_API/tests/Chunking/test_rolling_summarize_service_prompt.py \
  tldw_Server_API/tests/Chunking/test_chunking_endpoint.py \
  tldw_Server_API/tests/Chunking/test_chunker_v2.py \
  tldw_Server_API/tests/Chat/test_service_prompt_execution.py \
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py
```

- [ ] From `apps/tldw-frontend`, run `bunx vitest run ../packages/ui/src/components/Option/WritingPlayground/hooks/__tests__/useWritingFeedback.test.tsx ../packages/ui/src/services/__tests__/service-prompts.test.ts`, then the umbrella plan's full frontend gates because `apps/` changed.
- [ ] Run `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Service_Prompts tldw_Server_API/app/core/Chunking tldw_Server_API/app/api/v1/endpoints/chunking.py -f json -o /tmp/bandit_task_12113_5.json`; review and fix every new finding in touched code.
- [ ] Run the exact touched-Python checks from the repository root:

```bash
source .venv/bin/activate
PYTHON_CHECK_PATHS=(
  tldw_Server_API/app/api/v1/endpoints/chunking.py
  tldw_Server_API/app/core/Chunking/__init__.py
  tldw_Server_API/app/core/Chunking/chunker.py
  tldw_Server_API/app/core/Chunking/strategies/rolling_summarize.py
  tldw_Server_API/app/core/Service_Prompts/registry.py
  tldw_Server_API/tests/Chat/test_service_prompt_execution.py
  tldw_Server_API/tests/Chunking/test_chunker_v2.py
  tldw_Server_API/tests/Chunking/test_chunking_endpoint.py
  tldw_Server_API/tests/Chunking/test_rolling_summarize_service_prompt.py
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py
  tldw_Server_API/tests/Service_Prompts/test_extraction_chunking_contracts.py
  tldw_Server_API/tests/Service_Prompts/test_registry.py
)
python -m black --check "${PYTHON_CHECK_PATHS[@]}"
python -m ruff check "${PYTHON_CHECK_PATHS[@]}"
python -m mypy "${PYTHON_CHECK_PATHS[@]}"
```
- [ ] Because `apps/` changed, run the complete frontend gate exactly: `cd apps/tldw-frontend && bun run test:run && bunx vitest run -c vitest.extension.config.ts && cd ../packages/ui && bun run test && cd ../../tldw-frontend && bun run format:check && bun run lint && bunx tsc --noEmit -p ../packages/ui/tsconfig.json && bun run build`.
- [ ] Run the umbrella plan's full backend gate exactly from the repository root: `source .venv/bin/activate && python -m pytest -v`, followed by `git diff --check` and `node Helper_Scripts/validate_service_prompt_inventory.mjs .`. Record all outputs in TASK-12962. If any mandatory command is unavailable or fails for environmental or unrelated reasons, diagnose it under the three-attempt rule and stop; do not waive or commit around it.
- [ ] Commit: `docs: complete extraction chunking prompt migration (TASK-12962)`.

## Stop conditions

- Stop if TASK-12957 did not establish authenticated server-side browser execution with the exact shared resolver; do not expose hidden parts or implement resolution in TypeScript.
- Stop if rolling summarization cannot carry one immutable render-ready bundle through every segment without re-resolution.
- Stop if preserving the current explicit-system three-way behavior requires changing an undeclared contract; return to the inventory for human review.
- Stop if either output becomes prompt-dependent for a machine schema rather than independently normalized/validated.
