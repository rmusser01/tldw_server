# Moderation on every LLM surface (TASK-13950.27) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every LLM call that carries user content is moderated by one Moderation-owned pair of gateway hooks, under a named surface.
- Moderation stays off unless policy turns it on: the global flag, an explicit per-user override, or a Guardian overlay.
- Operators can switch it off one surface at a time.
- A shrink-only guard stops new unmoderated call sites.

**Architecture:** TASK-13950.13 gave `LLM_Calls/gateway.py` a process-global pre/post-call hook list. It is empty in production, and every call except `chat.completions` has `surface=None`.
- **Task 1** adds the guard and makes chat's sites name their surface.
- **Task 2** makes the moderation policy surface-aware.
- **Task 3** adds the hooks (`core/Moderation/gateway_hooks.py`). They are registered in two places, the app lifespan (services layer) and the `main()` of every LLM-calling worker process. Task 3 also moderates the chat auto-router.
- **Tasks 4-10** give every call site a surface id in batches, which switches moderation on for it, with tests:
  - direct provider-adapter calls move onto the gateway;
  - the raw-HTTP surfaces the owner named are moderated explicitly.
- **Task 11** moderates any call still without a surface as `unclassified`, exempts the internal calls with written reasons, and closes the task.

Chat keeps its own pipeline. The hooks skip `chat.completions`, so chat is moderated once (R1).

**Tech Stack:** Python 3, FastAPI, pytest, and the AST lint ratchets in `tldw_Server_API/tests/lint` (`_ceilings.py`).

**Spec:** `backlog/tasks/task-13950.27 - Apply-optional-policy-configured-moderation-to-every-LLM-surface.md` (AC1-AC6) and the owner decisions below. The facts come from `.superpowers/sdd/2026-11-26-task-13950-27-moderation-every-surface/survey.md` and `plan-review.md` (dev 1f5f1f05e0); line numbers are that tree's.

**Owner decisions:**
- **2026-10-25:** moderation applies on every LLM surface, optional and policy-configured.
- **2026-11-26:**
  - Evaluation judge calls are exempt as `internal.eval_judge` ("measurement; moderating judge input/output changes scores"). The evaluated model's generation calls stay moderated.
  - Slides standalone-HTML and MCP model completion, both raw HTTP, are moderated in `.27`. The `/llamacpp` passthrough is a follow-up. The OCR LLM backends are exempt, because they transcribe documents rather than generate text.

## Global Constraints

- **Moderation stays optional.** With `[Moderation] enabled = False` (the default), no per-user override and no Guardian overlay:
  - every surface makes zero policy-evaluator calls and creates no Guardian DB;
  - every surface's output is byte-identical to today, whatever the surface keys say.
- **Chat output is unchanged.** The TASK-13950.25 golden (`tests/Chat/unit/test_chat_executors_golden.py`) passes unmodified in every PR.
- **Layering.** `core/LLM_Calls` never imports `core/Chat` or `core/Moderation`.
  - Code inside `core/LLM_Calls` that must moderate outside the gateway entries (Summarization, Task 10) uses the gateway's public hook runners (Task 3).
  - The app lifespan registers the hooks from the services layer. Workers call `register_moderation_gateway_hooks()` from their own `main()`; that is a domain-to-domain import.
- **No secrets.** Hooks never log `call.request`, which carries `api_key`, `app_config` and the credential handle. A log line carries surface, phase, action, provider and the integer user id, never message text.
- **Hook rules.** A hook must not call the gateway. A post hook never raises. Hooks are sync, because chat and most surfaces use the sync entry. Policy is resolved once per call: `pre_call` stores it on the `GatewayCall` instance as a plain attribute, and `post_call` reads it.
- **Refusal text.**
  - Input block: `LLMPolicyRefusal("Input violates moderation policy")` (HTTP 400, chat's text).
  - Output block, non-stream: the content text becomes `"Output violates moderation policy"`, `finish_reason` becomes `"content_filter"` (Anthropic format: `stop_reason: "refusal"`), and provider `usage` is kept.
  - Output block, stream: the wrapper emits one delta with that text, then the `content_filter` finish chunk, then `[DONE]`.
  - A `content_filter`/`refusal` result is never retried or sent to a fallback provider. The retry/fallback sites that re-ask after a parse failure check for it, and each batch names its sites:
    - `chat_route_resolver` provider fallback (Task 5);
    - the RAG query classifier (Task 7);
    - claims extraction (Task 9).
- **No leaks.** When moderation changes a chunk's or response's content, drop that chunk's or choice's `provider_response`. After a block, drop it from every later chunk too.
- **Refusals propagate.** A surface surfaces a refusal as its own refusal: HTTP 400 for HTTP routes, an error frame or result with code `llm_policy_refused` for WS and voice, a failed step or job for background work. It never falls back to another provider, an empty answer or a default router choice, and it never marks the provider unhealthy. Each batch names its sites.
- **Surface ids.**
  - Every id is a lowercase dotted string in `core/Moderation/surfaces.py`.
  - Call sites pass a string literal (chat included: `llm_surface="chat.completions"`), or forward their own `llm_surface` parameter. No call site imports Moderation.
  - Injected default callables are surfaced partials, e.g. `fn = partial(perform_chat_api_call_async, llm_surface="chat.prompt_improvement")`.
- **Ratchets.**
  - The four frozen functions (`endpoints.chat:create_chat_completion`, `character_chat_sessions:character_chat_completion`, `audio_streaming:websocket_audio_chat_stream` and `._stream_llm_with_runtime`, `speech_chat_service:run_speech_chat_turn`) keep zero net lines, or bump `function_size_baseline.json` with a reason naming the PR.
  - New env reads go through `core/config_sections/moderation.py`, never `os.getenv` in a new module.
- **Tests.**
  - Stub the LLM at `gateway._get_llm_registry` (pattern: `tests/Chat/integration/test_gateway_policy_hooks_chat_endpoint.py:16-48`) so hooks run. Never patch `perform_chat_api_call*` in a moderation test.
  - Endpoint tests use a provider outside chat's TEST_MODE mock set `{openai, groq, mistral}` (`chat.py:5459-5527`), e.g. `anthropic`, or an unsupported name such as `e2e-fake` for gateway unit tests.
  - Unset `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`. Set `HTTP_PROXY`/`HTTPS_PROXY=http://127.0.0.1:9` and `NO_PROXY=localhost,127.0.0.1`. Run with `PYTHONPATH=. TEST_MODE=true DISABLE_HEAVY_STARTUP=1`. Set `RUN_JOBS=1` and `RUN_EVALUATIONS=1` where a step names jobs or evaluations tests.
- **Disk.**
  - Every pytest run uses `--basetemp` under `$TMPDIR`, deleted right after, and `-p no:cacheprovider`.
  - At most `-n 2`, one run at a time.
  - Refuse to start below 100 GiB free (`df -g /System/Volumes/Data`).
- **CI.** No new required job. The guards join backend-required's existing "Enforce CI contracts and code ratchets" step.
- **Process.**
  - PRs target `dev` and are merged as merge commits. ADR-004 Change summary waived (TASK-13950).
  - Backlog edits only via `PYTHONPATH=tools/backlog-py/src $PY -m backlog_py task ...`; stage with `git add -A backlog/tasks`.
  - Coordination: PR #3437 (DIRTY) rewrites `moderation_service`, `supervised_policy`, `chat_service`, `moderation_pipeline` and `streaming_pipeline`. Tasks 2 and 3 edit these lightly, so whichever lands second rebases.

## Rulings (controller; binding)

- **R1. Chat keeps its pipeline, and the hooks skip `chat.completions`.**
  - Why chat's checks do not move into hooks:
    - chat moderates and redacts `request_data` before it persists the user turn;
    - its mandatory audit is async and fail-closed (`chat_service.py:2404`; 503 at `chat.py:4647-4651`);
    - chat uses the sync entry, where a hook cannot await;
    - Guardian, self-monitoring and topic monitoring are request-bound.
  - `.13`'s plan allowed either option. AC2's intent, that chat is not moderated twice, holds through `CALLER_MODERATED_SURFACES`.
  - AC2 is reworded in PR C, where the skip lands.
  - Gap kept as today: server-side tool outputs on the tool continuation (`tool_execution_service.py:151`) stay unmoderated. This is a follow-up.
- **R2. Surfaces switch on one batch at a time.** Until Task 11 the hooks skip `surface=None`. Task 11 moderates `None` as `unclassified`, a runtime backstop.
- **R3. Hooks write no audit rows.** They emit `logger.warning("moderation {phase} {action} surface={surface} provider={provider} user_id={user_id}")`. Audit parity is a follow-up.
- **R4. Identity, Guardian and cost.**
  - The policy user is `str(call.user_id)`, else `None`, which gives the default policy.
  - **Guardian.** Applies when `is_guardian_enabled()` and `call.user_id` is not None, through `resolve_existing_guardian_db_for_user_id` (`guardian_db_resolver.py:52`). A `FileNotFoundError` means no overlay and creates no DB.
    - The engine (`SupervisedPolicyEngine`) is cached per `user_id` with its own 60 s TTL.
    - The overlay is built with `chat_type="regular"`.
    - An engine failure falls back to the base policy.
  - Self-monitoring and topic monitoring stay chat-only (follow-up). Background code without a content scope gets the default policy; a follow-up sets `scoped_context` in `core/Workflows`.
- **R5. Surface keys narrow only the default policy.**
  - `effective_default.enabled = global.enabled and surface_enabled.get(surface, True)`.
  - Then the explicit per-user override coalesces (`policy_compiler.py:138`), and the Guardian overlay forces `enabled=True`.
  - `get_effective_policy(user_id, *, surface)` requires the keyword. `None` means no narrowing: the admin effective-policy endpoints and the explicit workflow moderation step pass it.
  - When no narrowing applies, it returns the same policy object as today.
- **R6. Moving direct adapter calls onto the gateway.**
  - **Unchanged by the move:** quota (prechecked), usage accounting, and TASK-15581's denied-model refusal, which runs inside every adapter wrapper (`usage_accounting.py:365-415`).
  - **Accepted, recorded changes:**
    - model validation: `_validate_provider_model_selection`, a model-inventory request that fails closed;
    - egress remapping: `EgressPolicyError` → `ChatConfigurationError`/`ChatProviderError(503)`;
    - request rebuilding: `skip_keys` drops `base_url`, `timeout` unless passed, and so on;
    - the bounded sync pool.
  - Each PR lists these per site. Each moved site gets one test in which conftest's model-validation stub (`tests/conftest.py:315-344`) is replaced by a stub of the inventory lookup returning the site's model, so the validation path actually runs.
  - Never use `llm_call_context(exempt=True)` to preserve behaviour: it switches off TASK-15581's refusal and the quota check.
  - A site whose error taxonomy depends on raw `EgressPolicyError`, such as `Summarization_General_Lib._summarize_via_adapter` (`_summary_error_code` :163-169), stays direct and is moderated explicitly through the Task 3 text functions. Its guard entry carries that reason.
- **R7. Explicit moderation where there is no gateway.**
  - `/api/v1/messages` native providers (`anthropic`, `llama.cpp`; raw HTTP, `messages.py:2095-2150`) use Task 5's Anthropic-format functions.
  - Slides standalone-HTML (`core/Slides/standalone_html_provider.py:495-615`, OpenAI or Anthropic format) and MCP model completion (`core/MCP_unified/adapters/model_completion/transport.py:359`, OpenAI format) use the OpenAI or Anthropic functions (Task 9).
  - The `/llamacpp` passthrough, VN Play's unwired `GenerationModerationAdapter` and `Local_LLM/*_Handler.py`/`Local_Summarization_Lib.py`, which need a reachability check, become follow-ups.
  - The OCR backends are exempt by owner decision; Task 11 documents that.
- **R8. The voice assistant stays dead.**
  - `VoiceAssistant/router.py` and `intent_parser.py` import a missing module and pass legacy kwargs the gateway ignores.
  - Task 6 only adds `llm_surface="voice.assistant"` to both calls. A follow-up decides whether to revive the fallback: its provider, key and message shape.
- **R9. Guard counting.**
  - In each module, the guard counts:
    1. Calls to `perform_chat_api_call`, `perform_chat_api_call_async`, `chat_api_call`, `chat_api_call_async`, or an import alias of one, without an `llm_surface=` keyword. Both bare-name and attribute calls (`cs.perform_chat_api_call(...)`) count.
    2. `partial`/`functools.partial` of them without `llm_surface=`.
    3. Plain `Load` references to them, such as an injected default or a function passed as an argument. The exceptions are operands of `is`, `is not` and `isinstance` (`chat.py:4226`, `:5468`) and a module-level identity alias (`chat.py:264`).
  - On top of that, add the `.13` direct-adapter reference count (`_references(tree)`, `test_direct_llm_adapter_calls_ratchet.py:84`).
  - A `**splat` is not a surface.
  - `llm_surface=<name>` counts as surfaced when `<name>` is a parameter of the enclosing function (forwarding).
  - Local functions that shadow a gateway name, such as `WebSearch_APIs.py:464` and the `except ImportError` stubs, count by name. Their batch deals with them.
  - `chat_orchestrator.py` and `gateway.py` are excluded (they are the definitions), and so are in-app `tests` packages.

## Review Focus

1. **A refusal that is swallowed, retried or mislabelled.** Watch for `record_failure`, `routing/runtime.py:229` and `except HTTPException → RuntimeError`. Expected: Global Constraint "Refusals propagate", asserted by each block test, with the provider stub called zero times after the block.
2. **Moderated text that leaks through a side channel.** Watch for `provider_response`, raw SSE forwarding (`character_chat_sessions.py:7741-7756`) and a reasoning delta. Expected: no original text reaches the client after a redact or block. Real Anthropic, Google and Cohere frame shapes are used in tests.
3. **Stream chunks that are not plain content.** Watch for `n>1`, missing `index`, tool-call deltas, usage-only chunks, heartbeat or control lines, multi-line items, non-`str` items, error frames, and a stream without `[DONE]`. Expected: the original item object is yielded unless its content changed; held text is flushed at `[DONE]` or at exhaustion.
4. **Per-call cost with moderation off.** Watch for Guardian DB creation, DDL on the event loop, and repeated policy resolution. Expected: no Guardian DB created, at most one policy resolution per call, the engine cached.
5. **Non-text content.** Watch for list content parts, `None` content, tool-call arguments and `reasoning_content`. Expected: only the `text` of `user`/`tool` input messages and string `content` of output are moderated, with nothing crashing. The docs list what is not moderated.

---

### Task 1 (PR A): the unmoderated-call guard; chat names its surface (AC1, AC6)

**Files:**
- Create: `tldw_Server_API/tests/lint/test_unmoderated_llm_calls_guard.py`, `tldw_Server_API/tests/lint/unmoderated_llm_calls_baseline.json`.
- Modify: `.github/workflows/backend-required.yml`, the Enforce step's `run:` list next to `test_direct_llm_adapter_calls_ratchet.py` (:1007). Also `tests/CI/test_license_first_workflow_contracts.py`: add the guard path to the parametrization of `test_backend_required_enforces_llm_gateway_guards` (:1696-1716).
- Modify: chat's sites pass the literal `llm_surface="chat.completions"` explicitly, and `chat_service.build_call_params_from_request` (:1784) stops putting `llm_surface` in the dict:
  - `chat.py:5527, 5613, 5916, 5955, 6066`
  - `completion_pipeline.py:481`
  - `streaming_pipeline.py:722, 1151`
  - `tool_execution_service.py:151`
  - any other reader of `llm_surface` in `cleaned_args`/`refreshed_args` (`git grep -n llm_surface`)

  Keep zero net lines in `create_chat_completion`.
- Test: add `test_every_chat_gateway_call_carries_chat_surface` to `tests/Chat/integration/test_gateway_policy_hooks_chat_endpoint.py`. It records `call.surface` through a test pre-hook across non-stream, stream and tool continuation, and asserts each is `"chat.completions"`.
- Add this plan file.

**Interfaces:**
- Produces: `unsurfaced_gateway_references(tree: ast.Module) -> int` and `count_module(tree) -> int` (unsurfaced gateway references plus `_references(tree)`).
  - Baseline: `{"<app-relative module>": {"references": int, "reason": str|null}}` via `_ceilings.Ratchet`, with the `--tighten`/`--write-baseline` CLI and the raised-ceiling-needs-a-new-reason test (pattern copied from the `.13` ratchet).
  - Import `_references` from the `.13` ratchet; that file stays unmodified.
  - Do not copy its `b"get_adapter"` byte prefilter.
- Every later task runs `--tighten`.

- [ ] **Step 1: Failing self-tests on a seeded tmp tree:**
  - `test_bare_call_without_surface_counts`: `perform_chat_api_call(api_endpoint="x")` → 1.
  - `test_attribute_call_without_surface_counts`: `cs.perform_chat_api_call_async(x)` → 1.
  - `test_call_with_literal_surface_does_not_count`.
  - `test_forwarded_parameter_surface_does_not_count`: `def w(*, llm_surface): perform_chat_api_call(llm_surface=llm_surface)` → 0.
  - `test_non_parameter_name_surface_counts`: `perform_chat_api_call(llm_surface=SOME_GLOBAL)` → 1.
  - `test_splat_alone_counts`: `perform_chat_api_call(**args)` → 1.
  - `test_partial_without_and_with_surface`: → 1 and 0.
  - `test_import_alias_call_counts`.
  - `test_injected_default_reference_counts`: `def f(fn=perform_chat_api_call_async): ...` → 1.
  - `test_identity_comparison_does_not_count`: `x is perform_chat_api_call` → 0.
  - `test_direct_adapter_reference_counts`: `get_registry().get_adapter(p).chat(r)` → 1.
- [ ] **Step 2: Run** and watch them fail.
- [ ] **Step 3: Implement** the counter and ratchet. Make chat's surface explicit, then seed with `--write-baseline`. Expect about 69 modules: the dry run counted 117 before chat's 9 sites were surfaced, and the plain-reference modules add more. Put the real numbers in the PR body.
- [ ] **Step 4: Wire CI** (step and contract).
- [ ] **Step 5: Run:**
  - the guard and `.13` ratchet tests;
  - `tests/CI/test_license_first_workflow_contracts.py -k llm_gateway_guards`;
  - `tests/Chat/integration/test_gateway_policy_hooks_chat_endpoint.py`;
  - the golden, and the `.25` chat-executors gate selection;
  - `tests/lint/test_function_size_ratchet.py`.

  All pass, and the golden is unchanged.
- [ ] **Step 6: Commit:** `test(lint): guard unmoderated LLM call sites; chat names its surface explicitly (TASK-13950.27 PR A)`.

### Task 2 (PR B): surface registry and surface-aware policy (AC3)

**Files:**
- Create: `tldw_Server_API/app/core/Moderation/surfaces.py` and `tldw_Server_API/tests/Moderation/test_surface_policy.py`.
- Modify: `core/Moderation/moderation_service.py` (`get_effective_policy` :516), `supervised_policy.py` (proxy :389-390), and `core/config_sections/moderation.py` (surface keys, plus `stream_buffer_chars` from `MODERATION_STREAM_BUFFER_CHARS`, default 1024).
- Modify: the 10 app callers. Chat's pass `surface="chat.completions"`. `endpoints/moderation.py:601,626`, `moderation_service.py:638` and `Workflows/adapters/llm/moderation.py:110` pass `surface=None`.
- Modify the tests that break, keeping their assertions:
  - Positional real-service calls:
    - `Guardian/test_chat_integration.py:405,428`
    - `Guardian/test_supervised_policy.py:1545`
    - `unit/test_moderation_blocklist_parse.py:524`
    - `unit/test_moderation_policy_evaluator_characterization.py:473,479`
    - `unit/test_moderation_test_endpoint_sample.py:331,365`
    - `unit/test_moderation_user_override_validation.py:323`
    - `Chat_NEW/integration/test_moderation_categories.py:226,314`
  - Fixed-signature fakes: these gain `surface=None`:
    - `Chat_NEW/integration/test_moderation.py:203,223`
    - `test_moderation_categories.py:87,225,313`
    - `Guardian/test_chat_integration.py:85`
    - `Guardian/test_comprehensive_edge_cases.py:718`
    - `Guardian/test_supervised_policy.py:1533`
    - `Chat/unit/test_chat_service_queue_estimate.py:58`
    - `Workflows/adapters/test_llm_adapters.py:979`
    - `unit/test_moderation_event_capture.py:34`
    - `unit/test_moderation_test_endpoint_sample.py:219`
- Modify: the guard file gains `test_every_surface_literal_is_registered`. Each literal `llm_surface=` value must be in `KNOWN_SURFACES`; the only non-literal value allowed is a forwarded parameter.
- Docs: `Config_Files/config.txt` `[Moderation]` (a commented example), `Docs/Code_Documentation/Moderation-Guardrails.md` with its `Docs/Published` mirror (`Helper_Scripts/refresh_docs_published.sh`).

**Interfaces:**
- Produces, in `surfaces.py` (pure data):
  - `CALLER_MODERATED_SURFACES = frozenset({"chat.completions"})`.
  - `EXEMPT_SURFACES: dict[str, str]`, mapping each id to its reason: `internal.byok_probe` (fixed `max_tokens=1` health probe), `internal.first_chat_verifier` (fixed setup prompt), `internal.eval_judge` (owner 2026-11-26: measurement; moderating judge I/O changes scores), `moderation.classifier` (it is the moderation; recorded as an AC5 interpretation).
  - `UNCLASSIFIED_SURFACE = "unclassified"`.
  - `MODERATED_SURFACES`:
    - chat: `chat.completions.router`, `chat.prompt_improvement`, `chat.document_generation`, `chat.macros`, `chat.dialogue_workflows`, `sharing.workspace_chat`
    - messages and character: `messages`, `character.completion`, `character.roleplay_resume`, `character.router`, `character.memory_extraction`
    - voice: `voice.speech_chat`, `voice.chat_stream`, `voice.realtime`, `voice.assistant`
    - workflows, RAG and research: `workflows.step`, `rag.generation`, `rag.auxiliary`, `research`
    - study and prompts: `study.assistant`, `study.packs`, `study.quizzes`, `prompt_studio`
    - media and audio: `media.summarization`, `media.document_insights`, `audio.insights`
    - automation, documents and generation: `scheduled_tasks.automation`, `chunking.boundaries`, `claims.extraction`, `notes.graph_suggestions`, `rpg.rules`, `skills.execution`, `slides.generation`, `slides.standalone_html`
    - web and writing: `web.search`, `web.extraction`, `writing.manuscript`, `writing.note_title`
    - everything else: `vn_play`, `mcp.model_completion`, `evaluations`, `data_tables.generation`, `explainer`
  - `KNOWN_SURFACES` is the union of all of these plus `UNCLASSIFIED_SURFACE`.
- Produces: `get_effective_policy(self, user_id: str | None, *, surface: str | None) -> ModerationPolicy`, on both the service and the proxy, with R5's rule. It returns the unchanged object when no narrowing applies.
- Produces: the config keys `[Moderation] surface.<id> = true|false`, parsed into `surface_enabled: dict[str, bool]`. An unknown id logs one warning and is ignored. The keys are config-only.

- [ ] **Step 1: Failing tests** (`test_surface_policy.py`, with a real service built from a tmp config):
  - `test_global_off_surface_on_stays_disabled`
  - `test_global_on_surface_off_disables_default`
  - `test_global_on_surface_unset_inherits`
  - `test_user_override_enables_when_global_off_and_surface_off`
  - `test_guardian_overlay_enables_when_global_off_and_surface_off` (proxy with a stub engine)
  - `test_surface_none_means_no_narrowing`
  - `test_no_narrowing_returns_the_same_object`
  - `test_unknown_surface_key_is_ignored_with_one_warning`
  - `test_surface_is_a_required_keyword`
  - `test_stream_buffer_chars_reads_config_and_env`
- [ ] **Step 2: Run** and watch them fail.
- [ ] **Step 3: Implement**, then update the callers and the listed tests.
- [ ] **Step 4: Run:**
  - `tests/Moderation/`, `tests/unit/test_moderation_*`, `tests/Guardian/`;
  - `tests/Chat_NEW/integration/test_moderation.py` and `test_moderation_categories.py`;
  - `tests/Workflows/adapters/test_llm_adapters.py` and `tests/Chat/unit/test_chat_service_queue_estimate.py`;
  - the golden and the guard;
  - `tests/lint/test_env_read_ratchet.py`.

  All pass, and the golden is unchanged.
- [ ] **Step 5: Docs.** Add a "Per-surface keys" section (the id table, R5's rule and an example), refresh the mirror, and include the mirror diff.
- [ ] **Step 6: Commit:** `feat(moderation): surface-aware effective policy and [Moderation] surface keys (TASK-13950.27 PR B)`.

### Task 3 (PR C): the hooks; registration in app and workers; the chat auto-router (AC2, AC4)

**Files:**
- Create:
  - `core/Moderation/gateway_hooks.py`
  - `tests/Moderation/conftest.py`, with the `moderation_hooks` and `stub_llm_registry` fixtures. `stub_llm_registry` returns a canned dict or SSE list, records requests, and offers frame builders for OpenAI, Anthropic, Google and Cohere shapes copied from the adapters (`anthropic_adapter.py:544-547`, `google_adapter.py:484,556,582,596`, `cohere_adapter.py:317,413`).
  - `tests/Moderation/test_gateway_hooks.py`, `tests/Moderation/test_gateway_hooks_zero_calls.py`, `tests/lint/test_llm_worker_hook_registration.py`
- Modify the startup and shutdown paths:
  - `services/startup_core_initialization.py:21` (`initialize_startup_core_components`) calls `register_moderation_gateway_hooks()` after `_load_startup_catalogs` (:52), and stores the remover on `app.state`.
  - `services/lifespan_shutdown_sequence.py` calls the remover.
  - The `main()` of every LLM-calling worker calls `register_moderation_gateway_hooks()` too. The set is:
    - the modules launched by `Dockerfiles/docker-compose.workers.yml` (`command:` lines);
    - every module that uses `standalone_llm_provider_overrides`/`_sync` (`core/Provider_Credentials/llm_provider_overrides.py:619,636`, the TASK-15581 standalone-worker marker), except its own definition module;
    - `core/Explainer/jobs_worker.py` and `core/Research/jobs_worker.py`.
- Modify: `core/LLM_Calls/gateway.py` gains public runners `run_policy_pre_hooks(call: GatewayCall) -> None` and `run_policy_post_hooks(call: GatewayCall, result: Any) -> Any`. They are the existing loops at :1348-1353, extracted, with the entries calling them and no behaviour change. Task 10 consumes them.
- Modify: `tests/LLM_Calls/test_gateway_policy_hooks.py:64-65`. Remove registered moderation hooks before asserting `_POLICY_HOOKS == []`.
- Modify: `tests/lint/test_llm_calls_import_isolation.py` gains a case: importing `core.LLM_Calls.gateway` loads no `core.Moderation` module.
- Modify: `core/Chat/chat_orchestrator.py`. `chat_api_call` and `chat_api_call_async` (:235-273, :487-) gain `llm_surface: str | None = None`, forwarded to the gateway call.
- Modify the auto-router and routing runtime:
  - `chat.py:1767` passes `llm_surface="chat.completions.router"`.
  - At `chat.py:1794`, `except LLMPolicyRefusal: raise` goes before `except Exception` (:1798), so `record_failure` is not called.
  - `core/LLM_Calls/routing/runtime.py:229` re-raises `LLMPolicyRefusal` and keeps swallowing everything else; the `LLMQuotaExceededError` debug path is unchanged.
  - At `chat.py:1901`, `LLMPolicyRefusal` joins the re-raise tuple `(ByokResolutionError, ChatAuthenticationError, ChatConfigurationError)`, ahead of the noncritical catch at :1905. That catch includes `HTTPException`, so without the change the refusal would turn into `503 auto_routing_failed`.
  - The character router gets the same change in Task 5.
- Backlog: reword TASK-13950.27 AC2 per R1 via backlog-py.
- Test: `tests/Chat/integration/test_auto_router_moderation.py`

**Interfaces:**
- Consumes: `GatewayCall`, `add_policy_hooks` (`gateway.py:1233-1299`), `LLMPolicyRefusal` (`core/exceptions.py:2251`), `sse_data` (`LLM_Calls/sse.py:42`), and `surfaces.py`.
- Produces (Tasks 5, 9 and 10 consume the text functions; R1's follow-up would too):
  - `moderate_input_messages_for_surface(messages: list[dict], *, surface: str | None, user_id: int | None, policy=None) -> list[dict]`. It returns the same list object when nothing changes, returns a new list on redact, and raises `LLMPolicyRefusal` on block.
  - `moderate_output_text_for_surface(text: str, *, surface, user_id, policy=None) -> tuple[str, str]`, where the action is one of `pass`, `warn`, `redact`, `block`.
  - `class StreamModerator(policy)`, with `feed(text: str) -> tuple[str, bool]` (the text to emit, and whether it blocked) and `flush() -> str`. Holdback is `stream_buffer_chars`.
  - `resolve_policy(call_or_surface, user_id) -> ModerationPolicy | None`. It returns `None` for a skip: a `None` surface (until Task 11), a caller-moderated or exempt surface, re-entry, or a disabled policy or phase.
  - `pre_call(call)` and `post_call(call, result)`.
  - `register_moderation_gateway_hooks() -> Callable[[], None]`. It is idempotent while registered, and its remover resets the flag.
  - `gateway.run_policy_pre_hooks` and `run_policy_post_hooks`, described above.
- **Stream wrapper** (sync, or async when the result has `__aiter__`):
  - For each `str` item, parse only its `data:` line as JSON. Default a missing `index` to 0. Run one `StreamModerator` per choice index over `delta.content`.
  - An item is "unchanged" when its emitted text equals its own text and nothing is held for its choice. Unchanged items, including every non-content item, are yielded as the original object.
  - Any other content item is yielded as `sse_data(chunk)` with the emitted text and without `provider_response`. Drop a chunk whose whole text is held back if it carries nothing else; otherwise keep it with `content` removed.
  - Flush a choice's held text as `sse_data({"choices":[{"index":i,"delta":{"content":held}}]})` immediately before yielding any chunk for that choice that carries `finish_reason`. Also flush before `[DONE]`, and at iterator exhaustion when there is no `[DONE]`.
  - On block, emit a delta with the refusal text (Global Constraints), then `{"choices":[{"index":i,"delta":{},"finish_reason":"content_filter"}]}`, then `data: [DONE]`, then close the provider iterator. Usage is then estimated, as for a client disconnect (`usage_accounting.py:251-270`), which is the accepted trade-off.
  - The holdback (`stream_buffer_chars`) delays the first text of every moderated stream by up to that many characters. Task 11's docs state this.
  - Pass non-`str` items, error frames and non-`data:` lines through untouched.
- **Non-stream:** moderate each `choices[i].message.content` string; on any change, return a new dict without that choice's `provider_response`.

- [ ] **Step 1: Failing unit tests** (`test_gateway_hooks.py`, real gateway with `stub_llm_registry`, a block rule `forbidden`, a redact rule `secret`):
  - Input:
    - `test_input_block_raises_and_provider_not_called`
    - `test_input_redact_reaches_provider_and_caller_list_unchanged`
    - `test_input_moderates_only_user_and_tool_text_parts`
  - Output, non-stream:
    - `test_output_block_non_stream_content_filter_keeps_usage`
    - `test_output_redact_non_stream`
    - `test_non_stream_redact_drops_provider_response` (Google and Cohere shapes)
  - Output, stream:
    - `test_stream_redact_across_chunk_boundary`
    - `test_stream_redact_drops_provider_response` (Anthropic shape)
    - `test_stream_block_emits_content_filter_done_and_closes_provider`
    - `test_unchanged_items_are_the_same_objects`: non-content items, and content items with `stream_buffer_chars=0` (OpenAI compact frames, non-ASCII)
    - `test_flush_precedes_finish_reason_chunk`: a short answer is followed by `finish_reason: stop` → content, then the finish chunk, then usage, then `[DONE]`
    - `test_non_content_items_pass_through` (tool-call delta, usage chunk, heartbeat, `n=2`, missing index, multi-line item, error frame)
    - `test_stream_without_done_flushes_held_text`
    - `test_async_entry_stream_is_wrapped`
  - Skips and policy:
    - `test_caller_moderated_exempt_and_none_surfaces_skip`
    - `test_policy_resolved_once_per_call`
    - `test_guardian_overlay_enables_for_hook_surface`
    - `test_no_guardian_db_means_no_overlay_and_no_file_created`
    - `test_user_id_none_uses_default_policy`
    - `test_reentrant_call_is_not_moderated`
  - Registration:
    - `test_register_is_idempotent_remove_resets_and_reregisters`
    - `test_app_start_stop_start_leaves_one_registration`
- [ ] **Step 2: Zero-calls tests** (`test_gateway_hooks_zero_calls.py`), parametrized over `sorted(MODERATED_SURFACES)`, stream and non-stream:
  - global off with the surface key on, and global off with no key → the evaluator spy (`policy_evaluator.evaluate_text`/`evaluate_action*`) records 0 calls, output equals the stub's exactly, and no Guardian DB file is created;
  - sanity: global on → more than 0 calls;
  - `test_chat_completions_global_off_zero_moderation_calls[stream|non-stream]`, through `/api/v1/chat/completions` with provider `anthropic` and the stub registry.
- [ ] **Step 3: Worker lint** (`test_llm_worker_hook_registration.py`). For every module in the worker set (the compose `command:` modules, the `standalone_llm_provider_overrides*` users, and the two named `jobs_worker.py` files), assert its source calls `register_moderation_gateway_hooks`. An allowlist names the workers that make no LLM call, each with a reason. Self-test it on a seeded compose file and a seeded overrides user.
- [ ] **Step 4: Auto-router tests** (`test_auto_router_moderation.py`; fixtures `authenticated_client`, `setup_dependencies`, `mock_chacha_db` from `tests/_plugins/chat_fixtures.py:413-470`, plus `test_chat_endpoint_auto_routing.py`'s routing setup minus its `perform_chat_api_call_async` patch):
  - `test_router_input_block_returns_400_without_record_failure`. Set global on and `surface.chat.completions = false`, so only the router can block. Assert 400 (not 503), router-provider stub calls 0, and `record_failure` not called.
  - `test_runtime_still_swallows_other_router_errors`. A generic router error still falls back to rules routing, and the quota debug path is unchanged.
  - `test_router_receives_redacted_latest_turn`.
  - `test_router_global_off_zero_moderation_calls`.
- [ ] **Step 5: Run** each step's tests and watch them fail.
- [ ] **Step 6: Implement** the hooks, the gateway runners (with `tests/LLM_Calls/test_gateway_policy_hooks.py` unchanged and passing), registration (app and workers), the teardown, the orchestrator parameter, the router and runtime fixes, and the AC2 reword.
- [ ] **Step 7: Run:**
  - the new tests;
  - `tests/LLM_Calls/test_gateway_policy_hooks.py`;
  - `tests/Chat/integration/test_gateway_policy_hooks_chat_endpoint.py` and `test_chat_endpoint_auto_routing.py`;
  - `tests/Character_Chat_NEW/integration/test_character_chat_auto_routing.py`;
  - the golden and the `.25` gate selection;
  - `tests/lint/` (isolation, layering, guard `--tighten`, function size, env read).

  All pass.
- [ ] **Step 8: Commit:** `feat(moderation): gateway input/output hooks registered in app and workers; moderate the chat auto-router (TASK-13950.27 PR C)`.

### Task 4 (PR D): `/messages`

**Files:**
- `api/v1/endpoints/messages.py`:
  - The gateway path (:2171, :2198) passes `llm_surface="messages"`.
  - The native path is moderated in its caller in `_handle_messages` (:2095-2150), never inside `_native_post_json` or `_prepare_native_stream_iterator`. Those two are shared with `count_tokens` (:2263), and tests patch them.
  - The request is moderated before the POST, the response after it, and the stream before `_sanitize_native_messages_stream` wraps it.
- `core/Moderation/gateway_hooks.py` gains:
  - `moderate_anthropic_request(payload: dict, *, surface, user_id) -> dict`. It moderates `text` blocks of `user` messages and `tool_result` text, and leaves the top-level `system` alone.
  - `moderate_anthropic_response(resp: dict, *, surface, user_id) -> dict`. Block sets the text blocks to the refusal text and `stop_reason` to `"refusal"`.
  - `wrap_anthropic_stream(iterator, *, surface, user_id)`. It moderates `content_block_delta` events whose `delta.type == "text_delta"`, yields original items when nothing changes, and on block emits `message_delta` (`stop_reason: "refusal"`) followed by `message_stop`.
- Test: `tests/Moderation/surfaces/test_messages_moderation.py`, using `test_messages_endpoint.py`'s fixtures with the stub registry in place of its `chat_service` patch, and patching `_native_post_json` and `_prepare_native_stream_iterator` for the native path.

- [ ] **Step 1: Failing tests:**
  - gateway provider: `test_messages_input_block_400`, `test_messages_output_redact`;
  - native `anthropic`: `test_native_messages_input_block_400_no_post`, `test_native_messages_output_redact`, `test_native_messages_stream_redact_and_block`, `test_native_messages_global_off_zero_calls`;
  - `test_count_tokens_is_not_moderated` (global on, a blocked word in the body → normal count response).
- [ ] **Step 2: Run** and watch them fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the new tests, `tests/Chat/integration/test_messages_endpoint.py` and the guard (`--tighten`). All pass.
- [ ] **Step 5: Commit:** `feat(moderation): moderate /api/v1/messages, native providers included (TASK-13950.27 PR D)`.

### Task 5 (PR E): character chat and chat-adjacent surfaces

**Files:**
- `character_chat_sessions.py`:
  - `character.router` at :1141. `LLMPolicyRefusal` is re-raised in that function's except before any `record_failure`. It also goes before the noncritical catch at :1198, whose tuple includes `HTTPException`.
  - `character.roleplay_resume` at :6852-6853;
  - `character.completion` at :7449-7450 (zero net lines).
- `character_memory_extraction.py:224-231` → `chat_api_call(..., llm_surface="character.memory_extraction")`, using Task 3's orchestrator parameter.
- Chat-adjacent sites. Each injected default becomes a surfaced partial; callers that inject the raw function (`prompt_improvement_dispatch.py:118`) inject a surfaced partial. Grep existing test stubs for fixed signatures before adding the keyword.
  - `chat_route_resolver.py` (calls via `perform_chat_api_call_async_fn`, :146) → `chat.prompt_improvement`. Its provider fallback does not re-ask after a `content_filter` result.
  - `prompt_improvement_dispatch.py:211` → `chat.prompt_improvement`
  - `document_generator.py:761` (alias `chat_api_call`) → `chat.document_generation`
  - `branch_runner.py:67` (injected default) → `chat.macros`
  - `dialogue_orchestrator.py:6,20` (an orchestrator call, `self._llm_caller`) → `chat.dialogue_workflows`
  - `shared_workspace_chat_service.py:726` → `sharing.workspace_chat`
- Test: `tests/Moderation/surfaces/test_character_moderation.py`, using the fixtures of `tests/Character_Chat/test_complete_v2_streaming_e2e_mock.py` and `tests/Chat_NEW/integration/test_roleplay_resume_completion.py` with the stub registry. Also `test_chat_adjacent_surfaces.py`.

- [ ] **Step 1: Failing tests:**
  - complete-v2: `test_complete_v2_input_block_400`, `test_complete_v2_output_redact`, `test_complete_v2_stream_redact_with_anthropic_frames_leaks_nothing` (the raw-line forwarding at :7741-7756);
  - resume and router: `test_roleplay_resume_input_block`, `test_character_router_block_400_no_fallback` (global on, `surface.character.completion = false`);
  - chat-adjacent: `test_surface_reaches_gateway[<site>]`, one case per site;
  - prompt improvement: `test_prompt_improvement_router_block_propagates` and `test_prompt_improvement_content_filter_not_refallen_back`.
- [ ] **Step 2: Run** and watch them fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the new tests, `tests/Character_Chat*`, `tests/Chat_NEW/integration/test_roleplay_resume_completion.py`, the touched chat-adjacent modules' tests, the guard (`--tighten`) and function size. All pass.
- [ ] **Step 5: Commit:** `feat(moderation): moderate character chat and chat-adjacent surfaces (TASK-13950.27 PR E)`.

### Task 6 (PR F): voice

**Files:**
- `core/Streaming/speech_chat_service.py`:
  - Delete the direct adapter branches (:773-799).
  - The gateway call (:812) passes `llm_surface="voice.speech_chat"`, keeping `timeout=SPEECH_CHAT_LLM_TIMEOUT_SECONDS`, `response_format` and `user`.
  - `except HTTPException → RuntimeError` (:808, :845) lets `LLMPolicyRefusal` through.
- `api/v1/endpoints/audio/audio_streaming.py`:
  - Delete the direct `astream` (:2476).
  - `_fallback_stream` (:2427) passes `llm_surface="voice.chat_stream"` and `provider_llm_extra_params` top-level, as the direct path did (:2423-2426).
  - `except Exception` (:2542) maps `LLMPolicyRefusal` to an error frame with code `llm_policy_refused`.
- `core/Audio/Realtime/default_pipeline.py:368` → `voice.realtime`.
- `VoiceAssistant/router.py:1153` and `intent_parser.py:404`: add `llm_surface="voice.assistant"` only (R8).
- Test: `tests/Moderation/surfaces/test_voice_moderation.py`, with the STT/TTS stubs of `tests/Audio/test_audio_chat_endpoint.py` and the WS harness of `tests/Audio/test_ws_audio_chat_stream.py`.

- [ ] **Step 1: Failing tests:**
  - `/audio/chat`: `test_audio_chat_input_block_is_a_refusal_not_provider_failure`, `test_audio_chat_output_redact`;
  - WS: `test_ws_chat_stream_output_redact`, `test_ws_chat_stream_block_frame_code_llm_policy_refused`;
  - other: `test_realtime_pipeline_surface`, and R6's validation-path test for each migrated site.
- [ ] **Step 2: Run** and watch them fail.
- [ ] **Step 3: Implement**, listing R6's accepted changes in the PR.
- [ ] **Step 4: Run** the new tests, `tests/Audio/test_speech_chat_service.py`, `test_audio_chat_endpoint.py`, `test_ws_audio_chat_stream.py`, `tests/VoiceAssistant/`, the guard, the `.13` ratchet (`--tighten`) and function size. All pass. Patch targets the migration moved are listed in the PR.
- [ ] **Step 5: Commit:** `feat(moderation): moderate the voice surfaces (TASK-13950.27 PR F)`.

### Task 7 (PR G): workflows, RAG, research

**Files:**
- **Workflows.** Every gateway call in `core/Workflows/adapters/` gets `workflows.step`: `llm/llm.py`, `content/generation.py` (its `except ImportError` stub at :47 gains the parameter), `text/nlp.py`, `content/audio_briefing.py`, `rag/query.py`, `research/bibliography.py`, `content/image.py`, `integration/messaging.py`, `media/documents.py`. `llm/translate.py`'s direct call moves to the gateway (R6).
- **RAG.**
  - `rag_service/generation.py:532` → `rag.generation`.
  - `query_classifier.py:317,471`, `media_search.py:108`, `research_agent.py:1341` and `suggestion_generator.py:234` → `rag.auxiliary`.
- **Research.** `Research/providers/synthesis.py` (an injected `self._chat_fn` → surfaced partial) and `Research_Workspace/artifact_generation.py:81` → `research`.
- Name each refusal site in the PR. A workflow step fails with the refusal text; RAG returns 400.
- Test: `tests/Moderation/surfaces/test_workflow_moderation.py`, using the setup of `tests/Workflows/adapters/test_content_adapters.py`. Also `test_rag_moderation.py`, using `tests/RAG_NEW/integration/test_rag_stream_parity.py:380-397` with the stub registry.

- [ ] **Step 1: Failing tests:**
  - workflows: `test_workflow_llm_step_input_block_fails_step` (refusal text, provider 0 calls), `test_workflow_llm_step_output_redact`;
  - RAG: `test_rag_generation_input_block_400`, `test_rag_generation_answer_redacted`, `test_rag_stream_answer_redacted`;
  - other: `test_surface_reaches_gateway[<site>]` for translate, `rag.auxiliary` and research, plus `test_rag_auxiliary_content_filter_not_retried` (query classifier).
- [ ] **Step 2: Run** and watch them fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the new tests, `tests/Workflows/adapters/`, `tests/RAG_NEW/unit/test_generation_*` and `tests/RAG_NEW/integration/test_rag_stream_parity.py`, plus the guard and the `.13` ratchet (`--tighten`). All pass.
- [ ] **Step 5: Commit:** `feat(moderation): moderate workflow LLM steps, RAG and research (TASK-13950.27 PR G)`.

### Task 8 (PR H): study tools and Prompt Studio

**Files:**
- **Study.**
  - `Flashcards/study_assistant.py:473` → `study.assistant`. Its `except ImportError` stub at :22 gains the parameter.
  - `StudyPacks/generation_service.py:597` → `study.packs`, with the same change to its stub at :55.
  - `MCP_unified/.../quizzes_module.py:1097` and `services/quiz_generator.py:1535` (direct → gateway) → `study.quizzes`.
- **Prompt Studio.**
  - `prompt_studio/{prompt_executor,evaluation_manager,prompt_generator,prompt_improver,test_runner}.py`: direct calls move to the gateway as `prompt_studio`. The `prompt_executor.py:277-345` request maps one-to-one onto gateway kwargs. Its legacy `_legacy_call` (:519-522) passes `llm_surface` through Task 3's orchestrator parameter.
  - `Prompt_Engineering.py`: the wrapper (:45-63) takes and forwards `llm_surface`; :561 and :661 and the direct call → `prompt_studio`.
  - The Prompt Studio jobs worker is covered by Task 3's worker registration.
- Test: `tests/Moderation/surfaces/test_study_moderation.py`, using `tests/Flashcards/test_flashcards_endpoint_integration.py` and `tests/DB_Management/test_study_pack_worker_*` with `RUN_JOBS=1`. Also `test_prompt_studio_moderation.py`, using the `client` of `tests/prompt_studio/integration/test_api_endpoints.py:192`.

- [ ] **Step 1: Failing tests:**
  - study: `test_study_assistant_input_block_400`, `test_study_assistant_output_redact`, `test_study_pack_generation_block_fails_job`, `test_quiz_generation_surface`;
  - `test_study_pack_worker_call_carries_owner_user_id`. The worker's `scoped_context(user_id=owner)` (`study_pack_jobs_worker.py:65`) reaches `call.user_id`, so a per-user override applies in a worker;
  - Prompt Studio: `test_prompt_studio_execute_input_block`, `test_prompt_studio_execute_output_redact`, `test_surface_reaches_gateway[<prompt studio module>]`;
  - R6's validation-path test for each migrated site.
- [ ] **Step 2: Run** and watch them fail.
- [ ] **Step 3: Implement**, listing R6's changes in the PR.
- [ ] **Step 4: Run** the new tests, `tests/Flashcards/`, `tests/prompt_studio/`, the StudyPacks worker tests (`RUN_JOBS=1`), the guard and the `.13` ratchet (`--tighten`). All pass.
- [ ] **Step 5: Commit:** `feat(moderation): moderate study tools and Prompt Studio (TASK-13950.27 PR H)`.

### Task 9 (PR I): remaining gateway surfaces; Slides standalone and MCP model completion

**Files:**
- **Gateway calls**, with surfaced partials for injected defaults (`auto_boundary_assistant.py`, `slides_generator.py`, `extraction/dependencies.py` → `strategies/llm.py:170`):
  - `Scheduled_Tasks/automation_executors.py:267,331` → `scheduled_tasks.automation`
  - `Chunking/auto_boundary_assistant.py` → `chunking.boundaries`
  - `Claims_Extraction/claims_service.py:621` → `claims.extraction`
  - `Claims_Extraction/ingestion_claims.py:361-363` → `claims.extraction`. Its `_cac` orchestrator call takes the parameter, and its direct call moves to the gateway (R6).
  - `Notes_Graph/suggestion_generation.py:793` → `notes.graph_suggestions`
  - `RPG/rules/answering.py:84` → `rpg.rules`
  - `Skills/skill_executor.py:436` → `skills.execution`
  - `Slides/slides_generator.py` → `slides.generation`
  - `Web_Scraping/extraction/` → `web.extraction`
  - `Writing/manuscript_analysis.py:143`, `manuscript_annotation_jobs.py:147` and `manuscript_annotations.py:227` → `writing.manuscript`
  - `VN_Play/adapters.py`: the wrapper (:87) forwards the parameter; :142 and :199 → `vn_play`
- **Raw HTTP** (R7):
  - `Slides/standalone_html_provider.py` → `slides.standalone_html`, using the OpenAI or Anthropic functions by target. The request is moderated after `_build_request` (:402), and the text after `_extract_text` (:450), both in the `_request_once` path (:555). A refusal maps to that module's `_fail("llm_policy_refused")`.
  - `MCP_unified/adapters/model_completion/transport.py` → `mcp.model_completion`, using the OpenAI functions. The payload is moderated before the POST (:359) and the response after it. A refusal maps to a `ModelFailureDomain` value named for policy refusal; add one if none exists.
- Test: `tests/Moderation/surfaces/test_other_surfaces.py` and `test_raw_http_surfaces.py`

- [ ] **Step 1: Failing tests:**
  - `test_surface_reaches_gateway[<site>]`, one case per site;
  - `test_claims_extraction_block_propagates_and_is_not_retried`;
  - `test_slides_generation_output_redact`;
  - `test_slides_standalone_input_block_no_post[openai|anthropic]` (asserts the `llm_policy_refused` code), `test_slides_standalone_output_redact`;
  - `test_mcp_model_completion_input_block_and_output_redact` (asserts the refusal domain), `test_raw_http_global_off_zero_calls`.
- [ ] **Step 2: Run** and watch them fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the new tests, the touched packages' test directories, the guard and the `.13` ratchet (`--tighten`). All pass.
- [ ] **Step 5: Commit:** `feat(moderation): moderate the remaining gateway surfaces, Slides standalone and MCP model completion (TASK-13950.27 PR I)`.

### Task 10 (PR J): remaining direct-adapter surfaces

**Files:**
- **Evaluations** (owner decision): `Evaluations/{eval_runner,ms_g_eval,wordbench_runner}.py` and `Evaluations/cli/{evals_cli,evals_cli_enhanced}.py`.
  - Judge calls → `internal.eval_judge`; evaluated-model generation calls → `evaluations`.
  - Direct calls move to the gateway (R6).
  - The CLIs are operator tools and do not register hooks. Their judge calls are exempt anyway; this is recorded in the PR.
- **Data tables:** `Data_Tables/jobs_worker.py` → `data_tables.generation`.
- **Explainer:** `Explainer/jobs.py` → `explainer`.
- **Documents and audio:** `endpoints/media/document_insights.py` → `media.document_insights`; `Audio/Audio_Streaming_Insights.py` → `audio.insights`.
- **Web search:** the whole of `Web_Scraping/WebSearch_APIs.py`, where the local `chat_api_call` (:464) and `_call_adapter_text` move to the gateway and :944 and :1386 pass `web.search`, and `WebSearch/Web_Search.py` → `web.search`.
- **Writing:** `Writing/note_title.py` → `writing.note_title`.
- **Summarization** (R6 exception): `LLM_Calls/Summarization_General_Lib._summarize_via_adapter` stays direct. It never imports Moderation (Layering).
  - It builds `GatewayCall(surface=llm_surface, provider=..., request=..., stream=..., user_id=current_llm_user_id())`.
  - It calls `gateway.run_policy_pre_hooks(call)` before the adapter call (:385, :416), then `run_policy_post_hooks(call, result)` on the adapter's raw response or raw SSE stream, before its own closure parses that into text.
  - `LLMPolicyRefusal` joins `_SUMMARY_PASSTHROUGH_EXCEPTIONS` (:122), so it is not wrapped as a provider failure.
  - `analyze()` takes `llm_surface` from its caller, defaulting to `media.summarization`. The RAG gap analysis (`pipeline/augment.py:196-240`) passes `rag.auxiliary`.
  - The PR lists every `analyze()` importer (`git grep -l "analyze(" -- tldw_Server_API/app`, about 46) whose broad `except` would swallow the refusal, and how each now surfaces it, per Global Constraint "Refusals propagate".
  - Its guard entry carries the R6 reason.
- Test: `tests/Moderation/surfaces/test_direct_adapter_migrations.py`

- [ ] **Step 1: Failing tests:**
  - `test_surface_reaches_gateway[<site>]`, one case per site;
  - `test_eval_judge_is_exempt_and_generation_is_moderated`;
  - `test_analyze_input_block_and_output_redact` (stream and non-stream), `test_summarization_imports_no_moderation` (a fresh-process import check);
  - `test_workflow_summarize_step_block_fails_step`, `test_media_ingest_summary_block_is_recorded_not_retried`;
  - `test_analyze_still_refuses_disabled_model_and_keeps_error_codes` (`dns_changed` and `tls_pin_error` stay `provider_failure`);
  - `test_note_title_input_block_propagates`;
  - R6's validation-path test for each migrated site.
- [ ] **Step 2: Run** and watch them fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run:**
  - the new tests;
  - `tests/Evaluations/` with `RUN_EVALUATIONS=1`;
  - the TASK-15581 and summarization tests (`git grep -l _summarize_via_adapter tldw_Server_API/tests`);
  - the touched packages' tests (Data Tables with `RUN_JOBS=1`);
  - both ratchets (`--tighten`).

  All pass. The `.13` baseline then holds only `byok_testing.py` and `Summarization_General_Lib.py` (with reasons).
- [ ] **Step 5: Commit:** `feat(moderation): move the remaining direct adapter calls onto the gateway; moderate summarization explicitly (TASK-13950.27 PR J)`.

### Task 11 (PR K): backstop, exemptions, close-out (AC4, AC5)

**Files:**
- **Exemptions:**
  - `Provider_Credentials/byok_testing.py:657` → `internal.byok_probe`. Its direct call moves per R6 or keeps that reason.
  - `Setup/first_chat_verifier.py:71` → `internal.first_chat_verifier`.
  - `Moderation/classification.py:154` → `moderation.classifier`.
  - `semantic_matcher.classify_with_llm` (no callers, legacy kwargs) gets the same exempt id and a follow-up to delete it.
- `gateway_hooks.py`: `surface=None` is moderated as `UNCLASSIFIED_SURFACE`, with one `logger.warning("unsurfaced LLM call provider={provider}")` per provider per process.
- **Raw-HTTP sweep:** grep `tldw_Server_API/app` for provider endpoint POSTs (`/chat/completions`, `/v1/messages`, `/completion`, `generateContent`) outside `LLM_Calls/providers`. Each hit is either moderated already (Tasks 4 and 9), exempt by owner decision (OCR), or listed as a follow-up in the PR body.
- **Baselines:** both are tightened. The guard baseline ends empty except the Summarization R6 entry and any exempt-internal module, each with its reason (AC5).
- **Backlog** (backlog-py):
  - Check off the TASK-13950.27 ACs, and add notes and a final summary. Record as interpretations: the AC5 exemptions (`moderation.classifier`, `internal.eval_judge`) and the R9 counting rule.
  - File follow-ups:
    - audit parity on hook surfaces (R3);
    - self-monitoring and topic monitoring on hook surfaces, and `scoped_context` in `core/Workflows` (R4);
    - the `/llamacpp` passthrough, VN Play's moderation adapter, and the `Local_LLM` handlers' reachability (R7);
    - the voice assistant revival decision (R8);
    - converging chat onto the hook functions, and moderating tool-continuation outputs (R1);
    - moderating list content parts, tool-call arguments and `reasoning_content`;
    - deleting `semantic_matcher.classify_with_llm`.
- **Docs:** Moderation-Guardrails.md gets a full refresh, then the mirror refresh. It covers:
  - the final surface table, and the exemptions with their reasons: OCR, and the evals CLIs, which are operator tools that register no hooks;
  - `unclassified`, and what is not moderated;
  - the stream latency cost of `stream_buffer_chars`;
  - the known difference that hook surfaces moderate only the provider request, so a surface may persist unredacted user text.
- **Tests:**
  - `test_gateway_hooks_zero_calls.py` is extended to `sorted(MODERATED_SURFACES | {UNCLASSIFIED_SURFACE})`.
  - `test_unsurfaced_call_is_moderated_as_unclassified`
  - `test_exempt_surfaces_are_never_moderated`
  - `test_exempt_classifier_call_is_not_moderated`

- [ ] **Step 1: Failing tests** (as listed).
- [ ] **Step 2: Run** and watch them fail.
- [ ] **Step 3: Implement** the backstop, exemptions and sweep, then tighten both baselines.
- [ ] **Step 4: Run** `tests/Moderation/`, `tests/LLM_Calls/test_gateway_policy_hooks.py`, the `.25` gate selection, `tests/Chat_NEW/integration/test_moderation*.py` and `tests/lint/`. All pass.
- [ ] **Step 5: Commit:** `feat(moderation): moderate unsurfaced LLM calls as unclassified; exempt internal calls; close TASK-13950.27 (PR K)`.

---

## Review dispositions (plan-review.md, round 1)

| Finding | Where handled |
|---|---|
| C1 workers | Task 3 (`register_moderation_gateway_hooks()` in the app lifespan and every worker `main()`, worker lint) |
| C2 `provider_response` | Global Constraint "No leaks"; Task 3 stream and non-stream; Task 5 complete-v2 raw-line test |
| I1 Guardian cost | R4; Task 3 tests (no DB created, resolved once) |
| I2 refusal sites | Global Constraint "Refusals propagate"; Task 3 (router, runtime); Tasks 5-6 named sites |
| I3 breaking tests | Task 2 Files and Step 4 |
| I4 orchestrator signature | Task 3 (`llm_surface` parameter) |
| I5 forwarding | R9; Task 1 test |
| I6 plain references | R9 counts them; surfaced partials (Tasks 5, 7, 9) |
| I7 WebSearch_APIs | Task 10, whole module |
| I8 R6 | R6 rewritten; Summarization stays direct (Task 10) |
| I9 stream format | Task 3 stream wrapper spec and tests |
| I10 teardown | Task 3 (remover on app.state, shutdown, reset) |
| I11 voice assistant | R8 (surface only, follow-up) |
| I12 raw HTTP | Owner 2026-11-26; R7; Tasks 9 and 11 sweep |
| I13 eval judges | Owner 2026-11-26; Task 10 (`internal.eval_judge`) |
| I14 chat zero calls | Task 3 Step 2 endpoint test |
| M1 `_references` | Task 1 Interfaces |
| M2 call lines | Tasks 5, 7, 9 Files |
| M3 env read | Task 2 (`stream_buffer_chars`) |
| M4 function size | Global Constraint "Ratchets" |
| M5 usage on block | Task 3 stream wrapper (close; estimate) |
| M6 no retry | Global Constraint; Tasks 7, 9 tests |
| M7 unmoderated parts | Review Focus 5; Task 11 docs and follow-up |
| M8 log fields | R3 |
| M9 classifier recursion | Task 11 test renamed; contextvar kept |
| M10 exempt interpretation | Task 11 final summary |
| M11 analyze surface | Task 10 (caller-supplied) |
| M12 env gates | Global Constraint; Tasks 8, 10 |
| M13 #3437 | Global Constraint "Process" |
| M14 AC2 reword | Task 3 |
| M15 persisted text | Task 11 docs |
| M16 chat_type | R4 (`"regular"`) |

## Review dispositions (plan-rereview.md, round 2)

| Finding | Where handled |
|---|---|
| N1 router swallow | Task 3 (`chat.py:1794` and `:1901` re-raise, runtime re-raise, nonlocal dropped); Task 5 (character `:1141`/`:1198`, prompt-improvement test) |
| N2 flush order and "unchanged" | Task 3 stream wrapper spec and tests; Task 11 docs (latency) |
| N3 Summarization layering | Task 3 gateway runners; Task 10 uses them; `analyze()` importers listed |
| N4 chat surface argument | Literal `"chat.completions"` (Task 1; Global "Surface ids"; Task 2 literal test) |
| N5 native `/messages` placement | Task 4 (caller at :2095-2150; count_tokens test) |
| N6 standalone workers | Task 3 worker set and lint (`standalone_llm_provider_overrides*` users plus Explainer and Research) |
| m1 raw-HTTP lines and taxonomies | Task 9 |
| m2 stream block text | Global "Refusal text"; Task 3 |
| m3 retry owners | Global "Refusal text" (named sites); Tasks 5, 7, 9 |
| m4 evals CLIs | Task 11 docs |
| m5 worker identity | Task 8 test |

## Execution notes

Added at close-out (PR K). The plan above is unchanged; this records what execution added or changed.

- **R10 was added during execution** (from Task 6): a client- or config-controlled mapping must never be able to set `llm_surface`, because a later `**splat` can override a bound partial and an exempt id smuggled through user data would skip moderation. Tasks 7-10 drop the key at each such site with a test, and Task 11 audited every `**splat` gateway call branch-wide and added a static lint that lets each exempt or caller-moderated id appear as an `llm_surface` literal only in its designated modules.
- **Task 3 was split into parts a and b** (PR C holds the hooks and policy; the app wiring, worker registration and chat endpoint tests followed in part b).
- **PRs**: A-J are #3649-#3662 (B #3650, C #3652, D #3656, E #3657, F #3658, G #3659, H #3660, I #3661, J #3662); K is the close-out. They were stacked, merged bottom-up, and each stacked base was retargeted to `dev` before its predecessor's branch was deleted.
- **Test-quality lesson**: the test-quality ratchet counts tautology-suspect assertions, so a test whose assertion cannot fail (a split word in a stream-leak test, an evader fixture that dodges the endpoint-monkeypatch counter) is rejected by review or CI; write the failing case first and drive the real call.
- **Shard-coverage lesson**: a new test directory belongs to no full-suite shard until it is added to one; the non-required Shard coverage guard caught `tests/Moderation` being silently skipped, so add the directory to a shard in the PR that creates it.
