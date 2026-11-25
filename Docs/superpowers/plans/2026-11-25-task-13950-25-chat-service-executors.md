# Decompose the chat_service executors (TASK-13950.25) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
- `execute_streaming_call` and `_execute_non_stream_call_impl` become kwargs shims of about 50 and 38 lines.
  - Each shim calls a stage orchestrator of at most 280 lines: `streaming_pipeline.run_streaming_call(cst)` (~228) and `completion_pipeline.run_non_stream_call(cst)` (~107).
  - The orchestrators' bodies move into the modules the 2026-06-24 design names.
- `build_context_and_messages` moves to `history_context.py`.
- `ChatCompletionPipeline` is deleted.
- `chat_service.py` drops from 6,835 lines to about 3,460.
- Chat output stays identical throughout.

**Architecture:** Eleven tasks and eleven PRs, merged strictly in order. The method is TASK-13950.24's, reused.
1. A golden-output test and a path-gated required job land first (Task 1).
2. The .24 move tool is generalised in three PRs. Each lands just before the PR that first uses its features (Tasks 2, 4 and 6).
3. For each executor in turn, non-stream first because it sits at the file's tail:
   - a pure rewrite moves every name that crosses a planned boundary onto a per-call state object, `cst` (Tasks 3 and 7);
   - closures then *lift* to module functions, and statement spans *move*;
   - the remaining body becomes the orchestrator behind a shim (Tasks 5, 8 and 9).
4. `build_context_and_messages` relocates (Task 10), and Task 11 closes out.

`Helper_Scripts/rag_split/rag_split.py` produces every rewrite, lift, move and relocate. Its `verify` proves each one against the PR's base, independently of `apply`'s text edits.

**Tech Stack:** Python 3.12 (stdlib `ast`, `symtable`, `functools`), pytest, GitHub Actions (`backend-required.yml`), and the AST lint ratchets under `tldw_Server_API/tests/lint/`.

**Spec:**
- `backlog/tasks/task-13950.25 - Decompose-chat-service-executors-and-make-ChatCompletionPipeline-real.md`, AC #1-#5.
- `Docs/Design/2026-06-24-chat-completion-pipeline-refactor-design.md`, which names the modules.
- The roadmap's Phase 3 `.25` row: `Docs/superpowers/plans/2026-10-25-architecture-remediation-roadmap.md`.
- Two research files at dev `8f33908da4`, both git-ignored, both in the ledger workspace `.superpowers/sdd/2026-11-25-task-13950-25-chat-service-executors/`:
  - `survey.md`, cited below as §n;
  - `plan-review.md`, the plan review and dry-run, cited as Dn/Cn/In/Mn. Its corrected spans and measurements are folded in below.

## Measured (dev 8f33908da4)

- **File.** `core/Chat/chat_service.py` is 6,835 lines with 103 top-level defs.
  - `execute_streaming_call` is `:4227-5768` (1,542 lines, 38 keyword-only params).
  - `_execute_non_stream_call_impl` is `:5771-6824` (1,054 lines, 30 params).
  - `build_context_and_messages` is `:3056-3871` (816 lines, 8 params).
  - Their `function_size_baseline.json` ceilings are 1,545/38, 1,054/30 and 821/8. THRESHOLD is 300.
- **Fit (D1).** After Appendix B's lifts and moves (corrected spans):
  - `run_non_stream_call` is ~107 lines and `run_streaming_call` ~228. The shims are 38 and 50.
  - The largest new function is `run_non_stream_tool_autoexec` (~232).
  - `chat_service.py` after PR J is ~3,457 lines.
  - The rewrites alone change the executors by −2 and −7 lines (prologue +2, deleted `nonlocal`s), so neither ceiling is at risk (D3).
- **Shape (§1-§3, D2).**
  - Neither executor is a generator, and neither has a failover loop: failover is one attempt inside an `except`.
  - The streaming executor returns a `StreamingResponse`, and 849 lines of its callbacks (S10-S14) run after it has returned.
  - Rebound parameters: `cleaned_args`, `model`, `selected_provider` and `structured_request_context` in both executors, plus `llm_call_func` in streaming. Every other parameter is read-only.
  - Streaming's `_queued_processor` rebinds six names from the queue worker, but only on its fallback branch (`:4453-4538`).
  - `tracked_streaming_generator` (`:5390`) owns 9 locals that its nested defs read: `chunk_seq`, `stream_holdback`, `eff_policy`, `moderation`, `req_user_id`, `stream_id`, `mandatory_moderation_audit_tasks` and `stream_buffer_limit`, plus `StopStreamWithError`, from a function-local import at `:5402`. None of them collides with an executor local.
  - No `locals()`, `del`, `global` or `in locals()` appears in the three functions.
  - `return` appears at streaming `:4694`, `:4768`, `:4828`, `:5750` and `:5760`, and at non-stream `:6824` only.
  - Handler variables: non-stream `:6156` and streaming `:4695` bind `e`; streaming `:4649` binds `he`.
- **Errors.** `_CHAT_NONCRITICAL_EXCEPTIONS` (`:238`) holds `AttributeError`, `KeyError`, `TypeError`, `RuntimeError`, `OSError` and others, but not `NameError`.
- **Design modules (§5).**
  - Non-stream moderation and response processing already delegate.
  - Streaming moderation is about 380 inline lines.
  - `streaming_pipeline.py` (79 lines) is a kwargs forwarder.
  - `tool_execution_service.py` (77 lines) is only a request guard; non-stream autoexec plus auto-continue is 231 inline lines.
  - `command_authorization.py` and `chat_logging.py` have no executor lines to take.
- **ChatCompletionPipeline (§6).**
  - It is a 26-line frozen dataclass of two callables, built on every call (`:6827-6831`); `completion_pipeline.py:26` exports it in `__all__`.
  - It forwards `execute_non_stream_call(**kwargs)` to `_execute_non_stream_call_impl`. `:5655` calls `get_chat_completion_pipeline().execute_streaming(...)`, which forwards to `create_chat_streaming_response`, re-read as a global on every call.
  - Its only other users are 2 tests (`tests/Chat/unit/test_chat_service_content.py:78-124`), `core/Chat/README.md:21`, and design `:45`, `:67` and `:235`.
- **Patching (§7b).** Tests patch 60 `chat_service` names at 1,155 sites (a floor).
  - The top ones are `perform_chat_api_call` (295), `perform_chat_api_call_async` (242), `log_llm_usage` (80) and `get_request_queue` (78).
  - `time` (4 sites) is the only stdlib name patched on the module.
- **Imports (C1).**
  - `chat_service` imports `completion_pipeline`, `moderation_pipeline`, `persistence_service`, `response_processor`, `streaming_pipeline` and `tool_execution_service` **by name** (`:75-155`).
  - `moderation_pipeline.py:24-30` imports five names from `response_processor` by name; all five are read only inside function bodies.
  - None of those modules imports `chat_service`, and the design's import rule (`:59`) forbids it.
  - All seven destination modules have `from __future__ import annotations`.
  - `E402` and `E501` are ignored. CI's ruff step is `continue-on-error`, and nothing reformats these files.
- **Unused imports (I5).** 58 names `chat_service` imports are read only inside the three moving functions. Once moved code reads them as `cs.<name>`, ruff reports them as `F401`.
- **Names.**
  - In `chat_service.py`, `cst` and `cs` occur 0 times; `state` occurs 35 times.
  - In the seven destination modules, `cs` occurs 0 times.
  - None of the 29 lifts or 13 moves clashes with a destination's top-level names (D2).
- **Ratchets.**
  - BLE001 is clean in `chat_service` and in every destination module.
  - `env_read_baseline.json:118` pins `chat_service` at 21 reads, 10 of them at import time. Three of those reads are in the streaming body (`CHAT_STREAM_CHANNEL_MAXSIZE` `:4382`, `MODERATION_STREAM_BUFFER_CHARS` `:5423` inside `tracked_streaming_generator`, `STREAMS_UNIFIED` `:5694`), and one is in `build_context_and_messages` (`CHAT_TRIM_USER_ONLY_OVERLAP` `:3707`).
  - `charactersragdb_delegated_callers_baseline.json:30` counts 23 calls in `chat_service`, 15 of them in `build_context_and_messages`.
- **Dead code (§10).**
  - `_apply_redaction_to_content` (`:2116`, 36 lines) and `_resolve_run_first_completion_outcome` (`:1305`, 9 lines) have zero references.
  - `write_mandatory_moderation_audit` (`:2396`) is a divergent duplicate of `moderation_pipeline.py:139`; the `chat_service` copy is the live one.
- **Callback introspection (M7).**
  - `streaming_utils.py:964` (`_accepts_keyword`) calls `inspect.signature(callback)` without `eval_str`. That is partial-safe: the signature drops the bound argument.
  - Nothing else reads `__name__`, `__code__` or `__qualname__` from a callback or processor. The places checked are `request_queue`, the bounded daemon, `with_llm_call_context`, `SSEStream` and `chat.py`.
  - Every lifted def's defaults are constants.
- **No Chat test runs on a dev PR (§9).**
  - `ci.yml`'s Chat shards run only at `nightly`/`release` tier (`ci.yml:594`).
  - A PR touching `chat_service` gets only backend-required's Enforce step.
  - There are no golden or snapshot tests of non-stream JSON or of SSE frames (§7c).
- **The gate list (D6), serial at dev:** 1,673 passed, 2 failed, 1 skipped, in 965 s.
  - **Failures.** Both are `[postgres]` cases. Each raises `NotFoundError: Conversation not found` in `message_store._lock_history_owner`:
    - `test_chat_retry_system_messages.py::test_retry_context_does_not_expose_another_owners_instructions[postgres]`;
    - `test_chat_queued_resume_history.py::test_foreign_conversation_is_not_used_as_history_evidence[postgres]`.
  - **Skip.** A static one, at `test_chat_endpoint_simplified.py:763`.
  - **Slowest files:** `test_roleplay_resume_contract` takes 325 s and touches the targets once, through a patch of `_format_selected_history_messages`. `test_chat_endpoint_simplified` takes 283 s.
  - **Postgres.** Five files parametrise `sqlite`+`postgres`. No file needs a plugin that CI's `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 -p pytest_asyncio.plugin -p pytest_timeout` form lacks.
- **Trigger rate.** Of 216 first-parent merges to dev in the 30 days to 2026-11-25, 15 (7%) touched `chat_service.py`, the six destination modules, `history_context.py` or `streaming_utils.py`.
- **Open PRs editing `chat_service.py`:**

  | PR | change | last updated |
  |---|---|---|
  | #3611 | +5/−0 | 2026-11-25 |
  | #3538 | +153/−34 | 2026-11-25 |
  | #3535 | +57/−10 | 2026-11-25 |
  | #3437 | +153/−21 | 2026-11-19 |
  | #854 | +57/−12 | 2026-11-19 |

- **Backlog.**
  - The chat-pipeline task the AC names as TASK-12013 was renumbered to **TASK-15032** on 2026-10-31. It is still `Done`, with AC#8 ticked and no discrepancy note.
  - `backlog_py task edit` has `--remove-ac`, `--ac` and `--append-notes`.

## Rulings

R1-R8 are the controller's and binding. R9-R21 are the planner's. Each gives its why and, where it matters, its cost if wrong.

1. **R1, no behaviour change.**
   - Defects found on the way are filed, never fixed here. Examples: the duplicate `write_mandatory_moderation_audit`, the duplicated structured-prep/guardrail/fallback blocks between the executors (§11), and the two `[postgres]` failures (R18).
   - Deleting the two zero-reference helpers (§10) and the forwarding-only `ChatCompletionPipeline` is not a behaviour change.
2. **R2, the golden test lands first** (Task 1, Appendix C).
   - 9 non-stream and 10 streaming cases record:
     - the returned payload, or the SSE frame list;
     - persistence calls;
     - usage rows and metric calls;
     - a call-order spy over the stubbed collaborators;
     - background-task calls, kept as a separate unordered multiset.
   - The golden JSON is committed with `git add -f`, because `.gitignore` ignores `*.json`.
   - Seeded-change tests prove the comparison is sensitive.
   - The golden never changes after PR A.
   - **Why:** the design's compatibility guarantees (SSE event shapes, `tldw_*` metadata) are pinned today only by ad-hoc assertions.
3. **R3, the gate.**
   - A new path-gated required job, `chat-executors`, lands in PR A, following the `rag-pipeline` precedent (R18). It runs the golden and Appendix D's list.
   - The structure pins run in the Enforce step (R20).
4. **R4, patch compatibility: the `cs.` seam.**
   - Moved code reads every `chat_service` module global as `cs.<name>` through the module object, so the 1,155 patch sites stay live with no test churn. This is .24's `up.` rule (V2).
   - Names bound by a stdlib import are imported directly, so the env-read ratchet still sees `os.getenv`. The exception is `time`, which reads as `cs.time` (manifest `facade_stdlib`).
   - `chat_service`'s aliases of destination modules (`_completion_pipeline`, `_streaming_pipeline`, …) are not read through `cs` (R5).
   - A structure pin fails if moved or lifted code reads a patched name bare (R20).
   - **F401 consequence.** `chat_service.py` gets a per-file `F401` ignore in `pyproject.toml`, with the comment `# the chat patch facade: moved code reads these imports as cs.<name> (TASK-13950.25 R4)`. It lands in PR E, the first PR that orphans an import. Ruff runs C2 without `--isolated`, so the ignore applies. Per-import `noqa` would cost about 58 comments.
5. **R5, the late-import block** (fixes C1).
   - **Layout.** A destination module that holds moved code ends with a *late-import block*:
     - first, module imports of every other `core.Chat` destination module that the moved code calls, as `from tldw_Server_API.app.core.Chat import <dest> as _<dest>`;
     - then, last, `from tldw_Server_API.app.core.Chat import chat_service as cs`;
     - the block opens with the comment `# late imports: moved code reads these at call time (TASK-13950.25 R5)`.
   - **Placement.**
     - Appended defs and any added top-level statements go above the block. Later PRs insert above it too, never at EOF.
     - A call into the same module is a bare name.
     - Stdlib imports, `functools` and `free_imports` go after the module's last top import.
   - **PR E** moves `moderation_pipeline.py:24-30`'s by-name `response_processor` import into that module's late block, as a declared manifest item (`late_imports`).
   - **Why it works:** `chat_service` imports these modules by name, so any module it imports must define its names before it starts `chat_service`'s import. With every cross-module import at the end, each module's defs exist before any cycle closes. The plan review simulated all 9 first-import orders (the 8 modules plus `api.v1.endpoints.chat`): every order imports, and every `<mod>.cs is chat_service`.
   - **Consequence:** no module-level code may read `cs.` or `_<dest>.`. So a lifted def's decorators and defaults must not read them (V-rule). Annotations are exempt, because every destination has `from __future__ import annotations` (the tool checks this); `_StreamUsage` on `save_callback`/`_stream_usage_recorder` becomes `cs._StreamUsage`.
   - **Pin:** `test_each_chat_module_imports_first_in_a_fresh_interpreter` (R20).
   - **Docs:** PR E amends design `:59` to say this. The rule's purpose was cycle-safety, and this keeps it.
   - **Cost if wrong:** the modules stay coupled to the facade. A follow-up can retarget the patches per name and drop the `cs.` reads.
6. **R6, the call state.** `core/Chat/call_state.py` (new, stdlib only) defines `NonStreamCall` and `StreamCall`.
   - Each is a plain class with `__slots__`: one slot per parameter and rewritten local, plus `__weakref__`. No defaults.
   - `from_call_locals(values)` sets the parameter slots from `locals()`, taken as the executor's first statement.
   - Reading an unset slot raises `UnboundLocalError` from `__getattr__`. That is what the local read raises today, and it is not in `_CHAT_NONCRITICAL_EXCEPTIONS`. A plain `AttributeError` *is* in it, and would be swallowed.
   - The variable is `cst`.
   - **No frozen config object:** no AC asks for one, and a read-only/mutated split adds a second object with no consumer.
   - **No `publish`:** no callable that captures `cst` is stored on `cst`, except where today's code already stores one.
     - Those are `cst.llm_call_func = llm_call_func_fb` (S8), `cst.raw_stream_iter` and `cst.streaming_generator`, which `_produce` reads. They make a `cst` ↔ frame cycle that cyclic GC collects once the response is consumed.
     - Non-stream stores none, so neither shim needs a `release()`.
     - Pinned by `test_stream_call_state_is_collected_after_the_response_is_consumed` (Task 9), through a `weakref` to the one call's `cst` after the golden's record is cleared.
7. **R7, AC2 and AC3.**
   - **Shims.** Both executors end as kwargs shims: signature + docstring + `return await _<mod>.run_<x>(_call_state.<Cls>.from_call_locals(locals()))`.
   - **Orchestrators and stages.** The orchestrators are each at most 280 lines, and so is every new function.
   - **`ChatCompletionPipeline` is deleted** (AC2, "or is deleted"):
     - `execute_non_stream_call(**kwargs)` returns `await _execute_non_stream_call_impl(**kwargs)`;
     - `:5655` calls `create_chat_streaming_response(...)` directly, which is behaviour-identical (M6);
     - `get_chat_completion_pipeline`, `__all__`'s entry and the two wrapper tests go;
     - `completion_pipeline.py` then holds the real non-stream stages.
   - **Ratchet.** Every PR `--tighten`s the executor ceilings.
8. **R8, slicing.** One PR per row of the slices table, merged in order. Non-stream goes first (the file's tail), then streaming, then `build_context_and_messages`. Each PR passes:
   - the golden, unchanged;
   - Appendix D's gate list;
   - its own extra files;
   - every ratchet;
   - `rag_split verify`.
9. **R9, rewrite purely, then lift and move verbatim** (the .24 R10).
   - **The rewrites** (Tasks 3 and 7) rewrite, into `cst.<name>`, every name that crosses a planned function boundary:
     - every parameter;
     - every local that crosses a span boundary;
     - every enclosing local that a to-be-lifted def reads or rebinds.

     `nonlocal` statements for those names are deleted. `rewrite_in` also rewrites `tracked_streaming_generator`'s 8 data names. Its local import of `StopStreamWithError` becomes a `free_imports` entry in PR H.
   - **Def-bound names** (closures to be lifted) are never rewritten (I1).
   - The rewrite PRs carry no lifts. The rewrite alone does not grow either executor (D3).
   - After that, a lift or move only dedents, prefixes globals with `cs.`, threads `cst` and rewrites lifted-name references.
10. **R10, lifting** (tool feature, Task 4; depth 2 in Task 6).
    - A nested def at any depth becomes a module-level function in its destination module, with `cst` prepended to its parameters.
    - Its def statement is replaced, at the same position, by `<name> = functools.partial(<ref>, cst)`. Here `<ref>` is the bare name in the same module, or `_<dest>.<name>`. Every other reference in the remaining function keeps working; partials accept attributes, so `_out_transform.flush = _flush_holdback` works too.
    - **In every lifted body, move body and entry body,** a reference to a lifted name that the code does not bind itself becomes `<ref>(cst, …)` when it is called and `functools.partial(<ref>, cst)` otherwise (I1).
    - The def's remaining free variables must be `cst` or lifted names (V-rule), and `Cells.crossing` ignores lifted names.
    - Lift order is callee-first. A lift may sit inside a move span, in which case it is applied first.
    - **Equivalence:** a partial of an async function or async generator function satisfies `inspect.iscoroutinefunction`/`isasyncgenfunction` on 3.12, and no consumer reads `__name__` (Measured).
11. **R11, moving spans** (the .24 `moves`, plus `args`).
    - A run of whole statements of one block becomes `[async] def <name>(cst[, *args]) -> None` and is replaced by `[await] <ref>(cst[, *args])`.
    - `args` passes read-only names local to the enclosing block. These are the handler variables `e` (non-stream N10, streaming S8) and `he` (streaming S7), which cannot become `cst` attributes.
    - V1 still forbids `return`, `yield` and `nonlocal` in a span, and `break`/`continue` leaving it. A bare `raise` inside a moved handler span re-raises the handled exception, across suspensions too.
    - **Correction (final review, 2026-11-25):** the last claim was wrong. When a task resumes a coroutine chain by throwing a failed future's exception into it, the outer coroutines' exception state is not pushed, so a moved span loses the handled exception after such an await and its own raises are re-chained at the new boundary. Tool rule V18 now refuses handler spans that contain an await.
12. **R12, the entry split** (Tasks 5 and 9). The remaining body moves whole, `return`s included, into the orchestrator in the destination module. The executor becomes the R7 shim. This is .24's `entry_split`, generalised to write the entry into another module.
13. **R13, destinations (Appendix B).**
    - **Non-stream:**
      - the provider call, failover and usage go to `completion_pipeline.py`;
      - choice normalisation, output safety and encoding/metadata go to `response_processor.py`;
      - first save and deferred-flush persistence go to `persistence_service.py`;
      - tool autoexec plus auto-continue goes to `tool_execution_service.py`.
    - **Streaming:**
      - output policies and chunk moderation go to `moderation_pipeline.py`;
      - the queue bridge, provider call and failover, save callbacks, moderation holdback, the audit wait, the tracked generator and the SSE response go to `streaming_pipeline.py` (design `:52`).
    - **`build_context_and_messages`** goes to `history_context.py`.
    - **Exceptions:** the design leaves the non-stream provider call (`:65`) and history building to `chat_service`. They leave the facade only for AC3, and the PR bodies say so.
14. **R14, the open PRs.** Every PR from C on edits `chat_service.py`, so the five open PRs listed under Measured will conflict with whichever side lands second.
    - Don't wait for them.
    - Before opening each PR, list the open PRs that touch `chat_service.py`. If one has merged, rebase ours by re-running `apply` on the new dev, never by a hand merge. Moves are anchored by line, so re-anchor every move span and `base`; lifts are by name.
    - Appendix B maps old line ranges to their new homes for those PRs' owners.
15. **R15, `build_context_and_messages` relocates whole** (Task 10, tool feature `relocate`).
    - The def moves verbatim, with `cs.` globals, above `history_context.py`'s late-import block.
    - `chat_service` re-exports the name with `from tldw_Server_API.app.core.Chat.history_context import build_context_and_messages`, so the attribute, the 4 patch sites and `chat.py:994`'s compat wrapper keep working.
16. **R16, the verify check is extended** (.24's R16).
    - Every chat manifest lives under `Helper_Scripts/rag_split/manifests/`, prefixed `c25`, which is where the verify-chain test discovers manifests.
    - Each manifest's verify output goes in its PR body.
    - For an existing module, `verify` compares the whole module AST with the base module plus the added imports, appended defs and late block (I3).
17. **R17, AC4 is amended to the renumbered id** (PR A). Its text becomes: "The AC#8 discrepancy is recorded on TASK-15032 'Refactor Chat completion pipeline and fix validated review findings' (renumbered from TASK-12013 on 2026-10-31)". The note goes on TASK-15032 through `task edit --append-notes`.
18. **R18, the job.**
    - Job id `chat-executors`, named "Chat executors (golden + chat service tests)". Flag `chat_executors_changed`.
    - **Globs:**
      - `chat_service.py`, `call_state.py`, `completion_pipeline.py`, `streaming_pipeline.py`, `moderation_pipeline.py`, `persistence_service.py`, `response_processor.py`, `tool_execution_service.py`, `history_context.py` and `streaming_utils.py` (all under `tldw_Server_API/app/core/Chat/`);
      - the golden test and data, every gate-list file, `Helper_Scripts/rag_split/**` and `tldw_Server_API/tests/_plugins/postgres.py`;
      - `.github/workflows/backend-required.yml`, `.github/actions/detect-required-gate-changes/action.yml` and `Helper_Scripts/ci/path_classifier.py`.
    - `CHAT_EXECUTORS_GATE_TESTS` lists the files. `CHAT_EXECUTORS_GATE_DESELECT` maps a test id to its reason (passed as `--deselect`), and `CHAT_EXECUTORS_GATE_EXCLUDED` maps a file to its reason. All three live in `path_classifier.py`, and the workflow lists the same items literally.
    - **Run:** serial, with a `postgres:18` service and `TLDW_TEST_POSTGRES_REQUIRED=1`, and chacha's junit check (every listed file has ≥1 pass and 0 skips).
    - **Excluded:** `test_roleplay_resume_contract.py`. Its 325 s run touches the targets only through one patch, which the R20 pin guards, and it runs nightly in the character shard; Task 10 runs it locally as C4.
    - **Deselected:** the static skip at `test_chat_endpoint_simplified.py:763`, and the two `[postgres]` failures *if* they also fail under Docker `postgres:18`. Task 1 checks; each deselect names the bug task Task 1 files.
    - It runs in parallel with `rag-pipeline`, and `backend-required` waits for it. `backend-required` also gets its own "Require chat-executors gate to have passed" step (the `:621-637` pattern), so a failed job fails it. `timeout-minutes: 40`, because the gate measured 16 min locally before trimming.
    - **Contract.** The rag contract bans `--deselect`, `-k`, `-m` and `--ignore` (`test_license_first_workflow_contracts.py:2421-2423`, `:2524-2532`). The chat contract is new, and its only allowance is a `--deselect` for each id in `CHAT_EXECUTORS_GATE_DESELECT`. It still bans `-k`, `-m` and `--ignore`.
    - **Fast-lane cost:** about 10-11 min of serial tests (965 s measured, minus the roleplay file's 325 s), plus about 2 min of setup.
      - It is off the critical path: backend-required's longest job is 16-22 min.
      - It costs runner minutes on the ~7% of dev merges that touch the code globs. The rate is higher with the workflow and classifier globs: 24% in .24's measurement.
      - PR A records the CI wall time.
    - **Why not the Enforce step:** that runs on every backend PR. **Why not nightly:** that leaves the moves unverified.
19. **R19, ratchets.**
    - **`function_size`:** every PR `--tighten`s, with no new entries except one. In PR J, `…chat_service:build_context_and_messages` is deleted and `…history_context:build_context_and_messages` is added at its measured lines/8, with reason `"moved from tldw_Server_API.app.core.Chat.chat_service:build_context_and_messages (TASK-13950.25 PR J)"`. This is the ratchet's documented move fix (`test_function_size_ratchet.py:33-36`).
    - **`env_read`:** a moved read keeps its code. The destination module gets an entry, or an incremented one, with reason `"moved from core/Chat/chat_service.py (TASK-13950.25 PR <x>)"`, and `chat_service.py`'s entry is `--tighten`ed. The reads and their PRs:

      | read | destination | PR |
      |---|---|---|
      | `MODERATION_STREAM_BUFFER_CHARS` | `streaming_pipeline` | H |
      | `CHAT_STREAM_CHANNEL_MAXSIZE` | `streaming_pipeline` | I |
      | `STREAMS_UNIFIED` | `streaming_pipeline` | I |
      | `CHAT_TRIM_USER_ONLY_OVERLAP` | `history_context` | J |

    - **`charactersragdb_delegated_callers`:** re-keyed per destination with the same reason shape, and the old count lowered. `build_context_and_messages`' 15 calls go in PR J; any in moved executor code go in its PR.
    - **`scope_predicate`:** `--write-baseline` re-keys any predicate a move carries; its baseline is qualname-keyed.
    - **Untouched:** `private_coercion` (its two coercers don't move), `global_statements`, `test_mode_branch`, `test_quality` and `layering` (moves stay inside `core.Chat`).
20. **R20, AC5.**
    - **Structure pins.** They live in `tldw_Server_API/tests/lint/test_chat_executor_structure_pins.py` and run in the Enforce step, added to the contract list at `test_license_first_workflow_contracts.py:1696-1701` (the .22 and .24 precedent). They have to run on every backend PR, because the patched-name list comes from every test file.
    - **`chat-executors` contract.** The golden and the gate list run in `chat-executors`. Its contract test fails if the job stops running them, or if the classifier lists and the workflow lists diverge. It extends the per-job tables in `test_license_first_workflow_contracts.py` (`:113, :171, :237, :272, :294, :752, :2389-2486`), as `rag-pipeline` did.
    - **Tool tests.** The extended tool's tests stay in the Enforce step, which is already pinned.
21. **R21, the tool lands in three parts, each before its first use** (I11):
    - **B** (before C): parametrisation and the pins;
    - **D** (before E): depth-1 lift, existing-module append with the late block, `args`, lift-inside-move, `entry`, `drop`, `late_imports`;
    - **F** (before G): `rewrite_in`, `free_imports` and depth-2 lift;
    - `relocate` lands with its first use, in J.

    Each feature is proven on its first real use.

## Global Constraints

Copied into every implementer brief.
- **PRs:** target `dev` and branch from the latest `origin/dev`. One PR per slice, merged strictly in order.
  - Carry a `## Change summary`; the owner waived ADR-004 sign-off for TASK-13950.
  - Get a clean-subagent code review before merge.
  - Never arm auto-merge.
  - Compare non-required E2E results with dev before merging.
  - Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Git:** never `git stash`, `--no-verify`, force-push or `git clean -fdx`.
- **Python:** `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. There is no `python` on PATH. The shell is zsh, so `$VAR` lists don't word-split; use `${=VAR}`, arrays or `$(…)`.
- **Disk:**
  - Refuse to start a pytest run under 100 GiB free (`df -g /System/Volumes/Data`).
  - Each run uses `-p no:cacheprovider --basetemp "$TMPDIR/t25-<task>-<n>"`, deleted right after by its literal path.
  - At most `-n 2`, one run at a time; the gate list runs serially.
  - No repo exports, no extra worktrees, no `/tmp` files.
- **Tests and network:**
  - `unset OPENAI_API_KEY ANTHROPIC_API_KEY`.
  - `HTTP_PROXY`/`HTTPS_PROXY=http://127.0.0.1:9`, `NO_PROXY=localhost,127.0.0.1`.
  - Never pass `-p pytest_asyncio.plugin` or `-p xdist.plugin`. Never delenv `PYTEST_CURRENT_TEST`.
  - A chat path with a commercial provider calls the real API unless the adapter is stubbed. Every golden case and every new test stubs `perform_chat_api_call`/`perform_chat_api_call_async`, or passes a fake `llm_call_func`.
  - PostgreSQL runs set `TLDW_TEST_POSTGRES_REQUIRED=1`. The plugin auto-starts Docker `postgres:18`; don't set `TLDW_TEST_NO_DOCKER`.
- **Known local failures (not regressions):**
  - `test_runtime_package_discovery` (setuptools);
  - 3 in `test_merge_queue_gate_workflows` and 3 in `test_required_workflow_contracts`;
  - `test_rg_route_map_lint` and `test_route_auth_ratchet` under `-n 2`;
  - plus Task 1's deselects.

  Confirm each against a dev baseline before dismissing it.
- **Bandit:** `uvx --offline bandit -q` on changed app and `Helper_Scripts` files. No new findings against the base. Moved code keeps its `# nosec` comments.
- **Backlog:** `PYTHONPATH=tools/backlog-py/src $PY -m backlog_py task …` only, with explicit ids. Stage with `git add -A backlog/tasks`.
- **Subagents:** implementers never dispatch subagents.
- **Baselines:** never widen any. The only new or changed entries allowed are R19's. `function_size` is only ever `--tighten`ed, apart from R19's PR J re-key.
- **Moves:**
  - Every rewrite, lift, move and relocate is `rag_split apply` output, committed with its `c25*.py` manifest.
  - A hand edit after `apply` must be a declared manifest item, or `verify` fails.
  - A rebase re-runs `apply` on the new dev (R14).
  - Before opening a PR, run `gh pr list --base dev --state open --json number,files` and check which open PRs touch `chat_service.py`.
- **Golden:** the golden JSON never changes after PR A, and no test is edited to pass. A golden diff is a behaviour change: find it, never regenerate. The exception is the wrapper's two tests that R7 deletes.

## Review Focus

1. **An inert patch after a move.** A test patches a `chat_service` name that moved code no longer reads there, and passes vacuously.
   - Pinned by `test_moved_chat_code_reads_patched_names_through_chat_service` (R20). It selects functions from the `c25*` manifests (lift, move, entry and relocate names), so lifted code is covered. Its seeded self-test uses a lifted def.
   - Also pinned by `verify`'s `cs.` check. `time` is the one stdlib name on the list (R4).
2. **An import cycle, or an import-time `cs.`/`_<dest>.` read.**
   - Pinned by `test_each_chat_module_imports_first_in_a_fresh_interpreter` (R20, subprocess). It covers each destination module, `chat_service` and `api.v1.endpoints.chat`, plus `call_state` from PR C.
   - Also by the R5 V-rule on decorators and defaults, and the tool's future-import check.
3. **Late binding across the queue worker or the post-return callbacks.** Closures read `model`, `selected_provider`, `llm_call_func` and `cleaned_args` at call time, and `_queued_processor` rebinds them from the worker on its fallback branch.
   - `cst.<name>` keeps one slot per name, so the semantics are the same.
   - Golden S02 (queued with worker failover) and S03 (direct failover) record which provider, model and credential source the save callback and the usage row see.
   - `test_unset_call_slot_reads_like_an_unbound_local` (Task 3) covers reads before assignment.
4. **Exceptions and `return`.**
   - Handler bodies keep their `return`s in the orchestrator (V1).
   - Moved handler spans take the handler variable as `args`, and their bare `raise` re-raises it (R11).
   - Golden N04 (no fallback → raise), S04 (HTTPException → terminal error stream) and S05 (noncritical → terminal error) pin both shapes.
5. **Moderation or persistence drift on a branch the golden can't see.** Examples: the holdback buffer limit, a mandatory audit task pending at stream end, a partial save on disconnect.
   - Golden S06 covers holdback, block and redact, and the audit wait.
   - `test_chat_service_fallback.py`, `test_chat_service_streaming_tool_autoexec.py` and `test_chat_history_and_streaming.py` are in the gate.
   - `verify` covers every branch of every move.

## CI-equivalent checks

- **C1:** `tldw_Server_API/tests/lint`, `tldw_Server_API/tests/Helper_Scripts/test_rag_split.py` and `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`. Add the whole `tldw_Server_API/tests/CI` when a workflow, the classifier, an action or a contract file changes.
- **C2 (rewrite, lift, move and relocate PRs):** paste both outputs into the PR body.
  - `$PY -m Helper_Scripts.rag_split.rag_split verify <manifest> --base origin/dev` reports 0 differences;
  - `ruff check --select F821,F841,F401,F811,F822,BLE001` on the changed Chat modules is clean. Run it with the repo config, not `--isolated`, so R4's per-file ignore applies.
- **C3:** Appendix D's gate list, serially, with `TLDW_TEST_POSTGRES_REQUIRED=1` and the deselects: 0 failed apart from the known local failures, 0 skipped. Record the wall time.
- **C4:** the task's extra files.
- **C5:** bandit.

A run:
```bash
df -g /System/Volumes/Data   # refuse under 100 GiB free
unset OPENAI_API_KEY ANTHROPIC_API_KEY
export HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 NO_PROXY=localhost,127.0.0.1
GATE=$(PYTHONPATH=. $PY -c 'from Helper_Scripts.ci.path_classifier import CHAT_EXECUTORS_GATE_TESTS as t; print(" ".join(t))')
DESEL=$(PYTHONPATH=. $PY -c 'from Helper_Scripts.ci.path_classifier import CHAT_EXECUTORS_GATE_DESELECT as d; print(" ".join("--deselect=" + k for k in d))')
BT="$TMPDIR/t25-<task>-<n>"
PYTHONPATH=. TEST_MODE=true DISABLE_HEAVY_STARTUP=1 TLDW_TEST_POSTGRES_REQUIRED=1 \
  $PY -m pytest -q -p no:cacheprovider --basetemp "$BT" -rs ${=DESEL} ${=GATE}
rm -rf "$BT"
```

## PR slices and projections

"~" values are estimates (D1). Each PR measures and records the real ones in its body.

| PR | Task | content | non-stream / streaming lines after | `chat_service.py` after | changed lines (est.) |
|---|---|---|---|---|---|
| A | 1 | plan; TASK-15032 note and AC4 amendment; bug tasks for any deselects; golden; `chat-executors` job | 1,054 / 1,542 | 6,835 | ~1,700 + golden data |
| B | 2 | tool part 1: parametrisation (`ctx`, `facade`, `facade_stdlib`, `task`, names); structure pins | same | 6,835 | ~700 |
| C | 3 | `call_state.NonStreamCall`; pure non-stream rewrite to `cst.` | ~1,052 / 1,542 | ~6,835 | ~800 |
| D | 4 | tool part 2: depth-1 lift with lifted-name rewrites, existing-module append and late block, `args`, lift-inside-move, `entry`, `drop`, `late_imports` | same | same | ~1,200 |
| E | 5 | non-stream lifts and moves; entry split; delete `ChatCompletionPipeline`; F401 ignore; design `:45`/`:59`/`:67`/`:235` | 38 / 1,542 | ~5,825 | ~2,300 |
| F | 6 | tool part 3: `rewrite_in`, `free_imports`, depth-2 lift | same | same | ~500 |
| G | 7 | `call_state.StreamCall`; pure streaming rewrite to `cst.` (with `rewrite_in`) | 38 / ~1,535 | ~5,820 | ~1,500 |
| H | 8 | streaming callbacks: S10-S14 lifts | 38 / ~700 | ~4,990 | ~2,000 |
| I | 9 | streaming S2/S4/S15 lifts; S3/S4/S7/S8 moves; entry split | 38 / 50 | ~4,340 | ~1,300 |
| J | 10 | relocate `build_context_and_messages` (with the tool's `relocate`); delete 2 dead helpers | 38 / 50 | ~3,457, **AC3 met** | ~1,900 |
| K | 11 | close-out | | | backlog |

---

### Task 1: Golden test, `chat-executors` job, backlog (PR A; R2, R3, R17, R18, AC4, AC5)

**Files:**
- Create: `tldw_Server_API/tests/Chat/unit/test_chat_executors_golden.py`
- Create: `tldw_Server_API/tests/Chat/unit/golden/chat_executors_golden.json` (`git add -f`)
- Modify: `Helper_Scripts/ci/path_classifier.py` (globs, flag, `CHAT_EXECUTORS_GATE_TESTS`, `_DESELECT`, `_EXCLUDED`) and its tests
- Modify: `.github/actions/detect-required-gate-changes/action.yml` (flag output)
- Modify: `.github/workflows/backend-required.yml`. The changes are:
  - job `chat-executors`, with a `postgres:18` service and `timeout-minutes: 40`;
  - the `changes` job's `chat_executors_changed` output (`:50` pattern);
  - `backend-required`'s `needs`;
  - its "Require chat-executors gate to have passed" step (`:621-637` pattern).
- Modify: `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py` (R20's per-job tables, mirroring `rag-pipeline`)
- Backlog: TASK-13950.25 (AC4, R17), TASK-15032 (notes), and a bug task per deselected failure
- Add: this plan

**Interfaces:**
- Produces: `run_case(case_id, monkeypatch) -> dict`, the normalised record; `GOLDEN_CASES: dict[str, dict]`; `CHAT_EXECUTORS_GATE_TESTS: tuple[str, ...]`; `CHAT_EXECUTORS_GATE_DESELECT: dict[str, str]`; `CHAT_EXECUTORS_GATE_EXCLUDED: dict[str, str]`.

- [ ] **Step 1: Baseline the gate.** Run Appendix D's list serially at dev (C3 form, Docker `postgres:18`).
  - If the two `[postgres]` failures reproduce, file one bug task for them and deselect both, each reason naming that task.
  - Deselect the static skip with the reason `static skip: "Streaming tests hang with TestClient"`.
  - Record the counts and the wall time.
- [ ] **Step 2: Write the golden harness.** `run_case(case_id, monkeypatch)`:
  - **Calls:** `chat_service.execute_non_stream_call(**kwargs)` for N cases. For S cases, it calls `chat_service.execute_streaming_call(**kwargs)` and drains the returned response body.
  - **Stubs**, all on `chat_service` names so they stay live after moves (R4):
    - a fake `llm_call_func`, `perform_chat_api_call(_async)` and the request queue;
    - `log_llm_usage` and `save_message_fn`;
    - the moderation and topic-monitoring getters;
    - `submit_accounting`, through a recorder that awaits `make_coro()` inline.
  - **Fakes.** Reuse the module-level fakes in `tests/Chat/unit/test_chat_service_fallback.py:61-313, 2337`. Streaming queue fakes exist only inside test bodies, so add one module-level streaming queue fake to the golden module.
  - **Record:**
    - the returned JSON or the SSE frame list;
    - persisted messages, usage rows and metric calls;
    - the collaborator call order.
  - **Background tasks:** await every task scheduled during the case (topic monitoring, mandatory audits) before the snapshot. Record those calls as an unordered multiset, kept apart from the call order.
  - **Normalise:**
    - ids, `stream_id`, `loop_compat_run_id`, `created` and timestamps get stable placeholders in payloads and SSE frames;
    - latencies are dropped;
    - dict insertion order is kept.
  - **Stdlib patches:** only `chat_service.time` may be patched, with a fixed clock.
- [ ] **Step 3: Write the tests.**
  - `test_golden_case_matches[<case>]`, one per Appendix C case;
  - `test_seeded_change_in_<payload|frames|persistence|usage>_fails_the_comparison`, four tests;
  - `test_every_case_stubs_the_provider`: a socket guard sees no connect attempt.
- [ ] **Step 4: Generate the golden once.** Use an env-flagged write helper in the test module, and commit with `git add -f`. Then run the golden 3 times to prove it is stable.
- [ ] **Step 5: Wire the job.** Follow R18, with `rag-pipeline`'s job, action output and contract rows as the template. The contract asserts:
  - the job exists, has the Postgres service, and `backend-required` needs it;
  - the workflow's literal file, deselect and exclude lists equal the classifier's;
  - the junit check is present;
  - the flag's globs equal R18's.
- [ ] **Step 6: Edit the backlog.**
  - TASK-15032: `task edit TASK-15032 --append-notes "AC#8 discrepancy (TASK-13950.25, 2026-11-25): AC#8 claims chat_service.py is a compatibility facade while focused modules own response processing, moderation, persistence and streaming orchestration. At dev 8f33908da4 chat_service.py is 6,835 lines; execute_streaming_call is 1,542 lines and _execute_non_stream_call_impl 1,054, with streaming moderation (~380 lines), streaming orchestration and non-stream tool continuation still inline, and ChatCompletionPipeline only forwarding. TASK-13950.25 completes the extraction."`
  - TASK-13950.25: remove AC4 and the later ACs, then re-add them in order, keeping their numbers and checked state (the TASK-13950.22 precedent). The new AC4 text is R17's.
- [ ] **Step 7: Run C1 (with the whole `tests/CI`), the golden and C3.** Commit, push, open PR A, and record the `chat-executors` CI wall time in the body.

### Task 2: Tool part 1; structure pins (PR B; R4, R16, R20, R21)

**Files:**
- Modify: `Helper_Scripts/rag_split/rag_split.py`, `tldw_Server_API/tests/Helper_Scripts/test_rag_split.py`
- Create: `tldw_Server_API/tests/lint/test_chat_executor_structure_pins.py`
- Modify: `.github/workflows/backend-required.yml`, adding the pins file to the Enforce step's literal pytest list (`:799-850`). `TEST_SEAM_GATE_SUITES` checks that list token by token.
- Modify: `test_license_first_workflow_contracts.py`, adding the pins file to `TEST_SEAM_GATE_SUITES` (`:1686-1722`).

**Interfaces:**
- Produces Appendix A's part-1 keys, each defaulting to the RAG value.
- Produces `PATCHED_CHAT_SERVICE_NAMES: frozenset[str]`, from an AST scan of the tests: the §7b method plus aliases bound by `x = chat_service`.
- Produces `c25_manifest_functions() -> dict[str, set[str]]` (module → function names), read from the `c25*` manifests.

- [ ] **Step 1: Pin RAG compatibility first.** Add `test_every_rag_manifest_still_verifies_after_generalisation`: the verify chain over `t03`…`t10l` at their recorded bases is unchanged. CI has `fetch-depth: 0`.
- [ ] **Step 2: Write failing tool tests for each part-1 key, then implement them.** Part 1 is:
  - parametrising `STATE_CLASS`, the state module, the `NOQA`/header/V9 texts, `_is_pkg`'s package name and V8's reserved names (`ctx` names, `facade`);
  - `ctx`, `ctx_import`, `facade`, `facade_stdlib` and `task`.
- [ ] **Step 3: Write the pins.**
  - `test_moved_chat_code_reads_patched_names_through_chat_service`: in every function `c25_manifest_functions()` names, there is no bare load of a name in `PATCHED_CHAT_SERVICE_NAMES` that is bound by an import of the same object. A seeded self-test uses a mini lifted def. It is vacuous until PR E, and its docstring says so. From PR E on it also asserts that `c25_manifest_functions()` is non-empty; Task 5 adds that line.
  - `test_each_chat_module_imports_first_in_a_fresh_interpreter`: parametrised over the seven destination modules, `chat_service` and `api.v1.endpoints.chat`. A subprocess imports the module first, then `chat_service`, and asserts `<mod>.cs is chat_service` for every module that binds `cs`.
- [ ] **Step 4: Run C1 and `rag_split shared` on `_execute_non_stream_call_impl` with Appendix B's non-stream partition.** That run produces Task 3's rewrite set, saved to the ledger workspace. Commit and open PR B.

### Task 3: `NonStreamCall`; pure non-stream rewrite (PR C; R6, R9)

**Files:**
- Create: `tldw_Server_API/app/core/Chat/call_state.py` and `Helper_Scripts/rag_split/manifests/c25c_nonstream_state.py`
- Modify: `chat_service.py`, the `function_size` baseline (`--tighten`), the pins (add `call_state` to the subprocess list)

**Interfaces:**
- Produces: `class NonStreamCall` (slots `PARAMS + LOCALS + ("__weakref__",)`), `NonStreamCall.from_call_locals(values: Mapping[str, Any]) -> NonStreamCall`, and `PARAMS: tuple[str, ...]` equal to the executor's 30 keyword-only names in order.

- [ ] **Step 1: Write the pins.**
  - `test_nonstream_call_params_match_the_executor_signature`;
  - `test_unset_call_slot_reads_like_an_unbound_local`: an unset slot raises `UnboundLocalError`, and `_CHAT_NONCRITICAL_EXCEPTIONS` does not catch it. A name that is not a slot still raises `AttributeError`, so `hasattr(cst, "zzz")` is False and `copy.copy(cst)` works.
  - `__getattr__` therefore raises `UnboundLocalError` only when `name in type(self).__slots__`.

  Both go in `tests/lint/test_chat_executor_structure_pins.py`.
- [ ] **Step 2: Write the manifest.**
  - `prologue: "cst = _call_state.NonStreamCall.from_call_locals(locals())"`;
  - `imports: ["from tldw_Server_API.app.core.Chat import call_state as _call_state"]`;
  - `rewrite`: every parameter and every reported local → `cst.<name>`, excluding def-bound names (R9).
- [ ] **Step 3: Apply, then run C2, then `function_size --tighten`.**
- [ ] **Step 4: Run C1, the golden, C3 and C5.** Commit and open PR C.

### Task 4: Tool part 2 (PR D; R5, R10, R11, R12, R21)

**Files:** `rag_split.py` and its tests only.

Implement Appendix A's part-2 keys test-first, one positive and one refused case each, on mini sources under `tmp_path`:
- `lift` at depth 1;
- the R10 lifted-name rewrite in lift, move and entry bodies;
- lift inside a move;
- the existing-module append with the late block, and its whole-module `verify`;
- `args`, `entry`, `drop` (including `__all__` entries) and `late_imports`.

Also add the V-rules:
- decorators or defaults that read `cs.`/`_<dest>.`;
- annotations, which are allowed only under `from __future__ import annotations`;
- free variables outside `cst` and lifted names;
- destination name clashes;
- `publish` refused under `facade="cs"`.

Then run a dry `apply` of Task 5's draft manifest against a scratch copy of the `chat_service` and destination sources under `$TMPDIR`. It must `verify` clean, and the scratch copy is deleted afterwards.

### Task 5: Non-stream stages; entry split; delete `ChatCompletionPipeline` (PR E; R4, R5, R7, R12, R13, AC1, AC2, AC3 part)

**Files:**
- Modify: `chat_service.py`, `completion_pipeline.py`, `response_processor.py`, `persistence_service.py`, `tool_execution_service.py`, `moderation_pipeline.py` (`late_imports` only)
- Create: `Helper_Scripts/rag_split/manifests/c25e_nonstream_stages.py`
- Modify: `pyproject.toml` (R4's F401 per-file ignore)
- Modify: the design (`:45`, `:59`, `:67`, `:235`) and `core/Chat/README.md:21`
- Modify: `tests/Chat/unit/test_chat_service_content.py` (delete the 2 wrapper tests, `:78-124`)

**Interfaces:**
- Consumes: `NonStreamCall` (Task 3).
- Produces:
  - `completion_pipeline.run_non_stream_call(cst: "NonStreamCall") -> Any`, at most 280 lines;
  - `_execute_non_stream_call_impl`, now the R7 shim;
  - `execute_non_stream_call(**kwargs)`, which returns `await _execute_non_stream_call_impl(**kwargs)`.

- [ ] **Step 1: Write the manifest.** It covers Appendix B's non-stream rows:
  - lifts first;
  - moves with `args: ["e"]` for N10;
  - `entry` into `completion_pipeline.run_non_stream_call`;
  - `late_imports` for `moderation_pipeline.py:24-30`;
  - `drop` for:
    - `ChatCompletionPipeline` and `get_chat_completion_pipeline`;
    - the `:75` import;
    - `completion_pipeline.py:7-9`'s now-unused `dataclass`, `Awaitable` and `Callable` imports;
    - `__all__`, which becomes `["run_non_stream_call"]`.

    `:5655` is replaced by a direct `create_chat_streaming_response(request=…, stream_factory=…)` call.
- [ ] **Step 2: Apply, run C2, add the F401 ignore, and delete the 2 wrapper tests.**
- [ ] **Step 3: Edit the docs.**
  - The design `:59` rule becomes R5's text.
  - `:45`, `:67` and `:235` say the class was deleted and `completion_pipeline.run_non_stream_call` owns the non-stream stages (TASK-13950.25).
  - `README.md:21` is updated to match.
  - `Docs/Design` is not mirrored under `Docs/Published`. Check, and if it is, run `Helper_Scripts/refresh_docs_published.sh`.
- [ ] **Step 4: Run C1, the golden, C3 and C5.** Record the line counts for `_execute_non_stream_call_impl` (~38) and `run_non_stream_call` (≤280). Commit and open PR E.

### Task 6: Tool part 3 (PR F; R9, R10, R21)

**Files:** `rag_split.py` and its tests only.

Add, test-first:
- `rewrite_in`: a rewrite plus `nonlocal` deletion for names resolving to a named nested def's scope. Its V-rule: no collision with executor-scope rewritten names.
- `free_imports`: imports added at a destination's top for names that a function-local import bound in the source.
- depth-2 lift, of a def nested in a def that is itself lifted in the same or a later manifest.

Then dry-run Task 7's and Task 8's draft manifests on a scratch copy, as in Task 4.

### Task 7: `StreamCall`; pure streaming rewrite (PR G; R6, R9)

Task 3's pattern, applied to `execute_streaming_call`:
- `rewrite_in: {"tracked_streaming_generator": {<the 8 data names of Measured/Shape>: "cst.<name>"}}`, with their `nonlocal`s deleted. `StopStreamWithError` stays a local import; it becomes a `free_imports` entry in Task 8.
- The manifest is `c25g_stream_state.py`. The pin is `test_stream_call_params_match_the_executor_signature`.
- `nonlocal` statements for rewritten names are deleted. A `nonlocal` that names a local of a def that isn't lifted stays.

### Task 8: Streaming callbacks (PR H; R10, R13, AC1)

**Lifts:** Appendix B's S10-S14 rows (manifest `c25h_stream_callbacks.py`), callee-first, depth 2 included.
- To `moderation_pipeline`: `_apply_stream_output_policies` and `_out_transform`.
- To `streaming_pipeline`: everything else.
- `_finalize_stream` stays nested in `tracked_streaming_generator`.

**Free imports:** `free_imports: {"StopStreamWithError": "from tldw_Server_API.app.core.Chat.streaming_utils import StopStreamWithError"}` for both modules that need it. `streaming_utils` imports no destination module and not `chat_service`.

**Env reads:** `MODERATION_STREAM_BUFFER_CHARS` moves with `tracked_streaming_generator`; R19 re-keys it to `streaming_pipeline`.

**Checks:** C1-C5 and the golden.

### Task 9: Streaming provider, queue and SSE; entry split (PR I; R11, R12, AC1, AC3 part)

**Lifts and moves:** Appendix B's S2, S3, S4, S7, S8 and S15 rows (manifest `c25i_stream_entry.py`), with lifts before the moves that read them. Then `entry` into `streaming_pipeline.run_streaming_call`. Handler spans take `args: ["he"]` (S7) and `args: ["e"]` (S8).

**Env reads:** `CHAT_STREAM_CHANNEL_MAXSIZE` and `STREAMS_UNIFIED` move, and R19 re-keys them.

**New pin** (golden module): `test_stream_call_state_is_collected_after_the_response_is_consumed`.
- Monkeypatch `call_state.StreamCall.from_call_locals` with a wrapper that calls the original and keeps a `weakref.ref` to the object it returns.
- Run golden S01 and drain its response.
- Clear the record, which drops any spy that holds a callback.
- Call `gc.collect()`, then assert that a `weakref` to that call's `cst` is dead.

**Checks:** C1-C5 and the golden. Record the line counts for `execute_streaming_call` (~50) and `run_streaming_call` (≤280).

### Task 10: Relocate `build_context_and_messages`; dead helpers (PR J; R15, R19, AC3)

- **Tool.** Add `relocate` to the tool, test-first: the def goes above the destination's late block, and the source gets a re-export import.
- **Manifest** `c25j_history.py`:
  - `relocate: [{"name": "build_context_and_messages", "module": "history_context"}]`;
  - `drop: ["_apply_redaction_to_content", "_resolve_run_first_completion_outcome"]`. Both have zero references; the `git grep -n` evidence goes in the PR body.
- **Ratchets (R19):**
  - the `function_size` re-key;
  - `env_read` `CHAT_TRIM_USER_ONLY_OVERLAP` to `history_context`;
  - 15 `charactersragdb_delegated_callers` calls to `history_context`.
- **Checks:**
  - `wc -l chat_service.py` is under 4,000;
  - C4 runs `Character_Chat_NEW/unit/test_roleplay_resume_contract.py`, the file R18 excludes from the gate;
  - C1-C5 and the golden.

### Task 11: Close-out (PR K; AC1-AC5, DoD)

- **Step 1: Measure each AC at this branch's tree, and write each measurement into the notes.**
  - **AC1:** the Appendix B table, with each destination's function names.
  - **AC2:** `git grep -n ChatCompletionPipeline` hits only the design history, the plans and the backlog.
  - **AC3:** the two shims' line counts and `wc -l` of `chat_service.py`.
  - **AC4:** TASK-15032's note.
  - **AC5:** the job, its contract, and the Enforce-step pins and their contract entry.
- **Step 2: Check the ACs and fill in the Final Summary and DoD.**
- **Step 3: File the follow-ups, one task each, each depending on TASK-13950.25.**
  - deduplicate the structured-prep/guardrail/fallback blocks between the two orchestrators;
  - delete the divergent `chat_service.write_mandatory_moderation_audit`;
  - retarget the patch sites so the `cs.` seam and the F401 ignore can go (R5's cost);
  - the R6 streaming cycle, if a `release()` is wanted;
  - any defects found on the way (R1).
- **Step 4: Update `core/Chat/README.md`'s module roles** to the final layout.
- **Step 5: Commit, as one commit.**

## Self-review

- **Spec coverage.**
  - **AC1:** Tasks 5, 8 and 9.
  - **AC2:** Task 5.
  - **AC3:** Tasks 5, 9 and 10. The arithmetic is in D1, and Task 10 checks it.
  - **AC4:** Task 1, with R17.
  - **AC5:** R20 and Tasks 1-2.
- **Size.** The executors never grow (D3). Every new function is ≤ 280 (D1).
- **Type consistency.** `cst`, `cs`, `_call_state`, `_<dest>`, `NonStreamCall`, `StreamCall`, `run_non_stream_call` and `run_streaming_call` are used the same way in every task and appendix.
- **Review dispositions.** See the end of this plan.

## Appendix A: `rag_split` extensions

All keys are optional. Each default is the .24 behaviour, so the 14 RAG manifests verify byte for byte.

| key | part | default | meaning |
|---|---|---|---|
| `ctx` | 1 | `[["rcfg","RagRunConfig"],["state","PipelineState"]]` | Leading parameters of every moved or lifted function, as `[name, annotation]`. Chat: `[["cst","NonStreamCall"]]` or `[["cst","StreamCall"]]`. |
| `ctx_import` | 1 | the RAG `TYPE_CHECKING` imports | Header lines, put in an existing `TYPE_CHECKING` block or a new one. |
| `facade` | 1 | `"up"` | Alias for the source module in moved code. Chat: `"cs"`. |
| `facade_stdlib` | 1 | `[]` | Stdlib-bound names still read through the facade. Chat: `["time"]`. |
| `task` | 1 | `"TASK-13950.24"` | Used in moved-code docstrings and in the `noqa` text. |
| `lift` | 2 (depth 1), 3 (depth 2) | `{}` | `{nested def name: module}`. The def site becomes `name = functools.partial(<ref>, cst)`, and the body goes above the module's late block with `cst` prepended. Lifted-name references are rewritten in every lift, move and entry body (R10). A lift may sit inside a move span, and is applied first. |
| existing-module append | 2 | | Appended defs go above the late block. Stdlib, `functools`, `ctx_import` and `free_imports` lines go after the module's last top import. `_<dest>` module imports and the `cs` import go in the late block (R5). `verify` compares the whole module AST with the base module plus those additions, and skips the stage-shape rules. V9 parity covers only the stdlib names the moved code reads. A call into the same module is a bare name. |
| `moves[].args` | 2 | `[]` | Extra positional parameters after `cst`. Each must be read-only in the span and bound in the enclosing block. |
| `entry` | 2 | unset | `{"name", "module", "doc"}`. The body after the docstring and the prologue statement moves whole to `module.name(cst)`; it is `async` if the source function is. The source keeps its signature and docstring. The prologue statement (`cst = <expr>`) is consumed: the body becomes the single statement `return await _<module>.name(<expr>)`. |
| `drop` | 2 | `[]` | Top-level defs, classes and `__all__` entries to delete, with declared import or call-site replacements. Each must have zero references outside itself, apart from the declared replacements. |
| `late_imports` | 2 | `[]` | `[{"module", "lines": [a, b]}]`. An existing top import moves into that module's late block. |
| `rewrite_in` | 3 | `{}` | `{nested def: {name: expr}}`. A rewrite plus `nonlocal` deletion for names resolving to that def's scope. |
| `free_imports` | 3 | `{}` | `{name: import line}`, added at a destination's top for a lifted body's names that a function-local import bound in the source. |
| `relocate` | in J | `[]` | `[{"name", "module"}]`. A top-level def moves verbatim, with `cs.` globals, above the module's late block. The source gets `from <pkg>.<module> import <name>` after its last import. |
| `publish` | | | Refused when `facade` is `"cs"` (R6). |

## Appendix B: the partition (dev 8f33908da4; the survey's §1 phase ids; spans corrected by the review's M1)

| id | lines | kind | destination | PR |
|---|---|---|---|---|
| N3 `_record_provider_use` | 5833-5849 | lift | completion_pipeline | E |
| N4 `_record_dict_response_usage` | 5851-5976 | lift | completion_pipeline | E |
| N6 `_record_provider_use_after_cancel` | 6019-6023 | lift | completion_pipeline | E |
| N7 `_queued_processor`, `_invoke_non_stream_provider` | 6027-6064, 6109-6115 | lift (inside the N7 move) | completion_pipeline | E |
| N10 `_record_fallback_use_after_cancel` | 6244-6246 | lift (inside the N10 move) | completion_pipeline | E |
| N12 `_refresh_primary_choice_state`, `_apply_output_safety_to_current_choices` | 6367-6405 | lift | response_processor | E |
| N5 structured prep, tool guard, guardrail | 5979-6007 | move `prepare_non_stream_request` | completion_pipeline | E |
| N7 provider call (the `if`/`else`) | 6025-6133 | move `call_non_stream_provider` | completion_pipeline | E |
| N10 handler body | 6157-6325 | move `fail_over_non_stream`, `args: ["e"]` | completion_pipeline | E |
| N11 normalise and collect choices | 6327-6365 | move `collect_non_stream_reply` | response_processor | E |
| N13 structured validate, first save or defer | 6409-6477 | move `save_first_non_stream_reply` | persistence_service | E |
| N14 tool autoexec and auto-continue | 6479-6709 | move `run_non_stream_tool_autoexec` | tool_execution_service | E |
| N15 tail validate, flush deferred persistence | 6711-6746 | move `flush_deferred_non_stream_persistence` | persistence_service | E |
| N16-N17 run-first, assistant name, encode, `tldw_*` metadata | 6748-6803 | move `encode_non_stream_payload` | response_processor | E |
| N rest | | `entry` → `run_non_stream_call` | completion_pipeline | E |
| S10 once-recorders | 4852-4884 | lift | streaming_pipeline | H |
| S11 `_apply_stream_output_policies` | 4894-5085 | lift | moderation_pipeline | H |
| S12 `_stream_generation_metadata`, `_prepare_saved_stream_reply`, `partial_save_callback`, `_stream_usage_recorder` | 5087-5170 | lift | streaming_pipeline | H |
| S13 `save_callback` | 5172-5388 | lift | streaming_pipeline | H |
| S14 `_emit_with_holdback`, `_flush_holdback`, `_await_mandatory_moderation_audits` | 5429-5474 | lift (depth 2) | streaming_pipeline | H |
| S14 `_out_transform` | 5476-5617 | lift (depth 2) | moderation_pipeline | H |
| S14 `tracked_streaming_generator` (keeps `_finalize_stream`) | 5390-5688 | lift | streaming_pipeline | H |
| S2 `_remaining_provider_factory_timeout`, `_public_stream_error_payload`, `_terminal_stream_error` | 4285-4336 | lift | streaming_pipeline | I |
| S4 `_refresh_params_sync`, `_queued_processor`, `_channel_stream` | 4395-4629 (by name; the range only brackets the three defs) | lift (inside the S4 move) | streaming_pipeline | I |
| S15 `_produce`, `_gen` | 5709-5748 | lift | streaming_pipeline | I |
| S3 structured prep, guard, guardrail (the inner `try`) | 4339-4367 | move `prepare_stream_request` | streaming_pipeline | I |
| S4 statements (channel size, enqueue and admission, `raw_stream_iter`) | 4379-4631 | move `start_queued_stream` | streaming_pipeline | I |
| S7 pre-return statements | 4650-4680 | move `record_stream_http_error`, `args: ["he"]` | streaming_pipeline | I |
| S8 refresh | 4719-4755 | move `refresh_fallback_params`, `args: ["e"]` | streaming_pipeline | I |
| S8 fallback call (holds `llm_call_func_fb`'s def) | 4778-4817 | move `call_stream_fallback`, `args: ["e"]` | streaming_pipeline | I |
| S rest | | `entry` → `run_streaming_call` | streaming_pipeline | I |
| `build_context_and_messages` | 3056-3871 | relocate | history_context | J |

The S8 guard (`:4757-4777`, with a `return`) and every `return` stay in `run_streaming_call`. Line ranges are at the base, and each PR re-anchors them (R14).

## Appendix C: golden cases

Every case stubs the provider. `P`/`F` are the primary and fallback fakes.

| id | path | what it pins |
|---|---|---|
| N01 | non-stream, direct, persist | payload, `tldw_*` metadata order, assistant save, usage row |
| N02 | non-stream, queued (fake queue) | queued worker path, provider-use recording |
| N03 | non-stream, `P` raises `ChatProviderError`, `F` succeeds | failover rebinds (`model`, `selected_provider`) seen by usage and the save |
| N04 | non-stream, `P` raises, fallback off | exception type and status |
| N05 | non-stream, string response, force-normalise on | string wrap |
| N06 | non-stream, JSON-schema structured request, deferred persistence | `tldw_structured`, deferred flush order |
| N07 | non-stream, output moderation redacts, topic monitoring on | redacted payload; audit and topic calls (multiset) |
| N08 | non-stream, legacy tool autoexec plus auto-continue once | tool messages, second provider call, final save |
| N09 | non-stream, guardrail decision present, `INJECT_ASSISTANT_NAME` on | `tldw_prompt_guardrails`, assistant name |
| S01 | stream, direct | SSE frames, final save, usage row, metadata frame |
| S02 | stream, queued; `P`'s factory raises a replayable `ChatProviderError` in the worker (`_certified_pre_dispatch`); `F` streams | worker rebinds: `F`'s provider, model and credential source seen by the save callback and the usage row |
| S03 | stream, direct; `P`'s factory raises; `F` streams | direct failover frames, `llm_call_func` rebind |
| S04 | stream, factory raises `HTTPException` | terminal error stream (S7), refund call |
| S05 | stream, noncritical error, fallback off | terminal error stream (S8 else) |
| S06 | stream, chunk moderation with holdback: one redact, one block | frames; mandatory audit wait (multiset) |
| S07 | stream, legacy tool autoexec in the save callback, loop-compat on | tool frames, loop events |
| S08 | stream, `STREAMS_UNIFIED=1` | `SSEStream` frames, `_produce`/`_gen` path |
| S09 | stream, structured request | structured SSE events, structured validation on save |
| S10 | stream, queued, success | frames through `_channel_stream` |

## Appendix D: the gate list (under `tldw_Server_API/tests/`; Task 1 baselines it)

- `Chat/unit/`:
  - `test_chat_executors_golden.py`;
  - `test_chat_service_fallback.py`, `test_chat_service_fallback_credential_taxonomy.py`, `test_chat_service_content.py`, `test_chat_service_streaming_tool_autoexec.py`, `test_chat_service_tool_autoexec.py`, `test_chat_service_queue_estimate.py`, `test_chat_service_audit_tasks.py`;
  - `test_streaming_structured_events.py`, `test_prompt_cost_guardrails.py`, `test_chat_history_and_streaming.py`, `test_history_context.py`, `test_history_selected_images.py`, `test_native_history_frozen_worldbooks.py`, `test_failed_retry_provider_order.py`.
- `Chat/integration/`: `test_chat_endpoint_simplified.py`, `test_chat_endpoint_auto_routing.py`, `test_chat_endpoint_streaming_normalization.py`, `test_chat_run_first_rollout.py`.
- `Chat_NEW/unit/`: `test_chat_continuation_controls.py`, `test_chat_history_dedup.py`, `test_new_chat_title.py`, `test_tool_message_persistence.py`, `test_chat_queue_integration.py`.
- `Chat_NEW/integration/`: `test_chat_rg_refund.py`, `test_native_character_prompt_delivery.py`, `test_server_settled_generation_metadata.py`.
- `DB_Management/`: `test_chat_retry_system_messages.py`, `test_chat_successful_regeneration.py`, `test_chat_queued_resume_history.py`.
- `Guardian/test_chat_integration.py`, `Usage/test_llm_quota_byok_calls.py`, `unit/test_moderation_event_capture.py`.
- **Excluded** (`CHAT_EXECUTORS_GATE_EXCLUDED`): `Character_Chat_NEW/unit/test_roleplay_resume_contract.py`, for R18's reason.

## Review dispositions (plan review and dry-run, round 1: `plan-review.md`)

**C1:** R5 now uses the late-import block, with the measured 9-of-9 import orders; `late_imports` moves `moderation_pipeline.py:24-30`.

**I1-I12:**

| finding | disposition |
|---|---|
| I1 | R10's lifted-name rewrite applies in lift, move and entry bodies; `crossing` ignores lifted names; R9 excludes def-bound names |
| I2 | `rewrite_in` and `free_imports` (Appendix A part 3; Tasks 7 and 8) |
| I3 | Appendix A's existing-module append row; R16 |
| I4 | R5's V-rule covers decorators and defaults only; annotations need the future import |
| I5 | R4's F401 per-file ignore (PR E) |
| I6 | S02 is queued with worker failover; S10 added |
| I7 | `submit_accounting` stub, awaited background tasks, unordered multiset (Task 1 Step 2) |
| I8 | R19's PR J `function_size` re-key; Task 10 lists the `env_read` entry |
| I9 | the pins moved to `tests/lint/` and the Enforce step (R20) |
| I10 | the pin selects from the `c25*` manifests |
| I11 | the tool is split into B, D and F (R21) |
| I12 | per-test deselects; the roleplay file is excluded; Postgres service; stated cost (R18) |

**M1-M13:**

| finding | disposition |
|---|---|
| M1 | Appendix B's spans and handler names corrected |
| M2 | holdback and the audit wait go to `streaming_pipeline` (R13) |
| M3 | Task 8 and R19 |
| M4 | `private_coercion` removed from R19 |
| M5 | `_<dest>` aliases throughout |
| M6 | design `:67` and `__all__` in Task 5's `drop` |
| M7 | Measured corrected |
| M8 | Task 1 Step 2 names the reusable fakes and the new streaming queue fake |
| M9 | the unset-slot pin lands in Task 3; `call_state` joins the subprocess pin then |
| M10 | R14 |
| M11 | the rewrite PRs are pure |
| M12 | Task 1's files and R18's globs |
| M13 | Task 9's pin clears the record and uses a `weakref` |

## Review dispositions (scoped re-review of v2: `plan-rereview.md`)

All 26 round-1 findings were confirmed resolved.

| finding | disposition |
|---|---|
| I-1 | Task 2's Files add the Enforce step's literal list and `TEST_SEAM_GATE_SUITES` |
| I-2 | R18: a new chat contract allowing only the listed `--deselect` ids; `timeout-minutes: 40` |
| I-3 | Task 1's Files and R18: the `changes` output and the "Require … gate" step |
| M-1 | R9 |
| M-2 | Task 2 |
| M-3 | Appendix A `entry` |
| M-4 | Task 9's pin hook |
| M-5 | Task 5's `drop` |
| M-6 | Task 2's pin |
| M-7 | Task 3's pin and `__getattr__` rule |
| M-8 | Appendix B's S4 and S8 rows |
