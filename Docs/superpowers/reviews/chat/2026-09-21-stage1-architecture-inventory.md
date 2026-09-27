# Stage 1 — Architecture survey and inventory (`core/Chat`)

Date: 2026-09-21. Reviewer role: principal engineer, read-only audit.

## Scope

Establish what the module actually is before judging it: file inventory, size x churn, the recent
defect history, and — the decisive question for this module — which entrypoints are live and which are
kept alive only by a test. Findings that depend on reachability (`chat-4`, `chat-10`) are grounded here.

## Code Paths Reviewed

- `core/Chat/chat_service.py` (whole file surveyed; 7,285 LOC, 148 commits/12mo)
  - `execute_streaming_call` (4788-6249) — one function, 1,462 lines
  - `_execute_non_stream_call_impl` (6250-7276) — one function, 1,027 lines
  - `build_context_and_messages` (4009-4518) — one function, 510 lines
- `core/Chat/chat_orchestrator.py:chat_api_call (342-593)`, `:chat_api_call_async (594-703)`,
  `:_chat_sync_impl (869-1190)`, `:chat (1191-1378)`, `:achat (1379-1670)`
- `core/Chat/streaming_utils.py:StreamingResponseHandler (1281-2285)`,
  `:create_streaming_response_with_timeout (2286-2531)`
- `core/Chat/completion_pipeline.py (1-26)`, `core/Chat/streaming_pipeline.py (1-71)`,
  `core/Chat/orchestrator/{__init__,error_mapping,provider_resolution,request_validation,stream_execution}.py`
  (17 + 11 + 17 + 22 + 52 LOC)
- `core/Chat/Workflows.py:1-60` (loads `./App_Function_Libraries/Workflows/Workflows.json`)
- `core/Chat/README.md`, `core/Chat/REFACTORING_PLAN.md`

## Tests Reviewed

Located by import-grep only (`grep -rl "core\.Chat" tldw_Server_API/tests` → **288 files**). Full
per-module table in `2026-09-21-stage3-test-inventory.txt`. Stage-1-relevant entries:

| Test file | Protects | Downgrades risk? |
| --- | --- | --- |
| `tests/Chat/unit/test_chat_workflows.py:7` | the *only* importer of `core.Chat.Workflows`, which is in turn the only importer of `chat_orchestrator.chat` | No — it is what keeps the dead path alive; it does not exercise production reachability |
| `tests/Chat/unit/test_chat_orchestrator_contract.py`, `test_chat_orchestrator_bedrock.py` | `chat_api_call` dispatch contract | Partially — covers the live `chat_api_call`, not `_chat_sync_impl`/`achat` |
| `tests/Chat_NEW/unit/test_chat_sync_wrapper.py` | `chat()` sync-wrapper guard rails (event-loop refusal, `CHAT_COMMANDS_ASYNC_ONLY`) | Yes for the wrapper; no for the 320-line body |

## Validation Commands

```
$ find tldw_Server_API/app/core/Chat -name '*.py' | xargs wc -l | sort -rn | head -6
   25698 total
    7285 tldw_Server_API/app/core/Chat/chat_service.py
    2541 tldw_Server_API/app/core/Chat/streaming_utils.py
    1670 tldw_Server_API/app/core/Chat/chat_orchestrator.py
    1503 tldw_Server_API/app/core/Chat/document_generator.py
    1335 tldw_Server_API/app/core/Chat/request_queue.py
```

```
$ git log --since='12 months ago' --name-only --pretty=format: -- tldw_Server_API/app/core/Chat/ \
    | grep '\.py$' | sort | uniq -c | sort -rn | head -6
 148 chat_service.py
  42 streaming_utils.py
  34 chat_orchestrator.py
  23 rate_limiter.py
  22 provider_config.py        <- file no longer exists (deleted during the window)
  22 document_generator.py
```

```
$ grep -rn 'Chat.chat_orchestrator import\|Chat import chat_orchestrator' --include='*.py' tldw_Server_API/app/
core/Chat/Workflows.py:34:from ...Chat.chat_orchestrator import chat
core/Chat_Workflows/dialogue_orchestrator.py:6:... import chat_api_call_async
core/Claims_Extraction/ingestion_claims.py:360:... import chat_api_call as _cac
core/Character_Chat/modules/character_memory_extraction.py:222:... import chat_api_call
core/AuthNZ/byok_testing.py:34:... import chat_api_call
core/Prompt_Management/prompt_studio/prompt_executor.py:516:... import chat_api_call as _legacy_call
(6 hits)
```

```
$ grep -rn 'Chat.Workflows\|Chat import Workflows' --include='*.py' tldw_Server_API/ | grep -v Chat_Workflows
tldw_Server_API/tests/Chat/unit/test_chat_workflows.py:7:from tldw_Server_API.app.core.Chat import Workflows as workflows
(1 hit — the only importer, and it is a test)

$ ls -d App_Function_Libraries
ls: App_Function_Libraries: No such file or directory
```

```
$ git log --since='3 months ago' --oneline -- tldw_Server_API/app/core/Chat/ | wc -l
50
$ git log --since='3 months ago' --oneline -- tldw_Server_API/app/core/Chat/ | grep -c '^[0-9a-f]* fix'
(the first 12 of 50 are all `fix(chat|uat|sharing|llm)`; 8 of the first 12 are the
 failed-turn-retry / image-identity / provider-context cluster)
```

## Findings

### FINDING chat-9
```
axis:        encapsulation
class:       n/a
severity:    Medium
sites:       core/Chat/chat_service.py:execute_streaming_call (4788-6249, 1462 lines);
             core/Chat/chat_service.py:_execute_non_stream_call_impl (6250-7276, 1027 lines);
             core/Chat/chat_service.py:build_context_and_messages (4009-4518, 510 lines)
canonical:   NONE
destination: follow the shipped in-repo template: core/DB_Management/Media_DB_v2.py (121 commits of
             churn) was split into the core/DB_Management/media_db/ package (api.py, constants.py,
             errors.py, legacy_content_queries.py, runtime/). The equivalent here is a
             core/Chat/completion/ package with one responsibility per module. The named-but-empty
             boundaries already exist and are the natural seams: completion_pipeline.py (26 LOC),
             streaming_pipeline.py (71 LOC), orchestrator/ (4 files, 102 LOC total).
knowledge:   The turn-assembly rules (history windowing, retry/failed-turn recovery, image identity,
             overlap trimming, continuation anchors) and the streaming lifecycle (queue bridging,
             credential recording, moderation, persistence callbacks, SSE finalization) are each spread
             through one enormous function body with no named seam. Every rule change has to be made
             inside a 1,000+ line `try` block, so reviewers cannot see the whole state machine.
scenario:    n/a (encapsulation axis)
impact:      Three functions carry 3,000 of the file's 7,285 lines, and the file carries 148 of the
             module's ~320 commits in 12 months. The last 3 months of `fix(chat): ...` commits are
             concentrated in exactly this surface — `preserve image identity across failed-turn
             recovery`, `reuse the unanswered user on an explicit failed-turn retry`, `place accepted
             retry turn last in provider context`, `correlate persisted failed turns before retry
             hydration`, `distinguish new repeated questions during retry`. Five separate fixes in one
             quarter to one un-named state machine is the cost, measured.
cost-driver: n/a
tests:       96 test files import chat_service (import-grep reachability, not measured coverage);
             the retry cluster specifically: tests/Chat/unit/test_failed_retry_provider_order.py,
             tests/Chat/unit/test_chat_persistence_content.py,
             tests/Chat/unit/test_chat_history_multi_image.py,
             tests/Chat_NEW/unit/test_chat_continuation_controls.py,
             tests/Chat_NEW/integration/test_chat_continuation_controls_integration.py
effort:      expensive — needs a Docs/Design/ doc, an ADR, a Backlog task and a staged
             IMPLEMENTATION_PLAN. Coverage is good enough that the extraction is low-risk; the cost is
             sequencing, not danger.
owner-only:  no (core/Chat only), but the seam touches api/v1/endpoints/chat.py imports → coordinate.
confidence:  confirmed (the sizes and the churn); probable-risk (that decomposition reduces the defect
             rate — argued from the fix history, not measured).
```

### FINDING chat-13
```
axis:        duplication
class:       divergent-copies
severity:    Low
sites:       core/Chat/REFACTORING_PLAN.md:31 ("chat_orchestrator.py - source of truth for
             orchestration (`achat` canonical, `chat` sync wrapper)");
             core/Chat/REFACTORING_PLAN.md:36 ("chat_service.py - compatibility facade");
             core/Chat/REFACTORING_PLAN.md:83-84 (instructs "extending
             tests/Chat/test_chat_functions.py", a file that does not exist);
             core/Chat/README.md:180,329 (points at tests/Chat only, never mentions tests/Chat_NEW);
             contradicted by core/Chat/chat_orchestrator.py:627-630
             (`log_legacy_once("chat_orchestrator.chat_api_call_async is deprecated; use
             chat_service.perform_chat_api_call_async instead")`)
canonical:   Docs/Architecture.md is the operative architecture record.
destination: n/a — delete REFACTORING_PLAN.md or fold its residue into core/Chat/README.md, which is
             the module's one documentation owner.
knowledge:   "Which module owns chat orchestration." The in-source plan says chat_orchestrator;
             the code says chat_service (7,285 LOC / 148 commits vs 1,670 / 34, and the orchestrator's
             own deprecation log line points the other way).
scenario:    n/a
impact:      Low in isolation, but it is the document a new contributor reads first, and it sends them
             to the wrong module and to a test file that was deleted. It also states a "Current Status
             (May 2025)" against a 2026 codebase.
cost-driver: n/a
tests:       n/a (documentation)
effort:      cheap — delete or rewrite one file.
owner-only:  no
confidence:  confirmed
```

## Suggested Refactor/Actions

1. **Delete `core/Chat/Workflows.py` and the `chat`/`achat`/`_chat_sync_impl` trio** unless a consumer
   is produced. Evidence: `Workflows.py` loads `./App_Function_Libraries/Workflows/Workflows.json`, a
   path from the retired Gradio UI that does not exist in the repo; its only importer is
   `tests/Chat/unit/test_chat_workflows.py`. That removes ~640 of `chat_orchestrator.py`'s 1,670 lines
   and resolves findings `chat-4` and `chat-10` by deletion rather than by refactor. Deletion over
   addition: this is the cheapest high-value action in the module. Propose as a Backlog task; do not
   delete without owner sign-off, because `chat_api_call` in the same file *is* live (5 core callers).
2. **Rewrite or delete `core/Chat/REFACTORING_PLAN.md`** (finding `chat-13`). One file, no code risk.
3. **Sequence `chat-9` behind the `media_db/` template**, not ahead of it — read
   `core/DB_Management/media_db/` package boundaries first, then propose `core/Chat/completion/`
   with the same shape. Needs `Docs/Design/2026-MM-DD-chat-completion-decomposition-design.md`, an ADR,
   a Backlog task and `IMPLEMENTATION_PLAN_chat-completion.md` with 3-5 staged goals.
