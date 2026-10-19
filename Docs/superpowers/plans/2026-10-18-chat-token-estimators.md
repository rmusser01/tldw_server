# Chat Token Estimators: One Redactor, One Rounding Rule (TASK-13843) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every prompt-size estimate on the chat path (Resource Governor reservation, legacy rate limiter, billing reservation, usage-quota check, queue admission, prompt-cost guardrail, post-call stream estimates), and every recorded provider call's estimate row (`usage_accounting`) (Review M4), comes from one function with one data-URI redactor and one rounding rule. The guardrail redactor stops mis-slicing text whose lowercase is longer. The endpoint's redundant redaction pass goes. Data:image text that is not a real image is counted (the recommended answer to the owner's AC4 decision). Character chat's preview-only `_estimate_tokens` is out of scope (see D2) (Review M3).

**Architecture:** The one estimate already exists: `prompt_cost_envelope.estimate_segment_tokens`, rounded up. Task 1 makes its redactor, `prompt_cost_envelope._sanitize_data_uris`, a one-line wrapper over the shared `DATA_URI_REDACT_RE` from PR #3247, which fixes the position bug. Task 2 routes `chat_service.estimate_tokens_from_json`, `chat_service._estimate_tokens_from_messages` and `usage_accounting._prompt_estimate` through it, and deletes the endpoint's `_sanitize_json_for_rate_limit` and `_estimate_tokens_for_queue`, plus `chat_service._sanitize_data_uris` and `_sanitize_messages_for_token_estimate`. Task 3 narrows `DATA_URI_REDACT_RE` to real base64 images, if the owner picks option B. Its payload run is possessive, so image bulk redacts faster than `json.dumps` builds it, and a timing guard pins that (Review B1). Finding chat-7 proposed a new `core/Chat/token_estimation.py`. This plan adds no module: `prompt_cost_envelope.py` already holds the estimate, and `chat_service` already imports from it.

**Tech Stack:** Python 3.12+ (`requires-python = ">=3.12"`; Task 3's possessive quantifier `++` needs 3.11 or later) (Review B1), `re`, `json`, pytest (+ pytest-xdist for the sweep), backlog-py.

**Spec:** `backlog/tasks/task-13843 - Chat-token-estimators-still-diverge-after-the-shared-data-URI-pattern-chat-7-remainder.md` (origin/dev); finding chat-7 in `Docs/superpowers/reviews/chat/2026-09-21-stage2-orchestration-streaming.md` (lines 361-421); the TASK-13560 plan `Docs/superpowers/plans/2026-10-15-data-uri-redaction-linear.md` (PR #3247), which left these items open.

All line numbers, counts and outputs below were measured on origin/dev `e1f7770953` (2026-10-18), and on a prototype of this plan in a scratch worktree of that commit. The counts that the review fixes change were re-measured on origin/dev `7c94a2efb1`, whose 11 newer commits change no file this plan touches.

Independent review and dry run (2026-10-18): 1 Blocker, 1 Important, 9 Minor; all folded in.

## Global Constraints

- Branch `fix/chat-token-estimators` from `origin/dev`. One PR, base `dev`.
- `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. Tests: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q <paths>`.
- Test first. Each RED step must fail on the unchanged code for the reason stated before the fix goes in.
- Edit the backlog only with backlog-py (ADR-059): `PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task ...`. Use `--append-notes`, never `--notes`, and stage with `git add -A backlog/tasks`. Never hand-edit a task file.
- Every commit ends with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `--no-verify`.
- No new dependency and no new module.
- `FINGERPRINT_VERSION` stays `prompt-v1` (see Compatibility).
- Redaction stays linear. Every crafted shape in `test_data_uri_redaction_is_linear.py` must finish in under 2 s.
- Redaction stays cheap on image bulk: the ten-image, 42 MB request body redacts in at most four times the time `json.dumps` takes to build it (Task 3's guard). Each estimate site redacts the whole serialized request synchronously inside an `async` handler, so a slow pattern stalls every request on that worker (Review B1).
- No touched file is listed in `Helper_Scripts/ci/scope_predicate_baseline.txt`. No touched doc is mirrored under `Docs/Published`: `refresh_docs_published.sh` copies `Docs/Architecture.md`, not `Docs/Architecture/`, and does not copy `Docs/Product/` or `Docs/superpowers/`. So no baseline regeneration and no docs refresh are needed.
- The lint ratchets that name `chat.py` and `chat_service.py` (`test_private_coercion_ratchet.py`, `test_core_to_api_import_boundary.py`, `test_no_inline_sse_frames.py`) passed on the prototype: 9 passed. The deleted helpers are not coercers.
- dev enforces `tldw_Server_API/tests/CI/test_test_quality_ratchet.py` since #3243: new tests must not be status-only or tautological. Task 4 Step 1 runs it in the final sweep. On the prototype it passes, and `Helper_Scripts/ci/test_quality_triage.py` flags none of the six touched test files; none is in `test_quality_baseline.txt`.
- ADR-004: Change summary: waived (the owner's 2026-09-22 waiver, confirmed for burn-down PRs on 2026-10-18; recorded in the PR body).

## Decisions for the owner

### D1 (AC4): Does data:image text that is not a real image count toward pre-call estimates?

**Recommendation: B. Count it unless it is a real base64 image.** This is the smallest change that answers the question the task asks. It is one regex in `core/Utils/image_validation.py`, and every redaction site follows it. Every real data-URI image stays left out, and every ordinary prompt fingerprint stays the same, except where a markdown-embedded image sits in message text (Review M5, M6). It also keeps the behaviour PR #3247 pinned in a test: a real base64 image anywhere in the body is not counted.

**What happens today (measured).** `DATA_URI_REDACT_RE = (data:image[^,"\s]{0,128},)[^"\s]+` drops everything after `data:image…,` up to the next quote or whitespace. The estimate runs on the serialized request (`json.dumps(request_data.model_dump())`, `chat.py:3989`). `json.dumps` escapes newlines, tabs and every non-ASCII character, so in that string only a literal space or a quote ends the run. Measured on the prototype (the RED output of Task 3):

- A user message `data:image,` + 4000 characters is estimated at 17 tokens.
- `data:image/png;base64,` + 800 words joined by non-breaking spaces (8000 characters) is estimated at 20 tokens.
- The size check (`chat_validators.py:338`) does not count either message toward its 1,000,000-character limit.

Each message can carry up to 400,000 characters (`chat.py:498`), and a request up to 1000 messages (`chat.py:499`). The same redactor feeds:

- the RG reservation (`chat.py:4071`);
- the legacy rate limiter (`:4186`);
- the billing reservation and the usage-quota check (`:4297`);
- queue admission (`:5192`, `chat_service.py:5402`, `:7022`);
- the size check;
- the prompt-cost guardrail;
- the recorded usage of every streamed chat. ADR-067's Consequences say stream prompt tokens are estimated, so `chat_service.py:6265` estimates them and passes the result to `log_llm_usage` and to `rg_commit_cb`, which records billing actuals.

Non-streaming calls are corrected after the call by the provider's usage. Streams are not.

| Option | What is left out of an estimate | Consequences |
| --- | --- | --- |
| A. Keep | Any `data:image…,<run without quote or whitespace>` | No behaviour change from the pattern. The gap above stays, and for streams it is never billed. Task 2 also widens it a little, in list-content text parts and in string content across newlines or tabs (Review M2). Today `_estimate_tokens_from_messages` counts such text inside list-content `text` parts (`chat_service.py:3059-3075` rewrites only `image_url` parts and string content), and it redacts string content as raw text, where a real newline or tab ends the run. After Task 2 it redacts the `json.dumps` output, where those become `\n` and `\t`, so the run continues to the next space or quote, and it drops list-content text too, because every estimator shares one redactor. |
| **B. Real base64 only (recommended)** | `data:image/<type>[;params];base64,` followed by the base64 alphabet. JSON-escaped `\n` and `\r` are allowed, for wrapped base64. | Counted now: `data:image,…`, `data:image/png,…`, URL-encoded SVG (`data:image/svg+xml,%3Csvg…`, common in pasted HTML and CSS), and words joined by non-breaking spaces or punctuation after a real header. Such requests reach rate limits and the size check sooner, which matches what the provider bills. Real data-URI images in any position, wrapped or not, are still left out. An Anthropic-native base64 block (`source.data`) has no `data:` prefix, so `_prompt_estimate` counts it in full, before and after this plan (Review M6). Fingerprints of ordinary prompts do not change (Task 1 pins one); a markdown-embedded image in message text is the exception (see Compatibility) (Review M5). **Residual:** a run of base64-alphabet text after a valid header in a text position is still free before the call, and for streams after it too. A client could base64-encode a prompt to evade the estimate. |
| C. Image positions only | Payloads under an image key (`image_url.url`, `input_image.image_url`) in serialized JSON, matched by a key-anchored pattern. Plain-text segments are never redacted. | Closes the residual: text is always counted. It is a larger change: a second pattern (fingerprints keep the free-text one), and the endpoint test `test_endpoint_queue_estimate_ignores_a_payload_outside_a_url_key`, added by #3247, is reversed. A base64 image sent as bare string content would then be counted, which is correct, since the provider receives it as text. Task 3 would need re-planning. |

If the owner picks A, skip Task 3's code steps. Record the decision with `--append-notes "AC4 decision: A (keep) ..."`, and note the Task 2 widening in the PR body. If the owner picks C, stop after Task 2 and re-plan Task 3. If the owner wants B now and C later, file C as a follow-up task with an explicit id (AGENTS.md, Backlog section).

### D2: Rounding rule for every chat prompt estimate (Review I1)

**Recommendation: round up, `(n + 3) // 4`, never below 1 for a request.** Finding chat-7 calls this a decision for the owner: "changing a rounding rule moves rate-limit and billing numbers, so it needs a Docs/Design note and a decision on which rule wins (the conservative `(len+3)//4` is the defensible one for anything that gates money)". Chat-7 also marks the endpoint sites owner-only: "owner-only: YES for the api/v1/endpoints/chat.py sites (:3958, :4073, :4183, :6129)". On origin/dev those are `chat.py:4071` (RG reservation), `:4186` (legacy limiter), `:4297` (billing and quota) and `_sanitize_json_for_rate_limit` (`:6323`). Task 2 rewrites all four. So the owner's approval of this plan, with D2, is the approval those sites need. The design note chat-7 asks for is this plan (see ADR assessment).

Today `estimate_tokens_from_json`, `_estimate_tokens_from_messages`, `_estimate_tokens_for_queue` and `usage_accounting._prompt_estimate` round down (`max(1, n // 4)`). `estimate_segment_tokens` rounds up and returns 0 for empty text. This plan keeps rounding up, because of how the estimates are used:

- **Quota and rate pre-checks** (RG reservation, legacy limiter, `LimitEnforcer(LLM_TOKENS_MONTH)` reservation, the usage-quota check, queue admission) are admission gates. An under-count admits a request that should be refused; an over-count of under one token refuses nothing a user would notice. Rounding up is never lower than rounding down, and the two differ by exactly 1 when `n % 4 != 0` (3 lengths in 4). On a 2 KB request (500 tokens) that is 0.2%. Finding chat-7 calls the conservative rule "the defensible one for anything that gates money".
- **Post-call stream estimates** (`llm_usage_log` rows, billing actuals through `rg_commit_cb`) are billed. Rounding up bills at most one extra prompt token per stream; rounding down under-bills by 0.375 tokens on average. Both are negligible, so take the one that does not under-bill.
- **Recorded provider calls' estimate rows** (`usage_accounting._prompt_estimate`). `/chat/completions` runs its provider calls under `llm_call_context(records_itself=True)` (`chat.py:5549`, `:5822`), and so does character chat (`character_chat_sessions.py:1172`), so the provider recorder never writes their rows. `_prompt_estimate` sizes the estimate rows of every other instrumented provider call: claims, workflows, MCP tools, summarization, RAG and so on. Each row moves by at most 1 token, plus option B's redaction (Review M4).
- **Context-window trimming.** None of the estimators this plan changes trims context (`git grep`) (Review M3). World-book budgets use their own counters: `world_book_manager.count_tokens`, and `roleplay_resume_prompt.py:141`'s `(bytes + 3) // 4`, which this plan does not touch. If an estimator were ever used for trimming, rounding up would also be the safe side: it trims one token early, where rounding down overflows the window and gets a provider 400.
- `estimate_segment_tokens` already rounds up (`test_prompt_cost_envelope.py:119`, `estimate_segment_tokens("abcde") == 2`). So adopting its rule leaves the guardrail's numbers unchanged and moves only the round-down sites.

`max(1, …)` stays at the request-level estimators: a request always costs at least one token, as today. `estimate_segment_tokens("")` stays 0, so an empty segment adds nothing to an envelope.

Out of scope (Review M3):
- Completion-side estimates (`len(reply) // 4` in `chat_service.py:6268`, `:6928`, `response_processor.py:180`, `usage_accounting._approx_tokens`) are not prompt-size estimators, are outside AC1, and are unchanged.
- `character_chat_sessions._estimate_tokens` (`:6195`, `max(1, len(text) // 4)`) is preview-only: `prompt_assembly_preview` (`:6495-6514`) uses it to report section sizes and to truncate preview sections to a budget, and `get_author_note_info` (`:9528-9529`) to report counts. Neither gates, bills nor builds the prompt sent to a provider. `Character_Chat_NEW/unit/test_prd_round2_endpoints.py:152-153` pins only `== 0` and `>= 1`. It stays as it is.
- `chat.py:5301` is the TEST_MODE mock provider's usage block (`_mock_chat_call`), not an estimator.

If the owner prefers round-down, Tasks 1 and 2 need re-planning: the envelope's numbers move, and Task 1's `(len(redacted) + 3) // 4` and `total_estimated_tokens == 36` pins change with them.

## Compatibility: prompt fingerprints

The envelope's redactor feeds `canonicalize_messages`, `PromptSegment.fingerprint`, `aggregate_fingerprint` and the guardrail's `segment_fingerprints`. These are all the places that store or compare them (`git grep` on origin/dev):

| Consumer | Stores or compares? | Effect of a change |
| --- | --- | --- |
| `prompt_cost_guardrails._fingerprint_churn_warnings` | Compares the current fingerprints with `previous_fingerprints`. | The previous values come from request args (`prompt_guardrail_previous_fingerprints`, `previous_prompt_fingerprints`, `tldw_previous_prompt_fingerprints`; `chat_service.py:5216-5219`) or from `continuation_metadata["prompt_guardrail_fingerprints"]`. The latter is never written: `_resolve_tldw_continuation_history` returns only `applied`, `mode` and `from_message_id` (`chat_service.py:4016-4020`). The WebUI sends none: 0 hits under `apps/`. Character chat passes none. So only an external client that echoes fingerprints back is affected, and only when guardrails are on (default off), `warn_on_fingerprint_churn` is on (default on), and its prompt contains an input class whose redaction changed (table below). It then gets one `*_fingerprint_churn` warning. If `default_action=block`, that warning becomes a 413 for that one request. |
| `tldw_prompt_guardrails` in the chat response (`chat_service.py:7790`), and the 413 detail | Emits them. | Diagnostic only. |
| Character chat assistant-message metadata `prompt_guardrails` (`character_chat_sessions.py:7815`, ChaChaNotes) | Stores them, only when the action is not `allow`. | Never read back: no reader in `git grep`. Old rows stay as historical diagnostics. |
| `llm_usage_log.prompt_fingerprint`, `prompt_fingerprint_version`, `world_book_fingerprint` (AuthNZ migrations) | Columns exist. | No app caller passes these arguments to `log_llm_usage`, so they are always NULL. Nothing to migrate. |
| `world_book_prompt_context` fingerprint, VN Play `prompt_fingerprint`, document-insights `prompt_fingerprint` | Different hashes. | Unaffected: none goes through this redactor. |

No cache, dedupe key or quota is keyed on these fingerprints. **So no migration, and no `FINGERPRINT_VERSION` bump.** The comparator compares full strings and ignores the version, so a bump to `prompt-v2` would change every fingerprint. Every echoing client would then get the churn warning (or the 413) on its next turn, not just the few whose prompts hit a changed input class.

These inputs redact differently. The old scanner is origin/dev's `prompt_cost_envelope._sanitize_data_uris`. "Task 1" is the shared pattern as it is on dev. "Task 3" is the narrowed pattern (option B). Measured by a probe script on the prototype:

| Input | Old scanner vs Task 1 | Old scanner vs Task 3 |
| --- | --- | --- |
| Ordinary `data:image/png;base64,<base64>`, alone or in JSON | same | same |
| 30 × `İ` before a data URI | differs (old keeps the payload and puts `<omitted>` after a later comma) | differs (same reason) |
| Payload followed by `'` | differs (Task 1 eats past the quote) | same |
| Prose `see data:image docs, then more` | differs (old inserts `<omitted>`) | differs |
| Empty payload `data:image/png;base64, next` | differs | differs |
| Header longer than 128 characters | differs (old redacts it) | differs |
| `data:image,<run>`, `data:image/png,<run>` | same | differs (counted) |
| Base64 payload followed by `-`, `_` or another non-base64 character, including a markdown-embedded image: `![x](data:image/png;base64,iVBOR…==) caption` redacts to `…<omitted>) caption`, where the old scanner and Task 1 give `…<omitted> caption` (Review M5) | same | differs (the run stops there) |
| A line-wrapped base64 image (`base64.encodebytes`: raw `\n` or `\r\n` in a string) | differs (old keeps lines after the first) | differs (now redacted whole, so its fingerprint changes) |
| `data:ımage` (dotless i; `re.IGNORECASE` folds it to `i`) | differs (now redacted) | differs |

### Another plan edits `usage_accounting.py` (Review M6)

`plan_llm_usage_recording.md` (TASK-13585 and TASK-13610) also edits `usage_accounting.py`: the import at `:17`, `:103`, `:135-137`, and new functions before `:436`. This plan edits `:22`, `:30` and `:67-78`.
- No textual conflict: four unchanged lines separate the two import hunks. After either PR merges, find each edit by its quoted text, not its line number.
- No semantic conflict: every prompt-token value that plan pins comes from a provider usage block (`(21, 8)`, `(17, 9)`, `(6, 2)`, `(9, 3)`, `62003`), never from `_prompt_estimate`.
- That plan leaves Anthropic shapes (`system`, raw base64 `source.data` image blocks) out of `_prompt_estimate`, because this plan rewrites that function. This plan does not add them either; see D1 option B.

## ADR assessment

No ADR is needed. The related decisions are ADR-064 (usage quotas are per-user `limits.*` values; it does not define how prompts are estimated) and ADR-067 (accounting in the provider base class; "Stream prompt tokens are estimated"). Neither fixes a rounding rule or a redaction rule, and this plan changes neither decision. The estimator is a cheap heuristic that can be changed back in one line. Its rule is recorded in `estimate_segment_tokens`'s docstring, in the `DATA_URI_REDACT_RE` comment, in this plan (committed to `Docs/superpowers/plans/`, as TASK-13560's was) and in the task notes. ADR-059 governs the backlog edits.

Finding chat-7 asks for "a Docs/Design note and a decision on which rule wins" before the rounding rule changes. The committed plan, `Docs/superpowers/plans/2026-10-18-chat-token-estimators.md`, serves as that design note, for four reasons (Review I1):
- It holds what such a note would: the decision (D2, with D1 for redaction), every consumer of the estimate and how the change moves it, the alternatives, and the measured effect.
- The owner approves it, D1 and D2 included, before any code changes, which is the decision chat-7 asks for.
- It is committed in the same PR as the change (Task 4 Step 3), so the rationale and the code land together and stay findable from the task.
- The rule is a one-line, reversible heuristic whose live statement is `estimate_segment_tokens`'s docstring. A separate `Docs/Design` file would repeat this plan and drift from it.

## Review Focus

1. **A client that echoes fingerprints, with guardrails in block mode.** It gets one 413 after deploy if its prompt contains a changed input class. Ordinary prompts must keep their fingerprints: `test_fingerprints_of_ordinary_base64_images_do_not_change` (Task 1) pins a golden value taken from origin/dev. Under option B, a markdown-embedded image in message text is the one ordinary form whose fingerprint changes (Compatibility table) (Review M5).
2. **Wrapped base64 images** (JSON-escaped `\n` or `\r\n` in the payload) must stay left out. Otherwise a large wrapped image hits the 1,000,000-character size check and a 413 under option B. Pinned by `test_a_wrapped_base64_image_is_still_left_out[lf|crlf]` (Task 3).
3. **Pasted HTML or CSS with URL-encoded SVG** (`data:image/svg+xml,%3Csvg…`) now counts toward rate limits and the size check, by design under option B. Pinned by the `url-encoded-svg` case of `test_data_image_text_without_a_real_image_is_counted` (Task 3).
4. **A real base64 image whose header is longer than 128 characters, or contains whitespace or a quote,** is counted whole. Task 3 changes `test_media_type_over_128_chars_is_counted_not_redacted` to use a `;base64,` header, so it still pins the bound once the pattern requires `;base64,`.
5. **Crafted `data:image/` runs.** The narrowed header scans up to 128 characters and backs off looking for `;base64,`. Task 3 adds a `slash` shape (`"data:image/" * 46_000`) to the timing tests. It took 0.05 s on the prototype, at a machine load average near 80.
6. **Image bulk on the event loop (Review B1).** After Tasks 2 and 3, the size check (`chat.py:3995`), the RG or legacy-limiter estimate (`:4071` / `:4186`), the billing and quota estimate (`:4297`), queue admission (`:5192`, or `chat_service.py:5402` / `:7022`) and the stream-end or no-usage estimate (`chat_service.py:6265` / `:6921`) each redact the whole serialized request synchronously inside an `async` handler. The stream-end estimate is new bulk work: before Task 2, `_sanitize_messages_for_token_estimate` replaced image URLs before `json.dumps`. A non-possessive payload alternation is linear but took 14-44 times as long as `json.dumps` on ten 4 MB images (0.9-3.7 s per pass; the review totalled 3.6-4.5 s for one request at the default image limits); the possessive run takes 0.3-0.6 times as long. Pinned by `test_image_bulk_redacts_within_four_times_the_json_dumps_time` (Task 3).

## File map

| File | Change | Task |
| --- | --- | --- |
| `tldw_Server_API/app/core/Chat/prompt_cost_envelope.py` | `_sanitize_data_uris` uses `DATA_URI_REDACT_RE` (lines 181-200 → 3 lines); import; `estimate_segment_tokens` docstring | 1, 2 |
| `tldw_Server_API/app/core/Chat/chat_service.py` | import `estimate_segment_tokens` (line 99); drop `DATA_URI_REDACT_RE` import (226); rewrite `estimate_tokens_from_json` and `_estimate_tokens_from_messages`; delete `_sanitize_data_uris` and `_sanitize_messages_for_token_estimate` (3029-3086) | 2 |
| `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py` | `_prompt_estimate` uses `estimate_segment_tokens` (67-78); imports (30) | 2 |
| `tldw_Server_API/app/api/v1/endpoints/chat.py` | drop the outer pass at 4071, 4186, 4297; 5192 calls `estimate_tokens_from_json`; delete `_sanitize_json_for_rate_limit` (6323-6332) and `_estimate_tokens_for_queue` (6412-6418); drop the `DATA_URI_REDACT_RE` import (71) | 2 |
| `tldw_Server_API/app/core/Utils/image_validation.py` | narrow `DATA_URI_REDACT_RE` (205-209), with a possessive payload run (Review B1) | 3 |
| `tldw_Server_API/tests/Chat/unit/test_prompt_cost_envelope.py` | two tests | 1 |
| `tldw_Server_API/tests/Chat/unit/test_chat_token_estimators_agree.py` | new | 2, 3 |
| `tldw_Server_API/tests/Chat/unit/test_data_uri_redaction_is_linear.py`, `test_chat_endpoint_helpers.py`, `test_chat_service_token_estimates.py`, `tests/Resource_Governance/test_e2e_tokens_daily_cap.py` | repoint from the deleted helpers | 2, 3 |
| `tldw_Server_API/tests/Chat/unit/test_data_uri_redaction_is_linear.py` | the `slash` shape, the 128-bound input, and the image-bulk timing guard (Review B1) | 3 |
| `Docs/Product/Chat_Endpoint_Final_Split_PRD.md` | its "Required Symbols" list names the two deleted endpoint helpers (lines 92-93, 137-138) | 2 |
| `Docs/Architecture/llm-providers.md:15` | the accounting fallback's description | 2, 3 |
| `Docs/superpowers/plans/2026-10-18-chat-token-estimators.md` | this plan | 4 |

---

### Task 1: The guardrail redactor keeps positions (AC2)

**Files:**
- Modify: `tldw_Server_API/app/core/Chat/prompt_cost_envelope.py:9-15` (imports), `:181-200` (`_sanitize_data_uris`)
- Test: `tldw_Server_API/tests/Chat/unit/test_prompt_cost_envelope.py`

**Interfaces:**
- Consumes: `tldw_Server_API.app.core.Utils.image_validation.DATA_URI_REDACT_RE` (compiled `re.Pattern[str]`; group 1 is the prefix to keep).
- Produces: `prompt_cost_envelope._sanitize_data_uris(text: str) -> str` (same name and signature; it becomes the chat path's one redactor) and `estimate_segment_tokens(text: str) -> int` (unchanged signature).

- [ ] **Step 0: Branch and start the task**

Work in a worktree of your own (superpowers:using-git-worktrees), and run every later command from its root:

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_server fetch origin dev --quiet
git -C /Users/macbook-dev/Documents/GitHub/tldw_server worktree add -b fix/chat-token-estimators ../tldw-chat-token-estimators origin/dev
cd /Users/macbook-dev/Documents/GitHub/tldw-chat-token-estimators
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13843 -s "In Progress" --append-notes "Plan: Docs/superpowers/plans/2026-10-18-chat-token-estimators.md. Owner decisions: D1 (AC4 redaction) and D2 (round up, (n+3)//4, never below 1 for a request); the owner's approval of the plan covers chat-7's owner-only chat.py sites. ADR assessment: no ADR needed (ADR-064 and ADR-067 are related and unchanged; the estimator rule is a reversible heuristic recorded in the plan and docstrings). The committed plan is the design note chat-7 asks for."
git add -A backlog/tasks
```

(The task file is committed with Task 1.) The note names D1, D2 and the design-note decision; put the owner's actual answers in it (Review I1).

- [ ] **Step 1: Write the failing test and the fingerprint pin**

At the top of `tldw_Server_API/tests/Chat/unit/test_prompt_cost_envelope.py`, add `import json` and a blank line before the existing import:

```python
import json

from tldw_Server_API.app.core.Chat.prompt_cost_envelope import (
```

Append to the end of the file:

```python


def test_redaction_keeps_positions_when_lowercase_changes_length() -> None:
    # "İ".lower() is two characters. A scanner that finds "data:image" in text.lower() and
    # slices the original text with that index lands past the real comma: it keeps the
    # payload and puts "<omitted>" after the next comma instead.
    text = "İ" * 30 + " data:image/png;base64," + "A" * 400 + " tail, end"
    redacted = "İ" * 30 + " data:image/png;base64,<omitted> tail, end"

    canonical = json.loads(canonicalize_messages([{"role": "user", "content": text}]))
    envelope = build_prompt_cost_envelope([{"role": "user", "content": text}])

    assert canonical[0]["content"] == redacted
    assert envelope.total_estimated_tokens == (len(redacted) + 3) // 4


def test_fingerprints_of_ordinary_base64_images_do_not_change() -> None:
    # Pinned from origin/dev before TASK-13843: a well-formed base64 data URI redacts to the
    # same text under the old scanner and the shared pattern, so stored or echoed
    # fingerprints of ordinary prompts stay valid. Change this value only with a
    # FINGERPRINT_VERSION bump.
    messages = [
        {"role": "system", "content": "You are helpful."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo" + "A" * 400 + "=="}},
            ],
        },
        {"role": "assistant", "content": "It is a square."},
        {"role": "user", "content": "And this one? data:image/jpeg;base64,/9j/4AAQSkZJRg" + "B" * 400},
    ]
    envelope = build_prompt_cost_envelope(
        messages,
        world_book_text="World info: data:image/gif;base64,R0lGOD" + "C" * 100,
    )

    assert envelope.aggregate_fingerprint == (
        "prompt-v1:sha256:b21e1c8ad7bb61b6a4d7347b110559fa337f6a911c990fef890f368a06bd52ce"
    )
    assert envelope.total_estimated_tokens == 36
```

- [ ] **Step 2: Run it and watch the first test fail for the right reason**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q --tb=line tldw_Server_API/tests/Chat/unit/test_prompt_cost_envelope.py
```

Expected: `1 failed, 8 passed`. The failure is in `test_redaction_keeps_positions_when_lowercase_changes_length` at the canonical-content assert: `AssertionError: assert 'İİİİİİİİİİİİ...<omitted> end' == 'İİİİİİİİİİİİ...ed> tail, end'`. The old scanner kept the 400-character payload and put `<omitted>` after `tail,`. The fingerprint pin passes on the unchanged code, as it must.

- [ ] **Step 3: Replace the hand-rolled scanner with the shared pattern**

In `tldw_Server_API/app/core/Chat/prompt_cost_envelope.py`, after `from typing import Any`, add:

```python
from typing import Any

from tldw_Server_API.app.core.Utils.image_validation import DATA_URI_REDACT_RE
```

Replace the whole of `_sanitize_data_uris` (lines 181-200) with:

```python
def _sanitize_data_uris(text: str) -> str:
    """Redact data:image payloads: the chat path's one redactor, for fingerprints and estimates."""
    return DATA_URI_REDACT_RE.sub(r"\1<omitted>", text)
```

- [ ] **Step 4: Run the envelope and guardrail tests**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Chat/unit/test_prompt_cost_envelope.py tldw_Server_API/tests/Chat/unit/test_prompt_cost_guardrails.py tldw_Server_API/tests/Character_Chat/test_world_book_prompt_context.py
```

Expected: `23 passed` (the prototype). The fingerprint pin still passes: ordinary base64 URIs redact the same way.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Chat/prompt_cost_envelope.py tldw_Server_API/tests/Chat/unit/test_prompt_cost_envelope.py
git add -A backlog/tasks
git commit -m "fix(chat): the prompt-cost redactor keeps positions when lowercase changes length (TASK-13843)

prompt_cost_envelope._sanitize_data_uris found data:image in text.lower() and
sliced the original text with that index. 'İ'.lower() is two characters, so
after enough of them it skipped the real comma, kept the payload in the
fingerprint and the estimate, and put <omitted> after a later comma. It now
uses the shared DATA_URI_REDACT_RE. A pinned fingerprint shows that ordinary
base64 images fingerprint exactly as before; FINGERPRINT_VERSION stays prompt-v1.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: One estimate for the chat path, and no outer redaction pass (AC1, AC3)

**Files:**
- Create: `tldw_Server_API/tests/Chat/unit/test_chat_token_estimators_agree.py`
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py:99`, `:226`, `:3029-3086`
- Modify: `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py:20-30`, `:67-78`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py:70-74`, `:4071`, `:4186`, `:4297`, `:5192`, `:6323-6333`, `:6412-6419`
- Modify: `tldw_Server_API/app/core/Chat/prompt_cost_envelope.py:95-100` (docstring only)
- Test (repoint): `tldw_Server_API/tests/Chat/unit/test_data_uri_redaction_is_linear.py`, `tldw_Server_API/tests/Chat/unit/test_chat_endpoint_helpers.py:9-25`, `tldw_Server_API/tests/Chat/unit/test_chat_service_token_estimates.py`, `tldw_Server_API/tests/Resource_Governance/test_e2e_tokens_daily_cap.py:149-154`
- Docs: `Docs/Product/Chat_Endpoint_Final_Split_PRD.md:92-93`, `:137-138`; `Docs/Architecture/llm-providers.md:15`

**Interfaces:**
- Consumes: `prompt_cost_envelope.estimate_segment_tokens(text: str) -> int` (Task 1).
- Produces: `chat_service.estimate_tokens_from_json(request_json: str) -> int` (same signature; now `max(1, estimate_segment_tokens(request_json))`) and `chat_service._estimate_tokens_from_messages(messages: list[dict[str, Any]]) -> int` (now `estimate_tokens_from_json(json.dumps(messages, default=str))`). It removes `chat._sanitize_json_for_rate_limit`, `chat._estimate_tokens_for_queue`, `chat_service._sanitize_data_uris` and `chat_service._sanitize_messages_for_token_estimate`.

- [ ] **Step 1: Write the failing agreement test**

Create `tldw_Server_API/tests/Chat/unit/test_chat_token_estimators_agree.py`:

```python
"""The chat path's prompt-size estimators agree (TASK-13843, review finding chat-7).

Rate limiting, the billing reservation, the usage-quota check, queue admission, the
prompt-cost guardrail, the post-call stream estimates and every recorded provider call's
estimate row (usage_accounting) size a prompt with one estimate:
``prompt_cost_envelope.estimate_segment_tokens`` (data:image payloads redacted by one
pattern, four characters per token rounded up).
"""

import json

import pytest

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.prompt_cost_envelope import estimate_segment_tokens
from tldw_Server_API.app.core.LLM_Calls import usage_accounting

_IMAGE = "data:image/png;base64," + "A" * 4000
_MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this picture?"},
            {"type": "image_url", "image_url": {"url": _IMAGE}},
        ],
    },
]


@pytest.mark.unit
@pytest.mark.parametrize("extra", range(4), ids=lambda n: f"len_mod_4_plus_{n}")
def test_chat_prompt_estimators_agree_on_the_same_request(extra):
    messages = [*_MESSAGES, {"role": "user", "content": "x" * extra}]
    body = json.dumps(messages, default=str)
    expected = estimate_segment_tokens(body)

    assert chat_service.estimate_tokens_from_json(body) == expected
    assert chat_service._estimate_tokens_from_messages(messages) == expected
    assert usage_accounting._prompt_estimate({"messages": messages}) == expected


@pytest.mark.unit
def test_the_estimate_rounds_up():
    assert chat_service.estimate_tokens_from_json("abcde") == 2


@pytest.mark.unit
def test_an_empty_request_still_costs_one_token():
    assert chat_service.estimate_tokens_from_json("") == 1


@pytest.mark.unit
def test_the_endpoint_has_no_redactor_or_estimator_of_its_own():
    assert chat_endpoint.estimate_tokens_from_json is chat_service.estimate_tokens_from_json
    for name in ("_sanitize_json_for_rate_limit", "_estimate_tokens_for_queue", "DATA_URI_REDACT_RE"):
        assert not hasattr(chat_endpoint, name), name
    for name in ("_sanitize_data_uris", "_sanitize_messages_for_token_estimate", "DATA_URI_REDACT_RE"):
        assert not hasattr(chat_service, name), name
```

The parametrize covers every remainder of the length mod 4: rounding down and rounding up agree only when the length is a multiple of 4. The last test pins AC3. The endpoint keeps no redactor of its own, so a second pass cannot come back unnoticed. The module docstring names `usage_accounting`'s estimate rows separately: `/chat/completions` and character chat record themselves (`records_itself=True`), so `_prompt_estimate` sizes the rows of every other instrumented provider call, not the chat path's (Review M4).

- [ ] **Step 2: Run it and watch it fail**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q --tb=line tldw_Server_API/tests/Chat/unit/test_chat_token_estimators_agree.py
```

Expected: `6 failed, 1 passed` (`test_an_empty_request_still_costs_one_token` passes):
- Three `len_mod_4_plus_*` cases fail at the first assert, e.g. `assert 60 == 61` and `assert 61 == 62`: `estimate_tokens_from_json` rounds down.
- The fourth fails at the second assert, `assert 57 == 61`: there rounding down equals rounding up, but `_estimate_tokens_from_messages` writes `"data:image/omitted"`, 13 characters shorter than the shared redaction.
- `test_the_estimate_rounds_up`: `assert 1 == 2`.
- `test_the_endpoint_has_no_redactor_or_estimator_of_its_own`: `AssertionError: _sanitize_json_for_rate_limit`.

- [ ] **Step 3: Route chat_service through the one estimate**

In `tldw_Server_API/app/core/Chat/chat_service.py`, replace line 99:

```python
from tldw_Server_API.app.core.Chat.prompt_cost_envelope import build_prompt_cost_envelope
```

with:

```python
from tldw_Server_API.app.core.Chat.prompt_cost_envelope import (
    build_prompt_cost_envelope,
    estimate_segment_tokens,
)
```

Delete line 226 (its only user is the function deleted below; `validate_image_url` at line 4052 has its own local import):

```python
from tldw_Server_API.app.core.Utils.image_validation import DATA_URI_REDACT_RE
```

Replace everything from `def estimate_tokens_from_json` through the end of `_estimate_tokens_from_messages` (lines 3029-3086: `estimate_tokens_from_json`, `_sanitize_data_uris`, `_sanitize_messages_for_token_estimate`, `_estimate_tokens_from_messages`) with:

```python
def estimate_tokens_from_json(request_json: str) -> int:
    """Heuristic prompt size for rate limits, quotas, billing and queue admission.

    The chat path's one estimate (``estimate_segment_tokens``: data:image payloads
    redacted, four characters per token rounded up), never below 1.
    """
    try:
        return max(1, estimate_segment_tokens(request_json))
    except _CHAT_NONCRITICAL_EXCEPTIONS:
        return 1


def _estimate_tokens_from_messages(messages: list[dict[str, Any]]) -> int:
    """The same estimate for a message list, taken over its JSON."""
    try:
        return estimate_tokens_from_json(_json.dumps(messages, default=str))
    except _CHAT_NONCRITICAL_EXCEPTIONS:
        return 1
```

Callers keep their names and signatures: `chat.py:4071`, `:4186`, `:4297`, `:5192` (after Step 5), `chat_service.py:5402`, `:6265`, `:6921`, `:7022`.

- [ ] **Step 4: Route the accounting fallback through it**

In `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py`, add the import as the first `tldw_Server_API` import (after `from loguru import logger` and its blank line):

```python
from loguru import logger

from tldw_Server_API.app.core.Chat.prompt_cost_envelope import estimate_segment_tokens
from tldw_Server_API.app.core.Usage.llm_accounting import submit_accounting
```

Delete line 30:

```python
from tldw_Server_API.app.core.Utils.image_validation import DATA_URI_REDACT_RE
```

In `_prompt_estimate`, change the docstring and the return line:

```python
def _prompt_estimate(request: Any) -> int:
    """Estimated prompt tokens: the messages plus the system prompt, by the chat path's estimate."""
    try:
        if not isinstance(request, dict):
            return 1
        text = json.dumps(request.get("messages"), default=str)
        system_message = request.get("system_message")
        if isinstance(system_message, str):
            text += system_message
        return max(1, estimate_segment_tokens(text))
    except Exception:  # noqa: BLE001 - an estimate must never fail a call
        return 1
```

`LLM_Calls` already imports from `core.Chat` (`adapter_utils.py:12-13`), and `prompt_cost_envelope` imports only the stdlib and `image_validation`, so there is no import cycle. The prototype imports cleanly and passes `tests/LLM_Calls/test_usage_accounting.py`. Its docstring says it uses "the estimate the chat endpoint already uses"; after this step that is true again. `_approx_tokens` (completions) is unchanged.

- [ ] **Step 5: Drop the endpoint's own redactor and estimator**

In `tldw_Server_API/app/api/v1/endpoints/chat.py`:

Lines 70-74 become:

```python
from tldw_Server_API.app.core.Utils.image_validation import (
    get_max_base64_bytes,
    validate_image_url,
)
```

Line 4071:

```python
                estimated_tokens = estimate_tokens_from_json(request_json)
```

Line 4186:

```python
                    estimated_tokens = estimate_tokens_from_json(request_json)
```

Line 4297:

```python
                    estimate_tokens_from_json(request_json) if request_json else 1000
```

Line 5192:

```python
                        est_tokens_for_queue = estimate_tokens_from_json(request_json)
```

Delete `_sanitize_json_for_rate_limit` (lines 6323-6332) and the two blank lines after it, and `_estimate_tokens_for_queue` (lines 6412-6418) and the two blank lines after it. Then check that nothing still uses them:

```bash
git grep -n "_sanitize_json_for_rate_limit\|_estimate_tokens_for_queue\|DATA_URI_REDACT_RE" -- tldw_Server_API/app
```

Expected: only `tldw_Server_API/app/core/Utils/image_validation.py` (the definition), `tldw_Server_API/app/api/v1/schemas/chat_validators.py` (the size check), and `tldw_Server_API/app/core/Chat/prompt_cost_envelope.py`.

That removes the redundant pass. On every request, the RG or legacy limiter estimate and the billing/quota estimate each redacted the serialized request twice. Each estimate site now redacts it once. This is not "once per request" (Review M7): every site still redacts the request it estimates, and with guardrails on, the envelope sanitizes each text part in `_extract_text` (`prompt_cost_envelope.py:286-300` on origin/dev) and `estimate_segment_tokens` sanitizes the same text again. That second pass is idempotent under both patterns, runs only with guardrails on, and is outside AC3, so this plan leaves it.

- [ ] **Step 6: Say in the estimate's docstring that it is the one estimate**

In `prompt_cost_envelope.py`, replace the `estimate_segment_tokens` docstring line:

```python
    """Conservative deterministic estimate using the existing 4 chars/token heuristic."""
```

with:

```python
    """The chat path's one prompt-size estimate: data:image payloads redacted, 4 chars/token rounded up.

    Rate limits, quota and billing reservations, queue admission, the prompt-cost guardrail,
    the post-call stream estimates and every recorded provider call's estimate row
    (usage_accounting) all use it (TASK-13843). Rounded up because every one of them is
    safer over than under.
    """
```

The docstring names `usage_accounting`'s rows as their own consumer (Review M4).

- [ ] **Step 7: Run the new test**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Chat/unit/test_chat_token_estimators_agree.py tldw_Server_API/tests/Chat/unit/test_prompt_cost_envelope.py
```

Expected: `16 passed`.

- [ ] **Step 8: Repoint the tests that imported the deleted helpers**

Without this step, `test_data_uri_redaction_is_linear.py` fails at collection with `AttributeError: module 'tldw_Server_API.app.core.Chat.chat_service' has no attribute '_sanitize_data_uris'`. Each test keeps what it checked; only the function under test changes.

`tldw_Server_API/tests/Chat/unit/test_data_uri_redaction_is_linear.py`:

1. Replace the end of the module docstring and the imports:

```python
seconds to redact. Each chat call site must handle it in well under two seconds.
The call sites now share one pattern, so the endpoint's estimate also redacts a
payload outside a "url" key and the size validator an uppercase data URI.
Since TASK-13843 the chat path has one redactor (prompt_cost_envelope._sanitize_data_uris)
and one estimate (estimate_segment_tokens) behind every prompt-size estimate.
"""

import time

import pytest

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.api.v1.schemas import chat_validators
from tldw_Server_API.app.core.Chat import chat_service, prompt_cost_envelope
```

2. Replace `test_chat_endpoint_rate_limit_sanitizer_is_linear`:

```python
@pytest.mark.unit
@_CRAFTED_SHAPES
def test_prompt_cost_estimate_is_linear(body):
    _assert_fast(prompt_cost_envelope.estimate_segment_tokens, body)
```

3. Replace `test_ordinary_data_uri_payload_is_still_redacted`, `test_real_data_uri_after_crafted_markers_is_still_redacted` and `test_endpoint_queue_estimate_ignores_a_payload_outside_a_url_key`:

```python
@pytest.mark.unit
def test_ordinary_data_uri_payload_is_still_redacted():
    body = '{"url": "data:image/png;base64,' + "A" * 400 + '"}'
    redacted = prompt_cost_envelope._sanitize_data_uris(body)
    assert "A" * 400 not in redacted
    assert "data:image/png;base64," in redacted


@pytest.mark.unit
def test_real_data_uri_after_crafted_markers_is_still_redacted():
    body = _CRAFTED + '"url":"data:image/png;base64,' + "A" * 400 + '"'
    assert "A" * 400 not in prompt_cost_envelope._sanitize_data_uris(body)


@pytest.mark.unit
def test_endpoint_queue_estimate_ignores_a_payload_outside_a_url_key():
    body = '{"messages":[{"role":"user","content":"data:image/png;base64,' + "A" * 4000 + '"}]}'
    assert chat_endpoint.estimate_tokens_from_json(body) < 100
```

4. In `test_media_type_over_128_chars_is_counted_not_redacted`, change `chat_service._sanitize_data_uris(body)` to `prompt_cost_envelope._sanitize_data_uris(body)`.

`tldw_Server_API/tests/Chat/unit/test_chat_endpoint_helpers.py`: replace `test_queue_estimate_sanitizes_base64_payload` (lines 9-25) with:

```python
def test_queue_estimate_sanitizes_base64_payload():
    def _request(payload: str) -> str:
        return (
            "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\""
            + payload
            + "\"}}]}]}"
        )

    large = chat_endpoint.estimate_tokens_from_json(_request("data:image/png;base64," + "a" * 400))
    small = chat_endpoint.estimate_tokens_from_json(_request("data:image/png;base64," + "a" * 4))

    assert large == small
```

`tldw_Server_API/tests/Chat/unit/test_chat_service_token_estimates.py`: the old test compared against a hand-written `"data:image/omitted"` URL, which was the deleted helper's output. Replace the file with:

```python
from tldw_Server_API.app.core.Chat import chat_service


def _messages(base64_payload: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_payload}"}},
            ],
        }
    ]


def test_token_estimate_redacts_base64_images():
    est_with_payload = chat_service._estimate_tokens_from_messages(_messages("A" * 4000))
    est_small_payload = chat_service._estimate_tokens_from_messages(_messages("A" * 4))

    assert est_with_payload == est_small_payload
```

`tldw_Server_API/tests/Resource_Governance/test_e2e_tokens_daily_cap.py` (lines 149-154): delete the `_sanitize_json_for_rate_limit` import and compute the reserve the way the endpoint now does:

```python
    from tldw_Server_API.app.core.Chat.chat_service import estimate_tokens_from_json

    req_model = ChatCompletionRequest(**body)
    request_json = json.dumps(req_model.model_dump())
    est = int(estimate_tokens_from_json(request_json) or 1)
```

- [ ] **Step 9: Run every directly affected test**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Chat/unit/test_chat_token_estimators_agree.py \
  tldw_Server_API/tests/Chat/unit/test_prompt_cost_envelope.py \
  tldw_Server_API/tests/Chat/unit/test_prompt_cost_guardrails.py \
  tldw_Server_API/tests/Chat/unit/test_chat_endpoint_helpers.py \
  tldw_Server_API/tests/Chat/unit/test_chat_service_token_estimates.py \
  tldw_Server_API/tests/Chat/unit/test_data_uri_redaction_is_linear.py \
  tldw_Server_API/tests/Chat/unit/test_chat_service_queue_estimate.py \
  tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py \
  tldw_Server_API/tests/Resource_Governance/test_e2e_tokens_daily_cap.py
```

Expected: `84 passed` (the prototype).

- [ ] **Step 10: Update the two docs that name the deleted helpers or the old rule**

`Docs/Product/Chat_Endpoint_Final_Split_PRD.md`: replace these lines in "Required Symbols for Existing Tests/App":

```markdown
- `_save_message_turn_to_db`
- `_sanitize_json_for_rate_limit`
- `_estimate_tokens_for_queue`

These symbols remain importable from `chat.py` through wrappers/re-exports even after extraction.
```

with:

```markdown
- `_save_message_turn_to_db`

These symbols remain importable from `chat.py` through wrappers/re-exports even after extraction.

`_sanitize_json_for_rate_limit` and `_estimate_tokens_for_queue` were removed by TASK-13843: every chat
prompt-size estimate is `chat_service.estimate_tokens_from_json`, which `chat.py` imports.
```

The removal note goes after the "These symbols remain importable" sentence, so that sentence still points at the list above it, not at the removed helpers (Review M8).

Also delete these two lines under "Core Chat Modules (New)":

```markdown
- `core/Chat/queue_estimation.py` (optional if needed)
  - `_sanitize_json_for_rate_limit` and `_estimate_tokens_for_queue` logic with endpoint-level wrappers preserved.
```

The PRD's list exists so that tests keep their imports during a future split of `chat.py`. The only importers were the three tests repointed in Step 8. `git grep` shows no test or app code that reads the PRD.

`Docs/Architecture/llm-providers.md:15`: replace `estimation fallback (4 chars/token, data-URIs redacted);` with:

```markdown
estimation fallback (the chat path's `estimate_segment_tokens`: 4 chars/token rounded up, data-URIs redacted);
```

- [ ] **Step 11: Commit**

```bash
git add tldw_Server_API/tests/Chat/unit/test_chat_token_estimators_agree.py \
  tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/core/Chat/prompt_cost_envelope.py \
  tldw_Server_API/app/core/LLM_Calls/usage_accounting.py tldw_Server_API/app/api/v1/endpoints/chat.py \
  tldw_Server_API/tests/Chat/unit/test_data_uri_redaction_is_linear.py \
  tldw_Server_API/tests/Chat/unit/test_chat_endpoint_helpers.py \
  tldw_Server_API/tests/Chat/unit/test_chat_service_token_estimates.py \
  tldw_Server_API/tests/Resource_Governance/test_e2e_tokens_daily_cap.py \
  Docs/Product/Chat_Endpoint_Final_Split_PRD.md Docs/Architecture/llm-providers.md
# The message names usage_accounting's rows and says "each estimate site redacts once" (Review M4, M7).
git commit -m "fix(chat): one prompt-size estimate for the chat path; drop the endpoint's second redaction pass (TASK-13843)

Five estimators sized the same prompt: estimate_tokens_from_json,
_estimate_tokens_from_messages, the endpoint's _estimate_tokens_for_queue and
usage_accounting._prompt_estimate rounded down, estimate_segment_tokens rounded
up, and _estimate_tokens_from_messages also had its own structural redactor.
They now all return estimate_segment_tokens (one redactor, rounded up; never
below 1 for a request), so the rate limiter, the RG and billing reservations,
the quota check, queue admission, the guardrail, the stream usage estimate and
every recorded provider call's estimate row (usage_accounting) agree. Rounded
up because each of them gates or bills tokens: an estimate grows by one token
in three lengths out of four (owner decision D2).

The endpoint redacted the request twice before every estimate
(estimate_tokens_from_json(_sanitize_json_for_rate_limit(...))). The outer
helper and _estimate_tokens_for_queue are gone; the endpoint has no redactor
of its own, and each estimate site redacts once.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Count data:image text that is not a real image (AC4, option B; skip if the owner picks A or C)

**Files:**
- Modify: `tldw_Server_API/app/core/Utils/image_validation.py:205-209`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_token_estimators_agree.py`, `tldw_Server_API/tests/Chat/unit/test_data_uri_redaction_is_linear.py`
- Docs: `Docs/Architecture/llm-providers.md:15`

**Interfaces:**
- Consumes: Tasks 1-2. Every chat estimate, the size check (`chat_validators.validate_request_size`, string path) and the envelope fingerprints use `DATA_URI_REDACT_RE`.
- Produces: `DATA_URI_REDACT_RE`, the same name and group-1 contract (the prefix to keep, now ending in `;base64,`). This flips a shared default, so Task 4 Step 1 runs every test file that imports a module using it. Its payload run uses a possessive quantifier, which needs Python 3.11 or later; the project requires 3.12 (Review B1).

- [ ] **Step 1: Write the failing tests**

In `test_chat_token_estimators_agree.py`, add the size-check module to the imports:

```python
from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.api.v1.schemas import chat_validators
from tldw_Server_API.app.core.Chat import chat_service
```

Insert before `test_the_endpoint_has_no_redactor_or_estimator_of_its_own`. In the `nbsp-joined-words` case, spell the non-breaking space as the Python escape `\xa0`, exactly as shown: four ASCII characters, never a literal U+00A0 and never a backslash-u escape. The agent Write and Edit tools turn a backslash-u escape into the literal character, and an editor that normalizes the literal character to a plain space makes that case pass on dev, so RED would show 5 failures, not 6 (Review M1):

```python
@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "data:image," + "x" * 4000,
        "data:image/png," + "x" * 4000,
        "data:image/svg+xml," + "%3Csvg%3E" * 450,
        "data:image/png;base64," + "word\xa0" * 800,
        "data:image/png;base64," + "word-" * 800,
    ],
    ids=["no-media-type", "no-base64-marker", "url-encoded-svg", "nbsp-joined-words", "hyphen-joined-words"],
)
def test_data_image_text_without_a_real_image_is_counted(text):
    # AC4 decision (TASK-13843): only a real base64 image payload is left out of an estimate.
    body = json.dumps({"messages": [{"role": "user", "content": text}]})
    assert chat_service.estimate_tokens_from_json(body) >= 4000 // 4


@pytest.mark.unit
@pytest.mark.parametrize("line_break", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_a_wrapped_base64_image_is_still_left_out(line_break):
    wrapped = "data:image/png;base64," + ("A" * 76 + line_break) * 50
    body = json.dumps({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": wrapped}}]}]})
    assert chat_service.estimate_tokens_from_json(body) < 100


@pytest.mark.unit
def test_the_size_check_counts_data_image_text_without_a_real_image(monkeypatch):
    monkeypatch.setattr(chat_validators, "MAX_REQUEST_SIZE", 1000)
    body = json.dumps({"messages": [{"role": "user", "content": "data:image," + "x" * 2000}]})
    with pytest.raises(ValueError, match="Request too large"):
        chat_validators.validate_request_size(body)
```

In `test_data_uri_redaction_is_linear.py`, replace the `_CRAFTED_SHAPES` line with:

```python
# The third is the worst case for the base64-header pattern: every "data:image/" makes the
# bounded header scan 128 characters and back off looking for ";base64,".
_CRAFTED_SHAPES = pytest.mark.parametrize(
    "body",
    [_CRAFTED, "data:image" * 51_000, "data:image/" * 46_000],
    ids=["url-key", "quote-free", "slash"],
)
```

In `test_media_type_over_128_chars_is_counted_not_redacted`, make the over-long header a base64 one, so the test still pins the 128 bound once the pattern requires `;base64,`:

```python
    body = "data:image/" + "x" * 200 + ";base64,AAAA"
    assert prompt_cost_envelope._sanitize_data_uris(body) == body
```

Add the image-bulk guard (Review B1). In `test_data_uri_redaction_is_linear.py`, add `import json` above `import time`:

```python
import json
import time
```

After `_assert_fast`, add:

```python


def _best_time(fn, arg) -> float:
    times = []
    for _ in range(3):
        started = time.perf_counter()
        fn(arg)
        times.append(time.perf_counter() - started)
    return min(times)
```

Append to the end of the file:

```python


@pytest.mark.unit
def test_image_bulk_redacts_within_four_times_the_json_dumps_time():
    # Ten 4 MB base64 images, about 42 MB of request JSON: the usual large body, which every
    # estimate site redacts on the event loop. Timing json.dumps of the same request in the same
    # run keeps the bound independent of machine load. Measured ratios: dev's pattern 2.0-2.5x,
    # a non-possessive base64 payload alternation 14-44x, the possessive one 0.3-0.6x.
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * (4 * 1024 * 1024)}}
    request = {"messages": [{"role": "user", "content": [image] * 10}]}
    body = json.dumps(request)

    build = _best_time(json.dumps, request)
    redact = _best_time(chat_service.estimate_tokens_from_json, body)

    assert redact <= 4 * build, f"redaction took {redact:.3f}s, json.dumps {build:.3f}s"
```

The bound is four times, not twice. dev's own pattern takes 2.0-2.5 times the `json.dumps` time on this body (the review measured 1.98, this revision 2.2-2.5), so a 2x bound would pass or fail at random in this task's RED run. At 4x, dev's pattern passes with a 1.6x margin, the possessive pattern passes with a margin of 6.7x or more, and the non-possessive alternation (14-44x) fails. Taking the best of three runs of each, in the same process, keeps the ratio steady under load: it held at load averages of 17 and 80.

- [ ] **Step 2: Run them and watch them fail**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q --tb=line tldw_Server_API/tests/Chat/unit/test_chat_token_estimators_agree.py tldw_Server_API/tests/Chat/unit/test_data_uri_redaction_is_linear.py
```

Expected: `6 failed, 24 passed` (re-measured on the prototype, Review B1). The five `test_data_image_text_without_a_real_image_is_counted` cases fail with small estimates: `assert 17 >= (4000 // 4)` (no media type), `18` (no base64 marker), `19` (URL-encoded SVG) and `20` (both joined-words cases). `test_the_size_check_counts_data_image_text_without_a_real_image` fails with `Failed: DID NOT RAISE ValueError`. Both wrapped cases, the three `slash` timing cases, the updated 128-bound test and the image-bulk guard pass on the current pattern, because they are guards.

- [ ] **Step 3: Narrow the shared pattern to real base64 images**

In `tldw_Server_API/app/core/Utils/image_validation.py`, replace lines 205-209 (the comment and `DATA_URI_REDACT_RE`) with:

```python
# Finds base64 image payloads in free text (a request body, a prompt) for redaction; group 1 is
# the "data:image/...;base64," prefix to keep. Only a real base64 image is redacted: the header
# names an image type and base64, and the payload is the base64 alphabet (JSON-escaped line
# breaks allowed, for wrapped base64). Any other text after "data:image" is counted, because a
# provider bills it as text (TASK-13843). The header is bounded: an unbounded ``[^,]*`` rescans
# to the end of the text for every ``data:image`` with no comma after it, which is quadratic on
# crafted input. The payload run is possessive (``++``): it never gives characters back, so
# image bulk is scanned at about one character per step.
DATA_URI_REDACT_RE = re.compile(r'((?i:data:image/[^,"\s]{0,128};base64,))(?:[A-Za-z0-9+/=]++|\\[rn])+')
```

`(?i:…)` makes only the header case-insensitive. The payload alphabet is written out in full, so no case folding can add characters to it. The payload alternation is unambiguous (the class excludes `\`), so it cannot backtrack.

The payload run is possessive, `[A-Za-z0-9+/=]++` (Review B1). Without `++`, the engine enters the group and picks a branch for every payload character: still linear, but 14-44 times as slow as `json.dumps` on image bulk. With it, a run of base64 characters is one step. Possessive quantifiers exist since Python 3.11, and `pyproject.toml` has `requires-python = ">=3.12"`. The output does not change: the payload group is the last thing in the pattern, so a greedy match has nothing after it to give characters back for, and the possessive and plain forms match the same span on every input. Check that on every Task 3 case and every PR #3247 input (from the repo root):

```bash
$PY - <<'EOF' 2>&1 | tail -1
import json
import re

from tldw_Server_API.app.core.Utils.image_validation import DATA_URI_REDACT_RE

assert "++" in DATA_URI_REDACT_RE.pattern
plain = re.compile(DATA_URI_REDACT_RE.pattern.replace("]++|", "]|"))


def body(content):
    return json.dumps({"messages": [{"role": "user", "content": content}]})


def fill(unit):
    return unit * (510_000 // len(unit) + 1)


crafted = '"url":"data:image' * 30_000
image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * (4 * 1024 * 1024)}}
inputs = [
    # Task 3
    body("data:image," + "x" * 4000),
    body("data:image/png," + "x" * 4000),
    body("data:image/svg+xml," + "%3Csvg%3E" * 450),
    body("data:image/png;base64," + "word\xa0" * 800),
    body("data:image/png;base64," + "word-" * 800),
    *(body([{"type": "image_url", "image_url": {"url": "data:image/png;base64," + ("A" * 76 + b) * 50}}]) for b in ("\n", "\r\n")),
    body("data:image," + "x" * 2000),
    "data:image/" * 46_000,
    "data:image/" + "x" * 200 + ";base64,AAAA",
    json.dumps({"messages": [{"role": "user", "content": [image]}]}),
    # PR #3247 (TASK-13560)
    crafted,
    "data:image" * 51_000,
    '{"url": "data:image/png;base64,' + "A" * 400 + '"}',
    crafted + '"url":"data:image/png;base64,' + "A" * 400 + '"',
    '{"messages":[{"role":"user","content":"data:image/png;base64,' + "A" * 4000 + '"}]}',
    '{"url": "DATA:IMAGE/PNG;BASE64,' + "A" * 5000 + '"}',
    "data:image/" + "x" * 200 + ",payload",
    json.dumps([{"role": "user", "content": "data:image" * 50_000}]),
    *(fill(unit) for unit in (
        '"url":"data:image', "data:image", "data:image" + "x" * 118, "data:image" + ";" * 128,
        "DaTa:ImAgE", 'data:image,"', "data:image,", "data:imag", "d",
        'data:image/png;base64,' + "A" * 1000 + '" ',
    )),
    # Edge cases: a markdown image, escaped line breaks, a non-base64 escape
    "![x](data:image/png;base64,iVBOR" + "A" * 50 + "==) caption",
    "data:image/png;base64,AAAA\\nBBBB\\r\\nCCCC-tail",
    "data:image/png;base64,A\\x" * 21_000,
]
for text in inputs:
    assert DATA_URI_REDACT_RE.sub(r"\1<omitted>", text) == plain.sub(r"\1<omitted>", text), text[:60]
print(f"byte-identical on {len(inputs)} inputs")
EOF
```

Expected: `byte-identical on 32 inputs` (the prototype; `tail -1` drops the config log lines the import prints, and shows the `AssertionError` line if an input differs). The #3247 inputs are every body in `test_data_uri_redaction_is_linear.py` as merged, the crafted prompt in `test_usage_accounting.py`, and the ten adversarial shapes from the TASK-13560 probe at 510 KB. The image body is the one-image version of the guard's body: the plain form takes seconds on the full ten-image body.

Timings, best of 3, on the prototype (Review B1). The machine was shared with other test runs (load average 17-82), so read the ratios to `json.dumps`, not the milliseconds:

| Input | `json.dumps` | dev `[^"\s]+` | non-possessive option B | possessive option B (this step) |
| --- | --- | --- | --- | --- |
| 1 image, 4.2 MB request JSON | 6.6-7.6 ms | 16.3-17.0 ms (2.2-2.5x) | 127-255 ms (17-37x) | 3.4-4.0 ms (0.5x) |
| 10 images, 41.9 MB request JSON | 67-85 ms | 164-187 ms (2.2-2.5x) | 1,343-3,714 ms (20-44x) | 33-40 ms (0.5-0.6x) |

The review measured the ten-image body at 62 ms for `json.dumps`, 123 ms for dev, 886.5 ms for the non-possessive form and 21.3 ms for the possessive one. `_estimate_tokens_from_messages` on ten 4 MB images, which now serializes and redacts the bulk, took 146-252 ms here, against 922-989 ms in the review with the non-possessive form; on dev it is about 0 ms, because the structural redactor never serialized the bulk. Swapping the non-possessive form back in makes the guard fail, `AssertionError: redaction took 3.630s, json.dumps 0.090s`, while the other 29 tests in the two files pass.

Crafted shapes of about 500 KB, possessive form: `url-key` 0.003 s, `quote-free` 0.003 s, `slash` 0.047 s, `data:image/;` repeated 0.041 s, `data:image/png;base64` repeated with no comma 0.020 s, 250,000 `\n` escapes 0.011 s. The non-possessive form took the same within 0.01 s on each.

- [ ] **Step 4: Run the tests of every module that uses the pattern**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Chat/unit/test_chat_token_estimators_agree.py \
  tldw_Server_API/tests/Chat/unit/test_data_uri_redaction_is_linear.py \
  tldw_Server_API/tests/Chat/unit/test_prompt_cost_envelope.py \
  tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py \
  tldw_Server_API/tests/Utils/test_image_validation.py
```

Expected: `109 passed` (re-measured on the prototype: the 108 before the review plus the image-bulk guard, Review B1). The fingerprint pin from Task 1 still passes, and so does #3247's `test_endpoint_queue_estimate_ignores_a_payload_outside_a_url_key`: a real base64 image is still left out wherever it is. The guard took 0.92 s, most of it building the 42 MB body.

- [ ] **Step 5: Update the doc line**

`Docs/Architecture/llm-providers.md:15`: change `4 chars/token rounded up, data-URIs redacted` to `4 chars/token rounded up, base64 image payloads redacted`.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Utils/image_validation.py \
  tldw_Server_API/tests/Chat/unit/test_chat_token_estimators_agree.py \
  tldw_Server_API/tests/Chat/unit/test_data_uri_redaction_is_linear.py \
  Docs/Architecture/llm-providers.md
git commit -m "fix(chat): count data:image text that is not a real base64 image (TASK-13843)

DATA_URI_REDACT_RE dropped everything after data:image..., up to a quote or a
space. In a serialized request only those end the run, so data:image,<400,000
characters> cost about 2 tokens in the rate limits, the RG and billing
reservations, the quota check, queue admission and the size check, and for
streams in the recorded usage too. The pattern now matches only
data:image/<type>[;params];base64, and the base64 alphabet (JSON-escaped line
breaks allowed, for wrapped images). Real images are redacted as before, and
ordinary prompt fingerprints are unchanged. Text after a valid header that is
itself base64 is still left out (owner decision D1, option B).

The payload run is possessive. A plain alternation is linear but took 14 to
44 times as long as json.dumps on ten 4 MB images, and every estimate site
redacts on the event loop. The possessive run takes half as long as
json.dumps and gives the same output; a guard pins the ratio.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Verify, close the task, open the PR

**Files:**
- Create: `Docs/superpowers/plans/2026-10-18-chat-token-estimators.md` (this plan)
- Modify: `backlog/tasks/task-13843 - ….md` (with backlog-py only)

- [ ] **Step 1: Run every test file that imports a changed module, and compare with origin/dev**

```bash
mkdir -p .superpowers/sdd
git grep -l -P 'core\.Utils\.image_validation|core\.Utils import [^\n]*image_validation|core\.LLM_Calls\.usage_accounting|core\.LLM_Calls import [^\n]*usage_accounting|core\.Chat\.chat_service\b|core\.Chat import [^\n]*\bchat_service\b|core\.Chat\.prompt_cost_(envelope|guardrails)\b|core\.Chat import [^\n]*\bprompt_cost_(envelope|guardrails)\b|world_book_prompt_context\b|schemas\.chat_validators\b|schemas import [^\n]*\bchat_validators\b|v1\.endpoints\.chat\b|v1\.endpoints import [^\n]*\bchat\b' -- 'tldw_Server_API/tests/*.py' > .superpowers/sdd/t13843_files.txt
wc -l < .superpowers/sdd/t13843_files.txt
TLDW_TEST_NO_DOCKER=1 RUN_JOBS=1 RUN_EVALUATIONS=1 $PY -m pytest -p no:cacheprovider -q -n 4 -rfE --tb=line $(cat .superpowers/sdd/t13843_files.txt) tldw_Server_API/tests/CI/test_test_quality_ratchet.py
```

Use `-P`: with `-E`, macOS `git grep` reads `\b` as a literal `b` (from the TASK-13560 plan). The list is 200 files on the branch (199 on origin/dev, plus the new test file once committed). The last path adds the test-quality ratchet, which dev enforces since #3243: new tests must not be status-only or tautological. It imports no changed module, so the `git grep` does not list it.

Expected: no failure that does not also fail on unchanged `origin/dev`. The same command in a clean worktree of `origin/dev` `e1f7770953`, over its 199 files, gave `11 failed, 6346 passed, 35 skipped` in 47 minutes:
- `Character_Chat_NEW/integration/test_chat_import_local_chats.py::test_another_owner_reusing_ids_never_touches_the_first_owners_chat[postgres]`. It runs only when a Postgres is reachable, and also failed on the TASK-13560 baseline.
- Ten `Services/test_router_groups_contract.py::test_iter_minimal_optional_router_specs_*` (`*media*` and `*audio*`) cases. They fail whenever xdist runs a Claims test earlier on the same worker: `tests/Claims/conftest.py` sets `ROUTES_DISABLE` for the whole session (diagnosed in the TASK-13560 plan). Whether they fail depends on worker placement.

The first prototype, over 200 files, gave `2 failed, 6374 passed, 35 skipped` (19 more tests: the new file has 15, and the edited files 4). The failures:
- the Postgres owner-reuse import, as on the baseline;
- `Admin/test_admin_llm_provider_test_runtime.py::test_admin_health_blocking_adapter_timeout_bounds_capacity_and_recovers`. It is a timing test that passed when run alone on the prototype (`1 passed`). The TASK-13560 plan records it as flaky under xdist load: it failed on that plan's base run and passed with its change.

The review's dry run on origin/dev `7c94a2efb1` gave `1 failed, 6374 passed, 35 skipped, 1 error`: the owner-reuse import again, and a setup error in `Character_Chat_NEW/unit/test_roleplay_resume_contract.py::test_original_replay_does_not_rewrite_or_recheck_later_source_changes[postgres-snapshot-missing]` (`CharactersRAGDBError: Unexpected database initialization error`). That is Postgres initialization contention under xdist: the case passes alone on both trees.

Re-measured after the review fixes (Review B1), with the command above, ratchet included, on the revised prototype of `7c94a2efb1`: `6 failed, 6372 passed, 35 skipped` in 61 minutes, at a machine load average of 17-80. That is 6,413 tests, two more than the first prototype: the image-bulk guard and the ratchet, both of which passed. The failures:
- the Postgres owner-reuse import, as on the baseline;
- five `Services/test_router_groups_contract.py::test_iter_minimal_optional_router_specs_*media*` cases, from the Claims `ROUTES_DISABLE` leak above. The file passes alone on the prototype: `178 passed`.

Whether the router-contract cases fail depends on worker placement: ten failed on the base run, none on the first prototype, five here. For every failure outside these lists, run the test alone on both trees before blaming or clearing the change. If one fails only with the change, stop and fix it before going on.

- [ ] **Step 2: Lint and security**

```bash
$PY -m ruff check --output-format concise tldw_Server_API/app/core/Chat/prompt_cost_envelope.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/core/LLM_Calls/usage_accounting.py tldw_Server_API/app/core/Utils/image_validation.py tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/tests/Chat/unit/test_chat_token_estimators_agree.py tldw_Server_API/tests/Chat/unit/test_prompt_cost_envelope.py tldw_Server_API/tests/Chat/unit/test_data_uri_redaction_is_linear.py tldw_Server_API/tests/Chat/unit/test_chat_endpoint_helpers.py tldw_Server_API/tests/Chat/unit/test_chat_service_token_estimates.py tldw_Server_API/tests/Resource_Governance/test_e2e_tokens_daily_cap.py
uvx bandit -ll tldw_Server_API/app/core/Chat/prompt_cost_envelope.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/core/LLM_Calls/usage_accounting.py tldw_Server_API/app/core/Utils/image_validation.py tldw_Server_API/app/api/v1/endpoints/chat.py
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/lint/test_private_coercion_ratchet.py tldw_Server_API/tests/lint/test_core_to_api_import_boundary.py tldw_Server_API/tests/lint/test_no_inline_sse_frames.py
```

Expected (prototype, re-measured): ruff prints `Found 5 errors.`, all of them already on `origin/dev`, and no new one:

```text
tldw_Server_API/app/core/Chat/chat_service.py:11:1: I001 [*] Import block is un-sorted or un-formatted
tldw_Server_API/app/core/Chat/prompt_cost_envelope.py:146:14: C420 [*] Unnecessary dict comprehension for iterable; use `dict.fromkeys` instead
tldw_Server_API/app/core/Chat/prompt_cost_envelope.py:196:36: SIM118 Use `key in dict` instead of `key in dict.keys()`
tldw_Server_API/app/core/Chat/prompt_cost_envelope.py:229:37: SIM118 Use `key in dict` instead of `key in dict.keys()`
tldw_Server_API/tests/Resource_Governance/test_e2e_tokens_daily_cap.py:31:9: I001 [*] Import block is un-sorted or un-formatted
```

The `prompt_cost_envelope.py` findings are at lines 138, 205 and 238 on origin/dev; the changes shift them to 146, 196 and 229 (one line later than before the review, because the Step 6 docstring now names `usage_accounting`'s rows). `test_e2e_tokens_daily_cap.py` is linted too (Review M9). On origin/dev it has two findings, `:31:9 I001` and `:148:5 I001`; Task 2 Step 8 drops the out-of-order `_sanitize_json_for_rate_limit` import from the block at `:148`, which leaves that block sorted, so only `:31` is left. Bandit: `No issues identified.` (`bandit` is not in the venv; `uvx` fetches it.) The ratchets: `9 passed`.

- [ ] **Step 3: Commit the plan and close the task**

Copy this plan to `Docs/superpowers/plans/2026-10-18-chat-token-estimators.md` and check it for trailing whitespace (the pre-commit hook rejects it):

```bash
grep -n '[[:space:]]$' Docs/superpowers/plans/2026-10-18-chat-token-estimators.md
```

Expected: no output.

The notes below quote the prototype's numbers. Put the numbers your own runs printed in place of them, and the owner's actual D1 and D2 answers (Review I1).

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13843 --append-notes "AC2: prompt_cost_envelope._sanitize_data_uris now uses the shared DATA_URI_REDACT_RE. The hand-rolled scanner found positions in text.lower() and sliced the original text, so 30 'İ' before a data URI kept the 400-character payload. RED 1 failed, 8 passed (test_prompt_cost_envelope.py); GREEN 23 passed with the guardrail and world-book prompt-context tests. A pinned origin/dev fingerprint shows ordinary base64 images fingerprint as before. Fingerprints: nothing caches, dedupes or meters on them (guardrail churn compares only client-echoed values; the WebUI sends none; llm_usage_log.prompt_fingerprint is never written; character-chat metadata is write-only), so there is no migration and FINGERPRINT_VERSION stays prompt-v1." --append-notes "AC1 and AC3: estimate_tokens_from_json, _estimate_tokens_from_messages and usage_accounting._prompt_estimate all return prompt_cost_envelope.estimate_segment_tokens: one redactor, rounded up ((n+3)//4, never below 1 for a request; owner decision D2), because every use gates or bills tokens and none of these estimators trims context. That covers the chat path and the estimate rows of every recorded provider call (usage_accounting). Character chat's preview-only _estimate_tokens is out of scope. The endpoint's _sanitize_json_for_rate_limit (the redundant outer pass at the three estimate sites) and _estimate_tokens_for_queue are deleted, as are chat_service._sanitize_data_uris and _sanitize_messages_for_token_estimate; each estimate site redacts once. New test_chat_token_estimators_agree.py: RED 6 failed, 1 passed; GREEN 7 passed. Directly affected tests: 84 passed." --append-notes "AC4 decision (owner, D1): B. Only a real base64 image (data:image/<type>[;params];base64, plus the base64 alphabet, JSON-escaped line breaks allowed) is left out of an estimate or the size check. data:image text without one now counts. Residual: base64-alphabet text after a valid header is still left out; option C (redact image positions only) would close it. The payload run is possessive: same output as the plain alternation on 32 inputs (every Task 3 case and the PR #3247 inputs), and a 42 MB ten-image body redacts in about 0.5x the json.dumps time, against 14-44x for the plain form; test_image_bulk_redacts_within_four_times_the_json_dumps_time pins it. RED 6 failed, 24 passed; GREEN 109 passed across the five pattern-user test files. Sweep (200 files plus tests/CI/test_test_quality_ratchet.py, -n 4): 6 failed, 6372 passed, 35 skipped; the image-bulk guard and the ratchet passed. One failure is the Postgres owner-reuse import, which also fails on origin/dev. The other five are router-contract media cases from the Claims ROUTES_DISABLE leak; that file passes alone (178 passed). origin/dev e1f7770953 over its 199 files: 11 failed, 6346 passed, 35 skipped (that import, plus ten router-contract cases from the same leak). Ruff: no new findings. Bandit -ll on the five source files: no issues." --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 --final-summary "The chat path sizes prompts one way: every estimator (rate limits, RG and billing reservations, quota check, queue admission, prompt-cost guardrail, stream usage), and every recorded provider call's estimate row (usage_accounting), returns estimate_segment_tokens, with one data-URI redactor, rounded up. The guardrail redactor no longer mis-slices text whose lowercase is longer, each estimate site redacts the request once instead of twice, and data:image text that is not a real base64 image now counts. Ordinary prompt fingerprints are unchanged." -s Done
git add -A backlog/tasks
git add Docs/superpowers/plans/2026-10-18-chat-token-estimators.md
git commit -m "docs(chat): plan for one chat token estimate; close TASK-13843

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git show --stat HEAD
```

Expected: `git show --stat HEAD` lists the task file and the plan only.

The notes record D2, the possessive run and its guard, and the out-of-scope character-chat estimator; the final summary names `usage_accounting`'s rows and says "each estimate site redacts the request once" (Review I1, B1, M3, M4, M7).

- [ ] **Step 4: Push and open the PR against dev**

```bash
git push -u origin fix/chat-token-estimators
gh pr create --base dev --title "fix(chat): one prompt-size estimate for the chat path (TASK-13843)" --body "$(cat <<'EOF'
Closes TASK-13843 (the chat-7 remainder of Docs/superpowers/reviews/chat/2026-09-21-stage2-orchestration-streaming.md).

- AC1 (owner decision D2 = round up): every chat prompt-size estimator, and usage_accounting's estimate rows, return prompt_cost_envelope.estimate_segment_tokens: one redactor, rounded up, never below 1 for a request. test_chat_token_estimators_agree.py checks that they agree.
- AC2: the prompt-cost redactor uses the shared pattern and keeps positions for text whose lowercase is longer.
- AC3: the endpoint's second redaction pass (_sanitize_json_for_rate_limit) and _estimate_tokens_for_queue are gone; each estimate site redacts once.
- AC4 (owner decision D1 = B): data:image text that is not a real base64 image counts toward estimates and the size check. The payload run is possessive, so a 42 MB ten-image request redacts in about half the time json.dumps takes to build it; test_image_bulk_redacts_within_four_times_the_json_dumps_time pins that.

Fingerprints of ordinary prompts are unchanged (pinned); a markdown-embedded image is the one ordinary form whose fingerprint changes. FINGERPRINT_VERSION stays prompt-v1. Plan, which is also chat-7's design note: Docs/superpowers/plans/2026-10-18-chat-token-estimators.md.

## Change summary
(owner to write; ADR-004, no waiver for this task)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

The body names D2, `usage_accounting`'s rows, the bulk guard, the markdown-image exception and the design note (Review I1, B1, M4, M5, M7).

- [ ] **Step 5: Review and merge**

1. Run `/superpowers:requesting-code-review` in a clean subagent, with this plan and the PR diff, and address its findings before merging.
2. Change summary: waived (the owner's 2026-09-22 waiver, confirmed for burn-down PRs on 2026-10-18; recorded in the PR body).
3. Merge through the serial queue (the owner's `pr_queue.sh`), one PR at a time; MERGE_QUEUE is unset. Editing the PR body re-runs the required license status, so wait for it before merging.
4. Confirm the PR is MERGED before deleting the branch.
