# Backend Performance Remediation — Batch 2: Chat & Character Per-Turn Path Implementation Plan (2026-10-06)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan stage-by-stage. Steps use checkbox (`- [ ]`) syntax for tracking.

**Backlog task:** TASK-13514 · **Index:** [2026-10-06-perf-remediation-coordination-index.md](2026-10-06-perf-remediation-coordination-index.md)

**Goal:** Make each chat/character turn independent of conversation length wherever it is currently linear-or-worse in history size: batch metadata fetches (and stop committing DDL per message), cache Jinja templates, reuse world-book services, hash overlap signatures, and buffer streamed tool-call arguments.

**Architecture:** All changes are inside `app/core/Chat/`, `app/core/Character_Chat/`, and one DDL guard in `ChaChaNotes_DB.py`. No API contract changes; DB write behavior unchanged.

**Tech Stack:** existing `get_message_metadata_map` batch helper, `functools.lru_cache`, `hashlib.blake2b`, `weakref.WeakKeyDictionary`, Batch-0 query counter.

## Global constraints

- Backend only: never modify `apps/**`.
- Activate venv: `source .venv/bin/activate`. One commit per stage referencing TASK-13514.
- TDD: named test first (red), implement, green. Never disable existing tests.
- Bandit: `python -m bandit -r tldw_Server_API/app/core/Chat tldw_Server_API/app/core/Character_Chat tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py -f json -o /tmp/bandit_perf_b2.json`.
- The streaming-path fix (Stage 7) must not change the SSE wire format — deltas out look identical.
- Line numbers from the 2026-10-06 review; re-locate by symbol.

---

## Stage 1: Batch message-metadata fetch + hoist per-call DDL

**Goal:** Replace the per-message `get_message_metadata` loop (1 SELECT **plus one `CREATE TABLE ... commit=True`** per message) with one `IN (...)` query, and make the table-ensure run once per DB instance.

**Files:**
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py` (history loop, ~4251-4256)
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` (`_ensure_message_metadata_table`, ~25451)
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/message_store.py` (`get_message_metadata` ~1765; keep `get_message_metadata_map` ~1800 as-is)
- Test: `tldw_Server_API/tests/Chat/test_history_metadata_batching.py` (new)

**Change:**
1. `ChaChaNotes_DB._ensure_message_metadata_table`: guard with `self._message_metadata_table_ready: bool` instance flag (per backend); after first successful ensure, subsequent calls return immediately. Set the flag also during initial schema init if the table is created there.
2. `chat_service` history loop: fetch `ids = [m["id"] for m in raw_hist if m.get("id")]` once, then `metadata_map = await asyncio.to_thread(chat_db.get_message_metadata_map, ids)` (single thread hop), and build the per-message enrichment from the map exactly as the loop did. Preserve the try/except `_CHAT_NONCRITICAL_EXCEPTIONS` behavior around the batched call.

**Success criteria:** a 100-message history load issues 2 queries (messages + metadata map), 0 DDL statements; `bench_chat_history_assembly` from Batch 0 drops from ~200+ statements to ≤5.

**Tests:**
- [ ] `test_history_load_uses_two_queries_no_ddl` — Batch-0 `count_statements` around the history-load path with 100 messages: `count <= 5`, `count_matching("CREATE TABLE") == 0`.
- [ ] `test_ensure_message_metadata_table_runs_once` — call `get_message_metadata` twice; DDL statement count == 1.
- [ ] Existing chat service tests green (metadata still attached to messages).

**Status:** Not Started

## Stage 2: Jinja template cache + identity passthrough short-circuit

**Goal:** Stop compiling 2 ad-hoc regexes + a fresh Jinja `Template` per history message when the default template is the identity `"{{message_content}}"`.

**Files:**
- Modify: `tldw_Server_API/app/core/Chat/prompt_template_manager.py` (`safe_render` + `_normalize_template_syntax`, ~111-141; default templates ~196-197)
- Test: `tldw_Server_API/tests/Chat/test_prompt_template_cache.py` (new)

**Change:**
1. `@functools.lru_cache(maxsize=256)` on `_compile_template(normalized_source: str) -> jinja2.Template` — normalize first, cache the compiled object keyed on the normalized string.
2. Identity short-circuit: after normalization + whitespace-canonicalization (strip spaces inside `{{ }}`), if source equals `"{{message_content}}"`, return `str(variables.get("message_content", ""))` directly — no Jinja involvement.
3. The per-message loop in `chat_service.py` (~4771-4795) stays; it now hits the cache.
4. Sandbox note: `_SANDBOX.from_string` must remain the only instantiation path — cached objects come from the same sandbox.

**Success criteria:** rendering 200 messages with the default template performs 0 regex compilations and 0 Jinja parses after the first; rendered output byte-identical for identity and custom templates.

**Tests:**
- [ ] `test_identity_template_short_circuits` — mock counter on `_SANDBOX.from_string` == 0 for identity renders.
- [ ] `test_custom_templates_cached` — render same custom template 50× → `from_string` (and regex compile) count == 1.
- [ ] `test_render_output_unchanged` — golden-string comparison for identity + one custom template (expected literals in test).

**Status:** Not Started

## Stage 3: Default-character cache, single token estimate, bounded continuation walk

**Goal:** Kill the 2-4 duplicate default-character lookups, the duplicated full-request token estimates, and the unbounded ancestor walk.

**Files:**
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py` (`_resolve_default_character_id` ~789-810 and callers ~839, ~902; token estimate call sites ~5081, ~6634; continuation chain walk ~3892-3923)
- Modify: `tldw_Server_API/app/core/Chat/chat_helpers.py` (~192-198) and `chat_history.py` (~88, 112) if they repeat the lookup
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py` (token-estimate computation sites ~3964/4079/4190/5022 — thread the value down)
- Test: `tldw_Server_API/tests/Chat/test_default_character_and_token_estimate.py` (new)

**Change:**
1. Cache default character id per `chat_db` instance (attribute or `WeakKeyDictionary`); invalidate on character-card create/update/delete (grep `get_character_card_by_name` writers).
2. Token estimate: compute once at the endpoint layer and pass the value down as a parameter to `execute_streaming_call` / `_execute_non_stream_call_impl`; delete the recomputation at `chat_service.py:5081, 6634`.
3. Continuation walk: parent links are chronological — stop once `len(chain) >= history_limit`; the slice is then a no-op.

**Success criteria:** one chat request performs exactly 1 default-character lookup, 1 token estimate; a continuation request walks at most `history_limit` ancestors.

**Tests:**
- [ ] `test_default_character_looked_up_once_per_request` — mock counter on `get_character_card_by_name` across a full request path.
- [ ] `test_token_estimate_computed_once` — mock the sanitize/estimate entry point; count == 1 per request.
- [ ] `test_continuation_walk_bounded` — 50-message ancestry with `history_limit=10` → `get_message_by_id` count == 10.

**Status:** Not Started

## Stage 4: Overlap-trim hashing + moderation boundary

**Goal:** Turn the O(m²) signature-slice overlap scan into O(n+m), and stop re-moderating the entire resent history.

**Files:**
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py` (`_msg_sig` usage + trim loop ~4496-4527; moderation loop ~3807-3820)
- Test: `tldw_Server_API/tests/Chat/test_overlap_trim_and_moderation.py` (new)

**Change:**
1. `_msg_sig(m)` → return `hashlib.blake2b(json.dumps(..., sort_keys=True).encode(), digest_size=16).digest()`; compare digest lists. Find the overlap boundary with a **single backward scan** (walk from the end while digests match) instead of all (k, i) slice pairs. Preserve the exact semantic of the existing algorithm (longest overlap between history tail and request prefix) — port it, then assert parity on randomized fixtures.
2. Moderation: compute the overlap boundary (4.1) first; iterate `request_data.messages` **after** the boundary index only (newly-added user/tool turns). If no boundary is found, fall back to current behavior (moderate all) — safe default.

**Success criteria:** parity with old trim results on 200 randomized conversations; moderation regex-invocation count drops to O(new turns).

**Tests:**
- [ ] `test_overlap_trim_parity_randomized` — seeded random conversations; old implementation (reference copy kept in test) vs new → identical boundary.
- [ ] `test_moderation_skips_resent_history` — 20 resent messages + 1 new → moderation hook count == 1.
- [ ] `test_moderation_all_when_no_boundary` — fallback path still moderates everything.

**Status:** Not Started

## Stage 5: WorldBookService reuse + right-to-left scan window

**Goal:** Reuse one `WorldBookService` (its entry cache currently dies with each per-turn instance) and build the 2,000-char scan window from the tail instead of joining the whole conversation.

**Files:**
- Modify: `tldw_Server_API/app/core/Character_Chat/world_book_prompt_context.py` (`build_world_book_prompt_context` ~82-93; `_recent_scan_text` ~50-63)
- Modify: `tldw_Server_API/app/core/Character_Chat/world_book_manager.py` (entry cache keying ~1886-1897, ~265; keyword index ~1925-1966)
- Modify: `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py` (call site ~5881) — pass the shared service
- Test: `tldw_Server_API/tests/Character_Chat/test_world_book_service_reuse.py` (new)

**Change:**
1. Module-level `weakref.WeakKeyDictionary` keyed on the DB instance → `WorldBookService`; `build_world_book_prompt_context` uses it when `world_book_service is None`. Entry cache key becomes `(book_id, book_last_modified)` so book edits invalidate naturally.
2. `_recent_scan_text`: iterate `messages` right-to-left accumulating lengths; stop once ≥ `window_chars`; build via `"".join(reversed(parts))[-window_chars:]` — no full-conversation join.
3. Quick-keyword index: verify whether it is rebuilt per entry or per `process_context` call; if per entry, hoist to per-call.

**Success criteria:** 3 consecutive turns on the same conversation perform 1 world-book entry load total; scan-text allocation bounded by `window_chars + max_message_len`.

**Tests:**
- [ ] `test_world_book_service_reused_across_turns` — mock DB fetch counter for entries across 3 `build_world_book_prompt_context` calls.
- [ ] `test_book_edit_invalidates_entry_cache` — bump `last_modified` → refetch.
- [ ] `test_scan_window_matches_old_impl` — randomized message lists; old (reference copy) vs new window string equality.

**Status:** Not Started

## Stage 6: Character-chat tail window + alias-inference cache

**Goal:** Stop loading up to 2,000 full messages and re-running the O(aliases×messages) scan every turn.

**Files:**
- Modify: `tldw_Server_API/app/core/Character_Chat/modules/character_chat.py` (~625-730; `_compute_additional_char_aliases` ~171-233)
- Test: `tldw_Server_API/tests/Character_Chat/test_alias_and_history_window.py` (new)

**Change:**
1. Cap the load: fetch only the last `min(messages_limit, CONTEXT_WINDOW_MESSAGES)` messages needed for prompt assembly (module constant, default 400; keep `messages_limit` for callers that legitimately want more, e.g. exports).
2. Cache `_compute_additional_char_aliases` result keyed `(conversation_id, message_count, character_id)` — recompute only when the count changed (new message).

**Success criteria:** per-turn message load ≤ 400 rows regardless of conversation age; alias scan runs once per new message, not per turn.

**Tests:**
- [ ] `test_turn_loads_bounded_window` — 2,000-message conversation, one turn → `get_messages_for_conversation` called with limit ≤ 400.
- [ ] `test_alias_cache_hit_on_same_count` — two turns, no new messages → `_compute_additional_char_aliases` invoked once.

**Status:** Not Started

## Stage 7: Streamed tool-call argument buffering

**Goal:** Replace the dict-held `+=` accumulation (quadratic char copies per tool-call stream) with a list buffer joined at finalization.

**Files:**
- Modify: `tldw_Server_API/app/core/Chat/streaming_utils.py` (tool-call accumulator ~1460-1474; function_call variant ~1484-1496; `get_accumulated_tool_calls`; `MAX_TOOL_ARGUMENT_LENGTH` ~963)
- Test: `tldw_Server_API/tests/Chat/test_streaming_tool_args_buffer.py` (new)

**Change:**
1. Accumulator stores `{"arguments_parts": list[str], "arguments_len": int}` per tool index; each delta appends to the list and adds to the counter; enforce `MAX_TOOL_ARGUMENT_LENGTH` against the counter (preserve current truncation semantics exactly).
2. `get_accumulated_tool_calls()` (and any reader of `arguments`) returns `"".join(parts)`; the external dict shape stays identical.
3. Same treatment for the legacy `function_call` variant.

**Success criteria:** wire output for a synthetic 5,000-delta stream byte-identical pre/post (golden SSE capture); no O(L²) copying (list grows, single join).

**Tests:**
- [ ] `test_tool_args_accumulate_identically` — synthetic delta sequence; joined result + final SSE frames match a pre-change golden (record golden first).
- [ ] `test_length_cap_enforced_on_counter` — deltas beyond 50k chars truncated per existing semantics.

**Status:** Not Started

---

## Batch verification

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chat tldw_Server_API/tests/Character_Chat -v -m "not external_api and not local_llm_service"
python -m tldw_Server_API.tests.perf.benchmarks.bench_chat_history_assembly   # expect statements_per_100_messages <= 5
python -m bandit -r tldw_Server_API/app/core/Chat tldw_Server_API/app/core/Character_Chat -f json -o /tmp/bandit_perf_b2.json
```

Re-record deltas in `Docs/Reviews/PERF_BASELINE_2026_10.md`; update TASK-13514 (notes, touched files, verification, final summary, DOD).
