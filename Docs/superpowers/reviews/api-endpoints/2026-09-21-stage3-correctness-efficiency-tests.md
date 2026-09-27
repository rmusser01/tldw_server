# Stage 3 — Correctness, efficiency, test reachability, synthesis

## Scope

Axis 4 (correctness / latent bugs) and Axis 5 (efficiency) across
`tldw_Server_API/app/api/v1/endpoints/`, concentrated on the size x churn hot set from stage 1.
Every correctness finding here carries a concrete failure scenario; every efficiency finding names
the cost driver and what it scales with. Findings without one were dropped — the dropped list is at
the end.

Test reachability for all cited modules is in `2026-09-21-stage3-test-inventory.txt`. No suite was
executed; every `tests:` line in this ledger is import-grep reachability, not measured coverage.

## Code Paths Reviewed

- `chat.py:_decode_knowledge_qa_share_token (6496-6519)`, `:_urlsafe_b64decode (6478-6480)`,
  `:_get_knowledge_qa_share_signing_key (6462-6471)`,
  `:resolve_conversation_share_token (7670-7700+)`, `:_verify_conversation_ownership (6351-6380)`,
  `:get_conversation_citations (8047-8077)`, `:_replace_conversation_keywords (6405-6448)`,
  `:update_chat_conversation (7196-7204)`, `:save_chat_knowledge (6817-6824)`,
  `:_persist_with_transaction (3189-3213)`
- `persona.py:persona_catalog (7556-7585)`, `:_run_persona_db_call (891-894)`,
  `:_load_persona_buddy_rows_for_projection (3116)`, `:get_persona_voice_analytics (6707-6868)`,
  `:list_persona_visual_packs (4353-4359)`,
  `:list_persona_visual_generated_candidates (4751-4757)`,
  `:restore_persona_profile_state_entry (6328-6409)`, `:dry_run_persona_voice_command (7275-7367)`
- `files.py:_parse_iso_datetime (83-92)`, `:export_file_artifact (185-265)`,
  `:_clear_export_state (95-115)`; `core/File_Artifacts/file_artifacts_service.py:378`,
  `:_build_export_info_from_row (770-792)`;
  `core/DB_Management/Collections_DB.py:1270,1679` (the `export_expires_at TEXT` DDL)
- `workflows.py:2192-2210`, `:2593-2602`
- `api/v1/API_Deps/auth_deps.py:get_db_transaction (417-442)`;
  `core/AuthNZ/database.py:get_db_transaction (1908-1912)`, `:_normalize_sqlite_sql (1966)`
- `api/v1/schemas/chat_conversation_schemas.py:76` (`keywords: list[str] | None`)

## Tests Reviewed

See `2026-09-21-stage3-test-inventory.txt` for the full table and the per-test notes. The four that
materially bear on findings below:

- `tests/Chat/unit/test_chat_share_links_api.py:234-241` — the only malformed-share-token test.
  Sends `"not-a-valid-token"` (no `.`), caught by the arity check at `chat.py:6498`, never reaches
  the decode. **Does not downgrade api-endpoints-1.**
- `tests/Files/test_files_endpoint.py:146,173,221,728` — cover `export_expires_at` as `None`, past,
  and future, all produced by `datetime.isoformat()`. **Do not downgrade api-endpoints-11.**
- `tests/Admin/test_admin_rate_limits_api.py` — stub-driven dual-backend assertions.
  **Does not downgrade api-endpoints-12** (stage 1).
- `tests/lint/test_endpoint_auth_deps_import_boundary.py` — the ratchet precedent for the stage-1
  action list.

`chat.py` has 78 test files and `persona.py` 27 by import-grep, which is why several findings below
are graded `cheap` on effort: the route surface is pinned even where the specific branch is not.
No endpoint test was found for `GET /api/v1/chat/conversations/{id}/citations`.

## Validation Commands

```
$ sed -n '6496,6519p' tldw_Server_API/app/api/v1/endpoints/chat.py
    # confirms: provided_signature = _urlsafe_b64decode(encoded_signature)  is at :6507,
    # OUTSIDE the try: that starts at :6511

$ python3 - <<'EOF'
import base64
def dec(v): return base64.urlsafe_b64decode(f"{v}{'='*(-len(v)%4)}")
for c in ["AAAA!", "A", "AAAAA", "AA=A"]:
    try: print(repr(c), "->", dec(c))
    except Exception as e: print(repr(c), "->", type(e).__module__+"."+type(e).__name__+":", e)
EOF
'AAAA!'  -> b'\x00\x00\x00'
'A'      -> binascii.Error: Invalid base64-encoded string: number of data characters (1) cannot be 1 more than a multiple of 4
'AAAAA'  -> binascii.Error: Invalid base64-encoded string: number of data characters (5) cannot be 1 more than a multiple of 4
'AA=A'   -> binascii.Error: Incorrect padding

$ python3 -c "import binascii; print(binascii.Error.__mro__)"
(<class 'binascii.Error'>, <class 'ValueError'>, <class 'Exception'>, <class 'BaseException'>, <class 'object'>)

$ grep -rn 'shared/conversations' tldw_Server_API/tests --include='*.py'
tests/Chat/unit/test_chat_share_links_api.py:146
tests/Chat/unit/test_chat_share_links_api.py:164
tests/Chat/unit/test_chat_share_links_api.py:239      # "not-a-valid-token", expects 400

$ grep -c '_run_persona_db_call' tldw_Server_API/app/api/v1/endpoints/persona.py
     106
$ sed -n '7566,7580p' tldw_Server_API/app/api/v1/endpoints/persona.py
    # confirms: db.list_persona_profiles(..., limit=200) then a per-profile
    # db.list_persona_policy_rules(...) inside the loop, neither offloaded

$ grep -rn 'time\.sleep(\|requests\.\(get\|post\)(' tldw_Server_API/app/api/v1/endpoints --include='*.py'
embeddings_v5_production_enhanced.py:1104      # background cleanup thread
chat.py:3210                                   # inside _persist_with_transaction, which runs via
                                               # current_loop.run_in_executor at :3216 -- CORRECT,
                                               # dropped as a finding

$ grep -rn 'def [a-zA-Z_]*(.*=\s*\[\]\|def [a-zA-Z_]*(.*=\s*{}' tldw_Server_API/app/api/v1/endpoints --include='*.py' \
    | grep -v 'Query(\|Body(\|Field(\|Depends('
    (no output — no mutable default arguments in endpoints)
```

## Findings

### FINDING api-endpoints-1 — unauthenticated HTTP 500 on the public share-token route: a base64 decode outside the try block

```
axis:        correctness
class:       divergent-copies
severity:    High
sites:       chat.py:_urlsafe_b64decode (6478-6480)
             chat.py:_decode_knowledge_qa_share_token (6496-6519) — the decode at :6507 is OUTSIDE
               the try that begins at :6511
             chat.py:resolve_conversation_share_token (7670-7686) — the public route
               GET /api/v1/chat/shared/conversations/{share_token}, registered on both `router`
               and `conversations_alias_router`, with NO auth dependency in its signature
             correct sibling: notes.py:_decode_attachment_cursor (857-912)
canonical:   notes.py:857-912 is the least-wrong implementation in this layer and should be the
             model: size cap -> 413, schema pre-check, canonical-base64 re-encode check,
             hmac.compare_digest, all decodes inside one try whose except tuple includes ValueError
destination: n/a — the fix is 3 lines in chat.py; the shared form is api-endpoints-10
knowledge:   "What HTTP status a malformed signed token produces." chat.py answers 400 for a
             missing separator (:6498), 403 for a bad signature (:6509), 400 for a bad payload
             (:6513) — and 500 for a bad signature *encoding*, because that one path escapes.
scenario:    `GET /api/v1/chat/shared/conversations/AAAA.A` — an unauthenticated request with a
             two-part token whose signature segment is a single base64 character.
             `token.split(".")` gives 2 parts, so the arity check at :6498 passes. :6502 computes
             the expected HMAC. :6507 calls `_urlsafe_b64decode("A")`, which pads to `"A==="` and
             raises `binascii.Error: Invalid base64-encoded string: number of data characters (1)
             cannot be 1 more than a multiple of 4` (verified on the CI interpreter, 3.12.11).
             `binascii.Error` IS a `ValueError` and IS in the except tuple at :6513 — but the try
             does not start until :6511, so the exception escapes the function and the route,
             reaching main.py's unhandled-exception handler as
             `HTTP 500 {"detail": "Internal server error"}`. Any two-part token whose signature
             segment has `len % 4 == 1`, or contains a stray `=`, does the same. The correct
             response is 400 or 403.
impact:      High. Confirmed, reachable, zero-cost to trigger, on a public route with no
             authentication. It is not an auth bypass and leaks no data — the failure is upstream
             of the signature comparison — but it converts a malformed-input case into a server
             error on an unauthenticated surface, which pollutes error-rate monitoring and is a
             free 5xx generator. The same file's sibling `notes.py` implementation shows the
             correct handling already exists one directory over.
tests:       Import-grep reachability, not measured coverage. `chat` has 78 test files;
             `tests/Chat/unit/test_chat_share_links_api.py:234-241` is the only malformed-token
             test and it sends `"not-a-valid-token"`, which has no `.` and is rejected by the arity
             check before reaching the decode. The bug is untested.
effort:      Cheap. Move :6507 inside the try (or widen the try to start at :6501) and add
             `binascii.Error` explicitly for readability. One regression test with the payload
             `"AAAA.A"` asserting 400 or 403.
owner-only:  yes
confidence:  confirmed (the exception, the try boundary, the route being unauthenticated, the test
             gap)
```

### FINDING api-endpoints-4 — `persona_catalog` runs up to 201 blocking SQLite queries on the event loop, bypassing the file's own offload helper and its own batching precedent

```
axis:        efficiency
class:       adoption-gap
severity:    High
sites:       persona.py:persona_catalog (7556-7585) — `db.list_persona_profiles(..., limit=200)` at
               :7566 and `db.list_persona_policy_rules(...)` at :7571 inside the `for profile in
               profiles` loop, neither awaited nor offloaded
             persona.py:_run_persona_db_call (891-894) — the file's own
               `asyncio.to_thread` wrapper, docstring "Offload synchronous persona DB calls from
               async HTTP handlers", used 106 times elsewhere in the same file
             persona.py:_load_persona_buddy_rows_for_projection (3116), called at :7569 — the
               batched form, in the same function, one line above the N+1 loop
             same N+1 shape, offloaded but still per-item:
               persona.py:list_persona_visual_packs (4353-4359) — one
                 `list_persona_visual_assets` per pack
               persona.py:list_persona_visual_generated_candidates (4751-4757)
             other unoffloaded sync DB calls on async routes:
               persona.py:get_persona_voice_analytics (6707-6868) — 6 sync aggregates
               persona.py:restore_persona_profile_state_entry (6328-6409) — 4
               persona.py:dry_run_persona_voice_command (7275-7367) — 4
canonical:   persona.py:_run_persona_db_call (891-894) — exists, documented, used 106 times,
             bypassed here
destination: n/a — adoption, plus one batched `list_persona_policy_rules_for_personas` on the
             ChaChaNotes owner mirroring the buddy-row loader that already exists at :3116
knowledge:   "Persona DB reads must not block the loop, and per-profile projection data must be
             batch-loaded." Both rules are already written down in this file — one as a helper with
             106 users, one as a function called on the line above the violation.
scenario:    n/a (efficiency axis)
cost-driver: N+1 plus event-loop blocking. `GET /api/v1/persona/catalog` issues 1
             `list_persona_profiles` query plus one `list_persona_policy_rules` query per profile,
             all synchronous SQLite on the event loop thread. Scales linearly with the number of
             active personas, capped at 200 by the `limit=200` on :7566 — so up to **201 sequential
             blocking round-trips per request**. Because they are on the loop and not in a thread,
             the cost is not just this request's latency: every concurrent request on the worker,
             including the `persona_stream` websockets, stalls for the duration. The
             `get_persona_voice_analytics` sibling is worse per query (six `COUNT/SUM/AVG` scans of
             `voice_command_events` over a caller-controlled `days` window, `ge=1, le=365`) though
             fewer of them.
tests:       Import-grep reachability, not measured coverage: `persona` 27 test files. Route-level
             coverage exists; no test asserts query count or non-blocking behaviour.
effort:      Cheap for the offload half (wrap in `_run_persona_db_call`, the idiom is already used
             106 times in this file). Moderate for the N+1 half — it needs a batched
             policy-rule loader on the ChaChaNotes owner, but
             `_load_persona_buddy_rows_for_projection` is the template and it is in the same file.
owner-only:  yes for the endpoint change; no for the batched loader if it lands in
             `core/DB_Management/`
confidence:  confirmed
```

### FINDING api-endpoints-5 — `get_conversation_citations` is the one conversation-scoped handler in chat.py that skips the shared ownership guard

```
axis:        correctness
class:       divergent-copies
severity:    Medium
sites:       chat.py:get_conversation_citations (8047-8077) — `db.get_conversation_by_id(...)` at
               :8055 with a bare not-None check at :8056
             chat.py:_verify_conversation_ownership (6351-6380) — the shared guard
             the 8 sibling handlers that DO call it: chat.py:6717, :7086, :7152, :7283, :7472,
               :7552, :7636, :8009
canonical:   chat.py:_verify_conversation_ownership (6351-6380)
destination: n/a — adoption
knowledge:   "What it takes to be allowed to read a conversation." The guard encodes four rules:
             not soft-deleted (:6358), client_id matches the caller (:6360-6366), scope_type matches
             (:6373), and workspace_id matches when the scope is `workspace` (:6375-6378). The
             citations handler encodes one of the four.
scenario:    Two, both confirmed by reading:
             (a) Soft-delete a conversation, then
                 `GET /api/v1/chat/conversations/{id}/citations`. `get_conversation_by_id` returns
                 the row (the guard's own `conversation.get("deleted")` check at :6358 exists
                 precisely because it does), the not-None check passes, and the endpoint returns
                 200 with the full RAG bibliography — every retrieved document across every
                 message. Every sibling handler returns 404 for the same id.
             (b) The handler declares no `scope_type`/`workspace_id` parameters at all, so it
                 always evaluates against the default global scope. A workspace-scoped
                 conversation's citations are readable from global scope, where
                 `GET /api/v1/chat/conversations/{id}` on the same id returns 404 via :6373.
             NOT claimed: this is not a cross-user read. `db` is
             `Depends(get_chacha_db_for_user)`, a per-user ChaChaNotes database, so another user's
             conversation is not in scope. The missing `client_id` check is a lost defence-in-depth
             layer, not an open door.
impact:      Medium. The repo's own cross-user isolation audit (2026-09-21) found that isolation
             leaks fail to converge because guards are applied per-handler rather than centrally;
             this is one handler out of nine forgetting the central guard, which is that pattern
             exactly. It is graded Medium rather than High because the per-user DB dependency
             contains the blast radius to one user's own soft-deleted and cross-scope data.
tests:       Import-grep reachability, not measured coverage: `chat` 78 test files. No endpoint
             test was found for `/conversations/{id}/citations`; the route is unexercised.
effort:      Cheap. Replace :8055-8060 with a `_verify_conversation_ownership(db, conversation_id,
             current_user, scope)` call and accept the scope params the siblings accept. Add one
             test asserting 404 after soft-delete.
owner-only:  yes
confidence:  confirmed (the missing guard and both divergent behaviours); confirmed (no test)
```

### FINDING api-endpoints-6 — unbounded keyword list turns one PATCH into ~200,000 synchronous SQLite calls on the event loop

```
axis:        efficiency
class:       n/a
severity:    Medium
sites:       chat.py:_replace_conversation_keywords (6405-6448), the per-keyword call sequence at
               :6437
             chat.py:update_chat_conversation (7196-7204) — calls it bare from an `async def`
               handler: no `await`, no `run_in_executor`, no `to_thread`
             chat.py:save_chat_knowledge (6817-6824) — same shape via `tags`
             api/v1/schemas/chat_conversation_schemas.py:76 — `keywords: list[str] | None` with no
               `max_length`; the `_normalize_keywords` validator only dedupes
canonical:   chat.py:_persist_with_transaction (3189-3213) + `current_loop.run_in_executor(...)`
             at :3216 — the correct pattern for sync DB work, in the same file
destination: n/a — the fix is a `max_length` on the schema field plus the existing executor idiom
knowledge:   "Request-body collections that drive per-item DB work must be bounded, and sync DB
             work must leave the loop." Both are already practised in this file; neither is applied
             on this path.
scenario:    n/a (efficiency axis)
cost-driver: Per keyword, per request, with no upper bound. The loop at :6437 issues up to four
             sequential SQLite calls per new keyword — `get_keyword_by_text`, `add_keyword`,
             `get_keyword_by_id`, `link_conversation_to_keyword`. With no `max_length` on the
             schema field, `PATCH /api/v1/chat/conversations/{id}` carrying 50,000 distinct
             keywords produces on the order of 200,000 synchronous SQLite calls inside one request,
             on the event loop thread. That is a single-request worker wedge: every other coroutine
             on that worker, streaming responses included, is blocked for the duration. Scales
             linearly and unboundedly with attacker-controlled input.
tests:       Import-grep reachability, not measured coverage: `chat` 78 test files. None bounds the
             keyword list or asserts non-blocking behaviour on this path.
effort:      Cheap. `max_length` on `chat_conversation_schemas.py:76` (owner-only: the schema is
             under api/v1/) and route the loop through the executor idiom already at
             `chat.py:3216`. A batched `link_conversation_to_keywords` on the ChaChaNotes owner
             would be the durable fix.
owner-only:  yes
confidence:  confirmed (the unbounded field, the per-item call sequence, the missing offload);
             probable-risk (the 50,000-keyword request reaching the handler depends on any upstream
             body-size limit, which was not traced)
```

### FINDING api-endpoints-11 — a swallowing timestamp parser makes the file-artifact export expiry gate fail open

```
axis:        correctness
class:       divergent-copies
severity:    Medium
sites:       files.py:_parse_iso_datetime (83-92) — `except Exception: return None`
             files.py:export_file_artifact (185-265) — the two gates it drives:
               `consumed_at` at :211-213 and `expires_at` at :221-234
             core/File_Artifacts/file_artifacts_service.py:_build_export_info_from_row (770-792) —
               the same two gates, driven by that class's own `_parse_iso_datetime`
             the value's writer: core/File_Artifacts/file_artifacts_service.py:378
               (`expires_at.replace(microsecond=0).isoformat()`)
             the column: core/DB_Management/Collections_DB.py:1270 and :1679
               (`export_expires_at TEXT`, on both the SQLite and the second DDL block)
canonical:   NONE usable — see api-endpoints-7
destination: `app/api/v1/utils/iso_datetime.py` (same destination as api-endpoints-7), with a parse
             that distinguishes "absent" from "unparseable" instead of collapsing both to None
knowledge:   "What an unparseable expiry timestamp means." Both copies answer "treat it as no
             expiry". The safe answer for an access gate is the opposite.
scenario:    `files.py:221` reads `expires_at = _parse_iso_datetime(row.export_expires_at)` and
             :223 gates on `if expires_at is not None and expires_at <= now`. Because :88's
             `except Exception` returns `None` for any value `datetime.fromisoformat` rejects, a
             row whose `export_expires_at` is not ISO-8601 — a legacy value, a hand-edited row, a
             value written by a future migration in `"YYYY-MM-DD HH:MM:SS UTC"` or epoch form into
             a column typed only `TEXT` — makes the expiry check evaluate to False and
             `GET /api/v1/files/{file_id}/export` serve the artifact indefinitely. The
             single-use gate at :211 has the same shape, but it is independently defended by
             `cdb.consume_file_artifact_export(...)` returning False at :249, so only the expiry
             gate actually fails open. The core copy at
             `file_artifacts_service.py:776` repeats the same pattern for the status projection, so
             the artifact also keeps reporting `status="ready"`.
impact:      Medium. Confirmed fail-open direction on an access gate; probable-risk on reachability,
             because the only writer in the tree (`file_artifacts_service.py:378`) emits
             `.isoformat()`, which always parses. The severity is driven by the direction of the
             failure, not by a demonstrated trigger.
tests:       Import-grep reachability, not measured coverage: `files` 5 test files.
             `tests/Files/test_files_endpoint.py:173` (past) and `:221` (future) cover the parseable
             cases; `:146` and `tests/Files/test_files_export_gc.py:59,72` cover None. No test
             supplies an unparseable value.
effort:      Cheap. Narrow the except to `ValueError` (which `binascii`/`fromisoformat` raise) and
             make the gate fail closed: treat an unparseable expiry as expired. One test with
             `export_expires_at="not-a-timestamp"` asserting 404. Note the core copy must move in
             the same change or the status projection and the download gate disagree.
owner-only:  yes for files.py; no for core/File_Artifacts/file_artifacts_service.py
confidence:  confirmed (the fail-open code path, both copies); probable-risk (reachability — no
             writer in the tree produces an unparseable value today)
```

### FINDING api-endpoints-15 — share-link HMAC key falls back to a literal published in this repo, and `lru_cache` pins whichever key the first call produced

```
axis:        correctness
class:       n/a
severity:    Medium
sites:       chat.py:_get_knowledge_qa_share_signing_key (6462-6471) — `@lru_cache(maxsize=1)`,
               with the fallback
               `(os.getenv("JWT_SECRET_KEY") or "knowledge_qa_share_link_default")` at :6470-6471
             chat.py:_build_knowledge_qa_share_token (6483-6493) — the minting path
             chat.py:create_conversation_share_link (~7463) — the route that mints
             chat.py:resolve_conversation_share_token (7670+) — the route that verifies; the
               DB-side `share_id` lookup at ~:7715-7720 is the control that actually holds
             core/AuthNZ/crypto_utils.py:derive_hmac_key — raises ValueError when no key material
               is configured
             chat.py:_CHAT_ENDPOINT_NONCRITICAL_EXCEPTIONS (351-371) — includes ValueError, so the
               fallback arm is reachable, not theoretical
canonical:   core/AuthNZ/crypto_utils.py:derive_hmac_key
destination: n/a — the fix is to let the ValueError propagate at startup rather than fall back
knowledge:   "Where the share-link signing key comes from, and what to do when there isn't one."
             The answer today is "use a constant that is in the public repository", which is not a
             key.
scenario:    Two:
             (a) With neither `KNOWLEDGE_QA_SHARE_LINK_SECRET` nor `JWT_SECRET_KEY` set at the time
                 of the first call, `derive_hmac_key()` raises ValueError, the except arm is taken,
                 and every token minted by `create_conversation_share_link` is signed with the
                 literal `"knowledge_qa_share_link_default"` — a value anyone can read in this
                 file. Anyone can then forge a well-signed share token. The DB-side `share_id`
                 lookup at :7715-7720 is what stops it becoming a read of arbitrary conversations,
                 so this is a lost defence layer rather than an open door.
             (b) `@lru_cache(maxsize=1)` pins the first result for the process lifetime. If the
                 first call lands before AuthNZ settings load and takes the fallback, and a later
                 restart derives the real key, every share link issued by the first process returns
                 `403 Invalid share token` with no diagnostic — a silent mass invalidation whose
                 cause is a startup-ordering race.
impact:      Medium. A hardcoded signing-key fallback in a public repository is the kind of finding
             that must clear the Bandit HIGH/CRITICAL gate (`Docs/ADR/005-bandit-touched-scope-security-gate.md`)
             if the line is ever touched. It is not graded High only because the DB-side share_id
             lookup independently constrains what a forged token can reach.
tests:       Import-grep reachability, not measured coverage: `chat` 78 test files, including
             `tests/Chat/unit/test_chat_share_links_api.py`. No test asserts that the fallback arm
             is not taken, and none asserts the derived-key path.
effort:      Cheap to make it fail closed (drop the literal; let the ValueError surface at startup
             the way a missing JWT secret already does). Moderate if a migration path for links
             already minted under the fallback is required — which is itself a reason to fix it
             now rather than later.
owner-only:  yes
confidence:  confirmed (the literal, the lru_cache, and that the except arm is reachable because
             ValueError is in the noncritical tuple at :351-371); probable-risk (that a real
             deployment hits it — that depends on startup ordering, which was not traced)
```

## Findings considered and DROPPED

Recorded so the next reviewer does not re-derive them.

- **`chat.py:3210` `time.sleep(0.1 * (2 ** retries))` on a retry loop.** Not a blocking-loop bug:
  the enclosing `_persist_with_transaction` (3189-3213) is a sync function dispatched via
  `current_loop.run_in_executor(None, _persist_with_transaction)` at :3216. Correct as written.
- **`admin/admin_rate_limits.py:176,193,227` commits only on the SQLite branch.** Not a missing
  Postgres commit: the dependency is `get_db_transaction` (`core/AuthNZ/database.py:1908-1912`),
  which wraps the request in `pool.transaction()` and commits on exit. The explicit commit is
  redundant, not absent.
- **`workflows.py:2195`'s `+ b"=="` padding.** Differs in idiom from `:2597` but round-trips
  identically for every tested cursor length; CPython tolerates excess padding. Recorded as an
  idiom divergence under api-endpoints-10, not as a defect. (Correction to briefing cluster C1.)
- **`files.py:83` "fails on Z-suffixed input".** False on the supported Python floor
  (`pyproject.toml:15`, `>=3.11`); verified on 3.12.11. (Correction to briefing cluster C3.)
- **Negative `token_budget` in `character_chat_sessions.py:_truncate_to_budget (5651)`.** The
  missing `max(1, ...)` clamp would make `text[:budget*4]` slice from the end for a negative
  budget, but all budgets are the positive module constants `_TOKEN_BUDGET_*` at :5457-5463.
  Recorded as latent under api-endpoints-8; not a finding.
- **Mutable default arguments in endpoints.** Swept; none found outside FastAPI's
  `Query`/`Body`/`Field`/`Depends` sentinels, which are the documented pattern
  (`pyproject.toml` also globally ignores B008).
- **Blind `except` in the hot files.** `chat.py` and `persona.py` are both on the BLE001
  grandfather list (`pyproject.toml:801`, `:837`). Reported only where a specific swallowed error
  produces a concrete failure — api-endpoints-11 is the one case that cleared that bar.
- **Discord's OAuth callback trusting the `guild_id` query parameter** (`discord.py:409,415`).
  Discord documents `guild_id` on the install redirect, so the fallback is protocol-correct; the
  `state` record is validated first. Recorded in the stage-2 divergence table as
  justified-divergence with a note, not as a security finding.
- **Auth-dependency duplication across endpoints.** Owned by `auth-dependencies/`, which explicitly
  states duplicate checks are often intentional defence in depth. Not re-reported.

## Synthesis

The endpoints layer's problems are not distributed evenly across 248k LOC; they cluster in three
shapes, and all three have an in-repo fix already demonstrated somewhere else in the same layer.

**Shape 1 — the rule exists, the code does not follow it.** `Architecture.md:180` forbids raw SQL in
endpoints; 20 files contain it and 17 of those have a named core owner (stage 1, api-endpoints-2).
`persona.py` defines `_run_persona_db_call`, uses it 106 times, and skips it in 26 handlers
(api-endpoints-4). `chat.py` defines `_verify_conversation_ownership` and 8 of 9 handlers call it
(api-endpoints-5). `character_chat_sessions.py` imports the persona assembler and then re-implements
its truncator (api-endpoints-8). None of these needs a new abstraction; every one needs adoption of
an abstraction that already shipped. The proportionate control is the ratchet test pattern the repo
already runs at `tests/lint/test_endpoint_auth_deps_import_boundary.py`.

**Shape 2 — the same job done N ways, with the variance hiding the bug.** Four error-to-HTTP
conventions, one of which drops a real error code (api-endpoints-9). Four `_parse_iso_datetime`
copies, one of which fails open on an access gate (api-endpoints-11). Eight base64 cursor decoders
with six failure contracts, one of which returns 500 on a public route (api-endpoints-1). In each
cluster the correct implementation already exists in this layer — `map_db_error_to_http`,
`notes.py:_decode_attachment_cursor`, `mcp_unified_endpoint.py:_parse_safe_config_query` — and the
useful output of the enumeration was not the count but the outlier it isolated.

**Shape 3 — whole-module cloning, where the duplication cost is already being paid.** C9 is the
clearest case in the repo: three source pairs at 61-91% identity, four test pairs at 29-100%, and an
uncommitted security fix that had to be written four times with byte-identical comments
(api-endpoints-3). The divergent remainder has already produced two vocabularies for one policy
document. This is the module's single highest-value item and no other reviewer in this audit covers
it.

The cross-cutting risk the briefing named — SQLite/PostgreSQL divergence tested on one backend only
— is present here in concrete form: 10 endpoint files hand-roll the dialect branch, and the one test
that covers both branches asserts SQL strings against a stub (api-endpoints-12). That is the cheapest
gap to close, because the Postgres fixture already exists at
`tldw_Server_API/tests/AuthNZ/conftest.py`.

## Suggested Refactor/Actions

Ordered by value per unit of effort. Items 1-4 are one-liners or near-enough and should not wait for
any design document.

1. **`chat.py:6507`** — move the signature decode inside the try. One line. Add a `"AAAA.A"`
   regression test asserting 400/403. (api-endpoints-1)
2. **`persona.py:2469-2482`** — map `starter_copy_failed` to 409. One line. (api-endpoints-9)
3. **`chat.py:8055`** — call `_verify_conversation_ownership` and accept the scope params the 8
   sibling handlers accept. Add a soft-delete 404 test. (api-endpoints-5)
4. **`files.py:88`** — narrow `except Exception` to `ValueError` and make an unparseable expiry
   fail closed; mirror in `core/File_Artifacts/file_artifacts_service.py`. (api-endpoints-11)
5. **`persona.py:7566-7573`** — wrap both calls in `_run_persona_db_call` and add a batched
   policy-rule loader modelled on `_load_persona_buddy_rows_for_projection`. (api-endpoints-4)
6. **`chat_conversation_schemas.py:76`** — add `max_length`; route
   `_replace_conversation_keywords` through the executor idiom at `chat.py:3216`.
   (api-endpoints-6)
7. **`chat.py:6470`** — remove the hardcoded signing-key fallback; let the ValueError surface at
   startup. Must clear Bandit HIGH/CRITICAL per ADR-005 when that line is touched.
   (api-endpoints-15)
8. **Seed a raw-SQL AST ratchet** at `tests/lint/`, at the current 20 files. Not owner-only.
   (api-endpoints-2)
9. **Add real Postgres coverage for `admin/admin_rate_limits.py`** under `tests/AuthNZ_Postgres/`
   using the existing `isolated_test_environment` fixture. Not owner-only. (api-endpoints-12)
10. **Design-first work**, in priority order — each needs
    `Docs/Design/YYYY-MM-DD-<slug>-design.md`, an ADR entry, a Backlog task linking both, and
    `IMPLEMENTATION_PLAN_<slug>.md` with 3-5 stages. Backlog tasks are proposed here only; this
    audit is read-only and never hand-edits a task file.
    - `chatops-shell` — the Discord/Slack extraction, stage 1 = the 91.2% `*_oauth_admin.py` pair.
      (api-endpoints-3)
    - `endpoint-raw-sql-adoption` — `admin_rbac.py` (69) and `jobs_admin.py` (56), both of which
      reach past a core module's private API. (api-endpoints-2)
    - `persona-visual-packs-extraction` — the first god-module slice, following the shipped
      `core/DB_Management/media_db/` package split. (api-endpoints-14)
    - `api-iso-datetime` and `api-opaque-cursor` — the two new cohesive utility modules, explicitly
      NOT additions to `Utils.py` or `http_client.py`. (api-endpoints-7, -10, -16)

Base branch assumed for any of this: `dev`, per `CONTRIBUTING.md:86,121`. `origin/HEAD` resolves to
`main`; both exist.

## Not Covered

248k LOC across 269 files guarantees gaps. Honestly:

- **Roughly 200 of the 269 files were not opened.** The reading list was size x churn plus the
  mechanical rule sweeps. Files below the top ~25 by churn were touched only by grep. Notable
  unexamined bulk: `embeddings_v5_production_enhanced.py` (6,792 LOC / 127 commits — third-hottest
  in the module, read only for its error mappers and its `time.sleep`), `audio/audio_streaming.py`
  (4,642 / 58), `mcp_hub_management.py` (4,462 / 50), `auth.py` (3,970 / 91 — read only for its 12
  `utcnow()` sites), `agent_client_protocol.py` (3,958 / 61), `setup.py` (2,885 / 88),
  `sandbox.py` (2,725 / 86), `flashcards.py` (2,882 / 79), `rag_unified.py` (2,525 / 95),
  `slides.py`, `writing_manuscripts.py`, `vn_assets.py`, all of `evaluations/`, most of `media/`.
- **`watchlists.py` (8,739 LOC / 103 commits) was swept, not read.** It appears here only for its
  12 page-offset sites and one raw-SQL statement. It is inventoried but unaudited by
  `api-pagination/` too, and it tops `auth-dependencies/`'s legacy-user-dependency table at 64
  signals. It is the largest genuinely unreviewed surface in this module.
- **No test was executed.** Every `tests:` line is import-grep reachability. No coverage
  measurement, no `pytest --collect-only`, no reproduction of any scenario against a running app.
  The two empirically verified claims (the `binascii.Error`, the `Z` parse) were verified against
  the interpreter, not the application.
- **Axis 3 (sequential coupling) produced nothing that cleared its drop rule.** The multi-step
  flows examined — export consume/clear in `files.py`, the OAuth install state machine, the
  persona preview assembly — either enforce ordering through a context manager or make the invalid
  order unexpressible. Rather than pad, the axis is reported empty.
- **Efficiency was audited only in `chat.py`, `persona.py`, and the cursor/pagination paths.** No
  systematic N+1 sweep across the other 266 files, no query-plan analysis, no check for missing
  `LIMIT` behind paginated responses outside the files named above.
- **WebSocket and streaming handlers were not reviewed.** `audio/audio_streaming.py`,
  `persona.py:persona_stream (7903-11124)`, `prompt_studio/prompt_studio_websocket.py`, and the SSE
  paths in `chat.py` are untouched by this audit.
- **Resource Governor coverage per route was not verified.** ADR-018 requires governed routes to own
  a policy-store entry and a `route_map` entry; checking that per route across 2,188 route
  decorators was out of budget, and ADR-018 explicitly does not claim universal coverage for
  historical routes, so no finding is asserted either way.
- **The frontend/API contract was not checked.** Whether any client actually reads the `truncated`
  flag (api-endpoints-8) or parses `detail.code` (api-endpoints-9) is marked as an assumption in
  both findings and was not traced into `apps/`.
