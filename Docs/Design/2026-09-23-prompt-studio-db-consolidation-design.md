# Prompt Studio database consolidation

**Status:** Implemented · **Task:** TASK-13318 · **ADR:** [ADR-051](../ADR/051-prompt-studio-db-single-implementation.md) · **Review finding:** F20

## Problem

`core/DB_Management/PromptStudioDatabase.py` is 7,426 lines holding the same domain twice:

| Class | Lines | Used when |
|---|---|---|
| `_BackendPromptStudioDatabase` | 722–3843 (3,121) | a `DatabaseBackend` of type PostgreSQL is supplied |
| `_SQLitePromptStudioDatabase` | 3848–7141 (3,293) | everything else |
| `PromptStudioDatabase` (facade) | 7144–7426 | always; forwards `*args/**kwargs` plus `__getattr__` |

They share 60 methods over parallel SQL. The facade's `*args/**kwargs` forwarding erases every signature from mypy and IDEs, so the two drift silently. Measured drift so far:

- one missing method (`list_optimizations`, TASK-13290, fixed);
- **seven signature mismatches**, four on public methods (`get_prompt(include_deleted=)`, `create_bulk_test_cases(client_id=)`, positional `delete_signature` `hard_delete`, positional `list_evaluations` filters) — latent today, a `TypeError` on one backend for the first caller that uses them;
- `_format_test_case` differs in **arity**, `(row)` vs `(cursor, row)`, so no shared helper can call it on both;
- a busy-retry loop on the SQLite side only (around line 4129).

A signature-parity ratchet now guards against further drift: `tests/DB_Management/test_prompt_studio_backend_parity.py` (TASK-13318 AC2).

## Observations that shape the options

1. **The PostgreSQL class is already partly backend-neutral.** It branches on `self.backend_type` in several places (`ILIKE`/`LIKE` at 945, `deleted = FALSE`/`= 0` at 2731, boolean vs int binding at 2739/2781). It was written to run on either backend and is simply never constructed for SQLite.
2. **`DatabaseBackend` has a SQLite implementation.** The facade could hand the PostgreSQL class a SQLite `DatabaseBackend` rather than instantiating a separate class.
3. **The SQL it uses is largely portable.** It uses `RETURNING` and `ON CONFLICT`, both supported by SQLite ≥ 3.35. The PostgreSQL-flavoured constructs number about 34 and need an audit, not a rewrite.
4. **The domain divides into nine aggregates** by public method: jobs (11), test cases (10), optimizations (8), projects (6), signatures (6), prompts (5), evaluations (4), prompt versions (2), test runs (1).
5. **Precedent exists.** `media_db/` split `Media_DB_v2.py` into `repositories/` (one per aggregate, each over a session), `runtime/` (operations), `schema/`, and a thin `api.py`.

## Options

### A — Promote the backend class to the sole implementation, then split

Run `_BackendPromptStudioDatabase` on a SQLite `DatabaseBackend` too, delete `_SQLitePromptStudioDatabase`, then split the survivor into per-aggregate modules.

- **For:** fastest route to "business logic exists once" (AC3); the dialect branches already exist; the split then operates on one class, not two.
- **Against:** everything the SQLite class does that the backend class does not — the busy-retry loop, schema migrations (`_apply_prompt_studio_migrations`), `transaction()`, FTS setup — must be ported first, and SQLite users switch implementation in one step.
- **Risk:** highest single-step behavioural change for the default (SQLite) deployment.

### B — Split both into per-aggregate repositories, then merge per aggregate

Follow `media_db` directly: create `prompt_studio_db/repositories/<aggregate>.py`, and for each aggregate write one backend-neutral repository over `DatabaseBackend`, retiring that aggregate's methods from both classes.

- **For:** small, independently shippable steps; each aggregate's parity can be tested against both old implementations before they are deleted; matches the shipped precedent.
- **Against:** longer; both old classes stay alive until the last aggregate moves.

### C — Leave the two classes; fix drift only

Align the seven mismatches and rely on the parity ratchet.

- **For:** cheapest.
- **Against:** business logic stays duplicated (fails AC3); the ratchet only catches *signature* drift, not behavioural drift such as the one-sided retry.

## Recommendation

**Option B, starting with the smallest aggregates.** It is the only option where every step is independently reversible and verifiable against both existing implementations, and it matches the precedent the codebase already accepted. Option A's single switch of the default SQLite deployment is the riskiest change available here.

## Staged plan (Option B)

| Stage | Scope | Gate |
|---|---|---|
| 0 | Parity ratchet | **done** — `test_prompt_studio_backend_parity.py` |
| 1 | Package skeleton — **folded into Stage 3**: the package is created with its first repository rather than empty. The session is the legacy object itself (as in `media_db`): both classes already expose `transaction()`, `get_connection()`, `_cursor_exec(conn, sql, params)` with `?` placeholders, `_row_to_dict`, `_log_sync_event`, `client_id`, `backend_type` | no behaviour change; existing suites pass |
| 2 | Behavioural parity harness — **done**: `tests/prompt_studio/test_backend_behaviour_parity.py`, one scenario per aggregate on SQLite and live PostgreSQL. Found PostgreSQL leaking its `*_tsv` columns from 8 read paths; fixed in `_row_to_dict` | exists before any aggregate moves |
| 3 | Move the smallest aggregates first: test runs, prompt versions, evaluations — **done**: `prompt_studio_db/repositories/{test_runs,prompt_versions,evaluations}.py`, each write retried through `core/DB_Management/retry_policy.py` on both backends. `update_evaluation` now allowlists the columns it interpolates | parity harness green on both backends |
| 4 | Signatures, projects, prompts — **done**. Found: PostgreSQL uniqueness conflicts surfacing as `DatabaseError` (the cursor wrapper matched a now-redacted message), `TypeError` on missing rows, stub ids not advancing the sequence | as above |
| 5 | Test cases — **done**, arity resolved. Found: PostgreSQL rejecting every `is_golden`/`is_generated` update | as above |
| 6 | Optimizations and jobs — **done**. Found: SQLite `transaction()` override never issued `BEGIN` (reads outside the transaction, nested calls committing early); SQLite iteration rows returned as raw JSON. Sync-log and idempotency helpers now live once in `prompt_studio_db/session.py` | as above; job lease tests on both |
| 7 | Typed façade — **done**: every facade method that delegates to a repository has that method's exact signature (generated from the repository source, pinned by `test_facade_delegations_are_typed_like_their_repository`); only `update_project` keeps `**fields`, by design. The two classes remain as backend infrastructure (connections, schema, execution, row decoding) | mypy sees real signatures |

Each stage removes that aggregate's entries from `KNOWN_SIGNATURE_DRIFT` in the parity test as they are resolved.

## Decisions (2026-09-23)

1. Busy-retry: the unified implementation retries on both backends when the database reports busy (Stage 6).
2. The four latent public mismatches were aligned ahead of the refactor (commit bcd56dc9bc); `KNOWN_SIGNATURE_DRIFT` now holds only the three private entries.
3. PostgreSQL coverage: CI's `product-prompt-studio` shard runs in `full-suite-linux-312-shards`, which has a PostgreSQL service, so the Stage 2 harness runs against a real server there.
