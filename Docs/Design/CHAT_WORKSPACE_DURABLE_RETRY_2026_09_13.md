# Chat Workspace Durable Retry

Status: Initial contract approved on 2026-09-13. The user approved the
insertion-order migration and legacy-anchor rejection on 2026-09-29, requesting
review before implementation continues. Retry acceptance remains blocked.
Tracking: TASK-13255.8.1; provider routing prerequisite TASK-13255.16;
insertion-order migration proposal TASK-13255.8.1.1.

## Contract

Opted-in completion requests carry `tldw_turn: { user_message_id: <UUID> }`.
The client allocates the ID for a new composer submission, before retrieval,
and retains it for Retry and Switch model. Identical text submitted as a new
turn receives a different ID. Existing callers without this field are unchanged.

The extension requires an existing authorized conversation and `save_to_db=true`.
It contains one current user message plus optional request-local system context;
client history is not replayed into persistence. The server inserts that user
atomically or validates an existing matching row. Reuse with a different
conversation, role, canonical content, deleted row, or later user turn fails
without modifying historical data. Authorization precedes identity lookup.

History for generation contains conversation history through the identified
user exactly once, including legacy rows without parent links. Prior partial
assistant attempts after that user are excluded from generation context but
preserved in storage. The new assistant is linked to the same user. Do not
hold database transactions across model inference.

The user receipt is independent of any assistant-persistence receipt. Losing
the response after user commit must still permit safe retry with the original
client ID. RAG preflight failure does not imply that a completion user was saved.

This contract guarantees one durable user row, not exactly-once inference or
server-wide single-flight completion execution. It must not enable global
content-only overlap trimming or change generic message insertion semantics.

Enabled slash commands and macros are unsupported with `tldw_turn` and must
fail before execution. Their output cannot replace the durable canonical user
text, and macro short-circuits cannot bypass the persistence/authorization
contract. Disabled commands remain ordinary literal text. Final tool
auto-continuation answers retain the original user parent.

## Provider Identity

Normalize catalog alias `llama` to configured provider `llama.cpp`, including
qualified selections. Preserve the full local model identifier. Explicit local
selection must not silently fall through to a different provider when metadata
is missing. Reuse the shared resolver; do not add a second alias table in the UI.

## Validation

- SQLite and PostgreSQL atomic insert-or-validate, with concurrent same-ID calls.
- Duplicate retry, same-text new IDs, mismatched/deleted/foreign IDs, later turns.
- Failure after commit and before acknowledgement; empty and partial failures.
- Legacy history, unchanged callers, workspace and assistant authorization.
- Frontend original ID propagation through model switches and failed retrieval.
- Real browser, isolated FastAPI/database, and actual local Gemma UAT only.
- No mocks, authorization bypasses, readiness bypasses, or historical deletion in UAT.
- No commits or GitHub issue closures during this work.

## Review Blocker: Insertion Order

Independent review demonstrated that `(timestamp, id)` does not establish a
durable retry boundary. A future-dated imported message can push a new durable
user beyond subsequent ordinary messages, and an equal-time later message can
sort before its anchor by UUID. This can replay a partial assistant attempt or
miss a later-user conflict. The current implementation is not acceptance-ready.

No existing cross-backend insertion-order field was found. Conversation history
versions do not belong to individual messages, SQLite sync-create triggers do
not provide equivalent PostgreSQL evidence, and general message metadata can be
replaced by existing update paths. Do not substitute these as ordering guesses.

Approved extension: persist immutable message insertion order
transactionally across ordinary, durable, and import/sync insertion paths. Keep
existing IDs, content, timestamps, and generic duplicate behavior unchanged.
Preserve all historical records and reject legacy retry anchors whose ordering
cannot be verified. A migration cannot reconstruct missing historical insertion
order. Existing records must retain an explicit legacy distinction; the approved
policy rejects legacy retry anchors whose insertion order cannot be verified.
For PostgreSQL, order allocation must follow conversation-lock acquisition for
every writer. A sequence default alone does not establish commit order. Ordinary
chat, Sync v2, and chatbook imports share the existing message insertion path.

## Captured Request Scope

Durable generic normal/persona turns acquire the existing request-scope lease
before history or conversation-binding awaits, even without web search. The
normal-mode wrapper also acquires it for direct callers. The model factory
requires that captured scope. Its capability gate
queries only GET `/openapi.json` through the same target/principal guards used
for completion, instead of trusting the mutable general capability cache.
The generic UI capability cache remains unchanged.
