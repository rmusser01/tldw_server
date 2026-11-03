# Persona session transcript durability — 2026-11-02

Task: TASK-13260.278.18.83.45.
Plan: ../Plans/IMPLEMENTATION_PLAN_uat_all_fixes_20261006.md#s06-specialized-tools-and-discovery-selected-closure--2026-11-02.
ADR check: required, ../ADR/085-bounded-durable-persona-session-transcripts.md. This changes a lasting Persona persistence rule; ADR-046's Chat adapter and ADR-080's checkout ownership remain in force.

## Failure and intended behavior

A real S06 answer was visible and returned as two correlated turns through browser reload. The next private API process returned the same session with zero turns. Its check-in and session metadata remained. The current process-local transcript and telemetry-only personalization writes cannot provide restart recovery.

Future acknowledged turns must keep their exact IDs, order, timestamps, content and client-message correlation through API restart, detail, list counts, export and explicit resume. A fresh process must not need a new conversation or provider call to read saved history.

## Smallest storage boundary

Use the existing CharactersRAGDB PersonaStateStore seam, with a dedicated transcript table and normal SQLite v84→85 / PostgreSQL v89→90 migration. Do not put content in preferences, policy/scope snapshots, ordinary Chat history or telemetry.

One snapshot row per session holds the bounded ordered turn array, a stored count, a monotonic revision and server-derived last activity. Serialize writes in owner→session order: the existing PostgreSQL transaction-advisory-lock pattern supplies owner serialization; SQLite's existing BEGIN IMMEDIATE serializes writers. The owner lock also protects the cross-session retention cap. Read list counts without loading JSON transcript blobs.

An append validates the owned active session and bound Persona in that transaction, appends in server commit order, accepts an exact repeated turn ID idempotently before renewing timestamps, rejects conflicting reuse, and prunes by the existing runtime retention policy. Preserve all other session settings and version/transaction semantics. The stored turn is the actual prepared bounded runtime turn, never separately regenerated text or identity.

Every prune clears only transcript payload and advances its retained header revision. Return invalidations for pruned runtime sessions; revision-guarded publication rejects older snapshots. A read of expired history returns an authoritative empty/retired snapshot without updating storage or renewing TTL, and that revision cannot later accept a delayed nonempty publication. A subsequent new turn uses a newer revision. Durable-header presence suppresses legacy in-memory fallback even after eviction.

Bind an ordinary provider reply to its actual saved user-turn ID. Inside the same append transaction, refuse the reply if that originating turn has expired or been evicted. A held provider cannot create an orphaned assistant answer by obtaining a newer revision after retirement. Runtime publication updates only an existing owned session; it never recreates an evicted session. Explicit Connect creates the runtime container before loading the current durable revision and restores no pending authority.

The serialized maximum must account for worst-case JSON escaping and UTF-8, rather than the raw text character count. Validate the canonical fields and actual encoded byte length, and test worst-case bounded control-character and Unicode content. The 200-turn bound permits a larger blob than2MB; expose no per-user or session sizing knob.

Share an owned durable read among list counts, detail, export and cold materialization. Hydrate only an absent runtime session and only its retained turns; never restore pending plans, approvals or voice/tool authority. Preserve supported session modes and normal explicit Connect.

## Publication and failures

Prepare a bounded turn without exposing it as a completed runtime result, persist it through the existing owned ChaCha operation, then publish that exact turn to runtime. Join owned storage work on cancellation and retain the current-turn/terminal-session boundary. A failed store must surface an actionable error and must not emit a completed assistant delta. The provider's existing admission/accounting and cancellation behavior stay unchanged.

Retain the default 200-turn / 8192-content / 4096-metadata-payload bounds, session TTL and per-user session limit. Storage and runtime must agree on expiry and ordering. Retention or deletion never grants access to another owner's rows. Use existing abstractions, parameterized SQL only within DB_Management, official isolated PostgreSQL fixtures and restricted runtime roles.

## Verification sequence

1. Add a real WebSocket-producer regression with controlled inference, capture exact original turns, replace the manager, then require the same detail/list/export and explicit-resume transcript. Establish red before changing production.
2. Add database/store controls for foreign ownership and Persona, terminal/deleted sessions, duplicate ID versus conflicting payload, concurrent append ordering, bounded retention, schema upgrade and rollback, and injected storage failure.
3. Implement the minimum shared storage and hydration path. Run genuinely affected Persona/DB/migration suites on SQLite and official isolated PostgreSQL. Keep existing cancellation and no-approval-replay tests.
4. Complete scoped baseline/current lint/security, review every changed source/test hunk and migrations, normal hooks/local commit, and an isolated candidate wheel refresh with existing tools.
5. Validate a new real UAT turn, restart the private API, read and resume its exact saved transcript without regeneration. Preserve originals and earlier failed records. Finish Rowan guide and applicable mode cohorts before closing S06.

## Explicit limits

The existing unpersisted WebSocket scaffold remains temporary. Only a fresh successful global parent lookup inside the owned transaction can select that internal compatibility path; foreign, deleted, terminal, inactive or mismatched parents and database failures are rejected. A session known to be saved, or a plan requiring saved-session authority, never downgrades after its row disappears. No wire-level switch, provider conversation, automatic profile/policy creation or durable-history claim is added to the scaffold. This preserves an existing caller outside ADR085's owned persisted-session storage boundary.

The earlier private Companion answer has already been lost from runtime and was not stored as exact transcript text. Preserve the failed restart outcome; do not copy content from this plan, test journals or screenshots into canonical application state. The original user's API/frontend/data remain outside this repair's runtime actions. No GitHub publication, queue or merge acceptance is implied.

The metadata payload keeps its existing 4096-character limit plus at most 512 characters of retention diagnostics. This preserves the existing small internal-cap truncation markers; storage validates the complete serialized metadata and the full escaped snapshot size.

Expired reads mark the actual stored revision retired; they never allocate the next committed revision. Explicit resume renews only nonexpired history under the owner/session lock, advances the committed revision without changing any turn, and refreshes runtime activity after success. Ordinary GETs remain read-only.
