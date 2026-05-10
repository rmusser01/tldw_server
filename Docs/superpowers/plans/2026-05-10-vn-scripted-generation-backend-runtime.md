# VN Scripted Generation Backend Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement backend-owned scripted-story model generation for VN Play sessions from `Docs/superpowers/specs/2026-05-10-vn-scripted-model-generation-design.md`, including durable generation requests/revisions, strict output parsing, pinned profile snapshot resolution, idempotent API commands, and replay-safe active revision overlays.

**Architecture:** Extend the existing VN Play runtime rather than creating a parallel runtime. VN Scripts publish immutable profile snapshot maps; VN Play owns session-scoped generation points and revisions in the per-user ChaChaNotes DB; model output flows through a small provider/moderation adapter seam, strict parsers, existing visual resolver logic, immutable events, and read-time active-revision scene overlays.

**Tech Stack:** FastAPI, Pydantic, SQLite via `CharactersRAGDB`/DB management repositories, existing VN Play/VN Scripts/VN Policy services, existing chat provider adapter seam in `VN_Play/adapters.py`, pytest.

---

## Existing Backend Map

- Script runtime endpoints live in `tldw_Server_API/app/api/v1/endpoints/vn_play.py`.
- Script runtime schemas live in `tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py`.
- Script runtime service flow lives in `tldw_Server_API/app/core/VN_Play/service.py`.
- VN Play DB schema/repository methods live in `tldw_Server_API/app/core/DB_Management/VNPlay_DB.py`.
- Existing scene derivation lives in `tldw_Server_API/app/core/VN_Play/state.py`.
- Existing visual asset resolution lives in `tldw_Server_API/app/core/VN_Play/assets.py`.
- Existing VN Play provider adapter seam lives in `tldw_Server_API/app/core/VN_Play/adapters.py`.
- Script authoring/publish service lives in `tldw_Server_API/app/core/VN_Scripts/service.py`.
- Script validator lives in `tldw_Server_API/app/core/VN_Scripts/validator.py`.
- Script publish DB logic lives in `tldw_Server_API/app/core/DB_Management/VNScripts_DB.py`.
- Policy/generation profile service and snapshots live in `tldw_Server_API/app/core/VN_Policy/service.py` and `tldw_Server_API/app/core/DB_Management/VNPolicy_DB.py`.
- Pagination helpers live in `tldw_Server_API/app/api/v1/schemas/pagination.py`.

Backend tests to extend first:

- `tldw_Server_API/tests/VN_Play/test_vn_play_db.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_api.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_action_requests.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_turns.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_state.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_save_slots.py`
- `tldw_Server_API/tests/VN_Scripts/test_vn_script_validator.py`
- `tldw_Server_API/tests/VN_Scripts/test_vn_script_publish_snapshots.py`
- `tldw_Server_API/tests/VN_Platform/test_vn_capabilities_api.py`

## Out Of Scope

- No WebUI inspector implementation.
- No frontend routing, React components, browser QA, or client-side generation setup rules.
- No streaming generation transport.
- No realtime image generation.
- No manual editing of generated revisions.

---

## Task 1: Publish-Time Generation Profile Snapshot Map

**Goal:** Published script versions can pin multiple generation-profile snapshots and validate `generate.profile_key` against that immutable map.

**Files:**

- `tldw_Server_API/app/core/DB_Management/VNScripts_DB.py`
- `tldw_Server_API/app/core/VN_Scripts/service.py`
- `tldw_Server_API/app/core/VN_Scripts/validator.py`
- `tldw_Server_API/app/api/v1/endpoints/vn_scripts.py`
- `tldw_Server_API/app/api/v1/schemas/vn_script_schemas.py`
- `tldw_Server_API/tests/VN_Scripts/test_vn_script_validator.py`
- `tldw_Server_API/tests/VN_Scripts/test_vn_script_publish_snapshots.py`
- `tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py`

**Steps:**

- [ ] Define the authored profile-map source as script metadata, not runtime request data: add `generation_profiles`/`generation_profile_ids` map support to script create/update schemas while keeping the existing `generation_profile_id` as the `default` fallback.
- [ ] Add a backwards-compatible script-version field for `generation_profile_snapshots`, preserving the existing `generation_profile_snapshot_id` as the default profile snapshot.
- [ ] Extend publish input/service internals to build a map with required `default` plus additional authored profile keys from the script metadata.
- [ ] Include the profile map and resolved profile IDs in publish idempotency payload hashes so a reused idempotency key cannot publish with a different generation profile set.
- [ ] Update script API responses to expose the authored profile map and published-version snapshot map without exposing provider secrets.
- [ ] Persist all referenced snapshot rows in `vn_profile_snapshots` during publish and store the map on the script version.
- [ ] Extend `VNScriptValidationContext` with available generation profiles keyed by `profile_key`.
- [ ] Update `_validate_generation` to reject unknown `profile_key`, invalid key syntax, unsupported `output_schema`, missing `default` when omitted, `requires_user_confirm` shape errors, `on_cancel` target errors, `on_generated_choice` target errors, raw routing keys, and profile policy incompatibilities described by the spec.
- [ ] Keep literal generation valid for existing scripts when `narrative_text` or `regeneration_text` is present.

**Tests:**

- [ ] Validator rejects unknown/invalid profile keys.
- [ ] Validator rejects raw provider/model/API routing keys.
- [ ] Validator rejects `choice_set` without `on_generated_choice`.
- [ ] Validator rejects unsupported output schemas for a profile.
- [ ] Publish stores and returns default plus additional snapshot IDs.
- [ ] Publish idempotency rejects the same key when the authored profile map changed.
- [ ] Script create/update/list/detail API round-trips the authored profile map.
- [ ] Existing single-profile publish tests continue passing.

**Commit message:** `Add VN script generation profile snapshot maps`

---

## Task 2: VN Play Generation Repository Layer

**Goal:** Add durable generation point, request, action, and revision storage with idempotent command support.

**Files:**

- `tldw_Server_API/app/core/DB_Management/VNPlay_DB.py`
- `tldw_Server_API/app/core/VN_Play/models.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_db.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_action_requests.py`

**Steps:**

- [ ] Add schema initialization for `vn_play_generations`, `vn_play_generation_requests`, `vn_play_generation_actions`, and `vn_play_generation_revisions`.
- [ ] Add indexes and constraints for session ownership, generation point uniqueness, request lookup, action idempotency, and revision history pagination.
- [ ] Store JSON fields with the existing `_json_dump`/decode patterns and owner checks used by session actions.
- [ ] Implement repository helpers to create/get generation points by `(owner_user_id, session_id, generation_point_key)`.
- [ ] Implement idempotent generation action helpers mirroring `vn_play_session_actions` semantics for confirm, cancel, regenerate, and activate.
- [ ] Implement request/revision lifecycle helpers for statuses in the spec, including public error code and debug-only metadata.
- [ ] Implement active revision pointer update helpers that verify the revision belongs to the same generation and has `succeeded` status.

**Tests:**

- [ ] Schema initializes on fresh and upgraded DBs.
- [ ] Generation point uniqueness is enforced per session.
- [ ] Idempotency replay returns stored responses and conflicting payloads fail.
- [ ] Active revision cannot point to failed, blocked, or foreign revisions.
- [ ] Revision list returns stable offset pagination order.

**Commit message:** `Add VN Play generation persistence`

---

## Task 3: Strict Generation Output Parser And Adapter Seam

**Goal:** Parse model output into bounded, safe, schema-specific payloads before any revision can become active.

**Files:**

- `tldw_Server_API/app/core/VN_Play/generated_outputs.py` (new)
- `tldw_Server_API/app/core/VN_Play/adapters.py`
- `tldw_Server_API/app/core/VN_Play/service.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_generated_outputs.py` (new)
- `tldw_Server_API/tests/VN_Play/test_vn_play_turns.py`

**Steps:**

- [ ] Create Pydantic models with `extra="forbid"` for `narrative_dialogue`, `choice_set`, and `scene_update` outputs.
- [ ] Enforce max string lengths, array caps, choice ID regex/uniqueness, metadata 4 KB cap, visual label 4 KB cap, and attached-character validation hook.
- [ ] Add a generation-specific adapter interface to `VN_Play/adapters.py` that calls the existing chat provider seam using provider/model/max token/temperature from pinned profile snapshots.
- [ ] Keep adapter calls mockable by tests and late-bound like `perform_chat_api_call_async`.
- [ ] Pass VN usage/accounting metadata through the existing LLM call path: `vn_session_id`, `script_id`, `script_version_id`, `generation_id`, `generation_request_id`, `generation_revision_id`, `generation_profile_key`, `generation_profile_snapshot_id`, and `generation_point_key`.
- [ ] Persist returned usage metadata on generation revisions without introducing a VN-only usage ledger unless the existing accounting path cannot carry the metadata.
- [ ] Treat provider/rate-limit failures as model-call failures with stable public error codes and persisted debug metadata.
- [ ] Convert provider exceptions into stable public codes such as `provider_unavailable`, `model_timeout`, and `model_error`.
- [ ] Add moderation adapter seam that can fail closed for hosted/public profiles and record `moderation_skipped_by_policy` for local opt-out.

**Tests:**

- [ ] Unknown fields fail at root and nested levels.
- [ ] Empty narrative/dialogue output fails for `narrative_dialogue`.
- [ ] Invalid visual directive shapes fail.
- [ ] Choice IDs must be unique and valid.
- [ ] Oversized metadata and labels fail.
- [ ] Mocked provider failure maps to stable public code.
- [ ] Mocked usage metadata is passed through and persisted on the revision.
- [ ] Mocked rate-limit/provider-denial failure does not create an active revision.
- [ ] Moderation failure blocks activation for hosted profile.

**Commit message:** `Add strict VN generation output parsing`

---

## Task 4: Provider Call Transaction And Recovery Orchestration

**Goal:** Make nondeterministic provider calls recoverable and idempotent by splitting database state changes around the external call.

**Files:**

- `tldw_Server_API/app/core/VN_Play/service.py`
- `tldw_Server_API/app/core/DB_Management/VNPlay_DB.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_action_requests.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_turns.py`

**Steps:**

- [ ] Add service helpers for the two-transaction generation flow: preflight/lock/checkpoint/request/action creation, provider call outside transaction, and final revision/event/response persistence.
- [ ] In the first transaction, validate owner/session/mode/scene version, acquire the per-session action lock, create or reuse generation rows, create the checkpoint, create request/action rows, and mark action/request `in_progress`.
- [ ] Set `provider_call_started_at` immediately before invoking the provider and never start a second provider call for the same in-progress action once that field is set.
- [ ] In the second transaction, persist revision status, parser/moderation result, usage metadata, events, active revision pointer, scene version, and the completed action response.
- [ ] Implement stale lease recovery: if `provider_call_started_at` is set and no provider result was persisted by lease expiry, mark request/action `abandoned` with public error `generation_attempt_abandoned`; retry must use a new idempotency key/request.
- [ ] Allow the same idempotency key to reclaim a request only when the first transaction committed but `provider_call_started_at` is still unset.
- [ ] Replay the completed generation action response when the provider result was persisted but the HTTP response was lost.
- [ ] Reject stale `client_scene_version` before any provider invocation.

**Tests:**

- [ ] Duplicate same-key request while provider call is in progress returns `generation_request_in_progress` and does not call the provider twice.
- [ ] Same-key reclaim works before `provider_call_started_at`.
- [ ] Stale lease after provider start marks request/action abandoned and requires a new key.
- [ ] Completed request replay returns stored response without a provider call.
- [ ] Stale scene version returns 409 before provider invocation.

**Commit message:** `Add VN generation call recovery orchestration`

---

## Task 5: Interpreter Integration For Automatic And Confirmation-Gated Generation

**Goal:** Replace literal-only `generate` behavior with real generation execution while preserving deterministic literal behavior.

**Files:**

- `tldw_Server_API/app/core/VN_Play/service.py`
- `tldw_Server_API/app/core/VN_Play/constants.py`
- `tldw_Server_API/app/core/VN_Play/state.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_turns.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_state.py`

**Steps:**

- [ ] Refactor `_execute_script_program` so a `generate` opcode can return a pending generation descriptor instead of requiring literal text.
- [ ] Resolve `profile_key` through the session's published script-version snapshot map and copy snapshot ID/key onto generation records.
- [ ] Create or reuse the session generation point by deterministic point key such as `label:index:opcode_id`.
- [ ] For `requires_user_confirm=true`, create `pending_confirmation` request state, update script position with `waiting_generation_confirmation`, and return a normal script action response.
- [ ] For automatic generation, create a checkpoint before model invocation, execute up to the profile batch cap, and pause with `waiting_reason=generation_batch_limit` if the cap is reached.
- [ ] Persist successful output as a revision, activate it, append immutable generation events, and update public state from active revision output.
- [ ] For `scene_update`, route generated `visual_directives` through the existing visual resolver, persist applied/rejected resolver outcomes on the revision, and append audit events without letting unresolved directives fail the generation.
- [ ] Persist failed model/parse/moderation outcomes and leave the session at the generation point for retry.
- [ ] Preserve existing literal `narrative_text` and `regeneration_text` path with `model_invoked=false`.

**Tests:**

- [ ] Automatic generation creates request/revision/generation rows and advances scene.
- [ ] Confirmation-gated generation pauses without model call.
- [ ] Batch cap of one pauses before a second automatic generation.
- [ ] `scene_update` persists applied and rejected visual resolver outcomes on the active revision.
- [ ] Model failure persists failed revision/request and does not advance cursor.
- [ ] Existing literal generation tests still pass.

**Commit message:** `Run scripted VN generation through backend runtime`

---

## Task 6: Generated Choices And Authored Control Flow

**Goal:** Allow generated `choice_set` output while keeping branch targets authored by scripts.

**Files:**

- `tldw_Server_API/app/core/VN_Play/service.py`
- `tldw_Server_API/app/core/VN_Play/state.py`
- `tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_branch_navigation.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_turns.py`

**Steps:**

- [ ] Expose generated choices in public script state with `source`, `generation_id`, and `revision_id`.
- [ ] Update choice selection to recognize generated choices owned by the active revision at the current generation point.
- [ ] On selection, record branch/event metadata with generation/revision/choice IDs.
- [ ] Set system variables such as `last_generated_choice.id`, `.text`, and `.metadata`.
- [ ] Jump only to authored `on_generated_choice`; never accept model-provided next targets.
- [ ] Ensure branch navigation surfaces generated-choice metadata without debug prompt/raw output leakage.

**Tests:**

- [ ] Generated choice selection jumps to `on_generated_choice`.
- [ ] Generated choice metadata is stored in branch events.
- [ ] Choice from inactive revision cannot be selected.
- [ ] Model-provided target fields are rejected by parser.

**Commit message:** `Support generated VN script choices`

---

## Task 7: Revision Activation, Regeneration, Cancellation, And Checkpoint Restore

**Goal:** Implement the command semantics that make nondeterministic generation recoverable without rewriting events.

**Files:**

- `tldw_Server_API/app/core/VN_Play/service.py`
- `tldw_Server_API/app/core/DB_Management/VNPlay_DB.py`
- `tldw_Server_API/app/core/VN_Play/state.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_action_requests.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_save_slots.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_state.py`

**Steps:**

- [ ] Implement cancel for pending confirmation requests, including `on_cancel` jump and `generation_canceled` stable state.
- [ ] Implement regeneration for a generation point by creating a new request/revision while preserving history.
- [ ] Implement activation of older successful revisions with checkpoint-before-activation and active pointer update.
- [ ] Block activation when downstream material events exist after the active revision event, returning `revision_activation_blocked` with `downstream_material_event_exists`.
- [ ] Extend checkpoint/save-slot snapshots with active generation revision maps.
- [ ] On restore, update active revision pointers in one transaction, clear absent pointers, append `script_generation_checkpoint_restored`, and increment scene version.
- [ ] Update scene derivation to use the active-revision overlay rather than blindly replaying inactive generation event payloads.
- [ ] For `scene_update`, derive current scene from the active revision's stored applied/rejected resolver outcomes, not stale `visual_directive_applied` events from inactive revisions.

**Tests:**

- [ ] Cancel with `on_cancel` jumps and advances through normal script flow.
- [ ] Cancel without `on_cancel` leaves stable canceled state.
- [ ] Regenerate creates inactive history until activated or returns active result as specified by service command.
- [ ] Activation changes current public output without rewriting original events.
- [ ] Activating a `scene_update` revision changes current visuals using the revision's stored resolver outcomes.
- [ ] Activation is blocked after downstream material events.
- [ ] Checkpoint restore restores exact active revision map.

**Commit message:** `Add VN generation revision controls`

---

## Task 8: API Schemas And Endpoints

**Goal:** Expose backend-owned generation commands and history through stable VN API endpoints.

**Files:**

- `tldw_Server_API/app/api/v1/endpoints/vn_play.py`
- `tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py`
- `tldw_Server_API/app/api/v1/schemas/pagination.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_api.py`
- `tldw_Server_API/tests/VN_Platform/test_vn_platform_errors.py`

**Steps:**

- [ ] Add request/response schemas for generation confirmation, cancellation, regeneration, activation, history list, revision list/detail, and debug detail.
- [ ] Add `waiting_generation_confirmation` and generated-choice fields to public script state schemas.
- [ ] Add event type literals for generation lifecycle events.
- [ ] Implement `POST /api/v1/vn/vn-play/sessions/{session_id}/script/generation-requests/{generation_request_id}/confirm`.
- [ ] Implement `POST /api/v1/vn/vn-play/sessions/{session_id}/script/generation-requests/{generation_request_id}/cancel`.
- [ ] Implement `POST /api/v1/vn/vn-play/sessions/{session_id}/script/generations/{generation_id}/regenerate`.
- [ ] Implement `POST /api/v1/vn/vn-play/sessions/{session_id}/script/generations/{generation_id}/revisions/{revision_id}/activate`.
- [ ] Implement `GET /api/v1/vn/vn-play/sessions/{session_id}/script/generations` with offset pagination.
- [ ] Implement `GET /api/v1/vn/vn-play/sessions/{session_id}/script/generations/{generation_id}/revisions/{revision_id}/debug` as owner/admin debug-only detail.
- [ ] Implement explicit debug authorization for owner/admin access; do not assume the current owner-scoped `_service` dependency provides admin cross-user access.
- [ ] Redact moderation-blocked raw output by default and require explicit reveal parameters such as `include_blocked_raw=true&confirm=REVEAL_MODERATION_BLOCKED`.
- [ ] Emit `vn.script_generation.debug_read` through the existing audit/logging path for successful, denied, and moderation-blocked reveal reads where configured; in single-user/no-audit deployments, log a structured warning instead of blocking.
- [ ] Keep raw prompts, raw model output, parser diagnostics, and moderation diagnostics out of public list/state responses.

**Tests:**

- [ ] Endpoints enforce session owner access.
- [ ] Confirm/cancel/regenerate/activate are idempotent and reject stale `client_scene_version`.
- [ ] History list uses canonical offset pagination metadata.
- [ ] Debug endpoint includes diagnostics only for authorized owner/admin.
- [ ] Non-owner debug access is denied and recorded through the audit/logging path.
- [ ] Moderation-blocked raw output is redacted by default and revealed only with explicit confirmation parameters.
- [ ] Public responses never include raw prompt or raw model output.

**Commit message:** `Expose VN generation runtime API`

---

## Task 9: Capabilities, Setup Metadata, Docs, And Verification

**Goal:** Make the backend contract discoverable and verify the touched backend surface.

**Files:**

- `tldw_Server_API/app/api/v1/endpoints/vn_capabilities.py`
- `tldw_Server_API/app/api/v1/schemas/vn_capabilities_schemas.py`
- `tldw_Server_API/app/core/VN_Play/setup_options.py`
- `tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py`
- `Docs/API/VN.md` or the current VN API doc location if different
- `tldw_Server_API/tests/VN_Platform/test_vn_capabilities_api.py`
- Existing setup-options tests in `tldw_Server_API/tests/VN_Play/test_vn_play_api.py`, or add `tldw_Server_API/tests/VN_Play/test_vn_play_setup_options.py` if a focused split is cleaner.

**Steps:**

- [ ] Add capability flags/limits for scripted generation, output schemas, confirmation support, revision activation, history debug detail, and batch limits.
- [ ] Add setup-options and script-version metadata for profile key, immutable snapshot ID, provider class, max automatic generation batch count, moderation requirement, estimated cost class, supported output schemas, dynamic choice support, scene update support, and whether confirmation is required by profile or opcode.
- [ ] Add readiness warnings when required profile snapshots are missing, unavailable, or incompatible with a script's generated output requirements.
- [ ] Update API docs with endpoint list, public/debug response boundary, idempotency requirements, and examples for confirmation/cancel/regenerate/activate.
- [ ] Run focused VN test suites.
- [ ] Run compile check on touched backend packages.
- [ ] Run Bandit on touched backend scope.
- [ ] Run `git diff --check`.

**Verification Commands:**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/VN_Scripts tldw_Server_API/tests/VN_Play tldw_Server_API/tests/VN_Platform -q
python -m compileall tldw_Server_API/app/core/VN_Play tldw_Server_API/app/core/VN_Scripts tldw_Server_API/app/core/DB_Management tldw_Server_API/app/api/v1/endpoints/vn_play.py tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py
python -m bandit -r tldw_Server_API/app/core/VN_Play tldw_Server_API/app/core/VN_Scripts tldw_Server_API/app/core/DB_Management/VNPlay_DB.py tldw_Server_API/app/core/DB_Management/VNScripts_DB.py tldw_Server_API/app/api/v1/endpoints/vn_play.py -f json -o /tmp/bandit_vn_scripted_generation_backend.json
git diff --check
```

**Commit message:** `Document VN generation runtime capabilities`

---

## Implementation Notes

- Keep all generation metadata in the per-user ChaChaNotes database alongside existing VN Play session/event/checkpoint data.
- Do not add frontend logic or client-owned provider routing. The backend owns prompt construction, profile snapshot resolution, model invocation, moderation, parsing, and replay.
- Preserve existing public/debug boundary: public state shows active output and stable errors; debug endpoints show raw prompts, raw model output, parser diagnostics, moderation diagnostics, and provider usage.
- Prefer additive DB columns/tables with decode defaults so existing VN Play and VN Scripts tests continue to pass.
- Use deterministic idempotency keys scoped to session/action/request owner and replay stored responses for completed commands.
- Keep literal `generate` support as a compatibility path and as a useful deterministic test fixture.
- Checkpoint before nondeterministic model generation and before revision activation.
- Avoid implicit fallback providers. Provider/model unavailability is a persisted failure, not a routing decision.
- Keep provider calls outside open DB transactions. Provider call recovery is a first-class runtime requirement, not a best-effort retry wrapper.
- Store `scene_update` resolver outcomes on revisions so activation and checkpoint restore never depend on stale inactive visual events.
- Treat debug reveal as a sensitive read path with explicit owner/admin authorization, reveal confirmation, and audit/log hooks.
