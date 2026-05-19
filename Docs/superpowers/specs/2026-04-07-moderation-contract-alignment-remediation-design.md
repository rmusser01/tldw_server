# Moderation Contract Alignment Remediation Design

Date: 2026-04-07
Topic: Moderation backend remediation and contract alignment
Status: Proposed design

## Objective

Implement the backend moderation fixes and hardening work identified in the moderation review, including:

- all five confirmed findings
- the caller-path `chat_type` propagation risk
- the approved regression and hardening improvements

The implementation should fix the real defects without turning into a broad chat rewrite. The core theme is contract alignment: callers should see the same moderation behavior the running process actually enforces, and failed persistence should not silently change live state.

## Scope

This remediation is limited to the backend moderation surface and the live chat caller path that defines moderation behavior:

- `tldw_Server_API/app/core/Moderation/moderation_service.py`
- `tldw_Server_API/app/api/v1/endpoints/moderation.py`
- `tldw_Server_API/app/api/v1/schemas/moderation_schemas.py`
- `tldw_Server_API/app/api/v1/endpoints/chat.py`
- `tldw_Server_API/app/core/Chat/chat_service.py`
- `tldw_Server_API/app/core/Moderation/supervised_policy.py`
- moderation-focused backend tests under `tldw_Server_API/tests/`

This remediation includes:

- a canonical internal moderation evaluation contract
- compatibility wrappers for existing tuple-returning APIs
- phase-aware redaction behavior
- long-span regex detection correctness
- atomic persistence and commit-after-success semantics for user overrides and runtime overrides
- explicit admin test context for supervised and `chat_type`-aware evaluation
- deterministic moderation `chat_type` mapping in the chat caller path
- focused regression tests for the above

This remediation excludes:

- frontend moderation UI or service work
- unrelated chat refactors outside moderation call sites
- broad API redesign outside the moderation admin surface
- changing weak `If-Match` acceptance unless that becomes necessary during implementation

## Problems To Solve

The review identified five confirmed defects and one additional caller-path risk:

1. Failed user-override writes still change live in-memory moderation state.
2. Runtime settings persistence fails open when `persist=True`.
3. `/moderation/test` does not reflect the live supervised moderation path.
4. Phase-specific rules are ignored by redaction transforms.
5. Long-span regex matches can be missed on larger payloads.
6. `chat_type` may not propagate through `/chat/completions`, which can make guardian governance scoping wrong for character-context chat.

The fixes should address these directly rather than only patching symptoms.

## Approaches Considered

### Minimal surgical fixes

Patch each confirmed bug in place, add the smallest tests, and wire `chat_type` through the existing path with minimal structural change.

Trade-offs:

- smallest code diff
- lowest short-term disruption
- leaves duplicated moderation evaluation logic in chat and admin surfaces
- leaves caller contracts more implicit than explicit

### Targeted hardening pass

Fix the defects, make persistence semantics explicit, align the admin tester with the live caller path, and thread moderation caller context through chat execution.

Trade-offs:

- balances correctness and change size
- reduces repeated logic in the most fragile places
- still focused enough for one remediation pass

### Chosen: Deeper contract-alignment refactor

Refactor the moderation backend around a canonical evaluation contract, atomic state transitions, and explicit caller context propagation while preserving current external behavior where possible.

Why this is chosen:

- the confirmed defects share one root cause: different layers reconstruct moderation behavior independently
- the persistence defects are easier to fix correctly with shared state-transition helpers than with repeated local patches
- the admin tester and chat path should stop diverging by design, not just coincidence
- compatibility wrappers let the refactor reduce internal drift without forcing broad immediate API churn

## Design Principles

The implementation should follow these rules:

- introduce one canonical internal moderation evaluation shape
- keep existing tuple-returning moderation service methods as compatibility wrappers for now
- preserve current audit, metrics, and HTTP response shapes unless a reviewed defect requires change
- make supervised evaluation in `/moderation/test` explicit and opt-in
- define moderation `chat_type` mapping deterministically instead of inferring it ad hoc
- only commit in-memory durable state after successful persistence
- keep durable state transitions under the existing service lock through compute, persist, and swap
- keep chat refactoring focused on moderation branches, not unrelated request execution

## Architecture

The remediation has three coordinated seams.

### 1. Shared moderation evaluation contract

`moderation_service.py` will expose one canonical internal evaluation helper that computes:

- `action`
- `redacted_text`
- `matched_pattern`
- `category`
- `match_span`
- `sample`

This contract becomes the source of truth for:

- admin moderation testing
- chat input moderation decisions
- chat output moderation decisions
- phase-aware snippet and redaction behavior

The existing tuple-based service methods such as `evaluate_action()` and `evaluate_action_with_match()` should remain available as thin wrappers around the canonical helper so current callers do not break during the refactor. The refactor goal is to migrate chat and admin endpoint code onto the canonical result first, not to force a repo-wide return-shape conversion in one pass.

### 2. Atomic persistence and state transitions

`moderation_service.py` will centralize JSON persistence behind shared helpers:

- one atomic JSON write helper using temp-file plus `os.replace()`
- one write-and-commit helper per durable state family

The state transition rule is:

1. compute next state without mutating the live in-memory state
2. persist the next state atomically
3. only after successful persistence, swap the in-memory state
4. surface failure to the caller instead of silently logging and continuing

This rule applies to:

- per-user override updates
- per-user override deletions
- runtime override persistence for moderation settings

Blocklist writes already follow the atomic temp-file-and-replace pattern and should remain the model for the other JSON persistence paths.

### 3. Explicit moderation caller context

The live moderation path will carry a small moderation-specific context object through chat execution. It should include only:

- effective `user_id`
- optional `dependent_user_id`
- effective `chat_type`
- `apply_guardian_overlay`

This context exists to align policy resolution, not to become a general request bag.

For output moderation, the design should not rely on an implicit proxy that only overrides `get_effective_policy()` without carrying `chat_type`. The implementation should either extend that proxy contract explicitly or replace it on the main chat path with a helper that resolves the overlaid effective policy from the full moderation context. The key point is that guardian overlay behavior for output moderation must be driven by the same explicit context used for input moderation.

The same context model will be used to support explicit admin test simulation of the live caller path. `/moderation/test` should not silently infer supervised overlays from a bare `user_id`. Instead, guardian overlay and non-default `chat_type` behavior should only apply when the request explicitly asks for that context.

## Caller Context Rules

The remediation must define deterministic moderation `chat_type` mapping for the chat completions path.

For `/chat/completions`, the mapping must be based on the resolved assistant context, not only the raw request payload.

For the live chat path, the mapping is:

- if the effective assistant context is character-backed, moderation `chat_type` is `"character"`
- otherwise moderation `chat_type` is `"regular"`

Character-backed context includes both:

- requests that supply `character_id`
- requests that continue an existing character conversation whose stored context already resolves to a character assistant

This mapping should be computed once after assistant-context resolution and passed through the moderation flow for both:

- input moderation
- output moderation, including guardian overlays

The admin test endpoint should expose `chat_type` as an explicit optional request field. Its default should remain `"regular"` when omitted.

## Admin Test Contract

`/moderation/test` should align with live chat moderation behavior without becoming ambiguous.

The endpoint should be extended so supervised and `chat_type`-aware evaluation is explicit. The request contract should add optional fields for:

- `dependent_user_id`
- `chat_type`
- `apply_guardian_overlay`

Behavior rules:

- if `apply_guardian_overlay` is false or omitted, the endpoint evaluates the base moderation service path only
- if `apply_guardian_overlay` is true, the endpoint should use the same guardian overlay logic as live chat, using the target dependent user's guardian database rather than the caller principal
- if `apply_guardian_overlay` is true, `user_id` is required because the endpoint is simulating a live user moderation context
- if `apply_guardian_overlay` is true and `dependent_user_id` is omitted, it should default to `user_id`
- if both `user_id` and `dependent_user_id` are supplied while `apply_guardian_overlay` is true, they should resolve to the same dependent account or the endpoint should reject the request as an invalid simulation
- if `chat_type` is omitted, the endpoint defaults to `"regular"`
- the `effective` field in the response should reflect the actual evaluated policy for the test run, including guardian overlay when applied

This keeps the admin surface predictable and avoids quietly changing semantics for existing callers.

## Moderation Service Semantics

### Phase-aware redaction

Redaction helpers must stop applying rules that are ineligible for the current phase. The canonical evaluation helper should pass phase through to the redaction path so `redact_text()` and `redact_text_with_count()` use the same rule eligibility model as detection.

### Long-span regex correctness

The service should no longer treat the `4 * _max_scan_chars` fallback cutoff as a correctness boundary. Large-text scanning can still keep a chunked fast path, but the final detection contract should not silently miss a valid match solely because the payload is above an internal size threshold.

If performance protection is still needed, it should be applied in a way that is explicit and testable rather than an invisible correctness cutoff.

## Chat Path Refactor Boundaries

The biggest implementation risk is in `chat_service.py`, where moderation behavior is reconstructed multiple times. The refactor should reduce duplication, but it should not broadly rewrite chat execution.

Allowed change shape:

- replace repeated moderation decision reconstruction with shared helper usage
- factor any dependent-user guardian service bootstrap that must be shared between live chat and admin test into one helper instead of duplicating identity-to-database resolution logic
- keep audit emission in caller code
- keep metrics emission in caller code
- preserve current moderation-related HTTP errors and output mutation behavior unless required by a reviewed defect

Not allowed:

- unrelated chat execution restructuring
- audit schema changes unrelated to moderation defects
- response-shape changes for chat completions unrelated to moderation fixes

## Files Expected To Change

Primary implementation files:

- `tldw_Server_API/app/core/Moderation/moderation_service.py`
- `tldw_Server_API/app/api/v1/endpoints/moderation.py`
- `tldw_Server_API/app/api/v1/schemas/moderation_schemas.py`
- `tldw_Server_API/app/api/v1/endpoints/chat.py`
- `tldw_Server_API/app/core/Chat/chat_service.py`
- `tldw_Server_API/app/core/Moderation/supervised_policy.py`

Primary tests to update:

- `tldw_Server_API/tests/unit/test_moderation_check_text_snippet.py`
- `tldw_Server_API/tests/unit/test_moderation_redact_categories.py`
- `tldw_Server_API/tests/unit/test_moderation_runtime_overrides_bool.py`
- `tldw_Server_API/tests/unit/test_moderation_user_override_validation.py`
- `tldw_Server_API/tests/unit/test_moderation_test_endpoint_sample.py`
- `tldw_Server_API/tests/unit/test_moderation_etag_handling.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_moderation_permissions_claims.py`
- `tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py`
- `tldw_Server_API/tests/Chat_NEW/integration/test_moderation_categories.py`

Additional focused tests may be added only if an existing file becomes materially harder to understand than a small new test file.

## Test Strategy

Testing should be organized around the new contracts.

### Service regressions

- phase-aware redaction in `redact_text()`
- phase-aware redaction in `redact_text_with_count()`
- long-span regex detection beyond `4 * _max_scan_chars`
- failed `set_user_override()` persistence does not mutate live state
- failed `delete_user_override()` persistence does not mutate live state
- failed runtime override persistence surfaces failure and does not commit durable state

### Endpoint regressions

- `/moderation/test` base path remains compatible
- `/moderation/test` supervised path matches live caller behavior when explicit context is supplied
- `/moderation/test` rejects impossible overlay simulations or normalizes them to the same dependent target, per the final endpoint contract
- `/moderation/test` returns the overlaid effective policy snapshot when guardian overlay is applied
- mutating moderation route authz is covered, not only `GET /moderation/users`
- settings update endpoint fails when persistence fails

### Live caller-path regressions

- supervised-account chat behavior matches admin test output under equivalent explicit context
- character-context chat maps to moderation `chat_type="character"` both for direct `character_id` requests and for continued character conversations without a fresh `character_id`
- guardian `scope_chat_types` is honored through the live chat moderation path
- non-supervised chat output moderation remains correct for standard paths

## Compatibility

Compatibility requirements:

- existing tuple-returning moderation service methods remain callable
- existing chat completion response shapes remain unchanged
- existing blocklist ETag behavior remains unchanged in this pass unless implementation work forces a narrower change
- `/moderation/test` should remain usable without the new optional context fields

The only intended contract expansion is explicit optional context for the moderation admin test endpoint.

## Risks and Mitigations

### Risk: chat moderation refactor breaks audit or metrics behavior

Mitigation:

- keep audit and metrics emission in caller code
- only centralize moderation result computation
- add caller-path regressions for both streaming and non-streaming output moderation

### Risk: canonical evaluation helper creates broad API churn

Mitigation:

- keep compatibility wrappers
- migrate only the highest-risk callers in this pass
- defer repo-wide cleanup until after behavior is stable

### Risk: supervised admin testing exposes surprising semantics

Mitigation:

- make supervised overlay explicit and opt-in
- require explicit `dependent_user_id`
- default `chat_type` clearly to `"regular"`

## Success Criteria

This design is successful when:

- persistence failures no longer change live moderation state
- runtime settings persistence failures surface to callers
- phase-aware redaction is enforced consistently
- long-span regex matches are no longer silently missed due to the fallback cutoff
- `/moderation/test` can explicitly simulate the live supervised moderation path
- chat moderation consistently propagates deterministic `chat_type`
- new regression tests pin the defects that were previously missed
- existing compatible callers still work through wrapper behavior
