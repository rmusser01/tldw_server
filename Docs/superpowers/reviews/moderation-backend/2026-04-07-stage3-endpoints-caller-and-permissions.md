# Stage 3 Endpoints Caller and Permissions

## Scope
Review moderation endpoints, caller behavior, authz boundaries, and the real chat-site enforcement contract.

## Files Reviewed
- `tldw_Server_API/app/api/v1/endpoints/moderation.py`
- `tldw_Server_API/app/api/v1/schemas/moderation_schemas.py`
- `tldw_Server_API/app/api/v1/endpoints/chat.py`
- `tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py`
- `tldw_Server_API/app/core/Chat/chat_service.py`
- `tldw_Server_API/app/core/Moderation/supervised_policy.py`
- `tldw_Server_API/app/core/Moderation/governance_utils.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_moderation_permissions_claims.py`
- `tldw_Server_API/tests/unit/test_moderation_test_endpoint_sample.py`
- `tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py`
- `tldw_Server_API/tests/Chat_NEW/integration/test_moderation_categories.py`
- `tldw_Server_API/tests/Guardian/test_supervised_policy.py`

## Tests Reviewed
- `tldw_Server_API/tests/AuthNZ_Unit/test_moderation_permissions_claims.py`
- `tldw_Server_API/tests/unit/test_moderation_test_endpoint_sample.py`
- `tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py`
- `tldw_Server_API/tests/Chat_NEW/integration/test_moderation_categories.py`
- Guardian policy chat-type expectations were inspected in `tldw_Server_API/tests/Guardian/test_supervised_policy.py`
- Endpoint and caller-path verification slice passed: `14 passed, 228 warnings in 14.60s`

## Validation Commands
- `rg -n "moderat|check_text|evaluate_action|redact_text|effective_policy|If-Match|ETag|override|permissions|claim" tldw_Server_API/app/api/v1/endpoints/moderation.py tldw_Server_API/app/api/v1/schemas/moderation_schemas.py tldw_Server_API/app/api/v1/endpoints/chat.py`
- `sed -n '1,320p' tldw_Server_API/app/api/v1/endpoints/moderation.py`
- `sed -n '1,260p' tldw_Server_API/app/api/v1/schemas/moderation_schemas.py`
- `nl -ba tldw_Server_API/app/api/v1/endpoints/moderation.py | sed -n '1,430p'`
- `nl -ba tldw_Server_API/app/api/v1/endpoints/moderation.py | sed -n '402,490p'`
- `nl -ba tldw_Server_API/app/api/v1/endpoints/chat.py | sed -n '2988,3010p'`
- `nl -ba tldw_Server_API/app/api/v1/endpoints/chat.py | sed -n '3852,3860p'`
- `nl -ba tldw_Server_API/app/core/Chat/chat_service.py | sed -n '2294,2505p'`
- `nl -ba tldw_Server_API/app/core/Chat/chat_service.py | sed -n '3720,3825p'`
- `nl -ba tldw_Server_API/app/core/Chat/chat_service.py | sed -n '4155,4258p'`
- `nl -ba tldw_Server_API/app/core/Chat/chat_service.py | sed -n '4845,4952p'`
- `nl -ba tldw_Server_API/app/core/Moderation/supervised_policy.py | sed -n '360,460p'`
- `sed -n '1,220p' tldw_Server_API/app/core/Moderation/governance_utils.py`
- `nl -ba tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py | sed -n '896,916p'`
- `nl -ba tldw_Server_API/tests/AuthNZ_Unit/test_moderation_permissions_claims.py | sed -n '1,220p'`
- `nl -ba tldw_Server_API/tests/unit/test_moderation_test_endpoint_sample.py | sed -n '1,220p'`
- `nl -ba tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py | sed -n '1,260p'`
- `nl -ba tldw_Server_API/tests/Chat_NEW/integration/test_moderation_categories.py | sed -n '1,220p'`
- `nl -ba tldw_Server_API/tests/Guardian/test_supervised_policy.py | sed -n '1240,1315p'`
- `source ../../.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/AuthNZ_Unit/test_moderation_permissions_claims.py tldw_Server_API/tests/unit/test_moderation_test_endpoint_sample.py tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py tldw_Server_API/tests/Chat_NEW/integration/test_moderation_categories.py`

## Confirmed Findings
- Medium, high confidence: the admin `/moderation/test` endpoint does not model the live guardian-enforced moderation path, so it can report weaker effective policy and pass results than real chat execution for supervised accounts. The admin tester always uses the base moderation service and the base `get_effective_policy(payload.user_id)` result in [`moderation.py`](../../../../tldw_Server_API/app/api/v1/endpoints/moderation.py) lines 402-489. The live chat input path instead calls [`moderate_input_messages`](../../../../tldw_Server_API/app/core/Chat/chat_service.py) with `supervised_policy_engine` and `dependent_user_id` from [`chat.py`](../../../../tldw_Server_API/app/api/v1/endpoints/chat.py) lines 2989-3008, then overlays guardian policies and notifications in [`chat_service.py`](../../../../tldw_Server_API/app/core/Chat/chat_service.py) lines 2325-2359 before evaluating the effective policy in lines 2428-2496. The live chat output path similarly wraps the base moderation service in `GuardianModerationProxy` in [`chat.py`](../../../../tldw_Server_API/app/api/v1/endpoints/chat.py) lines 3852-3859, and all output moderation paths then evaluate the overlaid policy in [`chat_service.py`](../../../../tldw_Server_API/app/core/Chat/chat_service.py) lines 3721-3819, 4155-4254, and 4850-4951. `GuardianModerationProxy` proves that the live output path can change `get_effective_policy()` results by layering supervised rules on top of the base policy in [`supervised_policy.py`](../../../../tldw_Server_API/app/core/Moderation/supervised_policy.py) lines 360-377. Impact: the admin moderation tester is not trustworthy for supervised accounts because it can miss guardian-only blocks or redactions and return an `effective` policy snapshot that is not the one enforced by chat. The currently selected tests do not cover this mismatch: the admin endpoint sample test uses a plain `ModerationService` in [`test_moderation_test_endpoint_sample.py`](../../../../tldw_Server_API/tests/unit/test_moderation_test_endpoint_sample.py) lines 52-157, and the chat integration tests patch `get_moderation_service` with toy base services in [`test_moderation.py`](../../../../tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py) lines 55-118 and [`test_moderation_categories.py`](../../../../tldw_Server_API/tests/Chat_NEW/integration/test_moderation_categories.py) lines 54-95.

## Probable Risks
- Medium, medium confidence: governance `scope_chat_types` may not propagate through the main chat completions moderation path, which would make guardian rules scoped to non-`regular` chat types ineffective or misapplied. The moderation pipeline and supervised-policy engine both support a `chat_type` input in [`chat_service.py`](../../../../tldw_Server_API/app/core/Chat/chat_service.py) lines 2304-2307 and [`supervised_policy.py`](../../../../tldw_Server_API/app/core/Moderation/supervised_policy.py) lines 94-156, and governance matching treats `None` as `"regular"` in [`governance_utils.py`](../../../../tldw_Server_API/app/core/Moderation/governance_utils.py) lines 113-127. Tests explicitly show that guardian policies behave differently for `"character"` and `"regular"` chat types in [`test_supervised_policy.py`](../../../../tldw_Server_API/tests/Guardian/test_supervised_policy.py) lines 1244-1294. But the main chat endpoint calls `moderate_input_messages(..., supervised_policy_engine=_supervised_engine, dependent_user_id=_dep_user_id)` without a `chat_type` argument in [`chat.py`](../../../../tldw_Server_API/app/api/v1/endpoints/chat.py) lines 2996-3008, and the output path constructs `GuardianModerationProxy(base, _supervised_engine, _dep_user_id)` without any chat-type input in [`chat.py`](../../../../tldw_Server_API/app/api/v1/endpoints/chat.py) lines 3852-3859. Because chat completions does accept character context via `character_id` in [`chat_request_schemas.py`](../../../../tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py) lines 905-906 and validates it in [`chat.py`](../../../../tldw_Server_API/app/api/v1/endpoints/chat.py) lines 2315-2318, there is a plausible path where character-scoped guardian rules are not enforced as intended. Confidence is downgraded because I did not find an explicit contract stating that `/chat/completions` should map `character_id` or other modes to a non-`regular` moderation chat type.

## Improvements
- If `/moderation/test` is meant to reflect real chat behavior, add an optional supervised or dependent context and use the same guardian overlay path as live chat before returning `effective`, `action`, and `redacted_text`.
- If `/moderation/test` is intentionally base-policy only, document that limitation in the endpoint contract and UI copy so admins do not mistake it for a live-path simulator.
- Add one caller-path test where a supervised account has a guardian-only rule and assert that chat blocks or redacts while `/moderation/test` currently does not; that regression would pin the current mismatch and any future fix.
- Add at least one authz test on a mutating moderation route such as `PUT /moderation/users/{user_id}` or `POST /moderation/blocklist/append`, because the current permissions slice only exercises `GET /moderation/users`.

## Open Questions
- Does `/chat/completions` intentionally collapse all moderation governance contexts to `"regular"`, even when `character_id` is present, or is chat-type propagation simply not implemented yet?

## Exit Note
No direct router-auth bug was found in this pass: the moderation router is consistently gated by both `require_roles("admin")` and `require_permissions(SYSTEM_CONFIGURE)` in [`moderation.py`](../../../../tldw_Server_API/app/api/v1/endpoints/moderation.py) lines 37-42, and the narrow permission slice passed. The main endpoint-level issue is representational correctness: the admin test surface is not aligned with the live guardian-enforced caller path. Task 3 is complete and the review can now move into persistence, optimistic concurrency, and override or blocklist write verification.
