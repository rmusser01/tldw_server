# UAT133 independent design review

## Verdict

The bounded design is sound: after an explicit failed Retry has passed existing tail/content/correlation checks, place that exact identified saved user turn last exactly once. Preserve requested ordering for the other selected history rows. This fixes the actual final Custom OpenAI payload boundary without changing the general history-order contract or writing another canonical user.

## Independent provider-payload confirmation

Reviewed root's design and unchanged `/private/tmp/cycle5-controller-retry-order-probe.py` / `.log`. Extended a private copy through actual production functions:

`build_context_and_messages` → `apply_prompt_templating` → `inject_research_context_into_prompt` (no research context) → `build_call_params_from_request` → `_build_adapter_request_from_chat_args` → `CustomOpenAIAdapter._build_payload`.

Script `/private/tmp/cycle5-uat133-final-payload-review-probe.py`; log `/private/tmp/cycle5-uat133-final-payload-review-probe.log`. Run with activated project venv and `PYTHONDONTWRITEBYTECODE=1`. **2 RED descending / 2 GREEN ascending**, both full-client-history and user-only requests. At the final adapter payload, descending still ends with `Who coordinates Cedar?`; ascending ends with `Reply exactly: CEDAR RETRY READY.` All four retain canonical ID `failed` and perform zero duplicate writes.

The isolated process uses synthetic data/key, private config/data root, and blocked outbound sockets. It builds the actual request body but never calls an adapter transport or model. Initial extended-probe credential lookup hit the intentionally unavailable private credential store; supplying an unused synthetic adapter key made the construction independent of that store. Root's probe was not modified.

Source explains the result:

- `chat_service.py:4503–4511` returns requested-order history plus current turn; overlapping Retry can leave current turn empty.
- `apply_prompt_templating` removes/moves system text and transforms message contents while preserving non-system order (lines4631–4665).
- Research augmentation changes system context only (lines2988–3001).
- The endpoint passes that list to the call builder (`endpoints/chat.py:4789–4806`); the builder/adapter-request conversion retain the message list.
- `custom_openai_adapter.py:236–244` prepends the system message and extends with that unchanged list. Streaming and nonstreaming both use this payload builder.

This confirms application-provided wrong ordering. It is not a capture of the native provider's HTTP body or proof that every provider always answers the last question. The native wrong-answer observation remains evidence supplied by root.

## Implementation constraints

1. Reposition **after existing overlap and safety validation**, using `retry_user_message_id` matched to `historical_ids`, never content-only matching. Removing the row before overlap calculation could break full-prefix matching or weaken the extra-unsaved-user409 safeguard.
2. If the matched row is in the selected window, preserve its already-validated text/image representation and established persona/Character templating behavior while moving it. Neutral/persona literal Retry text and image MIME/bytes must survive. Do not accidentally introduce a second placeholder transformation or switch to synthetic image placeholder text.
3. If the exact saved retry row is outside the selected window, include the already-validated requested current user once. Maintain the existing rejection of extra unverified user entries. In particular, full client history plus a zero history window is not automatically authorized to bypass the extra-unsaved-user guard; test its existing accepted/rejected behavior explicitly.
4. Ordinary equal new user turns, two distinct equal historic messages, non-Retry requests and the general DESC history contract remain unchanged. Existing `test_build_context_uses_history_knobs` explicitly asserts descending order; globally reversing history is broader than this repair.
5. Do not create a new saved row, replace ACK identity, delete the saved error variant, or alter answered-tail409, correlation conflict, strict attachment and owner checks. Keep continuation/prefill behavior separate unless a test establishes that the flags may legally combine.

Required permanent coverage already identified by root is appropriate: final adapter-bound order/count for ASC/DESC, user-only/full-history, windows1/limited/0, unanswered tail and saved error envelope, literal/image controls, neutral/persona/Character, intentional identical turns, correlation conflict/answered tail and ownership. An endpoint/transport seam test should assert the final dispatched message, not merely `result[4]` count. No broader implementation request.

No repository/source/test/task/browser/runtime edits or inference. Waiting for explicit source-edit release. UAT134 remains under existing TASK13260.24 as instructed.
