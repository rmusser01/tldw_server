# UAT261 causal-evidence feasibility plan

## Recommendation

Do not change a provider setting, enable a runtime flag, or send a call now. The current `complete-v2` path has a suitable *in-process test seam* for a bounded paired capture, but it does not already retain the causal record needed to explain the historical anomaly.

Authorize one separate, test-only capture harness first. It should prove that a final `complete-v2` provider-bound call can produce the prescribed redacted projection. Only after that harness passes should a separately authorized, fixed two-call provider run be considered.

## What exists today

### Final outbound messages can be fingerprinted without changing them

`character_chat_completion` assembles the final `formatted` list, including prompt preset, history, memory, steering, and world-book context, before the provider call ([character_chat_sessions.py](../../../tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:6173), [character_chat_sessions.py](../../../tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:6240)). The same final list is passed as `messages_payload` to `perform_chat_api_call` ([character_chat_sessions.py](../../../tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:6497)).

The maintained envelope helper canonicalizes message order and keys and emits versioned SHA-256 fingerprints, lengths, roles, counts, and token estimates without emitting prompt text ([prompt_cost_envelope.py](../../../tldw_Server_API/app/core/Chat/prompt_cost_envelope.py:79), [prompt_cost_envelope.py](../../../tldw_Server_API/app/core/Chat/prompt_cost_envelope.py:103)). The existing streaming integration test already monkeypatches this exact module-level call boundary and collects provider call kwargs locally ([test_complete_v2_streaming_e2e_mock.py](../../../tldw_Server_API/tests/Character_Chat/test_complete_v2_streaming_e2e_mock.py:122)).

### Safe usage storage exists, but is not connected to this call

`llm_usage_log` supports prompt and world-book fingerprints plus sanitized provider usage metadata ([usage_tracker.py](../../../tldw_Server_API/app/core/Usage/usage_tracker.py:285), [usage_tracker.py](../../../tldw_Server_API/app/core/Usage/usage_tracker.py:435)). Its normalizer redacts credential and prompt-like fields while retaining numeric counters ([llm_usage_normalizer.py](../../../tldw_Server_API/app/core/Usage/llm_usage_normalizer.py:15), [llm_usage_normalizer.py](../../../tldw_Server_API/app/core/Usage/llm_usage_normalizer.py:87)).

However, `complete-v2` itself invokes neither `log_llm_usage` nor a durable equivalent: its only route-local usage log is for the *auto-router selection call*, not the final completion ([character_chat_sessions.py](../../../tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:1023)). The final non-stream response extracts assistant content and returns it, omitting provider `finish_reason`, `usage`, and provider-system fingerprint ([character_chat_sessions.py](../../../tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:6785), [character_chat_sessions.py](../../../tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:7129)). Streaming forwards provider SSE but does not persist terminal metadata ([character_chat_sessions.py](../../../tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:6785)).

The prompt-cost guardrail can construct an envelope, but only when its runtime configuration is enabled and it records response metadata only for a warning/block decision ([character_chat_sessions.py](../../../tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:6402)). Turning it on or changing its thresholds would change runtime behavior and is outside this investigation. The prompt-preview endpoint is also unsuitable as the causal record: it returns raw section contents and has a separately maintained assembly path ([character_chat_sessions.py](../../../tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:5779), [character_chat_sessions.py](../../../tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:5947)).

## Minimal isolated harness

Add a temporary, test-only harness in a future authorized task. It must patch only `character_chat_sessions.perform_chat_api_call` at the existing test seam and must never invoke a network adapter. Run two otherwise identical requests against the same fixture card and a fresh, isolated chat each time. Keep the current route's chosen request fields identical; do not supply alternative temperature, model, provider, or prompt settings.

At the patched boundary, calculate in memory and write only this projection to a dedicated evidence file:

```json
{
  "capture_version": "uat261-boundary-v1",
  "call_label": "A or B",
  "message_fingerprint_version": "prompt-v1",
  "message_fingerprint": "prompt-v1:sha256:...",
  "message_count": 0,
  "message_roles": ["system", "user"],
  "generation_settings_fingerprint": "sha256:...",
  "provider_response": {
    "finish_reason": "stop | length | null",
    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    "system_fingerprint_hash": "sha256:... | null"
  },
  "final_answer": {"present": true, "length": 0, "sha256": "..."}
}
```

The generation-settings digest must use only the actual outbound, non-secret fields: provider identifier, model identifier, temperature, top-p, repetition penalty, stop value, max tokens, stream flag, and hashes of tool/cache-intent objects if present. It must exclude `api_key`, `app_config`, credential runtime/context objects, `credentials_resolved`, user identifiers, headers, and all message content. The provider fake should return distinct synthetic terminal metadata for the two calls so the harness proves both extraction and null handling; it must not claim anything about a real provider.

For streaming coverage, the fake should emit a normal terminal chunk with `finish_reason`, optional usage if the provider supplies it, and `[DONE]`. The harness records only the accumulated final-answer hash/length and terminal metadata. It must explicitly prove that no raw message, provider body, reasoning content, credential, config value, or response text reaches the evidence JSON or test output.

## Later, bounded live evidence only if authorized

The harness cannot recover the historic request. If it passes, one user-authorized live pair may be useful: two fresh non-persisted conversations using the same existing tagged card, the same current explicit provider/model, exactly the same request body, and the established streaming mode. A temporary in-process boundary wrapper should produce the same projection, then delegate to the normal provider call. It must stop after those two calls, make no retries, and retain no raw streams or prompts.

Interpretation is limited:

- different message/settings fingerprints establish an application-controlled request difference;
- identical fingerprints with differing final-answer hashes show the application passed equivalent observed input, but do not by themselves prove provider fault;
- missing finish/usage/system metadata remains a provider-observability absence, not a fabricated value.

This cannot establish historical recovery because the anomalous request did not retain its outbound fingerprint. A production observability feature would be a separate authorization and design task; this plan proposes no production change.

## Frozen inputs examined

| File | SHA-256 |
| --- | --- |
| `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py` | `1e55a255af414653f4df2f7c6a18b220f26353b1fb252d5035d8a4e6aa0bcb1b` |
| `tldw_Server_API/app/core/Chat/prompt_cost_envelope.py` | `e8899904e61bd1d8dc403f5a467cb504f4fd525fe86626bf4c4f0bf548ea54ba` |
| `tldw_Server_API/app/core/Usage/usage_tracker.py` | `755073cebd8089a5aa92bf6bb40ea7555f27dea9f679caa1f3f3d8c15155ff19` |
| `tldw_Server_API/app/core/Usage/llm_usage_normalizer.py` | `3b983afbfe6c4fe1422484a5515b5413129083200fc60275f6cf88fba9defaaa` |
| `tldw_Server_API/tests/Character_Chat/test_complete_v2_streaming_e2e_mock.py` | `eca9ee8589f2b367f60ed71eecf4719e0bff5d31d0a3d185ae2211c1120aefe3` |

No runtime, provider, network, source, test, task, or Git mutation was made for this feasibility review.
