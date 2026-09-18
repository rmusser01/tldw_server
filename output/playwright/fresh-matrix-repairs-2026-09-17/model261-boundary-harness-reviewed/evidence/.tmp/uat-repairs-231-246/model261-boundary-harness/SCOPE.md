# UAT261 boundary-harness scope

This diagnostic creates only files in this directory:

1. `boundary_harness.py`, an isolated two-call ASGI harness;
2. `boundary-harness-evidence.json`, a redacted projection; and
3. `RUN.md`, the command, result, and hashes.

The harness patches only the module-level
`character_chat_sessions.perform_chat_api_call` boundary. It invokes the
existing `complete-v2` request assembly for two fresh chats, never delegates
to a provider adapter, and writes no source, maintained test, runtime, or
provider-setting change. It uses `build_prompt_cost_envelope` on the actual
`messages_payload` at that boundary and hashes only the allowed outbound
settings. Its fake stream supplies synthetic terminal metadata solely to prove
the redacted extraction shape, including a missing-usage `null` case.

The projection must exclude all prompt text, streamed answer text, provider
reasoning, credentials, headers, user identifiers, and configuration values.
It cannot recover or explain the historical UAT261 request.
