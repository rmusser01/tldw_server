# Persona Live preserves speech provider selection

Date: 2026-09-07
Status: Accepted
Task: TASK-13214

Persona Live preparation incorrectly restricted all users to the Kokoro setup used
for UAT, while its synthesis route already used the shared TTS service. Generic
voice and model defaults also crossed provider boundaries.

Use one Persona TTS resolution path for preparation and synthesis. Preserve the
selected registered provider, optional model and voice; use that adapter's own
configured defaults when fields are omitted. Keep the existing Persona `tldw`
alias for Kokoro to avoid silently changing saved profiles. Browser speech is
prepared and played by the browser. Explicit configured gateways use existing
server-owned gateway validation and credential resolution, never client URLs.

Both preparation and synthesis enter the existing authenticated audio credential
scope, applying effective BYOK and provider restrictions. Remote credentials are
request-owned; their temporary adapters are closed on failure, completion and
cancellation. Chat preparation validates the target only; actual Chat dispatch
retains credential, moderation and budget enforcement.

Preparation initializes the selected adapter without generating speech. Kitten
initialization stays model-independent so preparation and synthesis load the
selected model through its cached loader, without requiring default-model assets.
OpenAI preparation disables optional synthetic key-verification speech. Kokoro's
additional lazy model load remains a Kokoro-specific readiness step, not a
provider allowlist. Synthesis continues through the shared TTS service with
fallback disabled and the authenticated user context for owned voices/gateways.
Runtime failures invalidate readiness and preserve text-only recovery and Stop
ownership. Optional `tts_model` is stored in existing JSON defaults; no schema
migration is needed.

We reject both broad removal of all readiness checks and a second hard-coded
provider allowlist. An installed/configured adapter is the runtime boundary;
physical qualification of one provider does not establish support for only that
provider or prove every other deployment works.
