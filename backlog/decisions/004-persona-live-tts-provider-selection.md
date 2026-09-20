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

A blank Persona voice is not an instruction to copy another browser surface's
voice preference. The browser leaves it unset; the server uses its configured
global voice only when the selected provider matches the configured default
provider, otherwise the selected adapter supplies its own default. Global audio
provider aliases retain their own meaning during that comparison (global `tldw`
means Kitten; the legacy Persona alias remains Kokoro). Explicit Persona voices
remain authoritative. Kitten readiness validates the voice against the loaded
model without synthesizing, and speech streams close inside their credential
scope even when an error arrives after partial audio. Connected Live settings
remain fixed for that connection; saving defaults requires reconnecting, and
Live displays the active model override to make this visible.

Kitten's cached runtime selection does not replace its configured default model
or revision. Registry initialization makes the lazy adapter routable; public
audio health separately reports an unprepared, failed, or loaded runtime. This
keeps selected-model preparation independent of default-model assets without
claiming that an untested runtime is healthy. Browser playback callbacks also
belong to one utterance generation, invalidated before cancellation, so an old
utterance cannot end replacement playback within the same voice turn.
