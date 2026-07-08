# TTS Settings Voice Preview Design

Date: 2026-07-07
Status: Approved for implementation planning
Backlog: TASK-12178

## Summary

Add an inline voice/backend preview action to the shared TTS settings component
used by WebUI and the browser extension. Users can test the currently selected
provider, model, voice, and related form values before saving.

The implementation should reuse the existing TTS synthesis path instead of
adding a new backend endpoint or embedding the full TTS playground in settings.

## Goals

- Let users preview the active unsaved TTS settings before saving.
- Cover Browser, tldw server TTS, OpenAI-compatible TTS, and ElevenLabs.
- Keep WebUI and extension behavior aligned by changing the shared
  `TTSModeSettings` surface.
- Avoid persisting settings, credentials, or key-validation state during
  preview.
- Show clear, sanitized failure feedback.

## Non-Goals

- Do not add a new backend preview endpoint.
- Do not turn settings into a mini TTS playground with custom sample text,
  segment lists, downloads, or history.
- Do not claim OpenAI-compatible preview validates unsaved frontend Base URL or
  API key fields unless implementation confirms those fields are honored by the
  current server speech path.
- Do not redesign the TTS settings page beyond the preview affordance.

## Existing Path Review

`TTSModeSettings` is the right UI insertion point because it is shared by
WebUI `/settings/speech` and the browser-extension `/settings/speech` route.

`resolveTtsProviderContext` is the right base path for server-backed preview
because it already handles text normalization, provider branching, tldw,
OpenAI-compatible, ElevenLabs, audio formats, and synthesis. It needs small
preview-specific additions or local handling for unsaved Browser voice and
unsaved ElevenLabs API key.

`tldwClient.synthesizeSpeech` already maps tldw form values into
`/api/v1/audio/speech`: `model`, `voice`, `response_format`, `speed`,
`lang_code`, `normalization_options`, `extra_params`, and `stream=false`.

The existing `VoicePreviewButton` should not be reused as-is. It calls the
server speech endpoint directly, disables Browser TTS, ignores several
provider-specific settings, and swallows errors. Its small button shape can be
copied, but behavior should use the shared TTS resolution path.

## UX

Add a compact preview row near the active provider's voice/model controls. The
button label should reflect state:

- `Preview voice`
- `Generating...`
- `Stop preview`

The preview uses a fixed short sample sentence. A fixed sample keeps this slice
small and avoids duplicating the TTS playground.

If required provider-specific fields are missing, block preview before making a
request and show a short inline message. For OpenAI-compatible TTS, copy should
be precise: preview tests the selected model and voice through the server speech
API, and server/provider credentials may still be required.

## Data Flow

1. User edits TTS provider controls.
2. User clicks `Preview voice`.
3. The preview controller builds overrides from current `form.values`, not
   saved settings.
4. For server-backed providers, it resolves a provider context for the fixed
   sample text, synthesizes one sample, creates an object URL, and plays it.
5. For Browser TTS, it creates a `SpeechSynthesisUtterance`, applies the
   unsaved browser voice when available, applies unsaved playback speed, and
   speaks it directly.
6. `Stop preview`, unmount, or provider change cancels the active preview,
   aborts in-flight requests, cancels browser speech, pauses audio, and revokes
   object URLs.

## Provider Behavior

Browser:

- Requires browser speech synthesis support.
- Uses unsaved `voice` from the form.
- Applies unsaved playback speed to the utterance rate.

tldw:

- Requires unsaved `tldwTtsModel` and `tldwTtsVoice`.
- Uses unsaved response format, synthesis speed, language, and normalization
  options where the existing synthesis path already supports those fields.
- Provider-specific extras may be passed only for controls the current
  `/audio/speech` path already consumes; do not invent new backend semantics for
  preview.
- Sends `stream=false` for preview.

ElevenLabs:

- Requires unsaved API key, model, and voice.
- Must not persist the key or key validation result.
- Uses unsaved model and voice.

OpenAI-compatible:

- Requires unsaved model and voice.
- Previews those values through the existing server speech API.
- Does not promise to validate unsaved frontend Base URL/API key unless the
  implementation proves the current path supports those fields.

## Error Handling

- Use existing audio error classification where possible.
- Keep provider errors sanitized; do not show raw API keys or raw backend
  payloads.
- Treat browser autoplay/playback rejection as a playback issue, not a provider
  synthesis failure.
- Preview failures do not call `setTTSSettings`, do not update key validation
  fields, and do not change saved TTS state.

## Testing

Add focused unit coverage around `TTSModeSettings`:

- Preview uses unsaved form values.
- Preview does not call `setTTSSettings`.
- Browser preview uses the unsaved browser voice and playback speed.
- Missing required provider fields block preview.
- Stop/unmount cleanup cancels or revokes active preview resources.

No broad e2e test is required for this slice unless existing `/settings/speech`
route smoke coverage needs a stable selector. Add
`data-testid="tts-preview-button"` for targeted coverage and future smoke tests.

## Implementation Notes

Keep the change local to the shared settings/TTS helper path. Do not add a
generic preview framework unless the local controller becomes materially harder
to read than a tiny helper.
