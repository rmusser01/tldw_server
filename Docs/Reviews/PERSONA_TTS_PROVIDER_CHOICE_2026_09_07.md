# Persona TTS provider choice correction — 2026-09-07

Task: TASK-13214. Decision: [preserve provider selection](../../backlog/decisions/004-persona-live-tts-provider-selection.md).

The Migu UAT preparation path incorrectly required Kokoro while Persona offered
other speech providers. This correction preserves configured provider, optional
model and voice, uses existing effective audio credential/policy resolution,
and disables cross-provider fallback. The legacy Persona `tldw` alias remains
Kokoro for saved-profile compatibility. Browser speech uses native synthesis.

Preparation validates the selected provider request and loads local assets where
its adapter supports readiness checks. Kitten's cached loader warms the selected
model without first requiring default-model assets. Preparation does not generate
speech. Provider clients created for credential overrides are closed on completion,
validation/init failure, Stop and cancellation; cached local models remain owned
by the registry. Chat credentials, moderation and budgets remain enforced at the
existing authenticated Chat dispatch boundary. Preparation checks the Chat target,
not the availability of a static server API key.

## Verification

- 265 focused Python tests passed: Persona routing/model/voice/readiness/WS,
  real authenticated Chat admission boundary, credential runtime, Kitten adapter
  selection and request-adapter lifecycle.
- 111 focused frontend tests passed: defaults, provider catalog, model transport,
  browser exceptions and unavailable explicit voices, ownership/Stop and Persona panel localization guards.
- Initial provider regressions failed on the original restriction; subsequent
  invalid model/voice and credential-override tests failed before their fixes.
  Four real-registry Kitten model regressions failed before selected-model loading.
- Bandit: zero findings across all seven changed production Python files.
- New Python modules/tests pass targeted Ruff and Black checks. Existing large
  production files retain baseline formatting/lint debt; no broad reformat was
  applied. Frontend ESLint: zero errors, 58 pre-existing warnings, matching HEAD.
- MkDocs build passed with the existing date plugin's parallel processing disabled
  only for this local build. Canonical/Published guide mirrors match; guide links
  were checked.

## Real runtime checks

Used isolated source worktree `/private/tmp/tldw-server-migu-buddy-uat`, backend
on `127.0.0.1:9101`, WebUI on `127.0.0.1:18384`. The backend was restarted from
the corrected code; its launcher recorded revision, diff and application source
hashes. Existing normal STT selection was Parakeet ONNX. No configuration secrets
are included in this record.

1. Real Persona Profiles UI loaded the server provider catalog, selected `browser`,
   cleared the previous Kokoro voice, and saved assistant defaults successfully.
2. A real authenticated Persona WebSocket session prepared Parakeet with browser
   TTS in 0.06 seconds. A controlled committed transcript entered the existing
   DeepSeek Chat route and returned “Browser speech is working. The blue notebook
   is ready.” Voice Stop was acknowledged and session Stop returned HTTP 200.
   No server audio bytes were emitted for the browser provider.
3. A separate browser harness bundled the production voice controller and used
   **real native SpeechSynthesis**. Its input/WS boundary was controlled, with no
   microphone access or Chat call. Native `start` then `end` occurred; controller
   states were idle → listening → thinking → speaking → idle, with active/ready,
   native speaking and pending all false afterward. Stop interrupted a second
   long reply, produced native `error:interrupted`, and left no warning or queued
   speech. Native synthesis was observed, not replaced with a mock.

Raw local artifacts, harness source and server identity are under
`/private/tmp/persona-provider-uat-20260907/`. Test logs are
`/private/tmp/persona-provider-final-python.log`,
`/private/tmp/persona-provider-final-frontend.log` and
`/private/tmp/persona-provider-i18n.log`.

## Limits and remaining work

These checks do not establish human audibility, microphone transcription accuracy,
or a single physical end-to-end voice turn for every provider. OpenAI/ElevenLabs
and gateway contracts were verified with controlled adapters/credentials, not paid
remote synthesis calls. Remote provider readiness may still fail at actual
synthesis. A configured native browser voice that is not in the current voice
list fails clearly instead of substituting another voice; retry after browser
voices load or choose an installed voice. Floating Buddy animation qualification and previously tracked voice
responsiveness follow-ups remain separate.

The broader audio defaults resolver has its own `tldw` alias/default policy;
Persona deliberately preserves its existing alias instead of silently adopting
a different provider default. This correction does not redefine other audio
surfaces or claim a common alias across the server and Chatbook.
