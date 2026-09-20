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

## Follow-up review corrections

A second review found four gaps after the initial commit. All four are corrected:

- Blank Persona voices now stay unset in the browser. The server uses the global
  configured voice only for its configured default provider; other providers
  retain their own defaults. Explicit voices are preserved. The global `tldw`
  audio alias does not cause a Kitten voice to cross into legacy Persona Kokoro.
- Kitten preparation resolves the voice against the loaded runtime. Invalid or
  stale voices fail before recording, without generating speech.
- Speech generators close before their credential scope exits, including an
  error after partial audio. Such errors discard the incomplete output.
- Both guides and Live itself explain Disconnect → Connect after saving voice
  defaults. Live displays its current provider, model selection and voice; the
  Profiles preview labels an omitted voice as Provider default.

The failure-first run reproduced five backend and seven frontend failures. After
correction, 278 focused Python tests and 116 frontend tests passed. The regressions
exercise configured/default/explicit voice precedence, cross-provider isolation,
real Kitten voice resolution with model I/O controlled, generator cleanup order,
partial-output rejection and cancellation. Bandit found no issues in the changed
production Python module; targeted Ruff and Black passed. Frontend ESLint found
zero errors and two existing warnings in unchanged hook dependency code.

Two scoped independent re-reviews reported no residual findings. Both Published
mirrors match their canonical guides, local links resolve, and MkDocs built with
the same local serial-plugin workaround. The running browser visibly showed the
new reconnect guidance and `browser · default model · default voice` summary.
This follow-up did not open a microphone or repeat physical speech playback.

Logs: `/private/tmp/persona-review-fixes-final-python.log`,
`/private/tmp/persona-review-fixes-final-frontend.log`,
`/private/tmp/persona-review-fixes-bandit.json`, and
`/private/tmp/persona-review-fixes-docs-build.log`. The restarted isolated backend's
source manifest is under `/private/tmp/persona-review-fixes-20260907/`.

## PR #2928 review and CI corrections

Qodo identified six issues that were corrected:

- Kitten model loads no longer overwrite the cached adapter's configured default
  model or revision. Later model-less requests retain the configured selection.
- Public audio health distinguishes an unprepared or failed Kitten runtime from
  one loaded successfully, while lazy registry initialization remains routable
  for selected-model preparation without requiring default-model assets.
- Native browser callbacks carry an utterance generation invalidated before
  cancellation. Synchronous and delayed callbacks from replaced utterances
  cannot cancel their replacement or finish its turn.
- Provider catalog failures are visible and retryable, preserving the selected
  provider and unsaved form edits. Stale requests cannot replace newer results.
- The normalization helper and endpoint wrappers have docstrings.
- Kitten preparation fixtures and tests have explicit parameter and return types.

The seventh finding claimed missing test category markers. The existing
module-level `pytestmark = pytest.mark.unit` already classifies the file;
`pytest --collect-only -m unit` selected all 14 current preparation cases.

CI also exposed two integration omissions. The documentation checker treated
external Chatbook URL paths as local server paths; URL exclusion now preserves
missing-local-path detection, including repeated local slash separators. The
OpenAPI fingerprint was regenerated and frontend types rebuilt. Removing only
the optional `PersonaVoiceDefaults.tts_model` property from the export reproduces
the old fingerprint, confirming that it is the sole schema change.

Failure-first evidence reproduced six backend and five frontend review failures,
five initial documentation failures, and one additional repeated-slash local-path
failure found during independent review. That review found no remaining
actionable issues after the correction. These tests use controlled runtime/model
I/O and native speech API doubles; physical microphone/playback UAT was not
repeated for this review pass.

Final combined checks passed 284 Persona/TTS Python tests and 124 frontend tests.
Additional scoped adapter/health coverage passed 66 tests (overlapping the
combined run), and the corrected documentation checker passed all six cases.
The wider Docs run passed 210 cases; its strict-build subprocess encountered the
existing local macOS semaphore limit. A separate strict MkDocs build passed using
the date plugin's supported serial mode. CI remains the check of the unmodified
parallel build path. Bandit found zero issues across all five production Python
files changed in this review pass. Scoped lint checks found no new diagnostics.
OpenAPI drift validation and frontend schema generation passed.

Logs: `/private/tmp/pr2928-final-python.log`,
`/private/tmp/pr2928-final-frontend.log`, `/private/tmp/pr2928-final-bandit.json`,
`/private/tmp/pr2928-docs-edge-green.log`, `/private/tmp/pr2928-docs-build.log`,
and `/private/tmp/pr2928-openapi-check.log`.
