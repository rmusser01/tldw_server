# PRD: TTS/STT WebUI And Extension Workflow Remediation

Owner: Product / WebUI / Audio
Status: Draft
Date: 2026-05-18
Backlog: TASK-427

## 1. Executive Summary

The current TTS and STT surfaces are functional but do not yet behave like a reliable speech workflow for first-time exploration or expert comparison. The WebUI exposes useful TTS and STT controls, but provider readiness, model capability, setup recovery, result provenance, and repeat-use patterns are not consistently visible. The browser extension also diverges from the WebUI: extension `#/stt` currently opens the combined speech page instead of the dedicated STT comparison page.

This PRD defines one staged remediation program for WebUI and extension TTS/STT only. It does not replace the existing backend TTS or STT module PRDs. It focuses on visible user workflows: finding the pages, understanding setup, trying providers and models, comparing outputs, saving preferred configurations, and repeating successful workflows.

The recommended product direction is incremental:

1. Correct route parity and visible mismatch bugs first.
2. Add readiness and capability disclosure using current backend sources, with explicit unknown states where metadata is incomplete.
3. Make comparison runs auditable by preserving the configuration that produced each audio output or transcript.
4. Add per-user server-side presets after the metadata model is stable.
5. Treat Browser TTS as a no-setup escape hatch, not as a first-class provider in backend comparison.

The current experience partially supports first-time exploration and power-user comparison. It lets users generate or transcribe, but it does not yet help them understand why a provider is unavailable, what a model can do, which settings produced a result, or how to reuse a successful setup.

## 2. Evidence Base

### 2.1 Product Decisions Already Resolved

The following decisions are treated as settled inputs for this PRD:

1. Extension `#/stt` should match the WebUI `/stt` workflow rather than remain a quick dictation-only page.
2. Browser TTS is a no-setup escape hatch and preview path, not a first-class backend comparison provider.
3. TTS/STT presets should be per-user server state.
4. STT model capability metadata and comparison metadata must be grounded in code inspection and existing backend contracts, not assumed.

### 2.2 Observed WebUI And Extension Surfaces

Evidence comes from the prior browser-observed UX/HCI audit plus code inspection in this repository.

Observed routes and surfaces:

- WebUI `/tts`: TTS Playground with provider/model/voice controls, presets, configuration drawer, render strips, and history.
- WebUI `/stt`: STT Playground with record/upload, model multi-select, settings, comparison cards, and local history.
- WebUI `/speech`: combined round-trip/speak/listen page.
- WebUI `/settings/speech`: speech settings with STT model health and TTS provider settings.
- Extension `#/tts`: routes to `SpeechPlaygroundPage initialMode="listen"`.
- Extension `#/stt`: routes to `SpeechPlaygroundPage initialMode="speak"`, not the dedicated `SttPlaygroundPage`.

Key source files:

- `apps/packages/ui/src/routes/option-tts.tsx`
- `apps/packages/ui/src/routes/option-stt.tsx`
- `apps/tldw-frontend/extension/routes/option-tts.tsx`
- `apps/tldw-frontend/extension/routes/option-stt.tsx`
- `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx`
- `apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx`
- `apps/packages/ui/src/components/Option/STT/ComparisonPanel.tsx`
- `apps/packages/ui/src/hooks/useComparisonTranscribe.ts`
- `apps/packages/ui/src/components/Option/Speech/RenderStrip.tsx`

Existing product and review documents:

- `Docs/Product/PRD_TTS_Speech_UX_Upgrades.md`: TTS reliability, long-form generation, progress, history, and TTS UX upgrades.
- `Docs/Product/STT_Module_PRD.md`: current STT backend contract, provider registry, normalized artifacts, WS control, retention, and metrics.
- `Docs/Product/TTS_Module_PRD.md`: TTS backend architecture, provider registry, voices, validation, and endpoints.
- `Docs/Reviews/TTS_UX_REVIEW.md`: older TTS UX review. Some findings are stale, but provider naming, Browser TTS limitations, voice cloning visibility, output format clarity, and normalization remain useful.
- `Docs/Reviews/WEBUI_EXTENSION_UX_HCI_AUDIT_2026_05_17.md`: broader WebUI/extension audit with `/speech`, `/stt`, `/tts`, and `/audio` findings.

### 2.3 Issue-To-Phase Traceability

| Observed issue or improvement | Severity | Phase |
|---|---|---|
| Extension `#/stt` does not use the dedicated WebUI STT comparison page | P1 | Phase 1 |
| TTS route can show provider/model/voice combinations that appear inconsistent | P1 | Phase 1 |
| Settings copy points users to the wrong speech settings route | P2 | Phase 1 |
| Missing first-run page orientation and semantic titles weaken discoverability and accessibility | P2 | Phase 1 |
| Missing credential, model, and microphone recovery states are not actionable enough | P1 | Phase 1 and Phase 2A |
| TTS/STT readiness and health are not visible before users run actions | P1 | Phase 2 |
| STT model selector drops catalog labels, descriptions, and capability context | P2 | Phase 2 |
| STT capability metadata is not fully authoritative at per-model level | P1 risk | Phase 2A and Phase 2B |
| Browser TTS can be mistaken for a comparable server provider | P2 | Phase 2 and Phase 3 |
| TTS and STT results do not preserve enough visible provenance for expert comparison | P1 | Phase 3 |
| Comparison metadata such as backend latency, artifact id, cost, and model version can be unavailable | P1 risk | Phase 3 |
| Presets are not per-user server state for TTS/STT workflows | P2 | Phase 4 |
| Local history, server history, presets, and comparison runs are easy to conflate | P1 risk | Phase 4 |
| Batch and repeat-use workflows require too much manual rebuilding | P2 | Phase 5 |
| `/audio` route ambiguity could pull the work into an audio hub redesign | P2 risk | Phase 0 |

### 2.4 Current Backend Capability Sources

TTS sources available today:

- `GET /api/v1/audio/providers`: provider capability surface exposed by `audio_tts.py`.
- `GET /api/v1/audio/voices/catalog`: voice catalog across providers.
- TTS history schemas expose provider, model, voice, format, duration, generation time, params, segments, status, favorite, output id, and artifact state.
- The non-streaming `/api/v1/audio/speech` response returns audio bytes. It records history server-side, but the frontend may not receive a history id or rich metadata in the same response.

STT sources available today:

- Static model catalog from `transcription_models.py`, including labels and descriptions.
- Frontend currently consumes mostly model names through `useTranscriptionModelsCatalog`, which drops category and description detail.
- `GET /api/v1/audio/transcriptions/health`: model health, availability, on-demand state, message, estimated size, and warm-state fields.
- Provider registry capabilities in `stt_provider_adapter.py`: provider name, batch support, streaming support, diarization support, and notes.
- REST transcription response schema includes text, language, duration, words, and segments. Verbose formats can expose more timing detail.
- Internal normalized STT artifacts are richer than the public OpenAI-compatible response and include text, language, segments, diarization, usage, and metadata.

Important metadata limitation:

STT capability data is not fully authoritative at per-model level today. Some fields are provider-level, some come from model health, and some are static catalog descriptions. The UI must show metadata source and confidence rather than pretending every model has exact, live capability data.

## 3. Problem Statement

First-time users need to understand what TTS and STT do, whether their system is ready, what setup is missing, and how to recover from common failures. Today they can reach the pages, but the pages do not consistently explain provider readiness, missing credentials, model download state, microphone permission recovery, or why a selected provider/model/voice combination is invalid.

Experienced users need to compare providers, voices, models, formats, latency, quality, and STT options. Today they can run some comparisons, but the result cards do not preserve enough visible provenance, comparison history is fragmented, and server-side reusable presets do not exist for TTS/STT workflows.

The product problem is not that speech capabilities are absent. The problem is that the WebUI and extension do not yet make those capabilities legible, comparable, recoverable, and reusable.

## 4. Goals And Non-Goals

### Goals

1. Make `/tts`, `/stt`, and extension equivalents understandable on first visit.
2. Make provider/model/voice readiness visible before a user runs a request.
3. Make TTS and STT failures actionable without exposing secrets or overwhelming users with backend detail.
4. Support quick experimentation across TTS providers, voices, output formats, and quality presets.
5. Support STT comparison across available models and exposed options such as language, diarization, timestamps, segmentation, chunking, and streaming where backend capability is known.
6. Preserve result provenance: every generated audio output or transcript should show the configuration that produced it.
7. Add per-user server-side presets for repeat workflows after the configuration and metadata model is stable.
8. Bring extension `#/stt` into parity with WebUI `/stt`, adapted for extension constraints.

### Non-Goals

1. Do not redesign unrelated routes, settings pages, media ingestion, RAG, chat, or admin tools.
2. Do not replace existing TTS or STT backend architecture.
3. Do not require every STT model to expose exact per-model diarization, language, timestamp, or streaming capability in Phase 1.
4. Do not make Browser TTS a first-class backend provider. It remains a no-setup local escape hatch.
5. Do not add broad voice cloning workflows to this PRD unless they are limited to discoverability and provider capability labeling.
6. Do not create a new design system or large visual restyle. Use existing WebUI patterns.

## 5. User Journeys

### 5.1 First-Time User Journey

The first-time user should be able to:

1. Find `/tts` and `/stt` from navigation or audio entry points.
2. Understand that TTS turns text into audio and STT turns speech/audio into text.
3. See whether local and remote TTS providers are ready.
4. See whether STT models are installed, downloadable on demand, unavailable, or missing dependencies.
5. Try a no-setup path:
   - Browser TTS for quick local speech preview.
   - An installed or on-demand STT model if available.
6. Understand missing setup:
   - Missing OpenAI or ElevenLabs credentials.
   - Missing local model files.
   - Model is downloadable but not installed.
   - Microphone permission is denied.
   - Diarization or streaming is unsupported for the selected engine.
7. Recover using visible next steps:
   - Open the correct settings route.
   - Test credentials.
   - Install or warm a model if supported.
   - Retry after permission changes.
   - Choose a supported provider/model alternative.

### 5.2 Power-User Comparison Journey

The experienced user should be able to:

1. Build a comparison run with multiple TTS provider/model/voice/format/speed combinations.
2. Generate outputs and review them side by side.
3. See metadata per result:
   - Provider, model, voice, format, speed, preset.
   - Browser-measured latency in early phases.
   - Backend duration, generated file size, generation time, history id, and artifact link when the API can supply them.
   - Error category and recovery action when generation fails.
4. Build a comparison run with multiple STT model/configuration combinations.
5. Compare STT results by text, segments, timestamps, diarization, language, duration, word count, and latency where exposed.
6. Repeat a successful run with one click.
7. Save preferred setups as per-user server-side presets.
8. Keep track of what configuration produced which result.

## 6. Product Requirements By Phase

### Phase 0: Scope Alignment And Baseline Contract

Goal: create a shared product and engineering contract before UI changes begin.

Requirements:

1. Reference existing backend PRDs and avoid duplicating their architecture scope.
2. Define five separate concepts:
   - Capability metadata: what the system says a provider/model can do.
   - Readiness: whether the provider/model can run now.
   - Preset: reusable saved user configuration.
   - Comparison run: a temporary or persisted test session with one or more configurations.
   - Result artifact: generated audio or transcript plus provenance.
3. Inventory the current visible controls and which backend fields they map to.
4. Decide the canonical route relationship:
   - `/tts`: dedicated text-to-speech generation and comparison.
   - `/stt`: dedicated speech-to-text transcription and comparison.
   - `/speech`: combined workflow for users who want speak/listen together.
   - `/audio`: clear alias or redirect to `/speech` for this PRD. A standalone audio hub is out of scope and requires a separate decision/spec.
5. Mark Browser TTS as "Browser preview" or equivalent copy anywhere it appears.

Acceptance tests:

1. Product spec references `PRD_TTS_Speech_UX_Upgrades.md`, `STT_Module_PRD.md`, and `TTS_Module_PRD.md`.
2. Implementation plan derived from this PRD has separate tasks for route parity, readiness, comparison, presets, and persistence.
3. No phase depends on unrelated app-wide redesign.

### Phase 1: Correctness, Route Parity, And Obvious Recovery

Goal: remove visible contradictions and make the current UI truthful.

Requirements:

1. Extension `#/stt` must expose the same dedicated STT comparison experience as WebUI `/stt`, adapted for extension viewport constraints.
2. WebUI `/tts` and extension `#/tts` should both behave as dedicated TTS views:
   - Mode switcher hidden or constrained when route is TTS-specific.
   - Provider/model/voice controls must not silently mix one provider with another provider's local model or voice.
3. Fix wrong settings copy from speech pages. If the correct destination is `/settings/speech`, copy must point there.
4. Add semantic page titles:
   - "Text to Speech"
   - "Speech to Text"
5. Add first-run empty state copy:
   - TTS: "Choose a provider, enter text, and generate audio. Browser preview works without server setup."
   - STT: "Upload audio or record from your microphone, then compare transcription models."
6. Add permission recovery for microphone denial:
   - Explain that the browser blocked microphone access.
   - Include retry and browser settings guidance.
7. Improve visible credential/model errors:
   - Missing credentials: "This provider needs an API key before it can generate audio."
   - Missing model: "This model is not installed."
   - Unavailable local engine: "The local engine is configured but not ready."
   - Unsupported capability: "This model does not support diarization."

Acceptance tests:

1. WebUI `/stt` and extension `#/stt` render the same core STT comparison workflow.
2. Changing a TTS provider updates or resets incompatible model and voice controls.
3. A TTS render strip never displays a provider paired with an incompatible model/voice from another provider.
4. Settings links from speech pages navigate to `/settings/speech`.
5. A denied microphone permission state is visible, actionable, and keyboard reachable.

### Phase 2: Readiness And Capability Disclosure

Goal: show users what can run now and what each visible option means.

Requirements:

1. Add a compact readiness summary at the top of `/tts` and `/stt`.

Phase 2A must use current APIs and client-side composition first:

2. TTS readiness should use:
   - `/api/v1/audio/providers`
   - `/api/v1/audio/voices/catalog`
   - provider-specific settings state where already available
3. STT readiness should use:
   - `/api/v1/audio/transcriptions/health`
   - static model catalog labels and descriptions
   - existing frontend catalog data
4. Capability rows must show metadata confidence:
   - Confirmed by health check.
   - Inferred from static catalog.
   - Unknown.
5. STT model selector should include useful labels, not only raw model ids:
   - Installed or available on demand.
   - Local or remote when known.
   - Batch support when known.
   - Streaming support when known.
   - Diarization support when known.
   - Timestamp or segment support when known.
   - Language notes when available.
6. TTS provider selector should include:
   - Local/cloud/browser type.
   - Credentials required.
   - Supports streaming.
   - Supported formats.
   - Voice catalog state.
7. Browser TTS must be labeled as a local browser preview:
   - No server history guarantee.
   - No provider latency comparison.
   - Download/export only if the implementation actually supports it.
   - Excluded from server-backed presets, server-backed batch runs, and backend history unless explicitly stored as a `browser_local` client-only configuration that is revalidated in the browser before use.
8. Unknown states must be explicit. Do not hide an option solely because metadata is incomplete.

Phase 2B is optional and should only start if Phase 2A leaves material gaps:

9. If client composition cannot make STT capability states clear enough, add an STT capability summary endpoint.
10. The summary endpoint must combine model health, static catalog labels/descriptions, provider capability data, and source/confidence fields.
11. The summary endpoint must distinguish unsupported from unknown.

Acceptance tests:

1. When a TTS provider lacks credentials, the page shows that before generation.
2. When STT health reports unavailable or on-demand, the model selector displays that state.
3. A model with unknown diarization support is labeled unknown, not unsupported.
4. Readiness states have accessible text, not color-only badges.
5. Extension readiness display fits popup/options layouts without horizontal overflow.
6. Phase 2A can ship without a new backend capability endpoint.
7. If Phase 2B ships, capability fields show their source and never convert unknown into unsupported.

### Phase 3: Comparison Runs And Result Provenance

Goal: make repeated testing and side-by-side evaluation credible.

Requirements:

1. Add a visible "comparison run" model to TTS and STT pages.
2. Each TTS render row must preserve:
   - Provider.
   - Model.
   - Voice.
   - Format.
   - Speed.
   - Preset name if used.
   - Input text hash or short text preview.
   - Created time.
   - Status.
   - Error category if failed.
3. Each TTS result should show available metadata:
   - Audio duration if known.
   - Byte size if known.
   - Browser-measured latency.
   - Backend generation time if returned or linked by history.
   - History id or artifact id only when reliably available from the API.
4. Each STT result card must preserve:
   - Model.
   - Provider if known.
   - Language setting.
   - Diarization setting.
   - Timestamp/segment settings.
   - Chunking/segmentation settings.
   - Audio source identity.
   - Created time.
   - Status.
   - Error category if failed.
5. STT comparison cards should show:
   - Transcript text.
   - Word count.
   - Latency.
   - Language.
   - Duration.
   - Segments/timestamps when available.
   - Diarization labels when available.
6. Add repeat controls:
   - Retry failed result.
   - Duplicate configuration.
   - Generate/transcribe all.
   - Disable one row without deleting it.
7. Add review controls:
   - Copy transcript.
   - Download audio where supported.
   - Save transcript to Notes where current capability exists.
   - Export run summary as JSON or markdown in a later phase if low risk.
8. Keep local in-session comparison useful even before server persistence exists.

Acceptance tests:

1. A user can compare three TTS configurations and see which provider/model/voice produced each audio output.
2. A user can compare three STT models and see which model/configuration produced each transcript.
3. Retrying a failed result preserves the original configuration unless the user changes it.
4. Browser-measured latency is clearly labeled as client-measured if backend latency is unavailable.
5. Result metadata never claims cost, token usage, model version, or backend duration unless the backend returns it.

### Phase 4: Per-User Server Presets And Workflow Reuse

Goal: let users save and reuse preferred speech setups without relying on local-only browser state.

Requirements:

1. Complete a storage ownership decision gate before implementing preset CRUD:
   - Choose the backend service/module that owns audio presets.
   - Choose the database boundary and table or document shape.
   - Define migration behavior for new installs and existing users.
   - Define AuthNZ principal resolution for single-user and multi-user modes.
   - Confirm presets are not stored in TTS history, STT transcript rows, or generated artifact history.
   - Confirm Browser TTS presets are either excluded from server storage or stored only as explicitly marked `browser_local` client configurations.
2. Add per-user server-side preset storage for TTS and STT after the decision gate is complete.
3. In single-user mode, presets must still resolve through the authenticated/default principal rather than a global shared config.
4. Preset types:
   - TTS preset.
   - STT preset.
   - Combined speech preset, optional and only if `/speech` remains a combined workflow.
5. Preset fields:
   - Name.
   - Description.
   - Kind: `tts`, `stt`, or `speech`.
   - Provider/model/voice/config fields.
   - Capability assumptions captured at save time.
   - Created/updated timestamps.
   - Owner user id or principal.
   - Favorite/default flags.
6. Preset UX:
   - Save current setup.
   - Apply preset.
   - Duplicate preset.
   - Rename preset.
   - Delete preset.
   - Mark favorite/default.
7. Applying a preset must validate readiness against current backend state:
   - Available: apply normally.
   - Partially available: apply supported fields and show recovery.
   - Unavailable: show what changed and offer alternatives.
8. Migration:
   - Existing local storage or Dexie comparison history should not break.
   - Server presets start empty unless a deliberate import path is implemented.
   - If import is added, it must be explicit.

Acceptance tests:

1. A written storage ownership decision exists before preset CRUD implementation starts.
2. A signed-in or single-user principal can save a TTS preset and see it after page reload.
3. A saved STT preset can be applied from WebUI `/stt` and extension `#/stt`.
4. A preset referencing an unavailable provider displays a recovery state instead of silently failing.
5. Presets do not leak across users in multi-user mode.
6. Deleting a preset does not delete generated audio or transcripts.
7. Browser-local presets are not treated as portable server presets unless explicitly marked and revalidated.

### Phase 5: Batch, History, And Expert Tools

Goal: improve throughput for heavy repeat users after the core comparison and preset model is stable.

Requirements:

1. Add batch-oriented controls where the backend and UI can support them:
   - Multiple TTS texts against one preset.
   - One text against many TTS configurations.
   - Multiple audio files against one STT preset.
   - One audio file against many STT configurations.
2. Add stronger history filtering:
   - Provider.
   - Model.
   - Voice.
   - Preset.
   - Status.
   - Date.
   - Favorite.
3. Add richer review metadata only when reliably available:
   - Provider latency.
   - Backend generation time.
   - Audio duration.
   - Byte size.
   - Transcript segment count.
   - Diarization speaker count.
   - Cost or token usage only if backend/provider returns it.
4. Add comparison export:
   - Markdown summary for human review.
   - JSON summary for reproducibility.
5. Consider saved comparison templates after server presets prove useful.

Acceptance tests:

1. A user can run a repeated TTS test without manually rebuilding every row.
2. A user can filter prior TTS outputs by provider/model/voice.
3. A user can export a comparison run with enough metadata to reproduce it.
4. Batch controls degrade cleanly when a provider does not support batch behavior.

### 6.6 Quick Wins Versus Larger Product Work

Quick wins that should stay in the first implementation slice:

1. Correct extension `#/stt` route parity.
2. Fix `/settings/speech` copy and links.
3. Add semantic page titles and first-run empty states.
4. Prevent visible TTS provider/model/voice mismatches.
5. Label Browser TTS as "Browser preview."
6. Add plain-language credential, model, and microphone recovery copy.

Medium product improvements:

1. Phase 2A readiness strips for `/tts` and `/stt` using existing APIs.
2. Capability labels with source/confidence where current data supports them.
3. Better STT model labels from the static catalog and health endpoint.
4. Comparison rows that preserve configuration provenance.
5. Retry, duplicate, and disable-row controls.

Larger product and backend work:

1. Phase 2B STT capability summary endpoint if client composition is insufficient.
2. TTS response metadata linkage to history or artifact ids.
3. Per-user server-side preset CRUD.
4. Server-backed comparison run persistence.
5. Batch runs, export, and history filtering.

## 7. UX Requirements

### 7.1 Information Architecture

Canonical route model:

- `/tts`: dedicated TTS generation and comparison.
- `/stt`: dedicated STT transcription and comparison.
- `/speech`: combined round-trip workflow for users who want speak/listen together.
- `/audio`: clear alias or redirect to `/speech` in this PRD. A lightweight audio hub is a separate future decision.
- Extension `#/tts`: same dedicated TTS workflow as WebUI `/tts`, adapted to extension dimensions.
- Extension `#/stt`: same dedicated STT workflow as WebUI `/stt`, adapted to extension dimensions.

Do not require a first-time user to discover `/settings/speech` before they can understand the TTS/STT pages. The pages should explain setup in context and then link to settings for credential/model management.

### 7.2 TTS Page Layout

Recommended structure:

1. Page title and one-line purpose.
2. Readiness strip:
   - Providers ready.
   - Providers needing setup.
   - Browser preview available.
3. Text input and character count.
4. Configuration builder:
   - Provider.
   - Model.
   - Voice.
   - Format.
   - Speed.
   - Preset.
5. Advanced options collapsed by default.
6. Render list for comparison rows.
7. Generated audio/history panel.

Copy examples:

- Browser TTS label: "Browser preview"
- Browser TTS help: "Uses your browser's built-in speech voice. No API key needed. Not comparable to server providers."
- Missing key: "OpenAI needs an API key before audio generation can run."
- Voice mismatch: "This voice belongs to another provider. Choose an OpenAI voice or switch providers."

### 7.3 STT Page Layout

Recommended structure:

1. Page title and one-line purpose.
2. Readiness strip:
   - Models ready.
   - Models available on demand.
   - Models blocked.
3. Source input:
   - Upload audio.
   - Record microphone.
   - Reuse recent source if available.
4. Model/configuration selector.
5. Advanced configuration:
   - Language.
   - Diarization.
   - Timestamps.
   - Chunking.
   - Segmentation.
   - Streaming/file mode if exposed.
6. Compare/transcribe actions.
7. Result cards with transcript, metadata, and actions.

Copy examples:

- Upload empty state: "Upload audio or record from your microphone to compare transcription models."
- Permission denied: "Microphone access is blocked in this browser. Allow microphone access, then retry recording."
- Capability unknown: "Diarization support is unknown for this model."
- On-demand model: "This model can be prepared on demand. First run may be slower."

### 7.4 Progressive Disclosure

Default controls should be safe and short:

- Provider/model/voice or model selection.
- Input source.
- Primary action.
- Last result or comparison rows.

Advanced controls should be grouped:

- TTS: format, speed, normalization, SSML, provider-specific options.
- STT: language, diarization, timestamps, segmentation, embeddings/TreeSeg controls, chunking, streaming behavior.

Do not show low-level STT segmentation fields such as `K`, lambda balance, expansion width, or embeddings provider/model without labels that explain their role. If they remain visible, put them behind "Advanced segmentation" and include concise help text.

### 7.5 Accessibility

Requirements:

1. Page has one semantic `h1`.
2. Status badges include text and ARIA-readable labels.
3. Readiness and error states do not rely on color only.
4. Provider/model/voice selects have labels and descriptions.
5. Result cards expose status and actions in keyboard order.
6. Audio players are reachable by keyboard.
7. Loading states announce progress changes when practical.
8. Extension layouts avoid horizontal scrolling and preserve touch targets.

## 8. Data And State Model

### 8.1 Capability Metadata

Capability metadata describes what a provider/model can do.

Suggested shape:

```json
{
  "scope": "provider | model | voice",
  "id": "string",
  "provider": "string",
  "model": "string",
  "capabilities": {
    "batch": "supported | unsupported | unknown",
    "streaming": "supported | unsupported | unknown",
    "diarization": "supported | unsupported | unknown",
    "timestamps": "supported | unsupported | unknown",
    "languages": ["en"],
    "formats": ["mp3", "wav"]
  },
  "sources": {
    "batch": "provider-capability",
    "diarization": "provider-capability",
    "timestamps": "static-catalog",
    "languages": "static-catalog"
  },
  "updated_at": "iso timestamp"
}
```

The UI may compose this client-side in early phases. A backend endpoint is preferred if backend owners want one authoritative summary.

### 8.2 Readiness

Readiness describes whether the option can run now.

Suggested states:

- `ready`
- `needs_credentials`
- `not_installed`
- `available_on_demand`
- `warming`
- `disabled`
- `unsupported`
- `error`
- `unknown`

Readiness must include user-facing next-step copy.

### 8.3 Preset

Presets are reusable, per-user server-side configurations.

Suggested shape:

```json
{
  "id": "string",
  "user_id": "string",
  "kind": "tts | stt | speech",
  "name": "string",
  "description": "string",
  "config": {},
  "captured_capabilities": {},
  "favorite": false,
  "default": false,
  "created_at": "iso timestamp",
  "updated_at": "iso timestamp"
}
```

Presets should not be stored in TTS generation history. History and presets have different lifecycles.

Browser TTS preset rule:

- Browser TTS settings are not portable server-side presets by default.
- If a browser voice configuration is saved, it must be marked `browser_local`, must not be included in backend batch jobs, and must be revalidated against the current browser voice list before use.

### 8.4 Comparison Run

Comparison runs organize a set of configurations and results.

Suggested shape:

```json
{
  "id": "string",
  "kind": "tts | stt",
  "source": {
    "text_preview": "string",
    "text_hash": "string",
    "audio_source_id": "string"
  },
  "configs": [],
  "results": [],
  "created_at": "iso timestamp"
}
```

Comparison runs may remain local in Phase 3. Server persistence is optional until Phase 4 or Phase 5.

Privacy and retention rules:

- Phase 3 comparison runs are local/in-session by default.
- Full input text, uploaded audio, raw microphone blobs, and transcript source metadata should not be server-persisted unless the user explicitly saves, exports, or submits a server-backed job.
- If server-backed comparison persistence is added later, avoid global reusable hashes for user text or audio. Prefer opaque run ids, short previews, and user-owned artifact references. If hashing is needed for deduplication, it must be scoped so it cannot become a cross-user content fingerprint.

### 8.5 Result Artifact

Result artifacts are generated outputs plus provenance.

TTS result:

- Provider, model, voice, format, speed, preset id.
- Status, error category, recovery action.
- Audio URL/blob reference.
- Audio duration and byte size when known.
- Client latency.
- Backend generation time when returned.
- Server history id or output id only when reliably returned.

STT result:

- Model, provider if known, language, diarization, timestamps, chunking, segmentation.
- Status, error category, recovery action.
- Text, word count, language, duration.
- Segments and speakers when returned.
- Client latency.
- Backend metadata when returned.

## 9. Error, Empty, Loading, And Recovery States

### Error Categories

Use a UI-facing error category before displaying raw backend detail:

- Missing credentials.
- Invalid credentials.
- Provider disabled.
- Provider unavailable.
- Model not installed.
- Model download required.
- Model warming.
- Unsupported capability.
- Unsupported format.
- Permission denied.
- File invalid.
- File too large.
- Network/server failure.
- Unknown error.

Each error should include:

1. Plain-language cause.
2. Next action.
3. Diagnostic detail behind disclosure where safe.

The UI should implement an error-classification layer between backend/client exceptions and visible copy. It should map known backend failures into the categories above and preserve unknown failures as "Unknown error" with safe diagnostics. Required test fixtures should cover missing credentials, invalid credentials, model unavailable, model not installed, permission denied, unsupported capability, file invalid, network/server failure, and unknown error.

### Empty States

TTS empty state:

"Enter text, choose a voice, and generate audio. Browser preview works without server setup. Server providers may need an API key or local model."

STT empty state:

"Upload audio or record from your microphone, then compare transcription models. Some models may need setup or first-run preparation."

### Loading States

TTS:

- "Checking provider readiness..."
- "Loading voices..."
- "Generating audio..."
- "Receiving audio..."

STT:

- "Checking model readiness..."
- "Preparing model..."
- "Uploading audio..."
- "Transcribing..."
- "Comparing results..."

## 10. Backend And API Requirements

This PRD should reuse existing APIs first.

### Reuse First

Use existing endpoints where possible:

- `/api/v1/audio/providers`
- `/api/v1/audio/voices/catalog`
- `/api/v1/audio/transcriptions/health`
- `/api/v1/audio/transcriptions`
- `/api/v1/audio/speech`
- Existing TTS history endpoints under `audio_history.py`

### Possible New Or Extended APIs

Add only if reuse is insufficient:

1. STT capability summary endpoint:
   - Exposes provider capability plus model health plus static catalog metadata.
   - Must include source/confidence per capability field.
   - Belongs to Phase 2B only, after Phase 2A proves existing APIs are insufficient.
2. Audio preset CRUD:
   - `GET /api/v1/audio/presets`
   - `POST /api/v1/audio/presets`
   - `PATCH /api/v1/audio/presets/{id}`
   - `DELETE /api/v1/audio/presets/{id}`
   - Belongs to Phase 4 only, after the storage ownership decision gate is complete.
3. TTS response metadata link:
   - Header or JSON metadata endpoint that lets frontend connect generated audio to server history id/output id.
   - Required only if Phase 3 or Phase 5 needs reliable server-linked result metadata.

### API Guardrails

1. Do not expose secrets in readiness responses.
2. Do not expose local filesystem paths in user-facing payloads.
3. Do not mark a capability unsupported if the backend cannot distinguish unsupported from unknown.
4. Do not make Browser TTS flow depend on server APIs.
5. Keep OpenAI-compatible endpoints compatible. Rich metadata should be additive.
6. Do not persist Browser TTS provider choices as portable server presets unless they are explicitly marked browser-local.

## 11. State Management Requirements

1. Presets are per-user server state.
2. Result comparison runs can be local first, server-backed later.
3. Existing local/Dexie history should continue to load.
4. Server presets must validate current readiness on apply.
5. Extension and WebUI should share as much UI and client contract as practical.
6. Extension must handle offline or server-disconnected states without corrupting saved presets.

## 12. Success Metrics

First-time success:

- User can identify whether TTS/STT is ready within 10 seconds of route render in browser QA, measured from route navigation to visible readiness summary using mock readiness states plus at least one connected local-server run.
- User can run one no-setup or ready provider path without visiting settings when such a path exists.
- Missing credential/model errors lead to the correct settings or setup action.

Power-user success:

- User can compare at least three TTS configurations and identify which settings produced each output.
- User can compare at least three STT configurations and identify which settings produced each transcript.
- User can save and reapply a preset after Phase 4.
- User can repeat or retry a failed comparison row without rebuilding it manually.

Quality gates:

- No provider/model/voice mismatch in visible result rows.
- No color-only readiness indicators.
- No extension horizontal overflow in supported views.
- No raw provider error shown without plain-language summary.

## 13. Validation Plan

Phase 1 validation:

- Unit tests for route wiring and provider/model/voice reset behavior.
- Component tests for empty/error states.
- Browser tests for WebUI `/tts` and `/stt`.
- Extension route test for `#/stt` parity.

Phase 2 validation:

- Mock provider readiness payload tests.
- Mock STT health states: ready, unavailable, on-demand, error.
- Accessibility checks for readiness badges.
- Verify Phase 2A works without adding a new STT capability endpoint.
- If Phase 2B is implemented, add backend/client contract tests for capability source/confidence and unknown-vs-unsupported behavior.

Phase 3 validation:

- TTS comparison run test with three rows.
- STT comparison run test with three results.
- Retry/duplicate action tests.
- Metadata truthfulness tests for unknown/unavailable fields.
- Privacy tests or review checklist proving local comparison runs do not server-persist full text/audio source data without explicit save/export/job submission.

Phase 4 validation:

- Storage ownership decision record reviewed before implementation.
- Backend unit/integration tests for preset CRUD.
- Auth tests for per-user isolation.
- Frontend tests for save/apply/delete preset.
- Extension apply-preset test.
- Browser-local preset handling test, if Browser TTS configurations can be saved.

Manual browser QA:

- WebUI desktop.
- WebUI narrow/mobile viewport.
- Extension options page.
- Extension popup/sidepanel if applicable.
- Credential missing state.
- Microphone permission denied state.
- Local model unavailable state.

Implementation definition of done for each phase:

1. Relevant WebUI and extension routes are browser-verified.
2. Focused component and hook tests cover changed state behavior.
3. API changes, if any, include contract tests and generated client updates.
4. Accessibility basics are checked for headings, labels, keyboard order, and non-color-only status.
5. Documentation or settings copy is updated where route labels or setup guidance changes.
6. Known backend metadata gaps are documented as unknown states, not hidden assumptions.
7. Error mapping tests cover known backend/client failure categories.

## 14. Risks And Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| STT metadata is incomplete or provider-level only | UI may mislead users | Show capability source/confidence and explicit unknown states |
| Presets expand backend scope | Phase 4 could block quick fixes | Keep presets independent from Phases 1-3 |
| Extension parity breaks constrained layouts | STT page may be unusable in extension | Add extension-specific responsive tests |
| TTS history id is not returned with audio response | Result provenance may be incomplete | Use client metadata first; add server-linked metadata only when API supports it |
| Browser TTS gets confused with server providers | Users compare non-equivalent outputs | Label as Browser preview and exclude from default backend comparison |
| Advanced STT controls overload first-time users | Cognitive load remains high | Collapse advanced controls and add plain labels |
| Existing local history conflicts with server presets | Data loss or duplicate states | Keep local history and server presets separate |
| Comparison runs accidentally persist sensitive text or audio metadata | Privacy regression | Keep Phase 3 runs local by default; require explicit save/export/job submission for server persistence |
| Browser TTS presets are treated as portable backend presets | Broken reuse across devices or browsers | Exclude Browser TTS from server presets/batch by default or mark as `browser_local` and revalidate client-side |

## 15. Open Questions

1. After Phase 2A, is a dedicated STT capability summary endpoint still needed?
2. Should comparison runs become server-persisted in Phase 5, or remain local/exportable?
3. Should saved presets be exportable with chatbooks/workspaces later, after per-user server presets ship?
4. Which TTS metadata should be returned directly from `/api/v1/audio/speech` without breaking OpenAI compatibility: request id, history id, generation time, duration, byte size, or artifact id?
5. Which STT result metadata from the internal normalized artifact should become visible on the public WebUI comparison cards?
6. What default STT options should be shown for first-time users, given provider differences in diarization, timestamps, and streaming?
7. Should voice cloning discoverability be included in this remediation, or handled by the existing TTS UX upgrade PRD?
8. If a standalone `/audio` hub is still desired, what separate scope and success criteria should govern it outside this PRD?

## 16. Implementation Boundaries

Implementation should proceed in small PR slices:

1. Route parity and visible copy fixes.
2. TTS provider/model/voice state correctness.
3. Phase 2A readiness strip and capability labels using existing APIs.
4. Phase 2B STT capability summary endpoint only if Phase 2A is insufficient.
5. Comparison run provenance and local privacy guardrails.
6. Server preset ownership decision gate, then preset API and UI.
7. Batch/history/export enhancements.

Each slice should preserve existing APIs unless the phase explicitly requires an additive backend endpoint.

Do not begin large backend refactors, storage redesigns, provider rewrites, or app-wide navigation redesign from this PRD. If a backend/API issue blocks a visible workflow, document it as a narrow prerequisite and keep the implementation tied to the blocked user journey.
