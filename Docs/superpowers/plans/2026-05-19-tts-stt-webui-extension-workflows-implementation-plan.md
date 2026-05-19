# TTS/STT WebUI And Extension Workflow Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the staged TTS/STT WebUI and extension workflow remediation defined in the hardened PRD, so first-time users can recover from setup gaps and experienced users can compare speech configurations with trustworthy provenance.

**Architecture:** Preserve the existing route and backend ownership model. Use shared `apps/packages/ui` components for WebUI and extension parity, derive readiness from existing audio APIs in Phase 2A, add any STT capability endpoint only behind the Phase 2B gate, and add per-user server presets only after a storage ownership decision gate. Keep Browser TTS as a local browser preview path, not a server-backed provider.

**Tech Stack:** React, Next.js route wrappers, extension route wrappers, shared `apps/packages/ui`, TanStack Query, Plasmo storage, Dexie local history, existing tldw audio API client, Vitest, React Testing Library, Playwright, FastAPI and pytest only for optional Phase 2B or Phase 4 backend slices.

---

## Source Documents

- Hardened PRD: `Docs/superpowers/specs/2026-05-18-tts-stt-webui-extension-workflows-prd-design.md`
- Backlog task for this implementation plan: `TASK-428`
- PRD creation and hardening task: `TASK-427`
- Related route identity plan: `Docs/superpowers/plans/2026-05-17-webui-audio-routes-implementation-plan.md`
- Related capability and error state plan: `Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md`
- Related parent UX remediation plan: `Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md`

## Current Evidence Snapshot

The plan is grounded in the current repository state, not invented routes or backend behavior.

Observed route files:

- `apps/packages/ui/src/routes/option-stt.tsx` lazy-loads `SttPlaygroundPage` for WebUI `/stt`.
- `apps/packages/ui/src/routes/option-tts.tsx` renders `SpeechPlaygroundPage lockedMode="listen" hideModeSwitcher` for WebUI `/tts`.
- `apps/tldw-frontend/extension/routes/option-stt.tsx` currently renders `SpeechPlaygroundPage initialMode="speak"`, which is the known parity defect.
- `apps/tldw-frontend/extension/routes/option-tts.tsx` currently renders `SpeechPlaygroundPage initialMode="listen"`, which should be route-locked like WebUI `/tts`.

Observed TTS files:

- `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx`
- `apps/packages/ui/src/components/Option/Speech/RenderStrip.tsx`
- `apps/packages/ui/src/hooks/useTtsProviderData.ts`
- `apps/packages/ui/src/services/tldw/audio-providers.ts`
- `apps/packages/ui/src/services/tldw/domains/models-audio.ts`

Observed STT files:

- `apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx`
- `apps/packages/ui/src/components/Option/STT/ComparisonPanel.tsx`
- `apps/packages/ui/src/components/Option/STT/RecordingStrip.tsx`
- `apps/packages/ui/src/components/Option/STT/InlineSettingsPanel.tsx`
- `apps/packages/ui/src/components/Option/STT/HistoryPanel.tsx`
- `apps/packages/ui/src/hooks/useComparisonTranscribe.ts`
- `apps/packages/ui/src/hooks/useTranscriptionModelsCatalog.ts`
- `apps/packages/ui/src/hooks/useTldwAudioStatus.tsx`
- `apps/packages/ui/src/services/tldw/domains/models-audio.ts`

Known backend sources:

- TTS readiness inputs exist today through `/api/v1/audio/providers` and `/api/v1/audio/voices/catalog`.
- STT model list exists today through `/api/v1/media/transcription-models`.
- STT health exists today through `/api/v1/audio/transcriptions/health?model=...`.
- STT REST transcription can return `text`, `language`, `duration`, `words`, and `segments` depending on response format and backend response shape.
- STT per-model capability metadata is not fully authoritative today; some values are static catalog information, some are provider-level, and some are health-derived.

## Scope Boundaries

- Do not replace the existing audio backend architecture.
- Do not redesign unrelated WebUI routes.
- Do not make `/audio` a new hub in this plan. Keep it as an alias or redirect to `/speech` unless a separate product decision changes that.
- Do not treat Browser TTS as a server provider. It is a local "Browser preview" escape hatch.
- Do not implement server-side presets until the Phase 4 storage ownership decision is written and approved.
- Do not implement a new STT capability endpoint unless Phase 2A leaves material UX gaps that cannot be solved with existing APIs.
- Do not claim cost, provider latency, model version, backend generation time, history id, or artifact id unless the API returns or reliably links that value.
- Do not erase or migrate existing local/Dexie history as part of route or comparison work.

## Release Strategy

Ship this work as reviewable slices. Each slice should get its own Backlog task before file edits begin.

1. **Slice 1: Route parity and TTS configuration truthfulness**
   - Fix extension route ownership, TTS mode locking, incorrect settings copy, first-run labels, and provider/model/voice mismatch bugs.
   - Frontend only.
2. **Slice 2: Phase 2A readiness and error classification using existing APIs**
   - Add readiness summaries and explicit metadata confidence labels without backend changes.
   - Frontend only unless evidence proves an existing API client type needs adjustment.
3. **Slice 3: Comparison provenance and repeat controls**
   - Add visible TTS/STT result configuration metadata, client-measured latency labels, retry/duplicate/disable controls, and privacy-safe text previews or hashes.
   - Frontend only.
4. **Slice 4: Optional Phase 2B STT capability endpoint**
   - Only if Slice 2 cannot show material STT capability states clearly with existing APIs.
   - Backend plus frontend contract.
5. **Slice 5: Phase 4 preset storage decision and server CRUD**
   - First deliver a storage ownership decision document. Implement CRUD only after approval.
   - Backend plus frontend, likely split again into API, UI, and extension parity tasks.

## Shared Data Contracts

### TTS Render Configuration

Add a small tested helper so provider-specific defaults and labels are no longer scattered through `SpeechPlaygroundPage`.

New file:

- `apps/packages/ui/src/components/Option/Speech/tts-render-config.ts`

Suggested shape:

```ts
export type TtsProviderId = "browser" | "tldw" | "openai" | "elevenlabs" | string

export type TtsRenderConfigSource =
  | "settings"
  | "voice_picker"
  | "render_strip"
  | "browser_preview"
  | "preset"

export type TtsRenderConfig = {
  provider: TtsProviderId
  model?: string
  voice?: string
  format: string
  speed: number
  source: TtsRenderConfigSource
}

export type TtsProviderDefaults = {
  provider: TtsProviderId
  tldwModel?: string
  tldwVoice?: string
  openAiModel?: string
  openAiVoice?: string
  elevenLabsModel?: string
  elevenLabsVoice?: string
  format?: string
  speed?: number
}

export function buildTtsRenderConfig(input: TtsProviderDefaults): TtsRenderConfig {
  // Tests define the exact branch behavior before implementation.
}
```

Required behavior:

- `provider === "browser"` produces a config labeled as Browser preview and does not inherit `tldw` model or voice.
- `provider === "openai"` uses OpenAI model and voice defaults only.
- `provider === "elevenlabs"` uses ElevenLabs model and voice defaults only.
- `provider === "tldw"` uses tldw model and voice defaults only.
- Unknown/custom providers do not borrow another provider's voice unless the voice catalog explicitly identifies the provider.

### STT Model Option Metadata

Extend the model catalog hook without breaking existing call sites.

Modified file:

- `apps/packages/ui/src/hooks/useTranscriptionModelsCatalog.ts`

Suggested additions:

```ts
export type MetadataConfidence = "health" | "static_catalog" | "provider" | "unknown"

export type SttCapabilityValue = "supported" | "unsupported" | "unknown"

export type SttModelOption = {
  id: string
  label: string
  description?: string
  category?: string
  availability: "ready" | "on_demand" | "unavailable" | "unknown"
  readinessMessage?: string
  capabilities: {
    batch: SttCapabilityValue
    streaming: SttCapabilityValue
    diarization: SttCapabilityValue
    timestamps: SttCapabilityValue
    segments: SttCapabilityValue
  }
  sources: Partial<Record<keyof SttModelOption["capabilities"] | "availability", MetadataConfidence>>
}
```

Required behavior:

- Preserve `serverModels: string[]` for current consumers.
- Add `modelOptions: SttModelOption[]` for enhanced selectors and readiness UI.
- Do not convert missing metadata into "unsupported".
- Fetch health in a bounded way. Start with selected/default/visible models instead of launching unbounded health requests for every catalog model.

### Audio Error Classification

Use or extend existing shared capability/error-state patterns instead of creating isolated one-off alert copy.

Potential new file:

- `apps/packages/ui/src/components/Option/Audio/audio-error-classification.ts`

Suggested shape:

```ts
export type AudioErrorCategory =
  | "missing_credentials"
  | "missing_model"
  | "engine_unavailable"
  | "unsupported_capability"
  | "microphone_blocked"
  | "network"
  | "timeout"
  | "unknown"

export type AudioErrorClassification = {
  category: AudioErrorCategory
  title: string
  recovery: string
  settingsHref?: "/settings/speech"
}

export function classifyAudioError(error: unknown): AudioErrorClassification {
  // Map known API and browser errors to stable UX categories.
}
```

Required behavior:

- Preserve raw debug detail for development logs where current patterns allow it.
- User-facing cards show category, plain-language recovery, and safe next action.
- Do not expose API keys, credential names beyond provider names, or raw stack traces.

### Comparison Provenance

Extend current TTS and STT result structures with visible, privacy-aware metadata.

Suggested STT extension:

```ts
export type SttComparisonConfig = {
  model: string
  language?: string
  task?: string
  responseFormat?: string
  timestampGranularities?: string[]
  segmentationEnabled?: boolean
  diarizationRequested?: boolean
}

export type SttComparisonMetadata = {
  createdAt: string
  audioSourceLabel: string
  audioSizeBytes?: number
  clientLatencyMs?: number
  language?: string
  durationSeconds?: number
  segmentCount?: number
  errorCategory?: AudioErrorCategory
}
```

Suggested TTS extension:

```ts
export type TtsResultMetadata = {
  createdAt: string
  inputTextPreview?: string
  inputTextHash?: string
  inputTextLength: number
  clientLatencyMs?: number
  audioSizeBytes?: number
  audioDurationSeconds?: number
  backendGenerationMs?: number
  historyId?: string
  artifactId?: string
  errorCategory?: AudioErrorCategory
}
```

Privacy rules:

- Short text previews must be visibly labeled as previews.
- Hashes should be deterministic only for local comparison unless a server persistence decision explicitly requires otherwise.
- Do not store full input text in new comparison metadata unless the existing history path already stores it for that feature.
- Browser TTS metadata must remain `browser_local` and must be revalidated in the browser before reuse.

## Stage 0: Baseline And Planning Handoff

**Goal:** Prepare implementation slices without broadening scope.

**Success Criteria:**

- The implementation PRD and this plan are linked from the new implementation Backlog tasks.
- Current route ownership and test names are rechecked before coding.
- No code changes happen under this task unless explicitly changing this plan.

**Tests:** Documentation verification only for this plan.

**Status:** Complete via `TASK-428`; follow-up implementation tasks were created for Slice 1 and Slice 2A before file edits.

### Tasks

- [ ] Create follow-up Backlog tasks for Slice 1 through Slice 3 before implementation starts.
- [ ] Create optional Backlog tasks for Phase 2B and Phase 4 only when gates are satisfied.
- [ ] Confirm current route files still match the evidence snapshot.
- [ ] Keep this plan as the implementation source of truth for the first coding slice.

## Stage 1: Route Parity, Copy, And TTS Configuration Truthfulness

**Goal:** Remove visible contradictions and make `/tts`, `/stt`, extension `#/tts`, and extension `#/stt` align with the hardened PRD.

**Success Criteria:**

- Extension `#/stt` renders the dedicated `SttPlaygroundPage`.
- Extension `#/tts` renders the same locked TTS workflow as WebUI `/tts`.
- TTS route copy uses "Text to Speech" and STT route copy uses "Speech to Text" where a route-specific title is shown.
- Settings copy links or points to `/settings/speech`, not "Settings -> General -> Speech-to-Text".
- TTS render rows never pair a provider with another provider's model or voice.
- Browser TTS is labeled "Browser preview" and described as no-setup/local.

**Tests:**

- `cd apps/packages/ui && bunx vitest run src/routes/__tests__/option-audio-route-identity.test.tsx`
- `cd apps/packages/ui && bunx vitest run src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx src/components/Option/Speech/__tests__/RenderStrip.test.tsx`
- `cd apps/tldw-frontend && bun run test:extension -- extension/__tests__/audio-route-parity.guard.test.tsx`

**Status:** Complete via `TASK-429` / commit `9fcf83198`.

### Files

Modify:

- `apps/tldw-frontend/extension/routes/option-stt.tsx`
- `apps/tldw-frontend/extension/routes/option-tts.tsx`
- `apps/packages/ui/src/routes/option-stt.tsx`
- `apps/packages/ui/src/routes/option-tts.tsx`
- `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx`
- `apps/packages/ui/src/components/Option/Speech/RenderStrip.tsx`
- `apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx`

Add or extend tests:

- `apps/tldw-frontend/extension/__tests__/audio-route-parity.guard.test.tsx`
- `apps/packages/ui/src/routes/__tests__/option-audio-route-identity.test.tsx`
- `apps/packages/ui/src/components/Option/Speech/__tests__/tts-render-config.test.ts`
- `apps/packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx`
- `apps/packages/ui/src/components/Option/Speech/__tests__/RenderStrip.test.tsx`

Optional new helper:

- `apps/packages/ui/src/components/Option/Speech/tts-render-config.ts`

### Implementation Steps

- [x] Write failing route parity tests for extension `#/stt` and `#/tts`.
  - `#/stt` must import/render `SttPlaygroundPage`.
  - `#/tts` must pass `lockedMode="listen"` and `hideModeSwitcher` to `SpeechPlaygroundPage`.
- [x] Update `apps/tldw-frontend/extension/routes/option-stt.tsx` to use `SttPlaygroundPage`.
- [x] Update `apps/tldw-frontend/extension/routes/option-tts.tsx` to mirror WebUI `/tts` mode locking.
- [ ] Add `RouteErrorBoundary` to shared `/stt` only if tests confirm it is still missing and the local route-boundary pattern expects it.
- [ ] Add `tts-render-config.ts` and tests for provider-specific defaults. Deferred because Slice 1 fixed provider-specific selection in-place without requiring a new helper.
- [ ] Replace ad hoc provider/model/voice construction in `handleAddRenderStrip` with `buildTtsRenderConfig`. Deferred with the helper above.
- [x] Replace route-level TTS provider strip values so OpenAI and ElevenLabs do not display tldw model or voice values.
- [x] Update `RenderStrip` labels so Browser TTS says "Browser preview" and custom/tldw providers do not mask the provider as the model.
- [x] Fix speech settings copy to `/settings/speech`.
- [x] Add first-run empty copy for dedicated TTS and STT routes using the PRD language.
- [x] Verify keyboard focus order did not regress for route headings, primary inputs, and result rows through focused render tests.

### Stage 1 Commit Guidance

Commit after tests pass:

```bash
git add apps/tldw-frontend/extension/routes/option-stt.tsx \
  apps/tldw-frontend/extension/routes/option-tts.tsx \
  apps/tldw-frontend/extension/__tests__/audio-route-parity.guard.test.tsx \
  apps/packages/ui/src/routes/option-stt.tsx \
  apps/packages/ui/src/routes/__tests__/option-audio-route-identity.test.tsx \
  apps/packages/ui/src/components/Option/Speech \
  apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx
git commit -m "fix speech route parity and tts config provenance"
```

## Stage 2A: Readiness And Capability Disclosure From Existing APIs

**Goal:** Show what can run now and what metadata is known without adding backend endpoints.

**Success Criteria:**

- TTS readiness uses existing provider and voice catalog APIs.
- STT readiness uses existing model catalog and health APIs.
- Capability labels distinguish `supported`, `unsupported`, and `unknown`.
- Capability labels show source or confidence where useful.
- First-run users see setup needs before generation/transcription.
- Extension layouts do not overflow horizontally.

**Tests:**

- `cd apps/packages/ui && bunx vitest run src/hooks/__tests__/useTranscriptionModelsCatalog.test.tsx`
- `cd apps/packages/ui && bunx vitest run src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx src/components/Option/STT/__tests__/ComparisonPanel.test.tsx`
- `cd apps/packages/ui && bunx vitest run src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx src/components/Option/Speech/__tests__/TtsProviderStrip.test.tsx`
- `cd apps/tldw-frontend && bun run test:extension -- extension/__tests__/audio-route-parity.guard.test.tsx`

**Status:** Complete via `TASK-430` / commit `c46563eaa`.

### Files

Modify:

- `apps/packages/ui/src/hooks/useTranscriptionModelsCatalog.ts`
- `apps/packages/ui/src/hooks/useTldwAudioStatus.tsx`
- `apps/packages/ui/src/hooks/useTtsProviderData.ts`
- `apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx`
- `apps/packages/ui/src/components/Option/STT/ComparisonPanel.tsx`
- `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx`
- `apps/packages/ui/src/components/Option/Speech/TtsProviderStrip.tsx`
- `apps/packages/ui/src/services/tldw/domains/models-audio.ts`

Add:

- `apps/packages/ui/src/components/Option/Audio/audio-readiness.ts`
- `apps/packages/ui/src/components/Option/Audio/AudioReadinessStrip.tsx`
- `apps/packages/ui/src/components/Option/Audio/__tests__/audio-readiness.test.ts`
- `apps/packages/ui/src/components/Option/Audio/__tests__/AudioReadinessStrip.test.tsx`
- `apps/packages/ui/src/hooks/__tests__/useTranscriptionModelsCatalog.test.tsx`

### Implementation Steps

- [x] Add failing tests for readiness summary states:
  - TTS provider ready.
  - TTS provider missing credentials.
  - Browser preview available.
  - STT model ready.
  - STT model on demand.
  - STT model unavailable.
  - Unknown diarization support remains unknown.
- [x] Add `audio-readiness.ts` pure functions for formatting readiness and confidence labels.
- [x] Extend `useTranscriptionModelsCatalog` to expose `modelOptions` while preserving `serverModels`.
- [ ] Add typed client responses in `models-audio.ts` for transcription model catalog and health if the existing `any` return makes tests fragile. Deferred because tests remained stable through the hook-level type boundary.
- [x] Compose STT model options from static catalog response and bounded health checks.
- [x] Add readiness strip to `SttPlaygroundPage` above source input.
- [x] Add readiness strip to `SpeechPlaygroundPage` when in locked TTS mode and in combined mode where TTS controls are visible.
- [x] Keep advanced capability details compact, with accessible text on every badge.
- [ ] Add extension-width tests or snapshots to catch horizontal overflow in the readiness strip. Deferred to browser QA Stage 8; the component uses wrapping layout and focused render coverage in Slice 2A.
- [x] Ensure unknown states are visible instead of hidden.

### Phase 2A Guardrails

- Do not add backend code in this stage.
- Do not fetch health for every catalog model at once if the catalog can be large.
- Do not block users from selecting a model solely because metadata is incomplete.
- Do not show "unsupported" unless a source explicitly says unsupported.
- Do not persist readiness assumptions as presets in this stage.

## Stage 3: Audio Error Classification And Recovery

**Goal:** Make missing credentials, missing models, local engine failures, unsupported features, microphone denial, network failures, and timeouts recoverable without raw backend noise.

**Success Criteria:**

- User-facing errors map to stable categories.
- Errors include plain-language recovery and safe settings links.
- Microphone permission denial includes retry and browser settings guidance.
- Result rows preserve error category as comparison metadata.
- Raw error strings do not become the only visible recovery guidance.

**Tests:**

- `cd apps/packages/ui && bunx vitest run src/components/Option/Audio/__tests__/audio-error-classification.test.ts`
- `cd apps/packages/ui && bunx vitest run src/components/Option/STT/__tests__/RecordingStrip.test.tsx src/components/Option/STT/__tests__/ComparisonPanel.test.tsx`
- `cd apps/packages/ui && bunx vitest run src/components/Option/Speech/__tests__/SpeechPlaygroundPage.audio-source.test.tsx src/components/Option/Speech/__tests__/RenderStrip.test.tsx`

**Status:** Complete via `TASK-431`.

### Files

Modify:

- `apps/packages/ui/src/components/Option/STT/RecordingStrip.tsx`
- `apps/packages/ui/src/components/Option/STT/ComparisonPanel.tsx`
- `apps/packages/ui/src/hooks/useComparisonTranscribe.ts`
- `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx`
- `apps/packages/ui/src/components/Option/Speech/RenderStrip.tsx`
- `apps/packages/ui/src/hooks/useMultiRenderState.ts`

Add:

- `apps/packages/ui/src/components/Option/Audio/audio-error-classification.ts`
- `apps/packages/ui/src/components/Option/Audio/__tests__/audio-error-classification.test.ts`

### Implementation Steps

- [x] Write classification tests for known error shapes and browser `NotAllowedError`.
- [x] Implement `classifyAudioError(error)`.
- [x] Apply classification to STT comparison errors in `useComparisonTranscribe`.
- [x] Apply classification to TTS render errors in `useMultiRenderState` or at the closest existing render failure boundary.
- [x] Update `ComparisonPanel` and `RenderStrip` to show category title and recovery copy.
- [x] Add `/settings/speech` as the recovery link for missing credentials or setup where appropriate.
- [x] Add microphone-denied UI to `RecordingStrip` with retry and browser settings guidance.
- [x] Confirm errors remain keyboard reachable and screen-reader-readable.

### Stage 3 Verification Notes

- Added shared classifier coverage for credentials, missing model, microphone permission, network, timeout, engine unavailable, unsupported, and unknown failures.
- Added STT comparison and TTS render recovery links to `/settings/speech` when the classifier returns a settings recovery target.
- Verified with focused Stage 3 and audio readiness/parity suites; full package TypeScript remains blocked by existing unrelated frontend baseline errors outside the touched audio files.

## Stage 4: Comparison Run Provenance And Power-User Controls

**Goal:** Make side-by-side TTS and STT testing credible by showing what configuration produced each output.

**Success Criteria:**

- TTS result rows show provider, model, voice, format, speed, created time, status, and client-measured latency.
- STT result cards show model, language, task, response format, timestamp/segment settings, audio source label, created time, status, and client-measured latency.
- Available response metadata such as language, duration, segment count, byte size, and word count is shown only when actually available.
- Retry preserves the original row configuration.
- Duplicate row creates a new editable row from the original configuration.
- Disable row removes a row from "run all" without deleting it.
- Text preview/hash follows the PRD privacy rules.

**Tests:**

- `cd apps/packages/ui && bunx vitest run src/hooks/__tests__/useComparisonTranscribe.test.ts`
- `cd apps/packages/ui && bunx vitest run src/components/Option/STT/__tests__/ComparisonPanel.test.tsx src/components/Option/STT/__tests__/HistoryPanel.test.tsx`
- `cd apps/packages/ui && bunx vitest run src/components/Option/Speech/__tests__/RenderStrip.test.tsx src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx`
- `cd apps/tldw-frontend && bun run e2e:smoke:audio`

**Status:** Not Started

### Files

Modify:

- `apps/packages/ui/src/hooks/useComparisonTranscribe.ts`
- `apps/packages/ui/src/components/Option/STT/ComparisonPanel.tsx`
- `apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx`
- `apps/packages/ui/src/components/Option/STT/HistoryPanel.tsx`
- `apps/packages/ui/src/hooks/useMultiRenderState.ts`
- `apps/packages/ui/src/components/Option/Speech/RenderStrip.tsx`
- `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx`

Add:

- `apps/packages/ui/src/components/Option/Audio/comparison-provenance.ts`
- `apps/packages/ui/src/components/Option/Audio/__tests__/comparison-provenance.test.ts`

### Implementation Steps

- [ ] Add pure helpers for text preview/hash, created-time formatting, byte-size formatting, and client-latency labeling.
- [ ] Extend `ComparisonResult` to include `config` and `metadata` while preserving current fields.
- [ ] Update `extractText` or a new response normalizer to also extract `language`, `duration`, `segments`, and `word` metadata when present.
- [ ] Store STT comparison history with configuration provenance, not only model/text/latency/word count.
- [ ] Add STT card metadata rows for language, duration, segment count, timestamp settings, and source label.
- [ ] Add TTS render row metadata for provider/model/voice/format/speed, created time, input length, input preview/hash, byte size, and client latency.
- [ ] Label client-measured latency as "Client measured" or equivalent.
- [ ] Add retry, duplicate, and disable controls to TTS and STT rows using existing icon/button patterns.
- [ ] Preserve current copy, save-to-notes, download, and history actions.
- [ ] Ensure result metadata cannot resize rows unpredictably at extension widths.

## Stage 5: Optional Phase 2B STT Capability Summary Endpoint

**Goal:** Add a backend capability summary only if Phase 2A leaves material gaps that block clear UX.

**Success Criteria:**

- A written Phase 2A gap note identifies which UX states cannot be derived from existing APIs.
- New endpoint combines health, static catalog, provider capability, and source/confidence fields.
- Endpoint distinguishes unsupported from unknown.
- Frontend uses the endpoint as an enhancement, not as a hard dependency for basic `/stt`.

**Tests:**

- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/audio -k capability -v`
- `cd apps/packages/ui && bunx vitest run src/hooks/__tests__/useTranscriptionModelsCatalog.test.tsx`
- `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints tldw_Server_API/app/core -f json -o /tmp/bandit_tts_stt_capabilities.json`

**Status:** Gated

### Potential Files

Modify or add only after the gate:

- `tldw_Server_API/app/api/v1/endpoints/audio.py` or the current audio endpoint owner.
- `tldw_Server_API/app/api/v1/schemas/audio.py` or the current audio schema owner.
- `tldw_Server_API/app/core/STT/*` capability composition module if one already exists.
- `tldw_Server_API/tests/audio/test_stt_capabilities.py`
- `apps/packages/ui/src/services/tldw/domains/models-audio.ts`
- `apps/packages/ui/src/hooks/useTranscriptionModelsCatalog.ts`

### Endpoint Shape To Validate Before Coding

Candidate route:

```http
GET /api/v1/audio/transcriptions/capabilities
```

Candidate response:

```json
{
  "models": [
    {
      "id": "faster-whisper-large-v3",
      "label": "Faster Whisper Large v3",
      "provider": "faster_whisper",
      "availability": "ready",
      "availability_source": "health",
      "capabilities": {
        "batch": "supported",
        "streaming": "unknown",
        "diarization": "unknown",
        "timestamps": "supported",
        "segments": "supported"
      },
      "sources": {
        "batch": "provider",
        "streaming": "provider",
        "diarization": "unknown",
        "timestamps": "static_catalog",
        "segments": "response_schema"
      },
      "message": "Ready"
    }
  ]
}
```

### Gate Checklist

- [ ] Document the Phase 2A gap.
- [ ] Confirm existing backend owner for audio schemas and endpoints.
- [ ] Confirm AuthNZ dependency and rate-limit behavior by matching current audio endpoint patterns.
- [ ] Confirm response does not require downloading or warming models just to inspect metadata.
- [ ] Add API tests before implementation.
- [ ] Run backend tests and Bandit on touched backend scope.

## Stage 6: Phase 4 Preset Ownership Decision

**Goal:** Decide where per-user speech presets live before implementing server-side CRUD.

**Success Criteria:**

- A decision document exists before backend or frontend preset CRUD starts.
- It identifies backend owner, DB boundary, schema, AuthNZ principal behavior, migration behavior, and Browser TTS rules.
- It explicitly says presets are not TTS history, STT transcript rows, generated artifacts, or comparison history.
- It defines how WebUI and extension share preset state.

**Tests:** Documentation review plus any architecture tests defined by the decision document.

**Status:** Gated

### Decision Document

Add:

- `Docs/Design/Audio_Presets_Ownership_2026_05.md`

Minimum contents:

- Owner module and endpoint namespace.
- DB and table/document shape.
- Principal resolution in single-user and multi-user modes.
- Preset kind model: `tts`, `stt`, optional `speech`.
- Browser TTS server persistence rule.
- Import/export stance.
- Migration stance for existing local history.
- Deletion semantics.
- Rate-limit/security considerations.
- Frontend API client responsibilities.
- Extension parity responsibilities.

### Preset CRUD Candidate Shape

Do not implement until the decision document is accepted.

Candidate endpoints:

```http
GET /api/v1/audio/presets
POST /api/v1/audio/presets
PATCH /api/v1/audio/presets/{preset_id}
DELETE /api/v1/audio/presets/{preset_id}
POST /api/v1/audio/presets/{preset_id}/validate
```

Candidate schema:

```ts
export type AudioPresetKind = "tts" | "stt" | "speech"

export type AudioPreset = {
  id: string
  ownerUserId: string
  kind: AudioPresetKind
  name: string
  description?: string
  favorite: boolean
  isDefault: boolean
  config: Record<string, unknown>
  capabilityAssumptions: Record<string, unknown>
  createdAt: string
  updatedAt: string
}
```

## Stage 7: Preset CRUD And Reuse UX

**Goal:** Add server-side per-user TTS/STT presets after Stage 5 is complete.

**Success Criteria:**

- Users can save, apply, duplicate, rename, favorite/default, and delete TTS and STT presets.
- Presets survive reload and are available in both WebUI and extension.
- Presets validate current readiness before applying.
- Presets do not leak across users.
- Deleting a preset does not delete generated audio, transcripts, or history.

**Tests:**

- Backend API tests for CRUD, AuthNZ isolation, validation, and deletion semantics.
- Frontend Vitest tests for save/apply/validate flows.
- Extension route parity tests for applying a saved STT preset.
- Bandit on touched backend scope.

**Status:** Gated

### Potential Files

Backend, exact owner to be confirmed in Stage 5:

- `tldw_Server_API/app/api/v1/endpoints/audio_presets.py`
- `tldw_Server_API/app/api/v1/schemas/audio_presets.py`
- `tldw_Server_API/app/core/DB_Management/*`
- `tldw_Server_API/tests/audio/test_audio_presets.py`

Frontend:

- `apps/packages/ui/src/services/tldw/domains/models-audio.ts`
- `apps/packages/ui/src/hooks/useAudioPresets.ts`
- `apps/packages/ui/src/components/Option/Audio/AudioPresetPicker.tsx`
- `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx`
- `apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx`
- `apps/tldw-frontend/extension/routes/option-stt.tsx`
- `apps/tldw-frontend/extension/routes/option-tts.tsx`

### Implementation Steps

- [ ] Write backend CRUD and AuthNZ tests first.
- [ ] Implement storage migration and API endpoints using the decision document.
- [ ] Add frontend API client methods.
- [ ] Add `useAudioPresets` with query invalidation and validation behavior.
- [ ] Add preset picker/save/apply controls to TTS and STT pages.
- [ ] Add preset validation warnings for unavailable providers/models.
- [ ] Add extension tests proving saved STT presets apply in `#/stt`.
- [ ] Run backend, frontend, extension, and Bandit verification.

## Stage 8: Browser QA And Accessibility Verification

**Goal:** Validate the visible workflows in a running browser, especially because the original UX findings were browser-observed.

**Success Criteria:**

- WebUI `/tts` supports first-time Browser preview, configured server provider attempts, and comparison rows.
- WebUI `/stt` supports upload/record, model selection, readiness, and comparison results.
- Extension `#/tts` and `#/stt` use the same core surfaces without overflow.
- Errors and readiness states are visible, accessible, and actionable.
- Result metadata does not overlap or create unstable layouts.

**Tests:**

- `cd apps/tldw-frontend && bun run e2e:smoke:audio`
- `cd apps/tldw-frontend && bunx playwright test e2e/workflows/tier-2-features/stt-transcription.spec.ts --reporter=line`
- `cd apps/tldw-frontend && bunx playwright test e2e/workflows/tier-2-features/tts-synthesis.spec.ts --reporter=line`

**Status:** Not Started

### Manual Browser Checklist

- [ ] `/tts` first visit: page title, no-setup Browser preview label, provider readiness, text input, add row, generate path.
- [ ] `/tts` provider switch: model/voice do not remain from another provider.
- [ ] `/tts` result: config metadata, client latency label, retry/duplicate/disable controls.
- [ ] `/stt` first visit: page title, upload/record prompt, model readiness, settings discoverability.
- [ ] `/stt` model comparison: three models or mocked models show distinct config/result provenance.
- [ ] `/stt` microphone denial: recovery text and retry are visible.
- [ ] Extension `#/tts`: locked TTS surface fits extension viewport.
- [ ] Extension `#/stt`: dedicated STT comparison surface fits extension viewport.
- [ ] Keyboard only: controls and result actions are reachable in logical order.

## Verification Matrix

Run the narrow command for the touched slice first, then the broader checks before PR closeout.

Frontend unit and route tests:

```bash
cd apps/packages/ui
bunx vitest run src/routes/__tests__/option-audio-route-identity.test.tsx
bunx vitest run src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx src/components/Option/Speech/__tests__/RenderStrip.test.tsx
bunx vitest run src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx src/components/Option/STT/__tests__/ComparisonPanel.test.tsx
```

Extension unit tests:

```bash
cd apps/tldw-frontend
bun run test:extension -- extension/__tests__/audio-route-parity.guard.test.tsx
```

Smoke E2E:

```bash
cd apps/tldw-frontend
bun run e2e:smoke:audio
```

Backend, only for Phase 2B or Phase 4/6:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/audio -v
python -m bandit -r tldw_Server_API/app/api/v1/endpoints tldw_Server_API/app/core -f json -o /tmp/bandit_tts_stt_audio.json
```

Repo hygiene:

```bash
git diff --check
```

## Open Risks And Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| STT catalog response lacks enough structured metadata for useful labels. | Users still see raw model ids and vague capability states. | Ship Phase 2A with explicit unknown labels, then use the Phase 2B gate if the remaining ambiguity blocks task success. |
| Health checks are expensive if run for every model. | Slow first render and unnecessary backend work. | Fetch health for selected/default/visible models first; add lazy expansion for the full list. |
| TTS provider state remains centralized in `SpeechPlaygroundPage`. | Provider mismatch bugs can recur. | Move config derivation to tested pure helpers before changing UI controls. |
| Browser TTS appears in histories or presets as a portable provider. | Users expect server-backed repeatability that cannot exist. | Label as Browser preview, mark `browser_local`, exclude from server presets unless explicitly revalidated. |
| Preset CRUD chooses the wrong persistence boundary. | Data leakage or later migration pain. | Require Stage 5 decision document before any CRUD implementation. |
| Extension viewport cannot fit all WebUI STT controls. | Parity exists technically but is unusable. | Use the same core workflow with responsive grouping; validate extension-width tests and browser QA. |
| Comparison metadata implies more precision than the backend returns. | Power users make false provider/quality conclusions. | Label client latency explicitly and omit cost/version/backend duration unless returned. |

## Definition Of Done For Implementation Program

- [x] Slice 1 route parity and TTS config truthfulness shipped with tests.
- [x] Slice 2A readiness shipped with current APIs and explicit unknown states.
- [ ] Slice 3 comparison provenance shipped with privacy-safe metadata.
- [ ] Phase 2B either shipped with backend tests or explicitly closed as unnecessary after Phase 2A.
- [ ] Phase 4 preset ownership decision completed before CRUD work.
- [ ] Preset CRUD shipped only after storage/AuthNZ/migration ownership is approved.
- [ ] Browser-observed QA completed for WebUI `/tts`, WebUI `/stt`, extension `#/tts`, and extension `#/stt`.
- [ ] No unrelated WebUI, backend, media ingestion, RAG, chat, or app-wide redesign changes included.
