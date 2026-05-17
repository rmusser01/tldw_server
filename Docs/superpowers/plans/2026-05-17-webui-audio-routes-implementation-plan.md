# WebUI Audio Routes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the WebUI and extension audio routes self-explanatory, provider-aware, and recoverable across `/audio`, `/speech`, `/stt`, `/tts`, and `/audiobook-studio` without adding audio engines or changing backend APIs.

**Architecture:** Add a small audio route-job contract, then align each route wrapper and route-owned page to the same route identity, readiness, empty-state, and recovery vocabulary established by WP1, WP2, and WP4. Preserve current route ownership unless tests prove an intentional switch is required: `/speech` stays the unified speech route, `/stt` stays the dedicated transcription route in shared WebUI, `/tts` stays the shared speech route locked to listen mode, `/audio` stays an alias to `/speech`, and `/audiobook-studio` stays the long-form generation studio.

**Tech Stack:** React, Next.js pages, shared `apps/packages/ui` route shells, extension route wrappers, TanStack Query, existing audio hooks, existing design-system state primitives, Vitest, React Testing Library, Playwright.

---

## Source Documents

- Backlog task: `TASK-418.8`
- Parent implementation plan: `Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md`
- UX remediation spec: `Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md`
- Dependency plans:
  - `Docs/superpowers/plans/2026-05-17-webui-route-contract-visibility-implementation-plan.md`
  - `Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md`
  - `Docs/superpowers/plans/2026-05-17-webui-responsive-landmarks-implementation-plan.md`

## Audit Findings Addressed

- `F2 support`: Page purpose and canonical route identity are unclear when audio routes overlap.
- `F9 support`: Missing, unsupported, unavailable, unauthorized, degraded, and not-configured audio states are inconsistent.
- `F15 support`: Advanced model, voice, comparison, and output controls need to stay available without dominating first-use flows.
- `F18 support`: Hosted, beta, placeholder, unsupported, and debug states need explicit visibility language.
- `F19 support`: Audio routes need resilient responsive behavior and consistent landmarks.

## Route Inventory And Ownership

| Route | Primary user goal | Current ownership | Primary workflows | UX contract |
| --- | --- | --- | --- | --- |
| `/audio` | Reach the canonical combined audio route | `apps/tldw-frontend/pages/audio.tsx` redirects to `/speech` | Navigate from old links, bookmarks, docs, and search results | Keep as an alias. Test that it lands on `/speech` and does not imply a separate product surface. |
| `/speech` | Record speech, edit transcripts, and synthesize audio in one combined workspace | `apps/packages/ui/src/routes/option-speech.tsx`, `SpeechPlaygroundPage` | Round-trip, speak, listen, choose input source, inspect provider readiness, review generated audio | Treat as the canonical combined route. Keep mode switching visible and keep readiness close to the controls it affects. |
| `/stt` | Transcribe audio and compare transcription results | Shared WebUI: `option-stt.tsx`, `SttPlaygroundPage`. Extension: `extension/routes/option-stt.tsx`, `SpeechPlaygroundPage initialMode="speak"` | Record or upload audio, select models, run comparison, save transcript, inspect history | Keep a dedicated transcription identity. Resolve shared WebUI and extension route parity with explicit tests before changing ownership. |
| `/tts` | Generate audio from text with voice and provider controls | Shared WebUI: `option-tts.tsx`, `SpeechPlaygroundPage lockedMode="listen" hideModeSwitcher`. Legacy component exists at `components/Option/TTS/TtsPlaygroundPage.tsx` | Draft text, choose provider, choose voice/model, generate audio, inspect output history | Keep the route locked to listen mode unless a parity test proves switching to the legacy TTS page is safer. Prevent duplicate TTS surfaces from drifting. |
| `/audiobook-studio` | Build long-form audiobook projects from text and generated audio | `option-audiobook-studio.tsx`, `AudiobookStudioPage` | Create project, paste content, split chapters, generate audio, export output, recover autosaved work | Keep as a beta long-form production studio. Add route-level safety, project status, provider readiness, and recovery clarity without replacing the studio flow. |

## Frontend-Only Versus Backend-Gated Work

### Frontend-Only Work

Use frontend-only changes when the route can already derive state from:

- `useServerCapabilities`.
- `useTldwAudioStatus`.
- `useTtsProviderData`.
- `useTranscriptionModelsCatalog`.
- Existing hosted deployment checks in `isHostedTldwDeployment`.
- Existing STT recording, comparison, and history state.
- Existing speech provider, voice, model, output, `ffmpeg`, and generated-audio state.
- Existing audiobook project, autosave, chapter, generation, and output state.

Frontend-only changes include:

- Route labels, headings, and subheadings.
- Empty, loading, unavailable, unsupported, not-configured, degraded, and partial states.
- Local retry and recovery actions.
- Route error boundaries and accessible loading states.
- Route metadata and route ownership tests.
- Browser QA and Playwright assertions.

### Backend-Gated Work

Create a separate backend contract task before implementation if a route needs state that is not exposed by current frontend inputs.

Backend-gated examples:

- A new audio provider inventory endpoint.
- New STT or TTS engine support.
- A unified audiobook generation readiness endpoint.
- Server-side route alias metadata.
- New job APIs for audiobook batch generation.

Do not add backend API changes inside a WP11A implementation PR unless the Backlog task explicitly broadens scope and this plan is updated first.

## Non-Goals

- Do not add new STT engines.
- Do not add new TTS engines.
- Do not rename route paths.
- Do not replace `SpeechPlaygroundPage`, `SttPlaygroundPage`, or `AudiobookStudioPage` wholesale.
- Do not create a new design system.
- Do not hide advanced audio controls from returning users.
- Do not remove the `/audio` alias.
- Do not merge audiobook studio into the speech playground.
- Do not change backend APIs in this slice.
- Do not use visible explanatory prose as a substitute for route identity, status placement, disabled control reasons, and recovery actions.

## File Structure

### New Files

- `apps/packages/ui/src/routes/audio-route-jobs.ts`
  - Owns route labels, canonical jobs, canonical component ownership, feature family, and expected capability states for audio routes.
  - Includes `/audio` as a Next-page alias even though it is not a shared route-registry entry.
- `apps/packages/ui/src/routes/__tests__/audio-route-jobs.test.ts`
  - Verifies coverage for `/audio`, `/speech`, `/stt`, `/tts`, and `/audiobook-studio`.
  - Verifies finding coverage for `F2 support`, `F9 support`, `F15 support`, `F18 support`, and `F19 support`.
- `apps/packages/ui/src/routes/__tests__/audio-route-boundaries.test.tsx`
  - Extends or replaces `option-audio-route-identity.test.tsx` if a broader route-boundary test is clearer.
  - Verifies `OptionLayout`, route error boundaries, hosted gates, canonical labels, and page ownership.
- `apps/tldw-frontend/e2e/workflows/tier-2-features/audio-alias.spec.ts`
  - Add only if no existing route-alias coverage can be extended cleanly.
  - Verifies `/audio` redirects or resolves to the canonical speech route.

### Modified Files

- `apps/tldw-frontend/pages/audio.tsx`
  - Keep redirect behavior.
  - Add a regression test around the alias rather than changing UI unless route infrastructure requires a route metadata hook.
- `apps/tldw-frontend/pages/speech.tsx`
  - Verify dynamic import route wrapper still maps to the shared `OptionSpeech` route.
- `apps/tldw-frontend/pages/stt.tsx`
  - Verify dynamic import route wrapper still maps to the shared `OptionStt` route.
- `apps/tldw-frontend/pages/tts.tsx`
  - Verify dynamic import route wrapper still maps to the shared `OptionTts` route.
- `apps/tldw-frontend/pages/audiobook-studio.tsx`
  - Verify dynamic import route wrapper still maps to the shared `OptionAudiobookStudio` route.
- `apps/tldw-frontend/extension/routes/option-speech.tsx`
  - Keep extension route identity aligned with shared WebUI.
- `apps/tldw-frontend/extension/routes/option-stt.tsx`
  - Align route identity with the dedicated STT job after testing the current `SpeechPlaygroundPage initialMode="speak"` mapping.
- `apps/tldw-frontend/extension/routes/option-tts.tsx`
  - Align route identity with the canonical listen-mode TTS job.
- `apps/tldw-frontend/extension/routes/route-registry.tsx`
  - Verify audio routes have labels and paths consistent with shared route registry.
- `apps/packages/ui/src/routes/route-registry.tsx`
  - Verify shared audio route entries are covered by route-job tests.
- `apps/packages/ui/src/routes/option-speech.tsx`
  - Preserve `RouteErrorBoundary routeId="speech" routeLabel="Speech Playground"`.
- `apps/packages/ui/src/routes/option-stt.tsx`
  - Add `RouteErrorBoundary routeId="stt" routeLabel="STT Playground"` if tests confirm the missing boundary is still present.
  - Preserve hosted deployment fallback with `HostedAudioFeatureMessage`.
- `apps/packages/ui/src/routes/option-tts.tsx`
  - Preserve `RouteErrorBoundary routeId="tts" routeLabel="TTS Playground"`.
  - Preserve `SpeechPlaygroundPage lockedMode="listen" hideModeSwitcher`.
- `apps/packages/ui/src/routes/option-audiobook-studio.tsx`
  - Add `RouteErrorBoundary routeId="audiobook-studio" routeLabel="Audiobook Studio"` if tests confirm the missing boundary is still present.
- `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx`
  - Align route title, current mode status, provider readiness, voice/model status, generated output status, and recovery actions.
- `apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx`
  - Align title, model catalog loading, no-model state, failed catalog state, recording source status, comparison state, save-to-notes state, history state, and recovery actions.
- `apps/packages/ui/src/components/Option/TTS/TtsPlaygroundPage.tsx`
  - Treat as legacy or secondary unless route ownership changes through tests.
  - If kept, keep its tests focused on shared TTS utilities and do not let it become a second route owner.
- `apps/packages/ui/src/components/Option/AudiobookStudio/AudiobookStudioPage.tsx`
  - Align beta identity, project status, autosave status, chapter progress, generation readiness, output readiness, and recovery actions.

### Existing Tests To Extend

- `apps/packages/ui/src/routes/__tests__/option-audio-route-identity.test.tsx`
- `apps/packages/ui/src/routes/__tests__/option-audio-hosted-message.test.tsx`
- `apps/packages/ui/src/routes/__tests__/option-route-visibility.test.ts`
- `apps/packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx`
- `apps/packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.audio-source.test.tsx`
- `apps/packages/ui/src/components/Option/Speech/__tests__/TtsProviderStrip.test.tsx`
- `apps/packages/ui/src/components/Option/Speech/__tests__/TtsInspectorPanel.test.tsx`
- `apps/packages/ui/src/components/Option/Speech/__tests__/TtsInspectorTabs.test.tsx`
- `apps/packages/ui/src/components/Option/Speech/__tests__/RenderStrip.test.tsx`
- `apps/packages/ui/src/components/Option/Speech/__tests__/TtsStickyActionBar.test.tsx`
- `apps/packages/ui/src/components/Option/Speech/__tests__/VoicePickerModal.test.tsx`
- `apps/packages/ui/src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx`
- `apps/packages/ui/src/components/Option/STT/__tests__/RecordingStrip.test.tsx`
- `apps/packages/ui/src/components/Option/STT/__tests__/InlineSettingsPanel.test.tsx`
- `apps/packages/ui/src/components/Option/STT/__tests__/ComparisonPanel.test.tsx`
- `apps/packages/ui/src/components/Option/STT/__tests__/HistoryPanel.test.tsx`
- `apps/packages/ui/src/components/Option/STT/__tests__/keyboard-shortcuts.test.tsx`
- `apps/packages/ui/src/components/Option/TTS/__tests__/TtsPlaygroundPage.defaults.test.tsx`
- `apps/tldw-frontend/e2e/workflows/tier-2-features/speech-playground.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-2-features/stt-transcription.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-2-features/tts-synthesis.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-2-features/audiobook-studio.spec.ts`
- `apps/tldw-frontend/e2e/smoke/stage7-audio-regression.spec.ts`

## Audio Route Job Contract

Create a route metadata file that makes overlapping route intent testable:

```ts
export type AudioRouteConcept =
  | "audio_alias"
  | "speech_combined"
  | "stt"
  | "tts"
  | "audiobook"

export type AudioRouteOwner =
  | "next_alias"
  | "shared_route"
  | "extension_route"

export type AudioRouteCapability =
  | "hosted_gate"
  | "server_capability"
  | "provider_config"
  | "model_catalog"
  | "voice_catalog"
  | "recording_source"
  | "project_state"
  | "generation_state"

export type AudioRouteJob = {
  route: "/audio" | "/speech" | "/stt" | "/tts" | "/audiobook-studio"
  concept: AudioRouteConcept
  label: string
  primaryJob: string
  primaryActionLabel: string
  routeOwner: AudioRouteOwner
  canonicalComponent: string
  capabilities: AudioRouteCapability[]
  routeStatePolicy: "alias" | "ready_or_recoverable" | "beta_ready_or_recoverable"
  findings: Array<"F2 support" | "F9 support" | "F15 support" | "F18 support" | "F19 support">
}
```

Initial inventory:

```ts
export const AUDIO_ROUTE_JOBS: AudioRouteJob[] = [
  {
    route: "/audio",
    concept: "audio_alias",
    label: "Audio",
    primaryJob: "Open the canonical combined speech route from old links and bookmarks.",
    primaryActionLabel: "Open Speech Playground",
    routeOwner: "next_alias",
    canonicalComponent: "RouteRedirect:/speech",
    capabilities: [],
    routeStatePolicy: "alias",
    findings: ["F2 support", "F18 support", "F19 support"]
  },
  {
    route: "/speech",
    concept: "speech_combined",
    label: "Speech Playground",
    primaryJob: "Record, transcribe, edit, and synthesize audio in one workspace.",
    primaryActionLabel: "Start audio workflow",
    routeOwner: "shared_route",
    canonicalComponent: "SpeechPlaygroundPage",
    capabilities: ["server_capability", "provider_config", "model_catalog", "voice_catalog", "recording_source"],
    routeStatePolicy: "ready_or_recoverable",
    findings: ["F2 support", "F9 support", "F15 support", "F18 support", "F19 support"]
  },
  {
    route: "/stt",
    concept: "stt",
    label: "STT Playground",
    primaryJob: "Transcribe audio and compare transcription results.",
    primaryActionLabel: "Start transcription",
    routeOwner: "shared_route",
    canonicalComponent: "SttPlaygroundPage",
    capabilities: ["hosted_gate", "server_capability", "model_catalog", "recording_source"],
    routeStatePolicy: "ready_or_recoverable",
    findings: ["F2 support", "F9 support", "F15 support", "F18 support", "F19 support"]
  },
  {
    route: "/tts",
    concept: "tts",
    label: "TTS Playground",
    primaryJob: "Generate audio from text with provider, voice, and model controls.",
    primaryActionLabel: "Generate speech",
    routeOwner: "shared_route",
    canonicalComponent: "SpeechPlaygroundPage:listen",
    capabilities: ["hosted_gate", "server_capability", "provider_config", "voice_catalog"],
    routeStatePolicy: "ready_or_recoverable",
    findings: ["F2 support", "F9 support", "F15 support", "F18 support", "F19 support"]
  },
  {
    route: "/audiobook-studio",
    concept: "audiobook",
    label: "Audiobook Studio",
    primaryJob: "Create long-form audiobook projects from text and generated speech.",
    primaryActionLabel: "Create project",
    routeOwner: "shared_route",
    canonicalComponent: "AudiobookStudioPage",
    capabilities: ["provider_config", "voice_catalog", "project_state", "generation_state"],
    routeStatePolicy: "beta_ready_or_recoverable",
    findings: ["F2 support", "F9 support", "F15 support", "F18 support", "F19 support"]
  }
]
```

## Route State Vocabulary

Use the WP2 shared states. Do not invent audio-only names for equivalent states.

| State | Audio meaning | Required UI behavior |
| --- | --- | --- |
| `loading` | Model, provider, voice, project, or server capability query is in flight | Keep the landmark and primary controls stable. Use role status text and skeleton or inline busy states. |
| `ready` | Required route capability is available | Primary action is enabled. Current model, provider, source, project, or output state is visible. |
| `not_configured` | A required provider, voice, API key, local engine, model, or project is missing | Keep primary action disabled. Show the setup target and retry path in user language. |
| `unsupported` | Hosted mode or the current server does not expose the feature | Use `HostedAudioFeatureMessage` or the shared unsupported state. Do not expose raw endpoint errors first. |
| `unavailable` | The expected endpoint or local dependency is unreachable | Show recovery, retry, and diagnostics disclosure. |
| `degraded` | Some providers, voices, models, or local tools are missing while a narrower path still works | Keep working controls enabled and label what is unavailable. |
| `partial` | Some audiobook chapters, STT comparisons, or generated segments succeeded and others failed | Preserve completed work, mark failed items, and offer item-level retry. |
| `error` | User action failed | Explain what failed, preserve user input, and offer retry or recovery. |

## Implementation Tasks

### Task 1: Lock The Audio Route Contract

**Files:**
- Create: `apps/packages/ui/src/routes/audio-route-jobs.ts`
- Create: `apps/packages/ui/src/routes/__tests__/audio-route-jobs.test.ts`
- Modify: `apps/packages/ui/src/routes/__tests__/option-audio-route-identity.test.tsx`
- Modify: `apps/packages/ui/src/routes/route-registry.tsx`
- Modify: `apps/tldw-frontend/extension/routes/route-registry.tsx`

- [ ] **Step 1: Write the failing route-job coverage test**

Create `apps/packages/ui/src/routes/__tests__/audio-route-jobs.test.ts`:

```ts
import { describe, expect, it } from "vitest"
import { AUDIO_ROUTE_JOBS } from "../audio-route-jobs"

const routes = ["/audio", "/speech", "/stt", "/tts", "/audiobook-studio"] as const
const findings = ["F2 support", "F9 support", "F15 support", "F18 support", "F19 support"] as const

describe("audio route jobs", () => {
  it("covers every WP11A root audio route once", () => {
    expect(AUDIO_ROUTE_JOBS.map((job) => job.route).sort()).toEqual(Array.from(routes).sort())
  })

  it("keeps route labels and primary jobs usable", () => {
    for (const job of AUDIO_ROUTE_JOBS) {
      expect(job.label).not.toHaveLength(0)
      expect(job.primaryJob).not.toHaveLength(0)
      expect(job.primaryActionLabel).not.toHaveLength(0)
      expect(job.canonicalComponent).not.toHaveLength(0)
    }
  })

  it("maps the audit findings into implementation coverage", () => {
    const covered = new Set(AUDIO_ROUTE_JOBS.flatMap((job) => job.findings))
    for (const finding of findings) {
      expect(covered.has(finding)).toBe(true)
    }
  })
})
```

- [ ] **Step 2: Run the route-job test to verify it fails**

Run:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/audio-route-jobs.test.ts
```

Expected: FAIL because `audio-route-jobs.ts` does not exist.

- [ ] **Step 3: Add the audio route-job metadata**

Create `apps/packages/ui/src/routes/audio-route-jobs.ts` with the route contract from the "Audio Route Job Contract" section. Keep it pure data with no React imports.

- [ ] **Step 4: Run the route-job test to verify it passes**

Run:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/audio-route-jobs.test.ts
```

Expected: PASS.

- [ ] **Step 5: Extend route identity tests**

Extend `apps/packages/ui/src/routes/__tests__/option-audio-route-identity.test.tsx` so it also asserts:

- `/stt` has `RouteErrorBoundary routeId="stt"` after Task 3.
- `/audiobook-studio` has `RouteErrorBoundary routeId="audiobook-studio"` after Task 5.
- `/tts` still renders `SpeechPlaygroundPage lockedMode="listen" hideModeSwitcher`.
- `/tts` does not render `TtsPlaygroundPage` unless the implementation intentionally changes the route contract and updates `audio-route-jobs.ts` in the same commit.

- [ ] **Step 6: Commit the route contract**

```bash
git add apps/packages/ui/src/routes/audio-route-jobs.ts apps/packages/ui/src/routes/__tests__/audio-route-jobs.test.ts apps/packages/ui/src/routes/__tests__/option-audio-route-identity.test.tsx
git commit -m "test: lock audio route ownership"
```

### Task 2: Preserve `/audio` As A Canonical Alias To `/speech`

**Files:**
- Modify: `apps/tldw-frontend/pages/audio.tsx`
- Modify: `apps/tldw-frontend/pages/speech.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/tier-2-features/audio-alias.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/tier-2-features/speech-playground.spec.ts`

- [ ] **Step 1: Write the alias regression test**

Create or extend a Playwright route test:

```ts
import { expect, test } from "@playwright/test"

test("audio alias opens the canonical speech playground", async ({ page }) => {
  await page.goto("/audio")
  await expect(page).toHaveURL(/\/speech/)
  await expect(page.getByRole("heading", { name: /speech playground/i })).toBeVisible()
})
```

- [ ] **Step 2: Run the alias test to capture current behavior**

Run:

```bash
bunx playwright test apps/tldw-frontend/e2e/workflows/tier-2-features/audio-alias.spec.ts --reporter=line
```

Expected: PASS if the current `RouteRedirect` behavior is intact. If it fails, inspect whether Next routing base paths or hosted routing changed before editing.

- [ ] **Step 3: Keep `/audio` UI-free unless the alias is broken**

Leave `apps/tldw-frontend/pages/audio.tsx` as:

```tsx
import RouteRedirect from "~/components/Routes/RouteRedirect";

export default function AudioPage() {
  return <RouteRedirect to="/speech" />;
}
```

Only adjust imports or route plumbing if the test exposes an actual alias failure.

- [ ] **Step 4: Verify `/speech` first screen route identity**

Extend `speech-playground.spec.ts` to assert:

- Heading is `Speech Playground`.
- The route exposes the combined workflow modes.
- A user can see current audio source or recording readiness without opening settings.
- TTS provider readiness is visible near TTS controls.
- Generated audio history or empty output state is present after route load.

- [ ] **Step 5: Run speech route E2E**

Run:

```bash
bunx playwright test apps/tldw-frontend/e2e/workflows/tier-2-features/speech-playground.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 6: Commit the alias and speech route coverage**

```bash
git add apps/tldw-frontend/e2e/workflows/tier-2-features/audio-alias.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/speech-playground.spec.ts apps/tldw-frontend/pages/audio.tsx
git commit -m "test: preserve audio alias to speech playground"
```

### Task 3: Make `/stt` A Recoverable Transcription Route

**Files:**
- Modify: `apps/packages/ui/src/routes/option-stt.tsx`
- Modify: `apps/tldw-frontend/extension/routes/option-stt.tsx`
- Modify: `apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/STT/__tests__/RecordingStrip.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/STT/__tests__/InlineSettingsPanel.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/STT/__tests__/ComparisonPanel.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/STT/__tests__/HistoryPanel.test.tsx`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/stt-transcription.spec.ts`

- [ ] **Step 1: Add failing route-boundary coverage**

Extend `option-audio-route-identity.test.tsx`:

```ts
it("wraps the dedicated STT route in a route boundary", async () => {
  render(<OptionStt />)

  const boundary = screen.getByTestId("route-boundary")
  expect(boundary).toHaveAttribute("data-route-id", "stt")
  expect(boundary).toHaveAttribute("data-route-label", "STT Playground")
  expect(await screen.findByTestId("stt-playground")).toBeVisible()
})
```

- [ ] **Step 2: Run the route-boundary test to verify it fails**

Run:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/option-audio-route-identity.test.tsx
```

Expected: FAIL while `option-stt.tsx` lacks `RouteErrorBoundary`.

- [ ] **Step 3: Add the route boundary and keep hosted behavior**

Wrap `OptionStt` in:

```tsx
<RouteErrorBoundary routeId="stt" routeLabel="STT Playground">
  <OptionLayout>{/* existing hosted gate and Suspense */}</OptionLayout>
</RouteErrorBoundary>
```

Keep `HostedAudioFeatureMessage` as the hosted-mode path.

- [ ] **Step 4: Verify extension route parity before changing ownership**

Add tests or route-registry assertions that document the current extension `/stt` mapping. If the extension must match the shared WebUI dedicated STT route, update `apps/tldw-frontend/extension/routes/option-stt.tsx` to render `SttPlaygroundPage` with the same hosted, loading, and boundary behavior. If extension bundle constraints require `SpeechPlaygroundPage initialMode="speak"`, record that exception in `audio-route-jobs.ts` with a distinct `extension_route` entry and route-specific test.

- [ ] **Step 5: Add STT page readiness tests**

Extend `SttPlaygroundPage.test.tsx` to cover:

- Catalog loading state uses a stable route landmark.
- Catalog load failure offers retry.
- No models state uses user-facing setup language and keeps recording disabled.
- Recording source state is visible before a user presses record.
- Comparison state preserves successful transcript results when another model fails.
- Save-to-notes preserves transcript text on failure.
- History empty state explains what will appear after transcription.

- [ ] **Step 6: Implement the minimal STT page adjustments**

Use existing STT components and hooks. Keep advanced controls inside the existing `InlineSettingsPanel`. Do not move comparison, history, or shortcuts into new global chrome.

- [ ] **Step 7: Run STT component tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx apps/packages/ui/src/components/Option/STT/__tests__/RecordingStrip.test.tsx apps/packages/ui/src/components/Option/STT/__tests__/InlineSettingsPanel.test.tsx apps/packages/ui/src/components/Option/STT/__tests__/ComparisonPanel.test.tsx apps/packages/ui/src/components/Option/STT/__tests__/HistoryPanel.test.tsx apps/packages/ui/src/components/Option/STT/__tests__/keyboard-shortcuts.test.tsx
```

Expected: PASS.

- [ ] **Step 8: Run STT E2E**

Run:

```bash
bunx playwright test apps/tldw-frontend/e2e/workflows/tier-2-features/stt-transcription.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 9: Commit STT route readiness**

```bash
git add apps/packages/ui/src/routes/option-stt.tsx apps/tldw-frontend/extension/routes/option-stt.tsx apps/packages/ui/src/components/Option/STT apps/tldw-frontend/e2e/workflows/tier-2-features/stt-transcription.spec.ts apps/packages/ui/src/routes/__tests__/option-audio-route-identity.test.tsx
git commit -m "feat: clarify stt route readiness"
```

### Task 4: Make `/tts` A Single Canonical Synthesis Route

**Files:**
- Modify: `apps/packages/ui/src/routes/option-tts.tsx`
- Modify: `apps/tldw-frontend/extension/routes/option-tts.tsx`
- Modify: `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/TTS/TtsPlaygroundPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Speech/__tests__/TtsProviderStrip.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Speech/__tests__/TtsInspectorPanel.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Speech/__tests__/TtsInspectorTabs.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Speech/__tests__/RenderStrip.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Speech/__tests__/TtsStickyActionBar.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Speech/__tests__/VoicePickerModal.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/TTS/__tests__/TtsPlaygroundPage.defaults.test.tsx`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/tts-synthesis.spec.ts`

- [ ] **Step 1: Add route-owner guard coverage**

Extend `option-audio-route-identity.test.tsx` so `/tts` asserts:

- `RouteErrorBoundary routeId="tts"`.
- `RouteErrorBoundary routeLabel="TTS Playground"`.
- `SpeechPlaygroundPage` renders with `lockedMode="listen"`.
- Mode switcher is hidden.
- `TtsPlaygroundPage` is not rendered by the route.

- [ ] **Step 2: Run route-owner guard coverage**

Run:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/option-audio-route-identity.test.tsx
```

Expected: PASS before implementation. This protects current ownership while the page content changes.

- [ ] **Step 3: Add TTS readiness tests to the routed page**

Extend Speech tests for the locked listen route:

- Missing provider disables generation and points to setup.
- Missing voice catalog disables generation and keeps typed text.
- ElevenLabs loading, timeout, and missing-key states use the same recovery vocabulary as provider states.
- `ffmpeg` warning is visible as degraded output capability, not as a full route failure.
- Browser TTS fallback is labeled as local browser output when active.
- Generated audio segments remain inspectable after generation.
- Advanced voice and model controls remain accessible through current inspector tabs.

- [ ] **Step 4: Implement TTS page adjustments in `SpeechPlaygroundPage` and child components**

Keep the current locked listen route. Use existing provider strip, voice picker, inspector, sticky action bar, render strip, generated segment list, and fallback components. Avoid adding a second TTS route surface through `TtsPlaygroundPage`.

- [ ] **Step 5: Keep legacy `TtsPlaygroundPage` from drifting**

If `TtsPlaygroundPage` stays in the repository, keep its default test focused on component defaults and add a comment-free test assertion that the route owner remains `SpeechPlaygroundPage:listen` in `audio-route-jobs.ts`. Do not add new user-facing copy to explain why two components exist.

- [ ] **Step 6: Run TTS component tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx apps/packages/ui/src/components/Option/Speech/__tests__/TtsProviderStrip.test.tsx apps/packages/ui/src/components/Option/Speech/__tests__/TtsInspectorPanel.test.tsx apps/packages/ui/src/components/Option/Speech/__tests__/TtsInspectorTabs.test.tsx apps/packages/ui/src/components/Option/Speech/__tests__/RenderStrip.test.tsx apps/packages/ui/src/components/Option/Speech/__tests__/TtsStickyActionBar.test.tsx apps/packages/ui/src/components/Option/Speech/__tests__/VoicePickerModal.test.tsx apps/packages/ui/src/components/Option/TTS/__tests__/TtsPlaygroundPage.defaults.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Run TTS E2E**

Run:

```bash
bunx playwright test apps/tldw-frontend/e2e/workflows/tier-2-features/tts-synthesis.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 8: Commit TTS route readiness**

```bash
git add apps/packages/ui/src/routes/option-tts.tsx apps/tldw-frontend/extension/routes/option-tts.tsx apps/packages/ui/src/components/Option/Speech apps/packages/ui/src/components/Option/TTS apps/tldw-frontend/e2e/workflows/tier-2-features/tts-synthesis.spec.ts apps/packages/ui/src/routes/__tests__/option-audio-route-identity.test.tsx
git commit -m "feat: clarify tts route readiness"
```

### Task 5: Make `/audiobook-studio` Recoverable And Status-First

**Files:**
- Modify: `apps/packages/ui/src/routes/option-audiobook-studio.tsx`
- Modify: `apps/packages/ui/src/components/Option/AudiobookStudio/AudiobookStudioPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/AudiobookStudio/ContentInput/TextEditor.tsx`
- Modify: `apps/packages/ui/src/components/Option/AudiobookStudio/ChapterEditor/ChapterList.tsx`
- Modify: `apps/packages/ui/src/components/Option/AudiobookStudio/Generation/GenerationPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/AudiobookStudio/Output/OutputPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/AudiobookStudio/ProjectManagement/ProjectListView.tsx`
- Modify: `apps/packages/ui/src/components/Option/AudiobookStudio/ProjectManagement/ProjectMetadataForm.tsx`
- Test: add or extend `apps/packages/ui/src/components/Option/AudiobookStudio/__tests__/AudiobookStudioPage.test.tsx`
- Test: extend `apps/tldw-frontend/e2e/workflows/tier-2-features/audiobook-studio.spec.ts`

- [ ] **Step 1: Add failing route-boundary coverage**

Extend the audio route identity test so `/audiobook-studio` requires:

- `RouteErrorBoundary routeId="audiobook-studio"`.
- `RouteErrorBoundary routeLabel="Audiobook Studio"`.
- `AudiobookStudioPage` as the route-owned component.

- [ ] **Step 2: Run route-boundary coverage to verify it fails**

Run:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/option-audio-route-identity.test.tsx
```

Expected: FAIL while `option-audiobook-studio.tsx` lacks `RouteErrorBoundary`.

- [ ] **Step 3: Add the route boundary**

Wrap `OptionAudiobookStudio` in:

```tsx
<RouteErrorBoundary routeId="audiobook-studio" routeLabel="Audiobook Studio">
  <OptionLayout>
    <AudiobookStudioPage />
  </OptionLayout>
</RouteErrorBoundary>
```

- [ ] **Step 4: Add audiobook studio status tests**

Create or extend `AudiobookStudioPage.test.tsx` to cover:

- Beta identity is visible without obscuring the project workflow.
- New project, project list, project title, metadata, save, and saved states are visible.
- Autosave status and unsaved state are represented with control state, not hidden messages.
- Tabs stay visible and ordered as Content, Chapters, Generate, Output.
- Chapter counts, completed counts, pending counts, and generation state are visible near the workflow.
- Generation unavailable state keeps content and chapters intact.
- Output empty state explains that generated chapters appear after generation.

- [ ] **Step 5: Implement minimal audiobook studio adjustments**

Use current `AudiobookStudioPage` layout, `DismissibleBetaAlert`, tab components, project store, autosave state, and generation state. Keep dense project controls for returning users. Add route-level readiness or recovery only where existing state exists.

- [ ] **Step 6: Run audiobook component tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/AudiobookStudio/__tests__/AudiobookStudioPage.test.tsx apps/packages/ui/src/routes/__tests__/option-audio-route-identity.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Run audiobook E2E**

Run:

```bash
bunx playwright test apps/tldw-frontend/e2e/workflows/tier-2-features/audiobook-studio.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 8: Commit audiobook route readiness**

```bash
git add apps/packages/ui/src/routes/option-audiobook-studio.tsx apps/packages/ui/src/components/Option/AudiobookStudio apps/tldw-frontend/e2e/workflows/tier-2-features/audiobook-studio.spec.ts apps/packages/ui/src/routes/__tests__/option-audio-route-identity.test.tsx
git commit -m "feat: clarify audiobook studio readiness"
```

### Task 6: Verify Audio Routes Across Browser, Tests, And Responsive States

**Files:**
- Modify: `apps/tldw-frontend/e2e/smoke/stage7-audio-regression.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/speech-playground.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/stt-transcription.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/tts-synthesis.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/audiobook-studio.spec.ts`

- [ ] **Step 1: Extend stage 7 smoke coverage**

Add assertions that each route exposes:

- Stable heading and route landmark.
- Primary action state.
- Capability or provider status.
- Recovery path when the feature is unavailable.
- Mobile viewport layout without overlapping controls.

- [ ] **Step 2: Run route unit tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/audio-route-jobs.test.ts apps/packages/ui/src/routes/__tests__/option-audio-route-identity.test.tsx apps/packages/ui/src/routes/__tests__/option-audio-hosted-message.test.tsx apps/packages/ui/src/routes/__tests__/option-route-visibility.test.ts
```

Expected: PASS.

- [ ] **Step 3: Run focused audio component tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Speech/__tests__ apps/packages/ui/src/components/Option/STT/__tests__ apps/packages/ui/src/components/Option/TTS/__tests__ apps/packages/ui/src/components/Option/AudiobookStudio/__tests__
```

Expected: PASS.

- [ ] **Step 4: Run required WP11A Playwright coverage**

Run:

```bash
bunx playwright test apps/tldw-frontend/e2e/workflows/tier-2-features/stt-transcription.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/tts-synthesis.spec.ts apps/tldw-frontend/e2e/smoke/stage7-audio-regression.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 5: Run expanded audio workflow Playwright coverage**

Run:

```bash
bunx playwright test apps/tldw-frontend/e2e/workflows/tier-2-features/speech-playground.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/stt-transcription.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/tts-synthesis.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/audiobook-studio.spec.ts apps/tldw-frontend/e2e/smoke/stage7-audio-regression.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 6: Perform browser QA**

With the WebUI running, inspect these routes in desktop and mobile viewports:

- `/audio`: alias resolves to `/speech`.
- `/speech`: first-time route purpose, mode switcher, source status, provider status, generated output state.
- `/stt`: model catalog, recording controls, comparison panel, save/retry path, history.
- `/tts`: text input, provider and voice selection, generation action, output state, degraded local dependency state.
- `/audiobook-studio`: beta identity, project controls, autosave, tabs, generation status, output state.

Capture observations in the Backlog task and PR description. If a route has a browser-only issue, add the smallest route-specific follow-up test before fixing.

- [ ] **Step 7: Run final repository hygiene checks for touched scope**

Run:

```bash
git diff --check
```

Expected: PASS.

Run:

```bash
bunx tsc --noEmit
```

Expected: PASS, or document pre-existing TypeScript failures with exact file and error evidence.

- [ ] **Step 8: Commit verification updates**

```bash
git add apps/tldw-frontend/e2e/smoke/stage7-audio-regression.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/speech-playground.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/stt-transcription.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/tts-synthesis.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/audiobook-studio.spec.ts
git commit -m "test: verify audio route ux states"
```

## Acceptance Criteria

- `/audio`, `/speech`, `/stt`, `/tts`, and `/audiobook-studio` are represented in `audio-route-jobs.ts`.
- `/audio` remains an alias to `/speech` and is covered by browser or route tests.
- `/speech` presents itself as the combined speech route with visible mode, provider, source, and output state.
- `/stt` presents itself as a dedicated transcription route with model catalog, recording, comparison, history, save, retry, and no-model states covered.
- `/tts` presents itself as a dedicated synthesis route while staying routed through `SpeechPlaygroundPage lockedMode="listen" hideModeSwitcher` unless route contract tests are intentionally updated.
- `/audiobook-studio` presents itself as a beta long-form production route with project, autosave, chapter, generation, output, and recovery states covered.
- Shared WebUI and extension route wrappers have explicit route identity tests or documented intentional differences.
- Hosted, unsupported, not-configured, degraded, partial, and error states use the WP2 shared capability vocabulary.
- Desktop and mobile browser QA confirms no route has overlapping controls, hidden primary actions, or inaccessible recovery paths.
- No backend API change is included without a separate Backlog task and updated plan.

## Verification Commands

Run these before considering WP11A complete:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/audio-route-jobs.test.ts apps/packages/ui/src/routes/__tests__/option-audio-route-identity.test.tsx apps/packages/ui/src/routes/__tests__/option-audio-hosted-message.test.tsx apps/packages/ui/src/routes/__tests__/option-route-visibility.test.ts
```

```bash
bunx vitest run apps/packages/ui/src/components/Option/Speech/__tests__ apps/packages/ui/src/components/Option/STT/__tests__ apps/packages/ui/src/components/Option/TTS/__tests__ apps/packages/ui/src/components/Option/AudiobookStudio/__tests__
```

```bash
bunx playwright test apps/tldw-frontend/e2e/workflows/tier-2-features/stt-transcription.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/tts-synthesis.spec.ts apps/tldw-frontend/e2e/smoke/stage7-audio-regression.spec.ts --reporter=line
```

```bash
bunx playwright test apps/tldw-frontend/e2e/workflows/tier-2-features/speech-playground.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/stt-transcription.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/tts-synthesis.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/audiobook-studio.spec.ts apps/tldw-frontend/e2e/smoke/stage7-audio-regression.spec.ts --reporter=line
```

```bash
git diff --check
```

```bash
bunx tsc --noEmit
```

## Review Notes For Implementers

- Start with the route contract and tests before changing route wrappers.
- Treat `/tts` ownership as a regression-sensitive decision. The current route intentionally uses the shared speech page locked to listen mode while the legacy `TtsPlaygroundPage` still exists.
- Treat extension route parity as an explicit design decision. If extension constraints require a different component, test and document the difference in route metadata.
- Keep provider diagnostics available behind disclosure. First-use UI needs status and recovery, not raw endpoint detail.
- Preserve expert workflows: model comparison, advanced voice settings, provider selection, history, project save, chapter retry, and output export must remain fast to reach.
- Prefer disabled-control reasons, inline retry, and status placement over adding explanatory paragraphs.
- Keep all changes route-local unless a shared WP2 or WP4 primitive already exists for the state.
