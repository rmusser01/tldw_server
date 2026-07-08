# TTS Settings Voice Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small inline preview control to shared TTS settings so users can test the unsaved provider, model, voice, and speed before saving.

**Architecture:** Keep the feature in `TTSModeSettings`; avoid new routes, new backend endpoints, and websocket paths. Use browser `speechSynthesis` for Browser TTS and the existing one-shot synthesis helpers for server-backed providers, with one `AbortController` and one `HTMLAudioElement` owned by the component.

**Tech Stack:** React, Ant Design controls, Vitest, Testing Library, existing `tldwClient.synthesizeSpeech`, existing ElevenLabs/OpenAI TTS helpers.

**Backlog:** TASK-12920

**Spec:** `Docs/superpowers/specs/2026-07-07-tts-settings-voice-preview-design.md`

---

## File Map

- Modify: `apps/packages/ui/src/components/Option/Settings/TTSModeSettings.tsx`
  - Add preview state, cleanup, validation, browser playback, and server-backed playback.
  - Add a compact preview row near provider-specific voice/model controls.
- Modify: `apps/packages/ui/src/components/Option/Settings/__tests__/TTSModeSettings.test.tsx`
  - Extend the existing component harness with mocks for `tldwClient`, `generateSpeech`, `generateOpenAITTS`, `Audio`, `URL`, and `speechSynthesis`.
- Modify only if needed: `apps/packages/ui/src/services/tts-provider.ts`
  - Add an `elevenLabsApiKey` override only if keeping the preview through `resolveTtsProviderContext` is smaller than direct component calls.
- Update: `backlog/tasks/task-12920 - Implement-TTS-settings-voice-preview.md`
  - Record touched files and verification.

## Task 1: Add Preview Tests First

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Settings/__tests__/TTSModeSettings.test.tsx`

- [x] **Step 1: Add mocks for preview dependencies**

Mock:

```tsx
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { synthesizeSpeech: synthesizeSpeechMock },
}))

vi.mock("@/services/elevenlabs", () => ({
  getVoices: getVoicesMock,
  getModels: getModelsMock,
  generateSpeech: generateSpeechMock,
}))

vi.mock("@/services/openai-tts", () => ({
  generateOpenAITTS: generateOpenAITTSMock,
}))
```

- [x] **Step 2: Add focused failing tests**

Cover these behaviors in the existing `describe` block or a new adjacent one:

```tsx
it("previews browser TTS with the unsaved voice and playback speed", async () => {})
it("previews tldw TTS with unsaved form values without saving settings", async () => {})
it("previews ElevenLabs with the unsaved API key instead of the saved key", async () => {})
it("stops active server preview by aborting the request and revoking the object URL", async () => {})
it("does not use websocket APIs when tldw streaming is enabled", async () => {})
```

- [x] **Step 3: Run the failing test file**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Settings/__tests__/TTSModeSettings.test.tsx --maxWorkers=1
```

Expected: new preview tests fail because the button/behavior does not exist.

## Task 2: Implement Minimal Preview in Settings

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Settings/TTSModeSettings.tsx`

- [x] **Step 1: Import the existing synthesis helpers**

Add only the needed imports:

```tsx
import { Play, Square, Loader2 } from "lucide-react"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { generateSpeech } from "@/services/elevenlabs"
import { generateOpenAITTS } from "@/services/openai-tts"
```

- [x] **Step 2: Add tiny preview state and cleanup**

Inside `TTSModeSettings`:

```tsx
const [previewState, setPreviewState] = React.useState<"idle" | "loading" | "playing">("idle")
const [previewError, setPreviewError] = React.useState("")
const previewAudioRef = React.useRef<HTMLAudioElement | null>(null)
const previewUrlRef = React.useRef<string | null>(null)
const previewAbortRef = React.useRef<AbortController | null>(null)
```

Add `stopPreview` that aborts, pauses, revokes, cancels browser speech, clears refs, and returns to idle.

- [x] **Step 3: Add provider-specific preview**

Use one fixed sample sentence.

Browser path:

```tsx
const utterance = new SpeechSynthesisUtterance(PREVIEW_TEXT)
utterance.voice = window.speechSynthesis.getVoices().find((voice) => voice.name === form.values.voice) || null
utterance.rate = Number(form.values.playbackSpeed) || 1
window.speechSynthesis.speak(utterance)
```

Server paths:

- `tldw`: call `tldwClient.synthesizeSpeech(PREVIEW_TEXT, { model, voice, responseFormat, speed, language, normalizationOptions, stream: false, signal })`
- `openai`: call `generateOpenAITTS({ text: PREVIEW_TEXT, model, voice, signal })`
- `elevenlabs`: call `generateSpeech(unsavedKey, PREVIEW_TEXT, voiceId, modelId, undefined, { signal })`

After a buffer returns, create a blob URL, play it with `new Audio(url)`, and revoke the URL on stop/end/error.

- [x] **Step 4: Add the UI row**

Place one compact row after provider-specific voice/model controls and before global response splitting:

```tsx
<Button
  data-testid="tts-preview-button"
  type="default"
  icon={previewIcon}
  loading={previewState === "loading"}
  onClick={previewState === "idle" ? handlePreview : stopPreview}
>
  {previewState === "playing" ? "Stop preview" : previewState === "loading" ? "Generating..." : "Preview voice"}
</Button>
```

Render `previewError` as a small `Alert` or text line below the button.

- [x] **Step 5: Run the focused tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Settings/__tests__/TTSModeSettings.test.tsx --maxWorkers=1
```

Expected: PASS.

## Task 3: Tighten and Verify

**Files:**
- Modify as needed: `apps/packages/ui/src/components/Option/Settings/TTSModeSettings.tsx`
- Modify as needed: `apps/packages/ui/src/components/Option/Settings/__tests__/TTSModeSettings.test.tsx`

- [x] **Step 1: Self-review for unnecessary code**

Remove any helper or state that is not needed by the tests and spec.

- [x] **Step 2: Run targeted UI test**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Settings/__tests__/TTSModeSettings.test.tsx --maxWorkers=1
```

Expected: PASS.

- [x] **Step 3: Run diff checks**

Run:

```bash
git diff --check
```

Expected: no output.

- [x] **Step 4: Run Bandit on touched Python scope**

No Python code should be touched. Record "Bandit skipped: frontend-only change, no Python touched" in TASK-12920.

## Task 4: Finish Tracking and Commit

**Files:**
- Modify: `backlog/tasks/task-12920 - Implement-TTS-settings-voice-preview.md`

- [x] **Step 1: Update Backlog**

Record modified files, verification commands, test result, and Bandit skip.

- [x] **Step 2: Commit**

Run:

```bash
git add apps/packages/ui/src/components/Option/Settings/TTSModeSettings.tsx apps/packages/ui/src/components/Option/Settings/__tests__/TTSModeSettings.test.tsx backlog/tasks/task-12920\ -\ Implement-TTS-settings-voice-preview.md
git commit -m "feat: add TTS settings voice preview"
```
