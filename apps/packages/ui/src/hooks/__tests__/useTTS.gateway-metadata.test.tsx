// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import type { TtsProviderContext } from "@/services/tts-provider"
import type { TtsClip } from "@/db/dexie/types"

const testState = vi.hoisted(() => ({
  context: null as TtsProviderContext | null,
  savedClip: null as TtsClip | null
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ error: vi.fn() })
}))

vi.mock("@/config/platform", () => ({ isChromiumTarget: false }))

vi.mock("@/utils/tts", () => ({
  splitMessageContent: () => ["First.", "Second.", "Third."]
}))

vi.mock("@/services/tts", () => ({
  getElevenLabsModel: vi.fn(async () => "eleven-model"),
  getElevenLabsVoiceId: vi.fn(async () => "eleven-voice"),
  getOpenAITTSModel: vi.fn(async () => "openai-model"),
  getOpenAITTSVoice: vi.fn(async () => "openai-voice"),
  getTldwTTSModel: vi.fn(async () => "Vendor/Model"),
  getTldwTTSVoice: vi.fn(async () => "Narrator"),
  getVoice: vi.fn(async () => "Browser Voice")
}))

vi.mock("@/services/tts-provider", () => ({
  resolveTtsProviderContext: vi.fn(async () => testState.context)
}))

vi.mock("@/db/dexie/tts-clips", () => ({
  saveTtsClip: vi.fn(async (clip: TtsClip) => {
    testState.savedClip = clip
  })
}))

import { useTTS } from "../useTTS"

class MockAudio {
  playbackRate = 1
  currentTime = 0
  onended: (() => void) | null = null
  onerror: (() => void) | null = null
  error: MediaError | null = null

  canPlayType = vi.fn(() => "probably")
  pause = vi.fn()
  play = vi.fn(async () => {
    queueMicrotask(() => this.onended?.())
  })
}

const audioResult = (
  actualBackend?: string,
  fallbackUsed = false
) => ({
  buffer: new Uint8Array([1, 2, 3]).buffer,
  format: "mp3",
  mimeType: "audio/mpeg",
  actualBackend,
  fallbackUsed
})

describe("useTTS gateway metadata", () => {
  beforeEach(() => {
    testState.savedClip = null
    vi.clearAllMocks()
    vi.stubGlobal("Audio", MockAudio)
    vi.stubGlobal("crypto", {
      ...globalThis.crypto,
      randomUUID: vi.fn(() => "clip-1")
    })
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:tts")
    })
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: vi.fn()
    })
  })

  it("persists requested, per-segment, ordered actual backend, and fallback provenance", async () => {
    const results = [
      audioResult("gateway:company-proxy"),
      audioResult("openrouter", true),
      audioResult("gateway:company-proxy")
    ]
    testState.context = {
      provider: "tldw",
      utterance: "First. Second. Third.",
      playbackSpeed: 1,
      supported: true,
      normalizeText: (text) => text,
      cacheSettings: {
        provider: "tldw",
        backend: "gateway:company-proxy",
        cacheable: false,
        model: "Vendor/Model",
        voice: "Narrator",
        format: "mp3"
      },
      formatInfo: { requested: "mp3", resolved: "mp3", isFallback: false },
      synthesize: vi.fn(async () => results.shift()!)
    }
    const { result } = renderHook(() => useTTS())

    await act(async () => {
      await result.current.speak({ utterance: "ignored", saveClip: true })
    })

    expect(testState.savedClip).toMatchObject({
      requestedBackend: "gateway:company-proxy",
      actualBackends: ["gateway:company-proxy", "openrouter"],
      fallbackUsed: true,
      segments: [
        {
          actualBackend: "gateway:company-proxy",
          fallbackUsed: false
        },
        { actualBackend: "openrouter", fallbackUsed: true },
        {
          actualBackend: "gateway:company-proxy",
          fallbackUsed: false
        }
      ]
    })
  })

  it("keeps legacy clip records free of gateway-only provenance", async () => {
    testState.context = {
      provider: "tldw",
      utterance: "First. Second. Third.",
      playbackSpeed: 1,
      supported: true,
      normalizeText: (text) => text,
      cacheSettings: {
        provider: "tldw",
        cacheable: true,
        model: "Vendor/Model",
        voice: "Narrator",
        format: "mp3"
      },
      formatInfo: { requested: "mp3", resolved: "mp3", isFallback: false },
      synthesize: vi.fn(async () => audioResult())
    }
    const { result } = renderHook(() => useTTS())

    await act(async () => {
      await result.current.speak({ utterance: "ignored", saveClip: true })
    })

    expect(testState.savedClip?.requestedBackend).toBeUndefined()
    expect(testState.savedClip?.actualBackends).toBeUndefined()
    expect(testState.savedClip?.fallbackUsed).toBeUndefined()
    expect(testState.savedClip?.segments[0].actualBackend).toBeUndefined()
    expect(testState.savedClip?.segments[0].fallbackUsed).toBeUndefined()
  })
})
