// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import type { TtsProviderContext } from "@/services/tts-provider"
import type { AudioChapter } from "@/store/audiobook-studio"

const testState = vi.hoisted(() => ({
  context: null as TtsProviderContext | null
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ warning: vi.fn() })
}))

vi.mock("@/services/tts-provider", () => ({
  resolveTtsProviderContext: vi.fn(async () => testState.context)
}))

import { serializeChapters } from "@/db/dexie/audiobook-projects"
import { useAudiobookGeneration } from "../useAudiobookGeneration"
import { useAudiobookStudioStore } from "@/store/audiobook-studio"

class MockAudio extends EventTarget {
  duration = 12.5

  addEventListener(
    type: string,
    listener: EventListenerOrEventListenerObject | null,
    options?: boolean | AddEventListenerOptions
  ): void {
    super.addEventListener(type, listener, options)
    if (type === "loadedmetadata") {
      queueMicrotask(() => this.dispatchEvent(new Event("loadedmetadata")))
    }
  }
}

const makeChapter = (overrides: Partial<AudioChapter> = {}): AudioChapter => ({
  id: "chapter-1",
  title: "Chapter 1",
  content: "Narrate this chapter.",
  order: 0,
  voiceConfig: {
    provider: "tldw",
    tldwBackend: "gateway:company-proxy",
    tldwAllowFallback: true
  },
  status: "pending",
  ...overrides
})

describe("useAudiobookGeneration gateway metadata", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal("Audio", MockAudio)
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:chapter")
    })
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: vi.fn()
    })
    useAudiobookStudioStore.setState({
      chapters: [],
      isGenerating: false,
      generationQueue: [],
      currentGeneratingId: null
    })
  })

  it("persists requested, actual backend, and fallback metadata on generated chapters and project JSON", async () => {
    const chapter = makeChapter()
    useAudiobookStudioStore.setState({ chapters: [chapter] })
    testState.context = {
      provider: "tldw",
      utterance: chapter.content,
      playbackSpeed: 1,
      supported: true,
      normalizeText: (text) => text,
      cacheSettings: {
        provider: "tldw",
        backend: "gateway:company-proxy",
        cacheable: false
      },
      synthesize: vi.fn(async () => ({
        buffer: new Uint8Array([1, 2, 3]).buffer,
        format: "mp3",
        mimeType: "audio/mpeg",
        actualBackend: "openrouter",
        fallbackUsed: true
      }))
    }
    const { result } = renderHook(() => useAudiobookGeneration())

    await act(async () => {
      await result.current.generateSingleChapter(chapter)
    })

    const completed = useAudiobookStudioStore.getState().chapters[0]
    expect(completed).toMatchObject({
      status: "completed",
      requestedBackend: "gateway:company-proxy",
      actualBackend: "openrouter",
      fallbackUsed: true
    })
    expect(serializeChapters([completed])).toMatchObject([
      {
        requestedBackend: "gateway:company-proxy",
        actualBackend: "openrouter",
        fallbackUsed: true
      }
    ])
  })

  it("keeps old project chapters without provenance readable and serializable", () => {
    const oldChapter = makeChapter({
      voiceConfig: { provider: "tldw" },
      status: "completed"
    })

    useAudiobookStudioStore.getState().setChapters([oldChapter])

    const restored = useAudiobookStudioStore.getState().chapters[0]
    expect(restored.requestedBackend).toBeUndefined()
    expect(restored.actualBackend).toBeUndefined()
    expect(restored.fallbackUsed).toBeUndefined()
    expect(serializeChapters([restored])[0]).toMatchObject({
      id: "chapter-1",
      status: "completed"
    })
  })
})
