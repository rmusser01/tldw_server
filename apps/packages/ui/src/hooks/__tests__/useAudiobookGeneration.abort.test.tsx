import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { useAudiobookGeneration } from "@/hooks/useAudiobookGeneration"

const h = vi.hoisted(() => ({
  synthesize: vi.fn((_text: string, _opts?: { signal?: AbortSignal }) =>
    // Never resolves: keeps the chapter "in flight" so cancel can abort it.
    new Promise<{ buffer: ArrayBuffer; mimeType: string; format: string }>(
      () => {}
    )
  ),
  storeState: {
    updateChapter: vi.fn(),
    setIsGenerating: vi.fn(),
    setCurrentGeneratingId: vi.fn(),
    setGenerationQueue: vi.fn(),
    generationQueue: [] as string[]
  }
}))

vi.mock("@/services/tts-provider", () => ({
  resolveTtsProviderContext: vi.fn(async () => ({
    supported: true,
    provider: "tldw",
    utterance: "chapter text",
    synthesize: h.synthesize
  }))
}))

vi.mock("@/utils/tts-speed", () => ({
  applyVoiceSpeedOverrides: (config: unknown) => config
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_key: string, fallback?: string) => fallback ?? _key })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ warning: vi.fn(), error: vi.fn() })
}))

vi.mock("@/store/audiobook-studio", () => ({
  useAudiobookStudioStore: Object.assign(
    (selector: (state: typeof h.storeState) => unknown) => selector(h.storeState),
    { getState: () => h.storeState }
  )
}))

describe("useAudiobookGeneration cancel aborts in-flight synthesis", () => {
  afterEach(() => {
    vi.clearAllMocks()
  })

  it("threads an abort signal into synthesize and aborts it on cancelGeneration", async () => {
    const chapter = {
      id: "c1",
      title: "Chapter 1",
      content: "hello",
      order: 0,
      status: "pending",
      voiceConfig: {}
    } as any

    const { result } = renderHook(() => useAudiobookGeneration())

    act(() => {
      void result.current.generateAllChapters({ chapters: [chapter] })
    })

    await waitFor(() => expect(h.synthesize).toHaveBeenCalled())

    const options = h.synthesize.mock.calls[0][1]
    expect(options?.signal).toBeInstanceOf(AbortSignal)
    expect(options?.signal?.aborted).toBe(false)

    act(() => {
      result.current.cancelGeneration()
    })

    expect(options?.signal?.aborted).toBe(true)
  })
})
