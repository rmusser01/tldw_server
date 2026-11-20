import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, describe, expect, it, vi } from "vitest"

import { useDocumentTTS } from "@/hooks/document-workspace/useDocumentTTS"

const mocks = vi.hoisted(() => ({
  fetchTldwVoices: vi.fn(async () => []),
  synthesizeSpeech: vi.fn(async () => new ArrayBuffer(8)),
  getTldwTTSModel: vi.fn(async () => "KittenML/kitten-tts-nano-0.8"),
  getTldwTTSVoice: vi.fn(async () => "Bella")
}))

const requestScope = { config: { serverUrl: "http://fixture.test", authMode: "multi-user" }, userId: "alice" }
vi.mock("@/services/tldw", () => ({ tldwClient: { synthesizeSpeech: mocks.synthesizeSpeech } }))
vi.mock("@/hooks/document-workspace/useDocumentOperation", () => ({
  useDocumentOperation: () => ({
    requestScope,
    signal: new AbortController().signal,
    isCurrent: () => true
  })
}))

vi.mock("@/services/tldw/audio-voices", () => ({
  fetchTldwVoices: mocks.fetchTldwVoices
}))

vi.mock("@/services/tts", () => ({
  DEFAULT_TLDW_TTS_VOICE: "Bella",
  getTldwTTSModel: mocks.getTldwTTSModel,
  getTldwTTSVoice: mocks.getTldwTTSVoice
}))

const buildWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false
      }
    }
  })

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

describe("useDocumentTTS", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it("falls back to defaults when localStorage reads throw", async () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("storage blocked")
    })

    const { result } = renderHook(() => useDocumentTTS(), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.voicesLoading).toBe(false)
    })

    expect(result.current.voice).toBe("Bella")
    expect(result.current.speed).toBe(1)
    expect(result.current.volume).toBe(1)
  })

  it("uses configured tldw model and voice when speaking", async () => {
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:tts-audio")
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {})

    class MockAudio {
      paused = true
      ended = false
      duration = 0
      currentTime = 0
      volume = 1
      onended: (() => void) | null = null
      onerror: (() => void) | null = null
      onplay: (() => void) | null = null
      onpause: (() => void) | null = null
      ontimeupdate: (() => void) | null = null

      constructor(public src: string) {}

      pause() {
        this.paused = true
        this.onpause?.()
      }

      play() {
        this.paused = false
        this.onplay?.()
        return Promise.resolve()
      }
    }

    vi.stubGlobal("Audio", MockAudio)

    const { result } = renderHook(() => useDocumentTTS(), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.voicesLoading).toBe(false)
    })

    await act(async () => {
      await result.current.speak("Read this text")
    })

    expect(mocks.synthesizeSpeech).toHaveBeenCalledWith(
      "Read this text",
      expect.objectContaining({
        voice: "Bella",
        model: "KittenML/kitten-tts-nano-0.8",
        speed: 1,
        responseFormat: "mp3",
        requestScope
      })
    )

    expect(result.current.voice).toBe("Bella")
  })

  it("falls back to the configured tldw voice when local storage has no voice", async () => {
    mocks.getTldwTTSVoice.mockResolvedValue("Jasper")
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:tts-audio")
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {})

    class MockAudio {
      paused = true
      ended = false
      duration = 0
      currentTime = 0
      volume = 1
      onended: (() => void) | null = null
      onerror: (() => void) | null = null
      onplay: (() => void) | null = null
      onpause: (() => void) | null = null
      ontimeupdate: (() => void) | null = null

      constructor(public src: string) {}

      pause() {
        this.paused = true
        this.onpause?.()
      }

      play() {
        this.paused = false
        this.onplay?.()
        return Promise.resolve()
      }
    }

    vi.stubGlobal("Audio", MockAudio)

    const { result } = renderHook(() => useDocumentTTS(), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.voicesLoading).toBe(false)
    })

    await act(async () => {
      await result.current.speak("Configured voice text")
    })

    expect(mocks.synthesizeSpeech).toHaveBeenCalledWith(
      "Configured voice text",
      expect.objectContaining({
        voice: "Jasper",
        model: "KittenML/kitten-tts-nano-0.8",
        speed: 1,
        responseFormat: "mp3",
        requestScope
      })
    )
    expect(result.current.voice).toBe("Jasper")
  })

  it("revokes the object URL when audio playback errors", async () => {
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:tts-error")
    const revokeSpy = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => {})

    let lastAudio: any = null
    class MockAudio {
      paused = true
      ended = false
      duration = 0
      currentTime = 0
      volume = 1
      onended: (() => void) | null = null
      onerror: (() => void) | null = null
      onplay: (() => void) | null = null
      onpause: (() => void) | null = null
      ontimeupdate: (() => void) | null = null

      constructor(public src: string) {
        lastAudio = this
      }

      pause() {
        this.paused = true
        this.onpause?.()
      }

      play() {
        this.paused = false
        this.onplay?.()
        return Promise.resolve()
      }
    }

    vi.stubGlobal("Audio", MockAudio)

    const { result } = renderHook(() => useDocumentTTS(), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.voicesLoading).toBe(false)
    })

    await act(async () => {
      await result.current.speak("Broken audio")
    })

    expect(revokeSpy).not.toHaveBeenCalledWith("blob:tts-error")

    act(() => {
      lastAudio?.onerror?.()
    })

    expect(revokeSpy).toHaveBeenCalledWith("blob:tts-error")
    expect(result.current.state.error).toBe("Failed to play audio")
  })

  it("does not overwrite a user-selected voice when configured voice resolves later", async () => {
    let resolveVoice: ((value: string) => void) | undefined
    const pendingVoice = new Promise<string>((resolve) => {
      resolveVoice = resolve
    })
    mocks.getTldwTTSVoice.mockReturnValue(pendingVoice)

    const { result } = renderHook(() => useDocumentTTS(), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.voicesLoading).toBe(false)
    })

    act(() => {
      result.current.setVoice("Luna")
    })

    expect(result.current.voice).toBe("Luna")

    await act(async () => {
      resolveVoice?.("Bella")
      await pendingVoice
    })

    await waitFor(() => {
      expect(result.current.voice).toBe("Luna")
    })
  })
})
