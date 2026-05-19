import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, describe, expect, it, vi } from "vitest"

import { useDocumentTTS } from "@/hooks/document-workspace/useDocumentTTS"

const mocks = vi.hoisted(() => ({
  fetchTldwVoices: vi.fn(async () => []),
  getTldwTTSModel: vi.fn(async () => "KittenML/kitten-tts-nano-0.8"),
  getTldwTTSVoice: vi.fn(async () => "Bella")
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
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      blob: vi.fn().mockResolvedValue(new Blob(["audio"]))
    })
    vi.stubGlobal("fetch", fetchMock)

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

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/audio/speech",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          input: "Read this text",
          voice: "Bella",
          model: "KittenML/kitten-tts-nano-0.8",
          speed: 1,
          response_format: "mp3"
        })
      })
    )

    expect(result.current.voice).toBe("Bella")
  })

  it("falls back to the configured tldw voice when local storage has no voice", async () => {
    mocks.getTldwTTSVoice.mockResolvedValue("Jasper")
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:tts-audio")
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {})
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      blob: vi.fn().mockResolvedValue(new Blob(["audio"]))
    })
    vi.stubGlobal("fetch", fetchMock)

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

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/audio/speech",
      expect.objectContaining({
        body: JSON.stringify({
          input: "Configured voice text",
          voice: "Jasper",
          model: "KittenML/kitten-tts-nano-0.8",
          speed: 1,
          response_format: "mp3"
        })
      })
    )
    expect(result.current.voice).toBe("Jasper")
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
