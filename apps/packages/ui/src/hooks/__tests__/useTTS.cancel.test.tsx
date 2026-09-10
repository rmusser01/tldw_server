import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { useTTS } from "@/hooks/useTTS"
import { resolveTtsProviderContext } from "@/services/tts-provider"
import { getVoice } from "@/services/tts"

const mocks = vi.hoisted(() => ({
  synthesize: vi.fn(async () => ({
    buffer: new ArrayBuffer(8),
    mimeType: "audio/mpeg",
    format: "mp3"
  }))
}))

vi.mock("@/services/tts-provider", () => ({
  resolveTtsProviderContext: vi.fn(async () => ({
    provider: "openai",
    utterance: "Hello world.",
    playbackSpeed: 1,
    synthesize: mocks.synthesize,
    supported: true,
    formatInfo: { resolved: "mp3" }
  }))
}))

vi.mock("@/utils/tts", () => ({
  splitMessageContent: () => ["Hello world."]
}))

vi.mock("@/services/tts", () => ({
  getElevenLabsModel: vi.fn(async () => "el-model"),
  getElevenLabsVoiceId: vi.fn(async () => "el-voice"),
  getOpenAITTSModel: vi.fn(async () => "oa-model"),
  getOpenAITTSVoice: vi.fn(async () => "oa-voice"),
  getTldwTTSModel: vi.fn(async () => "tldw-model"),
  getTldwTTSVoice: vi.fn(async () => "tldw-voice"),
  getVoice: vi.fn(async () => "voice")
}))

vi.mock("@/config/platform", () => ({ isChromiumTarget: false }))

vi.mock("@/db/dexie/tts-clips", () => ({ saveTtsClip: vi.fn() }))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    error: vi.fn(),
    warning: vi.fn(),
    success: vi.fn(),
    info: vi.fn(),
    open: vi.fn()
  })
}))

// Audio whose play() never settles, simulating playback still in progress when
// the user hits Stop.
class HangingAudio {
  src: string
  playbackRate = 1
  currentTime = 0
  onended: (() => void) | null = null
  onerror: (() => void) | null = null
  error: unknown = null
  constructor(src?: string) {
    this.src = src ?? ""
  }
  canPlayType() {
    return "probably"
  }
  play() {
    return new Promise<void>(() => {})
  }
  pause() {}
}

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}
const audioResult = () => ({
  buffer: new ArrayBuffer(8),
  mimeType: "audio/mpeg",
  format: "mp3"
})

describe("useTTS cancel-during-playback", () => {
  afterEach(() => {
    vi.clearAllMocks()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it("frees the segment object URL and settles the playback promise when cancelled mid-playback", async () => {
    const createObjectURLSpy = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:seg-0")
    const revokeSpy = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => {})
    vi.stubGlobal("Audio", HangingAudio)

    const { result } = renderHook(() => useTTS())

    let speakPromise: Promise<void> | undefined
    act(() => {
      speakPromise = result.current.speak({ utterance: "Hello world." })
    })

    // Wait until playback has actually started (object URL created for the segment).
    await waitFor(() => expect(createObjectURLSpy).toHaveBeenCalled())
    expect(revokeSpy).not.toHaveBeenCalled()

    // Stop mid-playback: the URL must be revoked and the in-flight promise settled.
    act(() => {
      result.current.cancel()
    })

    expect(revokeSpy).toHaveBeenCalledWith("blob:seg-0")

    // The generator must unwind rather than hang forever.
    await act(async () => {
      await speakPromise
    })

    expect(result.current.isSpeaking).toBe(false)
  })

  it("does not start synthesis after cancellation while provider settings are loading", async () => {
    const context =
      deferred<Awaited<ReturnType<typeof resolveTtsProviderContext>>>()
    vi.mocked(resolveTtsProviderContext).mockReturnValueOnce(context.promise)
    const create = vi.spyOn(URL, "createObjectURL")
    const { result } = renderHook(() => useTTS())
    let pending!: Promise<void>
    act(() => {
      pending = result.current.speak({ utterance: "old" })
      result.current.cancel()
    })
    await act(async () => {
      context.resolve({
        provider: "openai",
        utterance: "old",
        playbackSpeed: 1,
        supported: true,
        synthesize: mocks.synthesize
      })
      await pending
    })
    expect(mocks.synthesize).not.toHaveBeenCalled()
    expect(create).not.toHaveBeenCalled()
  })

  it.each(["cancel", "unmount"] as const)(
    "does not create or play delayed audio after %s",
    async (action) => {
      const synthesis = deferred<ReturnType<typeof audioResult>>()
      mocks.synthesize.mockReturnValueOnce(synthesis.promise)
      const create = vi
        .spyOn(URL, "createObjectURL")
        .mockReturnValue("blob:late")
      const play = vi
        .spyOn(HangingAudio.prototype, "play")
        .mockImplementation(function () {
          queueMicrotask(() => this.onended?.())
          return Promise.resolve()
        })
      vi.stubGlobal("Audio", HangingAudio)
      const { result, unmount } = renderHook(() => useTTS())
      let pending!: Promise<void>
      act(() => {
        pending = result.current.speak({ utterance: "old" })
      })
      await waitFor(() => expect(mocks.synthesize).toHaveBeenCalled())
      act(() => {
        if (action === "cancel") result.current.cancel()
        else unmount()
      })
      await act(async () => {
        synthesis.resolve(audioResult())
        await pending
      })
      expect(create).not.toHaveBeenCalled()
      expect(play).not.toHaveBeenCalled()
    }
  )

  it("does not let superseded synthesis play or clear the newer playback state", async () => {
    const oldSynthesis = deferred<ReturnType<typeof audioResult>>()
    mocks.synthesize.mockReturnValueOnce(oldSynthesis.promise)
    const create = vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:new")
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {})
    const play = vi.spyOn(HangingAudio.prototype, "play")
    vi.stubGlobal("Audio", HangingAudio)
    const { result } = renderHook(() => useTTS())
    let oldSpeak!: Promise<void>
    let newSpeak!: Promise<void>
    act(() => {
      oldSpeak = result.current.speak({ utterance: "old" })
    })
    await waitFor(() => expect(mocks.synthesize).toHaveBeenCalledTimes(1))
    act(() => {
      newSpeak = result.current.speak({ utterance: "new" })
    })
    await waitFor(() => expect(play).toHaveBeenCalledTimes(1))
    await act(async () => {
      oldSynthesis.resolve(audioResult())
      await Promise.resolve()
    })
    expect(create).toHaveBeenCalledTimes(1)
    expect(play).toHaveBeenCalledTimes(1)
    expect(result.current.isSpeaking).toBe(true)
    act(() => result.current.cancel())
    await act(async () => {
      await Promise.all([oldSpeak, newSpeak])
    })
    expect(result.current.isSpeaking).toBe(false)
  })

  it("checks cancellation again after asynchronous audio buffer conversion", async () => {
    const buffer = deferred<ArrayBuffer>()
    const convert = vi.fn(() => buffer.promise)
    mocks.synthesize.mockResolvedValueOnce({
      ...audioResult(),
      buffer: { arrayBuffer: convert } as unknown as ArrayBuffer
    })
    const create = vi.spyOn(URL, "createObjectURL")
    const { result } = renderHook(() => useTTS())
    let pending!: Promise<void>
    act(() => {
      pending = result.current.speak({ utterance: "old" })
    })
    await waitFor(() => expect(convert).toHaveBeenCalled())
    act(() => result.current.cancel())
    await act(async () => {
      buffer.resolve(new ArrayBuffer(8))
      await pending
    })
    expect(create).not.toHaveBeenCalled()
  })

  it("does not start browser speech after cancellation while loading the voice", async () => {
    vi.mocked(resolveTtsProviderContext).mockResolvedValueOnce({
      provider: "browser",
      utterance: "browser voice",
      playbackSpeed: 1,
      supported: true
    })
    const voice = deferred<string>()
    vi.mocked(getVoice).mockReturnValueOnce(voice.promise)
    const browserSpeak = vi.fn()
    vi.stubGlobal("speechSynthesis", { speak: browserSpeak, cancel: vi.fn() })
    const { result } = renderHook(() => useTTS())
    let pending!: Promise<void>
    act(() => {
      pending = result.current.speak({ utterance: "browser voice" })
    })
    await waitFor(() => expect(getVoice).toHaveBeenCalled())
    act(() => result.current.cancel())
    await act(async () => {
      voice.resolve("voice")
      await pending
    })
    expect(browserSpeak).not.toHaveBeenCalled()
  })

  it.each(["end", "cancel"] as const)(
    "waits for browser playback to %s and ignores late callbacks",
    async (completion) => {
      vi.mocked(resolveTtsProviderContext).mockResolvedValueOnce({
        provider: "browser",
        utterance: "browser voice",
        playbackSpeed: 1,
        supported: true
      })
      class Utterance {
        constructor(public text: string) {}
      }
      let utterance: SpeechSynthesisUtterance
      const browserSpeak = vi.fn((value: SpeechSynthesisUtterance) => {
        utterance = value
      })
      vi.stubGlobal("SpeechSynthesisUtterance", Utterance)
      vi.stubGlobal("speechSynthesis", {
        speak: browserSpeak,
        cancel: vi.fn(),
        getVoices: () => []
      })
      const { result } = renderHook(() => useTTS())
      const settled = vi.fn()
      let pending!: Promise<void>
      act(() => {
        pending = result.current
          .speak({ utterance: "browser voice" })
          .then(settled)
      })
      await waitFor(() => expect(browserSpeak).toHaveBeenCalledTimes(1))
      expect(settled).not.toHaveBeenCalled()
      act(() => utterance.onstart?.({} as SpeechSynthesisEvent))
      expect(result.current.isSpeaking).toBe(true)
      const staleStart = utterance.onstart
      act(() => {
        if (completion === "cancel") result.current.cancel()
        else utterance.onend?.({} as SpeechSynthesisEvent)
      })
      await act(async () => {
        await pending
      })
      expect(settled).toHaveBeenCalledTimes(1)
      act(() => staleStart?.call(utterance, {} as SpeechSynthesisEvent))
      expect(result.current.isSpeaking).toBe(false)
    }
  )
})
