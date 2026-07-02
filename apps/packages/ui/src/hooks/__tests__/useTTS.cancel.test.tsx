import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { useTTS } from "@/hooks/useTTS"

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
  useTranslation: () => ({ t: (_key: string, fallback?: string) => fallback ?? _key })
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

describe("useTTS cancel-during-playback", () => {
  afterEach(() => {
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
})
