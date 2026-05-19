import { describe, expect, it, vi, beforeEach } from "vitest"
import { renderHook, act } from "@testing-library/react"
import { useMultiRenderState } from "../useMultiRenderState"
import { resolveTtsProviderContext } from "@/services/tts-provider"

vi.mock("@/services/tts-provider", () => ({
  resolveTtsProviderContext: vi.fn().mockResolvedValue({
    provider: "tldw",
    utterance: "test text",
    playbackSpeed: 1,
    supported: true,
    synthesize: vi.fn().mockResolvedValue({
      buffer: new ArrayBuffer(100),
      format: "mp3",
      mimeType: "audio/mpeg"
    })
  })
}))

// Mock URL.createObjectURL and revokeObjectURL
const mockCreateObjectURL = vi.fn().mockReturnValue("blob:mock-url")
const mockRevokeObjectURL = vi.fn()
Object.defineProperty(globalThis, "URL", {
  value: {
    createObjectURL: mockCreateObjectURL,
    revokeObjectURL: mockRevokeObjectURL
  },
  writable: true
})

describe("useMultiRenderState", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("starts with empty renders", () => {
    const { result } = renderHook(() => useMultiRenderState())
    expect(result.current.renders).toEqual([])
    expect(result.current.playingId).toBeNull()
  })

  it("addRender creates a new entry with idle state", () => {
    const { result } = renderHook(() => useMultiRenderState())
    let id: string
    act(() => {
      id = result.current.addRender({
        provider: "tldw",
        voice: "af_heart",
        model: "kokoro",
        format: "mp3"
      })
    })
    expect(result.current.renders).toHaveLength(1)
    expect(result.current.renders[0].state).toBe("idle")
    expect(result.current.renders[0].config.voice).toBe("af_heart")
  })

  it("removeRender removes the entry", () => {
    const { result } = renderHook(() => useMultiRenderState())
    let id: string
    act(() => {
      id = result.current.addRender({
        provider: "tldw",
        voice: "af_heart"
      })
    })
    act(() => {
      result.current.removeRender(result.current.renders[0].id)
    })
    expect(result.current.renders).toHaveLength(0)
  })

  it("updateConfig changes the config of a render", () => {
    const { result } = renderHook(() => useMultiRenderState())
    act(() => {
      result.current.addRender({ provider: "tldw", voice: "af_heart" })
    })
    const id = result.current.renders[0].id
    act(() => {
      result.current.updateConfig(id, {
        provider: "tldw",
        voice: "am_adam",
        model: "kokoro"
      })
    })
    expect(result.current.renders[0].config.voice).toBe("am_adam")
  })

  it("clearAll removes all renders", () => {
    const { result } = renderHook(() => useMultiRenderState())
    act(() => {
      result.current.addRender({ provider: "tldw", voice: "v1" })
      result.current.addRender({ provider: "tldw", voice: "v2" })
      result.current.addRender({ provider: "tldw", voice: "v3" })
    })
    expect(result.current.renders).toHaveLength(3)
    act(() => {
      result.current.clearAll()
    })
    expect(result.current.renders).toHaveLength(0)
  })

  it("startPlaying sets playingId", () => {
    const { result } = renderHook(() => useMultiRenderState())
    act(() => {
      result.current.addRender({ provider: "tldw", voice: "af_heart" })
    })
    const id = result.current.renders[0].id
    act(() => {
      result.current.startPlaying(id)
    })
    expect(result.current.playingId).toBe(id)
  })

  it("stopPlaying clears playingId for that strip", () => {
    const { result } = renderHook(() => useMultiRenderState())
    act(() => {
      result.current.addRender({ provider: "tldw", voice: "af_heart" })
    })
    const id = result.current.renders[0].id
    act(() => {
      result.current.startPlaying(id)
    })
    act(() => {
      result.current.stopPlaying(id)
    })
    expect(result.current.playingId).toBeNull()
  })

  it("hasIdle is true when idle renders exist", () => {
    const { result } = renderHook(() => useMultiRenderState())
    act(() => {
      result.current.addRender({ provider: "tldw", voice: "af_heart" })
    })
    expect(result.current.hasIdle).toBe(true)
  })

  it("hasReady is false when no audio generated", () => {
    const { result } = renderHook(() => useMultiRenderState())
    act(() => {
      result.current.addRender({ provider: "tldw", voice: "af_heart" })
    })
    expect(result.current.hasReady).toBe(false)
  })

  it("handleStripEnded clears playingId when no queue", () => {
    const { result } = renderHook(() => useMultiRenderState())
    act(() => {
      result.current.addRender({ provider: "tldw", voice: "af_heart" })
    })
    const id = result.current.renders[0].id
    act(() => {
      result.current.startPlaying(id)
    })
    expect(result.current.playingId).toBe(id)
    act(() => {
      result.current.handleStripEnded(id)
    })
    expect(result.current.playingId).toBeNull()
  })

  it("playAllSequentially sets first strip as playing", () => {
    const { result } = renderHook(() => useMultiRenderState())
    act(() => {
      result.current.addRender({ provider: "tldw", voice: "v1" })
      result.current.addRender({ provider: "tldw", voice: "v2" })
    })
    // Manually set both strips to ready state with audioUrl
    act(() => {
      result.current.updateRender(result.current.renders[0].id, {
        state: "ready",
        audioUrl: "blob:url1"
      })
      result.current.updateRender(result.current.renders[1].id, {
        state: "ready",
        audioUrl: "blob:url2"
      })
    })
    act(() => {
      result.current.playAllSequentially()
    })
    expect(result.current.playingId).toBe(result.current.renders[0].id)
  })

  it("generateRender transitions to generating then ready", async () => {
    const { result } = renderHook(() => useMultiRenderState())
    act(() => {
      result.current.addRender({
        provider: "tldw",
        voice: "af_heart",
        model: "kokoro"
      })
    })
    const id = result.current.renders[0].id
    await act(async () => {
      await result.current.generateRender(id, "Hello world")
    })
    expect(result.current.renders[0].state).toBe("ready")
    expect(result.current.renders[0].audioUrl).toBe("blob:mock-url")
  })

  it("classifies and redacts render generation failures", async () => {
    vi.mocked(resolveTtsProviderContext).mockResolvedValueOnce({
      provider: "tldw",
      utterance: "Hello world",
      playbackSpeed: 1,
      supported: true,
      synthesize: vi.fn().mockRejectedValue(
        new Error("Request failed for sk_secret_inline")
      )
    })

    const { result } = renderHook(() => useMultiRenderState())
    act(() => {
      result.current.addRender({
        provider: "tldw",
        voice: "af_heart",
        model: "kokoro"
      })
    })
    const id = result.current.renders[0].id

    await act(async () => {
      await result.current.generateRender(id, "Hello world")
    })

    expect(result.current.renders[0].state).toBe("error")
    expect(result.current.renders[0].errorMessage).toContain(
      "Credentials need attention"
    )
    expect(result.current.renders[0].errorMessage).toContain(
      "Settings -> Speech"
    )
    expect(result.current.renders[0].errorSettingsHref).toBe("/settings/speech")
    expect(result.current.renders[0].errorMessage).not.toContain(
      "sk_secret_inline"
    )
  })
})
