import { act, renderHook } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { useStreamingAudioPlayer } from "@/hooks/useStreamingAudioPlayer"

// MediaSource that reports streaming as supported but never opens its source
// buffer, so we rely on the play() rejection below to flip into the fallback.
class MockMediaSource {
  static isTypeSupported() {
    return true
  }
  readyState = "closed"
  addEventListener() {}
  removeEventListener() {}
  addSourceBuffer() {
    return {
      updating: false,
      onupdateend: null,
      appendBuffer() {}
    }
  }
  endOfStream() {}
}

// Audio whose play() rejects, which flips streamFailedRef and forces finish()
// down the buffered-fallback path.
class RejectingAudio {
  src = ""
  autoplay = false
  onended: (() => void) | null = null
  onerror: (() => void) | null = null
  play(): Promise<void> {
    return Promise.reject(new Error("blocked"))
  }
  pause() {}
  canPlayType() {
    return "probably"
  }
}

describe("useStreamingAudioPlayer stream->buffer fallback", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it.each([true, false])("settles a rejected play promise when streaming=%s", async (streaming) => {
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:audio")
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {})
    vi.stubGlobal("MediaSource", MockMediaSource)
    vi.stubGlobal("Audio", RejectingAudio)
    const { result } = renderHook(() => useStreamingAudioPlayer())
    await act(async () => {
      result.current.start("mp3", streaming)
      if (!streaming) result.current.finish()
      await Promise.resolve()
    })
    expect(result.current.state.error).toBe("Audio playback blocked")
    expect(result.current.state.playing).toBe(false)
  })

  it("ignores a retired player's late rejection and decode error after a newer start", async () => {
    let rejectOld!: (error: Error) => void
    const instances: DeferredAudio[] = []
    class DeferredAudio extends RejectingAudio {
      constructor() { super(); instances.push(this) }
      play() {
        return instances.length === 1
          ? new Promise<void>((_resolve, reject) => { rejectOld = reject })
          : Promise.resolve()
      }
    }
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:audio")
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {})
    vi.stubGlobal("MediaSource", MockMediaSource)
    vi.stubGlobal("Audio", DeferredAudio)
    const { result } = renderHook(() => useStreamingAudioPlayer())
    act(() => result.current.start("mp3", true))
    const oldError = instances[0].onerror
    act(() => result.current.start("mp3", true))
    await act(async () => {
      rejectOld(new Error("late rejection"))
      oldError?.()
      await Promise.resolve()
    })
    expect(result.current.state.error).toBeNull()
    expect(result.current.state.playing).toBe(true)
  })

  it("revokes the MediaSource blob URL before overwriting it with the fallback blob URL", async () => {
    const created: string[] = []
    let counter = 0
    vi.spyOn(URL, "createObjectURL").mockImplementation(() => {
      const url = `blob:mock-${counter++}`
      created.push(url)
      return url
    })
    const revokeSpy = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => {})

    vi.stubGlobal("MediaSource", MockMediaSource)
    vi.stubGlobal("Audio", RejectingAudio)

    const { result } = renderHook(() => useStreamingAudioPlayer())

    await act(async () => {
      result.current.start("mp3", true)
      // Flush the play() rejection so streamFailedRef flips to true.
      await Promise.resolve()
      await Promise.resolve()
      await Promise.resolve()
    })

    // The first URL created is the MediaSource blob URL.
    const mediaSourceUrl = created[0]
    expect(mediaSourceUrl).toBe("blob:mock-0")
    expect(revokeSpy).not.toHaveBeenCalledWith(mediaSourceUrl)

    await act(async () => {
      result.current.finish()
    })

    // The fallback path must revoke the previous (MediaSource) URL, not leak it.
    expect(revokeSpy).toHaveBeenCalledWith(mediaSourceUrl)
    // And a fresh fallback blob URL is now in use.
    const fallbackUrl = created[created.length - 1]
    expect(fallbackUrl).not.toBe(mediaSourceUrl)

    // Cleanup on stop revokes the fallback URL too (no dangling blob).
    await act(async () => {
      result.current.stop()
    })
    expect(revokeSpy).toHaveBeenCalledWith(fallbackUrl)
  })
})
