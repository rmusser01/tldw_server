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
  play() {
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
