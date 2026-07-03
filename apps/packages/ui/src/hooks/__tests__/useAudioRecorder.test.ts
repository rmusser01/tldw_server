import { describe, expect, it, vi, beforeEach, afterEach } from "vitest"
import { renderHook, act } from "@testing-library/react"

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockTrackStop = vi.fn()
const mockGetUserMedia = vi.fn().mockResolvedValue({
  getTracks: () => [{ stop: mockTrackStop }]
})

const AUDIO_CAPTURE_COORDINATOR_KEY = Symbol.for(
  "tldw.audioCaptureSessionCoordinator"
)

// Track constructed recorders so tests can trigger lifecycle events (onerror).
const recorderInstances: MockMediaRecorder[] = []
let failNextRecorderStart = false

class MockMediaRecorder {
  ondataavailable: ((e: { data: Blob }) => void) | null = null
  onstop: (() => void) | null = null
  onerror: ((e: unknown) => void) | null = null
  mimeType = "audio/webm"
  state = "inactive" as "inactive" | "recording"

  constructor() {
    recorderInstances.push(this)
  }

  start = vi.fn(() => {
    if (failNextRecorderStart) {
      failNextRecorderStart = false
      throw new Error("MediaRecorder start failed")
    }
    this.state = "recording"
  })

  stop = vi.fn(() => {
    this.state = "inactive"
    if (this.ondataavailable) {
      this.ondataavailable({
        data: new Blob(["audio-data"], { type: "audio/webm" })
      })
    }
    if (this.onstop) {
      this.onstop()
    }
  })

  emitError() {
    this.state = "inactive"
    this.onerror?.(new Event("error"))
  }
}

vi.stubGlobal("MediaRecorder", MockMediaRecorder)
vi.stubGlobal("navigator", {
  mediaDevices: { getUserMedia: mockGetUserMedia }
})

// Import after mocks are in place
import { useAudioRecorder } from "../useAudioRecorder"

describe("useAudioRecorder", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.clearAllMocks()
    recorderInstances.length = 0
    failNextRecorderStart = false
    mockGetUserMedia.mockResolvedValue({
      getTracks: () => [{ stop: mockTrackStop }]
    })
    // Reset the process-global capture coordinator so an owner left claimed by
    // one test (e.g. a still-"recording" hook) cannot make the next test's
    // reserveCaptureOwner() throw a capture-busy error.
    delete (
      globalThis as typeof globalThis & {
        [AUDIO_CAPTURE_COORDINATOR_KEY]?: unknown
      }
    )[AUDIO_CAPTURE_COORDINATOR_KEY]
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("starts in idle state with no blob and zero duration", () => {
    const { result } = renderHook(() => useAudioRecorder())

    expect(result.current.status).toBe("idle")
    expect(result.current.blob).toBeNull()
    expect(result.current.durationMs).toBe(0)
  })

  it("transitions to recording state after startRecording", async () => {
    const { result } = renderHook(() => useAudioRecorder())

    await act(async () => {
      await result.current.startRecording()
    })

    expect(result.current.status).toBe("recording")
    expect(mockGetUserMedia).toHaveBeenCalledWith({ audio: true })
  })

  it("passes the selected deviceId to getUserMedia when recording starts", async () => {
    const { result } = renderHook(() => useAudioRecorder())

    await act(async () => {
      await result.current.startRecording({ deviceId: "usb-1" })
    })

    expect(mockGetUserMedia).toHaveBeenCalledWith({
      audio: { deviceId: { exact: "usb-1" } }
    })
  })

  it("falls back to the default microphone when no deviceId is provided", async () => {
    const { result } = renderHook(() => useAudioRecorder())

    await act(async () => {
      await result.current.startRecording({})
    })

    expect(mockGetUserMedia).toHaveBeenCalledWith({ audio: true })
  })

  it("increments durationMs while recording", async () => {
    const { result } = renderHook(() => useAudioRecorder())

    await act(async () => {
      await result.current.startRecording()
    })

    expect(result.current.durationMs).toBe(0)

    act(() => {
      vi.advanceTimersByTime(200)
    })
    expect(result.current.durationMs).toBe(200)

    act(() => {
      vi.advanceTimersByTime(200)
    })
    expect(result.current.durationMs).toBe(400)
  })

  it("produces a blob on stop and returns to idle", async () => {
    const { result } = renderHook(() => useAudioRecorder())

    await act(async () => {
      await result.current.startRecording()
    })

    act(() => {
      vi.advanceTimersByTime(600)
    })

    act(() => {
      result.current.stopRecording()
    })

    expect(result.current.status).toBe("idle")
    expect(result.current.blob).toBeInstanceOf(Blob)
    expect(result.current.durationMs).toBe(600)
  })

  it("stops media tracks on stop", async () => {
    const { result } = renderHook(() => useAudioRecorder())

    await act(async () => {
      await result.current.startRecording()
    })

    act(() => {
      result.current.stopRecording()
    })

    expect(mockTrackStop).toHaveBeenCalled()
  })

  it("clears blob and duration with clearRecording", async () => {
    const { result } = renderHook(() => useAudioRecorder())

    await act(async () => {
      await result.current.startRecording()
    })
    act(() => {
      vi.advanceTimersByTime(400)
    })
    act(() => {
      result.current.stopRecording()
    })

    expect(result.current.blob).not.toBeNull()

    act(() => {
      result.current.clearRecording()
    })

    expect(result.current.blob).toBeNull()
    expect(result.current.durationMs).toBe(0)
  })

  it("accepts a blob via loadBlob", () => {
    const { result } = renderHook(() => useAudioRecorder())
    const externalBlob = new Blob(["external"], { type: "audio/webm" })

    act(() => {
      result.current.loadBlob(externalBlob, 5000)
    })

    expect(result.current.blob).toBe(externalBlob)
    expect(result.current.durationMs).toBe(5000)
    expect(result.current.status).toBe("idle")
  })

  it("does nothing if stopRecording is called while idle", () => {
    const { result } = renderHook(() => useAudioRecorder())

    // Should not throw
    act(() => {
      result.current.stopRecording()
    })

    expect(result.current.status).toBe("idle")
  })

  it("cleans up on unmount during recording", async () => {
    const { result, unmount } = renderHook(() => useAudioRecorder())

    await act(async () => {
      await result.current.startRecording()
    })

    unmount()

    expect(mockTrackStop).toHaveBeenCalled()
  })

  it("stops media tracks and returns to idle when the recorder errors", async () => {
    const { result } = renderHook(() => useAudioRecorder())

    await act(async () => {
      await result.current.startRecording()
    })
    expect(result.current.status).toBe("recording")

    // A MediaRecorder error that does not also fire onstop must still stop the
    // mic track (privacy: browser indicator must go off).
    act(() => {
      recorderInstances[recorderInstances.length - 1].emitError()
    })

    expect(mockTrackStop).toHaveBeenCalled()
    expect(result.current.status).toBe("idle")
  })

  it("releases the capture owner on recorder error so a new recording can start", async () => {
    const { result } = renderHook(() => useAudioRecorder())

    await act(async () => {
      await result.current.startRecording()
    })
    act(() => {
      recorderInstances[recorderInstances.length - 1].emitError()
    })

    // If the owner was released, a fresh start must acquire the mic again
    // instead of throwing a capture-busy error.
    await act(async () => {
      await result.current.startRecording()
    })

    expect(result.current.status).toBe("recording")
    expect(mockGetUserMedia).toHaveBeenCalledTimes(2)
  })

  it("does not orphan a stream on a rapid double-start", async () => {
    const streams = [
      { getTracks: () => [{ stop: vi.fn() }] },
      { getTracks: () => [{ stop: vi.fn() }] }
    ]
    let call = 0
    mockGetUserMedia.mockImplementation(() =>
      Promise.resolve(streams[call++] ?? streams[0])
    )

    const { result } = renderHook(() => useAudioRecorder())

    await act(async () => {
      // Fire two starts before the first getUserMedia resolves.
      const first = result.current.startRecording()
      const second = result.current.startRecording()
      await Promise.all([first, second])
    })

    // The synchronous re-entry guard must short-circuit the second start, so
    // only one stream is ever acquired (no orphaned, un-stopped track).
    expect(mockGetUserMedia).toHaveBeenCalledTimes(1)
    expect(recorderInstances).toHaveLength(1)
    expect(result.current.status).toBe("recording")
  })

  it("ignores a start while already recording", async () => {
    const { result } = renderHook(() => useAudioRecorder())

    await act(async () => {
      await result.current.startRecording()
    })
    await act(async () => {
      await result.current.startRecording()
    })

    // A start while already recording must not acquire a second stream.
    expect(mockGetUserMedia).toHaveBeenCalledTimes(1)
    expect(recorderInstances).toHaveLength(1)
  })

  it("clears recorder state when MediaRecorder.start throws synchronously", async () => {
    const { result } = renderHook(() => useAudioRecorder())

    failNextRecorderStart = true

    await act(async () => {
      await expect(result.current.startRecording()).rejects.toThrow(
        "MediaRecorder start failed"
      )
    })

    await act(async () => {
      await result.current.startRecording()
    })

    expect(mockGetUserMedia).toHaveBeenCalledTimes(2)
    expect(result.current.status).toBe("recording")
  })
})
