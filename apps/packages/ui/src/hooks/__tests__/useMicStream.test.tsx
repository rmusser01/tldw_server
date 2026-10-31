import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useMicStream } from "../useMicStream"

const AUDIO_CAPTURE_COORDINATOR_KEY = Symbol.for(
  "tldw.audioCaptureSessionCoordinator"
)
const mockTrackStop = vi.fn()
const mockGetUserMedia = vi.fn()
const mockProcessorDisconnect = vi.fn()
const mockSourceDisconnect = vi.fn()
const mockContextClose = vi.fn()
let mockProcessor: ScriptProcessorNode | null = null

const createMockStream = () =>
  ({
    getTracks: () => [{ stop: mockTrackStop }]
  }) as unknown as MediaStream

class MockAudioContext {
  destination = { kind: "destination" } as unknown as AudioDestinationNode

  createMediaStreamSource() {
    return {
      connect: vi.fn(),
      disconnect: mockSourceDisconnect
    } as unknown as MediaStreamAudioSourceNode
  }

  createScriptProcessor() {
    mockProcessor = {
      connect: vi.fn(),
      disconnect: mockProcessorDisconnect,
      onaudioprocess: null
    } as unknown as ScriptProcessorNode
    return mockProcessor
  }

  close = mockContextClose
}

const emitAudioFrame = (samples: Float32Array) => {
  mockProcessor?.onaudioprocess?.({
    inputBuffer: {
      getChannelData: () => samples
    }
  } as unknown as AudioProcessingEvent)
}

const getActiveAudioCaptureOwner = () =>
  (globalThis as any)[AUDIO_CAPTURE_COORDINATOR_KEY]?.getActiveOwner?.()

vi.stubGlobal("AudioContext", MockAudioContext)
vi.stubGlobal("navigator", {
  mediaDevices: { getUserMedia: mockGetUserMedia }
})

describe("useMicStream", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    delete (globalThis as any)[AUDIO_CAPTURE_COORDINATOR_KEY]
    mockProcessor = null
    mockGetUserMedia.mockResolvedValue(createMockStream())
  })

  it("passes the selected deviceId to getUserMedia for PCM streaming", async () => {
    const { result } = renderHook(() => useMicStream(vi.fn()))

    await act(async () => {
      await result.current.start({ deviceId: "usb-1" })
    })

    expect(mockGetUserMedia).toHaveBeenCalledWith({
      audio: { deviceId: { exact: "usb-1" } }
    })
  })

  it("falls back to the default microphone when no deviceId is provided", async () => {
    const { result } = renderHook(() => useMicStream(vi.fn()))

    await act(async () => {
      await result.current.start()
    })

    expect(mockGetUserMedia).toHaveBeenCalledWith({ audio: true })
  })

  it("emits PCM16 chunks by default for existing callers", async () => {
    const onChunk = vi.fn()
    const { result } = renderHook(() => useMicStream(onChunk))

    await act(async () => {
      await result.current.start()
    })

    emitAudioFrame(new Float32Array([-1, -0.5, 0, 0.5, 1, 2]))

    const chunk = onChunk.mock.calls[0]?.[0] as ArrayBuffer
    const view = new DataView(chunk)
    expect(chunk.byteLength).toBe(12)
    expect(view.getInt16(0, true)).toBe(-32768)
    expect(view.getInt16(2, true)).toBe(-16384)
    expect(view.getInt16(4, true)).toBe(0)
    expect(view.getInt16(6, true)).toBe(16383)
    expect(view.getInt16(8, true)).toBe(32767)
    expect(view.getInt16(10, true)).toBe(32767)
  })

  it("rejects a retired processor's queued frame after Stop and a new capture", async () => {
    const onChunk = vi.fn()
    const { result } = renderHook(() => useMicStream(onChunk))
    await act(async () => { await result.current.start() })
    const retiredProcessor = mockProcessor
    act(() => result.current.stop())
    await act(async () => { await result.current.start() })
    retiredProcessor?.onaudioprocess?.({
      inputBuffer: { getChannelData: () => new Float32Array([0.5]) }
    } as unknown as AudioProcessingEvent)
    expect(onChunk).not.toHaveBeenCalled()
    emitAudioFrame(new Float32Array([0.25]))
    expect(onChunk).toHaveBeenCalledOnce()
  })

  it("a retired permission rejection cannot stop a newer microphone owner", async () => {
    let rejectOld!: (error: Error) => void
    mockGetUserMedia.mockImplementationOnce(() => new Promise((_resolve, reject) => { rejectOld = reject }))
    const { result } = renderHook(() => useMicStream(vi.fn()))
    let oldStart!: Promise<void>
    await act(async () => { oldStart = result.current.start(); await Promise.resolve() })
    act(() => result.current.stop())
    await act(async () => { await result.current.start() })
    await act(async () => {
      rejectOld(new Error("old permission rejected"))
      await expect(oldStart).rejects.toThrow("old permission rejected")
    })
    expect(result.current.active).toBe(true)
    expect(getActiveAudioCaptureOwner()).toBe("live_voice")
  })

  it("can emit Float32 chunks for backend VAD/STT streams", async () => {
    const onChunk = vi.fn()
    const { result } = renderHook(() =>
      useMicStream(onChunk, { format: "float32" })
    )

    await act(async () => {
      await result.current.start()
    })

    emitAudioFrame(new Float32Array([-0.25, 0, 0.75]))

    const chunk = onChunk.mock.calls[0]?.[0] as ArrayBuffer
    expect(chunk.byteLength).toBe(12)
    expect(Array.from(new Float32Array(chunk))).toEqual([-0.25, 0, 0.75])
  })

  it("claims and releases the configured capture owner", async () => {
    const { result } = renderHook(() =>
      useMicStream(vi.fn(), { owner: "voice_chat" })
    )

    await act(async () => {
      await result.current.start()
    })

    expect(getActiveAudioCaptureOwner()).toBe("voice_chat")

    act(() => {
      result.current.stop()
    })

    expect(getActiveAudioCaptureOwner()).toBeNull()
  })
})
