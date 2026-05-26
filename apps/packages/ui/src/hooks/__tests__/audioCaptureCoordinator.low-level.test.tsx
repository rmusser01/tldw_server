import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const AUDIO_CAPTURE_COORDINATOR_KEY = Symbol.for(
  "tldw.audioCaptureSessionCoordinator"
)

const mockTrackStop = vi.fn()
const mockGetUserMedia = vi.fn()
const mockProcessorDisconnect = vi.fn()
const mockSourceDisconnect = vi.fn()
const mockContextClose = vi.fn()

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
    return {
      connect: vi.fn(),
      disconnect: mockProcessorDisconnect,
      onaudioprocess: null
    } as unknown as ScriptProcessorNode
  }

  close = mockContextClose
}

class MockMediaRecorder {
  ondataavailable: ((e: { data: Blob }) => void) | null = null
  onstop: (() => void | Promise<void>) | null = null
  onerror: ((e: unknown) => void) | null = null
  mimeType = "audio/webm"
  state = "inactive" as "inactive" | "recording"

  start = vi.fn(() => {
    this.state = "recording"
  })

  stop = vi.fn(() => {
    this.state = "inactive"
    this.onstop?.()
  })
}

vi.stubGlobal("AudioContext", MockAudioContext)
vi.stubGlobal("MediaRecorder", MockMediaRecorder)
vi.stubGlobal("navigator", {
  mediaDevices: { getUserMedia: mockGetUserMedia }
})

import { useAudioRecorder } from "../useAudioRecorder"
import { useMicStream } from "../useMicStream"

describe("low-level audio capture coordinator behavior", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockGetUserMedia.mockReset()
    mockGetUserMedia.mockResolvedValue(createMockStream())
    delete (
      globalThis as typeof globalThis & {
        [AUDIO_CAPTURE_COORDINATOR_KEY]?: unknown
      }
    )[AUDIO_CAPTURE_COORDINATOR_KEY]
  })

  it("blocks a second low-level hook while the first capture is active", async () => {
    const micHook = renderHook(() => useMicStream(vi.fn()))
    const recorderHook = renderHook(() => useAudioRecorder())

    await act(async () => {
      await micHook.result.current.start({ deviceId: "usb-1" })
    })

    let blockedError: unknown
    await act(async () => {
      try {
        await recorderHook.result.current.startRecording({ deviceId: "usb-2" })
      } catch (error) {
        blockedError = error
      }
    })

    expect(blockedError).toBeInstanceOf(Error)
    expect(mockGetUserMedia).toHaveBeenCalledTimes(1)
    expect(recorderHook.result.current.status).toBe("idle")
  })

  it("releases a failed startup reservation so a later hook can acquire capture", async () => {
    mockGetUserMedia
      .mockRejectedValueOnce(new Error("permission denied"))
      .mockResolvedValueOnce(createMockStream())

    const micHook = renderHook(() => useMicStream(vi.fn()))
    const recorderHook = renderHook(() => useAudioRecorder())

    let startupError: unknown
    await act(async () => {
      try {
        await micHook.result.current.start({ deviceId: "usb-1" })
      } catch (error) {
        startupError = error
      }
    })

    await act(async () => {
      await recorderHook.result.current.startRecording({ deviceId: "usb-2" })
    })

    expect(startupError).toBeInstanceOf(Error)
    expect(mockGetUserMedia).toHaveBeenCalledTimes(2)
    expect(recorderHook.result.current.status).toBe("recording")
  })
})
