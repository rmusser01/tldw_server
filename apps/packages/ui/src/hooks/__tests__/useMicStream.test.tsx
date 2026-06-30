import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useMicStream } from "../useMicStream"

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

vi.stubGlobal("AudioContext", MockAudioContext)
vi.stubGlobal("navigator", {
  mediaDevices: { getUserMedia: mockGetUserMedia }
})

describe("useMicStream", () => {
  beforeEach(() => {
    vi.clearAllMocks()
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
})
