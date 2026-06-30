import { beforeEach, describe, expect, it, vi } from "vitest"

const mockBgRequestClient = vi.hoisted(() => vi.fn())
const mockBgUpload = vi.hoisted(() => vi.fn())

vi.mock("@/services/background-proxy", () => ({
  bgRequestClient: mockBgRequestClient,
  bgUpload: mockBgUpload
}))

import { convertChatterboxVoice, unloadTtsProvider } from "../voice-cloning"

const makeFile = (name = "source.wav", type = "audio/wav", bytes = [1, 2, 3]) => {
  const data = Uint8Array.from(bytes).buffer
  return {
    name,
    type,
    arrayBuffer: vi.fn(async () => data)
  } as unknown as File
}

describe("voice cloning service", () => {
  beforeEach(() => {
    mockBgRequestClient.mockReset()
    mockBgUpload.mockReset()
  })

  it("converts Chatterbox voice using a stored target voice id", async () => {
    const output = Uint8Array.from([9, 8, 7]).buffer
    mockBgUpload.mockResolvedValue(output)
    const sourceAudio = makeFile()

    const result = await convertChatterboxVoice({
      sourceAudio,
      targetVoiceId: "voice-1",
      responseFormat: "wav",
      stream: false
    })

    expect(new Uint8Array(result)).toEqual(Uint8Array.from([9, 8, 7]))
    expect(mockBgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/audio/voice-conversion",
        method: "POST",
        responseType: "arrayBuffer",
        fields: {
          target_voice_id: "voice-1",
          response_format: "wav",
          stream: false
        },
        files: [
          expect.objectContaining({
            fieldName: "source_audio",
            name: "source.wav",
            type: "audio/wav"
          })
        ]
      })
    )
    expect(sourceAudio.arrayBuffer).toHaveBeenCalledTimes(1)
  })

  it("can omit targetVoiceId to use the backend default target reference", async () => {
    mockBgUpload.mockResolvedValue(Uint8Array.from([1, 2]).buffer)
    const sourceAudio = makeFile("speech.webm", "audio/webm", [4, 5, 6])

    await convertChatterboxVoice({
      sourceAudio,
      responseFormat: "mp3",
      stream: true
    })

    const call = mockBgUpload.mock.calls[0][0]
    expect(call.files).toHaveLength(1)
    expect(call.files[0].fieldName).toBe("source_audio")
    expect(call.files[0].name).toBe("speech.webm")
    expect(call.files[0].type).toBe("audio/webm")
    expect(call.fields).toEqual({
      response_format: "mp3",
      stream: true
    })
  })

  it("can upload a direct target voice reference file", async () => {
    mockBgUpload.mockResolvedValue(Uint8Array.from([3, 4]).buffer)
    const sourceAudio = makeFile("speech.wav", "audio/wav", [1, 2])
    const targetVoice = makeFile("target.wav", "audio/wav", [9, 9])

    await convertChatterboxVoice({
      sourceAudio,
      targetVoice,
      responseFormat: "wav"
    })

    const call = mockBgUpload.mock.calls[0][0]
    expect(call.file).toBeUndefined()
    expect(call.fileFieldName).toBeUndefined()
    expect(call.files).toEqual([
      expect.objectContaining({
        fieldName: "source_audio",
        name: "speech.wav",
        type: "audio/wav"
      }),
      expect.objectContaining({
        fieldName: "target_voice",
        name: "target.wav",
        type: "audio/wav"
      })
    ])
    expect(call.fields).toEqual({
      response_format: "wav",
      stream: false
    })
  })

  it("rejects target voice file and stored target voice id together", async () => {
    await expect(
      convertChatterboxVoice({
        sourceAudio: makeFile("speech.wav"),
        targetVoice: makeFile("target.wav"),
        targetVoiceId: "voice-1"
      })
    ).rejects.toThrow("Provide either targetVoice or targetVoiceId")

    expect(mockBgUpload).not.toHaveBeenCalled()
  })

  it("rejects unsupported response formats before upload", async () => {
    await expect(
      convertChatterboxVoice({
        sourceAudio: makeFile(),
        responseFormat: "aiff" as any
      })
    ).rejects.toThrow("Unsupported Chatterbox voice conversion response format")

    expect(mockBgUpload).not.toHaveBeenCalled()
  })

  it("unloads a TTS provider through the background request client", async () => {
    mockBgRequestClient.mockResolvedValue({ provider: "chatterbox", unloaded: true })

    const result = await unloadTtsProvider(" chatterbox ")

    expect(result).toEqual({ provider: "chatterbox", unloaded: true })
    expect(mockBgRequestClient).toHaveBeenCalledWith({
      path: "/api/v1/audio/tts/providers/chatterbox/unload",
      method: "POST"
    })
  })

  it("rejects empty provider ids before unloading a TTS provider", async () => {
    await expect(unloadTtsProvider("   ")).rejects.toThrow("provider is required")

    expect(mockBgRequestClient).not.toHaveBeenCalled()
  })
})
