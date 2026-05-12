import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { WakeDetector, WakeDetectorConfig } from "../personaWakeDetector"
import { usePersonaLiveVoiceController } from "../usePersonaLiveVoiceController"

const hookMocks = vi.hoisted(() => ({
  audioStart: vi.fn(),
  audioAppend: vi.fn(),
  audioFinish: vi.fn(),
  audioStop: vi.fn(),
  audioState: { playing: false },
  micStart: vi.fn(() => {
    hookMocks.micActive = true
    return Promise.resolve()
  }),
  micStop: vi.fn(() => {
    hookMocks.micActive = false
  }),
  micActive: false,
  micChunkHandler: null as ((chunk: ArrayBuffer) => void) | null
}))

const audioCatalogState = vi.hoisted(() => ({
  devices: [] as Array<{ deviceId: string; label: string }>,
  isSettled: true
}))

const { storageValues, useStorageMock } = vi.hoisted(() => ({
  storageValues: new Map<string, unknown>(),
  useStorageMock: vi.fn()
}))

vi.mock("@/hooks/useStreamingAudioPlayer", () => ({
  useStreamingAudioPlayer: () => ({
    start: hookMocks.audioStart,
    append: hookMocks.audioAppend,
    finish: hookMocks.audioFinish,
    stop: hookMocks.audioStop,
    state: hookMocks.audioState
  })
}))

vi.mock("@/hooks/useMicStream", () => ({
  useMicStream: (onChunk: (chunk: ArrayBuffer) => void) => {
    hookMocks.micChunkHandler = onChunk
    return {
      start: hookMocks.micStart,
      stop: hookMocks.micStop,
      active: hookMocks.micActive
    }
  }
}))

vi.mock("@/hooks/useAudioSourceCatalog", () => ({
  useAudioSourceCatalog: () => audioCatalogState
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: useStorageMock
}))

describe("usePersonaLiveVoiceController", () => {
  beforeEach(() => {
    hookMocks.audioStart.mockReset()
    hookMocks.audioAppend.mockReset()
    hookMocks.audioFinish.mockReset()
    hookMocks.audioStop.mockReset()
    hookMocks.audioState.playing = false
    hookMocks.micActive = false
    hookMocks.micChunkHandler = null
    hookMocks.micStart.mockReset()
    hookMocks.micStart.mockImplementation(() => {
      hookMocks.micActive = true
      return Promise.resolve()
    })
    hookMocks.micStop.mockReset()
    hookMocks.micStop.mockImplementation(() => {
      hookMocks.micActive = false
    })
    audioCatalogState.devices = [
      { deviceId: "default", label: "Default microphone" },
      { deviceId: "usb-1", label: "USB microphone" }
    ]
    audioCatalogState.isSettled = true
    storageValues.clear()
    useStorageMock.mockReset()
    useStorageMock.mockImplementation((key: string, defaultValue: unknown) => [
      storageValues.has(key) ? storageValues.get(key) : defaultValue,
      vi.fn(),
      { isLoading: false }
    ])
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  const resolvedDefaults = {
    sttLanguage: "en-US",
    sttModel: "whisper-1",
    ttsProvider: "openai",
    ttsVoice: "alloy",
    confirmationMode: "destructive_only" as const,
    wakeBehavior: "one_shot" as const,
    voiceChatTriggerPhrases: ["hey helper"],
    autoResume: true,
    bargeIn: false,
    autoCommitEnabled: true,
    vadThreshold: 0.5,
    minSilenceMs: 250,
    turnStopSecs: 0.2,
    minUtteranceSecs: 0.4
  }

  const getSentPayloads = (ws: WebSocket & { send: ReturnType<typeof vi.fn> }) =>
    ws.send.mock.calls.map(([payload]) => JSON.parse(String(payload)))

  const createWakeDetectorHarness = () => {
    let config: WakeDetectorConfig | null = null
    const detector: WakeDetector = {
      isAvailable: vi.fn(async () => true),
      start: vi.fn(async (nextConfig) => {
        config = nextConfig
        nextConfig.onStateChange?.("listening")
      }),
      stop: vi.fn(async () => {
        config = null
      })
    }
    return {
      detector,
      fireWake: (canonicalPhrase = "hey helper") => {
        config?.onWake({
          canonicalPhrase,
          transcript: `${canonicalPhrase} status`,
          detectedAtMs: 1714500000000,
          detectorKind: "browser_transcript"
        })
      }
    }
  }

  it("sends persona-scoped voice_config when the live websocket is connected", async () => {
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    await waitFor(() => {
      expect(ws.send).toHaveBeenCalledWith(
        JSON.stringify({
          type: "voice_config",
          session_id: "sess-voice",
          voice: {
            trigger_phrases: ["hey helper"],
            auto_resume: true,
            barge_in: false,
            wake_behavior: "one_shot"
          },
          stt: {
            language: "en-US",
            model: "whisper-1",
            enable_vad: true,
            vad_threshold: 0.5,
            min_silence_ms: 250,
            turn_stop_secs: 0.2,
            min_utterance_secs: 0.4
          },
          tts: {
            provider: "openai",
            voice: "alloy"
          }
        })
      )
    })
  })

  it("initializes session turn detection to the balanced preset", () => {
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    const controller = result.current as any
    expect(controller.vadPreset).toBe("balanced")
    expect(controller.autoCommitEnabled).toBe(true)
    expect(controller.vadThreshold).toBe(0.5)
    expect(controller.minSilenceMs).toBe(250)
    expect(controller.turnStopSecs).toBe(0.2)
    expect(controller.minUtteranceSecs).toBe(0.4)
  })

  it("blocks wake arming when the selected profile has no saved trigger phrases", async () => {
    const wakeHarness = createWakeDetectorHarness()
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-1",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true,
        wakeTriggerPhrases: [],
        wakeDetectorFactory: () => wakeHarness.detector
      })
    )

    await act(async () => {
      await result.current.toggleWakeArmed()
    })

    expect(wakeHarness.detector.start).not.toHaveBeenCalled()
    expect(result.current.wakeWarning).toMatch(/trigger phrase/i)
    expect(result.current.wakeWarningReasonCode).toBe("wake_not_configured")
  })

  it("marks unavailable wake detector state with a stable recovery reason", async () => {
    const detector: WakeDetector = {
      isAvailable: vi.fn(async () => false),
      start: vi.fn(async () => undefined),
      stop: vi.fn(async () => undefined)
    }
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-1",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true,
        wakeTriggerPhrases: ["hey helper"],
        wakeDetectorFactory: () => detector
      })
    )

    await act(async () => {
      await result.current.toggleWakeArmed()
    })

    expect(detector.start).not.toHaveBeenCalled()
    expect(result.current.wakeArmed).toBe(false)
    expect(result.current.wakeDetectorState).toBe("unavailable")
    expect(result.current.wakeWarning).toMatch(/unavailable/i)
    expect(result.current.wakeWarningReasonCode).toBe("wake_detector_unavailable")
  })

  it("maps wake detector permission errors to a recovery reason", async () => {
    const detector: WakeDetector = {
      isAvailable: vi.fn(async () => true),
      start: vi.fn(async (nextConfig) => {
        nextConfig.onStateChange?.("error")
        nextConfig.onError?.({
          code: "not-allowed",
          message: "Microphone permission is blocked."
        })
      }),
      stop: vi.fn(async () => undefined)
    }
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-1",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true,
        wakeTriggerPhrases: ["hey helper"],
        wakeDetectorFactory: () => detector
      })
    )

    await act(async () => {
      await result.current.toggleWakeArmed()
    })

    expect(result.current.wakeDetectorState).toBe("error")
    expect(result.current.wakeWarning).toMatch(/permission/i)
    expect(result.current.wakeWarningReasonCode).toBe(
      "wake_detector_permission_denied"
    )
  })

  it("starts the wake detector with raw saved phrases", async () => {
    const wakeHarness = createWakeDetectorHarness()
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-1",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true,
        wakeTriggerPhrases: ["hey helper"],
        wakeDetectorFactory: () => wakeHarness.detector
      })
    )

    await act(async () => {
      await result.current.toggleWakeArmed()
    })

    expect(wakeHarness.detector.start).toHaveBeenCalledWith(
      expect.objectContaining({
        phrases: ["hey helper"]
      })
    )
    expect(result.current.wakeArmed).toBe(true)
    expect(result.current.wakeDetectorState).toBe("listening")
  })

  it("sends wake activation when the detector hears a configured phrase", async () => {
    const wakeHarness = createWakeDetectorHarness()
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-1",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true,
        wakeTriggerPhrases: ["hey helper"],
        wakeDetectorFactory: () => wakeHarness.detector
      })
    )

    await act(async () => {
      await result.current.toggleWakeArmed()
    })
    ;(ws as WebSocket & { send: ReturnType<typeof vi.fn> }).send.mockClear()

    act(() => {
      wakeHarness.fireWake()
    })

    expect(getSentPayloads(ws as WebSocket & { send: ReturnType<typeof vi.fn> })).toEqual(
      expect.arrayContaining([
        {
          type: "wake_activation",
          session_id: "sess-1",
          matched_phrase: "hey helper",
          detector_kind: "browser_transcript",
          detected_at_ms: 1714500000000
        }
      ])
    )
  })

  it("keeps wake armed when activation send fails", async () => {
    const wakeHarness = createWakeDetectorHarness()
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn((payload: string) => {
        const parsed = JSON.parse(String(payload))
        if (parsed.type === "wake_activation") {
          throw new Error("socket send failed")
        }
      })
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-1",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true,
        wakeTriggerPhrases: ["hey helper"],
        wakeDetectorFactory: () => wakeHarness.detector
      })
    )

    await act(async () => {
      await result.current.toggleWakeArmed()
    })
    vi.mocked(wakeHarness.detector.stop).mockClear()

    act(() => {
      wakeHarness.fireWake()
    })

    expect(result.current.wakeArmed).toBe(true)
    expect(result.current.wakeWarning).toMatch(/activation could not be sent/i)
    expect(result.current.wakeWarningReasonCode).toBe("wake_activation_send_failed")
    expect(wakeHarness.detector.stop).not.toHaveBeenCalled()
    expect(hookMocks.micStart).not.toHaveBeenCalled()
  })

  it("surfaces a profile-specific wake rejection reason from the server", async () => {
    const wakeHarness = createWakeDetectorHarness()
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-1",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true,
        wakeTriggerPhrases: ["hey helper"],
        wakeDetectorFactory: () => wakeHarness.detector
      })
    )

    await act(async () => {
      await result.current.toggleWakeArmed()
    })

    act(() => {
      wakeHarness.fireWake("runtime only")
    })

    act(() => {
      result.current.handlePayload({
        event: "notice",
        reason_code: "WAKE_ACTIVATION_REJECTED",
        wake_rejection_reason: "not_saved_in_profile",
        message: "Wake activation was rejected for this persona session."
      })
    })

    await waitFor(() => {
      expect(result.current.wakeWarning).toMatch(/saved trigger phrase/i)
    })
    expect(result.current.wakeWarningReasonCode).toBe(
      "wake_activation_rejected_not_saved_in_profile"
    )
    expect(result.current.wakeArmed).toBe(true)
  })

  it("surfaces a runtime wake configuration rejection reason from the server", async () => {
    const wakeHarness = createWakeDetectorHarness()
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-1",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true,
        wakeTriggerPhrases: ["hey helper"],
        wakeDetectorFactory: () => wakeHarness.detector
      })
    )

    await act(async () => {
      await result.current.toggleWakeArmed()
    })

    act(() => {
      wakeHarness.fireWake("hey helper")
    })

    act(() => {
      result.current.handlePayload({
        event: "notice",
        reason_code: "WAKE_ACTIVATION_REJECTED",
        wake_rejection_reason: "missing_from_runtime_config",
        message: "Wake activation was rejected for this persona session."
      })
    })

    await waitFor(() => {
      expect(result.current.wakeWarning).toMatch(/live voice configuration/i)
    })
    expect(result.current.wakeWarningReasonCode).toBe(
      "wake_activation_rejected_missing_from_runtime_config"
    )
    expect(result.current.wakeArmed).toBe(true)
  })

  it("sends wake deactivation when wake listening is disarmed", async () => {
    const wakeHarness = createWakeDetectorHarness()
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-1",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true,
        wakeTriggerPhrases: ["hey helper"],
        wakeDetectorFactory: () => wakeHarness.detector
      })
    )

    await act(async () => {
      await result.current.toggleWakeArmed()
    })
    ;(ws as WebSocket & { send: ReturnType<typeof vi.fn> }).send.mockClear()

    await act(async () => {
      await result.current.toggleWakeArmed()
    })

    expect(getSentPayloads(ws as WebSocket & { send: ReturnType<typeof vi.fn> })).toEqual([
      {
        type: "wake_deactivation",
        session_id: "sess-1",
        reason: "disarmed"
      }
    ])
    expect(wakeHarness.detector.stop).toHaveBeenCalled()
    expect(result.current.wakeArmed).toBe(false)
  })

  it("cleans up wake state when detector stop rejects during disarm", async () => {
    const stopError = new Error("wake stop failed")
    const detector: WakeDetector = {
      isAvailable: vi.fn(async () => true),
      start: vi.fn(async (nextConfig) => {
        nextConfig.onStateChange?.("listening")
      }),
      stop: vi.fn(async () => {
        throw stopError
      })
    }
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-1",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true,
        wakeTriggerPhrases: ["hey helper"],
        wakeDetectorFactory: () => detector
      })
    )

    await act(async () => {
      await result.current.toggleWakeArmed()
    })
    ;(ws as WebSocket & { send: ReturnType<typeof vi.fn> }).send.mockClear()

    let disarmResult: unknown
    await act(async () => {
      disarmResult = await result.current.toggleWakeArmed()
    })

    expect(disarmResult).toBeUndefined()
    await waitFor(() => {
      expect(result.current.wakeArmed).toBe(false)
    })
    expect(result.current.wakeDetectorState).toBe("idle")
    expect(getSentPayloads(ws as WebSocket & { send: ReturnType<typeof vi.fn> })).toEqual([
      {
        type: "wake_deactivation",
        session_id: "sess-1",
        reason: "disarmed"
      }
    ])
  })

  it("sends wake deactivation on unmount when wake listening is armed", async () => {
    const wakeHarness = createWakeDetectorHarness()
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result, unmount } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-1",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true,
        wakeTriggerPhrases: ["hey helper"],
        wakeDetectorFactory: () => wakeHarness.detector
      })
    )

    await act(async () => {
      await result.current.toggleWakeArmed()
    })
    ;(ws as WebSocket & { send: ReturnType<typeof vi.fn> }).send.mockClear()

    unmount()

    expect(getSentPayloads(ws as WebSocket & { send: ReturnType<typeof vi.fn> })).toEqual([
      {
        type: "wake_deactivation",
        session_id: "sess-1",
        reason: "route_leave"
      }
    ])
  })

  it("sends wake deactivation for the previous session on persona switch", async () => {
    const wakeHarness = createWakeDetectorHarness()
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result, rerender } = renderHook(
      ({
        sessionId,
        personaId
      }: {
        sessionId: string
        personaId: string
      }) =>
        usePersonaLiveVoiceController({
          ws,
          connected: true,
          sessionId,
          personaId,
          resolvedDefaults,
          canUseServerStt: true,
          wakeTriggerPhrases: ["hey helper"],
          wakeDetectorFactory: () => wakeHarness.detector
        }),
      {
        initialProps: {
          sessionId: "sess-old",
          personaId: "persona-1"
        }
      }
    )

    await act(async () => {
      await result.current.toggleWakeArmed()
    })
    ;(ws as WebSocket & { send: ReturnType<typeof vi.fn> }).send.mockClear()

    rerender({
      sessionId: "sess-new",
      personaId: "persona-2"
    })

    await waitFor(() => {
      expect(getSentPayloads(ws as WebSocket & { send: ReturnType<typeof vi.fn> })).toEqual(
        expect.arrayContaining([
          {
            type: "wake_deactivation",
            session_id: "sess-old",
            reason: "persona_switch"
          }
        ])
      )
    })
  })

  it("stops armed wake listening when Persona Live disconnects", async () => {
    const wakeHarness = createWakeDetectorHarness()
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result, rerender } = renderHook(
      ({ connected }: { connected: boolean }) =>
        usePersonaLiveVoiceController({
          ws,
          connected,
          sessionId: "sess-1",
          personaId: "persona-1",
          resolvedDefaults,
          canUseServerStt: true,
          wakeTriggerPhrases: ["hey helper"],
          wakeDetectorFactory: () => wakeHarness.detector
        }),
      {
        initialProps: { connected: true }
      }
    )

    await act(async () => {
      await result.current.toggleWakeArmed()
    })
    ;(ws as WebSocket & { send: ReturnType<typeof vi.fn> }).send.mockClear()
    vi.mocked(wakeHarness.detector.stop).mockClear()

    rerender({ connected: false })

    await waitFor(() => {
      expect(result.current.wakeArmed).toBe(false)
    })
    expect(wakeHarness.detector.stop).toHaveBeenCalledTimes(1)
    expect(getSentPayloads(ws as WebSocket & { send: ReturnType<typeof vi.fn> })).toEqual(
      expect.arrayContaining([
        {
          type: "wake_deactivation",
          session_id: "sess-1",
          reason: "session_close"
        }
      ])
    )
  })

  it("cancels stale wake starts when disarmed before availability resolves", async () => {
    let resolveAvailable: ((available: boolean) => void) | null = null
    const detector: WakeDetector = {
      isAvailable: vi.fn(
        () =>
          new Promise<boolean>((resolve) => {
            resolveAvailable = resolve
          })
      ),
      start: vi.fn(async () => undefined),
      stop: vi.fn(async () => undefined)
    }
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-1",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true,
        wakeTriggerPhrases: ["hey helper"],
        wakeDetectorFactory: () => detector
      })
    )

    let armPromise: Promise<void> = Promise.resolve()
    act(() => {
      armPromise = result.current.toggleWakeArmed()
    })

    await waitFor(() => {
      expect(result.current.wakeArmed).toBe(true)
    })

    await act(async () => {
      await result.current.toggleWakeArmed()
    })

    await act(async () => {
      resolveAvailable?.(true)
      await armPromise
    })

    expect(detector.start).not.toHaveBeenCalled()
    expect(result.current.wakeArmed).toBe(false)
  })

  it("stops the wake detector before live mic capture starts", async () => {
    const wakeHarness = createWakeDetectorHarness()
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-1",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true,
        wakeTriggerPhrases: ["hey helper"],
        wakeDetectorFactory: () => wakeHarness.detector
      })
    )

    await act(async () => {
      await result.current.toggleWakeArmed()
    })
    const stopCallsBeforeMic = vi.mocked(wakeHarness.detector.stop).mock.calls.length

    await act(async () => {
      await result.current.startListening()
    })

    expect(vi.mocked(wakeHarness.detector.stop).mock.calls.length).toBe(
      stopCallsBeforeMic + 1
    )
    expect(hookMocks.micStart).toHaveBeenCalled()
    expect(result.current.wakeArmed).toBe(true)
  })

  it("restarts wake listening after manually stopped mic capture while armed", async () => {
    const wakeHarness = createWakeDetectorHarness()
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-1",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true,
        wakeTriggerPhrases: ["hey helper"],
        wakeDetectorFactory: () => wakeHarness.detector
      })
    )

    await act(async () => {
      await result.current.toggleWakeArmed()
    })
    expect(wakeHarness.detector.start).toHaveBeenCalledTimes(1)

    await act(async () => {
      await result.current.startListening()
    })
    await waitFor(() => {
      expect(result.current.isListening).toBe(true)
    })

    act(() => {
      result.current.stopListening()
    })

    await waitFor(() => {
      expect(wakeHarness.detector.start).toHaveBeenCalledTimes(2)
    })
    expect(result.current.wakeArmed).toBe(true)
  })

  it("restarts wake listening after push-to-talk wake turn finishes", async () => {
    const wakeHarness = createWakeDetectorHarness()
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket
    const pushToTalkDefaults = {
      ...resolvedDefaults,
      autoResume: false,
      wakeBehavior: "push_to_talk_after_wake" as const
    }

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-1",
        personaId: "persona-1",
        resolvedDefaults: pushToTalkDefaults,
        canUseServerStt: true,
        wakeTriggerPhrases: ["hey helper"],
        wakeDetectorFactory: () => wakeHarness.detector
      })
    )

    await act(async () => {
      await result.current.toggleWakeArmed()
    })
    expect(wakeHarness.detector.start).toHaveBeenCalledTimes(1)

    act(() => {
      wakeHarness.fireWake()
    })

    act(() => {
      result.current.handlePayload({
        event: "notice",
        reason_code: "TTS_UNAVAILABLE_TEXT_ONLY",
        message: "text only"
      })
    })

    await waitFor(() => {
      expect(wakeHarness.detector.start).toHaveBeenCalledTimes(2)
    })
    expect(result.current.wakeArmed).toBe(true)
  })

  it("marks the preset as custom after an advanced runtime edit", () => {
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    act(() => {
      ;(result.current as any).setVadThreshold(0.61)
    })

    expect((result.current as any).vadPreset).toBe("custom")
    expect((result.current as any).vadThreshold).toBe(0.61)
  })

  it("resets session turn detection tuning on persona switch", () => {
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result, rerender } = renderHook(
      ({
        sessionId,
        personaId
      }: {
        sessionId: string
        personaId: string
      }) =>
        usePersonaLiveVoiceController({
          ws,
          connected: true,
          sessionId,
          personaId,
          resolvedDefaults,
          canUseServerStt: true
        }),
      {
        initialProps: {
          sessionId: "sess-voice",
          personaId: "persona-1"
        }
      }
    )

    act(() => {
      ;(result.current as any).setVadPreset("fast")
    })

    expect((result.current as any).vadPreset).toBe("fast")
    expect((result.current as any).minSilenceMs).toBe(150)

    rerender({
      sessionId: "sess-voice-2",
      personaId: "persona-2"
    })

    expect((result.current as any).vadPreset).toBe("balanced")
    expect((result.current as any).autoCommitEnabled).toBe(true)
    expect((result.current as any).vadThreshold).toBe(0.5)
    expect((result.current as any).minSilenceMs).toBe(250)
    expect((result.current as any).turnStopSecs).toBe(0.2)
    expect((result.current as any).minUtteranceSecs).toBe(0.4)
  })

  it("sends updated voice_config when the preset changes while connected", async () => {
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    await waitFor(() => {
      expect(ws.send).toHaveBeenCalled()
    })
    ;(ws as WebSocket & { send: ReturnType<typeof vi.fn> }).send.mockClear()

    act(() => {
      ;(result.current as any).setVadPreset("fast")
    })

    await waitFor(() => {
      expect(getSentPayloads(ws as WebSocket & { send: ReturnType<typeof vi.fn> })).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            type: "voice_config",
            session_id: "sess-voice",
            stt: expect.objectContaining({
              enable_vad: true,
              vad_threshold: 0.35,
              min_silence_ms: 150,
              turn_stop_secs: 0.1,
              min_utterance_secs: 0.25
            })
          })
        ])
      )
    })
  })

  it("sends enable_vad false when auto-commit is turned off", async () => {
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    await waitFor(() => {
      expect(ws.send).toHaveBeenCalled()
    })
    ;(ws as WebSocket & { send: ReturnType<typeof vi.fn> }).send.mockClear()

    act(() => {
      ;(result.current as any).setAutoCommitEnabled(false)
    })

    await waitFor(() => {
      expect(getSentPayloads(ws as WebSocket & { send: ReturnType<typeof vi.fn> })).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            type: "voice_config",
            session_id: "sess-voice",
            stt: expect.objectContaining({
              enable_vad: false
            })
          })
        ])
      )
    })
  })

  it("sends updated advanced values when turn detection tuning changes", async () => {
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    await waitFor(() => {
      expect(ws.send).toHaveBeenCalled()
    })
    ;(ws as WebSocket & { send: ReturnType<typeof vi.fn> }).send.mockClear()

    act(() => {
      ;(result.current as any).setVadThreshold(0.61)
      ;(result.current as any).setMinSilenceMs(640)
      ;(result.current as any).setTurnStopSecs(0.48)
      ;(result.current as any).setMinUtteranceSecs(0.82)
    })

    await waitFor(() => {
      expect(getSentPayloads(ws as WebSocket & { send: ReturnType<typeof vi.fn> })).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            type: "voice_config",
            session_id: "sess-voice",
            stt: expect.objectContaining({
              enable_vad: true,
              vad_threshold: 0.61,
              min_silence_ms: 640,
              turn_stop_secs: 0.48,
              min_utterance_secs: 0.82
            })
          })
        ])
      )
    })
  })

  it("streams persona audio chunks and does not send a routine manual commit when listening stops", async () => {
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result, rerender } = renderHook(
      ({
        sessionId,
        personaId
      }: {
        sessionId: string
        personaId: string
      }) =>
        usePersonaLiveVoiceController({
          ws,
          connected: true,
          sessionId,
          personaId,
          resolvedDefaults,
          canUseServerStt: true
        }),
      {
        initialProps: {
          sessionId: "sess-voice",
          personaId: "persona-1"
        }
      }
    )

    await act(async () => {
      await result.current.startListening()
    })
    rerender({
      sessionId: "sess-voice",
      personaId: "persona-1"
    })

    expect(hookMocks.micStart).toHaveBeenCalledTimes(1)
    expect(result.current.isListening).toBe(true)
    expect(hookMocks.micChunkHandler).toBeTypeOf("function")

    act(() => {
      hookMocks.micChunkHandler?.(new Uint8Array([1, 2, 3, 4]).buffer)
    })

    expect(getSentPayloads(ws as WebSocket & { send: ReturnType<typeof vi.fn> })).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "audio_chunk",
          session_id: "sess-voice",
          audio_format: "pcm16",
          bytes_base64: expect.any(String)
        })
      ])
    )

    act(() => {
      result.current.handlePayload({
        event: "partial_transcript",
        text_delta: "hey helper search my notes"
      })
    })

    expect(result.current.heardText).toBe("hey helper search my notes")

    const stopCallsBeforeStop = hookMocks.micStop.mock.calls.length

    act(() => {
      result.current.stopListening()
    })
    rerender({
      sessionId: "sess-voice",
      personaId: "persona-1"
    })

    expect(hookMocks.micStop.mock.calls.length).toBe(stopCallsBeforeStop + 1)
    expect(
      getSentPayloads(ws as WebSocket & { send: ReturnType<typeof vi.fn> }).filter(
        (payload) => payload.type === "voice_commit"
      )
    ).toEqual([])
  })

  it("starts persona live voice with the remembered live_voice mic device", async () => {
    storageValues.set("liveVoiceAudioSourcePreference", {
      featureGroup: "live_voice",
      sourceKind: "mic_device",
      deviceId: "usb-1",
      lastKnownLabel: "USB microphone"
    })

    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    await act(async () => {
      await result.current.startListening()
    })

    expect(hookMocks.micStart).toHaveBeenCalledWith({ deviceId: "usb-1" })
  })

  it("waits for the live_voice preference to hydrate before starting listening", async () => {
    storageValues.set("liveVoiceAudioSourcePreference", {
      featureGroup: "live_voice",
      sourceKind: "mic_device",
      deviceId: "usb-1",
      lastKnownLabel: "USB microphone"
    })

    let preferenceLoading = true
    useStorageMock.mockImplementation((key: string, defaultValue: unknown) => [
      storageValues.has(key) ? storageValues.get(key) : defaultValue,
      vi.fn(),
      { isLoading: key === "liveVoiceAudioSourcePreference" ? preferenceLoading : false }
    ])

    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result, rerender } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    await act(async () => {
      await result.current.startListening()
    })

    expect(result.current.isListening).toBe(true)
    expect(hookMocks.micStart).not.toHaveBeenCalled()

    preferenceLoading = false

    await act(async () => {
      rerender({
        sessionId: "sess-voice",
        personaId: "persona-1"
      })
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(hookMocks.micStart).toHaveBeenCalledWith({ deviceId: "usb-1" })
  })

  it("cancels a queued live_voice start before hydration finishes", async () => {
    storageValues.set("liveVoiceAudioSourcePreference", {
      featureGroup: "live_voice",
      sourceKind: "mic_device",
      deviceId: "usb-1",
      lastKnownLabel: "USB microphone"
    })

    let preferenceLoading = true
    useStorageMock.mockImplementation((key: string, defaultValue: unknown) => [
      storageValues.has(key) ? storageValues.get(key) : defaultValue,
      vi.fn(),
      { isLoading: key === "liveVoiceAudioSourcePreference" ? preferenceLoading : false }
    ])

    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result, rerender } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    await act(async () => {
      await result.current.startListening()
    })

    expect(result.current.isListening).toBe(true)
    expect(hookMocks.micStart).not.toHaveBeenCalled()

    act(() => {
      result.current.toggleListening()
    })

    expect(result.current.isListening).toBe(false)

    await act(async () => {
      preferenceLoading = false
      rerender({
        sessionId: "sess-voice",
        personaId: "persona-1"
      })
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(hookMocks.micStart).not.toHaveBeenCalled()
    expect(result.current.isListening).toBe(false)
  })

  it("falls back to the default microphone when the remembered live_voice device is missing", async () => {
    storageValues.set("liveVoiceAudioSourcePreference", {
      featureGroup: "live_voice",
      sourceKind: "mic_device",
      deviceId: "usb-missing",
      lastKnownLabel: "Studio microphone"
    })
    audioCatalogState.devices = [{ deviceId: "default", label: "Default microphone" }]

    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    await act(async () => {
      await result.current.startListening()
    })

    expect(hookMocks.micStart).toHaveBeenCalledWith({ deviceId: null })
  })

  it("shows listening recovery after 4 seconds with transcript but no commit", async () => {
    vi.useFakeTimers()

    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result, rerender } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    await act(async () => {
      await result.current.startListening()
    })
    rerender({
      sessionId: "sess-voice",
      personaId: "persona-1"
    })

    act(() => {
      result.current.handlePayload({
        event: "partial_transcript",
        text_delta: "hey helper"
      })
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(4000)
    })

    expect((result.current as any).recoveryMode).toBe("listening_stuck")
  })

  it("increments listeningRecoveryCount when listening recovery triggers", async () => {
    vi.useFakeTimers()

    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result, rerender } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    await act(async () => {
      await result.current.startListening()
    })
    rerender({
      sessionId: "sess-voice",
      personaId: "persona-1"
    })

    act(() => {
      result.current.handlePayload({
        event: "partial_transcript",
        text_delta: "hey helper"
      })
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(4000)
    })

    expect((result.current as any).listeningRecoveryCount).toBe(1)
  })

  it("restarts listening recovery when new transcript deltas arrive", async () => {
    vi.useFakeTimers()

    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result, rerender } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    await act(async () => {
      await result.current.startListening()
    })
    rerender({
      sessionId: "sess-voice",
      personaId: "persona-1"
    })

    act(() => {
      result.current.handlePayload({
        event: "partial_transcript",
        text_delta: "hey helper"
      })
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3500)
    })

    act(() => {
      result.current.handlePayload({
        event: "partial_transcript",
        text_delta: "search my notes"
      })
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })

    expect((result.current as any).recoveryMode).toBe("none")

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000)
    })

    expect((result.current as any).recoveryMode).toBe("listening_stuck")
  })

  it("keep listening dismisses and restarts listening recovery", async () => {
    vi.useFakeTimers()

    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result, rerender } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    await act(async () => {
      await result.current.startListening()
    })
    rerender({
      sessionId: "sess-voice",
      personaId: "persona-1"
    })

    act(() => {
      result.current.handlePayload({
        event: "partial_transcript",
        text_delta: "hey helper search my notes"
      })
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(4000)
    })

    expect((result.current as any).recoveryMode).toBe("listening_stuck")

    act(() => {
      ;(result.current as any).keepListening()
    })

    expect((result.current as any).recoveryMode).toBe("none")

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3999)
    })

    expect((result.current as any).recoveryMode).toBe("none")

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1)
    })

    expect((result.current as any).recoveryMode).toBe("listening_stuck")
  })

  it("reset turn clears heard transcript and returns to idle", async () => {
    vi.useFakeTimers()

    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result, rerender } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    await act(async () => {
      await result.current.startListening()
    })
    rerender({
      sessionId: "sess-voice",
      personaId: "persona-1"
    })

    act(() => {
      result.current.handlePayload({
        event: "partial_transcript",
        text_delta: "hey helper search my notes"
      })
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(4000)
    })

    const stopCallsBeforeReset = hookMocks.micStop.mock.calls.length

    act(() => {
      ;(result.current as any).resetTurn()
    })
    rerender({
      sessionId: "sess-voice",
      personaId: "persona-1"
    })

    expect(hookMocks.micStop.mock.calls.length).toBe(stopCallsBeforeReset + 1)
    expect(result.current.heardText).toBe("")
    expect(result.current.state).toBe("idle")
    expect((result.current as any).recoveryMode).toBe("none")
  })

  it("switches to thinking and stops the mic when the server auto-commits a voice turn", async () => {
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result, rerender } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    await act(async () => {
      await result.current.startListening()
    })
    rerender({
      sessionId: "sess-voice",
      personaId: "persona-1"
    })

    act(() => {
      result.current.handlePayload({
        event: "partial_transcript",
        text_delta: "hey helper search my notes"
      })
    })

    const stopCallsBeforeCommit = hookMocks.micStop.mock.calls.length

    act(() => {
      result.current.handlePayload({
        event: "notice",
        reason_code: "VOICE_TURN_COMMITTED",
        transcript: "search my notes",
        commit_source: "vad_auto"
      })
    })
    rerender({
      sessionId: "sess-voice",
      personaId: "persona-1"
    })

    expect(hookMocks.micStop.mock.calls.length).toBe(stopCallsBeforeCommit + 1)
    expect(result.current.state).toBe("thinking")
    expect(result.current.lastCommittedText).toBe("search my notes")
    expect(result.current.heardText).toBe("hey helper search my notes")
  })

  it("shows thinking recovery after 8 seconds after commit with no assistant progress", async () => {
    vi.useFakeTimers()

    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    act(() => {
      result.current.handlePayload({
        event: "notice",
        reason_code: "VOICE_TURN_COMMITTED",
        transcript: "search my notes",
        commit_source: "vad_auto"
      })
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(8000)
    })

    expect((result.current as any).recoveryMode).toBe("thinking_stuck")
  })

  it("increments thinkingRecoveryCount when thinking recovery triggers", async () => {
    vi.useFakeTimers()

    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    act(() => {
      result.current.handlePayload({
        event: "notice",
        reason_code: "VOICE_TURN_COMMITTED",
        transcript: "search my notes",
        commit_source: "vad_auto"
      })
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(8000)
    })

    expect((result.current as any).thinkingRecoveryCount).toBe(1)
  })

  it("assistant progress clears thinking recovery", async () => {
    vi.useFakeTimers()

    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    act(() => {
      result.current.handlePayload({
        event: "notice",
        reason_code: "VOICE_TURN_COMMITTED",
        transcript: "search my notes",
        commit_source: "vad_auto"
      })
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(8000)
    })

    expect((result.current as any).recoveryMode).toBe("thinking_stuck")

    act(() => {
      result.current.handlePayload({
        event: "assistant_delta",
        text_delta: "Working on it"
      })
    })

    expect((result.current as any).recoveryMode).toBe("none")
  })

  it("re-arms thinking recovery when VOICE_TURN_PROCESSING arrives", async () => {
    vi.useFakeTimers()

    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    act(() => {
      result.current.handlePayload({
        event: "notice",
        reason_code: "VOICE_TURN_COMMITTED",
        transcript: "search my notes",
        commit_source: "vad_auto"
      })
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(7000)
    })

    expect((result.current as any).recoveryMode).toBe("none")

    act(() => {
      result.current.handlePayload({
        event: "notice",
        reason_code: "VOICE_TURN_PROCESSING",
        message: "Still processing this voice turn."
      })
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1500)
    })

    expect((result.current as any).recoveryMode).toBe("none")
  })

  it("still enters thinking_stuck after a renewed quiet window", async () => {
    vi.useFakeTimers()

    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    act(() => {
      result.current.handlePayload({
        event: "notice",
        reason_code: "VOICE_TURN_COMMITTED",
        transcript: "search my notes",
        commit_source: "vad_auto"
      })
    })

    act(() => {
      result.current.handlePayload({
        event: "notice",
        reason_code: "VOICE_TURN_PROCESSING",
        message: "Still processing this voice turn."
      })
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(8000)
    })

    expect((result.current as any).recoveryMode).toBe("thinking_stuck")
  })

  it("sets activeToolStatus and re-arms thinking recovery on tool_call", async () => {
    vi.useFakeTimers()

    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    act(() => {
      result.current.handlePayload({
        event: "notice",
        reason_code: "VOICE_TURN_COMMITTED",
        transcript: "search my notes",
        commit_source: "vad_auto"
      })
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(7000)
    })

    expect((result.current as any).recoveryMode).toBe("none")

    act(() => {
      result.current.handlePayload({
        event: "tool_call",
        tool: "search_notes",
        why: "Looking through your notes"
      })
    })

    expect((result.current as any).activeToolStatus).toBe(
      "Running search_notes: Looking through your notes"
    )
    expect((result.current as any).recoveryMode).toBe("none")

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1500)
    })

    expect((result.current as any).recoveryMode).toBe("none")
  })

  it("re-arms thinking recovery when VOICE_TOOL_EXECUTION_PROCESSING arrives", async () => {
    vi.useFakeTimers()

    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    await act(async () => {
      result.current.handlePayload({
        event: "notice",
        reason_code: "VOICE_TURN_COMMITTED",
        transcript: "search my notes",
        commit_source: "vad_auto"
      })
    })

    act(() => {
      result.current.handlePayload({
        event: "tool_call",
        tool: "search_notes",
        why: "Looking through your notes"
      })
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(7000)
    })

    expect((result.current as any).recoveryMode).toBe("none")

    act(() => {
      result.current.handlePayload({
        event: "notice",
        reason_code: "VOICE_TOOL_EXECUTION_PROCESSING",
        tool: "search_notes",
        step_idx: 0,
        why: "Looking through your notes"
      })
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1500)
    })

    expect((result.current as any).recoveryMode).toBe("none")
  })

  it("clears activeToolStatus and re-arms recovery on tool_result", async () => {
    vi.useFakeTimers()

    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    act(() => {
      result.current.handlePayload({
        event: "notice",
        reason_code: "VOICE_TURN_COMMITTED",
        transcript: "search my notes",
        commit_source: "vad_auto"
      })
    })

    act(() => {
      result.current.handlePayload({
        event: "tool_call",
        tool: "search_notes",
        why: "Looking through your notes"
      })
    })

    expect((result.current as any).activeToolStatus).toBeTruthy()

    act(() => {
      result.current.handlePayload({
        event: "tool_result",
        ok: true,
        tool: "search_notes",
        output: { ok: true }
      })
    })

    expect((result.current as any).activeToolStatus).toBe("")
    expect((result.current as any).recoveryMode).toBe("none")

    await act(async () => {
      await vi.advanceTimersByTimeAsync(8000)
    })

    expect((result.current as any).recoveryMode).toBe("thinking_stuck")
  })

  it("clears activeToolStatus and recovery on approval tool_result", async () => {
    vi.useFakeTimers()

    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    act(() => {
      result.current.handlePayload({
        event: "notice",
        reason_code: "VOICE_TURN_COMMITTED",
        transcript: "search my notes",
        commit_source: "vad_auto"
      })
    })

    act(() => {
      result.current.handlePayload({
        event: "tool_call",
        tool: "search_notes",
        why: "Looking through your notes"
      })
    })

    expect((result.current as any).activeToolStatus).toBeTruthy()

    act(() => {
      result.current.handlePayload({
        event: "tool_result",
        approval: {
          tool_name: "search_notes",
          context_key: "ctx",
          scope_key: "once",
          duration_options: ["once"]
        }
      })
    })

    expect((result.current as any).activeToolStatus).toBe("")
    expect((result.current as any).recoveryMode).toBe("none")
  })

  it("enters manual mode when the server cannot auto-commit and sends the current transcript on demand", async () => {
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result, rerender } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    await act(async () => {
      await result.current.startListening()
    })
    rerender({
      sessionId: "sess-voice",
      personaId: "persona-1"
    })

    act(() => {
      result.current.handlePayload({
        event: "partial_transcript",
        text_delta: "hey helper search my notes"
      })
    })
    act(() => {
      result.current.handlePayload({
        event: "notice",
        reason_code: "VOICE_MANUAL_MODE_REQUIRED",
        message:
          "Server VAD unavailable for this live session. Use Send now to commit heard speech manually."
      })
    })

    expect(result.current.manualModeRequired).toBe(true)
    expect(result.current.canSendNow).toBe(true)
    expect(result.current.warning).toContain("Use Send now")
    expect(result.current.warningReasonCode).toBe("voice_manual_mode_required")

    const stopCallsBeforeSend = hookMocks.micStop.mock.calls.length

    act(() => {
      result.current.sendCurrentTranscriptNow()
    })
    rerender({
      sessionId: "sess-voice",
      personaId: "persona-1"
    })

    expect(hookMocks.micStop.mock.calls.length).toBe(stopCallsBeforeSend + 1)
    expect(ws.send).toHaveBeenLastCalledWith(
      JSON.stringify({
        type: "voice_commit",
        session_id: "sess-voice",
        transcript: "hey helper search my notes",
        source: "persona_live_voice_manual"
      })
    )
    expect(result.current.state).toBe("thinking")
  })

  it("auto-resumes listening after a recoverable text-only tts warning", async () => {
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result, rerender } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    act(() => {
      result.current.handlePayload({
        event: "notice",
        reason_code: "TTS_UNAVAILABLE_TEXT_ONLY",
        message: "Live TTS unavailable for this session. Continuing in text-only mode."
      })
    })
    rerender({
      sessionId: "sess-voice",
      personaId: "persona-1"
    })

    await waitFor(() => {
      expect(hookMocks.micStart).toHaveBeenCalledTimes(1)
    })
    expect(result.current.textOnlyDueToTtsFailure).toBe(true)
    expect(result.current.warning).toContain("Continuing in text-only mode.")
    expect(result.current.warningReasonCode).toBe("voice_tts_unavailable_text_only")
  })

  it("cancels a queued auto-resume before hydration finishes", async () => {
    storageValues.set("liveVoiceAudioSourcePreference", {
      featureGroup: "live_voice",
      sourceKind: "mic_device",
      deviceId: "usb-1",
      lastKnownLabel: "USB microphone"
    })

    let preferenceLoading = true
    useStorageMock.mockImplementation((key: string, defaultValue: unknown) => [
      storageValues.has(key) ? storageValues.get(key) : defaultValue,
      vi.fn(),
      { isLoading: key === "liveVoiceAudioSourcePreference" ? preferenceLoading : false }
    ])

    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result, rerender } = renderHook(() =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId: "sess-voice",
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    act(() => {
      result.current.handlePayload({
        event: "notice",
        reason_code: "TTS_UNAVAILABLE_TEXT_ONLY",
        message: "Live TTS unavailable for this session. Continuing in text-only mode."
      })
    })

    expect(result.current.isListening).toBe(true)
    expect(hookMocks.micStart).not.toHaveBeenCalled()

    act(() => {
      result.current.toggleListening()
    })

    expect(result.current.isListening).toBe(false)

    await act(async () => {
      preferenceLoading = false
      rerender({
        sessionId: "sess-voice",
        personaId: "persona-1"
      })
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(hookMocks.micStart).not.toHaveBeenCalled()
    expect(result.current.isListening).toBe(false)
  })

  it("resets session-local overrides when the persona session changes", async () => {
    const ws = {
      readyState: WebSocket.OPEN,
      send: vi.fn()
    } as unknown as WebSocket

    const { result, rerender } = renderHook(
      ({
        personaId,
        sessionId
      }: {
        personaId: string
        sessionId: string
      }) =>
        usePersonaLiveVoiceController({
          ws,
          connected: true,
          sessionId,
          personaId,
          resolvedDefaults,
          canUseServerStt: true
        }),
      {
        initialProps: {
          personaId: "persona-1",
          sessionId: "sess-voice"
        }
      }
    )

    act(() => {
      result.current.setSessionAutoResume(false)
      result.current.setSessionBargeIn(true)
    })

    expect(result.current.sessionAutoResume).toBe(false)
    expect(result.current.sessionBargeIn).toBe(true)

    rerender({
      personaId: "persona-2",
      sessionId: "sess-voice-2"
    })

    await waitFor(() => {
      expect(result.current.sessionAutoResume).toBe(true)
      expect(result.current.sessionBargeIn).toBe(false)
    })
  })

  it("logs browser speech cancellation failures", () => {
    const speechSynthesisDescriptor = Object.getOwnPropertyDescriptor(
      window,
      "speechSynthesis"
    )
    const cancelError = new Error("cancel failed")
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {})

    Object.defineProperty(window, "speechSynthesis", {
      configurable: true,
      value: {
        cancel: () => {
          throw cancelError
        },
        getVoices: () => [],
        speak: vi.fn()
      }
    })

    renderHook(() =>
      usePersonaLiveVoiceController({
        ws: null,
        connected: false,
        sessionId: null,
        personaId: "persona-1",
        resolvedDefaults,
        canUseServerStt: true
      })
    )

    expect(errorSpy).toHaveBeenCalledWith(
      "stopBrowserSpeech: speechSynthesis.cancel failed",
      cancelError
    )

    errorSpy.mockRestore()
    if (speechSynthesisDescriptor) {
      Object.defineProperty(window, "speechSynthesis", speechSynthesisDescriptor)
      return
    }
    delete (window as { speechSynthesis?: unknown }).speechSynthesis
  })
})
