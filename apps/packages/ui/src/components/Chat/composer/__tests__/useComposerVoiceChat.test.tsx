import React from "react"
import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

// ---------------------------------------------------------------------------
// Mocks for the underlying primitives. The shared hook is pure orchestration —
// real STT requires browser APIs (MediaRecorder, SpeechRecognition) — so we
// replace each primitive with a controllable fake and assert the wiring.
// ---------------------------------------------------------------------------

const speechRecognitionState = {
  transcript: "",
  isListening: false,
  resetTranscript: vi.fn(),
  start: vi.fn(),
  stop: vi.fn(),
  supported: true,
  capturedOptions: undefined as
    | { autoStop?: boolean; autoStopTimeout?: number; onEnd?: () => void | Promise<void> }
    | undefined
}

vi.mock("@/hooks/useSpeechRecognition", () => ({
  useSpeechRecognition: (
    options?: { autoStop?: boolean; autoStopTimeout?: number; onEnd?: () => void | Promise<void> }
  ) => {
    speechRecognitionState.capturedOptions = options
    return {
      transcript: speechRecognitionState.transcript,
      isListening: speechRecognitionState.isListening,
      resetTranscript: speechRecognitionState.resetTranscript,
      start: speechRecognitionState.start,
      stop: speechRecognitionState.stop,
      supported: speechRecognitionState.supported
    }
  }
}))

const serverDictationState = {
  isServerDictating: false,
  startServerDictation: vi.fn(),
  stopServerDictation: vi.fn(),
  capturedOptions: undefined as
    | {
        canUseServerStt: boolean
        speechToTextLanguage: string
        sttSettings: any
        onTranscript: (text: string) => void
        onError: (err: unknown) => void
        onSuccess: () => void
      }
    | undefined
}

vi.mock("@/hooks/useServerDictation", () => ({
  useServerDictation: (options: any) => {
    serverDictationState.capturedOptions = options
    return {
      isServerDictating: serverDictationState.isServerDictating,
      startServerDictation: serverDictationState.startServerDictation,
      stopServerDictation: serverDictationState.stopServerDictation
    }
  }
}))

const dictationStrategyState = {
  recordServerError: vi.fn(),
  recordServerSuccess: vi.fn(),
  result: {
    requestedMode: "auto" as const,
    resolvedMode: "server" as const,
    speechAvailable: true,
    speechUsesServer: true,
    isDictating: false,
    toggleIntent: "start_server" as "start_server" | "start_browser",
    autoFallbackActive: false,
    autoFallbackErrorClass: null as null | string,
    clearAutoFallback: vi.fn()
  }
}

vi.mock("@/hooks/useDictationStrategy", async () => {
  return {
    useDictationStrategy: () => ({
      ...dictationStrategyState.result,
      recordServerError: dictationStrategyState.recordServerError,
      recordServerSuccess: dictationStrategyState.recordServerSuccess
    })
  }
})

const audioPrefsState = {
  preference: {
    featureGroup: "dictation" as const,
    sourceKind: "default_mic" as const,
    deviceId: null,
    lastKnownLabel: null
  },
  isLoading: false,
  setPreference: vi.fn()
}
vi.mock("@/hooks/useAudioSourcePreferences", () => ({
  useAudioSourcePreferences: () => audioPrefsState
}))

const audioCatalogState = {
  devices: [] as Array<{ deviceId: string; label: string }>,
  isSettled: true
}
vi.mock("@/hooks/useAudioSourceCatalog", () => ({
  useAudioSourceCatalog: () => audioCatalogState
}))

vi.mock("@/audio", () => ({
  resolveAudioCapturePlan: ({
    requestedSource
  }: any) => ({
    requestedSourceKind: requestedSource.sourceKind,
    resolvedSourceKind: requestedSource.sourceKind,
    speechPath: "server_dictation"
  })
}))

const emitDictationDiagnostics = vi.fn()
vi.mock("@/utils/dictation-diagnostics", () => ({
  emitDictationDiagnostics: (input: any) => emitDictationDiagnostics(input)
}))

// Imported after mocks so the mocks bind to the right module paths.
import { useComposerVoiceChat } from "../hooks/useComposerVoiceChat"

const SAMPLE_STT_SETTINGS = {
  model: "whisper-1",
  temperature: 0,
  task: "transcribe",
  responseFormat: "json",
  timestampGranularities: "segment",
  prompt: "",
  useSegmentation: false,
  segK: 1,
  segMinSegmentSize: 0,
  segLambdaBalance: 0,
  segUtteranceExpansionWidth: 0,
  segEmbeddingsProvider: "",
  segEmbeddingsModel: ""
} as const

const baseOptions = (overrides: Record<string, unknown> = {}) =>
  ({
    surface: "playground" as const,
    canUseServerStt: true,
    speechToTextLanguage: "en-US",
    sttSettings: SAMPLE_STT_SETTINGS,
    dictationModeOverride: null,
    dictationAutoFallbackEnabled: false,
    onTranscript: vi.fn(),
    ...overrides
  }) as const

const resetMocks = () => {
  speechRecognitionState.transcript = ""
  speechRecognitionState.isListening = false
  speechRecognitionState.supported = true
  speechRecognitionState.resetTranscript.mockReset()
  speechRecognitionState.start.mockReset()
  speechRecognitionState.stop.mockReset()
  speechRecognitionState.capturedOptions = undefined

  serverDictationState.isServerDictating = false
  serverDictationState.startServerDictation.mockReset()
  serverDictationState.stopServerDictation.mockReset()
  serverDictationState.capturedOptions = undefined

  dictationStrategyState.recordServerError.mockReset().mockReturnValue({
    errorClass: "transient_failure",
    appliedFallback: false,
    requestedMode: "auto",
    resolvedModeBeforeError: "server",
    speechAvailableBeforeError: true,
    speechUsesServerBeforeError: true,
    browserSupportsSpeechRecognition: true,
    browserDictationCompatible: true,
    autoFallbackEnabled: false
  })
  dictationStrategyState.recordServerSuccess.mockReset()
  dictationStrategyState.result = {
    requestedMode: "auto",
    resolvedMode: "server",
    speechAvailable: true,
    speechUsesServer: true,
    isDictating: false,
    toggleIntent: "start_server",
    autoFallbackActive: false,
    autoFallbackErrorClass: null,
    clearAutoFallback: vi.fn()
  }

  audioPrefsState.preference = {
    featureGroup: "dictation",
    sourceKind: "default_mic",
    deviceId: null,
    lastKnownLabel: null
  }
  audioPrefsState.isLoading = false
  audioPrefsState.setPreference.mockReset()

  audioCatalogState.devices = []
  audioCatalogState.isSettled = true

  emitDictationDiagnostics.mockReset()
}

describe("useComposerVoiceChat", () => {
  beforeEach(() => {
    resetMocks()
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it("tags diagnostics with the surface passed in options", () => {
    const { result } = renderHook(() =>
      useComposerVoiceChat(baseOptions({ surface: "sidepanel" }))
    )

    act(() => {
      result.current.handleDictationToggle()
    })

    expect(emitDictationDiagnostics).toHaveBeenCalledWith(
      expect.objectContaining({
        surface: "sidepanel",
        kind: "toggle",
        toggleIntent: "start_server"
      })
    )
  })

  it("forwards server dictation transcripts to onTranscript", () => {
    const onTranscript = vi.fn()
    renderHook(() => useComposerVoiceChat(baseOptions({ onTranscript })))

    // useServerDictation captured the wired callback when the hook ran.
    expect(serverDictationState.capturedOptions).toBeDefined()
    serverDictationState.capturedOptions!.onTranscript("hello from server")
    expect(onTranscript).toHaveBeenCalledWith("hello from server")
  })

  it("emits browser-dictation transcripts via the streaming effect", () => {
    const onTranscript = vi.fn()
    speechRecognitionState.transcript = "partial text"
    speechRecognitionState.isListening = true

    renderHook(() => useComposerVoiceChat(baseOptions({ onTranscript })))

    expect(onTranscript).toHaveBeenCalledWith("partial text")
  })

  it("wires server dictation success/error bridges through the dictation strategy", () => {
    renderHook(() => useComposerVoiceChat(baseOptions()))
    expect(serverDictationState.capturedOptions).toBeDefined()

    // Trigger success — the recorded strategy callback should fire and a
    // server_success diagnostic should be emitted.
    act(() => {
      serverDictationState.capturedOptions!.onSuccess()
    })
    expect(dictationStrategyState.recordServerSuccess).toHaveBeenCalled()
    expect(emitDictationDiagnostics).toHaveBeenCalledWith(
      expect.objectContaining({ kind: "server_success" })
    )

    emitDictationDiagnostics.mockReset()

    // Trigger error — the strategy classifies it and we emit server_error.
    act(() => {
      serverDictationState.capturedOptions!.onError(new Error("boom"))
    })
    expect(dictationStrategyState.recordServerError).toHaveBeenCalled()
    expect(emitDictationDiagnostics).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "server_error",
        errorClass: "transient_failure"
      })
    )
  })

  it("includes the resolved source kind in the diagnostic snapshot", () => {
    renderHook(() => useComposerVoiceChat(baseOptions()))
    expect(serverDictationState.capturedOptions).toBeDefined()

    act(() => {
      serverDictationState.capturedOptions!.onSuccess()
    })

    expect(emitDictationDiagnostics).toHaveBeenCalledWith(
      expect.objectContaining({
        requestedSourceKind: "default_mic",
        resolvedSourceKind: "default_mic",
        speechAvailable: true,
        speechUsesServer: true
      })
    )
  })

  it("delegates start_server toggle to useServerDictation", () => {
    const { result } = renderHook(() => useComposerVoiceChat(baseOptions()))

    act(() => {
      result.current.handleDictationToggle()
    })

    expect(serverDictationState.startServerDictation).toHaveBeenCalledTimes(1)
  })

  it("delegates start_browser toggle to startListening + resetTranscript", () => {
    dictationStrategyState.result.toggleIntent = "start_browser"
    const { result } = renderHook(() => useComposerVoiceChat(baseOptions()))

    act(() => {
      result.current.handleDictationToggle()
    })

    expect(speechRecognitionState.resetTranscript).toHaveBeenCalled()
    expect(speechRecognitionState.start).toHaveBeenCalledWith({
      continuous: true,
      lang: "en-US"
    })
  })

  it("defers the start when dictation source is not yet ready", () => {
    audioCatalogState.isSettled = false
    const { result, rerender } = renderHook(() =>
      useComposerVoiceChat(baseOptions())
    )

    act(() => {
      result.current.handleDictationToggle()
    })
    expect(serverDictationState.startServerDictation).not.toHaveBeenCalled()

    audioCatalogState.isSettled = true
    rerender()

    expect(serverDictationState.startServerDictation).toHaveBeenCalledTimes(1)
  })

  it("invokes onAutoSubmit through the speech-recognition onEnd hook when autoSubmit is enabled", async () => {
    const onAutoSubmit = vi.fn()
    renderHook(() =>
      useComposerVoiceChat(
        baseOptions({
          autoSubmitVoiceMessage: true,
          autoStopTimeout: 1500,
          onAutoSubmit
        })
      )
    )

    expect(speechRecognitionState.capturedOptions).toBeDefined()
    expect(speechRecognitionState.capturedOptions?.autoStop).toBe(true)
    expect(speechRecognitionState.capturedOptions?.autoStopTimeout).toBe(1500)

    await act(async () => {
      await speechRecognitionState.capturedOptions!.onEnd!()
    })
    expect(onAutoSubmit).toHaveBeenCalledTimes(1)
  })
})
