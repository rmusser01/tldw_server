import React from "react"
import {
  resolveAudioCapturePlan,
  type AudioCaptureRequestedSource
} from "@/audio"
import { useAudioSourceCatalog } from "@/hooks/useAudioSourceCatalog"
import { useAudioSourcePreferences } from "@/hooks/useAudioSourcePreferences"
import { useSpeechRecognition } from "@/hooks/useSpeechRecognition"
import {
  useDictationStrategy,
  type DictationErrorClass,
  type DictationModePreference,
  type DictationResolvedMode,
  type DictationServerErrorTransition,
  type DictationToggleIntent,
  type UseDictationStrategyResult
} from "@/hooks/useDictationStrategy"
import { useServerDictation } from "@/hooks/useServerDictation"
import type { SttSettings } from "@/hooks/useSttSettings"
import { emitDictationDiagnostics } from "@/utils/dictation-diagnostics"
import type { ChatComposerSurface } from "../types"

/**
 * Shared dictation orchestration consumed by both Playground and Sidepanel.
 *
 * Both surfaces stitched the same five primitives together with ~300 lines of
 * near-identical wiring:
 *   - `useSpeechRecognition`        (browser SpeechRecognition API)
 *   - `useServerDictation`          (server-side STT via MediaRecorder)
 *   - `useDictationStrategy`        (which engine to use + auto-fallback)
 *   - `useAudioSourcePreferences`   ("dictation" feature group)
 *   - `useAudioSourceCatalog`       (enumerated input devices)
 *
 * Differences between the two surfaces, intentionally pushed back to callers:
 *   - Diagnostics tagging — `surface` arg flows to `emitDictationDiagnostics`.
 *   - Transcript application — Playground collapses large pastes; Sidepanel
 *     just calls `form.setFieldValue("message", text)`. Surfaces pass
 *     `onTranscript` and decide.
 *   - Voice-conversation back-and-forth (`voiceChatEnabled`, voice stream
 *     state, status label, tooltip) — only Playground uses the conversation
 *     feature today; Sidepanel does too but composes those pieces locally.
 *     Keeping voice-conversation orchestration outside this hook avoids
 *     coupling i18n keys, availability resolvers, and the conversation
 *     stream lifecycle to a primitive that should stay focused on dictation.
 *
 * Auto-stop / auto-submit (`autoStopTimeout`, `onAutoSubmit`) are passed
 * through to `useSpeechRecognition` so Playground keeps its existing
 * "stop talking → fire submit" flow; Sidepanel simply omits them.
 */

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface UseComposerVoiceChatOptions {
  /** Which surface this hook is mounted under — drives diagnostics tagging. */
  surface: ChatComposerSurface
  /** Server-side STT availability, computed from server capabilities + health. */
  canUseServerStt: boolean
  /** Language code passed to both server STT and browser recognition. */
  speechToTextLanguage: string
  /** Consolidated STT settings (model, task, segmentation params, ...). */
  sttSettings: SttSettings
  /** User's preferred dictation engine; null means use auto/server default. */
  dictationModeOverride: DictationModePreference | null
  /** Whether to fall back to browser dictation when server fails (auto only). */
  dictationAutoFallbackEnabled: boolean
  /**
   * Called whenever a transcript is produced — by either engine. Surfaces
   * use this to write into their composer state. Server dictation calls this
   * once with the full transcript; browser dictation streams via the
   * `transcript` field and an internal effect re-emits the latest value
   * while listening.
   */
  onTranscript: (text: string) => void
  /**
   * Optional auto-submit on browser-dictation `onEnd`. When provided and
   * `autoStopTimeout` is set, the browser engine stops itself and this fires.
   * Sidepanel does not use this today.
   */
  onAutoSubmit?: () => void
  /** Browser SpeechRecognition autoStop timeout (ms). */
  autoStopTimeout?: number
  /** When true, the browser engine auto-stops after silence and fires onEnd. */
  autoSubmitVoiceMessage?: boolean
}

export interface UseComposerVoiceChatResult {
  // --- Browser speech recognition ---
  transcript: string
  isListening: boolean
  resetTranscript: () => void
  browserSupportsSpeechRecognition: boolean
  // --- Server dictation ---
  isServerDictating: boolean
  startServerDictation: (
    source?: AudioCaptureRequestedSource
  ) => Promise<void>
  stopServerDictation: () => void
  // --- Audio source preferences ---
  dictationAudioSourcePreference: ReturnType<
    typeof useAudioSourcePreferences
  >["preference"]
  setDictationAudioSourcePreference: ReturnType<
    typeof useAudioSourcePreferences
  >["setPreference"]
  dictationResolvedSourceKind: "default_mic" | "mic_device" | "tab_audio" | "system_audio"
  /**
   * Live enumeration of input devices — surfaced for the source picker UI.
   * Sourced from `useAudioSourceCatalog()` inside this hook; exposed so
   * surfaces don't need a second subscription.
   */
  audioInputDevices: ReturnType<typeof useAudioSourceCatalog>["devices"]
  /** Output of `resolveAudioCapturePlan({...})`. */
  dictationCapturePlan: ReturnType<typeof resolveAudioCapturePlan>
  // --- Strategy / unified state ---
  speechAvailable: boolean
  speechUsesServer: boolean
  dictationToggleIntent: DictationToggleIntent
  dictationStrategy: UseDictationStrategyResult
  // --- Handlers ---
  startBrowserDictation: () => void
  stopListening: () => Promise<void>
  handleDictationToggle: () => void
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Wires the shared dictation primitives together. See file header for the
 * design rationale and the explicit list of surface-specific bits that stay
 * outside this hook.
 */
export function useComposerVoiceChat(
  options: UseComposerVoiceChatOptions
): UseComposerVoiceChatResult {
  const {
    surface,
    canUseServerStt,
    speechToTextLanguage,
    sttSettings,
    dictationModeOverride,
    dictationAutoFallbackEnabled,
    onTranscript,
    onAutoSubmit,
    autoStopTimeout,
    autoSubmitVoiceMessage = false
  } = options

  // Keep latest callbacks in refs so we don't tear down the underlying
  // recognizer just because the parent re-rendered with a new closure.
  const onTranscriptRef = React.useRef(onTranscript)
  onTranscriptRef.current = onTranscript
  const onAutoSubmitRef = React.useRef(onAutoSubmit)
  onAutoSubmitRef.current = onAutoSubmit

  // --- Browser SpeechRecognition ---
  const {
    transcript,
    isListening,
    resetTranscript,
    start: startListening,
    stop: stopSpeechRecognition,
    supported: browserSupportsSpeechRecognition
  } = useSpeechRecognition({
    autoStop: autoSubmitVoiceMessage,
    autoStopTimeout,
    onEnd: async () => {
      if (autoSubmitVoiceMessage) {
        onAutoSubmitRef.current?.()
      }
    }
  })

  // --- Audio source preferences + device catalog ---
  const {
    preference: dictationAudioSourcePreference,
    isLoading: dictationSourceLoading,
    setPreference: setDictationAudioSourcePreference
  } = useAudioSourcePreferences("dictation")
  const {
    devices: audioInputDevices,
    isSettled: hasAudioCatalogSettled
  } = useAudioSourceCatalog()

  // --- Capture plan ---
  const dictationCapturePlan = React.useMemo(
    () =>
      resolveAudioCapturePlan({
        featureGroup: "dictation",
        requestedSource: dictationAudioSourcePreference,
        requestedSpeechPath:
          dictationModeOverride === "browser"
            ? "browser_dictation"
            : "server_dictation",
        capabilities: {
          browserDictationSupported: browserSupportsSpeechRecognition,
          serverDictationSupported: canUseServerStt,
          liveVoiceSupported: false,
          secureContextAvailable:
            typeof window === "undefined" ? true : window.isSecureContext
        }
      }),
    [
      browserSupportsSpeechRecognition,
      canUseServerStt,
      dictationAudioSourcePreference,
      dictationModeOverride
    ]
  )

  const dictationSourceReady = hasAudioCatalogSettled && !dictationSourceLoading

  // Resolve the preferred device against the live catalog. If a previously
  // selected mic is no longer enumerated, fall back to default.
  const resolvedDictationSourcePreference = React.useMemo(() => {
    if (!dictationSourceReady) {
      return dictationAudioSourcePreference
    }

    if (dictationAudioSourcePreference.sourceKind !== "mic_device") {
      return dictationAudioSourcePreference
    }

    const requestedDeviceId = String(
      dictationAudioSourcePreference.deviceId || ""
    ).trim()
    const deviceStillAvailable = audioInputDevices.some(
      (device) => device.deviceId === requestedDeviceId
    )

    if (deviceStillAvailable) {
      return dictationAudioSourcePreference
    }

    return {
      featureGroup: "dictation" as const,
      sourceKind: "default_mic" as const,
      deviceId: null,
      lastKnownLabel: null
    }
  }, [audioInputDevices, dictationAudioSourcePreference, dictationSourceReady])

  const resolvedDictationSourceKind = resolvedDictationSourcePreference.sourceKind
  const browserDictationCompatible =
    resolvedDictationSourcePreference.sourceKind === "default_mic"
  const resolvedModeOverride =
    dictationModeOverride === "browser" && !browserDictationCompatible
      ? canUseServerStt
        ? ("server" as const)
        : ("unavailable" as const)
      : null
  const requestedServerDictationSource = React.useMemo<
    AudioCaptureRequestedSource | undefined
  >(
    () =>
      resolvedDictationSourcePreference.sourceKind === "mic_device"
        ? resolvedDictationSourcePreference
        : undefined,
    [resolvedDictationSourcePreference]
  )

  // --- Diagnostics snapshot + bridge refs ---
  const dictationDiagnosticsSnapshotRef = React.useRef<{
    requestedMode: DictationModePreference
    resolvedMode: DictationResolvedMode
    requestedSourceKind: "default_mic" | "mic_device" | "tab_audio" | "system_audio"
    resolvedSourceKind: "default_mic" | "mic_device" | "tab_audio" | "system_audio"
    speechAvailable: boolean
    speechUsesServer: boolean
    fallbackReason: DictationErrorClass | null
  }>({
    requestedMode: "auto",
    resolvedMode: "unavailable",
    requestedSourceKind: "default_mic",
    resolvedSourceKind: "default_mic",
    speechAvailable: false,
    speechUsesServer: false,
    fallbackReason: null
  })

  const serverDictationErrorBridgeRef = React.useRef<
    (error: unknown) => DictationServerErrorTransition
  >(() => ({
    errorClass: "unknown_error",
    appliedFallback: false,
    requestedMode: "auto",
    resolvedModeBeforeError: "unavailable",
    speechAvailableBeforeError: false,
    speechUsesServerBeforeError: false,
    browserSupportsSpeechRecognition: false,
    browserDictationCompatible: false,
    autoFallbackEnabled: false
  }))
  const serverDictationSuccessBridgeRef = React.useRef<() => void>(() => {})

  const handleServerDictationError = React.useCallback(
    (error: unknown) => {
      const transition = serverDictationErrorBridgeRef.current(error)
      const snapshot = dictationDiagnosticsSnapshotRef.current
      emitDictationDiagnostics({
        surface,
        kind: "server_error",
        requestedMode: transition.requestedMode,
        resolvedMode: transition.resolvedModeBeforeError,
        requestedSourceKind: snapshot.requestedSourceKind,
        resolvedSourceKind: snapshot.resolvedSourceKind,
        speechAvailable: transition.speechAvailableBeforeError,
        speechUsesServer: transition.speechUsesServerBeforeError,
        errorClass: transition.errorClass,
        fallbackApplied: transition.appliedFallback,
        fallbackReason: transition.appliedFallback ? transition.errorClass : null
      })
    },
    [surface]
  )

  const handleServerDictationSuccess = React.useCallback(() => {
    serverDictationSuccessBridgeRef.current()
    const snapshot = dictationDiagnosticsSnapshotRef.current
    emitDictationDiagnostics({
      surface,
      kind: "server_success",
      requestedMode: snapshot.requestedMode,
      resolvedMode: snapshot.resolvedMode,
      requestedSourceKind: snapshot.requestedSourceKind,
      resolvedSourceKind: snapshot.resolvedSourceKind,
      speechAvailable: snapshot.speechAvailable,
      speechUsesServer: snapshot.speechUsesServer,
      fallbackReason: snapshot.fallbackReason
    })
  }, [surface])

  // --- Server dictation ---
  const {
    isServerDictating,
    startServerDictation,
    stopServerDictation
  } = useServerDictation({
    canUseServerStt,
    speechToTextLanguage,
    sttSettings,
    onTranscript: (text) => onTranscriptRef.current(text),
    onError: handleServerDictationError,
    onSuccess: handleServerDictationSuccess
  })

  // --- Strategy ---
  const dictationStrategy = useDictationStrategy({
    canUseServerStt,
    browserSupportsSpeechRecognition,
    browserDictationCompatible,
    resolvedModeOverride,
    isServerDictating,
    isBrowserDictating: isListening,
    modeOverride: dictationModeOverride,
    autoFallbackEnabled: Boolean(dictationAutoFallbackEnabled)
  })

  // Bridge strategy → server dictation hook callbacks. These refs are updated
  // every render so the stable `handleServerDictation*` callbacks always see
  // the current strategy. (Same pattern both surfaces used inline.)
  serverDictationErrorBridgeRef.current = dictationStrategy.recordServerError
  serverDictationSuccessBridgeRef.current = dictationStrategy.recordServerSuccess
  dictationDiagnosticsSnapshotRef.current = {
    requestedMode: dictationStrategy.requestedMode,
    resolvedMode: dictationStrategy.resolvedMode,
    requestedSourceKind: dictationCapturePlan.requestedSourceKind,
    resolvedSourceKind: resolvedDictationSourceKind,
    speechAvailable: dictationStrategy.speechAvailable,
    speechUsesServer: dictationStrategy.speechUsesServer,
    fallbackReason: dictationStrategy.autoFallbackErrorClass
  }

  const speechAvailable = dictationStrategy.speechAvailable
  const speechUsesServer = dictationStrategy.speechUsesServer
  const dictationToggleIntent = dictationStrategy.toggleIntent

  // --- Browser dictation ---
  const startBrowserDictation = React.useCallback(() => {
    resetTranscript()
    startListening({
      continuous: true,
      lang: speechToTextLanguage
    })
  }, [resetTranscript, speechToTextLanguage, startListening])

  const stopListening = React.useCallback(async () => {
    if (isListening) {
      stopSpeechRecognition()
    }
  }, [isListening, stopSpeechRecognition])

  // --- Toggle handler with deferred-start when sources aren't ready yet ---
  const [pendingDictationStart, setPendingDictationStart] = React.useState(false)

  const runPendingDictationStart = React.useCallback(() => {
    switch (dictationToggleIntent) {
      case "start_server":
        void startServerDictation(requestedServerDictationSource)
        return true
      case "start_browser":
        startBrowserDictation()
        return true
      default:
        return false
    }
  }, [
    dictationToggleIntent,
    requestedServerDictationSource,
    startBrowserDictation,
    startServerDictation
  ])

  const handleDictationToggle = React.useCallback(() => {
    if (pendingDictationStart) {
      setPendingDictationStart(false)
      return
    }

    switch (dictationToggleIntent) {
      case "start_server":
        if (!dictationSourceReady) {
          setPendingDictationStart(true)
          return
        }
        void startServerDictation(requestedServerDictationSource)
        break
      case "stop_server":
        setPendingDictationStart(false)
        stopServerDictation()
        break
      case "start_browser":
        if (!dictationSourceReady) {
          setPendingDictationStart(true)
          return
        }
        startBrowserDictation()
        break
      case "stop_browser":
        setPendingDictationStart(false)
        stopSpeechRecognition()
        break
      default:
        break
    }
    const snapshot = dictationDiagnosticsSnapshotRef.current
    emitDictationDiagnostics({
      surface,
      kind: "toggle",
      requestedMode: snapshot.requestedMode,
      resolvedMode: snapshot.resolvedMode,
      requestedSourceKind: snapshot.requestedSourceKind,
      resolvedSourceKind: snapshot.resolvedSourceKind,
      speechAvailable: snapshot.speechAvailable,
      speechUsesServer: snapshot.speechUsesServer,
      toggleIntent: dictationToggleIntent,
      fallbackReason: snapshot.fallbackReason
    })
  }, [
    dictationSourceReady,
    dictationToggleIntent,
    pendingDictationStart,
    requestedServerDictationSource,
    startBrowserDictation,
    startServerDictation,
    stopServerDictation,
    stopSpeechRecognition,
    surface
  ])

  React.useEffect(() => {
    if (!pendingDictationStart) return
    if (!dictationSourceReady) return
    if (!runPendingDictationStart()) {
      setPendingDictationStart(false)
      return
    }
    setPendingDictationStart(false)
  }, [dictationSourceReady, pendingDictationStart, runPendingDictationStart])

  // --- Browser-dictation transcript stream → onTranscript ---
  React.useEffect(() => {
    if (isListening) {
      onTranscriptRef.current(transcript)
    }
  }, [transcript, isListening])

  return {
    transcript,
    isListening,
    resetTranscript,
    browserSupportsSpeechRecognition,
    isServerDictating,
    startServerDictation,
    stopServerDictation,
    dictationAudioSourcePreference,
    setDictationAudioSourcePreference,
    dictationResolvedSourceKind: resolvedDictationSourceKind,
    audioInputDevices,
    dictationCapturePlan,
    speechAvailable,
    speechUsesServer,
    dictationToggleIntent,
    dictationStrategy,
    startBrowserDictation,
    stopListening,
    handleDictationToggle
  }
}
