import React from "react"
import { useAudioSourceCatalog } from "@/hooks/useAudioSourceCatalog"
import { useAudioSourcePreferences } from "@/hooks/useAudioSourcePreferences"
import { useMicStream } from "@/hooks/useMicStream"
import {
  BrowserTranscriptWakeDetector,
  type WakeDetectedEvent,
  type WakeDetector,
  type WakeDetectorState
} from "@/hooks/personaWakeDetector"
import type {
  PersonaWakeBehavior,
  ResolvedPersonaVoiceDefaults
} from "@/hooks/useResolvedPersonaVoiceDefaults"
import { useStreamingAudioPlayer } from "@/hooks/useStreamingAudioPlayer"
import { arrayBufferToBase64 } from "@/utils/compress"

export type PersonaLiveVoiceState =
  | "idle"
  | "listening"
  | "thinking"
  | "speaking"
  | "error"

export type PersonaLiveVoiceRecoveryMode = "none" | "listening_stuck" | "thinking_stuck"
export type PersonaLiveVadPreset = "conservative" | "balanced" | "fast" | "custom"
export type PersonaLiveVoiceWarningReasonCode =
  | "barge_in_disabled"
  | "live_voice_disconnected"
  | "server_stt_unavailable"
  | "voice_capture_error"
  | "voice_no_transcript"
  | "voice_manual_mode_required"
  | "voice_tts_unavailable_text_only"
  | "voice_commit_ignored_already_committed"
  | "voice_trigger_not_heard"
  | "voice_empty_command_after_trigger"

export type PersonaWakeWarningReasonCode =
  | "wake_not_configured"
  | "wake_detector_unavailable"
  | "wake_detector_permission_denied"
  | "wake_detector_error"
  | "wake_activation_disconnected"
  | "wake_activation_send_failed"
  | "wake_activation_rejected_not_saved_in_profile"
  | "wake_activation_rejected_missing_from_runtime_config"
  | "wake_activation_rejected_phrase_not_configured"
  | "wake_activation_rejected"

type PersonaWakeStopReason =
  | "disarmed"
  | "route_leave"
  | "stop_live_voice"
  | "persona_switch"
  | "tab_switch"
  | "session_close"

type UsePersonaLiveVoiceControllerArgs = {
  ws: WebSocket | null
  connected: boolean
  sessionId: string
  personaId: string
  resolvedDefaults: ResolvedPersonaVoiceDefaults
  canUseServerStt: boolean
  wakeTriggerPhrases?: string[]
  wakeDetectorFactory?: () => WakeDetector
}

type PersonaLiveVoicePayload = Record<string, unknown> | null | undefined

const LISTENING_RECOVERY_TIMEOUT_MS = 4_000
const THINKING_RECOVERY_TIMEOUT_MS = 8_000

const LIVE_VAD_PRESETS = {
  conservative: {
    autoCommitEnabled: true,
    vadThreshold: 0.65,
    minSilenceMs: 450,
    turnStopSecs: 0.35,
    minUtteranceSecs: 0.6
  },
  balanced: {
    autoCommitEnabled: true,
    vadThreshold: 0.5,
    minSilenceMs: 250,
    turnStopSecs: 0.2,
    minUtteranceSecs: 0.4
  },
  fast: {
    autoCommitEnabled: true,
    vadThreshold: 0.35,
    minSilenceMs: 150,
    turnStopSecs: 0.1,
    minUtteranceSecs: 0.25
  }
} as const

const isMatchingLiveVadPreset = (
  candidate: {
    autoCommitEnabled: boolean
    vadThreshold: number
    minSilenceMs: number
    turnStopSecs: number
    minUtteranceSecs: number
  },
  preset: (typeof LIVE_VAD_PRESETS)[keyof typeof LIVE_VAD_PRESETS]
): boolean =>
  candidate.autoCommitEnabled === preset.autoCommitEnabled &&
  candidate.vadThreshold === preset.vadThreshold &&
  candidate.minSilenceMs === preset.minSilenceMs &&
  candidate.turnStopSecs === preset.turnStopSecs &&
  candidate.minUtteranceSecs === preset.minUtteranceSecs

const normalizeTtsProvider = (provider: string): string =>
  String(provider || "").trim().toLowerCase()

const browserSpeechSupported = (): boolean =>
  typeof window !== "undefined" && "speechSynthesis" in window

const formatActiveToolStatus = (tool: unknown, why: unknown): string => {
  const toolName = String(tool || "").trim()
  const whyText = String(why || "").trim()
  if (!toolName) return ""
  if (!whyText) return `Running ${toolName}...`
  return `Running ${toolName}: ${whyText}`
}

const getRecordValue = (value: unknown, key: string): unknown => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined
  return (value as Record<string, unknown>)[key]
}

const normalizeToolNameCandidate = (value: unknown): string => {
  if (typeof value === "string" || typeof value === "number") {
    return String(value).trim()
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return ""
  }
  return (
    normalizeToolNameCandidate(getRecordValue(value, "tool_name")) ||
    normalizeToolNameCandidate(getRecordValue(value, "name")) ||
    normalizeToolNameCandidate(getRecordValue(getRecordValue(value, "function"), "name"))
  )
}

const getPayloadToolName = (payload: PersonaLiveVoicePayload): string => {
  if (!payload || typeof payload !== "object") return ""
  return (
    normalizeToolNameCandidate(payload.tool_name) ||
    normalizeToolNameCandidate(payload.tool) ||
    normalizeToolNameCandidate(payload.name)
  )
}

const WAKE_REJECTION_MESSAGES: Record<string, string> = {
  not_saved_in_profile:
    "Wake phrase was heard, but it is not a saved trigger phrase for this " +
    "persona. Add it to the selected persona's trigger phrases, then arm " +
    "wake listening again.",
  missing_from_runtime_config:
    "Wake phrase was heard, but the live voice configuration did not load " +
    "that saved trigger phrase. Reconnect Persona Live or save voice defaults again.",
  phrase_not_configured:
    "Wake phrase was heard, but it is not configured for this Persona Live " +
    "session. Check the selected persona's saved trigger phrases."
}

const WAKE_REJECTION_REASON_CODES: Record<string, PersonaWakeWarningReasonCode> = {
  not_saved_in_profile: "wake_activation_rejected_not_saved_in_profile",
  missing_from_runtime_config: "wake_activation_rejected_missing_from_runtime_config",
  phrase_not_configured: "wake_activation_rejected_phrase_not_configured"
}

const formatWakeActivationRejectedMessage = (
  payload: PersonaLiveVoicePayload
): string => {
  const reason = String(payload?.wake_rejection_reason || "").trim()
  return (
    WAKE_REJECTION_MESSAGES[reason] ||
    String(payload?.message || "Wake activation was rejected.")
  )
}

const getWakeActivationRejectedReasonCode = (
  payload: PersonaLiveVoicePayload
): PersonaWakeWarningReasonCode => {
  const reason = String(payload?.wake_rejection_reason || "").trim()
  return WAKE_REJECTION_REASON_CODES[reason] || "wake_activation_rejected"
}

const getWakeDetectorErrorReasonCode = (
  code: string | null | undefined
): PersonaWakeWarningReasonCode => {
  const normalized = String(code || "").trim().toLowerCase()
  if (normalized === "not-allowed" || normalized === "service-not-allowed") {
    return "wake_detector_permission_denied"
  }
  return "wake_detector_error"
}

export const usePersonaLiveVoiceController = ({
  ws,
  connected,
  sessionId,
  personaId,
  resolvedDefaults,
  canUseServerStt,
  wakeTriggerPhrases = [],
  wakeDetectorFactory
}: UsePersonaLiveVoiceControllerArgs) => {
  const [state, setState] = React.useState<PersonaLiveVoiceState>("idle")
  const [heardText, setHeardText] = React.useState("")
  const [lastCommittedText, setLastCommittedText] = React.useState("")
  const [activeToolName, setActiveToolName] = React.useState("")
  const [activeToolStatus, setActiveToolStatus] = React.useState("")
  const [warning, setWarning] = React.useState<string | null>(null)
  const [warningReasonCode, setWarningReasonCode] =
    React.useState<PersonaLiveVoiceWarningReasonCode | null>(null)
  const [manualModeRequired, setManualModeRequired] = React.useState(false)
  const [textOnlyDueToTtsFailure, setTextOnlyDueToTtsFailure] = React.useState(false)
  const [sessionAutoResume, setSessionAutoResume] = React.useState(resolvedDefaults.autoResume)
  const [sessionBargeIn, setSessionBargeIn] = React.useState(resolvedDefaults.bargeIn)
  const [autoCommitEnabled, setAutoCommitEnabled] = React.useState(
    resolvedDefaults.autoCommitEnabled
  )
  const [vadThreshold, setVadThreshold] = React.useState(
    resolvedDefaults.vadThreshold
  )
  const [minSilenceMs, setMinSilenceMs] = React.useState(
    resolvedDefaults.minSilenceMs
  )
  const [turnStopSecs, setTurnStopSecs] = React.useState(
    resolvedDefaults.turnStopSecs
  )
  const [minUtteranceSecs, setMinUtteranceSecs] = React.useState(
    resolvedDefaults.minUtteranceSecs
  )
  const [recoveryMode, setRecoveryMode] =
    React.useState<PersonaLiveVoiceRecoveryMode>("none")
  const [listeningRecoveryCount, setListeningRecoveryCount] = React.useState(0)
  const [thinkingRecoveryCount, setThinkingRecoveryCount] = React.useState(0)
  const [listeningRecoveryRestartKey, setListeningRecoveryRestartKey] = React.useState(0)
  const [thinkingRecoveryArmed, setThinkingRecoveryArmed] = React.useState(false)
  const [thinkingRecoveryRestartKey, setThinkingRecoveryRestartKey] = React.useState(0)
  const { preference: liveVoiceSourcePreference, isLoading: liveVoiceSourceLoading } =
    useAudioSourcePreferences("live_voice")
  const {
    devices: audioInputDevices,
    isSettled: hasAudioCatalogSettled
  } = useAudioSourceCatalog()
  const [pendingStartRequest, setPendingStartRequest] = React.useState(false)
  const [wakeArmed, setWakeArmed] = React.useState(false)
  const [wakeDetectorState, setWakeDetectorState] =
    React.useState<WakeDetectorState>("idle")
  const [wakeWarning, setWakeWarning] = React.useState<string | null>(null)
  const [wakeWarningReasonCode, setWakeWarningReasonCode] =
    React.useState<PersonaWakeWarningReasonCode | null>(null)
  const [sessionWakeBehavior, setSessionWakeBehavior] =
    React.useState<PersonaWakeBehavior>(resolvedDefaults.wakeBehavior)

  const heardTranscriptRef = React.useRef("")
  const manualModeRequiredRef = React.useRef(false)
  const textOnlyDueToTtsFailureRef = React.useRef(false)
  const pendingBinaryFinishRef = React.useRef(false)
  const pendingResumeRef = React.useRef(false)
  const awaitingTtsTimeoutRef = React.useRef<number | null>(null)
  const browserUtteranceActiveRef = React.useRef(false)
  const listeningRecoveryTimeoutRef = React.useRef<number | null>(null)
  const thinkingRecoveryTimeoutRef = React.useRef<number | null>(null)
  const wakeDetectorRef = React.useRef<WakeDetector | null>(null)
  const wakeArmedRef = React.useRef(false)
  const wakeActiveRef = React.useRef(false)
  const wakeStartTokenRef = React.useRef(0)
  const stopWakeListeningRef = React.useRef<
    (reason?: PersonaWakeStopReason, targetSessionId?: string) => Promise<void>
  >(async () => undefined)
  const restartWakeListeningRef = React.useRef<(() => void) | null>(null)
  const personaSessionKeyRef = React.useRef(`${personaId}:${sessionId}`)
  const personaSessionIdRef = React.useRef(sessionId)

  const activeProvider = React.useMemo(
    () => normalizeTtsProvider(resolvedDefaults.ttsProvider),
    [resolvedDefaults.ttsProvider]
  )
  const liveVoiceSourceReady = hasAudioCatalogSettled && !liveVoiceSourceLoading
  const liveVoiceResolvedSource = React.useMemo(() => {
    if (!liveVoiceSourceReady) {
      return liveVoiceSourcePreference
    }

    if (liveVoiceSourcePreference.sourceKind !== "mic_device") {
      return liveVoiceSourcePreference
    }

    const requestedDeviceId = String(liveVoiceSourcePreference.deviceId || "").trim()
    const deviceStillAvailable = audioInputDevices.some(
      (device) => device.deviceId === requestedDeviceId
    )

    if (deviceStillAvailable) {
      return liveVoiceSourcePreference
    }

    return {
      featureGroup: "live_voice" as const,
      sourceKind: "default_mic" as const,
      deviceId: null,
      lastKnownLabel: null
    }
  }, [audioInputDevices, liveVoiceSourcePreference, liveVoiceSourceReady])
  const liveVoiceDeviceId =
    liveVoiceResolvedSource.sourceKind === "mic_device"
      ? liveVoiceResolvedSource.deviceId
      : null

  const vadPreset = React.useMemo<PersonaLiveVadPreset>(() => {
    const current = {
      autoCommitEnabled,
      vadThreshold,
      minSilenceMs,
      turnStopSecs,
      minUtteranceSecs
    }
    if (isMatchingLiveVadPreset(current, LIVE_VAD_PRESETS.conservative)) {
      return "conservative"
    }
    if (isMatchingLiveVadPreset(current, LIVE_VAD_PRESETS.balanced)) {
      return "balanced"
    }
    if (isMatchingLiveVadPreset(current, LIVE_VAD_PRESETS.fast)) {
      return "fast"
    }
    return "custom"
  }, [autoCommitEnabled, minSilenceMs, minUtteranceSecs, turnStopSecs, vadThreshold])

  const setVadPreset = React.useCallback((preset: Exclude<PersonaLiveVadPreset, "custom">) => {
    const next = LIVE_VAD_PRESETS[preset]
    setAutoCommitEnabled(next.autoCommitEnabled)
    setVadThreshold(next.vadThreshold)
    setMinSilenceMs(next.minSilenceMs)
    setTurnStopSecs(next.turnStopSecs)
    setMinUtteranceSecs(next.minUtteranceSecs)
  }, [])

  const setVoiceWarning = React.useCallback(
    (
      message: string | null,
      reasonCode: PersonaLiveVoiceWarningReasonCode | null = null
    ) => {
      setWarning(message)
      setWarningReasonCode(message ? reasonCode : null)
    },
    []
  )

  const setWakeRecoveryWarning = React.useCallback(
    (
      message: string | null,
      reasonCode: PersonaWakeWarningReasonCode | null = null
    ) => {
      setWakeWarning(message)
      setWakeWarningReasonCode(message ? reasonCode : null)
    },
    []
  )

  const clearTransientWarning = React.useCallback(() => {
    if (textOnlyDueToTtsFailureRef.current) return
    if (manualModeRequiredRef.current) return
    setVoiceWarning(null)
  }, [setVoiceWarning])

  const clearAwaitingTtsTimeout = React.useCallback(() => {
    if (awaitingTtsTimeoutRef.current != null && typeof window !== "undefined") {
      window.clearTimeout(awaitingTtsTimeoutRef.current)
    }
    awaitingTtsTimeoutRef.current = null
  }, [])

  const clearListeningRecoveryTimeout = React.useCallback(() => {
    if (listeningRecoveryTimeoutRef.current != null && typeof window !== "undefined") {
      window.clearTimeout(listeningRecoveryTimeoutRef.current)
    }
    listeningRecoveryTimeoutRef.current = null
  }, [])

  const clearThinkingRecoveryTimeout = React.useCallback(() => {
    if (thinkingRecoveryTimeoutRef.current != null && typeof window !== "undefined") {
      window.clearTimeout(thinkingRecoveryTimeoutRef.current)
    }
    thinkingRecoveryTimeoutRef.current = null
  }, [])

  const armThinkingRecovery = React.useCallback(() => {
    clearThinkingRecoveryTimeout()
    setRecoveryMode((current) => (current === "thinking_stuck" ? "none" : current))
    setThinkingRecoveryArmed(true)
    setThinkingRecoveryRestartKey((current) => current + 1)
  }, [clearThinkingRecoveryTimeout])

  const clearThinkingRecovery = React.useCallback(() => {
    clearThinkingRecoveryTimeout()
    setThinkingRecoveryArmed(false)
    setRecoveryMode((current) => (current === "thinking_stuck" ? "none" : current))
  }, [clearThinkingRecoveryTimeout])

  const handleVoiceError = React.useCallback((error: unknown) => {
    const message =
      error instanceof Error
        ? error.message
        : "Live voice capture failed. Check microphone permissions and audio setup."
    setVoiceWarning(message, "voice_capture_error")
    setState("error")
  }, [setVoiceWarning])

  const {
    start: audioStart,
    append: audioAppend,
    finish: audioFinish,
    stop: audioStop,
    state: audioState
  } = useStreamingAudioPlayer()

  const stopBrowserSpeech = React.useCallback(() => {
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      try {
        window.speechSynthesis.cancel()
      } catch (error) {
        console.error("stopBrowserSpeech: speechSynthesis.cancel failed", error)
      }
    }
    browserUtteranceActiveRef.current = false
  }, [])

  const stopCurrentPlayback = React.useCallback(() => {
    clearAwaitingTtsTimeout()
    pendingResumeRef.current = false
    pendingBinaryFinishRef.current = false
    audioStop()
    stopBrowserSpeech()
  }, [audioStop, clearAwaitingTtsTimeout, stopBrowserSpeech])

  const sendVoiceCommit = React.useCallback(
    (transcript: string, source = "persona_live_voice_manual") => {
      if (!connected || !sessionId || !ws || ws.readyState !== WebSocket.OPEN) {
        setVoiceWarning(
          "Live voice is disconnected. Reconnect Persona Garden to send spoken commands.",
          "live_voice_disconnected"
        )
        setState("error")
        return
      }

      const normalizedTranscript = String(transcript || "").trim()
      if (!normalizedTranscript) {
        setVoiceWarning(
          "No speech transcript was captured for that live turn.",
          "voice_no_transcript"
        )
        setState("idle")
        return
      }

      try {
        ws.send(
          JSON.stringify({
            type: "voice_commit",
            session_id: sessionId,
            transcript: normalizedTranscript,
            source
          })
        )
        clearTransientWarning()
        armThinkingRecovery()
        setState("thinking")
      } catch (error) {
        handleVoiceError(error)
      }
    },
    [
      armThinkingRecovery,
      clearTransientWarning,
      connected,
      handleVoiceError,
      sessionId,
      setVoiceWarning,
      ws
    ]
  )

  const sendWakeActivation = React.useCallback(
    (event: WakeDetectedEvent): boolean => {
      if (!connected || !sessionId || !ws || ws.readyState !== WebSocket.OPEN) {
        setWakeRecoveryWarning(
          "Wake phrase heard, but Persona Live is not connected.",
          "wake_activation_disconnected"
        )
        return false
      }

      try {
        ws.send(
          JSON.stringify({
            type: "wake_activation",
            session_id: sessionId,
            matched_phrase: event.canonicalPhrase,
            detector_kind: event.detectorKind,
            detected_at_ms: event.detectedAtMs
          })
        )
        return true
      } catch {
        setWakeRecoveryWarning(
          "Wake phrase heard, but activation could not be sent.",
          "wake_activation_send_failed"
        )
        return false
      }
    },
    [connected, sessionId, setWakeRecoveryWarning, ws]
  )

  const sendWakeDeactivation = React.useCallback(
    (reason: PersonaWakeStopReason, targetSessionId = sessionId) => {
      const normalizedSessionId = String(targetSessionId || "").trim()
      if (!normalizedSessionId || !ws || ws.readyState !== WebSocket.OPEN) return
      try {
        ws.send(
          JSON.stringify({
            type: "wake_deactivation",
            session_id: normalizedSessionId,
            reason
          })
        )
      } catch {
        // Ignore transient websocket send errors during teardown.
      }
    },
    [sessionId, ws]
  )

  const { start: startMicStream, stop: stopMicStream, active: micActive } = useMicStream(
    (chunk) => {
      if (!connected || !sessionId || !ws || ws.readyState !== WebSocket.OPEN) return
      try {
        ws.send(
          JSON.stringify({
            type: "audio_chunk",
            session_id: sessionId,
            audio_format: "pcm16",
            bytes_base64: arrayBufferToBase64(chunk)
          })
        )
      } catch {
        // Ignore transient websocket send errors while the session reconnects.
      }
    }
  )

  const startMicCapture = React.useCallback(async () => {
    if (!canUseServerStt) {
      setVoiceWarning(
        "This tldw connection does not expose server speech transcription.",
        "server_stt_unavailable"
      )
      setState("error")
      return false
    }
    if (!connected || !sessionId || !ws || ws.readyState !== WebSocket.OPEN) {
      setVoiceWarning(
        "Connect Persona Garden before starting live voice.",
        "live_voice_disconnected"
      )
      setState("error")
      return false
    }
    if (!liveVoiceSourceReady) {
      return false
    }

    clearTransientWarning()
    setHeardText("")
    heardTranscriptRef.current = ""
    setActiveToolName("")
    setActiveToolStatus("")
    setRecoveryMode("none")
    clearThinkingRecovery()
    setState("listening")
    try {
      await startMicStream({ deviceId: liveVoiceDeviceId })
      return true
    } catch (error) {
      handleVoiceError(error)
      return false
    }
  }, [
    canUseServerStt,
    clearThinkingRecovery,
    clearTransientWarning,
    connected,
    handleVoiceError,
    liveVoiceDeviceId,
    liveVoiceSourceReady,
    sessionId,
    startMicStream,
    setVoiceWarning,
    ws
  ])

  const suspendWakeDetectorForLiveCapture = React.useCallback(async () => {
    if (!wakeArmedRef.current) return
    wakeStartTokenRef.current += 1
    if (!wakeDetectorRef.current) {
      setWakeDetectorState("idle")
      return
    }
    try {
      await wakeDetectorRef.current.stop()
    } finally {
      wakeDetectorRef.current = null
      setWakeDetectorState("idle")
    }
  }, [])

  const stopWakeListening = React.useCallback(
    async (
      reason: PersonaWakeStopReason = "disarmed",
      targetSessionId?: string
    ) => {
      wakeStartTokenRef.current += 1
      wakeArmedRef.current = false
      wakeActiveRef.current = false
      const detector = wakeDetectorRef.current
      wakeDetectorRef.current = null
      sendWakeDeactivation(reason, targetSessionId)
      try {
        await detector?.stop()
      } catch {
        // Wake detector teardown is best-effort; state cleanup must still finish.
      } finally {
        setWakeDetectorState("idle")
        setWakeArmed(false)
      }
    },
    [sendWakeDeactivation]
  )

  const finishVoiceTurn = React.useCallback(() => {
    if (
      wakeArmedRef.current &&
      wakeActiveRef.current &&
      (sessionWakeBehavior === "one_shot" ||
        sessionWakeBehavior === "push_to_talk_after_wake")
    ) {
      wakeActiveRef.current = false
      pendingResumeRef.current = false
      setPendingStartRequest(false)
      setState("idle")
      restartWakeListeningRef.current?.()
      return
    }
    if (sessionAutoResume && canUseServerStt && connected) {
      if (liveVoiceSourceReady) {
        setPendingStartRequest(false)
        pendingResumeRef.current = false
        void startMicCapture()
        return
      }
      pendingResumeRef.current = true
      setPendingStartRequest(true)
      setState("idle")
      return
    }
    pendingResumeRef.current = false
    setPendingStartRequest(false)
    setState("idle")
  }, [
    canUseServerStt,
    connected,
    liveVoiceSourceReady,
    sessionAutoResume,
    sessionWakeBehavior,
    startMicCapture
  ])

  const resetTurn = React.useCallback(() => {
    clearListeningRecoveryTimeout()
    clearThinkingRecovery()
    if (micActive) {
      stopMicStream()
    }
    pendingResumeRef.current = false
    setPendingStartRequest(false)
    setRecoveryMode("none")
    setHeardText("")
    heardTranscriptRef.current = ""
    setLastCommittedText("")
    setActiveToolName("")
    setActiveToolStatus("")
    if (!manualModeRequiredRef.current && !textOnlyDueToTtsFailureRef.current) {
      setVoiceWarning(null)
    }
    setState("idle")
  }, [
    clearListeningRecoveryTimeout,
    clearThinkingRecovery,
    micActive,
    setVoiceWarning,
    stopMicStream
  ])

  const keepListening = React.useCallback(() => {
    clearListeningRecoveryTimeout()
    setRecoveryMode("none")
    setListeningRecoveryRestartKey((current) => current + 1)
  }, [clearListeningRecoveryTimeout])

  const waitOnRecovery = React.useCallback(() => {
    if (state !== "thinking") {
      setRecoveryMode("none")
      return
    }
    armThinkingRecovery()
  }, [armThinkingRecovery, state])

  React.useEffect(() => {
    wakeArmedRef.current = wakeArmed
  }, [wakeArmed])

  React.useEffect(() => {
    stopWakeListeningRef.current = stopWakeListening
  }, [stopWakeListening])

  React.useEffect(() => {
    const nextKey = `${personaId}:${sessionId}`
    const previousKey = personaSessionKeyRef.current
    const previousSessionId = personaSessionIdRef.current
    if (previousKey === nextKey) return
    if (wakeArmedRef.current) {
      void stopWakeListeningRef.current("persona_switch", previousSessionId)
    }
    personaSessionKeyRef.current = nextKey
    personaSessionIdRef.current = sessionId
  }, [personaId, sessionId])

  React.useEffect(() => {
    manualModeRequiredRef.current = manualModeRequired
  }, [manualModeRequired])

  React.useEffect(() => {
    textOnlyDueToTtsFailureRef.current = textOnlyDueToTtsFailure
  }, [textOnlyDueToTtsFailure])

  React.useEffect(() => {
    if (!pendingStartRequest) return
    if (!liveVoiceSourceReady) return
    setPendingStartRequest(false)
    void startMicCapture()
  }, [liveVoiceSourceReady, pendingStartRequest, startMicCapture])

  React.useEffect(() => {
    if (!pendingResumeRef.current) return
    if (audioState.playing) return
    if (!sessionAutoResume || !canUseServerStt || !connected) {
      pendingResumeRef.current = false
      setPendingStartRequest(false)
      setState("idle")
      return
    }
    if (!liveVoiceSourceReady) {
      setPendingStartRequest(true)
      return
    }
    setPendingStartRequest(false)
    pendingResumeRef.current = false
    void startMicCapture()
  }, [
    audioState.playing,
    canUseServerStt,
    connected,
    liveVoiceSourceReady,
    sessionAutoResume,
    startMicCapture
  ])

  const startListening = React.useCallback(async () => {
    if (state === "speaking") {
      if (!sessionBargeIn) {
        setVoiceWarning("Barge-in is off for this live session.", "barge_in_disabled")
        return
      }
      stopCurrentPlayback()
    }
    if (!canUseServerStt) {
      setVoiceWarning(
        "This tldw connection does not expose server speech transcription.",
        "server_stt_unavailable"
      )
      setState("error")
      return
    }
    if (!connected || !sessionId || !ws || ws.readyState !== WebSocket.OPEN) {
      setVoiceWarning(
        "Connect Persona Garden before starting live voice.",
        "live_voice_disconnected"
      )
      setState("error")
      return
    }
    if (!liveVoiceSourceReady) {
      clearTransientWarning()
      setHeardText("")
      heardTranscriptRef.current = ""
      setActiveToolName("")
      setActiveToolStatus("")
      setRecoveryMode("none")
      clearThinkingRecovery()
      setPendingStartRequest(true)
      setState("idle")
      return
    }

    setPendingStartRequest(false)
    await suspendWakeDetectorForLiveCapture()
    await startMicCapture()
  }, [
    canUseServerStt,
    clearThinkingRecovery,
    clearTransientWarning,
    connected,
    liveVoiceSourceReady,
    sessionBargeIn,
    sessionId,
    state,
    startMicCapture,
    stopCurrentPlayback,
    suspendWakeDetectorForLiveCapture,
    setVoiceWarning,
    ws
  ])

  const startWakeListening = React.useCallback(async (
    options: { preserveWarning?: boolean } = {}
  ) => {
    const phrases = (wakeTriggerPhrases || [])
      .map((phrase) => String(phrase || "").trim())
      .filter(Boolean)
    if (phrases.length === 0) {
      setWakeRecoveryWarning(
        "Add a persona trigger phrase before arming wake listening.",
        "wake_not_configured"
      )
      return
    }

    const startToken = wakeStartTokenRef.current + 1
    wakeStartTokenRef.current = startToken
    const isCurrentStart = () => wakeStartTokenRef.current === startToken
    wakeArmedRef.current = true
    wakeActiveRef.current = false
    setWakeArmed(true)
    if (!options.preserveWarning) {
      setWakeRecoveryWarning(null)
    }
    setWakeDetectorState("starting")

    const detector = wakeDetectorFactory?.() || new BrowserTranscriptWakeDetector()
    const available = await detector.isAvailable()
    if (!isCurrentStart()) {
      await detector.stop()
      return
    }
    if (!available) {
      wakeArmedRef.current = false
      wakeActiveRef.current = false
      setWakeArmed(false)
      setWakeDetectorState("unavailable")
      setWakeRecoveryWarning(
        "Wake listening is unavailable in this browser context.",
        "wake_detector_unavailable"
      )
      return
    }

    await wakeDetectorRef.current?.stop()
    if (!isCurrentStart()) {
      await detector.stop()
      return
    }
    wakeDetectorRef.current = detector
    wakeActiveRef.current = false

    try {
      await detector.start({
        phrases,
        locale: resolvedDefaults.sttLanguage,
        onStateChange: setWakeDetectorState,
        onError: (error) =>
          setWakeRecoveryWarning(
            error.message,
            getWakeDetectorErrorReasonCode(error.code)
          ),
        onWake: (event) => {
          if (!isCurrentStart() || wakeDetectorRef.current !== detector) return
          if (wakeActiveRef.current) return
          wakeActiveRef.current = true
          const activationSent = sendWakeActivation(event)
          if (!activationSent) {
            wakeActiveRef.current = false
            return
          }
          wakeDetectorRef.current = null
          void detector.stop()
          if (sessionWakeBehavior !== "push_to_talk_after_wake") {
            void startListening()
          }
        }
      })
      if (!isCurrentStart()) {
        if (wakeDetectorRef.current === detector) {
          wakeDetectorRef.current = null
        }
        await detector.stop()
      }
    } catch (error) {
      if (!isCurrentStart()) return
      wakeArmedRef.current = false
      wakeActiveRef.current = false
      wakeDetectorRef.current = null
      setWakeArmed(false)
      setWakeDetectorState("error")
      setWakeRecoveryWarning(
        error instanceof Error ? error.message : "Wake listening could not start.",
        "wake_detector_error"
      )
    }
  }, [
    resolvedDefaults.sttLanguage,
    sendWakeActivation,
    setWakeRecoveryWarning,
    sessionWakeBehavior,
    startListening,
    wakeDetectorFactory,
    wakeTriggerPhrases
  ])

  const toggleWakeArmed = React.useCallback(async () => {
    if (wakeArmedRef.current) {
      await stopWakeListening("disarmed")
      return
    }
    await startWakeListening()
  }, [startWakeListening, stopWakeListening])

  React.useEffect(() => {
    restartWakeListeningRef.current = () => {
      void startWakeListening()
    }
  }, [startWakeListening])

  const stopListening = React.useCallback(() => {
    if (!micActive && !pendingStartRequest) return
    setPendingStartRequest(false)
    pendingResumeRef.current = false
    clearListeningRecoveryTimeout()
    clearThinkingRecovery()
    setRecoveryMode("none")
    if (micActive) {
      stopMicStream()
    }
    setState("idle")
    if (wakeArmedRef.current && !wakeActiveRef.current && !wakeDetectorRef.current) {
      restartWakeListeningRef.current?.()
    }
  }, [
    clearListeningRecoveryTimeout,
    clearThinkingRecovery,
    micActive,
    pendingStartRequest,
    stopMicStream
  ])

  const sendCurrentTranscriptNow = React.useCallback(() => {
    if (micActive) {
      stopMicStream()
    }
    sendVoiceCommit(heardTranscriptRef.current, "persona_live_voice_manual")
  }, [micActive, sendVoiceCommit, stopMicStream])

  const toggleListening = React.useCallback(() => {
    if (micActive || pendingStartRequest) {
      stopListening()
      return
    }
    void startListening()
  }, [micActive, pendingStartRequest, startListening, stopListening])

  const playBrowserSpeech = React.useCallback(
    (text: string) => {
      const spokenText = String(text || "").trim()
      if (!spokenText) {
        finishVoiceTurn()
        return
      }
      if (!browserSpeechSupported()) {
        textOnlyDueToTtsFailureRef.current = true
        setTextOnlyDueToTtsFailure(true)
        setVoiceWarning(
          "Browser speech playback is unavailable. Continuing in text-only mode.",
          "voice_tts_unavailable_text_only"
        )
        finishVoiceTurn()
        return
      }

      stopBrowserSpeech()
      const synthesis = window.speechSynthesis
      const utterance = new SpeechSynthesisUtterance(spokenText)
      if (resolvedDefaults.sttLanguage) {
        utterance.lang = resolvedDefaults.sttLanguage
      }
      const availableVoices = synthesis.getVoices()
      const matchedVoice = availableVoices.find(
        (voice) =>
          voice.voiceURI === resolvedDefaults.ttsVoice ||
          voice.name === resolvedDefaults.ttsVoice
      )
      if (matchedVoice) {
        utterance.voice = matchedVoice
      }
      browserUtteranceActiveRef.current = true
      utterance.onend = () => {
        browserUtteranceActiveRef.current = false
        finishVoiceTurn()
      }
      utterance.onerror = () => {
        browserUtteranceActiveRef.current = false
        textOnlyDueToTtsFailureRef.current = true
        setTextOnlyDueToTtsFailure(true)
        setVoiceWarning(
          "Browser speech playback failed. Continuing in text-only mode.",
          "voice_tts_unavailable_text_only"
        )
        finishVoiceTurn()
      }
      setState("speaking")
      synthesis.speak(utterance)
    },
    [
      finishVoiceTurn,
      resolvedDefaults.sttLanguage,
      resolvedDefaults.ttsVoice,
      setVoiceWarning,
      stopBrowserSpeech
    ]
  )

  React.useEffect(() => {
    setSessionAutoResume(resolvedDefaults.autoResume)
    setSessionBargeIn(resolvedDefaults.bargeIn)
    setSessionWakeBehavior(resolvedDefaults.wakeBehavior)
    setAutoCommitEnabled(resolvedDefaults.autoCommitEnabled)
    setVadThreshold(resolvedDefaults.vadThreshold)
    setMinSilenceMs(resolvedDefaults.minSilenceMs)
    setTurnStopSecs(resolvedDefaults.turnStopSecs)
    setMinUtteranceSecs(resolvedDefaults.minUtteranceSecs)
    manualModeRequiredRef.current = false
    setManualModeRequired(false)
    textOnlyDueToTtsFailureRef.current = false
    setTextOnlyDueToTtsFailure(false)
    setVoiceWarning(null)
    setWakeRecoveryWarning(null)
    setHeardText("")
    heardTranscriptRef.current = ""
    setLastCommittedText("")
    setActiveToolName("")
    setActiveToolStatus("")
    setRecoveryMode("none")
    setListeningRecoveryCount(0)
    setThinkingRecoveryCount(0)
    setListeningRecoveryRestartKey(0)
    setThinkingRecoveryArmed(false)
    setThinkingRecoveryRestartKey(0)
    setState("idle")
    clearAwaitingTtsTimeout()
    clearListeningRecoveryTimeout()
    clearThinkingRecoveryTimeout()
    pendingBinaryFinishRef.current = false
    pendingResumeRef.current = false
    setPendingStartRequest(false)
    stopMicStream()
    stopCurrentPlayback()
  }, [
    clearAwaitingTtsTimeout,
    clearListeningRecoveryTimeout,
    clearThinkingRecoveryTimeout,
    personaId,
    resolvedDefaults.autoCommitEnabled,
    resolvedDefaults.autoResume,
    resolvedDefaults.bargeIn,
    resolvedDefaults.wakeBehavior,
    resolvedDefaults.minSilenceMs,
    resolvedDefaults.minUtteranceSecs,
    resolvedDefaults.turnStopSecs,
    resolvedDefaults.vadThreshold,
    sessionId,
    setVoiceWarning,
    setWakeRecoveryWarning,
    stopMicStream,
    stopCurrentPlayback
  ])

  React.useEffect(() => {
    if (!connected) {
      if (wakeArmedRef.current || wakeDetectorRef.current) {
        void stopWakeListeningRef.current("session_close")
      }
      setSessionWakeBehavior(resolvedDefaults.wakeBehavior)
      setAutoCommitEnabled(resolvedDefaults.autoCommitEnabled)
      setVadThreshold(resolvedDefaults.vadThreshold)
      setMinSilenceMs(resolvedDefaults.minSilenceMs)
      setTurnStopSecs(resolvedDefaults.turnStopSecs)
      setMinUtteranceSecs(resolvedDefaults.minUtteranceSecs)
      manualModeRequiredRef.current = false
      setManualModeRequired(false)
      setRecoveryMode("none")
      setListeningRecoveryCount(0)
      setThinkingRecoveryCount(0)
      setActiveToolName("")
      setActiveToolStatus("")
      setWakeRecoveryWarning(null)
      setListeningRecoveryRestartKey(0)
      setThinkingRecoveryArmed(false)
      setThinkingRecoveryRestartKey(0)
      setState("idle")
      pendingBinaryFinishRef.current = false
      pendingResumeRef.current = false
      setPendingStartRequest(false)
      clearAwaitingTtsTimeout()
      clearListeningRecoveryTimeout()
      clearThinkingRecoveryTimeout()
      stopMicStream()
      stopCurrentPlayback()
    }
  }, [
    clearAwaitingTtsTimeout,
    clearListeningRecoveryTimeout,
    clearThinkingRecoveryTimeout,
    connected,
    resolvedDefaults.autoCommitEnabled,
    resolvedDefaults.minSilenceMs,
    resolvedDefaults.minUtteranceSecs,
    resolvedDefaults.turnStopSecs,
    resolvedDefaults.vadThreshold,
    resolvedDefaults.wakeBehavior,
    setWakeRecoveryWarning,
    stopMicStream,
    stopCurrentPlayback
  ])

  React.useEffect(() => {
    const normalizedHeardText = String(heardText || "").trim()
    if (state !== "listening" || !normalizedHeardText) {
      clearListeningRecoveryTimeout()
      setRecoveryMode((current) => (current === "listening_stuck" ? "none" : current))
      return
    }
    setRecoveryMode((current) => (current === "listening_stuck" ? "none" : current))
    clearListeningRecoveryTimeout()
    if (typeof window === "undefined") return
    listeningRecoveryTimeoutRef.current = window.setTimeout(() => {
      setListeningRecoveryCount((current) => current + 1)
      setRecoveryMode("listening_stuck")
    }, LISTENING_RECOVERY_TIMEOUT_MS)
    return () => {
      clearListeningRecoveryTimeout()
    }
  }, [clearListeningRecoveryTimeout, heardText, listeningRecoveryRestartKey, state])

  React.useEffect(() => {
    if (state !== "thinking") {
      clearThinkingRecoveryTimeout()
      setRecoveryMode((current) => (current === "thinking_stuck" ? "none" : current))
      return
    }
    if (!thinkingRecoveryArmed) {
      clearThinkingRecoveryTimeout()
      return
    }
    clearThinkingRecoveryTimeout()
    if (typeof window === "undefined") return
    thinkingRecoveryTimeoutRef.current = window.setTimeout(() => {
      setThinkingRecoveryCount((current) => current + 1)
      setRecoveryMode("thinking_stuck")
      setThinkingRecoveryArmed(false)
    }, THINKING_RECOVERY_TIMEOUT_MS)
    return () => {
      clearThinkingRecoveryTimeout()
    }
  }, [clearThinkingRecoveryTimeout, state, thinkingRecoveryArmed, thinkingRecoveryRestartKey])

  React.useEffect(() => {
    if (!connected || !sessionId || !ws || ws.readyState !== WebSocket.OPEN) return
    try {
      ws.send(
        JSON.stringify({
          type: "voice_config",
          session_id: sessionId,
          voice: {
            trigger_phrases: resolvedDefaults.voiceChatTriggerPhrases,
            auto_resume: sessionAutoResume,
            barge_in: sessionBargeIn,
            wake_behavior: sessionWakeBehavior
          },
          stt: {
            language: resolvedDefaults.sttLanguage,
            model: resolvedDefaults.sttModel,
            enable_vad: autoCommitEnabled,
            vad_threshold: vadThreshold,
            min_silence_ms: minSilenceMs,
            turn_stop_secs: turnStopSecs,
            min_utterance_secs: minUtteranceSecs
          },
          tts: {
            provider: resolvedDefaults.ttsProvider,
            voice: resolvedDefaults.ttsVoice
          }
        })
      )
    } catch {
      // Ignore transient websocket send errors; the next runtime change will retry.
    }
  }, [
    connected,
    resolvedDefaults.sttLanguage,
    resolvedDefaults.sttModel,
    resolvedDefaults.ttsProvider,
    resolvedDefaults.ttsVoice,
    resolvedDefaults.voiceChatTriggerPhrases,
    autoCommitEnabled,
    vadThreshold,
    minSilenceMs,
    turnStopSecs,
    minUtteranceSecs,
    sessionAutoResume,
    sessionBargeIn,
    sessionWakeBehavior,
    sessionId,
    ws
  ])

  React.useEffect(() => {
    return () => {
      clearAwaitingTtsTimeout()
      clearListeningRecoveryTimeout()
      clearThinkingRecoveryTimeout()
      setPendingStartRequest(false)
      pendingResumeRef.current = false
      stopMicStream()
      stopCurrentPlayback()
    }
  }, [clearAwaitingTtsTimeout, clearListeningRecoveryTimeout, clearThinkingRecoveryTimeout, stopMicStream, stopCurrentPlayback])

  React.useEffect(() => {
    return () => {
      if (wakeArmedRef.current) {
        void stopWakeListeningRef.current("route_leave")
      }
    }
  }, [])

  const handlePayload = React.useCallback(
    (payload: PersonaLiveVoicePayload) => {
      const eventType = String(payload?.event || payload?.type || "").trim().toLowerCase()
      if (!eventType) return

      if (eventType === "assistant_delta") {
        const text = String(payload?.text_delta || "").trim()
        if (!text) return
        clearAwaitingTtsTimeout()
        setActiveToolName("")
        setActiveToolStatus("")
        clearThinkingRecovery()
        if (textOnlyDueToTtsFailure) {
          finishVoiceTurn()
          return
        }
        if (activeProvider === "browser") {
          playBrowserSpeech(text)
          return
        }
        if (typeof window !== "undefined") {
          awaitingTtsTimeoutRef.current = window.setTimeout(() => {
            finishVoiceTurn()
          }, 1200)
        }
        setState("thinking")
        return
      }

      if (eventType === "partial_transcript") {
        const delta = String(payload?.text_delta || "").trim()
        if (!delta) return
        setHeardText((current) => {
          const next = current ? `${current} ${delta}` : delta
          heardTranscriptRef.current = next
          return next
        })
        return
      }

      if (eventType === "tool_plan") {
        clearThinkingRecovery()
        return
      }

      if (eventType === "tool_call") {
        const toolName = getPayloadToolName(payload)
        setActiveToolName(toolName)
        setActiveToolStatus(formatActiveToolStatus(toolName, payload?.why))
        if (state === "thinking") {
          armThinkingRecovery()
        }
        return
      }

      if (eventType === "tool_result") {
        setActiveToolName("")
        setActiveToolStatus("")
        if (payload?.approval && typeof payload.approval === "object") {
          clearThinkingRecovery()
          return
        }
        if (state === "thinking") {
          armThinkingRecovery()
          return
        }
        clearThinkingRecovery()
        return
      }

      if (eventType === "tts_audio") {
        clearAwaitingTtsTimeout()
        setActiveToolName("")
        setActiveToolStatus("")
        clearThinkingRecovery()
        const chunkIndex =
          typeof payload?.chunk_index === "number"
            ? payload.chunk_index
            : Number.parseInt(String(payload?.chunk_index ?? "0"), 10)
        const chunkCount =
          typeof payload?.chunk_count === "number"
            ? payload.chunk_count
            : Number.parseInt(String(payload?.chunk_count ?? "1"), 10)
        const audioFormat = String(payload?.audio_format || "mp3")
        if (chunkIndex <= 0) {
          audioStart(audioFormat, true)
          setState("speaking")
        }
        pendingBinaryFinishRef.current = chunkIndex >= chunkCount - 1
        return
      }

      if (eventType === "notice") {
        const reasonCode = String(payload?.reason_code || "").trim().toUpperCase()
        if (reasonCode === "WAKE_ACTIVATION_ACCEPTED") {
          setWakeRecoveryWarning(null)
          return
        }
        if (reasonCode === "WAKE_ACTIVATION_REJECTED") {
          wakeActiveRef.current = false
          setWakeRecoveryWarning(
            formatWakeActivationRejectedMessage(payload),
            getWakeActivationRejectedReasonCode(payload)
          )
          if (wakeArmedRef.current) {
            void startWakeListening({ preserveWarning: true })
          }
          return
        }
        if (reasonCode === "WAKE_DEACTIVATED") {
          wakeActiveRef.current = false
          return
        }
        if (reasonCode === "VOICE_TURN_PROCESSING") {
          if (state === "thinking") {
            armThinkingRecovery()
          }
          return
        }
        if (reasonCode === "VOICE_TOOL_EXECUTION_PROCESSING") {
          if (state === "thinking" && String(activeToolStatus || "").trim()) {
            armThinkingRecovery()
          }
          return
        }
        if (reasonCode === "TTS_UNAVAILABLE_TEXT_ONLY") {
          clearAwaitingTtsTimeout()
          setActiveToolName("")
          setActiveToolStatus("")
          clearThinkingRecovery()
          textOnlyDueToTtsFailureRef.current = true
          setTextOnlyDueToTtsFailure(true)
          setVoiceWarning(
            String(payload?.message || "Live TTS is unavailable. Continuing in text-only mode."),
            "voice_tts_unavailable_text_only"
          )
          finishVoiceTurn()
          return
        }
        if (reasonCode === "VOICE_MANUAL_MODE_REQUIRED") {
          manualModeRequiredRef.current = true
          setManualModeRequired(true)
          setVoiceWarning(
            String(
              payload?.message ||
                "Server VAD unavailable for this live session. Use Send now to commit heard speech manually."
            ),
            "voice_manual_mode_required"
          )
          return
        }
        if (reasonCode === "VOICE_TURN_COMMITTED") {
          clearAwaitingTtsTimeout()
          clearListeningRecoveryTimeout()
          if (micActive) {
            stopMicStream()
          }
          const committedTranscript = String(payload?.transcript || "").trim()
          if (committedTranscript) {
            setLastCommittedText(committedTranscript)
          }
          setActiveToolName("")
          setActiveToolStatus("")
          if (!manualModeRequiredRef.current && !textOnlyDueToTtsFailureRef.current) {
            setVoiceWarning(null)
          }
          setRecoveryMode("none")
          armThinkingRecovery()
          setState("thinking")
          return
        }
        if (reasonCode === "VOICE_COMMIT_IGNORED_ALREADY_COMMITTED") {
          setActiveToolName("")
          setActiveToolStatus("")
          setVoiceWarning(
            String(payload?.message || "This utterance was already committed."),
            "voice_commit_ignored_already_committed"
          )
          setState("thinking")
          return
        }
        if (reasonCode === "VOICE_TRIGGER_NOT_HEARD") {
          setActiveToolName("")
          setActiveToolStatus("")
          setHeardText("")
          heardTranscriptRef.current = ""
          setVoiceWarning(
            String(payload?.message || "No trigger phrase was heard, so the transcript was ignored."),
            "voice_trigger_not_heard"
          )
          if (wakeArmedRef.current) {
            wakeActiveRef.current = false
            void startWakeListening()
          }
          setState(micActive ? "listening" : "idle")
          return
        }
        if (reasonCode === "VOICE_EMPTY_COMMAND_AFTER_TRIGGER") {
          setActiveToolName("")
          setActiveToolStatus("")
          setHeardText("")
          heardTranscriptRef.current = ""
          setVoiceWarning(
            String(
              payload?.message ||
                "The trigger phrase was removed, but no spoken command remained."
            ),
            "voice_empty_command_after_trigger"
          )
          if (wakeArmedRef.current) {
            wakeActiveRef.current = false
            void startWakeListening()
          }
          setState(micActive ? "listening" : "idle")
          return
        }
        if (reasonCode === "TRANSCRIPT_REQUIRED") {
          setActiveToolName("")
          setActiveToolStatus("")
          setVoiceWarning(
            "No speech transcript was captured for that live turn.",
            "voice_no_transcript"
          )
          setState(micActive ? "listening" : "idle")
        }
      }
    },
    [
      activeProvider,
      armThinkingRecovery,
      audioStart,
      clearAwaitingTtsTimeout,
      clearThinkingRecovery,
      clearListeningRecoveryTimeout,
      finishVoiceTurn,
      micActive,
      playBrowserSpeech,
      activeToolStatus,
      setVoiceWarning,
      setWakeRecoveryWarning,
      startWakeListening,
      state,
      stopMicStream,
      textOnlyDueToTtsFailure
    ]
  )

  const handleBinaryPayload = React.useCallback(
    (data: ArrayBuffer) => {
      if (!(data instanceof ArrayBuffer)) return
      audioAppend(data)
      if (pendingBinaryFinishRef.current) {
        pendingBinaryFinishRef.current = false
        pendingResumeRef.current = true
        audioFinish()
      }
    },
    [audioAppend, audioFinish]
  )

  return {
    state,
    recoveryMode,
    listeningRecoveryCount,
    thinkingRecoveryCount,
    heardText,
    lastCommittedText,
    activeToolName,
    activeToolStatus,
    warning,
    warningReasonCode,
    wakeArmed,
    wakeDetectorState,
    wakeWarning,
    wakeWarningReasonCode,
    sessionWakeBehavior,
    wakeTriggerPhrases,
    manualModeRequired,
    canSendNow: Boolean(String(heardText || heardTranscriptRef.current || "").trim()),
    speechAvailable: canUseServerStt,
    isListening: micActive || pendingStartRequest,
    sessionAutoResume,
    sessionBargeIn,
    autoCommitEnabled,
    vadPreset,
    vadThreshold,
    minSilenceMs,
    turnStopSecs,
    minUtteranceSecs,
    textOnlyDueToTtsFailure,
    startListening,
    stopListening,
    toggleListening,
    toggleWakeArmed,
    stopWakeListening,
    sendCurrentTranscriptNow,
    keepListening,
    waitOnRecovery,
    resetTurn,
    setSessionAutoResume,
    setSessionBargeIn,
    setSessionWakeBehavior,
    setAutoCommitEnabled,
    setVadPreset,
    setVadThreshold,
    setMinSilenceMs,
    setTurnStopSecs,
    setMinUtteranceSecs,
    handlePayload,
    handleBinaryPayload
  }
}
