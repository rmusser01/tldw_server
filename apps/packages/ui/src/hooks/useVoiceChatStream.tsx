import React from "react"
import { useStorage } from "@plasmohq/storage/hook"
import { useAudioSourcePreferences } from "@/hooks/useAudioSourcePreferences"
import { useAudioSourceCatalog } from "@/hooks/useAudioSourceCatalog"
import { useSttSettings } from "@/hooks/useSttSettings"
import { useVoiceChatSettings } from "@/hooks/useVoiceChatSettings"
import { useSelectedModel } from "@/hooks/chat/useSelectedModel"
import { useMicStream } from "@/hooks/useMicStream"
import {
  buildVoiceConversationPreflight,
  normalizeVoiceConversationRuntimeError
} from "@/services/tldw/voice-conversation"
import { arrayBufferToBase64 } from "@/utils/compress"
import {
  DEFAULT_TLDW_TTS_MODEL,
  DEFAULT_TLDW_TTS_VOICE,
  DEFAULT_TTS_PROVIDER
} from "@/services/tts"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { getRuntimeSingleUserApiKeyOverride } from "@/services/tldw/runtime-auth-override"
import { resolveApiProviderForModel } from "@/utils/resolve-api-provider"
import { useStreamingAudioPlayer } from "@/hooks/useStreamingAudioPlayer"

export type VoiceChatState =
  | "idle"
  | "connecting"
  | "listening"
  | "thinking"
  | "speaking"
  | "error"

export type VoiceChatStreamMode = "voice_chat" | "push_to_talk"

type VoiceChatCallbacks = {
  onPartial?: (text: string) => void
  onTranscript?: (text: string, meta?: { autoCommit?: boolean }) => void
  onAssistantDelta?: (delta: string) => void
  onAssistantMessage?: (text: string) => void
  onError?: (message: string) => void
  onWarning?: (message: string) => void
  onStateChange?: (state: VoiceChatState) => void
}

type VoiceChatOptions = VoiceChatCallbacks & {
  active: boolean
  mode?: VoiceChatStreamMode
}

// Drop mic frames once the socket's outbound buffer backs up past this many
// bytes so a slow uplink can't grow the buffer until the browser force-closes.
const MAX_WS_BUFFERED_BYTES = 512 * 1024
// Abort the handshake if `onopen` never fires so the UI can recover instead of
// wedging in "connecting". Mirrors the extension STT worker connect timer.
const VOICE_WS_CONNECT_TIMEOUT_MS = 10000

const formatToMime = (format: string): string => {
  const normalized = String(format || "").trim().toLowerCase()
  switch (normalized) {
    case "wav":
      return "audio/wav"
    case "opus":
      return "audio/opus"
    case "aac":
      return "audio/aac"
    case "flac":
      return "audio/flac"
    case "ogg":
      return "audio/ogg"
    case "webm":
      return "audio/webm"
    case "ulaw":
      return "audio/basic"
    case "pcm":
      return "audio/L16"
    case "mp3":
    default:
      return "audio/mpeg"
  }
}

const isPlayableFormat = (format: string): boolean => {
  if (typeof Audio === "undefined") return false
  try {
    const probe = new Audio()
    return Boolean(probe.canPlayType(formatToMime(format)))
  } catch {
    return false
  }
}

const normalizeTriggerList = (phrases: string[]) =>
  phrases.map((phrase) => phrase.trim()).filter(Boolean)

const stripTriggerPhrases = (text: string, phrases: string[]): string => {
  if (!phrases.length) return text
  let next = text
  for (const phrase of phrases) {
    if (!phrase) continue
    const escaped = phrase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
    const pattern = new RegExp(`\\b${escaped}\\b`, "ig")
    next = next.replace(pattern, " ")
  }
  return next.replace(/\s+/g, " ").trim()
}

const detectTriggerPhrase = (text: string, phrases: string[]): boolean => {
  if (!phrases.length) return false
  const lower = text.toLowerCase()
  return phrases.some((phrase) => lower.includes(phrase.toLowerCase()))
}

export const useVoiceChatStream = ({
  active,
  mode = "voice_chat",
  onPartial,
  onTranscript,
  onAssistantDelta,
  onAssistantMessage,
  onError,
  onWarning,
  onStateChange
}: VoiceChatOptions) => {
  const sttSettings = useSttSettings()
  const {
    voiceChatModel,
    voiceChatPauseMs,
    voiceChatTriggerPhrases,
    voiceChatAutoResume,
    voiceChatBargeIn,
    voiceChatTtsMode
  } = useVoiceChatSettings()
  const [speechToTextLanguage] = useStorage("speechToTextLanguage", "en-US")
  const { selectedModel } = useSelectedModel()
  const {
    preference: liveVoiceSourcePreference,
    isLoading: liveVoiceSourceLoading
  } =
    useAudioSourcePreferences("live_voice")
  const {
    devices: audioInputDevices,
    isSettled: hasAudioCatalogSettled
  } = useAudioSourceCatalog()

  const [ttsProvider] = useStorage("ttsProvider", DEFAULT_TTS_PROVIDER)
  const [tldwTtsModel] = useStorage("tldwTtsModel", DEFAULT_TLDW_TTS_MODEL)
  const [tldwTtsVoice] = useStorage("tldwTtsVoice", DEFAULT_TLDW_TTS_VOICE)
  const [tldwTtsResponseFormat] = useStorage("tldwTtsResponseFormat", "mp3")
  const [tldwTtsSpeed] = useStorage("tldwTtsSpeed", 1)
  const [openAITTSModel] = useStorage("openAITTSModel", "tts-1")
  const [openAITTSVoice] = useStorage("openAITTSVoice", "alloy")
  const [elevenLabsModel] = useStorage("elevenLabsModel", "")
  const [elevenLabsVoiceId] = useStorage("elevenLabsVoiceId", "")
  const [speechPlaybackSpeed] = useStorage("speechPlaybackSpeed", 1)

  const wsRef = React.useRef<WebSocket | null>(null)
  const connectingRef = React.useRef(false)
  const closingRef = React.useRef(false)
  const triggeredRef = React.useRef(false)
  const bargeInSentRef = React.useRef(false)
  const pendingResumeRef = React.useRef(false)
  const resolvedTtsFormatRef = React.useRef<string | null>(null)
  const errorRef = React.useRef<string | null>(null)
  const startAttemptRef = React.useRef(0)
  const connectTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const didRunActiveEffectRef = React.useRef(false)
  const manualSessionRef = React.useRef(false)

  const [state, setState] = React.useState<VoiceChatState>("idle")
  const [error, setError] = React.useState<string | null>(null)
  const [connected, setConnected] = React.useState(false)

  const callbacksRef = React.useRef({
    onPartial,
    onTranscript,
    onAssistantDelta,
    onAssistantMessage,
    onError,
    onWarning,
    onStateChange
  })
  const stateRef = React.useRef<VoiceChatState>(state)
  const activeRef = React.useRef(active)
  const streamMode: VoiceChatStreamMode =
    mode === "push_to_talk" ? "push_to_talk" : "voice_chat"

  React.useEffect(() => {
    callbacksRef.current = {
      onPartial,
      onTranscript,
      onAssistantDelta,
      onAssistantMessage,
      onError,
      onWarning,
      onStateChange
    }
  }, [
    onPartial,
    onTranscript,
    onAssistantDelta,
    onAssistantMessage,
    onError,
    onWarning,
    onStateChange
  ])

  React.useEffect(() => {
    stateRef.current = state
  }, [state])

  React.useEffect(() => {
    activeRef.current = active
  }, [active])

  const updateState = React.useCallback((next: VoiceChatState) => {
    setState(next)
    callbacksRef.current.onStateChange?.(next)
  }, [])

  const isStartAttemptCurrent = React.useCallback(
    (attemptId: number) => startAttemptRef.current === attemptId,
    []
  )

  const {
    start: audioStart,
    append: audioAppend,
    finish: audioFinish,
    stop: audioStop,
    state: audioState
  } = useStreamingAudioPlayer()

  const { start: micStart, stop: micStop, active: micActive } = useMicStream(
    (chunk) => {
      const ws = wsRef.current
      if (!ws || ws.readyState !== WebSocket.OPEN) return
      try {
        if (voiceChatBargeIn && stateRef.current === "speaking") {
          // Send the barge-in interrupt once per speaking turn instead of on
          // every mic frame (~4/sec). The guard is cleared on tts_start and
          // when the server acks the interruption.
          if (!bargeInSentRef.current) {
            bargeInSentRef.current = true
            ws.send(JSON.stringify({ type: "interrupt", reason: "barge_in" }))
          }
        }
        // Backpressure: drop this mic frame when the outbound buffer is backed
        // up rather than growing it unbounded on a slow uplink.
        if (
          typeof ws.bufferedAmount === "number" &&
          ws.bufferedAmount > MAX_WS_BUFFERED_BYTES
        ) {
          return
        }
        const base64 = arrayBufferToBase64(chunk)
        ws.send(JSON.stringify({ type: "audio", data: base64 }))
      } catch {
        // ignore chunk send failures
      }
    },
    { owner: streamMode === "push_to_talk" ? "push_to_talk" : "voice_chat" }
  )

  const normalizedTriggers = React.useMemo(
    () => normalizeTriggerList(voiceChatTriggerPhrases),
    [voiceChatTriggerPhrases]
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

  const cleanupSession = React.useCallback(() => {
    manualSessionRef.current = false
    startAttemptRef.current += 1
    if (connectTimeoutRef.current) {
      clearTimeout(connectTimeoutRef.current)
      connectTimeoutRef.current = null
    }
    try {
      micStop()
    } catch {}
    try {
      audioStop()
    } catch {}
    wsRef.current = null
    connectingRef.current = false
    triggeredRef.current = false
    bargeInSentRef.current = false
    pendingResumeRef.current = false
    resolvedTtsFormatRef.current = null
    setConnected(false)
  }, [audioStop, micStop])

  const stop = React.useCallback(() => {
    manualSessionRef.current = false
    const ws = wsRef.current
    startAttemptRef.current += 1
    closingRef.current = Boolean(ws)
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify({ type: "stop" }))
      } catch {}
    }
    // Detach handlers before closing so a late onclose/onerror can't fire state
    // updates after the session (or the component) has gone away.
    if (ws) {
      ws.onopen = null
      ws.onmessage = null
      ws.onerror = null
      ws.onclose = null
    }
    try {
      ws?.close()
    } catch {}
    cleanupSession()
    if (!ws) {
      closingRef.current = false
    }
  }, [cleanupSession])

  const handleError = React.useCallback(
    (message: string) => {
      errorRef.current = message
      setError(message)
      callbacksRef.current.onError?.(message)
      updateState("error")
      stop()
    },
    [stop, updateState]
  )

  const normalizeStartupErrorMessage = React.useCallback((error: unknown): string => {
    const message =
      error instanceof Error ? String(error.message || "").trim() : String(error || "").trim()
    if (message === "Voice conversation TTS configuration is incomplete") {
      return "Voice conversation needs a server TTS model and voice."
    }
    return message || "Voice chat failed to start"
  }, [])

  const sendCommit = React.useCallback(() => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    try {
      ws.send(JSON.stringify({ type: "commit" }))
    } catch {}
  }, [])

  const sendPushToTalkRelease = React.useCallback(() => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    try {
      ws.send(JSON.stringify({ type: "push_to_talk_release" }))
    } catch {}
  }, [])

  const handleMessage = React.useCallback(
    (data: any) => {
      if (!data || typeof data !== "object") return
      const callbacks = callbacksRef.current
      const msgType = String(data.type || "").toLowerCase()

      if (msgType === "partial") {
        const text = String(data.text || "")
        callbacks.onPartial?.(text)
        if (!triggeredRef.current && detectTriggerPhrase(text, normalizedTriggers)) {
          triggeredRef.current = true
          sendCommit()
        }
        return
      }

      if (msgType === "full_transcript") {
        const raw = String(data.text || "")
        const cleaned = stripTriggerPhrases(raw, normalizedTriggers)
        triggeredRef.current = false
        if (!cleaned) {
          updateState("listening")
          return
        }
        callbacks.onTranscript?.(cleaned, { autoCommit: Boolean(data.auto_commit) })
        updateState("thinking")
        if (!voiceChatBargeIn) {
          try {
            micStop()
          } catch {}
        }
        return
      }

      if (msgType === "llm_delta") {
        const delta = String(data.delta || "")
        if (delta) {
          callbacks.onAssistantDelta?.(delta)
        }
        if (stateRef.current !== "speaking") {
          updateState("thinking")
        }
        return
      }

      if (msgType === "llm_message") {
        const text = String(data.text || "")
        callbacks.onAssistantMessage?.(text)
        return
      }

      if (msgType === "tts_start") {
        const format = String(data.format || resolvedTtsFormatRef.current || "mp3")
        // New speaking turn: allow one barge-in interrupt for this turn.
        bargeInSentRef.current = false
        audioStart(format, voiceChatTtsMode === "stream")
        updateState("speaking")
        if (!voiceChatBargeIn) {
          try {
            micStop()
          } catch {}
        }
        return
      }

      if (msgType === "tts_done") {
        audioFinish()
        pendingResumeRef.current = true
        return
      }

      if (msgType === "interrupted") {
        pendingResumeRef.current = false
        bargeInSentRef.current = false
        // Barge-in: stop the already-buffered assistant audio immediately so it
        // does not keep playing over the user.
        try {
          audioStop()
        } catch {}
        updateState(activeRef.current ? "listening" : "idle")
        return
      }

      if (msgType === "warning") {
        const message = String(data.message || "")
        if (message) callbacks.onWarning?.(message)
        return
      }

      if (msgType === "error") {
        const message = String(data.message || "Streaming error")
        handleError(message)
        return
      }
    },
    [
      audioFinish,
      audioStart,
      audioStop,
      handleError,
      micStop,
      normalizedTriggers,
      sendCommit,
      updateState,
      voiceChatBargeIn,
      voiceChatTtsMode
    ]
  )

  const start = React.useCallback(async () => {
    if (!liveVoiceSourceReady) return
    if (connectingRef.current || wsRef.current) return
    manualSessionRef.current = !activeRef.current
    const attemptId = startAttemptRef.current + 1
    startAttemptRef.current = attemptId
    connectingRef.current = true
    errorRef.current = null
    setError(null)
    updateState("connecting")

    try {
      const config = await tldwClient.getConfig()
      const token =
        config?.authMode === "multi-user"
          ? String(config?.accessToken || "").trim()
          : String(
              getRuntimeSingleUserApiKeyOverride() || config?.apiKey || ""
            ).trim()
      const requestedModel = String(voiceChatModel || selectedModel || "").trim()
      const preflight = await buildVoiceConversationPreflight({
        serverUrl: String(config?.serverUrl || ""),
        token,
        authMode: config?.authMode,
        authSource: config?.authSource,
        requestedModel,
        ttsProvider,
        tldwTtsModel,
        tldwTtsVoice,
        tldwTtsSpeed,
        tldwTtsResponseFormat,
        openAITTSModel,
        openAITTSVoice,
        elevenLabsModel,
        elevenLabsVoiceId,
        speechPlaybackSpeed,
        voiceChatTtsMode,
        resolveProvider: ({ modelId }) =>
          resolveApiProviderForModel({
            modelId
          })
      })
      if (!isStartAttemptCurrent(attemptId)) {
        return
      }
      resolvedTtsFormatRef.current = isPlayableFormat(preflight.tts.format)
        ? preflight.tts.format
        : "mp3"

      const ws = new WebSocket(preflight.websocketUrl)
      ws.binaryType = "arraybuffer"
      wsRef.current = ws

      // Handshake timeout: if onopen never fires the UI would otherwise wedge in
      // "connecting" forever. Fail fast so the restart guard is released.
      if (connectTimeoutRef.current) {
        clearTimeout(connectTimeoutRef.current)
      }
      connectTimeoutRef.current = setTimeout(() => {
        connectTimeoutRef.current = null
        if (!isStartAttemptCurrent(attemptId)) return
        if (ws.readyState === WebSocket.OPEN) return
        handleError("Voice chat connection timed out")
      }, VOICE_WS_CONNECT_TIMEOUT_MS)

      ws.onmessage = (event) => {
        if (typeof event.data === "string") {
          try {
            const payload = JSON.parse(event.data)
            handleMessage(payload)
          } catch {}
        } else if (event.data instanceof ArrayBuffer) {
          audioAppend(event.data)
        }
      }

      ws.onerror = () => {
        handleError("Voice chat websocket error")
      }

      ws.onclose = () => {
        const wasClosing = closingRef.current
        closingRef.current = false
        cleanupSession()
        if (!wasClosing && activeRef.current && !errorRef.current) {
          const message = normalizeVoiceConversationRuntimeError(
            new Error("websocket disconnected")
          ).message
          errorRef.current = message
          setError(message)
          callbacksRef.current.onError?.(message)
          updateState("error")
          return
        }
        // Don't clobber a surfaced error state by resetting to idle on close.
        if (!errorRef.current) {
          updateState("idle")
        }
      }

      ws.onopen = () => {
        if (connectTimeoutRef.current) {
          clearTimeout(connectTimeoutRef.current)
          connectTimeoutRef.current = null
        }
        void (async () => {
          try {
            if (!isStartAttemptCurrent(attemptId)) {
              try {
                ws.close()
              } catch {}
              return
            }
            // TASK-12106: authenticate before any config/audio so the server
            // authorizes the connection first. The audio WS endpoint reads an
            // {type:"auth", token} first frame via receive_text()
            // (streaming_service.py:641-647 multi-user / 720-723 single-user).
            // `token` is the JWT (multi-user) or API key (single-user).
            if (token) {
              ws.send(JSON.stringify({ type: "auth", token }))
            }
            const sttConfig: Record<string, any> = {
              enable_vad: true,
              min_silence_ms: voiceChatPauseMs,
              language: speechToTextLanguage
            }
            const modelValue = String(sttSettings.model || "").trim()
            if (modelValue) {
              sttConfig.model = modelValue
            }
            if (sttSettings.timestampGranularities) {
              sttConfig.timestamp_granularities =
                sttSettings.timestampGranularities
            }
            if (sttSettings.prompt && sttSettings.prompt.trim().length > 0) {
              sttConfig.prompt = sttSettings.prompt.trim()
            }
            if (sttSettings.task) {
              sttConfig.task = sttSettings.task
            }
            if (sttSettings.responseFormat) {
              sttConfig.response_format = sttSettings.responseFormat
            }
            if (typeof sttSettings.temperature === "number") {
              sttConfig.temperature = sttSettings.temperature
            }
            if (sttSettings.useSegmentation) {
              sttConfig.segment = true
              if (typeof sttSettings.segK === "number") {
                sttConfig.seg_K = sttSettings.segK
              }
              if (typeof sttSettings.segMinSegmentSize === "number") {
                sttConfig.seg_min_segment_size = sttSettings.segMinSegmentSize
              }
              if (typeof sttSettings.segLambdaBalance === "number") {
                sttConfig.seg_lambda_balance = sttSettings.segLambdaBalance
              }
              if (typeof sttSettings.segUtteranceExpansionWidth === "number") {
                sttConfig.seg_utterance_expansion_width =
                  sttSettings.segUtteranceExpansionWidth
              }
              if (sttSettings.segEmbeddingsProvider?.trim()) {
                sttConfig.seg_embeddings_provider =
                  sttSettings.segEmbeddingsProvider.trim()
              }
              if (sttSettings.segEmbeddingsModel?.trim()) {
                sttConfig.seg_embeddings_model =
                  sttSettings.segEmbeddingsModel.trim()
              }
            }

            ws.send(
              JSON.stringify({
                type: "config",
                protocol_version: 1,
                mode: streamMode,
                audio_format: "pcm16",
                sample_rate: 16000,
                channels: 1,
                stt: sttConfig,
                llm: preflight.llm,
                tts: preflight.tts
              })
            )

            if (!isStartAttemptCurrent(attemptId)) {
              try {
                ws.close()
              } catch {}
              return
            }

            await micStart({ deviceId: liveVoiceDeviceId })
            if (!isStartAttemptCurrent(attemptId)) {
              try {
                micStop()
              } catch {}
              try {
                ws.close()
              } catch {}
              return
            }
            setConnected(true)
            connectingRef.current = false
            updateState("listening")
          } catch (err: any) {
            handleError(normalizeStartupErrorMessage(err))
          }
        })()
      }
    } catch (err: any) {
      handleError(normalizeStartupErrorMessage(err))
    }
  }, [
    audioAppend,
    cleanupSession,
    handleError,
    handleMessage,
    isStartAttemptCurrent,
    micStart,
    micStop,
    normalizeStartupErrorMessage,
    normalizeVoiceConversationRuntimeError,
    ttsProvider,
    tldwTtsModel,
    tldwTtsVoice,
    tldwTtsSpeed,
    tldwTtsResponseFormat,
    openAITTSModel,
    openAITTSVoice,
    elevenLabsModel,
    elevenLabsVoiceId,
    speechPlaybackSpeed,
    resolveApiProviderForModel,
    selectedModel,
    speechToTextLanguage,
    streamMode,
    liveVoiceDeviceId,
    sttSettings.model,
    sttSettings.temperature,
    sttSettings.task,
    sttSettings.responseFormat,
    sttSettings.timestampGranularities,
    sttSettings.prompt,
    sttSettings.useSegmentation,
    sttSettings.segK,
    sttSettings.segMinSegmentSize,
    sttSettings.segLambdaBalance,
    sttSettings.segUtteranceExpansionWidth,
    sttSettings.segEmbeddingsProvider,
    sttSettings.segEmbeddingsModel,
    liveVoiceSourceReady,
    updateState,
    voiceChatModel,
    voiceChatPauseMs,
    voiceChatTtsMode
  ])

  React.useEffect(() => {
    if (!didRunActiveEffectRef.current) {
      didRunActiveEffectRef.current = true
      if (!active) {
        errorRef.current = null
        setError(null)
        updateState("idle")
        return
      }
    }

    if (active) {
      manualSessionRef.current = false
      if (!liveVoiceSourceReady) return
      void start()
      return
    }
    if (manualSessionRef.current) {
      return
    }
    if (
      wsRef.current ||
      connectingRef.current ||
      stateRef.current !== "idle" ||
      errorRef.current
    ) {
      stop()
    }
    errorRef.current = null
    setError(null)
    updateState("idle")
  }, [active, liveVoiceSourceReady, start, stop, updateState])

  React.useEffect(() => {
    if (!pendingResumeRef.current) return
    if (audioState.playing) return
    if (!active) {
      pendingResumeRef.current = false
      updateState("idle")
      return
    }
    if (!liveVoiceSourceReady) {
      return
    }
    if (voiceChatAutoResume) {
      pendingResumeRef.current = false
      if (!voiceChatBargeIn) {
        void micStart({ deviceId: liveVoiceDeviceId })
          .then(() => updateState("listening"))
          .catch(() => {
            handleError("Unable to restart microphone")
          })
      } else {
        updateState("listening")
      }
    } else {
      pendingResumeRef.current = false
      updateState("idle")
    }
  }, [
    active,
    audioState.playing,
    handleError,
    liveVoiceDeviceId,
    liveVoiceSourceReady,
    micStart,
    updateState,
    voiceChatAutoResume,
    voiceChatBargeIn
  ])

  React.useEffect(() => {
    return () => {
      stop()
    }
  }, [stop])

  return {
    state,
    error,
    connected,
    micActive,
    start,
    stop,
    sendCommit,
    sendPushToTalkRelease
  }
}
