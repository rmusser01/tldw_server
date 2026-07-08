import React from "react"
import { useTranslation } from "react-i18next"
import type { AudioCaptureRequestedSource } from "@/audio"
import { useMicStream } from "@/hooks/useMicStream"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import type { SttSettings } from "@/hooks/useSttSettings"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { arrayBufferToBase64 } from "@/utils/compress"

export interface UseServerDictationOptions {
  canUseServerStt: boolean
  speechToTextLanguage: string
  sttSettings: SttSettings
  onTranscript: (text: string) => void
  onPartialTranscript?: (text: string) => void
  onError?: (error: unknown) => void
  onSuccess?: () => void
}

export interface UseServerDictationResult {
  isServerDictating: boolean
  startServerDictation: (source?: AudioCaptureRequestedSource) => Promise<void>
  stopServerDictation: () => void
}

const buildTranscribeWebSocketUrl = (serverUrl: string): string =>
  `${serverUrl.replace(/^http/i, "ws").replace(/\/$/, "")}/api/v1/audio/stream/transcribe`

const buildSttConfig = (
  sttSettings: SttSettings,
  speechToTextLanguage: string
): Record<string, unknown> => {
  const config: Record<string, unknown> = {
    language: speechToTextLanguage
  }

  if (sttSettings.model && sttSettings.model.trim().length > 0) {
    config.model = sttSettings.model.trim()
  }
  if (sttSettings.timestampGranularities) {
    config.timestamp_granularities = sttSettings.timestampGranularities
  }
  if (sttSettings.prompt && sttSettings.prompt.trim().length > 0) {
    config.prompt = sttSettings.prompt.trim()
  }
  if (sttSettings.task) {
    config.task = sttSettings.task
  }
  if (sttSettings.responseFormat) {
    config.response_format = sttSettings.responseFormat
  }
  if (typeof sttSettings.temperature === "number") {
    config.temperature = sttSettings.temperature
  }
  if (sttSettings.useSegmentation) {
    config.segment = true
    if (typeof sttSettings.segK === "number") {
      config.seg_K = sttSettings.segK
    }
    if (typeof sttSettings.segMinSegmentSize === "number") {
      config.seg_min_segment_size = sttSettings.segMinSegmentSize
    }
    if (typeof sttSettings.segLambdaBalance === "number") {
      config.seg_lambda_balance = sttSettings.segLambdaBalance
    }
    if (typeof sttSettings.segUtteranceExpansionWidth === "number") {
      config.seg_utterance_expansion_width =
        sttSettings.segUtteranceExpansionWidth
    }
    if (sttSettings.segEmbeddingsProvider?.trim()) {
      config.seg_embeddings_provider = sttSettings.segEmbeddingsProvider.trim()
    }
    if (sttSettings.segEmbeddingsModel?.trim()) {
      config.seg_embeddings_model = sttSettings.segEmbeddingsModel.trim()
    }
  }

  return config
}

export const useServerDictation = (
  options: UseServerDictationOptions
): UseServerDictationResult => {
  const { t } = useTranslation(["playground"])
  const notification = useAntdNotification()
  const {
    canUseServerStt,
    speechToTextLanguage,
    sttSettings,
    onTranscript,
    onPartialTranscript,
    onError,
    onSuccess
  } = options

  const wsRef = React.useRef<WebSocket | null>(null)
  const startingRef = React.useRef(false)
  const [isServerDictating, setIsServerDictating] = React.useState(false)

  const reportError = React.useCallback(
    (error: unknown) => {
      try {
        onError?.(error)
      } catch {}
    },
    [onError]
  )

  const notifyError = React.useCallback(
    (description: React.ReactNode) => {
      notification.error({
        message: t("playground:actions.speechErrorTitle", "Dictation failed"),
        description
      })
    },
    [notification, t]
  )

  const { start: micStart, stop: micStop, active: micActive } = useMicStream(
    (chunk) => {
      const ws = wsRef.current
      if (!ws || ws.readyState !== WebSocket.OPEN) return
      try {
        ws.send(
          JSON.stringify({
            type: "audio",
            data: arrayBufferToBase64(chunk)
          })
        )
      } catch {}
    },
    { owner: "dictation" }
  )

  const stopServerDictation = React.useCallback(() => {
    const ws = wsRef.current
    startingRef.current = false
    try {
      if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "stop" }))
      }
    } catch {}
    if (ws) {
      ws.onopen = null
      ws.onmessage = null
      ws.onerror = null
      ws.onclose = null
    }
    try {
      ws?.close()
    } catch {}
    wsRef.current = null
    micStop()
    setIsServerDictating(false)
  }, [micStop])

  const startServerDictation = React.useCallback(async (
    source?: AudioCaptureRequestedSource
  ) => {
    if (startingRef.current) return
    if (isServerDictating || micActive) {
      stopServerDictation()
      return
    }

    if (!canUseServerStt) {
      notification.error({
        message: t(
          "playground:actions.speechUnavailableTitle",
          "Dictation unavailable"
        ),
        description: t(
          "playground:actions.speechUnavailableBody",
          "Connect to a tldw server that exposes the audio transcriptions API to use dictation."
        )
      })
      return
    }

    startingRef.current = true
    const requestedDeviceId =
      source?.sourceKind === "mic_device" ? source.deviceId : null

    try {
      const config = await tldwClient.getConfig()
      const serverUrl = String(config?.serverUrl || "").trim()
      if (!serverUrl) {
        throw new Error("tldw server not configured")
      }
      const token =
        config?.authMode === "multi-user"
          ? String(config?.accessToken || "").trim()
          : String(config?.apiKey || "").trim()

      const ws = new WebSocket(buildTranscribeWebSocketUrl(serverUrl))
      wsRef.current = ws

      ws.onopen = () => {
        void (async () => {
          try {
            if (wsRef.current !== ws) return
            if (token) {
              ws.send(JSON.stringify({ type: "auth", token }))
            }
            ws.send(
              JSON.stringify({
                type: "config",
                protocol_version: 1,
                mode: "dictate",
                audio_format: "pcm16",
                sample_rate: 16000,
                channels: 1,
                ...buildSttConfig(sttSettings, speechToTextLanguage)
              })
            )
            await micStart({ deviceId: requestedDeviceId })
            if (wsRef.current !== ws) {
              micStop()
              return
            }
            setIsServerDictating(true)
          } catch (error) {
            reportError(error)
            notifyError(
              error instanceof Error && error.message
                ? error.message
                : t(
                    "playground:actions.speechMicError",
                    "Unable to access your microphone. Check browser permissions and try again."
                  )
            )
            try {
              ws.close()
            } catch {}
          } finally {
            startingRef.current = false
          }
        })()
      }

      ws.onmessage = (event) => {
        if (typeof event.data !== "string") return
        let payload: any
        try {
          payload = JSON.parse(event.data)
        } catch {
          return
        }
        if (!payload || typeof payload !== "object") return

        const type = String(payload.type || "")
        if (type === "partial") {
          onPartialTranscript?.(String(payload.text || ""))
          return
        }
        if (type === "full_transcript" || type === "transcription") {
          const text = String(payload.text || payload.transcript || "").trim()
          if (text) {
            onTranscript(text)
            onSuccess?.()
          }
          return
        }
        if (type === "error") {
          reportError(payload)
          notifyError(String(payload.message || "Dictation websocket error"))
        }
      }

      ws.onerror = () => {
        const error = new Error("Dictation websocket error")
        reportError(error)
        notifyError(error.message)
        try {
          ws.close()
        } catch {}
      }

      ws.onclose = () => {
        if (wsRef.current === ws) {
          wsRef.current = null
        }
        startingRef.current = false
        micStop()
        setIsServerDictating(false)
      }
    } catch (error) {
      startingRef.current = false
      reportError(error)
      notifyError(
        error instanceof Error && error.message
          ? error.message
          : t(
              "playground:actions.speechErrorBody",
              "Transcription request failed. Check tldw server health."
            )
      )
    }
  }, [
    canUseServerStt,
    isServerDictating,
    micActive,
    micStart,
    micStop,
    notification,
    notifyError,
    onPartialTranscript,
    onSuccess,
    onTranscript,
    reportError,
    speechToTextLanguage,
    sttSettings,
    stopServerDictation,
    t
  ])

  React.useEffect(() => {
    return () => {
      stopServerDictation()
    }
  }, [stopServerDictation])

  return {
    isServerDictating: isServerDictating || micActive,
    startServerDictation,
    stopServerDictation
  }
}
