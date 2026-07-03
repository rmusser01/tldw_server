import React from "react"
import { useTranslation } from "react-i18next"
import type { AudioCaptureRequestedSource } from "@/audio"
import {
  createAudioCaptureSessionCoordinator,
  type AudioCaptureSessionCoordinator
} from "@/audio"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { SttSettings } from "@/hooks/useSttSettings"
import { useAntdNotification } from "@/hooks/useAntdNotification"

export interface UseServerDictationOptions {
  canUseServerStt: boolean
  speechToTextLanguage: string
  sttSettings: SttSettings
  onTranscript: (text: string) => void
  onError?: (error: unknown) => void
  onSuccess?: () => void
}

export interface UseServerDictationResult {
  isServerDictating: boolean
  startServerDictation: (source?: AudioCaptureRequestedSource) => Promise<void>
  stopServerDictation: () => void
}

const AUDIO_CAPTURE_COORDINATOR_KEY = Symbol.for(
  "tldw.audioCaptureSessionCoordinator"
)

const getAudioCaptureSessionCoordinator = (): AudioCaptureSessionCoordinator => {
  const globalState = globalThis as typeof globalThis & {
    [AUDIO_CAPTURE_COORDINATOR_KEY]?: AudioCaptureSessionCoordinator
  }
  if (!globalState[AUDIO_CAPTURE_COORDINATOR_KEY]) {
    globalState[AUDIO_CAPTURE_COORDINATOR_KEY] =
      createAudioCaptureSessionCoordinator()
  }
  return globalState[AUDIO_CAPTURE_COORDINATOR_KEY]
}

function buildAudioConstraints(
  deviceId?: string | null
): MediaStreamConstraints["audio"] {
  return deviceId ? { deviceId: { exact: deviceId } } : true
}

const buildCaptureBusyError = (activeOwner: string) => ({
  message: `Audio capture is already active for ${activeOwner}.`,
  details: {
    detail: {
      dictation_error_class: "unknown_error",
      status: "capture_busy",
      message: `Audio capture is already active for ${activeOwner}.`
    }
  }
})

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
    onError,
    onSuccess
  } = options

  const serverRecorderRef = React.useRef<MediaRecorder | null>(null)
  const serverStreamRef = React.useRef<MediaStream | null>(null)
  const serverChunksRef = React.useRef<BlobPart[]>([])
  const captureOwnerRef = React.useRef(false)
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

  const stopServerDictation = React.useCallback(() => {
    const rec = serverRecorderRef.current
    if (rec && rec.state !== "inactive") {
      try {
        rec.stop()
      } catch {}
    }
  }, [])

  const releaseCaptureOwner = React.useCallback(() => {
    if (!captureOwnerRef.current) return
    captureOwnerRef.current = false
    getAudioCaptureSessionCoordinator().release("dictation")
  }, [])

  const reserveCaptureOwner = React.useCallback((): string | null => {
    if (captureOwnerRef.current) return null
    const coordinator = getAudioCaptureSessionCoordinator()
    const activeOwner = coordinator.getActiveOwner()
    if (activeOwner !== null) {
      return activeOwner
    }
    coordinator.claim("dictation")
    captureOwnerRef.current = true
    return null
  }, [])

  const startServerDictation = React.useCallback(async (
    source?: AudioCaptureRequestedSource
  ) => {
    // Synchronous re-entry guard: a double-clicked dictation button must not
    // run a second getUserMedia (and orphan the first stream) while a start is
    // still in flight. Checked/set synchronously before the first await.
    if (startingRef.current) return
    if (isServerDictating) {
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

    const activeOwner = reserveCaptureOwner()
    if (activeOwner) {
      const busyError = buildCaptureBusyError(activeOwner)
      reportError(busyError)
      notification.error({
        message: t("playground:actions.speechErrorTitle", "Dictation failed"),
        description: busyError.message
      })
      return
    }

    startingRef.current = true
    try {
      const requestedDeviceId =
        source?.sourceKind === "mic_device" ? source.deviceId : null
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: buildAudioConstraints(requestedDeviceId)
      })
      // Hold the acquired stream in a ref so the synchronous catch below can
      // stop its tracks if `new MediaRecorder(stream)` or recorder.start()
      // throws.
      serverStreamRef.current = stream
      const recorder = new MediaRecorder(stream)
      serverChunksRef.current = []

      recorder.ondataavailable = (ev: BlobEvent) => {
        if (ev.data && ev.data.size > 0) {
          serverChunksRef.current.push(ev.data)
        }
      }

      recorder.onerror = (event: Event) => {
        console.error("MediaRecorder error", event)
        notification.error({
          message: t("playground:actions.speechErrorTitle", "Dictation failed"),
          description: t(
            "playground:actions.speechErrorBody",
            "Microphone recording error. Check your permissions and try again."
          )
        })
        try {
          stream.getTracks().forEach((trk) => trk.stop())
        } catch {}
        serverStreamRef.current = null
        serverRecorderRef.current = null
        releaseCaptureOwner()
        setIsServerDictating(false)
      }

      recorder.onstop = async () => {
        try {
          const blob = new Blob(serverChunksRef.current, {
            type: recorder.mimeType || "audio/webm"
          })
          if (blob.size === 0) {
            return
          }

          // Build STT options from settings
          const sttOptions: Record<string, any> = {
            language: speechToTextLanguage
          }
          if (sttSettings.model && sttSettings.model.trim().length > 0) {
            sttOptions.model = sttSettings.model.trim()
          }
          if (sttSettings.timestampGranularities) {
            sttOptions.timestamp_granularities = sttSettings.timestampGranularities
          }
          if (sttSettings.prompt && sttSettings.prompt.trim().length > 0) {
            sttOptions.prompt = sttSettings.prompt.trim()
          }
          if (sttSettings.task) {
            sttOptions.task = sttSettings.task
          }
          if (sttSettings.responseFormat) {
            sttOptions.response_format = sttSettings.responseFormat
          }
          if (typeof sttSettings.temperature === "number") {
            sttOptions.temperature = sttSettings.temperature
          }
          if (sttSettings.useSegmentation) {
            sttOptions.segment = true
            if (typeof sttSettings.segK === "number") {
              sttOptions.seg_K = sttSettings.segK
            }
            if (typeof sttSettings.segMinSegmentSize === "number") {
              sttOptions.seg_min_segment_size = sttSettings.segMinSegmentSize
            }
            if (typeof sttSettings.segLambdaBalance === "number") {
              sttOptions.seg_lambda_balance = sttSettings.segLambdaBalance
            }
            if (typeof sttSettings.segUtteranceExpansionWidth === "number") {
              sttOptions.seg_utterance_expansion_width =
                sttSettings.segUtteranceExpansionWidth
            }
            if (sttSettings.segEmbeddingsProvider?.trim()) {
              sttOptions.seg_embeddings_provider =
                sttSettings.segEmbeddingsProvider.trim()
            }
            if (sttSettings.segEmbeddingsModel?.trim()) {
              sttOptions.seg_embeddings_model = sttSettings.segEmbeddingsModel.trim()
            }
          }

          const res = await tldwClient.transcribeAudio(blob, sttOptions)
          let text = ""
          if (res) {
            if (typeof res === "string") {
              text = res
            } else if (typeof (res as any).text === "string") {
              text = (res as any).text
            } else if (typeof (res as any).transcript === "string") {
              text = (res as any).transcript
            } else if (Array.isArray((res as any).segments)) {
              text = (res as any).segments
                .map((s: any) => s?.text || "")
                .join(" ")
                .trim()
            }
          }

          if (text) {
            onTranscript(text)
            onSuccess?.()
          } else {
            reportError({
              details: {
                detail: {
                  dictation_error_class: "empty_transcript",
                  status: "empty_transcript",
                  message: "The transcription did not return any text."
                }
              }
            })
            notification.error({
              message: t(
                "playground:actions.speechErrorTitle",
                "Dictation failed"
              ),
              description: t(
                "playground:actions.speechNoText",
                "The transcription did not return any text."
              )
            })
          }
        } catch (e: any) {
          reportError(e)
          notification.error({
            message: t(
              "playground:actions.speechErrorTitle",
              "Dictation failed"
            ),
            description:
              e?.message ||
              t(
                "playground:actions.speechErrorBody",
                "Transcription request failed. Check tldw server health."
              )
          })
        } finally {
          try {
            stream.getTracks().forEach((trk) => trk.stop())
          } catch {}
          serverStreamRef.current = null
          serverRecorderRef.current = null
          releaseCaptureOwner()
          setIsServerDictating(false)
        }
      }

      serverRecorderRef.current = recorder
      recorder.start()
      setIsServerDictating(true)
    } catch (e: any) {
      reportError(e)
      // Add permissions guidance for microphone errors
      const isChromeOrEdge =
        typeof chrome !== "undefined" && chrome.permissions
      notification.error({
        message: t("playground:actions.speechErrorTitle", "Dictation failed"),
        description: (
          <div>
            <p className="mb-2">
              {t(
                "playground:actions.speechMicError",
                "Unable to access your microphone. Check browser permissions and try again."
              )}
            </p>
            {isChromeOrEdge && (
              <span className="text-xs text-primary">
                {t(
                  "playground:actions.micPermissionsHint",
                  "Check Site Settings > Microphone in your browser"
                )}
              </span>
            )}
          </div>
        )
      })
      // Stop any stream acquired before the throw (e.g. a MediaRecorder ctor
      // or recorder.start() failure) so the mic indicator does not stay on.
      try {
        serverStreamRef.current?.getTracks().forEach((trk) => trk.stop())
      } catch {}
      serverStreamRef.current = null
      serverRecorderRef.current = null
      setIsServerDictating(false)
      releaseCaptureOwner()
    } finally {
      startingRef.current = false
    }
  }, [
    canUseServerStt,
    isServerDictating,
    onSuccess,
    reportError,
    reserveCaptureOwner,
    releaseCaptureOwner,
    speechToTextLanguage,
    sttSettings,
    stopServerDictation,
    onTranscript,
    t
  ])

  // Cleanup on unmount
  React.useEffect(() => {
    return () => {
      stopServerDictation()
    }
  }, [stopServerDictation])

  return {
    isServerDictating,
    startServerDictation,
    stopServerDictation
  }
}
