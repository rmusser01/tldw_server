import React from "react"
import { useStorage } from "@plasmohq/storage/hook"
import { useTranslation } from "react-i18next"
import { Link } from "react-router-dom"
import {
  Button,
  Card,
  Input,
  List,
  Segmented,
  Select,
  Space,
  Switch,
  Tag,
  Tooltip,
  Typography,
  notification
} from "antd"
import type { DefaultOptionType } from "antd/es/select"
import { ArrowRight, Copy, Lock, Mic, Pause, Play, Plus, Save, Star, Trash2, Unlock } from "lucide-react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import {
  createAudioCaptureBusyError,
  getGlobalAudioCaptureSessionCoordinator,
  resolveAudioCapturePlan
} from "@/audio"
import { AudioSourcePicker } from "@/components/Common/AudioSourcePicker"
import { PageShell } from "@/components/Common/PageShell"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import WaveformCanvas from "@/components/Common/WaveformCanvas"
import { inferTldwProviderFromModel, resolveTtsProviderContext } from "@/services/tts-provider"
import { getTtsProviderLabel } from "@/services/tts-providers"
import {
  type TldwTtsProviderCapabilities,
  type TldwTtsVoiceInfo
} from "@/services/tldw/audio-providers"
import { useTtsPlayground, TTS_PRESETS, type TtsPresetKey } from "@/hooks/useTtsPlayground"
import { useStreamingAudioPlayer } from "@/hooks/useStreamingAudioPlayer"
import { useTranscriptionModelsCatalog } from "@/hooks/useTranscriptionModelsCatalog"
import {
  OPENAI_TTS_MODELS,
  OPENAI_TTS_VOICES,
  useTtsProviderData
} from "@/hooks/useTtsProviderData"
import {
  DEFAULT_TLDW_TTS_MODEL,
  DEFAULT_TLDW_TTS_VOICE,
  DEFAULT_TTS_PROVIDER,
  getTTSProvider,
  getTTSSettings,
  setTTSSettings,
  SUPPORTED_TLDW_TTS_FORMATS,
  setTldwTTSSpeed,
  setTldwTTSResponseFormat,
  setTldwTTSStreamingEnabled,
  setResponseSplitting as persistResponseSplitting
} from "@/services/tts"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { copyToClipboard } from "@/utils/clipboard"
import { estimateTtsDurationSeconds, splitMessageContent } from "@/utils/tts"
import { markdownToText } from "@/utils/markdown-to-text"
import { isTimeoutLikeError } from "@/utils/request-timeout"
import { withTemplateFallback } from "@/utils/template-guards"
import { listCustomVoices, type TldwCustomVoice } from "@/services/tldw/voice-cloning"
import { normalizeTtsProviderKey } from "@/services/tldw/tts-provider-keys"
import { TtsJobProgress } from "@/components/Common/TtsJobProgress"
import { LongformDraftEditor } from "@/components/Common/LongformDraftEditor"
import { CharacterProgressBar } from "@/components/Common/CharacterProgressBar"
import { TtsProviderStrip } from "@/components/Option/Speech/TtsProviderStrip"
import { TtsStickyActionBar } from "@/components/Option/Speech/TtsStickyActionBar"
import { TtsInspectorPanel } from "@/components/Option/Speech/TtsInspectorPanel"
import { TtsVoiceTab } from "@/components/Option/Speech/TtsVoiceTab"
import { TtsOutputTab } from "@/components/Option/Speech/TtsOutputTab"
import { TtsAdvancedTab } from "@/components/Option/Speech/TtsAdvancedTab"
import { VoiceCloningManager } from "@/components/Option/TTS/VoiceCloningManager"
import { RenderStrip } from "@/components/Option/Speech/RenderStrip"
import { AudioReadinessStrip } from "@/components/Option/Audio/AudioReadinessStrip"
import { AudioPresetControls } from "@/components/Option/Audio/AudioPresetControls"
import { buildTtsReadinessItems } from "@/components/Option/Audio/audio-readiness"
import { classifyAudioError } from "@/components/Option/Audio/audio-error-classification"
import { VoicePickerModal, type VoiceSelection } from "@/components/Option/Speech/VoicePickerModal"
import { useAudioSourceCatalog } from "@/hooks/useAudioSourceCatalog"
import { useAudioSourcePreferences } from "@/hooks/useAudioSourcePreferences"
import { useMultiRenderState } from "@/hooks/useMultiRenderState"
import {
  getModels as getElevenLabsModels,
  getVoices as getElevenLabsVoices
} from "@/services/elevenlabs"

const { Text, Title, Paragraph } = Typography

type SpeechMode = "roundtrip" | "speak" | "listen"

type SpeechHistoryItem = {
  id: string
  type: "stt" | "tts"
  createdAt: string
  text: string
  favorite?: boolean
  durationMs?: number
  model?: string
  language?: string
  provider?: string
  voice?: string
  format?: string
  speed?: number
  responseSplitting?: string
  streaming?: boolean
  sttTask?: string
  sttTemperature?: number
  sttResponseFormat?: string
  sttUseSegmentation?: boolean
  mode?: "short" | "long"
}

const SAMPLE_TEXT =
  "Sample: Hi there, this is the speech playground reading a short passage so you can preview voice and speed."
const MAX_HISTORY_ITEMS = 100
const STREAMING_FORMATS = new Set(["mp3", "opus", "aac", "flac", "wav", "pcm"])
const TTS_CHAR_WARNING = 2000
const TTS_CHAR_LIMIT = 8000
const TTS_ESTIMATE_CHARS_PER_SEC = 15
const ELEVENLABS_INLINE_TEST_TIMEOUT_MS = 10_000
const TTS_JOB_STEPS = [
  { key: "tts_started", label: "Queued" },
  { key: "tts_synthesizing", label: "Synthesizing" },
  { key: "tts_synthesis_complete", label: "Post-processing" },
  { key: "tts_writing_output", label: "Saving output" },
  { key: "tts_completed", label: "Complete" }
]
const TTS_JOB_STEP_INDEX = TTS_JOB_STEPS.reduce<Record<string, number>>(
  (acc, step, idx) => {
    acc[step.key] = idx
    return acc
  },
  {}
)
const VOICE_ROLE_OPTIONS = [
  { label: "Narrator", value: "narrator" },
  { label: "Speaker A", value: "speaker_a" },
  { label: "Speaker B", value: "speaker_b" },
  { label: "Speaker C", value: "speaker_c" }
]
const DEFAULT_VOICE_ROLE = "narrator"

type VoiceRoleCard = {
  id: string
  role: string
  voiceId: string
}

const formatHistoryDate = (value: string) => {
  try {
    return new Date(value).toLocaleString()
  } catch {
    return value
  }
}

const formatBytes = (value?: number | null) => {
  if (!value || value <= 0) return "0 B"
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / (1024 * 1024)).toFixed(1)} MB`
}

const buildHistoryParamsSummary = (item: SpeechHistoryItem) => {
  const parts: string[] = []
  if (item.type === "tts") {
    if (item.format) parts.push(`fmt ${item.format.toUpperCase()}`)
    if (typeof item.speed === "number") parts.push(`speed ${item.speed.toFixed(2)}`)
    if (item.responseSplitting) parts.push(`split ${item.responseSplitting}`)
    if (typeof item.streaming === "boolean") {
      parts.push(`stream ${item.streaming ? "on" : "off"}`)
    }
  } else {
    if (item.language) parts.push(`lang ${item.language}`)
    if (item.sttTask) parts.push(`task ${item.sttTask}`)
    if (typeof item.sttTemperature === "number") {
      parts.push(`temp ${item.sttTemperature}`)
    }
    if (item.sttResponseFormat) parts.push(`format ${item.sttResponseFormat}`)
    if (typeof item.sttUseSegmentation === "boolean") {
      parts.push(`segment ${item.sttUseSegmentation ? "on" : "off"}`)
    }
  }
  return parts.length > 0 ? parts.join(" · ") : null
}

const buildHistoryDetailTooltip = (item: SpeechHistoryItem) => {
  const rows: Array<[string, string]> = []
  if (item.durationMs != null) rows.push(["Duration", `${(item.durationMs / 1000).toFixed(1)}s`])
  if (item.model) rows.push(["Model", item.model])
  if (item.provider) rows.push(["Provider", item.provider])
  if (item.voice) rows.push(["Voice", item.voice])
  if (item.format) rows.push(["Format", item.format.toUpperCase()])
  if (typeof item.speed === "number") rows.push(["Speed", item.speed.toFixed(2)])
  if (item.responseSplitting) rows.push(["Split", item.responseSplitting])
  if (typeof item.streaming === "boolean") rows.push(["Streaming", item.streaming ? "On" : "Off"])
  if (item.language) rows.push(["Language", item.language])
  if (item.sttTask) rows.push(["Task", item.sttTask])
  if (typeof item.sttTemperature === "number") rows.push(["Temperature", String(item.sttTemperature)])
  if (item.sttResponseFormat) rows.push(["Response", item.sttResponseFormat])
  if (typeof item.sttUseSegmentation === "boolean") {
    rows.push(["Segmentation", item.sttUseSegmentation ? "On" : "Off"])
  }
  if (item.mode) rows.push(["Mode", item.mode])
  if (rows.length === 0) return null
  return (
    <div className="text-xs">
      {rows.map(([label, value]) => (
        <div key={label} className="flex items-center justify-between gap-2">
          <Text type="secondary">{label}</Text>
          <Text code>{value}</Text>
        </div>
      ))}
    </div>
  )
}

type SpeechPlaygroundPageProps = {
  initialMode?: SpeechMode
  lockedMode?: SpeechMode
  hideModeSwitcher?: boolean
}

export const SpeechPlaygroundPage: React.FC<SpeechPlaygroundPageProps> = ({
  initialMode,
  lockedMode,
  hideModeSwitcher = false
}) => {
  const { t } = useTranslation(["playground", "settings", "option", "common"])
  const queryClient = useQueryClient()

  const [mode, setMode] = useStorage<SpeechMode>("speechPlaygroundMode", "roundtrip")
  const [historyItems, setHistoryItems] = useStorage<SpeechHistoryItem[]>(
    "speechPlaygroundHistory",
    []
  )
  const [historyFilter, setHistoryFilter] = React.useState<"all" | "stt" | "tts">(
    "all"
  )
  const [historyFavoritesOnly, setHistoryFavoritesOnly] = React.useState(false)
  const [historyQuery, setHistoryQuery] = React.useState("")
  const [ttsPreset, setTtsPreset] = useStorage<TtsPresetKey>("ttsPreset", "balanced")
  const { preference: audioSourcePreference, setPreference: setAudioSourcePreference } =
    useAudioSourcePreferences("speech_playground")
  const { devices: audioInputDevices } = useAudioSourceCatalog()
  const isLockedTtsRoute = lockedMode === "listen"
  const effectiveMode = lockedMode ?? mode
  const effectiveHistoryFilter = isLockedTtsRoute ? "tts" : historyFilter

  React.useEffect(() => {
    if (lockedMode) return
    if (initialMode && mode !== initialMode) {
      setMode(initialMode)
    }
  }, [initialMode, lockedMode, mode, setMode])

  const handleModeChange = React.useCallback(
    (value: SpeechMode) => {
      if (lockedMode) return
      setMode(value)
    },
    [lockedMode, setMode]
  )

  const addHistoryItem = React.useCallback(
    (item: SpeechHistoryItem) => {
      setHistoryItems((prev) => {
        const next = [item, ...(prev || [])]
        return next.slice(0, MAX_HISTORY_ITEMS)
      })
    },
    [setHistoryItems]
  )

  const removeHistoryItem = React.useCallback(
    (id: string) => {
      setHistoryItems((prev) => (prev || []).filter((item) => item.id !== id))
    },
    [setHistoryItems]
  )

  const clearHistory = React.useCallback(() => {
    setHistoryItems([])
  }, [setHistoryItems])

  const filteredHistory = React.useMemo(() => {
    const query = historyQuery.trim().toLowerCase()
    return (historyItems || []).filter((item) => {
      if (effectiveHistoryFilter !== "all" && item.type !== effectiveHistoryFilter) return false
      if (historyFavoritesOnly && !item.favorite) return false
      if (!query) return true
      return item.text.toLowerCase().includes(query)
    })
  }, [effectiveHistoryFilter, historyItems, historyQuery, historyFavoritesOnly])

  const toggleHistoryFavorite = React.useCallback(
    (id: string) => {
      setHistoryItems((prev) =>
        (prev || []).map((item) =>
          item.id === id ? { ...item, favorite: !item.favorite } : item
        )
      )
    },
    [setHistoryItems]
  )

  const [speechToTextLanguage] = useStorage("speechToTextLanguage", "en-US")
  const [sttModel] = useStorage("sttModel", "whisper-1")
  const [sttTask] = useStorage("sttTask", "transcribe")
  const [sttResponseFormat] = useStorage("sttResponseFormat", "json")
  const [sttTemperature] = useStorage("sttTemperature", 0)
  const [sttUseSegmentation] = useStorage("sttUseSegmentation", false)
  const [sttTimestampGranularities] = useStorage("sttTimestampGranularities", "segment")
  const [sttPrompt] = useStorage("sttPrompt", "")
  const [sttSegK] = useStorage("sttSegK", 6)
  const [sttSegMinSegmentSize] = useStorage("sttSegMinSegmentSize", 5)
  const [sttSegLambdaBalance] = useStorage("sttSegLambdaBalance", 0.01)
  const [sttSegUtteranceExpansionWidth] = useStorage("sttSegUtteranceExpansionWidth", 2)
  const [sttSegEmbeddingsProvider] = useStorage("sttSegEmbeddingsProvider", "")
  const [sttSegEmbeddingsModel] = useStorage("sttSegEmbeddingsModel", "")

  const [activeModel, setActiveModel] = React.useState<string | undefined>()
  const {
    serverModels,
    serverModelsLoading,
    serverModelsError,
    retryServerModels
  } = useTranscriptionModelsCatalog({
    activeModel,
    defaultModel: sttModel,
    enabled: effectiveMode !== "listen",
    onInitialModel: setActiveModel,
    warnLabel: "Speech Playground"
  })
  const [isRecording, setIsRecording] = React.useState(false)
  const [isTranscribing, setIsTranscribing] = React.useState(false)
  const [useLongRunning, setUseLongRunning] = React.useState(false)
  const [liveText, setLiveText] = React.useState("")
  const [lastTranscript, setLastTranscript] = React.useState("")
  const [transcriptLocked, setTranscriptLocked] = React.useState(true)
  const [recordingError, setRecordingError] = React.useState<string | null>(null)
  const [recordingStream, setRecordingStream] = React.useState<MediaStream | null>(null)
  const recorderRef = React.useRef<MediaRecorder | null>(null)
  const recordingStreamRef = React.useRef<MediaStream | null>(null)
  const captureOwnerRef = React.useRef(false)
  const chunksRef = React.useRef<BlobPart[]>([])
  const startedAtRef = React.useRef<number | null>(null)
  const liveTextRef = React.useRef<string>("")
  const [hasAudioCatalogSettled, setHasAudioCatalogSettled] = React.useState(false)
  const captureCapabilities = React.useMemo(
    () => ({
      browserDictationSupported:
        typeof navigator !== "undefined" &&
        typeof navigator.mediaDevices?.getUserMedia === "function",
      serverDictationSupported: true,
      liveVoiceSupported: false,
      secureContextAvailable:
        typeof window !== "undefined" ? window.isSecureContext : false
    }),
    []
  )
  React.useEffect(() => {
    const mediaDevices =
      typeof navigator !== "undefined" ? navigator.mediaDevices : undefined

    if (!mediaDevices?.enumerateDevices) {
      return
    }

    let active = true

    void mediaDevices
      .enumerateDevices()
      .catch(() => undefined)
      .finally(() => {
        if (active) {
          setHasAudioCatalogSettled(true)
        }
      })

    return () => {
      active = false
    }
  }, [])

  const releaseCaptureOwner = React.useCallback(() => {
    if (!captureOwnerRef.current) return
    captureOwnerRef.current = false
    getGlobalAudioCaptureSessionCoordinator().release("speech_playground")
  }, [])

  const reserveCaptureOwner = React.useCallback(() => {
    if (captureOwnerRef.current) return
    const coordinator = getGlobalAudioCaptureSessionCoordinator()
    const activeOwner = coordinator.getActiveOwner()
    if (activeOwner !== null) {
      throw createAudioCaptureBusyError(activeOwner)
    }
    coordinator.claim("speech_playground")
    captureOwnerRef.current = true
  }, [])

  const stopRecordingStreamTracks = React.useCallback(
    (stream?: MediaStream | null, syncState = true) => {
      const streamToStop = stream ?? recordingStreamRef.current
      if (streamToStop) {
        try {
          streamToStop.getTracks().forEach((track) => track.stop())
        } catch {}
      }
      if (!stream || recordingStreamRef.current === stream) {
        recordingStreamRef.current = null
      }
      if (syncState) {
        setRecordingStream(null)
      }
    },
    []
  )

  const resolvedAudioSource = React.useMemo(() => {
    if (
      hasAudioCatalogSettled &&
      audioSourcePreference.sourceKind === "mic_device" &&
      !audioInputDevices.some(
        (device) => device.deviceId === audioSourcePreference.deviceId
      )
    ) {
      return {
        sourceKind: "default_mic" as const,
        deviceId: null
      }
    }

    return audioSourcePreference
  }, [audioInputDevices, audioSourcePreference, hasAudioCatalogSettled])
  const capturePlan = React.useMemo(
    () =>
      resolveAudioCapturePlan({
        featureGroup: "speech_playground",
        requestedSource: audioSourcePreference,
        requestedSpeechPath: "speech_playground_recording",
        capabilities: captureCapabilities
      }),
    [audioSourcePreference, captureCapabilities]
  )
  const captureDeviceId =
    resolvedAudioSource.sourceKind === "mic_device"
      ? resolvedAudioSource.deviceId ?? capturePlan.resolvedDeviceId
      : null

  React.useEffect(() => {
    return () => {
      const recorder = recorderRef.current
      recorderRef.current = null

      if (recorder) {
        recorder.ondataavailable = null
        recorder.onerror = null
        recorder.onstop = null
        try {
          if (recorder.state === "recording") {
            recorder.stop()
          }
        } catch {}
      }

      stopRecordingStreamTracks(recordingStreamRef.current, false)
      releaseCaptureOwner()
    }
  }, [releaseCaptureOwner, stopRecordingStreamTracks])

  const appendLiveText = React.useCallback((textChunk: string) => {
    if (!textChunk) return
    setLiveText((prev) => {
      const next = prev ? `${prev} ${textChunk}` : textChunk
      liveTextRef.current = next
      return next
    })
  }, [])

  const canEditTranscript = !transcriptLocked && !isRecording && !isTranscribing

  const handleTranscriptChange = (value: string) => {
    setLiveText(value)
    liveTextRef.current = value
    setLastTranscript(value)
  }

  const transcribeBlob = React.useCallback(
    async (blob: Blob, modelOverride?: string): Promise<string> => {
      const sttOptions: Record<string, any> = {
        language: speechToTextLanguage
      }
      const modelToUse = modelOverride || activeModel || sttModel
      if (modelToUse && modelToUse.trim().length > 0) {
        sttOptions.model = modelToUse.trim()
      }
      if (sttTimestampGranularities) {
        sttOptions.timestamp_granularities = sttTimestampGranularities
      }
      if (sttPrompt && sttPrompt.trim().length > 0) {
        sttOptions.prompt = sttPrompt.trim()
      }
      if (sttTask) {
        sttOptions.task = sttTask
      }
      if (sttResponseFormat) {
        sttOptions.response_format = sttResponseFormat
      }
      if (typeof sttTemperature === "number") {
        sttOptions.temperature = sttTemperature
      }
      if (sttUseSegmentation) {
        sttOptions.segment = true
        if (typeof sttSegK === "number") {
          sttOptions.seg_K = sttSegK
        }
        if (typeof sttSegMinSegmentSize === "number") {
          sttOptions.seg_min_segment_size = sttSegMinSegmentSize
        }
        if (typeof sttSegLambdaBalance === "number") {
          sttOptions.seg_lambda_balance = sttSegLambdaBalance
        }
        if (typeof sttSegUtteranceExpansionWidth === "number") {
          sttOptions.seg_utterance_expansion_width = sttSegUtteranceExpansionWidth
        }
        if (sttSegEmbeddingsProvider?.trim()) {
          sttOptions.seg_embeddings_provider = sttSegEmbeddingsProvider.trim()
        }
        if (sttSegEmbeddingsModel?.trim()) {
          sttOptions.seg_embeddings_model = sttSegEmbeddingsModel.trim()
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
      return text
    },
    [
      activeModel,
      speechToTextLanguage,
      sttModel,
      sttPrompt,
      sttResponseFormat,
      sttSegEmbeddingsModel,
      sttSegEmbeddingsProvider,
      sttSegK,
      sttSegLambdaBalance,
      sttSegMinSegmentSize,
      sttSegUtteranceExpansionWidth,
      sttTask,
      sttTemperature,
      sttTimestampGranularities,
      sttUseSegmentation
    ]
  )

  const handleStartRecording = async () => {
    if (isTranscribing) return
    if (isRecording) {
      recorderRef.current?.stop()
      setIsRecording(false)
      setIsTranscribing(true)
      return
    }
    try {
      setRecordingError(null)
      reserveCaptureOwner()
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: captureDeviceId
          ? { deviceId: { exact: captureDeviceId } }
          : true
      })
      const recorder = new MediaRecorder(stream)
      recorderRef.current = recorder
      recordingStreamRef.current = stream
      chunksRef.current = []
      startedAtRef.current = Date.now()
      liveTextRef.current = ""
      setLiveText("")
      setRecordingStream(stream)

      recorder.ondataavailable = async (ev: BlobEvent) => {
        if (!ev.data || ev.data.size === 0) return
        if (useLongRunning) {
          try {
            const text = await transcribeBlob(ev.data)
            if (text) {
              appendLiveText(text)
            }
          } catch (e: any) {
            // eslint-disable-next-line no-console
            console.error("Streaming STT chunk failed", e)
          }
        } else {
          chunksRef.current.push(ev.data)
        }
      }

      recorder.onerror = (event: Event) => {
        // eslint-disable-next-line no-console
        console.error("MediaRecorder error", event)
        setRecordingError(
          t(
            "playground:actions.speechErrorBody",
            "Microphone recording error. Check your permissions and try again."
          )
        )
        notification.error({
          message: t("playground:actions.speechErrorTitle", "Dictation failed"),
          description: t(
            "playground:actions.speechErrorBody",
            "Microphone recording error. Check your permissions and try again."
          )
        })
        setIsRecording(false)
        setIsTranscribing(false)
        recorderRef.current = null
        stopRecordingStreamTracks(recorder.stream)
        releaseCaptureOwner()
      }

      recorder.onstop = async () => {
        try {
          const startedAt = startedAtRef.current
          startedAtRef.current = null
          if (useLongRunning) {
            const text = liveTextRef.current.trim()
            if (!text) {
              return
            }
            setLiveText(text)
            liveTextRef.current = text
            const nowIso = new Date().toISOString()
            const durationMs = startedAt != null ? Date.now() - startedAt : undefined
            addHistoryItem({
              id: `${nowIso}-${Math.random().toString(36).slice(2, 8)}`,
              createdAt: nowIso,
              durationMs,
              model: activeModel || sttModel,
              language: speechToTextLanguage,
              sttTask,
              sttTemperature,
              sttResponseFormat,
              sttUseSegmentation,
              text,
              type: "stt",
              mode: "long"
            })
            setLastTranscript(text)
          } else {
            const blob = new Blob(chunksRef.current, {
              type: recorder.mimeType || "audio/webm"
            })
            chunksRef.current = []
            if (blob.size === 0) return
            const text = await transcribeBlob(blob)
            if (!text) {
              notification.error({
                message: t("playground:actions.speechErrorTitle", "Dictation failed"),
                description: t(
                  "playground:actions.speechNoText",
                  "The transcription did not return any text."
                )
              })
              return
            }
            setLiveText(text)
            liveTextRef.current = text
            const nowIso = new Date().toISOString()
            const durationMs = startedAt != null ? Date.now() - startedAt : undefined
            addHistoryItem({
              id: `${nowIso}-${Math.random().toString(36).slice(2, 8)}`,
              createdAt: nowIso,
              durationMs,
              model: activeModel || sttModel,
              language: speechToTextLanguage,
              sttTask,
              sttTemperature,
              sttResponseFormat,
              sttUseSegmentation,
              text,
              type: "stt",
              mode: "short"
            })
            setLastTranscript(text)
          }
        } catch (e: unknown) {
          const classified = classifyAudioError(e)
          notification.error({
            message: t("playground:actions.speechErrorTitle", "Dictation failed"),
            description: classified.recovery ||
              t(
                "playground:actions.speechErrorBody",
                "Transcription request failed. Check tldw server health."
              )
          })
        } finally {
          recorderRef.current = null
          stopRecordingStreamTracks(recorder.stream)
          releaseCaptureOwner()
          setIsRecording(false)
          setIsTranscribing(false)
        }
      }

      recorder.start(useLongRunning ? 5000 : undefined)
      setIsRecording(true)
    } catch (e: unknown) {
      const classified = classifyAudioError(e)
      const description =
        classified.recovery ||
        t(
          "playground:actions.speechMicError",
          "Unable to access your microphone. Check browser permissions and try again."
        )
      setRecordingError(
        description
      )
      notification.error({
        message: t("playground:actions.speechErrorTitle", "Dictation failed"),
        description
      })
      recorderRef.current = null
      stopRecordingStreamTracks()
      releaseCaptureOwner()
      setIsRecording(false)
      setIsTranscribing(false)
    }
  }

  const handleSaveToNotes = async (item: SpeechHistoryItem) => {
    const title = `STT: ${new Date(item.createdAt).toLocaleString()}`
    try {
      await tldwClient.createNote(item.text, {
        title,
        metadata: {
          origin: "speech-playground",
          stt_model: item.model,
          stt_language: item.language
        }
      })
      notification.success({
        message: t("settings:healthPage.copyDiagnostics", "Saved to Notes"),
        description: t("playground:tts.savedToNotes", "Transcription saved as a note.")
      })
    } catch (e: any) {
      notification.error({
        message: t("error", "Error"),
        description: e?.message || t("somethingWentWrong", "Something went wrong")
      })
    }
  }

  const {
    segments,
    isGenerating,
    generateSegments,
    clearSegments,
    setSegments
  } = useTtsPlayground()
  const {
    start: streamStart,
    append: streamAppend,
    finish: streamFinish,
    stop: streamStop,
    state: streamState,
    getBufferedBlob
  } = useStreamingAudioPlayer()
  const wsRef = React.useRef<WebSocket | null>(null)
  const streamMetaRef = React.useRef<{
    provider: string
    model?: string
    voice?: string
    format?: string
  } | null>(null)
  const [streamStatus, setStreamStatus] = React.useState<
    "idle" | "connecting" | "streaming" | "complete" | "error"
  >("idle")
  const [streamError, setStreamError] = React.useState<string | null>(null)
  const streamErrorRef = React.useRef<string | null>(null)
  const [streamChunks, setStreamChunks] = React.useState(0)
  const [streamBytes, setStreamBytes] = React.useState(0)
  const [activeSegmentIndex, setActiveSegmentIndex] = React.useState<number | null>(null)
  const audioRef = React.useRef<HTMLAudioElement | null>(null)
  const [currentTime, setCurrentTime] = React.useState(0)
  const [duration, setDuration] = React.useState(0)
  const [ttsText, setTtsText] = React.useState("")
  const [useDraftEditor, setUseDraftEditor] = React.useState(false)
  const [outlineDraft, setOutlineDraft] = React.useState("")
  const [transcriptDraft, setTranscriptDraft] = React.useState("")
  const [draftErrors, setDraftErrors] = React.useState<{ outline?: string; transcript?: string }>({})
  const [voicePreviewText, setVoicePreviewText] = React.useState(
    "Hello, this is a preview of the selected voice."
  )
  const [useTtsJob, setUseTtsJob] = React.useState(false)
  const [ttsJobId, setTtsJobId] = React.useState<number | null>(null)
  const [ttsJobStatus, setTtsJobStatus] = React.useState<"idle" | "running" | "success" | "error">("idle")
  const [ttsJobProgress, setTtsJobProgress] = React.useState<number | null>(null)
  const [ttsJobMessage, setTtsJobMessage] = React.useState<string | null>(null)
  const [ttsJobEta, setTtsJobEta] = React.useState<number | null>(null)
  const [ttsJobError, setTtsJobError] = React.useState<string | null>(null)
  const ttsJobAbortRef = React.useRef<AbortController | null>(null)
  const [useVoiceRoles, setUseVoiceRoles] = React.useState(false)
  const [voiceCards, setVoiceCards] = React.useState<VoiceRoleCard[]>([])
  const [voicePreviewUrl, setVoicePreviewUrl] = React.useState<string | null>(null)
  const [voicePreviewCardId, setVoicePreviewCardId] = React.useState<string | null>(null)
  const [voicePreviewingId, setVoicePreviewingId] = React.useState<string | null>(null)

  // Compose & Compare: Multi-render strips + voice picker
  const [voicePickerOpen, setVoicePickerOpen] = React.useState(false)
  const [voicePickerTargetStripId, setVoicePickerTargetStripId] = React.useState<string | null>(null)
  const multiRender = useMultiRenderState()

  const { data: ttsSettings } = useQuery({
    queryKey: ["fetchTTSSettings"],
    queryFn: getTTSSettings
  })

  const [elevenVoiceId, setElevenVoiceId] = React.useState<string | undefined>(undefined)
  const [elevenModelId, setElevenModelId] = React.useState<string | undefined>(undefined)
  const [tldwModel, setTldwModel] = React.useState<string | undefined>(
    DEFAULT_TLDW_TTS_MODEL
  )
  const [tldwVoice, setTldwVoice] = React.useState<string | undefined>(
    DEFAULT_TLDW_TTS_VOICE
  )
  const [tldwFormat, setTldwFormat] = React.useState<string | undefined>(undefined)
  const [tldwLanguage, setTldwLanguage] = React.useState<string | undefined>(undefined)
  const [tldwStreaming, setTldwStreaming] = React.useState(false)
  const [tldwEmotion, setTldwEmotion] = React.useState<string | undefined>(undefined)
  const [tldwEmotionIntensity, setTldwEmotionIntensity] = React.useState<number>(1)
  const [tldwNormalize, setTldwNormalize] = React.useState(true)
  const [tldwNormalizeUnits, setTldwNormalizeUnits] = React.useState(false)
  const [tldwNormalizeUrls, setTldwNormalizeUrls] = React.useState(true)
  const [tldwNormalizeEmails, setTldwNormalizeEmails] = React.useState(true)
  const [tldwNormalizePhones, setTldwNormalizePhones] = React.useState(true)
  const [tldwNormalizePlurals, setTldwNormalizePlurals] = React.useState(true)
  const [responseSplitting, setResponseSplitting] = React.useState("punctuation")
  const [openAiModel, setOpenAiModel] = React.useState<string | undefined>(undefined)
  const [openAiVoice, setOpenAiVoice] = React.useState<string | undefined>(undefined)
  const [inlineElevenLabsApiKey, setInlineElevenLabsApiKey] = React.useState("")
  const [inlineElevenLabsSaving, setInlineElevenLabsSaving] = React.useState(false)
  const [inlineElevenLabsResult, setInlineElevenLabsResult] = React.useState<{
    ok: boolean
    message: string
  } | null>(null)
  const [inlineElevenLabsDetailsOpen, setInlineElevenLabsDetailsOpen] = React.useState(true)
  const provider = ttsSettings?.ttsProvider || DEFAULT_TTS_PROVIDER
  const isTldw = provider === "tldw"
  const inferredProviderKey = React.useMemo(() => {
    if (!isTldw) return null
    return inferTldwProviderFromModel(tldwModel || ttsSettings?.tldwTtsModel)
  }, [isTldw, tldwModel, ttsSettings?.tldwTtsModel])
  const {
    hasAudio,
    ffmpegAvailable,
    providersInfo,
    tldwTtsModels,
    tldwVoiceCatalog,
    elevenLabsData,
    elevenLabsLoading,
    elevenLabsError,
    refetchElevenLabs
  } = useTtsProviderData({
    provider,
    elevenLabsApiKey: ttsSettings?.elevenLabsApiKey,
    inferredProviderKey
  })

  const { data: customVoices = [] } = useQuery<TldwCustomVoice[]>({
    queryKey: ["tts-custom-voices"],
    queryFn: listCustomVoices,
    enabled: isTldw && hasAudio
  })

  const currentTtsSelection = React.useMemo(() => {
    const format = tldwFormat || ttsSettings?.tldwTtsResponseFormat || "mp3"
    const speed = ttsSettings?.tldwTtsSpeed ?? 1

    if (provider === "browser") {
      return {
        provider,
        model: "",
        voice: "",
        format,
        speed
      }
    }

    if (provider === "openai") {
      const model =
        openAiModel ||
        ttsSettings?.openAITTSModel ||
        OPENAI_TTS_MODELS[0]?.value ||
        "tts-1"
      const voice =
        openAiVoice ||
        ttsSettings?.openAITTSVoice ||
        OPENAI_TTS_VOICES[model]?.[0]?.value ||
        "alloy"
      return { provider, model, voice, format, speed }
    }

    if (provider === "elevenlabs") {
      const model =
        elevenModelId ||
        ttsSettings?.elevenLabsModel ||
        elevenLabsData?.models?.[0]?.model_id ||
        ""
      const voice =
        elevenVoiceId ||
        ttsSettings?.elevenLabsVoiceId ||
        elevenLabsData?.voices?.[0]?.voice_id ||
        ""
      return { provider, model, voice, format, speed }
    }

    return {
      provider,
      model:
        tldwModel ||
        ttsSettings?.tldwTtsModel ||
        DEFAULT_TLDW_TTS_MODEL,
      voice:
        tldwVoice ||
        ttsSettings?.tldwTtsVoice ||
        DEFAULT_TLDW_TTS_VOICE,
      format,
      speed
    }
  }, [
    elevenLabsData?.models,
    elevenLabsData?.voices,
    elevenModelId,
    elevenVoiceId,
    openAiModel,
    openAiVoice,
    provider,
    tldwFormat,
    tldwModel,
    tldwVoice,
    ttsSettings
  ])
  const readinessProvider = isTldw && inferredProviderKey ? inferredProviderKey : provider
  const ttsReadinessItems = React.useMemo(
    () =>
      buildTtsReadinessItems({
        provider: readinessProvider,
        hasAudio,
        providersInfo,
        elevenLabsApiKey: ttsSettings?.elevenLabsApiKey,
        elevenLabsData,
        elevenLabsLoading,
        elevenLabsError,
        ffmpegAvailable
      }),
    [
      elevenLabsData,
      elevenLabsError,
      elevenLabsLoading,
      ffmpegAvailable,
      hasAudio,
      providersInfo,
      readinessProvider,
      ttsSettings?.elevenLabsApiKey
    ]
  )

  React.useEffect(() => {
    if (provider !== "elevenlabs" || ttsSettings?.elevenLabsApiKey) {
      setInlineElevenLabsApiKey("")
      setInlineElevenLabsResult(null)
      setInlineElevenLabsDetailsOpen(true)
    }
  }, [provider, ttsSettings?.elevenLabsApiKey])

  const handleAddRenderStrip = React.useCallback(() => {
    // Try to use last-used voice config from localStorage
    let lastVoice: { provider?: string; voice?: string; model?: string } | null = null
    try {
      const stored = localStorage.getItem("tts-last-render-config")
      if (stored) lastVoice = JSON.parse(stored)
    } catch {}

    const matchingLastVoice = lastVoice?.provider === provider ? lastVoice : null
    const defaultConfig = {
      ...currentTtsSelection,
      voice: matchingLastVoice?.voice || currentTtsSelection.voice,
      model: matchingLastVoice?.model || currentTtsSelection.model
    }
    multiRender.addRender(defaultConfig)
  }, [provider, currentTtsSelection, multiRender])

  const handleVoicePickerSelect = React.useCallback(
    (selection: VoiceSelection) => {
      // Persist last-used voice config
      try {
        localStorage.setItem("tts-last-render-config", JSON.stringify(selection))
      } catch {}

      if (voicePickerTargetStripId) {
        // Update existing strip config
        const existing = multiRender.renders.find((r) => r.id === voicePickerTargetStripId)
        if (existing) {
          multiRender.updateConfig(voicePickerTargetStripId, {
            ...existing.config,
            provider: selection.provider,
            voice: selection.voice,
            // Only preserve old model if the provider didn't change; otherwise use selection.model
            // (which is undefined for openai/elevenlabs/browser — that's intentional)
            model: selection.provider === existing.config.provider
              ? (selection.model ?? existing.config.model)
              : selection.model
          })
        }
        setVoicePickerTargetStripId(null)
      } else {
        // Add new render strip with the selected voice
        multiRender.addRender({
          provider: selection.provider,
          voice: selection.voice,
          model: selection.model,
          format: tldwFormat || ttsSettings?.tldwTtsResponseFormat || "mp3",
          speed: ttsSettings?.tldwTtsSpeed ?? 1
        })
      }
    },
    [voicePickerTargetStripId, multiRender, tldwFormat, ttsSettings]
  )

  const handleRenderStripConfigTagClick = React.useCallback(
    (stripId: string, field: string) => {
      if (field === "voice" || field === "provider") {
        setVoicePickerTargetStripId(stripId)
        setVoicePickerOpen(true)
      }
      // format/speed tag clicks are handled after openInspectorAt is defined
    },
    []
  )

  const handleGenerateAllRenders = React.useCallback(
    async (text: string) => {
      if (!text.trim()) return
      await multiRender.generateAll(text)
    },
    [multiRender]
  )

  const handlePlayAllRenders = React.useCallback(() => {
    multiRender.playAllSequentially()
  }, [multiRender])

  React.useEffect(() => {
    return () => {
      if (ttsJobAbortRef.current) {
        try {
          ttsJobAbortRef.current.abort()
        } catch {}
        ttsJobAbortRef.current = null
      }
    }
  }, [])

  React.useEffect(() => {
    return () => {
      if (voicePreviewUrl) {
        try {
          URL.revokeObjectURL(voicePreviewUrl)
        } catch {}
      }
    }
  }, [voicePreviewUrl])

  React.useEffect(() => {
    if (!ttsSettings) return
    setElevenVoiceId(ttsSettings.elevenLabsVoiceId || undefined)
    setElevenModelId(ttsSettings.elevenLabsModel || undefined)
    setTldwModel(ttsSettings.tldwTtsModel || DEFAULT_TLDW_TTS_MODEL)
    setTldwVoice(ttsSettings.tldwTtsVoice || DEFAULT_TLDW_TTS_VOICE)
    setTldwFormat(ttsSettings.tldwTtsResponseFormat || undefined)
    setTldwLanguage(ttsSettings.tldwTtsLanguage || undefined)
    setTldwStreaming(Boolean(ttsSettings.tldwTtsStreaming))
    setTldwEmotion(ttsSettings.tldwTtsEmotion || undefined)
    setTldwEmotionIntensity(
      typeof ttsSettings.tldwTtsEmotionIntensity === "number"
        ? ttsSettings.tldwTtsEmotionIntensity
        : 1
    )
    setTldwNormalize(ttsSettings.tldwTtsNormalize !== false)
    setTldwNormalizeUnits(Boolean(ttsSettings.tldwTtsNormalizeUnits))
    setTldwNormalizeUrls(ttsSettings.tldwTtsNormalizeUrls !== false)
    setTldwNormalizeEmails(ttsSettings.tldwTtsNormalizeEmails !== false)
    setTldwNormalizePhones(ttsSettings.tldwTtsNormalizePhones !== false)
    setTldwNormalizePlurals(ttsSettings.tldwTtsNormalizePlurals !== false)
    setResponseSplitting(ttsSettings.responseSplitting || "punctuation")
    setOpenAiModel(ttsSettings.openAITTSModel || undefined)
    setOpenAiVoice(ttsSettings.openAITTSVoice || undefined)
  }, [ttsSettings])

  React.useEffect(() => {
    if (tldwStreaming && useTtsJob) {
      setUseTtsJob(false)
    }
  }, [tldwStreaming, useTtsJob])

  React.useEffect(() => {
    if (!isTldw && useVoiceRoles) {
      setUseVoiceRoles(false)
    }
  }, [isTldw, useVoiceRoles])

  React.useEffect(() => {
    if (!useDraftEditor) {
      setDraftErrors({})
    }
  }, [useDraftEditor])

  React.useEffect(() => {
    if (!useVoiceRoles) return
    const primary = voiceCards[0]?.voiceId
    if (primary && primary !== tldwVoice) {
      setTldwVoice(primary)
    }
  }, [useVoiceRoles, voiceCards, tldwVoice, setTldwVoice])

  const handleAudioTimeUpdate = () => {
    const el = audioRef.current
    if (!el) return
    setCurrentTime(el.currentTime || 0)
    setDuration(el.duration || 0)
  }

  const handleSegmentSelect = (idx: number) => {
    setActiveSegmentIndex(idx)
    setCurrentTime(0)
    setDuration(0)
  }

  const isTtsDisabled = ttsSettings?.ttsEnabled === false
  const normalizationOptions = React.useMemo(
    () => ({
      normalize: tldwNormalize,
      unit_normalization: tldwNormalizeUnits,
      url_normalization: tldwNormalizeUrls,
      email_normalization: tldwNormalizeEmails,
      phone_normalization: tldwNormalizePhones,
      optional_pluralization_normalization: tldwNormalizePlurals
    }),
    [
      tldwNormalize,
      tldwNormalizeUnits,
      tldwNormalizeUrls,
      tldwNormalizeEmails,
      tldwNormalizePhones,
      tldwNormalizePlurals
    ]
  )
  const currentTtsPresetConfig = React.useMemo(() => {
    const config: Record<string, unknown> = {
      provider: currentTtsSelection.provider,
      response_format: currentTtsSelection.format,
      speed: currentTtsSelection.speed,
      streaming: tldwStreaming,
      response_splitting: responseSplitting,
      normalization_options: normalizationOptions
    }
    if (currentTtsSelection.model) config.model = currentTtsSelection.model
    if (currentTtsSelection.voice) config.voice = currentTtsSelection.voice
    if (tldwLanguage) config.lang_code = tldwLanguage
    if (tldwEmotion) config.emotion = tldwEmotion
    if (typeof tldwEmotionIntensity === "number") {
      config.emotion_intensity = tldwEmotionIntensity
    }
    if (currentTtsSelection.provider === "browser") {
      config.browser_local = true
      config.requires_browser_revalidation = true
    }
    return config
  }, [
    currentTtsSelection,
    normalizationOptions,
    responseSplitting,
    tldwEmotion,
    tldwEmotionIntensity,
    tldwLanguage,
    tldwStreaming
  ])

  const handleApplyServerTtsPreset = React.useCallback(
    async (config: Record<string, unknown>) => {
      const normalizedProvider =
        typeof config.provider === "string" ? config.provider.trim().toLowerCase() : ""
      const nextProvider =
        normalizedProvider
          ? normalizedProvider
          : provider
      const nextModel =
        typeof config.model === "string" && config.model.trim()
          ? config.model.trim()
          : undefined
      const nextVoice =
        typeof config.voice === "string" && config.voice.trim()
          ? config.voice.trim()
          : undefined
      const nextFormat =
        typeof config.response_format === "string" && config.response_format.trim()
          ? config.response_format.trim()
          : typeof config.format === "string" && config.format.trim()
            ? config.format.trim()
            : undefined
      const nextSpeed =
        typeof config.speed === "number" && Number.isFinite(config.speed)
          ? config.speed
          : undefined
      const nextStreaming =
        typeof config.streaming === "boolean" ? config.streaming : undefined
      const nextSplitting =
        typeof config.response_splitting === "string"
          ? config.response_splitting
          : undefined
      const nextLanguage =
        typeof config.lang_code === "string"
          ? config.lang_code
          : typeof config.language === "string"
            ? config.language
            : undefined
      const nextEmotion = typeof config.emotion === "string" ? config.emotion : undefined
      const nextEmotionIntensity =
        typeof config.emotion_intensity === "number" &&
        Number.isFinite(config.emotion_intensity)
          ? config.emotion_intensity
          : undefined
      const normalizers =
        config.normalization_options &&
        typeof config.normalization_options === "object" &&
        !Array.isArray(config.normalization_options)
          ? (config.normalization_options as Record<string, unknown>)
          : {}
      const boolOr = (value: unknown, fallback: boolean) =>
        typeof value === "boolean" ? value : fallback

      if (nextProvider === "openai") {
        if (nextModel) setOpenAiModel(nextModel)
        if (nextVoice) setOpenAiVoice(nextVoice)
      } else if (nextProvider === "elevenlabs") {
        if (nextModel) setElevenModelId(nextModel)
        if (nextVoice) setElevenVoiceId(nextVoice)
      } else if (nextProvider !== "browser") {
        if (nextModel) setTldwModel(nextModel)
        if (nextVoice) setTldwVoice(nextVoice)
      }
      if (nextFormat) setTldwFormat(nextFormat)
      if (nextSplitting) setResponseSplitting(nextSplitting)
      if (nextLanguage) setTldwLanguage(nextLanguage)
      if (typeof nextStreaming === "boolean") setTldwStreaming(nextStreaming)
      if (nextEmotion) setTldwEmotion(nextEmotion)
      if (typeof nextEmotionIntensity === "number") {
        setTldwEmotionIntensity(nextEmotionIntensity)
      }
      setTldwNormalize(boolOr(normalizers.normalize, tldwNormalize))
      setTldwNormalizeUnits(
        boolOr(normalizers.unit_normalization, tldwNormalizeUnits)
      )
      setTldwNormalizeUrls(boolOr(normalizers.url_normalization, tldwNormalizeUrls))
      setTldwNormalizeEmails(
        boolOr(normalizers.email_normalization, tldwNormalizeEmails)
      )
      setTldwNormalizePhones(
        boolOr(normalizers.phone_normalization, tldwNormalizePhones)
      )
      setTldwNormalizePlurals(
        boolOr(
          normalizers.optional_pluralization_normalization,
          tldwNormalizePlurals
        )
      )

      try {
        const currentSettings = await getTTSSettings()
        await setTTSSettings({
          ...currentSettings,
          ttsProvider: nextProvider,
          responseSplitting: nextSplitting ?? currentSettings.responseSplitting,
          openAITTSModel:
            nextProvider === "openai" && nextModel
              ? nextModel
              : currentSettings.openAITTSModel,
          openAITTSVoice:
            nextProvider === "openai" && nextVoice
              ? nextVoice
              : currentSettings.openAITTSVoice,
          elevenLabsModel:
            nextProvider === "elevenlabs" && nextModel
              ? nextModel
              : currentSettings.elevenLabsModel,
          elevenLabsVoiceId:
            nextProvider === "elevenlabs" && nextVoice
              ? nextVoice
              : currentSettings.elevenLabsVoiceId,
          tldwTtsModel:
            nextProvider !== "openai" && nextProvider !== "elevenlabs" && nextProvider !== "browser" && nextModel
              ? nextModel
              : currentSettings.tldwTtsModel,
          tldwTtsVoice:
            nextProvider !== "openai" && nextProvider !== "elevenlabs" && nextProvider !== "browser" && nextVoice
              ? nextVoice
              : currentSettings.tldwTtsVoice,
          tldwTtsResponseFormat: nextFormat ?? currentSettings.tldwTtsResponseFormat,
          tldwTtsSpeed: nextSpeed ?? currentSettings.tldwTtsSpeed,
          tldwTtsLanguage: nextLanguage ?? currentSettings.tldwTtsLanguage,
          tldwTtsStreaming: nextStreaming ?? currentSettings.tldwTtsStreaming,
          tldwTtsEmotion: nextEmotion ?? currentSettings.tldwTtsEmotion,
          tldwTtsEmotionIntensity:
            nextEmotionIntensity ?? currentSettings.tldwTtsEmotionIntensity,
          tldwTtsNormalize: boolOr(
            normalizers.normalize,
            currentSettings.tldwTtsNormalize
          ),
          tldwTtsNormalizeUnits: boolOr(
            normalizers.unit_normalization,
            currentSettings.tldwTtsNormalizeUnits
          ),
          tldwTtsNormalizeUrls: boolOr(
            normalizers.url_normalization,
            currentSettings.tldwTtsNormalizeUrls
          ),
          tldwTtsNormalizeEmails: boolOr(
            normalizers.email_normalization,
            currentSettings.tldwTtsNormalizeEmails
          ),
          tldwTtsNormalizePhones: boolOr(
            normalizers.phone_normalization,
            currentSettings.tldwTtsNormalizePhones
          ),
          tldwTtsNormalizePlurals: boolOr(
            normalizers.optional_pluralization_normalization,
            currentSettings.tldwTtsNormalizePlurals
          )
        })
        await queryClient.invalidateQueries({ queryKey: ["fetchTTSSettings"] })
      } catch (error: unknown) {
        notification.error({
          message: t("playground:tts.presetApplyFailedTitle", "Preset apply failed"),
          description:
            error instanceof Error
              ? error.message
              : t(
                  "playground:tts.presetApplyFailedBody",
                  "Unable to apply this preset. Check settings and try again."
                )
        })
      }
    },
    [
      provider,
      queryClient,
      t,
      tldwNormalize,
      tldwNormalizeEmails,
      tldwNormalizePhones,
      tldwNormalizePlurals,
      tldwNormalizeUnits,
      tldwNormalizeUrls
    ]
  )
  const voiceRoleError = React.useMemo(() => {
    if (!useVoiceRoles) return null
    if (voiceCards.length < 1) return "Select at least one voice."
    if (voiceCards.length > 4) return "Select up to four voices."
    const roles = new Set<string>()
    const voices = new Set<string>()
    for (const card of voiceCards) {
      if (!card.voiceId) return "Each role needs a voice."
      if (roles.has(card.role)) return "Roles must be unique."
      if (voices.has(card.voiceId)) return "Voices must be unique."
      roles.add(card.role)
      voices.add(card.voiceId)
    }
    return null
  }, [useVoiceRoles, voiceCards])

  const voiceRolePayload = React.useMemo(() => {
    if (!useVoiceRoles || voiceRoleError) return null
    return voiceCards.map((card) => ({
      role: card.role,
      voice: card.voiceId
    }))
  }, [useVoiceRoles, voiceRoleError, voiceCards])

  const extraParams = React.useMemo(() => {
    const extras: Record<string, any> = {}
    if (tldwEmotion) extras.emotion = tldwEmotion
    if (typeof tldwEmotionIntensity === "number") {
      extras.emotion_intensity = tldwEmotionIntensity
    }
    if (voiceRolePayload) {
      extras.voice_roles = voiceRolePayload
    }
    return Object.keys(extras).length > 0 ? extras : undefined
  }, [tldwEmotion, tldwEmotionIntensity, voiceRolePayload])

  const setStreamErrorSafe = React.useCallback((message: string | null) => {
    streamErrorRef.current = message
    setStreamError(message)
  }, [])

  const stopStreaming = React.useCallback(() => {
    if (wsRef.current) {
      try {
        wsRef.current.close()
      } catch {}
      wsRef.current = null
    }
    streamMetaRef.current = null
    streamStop()
    setStreamStatus("idle")
    setStreamErrorSafe(null)
    setStreamChunks(0)
    setStreamBytes(0)
  }, [setStreamErrorSafe, streamStop])

  const stopTtsJob = React.useCallback(() => {
    if (ttsJobAbortRef.current) {
      try {
        ttsJobAbortRef.current.abort()
      } catch {}
      ttsJobAbortRef.current = null
    }
    setTtsJobStatus("idle")
    setTtsJobId(null)
    setTtsJobProgress(null)
    setTtsJobMessage(null)
    setTtsJobEta(null)
    setTtsJobError(null)
  }, [])

  const handleStreamPlay = React.useCallback(async () => {
    if (!ttsText.trim()) return
    stopStreaming()
    clearSegments()
    setActiveSegmentIndex(null)
    setCurrentTime(0)
    setDuration(0)
    setStreamStatus("connecting")
    setStreamErrorSafe(null)

    const config = await tldwClient.getConfig()
    const serverUrl = String(config?.serverUrl || "").trim()
    if (!serverUrl) {
      const classified = classifyAudioError("tldw server not configured")
      setStreamStatus("error")
      setStreamErrorSafe(classified.recovery)
      return
    }
    const token =
      config?.authMode === "multi-user"
        ? String(config?.accessToken || "").trim()
        : String(config?.apiKey || "").trim()
    if (!token) {
      const classified = classifyAudioError("Missing authentication token")
      setStreamStatus("error")
      setStreamErrorSafe(classified.recovery)
      return
    }

    const base = serverUrl.replace(/^http/i, "ws").replace(/\/$/, "")
    const wsUrl = `${base}/api/v1/audio/stream/tts?token=${encodeURIComponent(token)}`
    const ws = new WebSocket(wsUrl)
    ws.binaryType = "arraybuffer"
    wsRef.current = ws

    ws.onopen = () => {
      void (async () => {
        const requestedFormat = (tldwFormat || ttsSettings?.tldwTtsResponseFormat || "mp3").toLowerCase()
        const format = STREAMING_FORMATS.has(requestedFormat) ? requestedFormat : "mp3"
        if (format !== requestedFormat) {
          notification.warning({
            message: "Streaming format adjusted",
            description: `WebSocket streaming supports mp3, opus, aac, flac, wav, or pcm. Falling back to ${format.toUpperCase()}.`
          })
        }
        const model =
          tldwModel || ttsSettings?.tldwTtsModel || DEFAULT_TLDW_TTS_MODEL
        const voice =
          tldwVoice || ttsSettings?.tldwTtsVoice || DEFAULT_TLDW_TTS_VOICE
        const speed = ttsSettings?.tldwTtsSpeed ?? 1
        const langCode = tldwLanguage || ttsSettings?.tldwTtsLanguage
        let utterance = markdownToText(ttsText)
        try {
          const ctx = await resolveTtsProviderContext(ttsText, {
            provider: "tldw",
            tldwModel: model,
            tldwVoice: voice,
            tldwResponseFormat: format
          })
          utterance = ctx.utterance
        } catch {
          // fallback to markdownToText
        }
        streamMetaRef.current = {
          provider: "tldw",
          model,
          voice,
          format
        }
        const payload = {
          type: "prompt",
          text: utterance,
          model,
          voice,
          format,
          speed,
          lang_code: langCode || undefined,
          extra_params: extraParams
        }
        streamStart(format, true)
        setStreamStatus("streaming")
        ws.send(JSON.stringify(payload))
      })()
    }

    ws.onmessage = (event) => {
      if (typeof event.data === "string") {
        try {
          const payload = JSON.parse(event.data)
          if (payload?.type === "error") {
            const classified = classifyAudioError(payload?.message || "Streaming error")
            setStreamStatus("error")
            setStreamErrorSafe(classified.recovery)
          }
        } catch {
          // ignore non-JSON status frames
        }
        return
      }
      if (event.data instanceof ArrayBuffer) {
        streamAppend(event.data)
        setStreamChunks((prev) => prev + 1)
        setStreamBytes((prev) => prev + event.data.byteLength)
      }
    }

    ws.onerror = () => {
      const classified = classifyAudioError("Streaming connection error")
      setStreamStatus("error")
      setStreamErrorSafe(classified.recovery)
    }

    ws.onclose = () => {
      streamFinish()
      setStreamStatus((prev) => (prev === "error" ? prev : "complete"))
      const blob = getBufferedBlob()
      if (blob) {
        const url = URL.createObjectURL(blob)
        const format =
          streamMetaRef.current?.format ||
          (tldwFormat || ttsSettings?.tldwTtsResponseFormat || "mp3").toLowerCase()
        const segment = {
          id: `stream-${Date.now()}`,
          index: 0,
          text: ttsText,
          url,
          blob,
          format,
          mimeType: blob.type,
          source: "generated" as const
        }
        setSegments([segment])
        setActiveSegmentIndex(0)
      }
      if (!streamErrorRef.current) {
        const meta = streamMetaRef.current
        const nowIso = new Date().toISOString()
        addHistoryItem({
          id: `${nowIso}-${Math.random().toString(36).slice(2, 8)}`,
          createdAt: nowIso,
          type: "tts",
          text: ttsText,
          provider: meta?.provider || "tldw",
          model: meta?.model,
          voice: meta?.voice,
          format: meta?.format,
          speed: ttsSettings?.tldwTtsSpeed,
          responseSplitting,
          streaming: true,
          mode: "short"
        })
      }
    }
  }, [
    addHistoryItem,
    clearSegments,
    extraParams,
    getBufferedBlob,
    setSegments,
    setStreamErrorSafe,
    stopStreaming,
    streamAppend,
    streamFinish,
    streamStart,
    tldwFormat,
    tldwLanguage,
    tldwModel,
    tldwVoice,
    ttsSettings,
    ttsText,
    responseSplitting
  ])

  const handlePlay = async () => {
    const effectiveText = useDraftEditor ? transcriptDraft : ttsText
    if (!effectiveText.trim() || isTtsDisabled) return
    if (useDraftEditor) {
      const nextErrors: { outline?: string; transcript?: string } = {}
      if (!outlineDraft.trim()) nextErrors.outline = "Outline is required."
      if (!transcriptDraft.trim()) nextErrors.transcript = "Transcript is required."
      setDraftErrors(nextErrors)
      if (nextErrors.outline || nextErrors.transcript) return
    }
    stopStreaming()
    stopTtsJob()
    const effectiveProvider = ttsSettings?.ttsProvider || (await getTTSProvider())
    const shouldStream =
      effectiveProvider === "tldw" &&
      tldwStreaming &&
      Boolean(activeProviderCaps?.caps.supports_streaming) &&
      streamFormatSupported
    if (shouldStream) {
      await handleStreamPlay()
    } else {
      const shouldUseJob = effectiveProvider === "tldw" && useTtsJob
      if (shouldUseJob) {
        clearSegments()
        setActiveSegmentIndex(null)
        setCurrentTime(0)
        setDuration(0)
        setTtsJobStatus("running")
        setTtsJobProgress(0)
        setTtsJobMessage("tts_started")
        setTtsJobEta(null)
        setTtsJobError(null)
        try {
          const model =
            tldwModel || ttsSettings?.tldwTtsModel || DEFAULT_TLDW_TTS_MODEL
          const voice =
            tldwVoice || ttsSettings?.tldwTtsVoice || DEFAULT_TLDW_TTS_VOICE
          const responseFormat = (tldwFormat || ttsSettings?.tldwTtsResponseFormat || "mp3").toLowerCase()
          const speed = ttsSettings?.tldwTtsSpeed ?? 1
          const langCode = tldwLanguage || ttsSettings?.tldwTtsLanguage
          let utterance = markdownToText(effectiveText)
          try {
            const ctx = await resolveTtsProviderContext(effectiveText, {
              provider: "tldw",
              tldwModel: model,
              tldwVoice: voice,
              tldwResponseFormat: responseFormat
            })
            utterance = ctx.utterance
          } catch {
            // fallback to markdownToText
          }
          const job = await tldwClient.createTtsJob({
            input: utterance,
            model,
            voice,
            response_format: responseFormat,
            speed,
            lang_code: langCode || undefined,
            normalization_options: normalizationOptions,
            extra_params: extraParams
          })
          setTtsJobId(job.job_id)
          const controller = new AbortController()
          ttsJobAbortRef.current = controller
          for await (const payload of tldwClient.streamAudioJobProgress(job.job_id, {
            signal: controller.signal,
            streamIdleTimeoutMs: 120000
          })) {
            const eventType = payload?.event
            const attrs = payload?.attrs || {}
            if (eventType === "job.snapshot" || eventType === "job.progress") {
              if (typeof attrs.progress_percent === "number") {
                setTtsJobProgress(attrs.progress_percent)
              }
              if (typeof attrs.progress_message === "string") {
                setTtsJobMessage(attrs.progress_message)
              }
              if (typeof attrs.eta_seconds === "number") {
                setTtsJobEta(attrs.eta_seconds)
              }
              if (eventType === "job.snapshot" && typeof attrs.status === "string") {
                if (attrs.status === "failed" || attrs.status === "cancelled") {
                  setTtsJobStatus("error")
                }
              }
            }
          }
          const artifacts = await tldwClient.getTtsJobArtifacts(job.job_id)
          const first = artifacts?.artifacts?.find((item) => item.type === "tts_audio") || artifacts?.artifacts?.[0]
          if (first?.output_id) {
            const blob = await tldwClient.downloadOutput(String(first.output_id), first.format)
            const url = URL.createObjectURL(blob)
            setSegments([
              {
                id: `tts-job-${job.job_id}`,
                index: 0,
                text: effectiveText,
                url,
                blob,
                format: first.format,
                mimeType: blob.type || "audio/mpeg",
                source: "generated"
              }
            ])
            setActiveSegmentIndex(0)
            const nowIso = new Date().toISOString()
            addHistoryItem({
              id: `${nowIso}-${Math.random().toString(36).slice(2, 8)}`,
              createdAt: nowIso,
              type: "tts",
              text: effectiveText,
              provider: "tldw",
              model,
              voice,
              format: first.format,
              speed,
              responseSplitting,
              streaming: false,
              mode: "long"
            })
            setTtsJobStatus("success")
          } else {
            const classified = classifyAudioError("No audio artifact found for this job.")
            setTtsJobStatus("error")
            setTtsJobError(classified.recovery)
          }
        } catch (error: unknown) {
          const classified = classifyAudioError(error)
          setTtsJobStatus("error")
          setTtsJobError(classified.recovery || "Long-form TTS job failed.")
        } finally {
          ttsJobAbortRef.current = null
        }
        return
      }

      clearSegments()
      setActiveSegmentIndex(null)
      setCurrentTime(0)
      setDuration(0)
      const created = await generateSegments(effectiveText, {
        provider: effectiveProvider,
        elevenLabsModel: elevenModelId,
        elevenLabsVoiceId: elevenVoiceId,
        tldwModel,
        tldwVoice,
        tldwResponseFormat: tldwFormat,
        tldwSpeed: ttsSettings?.tldwTtsSpeed,
        tldwLanguage,
        tldwNormalizationOptions: normalizationOptions,
        tldwExtraParams: extraParams,
        splitBy: responseSplitting,
        openAiModel,
        openAiVoice
      })

      if (created.length > 0) {
        const nowIso = new Date().toISOString()
        addHistoryItem({
          id: `${nowIso}-${Math.random().toString(36).slice(2, 8)}`,
          createdAt: nowIso,
          type: "tts",
          text: effectiveText,
          provider: effectiveProvider,
          model: tldwModel || openAiModel || elevenModelId,
          voice: tldwVoice || openAiVoice || elevenVoiceId,
          format: created[0]?.format,
          speed: ttsSettings?.tldwTtsSpeed,
          responseSplitting,
          streaming: false,
          mode: "short"
        })
      }
    }

    if (ttsSettings) {
      void setTTSSettings({
        ttsEnabled: ttsSettings.ttsEnabled,
        ttsProvider: ttsSettings.ttsProvider,
        voice: ttsSettings.voice,
        ssmlEnabled: ttsSettings.ssmlEnabled,
        elevenLabsApiKey: ttsSettings.elevenLabsApiKey,
        elevenLabsVoiceId: elevenVoiceId ?? ttsSettings.elevenLabsVoiceId,
        elevenLabsModel: elevenModelId ?? ttsSettings.elevenLabsModel,
        responseSplitting,
        removeReasoningTagTTS: ttsSettings.removeReasoningTagTTS,
        openAITTSBaseUrl: ttsSettings.openAITTSBaseUrl,
        openAITTSApiKey: ttsSettings.openAITTSApiKey,
        openAITTSModel: openAiModel ?? ttsSettings.openAITTSModel,
        openAITTSVoice: openAiVoice ?? ttsSettings.openAITTSVoice,
        ttsAutoPlay: ttsSettings.ttsAutoPlay,
        playbackSpeed: ttsSettings.playbackSpeed,
        tldwTtsModel: tldwModel ?? ttsSettings.tldwTtsModel,
        tldwTtsVoice: tldwVoice ?? ttsSettings.tldwTtsVoice,
        tldwTtsResponseFormat: tldwFormat ?? ttsSettings.tldwTtsResponseFormat,
        tldwTtsSpeed: ttsSettings.tldwTtsSpeed,
        tldwTtsLanguage: tldwLanguage ?? ttsSettings.tldwTtsLanguage,
        tldwTtsStreaming: tldwStreaming,
        tldwTtsEmotion: tldwEmotion ?? ttsSettings.tldwTtsEmotion,
        tldwTtsEmotionIntensity:
          typeof tldwEmotionIntensity === "number"
            ? tldwEmotionIntensity
            : ttsSettings.tldwTtsEmotionIntensity,
        tldwTtsNormalize: tldwNormalize,
        tldwTtsNormalizeUnits: tldwNormalizeUnits,
        tldwTtsNormalizeUrls: tldwNormalizeUrls,
        tldwTtsNormalizeEmails: tldwNormalizeEmails,
        tldwTtsNormalizePhones: tldwNormalizePhones,
        tldwTtsNormalizePlurals: tldwNormalizePlurals
      }).then(() => {
        queryClient.invalidateQueries({ queryKey: ["fetchTTSSettings"] })
      })
    }

    setActiveSegmentIndex(0)
  }

  const handleStop = () => {
    stopStreaming()
    stopTtsJob()
    const el = audioRef.current
    if (el) {
      el.pause()
      el.currentTime = 0
    }
    setCurrentTime(0)
    setDuration(0)
  }

  const providerLabel = getTtsProviderLabel(provider)

  const activeProviderCaps = React.useMemo(
    (): { key: string; caps: TldwTtsProviderCapabilities } | null => {
      if (!providersInfo || !inferredProviderKey) return null
      const entries = Object.entries(providersInfo.providers || {})
      const target = normalizeTtsProviderKey(inferredProviderKey)
      const match = entries.find(
        ([k]) => normalizeTtsProviderKey(k) === target
      )
      if (!match) return null
      return { key: match[0], caps: match[1] }
    },
    [providersInfo, inferredProviderKey]
  )

  const activeVoices = React.useMemo((): TldwTtsVoiceInfo[] => {
    if (tldwVoiceCatalog && tldwVoiceCatalog.length > 0) {
      return tldwVoiceCatalog.slice(0, 4)
    }
    if (!providersInfo || !activeProviderCaps) return []
    const allVoices = providersInfo.voices || {}
    const direct = allVoices[activeProviderCaps.key]
    if (Array.isArray(direct) && direct.length > 0) {
      return direct.slice(0, 4)
    }
    const fallbackKey = normalizeTtsProviderKey(activeProviderCaps.key)
    const fallback = Object.entries(allVoices).find(
      ([key]) => normalizeTtsProviderKey(key) === fallbackKey
    )?.[1]
    if (Array.isArray(fallback) && fallback.length > 0) {
      return fallback.slice(0, 4)
    }
    return []
  }, [providersInfo, activeProviderCaps, tldwVoiceCatalog])

  const providerVoices = React.useMemo((): TldwTtsVoiceInfo[] => {
    if (tldwVoiceCatalog && tldwVoiceCatalog.length > 0) {
      return tldwVoiceCatalog
    }
    if (!providersInfo || !activeProviderCaps) return []
    const allVoices = providersInfo.voices || {}
    const direct = allVoices[activeProviderCaps.key]
    if (Array.isArray(direct) && direct.length > 0) {
      return direct
    }
    const fallbackKey = normalizeTtsProviderKey(activeProviderCaps.key)
    const fallback = Object.entries(allVoices).find(
      ([key]) => normalizeTtsProviderKey(key) === fallbackKey
    )?.[1]
    if (Array.isArray(fallback) && fallback.length > 0) {
      return fallback
    }
    return []
  }, [providersInfo, activeProviderCaps, tldwVoiceCatalog])

  const customVoiceOptions = React.useMemo<DefaultOptionType[]>(() => {
    if (!customVoices || customVoices.length === 0) return []
    return customVoices
      .filter((voice) => Boolean(voice.voice_id))
      .map((voice) => ({
        label: `Custom: ${voice.name || voice.voice_id}`,
        value: `custom:${voice.voice_id}`
      }))
  }, [customVoices])

  const tldwVoiceOptions = React.useMemo<DefaultOptionType[]>(() => {
    const providerPrefix = customVoiceOptions.length > 0 ? "Server: " : ""
    const providerOptions: DefaultOptionType[] = providerVoices.map((v, idx) => ({
      label: `${providerPrefix}${v.name || v.id || `Voice ${idx + 1}`}${v.language ? ` (${v.language})` : ""}`,
      value: v.id || v.name || ""
    }))
    if (customVoiceOptions.length === 0) return providerOptions
    return [...customVoiceOptions, ...providerOptions]
  }, [providerVoices, customVoiceOptions])

  React.useEffect(() => {
    if (!useVoiceRoles) return
    setVoiceCards((prev) => {
      if (prev.length > 0) return prev
      const defaultVoice =
        tldwVoice ||
        ttsSettings?.tldwTtsVoice ||
        (tldwVoiceOptions[0]?.value as string | undefined) ||
        ""
      return [
        {
          id: `voice-${Date.now()}`,
          role: DEFAULT_VOICE_ROLE,
          voiceId: defaultVoice
        }
      ]
    })
  }, [useVoiceRoles, tldwVoice, ttsSettings?.tldwTtsVoice, tldwVoiceOptions])

  const openAiVoiceOptions = React.useMemo(() => {
    if (!openAiModel) {
      const seen = new Set<string>()
      const all: { label: string; value: string }[] = []
      Object.values(OPENAI_TTS_VOICES).forEach((list) => {
        list.forEach((v) => {
          if (!seen.has(v.value)) {
            seen.add(v.value)
            all.push(v)
          }
        })
      })
      return all
    }
    return OPENAI_TTS_VOICES[openAiModel] || []
  }, [openAiModel])

  const tldwFormatOptions = React.useMemo(() => {
    const formats =
      activeProviderCaps?.caps.formats?.length
        ? activeProviderCaps.caps.formats
        : SUPPORTED_TLDW_TTS_FORMATS
    const unique = Array.from(
      new Set(formats.map((fmt) => String(fmt).toLowerCase()))
    )
    return unique.map((fmt) => ({
      label: fmt === "pcm" ? "pcm (raw)" : fmt,
      value: fmt
    }))
  }, [activeProviderCaps])

  const tldwLanguageOptions = React.useMemo(() => {
    const languages = activeProviderCaps?.caps.languages || []
    if (!languages.length) return []
    const labelMap: Record<string, string> = {
      en: "English",
      es: "Spanish",
      fr: "French",
      de: "German",
      it: "Italian",
      pt: "Portuguese",
      ru: "Russian",
      ja: "Japanese",
      ko: "Korean",
      zh: "Chinese",
      ar: "Arabic",
      hi: "Hindi",
      pl: "Polish"
    }
    return Array.from(new Set(languages)).map((lang) => ({
      label: labelMap[String(lang)] ? `${labelMap[String(lang)]} (${lang})` : String(lang),
      value: String(lang)
    }))
  }, [activeProviderCaps])

  const canStream = Boolean(isTldw && activeProviderCaps?.caps.supports_streaming)
  const applyTtsPreset = React.useCallback(
    async (presetKey: TtsPresetKey) => {
      const preset = TTS_PRESETS[presetKey]
      if (!preset) return
      setTtsPreset(presetKey)
      if (isTldw) {
        const availableFormats: string[] =
          activeProviderCaps?.caps.formats?.map((fmt) => String(fmt).toLowerCase()) ||
          [...SUPPORTED_TLDW_TTS_FORMATS]
        const nextFormat = availableFormats.includes(preset.responseFormat)
          ? preset.responseFormat
          : (availableFormats[0] || "mp3")
        setTldwFormat(nextFormat)
        setTldwStreaming(Boolean(preset.streaming) && canStream && STREAMING_FORMATS.has(nextFormat))
        setResponseSplitting(preset.splitBy)
        setTldwEmotionIntensity(1)
        setTldwNormalize(true)
        try {
          await Promise.all([
            setTldwTTSSpeed(preset.speed),
            setTldwTTSResponseFormat(nextFormat),
            setTldwTTSStreamingEnabled(Boolean(preset.streaming) && STREAMING_FORMATS.has(nextFormat)),
            persistResponseSplitting(preset.splitBy)
          ])
          queryClient.invalidateQueries({ queryKey: ["fetchTTSSettings"] })
        } catch {
          // ignore preset persistence failures
        }
      }
    },
    [
      activeProviderCaps,
      canStream,
      isTldw,
      persistResponseSplitting,
      setResponseSplitting,
      setTldwEmotionIntensity,
      setTldwFormat,
      setTldwNormalize,
      setTldwStreaming,
      setTtsPreset,
      queryClient
    ]
  )

  const [inspectorOpen, setInspectorOpen] = useStorage<boolean>("ttsInspectorOpen", false)
  const [inspectorTab, setInspectorTab] = useStorage<"voice" | "output" | "advanced">("ttsInspectorTab", "voice")
  const [inspectorFocusField, setInspectorFocusField] = React.useState<string | null>(null)

  const openInspectorAt = React.useCallback(
    (tab: "voice" | "output" | "advanced", field?: string) => {
      setInspectorOpen(true)
      setInspectorTab(tab)
      if (field) setInspectorFocusField(field)
    },
    [setInspectorOpen, setInspectorTab]
  )

  // Responsive: use drawer below 1024px
  const [useDrawerMode, setUseDrawerMode] = React.useState(false)
  React.useEffect(() => {
    const mq = window.matchMedia("(max-width: 1023px)")
    const handler = (e: MediaQueryListEvent | MediaQueryList) => setUseDrawerMode(e.matches)
    handler(mq)
    mq.addEventListener("change", handler)
    return () => mq.removeEventListener("change", handler)
  }, [])

  const requestedStreamFormat = (tldwFormat || ttsSettings?.tldwTtsResponseFormat || "mp3").toLowerCase()
  const streamFormatSupported = STREAMING_FORMATS.has(requestedStreamFormat)

  const normalizedPreviewText = React.useMemo(
    () => markdownToText(useDraftEditor ? transcriptDraft : ttsText),
    [useDraftEditor, transcriptDraft, ttsText]
  )
  const previewSegments = React.useMemo(
    () => splitMessageContent(normalizedPreviewText, responseSplitting),
    [normalizedPreviewText, responseSplitting]
  )
  const previewWordCount = React.useMemo(() => {
    const parts = normalizedPreviewText.trim().split(/\s+/).filter(Boolean)
    return parts.length
  }, [normalizedPreviewText])
  const previewCharCount = normalizedPreviewText.length
  const estimatedDurationSeconds = React.useMemo(
    () => estimateTtsDurationSeconds(normalizedPreviewText, TTS_ESTIMATE_CHARS_PER_SEC),
    [normalizedPreviewText]
  )
  const formatDuration = (seconds: number) => {
    if (!seconds || seconds <= 0) return "0s"
    const total = Math.round(seconds)
    const minutes = Math.floor(total / 60)
    const secs = total % 60
    if (!minutes) return `${secs}s`
    return `${minutes}m ${secs}s`
  }
  const [showSegmentsPreview, setShowSegmentsPreview] = React.useState(false)

  const hasElevenLabsKey = Boolean(ttsSettings?.elevenLabsApiKey)
  const selectedTldwProviderLabel =
    activeProviderCaps?.caps.provider_name || inferredProviderKey || providerLabel
  const selectedTldwProviderMissing =
    isTldw &&
    hasAudio &&
    Boolean(providersInfo) &&
    Boolean(inferredProviderKey) &&
    !activeProviderCaps
  const selectedTldwVoiceCatalogMissing =
    isTldw &&
    hasAudio &&
    Boolean(activeProviderCaps) &&
    tldwVoiceOptions.length === 0
  const elevenLabsCatalogEmpty =
    provider === "elevenlabs" &&
    !!elevenLabsData &&
    (!elevenLabsData.voices?.length || !elevenLabsData.models?.length)

  const playDisabledReason = (() => {
    if (isTtsDisabled) {
      return t(
        "playground:tts.playDisabledTtsOff",
        "Enable text-to-speech above to play audio."
      )
    }
    if (!(useDraftEditor ? transcriptDraft : ttsText).trim()) {
      return t("playground:tts.playDisabledNoText", "Enter text to enable Play.")
    }
    if (provider !== "browser" && provider !== "elevenlabs" && !hasAudio) {
      return t(
        "playground:tts.playDisabledServerAudioUnavailable",
        "Open Settings -> Speech to connect the tldw audio/speech API before generating server TTS."
      )
    }
    if (selectedTldwProviderMissing) {
      return t(
        "playground:tts.playDisabledProviderMissing",
        "The selected TTS provider is not reported by the server. Open Settings -> Speech and choose a configured provider."
      )
    }
    if (selectedTldwVoiceCatalogMissing) {
      return withTemplateFallback(
        t(
          "playground:tts.playDisabledVoiceCatalogMissing",
          "No voices reported for {{provider}}. Configure voices in Settings -> Speech before generating.",
          { provider: selectedTldwProviderLabel } as any
        ),
        `No voices reported for ${selectedTldwProviderLabel}. Configure voices in Settings -> Speech before generating.`
      )
    }
    if (provider === "elevenlabs" && !hasElevenLabsKey) {
      return t(
        "playground:tts.playDisabledElevenLabsKeyMissing",
        "Enter an ElevenLabs API key before generating audio."
      )
    }
    if (provider === "elevenlabs" && elevenLabsLoading) {
      return t(
        "playground:tts.playDisabledElevenLabsLoading",
        "Loading ElevenLabs voices and models before generation."
      )
    }
    if (provider === "elevenlabs" && elevenLabsError) {
      return isTimeoutLikeError(elevenLabsError)
        ? t(
            "playground:tts.playDisabledElevenLabsTimeout",
            "Retry ElevenLabs voice/model loading; the last request timed out."
          )
        : t(
            "playground:tts.playDisabledElevenLabsError",
            "Retry ElevenLabs voice/model loading before generating audio."
          )
    }
    if (elevenLabsCatalogEmpty) {
      return t(
        "playground:tts.playDisabledElevenLabsCatalogEmpty",
        "No ElevenLabs voices or models are available. Check your ElevenLabs account or API key."
      )
    }
    if (voiceRoleError) return voiceRoleError
    return draftErrors.outline || draftErrors.transcript || null
  })()
  const isStreamingActive = streamStatus === "connecting" || streamStatus === "streaming"
  const isTtsJobRunning = ttsJobStatus === "running"
  const isPlayDisabled = isGenerating || isStreamingActive || isTtsJobRunning || Boolean(playDisabledReason)
  const canStop = Boolean(segments.length || audioRef.current || isStreamingActive || isTtsJobRunning)
  const stopDisabledReason =
    !canStop && t("playground:tts.stopDisabled", "Stop activates after audio starts.")
  const showElevenLabsHint =
    provider === "elevenlabs" &&
    !elevenLabsData &&
    !elevenLabsLoading
  const hasElevenLabsLoadError =
    hasElevenLabsKey && Boolean(elevenLabsError)
  const elevenLabsHintTitle = hasElevenLabsKey
    ? t(
        "playground:tts.elevenLabsUnavailableTitle",
        "ElevenLabs voices unavailable"
      )
    : t(
        "playground:tts.elevenLabsMissingTitle",
        "ElevenLabs needs an API key"
      )
  const elevenLabsHintBody = hasElevenLabsKey
    ? t(
        "playground:tts.elevenLabsUnavailableBody",
        "We couldn't load voices or models. Check your API key and try again."
      )
    : t(
        "playground:tts.elevenLabsMissingBody",
        "Enter your ElevenLabs API key below to load voices and models. You can also manage it in Settings."
      )
  const elevenLabsTimeoutBody = t(
    "playground:tts.elevenLabsTimeoutBody",
    "Loading voices/models took longer than 10 seconds. Retry or verify network access."
  )
  const activeStreamError = streamError || streamState.error
  const streamStatusLabel = React.useMemo(() => {
    switch (streamStatus) {
      case "connecting":
        return "Connecting..."
      case "streaming":
        return streamState.mode === "stream" ? "Streaming..." : "Buffering..."
      case "complete":
        return "Stream complete"
      case "error":
        return "Streaming error"
      default:
        return "Idle"
    }
  }, [streamState.mode, streamStatus])
  const ttsJobStepIndex = React.useMemo(() => {
    if (ttsJobMessage && TTS_JOB_STEP_INDEX[ttsJobMessage] != null) {
      return TTS_JOB_STEP_INDEX[ttsJobMessage]
    }
    if (ttsJobStatus === "success") return TTS_JOB_STEPS.length - 1
    return 0
  }, [ttsJobMessage, ttsJobStatus])
  const streamStatusColor =
    streamStatus === "error"
      ? "red"
      : streamStatus === "complete"
        ? "green"
        : streamStatus === "streaming"
        ? "blue"
        : "default"

  const inspectorBadge = React.useMemo((): "none" | "gray" | "amber" | "red" => {
    if (inspectorOpen) return "none"
    if (isTtsDisabled) return "red"
    if (isTldw && !hasAudio) return "red"
    if (showElevenLabsHint && !hasElevenLabsKey) return "red"
    if (showElevenLabsHint && hasElevenLabsKey) return "amber"
    return "gray"
  }, [inspectorOpen, isTtsDisabled, isTldw, hasAudio, showElevenLabsHint, hasElevenLabsKey])

  // Keyboard shortcuts: Ctrl/Cmd+Enter (play/stop), Escape (stop), Ctrl/Cmd+. (toggle inspector)
  React.useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (effectiveMode === "speak") return
      const mod = e.metaKey || e.ctrlKey
      if (mod && e.key === "Enter") {
        e.preventDefault()
        if (isStreamingActive || isTtsJobRunning || segments.length > 0) {
          handleStop()
        } else if (!isPlayDisabled) {
          void handlePlay()
        }
      }
      if (e.key === "Escape") {
        handleStop()
      }
      if (mod && e.key === ".") {
        e.preventDefault()
        setInspectorOpen((prev) => !prev)
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [
    effectiveMode,
    handlePlay,
    handleStop,
    isPlayDisabled,
    isStreamingActive,
    isTtsJobRunning,
    segments.length,
    setInspectorOpen
  ])

  const handleInlineElevenLabsSave = React.useCallback(async () => {
    const trimmedKey = inlineElevenLabsApiKey.trim()
    if (!trimmedKey) {
      setInlineElevenLabsResult({
        ok: false,
        message: t(
          "playground:tts.elevenLabsInlineKeyRequired",
          "Enter an ElevenLabs API key first."
        ) as string
      })
      return
    }

    setInlineElevenLabsSaving(true)
    setInlineElevenLabsResult(null)
    try {
      const [voices, models] = await Promise.all([
        getElevenLabsVoices(trimmedKey, {
          timeoutMs: ELEVENLABS_INLINE_TEST_TIMEOUT_MS
        }),
        getElevenLabsModels(trimmedKey, {
          timeoutMs: ELEVENLABS_INLINE_TEST_TIMEOUT_MS
        })
      ])
      if (
        !Array.isArray(voices) ||
        voices.length === 0 ||
        !Array.isArray(models) ||
        models.length === 0
      ) {
        setInlineElevenLabsResult({
          ok: false,
          message: t(
            "playground:tts.elevenLabsInlineKeyNoResources",
            "API key accepted, but no voices or models were returned."
          ) as string
        })
        return
      }

      const currentSettings = await getTTSSettings()
      await setTTSSettings({
        ...currentSettings,
        elevenLabsApiKey: trimmedKey
      })
      await queryClient.invalidateQueries({ queryKey: ["fetchTTSSettings"] })
      const savedMessage = t(
        "playground:tts.elevenLabsInlineKeySaved",
        "API key saved. ElevenLabs voices can now load."
      ) as string
      notification.success({ message: savedMessage })
      setInlineElevenLabsResult(null)
    } catch (error: unknown) {
      const status =
        typeof error === "object" && error !== null
          ? Number((error as { response?: { status?: unknown } }).response?.status)
          : NaN
      const code =
        typeof error === "object" && error !== null
          ? String((error as { code?: unknown }).code ?? "")
          : ""
      const message = isTimeoutLikeError(error)
        ? t(
            "playground:tts.elevenLabsInlineKeyTimeout",
            "Request timed out while validating the API key."
          )
        : status === 401 || status === 403
          ? t(
              "playground:tts.elevenLabsInlineKeyInvalid",
              "Invalid ElevenLabs API key. Check the key and try again."
            )
          : code === "ERR_NETWORK" || code === "ENOTFOUND" || code === "ECONNREFUSED"
            ? t(
                "playground:tts.elevenLabsInlineKeyNetworkError",
                "Network error while validating the API key. Check your connection and try again."
              )
            : t(
                "playground:tts.elevenLabsInlineKeyFailed",
                "Unable to validate the ElevenLabs API key."
              )
      setInlineElevenLabsResult({
        ok: false,
        message: message as string
      })
    } finally {
      setInlineElevenLabsSaving(false)
    }
  }, [inlineElevenLabsApiKey, queryClient, t])

  const handleAddVoiceCard = () => {
    if (voiceCards.length >= 4) return
    setVoiceCards((prev) => [
      ...prev,
      {
        id: `voice-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        role: VOICE_ROLE_OPTIONS[Math.min(prev.length, VOICE_ROLE_OPTIONS.length - 1)].value,
        voiceId: ""
      }
    ])
  }

  const handleRemoveVoiceCard = (id: string) => {
    setVoiceCards((prev) => prev.filter((card) => card.id !== id))
  }

  const handleUpdateVoiceCard = (
    id: string,
    updates: Partial<VoiceRoleCard>
  ) => {
    setVoiceCards((prev) =>
      prev.map((card) => (card.id === id ? { ...card, ...updates } : card))
    )
  }

  const resolvePreviewModel = (voiceId: string) => {
    if (!voiceId)
      return tldwModel || ttsSettings?.tldwTtsModel || DEFAULT_TLDW_TTS_MODEL
    if (voiceId.startsWith("custom:")) {
      const key = voiceId.replace("custom:", "")
      const match = customVoices.find((voice) => voice.voice_id === key)
      if (match?.provider) {
        return match.provider
      }
    }
    return tldwModel || ttsSettings?.tldwTtsModel || DEFAULT_TLDW_TTS_MODEL
  }

  const handleVoicePreview = async (card: VoiceRoleCard) => {
    if (!card.voiceId) return
    setVoicePreviewingId(card.id)
    try {
      const text =
        voicePreviewText.trim() || "Hello, this is a preview of the selected voice."
      const model = resolvePreviewModel(card.voiceId)
      const buffer = await tldwClient.synthesizeSpeech(text, {
        model,
        voice: card.voiceId,
        responseFormat: "mp3"
      })
      const blob = new Blob([buffer], { type: "audio/mpeg" })
      const url = URL.createObjectURL(blob)
      if (voicePreviewUrl) {
        try {
          URL.revokeObjectURL(voicePreviewUrl)
        } catch {}
      }
      setVoicePreviewUrl(url)
      setVoicePreviewCardId(card.id)
    } catch (error: unknown) {
      const classified = classifyAudioError(error)
      notification.error({
        message: "Preview failed",
        description: classified.recovery || "Unable to generate preview audio."
      })
    } finally {
      setVoicePreviewingId(null)
    }
  }

  const downloadBlob = React.useCallback((blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = filename
    link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }, [])

  const handleCopy = React.useCallback(
    async (text: string) => {
      try {
        await copyToClipboard({ text })
        notification.success({
          message: t("playground:speech.copySuccess", "Copied to clipboard")
        })
      } catch (e: any) {
        notification.error({
          message: t("error", "Error"),
          description:
            e?.message ||
            t("playground:speech.copyError", "Failed to copy transcript.")
        })
      }
    },
    [t]
  )

  const handleDownloadSegment = React.useCallback(
    (segmentIndex?: number) => {
      const idx = typeof segmentIndex === "number" ? segmentIndex : activeSegmentIndex ?? 0
      const seg = segments[idx]
      if (!seg) return
      const stamp = new Date().toISOString().replace(/[:.]/g, "-")
      const base = `speech-tts-${stamp}-${provider}`
      const filename = `${base}-part-${idx + 1}.${seg.format || "mp3"}`
      downloadBlob(seg.blob, filename)
    },
    [activeSegmentIndex, downloadBlob, provider, segments]
  )

  const handleDownloadAll = React.useCallback(() => {
    segments.forEach((seg, idx) => handleDownloadSegment(idx))
  }, [handleDownloadSegment, segments])

  const handleReplayHistoryItem = React.useCallback(
    async (item: SpeechHistoryItem) => {
      if (item.type !== "tts") return
      const providerToUse =
        item.provider || ttsSettings?.ttsProvider || DEFAULT_TTS_PROVIDER
      setTtsText(item.text)
      stopStreaming()
      clearSegments()
      setActiveSegmentIndex(null)
      setCurrentTime(0)
      setDuration(0)

      if (providerToUse === "tldw") {
        if (item.model) setTldwModel(item.model)
        if (item.voice) setTldwVoice(item.voice)
        if (item.format) setTldwFormat(item.format)
      } else if (providerToUse === "openai") {
        if (item.model) setOpenAiModel(item.model)
        if (item.voice) setOpenAiVoice(item.voice)
      } else if (providerToUse === "elevenlabs") {
        if (item.model) setElevenModelId(item.model)
        if (item.voice) setElevenVoiceId(item.voice)
      }

      const created = await generateSegments(item.text, {
        provider: providerToUse,
        elevenLabsModel: item.model || elevenModelId,
        elevenLabsVoiceId: item.voice || elevenVoiceId,
        tldwModel: item.model || tldwModel,
        tldwVoice: item.voice || tldwVoice,
        tldwResponseFormat: item.format || tldwFormat,
        tldwSpeed: ttsSettings?.tldwTtsSpeed,
        tldwLanguage: tldwLanguage,
        tldwNormalizationOptions: normalizationOptions,
        tldwExtraParams: extraParams,
        splitBy: responseSplitting,
        openAiModel: item.model || openAiModel,
        openAiVoice: item.voice || openAiVoice
      })

      if (created.length > 0) {
        setActiveSegmentIndex(0)
      }
    },
    [
      clearSegments,
      extraParams,
      generateSegments,
      elevenModelId,
      elevenVoiceId,
      normalizationOptions,
      openAiModel,
      openAiVoice,
      responseSplitting,
      stopStreaming,
      tldwFormat,
      tldwLanguage,
      tldwModel,
      tldwVoice,
      ttsSettings?.ttsProvider,
      ttsSettings?.tldwTtsSpeed
    ]
  )

  const downloadMenu = {
    items: [
      {
        key: "download-active",
        label: t("playground:speech.downloadCurrent", "Download current segment"),
        disabled: segments.length === 0 || provider === "browser"
      },
      {
        key: "download-all",
        label: t("playground:speech.downloadAll", "Download all segments"),
        disabled: segments.length <= 1 || provider === "browser"
      }
    ],
    onClick: ({ key }: { key: string }) => {
      if (key === "download-active") handleDownloadSegment()
      if (key === "download-all") handleDownloadAll()
    }
  }

  const downloadDisabledReason =
    provider === "browser"
      ? t(
          "playground:speech.downloadDisabledBrowser",
          "Browser TTS does not create downloadable audio."
        )
      : isStreamingActive
        ? t(
            "playground:speech.downloadDisabledStreaming",
            "Downloads unlock after streaming completes."
          )
        : segments.length === 0
        ? t(
            "playground:speech.downloadDisabledEmpty",
            "Generate audio to enable downloads."
          )
        : null

  const handleSendToTts = () => {
    const text = liveText.trim() ? liveText : lastTranscript
    if (!text.trim()) return
    if (useDraftEditor) {
      setTranscriptDraft(text)
    } else {
      setTtsText(text)
    }
  }

  const pageTitle = isLockedTtsRoute
    ? t("playground:speech.textToSpeechRouteTitle", "Text to Speech")
    : t("playground:speech.title", "Speech Playground")
  const pageSubtitle = isLockedTtsRoute
    ? t(
        "playground:speech.ttsRouteSubtitle",
        "Draft text, choose a voice, and generate audio in one place."
      )
    : t(
        "playground:speech.subtitle",
        "Record speech, edit transcripts, and synthesize audio in one place."
      )
  const historyTitle = isLockedTtsRoute
    ? t("playground:speech.ttsHistoryTitle", "TTS history")
    : t("playground:speech.historyTitle", "Speech history")
  const historyEmptyState = isLockedTtsRoute
    ? t("playground:speech.ttsEmptyHistory", "Generate audio to see TTS history here.")
    : t(
        "playground:speech.emptyHistory",
        "Start a recording or generate audio to see history here."
      )

  return (
    <PageShell maxWidthClassName="max-w-5xl" className="py-6">
      <Title level={1} className="!mb-1 !text-2xl">{pageTitle}</Title>
      <Text type="secondary">{pageSubtitle}</Text>

      <div className="mt-4 space-y-4">
        {!hideModeSwitcher && (
          <Card>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <Space orientation="vertical" size={2}>
                <Text strong>{t("playground:speech.modeLabel", "Mode")}</Text>
                <Segmented
                  value={effectiveMode}
                  onChange={(value) => handleModeChange(value as SpeechMode)}
                  options={[
                    { label: t("playground:speech.modeRoundTrip", "Round-trip"), value: "roundtrip" },
                    { label: t("playground:speech.modeSpeak", "Speak"), value: "speak" },
                    { label: t("playground:speech.modeListen", "Listen"), value: "listen" }
                  ]}
                />
              </Space>
              <Text type="secondary" className="text-xs">
                {t(
                  "playground:speech.modeHint",
                  "Your last mode is remembered for this device."
                )}
              </Text>
            </div>
          </Card>
        )}

        <div className={effectiveMode === "roundtrip" ? "grid gap-4 lg:grid-cols-2" : "space-y-4"}>
          {effectiveMode !== "listen" && (
            <Card className="h-full">
              <Space orientation="vertical" className="w-full" size="middle">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="space-y-1">
                    <Text strong>
                      {t("playground:stt.currentModelLabel", "Current transcription model")}:
                    </Text>
                    <div className="flex flex-wrap items-center gap-2">
                      <Select
                        showSearch
                        allowClear
                        placeholder={sttModel || "whisper-1"}
                        loading={serverModelsLoading}
                        value={activeModel}
                        onChange={(value) => setActiveModel(value)}
                        style={{ minWidth: 220 }}
                        options={serverModels.map((m) => ({ label: m, value: m }))}
                      />
                      {sttModel && (
                        <Tag bordered>
                          {t("playground:stt.defaultModel", "Default from Settings")}:{" "}
                          <Text code className="ml-1">
                            {sttModel}
                          </Text>
                        </Tag>
                      )}
                    </div>
                    <div className="text-xs text-text-subtle">
                      {t(
                        "playground:stt.settingsNotice",
                        "Language, task, response format, segmentation, and prompt reuse your Speech-to-Text defaults from Settings."
                      )}
                    </div>
                    {serverModelsError && (
                      <DesignSystemAlert
                        variant="warning"
                        title={t("playground:stt.modelsLoadError", "Model load failed")}
                        action={{
                          label: t("common:retry", "Retry"),
                          onClick: retryServerModels,
                          disabled: serverModelsLoading
                        }}
                      >
                        {serverModelsError}
                      </DesignSystemAlert>
                    )}
                  </div>
                  <div className="flex flex-col gap-3">
                    <div className="space-y-1">
                      <Text type="secondary" className="block text-xs">
                        {t("playground:stt.sessionMode", "Session mode")}
                      </Text>
                      <div className="flex items-center gap-2">
                        <Switch checked={useLongRunning} onChange={setUseLongRunning} size="small" />
                        <span className="text-xs text-text-muted">
                          {useLongRunning
                            ? t("playground:stt.modeLong", "Long-running (chunked recording)")
                            : t("playground:stt.modeShort", "Short dictation (single clip)")}
                        </span>
                      </div>
                    </div>
                    <div className="space-y-1">
                      <Text type="secondary" className="block text-xs">
                        {t("playground:stt.sourcePickerTitle", "Input source")}
                      </Text>
                      <AudioSourcePicker
                        ariaLabel={t(
                          "playground:stt.sourcePickerLabel",
                          "Speech playground input source"
                        )}
                        className="min-w-[240px]"
                        devices={audioInputDevices}
                        requestedSourceKind={audioSourcePreference.sourceKind}
                        resolvedSourceKind={resolvedAudioSource.sourceKind}
                        requestedDeviceId={audioSourcePreference.deviceId}
                        lastKnownLabel={audioSourcePreference.lastKnownLabel}
                        onChange={(nextValue) =>
                          setAudioSourcePreference({
                            featureGroup: "speech_playground",
                            sourceKind: nextValue.sourceKind,
                            deviceId: nextValue.deviceId ?? null,
                            lastKnownLabel: nextValue.lastKnownLabel ?? null
                          })
                        }
                      />
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between gap-3 pt-2">
                  <div className="flex flex-wrap items-center gap-3">
                    <Tag color="blue" bordered>
                      {t("playground:stt.languageTag", "Language")}:{" "}
                      <Text code className="ml-1">
                        {speechToTextLanguage || "auto"}
                      </Text>
                    </Tag>
                    <Tag bordered>
                      {t("playground:stt.taskTag", "Task")}{" "}
                      <Text code className="ml-1">
                        {sttTask || "transcribe"}
                      </Text>
                    </Tag>
                    <Tag bordered>
                      {t("playground:stt.formatTag", "Format")}{" "}
                      <Text code className="ml-1">
                        {sttResponseFormat || "json"}
                      </Text>
                    </Tag>
                    {sttUseSegmentation && (
                      <Tag color="purple" bordered>
                        {t("playground:stt.segmentationEnabled", "Segmentation enabled")}
                      </Tag>
                    )}
                  </div>
                  <Tooltip
                    placement="left"
                    title={
                      isTranscribing
                        ? (t("playground:stt.transcribingTooltip", "Transcribing audio...") as string)
                        : isRecording
                          ? (t("playground:stt.stopTooltip", "Stop and send to server") as string)
                          : (t(
                              "playground:stt.startTooltip",
                              "Start recording audio for transcription"
                            ) as string)
                    }
                  >
                    <Button
                      aria-label={
                        isRecording
                          ? t("playground:stt.stopAriaLabel", "Stop dictation")
                          : isTranscribing
                            ? t("playground:stt.transcribingAriaLabel", "Transcribing dictation")
                            : t("playground:stt.startAriaLabel", "Start dictation")
                      }
                      type={isRecording || isTranscribing ? "default" : "primary"}
                      danger={isRecording}
                      loading={isTranscribing}
                      disabled={isTranscribing}
                      icon={
                        isRecording ? (
                          <Pause className="h-4 w-4" />
                        ) : !isTranscribing ? (
                          <Mic className="h-4 w-4" />
                        ) : undefined
                      }
                      onClick={handleStartRecording}
                    >
                      {isRecording
                        ? t("playground:stt.stopButton", "Stop")
                        : isTranscribing
                          ? t("playground:stt.transcribingButton", "Transcribing...")
                          : t("playground:stt.recordButton", "Record")}
                    </Button>
                  </Tooltip>
                </div>

                <div className="text-xs text-text-subtle">
                  {withTemplateFallback(
                    t(
                      "playground:tooltip.speechToTextDetails",
                      "Uses {{model}} · {{task}} · {{format}}. Configure in Settings -> Speech (/settings/speech).",
                      {
                        model: activeModel || sttModel || "whisper-1",
                        task: sttTask === "translate" ? "translate" : "transcribe",
                        format: (sttResponseFormat || "json").toUpperCase()
                      } as any
                    ),
                    `Uses ${
                      activeModel || sttModel || "whisper-1"
                    } · ${sttTask === "translate" ? "translate" : "transcribe"} · ${(
                      sttResponseFormat || "json"
                    ).toUpperCase()}. Configure in Settings -> Speech (/settings/speech).`
                  )}
                </div>

                <WaveformCanvas
                  stream={recordingStream}
                  active={isRecording || isTranscribing}
                  label={t("playground:speech.recordingWaveform", "Live recording waveform") as string}
                />

                {recordingError && (
                  <Text type="danger" className="text-xs">
                    {recordingError}
                  </Text>
                )}

                {(liveText || isRecording || isTranscribing) && (
                  <div className="pt-1">
                    <div className="flex items-center justify-between gap-2 mb-1">
                      <Text strong className="text-xs block">
                        {t("playground:stt.currentTranscriptTitle", "Current session transcript")}
                      </Text>
                      <Button
                        size="small"
                        type="text"
                        icon={transcriptLocked ? <Unlock className="h-3.5 w-3.5" /> : <Lock className="h-3.5 w-3.5" />}
                        onClick={() => setTranscriptLocked((prev) => !prev)}
                        disabled={isRecording || isTranscribing}
                      >
                        {transcriptLocked
                          ? t("playground:speech.transcriptUnlock", "Unlock")
                          : t("playground:speech.transcriptLock", "Lock")}
                      </Button>
                    </div>
                    <Input.TextArea
                      value={liveText}
                      readOnly={!canEditTranscript}
                      onChange={(e) => handleTranscriptChange(e.target.value)}
                      autoSize={{ minRows: 3, maxRows: 8 }}
                      placeholder={t(
                        "playground:stt.currentTranscriptPlaceholder",
                        "Live transcript will appear here while recording."
                      )}
                    />
                    <Text type="secondary" className="text-xs">
                      {isRecording || isTranscribing
                        ? t(
                            "playground:speech.transcriptRecordingHint",
                            "Recording in progress; transcript is locked."
                          )
                        : canEditTranscript
                          ? t(
                              "playground:speech.transcriptEditHint",
                              "Editing enabled for this transcript."
                            )
                          : t(
                              "playground:speech.transcriptLockedHint",
                              "Locked to live transcription updates."
                            )}
                    </Text>
                  </div>
                )}

                {effectiveMode === "roundtrip" && (
                  <div className="flex flex-wrap items-center gap-2 pt-1">
                    <Button
                      type="primary"
                      icon={<ArrowRight className="h-4 w-4" />}
                      onClick={handleSendToTts}
                      disabled={!lastTranscript && !liveText}
                    >
                      {t("playground:speech.sendToTts", "Send to TTS")}
                    </Button>
                    <Text type="secondary" className="text-xs">
                      {t(
                        "playground:speech.sendHint",
                        "Use the latest transcript as the TTS draft."
                      )}
                    </Text>
                  </div>
                )}
              </Space>
            </Card>
          )}

          {effectiveMode !== "speak" && (
            <Card className="h-full overflow-hidden">
              <div className="flex h-full">
                {/* Zone 1: Workspace */}
                <div className="flex-1 flex flex-col min-w-0">
                  <div className="flex-1 overflow-y-auto p-4 space-y-4">
                    <AudioReadinessStrip items={ttsReadinessItems} label="TTS readiness" />
                    <TtsProviderStrip
                      provider={provider}
                      model={currentTtsSelection.model}
                      voice={currentTtsSelection.voice}
                      format={currentTtsSelection.format}
                      speed={currentTtsSelection.speed}
                      presetValue={(ttsPreset as TtsPresetKey) || "balanced"}
                      onPresetChange={(preset) => void applyTtsPreset(preset)}
                      onLabelClick={openInspectorAt}
                      onGearClick={() => setInspectorOpen((prev) => !prev)}
                    />
                    <AudioPresetControls
                      kind="tts"
                      currentConfig={currentTtsPresetConfig}
                      capabilityAssumptions={{
                        provider,
                        model: currentTtsSelection.model,
                        voice: currentTtsSelection.voice,
                        browser_local: provider === "browser"
                      }}
                      onApply={(config) => {
                        void handleApplyServerTtsPreset(config)
                      }}
                    />

                    {/* Error banners */}
                    {isTldw && !hasAudio && (
                      <DesignSystemAlert
                        variant="warning"
                        title={t(
                          "playground:tts.serverTtsUnavailableTitle",
                          "Server text-to-speech is not connected"
                        )}
                      >
                        <div className="space-y-3">
                          <span>
                            {t(
                              "playground:tts.serverTtsUnavailableBody",
                              "Check the server connection and Speech settings before generating server TTS."
                            )}
                          </span>
                          <div>
                            <Link
                              to="/settings/speech"
                              className="inline-flex min-h-8 items-center rounded border border-border px-3 text-xs font-medium text-primary hover:text-primaryStrong"
                            >
                              {t(
                                "playground:tts.openSpeechSettings",
                                "Open Speech settings"
                              )}
                            </Link>
                          </div>
                        </div>
                      </DesignSystemAlert>
                    )}

                    {showElevenLabsHint && (
                      <DesignSystemAlert
                        variant={hasElevenLabsKey ? "warning" : "info"}
                        title={elevenLabsHintTitle}
                      >
                        <div className="space-y-3">
                          <span>
                            {hasElevenLabsLoadError && isTimeoutLikeError(elevenLabsError)
                              ? elevenLabsTimeoutBody
                              : elevenLabsHintBody}
                          </span>
                          {hasElevenLabsKey ? (
                            <div>
                              <Button
                                size="small"
                                type="link"
                                onClick={() => {
                                  void refetchElevenLabs()
                                }}
                              >
                                {t("common:retry", "Retry")}
                              </Button>
                            </div>
                          ) : (
                            <details
                              open={inlineElevenLabsDetailsOpen}
                              onToggle={(event) => {
                                setInlineElevenLabsDetailsOpen(event.currentTarget.open)
                              }}
                              className="rounded-md border border-border bg-background/60 px-3 py-2"
                            >
                              <summary className="cursor-pointer text-sm font-medium text-text">
                                {t(
                                  "playground:tts.elevenLabsInlineKeySummary",
                                  "Enter API key"
                                )}
                              </summary>
                              <div className="mt-3 space-y-2">
                                <Space.Compact className="w-full max-w-xl">
                                  <Input.Password
                                    id="elevenlabs-api-key"
                                    aria-label={t(
                                      "playground:tts.elevenLabsInlineKeyLabel",
                                      "ElevenLabs API key"
                                    ) as string}
                                    placeholder="sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                                    value={inlineElevenLabsApiKey}
                                    onChange={(event) => {
                                      setInlineElevenLabsApiKey(event.target.value)
                                      setInlineElevenLabsResult(null)
                                    }}
                                    onPressEnter={() => {
                                      void handleInlineElevenLabsSave()
                                    }}
                                  />
                                  <Button
                                    type="primary"
                                    loading={inlineElevenLabsSaving}
                                    disabled={!inlineElevenLabsApiKey.trim()}
                                    onClick={() => {
                                      void handleInlineElevenLabsSave()
                                    }}
                                  >
                                    {t(
                                      "playground:tts.elevenLabsInlineKeySave",
                                      "Test & Save"
                                    )}
                                  </Button>
                                </Space.Compact>
                                {inlineElevenLabsResult && (
                                  <DesignSystemAlert
                                    variant={inlineElevenLabsResult.ok ? "success" : "error"}
                                    title={inlineElevenLabsResult.message}
                                  />
                                )}
                              </div>
                            </details>
                          )}
                        </div>
                      </DesignSystemAlert>
                    )}

                    {/* Text input area */}
                    <div className="space-y-2">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <Paragraph className="!mb-2 !mr-2">
                          {t("playground:tts.inputLabel", "Enter some text to hear it spoken.")}
                        </Paragraph>
                        <Button
                          size="small"
                          onClick={() => {
                            if (useDraftEditor) {
                              setTranscriptDraft(SAMPLE_TEXT)
                            } else {
                              setTtsText(SAMPLE_TEXT)
                            }
                          }}
                          aria-label={t("playground:tts.sampleText", "Insert sample text") as string}
                        >
                          {t("playground:tts.sampleText", "Insert sample text")}
                        </Button>
                      </div>
                      {useDraftEditor ? (
                        <LongformDraftEditor
                          outline={outlineDraft}
                          transcript={transcriptDraft}
                          onOutlineChange={(value) => {
                            setOutlineDraft(value)
                            if (draftErrors.outline) {
                              setDraftErrors((prev) => ({ ...prev, outline: undefined }))
                            }
                          }}
                          onTranscriptChange={(value) => {
                            setTranscriptDraft(value)
                            if (draftErrors.transcript) {
                              setDraftErrors((prev) => ({ ...prev, transcript: undefined }))
                            }
                          }}
                          outlineError={draftErrors.outline}
                          transcriptError={draftErrors.transcript}
                          preview={normalizedPreviewText}
                        />
                      ) : (
                        <Input.TextArea
                          aria-label={t("playground:tts.inputLabel", "Enter some text to hear it spoken.") as string}
                          value={ttsText}
                          onChange={(e) => setTtsText(e.target.value)}
                          autoSize={{ minRows: 4, maxRows: 10 }}
                          placeholder={t(
                            "playground:tts.inputPlaceholder",
                            "Type or paste text here, then use Play to listen."
                          ) as string}
                        />
                      )}
                    </div>

                    {/* Character progress bar */}
                    <CharacterProgressBar
                      count={previewCharCount}
                      max={TTS_CHAR_LIMIT}
                      warnAt={TTS_CHAR_WARNING}
                      dangerAt={TTS_CHAR_LIMIT - 2000}
                    />

                    {/* Stats line */}
                    <div className="text-xs text-text-subtle">
                      {previewWordCount} words · {previewSegments.length} segments ({responseSplitting}) · Est. ~{formatDuration(estimatedDurationSeconds)}
                    </div>

                    {/* Compose & Compare: Render Strips Zone */}
                    {multiRender.renders.length > 0 && (
                      <div className="space-y-2" role="region" aria-label="Render strips for A/B comparison">
                        <div className="flex items-center justify-between">
                          <Text strong className="text-sm">
                            {t("playground:tts.renderStrips", "Render Strips")}
                          </Text>
                          <div className="flex items-center gap-1.5">
                            {multiRender.hasReady && (
                              <Button
                                size="small"
                                onClick={() => void handlePlayAllRenders()}
                                icon={<Play className="h-3 w-3" />}
                              >
                                {t("playground:tts.playAll", "Play All")}
                              </Button>
                            )}
                            {multiRender.hasIdle && (
                              <Button
                                size="small"
                                type="primary"
                                onClick={() => {
                                  const effectiveText = useDraftEditor ? transcriptDraft : ttsText
                                  void handleGenerateAllRenders(effectiveText)
                                }}
                                disabled={!(useDraftEditor ? transcriptDraft : ttsText).trim()}
                              >
                                {t("playground:tts.generateAll", "Generate All")}
                              </Button>
                            )}
                            <Button
                              size="small"
                              type="text"
                              danger
                              onClick={multiRender.clearAll}
                            >
                              {t("playground:tts.clearStrips", "Clear")}
                            </Button>
                          </div>
                        </div>
                        {multiRender.renders.map((render) => (
                          <RenderStrip
                            key={render.id}
                            id={render.id}
                            state={render.state}
                            config={render.config}
                            audioUrl={render.audioUrl}
                            audioBlob={render.audioBlob}
                            errorMessage={render.errorMessage}
                            errorSettingsHref={render.errorSettingsHref}
                            metadata={render.metadata}
                            disabled={render.disabled}
                            progress={render.progress}
                            isPlaying={multiRender.playingId === render.id}
                            forcePaused={multiRender.playingId !== null && multiRender.playingId !== render.id}
                            onGenerate={(id) => {
                              const effectiveText = useDraftEditor ? transcriptDraft : ttsText
                              void multiRender.generateRender(id, effectiveText)
                            }}
                            onRemove={multiRender.removeRender}
                            onEdit={(id) => {
                              setVoicePickerTargetStripId(id)
                              setVoicePickerOpen(true)
                            }}
                            onPlay={multiRender.startPlaying}
                            onPause={multiRender.stopPlaying}
                            onEnd={multiRender.handleStripEnded}
                            onDuplicate={multiRender.duplicateRender}
                            onToggleDisabled={multiRender.setRenderDisabled}
                            onRetry={(id) => {
                              const effectiveText = useDraftEditor ? transcriptDraft : ttsText
                              void multiRender.generateRender(id, effectiveText)
                            }}
                            onConfigTagClick={handleRenderStripConfigTagClick}
                          />
                        ))}
                      </div>
                    )}

                    {/* Add Render button */}
                    <div className="flex items-center gap-2">
                      <Button
                        size="small"
                        icon={<Plus className="h-3.5 w-3.5" />}
                        onClick={handleAddRenderStrip}
                      >
                        {t("playground:tts.addRender", "Add Render")}
                      </Button>
                      <Button
                        size="small"
                        onClick={() => {
                          setVoicePickerTargetStripId(null)
                          setVoicePickerOpen(true)
                        }}
                      >
                        {t("playground:tts.pickVoice", "Pick Voice")}
                      </Button>
                    </div>

                    {/* Streaming / Job status — when active */}
                    {canStream && streamStatus !== "idle" && (
                      <div className="flex flex-wrap items-center gap-2 text-xs" aria-live="polite">
                        <Tag color={streamStatusColor} bordered>
                          {streamStatusLabel}
                        </Tag>
                        {streamChunks > 0 && <span>{streamChunks} chunks</span>}
                        {streamBytes > 0 && <span>{formatBytes(streamBytes)}</span>}
                      </div>
                    )}
                    {isTtsJobRunning && (
                      <TtsJobProgress
                        title="Long-form TTS"
                        steps={TTS_JOB_STEPS}
                        currentStep={ttsJobStepIndex}
                        percent={ttsJobProgress}
                        message={ttsJobMessage}
                        etaSeconds={ttsJobEta}
                        status="running"
                        metrics={[
                          { label: "Segments", value: String(previewSegments.length) },
                          { label: "Chars", value: String(normalizedPreviewText.length) }
                        ]}
                      />
                    )}
                    {ttsJobStatus === "success" && ttsJobId != null && (
                      <TtsJobProgress
                        title="Long-form TTS"
                        steps={TTS_JOB_STEPS}
                        currentStep={ttsJobStepIndex}
                        percent={ttsJobProgress ?? 100}
                        message={ttsJobMessage || "tts_completed"}
                        etaSeconds={0}
                        status="success"
                        metrics={[
                          { label: "Job", value: String(ttsJobId) },
                          { label: "Segments", value: String(previewSegments.length) }
                        ]}
                      />
                    )}
                    {ttsJobStatus === "error" && ttsJobError && (
                      <DesignSystemAlert
                        variant="error"
                        title={t("playground:tts.longFormError", "Long-form TTS error")}
                      >
                        {ttsJobError}
                      </DesignSystemAlert>
                    )}
                    {activeStreamError && (
                      <DesignSystemAlert
                        variant="error"
                        title={t("playground:tts.streamingError", "Streaming error")}
                      >
                        {activeStreamError}
                      </DesignSystemAlert>
                    )}

                    {/* Waveform + Segments */}
                    {segments.length > 0 && (
                      <div className="mt-2 space-y-2 w-full">
                        <div>
                          <Text strong>{t("playground:tts.outputTitle", "Generated audio segments")}</Text>
                          <Paragraph className="!mb-1 text-xs text-text-subtle">
                            {t(
                              "playground:tts.outputHelp",
                              "Select a segment, then use the player controls to play, pause, or seek."
                            )}
                          </Paragraph>
                        </div>
                        <div className="border border-border rounded-md p-3 space-y-2">
                          <audio
                            ref={audioRef}
                            controls
                            className="w-full"
                            src={
                              activeSegmentIndex != null
                                ? segments[activeSegmentIndex]?.url
                                : segments[0]?.url
                            }
                            onTimeUpdate={handleAudioTimeUpdate}
                          />
                          <WaveformCanvas
                            audioRef={audioRef}
                            active={Boolean(segments.length)}
                            label={t("playground:speech.playbackWaveform", "Playback waveform") as string}
                          />
                          <div className="flex items-center justify-between text-xs text-text-subtle">
                            <span>
                              {activeSegmentIndex != null
                                ? t("playground:tts.currentSegment", "Segment") +
                                  ` ${activeSegmentIndex + 1}/${segments.length}`
                                : t("playground:tts.currentSegmentNone", "No segment selected")}
                            </span>
                            {duration > 0 && (
                              <span>
                                {t("playground:tts.timeLabel", "Time")}: {Math.floor(currentTime)}s /{" "}
                                {Math.floor(duration)}s
                              </span>
                            )}
                          </div>
                          {/* Segment navigation with text previews */}
                          {segments.length > 1 && (
                            <div className="flex gap-1.5 overflow-x-auto pb-1" role="tablist">
                              {segments.map((seg, idx) => {
                                const preview = seg.text
                                  ? seg.text.slice(0, 25) + (seg.text.length > 25 ? "..." : "")
                                  : `Segment ${idx + 1}`
                                return (
                                  <Button
                                    key={seg.id}
                                    role="tab"
                                    aria-selected={activeSegmentIndex === idx}
                                    size="small"
                                    type={
                                      idx === (activeSegmentIndex != null ? activeSegmentIndex : 0)
                                        ? "primary"
                                        : "default"
                                    }
                                    onClick={() => handleSegmentSelect(idx)}
                                  >
                                    {idx + 1}: &ldquo;{preview}&rdquo;
                                  </Button>
                                )
                              })}
                            </div>
                          )}
                          <Text type="secondary" className="text-xs">
                            {t("playground:speech.segmentFormat", "Format")}:{" "}
                            {(segments[0]?.format || "mp3").toUpperCase()}
                          </Text>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Sticky action bar */}
                  <TtsStickyActionBar
                    onPlay={() => { void handlePlay() }}
                    onStop={handleStop}
                    onDownloadSegment={() => handleDownloadSegment()}
                    onDownloadAll={handleDownloadAll}
                    onToggleInspector={() => setInspectorOpen((prev) => !prev)}
                    isPlayDisabled={isPlayDisabled}
                    isStopDisabled={!canStop}
                    isDownloadDisabled={Boolean(downloadDisabledReason)}
                    playDisabledReason={(playDisabledReason as string) || null}
                    stopDisabledReason={(stopDisabledReason as string) || null}
                    downloadDisabledReason={(downloadDisabledReason as string) || null}
                    streamStatus={streamStatus as "idle" | "connecting" | "streaming" | "complete" | "error"}
                    inspectorOpen={inspectorOpen ?? false}
                    inspectorBadge={inspectorBadge}
                    segmentCount={segments.length}
                    provider={provider}
                    onAddRender={handleAddRenderStrip}
                    onPlayAllRenders={() => void handlePlayAllRenders()}
                    renderStripCount={multiRender.renders.length}
                    hasReadyRenders={multiRender.hasReady}
                  />
                </div>

                {/* Zone 2: Inspector */}
                <TtsInspectorPanel
                  open={inspectorOpen ?? false}
                  activeTab={inspectorTab ?? "voice"}
                  onTabChange={setInspectorTab}
                  onClose={() => setInspectorOpen(false)}
                  useDrawer={useDrawerMode}
                  voiceTab={
                    <TtsVoiceTab
                      provider={provider}
                      model={currentTtsSelection.model}
                      voice={currentTtsSelection.voice}
                      onProviderChange={(val) => {
                        if (ttsSettings) {
                          void setTTSSettings({
                            ...ttsSettings,
                            ttsProvider: val
                          }).then(() => {
                            queryClient.invalidateQueries({ queryKey: ["fetchTTSSettings"] })
                          })
                        }
                      }}
                      onModelChange={(val) => setTldwModel(val)}
                      onVoiceChange={(val) => setTldwVoice(val)}
                      modelOptions={
                        tldwTtsModels && tldwTtsModels.length > 0
                          ? tldwTtsModels.map((m) => ({ label: m.label, value: m.id }))
                          : []
                      }
                      voiceOptions={tldwVoiceOptions as { label: string; value: string }[]}
                      language={tldwLanguage}
                      onLanguageChange={(val) => setTldwLanguage(val)}
                      languageOptions={tldwLanguageOptions}
                      emotion={tldwEmotion}
                      onEmotionChange={(val) => setTldwEmotion(val)}
                      emotionIntensity={tldwEmotionIntensity}
                      onEmotionIntensityChange={(val) => setTldwEmotionIntensity(val)}
                      supportsEmotion={Boolean(isTldw && activeProviderCaps?.caps.supports_emotion_control)}
                      useVoiceRoles={useVoiceRoles}
                      onVoiceRolesChange={setUseVoiceRoles}
                      voiceRolesContent={
                        <>
                          {voiceCards.map((card) => (
                            <div key={card.id} className="flex items-center gap-2">
                              <Select
                                size="small"
                                className="w-28"
                                value={card.role}
                                onChange={(val) => handleUpdateVoiceCard(card.id, { role: val })}
                                options={VOICE_ROLE_OPTIONS}
                              />
                              <Select
                                size="small"
                                className="flex-1"
                                value={card.voiceId || undefined}
                                onChange={(val) => handleUpdateVoiceCard(card.id, { voiceId: val })}
                                options={tldwVoiceOptions as { label: string; value: string }[]}
                                showSearch
                                optionFilterProp="label"
                                placeholder="Select voice"
                              />
                              <Button
                                size="small"
                                type="text"
                                danger
                                onClick={() => handleRemoveVoiceCard(card.id)}
                              >
                                <Trash2 className="h-3.5 w-3.5" />
                              </Button>
                            </div>
                          ))}
                          {voiceCards.length < 4 && (
                            <Button size="small" type="dashed" block onClick={handleAddVoiceCard}>
                              Add voice
                            </Button>
                          )}
                          {voiceRoleError && (
                            <div className="text-xs text-red-500">{voiceRoleError}</div>
                          )}
                        </>
                      }
                      focusField={inspectorFocusField}
                      onFocusHandled={() => setInspectorFocusField(null)}
                    />
                  }
                  outputTab={
                    <TtsOutputTab
                      format={tldwFormat || ttsSettings?.tldwTtsResponseFormat || "mp3"}
                      synthesisSpeed={ttsSettings?.tldwTtsSpeed ?? 1}
                      playbackSpeed={ttsSettings?.playbackSpeed ?? 1}
                      responseSplitting={responseSplitting}
                      streaming={tldwStreaming}
                      canStream={canStream}
                      streamFormatSupported={streamFormatSupported}
                      onFormatChange={(val) => setTldwFormat(val)}
                      onSynthesisSpeedChange={(val) => {
                        void setTldwTTSSpeed(val).then(() => {
                          queryClient.invalidateQueries({ queryKey: ["fetchTTSSettings"] })
                        })
                      }}
                      onPlaybackSpeedChange={(val) => {
                        if (ttsSettings) {
                          void setTTSSettings({ ...ttsSettings, playbackSpeed: val }).then(() => {
                            queryClient.invalidateQueries({ queryKey: ["fetchTTSSettings"] })
                          })
                        }
                      }}
                      onResponseSplittingChange={(val) => setResponseSplitting(val)}
                      onStreamingChange={(val) => setTldwStreaming(val)}
                      formatOptions={tldwFormatOptions}
                      normalize={tldwNormalize}
                      onNormalizeChange={setTldwNormalize}
                      normalizeUnits={tldwNormalizeUnits}
                      onNormalizeUnitsChange={setTldwNormalizeUnits}
                      normalizeUrls={tldwNormalizeUrls}
                      onNormalizeUrlsChange={setTldwNormalizeUrls}
                      normalizeEmails={tldwNormalizeEmails}
                      onNormalizeEmailsChange={setTldwNormalizeEmails}
                      normalizePhones={tldwNormalizePhones}
                      onNormalizePhonesChange={setTldwNormalizePhones}
                      normalizePlurals={tldwNormalizePlurals}
                      onNormalizePluralsChange={setTldwNormalizePlurals}
                      focusField={inspectorFocusField}
                      onFocusHandled={() => setInspectorFocusField(null)}
                    />
                  }
                  advancedTab={
                    <TtsAdvancedTab
                      useDraftEditor={useDraftEditor}
                      onDraftEditorChange={setUseDraftEditor}
                      useTtsJob={useTtsJob}
                      onTtsJobChange={setUseTtsJob}
                      ssmlEnabled={ttsSettings?.ssmlEnabled ?? false}
                      onSsmlChange={(val) => {
                        if (ttsSettings) {
                          void setTTSSettings({ ...ttsSettings, ssmlEnabled: val }).then(() => {
                            queryClient.invalidateQueries({ queryKey: ["fetchTTSSettings"] })
                          })
                        }
                      }}
                      removeReasoning={ttsSettings?.removeReasoningTagTTS ?? true}
                      onRemoveReasoningChange={(val) => {
                        if (ttsSettings) {
                          void setTTSSettings({ ...ttsSettings, removeReasoningTagTTS: val }).then(() => {
                            queryClient.invalidateQueries({ queryKey: ["fetchTTSSettings"] })
                          })
                        }
                      }}
                      isTldw={isTldw}
                      onOpenVoiceCloning={() => openInspectorAt("advanced")}
                      voiceCloningContent={
                        <VoiceCloningManager
                          providersInfo={providersInfo}
                          onSelectVoice={(val) => setTldwVoice(val)}
                        />
                      }
                    />
                  }
                />
              </div>
            </Card>
          )}
        </div>

        <Card>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <Text strong>{historyTitle}</Text>
            <Space size="small" className="flex flex-wrap">
              <Tooltip title="Show favorites only">
                <Button
                  size="small"
                  type={historyFavoritesOnly ? "primary" : "default"}
                  onClick={() => setHistoryFavoritesOnly((prev) => !prev)}
                >
                  {historyFavoritesOnly ? "Favorites" : "All items"}
                </Button>
              </Tooltip>
              {!isLockedTtsRoute && (
                <div data-testid="speech-history-type-filter">
                  <Select
                    size="small"
                    value={historyFilter}
                    onChange={(value) => setHistoryFilter(value)}
                    options={[
                      { label: t("playground:speech.historyAll", "All"), value: "all" },
                      { label: t("playground:speech.historyStt", "STT"), value: "stt" },
                      { label: t("playground:speech.historyTts", "TTS"), value: "tts" }
                    ]}
                  />
                </div>
              )}
              <Input
                size="small"
                placeholder={t("playground:speech.historySearch", "Search transcripts")}
                value={historyQuery}
                onChange={(e) => setHistoryQuery(e.target.value)}
                style={{ width: 200 }}
              />
              {filteredHistory.length > 0 && (
                <Button size="small" type="text" icon={<Trash2 className="h-3 w-3" />} onClick={clearHistory}>
                  {t("playground:stt.clearAll", "Clear all")}
                </Button>
              )}
            </Space>
          </div>
          <Text type="secondary" className="text-xs">
            {withTemplateFallback(
              t(
                "playground:speech.historyRetentionHint",
                "Keeps the most recent {{count}} items. Use Clear all to remove everything.",
                { count: MAX_HISTORY_ITEMS }
              ),
              `Keeps the most recent ${MAX_HISTORY_ITEMS} items. Use Clear all to remove everything.`
            )}
          </Text>

          {filteredHistory.length === 0 ? (
            <Text type="secondary" className="text-xs">
              {historyEmptyState}
            </Text>
          ) : (
            <List
              itemLayout="vertical"
              dataSource={filteredHistory}
              renderItem={(item) => {
                const paramsSummary = buildHistoryParamsSummary(item)
                const detailTooltip = buildHistoryDetailTooltip(item)
                const actions: React.ReactNode[] = []
                if (item.type === "stt") {
                  actions.push(
                    <Button
                      key="save"
                      size="small"
                      icon={<Save className="h-3 w-3" />}
                      onClick={() => handleSaveToNotes(item)}
                    >
                      {t("playground:stt.saveToNotes", "Save to Notes")}
                    </Button>
                  )
                }
                actions.push(
                  <Tooltip key="favorite" title={item.favorite ? "Unfavorite" : "Favorite"}>
                    <Button
                      size="small"
                      type={item.favorite ? "primary" : "default"}
                      icon={<Star className="h-3 w-3" />}
                      onClick={() => toggleHistoryFavorite(item.id)}
                      aria-label={item.favorite ? "Unfavorite" : "Favorite"}
                    >
                    </Button>
                  </Tooltip>
                )
                if (item.type === "tts") {
                  actions.push(
                    <Button
                      key="replay"
                      size="small"
                      onClick={() => handleReplayHistoryItem(item)}
                    >
                      {t("playground:tts.replay", "Replay")}
                    </Button>
                  )
                }
                actions.push(
                  <Button
                    key="use"
                    size="small"
                    onClick={() => setTtsText(item.text)}
                  >
                    {t("playground:speech.useInTts", "Use in TTS")}
                  </Button>
                )
                actions.push(
                  <Button
                    key="copy"
                    size="small"
                    icon={<Copy className="h-3 w-3" />}
                    onClick={() => handleCopy(item.text)}
                  >
                    {t("playground:speech.copy", "Copy")}
                  </Button>
                )
                actions.push(
                  <Button
                    key="delete"
                    size="small"
                    type="text"
                    icon={<Trash2 className="h-3 w-3" />}
                    onClick={() => removeHistoryItem(item.id)}
                  >
                    {t("playground:stt.delete", "Delete")}
                  </Button>
                )

                return (
                  <List.Item key={item.id} actions={actions}>
                    <List.Item.Meta
                      title={
                        <div className="flex flex-wrap items-center gap-2">
                          <Tag color={item.type === "stt" ? "blue" : "gold"} bordered>
                            {item.type.toUpperCase()}
                          </Tag>
                          <Text>{formatHistoryDate(item.createdAt)}</Text>
                          {item.durationMs != null && (
                            <Tag bordered>
                              {t("playground:stt.durationTag", "Duration")}:{" "}
                              <Text code className="ml-1">
                                {(item.durationMs / 1000).toFixed(1)}s
                              </Text>
                            </Tag>
                          )}
                          {item.model && (
                            <Tag bordered>
                              {t("playground:stt.modelTag", "Model")}{" "}
                              <Text code className="ml-1">
                                {item.model}
                              </Text>
                            </Tag>
                          )}
                          {item.provider && (
                            <Tag bordered>
                              {t("playground:speech.providerTag", "Provider")}{" "}
                              <Text code className="ml-1">
                                {item.provider}
                              </Text>
                            </Tag>
                          )}
                          {item.voice && (
                            <Tag bordered>
                              {t("playground:speech.voiceTag", "Voice")}{" "}
                              <Text code className="ml-1">
                                {item.voice}
                              </Text>
                            </Tag>
                          )}
                          {detailTooltip && (
                            <Tooltip title={detailTooltip} placement="top">
                              <Tag bordered className="cursor-help">
                                {t("playground:speech.historyDetails", "Details")}
                              </Tag>
                            </Tooltip>
                          )}
                        </div>
                      }
                      description={
                        <div className="space-y-1">
                          {item.language && (
                            <Text type="secondary" className="text-xs">
                              {t("playground:stt.languageTag", "Language")}: {item.language}
                            </Text>
                          )}
                          {paramsSummary && (
                            <Text type="secondary" className="text-xs">
                              {t("playground:speech.paramsSummary", "Params")}: {paramsSummary}
                            </Text>
                          )}
                        </div>
                      }
                    />
                    <Input.TextArea
                      value={item.text}
                      autoSize={{ minRows: 3, maxRows: 8 }}
                      readOnly
                    />
                  </List.Item>
                )
              }}
            />
          )}
        </Card>
      </div>
      {/* Voice Picker Modal */}
      <VoicePickerModal
        open={voicePickerOpen}
        onClose={() => {
          setVoicePickerOpen(false)
          setVoicePickerTargetStripId(null)
        }}
        onSelect={handleVoicePickerSelect}
        providersInfo={providersInfo}
      />
    </PageShell>
  )
}

export default SpeechPlaygroundPage
