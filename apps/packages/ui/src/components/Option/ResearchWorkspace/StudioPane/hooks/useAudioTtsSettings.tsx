import React, { useState, useEffect, useRef } from "react"
import type { MessageInstance } from "antd/es/message/interface"
import type { TFunction } from "i18next"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { fetchTldwVoiceCatalog, type TldwVoice } from "@/services/tldw/audio-voices"
import {
  fetchTtsProviders,
  type TldwTtsProvidersInfo
} from "@/services/tldw/audio-providers"
import {
  fetchTldwTtsModels,
  type TldwTtsModel
} from "@/services/tldw/audio-models"
import { toServerTtsProviderKey } from "@/services/tldw/tts-provider-keys"
import { inferTldwProviderFromModel } from "@/services/tts-provider"
import type { AudioTtsProvider } from "@/types/workspace"
import type { AudioGenerationSettings } from "@/types/workspace"

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

export const TTS_PROVIDERS: { value: AudioTtsProvider; label: string }[] = [
  { value: "tldw", label: "tldw Server" },
  { value: "openai", label: "OpenAI" },
  { value: "browser", label: "Browser" }
]

export const TLDW_TTS_MODELS = [
  { value: "kokoro", label: "Kokoro" },
  { value: "kitten_tts", label: "KittenTTS" },
  { value: "KittenML/kitten-tts-nano-0.8", label: "KittenTTS (Nano)" },
  {
    value: "KittenML/kitten-tts-nano-0.8-int8",
    label: "KittenTTS (Nano INT8)"
  },
  { value: "KittenML/kitten-tts-micro-0.8", label: "KittenTTS (Micro)" },
  { value: "KittenML/kitten-tts-mini-0.8", label: "KittenTTS (Mini)" }
]

export const OPENAI_TTS_MODELS = [
  { value: "tts-1", label: "tts-1" },
  { value: "tts-1-hd", label: "tts-1-hd" }
]

export const OPENAI_TTS_VOICES = [
  { value: "alloy", label: "Alloy" },
  { value: "echo", label: "Echo" },
  { value: "fable", label: "Fable" },
  { value: "onyx", label: "Onyx" },
  { value: "nova", label: "Nova" },
  { value: "shimmer", label: "Shimmer" }
]

export const AUDIO_FORMATS: { value: string; label: string }[] = [
  { value: "mp3", label: "MP3" },
  { value: "wav", label: "WAV" },
  { value: "opus", label: "Opus" },
  { value: "aac", label: "AAC" },
  { value: "flac", label: "FLAC" }
]

const VOICE_PREVIEW_TEXT =
  "This is a quick voice preview from your current audio settings."

const KOKORO_FALLBACK_VOICES = [
  { value: "af_heart", label: "Heart (Female)" },
  { value: "af_bella", label: "Bella (Female)" },
  { value: "am_adam", label: "Adam (Male)" },
  { value: "am_michael", label: "Michael (Male)" }
]

const KITTEN_FALLBACK_VOICES = [
  { value: "Bella", label: "Bella" },
  { value: "Jasper", label: "Jasper" },
  { value: "Luna", label: "Luna" },
  { value: "Bruno", label: "Bruno" },
  { value: "Rosie", label: "Rosie" },
  { value: "Hugo", label: "Hugo" },
  { value: "Kiki", label: "Kiki" },
  { value: "Leo", label: "Leo" }
]

const isAbortLikeError = (error: unknown): boolean => {
  const candidate = (error as {
    name?: string
    message?: string
    code?: string
  } | null) ?? { message: String(error ?? "") }

  if (candidate.name === "AbortError") {
    return true
  }

  if (
    typeof candidate.code === "string" &&
    /^(REQUEST_ABORTED|ERR_CANCELED|ERR_CANCELLED)$/i.test(candidate.code)
  ) {
    return true
  }

  const message = candidate.message ?? String(error ?? "")
  return /\babort(ed|error)?\b/i.test(message)
}

// ─────────────────────────────────────────────────────────────────────────────
// Hook interface
// ─────────────────────────────────────────────────────────────────────────────

export interface UseAudioTtsSettingsDeps {
  audioSettings: AudioGenerationSettings
  setAudioSettings: (patch: Partial<AudioGenerationSettings>) => void
  messageApi: MessageInstance
  t: TFunction
}

export function useAudioTtsSettings(deps: UseAudioTtsSettingsDeps) {
  const { audioSettings, setAudioSettings, messageApi, t } = deps

  const [showTtsSettings, setShowTtsSettings] = useState(false)
  const [tldwVoices, setTldwVoices] = useState<TldwVoice[]>([])
  const [providersInfo, setProvidersInfo] =
    useState<TldwTtsProvidersInfo | null>(null)
  const [tldwModels, setTldwModels] = useState<TldwTtsModel[]>([])
  const [loadingVoices, setLoadingVoices] = useState(false)
  const [voiceCatalogSettled, setVoiceCatalogSettled] = useState(false)
  const [previewingVoice, setPreviewingVoice] = useState(false)
  const previewAudioRef = useRef<HTMLAudioElement | null>(null)

  const inferredTldwProviderKey = inferTldwProviderFromModel(audioSettings.model)
  const explicitBackendSupported =
    providersInfo?.supports_explicit_backend === true
  const selectedBackend = explicitBackendSupported
    ? String(audioSettings.backend || "").trim()
    : ""
  const selectedBackendCapabilities = selectedBackend
    ? providersInfo?.providers?.[selectedBackend]
    : undefined
  const catalogProvider = selectedBackend
    ? selectedBackend
    : inferredTldwProviderKey
      ? toServerTtsProviderKey(inferredTldwProviderKey)
      : ""

  useEffect(() => {
    let cancelled = false

    if (audioSettings.provider !== "tldw") {
      setProvidersInfo(null)
      return
    }

    fetchTtsProviders()
      .then((info) => {
        if (!cancelled) setProvidersInfo(info)
      })
      .catch(() => {
        if (!cancelled) setProvidersInfo(null)
      })

    return () => {
      cancelled = true
    }
  }, [audioSettings.provider])

  useEffect(() => {
    let cancelled = false

    if (audioSettings.provider !== "tldw") {
      setTldwModels([])
      return
    }

    fetchTldwTtsModels(selectedBackend || undefined)
      .then((models) => {
        if (!cancelled) setTldwModels(models)
      })
      .catch(() => {
        if (!cancelled) setTldwModels([])
      })

    return () => {
      cancelled = true
    }
  }, [audioSettings.provider, selectedBackend])

  // Fetch voices when provider changes to tldw
  useEffect(() => {
    let cancelled = false

    if (audioSettings.provider !== "tldw") {
      setTldwVoices([])
      setLoadingVoices(false)
      setVoiceCatalogSettled(true)
      return
    }
    if (!catalogProvider) {
      setTldwVoices([])
      setLoadingVoices(false)
      setVoiceCatalogSettled(true)
      return
    }
    setVoiceCatalogSettled(false)
    setLoadingVoices(true)
    fetchTldwVoiceCatalog(
      catalogProvider,
      selectedBackend && audioSettings.model
        ? { model: audioSettings.model }
        : undefined
    )
      .then((voices) => {
        if (!cancelled) {
          setTldwVoices(voices)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setTldwVoices([])
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingVoices(false)
          setVoiceCatalogSettled(true)
        }
      })

    return () => {
      cancelled = true
    }
  }, [audioSettings.model, audioSettings.provider, catalogProvider, selectedBackend])

  // Cleanup preview audio on unmount
  useEffect(() => {
    return () => {
      if (previewAudioRef.current) {
        previewAudioRef.current.pause()
        previewAudioRef.current.src = ""
        previewAudioRef.current = null
      }
    }
  }, [])

  // Get voice options based on provider
  const getVoiceOptions = React.useCallback(() => {
    if (audioSettings.provider === "tldw") {
      if (tldwVoices.length > 0) {
        return tldwVoices.map((v) => ({
          value: v.voice_id || v.id || v.name || "",
          label: v.name || v.voice_id || v.id || "Unknown"
        }))
      }
      if (selectedBackendCapabilities) {
        const modelVoices = audioSettings.model
          ? selectedBackendCapabilities.model_capabilities?.[audioSettings.model]
              ?.voices
          : undefined
        const discoveredVoices =
          modelVoices?.length
            ? modelVoices.map((voice) => ({ value: voice, label: voice }))
            : (selectedBackendCapabilities.voices || []).map((voice) => ({
                value: voice.voice_id || voice.id || voice.name || "",
                label: voice.name || voice.voice_id || voice.id || "Unknown"
              }))
        return discoveredVoices.filter((voice) => voice.value)
      }
      if (inferredTldwProviderKey === "kitten_tts") {
        return KITTEN_FALLBACK_VOICES
      }
      return KOKORO_FALLBACK_VOICES
    }
    if (audioSettings.provider === "openai") {
      return OPENAI_TTS_VOICES
    }
    return []
  }, [
    audioSettings.model,
    audioSettings.provider,
    inferredTldwProviderKey,
    selectedBackendCapabilities,
    tldwVoices
  ])

  useEffect(() => {
    if (audioSettings.provider !== "tldw") {
      return
    }
    const voiceOptions = getVoiceOptions()
    if (!voiceOptions.length) {
      return
    }
    if (!voiceCatalogSettled && !tldwVoices.length) {
      return
    }
    const currentVoice = String(audioSettings.voice || "").trim()
    if (currentVoice && voiceOptions.some((option) => option.value === currentVoice)) {
      return
    }
    setAudioSettings({ voice: voiceOptions[0].value })
  }, [
    audioSettings.provider,
    audioSettings.voice,
    getVoiceOptions,
    tldwVoices.length,
    voiceCatalogSettled,
    setAudioSettings
  ])

  // Get model options based on provider
  const getModelOptions = React.useCallback(() => {
    if (audioSettings.provider === "tldw") {
      const discoveredModels =
        tldwModels.length > 0
          ? tldwModels
          : (selectedBackendCapabilities?.models || []).map((id) => ({
              id,
              label: id
            }))
      if (discoveredModels.length > 0) {
        return discoveredModels.map((model) => ({
          value: model.id,
          label: model.label || model.id
        }))
      }
      return TLDW_TTS_MODELS
    }
    if (audioSettings.provider === "openai") {
      return OPENAI_TTS_MODELS
    }
    return []
  }, [
    audioSettings.provider,
    selectedBackendCapabilities,
    tldwModels
  ])

  const backendOptions = React.useMemo(
    () => [
      { value: "", label: "Automatic (legacy model inference)" },
      ...Object.entries(providersInfo?.providers || {})
        .filter(
          ([backend, capabilities]) =>
            typeof capabilities.display_name === "string" ||
            backend === "openrouter" ||
            backend.startsWith("gateway:")
        )
        .map(([backend, capabilities]) => ({
          value: backend,
          label: capabilities.display_name || backend
        }))
    ],
    [providersInfo]
  )

  const handleBackendChange = React.useCallback(
    (backend: string) => {
      const capabilities = providersInfo?.providers?.[backend]
      const model =
        capabilities?.default_model || capabilities?.models?.[0] || ""
      const modelCapabilities = model
        ? capabilities?.model_capabilities?.[model]
        : undefined
      const firstVoice = capabilities?.voices?.[0]
      const voice =
        modelCapabilities?.default_voice ||
        modelCapabilities?.voices?.[0] ||
        firstVoice?.voice_id ||
        firstVoice?.id ||
        firstVoice?.name ||
        ""

      setAudioSettings({
        backend,
        allowFallback: audioSettings.allowFallback ?? true,
        model,
        voice
      })
    },
    [audioSettings.allowFallback, providersInfo, setAudioSettings]
  )

  const handlePreviewVoice = React.useCallback(async () => {
    if (audioSettings.provider === "browser") {
      if (typeof window === "undefined" || !("speechSynthesis" in window)) {
        messageApi.error(
          t(
            "playground:studio.voicePreviewUnavailable",
            "Voice preview is unavailable in this browser."
          )
        )
        return
      }
      window.speechSynthesis.cancel()
      const utterance = new SpeechSynthesisUtterance(VOICE_PREVIEW_TEXT)
      utterance.rate = audioSettings.speed
      window.speechSynthesis.speak(utterance)
      return
    }

    setPreviewingVoice(true)
    try {
      const result = await tldwClient.synthesizeSpeechDetailed(VOICE_PREVIEW_TEXT, {
        model: audioSettings.model,
        voice: audioSettings.voice,
        responseFormat: "mp3",
        speed: audioSettings.speed,
        backend: audioSettings.backend || undefined,
        allowFallback: audioSettings.allowFallback ?? true
      })
      const audioBlob = new Blob([result.buffer], { type: "audio/mpeg" })
      const audioUrl = URL.createObjectURL(audioBlob)

      if (previewAudioRef.current) {
        previewAudioRef.current.pause()
      }
      const previewAudio = new Audio(audioUrl)
      previewAudioRef.current = previewAudio
      previewAudio.onended = () => {
        URL.revokeObjectURL(audioUrl)
        if (previewAudioRef.current === previewAudio) {
          previewAudioRef.current = null
        }
      }
      void previewAudio.play()
    } catch (error) {
      if (!isAbortLikeError(error)) {
        messageApi.error(
          t(
            "playground:studio.voicePreviewFailed",
            "Unable to preview this voice right now."
          )
        )
      }
    } finally {
      setPreviewingVoice(false)
    }
  }, [audioSettings, messageApi, t])

  return {
    // state
    showTtsSettings,
    setShowTtsSettings,
    tldwVoices,
    loadingVoices,
    previewingVoice,
    previewAudioRef,
    // computed
    inferredTldwProviderKey,
    explicitBackendSupported,
    backendOptions,
    // callbacks
    getVoiceOptions,
    getModelOptions,
    handleBackendChange,
    handlePreviewVoice,
  }
}
