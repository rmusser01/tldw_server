import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import React, { useState, useEffect, useRef } from "react"
import type { MessageInstance } from "antd/es/message/interface"
import type { TFunction } from "i18next"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  fetchTldwVoiceCatalog,
  type TldwVoice
} from "@/services/tldw/audio-voices"
import {
  fetchTtsProviders,
  type TldwTtsProvidersInfo
} from "@/services/tldw/audio-providers"
import {
  fetchTldwTtsModels,
  type TldwTtsModel
} from "@/services/tldw/audio-models"
import {
  normalizeTtsProviderKey,
  toServerTtsProviderKey
} from "@/services/tldw/tts-provider-keys"
import {
  listCustomVoices,
  type TldwCustomVoice
} from "@/services/tldw/voice-cloning"
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
  const [customVoices, setCustomVoices] = useState<TldwCustomVoice[]>([])
  const [customVoiceCatalogSettled, setCustomVoiceCatalogSettled] = useState<
    string | null
  >(null)
  const [accountRevision, setAccountRevision] = useState(0)
  const accountRevisionRef = useRef(0)
  const [tldwVoices, setTldwVoices] = useState<TldwVoice[]>([])
  const [providersInfo, setProvidersInfo] =
    useState<TldwTtsProvidersInfo | null>(null)
  const [tldwModels, setTldwModels] = useState<TldwTtsModel[]>([])
  const [loadingVoices, setLoadingVoices] = useState(false)
  const [voiceCatalogSettled, setVoiceCatalogSettled] = useState<string | null>(
    null
  )
  const [previewingVoice, setPreviewingVoice] = useState(false)
  const previewAudioRef = useRef<HTMLAudioElement | null>(null)

  useEffect(
    () =>
      watchChatAccountChanges(() => {
        accountRevisionRef.current += 1
        setAccountRevision(accountRevisionRef.current)
        setCustomVoices([])
        setTldwVoices([])
        setTldwModels([])
        setProvidersInfo(null)
        setVoiceCatalogSettled(null)
      }),
    []
  )

  const inferredTldwProviderKey = inferTldwProviderFromModel(
    audioSettings.model
  )
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
    const revision = accountRevisionRef.current

    if (audioSettings.provider !== "tldw") {
      setProvidersInfo(null)
      return
    }

    fetchTtsProviders()
      .then((info) => {
        if (!cancelled && revision === accountRevisionRef.current)
          setProvidersInfo(info)
      })
      .catch(() => {
        if (!cancelled && revision === accountRevisionRef.current)
          setProvidersInfo(null)
      })

    return () => {
      cancelled = true
    }
  }, [audioSettings.provider, accountRevision])

  useEffect(() => {
    let cancelled = false
    const revision = accountRevisionRef.current

    if (audioSettings.provider !== "tldw") {
      setTldwModels([])
      return
    }

    fetchTldwTtsModels(selectedBackend || undefined)
      .then((models) => {
        if (!cancelled && revision === accountRevisionRef.current)
          setTldwModels(models)
      })
      .catch(() => {
        if (!cancelled && revision === accountRevisionRef.current)
          setTldwModels([])
      })

    return () => {
      cancelled = true
    }
  }, [audioSettings.provider, selectedBackend, accountRevision])

  // Settlement must belong to this catalog, not the previous render's completed request.
  const voiceCatalogKey = JSON.stringify([
    accountRevision,
    audioSettings.provider,
    catalogProvider,
    audioSettings.model,
    selectedBackend
  ])

  // Fetch voices when provider changes to tldw
  useEffect(() => {
    let cancelled = false
    const revision = accountRevisionRef.current

    if (audioSettings.provider !== "tldw") {
      setTldwVoices([])
      setLoadingVoices(false)
      setVoiceCatalogSettled(voiceCatalogKey)
      return
    }
    if (!catalogProvider) {
      setTldwVoices([])
      setLoadingVoices(false)
      setVoiceCatalogSettled(voiceCatalogKey)
      return
    }
    setVoiceCatalogSettled(null)
    setLoadingVoices(true)
    fetchTldwVoiceCatalog(
      catalogProvider,
      selectedBackend && audioSettings.model
        ? { model: audioSettings.model }
        : undefined
    )
      .then((voices) => {
        if (!cancelled && revision === accountRevisionRef.current) {
          setTldwVoices(voices)
        }
      })
      .catch(() => {
        if (!cancelled && revision === accountRevisionRef.current) {
          setTldwVoices([])
        }
      })
      .finally(() => {
        if (!cancelled && revision === accountRevisionRef.current) {
          setLoadingVoices(false)
          setVoiceCatalogSettled(voiceCatalogKey)
        }
      })

    return () => {
      cancelled = true
    }
  }, [
    audioSettings.model,
    audioSettings.provider,
    catalogProvider,
    selectedBackend,
    accountRevision,
    voiceCatalogKey
  ])

  const unresolvedBackend = Boolean(audioSettings.backend && !providersInfo)
  const customVoicesAllowed =
    audioSettings.provider === "tldw" &&
    !selectedBackend &&
    Boolean(providersInfo || !audioSettings.backend)
  const customVoiceCatalogKey = JSON.stringify([
    accountRevision,
    customVoicesAllowed,
    unresolvedBackend
  ])

  useEffect(() => {
    let cancelled = false
    const revision = accountRevisionRef.current
    setCustomVoices([])
    setCustomVoiceCatalogSettled(
      !customVoicesAllowed && !unresolvedBackend ? customVoiceCatalogKey : null
    )
    if (customVoicesAllowed) {
      listCustomVoices()
        .then((voices) => {
          if (!cancelled && revision === accountRevisionRef.current)
            setCustomVoices(voices)
        })
        .catch(() => {
          if (!cancelled && revision === accountRevisionRef.current)
            setCustomVoices([])
        })
        .finally(() => {
          if (!cancelled && revision === accountRevisionRef.current)
            setCustomVoiceCatalogSettled(customVoiceCatalogKey)
        })
    }
    return () => {
      cancelled = true
    }
  }, [
    customVoicesAllowed,
    unresolvedBackend,
    accountRevision,
    customVoiceCatalogKey
  ])

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
      let options = tldwVoices.map((v) => ({
        value: v.voice_id || v.id || v.name || "",
        label: v.name || v.voice_id || v.id || "Unknown"
      }))
      if (!options.length) {
        if (selectedBackendCapabilities) {
          const modelVoices = audioSettings.model
            ? selectedBackendCapabilities.model_capabilities?.[
                audioSettings.model
              ]?.voices
            : undefined
          options = modelVoices?.length
            ? modelVoices.map((voice) => ({ value: voice, label: voice }))
            : (selectedBackendCapabilities.voices || []).map((voice) => ({
                value: voice.voice_id || voice.id || voice.name || "",
                label: voice.name || voice.voice_id || voice.id || "Unknown"
              }))
          options = options.filter((voice) => voice.value)
        } else if (inferredTldwProviderKey === "kitten_tts") {
          options = KITTEN_FALLBACK_VOICES
        } else if (
          !inferredTldwProviderKey ||
          inferredTldwProviderKey === "kokoro"
        ) {
          options = KOKORO_FALLBACK_VOICES
        }
      }
      const provider = normalizeTtsProviderKey(catalogProvider)
      const ownedOptions = selectedBackend
        ? []
        : customVoices
            .filter(
              (voice) =>
                !voice.provider ||
                normalizeTtsProviderKey(voice.provider) === provider
            )
            .filter((voice) => Boolean(voice.voice_id))
            .map((voice) => ({
              value: `custom:${voice.voice_id}`,
              label: `Custom: ${voice.name || voice.voice_id}`
            }))
      return [
        ...ownedOptions,
        ...options.filter(
          (option) =>
            !ownedOptions.some((owned) => owned.value === option.value)
        )
      ]
    }
    if (audioSettings.provider === "openai") return OPENAI_TTS_VOICES
    return []
  }, [
    audioSettings.model,
    audioSettings.provider,
    catalogProvider,
    customVoices,
    inferredTldwProviderKey,
    selectedBackend,
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
    if (
      unresolvedBackend ||
      voiceCatalogSettled !== voiceCatalogKey ||
      (!selectedBackend && customVoiceCatalogSettled !== customVoiceCatalogKey)
    ) {
      return
    }
    const currentVoice = String(audioSettings.voice || "").trim()
    if (
      currentVoice &&
      voiceOptions.some((option) => option.value === currentVoice)
    ) {
      return
    }
    setAudioSettings({ voice: voiceOptions[0].value })
  }, [
    audioSettings.provider,
    audioSettings.voice,
    getVoiceOptions,
    tldwVoices.length,
    voiceCatalogSettled,
    voiceCatalogKey,
    customVoiceCatalogSettled,
    customVoiceCatalogKey,
    selectedBackend,
    unresolvedBackend,
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
  }, [audioSettings.provider, selectedBackendCapabilities, tldwModels])

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
      const result = await tldwClient.synthesizeSpeechDetailed(
        VOICE_PREVIEW_TEXT,
        {
          model: audioSettings.model,
          voice: audioSettings.voice,
          responseFormat: "mp3",
          speed: audioSettings.speed,
          backend: audioSettings.backend || undefined,
          allowFallback: audioSettings.allowFallback ?? true
        }
      )
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
    loadingVoices:
      loadingVoices ||
      (audioSettings.provider === "tldw" &&
        !selectedBackend &&
        customVoiceCatalogSettled !== customVoiceCatalogKey),
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
    handlePreviewVoice
  }
}
