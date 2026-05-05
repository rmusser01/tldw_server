import React, { useState, useEffect, useRef } from "react"
import type { MessageInstance } from "antd/es/message/interface"
import type { TFunction } from "i18next"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { fetchTldwVoiceCatalog, type TldwVoice } from "@/services/tldw/audio-voices"
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
  const [loadingVoices, setLoadingVoices] = useState(false)
  const [voiceCatalogSettled, setVoiceCatalogSettled] = useState(false)
  const [previewingVoice, setPreviewingVoice] = useState(false)
  const previewAudioRef = useRef<HTMLAudioElement | null>(null)

  const inferredTldwProviderKey = inferTldwProviderFromModel(audioSettings.model)

  // Fetch voices when provider changes to tldw
  useEffect(() => {
    let cancelled = false

    if (audioSettings.provider !== "tldw") {
      setTldwVoices([])
      setLoadingVoices(false)
      setVoiceCatalogSettled(true)
      return
    }
    if (!inferredTldwProviderKey) {
      setTldwVoices([])
      setLoadingVoices(false)
      setVoiceCatalogSettled(true)
      return
    }
    setVoiceCatalogSettled(false)
    setLoadingVoices(true)
    fetchTldwVoiceCatalog(inferredTldwProviderKey)
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
  }, [audioSettings.provider, inferredTldwProviderKey])

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
      if (inferredTldwProviderKey === "kitten_tts") {
        return KITTEN_FALLBACK_VOICES
      }
      return KOKORO_FALLBACK_VOICES
    }
    if (audioSettings.provider === "openai") {
      return OPENAI_TTS_VOICES
    }
    return []
  }, [audioSettings.provider, inferredTldwProviderKey, tldwVoices])

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
      return TLDW_TTS_MODELS
    }
    if (audioSettings.provider === "openai") {
      return OPENAI_TTS_MODELS
    }
    return []
  }, [audioSettings.provider])

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
      const audioBuffer = await tldwClient.synthesizeSpeech(VOICE_PREVIEW_TEXT, {
        model: audioSettings.model,
        voice: audioSettings.voice,
        responseFormat: "mp3",
        speed: audioSettings.speed
      })
      const audioBlob = new Blob([audioBuffer], { type: "audio/mpeg" })
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
    // callbacks
    getVoiceOptions,
    getModelOptions,
    handlePreviewVoice,
  }
}
