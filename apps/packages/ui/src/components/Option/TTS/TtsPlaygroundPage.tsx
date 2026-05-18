import React from "react"
import {
  Button,
  Input,
  Alert,
  Progress,
  Typography,
  Space,
  Card,
  Tag,
  Select
} from "antd"
import { Trans, useTranslation } from "react-i18next"
import { Link } from "react-router-dom"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import {
  DEFAULT_TLDW_TTS_MODEL,
  DEFAULT_TLDW_TTS_VOICE,
  DEFAULT_TTS_PROVIDER,
  getTTSProvider,
  getTTSSettings,
  setTTSSettings
} from "@/services/tts"
import { inferTldwProviderFromModel } from "@/services/tts-provider"
import { getTtsProviderLabel } from "@/services/tts-providers"
import {
  type TldwTtsProviderCapabilities,
  type TldwTtsVoiceInfo
} from "@/services/tldw/audio-providers"
import {
  useTtsPlayground,
  type TtsPlaygroundSegment
} from "@/hooks/useTtsPlayground"
import {
  OPENAI_TTS_MODELS,
  OPENAI_TTS_VOICES,
  useTtsProviderData
} from "@/hooks/useTtsProviderData"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { PageShell } from "@/components/Common/PageShell"
import { TtsProviderPanel } from "@/components/Option/TTS/TtsProviderPanel"
import { isTimeoutLikeError } from "@/utils/request-timeout"
import { withTemplateFallback } from "@/utils/template-guards"

const { Text, Title, Paragraph } = Typography
const SAMPLE_TEXT =
  "Sample: Hi there, this is the TTS playground reading a short passage so you can preview voice and speed."
const TtsPlaygroundPage: React.FC = () => {
  const { t } = useTranslation(["playground", "settings", "common"])
  const [text, setText] = React.useState("")
  const { data: ttsSettings } = useQuery({
    queryKey: ["fetchTTSSettings"],
    queryFn: getTTSSettings
  })
  const queryClient = useQueryClient()
  const [elevenVoiceId, setElevenVoiceId] = React.useState<string | undefined>(
    undefined
  )
  const [elevenModelId, setElevenModelId] = React.useState<string | undefined>(
    undefined
  )
  const [tldwModel, setTldwModel] = React.useState<string | undefined>(
    DEFAULT_TLDW_TTS_MODEL
  )
  const [tldwVoice, setTldwVoice] = React.useState<string | undefined>(
    DEFAULT_TLDW_TTS_VOICE
  )
  const [openAiModel, setOpenAiModel] = React.useState<string | undefined>(
    undefined
  )
  const [openAiVoice, setOpenAiVoice] = React.useState<string | undefined>(
    undefined
  )
  const provider = ttsSettings?.ttsProvider || DEFAULT_TTS_PROVIDER
  const isTldw = provider === "tldw"
  const inferredProviderKey = React.useMemo(() => {
    if (!isTldw) return null
    return inferTldwProviderFromModel(tldwModel || ttsSettings?.tldwTtsModel)
  }, [isTldw, tldwModel, ttsSettings?.tldwTtsModel])
  const {
    hasAudio,
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
  const { capabilities } = useServerCapabilities()

  const {
    segments,
    isGenerating,
    generationProgress,
    generateSegments,
    clearSegments
  } = useTtsPlayground()
  const [activeSegmentIndex, setActiveSegmentIndex] = React.useState<
    number | null
  >(null)
  const audioRef = React.useRef<HTMLAudioElement | null>(null)
  const [currentTime, setCurrentTime] = React.useState(0)
  const [duration, setDuration] = React.useState(0)
  const [browserActiveIndex, setBrowserActiveIndex] = React.useState<
    number | null
  >(null)
  const [browserIsSpeaking, setBrowserIsSpeaking] = React.useState(false)
  const [browserIsPaused, setBrowserIsPaused] = React.useState(false)
  const browserQueueRef = React.useRef<number[]>([])
  const browserSegmentsRef = React.useRef<TtsPlaygroundSegment[]>([])
  const browserUtteranceRef = React.useRef<SpeechSynthesisUtterance | null>(
    null
  )
  const controlIds = {
    textInput: "tts-playground-input",
    elevenVoice: "tts-playground-eleven-voice",
    elevenModel: "tts-playground-eleven-model",
    tldwVoice: "tts-playground-tldw-voice",
    tldwModel: "tts-playground-tldw-model",
    openAiModel: "tts-playground-openai-model",
    openAiVoice: "tts-playground-openai-voice"
  }

  React.useEffect(() => {
    if (!ttsSettings) return
    setElevenVoiceId(ttsSettings.elevenLabsVoiceId || undefined)
    setElevenModelId(ttsSettings.elevenLabsModel || undefined)
    setTldwModel(ttsSettings.tldwTtsModel || DEFAULT_TLDW_TTS_MODEL)
    setTldwVoice(ttsSettings.tldwTtsVoice || DEFAULT_TLDW_TTS_VOICE)
    setOpenAiModel(ttsSettings.openAITTSModel || undefined)
    setOpenAiVoice(ttsSettings.openAITTSVoice || undefined)
  }, [ttsSettings])

  React.useEffect(() => {
    browserSegmentsRef.current = segments
  }, [segments])

  const clampPlaybackRate = (value: number) => {
    if (!Number.isFinite(value)) return 1
    return Math.min(2, Math.max(0.5, value))
  }

  const resolveBrowserVoice = React.useCallback(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) return null
    const target = ttsSettings?.voice
    if (!target) return null
    const voices = window.speechSynthesis.getVoices()
    return voices.find((voice) => voice.name === target) || null
  }, [ttsSettings?.voice])

  const stopBrowserSpeech = React.useCallback(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) return
    try {
      window.speechSynthesis.cancel()
    } catch {}
    browserQueueRef.current = []
    browserUtteranceRef.current = null
    setBrowserIsSpeaking(false)
    setBrowserIsPaused(false)
    setBrowserActiveIndex(null)
  }, [])

  const speakBrowserSegment = React.useCallback(
    (
      index: number,
      queue: number[] = [],
      nextSegments?: TtsPlaygroundSegment[]
    ) => {
      if (typeof window === "undefined" || !window.speechSynthesis) return
      const list = nextSegments || browserSegmentsRef.current
      const segment = list[index]
      if (!segment?.text) return

      window.speechSynthesis.cancel()
      browserQueueRef.current = queue
      setBrowserActiveIndex(index)
      setBrowserIsSpeaking(true)
      setBrowserIsPaused(false)

      const utterance = new SpeechSynthesisUtterance(segment.text)
      const voice = resolveBrowserVoice()
      if (voice) utterance.voice = voice
      utterance.rate = clampPlaybackRate(ttsSettings?.playbackSpeed ?? 1)
      utterance.onend = () => {
        const next = browserQueueRef.current.shift()
        if (typeof next === "number") {
          speakBrowserSegment(next, browserQueueRef.current, list)
          return
        }
        setBrowserIsSpeaking(false)
        setBrowserActiveIndex(null)
      }
      utterance.onerror = () => {
        setBrowserIsSpeaking(false)
        setBrowserIsPaused(false)
        setBrowserActiveIndex(null)
      }
      browserUtteranceRef.current = utterance
      window.speechSynthesis.speak(utterance)
    },
    [resolveBrowserVoice, ttsSettings?.playbackSpeed]
  )

  const queueBrowserSegments = React.useCallback(
    (startIndex: number, nextSegments?: TtsPlaygroundSegment[]) => {
      const list = nextSegments || browserSegmentsRef.current
      if (!list.length || startIndex < 0 || startIndex >= list.length) return
      const queue: number[] = []
      for (let i = startIndex + 1; i < list.length; i += 1) {
        queue.push(i)
      }
      speakBrowserSegment(startIndex, queue, list)
    },
    [speakBrowserSegment]
  )

  const pauseBrowserSpeech = React.useCallback(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) return
    if (!window.speechSynthesis.speaking || window.speechSynthesis.paused) {
      return
    }
    window.speechSynthesis.pause()
    setBrowserIsPaused(true)
  }, [])

  const resumeBrowserSpeech = React.useCallback(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) return
    if (!window.speechSynthesis.paused) return
    window.speechSynthesis.resume()
    setBrowserIsPaused(false)
  }, [])

  React.useEffect(() => {
    if (provider !== "browser") {
      stopBrowserSpeech()
    }
  }, [provider, stopBrowserSpeech])

  React.useEffect(() => {
    return () => {
      stopBrowserSpeech()
    }
  }, [stopBrowserSpeech])

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
  const handlePlay = async () => {
    if (!text.trim() || isTtsDisabled) return
    stopBrowserSpeech()
    clearSegments()
    setActiveSegmentIndex(null)
    setCurrentTime(0)
    setDuration(0)

    const effectiveProvider = ttsSettings?.ttsProvider || (await getTTSProvider())

    const nextSegments = await generateSegments(text, {
      provider: effectiveProvider,
      elevenLabsModel: elevenModelId,
      elevenLabsVoiceId: elevenVoiceId,
      tldwModel,
      tldwVoice,
      openAiModel,
      openAiVoice
    })

    if (ttsSettings) {
      void setTTSSettings({
        ttsEnabled: ttsSettings.ttsEnabled,
        ttsProvider: ttsSettings.ttsProvider,
        voice: ttsSettings.voice,
        ssmlEnabled: ttsSettings.ssmlEnabled,
        elevenLabsApiKey: ttsSettings.elevenLabsApiKey,
        elevenLabsVoiceId: elevenVoiceId ?? ttsSettings.elevenLabsVoiceId,
        elevenLabsModel: elevenModelId ?? ttsSettings.elevenLabsModel,
        responseSplitting: ttsSettings.responseSplitting,
        removeReasoningTagTTS: ttsSettings.removeReasoningTagTTS,
        openAITTSBaseUrl: ttsSettings.openAITTSBaseUrl,
        openAITTSApiKey: ttsSettings.openAITTSApiKey,
        openAITTSModel: openAiModel ?? ttsSettings.openAITTSModel,
        openAITTSVoice: openAiVoice ?? ttsSettings.openAITTSVoice,
        ttsAutoPlay: ttsSettings.ttsAutoPlay,
        playbackSpeed: ttsSettings.playbackSpeed,
        tldwTtsModel: tldwModel ?? ttsSettings.tldwTtsModel,
        tldwTtsVoice: tldwVoice ?? ttsSettings.tldwTtsVoice,
        tldwTtsResponseFormat: ttsSettings.tldwTtsResponseFormat,
        tldwTtsSpeed: ttsSettings.tldwTtsSpeed,
        tldwTtsLanguage: ttsSettings.tldwTtsLanguage,
        tldwTtsStreaming: ttsSettings.tldwTtsStreaming,
        tldwTtsEmotion: ttsSettings.tldwTtsEmotion,
        tldwTtsEmotionIntensity: ttsSettings.tldwTtsEmotionIntensity,
        tldwTtsNormalize: ttsSettings.tldwTtsNormalize,
        tldwTtsNormalizeUnits: ttsSettings.tldwTtsNormalizeUnits,
        tldwTtsNormalizeUrls: ttsSettings.tldwTtsNormalizeUrls,
        tldwTtsNormalizeEmails: ttsSettings.tldwTtsNormalizeEmails,
        tldwTtsNormalizePhones: ttsSettings.tldwTtsNormalizePhones,
        tldwTtsNormalizePlurals: ttsSettings.tldwTtsNormalizePlurals
      }).then(() => {
        queryClient.invalidateQueries({ queryKey: ["fetchTTSSettings"] })
      })
    }

    if (effectiveProvider === "browser") {
      queueBrowserSegments(0, nextSegments)
    } else if (nextSegments.length > 0) {
      setActiveSegmentIndex(0)
    }
  }

  const handleStop = () => {
    stopBrowserSpeech()
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
      const match = entries.find(
        ([k]) => k.toLowerCase() === inferredProviderKey.toLowerCase()
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
    const fallbackKey = activeProviderCaps.key.toLowerCase()
    const fallback = allVoices[fallbackKey]
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
    const fallbackKey = activeProviderCaps.key.toLowerCase()
    const fallback = allVoices[fallbackKey]
    if (Array.isArray(fallback) && fallback.length > 0) {
      return fallback
    }
    return []
  }, [providersInfo, activeProviderCaps, tldwVoiceCatalog])

  React.useEffect(() => {
    if (!isTldw || providerVoices.length === 0) {
      return
    }
    const currentVoice = String(tldwVoice || "").trim()
    const availableVoices = providerVoices
      .map((voice) => voice.id || voice.name || "")
      .filter(Boolean)
    if (currentVoice && availableVoices.includes(currentVoice)) {
      return
    }
    setTldwVoice(availableVoices[0])
  }, [isTldw, providerVoices, tldwVoice])

  const openAiVoiceOptions = React.useMemo(() => {
    if (!openAiModel) {
      // If no model selected yet, show the union of known voices.
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

  const effectiveElevenVoice =
    elevenVoiceId ?? ttsSettings?.elevenLabsVoiceId
  const effectiveElevenModel =
    elevenModelId ?? ttsSettings?.elevenLabsModel
  const isElevenLabsConfigured = Boolean(
    ttsSettings?.elevenLabsApiKey &&
      effectiveElevenVoice &&
      effectiveElevenModel
  )
  const playDisabledReason = isTtsDisabled
    ? t(
        "playground:tts.playDisabledTtsOff",
        "Enable text-to-speech above to play audio."
      )
    : !text.trim()
      ? t("playground:tts.playDisabledNoText", "Enter text to enable Play.")
      : ttsSettings?.ttsProvider === "elevenlabs" &&
          !isElevenLabsConfigured
        ? t(
            "playground:tts.playDisabledElevenLabs",
            "Add an ElevenLabs API key, voice, and model to enable Play."
          )
        : null
  const isPlayDisabled = isGenerating || Boolean(playDisabledReason)
  const canStop =
    provider === "browser"
      ? browserIsSpeaking || browserIsPaused
      : Boolean(segments.length || audioRef.current)
  const stopDisabledReason =
    !canStop &&
    t(
      "playground:tts.stopDisabled",
      "Stop activates after audio starts."
    )
  const hasElevenLabsKey = Boolean(ttsSettings?.elevenLabsApiKey)
  const showElevenLabsHint =
    ttsSettings?.ttsProvider === "elevenlabs" &&
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
        "Add your ElevenLabs API key in Settings to load voices and models."
      )
  const elevenLabsTimeoutBody = t(
    "playground:tts.elevenLabsTimeoutBody",
    "Loading voices/models took longer than 10 seconds. Retry or verify network access."
  )

  const handleElevenLabsApiKeyFocus = () => {
    const el = document.getElementById("elevenlabs-api-key")
    if (!el) return
    try {
      el.scrollIntoView({ block: "center" })
    } catch {}
    ;(el as HTMLElement).focus()
  }

  return (
    <PageShell maxWidthClassName="max-w-3xl" className="py-6">
      <Title level={1} className="!mb-1 !text-2xl">
        {t("playground:textToSpeech", "Text to Speech")}
      </Title>
      <Text type="secondary">
        {t(
          "playground:tts.subtitle",
          "Try out text-to-speech and tweak providers, models, and voices."
        )}
      </Text>
      <p className="text-[11px] text-text-subtle mt-1">
        <Trans
          i18nKey="playground:tts.combinedWorkflowHint"
          defaults="For combined TTS + STT workflows, try the <speechLink>Speech Playground</speechLink>."
          components={{ speechLink: <Link to="/speech" className="underline" /> }}
        />
      </p>
      <div className="mt-1">
        <a
          href="https://github.com/rmusser01/tldw_server/blob/main/Docs/Getting_Started/First_Time_Audio_Setup_CPU.md"
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-text-muted underline hover:text-text"
        >
          {t("playground:tts.audioSetupGuide", "Audio Setup Guide")}
        </a>
      </div>

      {capabilities?.ffmpegAvailable === false && (
        <Alert
          type="warning"
          showIcon
          className="mt-3 mb-0"
          message={t("playground:tts.ffmpegWarningTitle", "ffmpeg not detected")}
          description={t("playground:tts.ffmpegWarningBody", "Some audio processing features require ffmpeg. Install it for full functionality.")}
        />
      )}

      <div className="mt-4 space-y-4">
        <div data-testid="tts-provider-selector">
          <TtsProviderPanel
            providerLabel={providerLabel}
            provider={provider}
            ttsSettings={ttsSettings}
            isTldw={isTldw}
            hasAudio={hasAudio}
            activeProviderCaps={activeProviderCaps}
            activeVoices={activeVoices}
            providersInfo={providersInfo}
          />
        </div>

        {isTldw && !hasAudio && (
          <Alert
            type="info"
            showIcon
            className="mb-4"
            message={t("playground:tts.noProviderTitle", "No TTS provider detected on your server")}
            description={
              <Trans
                i18nKey="playground:tts.noProviderDescription"
                defaults="Your tldw server doesn't have a TTS engine configured yet. <settingsLink>Open Speech Settings</settingsLink> to configure one, or switch to <strong>Browser</strong> TTS which works without any setup."
                components={{
                  settingsLink: <Link to="/settings/speech" />,
                  strong: <strong />
                }}
              />
            }
          />
        )}

        {isTldw && !hasAudio && (
          <Alert
            type="info"
            showIcon
            className="mb-4"
            message={t("playground:tts.noProviderTitle", "No TTS provider detected on your server")}
            description={
              <Trans
                i18nKey="playground:tts.noProviderDescription"
                defaults="Your tldw server doesn't have a TTS engine configured yet. <settingsLink>Open Speech Settings</settingsLink> to configure one, or switch to <strong>Browser</strong> TTS which works without any setup."
                components={{
                  settingsLink: <Link to="/settings/speech" />,
                  strong: <strong />
                }}
              />
            }
          />
        )}

        <Card>
          <Space orientation="vertical" className="w-full" size="middle">
            <div>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <Paragraph className="!mb-2 !mr-2">
                  {t(
                    "playground:tts.inputLabel",
                    "Enter some text to hear it spoken."
                  )}
                </Paragraph>
                <Button
                  size="small"
                  onClick={() => setText(SAMPLE_TEXT)}
                  aria-label={t(
                    "playground:tts.sampleText",
                    "Insert sample text"
                  ) as string}
                >
                  {t("playground:tts.sampleText", "Insert sample text")}
                </Button>
              </div>
              <Input.TextArea
                id={controlIds.textInput}
                data-testid="tts-text-input"
                aria-label={t(
                  "playground:tts.inputLabel",
                  "Enter some text to hear it spoken."
                ) as string}
                value={text}
                onChange={(e) => setText(e.target.value)}
                autoSize={{ minRows: 4, maxRows: 10 }}
                placeholder={t(
                  "playground:tts.inputPlaceholder",
                  "Type or paste text here, then use Play to listen."
                ) as string}
              />
            </div>

            {showElevenLabsHint && (
              <Alert
                type={hasElevenLabsKey ? "warning" : "info"}
                showIcon
                title={elevenLabsHintTitle}
                description={
                  <div className="flex flex-wrap items-center gap-2">
                    <span>
                      {hasElevenLabsLoadError && isTimeoutLikeError(elevenLabsError)
                        ? elevenLabsTimeoutBody
                        : elevenLabsHintBody}
                    </span>
                    {hasElevenLabsKey && (
                      <Button
                        size="small"
                        type="link"
                        onClick={() => {
                          void refetchElevenLabs()
                        }}
                      >
                        {t("common:retry", "Retry")}
                      </Button>
                    )}
                    <Button
                      size="small"
                      type="link"
                      onClick={handleElevenLabsApiKeyFocus}
                    >
                      {t(
                        "playground:tts.elevenLabsMissingCta",
                        "Set API key in Settings"
                      )}
                    </Button>
                  </div>
                }
              />
            )}

            {ttsSettings?.ttsProvider === "elevenlabs" && elevenLabsData && (
              <div className="flex flex-col gap-2" data-testid="tts-voice-picker-elevenlabs">
                <Text type="secondary">
                  {t(
                    "playground:tts.voiceSelector.elevenLabs",
                    "Choose an ElevenLabs voice and model for this run."
                  )}
                </Text>
                <Space className="flex flex-wrap" size="middle">
                  <div>
                    <label
                      className="block text-xs mb-1 text-text"
                      htmlFor={controlIds.elevenVoice}>
                      Voice
                    </label>
                    <Select
                      id={controlIds.elevenVoice}
                      aria-label="ElevenLabs voice"
                      style={{ minWidth: 160 }}
                      placeholder="Select voice"
                      className="focus-ring"
                      options={elevenLabsData.voices.map((v: any) => ({
                        label: v.name,
                        value: v.voice_id
                      }))}
                      value={elevenVoiceId}
                      onChange={(val) => setElevenVoiceId(val)}
                    />
                  </div>
                  <div>
                    <label
                      className="block text-xs mb-1 text-text"
                      htmlFor={controlIds.elevenModel}>
                      Model
                    </label>
                    <Select
                      id={controlIds.elevenModel}
                      aria-label="ElevenLabs model"
                      style={{ minWidth: 160 }}
                      placeholder="Select model"
                      className="focus-ring"
                      options={elevenLabsData.models.map((m: any) => ({
                        label: m.name,
                        value: m.model_id
                      }))}
                      value={elevenModelId}
                      onChange={(val) => setElevenModelId(val)}
                    />
                  </div>
                </Space>
              </div>
            )}

            {isTldw && providerVoices.length > 0 && (
              <div className="flex flex-col gap-2" data-testid="tts-voice-picker-tldw">
                <Text type="secondary">
                  {t(
                    "playground:tts.voiceSelector.tldw",
                    "Choose a server voice and model for this run."
                  )}
                </Text>
                <Space className="flex flex-wrap" size="middle">
                  <div>
                    <label
                      className="block text-xs mb-1 text-text"
                      htmlFor={controlIds.tldwVoice}>
                      Voice
                    </label>
                    <Select
                      id={controlIds.tldwVoice}
                      aria-label="tldw server voice"
                      style={{ minWidth: 200 }}
                      placeholder="Select voice"
                      className="focus-ring"
                      options={providerVoices.map((v, idx) => ({
                        label: `${v.name || v.id || `Voice ${idx + 1}`}${
                          v.language ? ` (${v.language})` : ""
                        }`,
                        value: v.id || v.name || ""
                      }))}
                      value={tldwVoice}
                      onChange={(val) => setTldwVoice(val)}
                    />
                  </div>
                  <div>
                    <label
                      className="block text-xs mb-1 text-text"
                      htmlFor={controlIds.tldwModel}>
                      Model
                    </label>
                    {tldwTtsModels && tldwTtsModels.length > 0 ? (
                      <Select
                        id={controlIds.tldwModel}
                        aria-label="tldw server model"
                        style={{ minWidth: 160 }}
                        placeholder="Select model"
                        showSearch
                        optionFilterProp="label"
                        className="focus-ring"
                        options={tldwTtsModels.map((m) => ({
                          label: m.label,
                          value: m.id
                        }))}
                        value={tldwModel}
                        onChange={(val) => setTldwModel(val)}
                      />
                    ) : (
                      <Input
                        aria-label="tldw server model"
                        style={{ minWidth: 160 }}
                        value={tldwModel || ""}
                        onChange={(e) => setTldwModel(e.target.value)}
                        placeholder={DEFAULT_TLDW_TTS_MODEL}
                      />
                    )}
                  </div>
                </Space>
              </div>
            )}

            {ttsSettings?.ttsProvider === "openai" && (
              <div className="flex flex-col gap-2" data-testid="tts-voice-picker-openai">
                <Text type="secondary">
                  {t(
                    "playground:tts.voiceSelector.openai",
                    "Choose an OpenAI TTS model and voice for this run."
                  )}
                </Text>
                <Space className="flex flex-wrap" size="middle">
                  <div>
                    <label
                      className="block text-xs mb-1 text-text"
                      htmlFor={controlIds.openAiModel}>
                      Model
                    </label>
                    <Select
                      id={controlIds.openAiModel}
                      aria-label="OpenAI TTS model"
                      style={{ minWidth: 160 }}
                      placeholder="Select model"
                      className="focus-ring"
                      options={OPENAI_TTS_MODELS}
                      value={openAiModel}
                      onChange={(val) => {
                        setOpenAiModel(val)
                        const voicesForModel =
                          OPENAI_TTS_VOICES[val] || openAiVoiceOptions
                        if (
                          voicesForModel.length > 0 &&
                          !voicesForModel.find((v) => v.value === openAiVoice)
                        ) {
                          setOpenAiVoice(voicesForModel[0].value)
                        }
                      }}
                    />
                  </div>
                  <div>
                    <label
                      className="block text-xs mb-1 text-text"
                      htmlFor={controlIds.openAiVoice}>
                      Voice
                    </label>
                    <Select
                      id={controlIds.openAiVoice}
                      aria-label="OpenAI TTS voice"
                      style={{ minWidth: 160 }}
                      placeholder="Select voice"
                      className="focus-ring"
                      options={openAiVoiceOptions}
                      value={openAiVoice}
                      onChange={(val) => setOpenAiVoice(val)}
                    />
                  </div>
                </Space>
              </div>
            )}

            <Space>
              <Button
                type="primary"
                data-testid="tts-play-button"
                onClick={handlePlay}
                disabled={isPlayDisabled}
                loading={isGenerating}
              >
                {isGenerating
                  ? t("playground:tts.playing", "Playing…")
                  : t("playground:tts.play", "Play")}
              </Button>
              <Button
                onClick={handleStop}
                disabled={!canStop}
              >
                {t("playground:tts.stop", "Stop")}
              </Button>
            </Space>
            <Text type="secondary" className="text-xs">
              {playDisabledReason ||
                t(
                  "playground:tts.playHelper",
                  "Play uses your selected provider, voice, and speed."
                )}
              {!canStop && stopDisabledReason ? ` ${stopDisabledReason}` : ""}
            </Text>

            {isGenerating && generationProgress && generationProgress.total > 1 && (
              <div className="w-full">
                <Progress
                  percent={Math.round(
                    (generationProgress.completed / generationProgress.total) * 100
                  )}
                  size="small"
                  status="active"
                  format={() =>
                    t("playground:tts.progressSegments", "{{completed}}/{{total}} segments", {
                      completed: generationProgress.completed,
                      total: generationProgress.total
                    })
                  }
                />
              </div>
            )}

            {provider === "browser" && segments.length > 0 && (
              <div className="mt-4 space-y-2 w-full" data-testid="tts-audio-output">
                <div>
                  <Text strong>
                    {t(
                      "playground:tts.browserSegmentsTitle",
                      "Browser TTS segments"
                    )}
                  </Text>
                  <Paragraph className="!mb-1 text-xs text-text-subtle">
                    {t(
                      "playground:tts.browserSegmentsHelp",
                      "Queue segments or play an individual segment using system audio."
                    )}
                  </Paragraph>
                </div>
                <div className="border border-border rounded-md p-3 space-y-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <Button
                      size="small"
                      type="primary"
                      onClick={() => queueBrowserSegments(0)}
                    >
                      {t("playground:tts.browserQueueAll", "Queue all")}
                    </Button>
                    <Button
                      size="small"
                      onClick={pauseBrowserSpeech}
                      disabled={!browserIsSpeaking || browserIsPaused}
                    >
                      {t("playground:tts.browserPause", "Pause")}
                    </Button>
                    <Button
                      size="small"
                      onClick={resumeBrowserSpeech}
                      disabled={!browserIsPaused}
                    >
                      {t("playground:tts.browserResume", "Resume")}
                    </Button>
                    <Button
                      size="small"
                      onClick={stopBrowserSpeech}
                      disabled={!browserIsSpeaking && !browserIsPaused}
                    >
                      {t("playground:tts.stop", "Stop")}
                    </Button>
                    {browserActiveIndex != null && (
                      <Text type="secondary" className="text-xs">
                        {browserIsPaused
                          ? `${t(
                              "playground:tts.browserStatusPaused",
                              "Paused"
                            )} · `
                          : ""}
                        {withTemplateFallback(
                          t(
                            "playground:tts.browserStatusSpeaking",
                            "Speaking segment {{current}}/{{total}}",
                            {
                              current: browserActiveIndex + 1,
                              total: segments.length
                            }
                          ),
                          `Speaking segment ${browserActiveIndex + 1}/${segments.length}`
                        )}
                      </Text>
                    )}
                  </div>
                  <div className="space-y-2">
                    {segments.map((seg, idx) => (
                      <div
                        key={seg.id}
                        className="rounded border border-border p-2 space-y-2"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <Text strong>
                            {t("playground:tts.segmentLabel", "Part")}{" "}
                            {idx + 1}
                          </Text>
                          {browserActiveIndex === idx && (
                            <Tag
                              color={browserIsPaused ? "gold" : "green"}
                              bordered
                            >
                              {browserIsPaused
                                ? t(
                                    "playground:tts.browserStatusPaused",
                                    "Paused"
                                  )
                                : t(
                                    "playground:tts.playing",
                                    "Playing…"
                                  )}
                            </Tag>
                          )}
                        </div>
                        <Paragraph
                          className="!mb-0 text-xs text-text-muted"
                          ellipsis={{ rows: 2 }}
                        >
                          {seg.text}
                        </Paragraph>
                        <Space size="small" wrap>
                          <Button
                            size="small"
                            type="primary"
                            onClick={() => speakBrowserSegment(idx)}
                          >
                            {t(
                              "playground:tts.browserSegmentPlay",
                              "Play segment"
                            )}
                          </Button>
                          <Button
                            size="small"
                            onClick={() => queueBrowserSegments(idx)}
                          >
                            {t(
                              "playground:tts.browserQueueFromHere",
                              "Queue from here"
                            )}
                          </Button>
                        </Space>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {provider !== "browser" && segments.length > 0 && (
              <div className="mt-4 space-y-2 w-full" data-testid="tts-audio-output">
                <div>
                  <Text strong>
                    {t(
                      "playground:tts.outputTitle",
                      "Generated audio segments"
                    )}
                  </Text>
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
                  <div className="flex items-center justify-between text-xs text-text-subtle">
                    <span>
                      {activeSegmentIndex != null
                        ? t("playground:tts.currentSegment", "Segment") +
                          ` ${activeSegmentIndex + 1}/${segments.length}`
                        : t(
                            "playground:tts.currentSegmentNone",
                            "No segment selected"
                          )}
                    </span>
                    {duration > 0 && (
                      <span>
                        {t("playground:tts.timeLabel", "Time")}:{" "}
                        {Math.floor(currentTime)}s / {Math.floor(duration)}s
                      </span>
                    )}
                  </div>
                  <div className="flex flex-wrap gap-2 pt-1">
                    {segments.map((seg, idx) => (
                      <Button
                        key={seg.id}
                        size="small"
                        type={
                          idx ===
                          (activeSegmentIndex != null
                            ? activeSegmentIndex
                            : 0)
                            ? "primary"
                            : "default"
                        }
                        onClick={() => handleSegmentSelect(idx)}
                      >
                        {t("playground:tts.segmentLabel", "Part")} {idx + 1}
                      </Button>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </Space>
        </Card>

      </div>
    </PageShell>
  )
}

export default TtsPlaygroundPage
