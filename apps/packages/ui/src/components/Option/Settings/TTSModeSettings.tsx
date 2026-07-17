import { SaveButton } from "@/components/Common/SaveButton"
import { generateSpeech, getModels, getVoices } from "@/services/elevenlabs"
import { generateOpenAITTS } from "@/services/openai-tts"
import {
  DEFAULT_TLDW_TTS_MODEL,
  getTTSSettings,
  setElevenLabsApiKey,
  setElevenLabsKeyTestedAt,
  setElevenLabsKeyValid,
  setTTSSettings,
  SUPPORTED_TLDW_TTS_FORMATS
} from "@/services/tts"
import { formatToMimeType, inferTldwProviderFromModel } from "@/services/tts-provider"
import { TTS_PROVIDER_OPTIONS } from "@/services/tts-providers"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  fetchTldwVoiceCatalog,
  fetchTldwVoices,
  type TldwVoice
} from "@/services/tldw/audio-voices"
import {
  fetchTtsProviders,
  type TldwTtsProvidersInfo,
  type TldwTtsVoiceInfo,
  type TldwTtsProviderCapabilities
} from "@/services/tldw/audio-providers"
import {
  fetchTldwTtsModels,
  type TldwTtsModel
} from "@/services/tldw/audio-models"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { normalizeTtsProviderKey, toServerTtsProviderKey } from "@/services/tldw/tts-provider-keys"
import { listCustomVoices, type TldwCustomVoice } from "@/services/tldw/voice-cloning"
import { useWebUI } from "@/store/webui"
import { Alert, Button, Input, InputNumber, Select, Skeleton, Switch, Space, Tag } from "antd"
import {
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  QuestionCircleOutlined
} from "@ant-design/icons"
import { Play, Square } from "lucide-react"
import { useTranslation } from "react-i18next"
import React, { useState } from "react"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import { useSimpleForm } from "@/hooks/useSimpleForm"
import { isTimeoutLikeError } from "@/utils/request-timeout"

const ELEVENLABS_KEY_DEBOUNCE_MS = 350
const TTS_PREVIEW_TEXT =
  "This is a quick voice preview from your current audio settings."
const isCanonicalBackendId = (value: unknown): value is string =>
  value === "openrouter" ||
  (typeof value === "string" &&
    /^gateway:[a-z0-9][a-z0-9-]{0,62}$/.test(value))

const formatValidationTimestamp = (value?: string | null) => {
  if (!value) return ""
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleString()
}

const getErrorSignature = (error: unknown) => {
  if (!error) return ""
  if (error instanceof Error) return error.message
  if (typeof error === "string") return error
  try {
    return JSON.stringify(error)
  } catch {
    return String(error)
  }
}

const isAbortError = (error: unknown) =>
  typeof DOMException !== "undefined" && error instanceof DOMException
    ? error.name === "AbortError"
    : error instanceof Error && error.name === "AbortError"

export const TTSModeSettings = ({ hideBorder }: { hideBorder?: boolean }) => {
  const { t } = useTranslation("settings")
  const message = useAntdMessage()
  const { setTTSEnabled } = useWebUI()
  const queryClient = useQueryClient()

  // API key test states
  const [elevenLabsTestResult, setElevenLabsTestResult] = useState<{ ok: boolean; message: string } | null>(null)
  const [testingElevenLabs, setTestingElevenLabs] = useState(false)
  const [debouncedElevenLabsApiKey, setDebouncedElevenLabsApiKey] = useState("")
  const lastPersistedElevenLabsValidationRef = React.useRef<string | null>(null)
  const [previewState, setPreviewState] = useState<"idle" | "loading" | "playing">("idle")
  const [previewError, setPreviewError] = useState("")
  const previewAudioRef = React.useRef<HTMLAudioElement | null>(null)
  const previewUrlRef = React.useRef<string | null>(null)
  const previewAbortRef = React.useRef<AbortController | null>(null)

  const ids = {
    ttsEnabled: "tts-enabled-toggle",
    ttsAutoPlay: "tts-auto-play-toggle",
    ttsProvider: "tts-provider-select",
    browserVoice: "browser-voice-select",
    elevenApiKey: "elevenlabs-api-key",
    elevenVoice: "elevenlabs-voice-select",
    elevenModel: "elevenlabs-model-select",
    tldwBackend: "tldw-backend-select",
    tldwModel: "tldw-model-select",
    tldwVoice: "tldw-voice-select",
    tldwResponseFormat: "tldw-response-format",
    tldwSpeed: "tldw-speed-input",
    tldwLanguage: "tldw-language-select",
    tldwStreaming: "tldw-streaming-toggle",
    tldwEmotion: "tldw-emotion-select",
    tldwEmotionIntensity: "tldw-emotion-intensity",
    tldwNormalize: "tldw-normalize-toggle",
    tldwNormalizeUnits: "tldw-normalize-units-toggle",
    tldwNormalizeUrls: "tldw-normalize-urls-toggle",
    tldwNormalizeEmails: "tldw-normalize-emails-toggle",
    tldwNormalizePhones: "tldw-normalize-phones-toggle",
    tldwNormalizePlurals: "tldw-normalize-plurals-toggle",
    ssmlEnabled: "tts-ssml-toggle",
    removeReasoning: "tts-remove-reasoning-toggle",
    playbackSpeed: "tts-playback-speed-input",
    openAiModel: "openai-model-select",
    openAiVoice: "openai-voice-select"
  }

  const form = useSimpleForm({
    initialValues: {
      ttsEnabled: false,
      ttsProvider: "",
      voice: "",
      ssmlEnabled: false,
      removeReasoningTagTTS: true,
      elevenLabsApiKey: "",
      elevenLabsVoiceId: "",
      elevenLabsModel: "",
      elevenLabsKeyValid: null as boolean | null,
      elevenLabsKeyTestedAt: "",
      responseSplitting: "",
      openAITTSBaseUrl: "",
      openAITTSApiKey: "",
      openAITTSModel: "",
      openAITTSVoice: "",
      openAITTSKeyValid: null as boolean | null,
      openAITTSKeyTestedAt: "",
      ttsAutoPlay: false,
      playbackSpeed: 1,
    tldwTtsBackend: "",
    tldwTtsModel: "",
    tldwTtsVoice: "",
    tldwTtsResponseFormat: "mp3",
    tldwTtsSpeed: 1,
    tldwTtsLanguage: "",
    tldwTtsStreaming: false,
    tldwTtsEmotion: "",
    tldwTtsEmotionIntensity: 1,
    tldwTtsNormalize: true,
    tldwTtsNormalizeUnits: false,
    tldwTtsNormalizeUrls: true,
    tldwTtsNormalizeEmails: true,
    tldwTtsNormalizePhones: true,
    tldwTtsNormalizePlurals: true
    },
    validate: {
      playbackSpeed: (value) =>
        value === null || value === undefined
          ? (t(
              "generalSettings.tts.playbackSpeed.required",
              "Playback speed is required"
            ) as string)
          : null
    }
  })
  const {
    setFieldValue: setFormFieldValue,
    setValues: setFormValues
  } = form

  const { status, data, refetch } = useQuery({
    queryKey: ["fetchTTSSettings"],
    queryFn: async () => {
      const data = await getTTSSettings()
      form.setValues(data)
      form.resetDirty(data)
      return data
    }
  })

  React.useEffect(() => {
    const key = String(form.values.elevenLabsApiKey || "").trim()
    const timer = window.setTimeout(() => {
      setDebouncedElevenLabsApiKey(key)
    }, ELEVENLABS_KEY_DEBOUNCE_MS)
    return () => window.clearTimeout(timer)
  }, [form.values.elevenLabsApiKey])

  const { data: elevenLabsData, error: elevenLabsError } = useQuery({
    queryKey: ["fetchElevenLabsData", debouncedElevenLabsApiKey],
    queryFn: async () => {
      const [voices, models] = await Promise.all([
        getVoices(debouncedElevenLabsApiKey),
        getModels(debouncedElevenLabsApiKey)
      ])
      return { voices, models }
    },
    enabled:
      form.values.ttsProvider === "elevenlabs" && !!debouncedElevenLabsApiKey
  })

  const persistElevenLabsValidation = React.useCallback(
    async (
      ok: boolean,
      validatedKey?: string,
      options?: { notifyOnError?: boolean }
    ) => {
      const currentKey = String(form.values.elevenLabsApiKey || "").trim()
      if (validatedKey && currentKey !== validatedKey) return
      const testedAt = new Date().toISOString()
      const validationValues = {
        elevenLabsApiKey: currentKey,
        elevenLabsKeyValid: ok,
        elevenLabsKeyTestedAt: testedAt
      }

      try {
        await Promise.all([
          setElevenLabsApiKey(currentKey),
          setElevenLabsKeyValid(ok),
          setElevenLabsKeyTestedAt(testedAt)
        ])
        const wasDirty = form.isDirty()
        const nextFormValues = {
          ...form.values,
          ...validationValues
        }
        form.setValues(validationValues)
        if (!wasDirty) {
          form.resetDirty(nextFormValues)
        }
        queryClient.setQueryData(
          ["fetchTTSSettings"],
          (previous: typeof form.values | undefined) => ({
            ...(previous || form.values),
            ...validationValues
          })
        )
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error("Failed to persist ElevenLabs validation status:", error)
        if (options?.notifyOnError) {
          message.error(
            t(
              "generalSettings.tts.apiKeyStatus.persistError",
              "API key validation completed, but the validation status could not be saved."
            ) as string
          )
        }
      }
    },
    [form, message, queryClient, t]
  )

  React.useEffect(() => {
    if (form.values.ttsProvider !== "elevenlabs" || !debouncedElevenLabsApiKey) {
      return
    }
    if (!elevenLabsData && !elevenLabsError) return
    const ok =
      Boolean(elevenLabsData) &&
      !elevenLabsError &&
      Array.isArray(elevenLabsData.voices) &&
      elevenLabsData.voices.length > 0 &&
      Array.isArray(elevenLabsData.models) &&
      elevenLabsData.models.length > 0
    const persistKey = `${debouncedElevenLabsApiKey}:${ok}:${getErrorSignature(
      elevenLabsError
    )}`
    if (lastPersistedElevenLabsValidationRef.current === persistKey) return
    lastPersistedElevenLabsValidationRef.current = persistKey
    void persistElevenLabsValidation(ok, debouncedElevenLabsApiKey)
  }, [
    debouncedElevenLabsApiKey,
    elevenLabsData,
    elevenLabsError,
    form.values.ttsProvider,
    persistElevenLabsValidation
  ])

  const inferredTldwProviderKey = React.useMemo(
    () => inferTldwProviderFromModel(form.values.tldwTtsModel),
    [form.values.tldwTtsModel]
  )

  const {
    data: tldwProvidersInfo,
    isLoading: tldwProvidersLoading,
    error: tldwProvidersError,
    refetch: refetchTldwProviders
  } = useQuery<TldwTtsProvidersInfo | null>({
    queryKey: ["fetchTldwTtsProviders"],
    queryFn: () => fetchTtsProviders({ throwOnError: true }),
    enabled: form.values.ttsProvider === "tldw",
    retry: false
  })

  const explicitBackendSupported =
    tldwProvidersInfo?.supports_explicit_backend === true
  const configuredBackend = String(form.values.tldwTtsBackend || "")
  const selectedBackend = explicitBackendSupported ? configuredBackend : ""
  const selectedBackendCaps = selectedBackend
    ? tldwProvidersInfo?.providers?.[selectedBackend] || null
    : null
  const selectedModelCaps = selectedBackendCaps?.model_capabilities?.[
    form.values.tldwTtsModel
  ]
  const selectedModelVoices = React.useMemo(
    () =>
      Array.isArray(selectedModelCaps?.voices)
        ? selectedModelCaps.voices.filter(
            (voice): voice is string =>
              typeof voice === "string" && Boolean(voice)
          )
        : [],
    [selectedModelCaps]
  )
  const selectedModelDefaultVoice =
    typeof selectedModelCaps?.default_voice === "string"
      ? selectedModelCaps.default_voice
      : ""
  const requiresFreeformVoice = Boolean(
    selectedBackend &&
      (!selectedModelCaps ||
        selectedModelCaps.requires_freeform_voice === true ||
        (!selectedModelDefaultVoice && selectedModelVoices.length === 0))
  )

  const tldwBackendOptions = React.useMemo(
    () => [
      {
        label: "Automatic (legacy model inference)",
        value: ""
      },
      ...Object.entries(tldwProvidersInfo?.providers || {})
        .filter(
          ([backend, caps]) =>
            typeof caps.display_name === "string" ||
            backend === "openrouter" ||
            backend.startsWith("gateway:")
        )
        .map(([backend, caps]) => ({
          label: caps.display_name || backend,
          value: backend
        }))
    ],
    [tldwProvidersInfo]
  )

  const catalogProvider = selectedBackend || inferredTldwProviderKey

  const {
    data: tldwVoices = [],
    isLoading: tldwVoicesLoading,
    error: tldwVoicesError,
    refetch: refetchTldwVoices
  } = useQuery<TldwVoice[]>({
    queryKey: [
      "fetchTldwVoices",
      catalogProvider,
      selectedBackend ? form.values.tldwTtsModel : undefined
    ],
    queryFn: async () => {
      if (catalogProvider) {
        const catalog = await fetchTldwVoiceCatalog(
          selectedBackend
            ? selectedBackend
            : toServerTtsProviderKey(catalogProvider),
          selectedBackend
            ? { model: form.values.tldwTtsModel }
            : undefined
        )
        if (selectedBackend || catalog.length > 0) return catalog
      }
      return fetchTldwVoices({ throwOnError: true })
    },
    enabled: form.values.ttsProvider === "tldw",
    retry: false
  })

  const { data: customVoices = [] } = useQuery<TldwCustomVoice[]>({
    queryKey: ["tts-custom-voices"],
    queryFn: listCustomVoices,
    enabled: form.values.ttsProvider === "tldw" && !selectedBackend
  })

  const activeProviderCaps = React.useMemo((): TldwTtsProviderCapabilities | null => {
    if (selectedBackendCaps) return selectedBackendCaps
    if (!tldwProvidersInfo || !inferredTldwProviderKey) return null
    const providers = tldwProvidersInfo.providers || {}
    const matchKey = Object.keys(providers).find(
      (key) =>
        toServerTtsProviderKey(key) === toServerTtsProviderKey(inferredTldwProviderKey)
    )
    return matchKey ? providers[matchKey] : null
  }, [inferredTldwProviderKey, selectedBackendCaps, tldwProvidersInfo])

  const tldwFormatOptions = React.useMemo(() => {
    const formats = selectedModelCaps?.formats?.length
      ? selectedModelCaps.formats
      : activeProviderCaps?.formats?.length
        ? activeProviderCaps.formats
        : SUPPORTED_TLDW_TTS_FORMATS
    const unique = Array.from(
      new Set(formats.map((f) => String(f).toLowerCase()))
    )
    return unique.map((fmt) => ({
      label: fmt === "pcm" ? "pcm (raw)" : fmt,
      value: fmt
    }))
  }, [activeProviderCaps, selectedModelCaps])

  const tldwLanguageOptions = React.useMemo(() => {
    const languages = activeProviderCaps?.languages || []
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

  const {
    data: tldwModels,
    error: tldwModelsError,
    refetch: refetchTldwModels
  } = useQuery<TldwTtsModel[]>({
    queryKey: ["fetchTldwTtsModels", selectedBackend || undefined],
    queryFn: () => fetchTldwTtsModels(selectedBackend || undefined),
    enabled: form.values.ttsProvider === "tldw",
    retry: false
  })

  const displayedTldwModels = selectedBackend
    ? (selectedBackendCaps?.models || [])
        .filter(
          (model): model is string =>
            typeof model === "string" && Boolean(model)
        )
        .map((model) => ({ id: model, label: model }))
    : tldwModels

  const providerVoices = React.useMemo((): TldwTtsVoiceInfo[] => {
    if (selectedBackend) {
      return selectedModelVoices.map((voice) => ({ id: voice, name: voice }))
    }
    if (!tldwProvidersInfo || !inferredTldwProviderKey) return []
    const allVoices = tldwProvidersInfo.voices || {}
    const direct = allVoices[inferredTldwProviderKey]
    if (Array.isArray(direct) && direct.length > 0) {
      return direct
    }
    const fallbackKey = inferredTldwProviderKey.toLowerCase()
    const fallback = allVoices[fallbackKey]
    if (Array.isArray(fallback) && fallback.length > 0) {
      return fallback
    }
    return []
  }, [
    inferredTldwProviderKey,
    selectedBackend,
    selectedModelVoices,
    tldwProvidersInfo
  ])

  const tldwVoiceOptions = React.useMemo(() => {
    const options: { label: string; value: string }[] = []
    const seen = new Set<string>()
    const customIds = new Set(
      customVoices
        .map((voice) => voice.voice_id)
        .filter((id): id is string => Boolean(id))
    )

    const pushOption = (value: string, label: string) => {
      if (!value || seen.has(value)) return
      seen.add(value)
      options.push({ label, value })
    }

    const normalizedProvider = catalogProvider
      ? normalizeTtsProviderKey(catalogProvider)
      : ""

    const filteredCustomVoices = selectedBackend
      ? []
      : normalizedProvider
      ? customVoices.filter((voice) => {
          const voiceProvider = normalizeTtsProviderKey(voice.provider)
          return !voiceProvider || voiceProvider === normalizedProvider
        })
      : customVoices

    filteredCustomVoices.forEach((voice) => {
      const id = voice.voice_id || voice.name
      if (!id) return
      pushOption(`custom:${id}`, `Custom: ${voice.name || id}`)
    })

    const providerKey = catalogProvider?.toLowerCase() || ""
    const scopedVoices = providerKey
      ? tldwVoices.filter((voice) => {
          const voiceProvider = String(voice.provider || "").toLowerCase()
          return !voiceProvider || voiceProvider === providerKey
        })
      : tldwVoices

    const pushVoice = (voice: TldwVoice | TldwTtsVoiceInfo, index: number) => {
      const value = voice.voice_id || voice.id || voice.name
      if (!value) return
      if (customIds.has(value)) return
      const label =
        voice.name ||
        voice.voice_id ||
        voice.id ||
        `Voice ${index + 1}`
      pushOption(value, label)
    }

    providerVoices.forEach(pushVoice)
    scopedVoices.forEach(pushVoice)

    if (requiresFreeformVoice) {
      return []
    }

    if (options.length > 0) {
      return options
    }

    const fallback = (form.values.tldwTtsVoice || "").trim()
    return fallback ? [{ label: fallback, value: fallback }] : []
  }, [
    customVoices,
    catalogProvider,
    form.values.tldwTtsVoice,
    providerVoices,
    requiresFreeformVoice,
    selectedBackend,
    tldwVoices
  ])

  React.useEffect(() => {
    if (form.values.ttsProvider !== "tldw") return
    if (requiresFreeformVoice) return
    if (tldwVoiceOptions.length === 0) return
    const current = String(form.values.tldwTtsVoice || "").trim()
    const values = tldwVoiceOptions.map((option) => option.value)
    if (current && values.includes(current)) return
    setFormFieldValue("tldwTtsVoice", values[0])
  }, [
    form.values.ttsProvider,
    form.values.tldwTtsVoice,
    requiresFreeformVoice,
    setFormFieldValue,
    tldwVoiceOptions
  ])

  const getBackendDefaults = React.useCallback(
    (backend: string) => {
      if (!backend) {
        return {
          tldwTtsBackend: "",
          tldwTtsModel: DEFAULT_TLDW_TTS_MODEL,
          tldwTtsVoice: ""
        }
      }
      const caps = tldwProvidersInfo?.providers?.[backend]
      const models = Array.isArray(caps?.models)
        ? caps.models.filter(
            (model): model is string =>
              typeof model === "string" && Boolean(model)
          )
        : []
      const model =
        typeof caps?.default_model === "string" &&
        models.includes(caps.default_model)
          ? caps.default_model
          : models[0] || ""
      const modelCaps = caps?.model_capabilities?.[model]
      const voice =
        (typeof modelCaps?.default_voice === "string"
          ? modelCaps.default_voice
          : "") ||
        modelCaps?.voices?.find(
          (candidate): candidate is string =>
            typeof candidate === "string" && Boolean(candidate)
        ) ||
        ""
      return {
        tldwTtsBackend: backend,
        tldwTtsModel: model,
        tldwTtsVoice: voice
      }
    },
    [tldwProvidersInfo]
  )

  const handleTldwBackendChange = React.useCallback(
    (backend: string) => {
      setFormValues(getBackendDefaults(backend))
    },
    [getBackendDefaults, setFormValues]
  )

  const handleTldwModelChange = React.useCallback(
    (model: string) => {
      const modelCaps = selectedBackendCaps?.model_capabilities?.[model]
      const voice =
        (typeof modelCaps?.default_voice === "string"
          ? modelCaps.default_voice
          : "") ||
        modelCaps?.voices?.find(
          (candidate): candidate is string =>
            typeof candidate === "string" && Boolean(candidate)
        ) ||
        ""
      setFormValues({
        tldwTtsModel: model,
        tldwTtsVoice: voice
      })
    },
    [selectedBackendCaps, setFormValues]
  )

  React.useEffect(() => {
    if (
      form.values.ttsProvider !== "tldw" ||
      !explicitBackendSupported ||
      !selectedBackend
    ) {
      return
    }
    const caps = tldwProvidersInfo?.providers?.[selectedBackend]
    const models = Array.isArray(caps?.models)
      ? caps.models.filter(
          (model): model is string =>
            typeof model === "string" && Boolean(model)
        )
      : []
    if (!caps || !models.includes(form.values.tldwTtsModel)) {
      setFormValues(getBackendDefaults(caps ? selectedBackend : ""))
      return
    }
    const modelCaps = caps.model_capabilities?.[form.values.tldwTtsModel]
    const allowedVoices = [
      typeof modelCaps?.default_voice === "string"
        ? modelCaps.default_voice
        : "",
      ...(modelCaps?.voices || [])
    ].filter(
      (voice): voice is string => typeof voice === "string" && Boolean(voice)
    )
    if (
      modelCaps?.requires_freeform_voice !== true &&
      allowedVoices.length > 0 &&
      !allowedVoices.includes(form.values.tldwTtsVoice)
    ) {
      setFormFieldValue("tldwTtsVoice", allowedVoices[0])
    }
  }, [
    explicitBackendSupported,
    form.values.tldwTtsModel,
    form.values.tldwTtsVoice,
    form.values.ttsProvider,
    getBackendDefaults,
    selectedBackend,
    setFormFieldValue,
    setFormValues,
    tldwProvidersInfo
  ])

  const fallbackTargets = (selectedBackendCaps?.fallback?.targets || []).filter(
    isCanonicalBackendId
  )

  const tldwCatalogError = React.useMemo(() => {
    const issue = tldwVoicesError || tldwProvidersError || tldwModelsError
    if (!issue) return null
    return isTimeoutLikeError(issue)
      ? (t(
          "generalSettings.tts.tldwCatalogTimeout",
          "tldw voice and model catalog took longer than 10 seconds. Retry to continue."
        ) as string)
      : (t(
          "generalSettings.tts.tldwCatalogError",
          "Unable to load tldw voices/models. Retry or verify server audio configuration."
        ) as string)
  }, [t, tldwModelsError, tldwProvidersError, tldwVoicesError])

  const retryTldwCatalog = React.useCallback(() => {
    void refetchTldwProviders()
    void refetchTldwVoices()
    void refetchTldwModels()
  }, [refetchTldwModels, refetchTldwProviders, refetchTldwVoices])

  // Save mutation with loading state
  const { mutate: saveTTSMutation, isPending: isSaving } = useMutation({
    mutationFn: async (values: typeof form.values) => {
      const normalizedValues = {
        ...values,
        elevenLabsApiKey: String(values.elevenLabsApiKey || "").trim(),
        openAITTSApiKey: String(values.openAITTSApiKey || "").trim()
      }
      await setTTSSettings(normalizedValues)
      setTTSEnabled(values.ttsEnabled)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["fetchTTSSettings"] })
    },
    onError: (error: unknown) => {
      // Surface a user-visible error and log for diagnostics
      // eslint-disable-next-line no-console
      console.error("Failed to save TTS settings:", error)
      const errorMessage =
        error instanceof Error
          ? error.message
          : (t(
              "generalSettings.tts.saveError",
              "Failed to save TTS settings. Please try again."
            ) as string)
      message.error(
        errorMessage
      )
    }
  })

  // Test ElevenLabs API key
  const testElevenLabsApiKey = async () => {
    const elevenLabsApiKey = String(form.values.elevenLabsApiKey || "").trim()
    if (!elevenLabsApiKey) {
      setElevenLabsTestResult({ ok: false, message: t("generalSettings.tts.apiKeyTest.enterKey", "Please enter an API key first") })
      return
    }
    setTestingElevenLabs(true)
    setElevenLabsTestResult(null)
    try {
      const [voices, models] = await Promise.all([
        getVoices(elevenLabsApiKey),
        getModels(elevenLabsApiKey)
      ])
      const hasVoices = Array.isArray(voices) && voices.length > 0
      const hasModels = Array.isArray(models) && models.length > 0

      if (hasVoices && hasModels) {
        const successMessage = t(
          "generalSettings.tts.apiKeyTest.success",
          "API key valid! Found {{voiceCount}} voices and {{modelCount}} models.",
          { voiceCount: voices.length, modelCount: models.length }
        )
        await persistElevenLabsValidation(true, elevenLabsApiKey, { notifyOnError: true })
        message.success(successMessage as string)
        setElevenLabsTestResult({
          ok: true,
          message: successMessage as string
        })
      } else {
        const noResourcesMessage = t(
          "generalSettings.tts.apiKeyTest.noVoices",
          "API key accepted but no voices or models found"
        )
        await persistElevenLabsValidation(false, elevenLabsApiKey, { notifyOnError: true })
        setElevenLabsTestResult({
          ok: false,
          message: noResourcesMessage as string
        })
      }
    } catch (e: unknown) {
      // eslint-disable-next-line no-console
      console.error("Failed to test ElevenLabs API key:", e)
      const baseMessage = t(
        "generalSettings.tts.apiKeyTest.failed",
        "Invalid API key or connection error"
      ) as string
      const errorDetail =
        e instanceof Error
          ? e.message
          : typeof e === "string"
            ? e
            : JSON.stringify(e)
      const failureMessage = `${baseMessage} (${errorDetail})`
      await persistElevenLabsValidation(false, elevenLabsApiKey, { notifyOnError: true })
      message.error(failureMessage)
      setElevenLabsTestResult({
        ok: false,
        message: failureMessage
      })
    } finally {
      setTestingElevenLabs(false)
    }
  }

  const elevenLabsValidationBadge = React.useMemo(() => {
    if (form.values.elevenLabsKeyValid === true) {
      return {
        color: "success",
        icon: <CheckCircleOutlined aria-label="valid" />,
        label: t("generalSettings.tts.apiKeyStatus.valid", "Valid") as string
      }
    }
    if (form.values.elevenLabsKeyValid === false) {
      return {
        color: "error",
        icon: <ExclamationCircleOutlined aria-label="failed" />,
        label: t("generalSettings.tts.apiKeyStatus.failed", "Failed") as string
      }
    }
    return {
      color: "default",
      icon: <QuestionCircleOutlined aria-label="untested" />,
      label: t("generalSettings.tts.apiKeyStatus.untested", "Untested") as string
    }
  }, [form.values.elevenLabsKeyValid, t])
  const elevenLabsLastTested = formatValidationTimestamp(
    form.values.elevenLabsKeyTestedAt
  )

  const openAIValidationBadge = React.useMemo(() => {
    if (form.values.openAITTSKeyValid === true) {
      return {
        color: "success",
        icon: <CheckCircleOutlined aria-label="valid" />,
        label: t("generalSettings.tts.apiKeyStatus.valid", "Valid") as string
      }
    }
    if (form.values.openAITTSKeyValid === false) {
      return {
        color: "error",
        icon: <ExclamationCircleOutlined aria-label="failed" />,
        label: t("generalSettings.tts.apiKeyStatus.failed", "Failed") as string
      }
    }
    return {
      color: "default",
      icon: <QuestionCircleOutlined aria-label="untested" />,
      label: t("generalSettings.tts.apiKeyStatus.untested", "Untested") as string
    }
  }, [form.values.openAITTSKeyValid, t])
  const openAILastTested = formatValidationTimestamp(
    form.values.openAITTSKeyTestedAt
  )

  const handleElevenLabsKeyChange = React.useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      form.setValues({
        elevenLabsApiKey: event.target.value.trim(),
        elevenLabsKeyValid: null,
        elevenLabsKeyTestedAt: ""
      })
      setElevenLabsTestResult(null)
    },
    [form]
  )

  const handleOpenAIKeyChange = React.useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      form.setValues({
        openAITTSApiKey: event.target.value.trim(),
        openAITTSKeyValid: null,
        openAITTSKeyTestedAt: ""
      })
    },
    [form]
  )

  const stopPreview = React.useCallback(() => {
    previewAbortRef.current?.abort()
    previewAbortRef.current = null
    previewAudioRef.current?.pause()
    previewAudioRef.current = null
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current)
      previewUrlRef.current = null
    }
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      try {
        window.speechSynthesis.cancel()
      } catch {
        // Ignore browser speech cleanup failures.
      }
    }
    setPreviewState("idle")
  }, [])

  React.useEffect(() => stopPreview, [stopPreview, form.values.ttsProvider])

  const handlePreview = React.useCallback(async () => {
    stopPreview()
    setPreviewError("")

    const provider = String(form.values.ttsProvider || "").trim().toLowerCase()
    const requireField = (value: unknown, messageText: string) => {
      if (String(value || "").trim()) return false
      setPreviewError(messageText)
      return true
    }

    if (provider === "browser") {
      if (typeof window === "undefined" || !("speechSynthesis" in window)) {
        setPreviewError(
          t(
            "generalSettings.tts.preview.browserUnavailable",
            "Voice preview is unavailable in this browser."
          ) as string
        )
        return
      }
      const utterance = new SpeechSynthesisUtterance(TTS_PREVIEW_TEXT)
      const voiceName = String(form.values.voice || "").trim()
      const voice = window.speechSynthesis
        .getVoices()
        .find((candidate) => candidate.name === voiceName)
      if (voice) utterance.voice = voice
      const rate = Number(form.values.playbackSpeed)
      utterance.rate = Number.isFinite(rate) && rate > 0 ? rate : 1
      utterance.onend = () => setPreviewState("idle")
      utterance.onerror = () => setPreviewState("idle")
      setPreviewState("playing")
      window.speechSynthesis.speak(utterance)
      return
    }

    const requiredPreviewFieldsByProvider: Record<
      string,
      Array<{ value: unknown; message: string }>
    > = {
      tldw: [
        {
          value: form.values.tldwTtsModel,
          message: t(
            "generalSettings.tts.preview.selectTldwModel",
            "Select a tldw TTS model first."
          ) as string
        },
        {
          value: form.values.tldwTtsVoice,
          message: t(
            "generalSettings.tts.preview.selectTldwVoice",
            "Select a tldw TTS voice first."
          ) as string
        }
      ],
      openai: [
        {
          value: form.values.openAITTSModel,
          message: t(
            "generalSettings.tts.preview.selectOpenAiModel",
            "Select an OpenAI-compatible TTS model first."
          ) as string
        },
        {
          value: form.values.openAITTSVoice,
          message: t(
            "generalSettings.tts.preview.selectOpenAiVoice",
            "Select an OpenAI-compatible TTS voice first."
          ) as string
        }
      ],
      elevenlabs: [
        {
          value: form.values.elevenLabsApiKey,
          message: t(
            "generalSettings.tts.preview.enterElevenLabsApiKey",
            "Enter an ElevenLabs API key first."
          ) as string
        },
        {
          value: form.values.elevenLabsModel,
          message: t(
            "generalSettings.tts.preview.selectElevenLabsModel",
            "Select an ElevenLabs model first."
          ) as string
        },
        {
          value: form.values.elevenLabsVoiceId,
          message: t(
            "generalSettings.tts.preview.selectElevenLabsVoice",
            "Select an ElevenLabs voice first."
          ) as string
        }
      ]
    }
    const requiredPreviewFields = requiredPreviewFieldsByProvider[provider] ?? []
    if (requiredPreviewFields.some((field) => requireField(field.value, field.message))) {
      return
    }

    const controller = new AbortController()
    previewAbortRef.current = controller
    setPreviewState("loading")

    try {
      let buffer: ArrayBuffer
      let mimeType = "audio/mpeg"

      if (provider === "elevenlabs") {
        buffer = await generateSpeech(
          String(form.values.elevenLabsApiKey || "").trim(),
          TTS_PREVIEW_TEXT,
          String(form.values.elevenLabsVoiceId || "").trim(),
          String(form.values.elevenLabsModel || "").trim(),
          undefined,
          { signal: controller.signal }
        )
      } else if (provider === "openai") {
        buffer = await generateOpenAITTS({
          text: TTS_PREVIEW_TEXT,
          model: String(form.values.openAITTSModel || "").trim(),
          voice: String(form.values.openAITTSVoice || "").trim(),
          signal: controller.signal
        })
      } else {
        const responseFormat = String(form.values.tldwTtsResponseFormat || "mp3").trim() || "mp3"
        mimeType = formatToMimeType(responseFormat)
        buffer = await tldwClient.synthesizeSpeech(TTS_PREVIEW_TEXT, {
          model: String(form.values.tldwTtsModel || "").trim(),
          voice: String(form.values.tldwTtsVoice || "").trim(),
          responseFormat,
          speed: Number(form.values.tldwTtsSpeed) || 1,
          language: String(form.values.tldwTtsLanguage || "").trim() || undefined,
          normalizationOptions: {
            normalize: Boolean(form.values.tldwTtsNormalize),
            unit_normalization: Boolean(form.values.tldwTtsNormalizeUnits),
            url_normalization: Boolean(form.values.tldwTtsNormalizeUrls),
            email_normalization: Boolean(form.values.tldwTtsNormalizeEmails),
            phone_normalization: Boolean(form.values.tldwTtsNormalizePhones),
            optional_pluralization_normalization: Boolean(form.values.tldwTtsNormalizePlurals)
          },
          stream: false,
          signal: controller.signal
        })
      }

      if (controller.signal.aborted) return
      const url = URL.createObjectURL(new Blob([buffer], { type: mimeType }))
      previewUrlRef.current = url
      const audio = new Audio(url)
      const playbackRate = Number(form.values.playbackSpeed)
      audio.playbackRate =
        Number.isFinite(playbackRate) && playbackRate > 0 ? playbackRate : 1
      previewAudioRef.current = audio
      audio.onended = stopPreview
      audio.onerror = stopPreview
      await audio.play()
      if (!controller.signal.aborted) setPreviewState("playing")
    } catch (error) {
      if (!isAbortError(error)) {
        setPreviewError(
          t(
            "generalSettings.tts.preview.failed",
            "Unable to preview this voice right now."
          ) as string
        )
      }
      stopPreview()
    }
  }, [form.values, stopPreview, t])

  if (status === "pending") {
    return <Skeleton active />
  }
  if (status === "error") {
    return (
      <Alert
        type="warning"
        showIcon
        title={t("generalSettings.tts.loadError", "Unable to load TTS settings")}
        description={t("generalSettings.tts.loadErrorDesc", "Check your server connection and try again.")}
        action={<Button size="small" onClick={() => refetch()}>{t("common:retry", "Retry")}</Button>}
      />
    )
  }

  const previewHelpText =
    form.values.ttsProvider === "openai"
      ? (t(
          "generalSettings.tts.preview.openAiHelp",
          "Preview tests this model and voice through the server speech API; server or provider credentials may still be required."
        ) as string)
      : ""

  return (
    <div>
      <div className="mb-5">
        <h2
          className={`${
            !hideBorder ? "text-base font-semibold leading-7" : "text-md"
          } text-text`}>
          {t("generalSettings.tts.heading")}
        </h2>
        {!hideBorder && (
          <div className="border-b border-border mt-3"></div>
        )}
      </div>
      <form
        onSubmit={form.onSubmit((values) => {
          saveTTSMutation(values)
        })}
        className="space-y-4">
        <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
          <label
            className="text-text "
            htmlFor={ids.ttsEnabled}>
            {t("generalSettings.tts.ttsEnabled.label")}
          </label>
          <div>
            <Switch
              id={ids.ttsEnabled}
              aria-label={t("generalSettings.tts.ttsEnabled.label") as string}
              className="mt-4 sm:mt-0 focus-ring"
              {...form.getInputProps("ttsEnabled", {
                type: "checkbox"
              })}
            />
          </div>
        </div>
        <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
          <label
            className="text-text "
            htmlFor={ids.ttsAutoPlay}>
            {t("generalSettings.tts.ttsAutoPlay.label")}
          </label>
          <div>
            <Switch
              id={ids.ttsAutoPlay}
              aria-label={t("generalSettings.tts.ttsAutoPlay.label") as string}
              className="mt-4 sm:mt-0 focus-ring"
              {...form.getInputProps("ttsAutoPlay", {
                type: "checkbox"
              })}
            />
          </div>
        </div>
        <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
          <label
            className="text-text "
            htmlFor={ids.ttsProvider}>
            {t("generalSettings.tts.ttsProvider.label")}
          </label>
          <div>
            <Select
              id={ids.ttsProvider}
              aria-label={t("generalSettings.tts.ttsProvider.label") as string}
              placeholder={t("generalSettings.tts.ttsProvider.placeholder")}
              className="w-full mt-4 sm:mt-0 sm:w-[200px] focus-ring"
              options={TTS_PROVIDER_OPTIONS.map(({ label, value }) => ({
                label,
                value
              }))}
              {...form.getInputProps("ttsProvider")}
            />
          </div>
        </div>
        {form.values.ttsProvider === "browser" && (
          <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
            <span className="text-text ">
              {t("generalSettings.tts.ttsVoice.label")}
            </span>
            <div>
              <Select
                id={ids.browserVoice}
                aria-label={t("generalSettings.tts.ttsVoice.label") as string}
                placeholder={t("generalSettings.tts.ttsVoice.placeholder")}
                className="w-full mt-4 sm:mt-0 sm:w-[200px] focus-ring"
                showSearch
                optionFilterProp="label"
                options={data?.browserTTSVoices?.map((voice) => ({
                  label: `${voice.voiceName} - ${voice.lang}`.trim(),
                  value: voice.voiceName
                }))}
                {...form.getInputProps("voice")}
              />
            </div>
          </div>
        )}
        {form.values.ttsProvider === "elevenlabs" && (
          <>
            <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
              <div className="flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <span className="text-text">
                    {t("generalSettings.tts.elevenLabs.apiKey", "API Key")}
                  </span>
                  <Tag
                    color={elevenLabsValidationBadge.color}
                    icon={elevenLabsValidationBadge.icon}
                  >
                    {elevenLabsValidationBadge.label}
                  </Tag>
                </div>
                {elevenLabsLastTested && (
                  <span className="text-xs text-text-subtle">
                    {t(
                      "generalSettings.tts.apiKeyStatus.lastTested",
                      "Last tested {{timestamp}}",
                      { timestamp: elevenLabsLastTested }
                    )}
                  </span>
                )}
              </div>
              <Space.Compact className="mt-4 sm:mt-0">
                <Input.Password
                  id={ids.elevenApiKey}
                  placeholder="sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                  className="!w-[220px]"
                  required
                  value={form.values.elevenLabsApiKey}
                  onChange={handleElevenLabsKeyChange}
                  onFocus={() => setElevenLabsTestResult(null)}
                />
                <Button
                  type="default"
                  aria-label={t("generalSettings.tts.apiKeyTest.test", "Test")}
                  onClick={testElevenLabsApiKey}
                  loading={testingElevenLabs}
                >
                  {t("generalSettings.tts.apiKeyTest.test", "Test")}
                </Button>
              </Space.Compact>
            </div>
            {elevenLabsTestResult && (
              <Alert
                type={elevenLabsTestResult.ok ? "success" : "error"}
                title={elevenLabsTestResult.message}
                showIcon
                closable
                onClose={() => setElevenLabsTestResult(null)}
                className="mt-2"
              />
            )}

            {elevenLabsError && (
              <Alert
                type="error"
                title={t("generalSettings.tts.elevenLabs.fetchError", "Failed to fetch voices and models")}
                description={t("generalSettings.tts.elevenLabs.fetchErrorHelp", "Check your API key and internet connection, then try again.")}
                showIcon
                className="mt-2"
              />
            )}

            {elevenLabsData && (
              <>
                <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
                  <span className="text-text">
                    {t("generalSettings.tts.elevenLabs.voice", "TTS Voice")}
                  </span>
                  <Select
                    id={ids.elevenVoice}
                    aria-label="ElevenLabs voice"
                    options={elevenLabsData.voices.map((v) => ({
                      label: v.name,
                      value: v.voice_id
                    }))}
                    className="w-full mt-4 sm:mt-0 sm:w-[200px] focus-ring"
                    placeholder="Select a voice"
                    showSearch
                    optionFilterProp="label"
                    {...form.getInputProps("elevenLabsVoiceId")}
                  />
                </div>

                <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
                  <span className="text-text">
                    {t("generalSettings.tts.elevenLabs.model", "TTS Model")}
                  </span>
                  <Select
                    id={ids.elevenModel}
                    aria-label="ElevenLabs model"
                    className="w-full mt-4 sm:mt-0 sm:w-[200px] focus-ring"
                    placeholder="Select a model"
                    options={elevenLabsData.models.map((m) => ({
                      label: m.name,
                      value: m.model_id
                    }))}
                    {...form.getInputProps("elevenLabsModel")}
                  />
                </div>
              </>
            )}
          </>
        )}
        {form.values.ttsProvider === "openai" && (
          <>
            <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
              <span className="text-text">
                Base URL
              </span>
              <Input
                placeholder="http://localhost:5000/v1"
                className=" mt-4 sm:mt-0 !w-[300px] sm:w-[200px]"
                required
                {...form.getInputProps("openAITTSBaseUrl")}
              />
            </div>

            <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
              <div className="flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <span className="text-text">
                    API Key
                  </span>
                  <Tag
                    color={openAIValidationBadge.color}
                    icon={openAIValidationBadge.icon}
                  >
                    {openAIValidationBadge.label}
                  </Tag>
                </div>
                {openAILastTested && (
                  <span className="text-xs text-text-subtle">
                    {t(
                      "generalSettings.tts.apiKeyStatus.lastTested",
                      "Last tested {{timestamp}}",
                      { timestamp: openAILastTested }
                    )}
                  </span>
                )}
              </div>
              <Input.Password
                placeholder="sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                className=" mt-4 sm:mt-0 !w-[300px] sm:w-[200px]"
                value={form.values.openAITTSApiKey}
                onChange={handleOpenAIKeyChange}
              />
            </div>

            <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
              <span className="text-text">
                TTS Voice
              </span>
              <Select
                id={ids.openAiVoice}
                aria-label="OpenAI TTS voice"
                className=" mt-4 sm:mt-0 !w-[300px] sm:w-[200px] focus-ring"
                placeholder="Select a voice"
                options={[
                  { label: "alloy", value: "alloy" },
                  { label: "echo", value: "echo" },
                  { label: "fable", value: "fable" },
                  { label: "onyx", value: "onyx" },
                  { label: "nova", value: "nova" },
                  { label: "shimmer", value: "shimmer" }
                ]}
                {...form.getInputProps("openAITTSVoice")}
              />
            </div>

            <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
              <span className="text-text">
                TTS Model
              </span>
              <Select
                id={ids.openAiModel}
                aria-label="OpenAI TTS model"
                className=" mt-4 sm:mt-0 !w-[300px] sm:w-[200px] focus-ring"
                placeholder="Select a model"
                options={[
                  { label: "tts-1", value: "tts-1" },
                  { label: "tts-1-hd", value: "tts-1-hd" }
                ]}
                {...form.getInputProps("openAITTSModel")}
              />
            </div>
          </>
        )}
        {form.values.ttsProvider === "tldw" && (
          <>
            {tldwCatalogError && (
              <Alert
                type="warning"
                showIcon
                title={tldwCatalogError}
                action={
                  <Button
                    size="small"
                    onClick={retryTldwCatalog}
                    disabled={tldwVoicesLoading || tldwProvidersLoading}
                  >
                    {t("common:retry", "Retry")}
                  </Button>
                }
              />
            )}
            {explicitBackendSupported && (
              <div className="flex sm:flex-row flex-col gap-2 sm:justify-between">
                <div className="flex flex-col gap-0.5">
                  <label className="text-text" htmlFor={ids.tldwBackend}>
                    TTS Backend
                  </label>
                  <span className="text-xs text-text-subtle">
                    Choose a configured gateway or keep legacy model inference.
                  </span>
                </div>
                <div className="flex flex-col gap-1 sm:items-end">
                  <Select
                    id={ids.tldwBackend}
                    aria-label="tldw TTS backend"
                    className="w-full sm:w-[300px] focus-ring"
                    options={tldwBackendOptions}
                    value={selectedBackend}
                    onChange={handleTldwBackendChange}
                  />
                  {fallbackTargets.length > 0 && (
                    <span className="text-xs text-text-subtle sm:max-w-[300px] sm:text-right">
                      Possible fallback targets: {fallbackTargets.join(", ")}
                    </span>
                  )}
                </div>
              </div>
            )}
            <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
              <label className="text-text" htmlFor={ids.tldwModel}>
                TTS Model
              </label>
              {displayedTldwModels && displayedTldwModels.length > 0 ? (
                <Select
                  id={ids.tldwModel}
                  aria-label="tldw TTS model"
                  className=" mt-4 sm:mt-0 !w-[300px] sm:w-[200px] focus-ring"
                  placeholder="Select a model"
                  options={displayedTldwModels.map((m: TldwTtsModel) => ({
                    label: m.label,
                    value: m.id
                  }))}
                  value={form.values.tldwTtsModel}
                  onChange={handleTldwModelChange}
                />
              ) : (
                <Input
                  id={ids.tldwModel}
                  aria-label="tldw TTS model"
                  placeholder={DEFAULT_TLDW_TTS_MODEL}
                  className=" mt-4 sm:mt-0 !w-[300px] sm:w-[200px]"
                  value={form.values.tldwTtsModel}
                  onChange={(event) => handleTldwModelChange(event.target.value)}
                />
              )}
            </div>
            <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
              <div className="flex flex-col gap-0.5">
                <label className="text-text" htmlFor={ids.tldwVoice}>
                  TTS Voice
                </label>
                {!selectedBackend && (
                  <a href="https://github.com/rmusser01/tldw_server/tree/main/Docs/STT-TTS/" target="_blank" rel="noopener noreferrer" className="text-xs text-text-muted underline">Voice cloning setup guide</a>
                )}
              </div>
              {requiresFreeformVoice ? (
                <Input
                  id={ids.tldwVoice}
                  aria-label="tldw TTS voice"
                  className="w-full mt-4 sm:mt-0 sm:w-[300px]"
                  placeholder="Enter the required voice ID"
                  required
                  {...form.getInputProps("tldwTtsVoice")}
                />
              ) : (
                <Select
                  id={ids.tldwVoice}
                  aria-label="tldw TTS voice"
                  className="w-full mt-4 sm:mt-0 sm:w-[200px] focus-ring"
                  placeholder="Select a voice"
                  options={tldwVoiceOptions}
                  loading={tldwVoicesLoading || tldwProvidersLoading}
                  disabled={
                    !tldwVoicesLoading &&
                    !tldwProvidersLoading &&
                    tldwVoiceOptions.length === 0
                  }
                  showSearch
                  optionFilterProp="label"
                  {...form.getInputProps("tldwTtsVoice")}
                />
              )}
            </div>
            <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
              <span className="text-text">
                Response format
              </span>
              <Select
                id={ids.tldwResponseFormat}
                aria-label="tldw response format"
                className="w-full mt-4 sm:mt-0 sm:w-[200px] focus-ring"
                options={tldwFormatOptions}
                {...form.getInputProps("tldwTtsResponseFormat")}
              />
            </div>
            <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
              <span className="text-text">
                Synthesis speed
              </span>
              <InputNumber
                id={ids.tldwSpeed}
                aria-label="tldw synthesis speed"
                placeholder="1"
                min={0.25}
                max={4}
                step={0.05}
                className=" mt-4 sm:mt-0 !w-[300px] sm:w-[200px]"
                {...form.getInputProps("tldwTtsSpeed")}
              />
            </div>
            {tldwLanguageOptions.length > 0 && (
              <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
                <span className="text-text">
                  Language
                </span>
                <Select
                  id={ids.tldwLanguage}
                  aria-label="tldw language"
                  className="w-full mt-4 sm:mt-0 sm:w-[200px] focus-ring"
                  options={tldwLanguageOptions}
                  allowClear
                  placeholder="Auto"
                  {...form.getInputProps("tldwTtsLanguage")}
                />
              </div>
            )}
            <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
              <div className="flex flex-col gap-0.5">
                <label className="text-text" htmlFor={ids.tldwStreaming}>
                  Stream audio (WebSocket)
                </label>
                <span className="text-xs text-text-subtle">
                  Low-latency playback while audio is generated.
                </span>
              </div>
              <div>
                <Switch
                  id={ids.tldwStreaming}
                  aria-label="Stream audio (WebSocket)"
                  className="mt-4 sm:mt-0 focus-ring"
                  {...form.getInputProps("tldwTtsStreaming", { type: "checkbox" })}
                />
              </div>
            </div>
            {activeProviderCaps?.supports_emotion_control && (
              <>
                <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
                  <span className="text-text">
                    Emotion preset
                  </span>
                  <Select
                    id={ids.tldwEmotion}
                    aria-label="tldw emotion"
                    className="w-full mt-4 sm:mt-0 sm:w-[200px] focus-ring"
                    allowClear
                    placeholder="Default"
                    options={[
                      { label: "Neutral", value: "neutral" },
                      { label: "Calm", value: "calm" },
                      { label: "Energetic", value: "energetic" },
                      { label: "Happy", value: "happy" },
                      { label: "Sad", value: "sad" },
                      { label: "Angry", value: "angry" }
                    ]}
                    {...form.getInputProps("tldwTtsEmotion")}
                  />
                </div>
                <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
                  <span className="text-text">
                    Emotion intensity
                  </span>
                  <InputNumber
                    id={ids.tldwEmotionIntensity}
                    aria-label="tldw emotion intensity"
                    placeholder="1"
                    min={0.1}
                    max={2}
                    step={0.1}
                    className=" mt-4 sm:mt-0 !w-[300px] sm:w-[200px]"
                    {...form.getInputProps("tldwTtsEmotionIntensity")}
                  />
                </div>
              </>
            )}
            <div className="rounded-md border border-border p-3 space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-text" htmlFor={ids.tldwNormalize}>
                  Smart normalization
                </label>
                <Switch
                  id={ids.tldwNormalize}
                  aria-label="Smart normalization"
                  className="focus-ring"
                  {...form.getInputProps("tldwTtsNormalize", { type: "checkbox" })}
                />
              </div>
              <div className="text-xs text-text-subtle">
                Expands units, URLs, emails, and phone numbers to improve pronunciation.
              </div>
              {form.values.tldwTtsNormalize && (
                <div className="grid gap-2 sm:grid-cols-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-subtle">Units</span>
                    <Switch
                      id={ids.tldwNormalizeUnits}
                      size="small"
                      {...form.getInputProps("tldwTtsNormalizeUnits", { type: "checkbox" })}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-subtle">URLs</span>
                    <Switch
                      id={ids.tldwNormalizeUrls}
                      size="small"
                      {...form.getInputProps("tldwTtsNormalizeUrls", { type: "checkbox" })}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-subtle">Emails</span>
                    <Switch
                      id={ids.tldwNormalizeEmails}
                      size="small"
                      {...form.getInputProps("tldwTtsNormalizeEmails", { type: "checkbox" })}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-subtle">Phone</span>
                    <Switch
                      id={ids.tldwNormalizePhones}
                      size="small"
                      {...form.getInputProps("tldwTtsNormalizePhones", { type: "checkbox" })}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-subtle">Pluralization</span>
                    <Switch
                      id={ids.tldwNormalizePlurals}
                      size="small"
                      {...form.getInputProps("tldwTtsNormalizePlurals", { type: "checkbox" })}
                    />
                  </div>
                </div>
              )}
            </div>
          </>
        )}
        <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
          <div className="flex flex-col gap-0.5">
            <span className="text-text">
              {t("generalSettings.tts.preview.label", "Voice preview")}
            </span>
            {previewError && (
              <span className="text-xs text-danger">
                {previewError}
              </span>
            )}
            {previewHelpText && (
              <span className="text-xs text-text-subtle">
                {previewHelpText}
              </span>
            )}
          </div>
          <Button
            data-testid="tts-preview-button"
            type="default"
            className="mt-4 sm:mt-0"
            icon={
              previewState !== "idle" ? (
                <Square className="h-3.5 w-3.5" />
              ) : (
                <Play className="h-3.5 w-3.5" />
              )
            }
            aria-busy={previewState === "loading"}
            onClick={previewState === "idle" ? handlePreview : stopPreview}
          >
            {previewState !== "idle"
              ? t("generalSettings.tts.preview.stop", "Stop preview")
              : t("generalSettings.tts.preview.play", "Preview voice")}
          </Button>
        </div>
        <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
          <span className="text-text ">
            {t("generalSettings.tts.responseSplitting.label")}
          </span>
          <div>
            <Select
              placeholder={t("generalSettings.tts.responseSplitting.placeholder")}
              className="w-full mt-4 sm:mt-0 sm:w-[200px]"
              options={[
                { label: "None", value: "none" },
                { label: "Punctuation", value: "punctuation" },
                { label: "Paragraph", value: "paragraph" }
              ]}
              {...form.getInputProps("responseSplitting")}
            />
          </div>
        </div>
        <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
          <label
            className="text-text "
            htmlFor={ids.ssmlEnabled}>
            {t("generalSettings.tts.ssmlEnabled.label")}
          </label>
          <div>
            <Switch
              id={ids.ssmlEnabled}
              aria-label={t("generalSettings.tts.ssmlEnabled.label") as string}
              className="mt-4 sm:mt-0 focus-ring"
              {...form.getInputProps("ssmlEnabled", {
                type: "checkbox"
              })}
            />
          </div>
        </div>

        <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
          <div className="flex flex-col gap-0.5">
            <label
              className="text-text"
              htmlFor={ids.removeReasoning}>
              {t("generalSettings.tts.removeReasoningTagTTS.label")}
            </label>
            <span className="text-xs text-text-subtle">
              {t("generalSettings.tts.removeReasoningTagTTS.description", "Strip <think>...</think> reasoning blocks before speaking.")}
            </span>
          </div>
          <div>
            <Switch
              id={ids.removeReasoning}
              aria-label={
                t("generalSettings.tts.removeReasoningTagTTS.label") as string
              }
              className="mt-4 sm:mt-0 focus-ring"
              {...form.getInputProps("removeReasoningTagTTS", {
                type: "checkbox"
              })}
            />
          </div>
        </div>

        <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
          <div className="flex flex-col gap-0.5">
            <label
              className="text-text"
              htmlFor={ids.playbackSpeed}>
              {t("generalSettings.tts.playbackSpeed.label", "Playback Speed")}
            </label>
            <span className="text-xs text-text-subtle">
              {t("generalSettings.tts.playbackSpeed.description", "Controls how fast audio plays back (does not affect generation).")}
            </span>
          </div>
          <div className="flex flex-col gap-1">
            <InputNumber
              id={ids.playbackSpeed}
              aria-label="Playback speed"
              placeholder="1"
              min={0.25}
              max={2}
              step={0.05}
              className=" mt-4 sm:mt-0 !w-[300px] sm:w-[200px]"
              {...form.getInputProps("playbackSpeed")}
            />
            <span className="text-xs text-text-subtle sm:text-right">
              {t("generalSettings.tts.playbackSpeed.range", "0.25-2x")}
            </span>
          </div>
        </div>

        <div className="flex justify-end">
          <SaveButton
            btnType="submit"
            disabled={!form.isDirty()}
            loading={isSaving}
            className="disabled:cursor-not-allowed"
            text={form.isDirty() ? "save" : "saved"}
            textOnSave="saved"
          />
        </div>
      </form>
    </div>
  )
}
