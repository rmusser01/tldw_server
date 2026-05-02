import { isChromiumTarget } from "@/config/platform"
import {
  defineSetting,
  getSetting,
  setSetting,
  coerceBoolean,
  coerceBooleanOrNull,
  coerceNumber,
  coerceOptionalString,
  coerceString
} from "@/services/settings/registry"
import {
  TTS_PROVIDER_VALUES,
  type TtsProviderValue
} from "@/services/tts-providers"

export const DEFAULT_TTS_PROVIDER: TtsProviderValue = "tldw"
export const DEFAULT_TLDW_TTS_MODEL = "KittenML/kitten-tts-nano-0.8"
export const DEFAULT_TLDW_TTS_VOICE = "Bella"

export const SUPPORTED_TLDW_TTS_FORMATS = [
  "mp3",
  "opus",
  "aac",
  "flac",
  "wav",
  "pcm",
  "ogg",
  "webm",
  "ulaw"
] as const
type SupportedTldwTtsFormat = (typeof SUPPORTED_TLDW_TTS_FORMATS)[number]
const SUPPORTED_TLDW_TTS_FORMAT_SET = new Set(SUPPORTED_TLDW_TTS_FORMATS)

const normalizeTldwTtsFormatInput = (format?: string | null): string => {
  const normalized = (format || "").trim().toLowerCase()
  if (normalized === "mulaw" || normalized === "mu-law") return "ulaw"
  return normalized
}

export const isSupportedTldwTtsResponseFormat = (
  format?: string | null
): format is SupportedTldwTtsFormat => {
  const normalized = normalizeTldwTtsFormatInput(format)
  return SUPPORTED_TLDW_TTS_FORMAT_SET.has(normalized as SupportedTldwTtsFormat)
}

export const normalizeTldwTtsResponseFormat = (
  format?: string | null
): SupportedTldwTtsFormat => {
  const normalized = normalizeTldwTtsFormatInput(format)
  return SUPPORTED_TLDW_TTS_FORMAT_SET.has(normalized as SupportedTldwTtsFormat)
    ? (normalized as SupportedTldwTtsFormat)
    : "mp3"
}

const coercePositiveNumber = (value: unknown, fallback: number): number => {
  const num = coerceNumber(value, fallback)
  return num > 0 ? num : fallback
}

const TTS_PROVIDER_SETTING = defineSetting(
  "ttsProvider",
  DEFAULT_TTS_PROVIDER,
  (value) => {
    const normalized = String(value || "").toLowerCase()
    return TTS_PROVIDER_VALUES.includes(normalized as TtsProviderValue)
      ? (normalized as TtsProviderValue)
      : DEFAULT_TTS_PROVIDER
  }
)
const VOICE_SETTING = defineSetting("voice", undefined as string | undefined, coerceOptionalString)
const TTS_ENABLED_SETTING = defineSetting("isTTSEnabled", true, (value) =>
  coerceBoolean(value, true)
)
const SSML_ENABLED_SETTING = defineSetting("isSSMLEnabled", false, (value) =>
  coerceBoolean(value, false)
)
const ELEVEN_LABS_API_KEY_SETTING = defineSetting(
  "elevenLabsApiKey",
  undefined as string | undefined,
  coerceOptionalString
)
const ELEVEN_LABS_VOICE_ID_SETTING = defineSetting(
  "elevenLabsVoiceId",
  undefined as string | undefined,
  coerceOptionalString
)
const ELEVEN_LABS_MODEL_SETTING = defineSetting(
  "elevenLabsModel",
  undefined as string | undefined,
  coerceOptionalString
)
const ELEVEN_LABS_KEY_VALID_SETTING = defineSetting(
  "elevenLabsKeyValid",
  null as boolean | null,
  coerceBooleanOrNull
)
const ELEVEN_LABS_KEY_TESTED_AT_SETTING = defineSetting(
  "elevenLabsKeyTestedAt",
  "",
  (value) => coerceOptionalString(value) || ""
)
const OPENAI_TTS_BASE_URL_SETTING = defineSetting(
  "openAITTSBaseUrl",
  "https://api.openai.com/v1",
  (value) => coerceString(value, "https://api.openai.com/v1")
)
const OPENAI_TTS_API_KEY_SETTING = defineSetting(
  "openAITTSApiKey",
  "",
  (value) => coerceString(value, "")
)
const OPENAI_TTS_MODEL_SETTING = defineSetting(
  "openAITTSModel",
  "tts-1",
  (value) => coerceString(value, "tts-1")
)
const OPENAI_TTS_VOICE_SETTING = defineSetting(
  "openAITTSVoice",
  "alloy",
  (value) => coerceString(value, "alloy")
)
const OPENAI_TTS_KEY_VALID_SETTING = defineSetting(
  "openAITTSKeyValid",
  null as boolean | null,
  coerceBooleanOrNull
)
const OPENAI_TTS_KEY_TESTED_AT_SETTING = defineSetting(
  "openAITTSKeyTestedAt",
  "",
  (value) => coerceOptionalString(value) || ""
)
const RESPONSE_SPLITTING_SETTING = defineSetting(
  "ttsResponseSplitting",
  "punctuation",
  (value) => coerceString(value, "punctuation")
)
const REMOVE_REASONING_TAG_SETTING = defineSetting(
  "removeReasoningTagTTS",
  true,
  (value) => coerceBoolean(value, true)
)
const TTS_AUTOPLAY_SETTING = defineSetting("isTTSAutoPlayEnabled", false, (value) =>
  coerceBoolean(value, false)
)
const SPEECH_PLAYBACK_SPEED_SETTING = defineSetting(
  "speechPlaybackSpeed",
  1,
  (value) => coercePositiveNumber(value, 1)
)
const TLDW_TTS_MODEL_SETTING = defineSetting(
  "tldwTtsModel",
  DEFAULT_TLDW_TTS_MODEL,
  (value) => coerceString(value, DEFAULT_TLDW_TTS_MODEL)
)
const TLDW_TTS_VOICE_SETTING = defineSetting(
  "tldwTtsVoice",
  DEFAULT_TLDW_TTS_VOICE,
  (value) => coerceString(value, DEFAULT_TLDW_TTS_VOICE)
)
const TLDW_TTS_RESPONSE_FORMAT_SETTING = defineSetting(
  "tldwTtsResponseFormat",
  "mp3" as SupportedTldwTtsFormat,
  (value) => normalizeTldwTtsResponseFormat(String(value || ""))
)
const TLDW_TTS_SPEED_SETTING = defineSetting(
  "tldwTtsSpeed",
  1,
  (value) => coercePositiveNumber(value, 1)
)
const TLDW_TTS_LANGUAGE_SETTING = defineSetting(
  "tldwTtsLanguage",
  "",
  (value) => coerceOptionalString(value) || ""
)
const TLDW_TTS_STREAMING_SETTING = defineSetting(
  "tldwTtsStreaming",
  false,
  (value) => coerceBoolean(value, false)
)
const TLDW_TTS_EMOTION_SETTING = defineSetting(
  "tldwTtsEmotion",
  "",
  (value) => coerceOptionalString(value) || ""
)
const TLDW_TTS_EMOTION_INTENSITY_SETTING = defineSetting(
  "tldwTtsEmotionIntensity",
  1,
  (value) => coerceNumber(value, 1)
)
const TLDW_TTS_NORMALIZE_SETTING = defineSetting(
  "tldwTtsNormalize",
  true,
  (value) => coerceBoolean(value, true)
)
const TLDW_TTS_NORMALIZE_UNITS_SETTING = defineSetting(
  "tldwTtsNormalizeUnits",
  false,
  (value) => coerceBoolean(value, false)
)
const TLDW_TTS_NORMALIZE_URLS_SETTING = defineSetting(
  "tldwTtsNormalizeUrls",
  true,
  (value) => coerceBoolean(value, true)
)
const TLDW_TTS_NORMALIZE_EMAILS_SETTING = defineSetting(
  "tldwTtsNormalizeEmails",
  true,
  (value) => coerceBoolean(value, true)
)
const TLDW_TTS_NORMALIZE_PHONES_SETTING = defineSetting(
  "tldwTtsNormalizePhones",
  true,
  (value) => coerceBoolean(value, true)
)
const TLDW_TTS_NORMALIZE_PLURALS_SETTING = defineSetting(
  "tldwTtsNormalizePlurals",
  true,
  (value) => coerceBoolean(value, true)
)

export const getTTSProvider = async (): Promise<TtsProviderValue> =>
  getSetting(TTS_PROVIDER_SETTING)

export const setTTSProvider = async (ttsProvider: string) => {
  await setSetting(TTS_PROVIDER_SETTING, ttsProvider as TtsProviderValue)
}

export const getBrowserTTSVoices = async () => {
  try {
    if (isChromiumTarget) {
      const api = (globalThis as any)?.chrome?.tts
      if (!api || typeof api.getVoices !== "function") {
        return []
      }
      const tts = await api.getVoices()
      return Array.isArray(tts) ? tts : []
    }

    const synth = (globalThis as any)?.speechSynthesis
    if (!synth || typeof synth.getVoices !== "function") {
      return []
    }
    const tts = await synth.getVoices()
    if (!Array.isArray(tts)) {
      return []
    }
    return tts.map((voice) => ({
      voiceName: voice.name,
      lang: voice.lang
    }))
  } catch {
    return []
  }
}

export const getVoice = async () => getSetting(VOICE_SETTING)

export const setVoice = async (voice: string) => {
  await setSetting(VOICE_SETTING, voice)
}

export const isTTSEnabled = async () => getSetting(TTS_ENABLED_SETTING)

export const setTTSEnabled = async (isTTSEnabled: boolean) => {
  await setSetting(TTS_ENABLED_SETTING, isTTSEnabled)
}

export const isSSMLEnabled = async () => getSetting(SSML_ENABLED_SETTING)

export const setSSMLEnabled = async (isSSMLEnabled: boolean) => {
  await setSetting(SSML_ENABLED_SETTING, isSSMLEnabled)
}

export const getElevenLabsApiKey = async () =>
  getSetting(ELEVEN_LABS_API_KEY_SETTING)

export const setElevenLabsApiKey = async (elevenLabsApiKey: string) => {
  await setSetting(ELEVEN_LABS_API_KEY_SETTING, elevenLabsApiKey)
}

export const getElevenLabsVoiceId = async () =>
  getSetting(ELEVEN_LABS_VOICE_ID_SETTING)

export const setElevenLabsVoiceId = async (elevenLabsVoiceId: string) => {
  await setSetting(ELEVEN_LABS_VOICE_ID_SETTING, elevenLabsVoiceId)
}

export const getElevenLabsModel = async () =>
  getSetting(ELEVEN_LABS_MODEL_SETTING)

export const setElevenLabsModel = async (elevenLabsModel: string) => {
  await setSetting(ELEVEN_LABS_MODEL_SETTING, elevenLabsModel)
}

export const getElevenLabsKeyValid = async () =>
  getSetting(ELEVEN_LABS_KEY_VALID_SETTING)

export const setElevenLabsKeyValid = async (
  elevenLabsKeyValid: boolean | null
) => {
  await setSetting(ELEVEN_LABS_KEY_VALID_SETTING, elevenLabsKeyValid)
}

export const getElevenLabsKeyTestedAt = async () =>
  getSetting(ELEVEN_LABS_KEY_TESTED_AT_SETTING)

export const setElevenLabsKeyTestedAt = async (
  elevenLabsKeyTestedAt: string
) => {
  await setSetting(ELEVEN_LABS_KEY_TESTED_AT_SETTING, elevenLabsKeyTestedAt)
}

export const getOpenAITTSBaseUrl = async () =>
  getSetting(OPENAI_TTS_BASE_URL_SETTING)

export const setOpenAITTSBaseUrl = async (openAITTSBaseUrl: string) => {
  await setSetting(OPENAI_TTS_BASE_URL_SETTING, openAITTSBaseUrl)
}

export const getOpenAITTSApiKey = async () =>
  getSetting(OPENAI_TTS_API_KEY_SETTING)

export const getOpenAITTSModel = async () =>
  getSetting(OPENAI_TTS_MODEL_SETTING)

export const setOpenAITTSModel = async (openAITTSModel: string) => {
  await setSetting(OPENAI_TTS_MODEL_SETTING, openAITTSModel)
}

export const setOpenAITTSApiKey = async (openAITTSApiKey: string) => {
  await setSetting(OPENAI_TTS_API_KEY_SETTING, openAITTSApiKey)
}

export const getOpenAITTSVoice = async () =>
  getSetting(OPENAI_TTS_VOICE_SETTING)

export const setOpenAITTSVoice = async (openAITTSVoice: string) => {
  await setSetting(OPENAI_TTS_VOICE_SETTING, openAITTSVoice)
}

export const getOpenAITTSKeyValid = async () =>
  getSetting(OPENAI_TTS_KEY_VALID_SETTING)

export const setOpenAITTSKeyValid = async (
  openAITTSKeyValid: boolean | null
) => {
  await setSetting(OPENAI_TTS_KEY_VALID_SETTING, openAITTSKeyValid)
}

export const getOpenAITTSKeyTestedAt = async () =>
  getSetting(OPENAI_TTS_KEY_TESTED_AT_SETTING)

export const setOpenAITTSKeyTestedAt = async (
  openAITTSKeyTestedAt: string
) => {
  await setSetting(OPENAI_TTS_KEY_TESTED_AT_SETTING, openAITTSKeyTestedAt)
}

export const getResponseSplitting = async () =>
  getSetting(RESPONSE_SPLITTING_SETTING)

export const getRemoveReasoningTagTTS = async () =>
  getSetting(REMOVE_REASONING_TAG_SETTING)

export const setResponseSplitting = async (responseSplitting: string) => {
  await setSetting(RESPONSE_SPLITTING_SETTING, responseSplitting)
}

export const setRemoveReasoningTagTTS = async (removeReasoningTagTTS: boolean) => {
  await setSetting(REMOVE_REASONING_TAG_SETTING, removeReasoningTagTTS)
}

export const isTTSAutoPlayEnabled = async () => getSetting(TTS_AUTOPLAY_SETTING)

export const setTTSAutoPlayEnabled = async (isTTSAutoPlayEnabled: boolean) => {
  await setSetting(TTS_AUTOPLAY_SETTING, isTTSAutoPlayEnabled)
}

export const getSpeechPlaybackSpeed = async () =>
  getSetting(SPEECH_PLAYBACK_SPEED_SETTING)

export const setSpeechPlaybackSpeed = async (speechPlaybackSpeed: number) => {
  await setSetting(SPEECH_PLAYBACK_SPEED_SETTING, speechPlaybackSpeed)
}

export const getTldwTTSModel = async () => getSetting(TLDW_TTS_MODEL_SETTING)

export const setTldwTTSModel = async (model: string) => {
  await setSetting(TLDW_TTS_MODEL_SETTING, model)
}

export const getTldwTTSVoice = async () => getSetting(TLDW_TTS_VOICE_SETTING)

export const setTldwTTSVoice = async (voice: string) => {
  await setSetting(TLDW_TTS_VOICE_SETTING, voice)
}

export const getTldwTTSResponseFormat = async () =>
  getSetting(TLDW_TTS_RESPONSE_FORMAT_SETTING)

export const setTldwTTSResponseFormat = async (fmt: string) => {
  await setSetting(
    TLDW_TTS_RESPONSE_FORMAT_SETTING,
    normalizeTldwTtsResponseFormat(fmt)
  )
}

export const getTldwTTSSpeed = async () => getSetting(TLDW_TTS_SPEED_SETTING)

export const setTldwTTSSpeed = async (speed: number) => {
  await setSetting(TLDW_TTS_SPEED_SETTING, speed)
}

export const getTldwTTSLanguage = async () =>
  getSetting(TLDW_TTS_LANGUAGE_SETTING)

export const setTldwTTSLanguage = async (language: string) => {
  await setSetting(TLDW_TTS_LANGUAGE_SETTING, language)
}

export const getTldwTTSStreamingEnabled = async () =>
  getSetting(TLDW_TTS_STREAMING_SETTING)

export const setTldwTTSStreamingEnabled = async (enabled: boolean) => {
  await setSetting(TLDW_TTS_STREAMING_SETTING, enabled)
}

export const getTldwTTSEmotion = async () =>
  getSetting(TLDW_TTS_EMOTION_SETTING)

export const setTldwTTSEmotion = async (emotion: string) => {
  await setSetting(TLDW_TTS_EMOTION_SETTING, emotion)
}

export const getTldwTTSEmotionIntensity = async () =>
  getSetting(TLDW_TTS_EMOTION_INTENSITY_SETTING)

export const setTldwTTSEmotionIntensity = async (value: number) => {
  await setSetting(TLDW_TTS_EMOTION_INTENSITY_SETTING, value)
}

export const getTldwTTSNormalize = async () =>
  getSetting(TLDW_TTS_NORMALIZE_SETTING)

export const setTldwTTSNormalize = async (value: boolean) => {
  await setSetting(TLDW_TTS_NORMALIZE_SETTING, value)
}

export const getTldwTTSNormalizeUnits = async () =>
  getSetting(TLDW_TTS_NORMALIZE_UNITS_SETTING)

export const setTldwTTSNormalizeUnits = async (value: boolean) => {
  await setSetting(TLDW_TTS_NORMALIZE_UNITS_SETTING, value)
}

export const getTldwTTSNormalizeUrls = async () =>
  getSetting(TLDW_TTS_NORMALIZE_URLS_SETTING)

export const setTldwTTSNormalizeUrls = async (value: boolean) => {
  await setSetting(TLDW_TTS_NORMALIZE_URLS_SETTING, value)
}

export const getTldwTTSNormalizeEmails = async () =>
  getSetting(TLDW_TTS_NORMALIZE_EMAILS_SETTING)

export const setTldwTTSNormalizeEmails = async (value: boolean) => {
  await setSetting(TLDW_TTS_NORMALIZE_EMAILS_SETTING, value)
}

export const getTldwTTSNormalizePhones = async () =>
  getSetting(TLDW_TTS_NORMALIZE_PHONES_SETTING)

export const setTldwTTSNormalizePhones = async (value: boolean) => {
  await setSetting(TLDW_TTS_NORMALIZE_PHONES_SETTING, value)
}

export const getTldwTTSNormalizePlurals = async () =>
  getSetting(TLDW_TTS_NORMALIZE_PLURALS_SETTING)

export const setTldwTTSNormalizePlurals = async (value: boolean) => {
  await setSetting(TLDW_TTS_NORMALIZE_PLURALS_SETTING, value)
}

export const getTTSSettings = async () => {
  const [
    ttsEnabled,
    ttsProvider,
    browserTTSVoices,
    voice,
    ssmlEnabled,
    elevenLabsApiKey,
    elevenLabsVoiceId,
    elevenLabsModel,
    elevenLabsKeyValid,
    elevenLabsKeyTestedAt,
    responseSplitting,
    removeReasoningTagTTS,
    // OPENAI
    openAITTSBaseUrl,
    openAITTSApiKey,
    openAITTSModel,
    openAITTSVoice,
    openAITTSKeyValid,
    openAITTSKeyTestedAt,
    // UTILS
    ttsAutoPlay,
    playbackSpeed,
    // tldw_server TTS
    tldwTtsModel,
    tldwTtsVoice,
    tldwTtsResponseFormat,
    tldwTtsSpeed,
    tldwTtsLanguage,
    tldwTtsStreaming,
    tldwTtsEmotion,
    tldwTtsEmotionIntensity,
    tldwTtsNormalize,
    tldwTtsNormalizeUnits,
    tldwTtsNormalizeUrls,
    tldwTtsNormalizeEmails,
    tldwTtsNormalizePhones,
    tldwTtsNormalizePlurals
  ] = await Promise.all([
    isTTSEnabled(),
    getTTSProvider(),
    getBrowserTTSVoices(),
    getVoice(),
    isSSMLEnabled(),
    getElevenLabsApiKey(),
    getElevenLabsVoiceId(),
    getElevenLabsModel(),
    getElevenLabsKeyValid(),
    getElevenLabsKeyTestedAt(),
    getResponseSplitting(),
    getRemoveReasoningTagTTS(),
    // OPENAI
    getOpenAITTSBaseUrl(),
    getOpenAITTSApiKey(),
    getOpenAITTSModel(),
    getOpenAITTSVoice(),
    getOpenAITTSKeyValid(),
    getOpenAITTSKeyTestedAt(),
    // UTILS
    isTTSAutoPlayEnabled(),
    getSpeechPlaybackSpeed(),
    // tldw_server TTS
    getTldwTTSModel(),
    getTldwTTSVoice(),
    getTldwTTSResponseFormat(),
    getTldwTTSSpeed(),
    getTldwTTSLanguage(),
    getTldwTTSStreamingEnabled(),
    getTldwTTSEmotion(),
    getTldwTTSEmotionIntensity(),
    getTldwTTSNormalize(),
    getTldwTTSNormalizeUnits(),
    getTldwTTSNormalizeUrls(),
    getTldwTTSNormalizeEmails(),
    getTldwTTSNormalizePhones(),
    getTldwTTSNormalizePlurals()
  ])

  return {
    ttsEnabled,
    ttsProvider,
    browserTTSVoices,
    voice,
    ssmlEnabled,
    elevenLabsApiKey,
    elevenLabsVoiceId,
    elevenLabsModel,
    elevenLabsKeyValid,
    elevenLabsKeyTestedAt,
    responseSplitting,
    removeReasoningTagTTS,
    // OPENAI
    openAITTSBaseUrl,
    openAITTSApiKey,
    openAITTSModel,
    openAITTSVoice,
    openAITTSKeyValid,
    openAITTSKeyTestedAt,
    ttsAutoPlay,
    playbackSpeed,
    tldwTtsModel,
    tldwTtsVoice,
    tldwTtsResponseFormat,
    tldwTtsSpeed,
    tldwTtsLanguage,
    tldwTtsStreaming,
    tldwTtsEmotion,
    tldwTtsEmotionIntensity,
    tldwTtsNormalize,
    tldwTtsNormalizeUnits,
    tldwTtsNormalizeUrls,
    tldwTtsNormalizeEmails,
    tldwTtsNormalizePhones,
    tldwTtsNormalizePlurals
  }
}

export const setTTSSettings = async ({
  ttsEnabled,
  ttsProvider,
  voice,
  ssmlEnabled,
  elevenLabsApiKey,
  elevenLabsVoiceId,
  elevenLabsModel,
  elevenLabsKeyValid,
  elevenLabsKeyTestedAt,
  responseSplitting,
  removeReasoningTagTTS,
  openAITTSBaseUrl,
  openAITTSApiKey,
  openAITTSModel,
  openAITTSVoice,
  openAITTSKeyValid,
  openAITTSKeyTestedAt,
  ttsAutoPlay,
  playbackSpeed,
  tldwTtsModel,
  tldwTtsVoice,
  tldwTtsResponseFormat,
  tldwTtsSpeed,
  tldwTtsLanguage,
  tldwTtsStreaming,
  tldwTtsEmotion,
  tldwTtsEmotionIntensity,
  tldwTtsNormalize,
  tldwTtsNormalizeUnits,
  tldwTtsNormalizeUrls,
  tldwTtsNormalizeEmails,
  tldwTtsNormalizePhones,
  tldwTtsNormalizePlurals
}: {
  ttsEnabled: boolean
  ttsProvider: string
  voice: string
  ssmlEnabled: boolean
  elevenLabsApiKey: string
  elevenLabsVoiceId: string
  elevenLabsModel: string
  elevenLabsKeyValid?: boolean | null
  elevenLabsKeyTestedAt?: string
  responseSplitting: string
  removeReasoningTagTTS: boolean
  openAITTSBaseUrl: string
  openAITTSApiKey: string
  openAITTSModel: string
  openAITTSVoice: string
  openAITTSKeyValid?: boolean | null
  openAITTSKeyTestedAt?: string
  ttsAutoPlay: boolean
  playbackSpeed: number
  tldwTtsModel: string
  tldwTtsVoice: string
  tldwTtsResponseFormat: string
  tldwTtsSpeed: number
  tldwTtsLanguage: string
  tldwTtsStreaming: boolean
  tldwTtsEmotion: string
  tldwTtsEmotionIntensity: number
  tldwTtsNormalize: boolean
  tldwTtsNormalizeUnits: boolean
  tldwTtsNormalizeUrls: boolean
  tldwTtsNormalizeEmails: boolean
  tldwTtsNormalizePhones: boolean
  tldwTtsNormalizePlurals: boolean
}) => {
  const updates = [
    setTTSEnabled(ttsEnabled),
    setTTSProvider(ttsProvider),
    setVoice(voice),
    setSSMLEnabled(ssmlEnabled),
    setElevenLabsApiKey(elevenLabsApiKey),
    setElevenLabsVoiceId(elevenLabsVoiceId),
    setElevenLabsModel(elevenLabsModel),
    setResponseSplitting(responseSplitting),
    setRemoveReasoningTagTTS(removeReasoningTagTTS),
    setOpenAITTSBaseUrl(openAITTSBaseUrl),
    setOpenAITTSApiKey(openAITTSApiKey),
    setOpenAITTSModel(openAITTSModel),
    setOpenAITTSVoice(openAITTSVoice),
    setTTSAutoPlayEnabled(ttsAutoPlay),
    setSpeechPlaybackSpeed(playbackSpeed),
    setTldwTTSModel(tldwTtsModel),
    setTldwTTSVoice(tldwTtsVoice),
    setTldwTTSResponseFormat(tldwTtsResponseFormat),
    setTldwTTSSpeed(tldwTtsSpeed),
    setTldwTTSLanguage(tldwTtsLanguage),
    setTldwTTSStreamingEnabled(tldwTtsStreaming),
    setTldwTTSEmotion(tldwTtsEmotion),
    setTldwTTSEmotionIntensity(tldwTtsEmotionIntensity),
    setTldwTTSNormalize(tldwTtsNormalize),
    setTldwTTSNormalizeUnits(tldwTtsNormalizeUnits),
    setTldwTTSNormalizeUrls(tldwTtsNormalizeUrls),
    setTldwTTSNormalizeEmails(tldwTtsNormalizeEmails),
    setTldwTTSNormalizePhones(tldwTtsNormalizePhones),
    setTldwTTSNormalizePlurals(tldwTtsNormalizePlurals)
  ]
  if (elevenLabsKeyValid !== undefined) {
    updates.push(setElevenLabsKeyValid(elevenLabsKeyValid))
  }
  if (elevenLabsKeyTestedAt !== undefined) {
    updates.push(setElevenLabsKeyTestedAt(elevenLabsKeyTestedAt))
  }
  if (openAITTSKeyValid !== undefined) {
    updates.push(setOpenAITTSKeyValid(openAITTSKeyValid))
  }
  if (openAITTSKeyTestedAt !== undefined) {
    updates.push(setOpenAITTSKeyTestedAt(openAITTSKeyTestedAt))
  }
  await Promise.all(updates)
}
