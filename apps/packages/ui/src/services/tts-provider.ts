import {
  getElevenLabsApiKey,
  getElevenLabsModel,
  getElevenLabsVoiceId,
  getOpenAITTSModel,
  getOpenAITTSVoice,
  getRemoveReasoningTagTTS,
  getSpeechPlaybackSpeed,
  getTTSProvider,
  getTldwTTSModel,
  getTldwTTSResponseFormat,
  getTldwTTSSpeed,
  getTldwTTSVoice,
  getVoice,
  isSSMLEnabled,
  isSupportedTldwTtsResponseFormat,
  normalizeTldwTtsResponseFormat
} from "@/services/tts"
import { markdownToSSML } from "@/utils/markdown-to-ssml"
import { removeReasoning } from "@/libs/reasoning"
import { markdownToText } from "@/utils/markdown-to-text"
import { generateSpeech } from "@/services/elevenlabs"
import { generateOpenAITTS } from "@/services/openai-tts"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { inferProviderFromModel } from "@/utils/provider-registry"
import {
  TTS_PROVIDER_VALUES,
  type TtsProviderValue
} from "@/services/tts-providers"

export type TtsProviderKey = TtsProviderValue

export type TtsProviderOverrides = {
  provider?: string
  elevenLabsModel?: string
  elevenLabsVoiceId?: string
  elevenLabsSpeed?: number
  tldwModel?: string
  tldwVoice?: string
  tldwResponseFormat?: string
  tldwSpeed?: number
  tldwLanguage?: string
  tldwNormalizationOptions?: {
    normalize?: boolean
    unit_normalization?: boolean
    url_normalization?: boolean
    email_normalization?: boolean
    phone_normalization?: boolean
    optional_pluralization_normalization?: boolean
  }
  tldwExtraParams?: Record<string, any>
  openAiModel?: string
  openAiVoice?: string
  openAiSpeed?: number
}

export type TtsSynthesisResult = {
  buffer: ArrayBuffer
  format: string
  mimeType: string
}

export type TtsSynthesizeOptions = {
  signal?: AbortSignal
}

export type TtsFormatInfo = {
  requested?: string | null
  resolved: string
  isFallback: boolean
}

export type TtsCacheSettings = {
  provider: string
  model?: string | null
  voice?: string | null
  speed?: number | null
  format?: string | null
  language?: string | null
}

export type TtsTextNormalizer = (text: string) => string

export type TtsProviderContext = {
  provider: string
  utterance: string
  normalizeText: TtsTextNormalizer
  playbackSpeed: number
  supported: boolean
  browserVoiceName?: string | null
  synthesize?: (
    text: string,
    options?: TtsSynthesizeOptions
  ) => Promise<TtsSynthesisResult>
  formatInfo?: TtsFormatInfo
  cacheSettings?: TtsCacheSettings
}

const SUPPORTED_TTS_PROVIDERS = new Set<TtsProviderKey>(TTS_PROVIDER_VALUES)

const formatToMimeType = (format: string): string => {
  switch (format) {
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
      return "audio/L16; rate=24000; channels=1"
    case "mp3":
    default:
      return "audio/mpeg"
  }
}

const normalizeUtteranceWithSettings = (
  text: string,
  {
    shouldRemoveReasoning,
    ssmlEnabled
  }: {
    shouldRemoveReasoning: boolean
    ssmlEnabled: boolean
  }
): string => {
  let utterance = text
  if (shouldRemoveReasoning) {
    utterance = removeReasoning(utterance)
  }

  if (ssmlEnabled) {
    return markdownToSSML(utterance)
  }

  return markdownToText(utterance)
}

export const resolveTtsTextNormalizer = async (): Promise<TtsTextNormalizer> => {
  const [shouldRemoveReasoning, ssmlEnabled] = await Promise.all([
    getRemoveReasoningTagTTS(),
    isSSMLEnabled()
  ])

  return (text: string) =>
    normalizeUtteranceWithSettings(text, { shouldRemoveReasoning, ssmlEnabled })
}

export const applyBrowserSpeechSynthesisVoice = (
  utterance: SpeechSynthesisUtterance,
  synthesis: SpeechSynthesis | null | undefined,
  voiceName?: string | null
): (() => void) => {
  const noop = () => undefined
  const targetVoiceName = String(voiceName || "").trim()
  if (
    !targetVoiceName ||
    !synthesis ||
    typeof synthesis.getVoices !== "function"
  ) {
    return noop
  }

  let cleanup: () => void = noop
  let cleanedUp = false
  const applyVoice = () => {
    if (cleanedUp) return
    const selectedVoice = synthesis
      .getVoices()
      .find((voice) => voice.name === targetVoiceName)
    if (selectedVoice) {
      utterance.voice = selectedVoice
    }
  }

  applyVoice()
  if (utterance.voice) return noop

  if (
    typeof synthesis.addEventListener !== "function" ||
    typeof synthesis.removeEventListener !== "function"
  ) {
    return noop
  }

  const handleVoicesChanged = () => {
    applyVoice()
    if (utterance.voice) {
      cleanup()
    }
  }

  cleanup = () => {
    if (cleanedUp) return
    cleanedUp = true
    synthesis.removeEventListener("voiceschanged", handleVoicesChanged)
  }
  synthesis.addEventListener("voiceschanged", handleVoicesChanged)
  return cleanup
}

export const inferTldwProviderFromModel = (
  model?: string | null
): string | null => inferProviderFromModel(model, "tts")

export const resolveTtsProviderContext = async (
  text: string,
  overrides?: TtsProviderOverrides
): Promise<TtsProviderContext> => {
  const rawProvider = overrides?.provider ?? (await getTTSProvider())
  const provider = String(rawProvider || '').trim().toLowerCase()
  const normalizeText = await resolveTtsTextNormalizer()
  const utterance = normalizeText(text)
  const playbackSpeed = await getSpeechPlaybackSpeed()

  if (!SUPPORTED_TTS_PROVIDERS.has(provider as TtsProviderKey)) {
    return {
      provider,
      utterance,
      normalizeText,
      playbackSpeed,
      supported: false
    }
  }

  if (provider === "browser") {
    const browserVoiceName = await getVoice()
    return {
      provider,
      utterance,
      normalizeText,
      playbackSpeed,
      supported: true,
      browserVoiceName
    }
  }

  if (provider === "elevenlabs") {
    const apiKey = await getElevenLabsApiKey()
    const baseModel = await getElevenLabsModel()
    const baseVoice = await getElevenLabsVoiceId()
    const modelId = overrides?.elevenLabsModel || baseModel
    const voiceId = overrides?.elevenLabsVoiceId || baseVoice
    const speed = overrides?.elevenLabsSpeed

    if (!apiKey || !modelId || !voiceId) {
      throw new Error("Missing ElevenLabs configuration")
    }

    return {
      provider,
      utterance,
      normalizeText,
      playbackSpeed,
      supported: true,
      cacheSettings: {
        provider,
        model: modelId,
        voice: voiceId,
        speed,
        format: "mp3"
      },
      synthesize: async (segment: string, _options?: TtsSynthesizeOptions) => ({
        buffer: await generateSpeech(apiKey, segment, voiceId, modelId, speed, {
          signal: _options?.signal
        }),
        format: "mp3",
        mimeType: "audio/mpeg"
      })
    }
  }

  if (provider === "openai") {
    const baseModel = await getOpenAITTSModel()
    const baseVoice = await getOpenAITTSVoice()
    const model = overrides?.openAiModel || baseModel
    const voice = overrides?.openAiVoice || baseVoice
    const speed = overrides?.openAiSpeed

    return {
      provider,
      utterance,
      normalizeText,
      playbackSpeed,
      supported: true,
      cacheSettings: {
        provider,
        model,
        voice,
        speed,
        format: "mp3"
      },
      synthesize: async (segment: string, _options?: TtsSynthesizeOptions) => ({
        buffer: await generateOpenAITTS({
          text: segment,
          model,
          voice,
          speed,
          signal: _options?.signal
        }),
        format: "mp3",
        mimeType: "audio/mpeg"
      })
    }
  }

  const baseModel = await getTldwTTSModel()
  const baseVoice = await getTldwTTSVoice()
  const rawResponseFormat =
    overrides?.tldwResponseFormat || (await getTldwTTSResponseFormat())
  const responseFormat = normalizeTldwTtsResponseFormat(rawResponseFormat)
  const model = overrides?.tldwModel || baseModel
  const voice = overrides?.tldwVoice || baseVoice
  let speed = overrides?.tldwSpeed ?? (await getTldwTTSSpeed())
  const language = overrides?.tldwLanguage
  const normalizationOptions = overrides?.tldwNormalizationOptions
  const extraParams = overrides?.tldwExtraParams
  if (inferTldwProviderFromModel(model) === "kokoro" && Number.isFinite(speed)) {
    speed = Math.min(2, Math.max(0.5, speed))
  }
  const mimeType = formatToMimeType(responseFormat)
  const formatInfo: TtsFormatInfo = {
    requested: rawResponseFormat,
    resolved: responseFormat,
    isFallback: Boolean(rawResponseFormat) &&
      !isSupportedTldwTtsResponseFormat(rawResponseFormat)
  }

  return {
    provider,
    utterance,
    normalizeText,
    playbackSpeed,
    supported: true,
    formatInfo,
    cacheSettings: {
      provider,
      model,
      voice,
      speed,
      format: responseFormat,
      language
    },
    synthesize: async (segment: string, options?: TtsSynthesizeOptions) => ({
      buffer: await tldwClient.synthesizeSpeech(segment, {
        model,
        voice,
        responseFormat,
        speed,
        language,
        normalizationOptions,
        extraParams,
        stream: false,
        signal: options?.signal
      }),
      format: responseFormat,
      mimeType
    })
  }
}
