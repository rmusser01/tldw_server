import { bgRequestClient } from "@/services/background-proxy"

const MAX_PROVIDERS = 256
const MAX_MODELS = 1_000
const MAX_MODEL_CAPABILITIES = 1_000
const MAX_VOICES = 1_000
const MAX_FORMATS = 64
const MAX_LANGUAGES = 256
const MAX_FALLBACK_TARGETS = 64

export type TldwTtsVoiceInfo = {
  id?: string
  voice_id?: string
  name?: string
  language?: string
  gender?: string
  description?: string | null
  preview_url?: string | null
}

export type TldwTtsModelCapabilities = {
  formats?: string[]
  native_formats?: string[]
  converted_formats?: string[]
  default_format?: string
  default_voice?: string | null
  voices?: string[]
  requires_freeform_voice?: boolean
  supports_speed?: boolean
  supports_language?: boolean
  supports_target_sample_rate?: boolean
  allow_octet_stream?: boolean
  max_input_characters?: number
  max_response_bytes?: number
}

export type TldwTtsProviderCapabilities = {
  provider_name?: string
  display_name?: string
  models?: string[]
  default_model?: string | null
  model_capabilities?: Record<string, TldwTtsModelCapabilities>
  fallback?: {
    available?: boolean
    targets?: string[]
  }
  voices?: TldwTtsVoiceInfo[]
  formats?: string[]
  default_format?: string
  languages?: string[]
  supports_streaming?: boolean
  supports_voice_cloning?: boolean
  supports_ssml?: boolean
  supports_speech_rate?: boolean
  supports_emotion_control?: boolean
}

export type TldwTtsProvidersInfo = {
  providers: Record<string, TldwTtsProviderCapabilities>
  voices: Record<string, TldwTtsVoiceInfo[]>
  supports_explicit_backend?: boolean
}

type FetchTtsProvidersOptions = {
  throwOnError?: boolean
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const hasOwn = (value: Record<string, unknown>, key: string) =>
  Object.prototype.hasOwnProperty.call(value, key)

const cleanString = (value: unknown): string | undefined =>
  typeof value === "string" && value.trim() ? value : undefined

const cleanStringArray = (value: unknown, limit: number): string[] => {
  if (!Array.isArray(value)) return []
  const values: string[] = []
  const seen = new Set<string>()
  for (const candidate of value) {
    const cleaned = cleanString(candidate)
    if (!cleaned || seen.has(cleaned)) continue
    seen.add(cleaned)
    values.push(cleaned)
    if (values.length === limit) break
  }
  return values
}

const copyString = (
  source: Record<string, unknown>,
  target: Record<string, unknown>,
  key: string
) => {
  const value = cleanString(source[key])
  if (value !== undefined) target[key] = value
}

const copyNullableString = (
  source: Record<string, unknown>,
  target: Record<string, unknown>,
  key: string
) => {
  if (source[key] === null) {
    target[key] = null
    return
  }
  copyString(source, target, key)
}

const copyBoolean = (
  source: Record<string, unknown>,
  target: Record<string, unknown>,
  key: string
) => {
  if (typeof source[key] === "boolean") target[key] = source[key]
}

const copyPositiveNumber = (
  source: Record<string, unknown>,
  target: Record<string, unknown>,
  key: string
) => {
  const value = source[key]
  if (typeof value === "number" && Number.isFinite(value) && value > 0) {
    target[key] = value
  }
}

const normalizeVoice = (value: unknown): TldwTtsVoiceInfo | null => {
  const legacyId = cleanString(value)
  if (legacyId) return { id: legacyId, name: legacyId }
  if (!isRecord(value)) return null

  const voice: TldwTtsVoiceInfo = {}
  for (const key of ["id", "voice_id", "name", "language", "gender"] as const) {
    const field = cleanString(value[key])
    if (field !== undefined) voice[key] = field
  }
  for (const key of ["description", "preview_url"] as const) {
    const field = value[key]
    if (field === null) voice[key] = null
    else if (typeof field === "string") voice[key] = field
  }

  return voice.id || voice.voice_id || voice.name ? voice : null
}

const normalizeVoices = (value: unknown): TldwTtsVoiceInfo[] => {
  if (!Array.isArray(value)) return []
  const voices: TldwTtsVoiceInfo[] = []
  for (const candidate of value) {
    const voice = normalizeVoice(candidate)
    if (!voice) continue
    voices.push(voice)
    if (voices.length === MAX_VOICES) break
  }
  return voices
}

const normalizeModelCapabilities = (
  value: unknown
): Record<string, TldwTtsModelCapabilities> => {
  if (!isRecord(value)) return {}
  const result: Record<string, TldwTtsModelCapabilities> = {}

  for (const [model, rawCapabilities] of Object.entries(value)) {
    if (Object.keys(result).length === MAX_MODEL_CAPABILITIES) break
    if (!model.trim() || !isRecord(rawCapabilities)) continue
    const capabilities: Record<string, unknown> = {}

    for (const key of ["formats", "native_formats", "converted_formats"] as const) {
      if (hasOwn(rawCapabilities, key)) {
        capabilities[key] = cleanStringArray(rawCapabilities[key], MAX_FORMATS)
      }
    }
    if (hasOwn(rawCapabilities, "voices")) {
      capabilities.voices = cleanStringArray(
        rawCapabilities.voices,
        MAX_VOICES
      )
    }
    copyString(rawCapabilities, capabilities, "default_format")
    copyNullableString(rawCapabilities, capabilities, "default_voice")
    for (const key of [
      "requires_freeform_voice",
      "supports_speed",
      "supports_language",
      "supports_target_sample_rate",
      "allow_octet_stream"
    ]) {
      copyBoolean(rawCapabilities, capabilities, key)
    }
    for (const key of ["max_input_characters", "max_response_bytes"]) {
      copyPositiveNumber(rawCapabilities, capabilities, key)
    }
    result[model] = capabilities as TldwTtsModelCapabilities
  }

  return result
}

const normalizeProvider = (
  value: unknown
): TldwTtsProviderCapabilities | null => {
  if (!isRecord(value)) return null
  const provider: Record<string, unknown> = {}

  copyString(value, provider, "provider_name")
  copyString(value, provider, "display_name")
  copyNullableString(value, provider, "default_model")
  copyString(value, provider, "default_format")

  if (hasOwn(value, "models")) {
    provider.models = cleanStringArray(value.models, MAX_MODELS)
  }
  if (hasOwn(value, "formats")) {
    provider.formats = cleanStringArray(value.formats, MAX_FORMATS)
  }
  if (hasOwn(value, "languages")) {
    provider.languages = cleanStringArray(value.languages, MAX_LANGUAGES)
  }
  if (hasOwn(value, "voices")) {
    provider.voices = normalizeVoices(value.voices)
  }
  if (hasOwn(value, "model_capabilities")) {
    provider.model_capabilities = normalizeModelCapabilities(
      value.model_capabilities
    )
  }

  if (isRecord(value.fallback)) {
    const fallback: Record<string, unknown> = {}
    copyBoolean(value.fallback, fallback, "available")
    if (hasOwn(value.fallback, "targets")) {
      fallback.targets = cleanStringArray(
        value.fallback.targets,
        MAX_FALLBACK_TARGETS
      )
    }
    provider.fallback = fallback
  }

  for (const key of [
    "supports_streaming",
    "supports_voice_cloning",
    "supports_ssml",
    "supports_speech_rate",
    "supports_emotion_control"
  ]) {
    copyBoolean(value, provider, key)
  }

  return provider as TldwTtsProviderCapabilities
}

/** Normalize untrusted provider discovery before it reaches UI consumers. */
export const normalizeTtsProvidersResponse = (
  value: unknown
): TldwTtsProvidersInfo | null => {
  if (!isRecord(value)) return null
  const rawProviders = value.providers ?? value
  const rawVoices = value.voices ?? {}
  const providers: Record<string, TldwTtsProviderCapabilities> = {}
  const voices: Record<string, TldwTtsVoiceInfo[]> = {}

  if (isRecord(rawProviders)) {
    for (const [key, rawProvider] of Object.entries(rawProviders)) {
      if (Object.keys(providers).length === MAX_PROVIDERS) break
      if (!key.trim()) continue
      const provider = normalizeProvider(rawProvider)
      if (!provider) continue
      providers[key] = provider
      if (provider.voices?.length) voices[key] = provider.voices
    }
  }

  if (isRecord(rawVoices)) {
    for (const [key, rawVoiceList] of Object.entries(rawVoices)) {
      if (Object.keys(voices).length === MAX_PROVIDERS) break
      if (!key.trim()) continue
      voices[key] = normalizeVoices(rawVoiceList)
    }
  }

  if (Object.keys(providers).length === 0 && Object.keys(voices).length === 0) {
    return null
  }

  return {
    providers,
    voices,
    supports_explicit_backend: value.supports_explicit_backend === true
  }
}

export const fetchTtsProviders = async (
  options?: FetchTtsProvidersOptions
): Promise<TldwTtsProvidersInfo | null> => {
  try {
    const response = await bgRequestClient<unknown>({
      path: "/api/v1/audio/providers",
      method: "GET"
    })
    return normalizeTtsProvidersResponse(response)
  } catch (error) {
    if (options?.throwOnError) throw error
    return null
  }
}
