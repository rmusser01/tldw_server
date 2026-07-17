import { bgRequestClient } from "@/services/background-proxy"

export type TldwTtsVoiceInfo = {
  id?: string
  name?: string
  language?: string
  gender?: string
  description?: string | null
  preview_url?: string | null
  [key: string]: any
}

export type TldwTtsProviderCapabilities = {
  provider_name?: string
  display_name?: string
  models?: unknown[]
  default_model?: string | null
  model_capabilities?: Record<
    string,
    {
      formats?: string[]
      default_voice?: string | null
      voices?: string[]
      requires_freeform_voice?: boolean
      [key: string]: unknown
    }
  >
  fallback?: {
    available?: boolean
    targets?: string[]
  }
  formats?: string[]
  default_format?: string
  languages?: string[]
  supports_streaming?: boolean
  supports_voice_cloning?: boolean
  supports_ssml?: boolean
  supports_speech_rate?: boolean
  supports_emotion_control?: boolean
  [key: string]: any
}

export type TldwTtsProvidersInfo = {
  providers: Record<string, TldwTtsProviderCapabilities>
  voices: Record<string, TldwTtsVoiceInfo[]>
  supports_explicit_backend?: boolean
}

type FetchTtsProvidersOptions = {
  throwOnError?: boolean
}

export const fetchTtsProviders = async (
  options?: FetchTtsProvidersOptions
): Promise<TldwTtsProvidersInfo | null> => {
  try {
    const res = await bgRequestClient<any>({
      path: "/api/v1/audio/providers",
      method: "GET"
    })

    if (!res) {
      return null
    }

    const rawProviders = res.providers ?? res
    const rawVoices = res.voices ?? {}

    const providers: Record<string, TldwTtsProviderCapabilities> = {}
    const voices: Record<string, TldwTtsVoiceInfo[]> = {}

    if (rawProviders && typeof rawProviders === "object") {
      for (const key of Object.keys(rawProviders)) {
        const value = rawProviders[key]
        if (value && typeof value === "object") {
          providers[key] = value as TldwTtsProviderCapabilities
          const providerVoices = (value as { voices?: unknown }).voices
          if (
            !voices[key] &&
            Array.isArray(providerVoices) &&
            providerVoices.length > 0
          ) {
            voices[key] = providerVoices as TldwTtsVoiceInfo[]
          }
        }
      }
    }

    if (rawVoices && typeof rawVoices === "object") {
      for (const key of Object.keys(rawVoices)) {
        const list = Array.isArray(rawVoices[key]) ? rawVoices[key] : []
        voices[key] = list as TldwTtsVoiceInfo[]
      }
    }

    if (Object.keys(providers).length === 0 && Object.keys(voices).length === 0) {
      return null
    }

    return {
      providers,
      voices,
      supports_explicit_backend: res.supports_explicit_backend === true
    }
  } catch (error) {
    if (options?.throwOnError) {
      throw error
    }
    return null
  }
}
