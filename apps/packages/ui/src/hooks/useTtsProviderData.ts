import { useQuery } from "@tanstack/react-query"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useServerOnline } from "@/hooks/useServerOnline"
import { getModels, getVoices, type Model, type Voice } from "@/services/elevenlabs"
import {
  fetchTtsProviders,
  type TldwTtsProvidersInfo,
  type TldwTtsVoiceInfo
} from "@/services/tldw/audio-providers"
import { fetchTldwTtsModels, type TldwTtsModel } from "@/services/tldw/audio-models"
import { fetchTldwVoiceCatalog } from "@/services/tldw/audio-voices"
import { toServerTtsProviderKey } from "@/services/tldw/tts-provider-keys"

const ELEVENLABS_METADATA_TIMEOUT_MS = 10_000

export const OPENAI_TTS_MODELS = [
  { label: "tts-1", value: "tts-1" },
  { label: "tts-1-hd", value: "tts-1-hd" }
]

export const OPENAI_TTS_VOICES: Record<string, { label: string; value: string }[]> = {
  "tts-1": [
    { label: "alloy", value: "alloy" },
    { label: "echo", value: "echo" },
    { label: "fable", value: "fable" },
    { label: "onyx", value: "onyx" },
    { label: "nova", value: "nova" },
    { label: "shimmer", value: "shimmer" }
  ],
  "tts-1-hd": [
    { label: "alloy", value: "alloy" },
    { label: "echo", value: "echo" },
    { label: "fable", value: "fable" },
    { label: "onyx", value: "onyx" },
    { label: "nova", value: "nova" },
    { label: "shimmer", value: "shimmer" }
  ]
}

type ElevenLabsData = {
  voices: Voice[]
  models: Model[]
} | null

type UseTtsProviderDataArgs = {
  provider?: string
  elevenLabsApiKey?: string | null
  inferredProviderKey?: string | null
}

export const useTtsProviderData = ({
  provider,
  elevenLabsApiKey,
  inferredProviderKey
}: UseTtsProviderDataArgs) => {
  const { capabilities, loading: capsLoading } = useServerCapabilities()
  const isOnline = useServerOnline()
  const hasAudio =
    isOnline &&
    !capsLoading &&
    Boolean(capabilities?.hasTts ?? capabilities?.hasAudio)

  const { data: providersInfo } = useQuery<TldwTtsProvidersInfo | null>({
    queryKey: ["tldw-tts-providers"],
    queryFn: () => fetchTtsProviders(),
    enabled: hasAudio
  })

  const { data: tldwTtsModels } = useQuery<TldwTtsModel[]>({
    queryKey: ["tldw-tts-models"],
    queryFn: fetchTldwTtsModels,
    enabled: hasAudio
  })

  const { data: tldwVoiceCatalog } = useQuery<TldwTtsVoiceInfo[]>({
    queryKey: ["tldw-voice-catalog", inferredProviderKey],
    queryFn: async () => {
      if (!inferredProviderKey) return []
      const voices = await fetchTldwVoiceCatalog(
        toServerTtsProviderKey(inferredProviderKey)
      )
      return voices.map((v) => ({
        id: v.voice_id || v.id || v.name,
        name: v.name || v.voice_id || v.id,
        language: (v as any)?.language,
        gender: (v as any)?.gender,
        description: v.description,
        preview_url: (v as any)?.preview_url
      })) as TldwTtsVoiceInfo[]
    },
    enabled: hasAudio && provider === "tldw" && Boolean(inferredProviderKey)
  })

  const {
    data: elevenLabsData,
    isLoading: elevenLabsLoading,
    error: elevenLabsError,
    refetch: refetchElevenLabs
  } =
    useQuery<ElevenLabsData>({
      queryKey: ["tts-playground-elevenlabs", provider, elevenLabsApiKey],
      queryFn: async () => {
        if (provider !== "elevenlabs" || !elevenLabsApiKey) {
          return null
        }
        const [voices, models] = await Promise.all([
          getVoices(elevenLabsApiKey, { timeoutMs: ELEVENLABS_METADATA_TIMEOUT_MS }),
          getModels(elevenLabsApiKey, { timeoutMs: ELEVENLABS_METADATA_TIMEOUT_MS })
        ])
        return { voices, models }
      },
      enabled: provider === "elevenlabs" && Boolean(elevenLabsApiKey),
      retry: false
    })

  return {
    hasAudio,
    ffmpegAvailable: capabilities?.ffmpegAvailable ?? null,
    providersInfo,
    tldwTtsModels,
    tldwVoiceCatalog,
    elevenLabsData,
    elevenLabsLoading,
    elevenLabsError,
    refetchElevenLabs
  }
}
