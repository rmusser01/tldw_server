import { useQuery } from "@tanstack/react-query"
import { apiSend } from "@/services/api-send"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import {
  fetchTldwVoiceCatalog,
  fetchTldwVoices,
  type TldwVoice
} from "@/services/tldw/audio-voices"
import { inferTldwProviderFromModel } from "@/services/tts-provider"
import { toServerTtsProviderKey } from "@/services/tldw/tts-provider-keys"

export type AudioHealthState =
  | "unknown"
  | "healthy"
  | "unhealthy"
  | "unavailable"

type Options = {
  enabled?: boolean
  requireVoices?: boolean
  ttsProvider?: string | null
  tldwTtsModel?: string | null
}

type AudioStatus = {
  hasAudio: boolean
  hasStt: boolean
  hasTts: boolean
  hasVoiceChat: boolean
  hasVoiceConversationTransport: boolean
  healthState: AudioHealthState
  healthLoading: boolean
  sttHealthState: AudioHealthState
  sttHealthLoading: boolean
  ttsHealthState: AudioHealthState
  ttsHealthLoading: boolean
  voices: TldwVoice[]
  voicesLoading: boolean
  voicesAvailable: boolean | null
}

type AudioHealthResponse = {
  ok?: boolean
  status?: number
  data?: {
    status?: string
    available?: boolean
    usable?: boolean
    on_demand?: boolean
    provider?: string
    providers?: {
      details?: Record<
        string,
        {
          status?: string
          availability?: string
          runtime?: string
        }
      >
    }
    capabilities_envelope?: Array<{
      provider?: string
      availability?: string
      runtime?: string | null
    }>
  }
}

const READY_TTS_STATUSES = new Set(["enabled", "available", "ready", "healthy", "ok"])

const normalizeHealthStatus = (value?: string | null): string => {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/_/g, "-")
}

const isReadyHealthStatus = (value?: string | null): boolean => {
  const normalized = normalizeHealthStatus(value)
  return normalized.length > 0 && READY_TTS_STATUSES.has(normalized)
}

const EXPLICITLY_UNAVAILABLE_TTS_STATUSES = new Set([
  "disabled",
  "failed",
  "unavailable"
])

const getProviderHealthStatus = (
  payload: AudioHealthResponse["data"],
  provider: string | null
): string => {
  if (!payload || !provider) return ""
  const details = payload.providers?.details
  const detail = details
    ? Object.entries(details).find(
        ([key]) => toServerTtsProviderKey(key) === provider
      )?.[1]
    : undefined
  const envelope = payload.capabilities_envelope?.find(
    (entry) => toServerTtsProviderKey(entry.provider) === provider
  )

  return normalizeHealthStatus(
    detail?.availability ??
      envelope?.availability ??
      detail?.status
  )
}

const TTS_HEALTH_PROBE_INTERVAL_MS = 60_000
const STT_HEALTH_PROBE_INTERVAL_MS = 45_000
const HEALTH_PROBE_RETRY_ATTEMPTS = 1
const HEALTH_PROBE_RETRY_DELAY_MS = 500

export const useTldwAudioStatus = (options: Options = {}): AudioStatus => {
  const { capabilities, loading } = useServerCapabilities()
  const probeEnabled = options.enabled ?? true
  const shouldProbeVoices = probeEnabled && Boolean(options.requireVoices)
  const requestedTtsProvider = String(options.ttsProvider || "")
    .trim()
    .toLowerCase()
  const inferredSelectedTtsProvider =
    !requestedTtsProvider ||
    requestedTtsProvider === "browser" ||
    requestedTtsProvider === "tldw"
      ? inferTldwProviderFromModel(options.tldwTtsModel)
      : null
  const selectedTtsProvider =
    requestedTtsProvider === "openai" || requestedTtsProvider === "elevenlabs"
      ? requestedTtsProvider
      : requestedTtsProvider &&
            requestedTtsProvider !== "browser" &&
            requestedTtsProvider !== "tldw"
        ? toServerTtsProviderKey(requestedTtsProvider)
        : inferredSelectedTtsProvider
          ? toServerTtsProviderKey(inferredSelectedTtsProvider)
          : null
  const inferredTldwProvider = options.requireVoices
    ? inferTldwProviderFromModel(options.tldwTtsModel)
    : null
  const catalogProvider = inferredTldwProvider
    ? toServerTtsProviderKey(inferredTldwProvider)
    : null
  const hasStt = !loading && Boolean(capabilities?.hasStt ?? capabilities?.hasAudio)
  const hasTts = !loading && Boolean(capabilities?.hasTts ?? capabilities?.hasAudio)
  const hasVoiceChat =
    !loading &&
    Boolean(
      capabilities?.hasVoiceChat ??
        (capabilities?.hasStt != null && capabilities?.hasTts != null
          ? capabilities.hasStt && capabilities.hasTts
          : capabilities?.hasAudio)
    )
  const hasVoiceConversationTransport =
    !loading && Boolean(capabilities?.hasVoiceConversationTransport)
  const hasAudio =
    !loading &&
    Boolean(capabilities?.hasAudio ?? (hasStt || hasTts || hasVoiceChat))

  const ttsHealthQuery = useQuery<AudioHealthResponse>({
    queryKey: ["audio-health", "tts"],
    queryFn: async () =>
      (await apiSend({
        path: "/api/v1/audio/health",
        method: "GET"
      })) as AudioHealthResponse,
    enabled: hasTts && probeEnabled,
    staleTime: TTS_HEALTH_PROBE_INTERVAL_MS,
    refetchInterval: TTS_HEALTH_PROBE_INTERVAL_MS,
    retry: HEALTH_PROBE_RETRY_ATTEMPTS,
    retryDelay: HEALTH_PROBE_RETRY_DELAY_MS,
    refetchOnMount: false,
    refetchOnWindowFocus: false
  })

  const sttHealthQuery = useQuery<AudioHealthResponse>({
    queryKey: ["audio-health", "stt"],
    queryFn: async () =>
      (await apiSend({
        path: "/api/v1/audio/transcriptions/health",
        method: "GET"
      })) as AudioHealthResponse,
    enabled: hasStt && probeEnabled,
    staleTime: STT_HEALTH_PROBE_INTERVAL_MS,
    refetchInterval: STT_HEALTH_PROBE_INTERVAL_MS,
    retry: HEALTH_PROBE_RETRY_ATTEMPTS,
    retryDelay: HEALTH_PROBE_RETRY_DELAY_MS,
    refetchOnMount: false,
    refetchOnWindowFocus: false
  })

  let sttHealthState: AudioHealthState = "unknown"
  if (!hasStt) {
    sttHealthState = loading ? "unknown" : "unavailable"
  } else if (!probeEnabled) {
    sttHealthState = "unknown"
  } else if (sttHealthQuery.isLoading) {
    sttHealthState = "unknown"
  } else if (sttHealthQuery.isError) {
    // Fail-open on probe transport errors; let real transcription attempts decide.
    sttHealthState = "unknown"
  } else if (sttHealthQuery.data?.ok) {
    const sttPayload = sttHealthQuery.data?.data
    const provider = String(sttPayload?.provider ?? "")
      .trim()
      .toLowerCase()
    const explicitlyUsable = sttPayload?.usable === true
    const onDemandReady = sttPayload?.on_demand === true
    const failOpenForNonWhisper =
      sttPayload?.available === false && provider.length > 0 && provider !== "whisper"
    if (sttPayload?.available === false && !explicitlyUsable && !onDemandReady && !failOpenForNonWhisper) {
      sttHealthState = "unhealthy"
    } else {
      sttHealthState = "healthy"
    }
  } else if (sttHealthQuery.data?.status === 404) {
    sttHealthState = "unknown"
  } else {
    sttHealthState = "unhealthy"
  }

  const voicesQuery = useQuery<TldwVoice[]>({
    queryKey: ["audio-voices", catalogProvider],
    queryFn: async () => {
      if (catalogProvider) {
        const catalogVoices = await fetchTldwVoiceCatalog(catalogProvider)
        if (catalogVoices.length > 0) {
          return catalogVoices
        }
      }
      return fetchTldwVoices()
    },
    enabled:
      shouldProbeVoices &&
      !loading &&
      !(
        catalogProvider &&
        hasTts &&
        probeEnabled &&
        ttsHealthQuery.isLoading
      ) &&
      !EXPLICITLY_UNAVAILABLE_TTS_STATUSES.has(
        getProviderHealthStatus(ttsHealthQuery.data?.data, catalogProvider)
      ),
    staleTime: 300_000,
    refetchOnWindowFocus: false
  })

  const voices = voicesQuery.data ?? []
  const voicesWaitingForCapabilities = shouldProbeVoices && loading
  const voicesWaitingForHealth = Boolean(
    shouldProbeVoices &&
      catalogProvider &&
      hasTts &&
      probeEnabled &&
      ttsHealthQuery.isLoading
  )
  const voicesLoading =
    voicesWaitingForCapabilities || voicesWaitingForHealth || voicesQuery.isLoading
  const voicesAvailable =
    options.requireVoices && !voicesLoading ? voices.length > 0 : null
  const voicesConfirmTts =
    Boolean(options.requireVoices) && !voicesLoading && voices.length > 0

  let ttsHealthState: AudioHealthState = "unknown"
  if (!hasTts) {
    if (loading || voicesQuery.isLoading) {
      ttsHealthState = "unknown"
    } else {
      ttsHealthState = voicesConfirmTts ? "unknown" : "unavailable"
    }
  } else if (!probeEnabled) {
    ttsHealthState = "unknown"
  } else if (ttsHealthQuery.isLoading) {
    ttsHealthState = "unknown"
  } else if (ttsHealthQuery.isError) {
    // Probe errors should not hard-disable audio features.
    ttsHealthState = "unknown"
  } else if (ttsHealthQuery.data?.ok) {
    const healthPayload = ttsHealthQuery.data.data
    const selectedProviderStatus = getProviderHealthStatus(
      healthPayload,
      selectedTtsProvider
    )
    const overallStatus = normalizeHealthStatus(healthPayload?.status)
    const overallReady = overallStatus ? isReadyHealthStatus(overallStatus) : true
    const selectedProviderReady =
      selectedTtsProvider == null
        ? overallReady
        : selectedProviderStatus
          ? isReadyHealthStatus(selectedProviderStatus)
          : overallReady

    ttsHealthState = selectedProviderReady ? "healthy" : "unhealthy"
  } else if (ttsHealthQuery.data?.status === 404) {
    ttsHealthState = "unknown"
  } else {
    ttsHealthState = "unhealthy"
  }

  return {
    hasAudio,
    hasStt,
    hasTts,
    hasVoiceChat,
    hasVoiceConversationTransport,
    healthState: ttsHealthState,
    healthLoading: probeEnabled ? ttsHealthQuery.isLoading : false,
    sttHealthState,
    sttHealthLoading: probeEnabled ? sttHealthQuery.isLoading : false,
    ttsHealthState,
    ttsHealthLoading: probeEnabled ? ttsHealthQuery.isLoading : false,
    voices,
    voicesLoading: probeEnabled ? voicesLoading : false,
    voicesAvailable
  }
}
