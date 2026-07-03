import { resolveBrowserWebSocketBase } from "@/services/tldw/browser-websocket"
import { normalizeTldwTtsResponseFormat } from "@/services/tts"
import { inferTldwProviderFromModel } from "@/services/tts-provider"
import { toServerTtsProviderKey } from "@/services/tldw/tts-provider-keys"
import type { VoiceChatTtsMode } from "@/services/settings/ui-settings"

type AudioHealthStateLike =
  | "unknown"
  | "healthy"
  | "unhealthy"
  | "unavailable"

type VoiceConversationTtsProvider =
  | "browser"
  | "tldw"
  | "openai"
  | "elevenlabs"
  | (string & {})

type VoiceConversationTtsConfigInput = {
  ttsProvider?: string | null
  tldwTtsModel?: string | null
  tldwTtsVoice?: string | null
  tldwTtsSpeed?: number | null
  tldwTtsResponseFormat?: string | null
  openAITTSModel?: string | null
  openAITTSVoice?: string | null
  elevenLabsModel?: string | null
  elevenLabsVoiceId?: string | null
  speechPlaybackSpeed?: number | null
  voiceChatTtsMode?: VoiceChatTtsMode | null
}

type VoiceConversationAvailabilityInput = {
  isConnectionReady: boolean
  hasVoiceConversationTransport: boolean
  authReady: boolean
  sttHealthState: AudioHealthStateLike
  ttsHealthState: AudioHealthStateLike
  selectedModel?: string | null
  allowBackendDefaultModel: boolean
  ttsConfigReady: boolean
}

type VoiceConversationAudioHealthProbeInput = {
  isConnectionReady: boolean
  hasServerVoiceChat: boolean
  hasServerStt: boolean
  optionalAudioHealthEnabled: boolean
}

type VoiceConversationPreflightInput = VoiceConversationTtsConfigInput & {
  serverUrl: string
  token: string
  requestedModel?: string | null
  resolveProvider: ({ modelId }: { modelId: string }) => Promise<string | undefined> | string | undefined
}

type VoiceConversationTtsConfig = {
  provider?: string
  model: string
  voice: string
  speed: number
  format: string
}

type VoiceConversationPreflight = {
  websocketUrl: string
  llm: { model?: string; provider?: string }
  tts: VoiceConversationTtsConfig
}

export type VoiceConversationReason =
  | "ok"
  | "not_connected"
  | "transport_missing"
  | "auth_missing"
  | "audio_unhealthy"
  | "tts_config_missing"
  | "tts_provider_unsupported"
  | "model_missing"
  | "voice_chat_disconnected"
  | "voice_chat_tts_error"
  | "voice_chat_error"

export type VoiceConversationAvailability = {
  available: boolean
  reason: VoiceConversationReason
  message: string | null
}

type VoiceConversationTtsResolution =
  | { ok: true; value: VoiceConversationTtsConfig }
  | {
      ok: false
      reason: Extract<
        VoiceConversationReason,
        "tts_config_missing" | "tts_provider_unsupported"
      >
    }

type VoiceConversationRuntimeError = {
  reason: Extract<
    VoiceConversationReason,
    "voice_chat_disconnected" | "voice_chat_tts_error" | "voice_chat_error"
  >
  message: string
}

const trimString = (value?: string | null): string => String(value || "").trim()

const resolveVoiceConversationFormat = (
  requestedFormat?: string | null,
  mode?: VoiceChatTtsMode | null
): string => {
  const normalized = normalizeTldwTtsResponseFormat(requestedFormat)
  if (mode === "stream" && normalized === "pcm") {
    return "mp3"
  }
  return normalized
}

const hasUsableAudioHealth = (audioHealthState: AudioHealthStateLike): boolean =>
  audioHealthState !== "unhealthy" && audioHealthState !== "unavailable"

const resolveSafeTldwProviderHint = (
  model: string,
  requestedProvider: VoiceConversationTtsProvider
): string | undefined => {
  if (requestedProvider === "browser") {
    return undefined
  }

  if (requestedProvider === "tldw") {
    return "tldw"
  }

  const inferredProvider = inferTldwProviderFromModel(model)
  if (!inferredProvider) {
    return undefined
  }

  return toServerTtsProviderKey(inferredProvider) || undefined
}

const buildTtsConfigMissingError = () =>
  new Error("Voice conversation TTS configuration is incomplete")

const buildTtsProviderUnsupportedError = () =>
  new Error("Voice conversation TTS provider is unsupported")

const resolveVoiceConversationAvailabilityMessage = (
  reason: VoiceConversationReason
): string | null => {
  switch (reason) {
    case "ok":
      return null
    case "not_connected":
      return "playground:voiceChat.unavailableReasons.notConnected"
    case "transport_missing":
      return "playground:voiceChat.unavailableReasons.transportMissing"
    case "auth_missing":
      return "playground:voiceChat.unavailableReasons.authMissing"
    case "audio_unhealthy":
      return "playground:voiceChat.unavailableReasons.audioUnhealthy"
    case "tts_config_missing":
      return "playground:voiceChat.unavailableReasons.ttsConfigMissing"
    case "tts_provider_unsupported":
      return "playground:voiceChat.unavailableReasons.ttsProviderUnsupported"
    case "model_missing":
      return "playground:voiceChat.unavailableReasons.modelMissing"
    case "voice_chat_disconnected":
      return "playground:voiceChat.unavailableReasons.disconnected"
    case "voice_chat_tts_error":
      return "playground:voiceChat.unavailableReasons.ttsError"
    case "voice_chat_error":
      return "playground:voiceChat.unavailableReasons.error"
    default:
      return "playground:voiceChat.unavailableBody"
  }
}

export const resolveVoiceConversationTtsConfig = (
  input: VoiceConversationTtsConfigInput
): VoiceConversationTtsResolution => {
  const provider = trimString(input.ttsProvider).toLowerCase() as VoiceConversationTtsProvider
  const normalizedProvider = provider || "browser"

  if (normalizedProvider === "browser" || normalizedProvider === "tldw") {
    const model = trimString(input.tldwTtsModel)
    const voice = trimString(input.tldwTtsVoice)
    if (!model || !voice) {
      return { ok: false, reason: "tts_config_missing" }
    }

    return {
      ok: true,
      value: {
        provider: resolveSafeTldwProviderHint(model, normalizedProvider),
        model,
        voice,
        speed:
          typeof input.tldwTtsSpeed === "number" && Number.isFinite(input.tldwTtsSpeed)
            ? input.tldwTtsSpeed
            : 1,
        format: resolveVoiceConversationFormat(
          input.tldwTtsResponseFormat,
          input.voiceChatTtsMode
        )
      }
    }
  }

  if (normalizedProvider === "openai") {
    const model = trimString(input.openAITTSModel)
    const voice = trimString(input.openAITTSVoice)
    if (!model || !voice) {
      return { ok: false, reason: "tts_config_missing" }
    }

    return {
      ok: true,
      value: {
        provider: "openai",
        model,
        voice,
        speed:
          typeof input.speechPlaybackSpeed === "number" &&
          Number.isFinite(input.speechPlaybackSpeed)
            ? input.speechPlaybackSpeed
            : 1,
        format: "mp3"
      }
    }
  }

  if (normalizedProvider === "elevenlabs") {
    const model = trimString(input.elevenLabsModel)
    const voice = trimString(input.elevenLabsVoiceId)
    if (!model || !voice) {
      return { ok: false, reason: "tts_config_missing" }
    }

    return {
      ok: true,
      value: {
        provider: "elevenlabs",
        model,
        voice,
        speed:
          typeof input.speechPlaybackSpeed === "number" &&
          Number.isFinite(input.speechPlaybackSpeed)
            ? input.speechPlaybackSpeed
            : 1,
        format: "mp3"
      }
    }
  }

  return { ok: false, reason: "tts_provider_unsupported" }
}

export const resolveVoiceConversationAvailability = (
  input: VoiceConversationAvailabilityInput
): VoiceConversationAvailability => {
  if (!input.isConnectionReady) {
    return {
      available: false,
      reason: "not_connected",
      message: resolveVoiceConversationAvailabilityMessage("not_connected")
    }
  }
  if (!input.hasVoiceConversationTransport) {
    return {
      available: false,
      reason: "transport_missing",
      message: resolveVoiceConversationAvailabilityMessage("transport_missing")
    }
  }
  if (!input.authReady) {
    return {
      available: false,
      reason: "auth_missing",
      message: resolveVoiceConversationAvailabilityMessage("auth_missing")
    }
  }
  if (
    !hasUsableAudioHealth(input.sttHealthState) ||
    !hasUsableAudioHealth(input.ttsHealthState)
  ) {
    return {
      available: false,
      reason: "audio_unhealthy",
      message: resolveVoiceConversationAvailabilityMessage("audio_unhealthy")
    }
  }
  if (!input.ttsConfigReady) {
    return {
      available: false,
      reason: "tts_config_missing",
      message: resolveVoiceConversationAvailabilityMessage("tts_config_missing")
    }
  }
  if (!trimString(input.selectedModel) && !input.allowBackendDefaultModel) {
    return {
      available: false,
      reason: "model_missing",
      message: resolveVoiceConversationAvailabilityMessage("model_missing")
    }
  }

  return { available: true, reason: "ok", message: null }
}

export const shouldProbeVoiceConversationAudioHealth = (
  input: VoiceConversationAudioHealthProbeInput
): boolean => {
  if (input.optionalAudioHealthEnabled) {
    return true
  }

  if (!input.isConnectionReady) {
    return false
  }

  return input.hasServerVoiceChat || input.hasServerStt
}

export const buildVoiceConversationPreflight = async (
  input: VoiceConversationPreflightInput
): Promise<VoiceConversationPreflight> => {
  const serverUrl = trimString(input.serverUrl)
  if (!serverUrl) {
    throw new Error("tldw server not configured")
  }

  const token = trimString(input.token)
  if (!token) {
    throw new Error("Not authenticated. Configure tldw credentials in Settings.")
  }

  const ttsResolution = resolveVoiceConversationTtsConfig(input)
  if ("reason" in ttsResolution) {
    const error =
      ttsResolution.reason === "tts_provider_unsupported"
        ? buildTtsProviderUnsupportedError()
        : buildTtsConfigMissingError()
    throw error
  }
  const ttsConfig = ttsResolution.value

  const requestedModel = trimString(input.requestedModel)
  const llm = requestedModel
    ? {
        model: requestedModel,
        provider: await input.resolveProvider({ modelId: requestedModel })
      }
    : {}

  return {
    // TASK-12106: the auth token is intentionally NOT placed in the URL (it would
    // leak into access/proxy logs). The audio WS endpoint authenticates from an
    // {type:"auth", token} first frame sent by the client after `onopen`
    // (streaming_service.py:641-647 multi-user / 720-723 single-user). The token
    // is still validated above so callers fail fast when it is missing.
    websocketUrl: `${resolveBrowserWebSocketBase(serverUrl)}/api/v1/audio/chat/stream`,
    llm,
    tts: ttsConfig
  }
}

export const normalizeVoiceConversationRuntimeError = (
  error: unknown
): VoiceConversationRuntimeError => {
  const message =
    error instanceof Error ? trimString(error.message) : trimString(String(error || ""))
  const normalized = message.toLowerCase()

  if (
    /disconnect|connection\s+lost|websocket.*close|socket.*close|stream.*close/.test(
      normalized
    )
  ) {
    return {
      reason: "voice_chat_disconnected",
      message: "Voice chat disconnected"
    }
  }

  if (/\btts\b|synth/.test(normalized)) {
    return {
      reason: "voice_chat_tts_error",
      message: message || "Voice chat TTS error"
    }
  }

  return {
    reason: "voice_chat_error",
    message: message || "Voice chat error"
  }
}
