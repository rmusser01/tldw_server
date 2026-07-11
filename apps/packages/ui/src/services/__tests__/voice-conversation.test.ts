import { describe, expect, it, vi } from "vitest"

import {
  buildAuthenticatedAudioWebSocketUrl,
  buildVoiceConversationPreflight,
  isVoiceConversationAuthReady,
  normalizeVoiceConversationRuntimeError,
  resolveVoiceConversationAvailability,
  resolveVoiceConversationTtsConfig,
  shouldProbeVoiceConversationAudioHealth
} from "@/services/tldw/voice-conversation"

type VoiceConversationTtsResolution = ReturnType<
  typeof resolveVoiceConversationTtsConfig
>
type VoiceConversationTtsConfig = Extract<
  VoiceConversationTtsResolution,
  { ok: true }
>["value"]
type VoiceConversationTtsErrorReason = Extract<
  VoiceConversationTtsResolution,
  { ok: false }
>["reason"]

const getTtsConfigValue = (
  result: VoiceConversationTtsResolution
): VoiceConversationTtsConfig => {
  expect(result.ok).toBe(true)
  if (!("value" in result)) {
    const reason = "reason" in result ? result.reason : "unknown"
    throw new Error(`Expected TTS config to be valid, got ${reason}`)
  }
  return result.value
}

const getTtsConfigErrorReason = (
  result: VoiceConversationTtsResolution
): VoiceConversationTtsErrorReason => {
  expect(result.ok).toBe(false)
  if (!("reason" in result)) {
    throw new Error("Expected TTS config to be invalid")
  }
  return result.reason
}

describe("voice conversation contract", () => {
  it("lets sidepanel and playground callers start cookie-authenticated audio and chat sockets", async () => {
    const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    const config = {
      serverUrl: window.location.origin,
      authMode: "single-user" as const,
      authSource: "cookie-session" as const
    }

    try {
      expect(isVoiceConversationAuthReady(config)).toBe(true)
      expect(
        resolveVoiceConversationAvailability({
          isConnectionReady: true,
          hasVoiceConversationTransport: true,
          authReady: isVoiceConversationAuthReady(config),
          sttHealthState: "healthy",
          ttsHealthState: "healthy",
          selectedModel: "",
          allowBackendDefaultModel: true,
          ttsConfigReady: true
        }).available
      ).toBe(true)
      await expect(
        buildVoiceConversationPreflight({
          ...config,
          token: "",
          requestedModel: "",
          ttsProvider: "browser",
          tldwTtsModel: "kokoro",
          tldwTtsVoice: "af_heart",
          tldwTtsSpeed: 1,
          tldwTtsResponseFormat: "mp3",
          resolveProvider: vi.fn()
        })
      ).resolves.toMatchObject({
        websocketUrl: expect.stringMatching(/\/api\/v1\/audio\/chat\/stream$/)
      })
    } finally {
      if (originalDeploymentMode === undefined) {
        delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
      } else {
        process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
      }
    }
  })

  it("builds a secret-free page-origin websocket for cookie sessions", async () => {
    const originalWindow = globalThis.window
    const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    Object.defineProperty(globalThis, "window", {
      configurable: true,
      value: {
        location: {
          origin: "https://webui.example.test",
          protocol: "https:"
        }
      }
    })

    try {
      const result = await buildVoiceConversationPreflight({
        serverUrl: "https://remote.example.test",
        token: "stale-token",
        authMode: "single-user",
        authSource: "cookie-session",
        requestedModel: "",
        ttsProvider: "browser",
        tldwTtsModel: "kokoro",
        tldwTtsVoice: "af_heart",
        tldwTtsSpeed: 1,
        tldwTtsResponseFormat: "mp3",
        resolveProvider: vi.fn()
      })

      expect(result.websocketUrl).toBe(
        "wss://webui.example.test/api/v1/audio/chat/stream"
      )
      expect(
        buildAuthenticatedAudioWebSocketUrl({
          serverUrl: "https://remote.example.test",
          token: "stale-token",
          authMode: "single-user",
          authSource: "cookie-session",
          path: "/api/v1/audio/stream/tts"
        })
      ).toBe("wss://webui.example.test/api/v1/audio/stream/tts")
    } finally {
      if (originalDeploymentMode === undefined) {
        delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
      } else {
        process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
      }
      Object.defineProperty(globalThis, "window", {
        configurable: true,
        value: originalWindow
      })
    }
  })

  it("keeps voice conversation unavailable when only broad audio flags exist", () => {
    const result = resolveVoiceConversationAvailability({
      isConnectionReady: true,
      hasVoiceConversationTransport: false,
      authReady: true,
      sttHealthState: "healthy",
      ttsHealthState: "healthy",
      selectedModel: "gpt-4o-mini",
      allowBackendDefaultModel: false,
      ttsConfigReady: true
    })

    expect(result.available).toBe(false)
    expect(result.reason).toBe("transport_missing")
    expect(result.message).toBe(
      "playground:voiceChat.unavailableReasons.transportMissing"
    )
  })

  it("maps browser TTS to server-backed tldw settings for voice conversation", () => {
    const result = resolveVoiceConversationTtsConfig({
      ttsProvider: "browser",
      tldwTtsModel: "kokoro",
      tldwTtsVoice: "af_heart",
      tldwTtsSpeed: 1.25,
      tldwTtsResponseFormat: "mp3",
      openAITTSModel: "tts-1",
      openAITTSVoice: "alloy",
      elevenLabsModel: "",
      elevenLabsVoiceId: "",
      speechPlaybackSpeed: 1,
      voiceChatTtsMode: "stream"
    })

    const tts = getTtsConfigValue(result)
    expect(tts.model).toBe("kokoro")
    expect(tts.voice).toBe("af_heart")
    expect(tts.speed).toBe(1.25)
    expect(tts.format).toBe("mp3")
    expect(tts.provider).toBeUndefined()
  })

  it("preserves explicit tldw provider on the voice conversation wire shape", () => {
    const result = resolveVoiceConversationTtsConfig({
      ttsProvider: "tldw",
      tldwTtsModel: "kokoro",
      tldwTtsVoice: "af_heart",
      tldwTtsSpeed: 1.25,
      tldwTtsResponseFormat: "mp3",
      openAITTSModel: "tts-1",
      openAITTSVoice: "alloy",
      elevenLabsModel: "",
      elevenLabsVoiceId: "",
      speechPlaybackSpeed: 1,
      voiceChatTtsMode: "stream"
    })

    const tts = getTtsConfigValue(result)
    expect(tts.model).toBe("kokoro")
    expect(tts.voice).toBe("af_heart")
    expect(tts.speed).toBe(1.25)
    expect(tts.format).toBe("mp3")
    expect(tts.provider).toBe("tldw")
  })

  it("rejects voice conversation availability when STT health is unhealthy", () => {
    const result = resolveVoiceConversationAvailability({
      isConnectionReady: true,
      hasVoiceConversationTransport: true,
      authReady: true,
      sttHealthState: "unhealthy",
      ttsHealthState: "healthy",
      selectedModel: "gpt-4o-mini",
      allowBackendDefaultModel: false,
      ttsConfigReady: true
    })

    expect(result.available).toBe(false)
    expect(result.reason).toBe("audio_unhealthy")
    expect(result.message).toBe(
      "playground:voiceChat.unavailableReasons.audioUnhealthy"
    )
  })

  it("probes audio health as soon as server voice controls are relevant", () => {
    expect(
      shouldProbeVoiceConversationAudioHealth({
        isConnectionReady: true,
        hasServerVoiceChat: true,
        hasServerStt: false,
        optionalAudioHealthEnabled: false
      })
    ).toBe(true)

    expect(
      shouldProbeVoiceConversationAudioHealth({
        isConnectionReady: true,
        hasServerVoiceChat: false,
        hasServerStt: true,
        optionalAudioHealthEnabled: false
      })
    ).toBe(true)
  })

  it("keeps audio health idle before connection unless the panel is explicitly engaged", () => {
    expect(
      shouldProbeVoiceConversationAudioHealth({
        isConnectionReady: false,
        hasServerVoiceChat: true,
        hasServerStt: true,
        optionalAudioHealthEnabled: false
      })
    ).toBe(false)

    expect(
      shouldProbeVoiceConversationAudioHealth({
        isConnectionReady: false,
        hasServerVoiceChat: false,
        hasServerStt: false,
        optionalAudioHealthEnabled: true
      })
    ).toBe(true)
  })

  it("requires explicit OpenAI TTS model and voice when openai is selected", () => {
    const result = resolveVoiceConversationTtsConfig({
      ttsProvider: "openai",
      tldwTtsModel: "kokoro",
      tldwTtsVoice: "af_heart",
      tldwTtsSpeed: 1,
      tldwTtsResponseFormat: "mp3",
      openAITTSModel: "",
      openAITTSVoice: "",
      elevenLabsModel: "",
      elevenLabsVoiceId: "",
      speechPlaybackSpeed: 1,
      voiceChatTtsMode: "stream"
    })

    expect(getTtsConfigErrorReason(result)).toBe("tts_config_missing")
  })

  it("requires explicit ElevenLabs model and voice id when elevenlabs is selected", () => {
    const result = resolveVoiceConversationTtsConfig({
      ttsProvider: "elevenlabs",
      tldwTtsModel: "kokoro",
      tldwTtsVoice: "af_heart",
      tldwTtsSpeed: 1,
      tldwTtsResponseFormat: "mp3",
      openAITTSModel: "tts-1",
      openAITTSVoice: "alloy",
      elevenLabsModel: "",
      elevenLabsVoiceId: "",
      speechPlaybackSpeed: 1,
      voiceChatTtsMode: "stream"
    })

    expect(getTtsConfigErrorReason(result)).toBe("tts_config_missing")
  })

  it("builds preflight with backend defaults when no model is requested", async () => {
    const resolveProvider = vi.fn(async ({ modelId }: { modelId: string }) => `provider:${modelId}`)

    const result = await buildVoiceConversationPreflight({
      serverUrl: "http://127.0.0.1:8000/",
      token: "secret-token",
      requestedModel: "   ",
      ttsProvider: "browser",
      tldwTtsModel: "kokoro",
      tldwTtsVoice: "af_heart",
      tldwTtsSpeed: 1,
      tldwTtsResponseFormat: "pcm",
      openAITTSModel: "tts-1",
      openAITTSVoice: "alloy",
      elevenLabsModel: "",
      elevenLabsVoiceId: "",
      speechPlaybackSpeed: 1,
      voiceChatTtsMode: "stream",
      resolveProvider
    })

    // TASK-12106: the token is sent as an {type:"auth"} first frame, never in the URL.
    expect(result.websocketUrl).toBe(
      "ws://127.0.0.1:8000/api/v1/audio/chat/stream"
    )
    expect(result.websocketUrl).not.toContain("secret-token")
    expect(result.llm).toEqual({})
    expect(result.tts).toEqual({
      model: "kokoro",
      voice: "af_heart",
      speed: 1,
      format: "mp3"
    })
    expect(resolveProvider).not.toHaveBeenCalled()
  })

  it("builds preflight with an explicit llm provider when a model is requested", async () => {
    const resolveProvider = vi.fn(async ({ modelId }: { modelId: string }) => `provider:${modelId}`)

    const result = await buildVoiceConversationPreflight({
      serverUrl: "http://127.0.0.1:8000",
      token: "abc123",
      requestedModel: "gpt-4o-mini",
      ttsProvider: "openai",
      tldwTtsModel: "kokoro",
      tldwTtsVoice: "af_heart",
      tldwTtsSpeed: 1,
      tldwTtsResponseFormat: "mp3",
      openAITTSModel: "tts-1",
      openAITTSVoice: "alloy",
      elevenLabsModel: "",
      elevenLabsVoiceId: "",
      speechPlaybackSpeed: 1.5,
      voiceChatTtsMode: "stream",
      resolveProvider
    })

    expect(result.llm).toEqual({
      model: "gpt-4o-mini",
      provider: "provider:gpt-4o-mini"
    })
    expect(result.tts).toEqual({
      provider: "openai",
      model: "tts-1",
      voice: "alloy",
      speed: 1.5,
      format: "mp3"
    })
    expect(resolveProvider).toHaveBeenCalledWith({ modelId: "gpt-4o-mini" })
  })

  it("normalizes disconnect-like runtime errors to a stable disconnected reason", () => {
    expect(normalizeVoiceConversationRuntimeError("websocket disconnected")).toEqual({
      reason: "voice_chat_disconnected",
      message: "Voice chat disconnected"
    })
  })

  it("normalizes TTS-like runtime errors to a stable TTS reason", () => {
    expect(
      normalizeVoiceConversationRuntimeError(new Error("TTS provider failed during synthesis"))
    ).toEqual({
      reason: "voice_chat_tts_error",
      message: "TTS provider failed during synthesis"
    })
  })
})
