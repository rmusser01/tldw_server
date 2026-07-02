import { describe, expect, it } from "vitest"

import { buildVoiceConversationPreflight } from "@/services/tldw/voice-conversation"

describe("buildVoiceConversationPreflight (TASK-12106 auth-out-of-url)", () => {
  const baseInput = {
    serverUrl: "http://127.0.0.1:8000",
    token: "secret-token-123",
    requestedModel: "",
    ttsProvider: "tldw",
    tldwTtsModel: "kokoro",
    tldwTtsVoice: "af_heart",
    tldwTtsSpeed: 1,
    tldwTtsResponseFormat: "mp3",
    voiceChatTtsMode: "stream" as const,
    resolveProvider: () => undefined
  }

  it("keeps the auth token out of the websocket url", async () => {
    const preflight = await buildVoiceConversationPreflight(baseInput)

    expect(preflight.websocketUrl).toContain("/api/v1/audio/chat/stream")
    expect(preflight.websocketUrl).not.toContain("token")
    expect(preflight.websocketUrl).not.toContain("secret-token-123")
    expect(preflight.websocketUrl).not.toContain("?")
  })

  it("still fails fast when the token is missing", async () => {
    await expect(
      buildVoiceConversationPreflight({ ...baseInput, token: "" })
    ).rejects.toThrow(/Not authenticated/i)
  })
})
