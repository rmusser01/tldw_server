import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readSource = (relativePath: string) =>
  fs.readFileSync(path.resolve(__dirname, relativePath), "utf8")

describe("voice conversation cross-surface contract", () => {
  it("keeps both surfaces on the shared availability resolver and message copy", () => {
    const playgroundFormSource = readSource("../PlaygroundForm.tsx")
    const sidepanelFormSource = readSource("../../../Sidepanel/Chat/form.tsx")
    const playgroundVoiceChatSource = readSource("../hooks/usePlaygroundVoiceChat.ts")

    expect(playgroundFormSource).toContain("shouldProbeVoiceConversationAudioHealth(")
    expect(sidepanelFormSource).toContain("shouldProbeVoiceConversationAudioHealth(")
    expect(playgroundFormSource).toContain("resolveVoiceConversationAvailability(")
    expect(sidepanelFormSource).toContain("resolveVoiceConversationAvailability(")
    expect(playgroundVoiceChatSource).toContain("voiceConversationAvailability.message")
    expect(sidepanelFormSource).toContain("voiceConversationAvailability.message")
    expect(playgroundFormSource).toContain("voiceConversationAvailability.available")
    expect(sidepanelFormSource).toContain("voiceConversationAvailability.available")
    expect(playgroundFormSource).toContain("isVoiceConversationAuthReady(")
    expect(sidepanelFormSource).toContain("isVoiceConversationAuthReady(")
    expect(playgroundFormSource).toContain("canonicalConnectionConfig?.authSource")
    expect(sidepanelFormSource).toContain("canonicalConnectionConfig?.authSource")
  })

  it("routes voice chat runtime failures through normalized interruption handling", () => {
    const playgroundFormSource = readSource("../PlaygroundForm.tsx")
    const sidepanelFormSource = readSource("../../../Sidepanel/Chat/form.tsx")

    expect(playgroundFormSource).toContain("normalizeVoiceConversationRuntimeError(")
    expect(sidepanelFormSource).toContain("normalizeVoiceConversationRuntimeError(")
    expect(playgroundFormSource).toContain("voiceChatMessages.failTurn(")
    expect(sidepanelFormSource).toContain("voiceChatMessages.failTurn(")
  })
})
