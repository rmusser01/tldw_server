import { beforeEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"
import { useStoreMessage } from "@/store"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { useStoreChatModelSettings } from "@/store/model"

const config = (user: string, revision = 1) => ({
  serverUrl: "http://localhost:8000", authMode: "multi-user", authSource: "manual",
  accessToken: `test.${btoa(JSON.stringify({ sub: user, revision }))}.signature`
})

describe("Chat state when its route is unmounted", () => {
  beforeEach(() => {
    const settings = useStoreChatModelSettings.getState()
    settings.reset()
    settings.updateSettings({ systemPrompt: "ALICE GLOBAL SECRET", systemPromptTemplateId: "alice-global", temperature: 0.4 })
    settings.setActiveSettingsScope("llama:one")
    settings.updateSettings({ systemPrompt: "ALICE MODEL SECRET", systemPromptTemplateId: "alice-model", topK: 12 })
    settings.setActiveSettingsScope("llama:two")
    const messages = [{ role: "user" as const, isBot: false, name: "Alice", message: "ALICE PRIVATE", sources: [] }]
    useStoreMessageOption.setState({ messages, history: [{ role: "user", content: "ALICE PRIVATE" }],
      historyId: "alice-history", serverChatId: "alice-chat", serverChatTitle: "Alice private title",
      selectedModel: "keep-model", contextFiles: [], replyTarget: null })
    useStoreMessage.setState({ messages, history: [{ role: "user", content: "ALICE PRIVATE" }], historyId: "alice-history" })
    usePlaygroundSessionStore.getState().saveSession({ serverChatId: "alice-chat", scopeKey: "alice" })
  })

  const expectCleared = () => {
    expect(useStoreMessageOption.getState()).toMatchObject({ messages: [], history: [], historyId: null,
      serverChatId: null, serverChatTitle: null, selectedModel: "keep-model" })
    expect(useStoreMessage.getState()).toMatchObject({ messages: [], history: [], historyId: null })
    expect(usePlaygroundSessionStore.getState().serverChatId).toBeNull()
    const settings = useStoreChatModelSettings.getState()
    for (const scope of ["llama:one", "llama:two"]) {
      expect(settings.getEffectiveSettings(scope)).toMatchObject({
        systemPrompt: undefined, systemPromptTemplateId: undefined, temperature: 0.4
      })
    }
    expect(settings.getEffectiveSettings("llama:one").topK).toBe(12)
  }

  it("clears both live surfaces and the recovery selection on logout without a mounted Chat hook", () => {
    window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } }))
    expectCleared()
  })

  it("clears on an in-tab server, organization or account change", () => {
    window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } }))
    expectCleared()
  })

  it("clears when another tab switches Alice to Bob", () => {
    window.dispatchEvent(new StorageEvent("storage", { key: "tldwConfig",
      oldValue: JSON.stringify(config("alice")), newValue: JSON.stringify(config("bob")) }))
    expectCleared()
  })

  it("preserves the current conversation during same-account token rotation", () => {
    window.dispatchEvent(new StorageEvent("storage", { key: "tldwConfig",
      oldValue: JSON.stringify(config("alice")), newValue: JSON.stringify(config("alice", 2)) }))
    window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: false } }))
    expect(useStoreMessageOption.getState().messages[0].message).toBe("ALICE PRIVATE")
    expect(usePlaygroundSessionStore.getState().serverChatId).toBe("alice-chat")
    expect(useStoreChatModelSettings.getState().getEffectiveSettings("llama:one").systemPrompt).toBe("ALICE MODEL SECRET")
  })
})
