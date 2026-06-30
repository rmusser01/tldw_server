import fs from "node:fs"
import path from "node:path"
import { beforeEach, describe, expect, it } from "vitest"
import {
  useSidepanelChatTabsStore,
  type ChatModelSettingsSnapshot,
  type SidepanelChatSnapshot
} from "../sidepanel-chat-tabs"
import { useStoreChatModelSettings } from "../model"

const LLAMA_SETTING_KEYS = [
  "llamaThinkingBudgetTokens",
  "llamaGrammarMode",
  "llamaGrammarId",
  "llamaGrammarInline",
  "llamaGrammarOverride"
] as const

const readSource = (relativePath: string) =>
  fs.readFileSync(path.resolve(__dirname, relativePath), "utf8")

describe("llama.cpp chat model settings state", () => {
  beforeEach(() => {
    useStoreChatModelSettings.getState().reset()
    useSidepanelChatTabsStore.getState().clear()
  })

  it("stores llama.cpp controls in the shared chat model settings store", () => {
    useStoreChatModelSettings.getState().updateSettings({
      llamaGrammarMode: "library",
      llamaGrammarId: "grammar_1",
      llamaGrammarOverride: 'root ::= "override"',
      llamaThinkingBudgetTokens: 64
    })

    const next = useStoreChatModelSettings.getState()

    expect(next.llamaGrammarMode).toBe("library")
    expect(next.llamaGrammarId).toBe("grammar_1")
    expect(next.llamaGrammarOverride).toBe('root ::= "override"')
    expect(next.llamaThinkingBudgetTokens).toBe(64)
  })

  it("round-trips llama.cpp controls through sidepanel snapshots and route allowlists", () => {
    const modelSettings: ChatModelSettingsSnapshot = {
      llamaThinkingBudgetTokens: 32,
      llamaGrammarMode: "inline",
      llamaGrammarId: "grammar_saved",
      llamaGrammarInline: 'root ::= "ok"',
      llamaGrammarOverride: 'root ::= "override"'
    }
    const snapshot: SidepanelChatSnapshot = {
      history: [] as any,
      messages: [],
      chatMode: "normal",
      historyId: null,
      webSearch: false,
      toolChoice: "none",
      selectedModel: "llama.cpp/local-model",
      selectedSystemPrompt: null,
      selectedQuickPrompt: null,
      temporaryChat: false,
      useOCR: false,
      serverChatId: null,
      serverChatState: null,
      serverChatTopic: null,
      serverChatClusterId: null,
      serverChatSource: null,
      serverChatExternalRef: null,
      queuedMessages: [],
      modelSettings
    }

    useSidepanelChatTabsStore.getState().upsertTab({
      id: "tab-1",
      label: "Tab 1",
      historyId: null,
      serverChatId: null,
      serverChatTopic: null,
      updatedAt: 1
    })
    useSidepanelChatTabsStore.getState().setSnapshot("tab-1", snapshot)
    const restored = useSidepanelChatTabsStore.getState().getSnapshot("tab-1")

    expect(restored?.modelSettings.llamaThinkingBudgetTokens).toBe(32)
    expect(restored?.modelSettings.llamaGrammarMode).toBe("inline")
    expect(restored?.modelSettings.llamaGrammarInline).toBe('root ::= "ok"')

    const sharedRouteSource = readSource("../../routes/sidepanel-chat.tsx")
    const extensionRouteSource = readSource(
      "../../../../../tldw-frontend/extension/routes/sidepanel-chat.tsx"
    )

    for (const key of LLAMA_SETTING_KEYS) {
      expect(sharedRouteSource).toContain(`"${key}"`)
      expect(extensionRouteSource).toContain(`"${key}"`)
    }
  })
})
