import { describe, expect, it } from "vitest"

import {
  buildSidepanelChatWebUiHandoffUrl,
  decodeSidepanelChatWebUiHandoff,
  resolveSidepanelChatWebUiBaseUrl,
  SIDEPANEL_CHAT_WEBUI_HANDOFF_PARAM,
  SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE
} from "@/services/tldw/sidepanel-chat-webui-handoff"
import { SETTINGS_SERVER_CHAT_ID_PARAM } from "@/utils/settings-return"

describe("sidepanel chat WebUI handoff", () => {
  it("builds a real WebUI /chat URL and preserves draft/context in the handoff", () => {
    const url = new URL(
      buildSidepanelChatWebUiHandoffUrl({
        config: {
          serverUrl: "http://127.0.0.1:8000",
          webUiUrl: "http://127.0.0.1:8080/"
        },
        payload: {
          source: SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE,
          createdAt: 1_762_000_000_000,
          draft: "Summarize the active page",
          serverChatId: "server-chat-123",
          historyId: "local-history-456",
          chatMode: "rag",
          webSearch: true,
          toolChoice: "auto",
          selectedModel: "openai/gpt-4o-mini",
          selectedSystemPrompt: "prompt-1",
          selectedQuickPrompt: "quick-1",
          temporaryChat: false,
          useOCR: true,
          title: "Research tab"
        }
      })
    )

    expect(url.origin).toBe("http://127.0.0.1:8080")
    expect(url.pathname).toBe("/chat")
    expect(url.href).not.toContain("/options.html")
    expect(url.searchParams.get(SETTINGS_SERVER_CHAT_ID_PARAM)).toBe(
      "server-chat-123"
    )

    const decoded = decodeSidepanelChatWebUiHandoff(
      url.searchParams.get(SIDEPANEL_CHAT_WEBUI_HANDOFF_PARAM)
    )
    expect(decoded).toMatchObject({
      source: SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE,
      draft: "Summarize the active page",
      serverChatId: "server-chat-123",
      historyId: "local-history-456",
      chatMode: "rag",
      webSearch: true,
      toolChoice: "auto",
      selectedModel: "openai/gpt-4o-mini",
      selectedSystemPrompt: "prompt-1",
      selectedQuickPrompt: "quick-1",
      temporaryChat: false,
      useOCR: true,
      title: "Research tab"
    })
  })

  it("infers the documented local WebUI port from the local API URL", () => {
    expect(
      resolveSidepanelChatWebUiBaseUrl({
        serverUrl: "http://127.0.0.1:8000"
      })
    ).toBe("http://127.0.0.1:8080")
  })
})
