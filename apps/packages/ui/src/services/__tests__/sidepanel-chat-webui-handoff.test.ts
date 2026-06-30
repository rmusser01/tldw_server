import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  buildSidepanelChatWebUiHandoffUrl,
  decodeSidepanelChatWebUiHandoff,
  encodeSidepanelChatWebUiHandoff,
  resolveSidepanelChatWebUiBaseUrl,
  SIDEPANEL_CHAT_WEBUI_HANDOFF_MAX_AGE_MS,
  SIDEPANEL_CHAT_WEBUI_HANDOFF_PARAM,
  SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE
} from "@/services/tldw/sidepanel-chat-webui-handoff"

describe("sidepanel chat WebUI handoff", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-06-01T12:00:00.000Z"))
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  const getFragmentHandoff = (url: URL) =>
    new URLSearchParams(url.hash.slice(1)).get(
      SIDEPANEL_CHAT_WEBUI_HANDOFF_PARAM
    )

  it("builds a real WebUI /chat URL and preserves draft/context in a fragment handoff", () => {
    const url = new URL(
      buildSidepanelChatWebUiHandoffUrl({
        config: {
          serverUrl: "http://127.0.0.1:8000",
          webUiUrl: "http://127.0.0.1:8080/"
        },
        payload: {
          source: SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE,
          createdAt: Date.now(),
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
    expect(url.searchParams.has(SIDEPANEL_CHAT_WEBUI_HANDOFF_PARAM)).toBe(false)
    expect(url.search).toBe("")

    const decoded = decodeSidepanelChatWebUiHandoff(getFragmentHandoff(url))
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

  it("preserves explicit WebUI subpaths when building the /chat route", () => {
    const url = new URL(
      buildSidepanelChatWebUiHandoffUrl({
        config: {
          webUiUrl: "https://example.test/tldw/"
        },
        payload: {
          source: SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE,
          createdAt: Date.now()
        }
      })
    )

    expect(url.origin).toBe("https://example.test")
    expect(url.pathname).toBe("/tldw/chat")
  })

  it.each([
    "file:///tmp/app.html",
    "chrome-extension://extension-id/options.html",
    "about:blank",
    "data:text/plain,hello"
  ])("ignores non-http WebUI config values and still returns a safe URL: %s", (webUiUrl) => {
    const url = new URL(
      buildSidepanelChatWebUiHandoffUrl({
        config: { webUiUrl },
        payload: {
          source: SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE,
          createdAt: Date.now()
        }
      })
    )

    expect(["http:", "https:"]).toContain(url.protocol)
    expect(url.pathname).toBe("/chat")
    expect(getFragmentHandoff(url)).toBeTruthy()
  })

  it("infers the documented local WebUI port from the local API URL", () => {
    expect(
      resolveSidepanelChatWebUiBaseUrl({
        serverUrl: "http://127.0.0.1:8000"
      })
    ).toBe("http://127.0.0.1:8080")
  })

  it("expires stale handoff payloads", () => {
    const encoded = encodeSidepanelChatWebUiHandoff({
      source: SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE,
      createdAt: Date.now() - SIDEPANEL_CHAT_WEBUI_HANDOFF_MAX_AGE_MS - 1,
      draft: "old draft"
    })

    expect(decodeSidepanelChatWebUiHandoff(encoded)).toBeNull()
  })

  it("preserves explicit null prompt clears when decoding", () => {
    const encoded = encodeSidepanelChatWebUiHandoff({
      source: SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE,
      createdAt: Date.now(),
      selectedSystemPrompt: null,
      selectedQuickPrompt: ""
    })

    expect(decodeSidepanelChatWebUiHandoff(encoded)).toMatchObject({
      selectedSystemPrompt: null,
      selectedQuickPrompt: null
    })
  })
})
