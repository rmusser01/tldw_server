import { beforeEach, describe, expect, it, vi } from "vitest"

import { pageAssistModel } from "../index"
import { useMcpToolsStore } from "@/store/mcp-tools"
import { useStoreChatModelSettings } from "@/store/model"
import { useStoreMessageOption } from "@/store/option"
import { buildChatToolFilterState } from "@/utils/chat-tools"

vi.mock("@/services/model-settings", () => ({
  getAllDefaultModelSettings: vi.fn(async () => ({})),
  getModelSettings: vi.fn(async () => ({}))
}))

vi.mock("@/services/tldw-server", () => ({
  getDefaultApiProvider: vi.fn(async () => "openai")
}))

vi.mock("@/services/tldw", () => ({
  tldwModels: {
    getModel: vi.fn(async () => ({ capabilities: ["tools"] }))
  },
  tldwChat: {
    sendMessage: vi.fn(),
    streamMessage: vi.fn()
  }
}))

vi.mock("@/utils/resolve-api-provider", () => ({
  resolveApiProviderForModel: vi.fn(async () => "openai")
}))

const buildResolvedTools = (tools: Record<string, unknown>[]) =>
  buildChatToolFilterState({ tools }).chatTools

describe("pageAssistModel MCP tools", () => {
  beforeEach(() => {
    useStoreChatModelSettings.getState().reset()
    useStoreMessageOption.setState({
      toolChoice: "auto",
      serverChatId: null,
      temporaryChat: true
    })
    useMcpToolsStore.setState({
      tools: [],
      discoveredTools: [],
      availableTools: [],
      chatTools: [],
      healthState: "healthy",
      toolsLoading: false,
      disabledToolPreferences: { version: 1, scopes: {} },
      activeToolPreferenceScope: "default",
      disabledToolNames: [],
      collisionToolNames: [],
      toolCounts: {
        discovered: 0,
        executable: 0,
        disabled: 0,
        colliding: 0,
        chatEnabled: 0
      }
    })
  })

  it("uses stored chatTools instead of all executable MCP tools", async () => {
    const notesTool = {
      name: "notes.search",
      description: "Search notes",
      parameters: { type: "object", properties: { q: { type: "string" } } },
      canExecute: true
    }
    const slidesTool = {
      name: "slides.list",
      description: "List slides",
      canExecute: true
    }

    useMcpToolsStore.setState({
      tools: [notesTool, slidesTool],
      chatTools: buildResolvedTools([notesTool])
    })

    const chat = await pageAssistModel({ model: "tool-model" })

    expect(chat.toolChoice).toBe("auto")
    expect(chat.tools).toEqual([
      {
        type: "function",
        function: {
          name: "notes_search",
          description: "Search notes",
          parameters: {
            type: "object",
            properties: { q: { type: "string" } }
          }
        }
      }
    ])
  })

  it("omits tool choice and tools when no chat tools remain", async () => {
    useMcpToolsStore.setState({
      tools: [
        {
          name: "notes.search",
          description: "Search notes",
          canExecute: true
        }
      ],
      chatTools: []
    })

    const chat = await pageAssistModel({ model: "tool-model" })

    expect(chat.toolChoice).toBeUndefined()
    expect(chat.tools).toBeUndefined()
    expect(chat.extraHeaders).toBeUndefined()
  })
})
