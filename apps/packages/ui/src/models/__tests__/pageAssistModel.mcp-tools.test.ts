import { beforeEach, describe, expect, it, vi } from "vitest"

import { pageAssistModel } from "../index"
import { tldwModels } from "@/services/tldw"
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
    vi.mocked(tldwModels.getModel).mockResolvedValue({ capabilities: ["tools"] })
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
    expect(chat.extraHeaders).toEqual({
      "X-TLDW-Loop-Compat": "1"
    })
    expect(chat.chatDebugMetadata).toMatchObject({
      toolChoice: "auto",
      toolCounts: {
        discovered: 1,
        executable: 1,
        disabled: 0,
        colliding: 0,
        chatEnabled: 1
      }
    })
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
    expect(chat.chatDebugMetadata?.toolOmissionReason).toBe(
      "no_enabled_executable_tools"
    )
  })

  it("omits tools and loop compatibility when the selected model does not support tools", async () => {
    vi.mocked(tldwModels.getModel).mockResolvedValueOnce({ capabilities: [] })
    useMcpToolsStore.setState({
      chatTools: buildResolvedTools([
        {
          name: "notes.search",
          description: "Search notes",
          canExecute: true
        }
      ])
    })

    const chat = await pageAssistModel({ model: "plain-model" })

    expect(chat.toolChoice).toBeUndefined()
    expect(chat.tools).toBeUndefined()
    expect(chat.extraHeaders).toBeUndefined()
    expect(chat.chatDebugMetadata?.toolOmissionReason).toBe(
      "model_lacks_tool_capability"
    )
  })

  it("omits tools and loop compatibility when MCP is unhealthy", async () => {
    useMcpToolsStore.setState({
      healthState: "unhealthy",
      chatTools: buildResolvedTools([
        {
          name: "notes.search",
          description: "Search notes",
          canExecute: true
        }
      ])
    })

    const chat = await pageAssistModel({ model: "tool-model" })

    expect(chat.toolChoice).toBeUndefined()
    expect(chat.tools).toBeUndefined()
    expect(chat.extraHeaders).toBeUndefined()
    expect(chat.chatDebugMetadata?.toolOmissionReason).toBe("mcp_unhealthy")
  })

  it("omits tools and loop compatibility when tool choice is none", async () => {
    useStoreMessageOption.setState({ toolChoice: "none" })
    useMcpToolsStore.setState({
      chatTools: buildResolvedTools([
        {
          name: "notes.search",
          description: "Search notes",
          canExecute: true
        }
      ])
    })

    const chat = await pageAssistModel({ model: "tool-model" })

    expect(chat.toolChoice).toBeUndefined()
    expect(chat.tools).toBeUndefined()
    expect(chat.extraHeaders).toBeUndefined()
    expect(chat.chatDebugMetadata?.toolOmissionReason).toBe("tool_choice_none")
  })

  it("omits collision-only tools and reports the resolver counts", async () => {
    const collisionState = buildChatToolFilterState({
      tools: [
        { name: "docs.search", canExecute: true },
        { name: "docs_search", canExecute: true }
      ]
    })
    useMcpToolsStore.setState({
      tools: collisionState.availableTools.map((tool) => tool.tool as any),
      chatTools: collisionState.chatTools,
      toolCounts: collisionState.counts
    })

    const chat = await pageAssistModel({ model: "tool-model" })

    expect(chat.toolChoice).toBeUndefined()
    expect(chat.tools).toBeUndefined()
    expect(chat.extraHeaders).toBeUndefined()
    expect(chat.chatDebugMetadata).toMatchObject({
      toolOmissionReason: "no_enabled_executable_tools",
      toolCounts: {
        discovered: 2,
        executable: 2,
        disabled: 0,
        colliding: 2,
        chatEnabled: 0
      }
    })
  })
})
