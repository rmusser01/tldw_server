import { beforeEach, describe, expect, it, vi } from "vitest"

import { pageAssistModel } from "../index"
import { tldwModels, tldwChat, type ModelInfo } from "@/services/tldw"
import { useMcpToolsStore } from "@/store/mcp-tools"
import { useStoreChatModelSettings } from "@/store/model"
import { useStoreMessageOption } from "@/store/option"
import { buildChatToolFilterState } from "@/utils/chat-tools"
import { getChatTurnIdentitySupport } from "@/services/tldw/server-capabilities"
import { getAllDefaultModelSettings, getModelSettings } from "@/services/model-settings"

vi.mock("@/services/tldw/server-capabilities", () => ({
  getServerCapabilities: vi.fn(async () => ({ hasChatTurnIdentity: true })),
  getChatTurnIdentitySupport: vi.fn(async () => true)
}))

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

vi.mock("@/utils/resolve-api-provider", async () => ({
  ...await vi.importActual<typeof import("@/utils/resolve-api-provider")>("@/utils/resolve-api-provider"),
  resolveApiProviderForModel: vi.fn(async () => "openai")
}))

const buildResolvedTools = (tools: Record<string, unknown>[]) =>
  buildChatToolFilterState({ tools }).chatTools

const modelInfo = (capabilities: string[] = ["tools"]): ModelInfo => ({
  id: "tool-model",
  name: "Tool Model",
  provider: "openai",
  type: "chat",
  capabilities
})

const durableScope = {
  config: { serverUrl: "https://research-one.test", authMode: "multi-user" as const },
  userId: 42
}

describe("pageAssistModel MCP tools", () => {
  beforeEach(() => {
    vi.mocked(getModelSettings).mockResolvedValue({})
    vi.mocked(getAllDefaultModelSettings).mockResolvedValue({})
    vi.mocked(getChatTurnIdentitySupport).mockResolvedValue(true)
    vi.mocked(tldwModels.getModel).mockResolvedValue(modelInfo(["tools"]))
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

  it("persists the first saved conversation without requiring an existing server id", async () => {
    useStoreMessageOption.setState({ temporaryChat: false, serverChatId: null })
    const chat = await pageAssistModel({ model: "tool-model" })
    expect(chat.saveToDb).toBe(true)
  })

  it("forwards failed-turn retry intent through the real model without changing messages", async () => {
    vi.mocked(tldwChat.streamMessage).mockImplementation(async function* () { yield "Recovered" })
    const chat = await pageAssistModel({ model: "tool-model", conversationId: "saved", saveToDb: true, retryFailedTurn: true, clientMessageId: "local-user" })
    for await (const _token of await chat.stream([])) { /* consume */ }
    expect(vi.mocked(tldwChat.streamMessage).mock.calls.at(-1)?.[1]).toMatchObject({ retryFailedTurn: true, clientMessageId: "local-user", conversationId: "saved" })
    vi.mocked(tldwChat.sendMessage).mockResolvedValue("Recovered")
    await chat.invoke([])
    expect(vi.mocked(tldwChat.sendMessage).mock.calls.at(-1)?.[1]).toMatchObject({ retryFailedTurn: true, clientMessageId: "local-user", conversationId: "saved" })
  })

  it("keeps temporary conversations unpersisted even with a stale server id", async () => {
    useStoreMessageOption.setState({ temporaryChat: true, serverChatId: "old-chat" })
    const chat = await pageAssistModel({ model: "tool-model" })
    expect({ saved: chat.saveToDb, id: chat.conversationId }).toEqual({ saved: false, id: undefined })
  })

  it.each([false, undefined, "true", 1])("rejects durable turns without explicit identity support (%s)", async (flag) => {
    vi.mocked(getChatTurnIdentitySupport).mockResolvedValue(flag as boolean)
    await expect(pageAssistModel({
      model: "tool-model", saveToDb: true, conversationId: "workspace-chat",
      requestScope: durableScope,
      tldwTurn: { user_message_id: "12803947-a1f4-4c49-b6eb-bf4218765f6a" }, originalUserMessage: "Question"
    })).rejects.toThrow(/durable.*turn/i)
  })

  it("rejects a durable turn without a captured server scope", async () => {
    await expect(pageAssistModel({
      model: "tool-model", saveToDb: true, conversationId: "workspace-chat",
      tldwTurn: { user_message_id: "12803947-a1f4-4c49-b6eb-bf4218765f6a" }, originalUserMessage: "Question"
    })).rejects.toThrow(/scope/i)
  })

  it("keeps the captured scope on a supported durable model", async () => {
    const model = await pageAssistModel({
      model: "tool-model", saveToDb: true, conversationId: "workspace-chat",
      requestScope: durableScope,
      tldwTurn: { user_message_id: "12803947-a1f4-4c49-b6eb-bf4218765f6a" }, originalUserMessage: "Question"
    })
    expect(model.requestScope).toBe(durableScope)
    expect(getChatTurnIdentitySupport).toHaveBeenCalledWith(durableScope, undefined)
  })

  it("does not bind a durable turn to a background global conversation", async () => {
    useStoreMessageOption.setState({ serverChatId: "unrelated-chat", temporaryChat: false })
    await expect(pageAssistModel({
      model: "tool-model", saveToDb: true,
      tldwTurn: { user_message_id: "12803947-a1f4-4c49-b6eb-bf4218765f6a" }, originalUserMessage: "Question"
    })).rejects.toThrow(/conversation/i)
  })

  it.each([false, true])("uses captured generation settings only when supplied (%s)", async (captured) => {
    useStoreChatModelSettings.getState().updateSettings({
      temperature: 0.9, topP: 0.95, numPredict: 128, reasoningEffort: "high",
      historyMessageLimit: 36, historyMessageOrder: "newest", slashCommandInjectionMode: "live",
      extraHeaders: '{"X-Request":"live"}', extraBody: '{"request_label":"live"}'
    })
    const chat = await pageAssistModel({
      model: "tool-model",
      ...(captured ? { generationSettings: {
        temperature: 0.42, topP: 0.8, numPredict: 96, reasoningEffort: "low",
        historyMessageLimit: 12, historyMessageOrder: "oldest", slashCommandInjectionMode: "captured",
        extraHeaders: '{"X-Request":"captured"}', extraBody: '{"request_label":"captured"}'
      } } : {})
    })
    expect(chat).toMatchObject(captured ? {
      temperature: 0.42, topP: 0.8, maxTokens: 96, reasoningEffort: "low",
      historyMessageLimit: 12, historyMessageOrder: "oldest", slashCommandInjectionMode: "captured",
      extraHeaders: { "X-Request": "captured" }, extraBody: { request_label: "captured" }
    } : {
      temperature: 0.9, topP: 0.95, maxTokens: 128, reasoningEffort: "high",
      historyMessageLimit: 36, historyMessageOrder: "newest", slashCommandInjectionMode: "live",
      extraHeaders: { "X-Request": "live" }, extraBody: { request_label: "live" }
    })
  })

  it.each([0, 0.2])("keeps per-model settings ahead of the captured defaults (temperature=%s)", async (temperature) => {
    useStoreChatModelSettings.getState().setTemperature(0.9)
    vi.mocked(getModelSettings).mockResolvedValueOnce({ temperature, topP: 0.5, numPredict: 64, reasoningEffort: "high" })
    const chat = await pageAssistModel({ model: "tool-model", generationSettings: {
      temperature: 0.42, topP: 0.8, numPredict: 96, reasoningEffort: "low"
    } })
    expect(chat).toMatchObject({ temperature, topP: 0.5, maxTokens: 64, reasoningEffort: "high" })
  })

  it("uses configured defaults instead of live values absent from the captured settings", async () => {
    useStoreChatModelSettings.getState().setTemperature(0.9)
    vi.mocked(getAllDefaultModelSettings).mockResolvedValueOnce({ temperature: 0.17 })
    const chat = await pageAssistModel({ model: "tool-model", generationSettings: {} })
    expect(chat.temperature).toBe(0.17)
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

  it("preserves the captured Service Prompt request scope on the model", async () => {
    const requestScope = {
      config: {
        serverUrl: "https://research-one.test",
        authMode: "multi-user" as const
      },
      userId: 42
    }

    const chat = await pageAssistModel({ model: "tool-model", requestScope })

    expect(chat.requestScope).toBe(requestScope)
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
    vi.mocked(tldwModels.getModel).mockResolvedValueOnce(modelInfo([]))
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

  it("reports absent MCP when the MCP capability is unavailable", async () => {
    useMcpToolsStore.setState({
      healthState: "unavailable",
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
    expect(chat.chatDebugMetadata?.toolOmissionReason).toBe("mcp_absent")
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

it("freezes resolved model options into a stateless prepared body before ambient state changes", async () => {
  useStoreChatModelSettings.getState().reset()
  useStoreChatModelSettings.getState().setTemperature(0.23)
  useStoreChatModelSettings.setState({ slashCommandInjectionMode: "preface" })
  useStoreMessageOption.setState({
    serverChatId: "ambient-chat",
    temporaryChat: false,
    toolChoice: "none"
  })
  const model = await pageAssistModel({
    model: "tool-model",
    clientManagedHistory: true
  })
  const { HumanMessage } = await import("@/types/messages")
  const body = model.prepareClientManagedRequest([new HumanMessage("question")])
  useStoreChatModelSettings.getState().setTemperature(1.9)
  useStoreMessageOption.setState({ serverChatId: "other-chat" })
  expect(body).toMatchObject({
    temperature: 0.23,
    slash_command_injection_mode: "preface",
    save_to_db: false,
    messages: [{ role: "user", content: "question" }]
  })
  expect(body.conversation_id).toBeUndefined()
  expect(body.history_message_limit).toBeUndefined()
  expect(body.history_message_order).toBeUndefined()
})

it("rejects ambiguous credential-bearing provider extensions before creating provenance", async () => {
  const model = await pageAssistModel({
    model: "tool-model",
    clientManagedHistory: true,
    extraBody: '{"api_key":"private-token"}'
  })
  const { HumanMessage } = await import("@/types/messages")
  expect(() =>
    model.prepareClientManagedRequest([new HumanMessage("question")])
  ).toThrow("unsupported_history_custom_provider_body")
})
