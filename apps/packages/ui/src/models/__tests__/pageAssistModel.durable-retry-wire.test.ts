import { beforeEach, describe, expect, it, vi } from "vitest"
import type { ChatCompletionRequest } from "@/services/tldw/TldwApiClient"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import type { ChatScope } from "@/types/chat-scope"
import { AIMessage, HumanMessage, SystemMessage } from "@/types/messages"
import { useStoreChatModelSettings } from "@/store/model"
import { pageAssistModel } from "../index"

type WireRequest = ChatCompletionRequest & { metadata?: Record<string, unknown> }

const transport = vi.hoisted(() => ({
  requests: [] as Array<{ body: WireRequest; requestScope?: ServicePromptRequestScope; scope?: ChatScope }>,
  failNext: false
}))

vi.mock("@/services/model-settings", () => ({
  getAllDefaultModelSettings: async () => ({}),
  getModelSettings: async () => ({})
}))
vi.mock("@/services/tldw-server", () => ({ getDefaultApiProvider: async () => "ollama" }))
vi.mock("@/services/tldw/server-capabilities", () => ({ getChatTurnIdentitySupport: async () => true }))
vi.mock("@/services/tldw/TldwApiClient", () => {
  const record = (body: ChatCompletionRequest, options: { requestScope?: ServicePromptRequestScope; scope?: ChatScope }) => {
    transport.requests.push({ body: JSON.parse(JSON.stringify(body)), requestScope: options.requestScope, scope: options.scope })
    if (transport.failNext) {
      transport.failNext = false
      throw Object.assign(new Error("Provider unavailable"), { status: 503 })
    }
  }
  return { tldwClient: {
    initialize: async () => undefined,
    getConfig: async () => null,
    createChatCompletion: async (body: ChatCompletionRequest, options: { requestScope?: ServicePromptRequestScope }) => {
      record(body, options)
      return new Response(JSON.stringify({ choices: [{ message: { content: "Recovered answer" } }] }))
    },
    streamChatCompletion: async function* (body: ChatCompletionRequest, options: { requestScope?: ServicePromptRequestScope; scope?: ChatScope }) {
      record(body, options)
      yield { choices: [{ delta: { content: "Recovered answer" } }] }
    }
  } }
})
vi.mock("@/services/tldw", async () => {
  const { tldwChat } = await import("@/services/tldw/TldwChat")
  return {
    tldwChat,
    tldwModels: { getModel: async () => ({ id: "gemma", provider: "ollama", capabilities: [] }) }
  }
})

const turnId = "12409645-7bce-4cba-b03b-bc4b0b27cc68"
const requestScope: ServicePromptRequestScope = {
  config: { serverUrl: "http://127.0.0.1:8000", authMode: "single-user", expectedUserId: "test-user" },
  userId: "test-user"
}
const durableOptions = {
  model: "gemma", apiProvider: "ollama", saveToDb: true,
  conversationId: "workspace-chat", requestScope,
  tldwTurn: { user_message_id: turnId }, originalUserMessage: "Exact user question"
}
const messages = [
  new SystemMessage("Use the supplied evidence."),
  new HumanMessage("Earlier question"),
  new AIMessage("Earlier answer"),
  new HumanMessage("Exact user question")
]

const dispatch = async (model: Awaited<ReturnType<typeof pageAssistModel>>, stream: boolean) => {
  if (!stream) return (await model.generateOnce(messages)).text
  let text = ""
  for await (const token of await model.stream(messages)) text += token
  return text
}

describe.each([true, false])("factory-to-wire retry projection (stream=%s)", stream => {
  beforeEach(() => {
    transport.requests.length = 0
    transport.failNext = false
    useStoreChatModelSettings.getState().reset()
  })

  it.each([undefined, "local-user-id"])("retries a failed durable turn without legacy flags (clientMessageId=%s)", async clientMessageId => {
    const initial = await pageAssistModel({ ...durableOptions, clientMessageId })
    transport.failNext = true
    await expect(dispatch(initial, stream)).rejects.toThrow(/completion failed/i)

    const retry = await pageAssistModel({ ...durableOptions, clientMessageId, retryFailedTurn: true })
    expect(await dispatch(retry, stream)).toBe("Recovered answer")
    expect(transport.requests).toHaveLength(2)
    const request = transport.requests[1]
    expect(request.body).toEqual(transport.requests[0].body)
    expect(request.body).toMatchObject({
      stream, save_to_db: true, conversation_id: "workspace-chat",
      tldw_turn: { user_message_id: turnId },
      messages: [{ role: "system", content: "Use the supplied evidence." }, { role: "user", content: "Exact user question" }]
    })
    if (clientMessageId) {
      expect(request.body.metadata).toEqual({ tldw_client_message_id: "local-user-id" })
    } else {
      expect(request.body).not.toHaveProperty("metadata")
    }
    expect(request.requestScope).toBe(requestScope)
  })

  it.each([false, true])("suppresses durable legacy regeneration metadata (retryFailedTurn=%s)", async retryFailedTurn => {
    const model = await pageAssistModel({
      ...durableOptions, retryFailedTurn, clientMessageId: "local-user-id",
      regenerateFromMessageId: "legacy-user-id"
    })
    await dispatch(model, stream)
    expect(transport.requests[0].body.metadata).toEqual({ tldw_client_message_id: "local-user-id" })
    expect(transport.requests[0].body.tldw_turn).toEqual({ user_message_id: turnId })
  })

  it.each([
    { retryFailedTurn: false, regenerateFromMessageId: undefined, metadata: { tldw_client_message_id: "local-user-id" }, roles: ["system", "user", "assistant", "user"] },
    { retryFailedTurn: true, regenerateFromMessageId: undefined, metadata: { tldw_client_message_id: "local-user-id", tldw_retry_failed_turn: true }, roles: ["system", "user", "assistant", "user"] },
    { retryFailedTurn: false, regenerateFromMessageId: "legacy-user-id", metadata: { tldw_client_message_id: "local-user-id", tldw_regenerate_from_message_id: "legacy-user-id" }, roles: ["system", "user"] }
  ])("preserves ordinary legacy metadata and history: %j", async ({ retryFailedTurn, regenerateFromMessageId, metadata, roles }) => {
    const model = await pageAssistModel({
      model: "gemma", apiProvider: "ollama", saveToDb: true,
      conversationId: "workspace-chat", clientMessageId: "local-user-id",
      retryFailedTurn, regenerateFromMessageId
    })
    await dispatch(model, stream)
    const request = transport.requests[0].body
    expect(request.metadata).toEqual(metadata)
    expect(request.messages.map(message => message.role)).toEqual(roles)
    expect(request).not.toHaveProperty("tldw_turn")
    expect(request.stream).toBe(stream)
  })
})

it("keeps native selected-history ownership incompatible with a durable turn at the real factory", async () => {
  await expect(pageAssistModel({ ...durableOptions, clientManagedHistory: true })).rejects.toThrow("unsupported_history_durable_turn")
})
it.each([true, false])("forwards explicit workspace scope from the factory through the real model and chat service (stream=%s)", async stream => {
  transport.requests.length = 0
  const scope = { type: "workspace" as const, workspaceId: "space" }
  const model = await pageAssistModel({ ...durableOptions, scope })
  await dispatch(model, stream)
  expect(transport.requests[0].scope).toBe(scope)
  expect(transport.requests[0].body).not.toHaveProperty("workspace_id")
})
