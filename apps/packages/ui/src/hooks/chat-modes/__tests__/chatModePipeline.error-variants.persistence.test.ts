import { beforeEach, describe, expect, it, vi } from "vitest"
import type { ChatHistory, Message } from "@/store/option"
import type { HistoryInfo, Message as StoredMessage } from "@/db/dexie/types"

// Control only the storage adapter. The pipeline, Retry handler, save helpers,
// PageAssistDatabase and restore formatter are production implementations.
const storage = vi.hoisted(() => ({
  messages: new Map<string, StoredMessage>(),
  histories: new Map<string, HistoryInfo>()
}))
vi.mock("@/db/dexie/schema", () => ({ db: {
  messages: {
    add: async (row: StoredMessage) => {
      if (storage.messages.has(row.id)) throw new Error("Duplicate message ID")
      storage.messages.set(row.id, structuredClone(row))
    },
    where: () => ({ equals: (historyId: string) => ({
      toArray: async () => [...storage.messages.values()]
        .filter(row => row.history_id === historyId).map(row => structuredClone(row))
    }) })
  },
  chatHistories: {
    add: async (row: HistoryInfo) => storage.histories.set(row.id, structuredClone(row)),
    update: async (id: string, patch: Partial<HistoryInfo>) => {
      const row = storage.histories.get(id)
      if (row) storage.histories.set(id, { ...row, ...patch })
    }
  }
} }))
vi.mock("@/db/dexie/nickname", () => ({
  getModelNicknameByID: async () => null,
  getAllModelNicknames: async () => ({})
}))
vi.mock("@/db", () => ({}))
vi.mock("@/services/tldw/TldwApiClient", () => ({}))
vi.mock("@/utils/mcp-disclosure", () => ({ applyMcpModuleDisclosureFromToolCalls: vi.fn() }))
vi.mock("@/services/title", () => ({ generateTitle: async () => "Variant test" }))
vi.mock("@/utils/update-page-title", () => ({ updatePageTitle: vi.fn() }))
vi.mock("@/models", async () => {
  const { ImageSupportUnconfirmedError } = await import("@/utils/chat-error-message")
  return { pageAssistModel: async () => ({
    stream: () => { throw new ImageSupportUnconfirmedError(false) }
  }) }
})

import { runChatPipeline, type ChatModeDefinition, type ChatModeParamsBase } from "../chatModePipeline"
import { saveMessageOnError } from "@/hooks/chat-helper"
import { createRegenerateLastMessage } from "@/hooks/handlers/messageHandlers"
import { formatToChatHistory, formatToMessage, saveHistory } from "@/db/dexie/helpers"
import { PageAssistDatabase } from "@/db/dexie/chat"

const mode: ChatModeDefinition<ChatModeParamsBase> = {
  id: "normal",
  buildUserMessage: ctx => ({
    isBot: false, name: "You", message: ctx.message, images: [ctx.image], sources: [],
    id: ctx.resolvedUserMessageId
  }),
  buildAssistantMessage: ctx => ({
    isBot: true, name: "Assistant", message: "", images: [], sources: [],
    id: ctx.resolvedAssistantMessageId, parentMessageId: ctx.resolvedAssistantParentMessageId
  }),
  preparePrompt: async () => ({ chatHistory: [], humanMessage: "Question", sources: [] })
}
const IMAGE = "data:image/png;base64,iVBORw0KGgo="

beforeEach(() => {
  storage.messages.clear()
  storage.histories.clear()
  let time = 1000
  vi.spyOn(Date, "now").mockImplementation(() => time += 10)
})

describe("failed Retry variant persistence", () => {
  it.each([false, true])("restores the latest of three failures as one assistant (reload between retries: %s)", async reloadBetweenRetries => {
    const history = await saveHistory("Variants", false, "web-ui")
    let messages: Message[] = []
    let transcript: ChatHistory = []
    let attempt = 0
    const setMessages: ChatModeParamsBase["setMessages"] = next => {
      messages = typeof next === "function" ? next(messages) : next
    }
    const setHistory = (next: ChatHistory) => { transcript = next }
    const submit = async (input: {
      message: string; image: string; isRegenerate?: boolean;
      regenerateFromMessage?: Message; controller?: AbortController
    }) => runChatPipeline(mode, input.message, input.image, Boolean(input.isRegenerate),
      messages, transcript, (input.controller ?? new AbortController()).signal, {
        selectedModel: "test-model", useOCR: false, setMessages, setHistory,
        setIsProcessing: vi.fn(), setStreaming: vi.fn(), setAbortController: vi.fn(),
        historyId: history.id, setHistoryId: vi.fn(), userMessageId: "user-local",
        assistantMessageId: `assistant-${attempt++}`,
        regenerateFromMessage: input.regenerateFromMessage,
        saveMessageOnError, saveMessageOnSuccess: vi.fn()
      })
    const restore = async () => {
      const rows = await new PageAssistDatabase().getChatHistory(history.id)
      messages = formatToMessage(rows)
      transcript = formatToChatHistory(rows)
    }

    await submit({ message: "Keep this image", image: IMAGE })
    for (let retry = 0; retry < 2; retry++) {
      if (reloadBetweenRetries) await restore()
      await createRegenerateLastMessage({
        validateBeforeSubmitFn: () => true, history: transcript, messages,
        setHistory, setMessages, onSubmit: submit
      })()
    }
    expect(messages.filter(message => message.isBot)).toHaveLength(1)
    await restore()
    expect(messages.filter(message => message.isBot)).toHaveLength(1)
    expect(messages.find(message => message.isBot)).toMatchObject({
      id: "assistant-2", parentMessageId: "user-local", activeVariantIndex: 2,
      variants: [{ id: "assistant-0" }, { id: "assistant-1" }, { id: "assistant-2" }]
    })
    expect(messages.filter(message => !message.isBot)).toEqual([
      expect.objectContaining({ id: "user-local", message: "Keep this image", images: [IMAGE] })
    ])
    expect([...storage.messages.values()].filter(row => row.role === "assistant")
      .map(row => row.parent_message_id)).toEqual(["user-local", "user-local", "user-local"])
  })

  it("does not guess a variant relationship for legacy unparented errors", async () => {
    const rows: StoredMessage[] = [0, 1].map(index => ({
      id: `legacy-${index}`, history_id: "old-history", role: "assistant", name: "Model",
      content: "Same error", images: [], createdAt: 1000 + index, parent_message_id: null
    }))
    for (const row of rows) await new PageAssistDatabase().addMessage(row)
    expect(formatToMessage(await new PageAssistDatabase().getChatHistory("old-history"))
      .map(message => ({ id: message.id, variants: message.variants }))).toEqual([
      { id: "legacy-0", variants: undefined }, { id: "legacy-1", variants: undefined }
    ])
  })
})
