import { beforeEach, describe, expect, it, vi } from "vitest"
import type { HistoryInfo, Message } from "@/db/dexie/types"

// Only IndexedDB's storage/transaction boundary is replaced; reconciliation is real.
const state = vi.hoisted(() => ({
  histories: new Map<string, HistoryInfo>(),
  messages: new Map<string, Message>()
}))
function table<T extends { id: string }>(rows: Map<string, T>) {
  return {
    get: async (id: string) => rows.get(id),
    add: async (row: T) => {
      if (rows.has(row.id)) throw new Error("duplicate primary key")
      rows.set(row.id, structuredClone(row))
      return row.id
    },
    put: async (row: T) => {
      rows.set(row.id, structuredClone(row))
      return row.id
    },
    update: async (id: string, changes: Partial<T>) => {
      if (!rows.has(id)) return 0
      rows.set(id, { ...rows.get(id)!, ...changes })
      return 1
    },
    where: (field: string) => ({
      equals: (value: unknown) => ({
        toArray: async () =>
          [...rows.values()].filter(
            (row) => (row as Record<string, unknown>)[field] === value
          )
      })
    })
  }
}
vi.mock("@/db/dexie/schema", () => ({
  db: {
    chatHistories: table(state.histories),
    messages: table(state.messages),
    modelNickname: {},
    sessionFiles: {},
    transaction: async (
      _mode: string,
      _tables: unknown[],
      operation: (transaction: { abort: () => void }) => Promise<unknown>
    ) => operation({ abort: vi.fn() })
  }
}))
vi.mock("@/db/dexie/helpers", () => ({
  generateID: () => `history-${state.histories.size + 1}`
}))

import type { Message as ChatMessage } from "@/store/option"
import { createBranchMessage } from "../messageHandlers"
import {
  reconcileServerChatMirror,
  serverChatMirrorOwnerKey
} from "@/db/dexie/server-chat-mirror"
const owner = vi.hoisted(() => ({
  revision: 0,
  controller: new AbortController(),
  release: vi.fn()
}))
const client = vi.hoisted(() => ({
  getChat: vi.fn(),
  createChat: vi.fn(),
  addChatMessage: vi.fn()
}))
const requestScope = {
  config: { serverUrl: "http://chat.test", authMode: "multi-user" as const },
  userId: "alice"
}
vi.mock("@/store/playground-session", () => ({
  usePlaygroundSessionStore: {
    getState: () => ({ restoreRevision: owner.revision })
  }
}))
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: async () => ({
    requestScope,
    scopeSignal: owner.controller.signal,
    scopeInvalidatedSignal: owner.controller.signal,
    release: owner.release
  })
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: client }))
vi.mock("@/db/dexie/branch", () => ({ generateBranchMessage: vi.fn() }))
vi.mock("@/db", () => ({ getPromptById: vi.fn(), getSessionFiles: vi.fn() }))

describe("server branch with real local mirror", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    state.histories.clear()
    state.messages.clear()
    owner.revision = 0
    owner.controller = new AbortController()
    client.getChat.mockResolvedValue({
      id: "parent-chat",
      title: "Parent",
      character_id: 2
    })
    client.createChat.mockResolvedValue({
      id: "fork-chat",
      title: "Fork",
      character_id: 2
    })
    let next = 0
    client.addChatMessage.mockImplementation(async () => ({
      id: "fork-" + next++,
      version: 1
    }))
  })
  it("keeps parent rows and alternatives intact while saving fork receipts, images and parent links", async () => {
    const ownerKey = serverChatMirrorOwnerKey({ requestScope } as Parameters<
      typeof serverChatMirrorOwnerKey
    >[0])
    const parent: HistoryInfo = {
      id: "parent-local",
      title: "Parent",
      is_rag: false,
      createdAt: 1,
      server_chat_id: "parent-chat",
      server_scope_key: ownerKey
    }
    const image = "data:image/png;base64,aW1hZ2U="
    const prefix: ChatMessage[] = [
      {
        id: "local-user",
        serverMessageId: "parent-user",
        role: "user",
        isBot: false,
        name: "You",
        message: "",
        images: [image],
        sources: []
      },
      {
        id: "local-answer",
        serverMessageId: "parent-answer",
        role: "assistant",
        isBot: true,
        name: "Character",
        message: "Answer",
        parentMessageId: "local-user",
        sources: [],
        modelId: "model",
        modelName: "Model",
        metadataExtra: { mood: "happy" },
        variants: [
          { id: "old-answer", serverMessageId: "parent-old", message: "Old" },
          {
            id: "local-answer",
            serverMessageId: "parent-answer",
            message: "Answer"
          }
        ],
        activeVariantIndex: 1
      }
    ]
    state.histories.set(parent.id, structuredClone(parent))
    for (const row of prefix)
      state.messages.set(row.id!, {
        id: row.id!,
        history_id: parent.id,
        role: row.role!,
        name: row.name,
        content: row.message,
        images: row.images || [],
        createdAt: 1,
        serverMessageId: row.serverMessageId
      })
    const original = structuredClone([...state.messages.values()])
    const originalProjection = structuredClone(prefix)
    const publish = vi.fn(),
      setHistoryId = vi.fn()
    const branch = createBranchMessage({
      historyId: parent.id,
      serverChatId: "parent-chat",
      characterId: 2,
      messages: prefix,
      history: prefix.map((row) => ({
        role: row.role!,
        content: row.message,
        images: row.images
      })),
      setMessages: publish,
      setHistory: vi.fn(),
      setHistoryId,
      serverOnly: true,
      notification: {
        error: vi.fn(), warning: vi.fn(), success: vi.fn(),
        info: vi.fn(), open: vi.fn(), destroy: vi.fn()
      },
      // Accepting our own branch legitimately advances route ownership synchronously.
      onServerChatBranchAccepted: () => {
        owner.revision++
      }
    })
    expect(await branch(1)).toBe("fork-chat")
    const forkHistory = setHistoryId.mock.calls[0][0]
    expect(forkHistory).not.toBe(parent.id)
    const forkRows = [...state.messages.values()].filter(
      (row) => row.history_id === forkHistory
    )
    expect(forkRows.map((row) => row.serverMessageId)).toEqual([
      "fork-0",
      "fork-1"
    ])
    expect(forkRows[0]).toMatchObject({
      id: "fork-0",
      content: "",
      images: [image]
    })
    expect(forkRows[1]).toMatchObject({
      id: "fork-1",
      parent_message_id: "fork-0",
      modelId: "model",
      metadataExtra: { mood: "happy" }
    })
    expect(publish.mock.calls[0][0][1]).toMatchObject({
      parentMessageId: "fork-0",
      variants: undefined,
      activeVariantIndex: undefined
    })
    expect(client.addChatMessage).toHaveBeenNthCalledWith(
      2,
      "fork-chat",
      expect.objectContaining({ parent_message_id: "fork-0" }),
      expect.anything()
    )
    // Loading the canonical fork again is idempotent and cannot overwrite its parent.
    await reconcileServerChatMirror({
      historyId: forkHistory,
      chatId: "fork-chat",
      ownerKey,
      messages: publish.mock.calls[0][0]
    })
    expect(
      [...state.messages.values()].filter(
        (row) => row.history_id === forkHistory
      )
    ).toHaveLength(2)
    expect(
      [...state.messages.values()].filter((row) => row.history_id === parent.id)
    ).toEqual(original)
    expect(state.histories.get(parent.id)).toEqual(parent)
    expect(prefix).toEqual(originalProjection)
    expect(owner.release).toHaveBeenCalledOnce()
  })
})
