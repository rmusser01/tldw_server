import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { ChatSettingsRecord } from "@/types/chat-session-settings"

const state = vi.hoisted(() => ({
  storage: new Map<string, ChatSettingsRecord>(),
  remoteSettings: null as ChatSettingsRecord | null
}))

const mocks = vi.hoisted(() => ({
  initialize: vi.fn(),
  getChatSettings: vi.fn(),
  updateChatSettings: vi.fn()
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    hasPersistentBackend: true,
    get: vi.fn(async (key: string) => state.storage.get(key)),
    set: vi.fn(async (key: string, value: ChatSettingsRecord) => {
      state.storage.set(key, value)
    }),
    remove: vi.fn(async (key: string) => {
      state.storage.delete(key)
    })
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: (...args: unknown[]) =>
      (mocks.initialize as (...args: unknown[]) => unknown)(...args),
    getChatSettings: (...args: unknown[]) =>
      (mocks.getChatSettings as (...args: unknown[]) => unknown)(...args),
    updateChatSettings: (...args: unknown[]) =>
      (mocks.updateChatSettings as (...args: unknown[]) => unknown)(...args)
  }
}))

import {
  getChatSettingsStorageKey,
  normalizeChatSettingsRecord,
  resolveChatSettingsKey,
  syncChatSettingsForServerChat as syncSettings
} from "@/services/chat-settings"

import { settingsTestOwner, installSettingsLocks } from "./chat-settings-fixture"
const syncChatSettingsForServerChat = (params: Parameters<typeof syncSettings>[0]) => syncSettings({ ...settingsTestOwner, ...params })
afterEach(() => vi.unstubAllGlobals())

const createSettings = (
  overrides: Partial<ChatSettingsRecord> = {}
): ChatSettingsRecord => ({
  schemaVersion: 2,
  updatedAt: "2026-03-08T00:00:00.000Z",
  authorNote: "Stay in character.",
  pinnedMessageIds: ["msg-1"],
  characterMemoryById: {
    "7": {
      note: "Favorite tea is jasmine.",
      updatedAt: "2026-03-08T00:00:00.000Z"
    }
  },
  ...overrides
})

describe("syncChatSettingsForServerChat", () => {
  beforeEach(() => {
    state.storage.clear()
    installSettingsLocks()
    state.remoteSettings = null

    mocks.initialize.mockReset()
    mocks.getChatSettings.mockReset()
    mocks.updateChatSettings.mockReset()

    mocks.initialize.mockResolvedValue(undefined)
    mocks.getChatSettings.mockImplementation(async (serverChatId: string) => ({
      conversation_id: serverChatId,
      settings: state.remoteSettings ?? {}
    }))
    mocks.updateChatSettings.mockImplementation(
      async (serverChatId: string, settings: ChatSettingsRecord) => ({
        conversation_id: serverChatId,
        settings
      })
    )
  })

  it("does not push chat settings back to the server when only sync timestamps differ", async () => {
    const serverChatId = "chat-1"
    const localSettings = createSettings({
      updatedAt: "2026-03-08T00:05:00.000Z",
      characterMemoryById: {
        "7": {
          note: "Favorite tea is jasmine.",
          updatedAt: "2026-03-08T00:05:00.000Z"
        }
      }
    })
    state.remoteSettings = createSettings({
      updatedAt: "2026-03-08T00:00:00.000Z",
      characterMemoryById: {
        "7": {
          note: "Favorite tea is jasmine.",
          updatedAt: "2026-03-08T00:00:00.000Z"
        }
      }
    })

    state.storage.set(
      getChatSettingsStorageKey(
        resolveChatSettingsKey({ historyId: null, serverChatId })
      ),
      localSettings
    )

    const result = await syncChatSettingsForServerChat({
      historyId: "history-1",
      serverChatId
    })

    expect(mocks.updateChatSettings).not.toHaveBeenCalled()
    expect(result).toEqual(state.remoteSettings)
  })

  it("loads canonical server settings without automatically replaying newer cached edits", async () => {
    const serverChatId = "chat-2"
    const localSettings = createSettings({
      updatedAt: "2026-03-08T00:10:00.000Z",
      authorNote: "Address the user as Captain."
    })
    state.remoteSettings = createSettings({
      updatedAt: "2026-03-08T00:00:00.000Z",
      authorNote: "Stay in character."
    })

    state.storage.set(
      getChatSettingsStorageKey(
        resolveChatSettingsKey({ historyId: null, serverChatId })
      ),
      localSettings
    )

    const result = await syncChatSettingsForServerChat({
      historyId: "history-2",
      serverChatId
    })

    expect(mocks.updateChatSettings).not.toHaveBeenCalled()
    expect(result).toEqual(state.remoteSettings)
  })

  it("does not promote a local mirror's overlay when the authenticated server has no overrides", async () => {
    const localSettings = createSettings({ assistantOverlay: {
      kind: "persona", id: "persona-11", name: "Research Guide", updatedAt: "2026-05-22T18:10:00Z"
    } })
    const localKey = getChatSettingsStorageKey("local:history-overlay-sync")
    state.storage.set(localKey, localSettings)
    const result = await syncChatSettingsForServerChat({ historyId: "history-overlay-sync", serverChatId: "chat-overlay-sync" })
    expect(result).toBeNull()
    expect(mocks.updateChatSettings).not.toHaveBeenCalled()
    expect(state.storage.get(localKey)).toEqual(localSettings)
  })

  it.each([404, undefined])("surfaces failed remote reads without writing, status=%s", async (status) => {
    const error = Object.assign(new Error("Chat settings not found"), { status })
    mocks.getChatSettings.mockRejectedValueOnce(error)
    await expect(syncChatSettingsForServerChat({ historyId: null, serverChatId: "chat-settings-missing" })).rejects.toBe(error)
    expect(mocks.updateChatSettings).not.toHaveBeenCalled()
  })

  it("passes workspace scope through server settings sync", async () => {
    const historyId = "history-workspace-sync"
    const serverChatId = "chat-workspace-sync"
    const scope = { type: "workspace", workspaceId: "workspace-1" } as const
    const localSettings = createSettings({
      authorNote: "Workspace-scoped note."
    })
    state.storage.set(
      getChatSettingsStorageKey(
        resolveChatSettingsKey({ historyId, serverChatId: null })
      ),
      localSettings
    )
    state.remoteSettings = localSettings

    const result = await syncChatSettingsForServerChat({
      historyId,
      serverChatId,
      scope
    })

    expect(mocks.getChatSettings).toHaveBeenCalledWith(serverChatId, { requestScope: settingsTestOwner.requestScope, scope })
    expect(mocks.updateChatSettings).not.toHaveBeenCalled()
    expect(result).toEqual(localSettings)
  })
})

describe("normalizeChatSettingsRecord", () => {
  it("keeps nested and legacy dictionary ids mirrored from nested context", () => {
    const settings = normalizeChatSettingsRecord({
      schemaVersion: 2,
      updatedAt: "2026-03-08T00:00:00.000Z",
      conversationContext: {
        world_book_ids: [9, "10", 0, "bad"],
        chat_dictionary_ids: [7, "8", 7]
      },
      chat_dictionary_ids: [99]
    })

    expect(settings?.conversationContext).toEqual({
      world_book_ids: [9, 10],
      chat_dictionary_ids: [7, 8]
    })
    expect(settings?.chat_dictionary_ids).toEqual([7, 8])
  })

  it("mirrors legacy dictionary ids into conversation context when nested ids are absent", () => {
    const settings = normalizeChatSettingsRecord({
      schemaVersion: 2,
      updatedAt: "2026-03-08T00:00:00.000Z",
      conversationContext: {
        world_book_ids: [1]
      },
      chat_dictionary_ids: [3, "4", 3]
    })

    expect(settings?.conversationContext).toEqual({
      world_book_ids: [1],
      chat_dictionary_ids: [3, 4]
    })
    expect(settings?.chat_dictionary_ids).toEqual([3, 4])
  })
})

vi.mock("@/db/dexie/schema", async () => ({
  db: (await import("@/hooks/chat/__tests__/local-history-fixture")).memory
}))
beforeEach(async () => {
  const { memory } = await import(
    "@/hooks/chat/__tests__/local-history-fixture"
  )
  memory.chatHistories.rows.clear()
  for (const id of [
    "history-1",
    "history-2",
    "history-overlay-sync",
    "history-workspace-sync"
  ])
    await memory.chatHistories.put({ id })
})
