import { beforeEach, describe, expect, it, vi } from "vitest"
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
  syncChatSettingsForServerChat
} from "@/services/chat-settings"

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
    state.remoteSettings = null

    mocks.initialize.mockReset()
    mocks.getChatSettings.mockReset()
    mocks.updateChatSettings.mockReset()

    mocks.initialize.mockResolvedValue(undefined)
    mocks.getChatSettings.mockImplementation(async () => ({
      settings: state.remoteSettings
    }))
    mocks.updateChatSettings.mockImplementation(
      async (_serverChatId: string, settings: ChatSettingsRecord) => ({
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
    expect(result).toEqual(localSettings)
  })

  it("pushes chat settings only when the merged result differs from the remote copy", async () => {
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

    expect(mocks.updateChatSettings).toHaveBeenCalledTimes(1)
    expect(mocks.updateChatSettings).toHaveBeenCalledWith(
      serverChatId,
      localSettings
    )
    expect(result).toEqual(localSettings)
  })

  it("reconciles locally persisted assistant overlay settings once a server chat id exists", async () => {
    const historyId = "history-overlay-sync"
    const serverChatId = "chat-overlay-sync"
    const localSettings = createSettings({
      updatedAt: "2026-05-22T18:10:00.000Z",
      assistantOverlay: {
        kind: "persona",
        id: "persona-11",
        name: "Research Guide",
        avatar_url: "https://example.com/persona-11.png",
        system_prompt_snapshot: "Keep answers structured and calm.",
        updatedAt: "2026-05-22T18:10:00.000Z"
      }
    })
    state.storage.set(
      getChatSettingsStorageKey(
        resolveChatSettingsKey({ historyId, serverChatId: null })
      ),
      localSettings
    )
    state.remoteSettings = null

    const result = await syncChatSettingsForServerChat({
      historyId,
      serverChatId
    })

    expect(mocks.updateChatSettings).toHaveBeenCalledTimes(1)
    expect(mocks.updateChatSettings).toHaveBeenCalledWith(
      serverChatId,
      expect.objectContaining({
        assistantOverlay: expect.objectContaining({
          id: "persona-11",
          system_prompt_snapshot: "Keep answers structured and calm."
        })
      })
    )
    expect(
      state.storage.get(
        getChatSettingsStorageKey(
          resolveChatSettingsKey({ historyId: null, serverChatId })
        )
      )
    ).toMatchObject({
      assistantOverlay: expect.objectContaining({ id: "persona-11" })
    })
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
