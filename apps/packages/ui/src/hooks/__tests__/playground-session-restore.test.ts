import { describe, expect, it } from "vitest"
import { shouldRestorePersistedPlaygroundSession } from "@/hooks/playground-session-restore"

describe("shouldRestorePersistedPlaygroundSession", () => {
  it("returns false when no persisted session is available", () => {
    expect(
      shouldRestorePersistedPlaygroundSession({
        hasPersistedSession: false,
        persistedHistoryId: "h-1",
        persistedServerChatId: "c-1",
        currentHistoryId: null,
        currentServerChatId: null,
        currentMessagesLength: 0,
        currentHistoryLength: 0
      })
    ).toBe(false)
  })

  it("restores when chat view has no active conversation state", () => {
    expect(
      shouldRestorePersistedPlaygroundSession({
        hasPersistedSession: true,
        persistedHistoryId: "h-1",
        persistedServerChatId: null,
        currentHistoryId: null,
        currentServerChatId: null,
        currentMessagesLength: 0,
        currentHistoryLength: 0
      })
    ).toBe(true)
  })

  it("restores when current conversation differs from persisted session", () => {
    expect(
      shouldRestorePersistedPlaygroundSession({
        hasPersistedSession: true,
        persistedHistoryId: "h-1",
        persistedServerChatId: "c-1",
        currentHistoryId: "h-2",
        currentServerChatId: "c-2",
        currentMessagesLength: 4,
        currentHistoryLength: 4
      })
    ).toBe(true)
  })

  it("does not restore when the conversation changes after the playground mounts", () => {
    expect(
      shouldRestorePersistedPlaygroundSession({
        hasPersistedSession: true,
        persistedHistoryId: null,
        persistedServerChatId: "persisted-chat",
        initialHistoryId: null,
        initialServerChatId: null,
        initialMessagesLength: 0,
        initialHistoryLength: 0,
        currentHistoryId: null,
        currentServerChatId: "selected-chat",
        currentMessagesLength: 0,
        currentHistoryLength: 0
      })
    ).toBe(false)
  })

  it("restores a stale conversation that was already present when the playground mounted", () => {
    expect(
      shouldRestorePersistedPlaygroundSession({
        hasPersistedSession: true,
        persistedHistoryId: null,
        persistedServerChatId: "persisted-chat",
        initialHistoryId: null,
        initialServerChatId: "stale-chat",
        initialMessagesLength: 2,
        initialHistoryLength: 2,
        currentHistoryId: null,
        currentServerChatId: "stale-chat",
        currentMessagesLength: 2,
        currentHistoryLength: 2
      })
    ).toBe(true)
  })

  it("does not restore when current conversation already matches persisted session", () => {
    expect(
      shouldRestorePersistedPlaygroundSession({
        hasPersistedSession: true,
        persistedHistoryId: "h-1",
        persistedServerChatId: "c-1",
        currentHistoryId: "h-1",
        currentServerChatId: "c-1",
        currentMessagesLength: 6,
        currentHistoryLength: 6
      })
    ).toBe(false)
  })
})
