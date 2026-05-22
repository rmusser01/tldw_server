import { describe, expect, it, vi } from "vitest"

import {
  DEFAULT_PERSONA_MEMORY_MODE,
  ensurePersonaServerChat
} from "../personaServerChat"

const createSetterBundle = () => ({
  setServerChatId: vi.fn(),
  setServerChatTitle: vi.fn(),
  setServerChatCharacterId: vi.fn(),
  setServerChatAssistantKind: vi.fn(),
  setServerChatAssistantId: vi.fn(),
  setServerChatPersonaMemoryMode: vi.fn(),
  setServerChatMetaLoaded: vi.fn(),
  setServerChatState: vi.fn(),
  setServerChatVersion: vi.fn(),
  setServerChatTopic: vi.fn(),
  setServerChatClusterId: vi.fn(),
  setServerChatSource: vi.fn(),
  setServerChatExternalRef: vi.fn()
})

describe("ensurePersonaServerChat", () => {
  it("creates a persona-backed chat with read_only default and updates chat state", async () => {
    const setters = createSetterBundle()
    const createChat = vi.fn().mockResolvedValue({
      id: "persona-chat-1",
      title: "Persona chat",
      assistant_kind: "persona",
      assistant_id: "garden-helper",
      persona_memory_mode: "read_only",
      state: "resolved",
      version: 8,
      topic_label: "Garden topic",
      character_id: null
    })
    const ensureServerChatHistoryId = vi.fn().mockResolvedValue("history-1")
    const invalidateServerChatHistory = vi.fn()

    const result = await ensurePersonaServerChat({
      assistant: {
        kind: "persona",
        id: "garden-helper",
        name: "Garden Helper"
      },
      serverChatId: null,
      serverChatTitle: null,
      serverChatAssistantKind: null,
      serverChatAssistantId: null,
      serverChatPersonaMemoryMode: null,
      serverChatState: "in-progress",
      serverChatTopic: null,
      serverChatClusterId: null,
      serverChatSource: null,
      serverChatExternalRef: null,
      historyId: "history-local",
      temporaryChat: false,
      createChat,
      ensureServerChatHistoryId,
      invalidateServerChatHistory,
      ...setters
    })

    expect(createChat).toHaveBeenCalledWith(
      expect.objectContaining({
        assistant_kind: "persona",
        assistant_id: "garden-helper",
        persona_memory_mode: DEFAULT_PERSONA_MEMORY_MODE
      }),
      undefined
    )
    expect(setters.setServerChatId).toHaveBeenCalledWith("persona-chat-1")
    expect(setters.setServerChatAssistantKind).toHaveBeenCalledWith("persona")
    expect(setters.setServerChatAssistantId).toHaveBeenCalledWith("garden-helper")
    expect(setters.setServerChatPersonaMemoryMode).toHaveBeenCalledWith(
      "read_only"
    )
    expect(setters.setServerChatMetaLoaded).toHaveBeenCalledWith(true)
    expect(invalidateServerChatHistory).toHaveBeenCalledTimes(1)
    expect(result).toEqual({
      chatId: "persona-chat-1",
      historyId: "history-1",
      personaMemoryMode: "read_only"
    })
  })

  it("passes workspace scope through when creating a persona-backed chat", async () => {
    const setters = createSetterBundle()
    const createChat = vi.fn().mockResolvedValue({
      id: "persona-chat-3",
      title: "Scoped persona chat",
      assistant_kind: "persona",
      assistant_id: "garden-helper",
      persona_memory_mode: "read_only"
    })

    await ensurePersonaServerChat({
      assistant: {
        kind: "persona",
        id: "garden-helper",
        name: "Garden Helper"
      },
      serverChatId: null,
      serverChatTitle: null,
      serverChatAssistantKind: null,
      serverChatAssistantId: null,
      serverChatPersonaMemoryMode: null,
      serverChatState: "in-progress",
      serverChatTopic: null,
      serverChatClusterId: null,
      serverChatSource: null,
      serverChatExternalRef: null,
      historyId: null,
      temporaryChat: false,
      scope: { type: "workspace", workspaceId: "workspace-1" },
      createChat,
      ensureServerChatHistoryId: vi.fn().mockResolvedValue("history-3"),
      invalidateServerChatHistory: vi.fn(),
      ...setters
    })

    expect(createChat).toHaveBeenCalledWith(
      expect.objectContaining({
        assistant_kind: "persona",
        assistant_id: "garden-helper"
      }),
      { scope: { type: "workspace", workspaceId: "workspace-1" } }
    )
  })

  it("reuses an existing matching persona chat without creating a new one", async () => {
    const setters = createSetterBundle()
    const createChat = vi.fn()
    const ensureServerChatHistoryId = vi.fn().mockResolvedValue("history-2")

    const result = await ensurePersonaServerChat({
      assistant: {
        kind: "persona",
        id: "garden-helper",
        name: "Garden Helper"
      },
      serverChatId: "persona-chat-2",
      serverChatTitle: "Garden chat",
      serverChatAssistantKind: "persona",
      serverChatAssistantId: "garden-helper",
      serverChatPersonaMemoryMode: "read_write",
      serverChatState: "in-progress",
      serverChatTopic: null,
      serverChatClusterId: null,
      serverChatSource: null,
      serverChatExternalRef: null,
      historyId: "history-2",
      temporaryChat: false,
      createChat,
      ensureServerChatHistoryId,
      invalidateServerChatHistory: vi.fn(),
      ...setters
    })

    expect(createChat).not.toHaveBeenCalled()
    expect(setters.setServerChatAssistantKind).toHaveBeenCalledWith("persona")
    expect(setters.setServerChatAssistantId).toHaveBeenCalledWith("garden-helper")
    expect(setters.setServerChatPersonaMemoryMode).toHaveBeenCalledWith(
      "read_write"
    )
    expect(result).toEqual({
      chatId: "persona-chat-2",
      historyId: "history-2",
      personaMemoryMode: "read_write"
    })
  })

  it("resets stale character-backed server chat metadata before creating persona chat", async () => {
    const setters = createSetterBundle()
    const createChat = vi.fn().mockResolvedValue({
      id: "persona-chat-4",
      title: "Garden persona chat",
      assistant_kind: "persona",
      assistant_id: "garden-helper",
      persona_memory_mode: "read_only",
      character_id: null
    })
    const ensureServerChatHistoryId = vi.fn().mockResolvedValue("history-4")

    const result = await ensurePersonaServerChat({
      assistant: {
        kind: "persona",
        id: "garden-helper",
        name: "Garden Helper"
      },
      serverChatId: "character-chat-1",
      serverChatTitle: "Old character chat",
      serverChatAssistantKind: "character",
      serverChatAssistantId: "42",
      serverChatPersonaMemoryMode: "read_write",
      serverChatState: "resolved",
      serverChatTopic: "Old topic",
      serverChatClusterId: "old-cluster",
      serverChatSource: "old-source",
      serverChatExternalRef: "old-ref",
      historyId: "history-old",
      temporaryChat: false,
      createChat,
      ensureServerChatHistoryId,
      invalidateServerChatHistory: vi.fn(),
      ...setters
    })

    expect(setters.setServerChatId.mock.calls[0]).toEqual([null])
    expect(setters.setServerChatTitle.mock.calls[0]).toEqual([null])
    expect(setters.setServerChatCharacterId.mock.calls[0]).toEqual([null])
    expect(setters.setServerChatAssistantKind.mock.calls[0]).toEqual([null])
    expect(setters.setServerChatAssistantId.mock.calls[0]).toEqual([null])
    expect(setters.setServerChatPersonaMemoryMode.mock.calls[0]).toEqual([null])
    expect(setters.setServerChatMetaLoaded.mock.calls[0]).toEqual([false])
    expect(createChat).toHaveBeenCalledWith(
      {
        assistant_kind: "persona",
        assistant_id: "garden-helper",
        persona_memory_mode: "read_only",
        state: "in-progress",
        topic_label: undefined,
        cluster_id: undefined,
        source: undefined,
        external_ref: undefined
      },
      undefined
    )
    expect(setters.setServerChatId).toHaveBeenLastCalledWith("persona-chat-4")
    expect(setters.setServerChatAssistantKind).toHaveBeenLastCalledWith(
      "persona"
    )
    expect(setters.setServerChatAssistantId).toHaveBeenLastCalledWith(
      "garden-helper"
    )
    expect(result).toEqual({
      chatId: "persona-chat-4",
      historyId: "history-4",
      personaMemoryMode: "read_only"
    })
  })

  it("does not carry read_write from a different stale persona chat into a new persona chat", async () => {
    const setters = createSetterBundle()
    const createChat = vi.fn().mockResolvedValue({
      id: "persona-chat-5",
      title: "New persona chat",
      assistant_kind: "persona",
      assistant_id: "research-helper",
      persona_memory_mode: "read_only",
      character_id: null
    })
    const ensureServerChatHistoryId = vi.fn().mockResolvedValue("history-5")

    const result = await ensurePersonaServerChat({
      assistant: {
        kind: "persona",
        id: "research-helper",
        name: "Research Helper"
      },
      serverChatId: "persona-chat-old",
      serverChatTitle: "Old persona chat",
      serverChatAssistantKind: "persona",
      serverChatAssistantId: "garden-helper",
      serverChatPersonaMemoryMode: "read_write",
      serverChatState: "in-progress",
      serverChatTopic: null,
      serverChatClusterId: null,
      serverChatSource: null,
      serverChatExternalRef: null,
      historyId: "history-old",
      temporaryChat: false,
      createChat,
      ensureServerChatHistoryId,
      invalidateServerChatHistory: vi.fn(),
      ...setters
    })

    expect(setters.setServerChatPersonaMemoryMode.mock.calls[0]).toEqual([null])
    expect(createChat).toHaveBeenCalledWith(
      {
        assistant_kind: "persona",
        assistant_id: "research-helper",
        persona_memory_mode: "read_only",
        state: "in-progress",
        topic_label: undefined,
        cluster_id: undefined,
        source: undefined,
        external_ref: undefined
      },
      undefined
    )
    expect(result).toEqual({
      chatId: "persona-chat-5",
      historyId: "history-5",
      personaMemoryMode: "read_only"
    })
  })
})
