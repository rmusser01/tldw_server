import { describe, expect, it, vi } from "vitest"
import { useStoreMessageOption } from "@/store/option"

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

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise
  })
  return { promise, resolve }
}

describe("ensurePersonaServerChat", () => {
  it("keeps a newly created temporary Persona session temporary", async () => {
    const previous = useStoreMessageOption.getState()
    useStoreMessageOption.setState({ temporaryChat: true, serverChatId: null })
    const ensureServerChatHistoryId = vi.fn()
    try {
      await ensurePersonaServerChat({
        assistant: { kind: "persona", id: "garden-helper", name: "Garden Helper" },
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
        temporaryChat: true,
        createChat: vi.fn().mockResolvedValue({ id: "temporary-persona-session" }),
        ensureServerChatHistoryId,
        invalidateServerChatHistory: vi.fn(),
        ...createSetterBundle(),
        setServerChatId: useStoreMessageOption.getState().setServerChatId
      })
      expect(useStoreMessageOption.getState()).toMatchObject({
        serverChatId: "temporary-persona-session", temporaryChat: true
      })
      expect(ensureServerChatHistoryId).not.toHaveBeenCalled()
    } finally {
      useStoreMessageOption.setState(previous, true)
    }
  })

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
      serverChatMetaLoaded: false,
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
    expect(setters.setServerChatId).toHaveBeenCalledWith("persona-chat-1", { preserveTemporaryChat: false })
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

  it("does not publish a newly created persona chat when scope changes during history linking", async () => {
    const setters = createSetterBundle()
    const historyLink = deferred<string | null>()
    const ensureServerChatHistoryId = vi.fn(() => historyLink.promise)
    const invalidateServerChatHistory = vi.fn()
    const scopeController = new AbortController()
    const pending = ensurePersonaServerChat({
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
      serverChatMetaLoaded: false,
      serverChatState: "in-progress",
      serverChatTopic: null,
      serverChatClusterId: null,
      serverChatSource: null,
      serverChatExternalRef: null,
      historyId: null,
      temporaryChat: false,
      scopeInvalidatedSignal: scopeController.signal,
      createChat: vi.fn().mockResolvedValue({
        id: "persona-chat-scoped",
        title: "Scoped persona chat",
        assistant_kind: "persona",
        assistant_id: "garden-helper",
        persona_memory_mode: "read_only",
        state: "resolved",
        version: 4
      }),
      ensureServerChatHistoryId,
      invalidateServerChatHistory,
      ...setters
    })

    await vi.waitFor(() =>
      expect(ensureServerChatHistoryId).toHaveBeenCalledTimes(1)
    )
    scopeController.abort()
    historyLink.resolve("history-stale")

    await expect(pending).rejects.toMatchObject({ status: 412 })
    expect(ensureServerChatHistoryId).toHaveBeenCalledWith(
      "persona-chat-scoped",
      undefined,
      scopeController.signal
    )
    for (const setter of Object.values(setters)) {
      expect(setter).not.toHaveBeenCalled()
    }
    expect(invalidateServerChatHistory).not.toHaveBeenCalled()
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
      serverChatMetaLoaded: false,
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

  it("uses the requested persona memory mode when creating a new persona-backed chat", async () => {
    const setters = createSetterBundle()
    const createChat = vi.fn().mockResolvedValue({
      id: "persona-chat-memory",
      title: "Memory persona chat",
      assistant_kind: "persona",
      assistant_id: "garden-helper",
      persona_memory_mode: "read_write"
    })

    const result = await ensurePersonaServerChat({
      assistant: {
        kind: "persona",
        id: "garden-helper",
        name: "Garden Helper"
      },
      requestedPersonaMemoryMode: "read_write",
      serverChatId: null,
      serverChatTitle: null,
      serverChatAssistantKind: null,
      serverChatAssistantId: null,
      serverChatPersonaMemoryMode: null,
      serverChatMetaLoaded: false,
      serverChatState: "in-progress",
      serverChatTopic: null,
      serverChatClusterId: null,
      serverChatSource: null,
      serverChatExternalRef: null,
      historyId: null,
      temporaryChat: false,
      createChat,
      ensureServerChatHistoryId: vi.fn().mockResolvedValue("history-memory"),
      invalidateServerChatHistory: vi.fn(),
      ...setters
    })

    expect(createChat).toHaveBeenCalledWith(
      expect.objectContaining({
        assistant_kind: "persona",
        assistant_id: "garden-helper",
        persona_memory_mode: "read_write"
      }),
      undefined
    )
    expect(setters.setServerChatPersonaMemoryMode).toHaveBeenCalledWith(
      "read_write"
    )
    expect(result.personaMemoryMode).toBe("read_write")
  })

  it("starts a brand new persona chat with fresh metadata even when stale state remains in the store", async () => {
    const setters = createSetterBundle()
    const createChat = vi.fn().mockResolvedValue({
      id: "persona-chat-fresh",
      title: "Fresh persona chat",
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
      serverChatPersonaMemoryMode: "read_write",
      serverChatMetaLoaded: false,
      serverChatState: "resolved",
      serverChatTopic: "Stale topic",
      serverChatClusterId: "stale-cluster",
      serverChatSource: "stale-source",
      serverChatExternalRef: "stale-ref",
      historyId: "history-local",
      temporaryChat: false,
      createChat,
      ensureServerChatHistoryId: vi.fn().mockResolvedValue("history-fresh"),
      invalidateServerChatHistory: vi.fn(),
      ...setters
    })

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
      serverChatMetaLoaded: true,
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

  it("does not publish reused persona metadata when scope changes during history linking", async () => {
    const setters = createSetterBundle()
    const historyLink = deferred<string | null>()
    const ensureServerChatHistoryId = vi.fn(() => historyLink.promise)
    const scopeController = new AbortController()
    const pending = ensurePersonaServerChat({
      assistant: {
        kind: "persona",
        id: "garden-helper",
        name: "Garden Helper"
      },
      serverChatId: "persona-chat-existing",
      serverChatTitle: "Existing persona chat",
      serverChatAssistantKind: "persona",
      serverChatAssistantId: "garden-helper",
      serverChatPersonaMemoryMode: "read_write",
      serverChatMetaLoaded: true,
      serverChatState: "in-progress",
      serverChatTopic: null,
      serverChatClusterId: null,
      serverChatSource: null,
      serverChatExternalRef: null,
      historyId: null,
      temporaryChat: false,
      scopeInvalidatedSignal: scopeController.signal,
      createChat: vi.fn(),
      ensureServerChatHistoryId,
      invalidateServerChatHistory: vi.fn(),
      ...setters
    })

    await vi.waitFor(() =>
      expect(ensureServerChatHistoryId).toHaveBeenCalledTimes(1)
    )
    scopeController.abort()
    historyLink.resolve("history-stale")

    await expect(pending).rejects.toMatchObject({ status: 412 })
    expect(ensureServerChatHistoryId).toHaveBeenCalledWith(
      "persona-chat-existing",
      "Existing persona chat",
      scopeController.signal
    )
    for (const setter of Object.values(setters)) {
      expect(setter).not.toHaveBeenCalled()
    }
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
      serverChatMetaLoaded: true,
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
    expect(setters.setServerChatId).toHaveBeenLastCalledWith("persona-chat-4", { preserveTemporaryChat: false })
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
      serverChatMetaLoaded: true,
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

  it("does not reset a restored server chat before assistant metadata hydrates", async () => {
    const setters = createSetterBundle()
    const createChat = vi.fn()
    const ensureServerChatHistoryId = vi.fn().mockResolvedValue("history-restored")

    const result = await ensurePersonaServerChat({
      assistant: {
        kind: "persona",
        id: "garden-helper",
        name: "Garden Helper"
      },
      serverChatId: "restored-chat",
      serverChatTitle: "Restored chat",
      serverChatAssistantKind: null,
      serverChatAssistantId: null,
      serverChatPersonaMemoryMode: "read_write",
      serverChatMetaLoaded: false,
      serverChatState: "resolved",
      serverChatTopic: "Restored topic",
      serverChatClusterId: "restored-cluster",
      serverChatSource: "restored-source",
      serverChatExternalRef: "restored-ref",
      historyId: "history-restored",
      temporaryChat: false,
      createChat,
      ensureServerChatHistoryId,
      invalidateServerChatHistory: vi.fn(),
      ...setters
    })

    expect(createChat).not.toHaveBeenCalled()
    expect(setters.setServerChatId).not.toHaveBeenCalledWith(null)
    expect(setters.setServerChatMetaLoaded).not.toHaveBeenCalledWith(false)
    expect(result).toEqual({
      chatId: "restored-chat",
      historyId: "history-restored",
      personaMemoryMode: "read_write"
    })
  })
})

// Create/mirror I/O are declared callbacks; the actual helper writes retained
// metadata so an obsolete publication tail can corrupt replacement state.
const ownedFixture = () => {
  let current = true
  let metadata: Record<string, unknown> = {}
  const setters = createSetterBundle()
  for (const [name, setter] of Object.entries(setters))
    setter.mockImplementation((value) => {
      metadata[name] = value
    })
  const createChat = vi.fn().mockResolvedValue({
    id: "created-persona",
    title: "Original title",
    assistant_kind: "persona",
    assistant_id: "original-persona",
    persona_memory_mode: "read_only",
    version: 3
  })
  const mirror = vi.fn().mockResolvedValue("owned-mirror")
  const invalidate = vi.fn()
  const args = {
    assistant: {
      kind: "persona" as const,
      id: "original-persona",
      name: "Original"
    },
    serverChatId: null as string | null,
    serverChatTitle: null,
    serverChatAssistantKind: null as "persona" | "character" | null,
    serverChatAssistantId: null as string | null,
    serverChatPersonaMemoryMode: null,
    serverChatMetaLoaded: false,
    serverChatState: "in-progress",
    serverChatTopic: null,
    serverChatClusterId: null,
    serverChatSource: null,
    serverChatExternalRef: null,
    historyId: null,
    temporaryChat: false,
    scope: { type: "workspace" as const, workspaceId: "original-workspace" },
    requestScope: {
      config: {
        serverUrl: "https://original.test",
        authMode: "multi-user" as const
      },
      userId: "alice"
    },
    isCurrent: () => current,
    createChat,
    ensureServerChatHistoryId: mirror,
    invalidateServerChatHistory: invalidate,
    ...setters
  }
  const retire = () => {
    current = false
    metadata = { replacement: "Keep replacement metadata" }
  }
  return {
    args,
    setters,
    createChat,
    mirror,
    invalidate,
    retire,
    metadata: () => metadata
  }
}

const publicationNames = [
  "setServerChatId",
  "setServerChatState",
  "setServerChatVersion",
  "setServerChatTopic",
  "setServerChatClusterId",
  "setServerChatSource",
  "setServerChatExternalRef",
  "setServerChatTitle",
  "setServerChatCharacterId",
  "setServerChatAssistantKind",
  "setServerChatAssistantId",
  "setServerChatPersonaMemoryMode",
  "setServerChatMetaLoaded"
] as const

describe("Persona original command ownership", () => {
  it("refuses an expired original source before create or mirror I/O", async () => {
    const fixture = ownedFixture()
    fixture.retire()
    await expect(ensurePersonaServerChat(fixture.args)).rejects.toMatchObject({
      status: 412
    })
    expect(fixture.createChat).not.toHaveBeenCalled()
    expect(fixture.mirror).not.toHaveBeenCalled()
    expect(fixture.metadata()).toEqual({
      replacement: "Keep replacement metadata"
    })
  })

  it("does not link or publish a held create after its original source retires", async () => {
    const fixture = ownedFixture()
    const held = deferred<{ id: string }>()
    fixture.createChat.mockReturnValue(held.promise)
    const result = ensurePersonaServerChat(fixture.args).then(
      () => null,
      (error) => error
    )
    await vi.waitFor(() => expect(fixture.createChat).toHaveBeenCalledOnce())
    fixture.retire()
    held.resolve({ id: "late-original" })
    expect(await result).toMatchObject({ status: 412 })
    expect(fixture.mirror).not.toHaveBeenCalled()
    expect(fixture.metadata()).toEqual({
      replacement: "Keep replacement metadata"
    })
  })

  it.each(["created", "reused"] as const)(
    "does not publish %s metadata after held mirror I/O retires its source",
    async (kind) => {
      const fixture = ownedFixture()
      const held = deferred<string | null>()
      fixture.mirror.mockReturnValue(held.promise)
      const result = ensurePersonaServerChat({
        ...fixture.args,
        ...(kind === "reused"
          ? {
              serverChatId: "existing-persona",
              serverChatMetaLoaded: true,
              serverChatAssistantKind: "persona" as const,
              serverChatAssistantId: "original-persona"
            }
          : {})
      }).then(
        () => null,
        (error) => error
      )
      await vi.waitFor(() => expect(fixture.mirror).toHaveBeenCalledOnce())
      fixture.retire()
      held.resolve("late-mirror")
      expect(await result).toMatchObject({ status: 412 })
      expect(fixture.metadata()).toEqual({
        replacement: "Keep replacement metadata"
      })
      expect(fixture.invalidate).not.toHaveBeenCalled()
    }
  )

  it.each(publicationNames)(
    "retires the created metadata tail after %s synchronously replaces its source",
    async (name) => {
      const fixture = ownedFixture()
      fixture.setters[name].mockImplementation(() => fixture.retire())
      await expect(ensurePersonaServerChat(fixture.args)).rejects.toMatchObject(
        { status: 412 }
      )
      expect(fixture.metadata()).toEqual({
        replacement: "Keep replacement metadata"
      })
      expect(fixture.invalidate).not.toHaveBeenCalled()
    }
  )

  it.each(publicationNames)(
    "retires the reset tail after %s synchronously replaces its source",
    async (name) => {
      const fixture = ownedFixture()
      fixture.setters[name].mockImplementation(() => fixture.retire())
      await expect(
        ensurePersonaServerChat({
          ...fixture.args,
          serverChatId: "old-character",
          serverChatMetaLoaded: true,
          serverChatAssistantKind: "character",
          serverChatAssistantId: "old-card"
        })
      ).rejects.toMatchObject({ status: 412 })
      expect(fixture.metadata()).toEqual({
        replacement: "Keep replacement metadata"
      })
      expect(fixture.createChat).not.toHaveBeenCalled()
      expect(fixture.mirror).not.toHaveBeenCalled()
    }
  )

  it.each([
    "setServerChatAssistantKind",
    "setServerChatAssistantId",
    "setServerChatPersonaMemoryMode",
    "setServerChatCharacterId"
  ] as const)(
    "retires the reused metadata tail after %s synchronously replaces its source",
    async (name) => {
      const fixture = ownedFixture()
      fixture.setters[name].mockImplementation(() => fixture.retire())
      await expect(
        ensurePersonaServerChat({
          ...fixture.args,
          serverChatId: "existing-persona",
          serverChatMetaLoaded: true,
          serverChatAssistantKind: "persona",
          serverChatAssistantId: "original-persona"
        })
      ).rejects.toMatchObject({ status: 412 })
      expect(fixture.metadata()).toEqual({
        replacement: "Keep replacement metadata"
      })
    }
  )

  it("retains the original account and workspace request while creation is held", async () => {
    const fixture = ownedFixture()
    const held = deferred<{ id: string }>()
    fixture.createChat.mockReturnValue(held.promise)
    const result = ensurePersonaServerChat(fixture.args)
    const sentOptions = fixture.createChat.mock.calls[0][1]
    fixture.args.requestScope.config.serverUrl = "https://replacement.test"
    fixture.args.requestScope.userId = "bob"
    fixture.args.scope.workspaceId = "replacement-workspace"
    expect(sentOptions).toEqual({
      scope: { type: "workspace", workspaceId: "original-workspace" },
      requestScope: {
        config: { serverUrl: "https://original.test", authMode: "multi-user" },
        userId: "alice"
      },
      signal: undefined
    })
    held.resolve({ id: "created-persona" })
    expect(await result).toEqual({
      chatId: "created-persona",
      historyId: "owned-mirror",
      personaMemoryMode: "read_only"
    })
  })

  it("does not link or publish after execution cancellation during creation", async () => {
    const fixture = ownedFixture()
    const held = deferred<{ id: string }>()
    const controller = new AbortController()
    fixture.createChat.mockReturnValue(held.promise)
    const result = ensurePersonaServerChat({
      ...fixture.args,
      signal: controller.signal
    }).then(
      () => null,
      (error) => error
    )
    await vi.waitFor(() => expect(fixture.createChat).toHaveBeenCalledOnce())
    controller.abort()
    held.resolve({ id: "cancelled-persona" })
    expect(await result).toMatchObject({ name: "AbortError" })
    expect(fixture.mirror).not.toHaveBeenCalled()
    expect(fixture.metadata()).toEqual({})
  })

  it("publishes a current source without converting an absent mirror into H1 acceptance", async () => {
    const fixture = ownedFixture()
    fixture.mirror.mockResolvedValue(null)
    expect(await ensurePersonaServerChat(fixture.args)).toEqual({
      chatId: "created-persona",
      historyId: null,
      personaMemoryMode: "read_only"
    })
    expect(fixture.metadata()).toMatchObject({
      setServerChatId: "created-persona",
      setServerChatAssistantKind: "persona",
      setServerChatMetaLoaded: true
    })
    expect(fixture.invalidate).toHaveBeenCalledOnce()
  })
})
