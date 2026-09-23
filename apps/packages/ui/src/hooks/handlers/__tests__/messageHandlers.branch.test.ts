import { beforeEach, describe, expect, it, vi } from "vitest"
import type { ChatHistory, Message } from "@/store/option"

const mockTldwClient = vi.hoisted(() => ({
  initialize: vi.fn(),
  getChat: vi.fn(),
  createChat: vi.fn(),
  addChatMessage: vi.fn()
}))
const ownership = vi.hoisted(() => ({
  revision: 0,
  loadSnapshot: vi.fn(),
  release: vi.fn(),
  controller: new AbortController()
}))
const mirror = vi.hoisted(() => ({ link: vi.fn(), reconcile: vi.fn() }))
vi.mock("@/db/dexie/server-chat-mirror", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/db/dexie/server-chat-mirror")>()),
  linkServerChatMirror: (...args: unknown[]) => mirror.link(...args),
  reconcileServerChatMirror: (...args: unknown[]) => mirror.reconcile(...args)
}))
vi.mock("@/store/playground-session", () => ({
  usePlaygroundSessionStore: {
    getState: () => ({ restoreRevision: ownership.revision })
  }
}))
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: (...args: unknown[]) =>
    ownership.loadSnapshot(...args)
}))
vi.mock("@/db/dexie/chat-persistence-transaction", () => ({
  runChatPersistenceTransaction: async (
    _signal: AbortSignal,
    operation: () => Promise<unknown>
  ) => operation()
}))

vi.mock("@/db/dexie/helpers", () => ({
  deleteChatForEdit: vi.fn(),
  formatToChatHistory: vi.fn((messages: unknown) => messages),
  formatToMessage: vi.fn((messages: unknown) => messages),
  saveHistory: vi.fn(),
  saveMessage: vi.fn(),
  updateMessageByIndex: vi.fn()
}))

vi.mock("@/db/dexie/branch", () => ({
  generateBranchMessage: vi.fn()
}))

vi.mock("@/db", () => ({
  getPromptById: vi.fn(),
  getSessionFiles: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: mockTldwClient
}))

vi.mock("@/utils/conversation-state", () => ({
  normalizeConversationState: (value?: string | null) => {
    if (value === "resolved" || value === "backlog" || value === "non-viable") {
      return value
    }
    return "in-progress"
  }
}))

import { createBranchMessage } from "../messageHandlers"
import { saveHistory, saveMessage } from "@/db/dexie/helpers"

describe("createBranchMessage", () => {
  const requestScope = {
    config: { serverUrl: "http://chat.test", authMode: "multi-user" },
    userId: "alice"
  }
  beforeEach(() => {
    vi.clearAllMocks()
    ownership.revision = 0
    ownership.controller = new AbortController()
    mirror.link.mockResolvedValue("fork-local-history")
    mirror.reconcile.mockResolvedValue({ localIds: new Map(), rows: [] })
    ownership.loadSnapshot.mockReset().mockImplementation(async () => ({
      requestScope,
      scopeSignal: ownership.controller.signal,
      scopeInvalidatedSignal: ownership.controller.signal,
      release: ownership.release
    }))
    vi.mocked(saveHistory).mockResolvedValue({
      id: "owned-snapshot-branch", title: "Branch · msg #1",
      is_rag: false, createdAt: 1
    })
    vi.mocked(saveMessage).mockImplementation(
      async (data) => ({ ...data, id: "saved-message", createdAt: 1 })
    )
  })

  const localOptions = () => ({
    notification: {
      error: vi.fn(), warning: vi.fn(), success: vi.fn(),
      info: vi.fn(), open: vi.fn(), destroy: vi.fn()
    },
    historyId: null,
    setHistory: vi.fn(),
    setMessages: vi.fn(),
    setHistoryId: vi.fn(),
    messages: [
      {
        isBot: false,
        name: "You",
        message: "Alice private draft conversation",
        sources: []
      }
    ] as Message[]
  })

  it("stamps a snapshot-only branch with its verified source owner", async () => {
    const options = localOptions()
    expect(await createBranchMessage(options)(0)).toBe("owned-snapshot-branch")
    expect(saveHistory).toHaveBeenCalledWith(
      "Branch · msg #1",
      false,
      "branch",
      undefined,
      undefined,
      requestScope
    )
    expect(ownership.release).toHaveBeenCalledTimes(1)
  })

  it("does not assign an old snapshot to the account selected while ownership resolves", async () => {
    const options = localOptions()
    ownership.loadSnapshot.mockImplementationOnce(async () => {
      ownership.revision++
      return {
        requestScope: { ...requestScope, userId: "bob" },
        scopeSignal: ownership.controller.signal,
        scopeInvalidatedSignal: ownership.controller.signal,
        release: ownership.release
      }
    })
    expect(await createBranchMessage(options)(0)).toBeNull()
    expect(saveHistory).not.toHaveBeenCalled()
    expect(options.setMessages).not.toHaveBeenCalled()
  })

  it("prefers the parent server chat character_id when local characterId is stale", async () => {
    mockTldwClient.initialize.mockResolvedValue(undefined)
    mockTldwClient.getChat.mockResolvedValue({
      id: "parent-chat-id",
      title: "Parent Chat",
      character_id: 2,
      state: "in-progress"
    })
    mockTldwClient.createChat.mockImplementation(async (payload: any) => {
      if (payload.character_id !== 2) {
        throw new Error(
          `unexpected character_id: ${String(payload.character_id)}`
        )
      }
      return {
        id: "new-branch-chat-id",
        character_id: 2,
        state: "in-progress",
        title: payload.title
      }
    })
    mockTldwClient.addChatMessage.mockResolvedValue({
      id: "msg-1"
    })

    const notification = {
      error: vi.fn(),
      warning: vi.fn()
    } as any

    const setMessages = vi.fn()
    const setHistory = vi.fn()
    const setHistoryId = vi.fn()
    const setServerChatId = vi.fn()
    const setServerChatState = vi.fn()
    const setServerChatVersion = vi.fn()
    const setServerChatTitle = vi.fn()
    const setServerChatCharacterId = vi.fn()
    const setServerChatMetaLoaded = vi.fn()
    const setServerChatTopic = vi.fn()
    const setServerChatClusterId = vi.fn()
    const setServerChatSource = vi.fn()
    const setServerChatExternalRef = vi.fn()

    const messages = [
      {
        isBot: false,
        name: "You",
        message: "hello",
        sources: []
      }
    ] as Message[]

    const history = [
      {
        role: "user",
        content: "hello"
      }
    ] as ChatHistory

    const branchMessage = createBranchMessage({
      notification,
      setMessages,
      setHistory,
      historyId: "local-history-id",
      setHistoryId,
      serverChatId: "parent-chat-id",
      scope: { type: "workspace", workspaceId: "workspace-1" },
      setServerChatId,
      setServerChatState,
      setServerChatVersion,
      setServerChatTitle,
      setServerChatCharacterId,
      setServerChatMetaLoaded,
      setServerChatTopic,
      setServerChatClusterId,
      setServerChatSource,
      setServerChatExternalRef,
      characterId: "stale-local-character-id",
      chatTitle: "local title",
      serverChatState: "in-progress",
      messages,
      history,
      serverOnly: true
    })

    const result = await branchMessage(0)

    expect(result).toBe("new-branch-chat-id")
    expect(mockTldwClient.getChat).toHaveBeenCalledWith("parent-chat-id", {
      scope: { type: "workspace", workspaceId: "workspace-1" },
      signal: ownership.controller.signal,
      requestScope
    })
    expect(mockTldwClient.createChat).toHaveBeenCalledWith(
      expect.objectContaining({
        parent_conversation_id: "parent-chat-id",
        character_id: 2,
        state: "in-progress"
      }),
      {
        scope: { type: "workspace", workspaceId: "workspace-1" },
        signal: ownership.controller.signal,
        requestScope
      }
    )
    expect(mockTldwClient.addChatMessage).toHaveBeenCalledWith(
      "new-branch-chat-id",
      expect.objectContaining({
        role: "user",
        content: "hello"
      }),
      {
        scope: { type: "workspace", workspaceId: "workspace-1" },
        signal: ownership.controller.signal,
        requestScope
      }
    )
    expect(notification.error).not.toHaveBeenCalled()
  })
  it.each([0, 2])(
    "forked prefix reconciles once with %i later turn rows",
    async (laterRows) => {
      const { reconcileServerChatMessages } =
        await import("@/db/dexie/server-chat-mirror")
      const prefix: Message[] = [
        "Greeting",
        "First question",
        "First answer",
        "Retry question"
      ].map((message, index) => ({
        id: `local-${index}`,
        serverMessageId: `parent-${index}`,
        serverMessageVersion: 1,
        message,
        isBot: index % 2 === 0,
        role: index % 2 === 0 ? "assistant" : "user",
        name: index % 2 === 0 ? "Character" : "You",
        sources: [],
        messageType: index === 0 ? "character:greeting" : undefined
      }))
      const history: ChatHistory = prefix.map((row) => ({
        role: row.role!,
        content: row.message
      }))
      const canonicalPrefix: Message[] = []
      mockTldwClient.initialize.mockResolvedValue(undefined)
      mockTldwClient.getChat.mockResolvedValue({
        id: "parent-chat",
        character_id: 2,
        title: "Parent"
      })
      mockTldwClient.createChat.mockResolvedValue({
        id: "fork-chat",
        character_id: 2,
        title: "Fork"
      })
      mockTldwClient.addChatMessage.mockImplementation(
        async (_chat, payload) => {
          const index = canonicalPrefix.length
          const id = `fork-${index}`
          canonicalPrefix.push({
            ...prefix[index],
            id,
            serverMessageId: id,
            message: payload.content
          })
          return { id, version: 1 }
        }
      )
      let visible: Message[] = []
      const options = {
        ...localOptions(),
        historyId: "parent-local-history",
        serverChatId: "parent-chat",
        characterId: 2,
        messages: prefix,
        history,
        serverOnly: true,
        setMessages: (rows: Message[]) => {
          visible = rows
        }
      }
      expect(await createBranchMessage(options)(3)).toBe("fork-chat")
      expect(canonicalPrefix).toHaveLength(4)
      const retry: Message = {
        id: "local-retry",
        serverMessageId: "fork-retry",
        serverMessageVersion: 1,
        message: "Retry answer",
        isBot: true,
        role: "assistant",
        name: "Character",
        sources: []
      }
      const later: Message[] = laterRows
        ? [
            {
              id: "local-new-user",
              serverMessageId: "fork-new-user",
              message: "Arithmetic question",
              isBot: false,
              role: "user",
              name: "You",
              sources: []
            },
            {
              id: "local-new-answer",
              serverMessageId: "fork-new-answer",
              message: "Four",
              isBot: true,
              role: "assistant",
              name: "Character",
              sources: []
            }
          ]
        : []
      const incoming = [...canonicalPrefix, retry, ...later]
      const merged = reconcileServerChatMessages(
        [...visible, retry, ...later],
        incoming
      )
      expect(new Set(incoming.map((row) => row.serverMessageId)).size).toBe(
        5 + laterRows
      )
      expect(merged.map((row) => row.serverMessageId)).toEqual(
        incoming.map((row) => row.serverMessageId)
      )
    }
  )
  it.each([0, 2])(
    "existing same-conversation receipt control with %i later rows",
    async (laterRows) => {
      const { reconcileServerChatMessages } =
        await import("@/db/dexie/server-chat-mirror")
      const incoming: Message[] = Array.from(
        { length: 5 + laterRows },
        (_, index) => ({
          id: `local-${index}`,
          serverMessageId: `server-${index}`,
          message: `Unique row ${index}`,
          isBot: index % 2 === 0,
          role: index % 2 === 0 ? "assistant" : "user",
          name: "Test",
          sources: []
        })
      )
      expect(reconcileServerChatMessages(incoming, incoming)).toHaveLength(
        5 + laterRows
      )
    }
  )

  it.each(["revision", "owner"] as const)(
    "held server create cannot publish after %s replacement",
    async (replacement) => {
      mockTldwClient.initialize.mockResolvedValue(undefined)
      mockTldwClient.getChat.mockResolvedValue({
        id: "parent-chat",
        character_id: 2,
        title: "Parent"
      })
      let release!: (value: Record<string, unknown>) => void
      mockTldwClient.createChat.mockReturnValue(
        new Promise((resolve) => {
          release = resolve
        })
      )
      mockTldwClient.addChatMessage.mockResolvedValue({
        id: "fork-user",
        version: 1
      })
      const options = {
        ...localOptions(),
        serverChatId: "parent-chat",
        characterId: 2,
        history: [{ role: "user" as const, content: "Question" }],
        serverOnly: true,
        setServerChatId: vi.fn()
      }
      const pending = createBranchMessage(options)(0)
      await vi.waitFor(() =>
        expect(mockTldwClient.createChat).toHaveBeenCalled()
      )
      if (replacement === "revision") ownership.revision++
      else ownership.controller.abort()
      release({ id: "fork-chat", character_id: 2, title: "Fork" })
      expect(await pending).toBeNull()
      expect(options.setServerChatId).not.toHaveBeenCalled()
      expect(options.setMessages).not.toHaveBeenCalled()
    }
  )

  it("returns the acknowledged fork snapshot to Retry without changing parent rows", async () => {
    const { createRegenerateLastMessage } = await import("../messageHandlers")
    const user: Message = {
      id: "local-user",
      serverMessageId: "parent-user",
      isBot: false,
      role: "user",
      name: "You",
      message: "Question",
      sources: []
    }
    const answer: Message = {
      id: "local-answer",
      serverMessageId: "parent-answer",
      isBot: true,
      role: "assistant",
      name: "Character",
      message: "Answer",
      sources: []
    }
    const rows = [user, answer]
    const history: ChatHistory = rows.map((row) => ({
      role: row.role!,
      content: row.message
    }))
    mockTldwClient.initialize.mockResolvedValue(undefined)
    mockTldwClient.getChat.mockResolvedValue({
      id: "parent-chat",
      character_id: 2
    })
    mockTldwClient.createChat.mockResolvedValue({
      id: "fork-chat",
      character_id: 2
    })
    mockTldwClient.addChatMessage.mockResolvedValue({
      id: "fork-user",
      version: 1
    })
    const options = {
      ...localOptions(),
      messages: rows,
      history,
      serverChatId: "parent-chat",
      characterId: 2,
      serverOnly: true
    }
    const branch = createBranchMessage(options)
    const onSubmit = vi.fn()
    await createRegenerateLastMessage({
      validateBeforeSubmitFn: () => true,
      messages: rows,
      history,
      setHistory: vi.fn(),
      setMessages: vi.fn(),
      onSubmit,
      beforeSubmit: async () => {
        let copied: Message[] | undefined
        const chatId = await branch(0, (snapshot) => {
          copied = snapshot.messages
        })
        return {
          messages: copied,
          submitExtras: { serverChatIdOverride: chatId }
        }
      }
    })()
    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        messages: [
          expect.objectContaining({
            id: "fork-user",
            serverMessageId: "fork-user"
          })
        ],
        serverChatIdOverride: "fork-chat"
      })
    )
    expect(user.serverMessageId).toBe("parent-user")
  })

  const buildBranchProjectionFixture = async () => {
    const actual =
      await vi.importActual<typeof import("@/db/dexie/helpers")>(
        "@/db/dexie/helpers"
      )
    const diagnostic = {
      mode: "rag",
      grounded: false,
      reason: "selected_source_evidence_not_found"
    }
    const image = "data:image/png;base64,aW1hZ2U="
    const rows = [
      { id: "s", role: "system", content: "System instructions" },
      {
        id: "g",
        role: "assistant",
        content: "Greeting",
        messageType: "character:greeting"
      },
      { id: "u", role: "user", content: "Earlier question" },
      {
        id: "a-old",
        role: "assistant",
        content: "Old answer",
        parent_message_id: "u",
        serverMessageId: "server-a-old"
      },
      {
        id: "a",
        role: "assistant",
        content: "<think>Internal reasoning</think>Latest answer",
        parent_message_id: "u",
        serverMessageId: "server-a",
        modelName: "Model",
        modelImage: "model.png",
        metadataExtra: { mood_label: "friendly" },
        generationInfo: { custom: "keep" },
        reasoning_time_taken: 3
      },
      {
        id: "du",
        role: "user",
        content: "Undispatched diagnostic question",
        generationInfo: diagnostic
      },
      {
        id: "da",
        role: "assistant",
        content: "No source evidence",
        parent_message_id: "du",
        generationInfo: diagnostic
      },
      { id: "iu", role: "user", content: "", images: [image] },
      {
        id: "ia",
        role: "assistant",
        content: "Answer AFTER selected branch point",
        parent_message_id: "iu"
      }
    ].map((row, index) => ({
      history_id: "parent-local",
      name:
        row.role === "user"
          ? "You"
          : row.role === "system"
            ? "System"
            : "Character",
      createdAt: index,
      images: [],
      ...row
    })) as Parameters<typeof actual.formatToMessage>[0]
    return {
      messages: actual.formatToMessage(rows),
      history: actual.formatToChatHistory(rows),
      image
    }
  }
  it("actual formatter projection preserves metadata but excludes diagnostic pairs only from history", async () => {
    const { messages, history, image } = await buildBranchProjectionFixture()
    const { excludeLocalRagDiagnostics } =
      await import("@/utils/local-rag-diagnostic")
    expect(messages.map((row) => row.id)).toEqual([
      "s",
      "g",
      "u",
      "a",
      "du",
      "da",
      "iu",
      "ia"
    ])
    expect(history.map((row) => row.content)).toEqual(
      excludeLocalRagDiagnostics(messages).map((row) => row.message)
    )
    expect(messages.find((row) => row.id === "a")).toMatchObject({
      modelName: "Model",
      modelImage: "model.png",
      metadataExtra: { mood_label: "friendly" },
      generationInfo: { custom: "keep" },
      reasoning_time_taken: 3,
      activeVariantIndex: 1,
      variants: [
        { serverMessageId: "server-a-old" },
        { serverMessageId: "server-a" }
      ]
    })
    expect(history.find((row) => row.content === "")).toMatchObject({
      images: [image]
    })
  })
  it.each(["boundary", "image"] as const)(
    "branch respects projected %s with system, variants and diagnostics",
    async (check) => {
      const { messages, history } = await buildBranchProjectionFixture()
      mockTldwClient.initialize.mockResolvedValue(undefined)
      mockTldwClient.getChat.mockResolvedValue({
        id: "parent-chat",
        character_id: 2,
        title: "Parent"
      })
      mockTldwClient.createChat.mockResolvedValue({
        id: "fork-chat",
        character_id: 2,
        title: "Fork"
      })
      let counter = 0
      mockTldwClient.addChatMessage.mockImplementation(async () => ({
        id: `fork-${counter++}`,
        version: 1
      }))
      const branchPoint = messages.findIndex((row) => row.id === "iu")
      expect(
        await createBranchMessage({
          ...localOptions(),
          historyId: "parent-local",
          serverChatId: "parent-chat",
          characterId: 2,
          messages,
          history,
          serverOnly: true
        })(branchPoint)
      ).toBe("fork-chat")
      const payloads = mockTldwClient.addChatMessage.mock.calls.map(
        (call) => call[1]
      )
      if (check === "boundary")
        expect(
          payloads.some(
            (payload) =>
              payload.content === "Answer AFTER selected branch point"
          )
        ).toBe(false)
      else
        expect(payloads).toContainEqual(
          expect.objectContaining({
            role: "user",
            image_base64: expect.any(String)
          })
        )
      expect(
        payloads.find((payload) =>
          payload.content?.includes("Internal reasoning")
        )?.content
      ).toBe(
        history.find((row) => row.content.includes("Internal reasoning"))
          ?.content
      )
    }
  )

  it("real Playground history setter preserves published server branch metadata", async () => {
    const { useStoreMessageOption } = await import("@/store/option")
    useStoreMessageOption.setState({
      historyId: "parent-local",
      serverChatId: "parent-chat",
      serverChatCharacterId: 2,
      serverChatTitle: "Parent",
      serverChatMetaLoaded: true
    })
    const state = useStoreMessageOption.getState()
    const user: Message = {
      id: "user",
      serverMessageId: "parent-user",
      isBot: false,
      role: "user",
      name: "You",
      message: "Question",
      sources: []
    }
    mockTldwClient.getChat.mockResolvedValue({
      id: "parent-chat",
      character_id: 2,
      title: "Parent"
    })
    mockTldwClient.createChat.mockResolvedValue({
      id: "fork-chat",
      character_id: 2,
      title: "Fork"
    })
    mockTldwClient.addChatMessage.mockResolvedValue({
      id: "fork-user",
      version: 1
    })
    expect(
      await createBranchMessage({
        ...localOptions(),
        historyId: "parent-local",
        serverChatId: "parent-chat",
        characterId: 2,
        messages: [user],
        history: [{ role: "user", content: "Question" }],
        serverOnly: true,
        setHistoryId: state.setHistoryId,
        setServerChatId: state.setServerChatId,
        setServerChatTitle: state.setServerChatTitle,
        setServerChatCharacterId: state.setServerChatCharacterId,
        setServerChatMetaLoaded: state.setServerChatMetaLoaded
      })(0)
    ).toBe("fork-chat")
    expect(useStoreMessageOption.getState()).toMatchObject({
      historyId: "fork-local-history",
      serverChatId: "fork-chat",
      serverChatCharacterId: 2,
      serverChatTitle: "Fork",
      serverChatMetaLoaded: true
    })
  })
})
