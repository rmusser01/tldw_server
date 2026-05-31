import { describe, expect, it, vi } from "vitest"
import type { Message } from "@/store/option"
import type { ServerChatMessage } from "@/services/tldw/TldwApiClient"
import {
  applyAssistantPresentationToMessages,
  fetchAllServerChatMessages,
  mapServerChatMessagesToPlaygroundMessages,
  reportDeferredAssistantPresentationError,
  resolveServerChatAssistantIdentity,
  shouldCommitServerChatLoadResult,
  shouldPreserveLocalMessagesForServerLoad,
  shouldSkipLoadedServerChatReload
} from "@/hooks/chat/useServerChatLoader"
import {
  buildImageGenerationEventMirrorContent,
  IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE
} from "@/utils/image-generation-chat"

const createMessage = (overrides: Partial<Message> = {}): Message => ({
  isBot: false,
  name: "You",
  role: "user",
  message: "hello",
  sources: [],
  ...overrides
})

describe("shouldPreserveLocalMessagesForServerLoad", () => {
  it("preserves local messages while streaming", () => {
    const currentMessages = [createMessage({ message: "draft response" })]
    expect(
      shouldPreserveLocalMessagesForServerLoad({
        currentMessages,
        serverMessages: [],
        isStreaming: true,
        isProcessing: false
      })
    ).toBe(true)
  })

  it("preserves local messages when unsynced content exists", () => {
    const currentMessages = [
      createMessage({
        isBot: true,
        role: "assistant",
        message: "fresh assistant reply",
        serverMessageId: undefined
      })
    ]
    expect(
      shouldPreserveLocalMessagesForServerLoad({
        currentMessages,
        serverMessages: [],
        isStreaming: false,
        isProcessing: false
      })
    ).toBe(true)
  })

  it("preserves local messages when persisted IDs are missing in server snapshot", () => {
    const currentMessages = [
      createMessage({
        isBot: true,
        role: "assistant",
        message: "new persisted reply",
        serverMessageId: "srv-2"
      })
    ]
    const serverMessages = [
      createMessage({
        serverMessageId: "srv-1",
        id: "srv-1"
      })
    ]
    expect(
      shouldPreserveLocalMessagesForServerLoad({
        currentMessages,
        serverMessages,
        isStreaming: false,
        isProcessing: false
      })
    ).toBe(true)
  })

  it("does not preserve when local messages are fully reflected in server snapshot", () => {
    const currentMessages = [
      createMessage({
        isBot: true,
        role: "assistant",
        message: "synced reply",
        serverMessageId: "srv-1"
      })
    ]
    const serverMessages = [
      createMessage({
        isBot: true,
        role: "assistant",
        message: "synced reply",
        serverMessageId: "srv-1",
        id: "srv-1"
      })
    ]
    expect(
      shouldPreserveLocalMessagesForServerLoad({
        currentMessages,
        serverMessages,
        isStreaming: false,
        isProcessing: false
      })
    ).toBe(false)
  })

  it("does not preserve when the only unsynced local content is a synthetic character greeting", () => {
    const currentMessages = [
      createMessage({
        isBot: true,
        role: "assistant",
        message: "Greetings, traveler.",
        messageType: "character:greeting",
        serverMessageId: undefined
      })
    ]
    expect(
      shouldPreserveLocalMessagesForServerLoad({
        currentMessages,
        serverMessages: [],
        isStreaming: false,
        isProcessing: false
      })
    ).toBe(false)
  })
})

describe("shouldSkipLoadedServerChatReload", () => {
  it("returns false when current messages are empty even if same chat is marked loaded", () => {
    expect(
      shouldSkipLoadedServerChatReload({
        activeServerChatId: "chat-1",
        loadedChatId: "chat-1",
        loaded: true,
        currentMessages: []
      })
    ).toBe(false)
  })

  it("returns true when same chat is loaded and local messages exist", () => {
    expect(
      shouldSkipLoadedServerChatReload({
        activeServerChatId: "chat-1",
        loadedChatId: "chat-1",
        loaded: true,
        currentMessages: [createMessage({ message: "synced" })]
      })
    ).toBe(true)
  })
})

describe("shouldCommitServerChatLoadResult", () => {
  it("returns true when the same chat is still active and the same controller owns the load", () => {
    const controller = new AbortController()

    expect(
      shouldCommitServerChatLoadResult({
        requestedChatId: "chat-1",
        activeServerChatId: "chat-1",
        requestController: controller,
        activeController: controller
      })
    ).toBe(true)
  })

  it("returns false when a newer chat selection has replaced the active chat", () => {
    const controller = new AbortController()

    expect(
      shouldCommitServerChatLoadResult({
        requestedChatId: "chat-a",
        activeServerChatId: "chat-b",
        requestController: controller,
        activeController: controller
      })
    ).toBe(false)
  })

  it("returns false when the active controller no longer matches the request controller", () => {
    expect(
      shouldCommitServerChatLoadResult({
        requestedChatId: "chat-1",
        activeServerChatId: "chat-1",
        requestController: new AbortController(),
        activeController: new AbortController()
      })
    ).toBe(false)
  })
})

describe("resolveServerChatAssistantIdentity", () => {
  it("preserves persona-backed assistant identity from chat metadata", () => {
    expect(
      resolveServerChatAssistantIdentity({
        assistant_kind: "persona",
        assistant_id: "garden-helper",
        persona_memory_mode: "read_only",
        character_id: null
      } as any)
    ).toEqual({
      assistantKind: "persona",
      assistantId: "garden-helper",
      characterId: null,
      personaMemoryMode: "read_only"
    })
  })

  it("preserves read-write persona restore identity without legacy character fallback", () => {
    expect(
      resolveServerChatAssistantIdentity({
        assistant_kind: "persona",
        assistant_id: "garden-helper",
        persona_memory_mode: "read_write",
        character_id: null
      } as any)
    ).toEqual({
      assistantKind: "persona",
      assistantId: "garden-helper",
      characterId: null,
      personaMemoryMode: "read_write"
    })
  })

  it("ignores legacy character_id when persona assistant metadata is present", () => {
    expect(
      resolveServerChatAssistantIdentity({
        assistant_kind: "persona",
        assistant_id: "garden-helper",
        persona_memory_mode: "read_only",
        character_id: 42
      } as any)
    ).toEqual({
      assistantKind: "persona",
      assistantId: "garden-helper",
      characterId: null,
      personaMemoryMode: "read_only"
    })
  })

  it("keeps persona identity even when memory mode metadata is invalid", () => {
    expect(
      resolveServerChatAssistantIdentity({
        assistant_kind: "persona",
        assistant_id: "garden-helper",
        persona_memory_mode: "session",
        character_id: null
      } as any)
    ).toEqual({
      assistantKind: "persona",
      assistantId: "garden-helper",
      characterId: null,
      personaMemoryMode: null
    })
  })

  it("keeps persona identity without synthesizing memory mode when metadata is missing", () => {
    expect(
      resolveServerChatAssistantIdentity({
        assistant_kind: "persona",
        assistant_id: "garden-helper",
        character_id: null
      } as any)
    ).toEqual({
      assistantKind: "persona",
      assistantId: "garden-helper",
      characterId: null,
      personaMemoryMode: null
    })
  })

  it("treats chats with only character_id and no tracked source as plain", () => {
    expect(
      resolveServerChatAssistantIdentity({
        character_id: 42
      } as any)
    ).toEqual({
      assistantKind: null,
      assistantId: null,
      characterId: null,
      personaMemoryMode: null
    })
  })

  it("backfills character identity from legacy tracked chat sources", () => {
    expect(
      resolveServerChatAssistantIdentity({
        character_id: 42,
        source: "webui-character-chat"
      } as any)
    ).toEqual({
      assistantKind: "character",
      assistantId: "42",
      characterId: 42,
      personaMemoryMode: null
    })
  })
})

const createServerMessage = (
  overrides: Partial<ServerChatMessage> = {}
): ServerChatMessage => ({
  id: "msg-1",
  role: "assistant",
  content: "hello",
  created_at: "2026-02-20T00:00:00.000Z",
  ...overrides
})

describe("fetchAllServerChatMessages", () => {
  it("fetches all pages so later-page greeting messages are included", async () => {
    const greeting = createServerMessage({
      id: "msg-2",
      role: "assistant",
      content: "Greetings, traveler.",
      created_at: "2026-02-20T00:00:02.000Z",
      metadata_extra: { message_type: "character:greeting" }
    })
    const userMessage = createServerMessage({
      id: "msg-1",
      role: "user",
      content: "Hello there",
      created_at: "2026-02-20T00:00:01.000Z"
    })
    const assistantReply = createServerMessage({
      id: "msg-3",
      role: "assistant",
      content: "How can I help?",
      created_at: "2026-02-20T00:00:03.000Z"
    })

    const pages = new Map<number, ServerChatMessage[]>([
      [0, [userMessage]],
      [1, [greeting, assistantReply]],
      [3, []]
    ])

    const messages = await fetchAllServerChatMessages(
      async ({ limit, offset }) => {
        expect(limit).toBe(1)
        return pages.get(offset) ?? []
      },
      {
        limit: 1,
        maxPages: 10
      }
    )

    expect(messages.map((message) => message.id)).toEqual([
      "msg-1",
      "msg-2",
      "msg-3"
    ])
    expect(messages.some((message) => message.content.includes("Greetings"))).toBe(
      true
    )
  })

  it("deduplicates repeated message ids across paginated responses", async () => {
    const first = createServerMessage({ id: "msg-1" })
    const second = createServerMessage({ id: "msg-2" })
    const duplicateSecond = createServerMessage({ id: "msg-2" })
    const third = createServerMessage({ id: "msg-3" })

    const pages = new Map<number, ServerChatMessage[]>([
      [0, [first, second]],
      [2, [duplicateSecond, third]],
      [4, []]
    ])

    const messages = await fetchAllServerChatMessages(
      async ({ limit, offset }) => {
        expect(limit).toBe(2)
        return pages.get(offset) ?? []
      },
      {
        limit: 2,
        maxPages: 10
      }
    )

    expect(messages.map((message) => message.id)).toEqual([
      "msg-1",
      "msg-2",
      "msg-3"
    ])
  })
})

describe("mapServerChatMessagesToPlaygroundMessages", () => {
  it("maps mirrored image event messages into assistant image event cards", () => {
    const mirroredContent = buildImageGenerationEventMirrorContent({
      kind: "image_generation_event",
      version: 1,
      eventId: "evt-1",
      request: {
        prompt: "portrait of Lana, cinematic lighting",
        backend: "flux-test-backend",
        width: 768,
        height: 1024
      },
      source: "generate-modal",
      imageDataUrl: "data:image/png;base64,abc123"
    })
    const mapped = mapServerChatMessagesToPlaygroundMessages({
      serverMessages: [
        createServerMessage({
          id: "srv-img-1",
          role: "assistant",
          content: mirroredContent,
          created_at: "2026-02-20T00:00:02.000Z"
        })
      ],
      assistantName: "Lana",
      characterId: 42
    })

    expect(mapped).toHaveLength(1)
    expect(mapped[0].messageType).toBe(IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE)
    expect(mapped[0].message).toBe("")
    expect(mapped[0].images).toEqual(["data:image/png;base64,abc123"])
    expect(mapped[0].generationInfo?.image_generation?.request?.backend).toBe(
      "flux-test-backend"
    )
    expect(mapped[0].generationInfo?.image_generation?.sync?.status).toBe("synced")
  })
})

describe("applyAssistantPresentationToMessages", () => {
  it("updates placeholder assistant labels after deferred enrichment", () => {
    const result = applyAssistantPresentationToMessages({
      messages: [
        createMessage({
          isBot: true,
          role: "assistant",
          name: "Assistant",
          modelName: "Assistant",
          modelImage: undefined
        }),
        createMessage({
          isBot: true,
          role: "assistant",
          name: "Narrator",
          modelName: "Narrator",
          modelImage: "existing-image"
        }),
        createMessage({
          isBot: false,
          role: "user",
          name: "You"
        })
      ],
      assistantName: "Archivist",
      assistantAvatarUrl: "avatar.png"
    })

    expect(result[0]).toMatchObject({
      name: "Archivist",
      modelName: "Archivist",
      modelImage: "avatar.png"
    })
    expect(result[1]).toMatchObject({
      name: "Narrator",
      modelName: "Narrator",
      modelImage: "existing-image"
    })
    expect(result[2]).toMatchObject({
      name: "You"
    })
  })

  it("applies generic Persona fallback presentation without rewriting explicit speaker labels", () => {
    const result = applyAssistantPresentationToMessages({
      messages: [
        createMessage({
          isBot: true,
          role: "assistant",
          name: "Assistant",
          modelName: "Assistant"
        }),
        createMessage({
          isBot: true,
          role: "assistant",
          name: "Garden Helper",
          modelName: "Garden Helper"
        })
      ],
      assistantName: "Persona",
      assistantAvatarUrl: null
    })

    expect(result[0]).toMatchObject({
      name: "Persona",
      modelName: "Persona",
      modelImage: undefined
    })
    expect(result[1]).toMatchObject({
      name: "Garden Helper",
      modelName: "Garden Helper"
    })
  })
})

describe("reportDeferredAssistantPresentationError", () => {
  it("logs deferred assistant hydration failures with contextual metadata", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined)
    const failure = new Error("character lookup failed")

    reportDeferredAssistantPresentationError({
      stage: "character-profile",
      assistantKind: "character",
      assistantId: "42",
      characterId: 42,
      error: failure
    })

    expect(warnSpy).toHaveBeenCalledWith(
      "[useServerChatLoader] Deferred assistant presentation failed",
      expect.objectContaining({
        stage: "character-profile",
        assistantKind: "character",
        assistantId: "42",
        characterId: 42,
        error: failure
      })
    )

    warnSpy.mockRestore()
  })

  it("logs persona profile fallback failures while preserving assistant metadata", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined)
    const failure = new Error("persona lookup failed")

    reportDeferredAssistantPresentationError({
      stage: "persona-profile",
      assistantKind: "persona",
      assistantId: "garden-helper",
      characterId: null,
      error: failure
    })

    expect(warnSpy).toHaveBeenCalledWith(
      "[useServerChatLoader] Deferred assistant presentation failed",
      expect.objectContaining({
        stage: "persona-profile",
        assistantKind: "persona",
        assistantId: "garden-helper",
        characterId: null,
        error: failure
      })
    )

    warnSpy.mockRestore()
  })
})
