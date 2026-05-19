import React from "react"
import { renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { Character } from "@/types/character"
import type { ChatHistory, Message } from "@/store/option/types"
import {
  buildGreetingOptionsFromEntries,
  buildGreetingsChecksumFromOptions,
  collectGreetingEntries
} from "@/utils/character-greetings"
import { useCharacterGreeting } from "../useCharacterGreeting"

const mocks = vi.hoisted(() => ({
  settings: null as Record<string, unknown> | null,
  updateSettings: vi.fn(),
  initialize: vi.fn(),
  getCharacter: vi.fn(),
  selectedCharacterStorageGet: vi.fn()
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [""]
}))

vi.mock("@/db/dexie/helpers", () => ({
  generateID: () => "generated-id"
}))

vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({
    settings: mocks.settings,
    updateSettings: mocks.updateSettings,
    chatKey: "chat:test"
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: mocks.initialize,
    getCharacter: mocks.getCharacter
  }
}))

vi.mock("@/utils/selected-character-storage", () => ({
  SELECTED_CHARACTER_STORAGE_KEY: "selectedCharacter",
  selectedCharacterStorage: {
    get: mocks.selectedCharacterStorageGet
  },
  parseSelectedCharacterValue: (value: unknown) =>
    value && typeof value === "object" ? value : null
}))

const applyMessageUpdate = (
  current: Message[],
  next: Message[] | ((prev: Message[]) => Message[])
) => (typeof next === "function" ? next(current) : next)

const applyHistoryUpdate = (
  current: ChatHistory,
  next: ChatHistory | ((prev: ChatHistory) => ChatHistory)
) => (typeof next === "function" ? next(current) : next)

const createDeferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

describe("useCharacterGreeting", () => {
  beforeEach(() => {
    mocks.settings = {
      greetingEnabled: true,
      greetingSelectionId: null,
      greetingsChecksum: null,
      useCharacterDefault: false
    }
    mocks.updateSettings.mockReset()
    mocks.updateSettings.mockResolvedValue(null)
    mocks.initialize.mockReset()
    mocks.initialize.mockResolvedValue(null)
    mocks.getCharacter.mockReset()
    mocks.selectedCharacterStorageGet.mockReset()
    mocks.selectedCharacterStorageGet.mockResolvedValue(null)
  })

  it("schedules greeting updates with React.startTransition", async () => {
    const transitionSpy = vi.spyOn(React, "startTransition")
    const selectedCharacter = {
      id: "char-1",
      name: "Guide",
      greeting: "Welcome",
      alternateGreetings: ["Good to see you"]
    } as Character
    let messageState: Message[] = []
    let historyState: ChatHistory = []

    const setMessages = vi.fn(
      (next: Message[] | ((prev: Message[]) => Message[])) => {
        messageState = applyMessageUpdate(messageState, next)
      }
    )
    const setHistory = vi.fn(
      (next: ChatHistory | ((prev: ChatHistory) => ChatHistory)) => {
        historyState = applyHistoryUpdate(historyState, next)
      }
    )
    const setSelectedCharacter = vi.fn()

    renderHook(() =>
      useCharacterGreeting({
        playgroundReady: true,
        selectedCharacter,
        serverChatId: null,
        historyId: "history-1",
        messagesLength: messageState.length,
        setMessages,
        setHistory,
        setSelectedCharacter
      })
    )

    await waitFor(() => {
      expect(messageState).toHaveLength(1)
      expect(historyState).toHaveLength(1)
    })

    expect(transitionSpy).toHaveBeenCalled()
    transitionSpy.mockRestore()
  })

  it("applies the latest selection when fetched greetings resolve", async () => {
    const initialCharacter = {
      id: "char-42",
      name: "Narrator",
      greeting: "Initial greeting"
    } as Character
    const fetchedCharacter = {
      ...initialCharacter,
      alternateGreetings: ["Selected alternative"]
    } as Character
    const fetchedOptions = buildGreetingOptionsFromEntries(
      collectGreetingEntries(fetchedCharacter)
    )
    const fetchedChecksum = buildGreetingsChecksumFromOptions(fetchedOptions)
    const selectedAlternativeId = fetchedOptions[1]?.id
    if (!selectedAlternativeId) {
      throw new Error("Test setup failed: missing alternate greeting option")
    }

    mocks.settings = {
      greetingEnabled: true,
      greetingSelectionId: null,
      greetingsChecksum: null,
      useCharacterDefault: true
    }

    const deferred = createDeferred<Character>()
    mocks.getCharacter.mockReturnValue(deferred.promise)

    let messageState: Message[] = []
    let historyState: ChatHistory = []
    const setMessages = vi.fn(
      (next: Message[] | ((prev: Message[]) => Message[])) => {
        messageState = applyMessageUpdate(messageState, next)
      }
    )
    const setHistory = vi.fn(
      (next: ChatHistory | ((prev: ChatHistory) => ChatHistory)) => {
        historyState = applyHistoryUpdate(historyState, next)
      }
    )
    const setSelectedCharacter = vi.fn()

    const { rerender } = renderHook(() =>
      useCharacterGreeting({
        playgroundReady: true,
        selectedCharacter: initialCharacter,
        serverChatId: null,
        historyId: "history-1",
        messagesLength: messageState.length,
        setMessages,
        setHistory,
        setSelectedCharacter
      })
    )

    await waitFor(() => {
      expect(messageState[0]?.message).toBe("Initial greeting")
    })

    mocks.settings = {
      greetingEnabled: true,
      greetingSelectionId: selectedAlternativeId,
      greetingsChecksum: fetchedChecksum,
      useCharacterDefault: false
    }
    rerender()

    deferred.resolve(fetchedCharacter)

    await waitFor(() => {
      expect(messageState[0]?.message).toBe("Selected alternative")
      expect(historyState[0]?.content).toBe("Selected alternative")
    })
  })

  it("honors legacy index-based greeting selection ids", async () => {
    const selectedCharacter = {
      id: "char-7",
      name: "Guide",
      greeting: "Primary",
      alternateGreetings: ["Alternate greeting"]
    } as Character

    mocks.settings = {
      greetingEnabled: true,
      greetingSelectionId: "greeting:1:selected",
      greetingsChecksum: buildGreetingsChecksumFromOptions(
        buildGreetingOptionsFromEntries(collectGreetingEntries(selectedCharacter))
      ),
      useCharacterDefault: false
    }

    let messageState: Message[] = []
    const setMessages = vi.fn(
      (next: Message[] | ((prev: Message[]) => Message[])) => {
        messageState = applyMessageUpdate(messageState, next)
      }
    )
    const setHistory = vi.fn()
    const setSelectedCharacter = vi.fn()

    renderHook(() =>
      useCharacterGreeting({
        playgroundReady: true,
        selectedCharacter,
        serverChatId: null,
        historyId: "history-legacy-selection",
        messagesLength: messageState.length,
        setMessages,
        setHistory,
        setSelectedCharacter
      })
    )

    await waitFor(() => {
      expect(messageState[0]?.message).toBe("Alternate greeting")
    })
  })

  it("does not inject greeting messages for server chats", async () => {
    const selectedCharacter = {
      id: "char-server",
      name: "Server Guide",
      greeting: "Server hello"
    } as Character
    let messageState: Message[] = []
    let historyState: ChatHistory = []
    const setMessages = vi.fn(
      (next: Message[] | ((prev: Message[]) => Message[])) => {
        messageState = applyMessageUpdate(messageState, next)
      }
    )
    const setHistory = vi.fn(
      (next: ChatHistory | ((prev: ChatHistory) => ChatHistory)) => {
        historyState = applyHistoryUpdate(historyState, next)
      }
    )
    const setSelectedCharacter = vi.fn()

    renderHook(() =>
      useCharacterGreeting({
        playgroundReady: true,
        selectedCharacter,
        serverChatId: "srv-chat-1",
        historyId: "history-server",
        messagesLength: messageState.length,
        setMessages,
        setHistory,
        setSelectedCharacter
      })
    )

    await waitFor(() => {
      expect(setMessages).not.toHaveBeenCalled()
    })
    expect(messageState).toHaveLength(0)
    expect(historyState).toHaveLength(0)
  })

  it("does not sync selected character from storage during server chat load", async () => {
    mocks.selectedCharacterStorageGet.mockResolvedValue({
      id: "char-stale",
      name: "Stale character",
      greeting: "Old greeting"
    })
    const setMessages = vi.fn()
    const setHistory = vi.fn()
    const setSelectedCharacter = vi.fn()

    renderHook(() =>
      useCharacterGreeting({
        playgroundReady: true,
        selectedCharacter: null,
        serverChatId: "srv-chat-2",
        historyId: "history-server",
        messagesLength: 0,
        setMessages,
        setHistory,
        setSelectedCharacter
      })
    )

    await waitFor(() => {
      expect(setSelectedCharacter).not.toHaveBeenCalled()
    })
    expect(mocks.selectedCharacterStorageGet).not.toHaveBeenCalled()
  })

  it("does not refetch a character after merging fetched details for the same id", async () => {
    const initialCharacter = {
      id: "char-loop",
      name: "Loop Guide",
      greeting: "Hello"
    } as Character
    const fetchedCharacter = {
      ...initialCharacter,
      avatar_url: "https://example.com/avatar.png",
      alternateGreetings: ["Hydrated greeting"]
    } as Character

    let selectedCharacter = initialCharacter
    let messageState: Message[] = []
    let historyState: ChatHistory = []
    const setMessages = vi.fn(
      (next: Message[] | ((prev: Message[]) => Message[])) => {
        messageState = applyMessageUpdate(messageState, next)
      }
    )
    const setHistory = vi.fn(
      (next: ChatHistory | ((prev: ChatHistory) => ChatHistory)) => {
        historyState = applyHistoryUpdate(historyState, next)
      }
    )
    const setSelectedCharacter = vi.fn((next: Character | null) => {
      if (next) {
        selectedCharacter = next
      }
    })

    mocks.getCharacter.mockResolvedValue(fetchedCharacter)

    const { rerender } = renderHook(
      ({ currentCharacter }: { currentCharacter: Character | null }) =>
        useCharacterGreeting({
          playgroundReady: true,
          selectedCharacter: currentCharacter,
          serverChatId: null,
          historyId: "history-loop",
          messagesLength: messageState.length,
          setMessages,
          setHistory,
          setSelectedCharacter
        }),
      {
        initialProps: {
          currentCharacter: selectedCharacter
        }
      }
    )

    await waitFor(() => {
      expect(setSelectedCharacter).toHaveBeenCalledTimes(1)
    })
    expect(mocks.getCharacter).toHaveBeenCalledTimes(1)

    rerender({ currentCharacter: selectedCharacter })

    await Promise.resolve()
    await Promise.resolve()

    rerender({ currentCharacter: { ...selectedCharacter } })
    await Promise.resolve()
    await Promise.resolve()

    expect(mocks.getCharacter).toHaveBeenCalledTimes(1)
    expect(messageState[0]?.message).toBeTruthy()
    expect(historyState[0]?.content).toBeTruthy()
  })

  it("does not rewrite an already injected greeting when the same character is rehydrated", async () => {
    const selectedCharacter = {
      id: "char-stable",
      name: "Stable Guide",
      greeting: "Hello again",
      alternateGreetings: ["Welcome back"]
    } as Character
    const greetingOptions = buildGreetingOptionsFromEntries(
      collectGreetingEntries(selectedCharacter)
    )
    const selectedGreeting = greetingOptions[0]
    if (!selectedGreeting) {
      throw new Error("Test setup failed: missing greeting option")
    }
    mocks.settings = {
      greetingEnabled: true,
      greetingSelectionId: selectedGreeting.id,
      greetingsChecksum: buildGreetingsChecksumFromOptions(greetingOptions),
      useCharacterDefault: false
    }

    let messageState: Message[] = []
    let historyState: ChatHistory = []
    const setMessages = vi.fn(
      (next: Message[] | ((prev: Message[]) => Message[])) => {
        messageState = applyMessageUpdate(messageState, next)
      }
    )
    const setHistory = vi.fn(
      (next: ChatHistory | ((prev: ChatHistory) => ChatHistory)) => {
        historyState = applyHistoryUpdate(historyState, next)
      }
    )
    const setSelectedCharacter = vi.fn()

    const { rerender } = renderHook(
      ({ currentCharacter }: { currentCharacter: Character | null }) =>
        useCharacterGreeting({
          playgroundReady: true,
          selectedCharacter: currentCharacter,
          serverChatId: null,
          historyId: "history-stable",
          messagesLength: messageState.length,
          setMessages,
          setHistory,
          setSelectedCharacter
        }),
      {
        initialProps: {
          currentCharacter: selectedCharacter
        }
      }
    )

    await waitFor(() => {
      expect(messageState[0]?.message).toBe(selectedGreeting.text)
    })
    const messageWrites = setMessages.mock.calls.length
    const historyWrites = setHistory.mock.calls.length

    rerender({ currentCharacter: { ...selectedCharacter } })
    await Promise.resolve()
    await Promise.resolve()

    expect(setMessages).toHaveBeenCalledTimes(messageWrites)
    expect(setHistory).toHaveBeenCalledTimes(historyWrites)
  })

  it("refreshes an injected greeting when the same character gains a new name or avatar", async () => {
    const selectedCharacter = {
      id: "char-hydrated-greeting",
      name: "Draft Guide",
      greeting: "Hello there"
    } as Character
    const hydratedCharacter = {
      ...selectedCharacter,
      name: "Hydrated Guide",
      avatar_url: "https://example.com/hydrated.png"
    } as Character

    mocks.settings = {
      greetingEnabled: true,
      greetingSelectionId: null,
      greetingsChecksum: null,
      useCharacterDefault: true
    }

    let messageState: Message[] = []
    let historyState: ChatHistory = []
    const setMessages = vi.fn(
      (next: Message[] | ((prev: Message[]) => Message[])) => {
        messageState = applyMessageUpdate(messageState, next)
      }
    )
    const setHistory = vi.fn(
      (next: ChatHistory | ((prev: ChatHistory) => ChatHistory)) => {
        historyState = applyHistoryUpdate(historyState, next)
      }
    )
    const setSelectedCharacter = vi.fn()

    const { rerender } = renderHook(
      ({
        character,
        messagesLength
      }: {
        character: Character
        messagesLength: number
      }) =>
        useCharacterGreeting({
          playgroundReady: true,
          selectedCharacter: character,
          serverChatId: null,
          historyId: "history-hydrated-greeting",
          messagesLength,
          setMessages,
          setHistory,
          setSelectedCharacter
        }),
      {
        initialProps: {
          character: selectedCharacter,
          messagesLength: 0
        }
      }
    )

    await waitFor(() => {
      expect(messageState[0]?.name).toBe("Draft Guide")
      expect(messageState[0]?.message).toBe("Hello there")
    })

    rerender({
      character: hydratedCharacter,
      messagesLength: messageState.length
    })

    await waitFor(() => {
      expect(messageState[0]?.name).toBe("Hydrated Guide")
      expect(messageState[0]?.modelName).toBe("Hydrated Guide")
      expect(messageState[0]?.modelImage).toBe(
        "https://example.com/hydrated.png"
      )
    })
    expect(historyState[0]?.content).toBe("Hello there")
  })
})
