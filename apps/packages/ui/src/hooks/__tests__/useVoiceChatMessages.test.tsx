// @vitest-environment jsdom
import React from "react"
import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { Message } from "@/store/option"

const saveMessageOnSuccessMock = vi.hoisted(() =>
  vi.fn(async (payload: unknown) => payload)
)
const createSaveMessageOnSuccessMock = vi.hoisted(() =>
  vi.fn(() => saveMessageOnSuccessMock)
)

const storeState = vi.hoisted(() => ({
  messages: [] as Message[],
  history: [] as { role: "user" | "assistant" | "system"; content: string }[],
  historyId: "history-1" as string | null,
  temporaryChat: false,
  setHistoryId: vi.fn(),
  setMessages: vi.fn(),
  setHistory: vi.fn()
}))

const voiceChatSettingsState = vi.hoisted(() => ({
  voiceChatModel: "",
  voiceChatTtsMode: "stream" as const
}))

const selectedModelState = vi.hoisted(() => ({
  selectedModel: "gpt-4o-mini"
}))

const idState = vi.hoisted(() => ({
  nextId: 0
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: () => ({
    messages: storeState.messages,
    setMessages: (updater: Message[] | ((prev: Message[]) => Message[])) => {
      storeState.messages =
        typeof updater === "function" ? updater(storeState.messages) : updater
      storeState.setMessages(updater)
    },
    history: storeState.history,
    setHistory: (
      updater:
        | { role: "user" | "assistant" | "system"; content: string }[]
        | ((prev: { role: "user" | "assistant" | "system"; content: string }[]) => { role: "user" | "assistant" | "system"; content: string }[])
    ) => {
      storeState.history =
        typeof updater === "function" ? updater(storeState.history) : updater
      storeState.setHistory(updater)
    },
    historyId: storeState.historyId,
    setHistoryId: (...args: unknown[]) => storeState.setHistoryId(...args),
    temporaryChat: storeState.temporaryChat
  })
}))

vi.mock("@/hooks/useVoiceChatSettings", () => ({
  useVoiceChatSettings: () => voiceChatSettingsState
}))

vi.mock("@/hooks/chat/useSelectedModel", () => ({
  useSelectedModel: () => selectedModelState
}))

vi.mock("@/hooks/utils/messageHelpers", () => ({
  createSaveMessageOnSuccess: (...args: unknown[]) =>
    (createSaveMessageOnSuccessMock as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("@/db/dexie/helpers", () => ({
  generateID: vi.fn(() => {
    idState.nextId += 1
    return `generated-id-${idState.nextId}`
  })
}))

import { useVoiceChatMessages } from "@/hooks/useVoiceChatMessages"

const resetStore = () => {
  storeState.messages = []
  storeState.history = []
  storeState.historyId = "history-1"
  storeState.temporaryChat = false
  storeState.setHistoryId.mockReset()
  storeState.setMessages.mockReset()
  storeState.setHistory.mockReset()
  idState.nextId = 0
  saveMessageOnSuccessMock.mockClear()
  createSaveMessageOnSuccessMock.mockClear()
}

describe("useVoiceChatMessages", () => {
  beforeEach(() => {
    resetStore()
    voiceChatSettingsState.voiceChatModel = ""
    selectedModelState.selectedModel = "gpt-4o-mini"
  })

  afterEach(() => {
    resetStore()
  })

  it("removes the empty assistant placeholder when the stream fails before assistant text arrives", async () => {
    const { result } = renderHook(() => useVoiceChatMessages())

    act(() => {
      result.current.beginTurn("hello there")
    })

    await act(async () => {
      await result.current.failTurn("voice_chat_error")
    })

    expect(storeState.messages.map((message) => message.role)).toEqual(["user"])
    expect(storeState.history).toEqual([{ role: "user", content: "hello there" }])
    expect(saveMessageOnSuccessMock).not.toHaveBeenCalled()
  })

  it("persists nothing when the stream fails before any transcript turn begins", async () => {
    const { result } = renderHook(() => useVoiceChatMessages())

    await act(async () => {
      await result.current.failTurn("voice_chat_disconnected")
    })

    expect(storeState.messages).toEqual([])
    expect(storeState.history).toEqual([])
    expect(saveMessageOnSuccessMock).not.toHaveBeenCalled()
  })

  it("marks partial assistant text as interrupted when the stream fails mid-turn", async () => {
    const { result } = renderHook(() => useVoiceChatMessages())

    act(() => {
      result.current.beginTurn("hello there")
      result.current.appendAssistantDelta("Partial answer")
    })

    await act(async () => {
      await result.current.failTurn("voice_chat_error")
    })

    const assistant = storeState.messages.find((message) => message.role === "assistant")
    expect(assistant?.message).toBe("Partial answer")
    expect(assistant?.generationInfo?.interrupted).toBe(true)
    expect(assistant?.generationInfo?.interruptionReason).toBe("voice_chat_error")
    expect(storeState.history.at(-1)).toEqual({
      role: "assistant",
      content: "Partial answer"
    })
    expect(saveMessageOnSuccessMock).toHaveBeenCalledWith(
      expect.objectContaining({
        fullText: "Partial answer",
        generationInfo: expect.objectContaining({
          interrupted: true,
          interruptionReason: "voice_chat_error"
        })
      })
    )
  })

  it("keeps the manual abandonment path non-persistent before assistant text arrives", () => {
    const { result } = renderHook(() => useVoiceChatMessages())

    act(() => {
      result.current.beginTurn("hello there")
      result.current.abandonTurn()
    })

    expect(storeState.messages.map((message) => message.role)).toEqual(["user"])
    expect(storeState.history).toEqual([{ role: "user", content: "hello there" }])
    expect(saveMessageOnSuccessMock).not.toHaveBeenCalled()
  })
})
