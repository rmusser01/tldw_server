// @vitest-environment jsdom
import React from "react"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { PlaygroundChat } from "../PlaygroundChat"

const useMessageOptionState = vi.hoisted(() => ({
  value: {
    messages: [
      {
        id: "c1-user",
        role: "user",
        isBot: false,
        name: "You",
        message: "Compare these two answers",
        messageType: "compare:user",
        clusterId: "cluster-1"
      },
      {
        id: "c1-a",
        role: "assistant",
        isBot: true,
        name: "Model A",
        modelId: "model-a",
        modelName: "Model A",
        message: "Model A response",
        messageType: "compare:reply",
        clusterId: "cluster-1"
      },
      {
        id: "c1-b",
        role: "assistant",
        isBot: true,
        name: "Model B",
        modelId: "model-b",
        modelName: "Model B",
        message: "Model B response",
        messageType: "compare:reply",
        clusterId: "cluster-1"
      }
    ],
    setMessages: vi.fn(),
    streaming: false,
    isProcessing: false,
    regenerateLastMessage: vi.fn(),
    isSearchingInternet: false,
    editMessage: vi.fn(),
    deleteMessage: vi.fn(),
    toggleMessagePinned: vi.fn(),
    ttsEnabled: false,
    onSubmit: vi.fn(),
    actionInfo: null,
    messageSteeringMode: "none",
    setMessageSteeringMode: vi.fn(),
    messageSteeringForceNarrate: false,
    setMessageSteeringForceNarrate: vi.fn(),
    clearMessageSteering: vi.fn(),
    createChatBranch: vi.fn(),
    createCompareBranch: vi.fn(),
    temporaryChat: false,
    serverChatId: "chat-1",
    serverChatCharacterId: null,
    stopStreamingRequest: vi.fn(),
    isEmbedding: false,
    compareMode: true,
    compareFeatureEnabled: true,
    compareSelectionByCluster: {
      "cluster-1": ["model-a", "model-b"]
    },
    setCompareSelectionForCluster: vi.fn(),
    compareActiveModelsByCluster: {
      "cluster-1": ["model-a", "model-b"]
    },
    setCompareActiveModelsForCluster: vi.fn(),
    setCompareSelectedModels: vi.fn(),
    historyId: "history-1",
    setSelectedModel: vi.fn(),
    setCompareMode: vi.fn(),
    sendPerModelReply: vi.fn(),
    compareCanonicalByCluster: {},
    setCompareCanonicalForCluster: vi.fn(),
    compareContinuationModeByCluster: {
      "cluster-1": "compare"
    },
    setCompareContinuationModeForCluster: vi.fn(),
    setCompareParentForHistory: vi.fn(),
    compareSplitChats: {},
    setCompareSplitChat: vi.fn(),
    compareMaxModels: 3
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValue?: string,
      options?: Record<string, unknown>
    ) => {
      const template = defaultValue || key
      if (!options) {
        return template
      }
      return template.replace(/\{\{\s*(\w+)\s*\}\}/g, (_, name: string) => {
        const value = options[name]
        return value == null ? "" : String(value)
      })
    }
  })
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({ data: [] })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [false]
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => useMessageOptionState.value
}))

vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [null]
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn()
  })
}))

vi.mock("@/components/Common/ChatGreetingPicker", () => ({
  ChatGreetingPicker: () => <div data-testid="chat-greeting-picker" />
}))

vi.mock("./PlaygroundEmpty", () => ({
  PlaygroundEmpty: () => <div data-testid="playground-empty" />
}))

vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: (props: { message: string; onNewBranch?: () => void }) => (
    <div data-testid="playground-message-mock">
      {props.message}
      <button onClick={props.onNewBranch}>Fork this message</button>
    </div>
  )
}))

const getCardElements = async (modelKey: "model-a" | "model-b") => {
  const identity = await screen.findByTestId(
    `compare-model-identity-cluster-1-${modelKey}`
  )
  const card = identity.closest('[role="article"]') as HTMLElement | null
  if (!card) {
    throw new Error(`Missing compare card for ${modelKey}`)
  }
  const input = within(card).getByRole("textbox") as HTMLInputElement
  const send = within(card).getByRole("button", { name: "Send" })
  return { card, input, send }
}

describe("PlaygroundChat per-model mini composer routing", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("routes follow-up prompts to the selected compare model", async () => {
    const user = userEvent.setup()

    render(<PlaygroundChat />)

    const { input, send } = await getCardElements("model-a")

    await user.type(input, "follow-up for model a")
    await user.click(send)

    expect(useMessageOptionState.value.sendPerModelReply).toHaveBeenCalledTimes(
      1
    )
    expect(useMessageOptionState.value.sendPerModelReply).toHaveBeenCalledWith({
      clusterId: "cluster-1",
      modelId: "model-a",
      message: "follow-up for model a"
    })
    expect(input.value).toBe("")
  })

  it("keeps per-model drafts isolated so sends do not leak between cards", async () => {
    const user = userEvent.setup()

    render(<PlaygroundChat />)

    const modelA = await getCardElements("model-a")
    const modelB = await getCardElements("model-b")

    await user.type(modelA.input, "alpha")
    await user.type(modelB.input, "beta")

    await user.click(modelA.send)

    expect(
      useMessageOptionState.value.sendPerModelReply
    ).toHaveBeenNthCalledWith(1, {
      clusterId: "cluster-1",
      modelId: "model-a",
      message: "alpha"
    })
    expect(modelA.input.value).toBe("")
    expect(modelB.input.value).toBe("beta")

    await user.click(modelB.send)

    expect(
      useMessageOptionState.value.sendPerModelReply
    ).toHaveBeenNthCalledWith(2, {
      clusterId: "cluster-1",
      modelId: "model-b",
      message: "beta"
    })
    expect(useMessageOptionState.value.sendPerModelReply).toHaveBeenCalledTimes(
      2
    )
    expect(modelB.input.value).toBe("")
  })
})

it("individual comparison response forks carry stable model-qualified boundaries", async () => {
  const user = userEvent.setup()
  render(<PlaygroundChat />)
  const { card } = await getCardElements("model-a")
  await user.click(
    within(card).getByRole("button", { name: "Fork this message" })
  )
  expect(useMessageOptionState.value.createChatBranch).toHaveBeenCalledWith(
    "c1-a",
    { model_id: "model-a", cluster_id: "cluster-1" },
    expect.any(Function)
  )
})
it.each(["blocked", "partial", "unknown"])(
  "%s comparison result does not leave compare mode or adopt a candidate child",
  async (state) => {
    useMessageOptionState.value.createCompareBranch.mockResolvedValue({
      state,
      operation_id: "op",
      owner_key: "owner",
      code: "failed",
      candidate_child_id: "candidate"
    })
    useMessageOptionState.value.setCompareMode.mockClear()
    useMessageOptionState.value.setCompareSplitChat.mockClear()
    const user = userEvent.setup()
    render(<PlaygroundChat />)
    const { card } = await getCardElements("model-a")
    await user.click(
      within(card).getByRole("button", { name: "Open as full chat" })
    )
    expect(useMessageOptionState.value.setCompareMode).not.toHaveBeenCalled()
    expect(
      useMessageOptionState.value.setCompareSplitChat
    ).not.toHaveBeenCalled()
  }
)
it("child receipt callback leaves comparison mode without changing source split records", async () => {
  useMessageOptionState.value.createCompareBranch.mockImplementation(
    async ({ onOpened }: any) => {
      onOpened("child")
      return {
        state: "committed",
        operation_id: "op",
        owner_key: "owner",
        child_id: "child",
        message_map: { "c1-a": "child-a" }
      }
    }
  )
  useMessageOptionState.value.setCompareSplitChat.mockClear()
  useMessageOptionState.value.setCompareMode.mockClear()
  const user = userEvent.setup()
  render(<PlaygroundChat />)
  const { card } = await getCardElements("model-a")
  await user.click(
    within(card).getByRole("button", { name: "Open as full chat" })
  )
  expect(useMessageOptionState.value.setCompareMode).toHaveBeenCalledWith(false)
  expect(useMessageOptionState.value.setCompareSplitChat).not.toHaveBeenCalled()
})
