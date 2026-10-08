import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useQuery } from "@tanstack/react-query"
import {
  CONVERSATION_TAB_QUERY_KEYS,
  ConversationTab
} from "../Settings/tabs/ConversationTab"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useChatSettingsRecord } from "@/hooks/chat/useChatSettingsRecord"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: any) => {
      if (typeof defaultValue === "string") return defaultValue
      if (defaultValue && typeof defaultValue.defaultValue === "string") {
        return defaultValue.defaultValue
      }
      return _key
    }
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    error: vi.fn(),
    success: vi.fn()
  })
}))

vi.mock("@/components/Common/Settings/PromptAssemblyPreview", () => ({
  PromptAssemblyPreview: () => null
}))

vi.mock("@/components/Common/Settings/LorebookDebugPanel", () => ({
  LorebookDebugPanel: () => null
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn().mockResolvedValue(undefined),
    listCharacters: vi.fn().mockResolvedValue([]),
    listChatMessages: vi.fn().mockResolvedValue([]),
    updateChat: vi.fn().mockResolvedValue({ version: 1 })
  }
}))

vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: vi.fn()
}))

vi.mock("@tanstack/react-query", async () => {
  const actual = await vi.importActual("@tanstack/react-query")
  return {
    ...actual,
    useQuery: vi.fn(),
    useQueryClient: vi.fn().mockReturnValue({
      invalidateQueries: vi.fn(),
      setQueryData: vi.fn()
    })
  }
})

vi.mock("antd", () => {
  return import("../../__tests__/mocks/antd").then((mod) =>
    mod.createFormSelectInputNumberAntdMock()
  )
})

describe("ConversationTab generation override controls", () => {
  const TEST_SERVER_CHAT_ID = "chat-1"
  const updateSettings = vi.fn().mockResolvedValue(undefined)
  const buildQueryResult = (overrides: Record<string, unknown> = {}) =>
    ({
      data: [],
      isLoading: false,
      isError: false,
      isFetching: false,
      error: null,
      refetch: vi.fn(),
      ...overrides
    }) as any
  const queryKeyEquals = (
    actualQueryKey: unknown,
    expectedQueryKey: readonly unknown[]
  ): boolean =>
    Array.isArray(actualQueryKey) &&
    actualQueryKey.length === expectedQueryKey.length &&
    actualQueryKey.every((value, index) => value === expectedQueryKey[index])

  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(useQuery).mockReturnValue(buildQueryResult())
    vi.mocked(useChatSettingsRecord).mockReturnValue({
      settings: {
        chatGenerationOverride: {
          enabled: false
        }
      },
      updateSettings,
      chatKey: "chat:test"
    } as any)
  })

  const renderConversationTab = (
    overrides: Partial<React.ComponentProps<typeof ConversationTab>> = {}
  ) =>
    render(
      <ConversationTab
        historyId="history-1"
        selectedSystemPrompt={null}
        onSystemPromptChange={() => {}}
        uploadedFiles={[]}
        onRemoveFile={() => {}}
        serverChatId={TEST_SERVER_CHAT_ID}
        serverChatState="in-progress"
        onStateChange={vi.fn()}
        serverChatTopic={null}
        onTopicChange={() => {}}
        onVersionChange={vi.fn()}
        {...overrides}
      />
    )

  it("identifies the system prompt as a model default with its separate save action", () => {
    renderConversationTab()
    expect(screen.getByText("Model default system prompt")).toBeInTheDocument()
    expect(screen.getByText("Saved with model settings. Applies to chats using this model; conversation Apply does not save it.")).toBeInTheDocument()
  })

  it("labels immediate attachment removal and keeps it out of form submission", () => {
    const onRemoveFile = vi.fn()
    renderConversationTab({ uploadedFiles: [{ id: "file-1", filename: "notes.txt", size: 1024 }], onRemoveFile })
    const remove = screen.getByRole("button", { name: "Remove file: notes.txt" })
    expect(remove).toHaveAttribute("type", "button")
    expect(screen.getByText("Attachment removal takes effect immediately.")).toBeInTheDocument()
    fireEvent.click(remove)
    expect(onRemoveFile).toHaveBeenCalledWith("file-1")
  })

  it("persists enabled chat generation override", async () => {
    renderConversationTab()

    const section = screen.getByTestId("chat-generation-override")
    const modeSelect = within(section).getByRole("combobox")
    fireEvent.change(modeSelect, { target: { value: "on" } })

    await waitFor(() => {
      expect(updateSettings).toHaveBeenCalledWith(
        expect.objectContaining({
          chatGenerationOverride: expect.objectContaining({
            enabled: true
          })
        })
      )
    })
  })

  it("parses and de-duplicates stop sequences on blur", async () => {
    renderConversationTab()

    const section = screen.getByTestId("chat-generation-override")
    const modeSelect = within(section).getByRole("combobox")
    fireEvent.change(modeSelect, { target: { value: "on" } })

    const stopInput = within(section).getByPlaceholderText(
      "Stop sequences, one per line"
    )
    await waitFor(() => {
      expect(stopInput).not.toBeDisabled()
    })
    fireEvent.change(stopInput, {
      target: { value: "END\nEND\nSTOP" }
    })
    fireEvent.blur(stopInput)

    await waitFor(() => {
      expect(updateSettings).toHaveBeenCalledWith(
        expect.objectContaining({
          chatGenerationOverride: expect.objectContaining({
            stop: ["END", "STOP"]
          })
        })
      )
    })
  })

  it("shows an inline warning when characters fail to load", async () => {
    vi.mocked(useQuery).mockImplementation((options: any) => {
      if (
        queryKeyEquals(
          options?.queryKey,
          CONVERSATION_TAB_QUERY_KEYS.listCharacters
        )
      ) {
        return buildQueryResult({ isError: true })
      }
      return buildQueryResult()
    })

    renderConversationTab()

    expect(
      await screen.findByText(
        "Failed to load character list. Participant options may be incomplete."
      )
    ).toBeInTheDocument()
  })

  it("shows an inline warning when pinned messages fail to load", async () => {
    vi.mocked(useQuery).mockImplementation((options: any) => {
      if (
        queryKeyEquals(
          options?.queryKey,
          CONVERSATION_TAB_QUERY_KEYS.pinnedMessages(TEST_SERVER_CHAT_ID)
        )
      ) {
        return buildQueryResult({ isError: true })
      }
      return buildQueryResult()
    })

    renderConversationTab()

    expect(
      await screen.findByText("Failed to load pinned messages.")
    ).toBeInTheDocument()
  })

  it("persists conversation state transitions and updates local state/version", async () => {
    const onStateChange = vi.fn()
    const onVersionChange = vi.fn()

    vi.mocked(tldwClient.updateChat).mockResolvedValueOnce({
      id: "chat-1",
      title: "Conversation",
      created_at: "2026-03-12T00:00:00.000Z",
      version: 8
    })
    vi.mocked(tldwClient.updateChat).mockResolvedValueOnce({
      id: "chat-1",
      title: "Conversation",
      created_at: "2026-03-12T00:00:00.000Z",
      version: 9
    })

    renderConversationTab({ onStateChange, onVersionChange })

    const stateSelect = screen.getByTestId("conversation-state-select")
    fireEvent.change(stateSelect, { target: { value: "resolved" } })
    fireEvent.change(stateSelect, { target: { value: "backlog" } })

    await waitFor(() => {
      expect(onStateChange).toHaveBeenNthCalledWith(1, "resolved")
      expect(onStateChange).toHaveBeenNthCalledWith(2, "backlog")
      expect(vi.mocked(tldwClient.updateChat)).toHaveBeenNthCalledWith(
        1,
        TEST_SERVER_CHAT_ID,
        { state: "resolved" }
      )
      expect(vi.mocked(tldwClient.updateChat)).toHaveBeenNthCalledWith(
        2,
        TEST_SERVER_CHAT_ID,
        { state: "backlog" }
      )
      expect(onVersionChange).toHaveBeenNthCalledWith(1, 8)
      expect(onVersionChange).toHaveBeenNthCalledWith(2, 9)
    })
  })
})
