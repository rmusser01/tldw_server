import { useChatSettingsRecord } from "@/hooks/chat/useChatSettingsRecord"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useStoreMessageOption } from "@/store/option"
import { useQuery } from "@tanstack/react-query"
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within
} from "@testing-library/react"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  CONVERSATION_TAB_QUERY_KEYS,
  ConversationTab
} from "../Settings/tabs/ConversationTab"

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
    updateChat: vi.fn().mockResolvedValue({ version: 1 }),
    getChat: vi
      .fn()
      .mockResolvedValue({
        id: "chat-1",
        version: 5,
        state: "in-progress",
        topic_label: null
      })
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
  const updateSettings = vi.fn()
  let current = true
  let ready = true
  const requestScope = {
    config: {
      serverUrl: "https://owned.test",
      authMode: "single-user" as const
    },
    userId: 1
  }
  const target = {
    historyId: "history-1",
    serverChatId: TEST_SERVER_CHAT_ID,
    requestScope,
    scope: { type: "workspace" as const, workspaceId: "W" },
    validate_lease: () => current
  }
  type DraftStatus = { dirty: boolean; pending: boolean; revision: number }
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
    actualQueryKey.length >= expectedQueryKey.length &&
    expectedQueryKey.every((value, index) => value === actualQueryKey[index])

  type OwnedReadQuery = {
    queryKey: readonly unknown[]
    enabled?: boolean
    queryFn: (context: { signal: AbortSignal }) => Promise<{ ownerKey: string; items: unknown[] }>
  }
  const readQueries = () => vi.mocked(useQuery).mock.calls.map(([options]) => options as unknown as OwnedReadQuery)

  beforeEach(() => {
    vi.clearAllMocks()
    current = true
    ready = true
    vi.mocked(tldwClient.getChat)
      .mockReset()
      .mockResolvedValue({
        id: TEST_SERVER_CHAT_ID,
        title: "Conversation",
        created_at: "2026-10-08T00:00:00Z",
        version: 5,
        state: "in-progress",
        topic_label: null
      })
    updateSettings
      .mockReset()
      .mockResolvedValue({
        schemaVersion: 2,
        updatedAt: "2026-10-08T00:00:00Z"
      })
    vi.mocked(tldwClient.updateChat)
      .mockReset()
      .mockImplementation(async (id, patch) => ({
        id: String(id),
        title: "Conversation",
        created_at: "2026-03-12T00:00:00Z",
        version: 5,
        ...patch
      }))
    useStoreMessageOption.setState({
      historyId: "history-1",
      serverChatId: TEST_SERVER_CHAT_ID,
      serverChatVersion: 4
    })
    vi.mocked(useQuery).mockReturnValue(buildQueryResult())
    vi.mocked(useChatSettingsRecord).mockReturnValue({
      settings: {
        chatGenerationOverride: {
          enabled: false
        }
      },
      updateSettings,
      chatKey: "chat:test",
      target,
      captureTarget: () => target,
      isReady: ready,
      isCurrent: () => current,
      mode: "ordinary"
    } as any)
  })

  const renderConversationTab = (
    overrides: Partial<
      React.ComponentProps<typeof ConversationTab> & {
        serverChatVersion: number
        isCurrent: () => boolean
        onDraftChange: (value: DraftStatus) => void
      }
    > = {}
  ) =>
    render(
      <ConversationTab
        historyId="history-1"
        selectedSystemPrompt={null}
        onSystemPromptChange={() => {}}
        uploadedFiles={[]}
        onRemoveFile={() => {}}
        serverChatId={TEST_SERVER_CHAT_ID}
        serverChatVersion={4}
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
    expect(
      screen.getByText(
        "Saved with model settings. Applies to chats using this model; conversation Apply does not save it."
      )
    ).toBeInTheDocument()
  })

  it("labels immediate attachment removal and keeps it out of form submission", () => {
    const onRemoveFile = vi.fn()
    renderConversationTab({
      uploadedFiles: [{ id: "file-1", filename: "notes.txt", size: 1024 }],
      onRemoveFile
    })
    const remove = screen.getByRole("button", {
      name: "Remove file: notes.txt"
    })
    expect(remove).toHaveAttribute("type", "button")
    expect(
      screen.getByText("Attachment removal takes effect immediately.")
    ).toBeInTheDocument()
    fireEvent.click(remove)
    expect(onRemoveFile).toHaveBeenCalledWith("file-1")
  })

  it("persists enabled chat generation override", async () => {
    renderConversationTab()

    const section = screen.getByTestId("chat-generation-override")
    const modeSelect = within(section).getByRole("combobox")
    fireEvent.change(modeSelect, { target: { value: "on" } })

    expect(updateSettings).not.toHaveBeenCalled()
    fireEvent.click(
      screen.getByRole("button", { name: "Apply conversation changes" })
    )
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

    expect(updateSettings).not.toHaveBeenCalled()
    fireEvent.click(
      screen.getByRole("button", { name: "Apply conversation changes" })
    )
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

  it("uses captured account and workspace for both settings catalog reads", async () => {
    renderConversationTab()
    const queries = readQueries()
    const signal = new AbortController().signal
    for (const options of queries) await options.queryFn({ signal })
    expect(tldwClient.initialize).not.toHaveBeenCalled()
    expect(tldwClient.listCharacters).toHaveBeenCalledWith({ limit: 200 }, expect.objectContaining({ requestScope, signal }))
    expect(tldwClient.listChatMessages).toHaveBeenCalledWith(TEST_SERVER_CHAT_ID, expect.objectContaining({ include_metadata: "true", limit: 200, offset: 0 }), expect.objectContaining({ requestScope, scope: target.scope, signal }))
  })

  it.each(["characters", "pinned"])("rejects a held %s read after its original source retires", async (kind) => {
    let release!: (value: Awaited<ReturnType<typeof tldwClient.listChatMessages>>) => void
    const held = new Promise<Awaited<ReturnType<typeof tldwClient.listChatMessages>>>(resolve => { release = resolve })
    const method = kind === "characters" ? tldwClient.listCharacters : tldwClient.listChatMessages
    vi.mocked(method).mockReturnValueOnce(held)
    renderConversationTab()
    const options = readQueries().find(value => value.queryKey[0] === (kind === "characters" ? "tldw:listCharacters" : "conversation-tab:pinned-messages"))
    const read = options.queryFn({ signal: new AbortController().signal })
    await waitFor(() => expect(method).toHaveBeenCalledOnce())
    current = false
    release([])
    await expect(read).rejects.toMatchObject({ status: 412 })
  })

  it("separates read keys across verified owners and does not display a foreign retained envelope", () => {
    const first = renderConversationTab()
    const before = readQueries().slice(-2).map(value => value.queryKey)
    const previous = vi.mocked(useChatSettingsRecord).mock.results.at(-1)!.value
    const nextTarget = { ...target, requestScope: { ...requestScope, userId: 2 }, scope: { type: "workspace", workspaceId: "Other" } }
    vi.mocked(useChatSettingsRecord).mockReturnValue({ ...previous, target: nextTarget, captureTarget: () => nextTarget })
    vi.mocked(useQuery).mockReturnValue(buildQueryResult({ data: { ownerKey: "foreign", items: [{ id: "secret", role: "user", content: "Foreign private pin" }] } }))
    first.rerender(<ConversationTab historyId="history-1" selectedSystemPrompt={null} onSystemPromptChange={vi.fn()} uploadedFiles={[]} onRemoveFile={vi.fn()} serverChatId={TEST_SERVER_CHAT_ID} serverChatVersion={4} serverChatState="in-progress" onStateChange={vi.fn()} serverChatTopic={null} onTopicChange={vi.fn()} onVersionChange={vi.fn()} />)
    const after = readQueries().slice(-2).map(value => value.queryKey)
    expect(after[0]).not.toEqual(before[0])
    expect(after[1]).not.toEqual(before[1])
    expect(screen.queryByText("Foreign private pin")).not.toBeInTheDocument()
  })

  it("does not read unqualified catalogs while the saved owner is unresolved", () => {
    const previous = vi.mocked(useChatSettingsRecord).mock.results.at(-1)?.value
    vi.mocked(useChatSettingsRecord).mockReturnValue({ ...previous, settings: null, updateSettings, isReady: false, mode: "ordinary", isCurrent: () => true, target: { historyId: null, serverChatId: TEST_SERVER_CHAT_ID }, captureTarget: () => { throw new Error("No owner") } } as ReturnType<typeof useChatSettingsRecord>)
    renderConversationTab()
    expect(readQueries().every(options => options.enabled === false)).toBe(true)
    expect(tldwClient.listCharacters).not.toHaveBeenCalled()
    expect(tldwClient.listChatMessages).not.toHaveBeenCalled()
  })

  it("includes pinned messages beyond the first server page", async () => {
    const page = Array.from({ length: 200 }, (_, id) => ({ id: String(id), role: "user", content: "Unpinned", pinned: false }))
    vi.mocked(tldwClient.listChatMessages).mockResolvedValueOnce(page as Awaited<ReturnType<typeof tldwClient.listChatMessages>>).mockResolvedValueOnce([{ id: "late-pin", role: "assistant", content: "Pinned on page two", pinned: true }] as Awaited<ReturnType<typeof tldwClient.listChatMessages>>)
    renderConversationTab()
    const options = readQueries().find(value => value.queryKey[0] === "conversation-tab:pinned-messages")
    const result = await options.queryFn({ signal: new AbortController().signal })
    expect(result.items).toEqual([{ id: "late-pin", role: "assistant", content: "Pinned on page two" }])
    expect(vi.mocked(tldwClient.listChatMessages).mock.calls[1][1]).toMatchObject({ limit: 200, offset: 200 })
  })

  it("stages conversation details and publishes one captured versioned update after Apply", async () => {
    const onStateChange = vi.fn(),
      onTopicChange = vi.fn(),
      onVersionChange = vi.fn()
    renderConversationTab({
      onStateChange,
      onTopicChange,
      onVersionChange,
      serverChatTopic: "Old tag"
    })
    fireEvent.change(screen.getByTestId("conversation-state-select"), {
      target: { value: "resolved" }
    })
    fireEvent.change(screen.getByTestId("conversation-state-select"), {
      target: { value: "backlog" }
    })
    fireEvent.change(
      screen.getByPlaceholderText("Conversation tag (optional)"),
      { target: { value: "" } }
    )
    expect(onStateChange).not.toHaveBeenCalled()
    expect(onTopicChange).not.toHaveBeenCalled()
    expect(tldwClient.updateChat).not.toHaveBeenCalled()
    fireEvent.click(
      screen.getByRole("button", { name: "Apply conversation changes" })
    )
    await waitFor(() => expect(onVersionChange).toHaveBeenCalledWith(5))
    expect(tldwClient.updateChat).toHaveBeenCalledExactlyOnceWith(
      TEST_SERVER_CHAT_ID,
      { state: "backlog", topic_label: null },
      expect.objectContaining({
        expectedVersion: 4,
        requestScope,
        scope: target.scope
      })
    )
    expect(onStateChange).toHaveBeenCalledExactlyOnceWith("backlog")
    expect(onTopicChange).toHaveBeenCalledExactlyOnceWith(null)
  })

  it("reports a typed conversation draft before blur and cancels without a write", () => {
    const onDraftChange = vi.fn()
    renderConversationTab({ onDraftChange })
    const editor = screen.getByPlaceholderText(
      "E.g., Keep responses grounded, avoid repetition, and progress the scene."
    )
    fireEvent.change(editor, { target: { value: "My unsaved note" } })
    expect(onDraftChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ dirty: true, pending: false })
    )
    fireEvent.blur(editor)
    expect(updateSettings).not.toHaveBeenCalled()
    fireEvent.click(
      screen.getByRole("button", { name: "Cancel conversation changes" })
    )
    expect(editor).toHaveValue("")
    expect(updateSettings).not.toHaveBeenCalled()
    expect(tldwClient.updateChat).not.toHaveBeenCalled()
    expect(onDraftChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ dirty: false, pending: false })
    )
  })

  it("keeps edited fields through external hydration and applies only the changed generation field", async () => {
    vi.mocked(useChatSettingsRecord).mockReturnValue({
      settings: {
        schemaVersion: 2,
        updatedAt: "2026-10-08T00:00:00Z",
        chatGenerationOverride: {
          enabled: true,
          temperature: 0.7,
          top_p: 0.9,
          stop: ["END"]
        }
      },
      updateSettings,
      chatKey: "chat:test",
      target,
      captureTarget: () => target,
      isReady: true,
      isCurrent: () => current,
      mode: "ordinary"
    } as ReturnType<typeof useChatSettingsRecord>)
    const view = renderConversationTab()
    const section = screen.getByTestId("chat-generation-override")
    const number = within(section).getAllByRole("spinbutton")[0]
    fireEvent.change(number, { target: { value: "0" } })
    const record = vi.mocked(useChatSettingsRecord).mock.results.at(-1)?.value
    vi.mocked(useChatSettingsRecord).mockReturnValue({
      ...record,
      settings: {
        ...record.settings,
        chatGenerationOverride: {
          enabled: true,
          temperature: 1.4,
          top_p: 0.8,
          stop: ["NEW"]
        }
      }
    })
    view.rerender(
      <ConversationTab
        historyId="history-1"
        selectedSystemPrompt={null}
        onSystemPromptChange={vi.fn()}
        uploadedFiles={[]}
        onRemoveFile={vi.fn()}
        serverChatId={TEST_SERVER_CHAT_ID}
        serverChatState="in-progress"
        onStateChange={vi.fn()}
        serverChatTopic={null}
        onTopicChange={vi.fn()}
        onVersionChange={vi.fn()}
      />
    )
    expect(number).toHaveValue(0)
    expect(updateSettings).not.toHaveBeenCalled()
    fireEvent.click(
      screen.getByRole("button", { name: "Apply conversation changes" })
    )
    await waitFor(() =>
      expect(updateSettings).toHaveBeenCalledExactlyOnceWith({
        chatGenerationOverride: { temperature: 0 }
      })
    )
  })

  it("locks duplicate Apply and Cancel while its captured settings step is held", async () => {
    let release!: () => void
    updateSettings.mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          release = resolve
        })
    )
    renderConversationTab()
    fireEvent.change(
      within(screen.getByTestId("chat-generation-override")).getByRole(
        "combobox"
      ),
      { target: { value: "on" } }
    )
    const apply = screen.getByRole("button", {
      name: "Apply conversation changes"
    })
    fireEvent.click(apply)
    fireEvent.click(apply)
    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1))
    expect(
      screen.getByRole("button", { name: "Cancel conversation changes" })
    ).toBeDisabled()
    release()
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Apply conversation changes" })
      ).toBeDisabled()
    )
  })

  it("applies metadata before settings when both advance the backend conversation version", async () => {
    let version = 4
    const order: string[] = []
    updateSettings.mockImplementation(async () => {
      order.push("settings")
      version++
      return null
    })
    vi.mocked(tldwClient.updateChat).mockImplementation(
      async (id, patch, options) => {
        order.push("metadata")
        if (options?.expectedVersion !== version)
          throw new Error("version conflict")
        return {
          id: String(id),
          title: "Conversation",
          created_at: "2026-10-08T00:00:00Z",
          version: ++version,
          ...patch
        }
      }
    )
    vi.mocked(tldwClient.getChat).mockImplementation(async () => ({
      id: TEST_SERVER_CHAT_ID,
      title: "Conversation",
      created_at: "2026-10-08T00:00:00Z",
      version,
      state: "resolved",
      topic_label: null
    }))
    const onVersionChange = vi.fn()
    renderConversationTab({ onVersionChange })
    fireEvent.change(
      within(screen.getByTestId("chat-generation-override")).getByRole(
        "combobox"
      ),
      { target: { value: "on" } }
    )
    fireEvent.change(screen.getByTestId("conversation-state-select"), {
      target: { value: "resolved" }
    })
    fireEvent.click(
      screen.getByRole("button", { name: "Apply conversation changes" })
    )
    await waitFor(() => expect(onVersionChange).toHaveBeenLastCalledWith(6))
    expect(order).toEqual(["metadata", "settings"])
    expect(tldwClient.updateChat).toHaveBeenCalledTimes(1)
    expect(updateSettings).toHaveBeenCalledTimes(1)
  })

  it("retains metadata acknowledgement when settings fails and Retry skips the acknowledged step", async () => {
    updateSettings.mockRejectedValueOnce(new Error("Settings unavailable"))
    renderConversationTab()
    fireEvent.change(
      within(screen.getByTestId("chat-generation-override")).getByRole(
        "combobox"
      ),
      { target: { value: "on" } }
    )
    fireEvent.change(screen.getByTestId("conversation-state-select"), {
      target: { value: "resolved" }
    })
    fireEvent.click(
      screen.getByRole("button", { name: "Apply conversation changes" })
    )
    await screen.findByRole("alert")
    expect(tldwClient.updateChat).toHaveBeenCalledTimes(1)
    fireEvent.click(
      screen.getByRole("button", { name: "Retry conversation changes" })
    )
    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(2))
    expect(tldwClient.updateChat).toHaveBeenCalledTimes(1)
    expect(
      vi.mocked(tldwClient.updateChat).mock.calls[0][2]?.expectedVersion
    ).toBe(4)
    expect(updateSettings.mock.calls[1][0]).toEqual(
      updateSettings.mock.calls[0][0]
    )
  })

  it("does not save settings until an unconfirmed metadata step is retried at its original version", async () => {
    vi.mocked(tldwClient.updateChat).mockRejectedValueOnce(
      new Error("Details unavailable")
    )
    vi.mocked(tldwClient.getChat).mockResolvedValueOnce({
      id: TEST_SERVER_CHAT_ID,
      title: "Conversation",
      created_at: "2026-10-08T00:00:00Z",
      version: 4,
      state: "in-progress",
      topic_label: null
    })
    renderConversationTab()
    fireEvent.change(
      within(screen.getByTestId("chat-generation-override")).getByRole(
        "combobox"
      ),
      { target: { value: "on" } }
    )
    fireEvent.change(screen.getByTestId("conversation-state-select"), {
      target: { value: "resolved" }
    })
    fireEvent.click(
      screen.getByRole("button", { name: "Apply conversation changes" })
    )
    await screen.findByRole("alert")
    expect(updateSettings).not.toHaveBeenCalled()
    fireEvent.click(
      screen.getByRole("button", { name: "Retry conversation changes" })
    )
    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1))
    expect(
      vi
        .mocked(tldwClient.updateChat)
        .mock.calls.map((call) => call[2]?.expectedVersion)
    ).toEqual([4, 4])
  })

  it("refreshes the version for a subsequent detail edit after a settings-only Apply", async () => {
    let version = 4
    updateSettings.mockImplementation(async () => {
      version++
      return null
    })
    vi.mocked(tldwClient.getChat).mockImplementation(async () => ({
      id: TEST_SERVER_CHAT_ID,
      title: "Conversation",
      created_at: "2026-10-08T00:00:00Z",
      version,
      state: "resolved",
      topic_label: null
    }))
    vi.mocked(tldwClient.updateChat).mockImplementation(
      async (id, patch, options) => {
        if (options?.expectedVersion !== version)
          throw new Error("version conflict")
        return {
          id: String(id),
          title: "Conversation",
          created_at: "2026-10-08T00:00:00Z",
          version: ++version,
          ...patch
        }
      }
    )
    const onVersionChange = vi.fn((value) =>
      useStoreMessageOption.setState({ serverChatVersion: value })
    )
    const Subject = () => {
      const observed = useStoreMessageOption((state) => state.serverChatVersion)
      return (
        <ConversationTab
          historyId="history-1"
          selectedSystemPrompt={null}
          onSystemPromptChange={vi.fn()}
          uploadedFiles={[]}
          onRemoveFile={vi.fn()}
          serverChatId={TEST_SERVER_CHAT_ID}
          serverChatVersion={observed}
          serverChatState="in-progress"
          onStateChange={vi.fn()}
          serverChatTopic={null}
          onTopicChange={vi.fn()}
          onVersionChange={onVersionChange}
        />
      )
    }
    render(<Subject />)
    fireEvent.change(
      within(screen.getByTestId("chat-generation-override")).getByRole(
        "combobox"
      ),
      { target: { value: "on" } }
    )
    fireEvent.click(
      screen.getByRole("button", { name: "Apply conversation changes" })
    )
    await waitFor(() => expect(onVersionChange).toHaveBeenCalledWith(5))
    fireEvent.change(screen.getByTestId("conversation-state-select"), {
      target: { value: "resolved" }
    })
    fireEvent.click(
      screen.getByRole("button", { name: "Apply conversation changes" })
    )
    await waitFor(() => expect(tldwClient.updateChat).toHaveBeenCalledTimes(1))
    expect(
      vi.mocked(tldwClient.updateChat).mock.calls[0][2]?.expectedVersion
    ).toBe(5)
    await waitFor(() => expect(onVersionChange).toHaveBeenLastCalledWith(6))
  })

  it("reconciles a committed metadata response lost in transport without a second PUT", async () => {
    vi.mocked(tldwClient.updateChat).mockRejectedValueOnce(
      new Error("Response lost")
    )
    vi.mocked(tldwClient.getChat).mockResolvedValue({
      id: TEST_SERVER_CHAT_ID,
      title: "Conversation",
      created_at: "2026-10-08T00:00:00Z",
      version: 5,
      state: "resolved",
      topic_label: null
    })
    const onVersionChange = vi.fn()
    renderConversationTab({ onVersionChange })
    fireEvent.change(screen.getByTestId("conversation-state-select"), {
      target: { value: "resolved" }
    })
    fireEvent.click(
      screen.getByRole("button", { name: "Apply conversation changes" })
    )
    await screen.findByRole("alert")
    fireEvent.click(
      screen.getByRole("button", { name: "Retry conversation changes" })
    )
    await waitFor(() => expect(onVersionChange).toHaveBeenCalledWith(5))
    expect(tldwClient.updateChat).toHaveBeenCalledTimes(1)
    expect(tldwClient.getChat).toHaveBeenCalledExactlyOnceWith(
      TEST_SERVER_CHAT_ID,
      expect.objectContaining({
        requestScope,
        scope: target.scope,
        requiredMetadata: ["state"]
      })
    )
  })

  it("does not replay unconfirmed metadata when its scoped reconciliation read fails", async () => {
    vi.mocked(tldwClient.updateChat).mockRejectedValueOnce(
      new Error("Response lost")
    )
    vi.mocked(tldwClient.getChat).mockRejectedValue(
      new Error("Cannot verify outcome")
    )
    renderConversationTab()
    fireEvent.change(screen.getByTestId("conversation-state-select"), {
      target: { value: "resolved" }
    })
    fireEvent.click(
      screen.getByRole("button", { name: "Apply conversation changes" })
    )
    await screen.findByRole("alert")
    fireEvent.click(
      screen.getByRole("button", { name: "Retry conversation changes" })
    )
    await waitFor(() => expect(tldwClient.getChat).toHaveBeenCalledTimes(1))
    expect(tldwClient.updateChat).toHaveBeenCalledTimes(1)
    expect(updateSettings).not.toHaveBeenCalled()
  })

  it.each(["metadata", "settings refresh"])("does not publish old metadata after a reentrant newer version during %s", async (path) => {
    useStoreMessageOption.setState({ serverChatState: "in-progress", serverChatTopic: null })
    vi.mocked(tldwClient.getChat).mockResolvedValue({ id: TEST_SERVER_CHAT_ID, title: "Conversation", created_at: "2026-10-08T00:00:00Z", version: 5, state: "resolved", topic_label: "Old proposal" })
    const onTopicChange = vi.fn((topic) => useStoreMessageOption.setState({ serverChatTopic: topic }))
    const onVersionChange = vi.fn((version) => useStoreMessageOption.setState({ serverChatVersion: version }))
    renderConversationTab({ onStateChange: () => {
      useStoreMessageOption.setState({ serverChatVersion: 6, serverChatTopic: "Newer topic", serverChatState: "backlog" })
    }, onTopicChange, onVersionChange })
    if (path === "metadata") {
      fireEvent.change(screen.getByTestId("conversation-state-select"), { target: { value: "resolved" } })
      fireEvent.change(screen.getByPlaceholderText("Conversation tag (optional)"), { target: { value: "Old proposal" } })
    } else {
      fireEvent.change(within(screen.getByTestId("chat-generation-override")).getByRole("combobox"), { target: { value: "on" } })
    }
    fireEvent.click(screen.getByRole("button", { name: "Apply conversation changes" }))
    await waitFor(() => expect(useStoreMessageOption.getState().serverChatState).toBe("backlog"))
    await waitFor(() => expect(screen.getByRole("button", { name: "Apply conversation changes" })).not.toHaveAttribute("aria-busy", "true"))
    expect(useStoreMessageOption.getState()).toMatchObject({ serverChatVersion: 6, serverChatTopic: "Newer topic", serverChatState: "backlog" })
    expect(onTopicChange).not.toHaveBeenCalled()
    expect(onVersionChange).not.toHaveBeenCalled()
  })

  it("does not admit writes from a retired rendered form", () => {
    renderConversationTab({ isCurrent: () => current })
    fireEvent.change(
      within(screen.getByTestId("chat-generation-override")).getByRole(
        "combobox"
      ),
      { target: { value: "on" } }
    )
    current = false
    fireEvent.click(
      screen.getByRole("button", { name: "Apply conversation changes" })
    )
    expect(updateSettings).not.toHaveBeenCalled()
    expect(tldwClient.updateChat).not.toHaveBeenCalled()
  })
})
