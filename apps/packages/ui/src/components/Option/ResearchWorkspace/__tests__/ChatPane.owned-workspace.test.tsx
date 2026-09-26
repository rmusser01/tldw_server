import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { StrictMode } from "react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { initialState, useWorkspaceStore } from "@/store/workspace"
import { useStoreMessageOption } from "@/store/option"
import type { OwnedWorkspaceScope } from "@/store/owned-workspace-state"
import { ConnectionPhase } from "@/types/connection"
import { ChatPane } from "../ChatPane"

const mocks = vi.hoisted(() => ({
  submit: vi.fn(),
  save: vi.fn(),
  get: vi.fn(),
  media: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : (fallback?.defaultValue ?? key)
  })
}))
vi.mock("@/hooks/useMediaQuery", () => ({ useMobile: () => false }))
vi.mock("@/hooks/useSmartScroll", () => ({
  useSmartScroll: () => ({
    containerRef: { current: null },
    isAutoScrollToBottom: true,
    autoScrollToBottom: vi.fn()
  })
}))
vi.mock("@/store/connection", () => ({
  useConnectionStore: (selector: (state: unknown) => unknown) =>
    selector({
      state: {
        phase: ConnectionPhase.CONNECTED,
        isChecking: false,
        lastError: null
      },
      checkOnce: vi.fn()
    })
}))
vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    ...useStoreMessageOption(),
    onSubmit: mocks.submit,
    stopStreamingRequest: vi.fn(),
    regenerateLastMessage: vi.fn(),
    createChatBranch: vi.fn(),
    deleteMessage: vi.fn(),
    editMessage: vi.fn()
  })
}))
vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: ({ message }: { message: string }) => <div>{message}</div>
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getMediaDetails: (...args: unknown[]) => mocks.media(...args),
    getChatLorebookDiagnostics: vi.fn(async () => ({
      turns: [],
      total_turns_with_diagnostics: 0
    }))
  }
}))
vi.mock("@/services/tldw-server", () => ({
  fetchChatModels: vi.fn(async () => [
    { id: "test-model", name: "Test", provider: "test" }
  ])
}))

const scope: OwnedWorkspaceScope = {
  serverBase: "https://one.test/tldw",
  principalId: "1",
  organizationId: null
}
const owned = (patch: Partial<typeof scope> = {}) => ({
  kind: "server-owned" as const,
  scope: { ...scope, ...patch }
})
const input = () => screen.getByLabelText("Chat message") as HTMLTextAreaElement
const mount = (props: Parameters<typeof ChatPane>[0] = {}) =>
  render(
    <StrictMode>
      <MemoryRouter>
        <ChatPane {...props} />
      </MemoryRouter>
    </StrictMode>
  )
const submitDraft = (text: string) => {
  fireEvent.change(input(), { target: { value: text } })
  fireEvent.keyDown(input(), { key: "Enter" })
}
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  useWorkspaceStore.setState({
    ...initialState,
    storeHydrated: true,
    workspaceId: "same-id",
    workspaceChatReferenceId: "same-id",
    activeWorkspaceOrigin: owned(),
    ownedWorkspaceComposer: "Recovered owned draft",
    saveWorkspaceChatSession: mocks.save,
    getWorkspaceChatSession: mocks.get
  })
  useStoreMessageOption.setState({
    messages: [],
    history: [],
    historyId: null,
    serverChatId: null,
    streaming: false,
    isProcessing: false,
    selectedModel: "test-model"
  })
  mocks.get.mockReturnValue(null)
  mocks.submit.mockResolvedValue({ status: "submitted" })
})

describe("ChatPane owned automatic cache and composer boundary", () => {
  it("clears owned history on unmount before a separate local pane can consume it", () => {
    const view = mount()
    act(() => {
      useStoreMessageOption.setState({
        history: [{ role: "user", content: "Outgoing owned history" }],
        historyId: "owned-history",
        serverChatId: "owned-chat"
      })
    })
    view.unmount()
    expect(useStoreMessageOption.getState().history).toEqual([])
    expect(useStoreMessageOption.getState().historyId).toBeNull()
    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
  })

  it("never reads or saves a legacy chat session on owned StrictMode mount or edits", () => {
    mount()
    fireEvent.change(input(), { target: { value: "Edited draft" } })
    expect(mocks.get).not.toHaveBeenCalled()
    expect(mocks.save).not.toHaveBeenCalled()
  })

  it("restores the scoped composer and keeps edits in the owned draft", () => {
    mount()
    expect(input().value).toBe("Recovered owned draft")
    fireEvent.change(input(), { target: { value: "New owned draft" } })
    expect(useWorkspaceStore.getState().ownedWorkspaceComposer).toBe(
      "New owned draft"
    )
  })

  it("reflects same-scope restored drafts without resubmitting or clearing them", () => {
    mount()
    act(() =>
      useWorkspaceStore.getState().setOwnedWorkspaceComposer("Restored again")
    )
    expect(input().value).toBe("Restored again")
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it.each([
    { serverBase: "https://two.test/tldw" },
    { serverBase: "https://one.test/other" },
    { principalId: "2" },
    { organizationId: "org-2" }
  ])(
    "clears child text and global history for the same ID in scope %j",
    (patch) => {
      mount()
      fireEvent.change(input(), { target: { value: "Outgoing draft" } })
      act(() => {
        useStoreMessageOption.setState({
          messages: [
            {
              id: "old",
              isBot: true,
              name: "Bot",
              message: "Old private answer",
              sources: []
            }
          ],
          history: [{ role: "user", content: "Old private question" }],
          historyId: "old-history",
          serverChatId: "old-chat"
        })
      })
      act(() => {
        useWorkspaceStore.setState({
          activeWorkspaceOrigin: owned(patch),
          ownedWorkspaceComposer: "Incoming draft"
        })
      })
      expect(input().value).toBe("Incoming draft")
      expect(screen.queryByText("Old private answer")).not.toBeInTheDocument()
      expect(useStoreMessageOption.getState().history).toEqual([])
      expect(useStoreMessageOption.getState().serverChatId).toBeNull()
      expect(useStoreMessageOption.getState().historyId).toBeNull()
    }
  )

  it("does not restore an old failed submission into the incoming empty composer", async () => {
    const pending = deferred<{ status: string }>()
    mocks.submit.mockReturnValue(pending.promise)
    mount()
    submitDraft("Outgoing submission")
    await waitFor(() => expect(mocks.submit).toHaveBeenCalledTimes(1))
    act(() => {
      useWorkspaceStore.setState({
        activeWorkspaceOrigin: owned({ principalId: "2" }),
        ownedWorkspaceComposer: ""
      })
    })
    await act(async () => pending.resolve({ status: "failed" }))
    expect(input().value).toBe("")
    expect(useWorkspaceStore.getState().ownedWorkspaceComposer).toBe("")
  })

  it("does not dispatch after capability preparation crosses scopes", async () => {
    const pending = deferred<never>()
    mount({
      researchWorkspaceCapabilitiesStale: true,
      onRefreshResearchWorkspaceCapabilities: () => pending.promise
    })
    submitDraft("Do not dispatch into the new account")
    act(() => {
      useWorkspaceStore.setState({
        activeWorkspaceOrigin: owned({ principalId: "2" }),
        ownedWorkspaceComposer: ""
      })
    })
    await act(async () => pending.resolve(undefined as never))
    expect(mocks.submit).not.toHaveBeenCalled()
    expect(input().value).toBe("")
  })

  it("coalesces repeated send attempts while capability preparation is pending", async () => {
    const pending = deferred<never>()
    const refresh = vi.fn(() => pending.promise)
    mount({
      researchWorkspaceCapabilitiesStale: true,
      onRefreshResearchWorkspaceCapabilities: refresh
    })
    submitDraft("Only send once")
    fireEvent.keyDown(input(), { key: "Enter" })
    fireEvent.click(screen.getByRole("button", { name: "Send" }))
    expect(refresh).toHaveBeenCalledTimes(1)
    expect(useWorkspaceStore.getState().ownedWorkspaceComposer).toBe("Only send once")
    await act(async () => pending.resolve(undefined as never))
    expect(mocks.submit).toHaveBeenCalledTimes(1)
  })

  it("does not dispatch full-source preparation after leaving and reopening the same scope", async () => {
    const pending = deferred<unknown>()
    mocks.media.mockReturnValue(pending.promise)
    useWorkspaceStore.setState({
      sources: [
        {
          id: "source-1",
          mediaId: 1,
          title: "Private source",
          type: "text",
          status: "ready",
          addedAt: new Date()
        }
      ],
      selectedSourceIds: ["source-1"]
    })
    const view = mount()
    fireEvent.click(
      screen.getByRole("switch", { name: "Include full source contents" })
    )
    submitDraft("Old source question")
    await waitFor(() => expect(mocks.media).toHaveBeenCalledTimes(1))
    view.unmount()
    mount()
    await act(async () =>
      pending.resolve({ content: { text: "Old private source contents" } })
    )
    expect(mocks.submit).not.toHaveBeenCalled()
    expect(input().value).toBe("Old source question")
    expect(useWorkspaceStore.getState().ownedWorkspaceComposer).toBe(
      "Old source question"
    )
  })

  it.each([false, true])(
    "retains the owned question until successful submission without clearing newer text (newer=%s)",
    async (newer) => {
      const pending = deferred<{ status: string }>()
      mocks.submit.mockReturnValue(pending.promise)
      mount()
      submitDraft("Pending question")
      await waitFor(() => expect(mocks.submit).toHaveBeenCalledTimes(1))
      expect(useWorkspaceStore.getState().ownedWorkspaceComposer).toBe(
        "Pending question"
      )
      if (newer)
        fireEvent.change(input(), { target: { value: "Newer question" } })
      await act(async () => pending.resolve({ status: "submitted" }))
      expect(input().value).toBe(newer ? "Newer question" : "")
      expect(useWorkspaceStore.getState().ownedWorkspaceComposer).toBe(
        newer ? "Newer question" : ""
      )
    }
  )

  it("does not show a stale submit error after changing workspace ID", async () => {
    const pending = deferred<{ status: string }>()
    mocks.submit.mockImplementation(async () => {
      await pending.promise
      throw new Error("Old request failed")
    })
    mount()
    submitDraft("Outgoing submission")
    await waitFor(() => expect(mocks.submit).toHaveBeenCalledTimes(1))
    act(() => {
      useWorkspaceStore.setState({
        workspaceId: "another-id",
        ownedWorkspaceComposer: "Next"
      })
    })
    await act(async () => pending.resolve({ status: "failed" }))
    expect(input().value).toBe("Next")
    expect(
      screen.queryByText(
        "Unable to reach server. Please check your connection and retry."
      )
    ).not.toBeInTheDocument()
  })

  it("does not carry an owned history or pending draft into local cache", async () => {
    const pending = deferred<{ status: string }>()
    mocks.submit.mockReturnValue(pending.promise)
    mount()
    submitDraft("Owned outgoing submission")
    await waitFor(() => expect(mocks.submit).toHaveBeenCalledTimes(1))
    act(() => {
      useStoreMessageOption.setState({
        history: [{ role: "user", content: "Owned history" }]
      })
    })
    act(() => {
      useWorkspaceStore.setState({
        activeWorkspaceOrigin: { kind: "legacy-local" }
      })
    })
    await act(async () => pending.resolve({ status: "failed" }))
    expect(input().value).toBe("")
    expect(mocks.get).toHaveBeenCalled()
    expect(
      mocks.save.mock.calls.every(([, session]) => session.history.length === 0)
    ).toBe(true)
  })

  it("does not carry a pending local submission into an owned draft", async () => {
    const pending = deferred<{ status: string }>()
    mocks.submit.mockReturnValue(pending.promise)
    useWorkspaceStore.setState({
      activeWorkspaceOrigin: { kind: "legacy-local" }
    })
    mount()
    submitDraft("Local outgoing submission")
    await waitFor(() => expect(mocks.submit).toHaveBeenCalledTimes(1))
    mocks.get.mockClear()
    mocks.save.mockClear()
    act(() => {
      useWorkspaceStore.setState({
        activeWorkspaceOrigin: owned(),
        ownedWorkspaceComposer: ""
      })
    })
    await act(async () => pending.resolve({ status: "failed" }))
    expect(input().value).toBe("")
    expect(useWorkspaceStore.getState().ownedWorkspaceComposer).toBe("")
    expect(mocks.get).not.toHaveBeenCalled()
    expect(mocks.save).not.toHaveBeenCalled()
  })

  it("does not clear messages or drafts when the owned scope is unchanged", () => {
    mount()
    fireEvent.change(input(), { target: { value: "Keep this draft" } })
    act(() => {
      useStoreMessageOption.setState({
        history: [{ role: "user", content: "Keep history" }]
      })
    })
    act(() => {
      useWorkspaceStore.setState({ activeWorkspaceOrigin: owned() })
    })
    expect(input().value).toBe("Keep this draft")
    expect(useStoreMessageOption.getState().history).toEqual([
      { role: "user", content: "Keep history" }
    ])
  })

  it("restores a failed submission in the same scope but preserves newer typing", async () => {
    const pending = deferred<{ status: string }>()
    mocks.submit.mockReturnValue(pending.promise)
    mount()
    submitDraft("Retry this")
    await waitFor(() => expect(mocks.submit).toHaveBeenCalledTimes(1))
    await act(async () => pending.resolve({ status: "failed" }))
    expect(input().value).toBe("Retry this")
    expect(useWorkspaceStore.getState().ownedWorkspaceComposer).toBe(
      "Retry this"
    )

    const next = deferred<{ status: string }>()
    mocks.submit.mockReturnValue(next.promise)
    submitDraft("Another request")
    await waitFor(() => expect(mocks.submit).toHaveBeenCalledTimes(2))
    fireEvent.change(input(), { target: { value: "New typing" } })
    await act(async () => next.resolve({ status: "failed" }))
    expect(input().value).toBe("New typing")
    expect(useWorkspaceStore.getState().ownedWorkspaceComposer).toBe(
      "New typing"
    )
  })

  it("preserves local automatic cache behavior and does not write owned drafts", () => {
    useWorkspaceStore.setState({
      activeWorkspaceOrigin: { kind: "legacy-local" }
    })
    mount()
    fireEvent.change(input(), { target: { value: "Local draft" } })
    expect(mocks.get).toHaveBeenCalled()
    expect(mocks.save).toHaveBeenCalled()
    expect(useWorkspaceStore.getState().ownedWorkspaceComposer).toBe(
      "Recovered owned draft"
    )
  })
})
